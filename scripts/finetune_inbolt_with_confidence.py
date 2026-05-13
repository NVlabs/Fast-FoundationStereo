"""
Fine-tune FastFoundationStereo on the Inbolt dataset with an additional confidence head.

The confidence head predicts pixel validity from the stereo pair:
  confidence = 1 → pixel has a valid Zivid depth measurement
  confidence = 0 → pixel has no valid Zivid measurement (specular, transparent, OOB)

Loss:
  - RAFT-style smooth-L1 sequence loss on valid pixels  (disparity)
  - Binary cross-entropy on all pixels                   (confidence, target = valid_mask)

At the end of training an evaluation section compares depth performance on the test split
between the original model and the newly trained confidence model.

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_inbolt_with_confidence.py
"""

import os, sys, logging
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from core.utils.utils import InputPadder
from core.foundation_stereo import normalize_image
import Utils as U
from scripts.data_manager_inbolt import DataSource


# ── constants ────────────────────────────────────────────────────────────────

INBOLT_DIR  = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
MODEL_PATH  = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
OUT_PATH    = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_with_confidence-20260507.pth'

BF          = 50.102706998586 * 385.509887695312   # focal_px * baseline_mm
EPOCHS      = 120
LR          = 2e-5
ITERS       = 8
GAMMA       = 0.9
CONF_WEIGHT = 1.0    # weight of confidence BCE loss relative to disparity loss
TRAIN_RATIO = 0.75
SPLIT_SEED  = 0


# ── confidence head ───────────────────────────────────────────────────────────

class ConfidenceHead(nn.Module):
    """Lightweight head: stem_2 features at H/2 → confidence map at H (sigmoid, 0–1)."""

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # ×2 upsample
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── model wrapper ─────────────────────────────────────────────────────────────

class FastFoundationStereoWithConfidence(nn.Module):
    """
    Wraps a pretrained FastFoundationStereo and adds a ConfidenceHead.

    forward(..., test_mode=False) → (init_disp, disp_preds, conf)
    forward(..., test_mode=True)  → (disp_up, conf)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        # stem_2 outputs 16 channels at H/2 resolution — cheap to re-evaluate
        self.conf_head = ConfidenceHead(in_channels=16)

    # expose base.feature so the freeze loop in main() still works
    @property
    def feature(self):
        return self.base.feature

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        test_mode: bool = False,
        **kwargs,
    ):
        result = self.base.forward(image1, image2, iters=iters, test_mode=test_mode, **kwargs)

        # Recompute stem_2x for the confidence head.
        # normalize_image is pure arithmetic; stem_2 is a 3-conv lightweight block.
        with torch.amp.autocast('cuda', enabled=self.base.args.mixed_precision, dtype=U.AMP_DTYPE):
            stem_2x = self.base.stem_2(normalize_image(image1))   # (B, 32, H/2, W/2)
        conf = self.conf_head(stem_2x.float())                     # (B, 1, H, W)

        if test_mode:
            return result, conf    # (disp_up, conf)
        else:
            init_disp, disp_preds = result
            return init_disp, disp_preds, conf


# ── dataset ───────────────────────────────────────────────────────────────────

class InboltDataset(Dataset):
    def __init__(self, root: str):
        self.source = DataSource()
        n = self.source.init_directory(input_rectified=root)
        logging.info(f"DataSource found {n} samples in {root}")

    def __len__(self):
        return len(self.source.imgs)

    def __getitem__(self, idx):
        data  = self.source.get_item_projected(idx)
        left  = data['left']
        right = data['right']
        depth = data['depth_zivid']   # float32, mm  (Zivid resolution)

        h, w = left.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)
        right = np.stack([right, right, right], axis=-1)

        disp  = np.zeros_like(depth, dtype=np.float32)
        valid = depth > 0
        disp[valid] = BF / depth[valid]

        left_t  = torch.from_numpy(left).permute(2, 0, 1).float()
        right_t = torch.from_numpy(right).permute(2, 0, 1).float()
        disp_t  = torch.from_numpy(disp).unsqueeze(0).float()
        valid_t = torch.from_numpy(valid).unsqueeze(0)

        return left_t, right_t, disp_t, valid_t


# ── loss ──────────────────────────────────────────────────────────────────────

def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted smooth-L1 loss on valid pixels."""
    n    = len(disp_preds)
    loss = 0.0
    for i, pred in enumerate(disp_preds):
        w = gamma ** (n - 1 - i)
        gt = disp_gt
        v  = valid
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            v  = F.interpolate(valid.float(), size=pred.shape[-2:], mode='nearest').bool()
        loss = loss + w * F.smooth_l1_loss(pred[v], gt[v])
    return loss


def sequence_loss_with_confidence(disp_preds, conf, disp_gt, valid, gamma=GAMMA, conf_weight=CONF_WEIGHT):
    """
    Combined loss:
      - Smooth-L1 disparity sequence loss on valid pixels.
      - BCE confidence loss on all pixels: target=1 where Zivid is valid, 0 otherwise.
    """
    disp_loss = sequence_loss(disp_preds, disp_gt, valid, gamma)

    conf_target = valid.float()
    if conf.shape[-2:] != conf_target.shape[-2:]:
        conf_target = F.interpolate(conf_target, size=conf.shape[-2:], mode='nearest')
    # BCE is blocked by PyTorch's autocast dispatcher regardless of dtype; disable it here.
    with torch.amp.autocast('cuda', enabled=False):
        conf_loss = F.binary_cross_entropy(conf.float(), conf_target.float())

    return disp_loss + conf_weight * conf_loss, disp_loss, conf_loss


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate_split_loss(model, dataloader):
    """Average combined loss over a dataloader (no grad)."""
    if len(dataloader) == 0:
        return float('nan')

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for left, right, disp_gt, valid in dataloader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds, conf = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss, _, _ = sequence_loss_with_confidence(disp_preds, conf, disp_gt, valid)

            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


def _depth_mae_m(model_or_fn, dataloader, with_confidence: bool):
    """
    Compute depth MAE (metres) on valid GT pixels for a model.
    model_or_fn: either a FastFoundationStereoWithConfidence (with_confidence=True)
                 or the original FastFoundationStereo (with_confidence=False).
    Returns (mae_mm, coverage_pct)
    """
    total_abs_err = 0.0
    total_valid_gt = 0
    total_valid_pred = 0
    total_pixels = 0

    with torch.no_grad():
        for left, right, disp_gt, valid in dataloader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                if with_confidence:
                    disp_up, _conf = model_or_fn.forward(left_p, right_p, iters=ITERS, test_mode=True)
                else:
                    disp_up = model_or_fn.forward(left_p, right_p, iters=ITERS, test_mode=True)

            disp_up = padder.unpad(disp_up.float())  # (1, 1, H, W)

            # Convert disparity → depth (mm) where disp > 0
            pred_disp_np = disp_up.squeeze().cpu().numpy().clip(0, None)
            gt_disp_np   = disp_gt.squeeze().cpu().numpy()
            valid_np     = valid.squeeze().cpu().numpy().astype(bool)

            pred_depth = np.zeros_like(pred_disp_np)
            ok = pred_disp_np > 0
            pred_depth[ok] = BF / pred_disp_np[ok]  # mm

            gt_depth = np.zeros_like(gt_disp_np)
            gt_ok = gt_disp_np > 0
            gt_depth[gt_ok] = BF / gt_disp_np[gt_ok]  # mm

            # Only evaluate where GT is valid
            mask = valid_np & ok
            if mask.any():
                total_abs_err   += float(np.abs(pred_depth[mask] - gt_depth[mask]).sum())
                total_valid_pred += int(mask.sum())
            total_valid_gt += int(valid_np.sum())
            total_pixels   += valid_np.size

    mae_mm = total_abs_err / max(total_valid_pred, 1)
    coverage = 100.0 * total_valid_pred / max(total_valid_gt, 1)
    return mae_mm, coverage


def _confidence_metrics(model, dataloader):
    """Compute confidence accuracy (treating conf>0.5 as valid prediction)."""
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for left, right, _disp_gt, valid in dataloader:
            left, right = left.cuda(), right.cuda()
            valid = valid.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _disp_up, conf = model.forward(left_p, right_p, iters=ITERS, test_mode=True)

            conf = padder.unpad(conf)
            pred_valid = (conf > 0.5).squeeze().cpu()
            gt_valid   = valid.squeeze().cpu().bool()

            tp += int((pred_valid & gt_valid).sum())
            tn += int((~pred_valid & ~gt_valid).sum())
            fp += int((pred_valid & ~gt_valid).sum())
            fn += int((~pred_valid & gt_valid).sum())

    accuracy  = 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = 100.0 * tp / max(tp + fp, 1)
    recall    = 100.0 * tp / max(tp + fn, 1)
    return accuracy, precision, recall


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    logging.info(f"Loading base model from {MODEL_PATH}")
    base_model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)

    model = FastFoundationStereoWithConfidence(base_model)

    # freeze the ViT-L backbone — conf_head and the rest of the model will train
    for param in model.feature.parameters():
        param.requires_grad = False
    logging.info("ViT backbone frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable: {trainable:,} / {total:,} parameters")

    model = model.cuda().train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    dataset  = InboltDataset(INBOLT_DIR)
    n_total  = len(dataset)

    if n_total < 2:
        raise RuntimeError(f"Need at least 2 samples, got {n_total}.")

    n_train = min(max(1, int(round(TRAIN_RATIO * n_total))), n_total - 1)
    n_test  = n_total - n_train

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0)

    logging.info(
        f"Split seed={SPLIT_SEED}: total={n_total}, "
        f"train={len(train_set)} ({100.0*len(train_set)/n_total:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total:.1f}%)"
    )

    best_loss    = float('inf')
    best_ckpt    = None

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_loss = epoch_disp_loss = epoch_conf_loss = 0.0

        for left, right, disp_gt, valid in train_loader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds, conf = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss, d_loss, c_loss = sequence_loss_with_confidence(
                    disp_preds, conf, disp_gt, valid
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss      += loss.item()
            epoch_disp_loss += d_loss.item()
            epoch_conf_loss += c_loss.item()

        n_batches = len(train_loader)
        train_loss = epoch_loss      / n_batches
        train_eval = evaluate_split_loss(model, train_loader)
        test_eval  = evaluate_split_loss(model, test_loader)

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"(disp={epoch_disp_loss/n_batches:.4f}  conf={epoch_conf_loss/n_batches:.4f})  "
            f"train_eval={train_eval:.4f}  test_eval={test_eval:.4f}"
        )

        if test_eval < best_loss:
            best_loss = test_eval
            best_ckpt = OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth')
            torch.save(model, best_ckpt)
            logging.info(f"  → saved best model (test_eval={best_loss:.4f})")

    logging.info(f"Training complete. Best test_eval={best_loss:.4f}")

    # ── depth performance comparison ──────────────────────────────────────────
    logging.info("\n── Depth performance comparison (test split) ──────────────────────────")

    model.eval()

    # confidence model (best checkpoint or final model)
    eval_model = model
    if best_ckpt is not None:
        logging.info(f"Loading best checkpoint for evaluation: {best_ckpt}")
        eval_model = torch.load(best_ckpt, map_location='cuda', weights_only=False)
        eval_model.eval()

    conf_mae, conf_cov = _depth_mae_m(eval_model, test_loader, with_confidence=True)
    conf_acc, conf_prec, conf_rec = _confidence_metrics(eval_model, test_loader)
    logging.info(
        f"[Confidence model]  depth MAE={conf_mae:.2f} mm  coverage={conf_cov:.1f}%  "
        f"conf_acc={conf_acc:.1f}%  conf_prec={conf_prec:.1f}%  conf_rec={conf_rec:.1f}%"
    )

    # original model for comparison
    logging.info(f"Loading original model for comparison: {MODEL_PATH}")
    orig_model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)
    orig_model.eval()
    orig_mae, orig_cov = _depth_mae_m(orig_model, test_loader, with_confidence=False)
    logging.info(
        f"[Original model]    depth MAE={orig_mae:.2f} mm  coverage={orig_cov:.1f}%"
    )

    mae_delta = conf_mae - orig_mae
    sign = "+" if mae_delta >= 0 else ""
    logging.info(
        f"\nDepth MAE delta (confidence − original): {sign}{mae_delta:.2f} mm  "
        f"(negative = confidence model is better)"
    )
    logging.info(f"Model saved to {OUT_PATH} (best: {best_ckpt})")


if __name__ == '__main__':
    main()
