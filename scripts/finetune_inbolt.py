"""
Fine-tune FastFoundationStereo on the Inbolt dataset.

The Inbolt dataset provides:
  - realsense/{idx}/mono0.png  : left IR image  (uint8, 480x640)
  - realsense/{idx}/mono1.png  : right IR image (uint8, 480x640)
  - zivid/{idx}/depthmap_mm.png: ground-truth depth in mm (Zivid scanner, 1024x1224)

Strategy:
  - Freeze the ViT-L backbone (model.feature) to prevent overfitting on small datasets.
  - Train everything else with RAFT-style sequence loss over GRU iterations.
  - IR uint8 images are replicated to 3 channels.
  - Zivid depth is resized to RealSense image resolution before disparity conversion.
  - Depth is converted to disparity: disp = BF / depth_mm.

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_inbolt.py
"""

import os, sys, logging
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from core.utils.utils import InputPadder
import Utils as U
from inbolt_data_manager import DataSource


# ── constants ────────────────────────────────────────────────────────────────

INBOLT_DIR   = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260322T091926Z-1-001/Data Collection'  # local path to the dataset
INBOLT_DIR   = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection' 
# MODEL_PATH = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
# OUT_PATH   = f'{code_dir}/../weights/20-30-48/model_finetuned_inbolt-20260415.pth'
MODEL_PATH = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
OUT_PATH   = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415.pth'


# BF         = 49.8624*385.73  # D435 - focal_px * baseline_mm (calibrated from camera)  # D435 - focal_px * baseline_mm (calibrated from camera)
BF         = 50.102706998586 * 385.509887695312 # new data
EPOCHS     = 120
LR         = 2e-5
ITERS      = 8          # GRU iterations (same as inference)
GAMMA      = 0.9        # sequence loss weight decay
TRAIN_RATIO = 0.75
SPLIT_SEED  = 0


# ── dataset ──────────────────────────────────────────────────────────────────

class InboltDataset(Dataset):
    def __init__(self, root):
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

        # Resize Zivid depth to match RealSense stereo image resolution
        h, w  = left.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # IR uint8 → float [0, 255], replicate to 3-channel pseudo-RGB
        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)  # H x W x 3
        right = np.stack([right, right, right], axis=-1)

        # depth (mm) → disparity (pixels):  disp = focal * baseline / depth
        disp  = np.zeros_like(depth, dtype=np.float32)
        valid = depth > 0
        disp[valid] = BF / depth[valid]

        left_t  = torch.from_numpy(left).permute(2, 0, 1).float()   # (3, H, W)
        right_t = torch.from_numpy(right).permute(2, 0, 1).float()  # (3, H, W)
        disp_t  = torch.from_numpy(disp).unsqueeze(0).float()       # (1, H, W)
        valid_t = torch.from_numpy(valid).unsqueeze(0)               # (1, H, W) bool

        return left_t, right_t, disp_t, valid_t


# ── loss ─────────────────────────────────────────────────────────────────────

def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted sum of smooth-L1 losses over GRU iterations."""
    n    = len(disp_preds)
    loss = 0.0
    for i, pred in enumerate(disp_preds):
        w  = gamma ** (n - 1 - i)
        gt = disp_gt
        v  = valid
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            v  = F.interpolate(valid.float(), size=pred.shape[-2:], mode='nearest').bool()
        loss = loss + w * F.smooth_l1_loss(pred[v], gt[v])
    return loss


def evaluate_split_loss(model, dataloader):
    """Evaluate average sequence loss over a dataloader (no gradient updates)."""
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
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    # load full model object (weights + architecture)
    logging.info(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)

    # freeze the ViT-L backbone — with only 24 samples it would overfit
    for param in model.feature.parameters():
        param.requires_grad = False
    logging.info("ViT backbone frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable: {trainable:,} / {total:,} parameters")

    model = model.cuda().train()
    logging.info("Model on single GPU.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    dataset = InboltDataset(INBOLT_DIR)
    n_total = len(dataset)

    if n_total < 2:
        raise RuntimeError(f"Need at least 2 samples for a 75/25 train/test split, got {n_total}.")

    n_train = int(round(TRAIN_RATIO * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test = n_total - n_train

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=split_generator)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    logging.info(
        f"Random split with seed={SPLIT_SEED}: total={n_total}, train={len(train_set)} ({100.0*len(train_set)/n_total:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total:.1f}%)"
    )

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for left, right, disp_gt, valid in train_loader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            # pad so H and W are divisible by 32
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_eval_error = evaluate_split_loss(model, train_loader)
        test_eval_error = evaluate_split_loss(model, test_loader)

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_eval_error={train_eval_error:.4f}  test_eval_error={test_eval_error:.4f}"
        )

        if test_eval_error < best_loss:
            best_loss = test_eval_error
            torch.save(model, OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth'))
            logging.info(f"  → saved best model (test_eval_error={best_loss:.4f})")

    final_train_error = evaluate_split_loss(model, train_loader)
    final_test_error = evaluate_split_loss(model, test_loader)
    logging.info(f"Final train error: {final_train_error:.4f}")
    logging.info(f"Final test error:  {final_test_error:.4f}")
    logging.info(f"Training complete. Best test error: {best_loss:.4f}")
    logging.info(f"Model saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
