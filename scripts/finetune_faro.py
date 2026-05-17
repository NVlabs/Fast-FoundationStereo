"""
Fine-tune FastFoundationStereo on the FARO dataset.

The FARO dataset provides:
  - img_left.png / img_right.png : uint16 IR stereo images
  - img_depth_faro.png           : ground-truth depth in mm (FARO scanner)

Strategy:
  - Freeze the ViT-L backbone (model.feature) to prevent overfitting on 24 samples.
  - Train everything else with RAFT-style sequence loss over GRU iterations.
  - IR uint16 images are clipped to [0,255] and replicated to 3 channels.
  - Depth is converted to disparity: disp = BF / depth_mm  (BF = 49470.45).

Usage:
  cd /home/administrato/dev/Fast-FoundationStereo
  python scripts/finetune_faro.py
"""

import os, sys, logging
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from core.utils.utils import InputPadder
import Utils as U
from scripts.data_manager_faro import DataSource


# ── constants ────────────────────────────────────────────────────────────────

FARO_DIR   = r'/mnt/algonas/Local/Data/Stereo/Faro/FARO_DATA_BASE'  # local path to the dataset
MODEL_PATH = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
OUT_PATH   = f'{code_dir}/../weights/20-30-48/model_finetuned_faro_office.pth'
#MODEL_PATH = f'{code_dir}/../weights/20-30-48/model_finetuned_faro_kitchen.pth'

BF         = 49470.45   # focal_px * baseline_mm (calibrated from camera)
EPOCHS     = 15
LR         = 2e-5
ITERS      = 8          # GRU iterations (same as inference)
GAMMA      = 0.9        # sequence loss weight decay


# ── dataset ──────────────────────────────────────────────────────────────────

class FaroDataset(Dataset):
    def __init__(self, root):
        self.source = DataSource()
        n = self.source.init_directory(input_rectified=root, test_keywords=['OFFICE'], split='train')
        logging.info(f"DataSource found {n} samples in {root}")

    def __len__(self):
        return len(self.source.imgs)

    def __getitem__(self, idx):
        data  = self.source.get_item(idx)
        left  = data['left']
        right = data['right']
        depth = data['depth_faro']   # float32, mm

        # uint16 IR → float [0, 255], replicate to 3-channel pseudo-RGB
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

        return left_t, right_t, disp_t, valid_t, idx


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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    # load full model object (weights + architecture)
    logging.info(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    # freeze the ViT-L backbone — with only 24 samples it would overfit
    for param in model.feature.parameters():
        param.requires_grad = False
    logging.info("ViT backbone frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable: {trainable:,} / {total:,} parameters")

    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda().train()
    logging.info("Using DataParallel on GPUs 0 and 1.")

    optimizer = torch.optim.AdamW(
        [p for p in model.module.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    dataset    = FaroDataset(FARO_DIR)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    best_loss = float('inf')

    for epoch in range(0,EPOCHS):
        epoch_loss = 0.0

        for left, right, disp_gt, valid, idx in dataloader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            # pad so H and W are divisible by 32
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if idx[0] % 10 == 0:
                logging.info(f"Epoch {epoch+1:3d}/{EPOCHS}  sample {idx[0]:3d}  loss={loss.item():.4f}")

        avg = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.module, OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth'))
            logging.info(f"  → saved best model (loss={best_loss:.4f})")

    logging.info(f"Training complete. Best loss: {best_loss:.4f}")
    logging.info(f"Model saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
