"""
Fine-tune FastFoundationStereo + RealSense depth fusion v2 on the Inbolt dataset.

Architecture (no DepthEncoder / no DepthFusionModule — feature extraction fully frozen):

  INPUTS
    left_IR  (B,3,H,W)
    right_IR (B,3,H,W)
    depth_rs_mm (B,1,H,W)

  FEATURE EXTRACTION  [FROZEN — stereo.feature + stereo.stem_2]
    EdgeNeXt backbone + FPN decoder
    → feat_left[x4,x8,x16,x32], feat_right[x4,x8,x16,x32], stem_2x

  DEPTH PREPROCESSING  [no learned params]
    validity M = (depth_rs_mm > 0) & isfinite
    depth_rs_m = depth_rs_mm / 1000.0 · M
    disp_prior = focal·baseline_m / depth_rs_m   (full-res pixels)
    disp_prior_14 = disp_prior / 4               (1/4-scale pixels)
    validity_14   = F.interpolate(M, H/4, W/4)

  COST VOLUME  [fine-tune 0.1× LR]
    feat_left[0]+feat_right[0] → GWC+Concat → hourglass → logits (B,D,H/4,W/4)

  STAGE 2 — GAUSSIAN LOGIT PRIOR  [learnable σ, α — full LR]
    prior_bias = −½·((d_idx − disp_prior_14) / σ)² · validity_14
    logits' = logits + α·prior_bias
    → stereo_init = softmax → disparity_regression   (B,1,H/4,W/4)

  STAGE 3a — GRU INIT BLEND  [DepthInitBlend ~5K — full LR]
    concat(stereo_init, disp_prior_14, validity_14)
    → sigmoid (3-layer CNN) → blend_w · validity_14
    → init_disp = blend_w·disp_prior_14 + (1−blend_w)·stereo_init

  CONTEXT NETWORK + GRU ITERATIONS  [fine-tune 0.1× LR]
    init_disp → 8× GRU update → disp_up (B,1,H,W)

  STAGE 3b — OUTPUT BLEND  [DepthOutputBlend ~8K — full LR]  ← NEW
    concat(disp_up, disp_prior_full, validity)
    → sigmoid (3-layer CNN) → blend_w_out · validity
    → disp_final = disp_up + blend_w_out·(disp_prior_full − disp_up)

  OUTPUT: disp_final (B,1,H,W)

Freezing strategy:
  Frozen    : stereo.feature (all submodules), stereo.stem_2
  Full LR   : DepthInitBlend, DepthOutputBlend, depth_sigma, depth_prior_scale
  0.1× LR   : all other trainable stereo params

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_inbolt_depthrs_2.py
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
from core.submodule import (
    BasicConv,
    disparity_regression,
    build_gwc_volume_optimized_pytorch1,
    build_concat_volume_optimized_pytorch1,
)
from core.geometry import Combined_Geo_Encoding_Volume
from core.foundation_stereo import normalize_image
import Utils as U
from scripts.data_manager_inbolt import DataSource


# ── constants ────────────────────────────────────────────────────────────────

INBOLT_DIR  = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
MODEL_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415_epoch_111.pth'
OUT_PATH    = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_depthrs_v2.pth'

BF            = 50.102706998586 * 385.509887695312   # focal_px * baseline_mm
FOCAL_PX      = 385.509887695312
BASELINE_MM   = 50.102706998586

EPOCHS        = 120
LR            = 2e-5
ITERS         = 8
GAMMA         = 0.9
TRAIN_RATIO   = 0.75
SPLIT_SEED    = 0


# ── depth fusion modules ──────────────────────────────────────────────────────

class DepthInitBlend(nn.Module):
    """Blend stereo soft-argmax init with depth prior at 1/4-scale for GRU warm start."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, padding=1, bn=True, relu=True),
            BasicConv(16, 16, kernel_size=3, padding=1, bn=True, relu=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)   # sigmoid(-5) ≈ 0 → no blend at init

    def forward(self, stereo_init, disp_prior, validity):
        # all inputs: (B, 1, H/4, W/4)
        x = torch.cat([stereo_init, disp_prior, validity], dim=1)
        blend_w = torch.sigmoid(self.net(x)) * validity
        return blend_w * disp_prior + (1.0 - blend_w) * stereo_init


class DepthOutputBlend(nn.Module):
    """Refine full-resolution GRU output with depth prior (residual, zero-init)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1, bn=True, relu=True),
            BasicConv(32, 32, kernel_size=3, padding=1, bn=True, relu=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)   # no correction at init

    def forward(self, disp_stereo, disp_prior_full, validity):
        # all inputs: (B, 1, H, W)
        x = torch.cat([disp_stereo, disp_prior_full, validity], dim=1)
        blend_w = torch.sigmoid(self.net(x)) * validity
        return disp_stereo + blend_w * (disp_prior_full - disp_stereo)


# ── model wrapper ─────────────────────────────────────────────────────────────

class FastFoundationStereoDepthRS_v2(nn.Module):
    """
    Wraps a pretrained FastFoundationStereo and adds RealSense depth fusion
    at three pipeline stages.  Feature extraction (stereo.feature + stereo.stem_2)
    is expected to be frozen before training.

    New modules (no DepthEncoder / no DepthFusionModule):
      depth_sigma, depth_prior_scale  — Stage 2 Gaussian prior on logits
      DepthInitBlend                  — Stage 3a GRU warm-start blend
      DepthOutputBlend                — Stage 3b full-resolution output correction
    """

    def __init__(self, stereo_model):
        super().__init__()
        self.stereo = stereo_model
        self.args   = stereo_model.args

        self.depth_init_blend   = DepthInitBlend()
        self.depth_output_blend = DepthOutputBlend()

        self.depth_sigma       = nn.Parameter(torch.tensor(4.0))
        self.depth_prior_scale = nn.Parameter(torch.tensor(0.1))

    # ------------------------------------------------------------------
    def _preprocess_depth(self, depth_rs_mm):
        """
        Returns:
          depth_rs_m  (B,1,H,W) metres, zeros where invalid
          validity    (B,1,H,W) float 0/1
          disp_prior  (B,1,H,W) full-res disparity in pixels, zeros where invalid
        """
        validity   = ((depth_rs_mm > 0) & torch.isfinite(depth_rs_mm)).float()
        depth_rs_m = (depth_rs_mm / 1000.0) * validity
        disp_prior = torch.where(
            validity > 0.5,
            (FOCAL_PX * BASELINE_MM / 1000.0) / depth_rs_m.clamp(min=1e-3),
            torch.zeros_like(depth_rs_m),
        )
        return depth_rs_m, validity, disp_prior

    # ------------------------------------------------------------------
    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        return self.stereo.upsample_disp(disp, mask_feat_4, stem_2x)

    # ------------------------------------------------------------------
    def forward(
        self,
        image1,
        image2,
        depth_rs_mm=None,
        iters=12,
        test_mode=False,
        low_memory=False,
    ):
        s = self.stereo
        B, C, H, W = image1.shape
        low_memory = low_memory or self.args.get('low_memory', False)

        image1 = normalize_image(image1)
        image2 = normalize_image(image2)

        with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):

            # ── feature extraction (frozen) ───────────────────────────
            out            = s.feature(torch.cat([image1, image2], dim=0))
            features_left  = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x        = s.stem_2(image1)

            # ── depth preprocessing ───────────────────────────────────
            disp_prior_14 = None
            validity_14   = None
            disp_prior_full = None
            validity_full   = None

            if depth_rs_mm is not None:
                depth_rs_m, validity_full, disp_prior_full = self._preprocess_depth(depth_rs_mm)
                disp_prior_14 = F.interpolate(disp_prior_full / 4.0, size=(H // 4, W // 4), mode='nearest')
                validity_14   = F.interpolate(validity_full,          size=(H // 4, W // 4), mode='nearest')

            # ── cost volume ───────────────────────────────────────────
            gwc_volume = build_gwc_volume_optimized_pytorch1(
                features_left[0], features_right[0],
                self.args.max_disp // 4, s.cv_group,
                normalize=self.args.normalize,
            )
            left_tmp      = s.proj_cmb(features_left[0])
            right_tmp     = s.proj_cmb(features_right[0])
            concat_volume = build_concat_volume_optimized_pytorch1(
                left_tmp, right_tmp, maxdisp=self.args.max_disp // 4
            )
            del left_tmp, right_tmp

            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del concat_volume, gwc_volume

            comb_volume = s.corr_stem(comb_volume)
            comb_volume = s.corr_feature_att(comb_volume, features_left[0])
            comb_volume = s.cost_agg(comb_volume, features_left)

            # ── Stage 2: Gaussian prior on logits ─────────────────────
            logits = s.classifier(comb_volume).squeeze(1)   # (B, D, H/4, W/4)

            if depth_rs_mm is not None:
                D     = logits.shape[1]
                d_idx = torch.arange(D, device=logits.device, dtype=logits.dtype).view(1, D, 1, 1)
                sigma = self.depth_sigma.abs().clamp(min=0.1)
                prior_bias = -0.5 * ((d_idx - disp_prior_14) / sigma) ** 2
                prior_bias = prior_bias * validity_14
                logits = logits + self.depth_prior_scale * prior_bias

            prob        = F.softmax(logits, dim=1)
            stereo_init = disparity_regression(prob, self.args.max_disp // 4)  # (B,1,H/4,W/4)

            # ── Stage 3a: GRU init blend ──────────────────────────────
            if depth_rs_mm is not None:
                init_disp = self.depth_init_blend(stereo_init, disp_prior_14, validity_14)
            else:
                init_disp = stereo_init

            # ── context network ───────────────────────────────────────
            cnet_list = s.cnet(features_left[0], features_left[1], features_left[2])
            cnet_list = list(cnet_list)
            net_list  = [torch.tanh(x[0]) for x in cnet_list]
            inp_list  = [torch.relu(x[1]) for x in cnet_list]
            inp_list  = [s.cam(x) * x for x in inp_list]
            att       = [s.sam(x) for x in inp_list]

        # ── geometry encoding volume ──────────────────────────────────
        geo_fn = Combined_Geo_Encoding_Volume(
            features_left[0].to(s.dtype),
            features_right[0].to(s.dtype),
            comb_volume.to(s.dtype),
            num_levels=self.args.corr_levels,
        )
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp   = init_disp.to(s.dtype)
        disp_preds = []

        del comb_volume, features_left, features_right, cnet_list

        # ── GRU iterations ────────────────────────────────────────────
        for itr in range(iters):
            disp     = disp.detach()
            geo_feat = geo_fn(disp, coords, dx=s.dx, low_memory=low_memory)

            with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
                net_list, mask_feat_4, delta_disp = s.update_block(
                    net_list, inp_list, geo_feat.to(s.dtype), disp, att
                )

            disp    = disp + delta_disp.to(s.dtype)
            if test_mode and itr < iters - 1:
                continue

            disp_up = self.upsample_disp(disp.to(s.dtype), mask_feat_4.to(s.dtype), stem_2x.to(s.dtype))
            disp_preds.append(disp_up)

        # ── Stage 3b: output blend (full resolution) ──────────────────
        if depth_rs_mm is not None:
            disp_preds = [
                self.depth_output_blend(p, disp_prior_full, validity_full)
                for p in disp_preds
            ]

        if test_mode:
            return disp_preds[-1]

        return init_disp, disp_preds


# ── dataset ───────────────────────────────────────────────────────────────────

class InboltDepthDataset(Dataset):
    """Returns (left, right, disp_gt, valid, depth_rs_mm) for each sample."""

    def __init__(self, root):
        self.source = DataSource()
        n = self.source.init_directory(input_rectified=root)
        logging.info(f"DataSource found {n} samples in {root}")

    def __len__(self):
        return len(self.source.imgs)

    def __getitem__(self, idx):
        data        = self.source.get_item_projected(idx)
        left        = data['left']
        right       = data['right']
        depth_zivid = data['depth_zivid']
        depth_rs    = data['depth_rs']

        h, w = left.shape[:2]
        if depth_zivid.shape != (h, w):
            depth_zivid = cv2.resize(depth_zivid, (w, h), interpolation=cv2.INTER_NEAREST)
        if depth_rs.shape != (h, w):
            depth_rs = cv2.resize(depth_rs, (w, h), interpolation=cv2.INTER_NEAREST)

        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)
        right = np.stack([right, right, right], axis=-1)

        disp        = np.zeros_like(depth_zivid, dtype=np.float32)
        valid       = depth_zivid > 0
        disp[valid] = BF / depth_zivid[valid]

        left_t     = torch.from_numpy(left).permute(2, 0, 1).float()
        right_t    = torch.from_numpy(right).permute(2, 0, 1).float()
        disp_t     = torch.from_numpy(disp).unsqueeze(0).float()
        valid_t    = torch.from_numpy(valid).unsqueeze(0)
        depth_rs_t = torch.from_numpy(depth_rs).unsqueeze(0).float()

        return left_t, right_t, disp_t, valid_t, depth_rs_t


# ── loss ──────────────────────────────────────────────────────────────────────

def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted smooth-L1 sum over GRU iterations."""
    n    = len(disp_preds)
    loss = 0.0
    for i, pred in enumerate(disp_preds):
        w = gamma ** (n - 1 - i)
        gt = disp_gt
        v  = valid
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            v  = F.interpolate(valid.float(), size=pred.shape[-2:], mode='nearest').bool()
        if not v.any():
            continue
        loss = loss + w * F.smooth_l1_loss(pred[v], gt[v])
    return loss


def evaluate_split_loss(model, dataloader):
    if len(dataloader) == 0:
        return float('nan')
    model.eval()
    total = 0.0
    with torch.no_grad():
        for left, right, disp_gt, valid, depth_rs in dataloader:
            left, right    = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()
            depth_rs       = depth_rs.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p, dr_p = padder.pad(left, right, depth_rs)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init, preds = model.forward(left_p, right_p, depth_rs_mm=dr_p,
                                             iters=ITERS, test_mode=False)
                preds = [padder.unpad(p) for p in preds]
                total += sequence_loss(preds, disp_gt, valid).item()

    model.train()
    return total / len(dataloader)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    logging.info(f"Loading base stereo model from {MODEL_PATH}")
    stereo_model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)
    model = FastFoundationStereoDepthRS_v2(stereo_model).cuda()

    # ── freeze entire feature extractor (backbone + FPN + stem_2) ─────
    for param in model.stereo.feature.parameters():
        param.requires_grad = False
    for param in model.stereo.stem_2.parameters():
        param.requires_grad = False
    logging.info("Feature extractor (stereo.feature + stereo.stem_2) fully frozen.")

    # ── parameter groups ──────────────────────────────────────────────
    new_params = (
        list(model.depth_init_blend.parameters())   +
        list(model.depth_output_blend.parameters()) +
        [model.depth_sigma, model.depth_prior_scale]
    )
    new_param_ids   = {id(p) for p in new_params}
    finetune_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in new_param_ids
    ]

    n_new      = sum(p.numel() for p in new_params)
    n_ft       = sum(p.numel() for p in finetune_params)
    n_total    = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Parameters — new (full LR): {n_new:,}  "
        f"fine-tune (0.1× LR): {n_ft:,}  "
        f"frozen: {n_total - n_new - n_ft:,}  "
        f"total: {n_total:,}"
    )

    optimizer = torch.optim.AdamW(
        [
            {'params': new_params,      'lr': LR},
            {'params': finetune_params, 'lr': LR * 0.1},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── dataset ───────────────────────────────────────────────────────
    dataset = InboltDepthDataset(INBOLT_DIR)
    n_total_data = len(dataset)
    if n_total_data < 2:
        raise RuntimeError(f"Need at least 2 samples, got {n_total_data}.")

    n_train = min(max(1, int(round(TRAIN_RATIO * n_total_data))), n_total_data - 1)
    n_test  = n_total_data - n_train

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0)

    logging.info(
        f"Split (seed={SPLIT_SEED}): total={n_total_data}, "
        f"train={len(train_set)} ({100.0*len(train_set)/n_total_data:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total_data:.1f}%)"
    )

    model.train()
    best_loss = float('inf')

    # ── training loop ─────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for left, right, disp_gt, valid, depth_rs in train_loader:
            left, right    = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()
            depth_rs       = depth_rs.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p, dr_p = padder.pad(left, right, depth_rs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init, preds = model.forward(
                    left_p, right_p, depth_rs_mm=dr_p, iters=ITERS, test_mode=False
                )
                preds = [padder.unpad(p) for p in preds]
                loss  = sequence_loss(preds, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_eval = evaluate_split_loss(model, train_loader)
        test_eval  = evaluate_split_loss(model, test_loader)

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"train_eval={train_eval:.4f}  "
            f"test_eval={test_eval:.4f}  "
            f"depth_sigma={model.depth_sigma.item():.3f}  "
            f"depth_prior_scale={model.depth_prior_scale.item():.4f}"
        )

        if test_eval < best_loss:
            best_loss = test_eval
            save_path = OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth')
            torch.save(model, save_path)
            logging.info(f"  → saved best model (test_eval={best_loss:.4f}) to {save_path}")

    final_train = evaluate_split_loss(model, train_loader)
    final_test  = evaluate_split_loss(model, test_loader)
    logging.info(f"Final train error: {final_train:.4f}")
    logging.info(f"Final test  error: {final_test:.4f}")
    logging.info(f"Best  test  error: {best_loss:.4f}")
    torch.save(model, OUT_PATH)
    logging.info(f"Final model saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
