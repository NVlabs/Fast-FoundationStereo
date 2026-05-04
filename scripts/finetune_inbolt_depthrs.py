"""
Fine-tune FastFoundationStereo + RealSense depth fusion on the Inbolt dataset.

Extends finetune_inbolt.py with three-stage depth fusion as specified in
docs/superpowers/specs/2026-04-30-depth-fusion-design.md:

  Stage 1 - DepthEncoder features fused into left image features (early)
  Stage 2 - Disparity prior Gaussian bias on cost-volume logits (mid)
  Stage 3 - Learned blend of stereo soft-argmax and depth prior for GRU init (late)

The Inbolt dataset provides:
  - realsense/{idx}/mono0.png        : left IR image  (uint8, 480x640)
  - realsense/{idx}/mono1.png        : right IR image (uint8, 480x640)
  - realsense/{idx}/depthmap_mm.png  : RealSense depth in mm  ← fusion input
  - zivid/{idx}/depthmap_mm.png      : GT depth in mm (Zivid scanner)

Freezing strategy:
  Frozen    : EdgeNeXt backbone (stereo.feature.stem, stereo.feature.stages)
  Full LR   : DepthEncoder, DepthFusionModule, DepthInitBlend, depth_sigma, depth_prior_scale
  0.1x LR   : all other trainable stereo parameters

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_inbolt_depthrs.py
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
OUT_PATH    = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_depthrs.pth'

BF            = 50.102706998586 * 385.509887695312   # focal_px * baseline_mm
FOCAL_PX      = 385.509887695312
BASELINE_MM   = 50.102706998586

DEPTH_NORM_M  = 5.0    # RealSense depth clipped and normalised to [0, 1] over 0–5 m

EPOCHS        = 120
LR            = 2e-5
ITERS         = 8
GAMMA         = 0.9
TRAIN_RATIO   = 0.75
SPLIT_SEED    = 0


# ── depth fusion modules ──────────────────────────────────────────────────────

class DepthEncoder(nn.Module):
    """4-scale CNN encoder for a (normalised depth + validity) 2-channel input."""
    depth_chans = [32, 64, 96, 128]

    def __init__(self):
        super().__init__()
        self.stem   = BasicConv(2,   32,  kernel_size=3, stride=2, padding=1, bn=True, relu=True)
        self.stage1 = BasicConv(32,  32,  kernel_size=3, stride=2, padding=1, bn=True, relu=True)
        self.stage2 = BasicConv(32,  64,  kernel_size=3, stride=2, padding=1, bn=True, relu=True)
        self.stage3 = BasicConv(64,  96,  kernel_size=3, stride=2, padding=1, bn=True, relu=True)
        self.stage4 = BasicConv(96,  128, kernel_size=3, stride=2, padding=1, bn=True, relu=True)

    def forward(self, x):
        # x: (B, 2, H, W) — ch0: normalised depth, ch1: validity mask
        validity = x[:, 1:2]  # (B, 1, H, W)

        # build per-scale validity masks via 2x2 max-pool chaining
        v2  = F.max_pool2d(validity, kernel_size=2, stride=2)   # H/2
        v4  = F.max_pool2d(v2,  kernel_size=2, stride=2)        # H/4
        v8  = F.max_pool2d(v4,  kernel_size=2, stride=2)        # H/8
        v16 = F.max_pool2d(v8,  kernel_size=2, stride=2)        # H/16
        v32 = F.max_pool2d(v16, kernel_size=2, stride=2)        # H/32

        x2  = self.stem(x)              # (B, 32,  H/2,  W/2)
        d4  = self.stage1(x2)  * v4    # (B, 32,  H/4,  W/4)
        d8  = self.stage2(d4)  * v8    # (B, 64,  H/8,  W/8)
        d16 = self.stage3(d8)  * v16   # (B, 96,  H/16, W/16)
        d32 = self.stage4(d16) * v32   # (B, 128, H/32, W/32)

        return [d4, d8, d16, d32]


class DepthFusionModule(nn.Module):
    """Per-scale zero-init residual projection that adds depth features into left RGB features."""

    def __init__(self, feat_dims):
        super().__init__()
        dc = DepthEncoder.depth_chans   # [32, 64, 96, 128]
        self.proj_4  = nn.Conv2d(dc[0], feat_dims[0], 1)
        self.proj_8  = nn.Conv2d(dc[1], feat_dims[1], 1)
        self.proj_16 = nn.Conv2d(dc[2], feat_dims[2], 1)
        self.proj_32 = nn.Conv2d(dc[3], feat_dims[3], 1)
        for proj in [self.proj_4, self.proj_8, self.proj_16, self.proj_32]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, left_feats, depth_feats):
        projs = [self.proj_4, self.proj_8, self.proj_16, self.proj_32]
        return [left_feats[i] + projs[i](depth_feats[i]) for i in range(4)]


class DepthInitBlend(nn.Module):
    """3-layer CNN that blends stereo soft-argmax and depth prior for GRU init."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, padding=1, bn=True, relu=True),
            BasicConv(16, 16, kernel_size=3, padding=1, bn=True, relu=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, stereo_init, disp_prior, validity):
        # all inputs: (B, 1, H/4, W/4)
        x = torch.cat([stereo_init, disp_prior, validity], dim=1)  # (B, 3, H/4, W/4)
        blend_w = torch.sigmoid(self.net(x))
        blend_w = blend_w * validity                                # 0 where depth invalid
        return blend_w * disp_prior + (1.0 - blend_w) * stereo_init


# ── model wrapper ─────────────────────────────────────────────────────────────

class FastFoundationStereoDepthRS(nn.Module):
    """
    Wraps a pretrained FastFoundationStereo and adds RealSense depth fusion
    at three stages of the pipeline.

    The wrapped stereo model is stored as self.stereo; all new modules live
    directly on this wrapper so parameter groups are easy to separate.
    """

    def __init__(self, stereo_model):
        super().__init__()
        self.stereo = stereo_model
        self.args   = stereo_model.args

        feat_dims = stereo_model.feature.d_out   # [224, 192, 320, 304] for vitl
        self.depth_encoder    = DepthEncoder()
        self.depth_fusion     = DepthFusionModule(feat_dims)
        self.depth_init_blend = DepthInitBlend()

        # Stage-2 learned scalars: prior sharpness (σ) and weight (α)
        self.depth_sigma       = nn.Parameter(torch.tensor(4.0))   # disparity units at 1/4 scale
        self.depth_prior_scale = nn.Parameter(torch.tensor(0.1))   # starts small → near-stereo behaviour

    # ------------------------------------------------------------------
    def _preprocess_depth(self, depth_rs_mm):
        """
        depth_rs_mm: (B, 1, H, W) float32, RealSense depth in millimetres.
        Returns:
          depth_rs_m   (B, 1, H, W) — depth in metres (invalid pixels = 0)
          validity     (B, 1, H, W) — 1 where depth is valid, else 0
          enc_input    (B, 2, H, W) — [normalised_depth, validity] for DepthEncoder
        """
        validity    = ((depth_rs_mm > 0) & torch.isfinite(depth_rs_mm)).float()
        depth_rs_m  = (depth_rs_mm / 1000.0) * validity
        d_norm      = depth_rs_m.clamp(0, DEPTH_NORM_M) / DEPTH_NORM_M
        enc_input   = torch.cat([d_norm, validity], dim=1)
        return depth_rs_m, validity, enc_input

    # ------------------------------------------------------------------
    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        return self.stereo.upsample_disp(disp, mask_feat_4, stem_2x)

    # ------------------------------------------------------------------
    def forward(
        self,
        image1,
        image2,
        depth_rs_mm=None,
        focal=FOCAL_PX,
        baseline_mm=BASELINE_MM,
        iters=12,
        test_mode=False,
        low_memory=False,
        optimize_build_volume='pytorch1',
    ):
        s = self.stereo   # shorthand
        B, C, H, W = image1.shape
        low_memory  = low_memory or self.args.get('low_memory', False)

        image1 = normalize_image(image1)
        image2 = normalize_image(image2)

        with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):

            # ── feature extraction ────────────────────────────────────
            out            = s.feature(torch.cat([image1, image2], dim=0))
            features_left  = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x        = s.stem_2(image1)

            # ── Stage 1: depth feature fusion ─────────────────────────
            disp_prior_14 = None
            validity_14   = None

            if depth_rs_mm is not None:
                depth_rs_m, validity, enc_input = self._preprocess_depth(depth_rs_mm)
                depth_feats    = self.depth_encoder(enc_input)
                features_left  = self.depth_fusion(features_left, depth_feats)

            # ── cost volume (unchanged) ───────────────────────────────
            gwc_volume = build_gwc_volume_optimized_pytorch1(
                features_left[0], features_right[0],
                self.args.max_disp // 4, s.cv_group,
                normalize=self.args.normalize,
            )
            left_tmp       = s.proj_cmb(features_left[0])
            right_tmp      = s.proj_cmb(features_right[0])
            concat_volume  = build_concat_volume_optimized_pytorch1(left_tmp, right_tmp, maxdisp=self.args.max_disp // 4)
            del left_tmp, right_tmp

            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del concat_volume, gwc_volume

            comb_volume = s.corr_stem(comb_volume)
            comb_volume = s.corr_feature_att(comb_volume, features_left[0])
            comb_volume = s.cost_agg(comb_volume, features_left)

            # ── Stage 2: disparity prior bias on logits ───────────────
            logits = s.classifier(comb_volume).squeeze(1)   # (B, D, H/4, W/4)

            if depth_rs_mm is not None:
                D = logits.shape[1]
                # Use torch.where so invalid pixels get disp_prior=0.
                # depth_rs_m is in metres; baseline_mm is in mm → divide by 1000
                # for consistent units: focal(px) * baseline_m(m) / depth_m(m) = px.
                disp_prior    = torch.where(
                    validity > 0.5,
                    (focal * baseline_mm / 1000.0) / depth_rs_m.clamp(min=1e-3),
                    torch.zeros_like(depth_rs_m),
                )
                disp_prior_14 = F.interpolate(disp_prior / 4.0, size=(H // 4, W // 4), mode='nearest')
                validity_14   = F.interpolate(validity,          size=(H // 4, W // 4), mode='nearest')

                d_idx      = torch.arange(D, device=logits.device, dtype=logits.dtype).view(1, D, 1, 1)
                sigma      = self.depth_sigma.abs().clamp(min=0.1)
                prior_bias = -0.5 * ((d_idx - disp_prior_14) / sigma) ** 2
                prior_bias = prior_bias * validity_14      # flat where depth invalid

                logits = logits + self.depth_prior_scale * prior_bias

            prob        = F.softmax(logits, dim=1)
            stereo_init = disparity_regression(prob, self.args.max_disp // 4)  # (B,1,H/4,W/4)

            # ── Stage 3: GRU init blend ───────────────────────────────
            if depth_rs_mm is not None:
                init_disp = self.depth_init_blend(stereo_init, disp_prior_14, validity_14)
            else:
                init_disp = stereo_init

            # ── context network (unchanged) ───────────────────────────
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

        # ── GRU iterations (unchanged) ────────────────────────────────
        for itr in range(iters):
            disp     = disp.detach()
            geo_feat = geo_fn(disp, coords, dx=s.dx, low_memory=low_memory)

            with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
                net_list, mask_feat_4, delta_disp = s.update_block(
                    net_list, inp_list, geo_feat.to(s.dtype), disp, att
                )

            disp = disp + delta_disp.to(s.dtype)
            if test_mode and itr < iters - 1:
                continue

            disp_up = self.upsample_disp(disp.to(s.dtype), mask_feat_4.to(s.dtype), stem_2x.to(s.dtype))
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        return init_disp, disp_preds


# ── dataset ───────────────────────────────────────────────────────────────────

class InboltDepthDataset(Dataset):
    """
    Like InboltDataset but also returns the RealSense depth map (mm) as a
    fourth tensor — used as the depth fusion input during training.
    """

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
        depth_zivid = data['depth_zivid']   # GT float32 mm (Zivid resolution)
        depth_rs    = data['depth_rs']       # RealSense float32 mm (model input)

        h, w = left.shape[:2]

        if depth_zivid.shape != (h, w):
            depth_zivid = cv2.resize(depth_zivid, (w, h), interpolation=cv2.INTER_NEAREST)
        if depth_rs.shape != (h, w):
            depth_rs = cv2.resize(depth_rs, (w, h), interpolation=cv2.INTER_NEAREST)

        # IR uint8 → float [0, 255], replicated to 3-channel pseudo-RGB
        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)
        right = np.stack([right, right, right], axis=-1)

        # Zivid GT depth (mm) → disparity (pixels)
        disp  = np.zeros_like(depth_zivid, dtype=np.float32)
        valid = depth_zivid > 0
        disp[valid] = BF / depth_zivid[valid]

        left_t     = torch.from_numpy(left).permute(2, 0, 1).float()      # (3, H, W)
        right_t    = torch.from_numpy(right).permute(2, 0, 1).float()     # (3, H, W)
        disp_t     = torch.from_numpy(disp).unsqueeze(0).float()          # (1, H, W)
        valid_t    = torch.from_numpy(valid).unsqueeze(0)                  # (1, H, W) bool
        depth_rs_t = torch.from_numpy(depth_rs).unsqueeze(0).float()      # (1, H, W) mm

        return left_t, right_t, disp_t, valid_t, depth_rs_t


# ── loss ──────────────────────────────────────────────────────────────────────

def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted sum of smooth-L1 losses over GRU iterations."""
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
    """Evaluate average sequence loss over a dataloader (no gradient updates)."""
    if len(dataloader) == 0:
        return float('nan')

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for left, right, disp_gt, valid, depth_rs in dataloader:
            left, right       = left.cuda(), right.cuda()
            disp_gt, valid    = disp_gt.cuda(), valid.cuda()
            depth_rs          = depth_rs.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p, depth_rs_p = padder.pad(left, right, depth_rs)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, depth_rs_mm=depth_rs_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    # ── load and wrap pretrained stereo model ─────────────────────────
    logging.info(f"Loading base stereo model from {MODEL_PATH}")
    stereo_model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)
    model = FastFoundationStereoDepthRS(stereo_model).cuda()
    logging.info("Model wrapped with depth fusion modules.")

    # ── freeze EdgeNeXt backbone ──────────────────────────────────────
    for param in model.stereo.feature.stem.parameters():
        param.requires_grad = False
    for param in model.stereo.feature.stages.parameters():
        param.requires_grad = False
    logging.info("EdgeNeXt backbone (stem + stages) frozen.")

    # ── two-group optimizer: new modules at full LR, rest at 0.1× LR ─
    new_params = (
        list(model.depth_encoder.parameters())    +
        list(model.depth_fusion.parameters())     +
        list(model.depth_init_blend.parameters()) +
        [model.depth_sigma, model.depth_prior_scale]
    )
    new_param_ids = {id(p) for p in new_params}
    finetune_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in new_param_ids
    ]

    trainable_new      = sum(p.numel() for p in new_params)
    trainable_finetune = sum(p.numel() for p in finetune_params)
    total              = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Parameters — new (full LR): {trainable_new:,}  "
        f"fine-tune (0.1× LR): {trainable_finetune:,}  "
        f"frozen: {total - trainable_new - trainable_finetune:,}  "
        f"total: {total:,}"
    )

    optimizer = torch.optim.AdamW(
        [
            {'params': new_params,       'lr': LR},
            {'params': finetune_params,  'lr': LR * 0.1},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── dataset and splits ────────────────────────────────────────────
    dataset = InboltDepthDataset(INBOLT_DIR)
    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError(f"Need at least 2 samples for train/test split, got {n_total}.")

    n_train = min(max(1, int(round(TRAIN_RATIO * n_total))), n_total - 1)
    n_test  = n_total - n_train

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0)

    logging.info(
        f"Split (seed={SPLIT_SEED}): total={n_total}, "
        f"train={len(train_set)} ({100.0*len(train_set)/n_total:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total:.1f}%)"
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
            left_p, right_p, depth_rs_p = padder.pad(left, right, depth_rs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, depth_rs_mm=depth_rs_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss       = epoch_loss / len(train_loader)
        train_eval_error = evaluate_split_loss(model, train_loader)
        test_eval_error  = evaluate_split_loss(model, test_loader)

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"train_eval={train_eval_error:.4f}  "
            f"test_eval={test_eval_error:.4f}  "
            f"depth_sigma={model.depth_sigma.item():.3f}  "
            f"depth_prior_scale={model.depth_prior_scale.item():.4f}"
        )

        if test_eval_error < best_loss:
            best_loss = test_eval_error
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
