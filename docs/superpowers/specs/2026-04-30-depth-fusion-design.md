# RealSense Depth Fusion — Design Spec

**Date:** 2026-04-30  
**Status:** Approved for implementation  
**Goal:** Fuse RealSense RGB-D depth into FastFoundationStereo at three pipeline stages to improve disparity accuracy, with no latency constraint and using fine-tuning of the pretrained model.

---

## 1. Overview

The modified model takes three inputs: `image1` (left RGB), `image2` (right RGB), and `depth` (RealSense depth map, single-channel metric in meters, hardware-registered to the left RGB camera). Depth is fused at three stages: early (feature level), mid (cost volume logits), and late (GRU initialization).

**Modified forward signature:**
```python
def forward(self, image1, image2, depth=None, focal=None, baseline=None, iters=12, ...)
```

When `depth=None`, all three stages are skipped and the model behaves identically to the original. No existing code paths are changed.

**Architecture diagram:**
```
INPUTS
  image1 (B,3,H,W) ──┐
  image2 (B,3,H,W) ──┤
  depth  (B,1,H,W) ──┼──────────────────────────────────────────────────┐
  focal, baseline    │                                                   │
                     │                                                   │
STAGE 1 — EARLY FEATURE FUSION                                         │
                     │                                                   │
  image1+2 ──► Feature (EdgeNeXt)      depth ──► DepthEncoder [NEW]    │
               [x4,x8,x16,x32]                   [d4,d8,d16,d32]       │
                    │                                    │               │
                    └──── DepthFusionModule [NEW] ───────┘               │
                         (concat + 1×1 conv per scale)                  │
                         → fused_left[0..3]                             │
                                                                        │
COST VOLUME (unchanged)                                                 │
                                                                        │
  fused_left[0] + features_right[0]                                    │
       ├── GWC volume                                                   │
       └── Concat volume                                                │
                │                                                       │
           corr_stem → corr_feature_att                                 │
           hourglass (3D, FeatureAtt + CostVolumeDisparityAttention)   │
           classifier → logits (B, D, H/4, W/4)                        │
                │                                                       │
STAGE 2 — DISPARITY PRIOR ON LOGITS                                    │
                │                                                       │
                │    depth ──► disp_prior = focal·baseline/depth ◄──────┘
                │                  │
                │    prior_bias = Gaussian(μ=disp_prior, σ=σ_learned)
                │                  │  (zeroed where depth is invalid)
                └──► logits + α·prior_bias   [NEW α, σ learnable]
                              │
                     softmax → prob → soft-argmax → stereo_init
                                                          │
STAGE 3 — GRU INITIALIZATION BLEND                       │
                                                          │
  disp_prior_14 ──┐                                       │
  validity_mask ──┼──► DepthInitBlend [NEW] ──► blend_w ─┤
  stereo_init ────┘    (3-layer CNN → sigmoid)            │
                                                          │
  init_disp = blend_w·disp_prior + (1-blend_w)·stereo_init ◄──────────┘

GRU ITERATIVE REFINEMENT (unchanged internals)
  for itr in range(iters):
      geo_feat = Combined_Geo_Encoding_Volume(disp)
      net_list, mask_feat_4, delta_disp = update_block(...)
      disp = disp + delta_disp
      disp_up = upsample_disp(disp, mask_feat_4, stem_2x)

OUTPUT: final disparity (B,1,H,W)
```

---

## 2. Depth Preprocessing

Applied before `DepthEncoder` and before computing `disp_prior`. Shared logic, computed once in `forward`.

- **Validity mask:** `M = (depth > 0) & torch.isfinite(depth)` → (B,1,H,W) float, 1=valid, 0=invalid
- **Normalization:** `d_norm = (depth - DEPTH_MEAN) / DEPTH_STD` applied only to valid pixels; invalid pixels set to 0
- **Encoder input:** `torch.cat([d_norm, M], dim=1)` → (B,2,H,W)
- **`DEPTH_MEAN`, `DEPTH_STD`:** dataset statistics, stored as registered buffers (not trained)

---

## 3. Stage 1 — DepthEncoder + DepthFusionModule

**File:** `core/extractor.py`

### DepthEncoder

Lightweight 4-scale CNN built from existing `BasicConv` primitives.

```
Input: (B, 2, H, W)  [normalized depth + validity mask]

stem:   BasicConv(2  → 32, k=3, stride=2, BN+ReLU)   → (B, 32, H/2, W/2)
stage1: BasicConv(32 → 32, k=3, stride=2, BN+ReLU)   → depth_x4  (B, 32,  H/4,  W/4)
stage2: BasicConv(32 → 64, k=3, stride=2, BN+ReLU)   → depth_x8  (B, 64,  H/8,  W/8)
stage3: BasicConv(64 → 96, k=3, stride=2, BN+ReLU)   → depth_x16 (B, 96,  H/16, W/16)
stage4: BasicConv(96 → 128,k=3, stride=2, BN+ReLU)   → depth_x32 (B, 128, H/32, W/32)

depth_chans = [32, 64, 96, 128]
```

Each stage's output is multiplied by a downsampled validity mask so invalid regions produce near-zero features and do not corrupt fusion.

### DepthFusionModule

One 1×1 conv per scale. Input is concat of left feature and depth feature; output has the same shape as the left feature.

```
d_out = Feature.d_out = [96+vit_feat_dim, 192, 320, 304]

fusion_4:  Conv2d(d_out[0]+32,  d_out[0],  1)
fusion_8:  Conv2d(d_out[1]+64,  d_out[1],  1)
fusion_16: Conv2d(d_out[2]+96,  d_out[2],  1)
fusion_32: Conv2d(d_out[3]+128, d_out[3],  1)
```

Applied only to `features_left`. Right image features are not fused — RealSense depth is registered to the left camera only.

**In `FastFoundationStereo.forward`:**
```python
if depth is not None:
    depth_input = self.preprocess_depth(depth)          # normalize + validity mask
    depth_feats = self.depth_encoder(depth_input)
    features_left = self.depth_fusion(features_left, depth_feats)
```

---

## 4. Stage 2 — Disparity Prior on Logits

**File:** `core/foundation_stereo.py`

Convert depth to disparity at 1/4 resolution, build a per-pixel Gaussian prior over the disparity axis, and add it as an additive bias to the cost volume logits before softmax.

```python
# depth → disparity prior
disp_prior = (focal * baseline) / depth.clamp(min=1e-3)    # (B,1,H,W)
disp_prior_14 = F.interpolate(disp_prior / 4, (H//4, W//4), mode='nearest')
validity_14   = F.interpolate(M.float(), (H//4, W//4), mode='nearest')

# Gaussian bias over disparity axis
d_indices = torch.arange(D, device=disp_prior.device).view(1, D, 1, 1)
prior_bias = -0.5 * ((d_indices - disp_prior_14) / self.depth_sigma.abs()) ** 2
prior_bias = prior_bias * validity_14    # flat (zero) where depth is invalid

# Inject into logits
logits = self.classifier(comb_volume).squeeze(1)
if depth is not None:
    logits = logits + self.depth_prior_scale * prior_bias
prob = F.softmax(logits, dim=1)
init_disp_stereo = disparity_regression(prob, self.args.max_disp // 4)
```

**New parameters:**
- `self.depth_sigma`: `nn.Parameter(torch.tensor(4.0))` — learned std in disparity units at 1/4 scale
- `self.depth_prior_scale`: `nn.Parameter(torch.tensor(0.1))` — initialized small so training starts near original stereo behavior

---

## 5. Stage 3 — GRU Initialization Blend

**File:** `core/foundation_stereo.py`

A small 3-layer CNN learns per-pixel blend weights between the stereo soft-argmax estimate and the depth-derived disparity prior.

```python
class DepthInitBlend(nn.Module):
    # Input:  concat(stereo_init, disp_prior_14, validity_14) → (B, 3, H/4, W/4)
    # Layers: BasicConv(3→16), BasicConv(16→16), Conv2d(16→1)
    # Output: blend_w ∈ [0,1] via sigmoid, masked to 0 where validity_14=0
```

**In `FastFoundationStereo.forward`:**
```python
if depth is not None:
    blend_w = self.depth_init_blend(
        torch.cat([init_disp_stereo, disp_prior_14, validity_14], dim=1)
    ) * validity_14
    init_disp = blend_w * disp_prior_14 + (1 - blend_w) * init_disp_stereo
else:
    init_disp = init_disp_stereo
```

The blend learns *where* to trust depth (nearby objects, textureless regions) vs. stereo (edges, far range), giving the GRU a better warm start.

---

## 6. New Modules Summary

| Module | File | Est. Params | Role |
|---|---|---|---|
| `DepthEncoder` | `core/extractor.py` | ~200K | Encode raw depth to 4-scale features |
| `DepthFusionModule` | `core/extractor.py` | ~80K | Fuse depth features into left image features |
| `depth_sigma`, `depth_prior_scale` | `core/foundation_stereo.py` | 2 | Learned prior sharpness and weight |
| `DepthInitBlend` | `core/foundation_stereo.py` | ~5K | Per-pixel GRU init blend |

---

## 7. Training Strategy

**Frozen — do not update:**
- `self.feature.stem` and `self.feature.stages` (EdgeNeXt backbone, ~7M params)
- Cost volume builders (no parameters)

**New modules — train from scratch at full LR:**
- `DepthEncoder`
- `DepthFusionModule`
- `depth_sigma`, `depth_prior_scale`
- `DepthInitBlend`

**Existing modules — fine-tune at 0.1× base LR:**
- `self.feature.deconv32_16`, `deconv16_8`, `deconv8_4`, `conv4`
- `self.cost_agg` (hourglass, all layers)
- `self.update_block` (GRU)
- `self.cnet`, `self.context_zqr_convs`
- `self.classifier`

**Loss:** standard sequence disparity loss (smooth L1 on all GRU iteration outputs, same as original training). No changes needed.

**Dataset requirement:** Triplets of `(left_RGB, right_RGB, RealSense_depth, GT_disparity)`. GT disparity can come from LiDAR accumulation, structured-light ground truth at close range, or synthetic renders with simulated RealSense noise (holes at edges and reflective surfaces).

---

## 8. Depth Hole Handling Summary

Invalid pixels (holes, out-of-range readings) are handled consistently at each stage:

| Stage | Invalid pixel behavior |
|---|---|
| DepthEncoder | Features multiplied by downsampled validity mask → near-zero output |
| DepthFusionModule | 1×1 conv learns to ignore near-zero depth features |
| Stage 2 prior bias | `prior_bias * validity_14` → flat (zero additive bias) for invalid pixels |
| Stage 3 blend | `blend_w * validity_14` → blend_w=0, falls back to stereo init |
