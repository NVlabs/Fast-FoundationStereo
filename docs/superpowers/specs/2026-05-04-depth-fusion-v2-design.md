# RealSense Depth Fusion v2 — Design Spec

**Date:** 2026-05-04  
**Status:** Approved for implementation  
**Goal:** Fuse RealSense depth into FastFoundationStereo without DepthEncoder or DepthFusionModule; fully freeze feature extraction; add a new post-GRU full-resolution output blend stage.

---

## 1. Overview

v1 (`finetune_inbolt_depthrs.py`) performed feature-level fusion via a learned depth encoder and per-scale residual projections. Despite fixing all initialization bugs and unit errors, the 13-sample Inbolt dataset was too small for the new modules to learn from scratch, and the model matched but did not beat the stereo fine-tuned baseline.

v2 removes the heavy early-fusion modules entirely and instead acts only at the **cost volume** and **disparity output** stages, where the depth prior carries direct geometric meaning with no learned feature extraction required.

**Forward signature (unchanged externally):**
```python
def forward(self, image1, image2, depth_rs_mm=None, iters=12, test_mode=False, ...)
```

When `depth_rs_mm=None`, all depth stages are skipped and the model behaves identically to the wrapped pretrained stereo model.

---

## 2. Architecture Diagram

```
INPUTS
  left_IR     (B,3,H,W) ──────────────────────────────────────────────────────┐
  right_IR    (B,3,H,W) ──────────────────────────────────────────────────────┤
  depth_rs_mm (B,1,H,W) ──────────────────────────────────────────────────────┤
                                                                                │
FEATURE EXTRACTION  [FULLY FROZEN]                                             │
  left+right → Feature (EdgeNeXt backbone + FPN decoder)                      │
  → feat_left[x4,x8,x16,x32], feat_right[x4,x8,x16,x32]                     │
  → stem_2x   (for convex upsampling)                                         │
                                                                                │
DEPTH PREPROCESSING  [no learned params]                                       │
  validity M = (depth_rs_mm > 0) & isfinite(depth_rs_mm)                      │
  depth_rs_m = depth_rs_mm / 1000.0 · M                                       │
  disp_prior = where(M, focal·baseline_m / depth_rs_m, 0)  ← full-res px     │
  disp_prior_14 = F.interpolate(disp_prior / 4, H/4, W/4, 'nearest')         │
  validity_14   = F.interpolate(M, H/4, W/4, 'nearest')                       │
                                                                                │
COST VOLUME  [fine-tune 0.1× LR]                                               │
  feat_left[0] + feat_right[0]                                                 │
       ├── GWC volume                                                           │
       └── Concat volume                                                        │
               │                                                                │
          corr_stem → corr_feature_att → hourglass → logits (B,D,H/4,W/4)    │
                                                                                │
STAGE 2 — GAUSSIAN LOGIT PRIOR  [learnable σ, α — full LR]                   │
  d_idx = arange(D).view(1,D,1,1)                                              │
  prior_bias = −½·((d_idx − disp_prior_14) / σ.abs().clamp(0.1))²            │
  prior_bias = prior_bias · validity_14                                         │
  logits' = logits + α · prior_bias                                            │
  → prob = softmax(logits')                                                     │
  → stereo_init = disparity_regression(prob, max_disp//4)   (B,1,H/4,W/4)    │
                                                                                │
STAGE 3a — GRU INIT BLEND  [DepthInitBlend ~5K — full LR]                    │
  x = cat(stereo_init, disp_prior_14, validity_14)    (B,3,H/4,W/4)          │
  blend_w = sigmoid(3-layer CNN(x)) · validity_14    [bias=-5 at init → 0]   │
  init_disp = blend_w·disp_prior_14 + (1−blend_w)·stereo_init                │
                                                                                │
CONTEXT NETWORK + GRU ITERATIONS  [fine-tune 0.1× LR]                         │
  cnet(feat_left) → net_list, inp_list, att                                    │
  for itr in 0..iters-1:                                                       │
      geo_feat = Geo_Encoding_Volume(disp, coords)                             │
      Δdisp = update_block(net_list, inp_list, geo_feat, disp, att)           │
      disp += Δdisp                                                             │
      disp_up = upsample_disp(disp, mask_feat_4, stem_2x)     (B,1,H,W)      │
                                                                                │
STAGE 3b — OUTPUT BLEND  [DepthOutputBlend ~8K — full LR]  ← NEW            ←┘
  for each disp_up in disp_preds:
      x = cat(disp_up, disp_prior_full, validity)     (B,3,H,W)
      blend_w = sigmoid(3-layer CNN(x)) · validity    [bias=-5 at init → 0]
      disp_final = disp_up + blend_w·(disp_prior_full − disp_up)

OUTPUT: disp_final (B,1,H,W)
```

---

## 3. New Modules

### DepthInitBlend (Stage 3a, ~5 K params)
```
Input:  cat(stereo_init, disp_prior_14, validity_14)  (B,3,H/4,W/4)
Layers: BasicConv(3→16, k=3, BN+ReLU)
        BasicConv(16→16, k=3, BN+ReLU)
        Conv2d(16→1, k=1)                     ← weight=0, bias=−5 at init
Output: blend_w = sigmoid(net(x)) · validity_14
Return: blend_w·disp_prior_14 + (1−blend_w)·stereo_init
```

### DepthOutputBlend (Stage 3b, ~8 K params)  — NEW
```
Input:  cat(disp_stereo, disp_prior_full, validity)  (B,3,H,W)
Layers: BasicConv(3→32, k=3, BN+ReLU)
        BasicConv(32→32, k=3, BN+ReLU)
        Conv2d(32→1, k=1)                     ← weight=0, bias=−5 at init
Output: blend_w = sigmoid(net(x)) · validity
Return: disp_stereo + blend_w·(disp_prior_full − disp_stereo)   [residual]
```

Both modules start with near-zero blend weight (sigmoid(−5) ≈ 0.007), so the model is numerically identical to the wrapped pretrained stereo model at epoch 0. Depth fusion is learned gradually.

Applied in **training mode** to every GRU iteration prediction (8 predictions per step → 8 sequence-loss terms, all passing through Stage 3b). Applied in **test mode** only to the final upsampled disparity.

---

## 4. Depth Preprocessing

All depth-prior computations use a consistent unit system:

```python
depth_rs_m  = depth_rs_mm / 1000.0 · validity         # metres, invalid → 0
disp_prior  = where(validity, focal_px·baseline_m / depth_rs_m.clamp(1e-3), 0)
              # focal(px) · baseline(m) / depth(m) = pixels
```

No separate normalization step or `DEPTH_NORM_M` constant needed (no DepthEncoder).

---

## 5. Parameter Groups

| Group | Modules | LR |
|---|---|---|
| Frozen (no grad) | `stereo.feature` (all submodules), `stereo.stem_2` | 0 |
| New — full LR | `depth_init_blend`, `depth_output_blend`, `depth_sigma`, `depth_prior_scale` | 2e-5 |
| Fine-tune — 0.1× LR | All remaining trainable stereo params | 2e-6 |

New learnable params: `depth_sigma` (σ, 4.0), `depth_prior_scale` (α, 0.1), plus ~13 K CNN params.  
Fully frozen: entire `Feature` module (backbone + FPN decoder) + `stem_2`.

---

## 6. Zero-Init Guarantee

At initialization, with all new module weights zero and biases −5:

- **Stage 2**: `α = 0.1` → small Gaussian prior added to logits. The prior is correctly centered at the depth-derived disparity and has radius σ=4 (in 1/4-scale pixels ≈ 16 full-scale pixels), giving a gentle push toward the depth estimate without overriding the cost volume.
- **Stage 3a**: `blend_w ≈ 0.007` → `init_disp ≈ stereo_init`. GRU warm start is unchanged.
- **Stage 3b**: `blend_w ≈ 0.007` → `disp_final ≈ disp_up`. Output is unchanged.

Epoch 1 training loss should be < 1.0 (same order as the fine-tuned stereo baseline).

---

## 7. Key Differences from v1

| | v1 | v2 |
|---|---|---|
| DepthEncoder | ✓ (4-scale CNN) | ✗ removed |
| DepthFusionModule | ✓ (residual proj) | ✗ removed |
| Feature extraction frozen | stem+stages only | entire Feature + stem_2 |
| Stage 2 logit prior | ✓ | ✓ |
| Stage 3a GRU init blend | ✓ | ✓ |
| Stage 3b output blend | ✗ | ✓ NEW |
| New params | ~288 K | ~13 K |
| Fine-tune params | ~12.4 M | ~11.4 M |
