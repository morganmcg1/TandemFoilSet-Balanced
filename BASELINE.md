# Baseline Metrics

## Current Baseline — PR #1473 (huber-relative-l2-compound)

**val_avg/mae_surf_p = 89.3940** (epoch 14 / 14 completed in 30-min cap) — **-0.24% vs previous 89.6121**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`
- `batch_size=4`, `surf_weight=10.0`, **Huber(δ=0.1) applied to relative-L2 normalized residuals**
- ~14 epochs in ~30 min
- Loss: `huber_relative_l2` — Huber on per-sample energy-normalized residuals (δ=0.1 in normalized space)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 109.01 |
| val_geom_camber_rc | 101.19 |
| val_geom_camber_cruise | **66.36** |
| val_re_rand | 81.02 |
| **val_avg** | **89.3940** |

**Test (all 4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 98.51 |
| test_geom_camber_rc | 88.12 |
| test_geom_camber_cruise | 54.80 |
| test_re_rand | 76.97 |
| **test_avg** | **79.5993** |

**Artifact**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-211810/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# Huber(delta=0.1) on relative-L2, lr=1e-3, T_max=14, grad_clip=1.0 now defaults on this branch
```

**Key insight**: Huber(δ=0.1) in normalized residual space compounds cleanly with relative-L2. The L2-fraction trajectory (33%→63%) shows Huber remains genuinely active throughout training — the delta=0.1 in normalized space is well-placed for intra-sample outlier capping without collapsing to MSE early. Grad clip_frac dropped from 1.0 to 0.075 by epoch 14 (vs ~0.98 on rel-L2-only) — the compound loss is significantly smoother. Val still falling at epoch 14.

---

## Previous Baseline — PR #1460 (relative-l2-loss)

**val_avg/mae_surf_p = 89.6121** (epoch 14 / 14 completed in 30-min cap) — **-7.20% vs previous 96.5587**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`
- `batch_size=4`, `surf_weight=10.0`, **per-sample relative L2 loss** (`||pred-y||²/||y||²`)
- ~131s/epoch; 14 epochs in ~30 min
- Loss: per-sample relative L2 in normalized space (replaces MSE)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 109.07 |
| val_geom_camber_rc | 97.99 |
| val_geom_camber_cruise | **67.09** |
| val_re_rand | 84.29 |
| **val_avg** | **89.6121** |

**Test (4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 91.14 |
| test_geom_camber_rc | 85.89 |
| test_geom_camber_cruise | 56.35 |
| test_re_rand | 79.18 |
| **test_avg** | **78.14** |

**Artifact**: `models/model-charliepai2g24h1-fern-relative-l2-loss-20260512-200551/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# loss=relative_l2, lr=1e-3, T_max=14, grad_clip=1.0 are now the default Config values on this branch
```

**Key insight**: Relative L2 loss (`||pred-y||²/||y||²`) normalizes by sample energy, automatically down-weighting high-energy (extreme-value) samples and up-weighting low-energy ones. This is a better inductive bias than MSE for flows with large Re variation — the loss landscape is flatter and more homogeneous across splits. Val still falling at epoch 14 (95.94 → 93.35 → 89.61 in last 3 epochs); more epochs would help.

**Gradient diagnostic**: clip_frac fell to 0.984 at ep 14 (was 1.0 throughout on MSE baseline) — relative-L2 is producing smaller raw gradient norms. The loss surface is smoother.

---

## Previous Baseline — PR #1518 (higher-lr-cosine-14)

**val_avg/mae_surf_p = 96.5587** (epoch 14 / 14 completed in 30-min cap) — **-17.6% vs previous 117.17**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, **`CosineAnnealingLR(T_max=14)`** ← key addition (was T_max=50)
- `batch_size=4`, `surf_weight=10.0`, `grad_clip=1.0`, MSE loss in normalized space
- ~131s/epoch; 14 epochs in ~30 min
- NaN scoring fix included: y-sanitization in `train.py:evaluate_split` (test_avg now 4-split clean)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 108.58 |
| val_geom_camber_rc | 110.59 |
| val_geom_camber_cruise | **74.35** |
| val_re_rand | 92.71 |
| **val_avg** | **96.5587** |

**Test (all 4 splits, NaN-free — scoring fix now in branch):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 94.97 |
| test_geom_camber_rc | 99.77 |
| test_geom_camber_cruise | 61.86 |
| test_re_rand | 86.88 |
| **test_avg** | **85.87** |

**Artifact**: `models/model-charliepai2g24h1-thorfinn-higher-lr-cosine-14-20260512-191045/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# lr=1e-3, T_max=14, grad_clip=1.0 are now the default Config values on this branch
```

**Key insight**: With `grad_clip=1.0` bounding the effective step size, pushing lr from 5e-4 to 1e-3 gave faster convergence. Reducing T_max from 50 to 14 made the model actually reach its low-LR fine-tuning phase within the 14-epoch wall-clock budget — the model was still improving at epoch 14. Val still falling at epoch 14 (100.34 → 98.66 → 96.56); a slightly longer or flatter cosine tail may yield additional gains.

**Key diagnostic**: Val crossed the old 117.17 baseline at epoch 10, reaching 96.56 by epoch 14. Pre-clip norms still 23–66 / 288–740 max. Clipping fires on ~100% of batches.

---

## Previous Baseline — PR #1479 (grad-clip-1)

**val_avg/mae_surf_p = 117.17** (epoch 13 / 14 completed in 30-min cap)

- `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=50)`, `grad_clip=1.0`
- **Artifact**: `models/model-charliepai2g24h1-thorfinn-grad-clip-1-20260512-180544/metrics.jsonl`

---

## Update Log

| Date | PR | val_avg/mae_surf_p | test_avg | Notes |
|------|----|--------------------|---------|-------|
| 2026-05-12 | #1518 | **96.5587** | **85.87** (4-split) | Round 2 winner. lr=1e-3, T_max=14; 14 epochs / 30 min |
| 2026-05-12 | #1479 | 117.17 | 116.17 (3-split) | Round 1 winner. grad_clip=1.0; 14 epochs / 30 min |
