# Baseline Metrics

## Current Baseline — PR #1456 (bf16-amp + cosine-eta-min)

**val_avg/mae_surf_p = 36.8778** (epoch 16 of 17 completed in 30-min cap) — **-7.51% vs previous 39.8693**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=17, eta_min=1e-5)`** ← key addition (was T_max=14)
- **bf16 AMP enabled** ← key addition (was fp32)
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- **17 epochs in ~30 min** (vs 13 previously, ~+29% throughput from bf16)
- Peak GPU 32.98 GB (room for larger batch/model)

**Per-split val at best epoch (16):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| val_single_in_dist | **42.92** | −4.89 (−10.2%) |
| val_geom_camber_rc | **47.78** | −4.50 (−8.6%) |
| val_geom_camber_cruise | **18.60** | −2.29 (−11.0%) |
| val_re_rand | 38.21 | −0.28 (−0.7%) |
| **val_avg** | **36.8778** | **−2.99 (−7.51%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| test_single_in_dist | **42.15** | −3.80 (−8.3%) |
| test_geom_camber_rc | **42.69** | −3.64 (−7.9%) |
| test_geom_camber_cruise | **15.26** | −1.98 (−11.5%) |
| test_re_rand | **27.53** | −3.84 (−12.2%) |
| **test_avg** | **31.9058** | **−3.32 (−9.42%)** |

**Convergence trace**: 172.42 → 161.06 → 135.82 → 106.21 → 88.47 → 79.09 → 76.91 → 72.14 → 61.82 → 58.93 → 52.57 → 51.18 → 43.74 → 39.71 → 38.44 → **36.88** → 36.97 (ep 16 best; ep 17 drifts back +0.09 at LR floor).

**Grad clip / norm trace**: clip_frac smoothly decays 0.98 → 0.34 across 17 epochs. `huber_l2_frac` rises 0.42 → 0.86 — Huber actively capping outliers throughout.

**Artifact**: `models/model-charliepai2g24h1-alphonse-bf16-amp-cosine-eta-min-20260513-005955/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP + bf16 + CosineAnnealingLR(T_max=17, eta_min=1e-5) are now defaults on this branch
```

**Key insight**: bf16 AMP gives ~+29% throughput at zero quality cost. T_max=17 aligns the cosine tail with the new 17-epoch budget. ALL 8 splits (4 val + 4 test) improved — the broad win signals the model genuinely benefits from more epochs, not just a per-split tuning. eta_min=1e-5 keeps the late epoch usable. Peak memory only 33/96 GB — substantial headroom for larger batch or model.

---

## Previous Baseline — PR #1630 (cosine-eta-min)

**val_avg/mae_surf_p = 39.8693** (epoch 13 / 13 completed in 30-min cap) — **-5.97% vs previous 42.4015**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=14, eta_min=1e-5)`** ← key addition (was `eta_min=0`)
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- ~13 epochs in ~30 min; peak GPU 42.15 GB

**Per-split val at best epoch (13):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| val_single_in_dist | 47.81 | +1.72 (worse) |
| val_geom_camber_rc | **52.28** | −3.70 |
| val_geom_camber_cruise | **20.89** | −3.43 |
| val_re_rand | **38.49** | −4.73 |
| **val_avg** | **39.8693** | **−2.53 (−5.97%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| test_single_in_dist | 45.95 | +4.19 (worse) |
| test_geom_camber_rc | **46.33** | −1.77 |
| test_geom_camber_cruise | **17.24** | −2.55 |
| test_re_rand | **31.37** | −4.60 |
| **test_avg** | **35.2214** | **−1.18 (−3.24%)** |

**LR trace (epoch 13)**: LR at ep 13 ≈ 5.90e-5 (vs 4.95e-5 without eta_min floor — +19% relative LR at the critical final epoch). Every earlier epoch is essentially identical to eta_min=0.

**Convergence trace**: 167.84 → 134.09 → 107.90 → 97.98 → 84.20 → 81.79 → 76.84 → 62.82 → 52.34 → 50.44 → 45.42 → 42.63 → **39.87** (monotone descent, still falling).

**Artifact**: `models/model-charliepai2g24h1-tanjiro-cosine-eta-min-20260512-231540/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP + CosineAnnealingLR(T_max=14, eta_min=1e-5) are now defaults on this branch
```

**Key insight**: `eta_min=1e-5` prevents the cosine schedule from reaching near-zero at epoch 13 (the run's last/best epoch). The +19% relative LR boost at the terminal epoch is enough to squeeze 3 additional OOD improvement without any other change. Single-line, zero-risk compounding on SOAP. Val still monotone descending at ep 13.

---

## Previous Baseline — PR #1613 (soap-optimizer)

**val_avg/mae_surf_p = 42.4015** (epoch 13 / 13 completed in 30-min cap) — **-52.6% vs previous 89.3940**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals (inherited)
- ~13 epochs in ~30 min (slight SOAP overhead vs 14 epoch baseline)
- SOAP vendored as `soap.py` (upstream commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`)

**Per-split val at best epoch (13):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 46.09 |
| val_geom_camber_rc | 55.98 |
| val_geom_camber_cruise | **24.32** |
| val_re_rand | 43.22 |
| **val_avg** | **42.4015** |

**Test (all 4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 41.76 |
| test_geom_camber_rc | 48.10 |
| test_geom_camber_cruise | 19.79 |
| test_re_rand | 35.97 |
| **test_avg** | **36.4017** |

**Convergence trace**: 163.69 → 107.26 → 83.20 → 75.45 → 72.44 → 58.88 → 52.97 → 49.79 → 44.55 → 42.40 (still falling at ep 13).

**Grad norm trace**: 38.87 → 27.04 → 19.66 → 14.39 → 9.16. SOAP's Kronecker-factored preconditioner is producing 4.2× gradient norm reduction. Clip frac: 1.000 through ep 10, then 0.997 → 0.987 → 0.984.

**Artifact**: `models/model-charliepai2g24h1-thorfinn-soap-optimizer-20260512-220030/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP optimizer, Huber+rel-L2, lr=1e-3, T_max=14, grad_clip=1.0 are now defaults on this branch
```

**Key insight**: SOAP's Kronecker-factored quasi-Newton preconditioner transforms this problem. The 4.2× grad norm reduction means each step is much better conditioned — the optimizer is following the loss surface curvature rather than a noisy first-order gradient. This is the largest single improvement in the programme (+52.6%). Val is still falling at ep 13 — more epochs (bf16-amp) would compound significantly.

---

## Previous Baseline — PR #1473 (huber-relative-l2-compound)

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
