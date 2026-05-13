# TandemFoilSet Baseline

Track: `icml-appendix-willow-pai2g-48h-r5`

## Current baseline

Stock `train.py` on `icml-appendix-willow-pai2g-48h-r5` — Transolver with the following config:

- `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- `epochs=50` (capped by `SENPAI_TIMEOUT_MINUTES=30` per-run wall clock)
- AdamW + CosineAnnealingLR, MSE loss in normalized space, vol + 10·surf

**Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the 4 val splits).
**Paper-facing metric:** `test_avg/mae_surf_p` (computed at end of run from the best-val checkpoint).

## 2026-05-13 00:05 — PR #1689: fern Huber β=0.5 (tighter MAE alignment)

Merged. Smooth L1 / Huber loss transition point reduced from β=1.0 → β=0.5 in both the training inner loop and `evaluate_split`. At β=0.5 the quadratic region covers only `|x| < 0.5` (in normalized space, the near-zero small-error regime), while moderate errors (0.5–1.0 MAE range) now receive a linear (L1-like) gradient. This directly aligns with the MAE primary metric over the bulk of the loss density, where most surface-pressure normalized errors live. EMA shadow absorbs the L1 kink noise near zero.

**New best (lower is better):**

| Metric | Value | vs PR #1606 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **85.9197** | −6.43 (−6.96%) |
| `test_avg/mae_surf_p` | **76.5495** | −5.08 (−6.22%) |

**Per-split test (best-val checkpoint, epoch 17):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 88.0317 |
| `test_geom_camber_rc` | 85.4633 |
| `test_geom_camber_cruise` | 56.3982 |
| `test_re_rand` | 76.3047 |
| **test_avg** | **76.5495** |

- **All 4 splits improved** (in_dist −7.6%, camber_rc −7.0%, camber_cruise −3.9%, re_rand −5.3%)
- **EMA-vs-live gap preserved:** EMA val=85.92 vs live val=96.41 (−10.5 MAE)
- **Code change:** `beta=1.0` → `beta=0.5` in two `F.smooth_l1_loss(...)` calls (train loop + evaluate_split)
- **W&B run:** `liurnqyo`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`

## 2026-05-12 22:10 — PR #1606: fern EMA of model weights (decay=0.999)

Merged. EMA shadow copy of model parameters updated after every optimizer step (`ema = 0.999 * ema + 0.001 * model`). Val and test evaluation uses EMA weights instead of live weights. EMA lags during warmup but consistently outperforms the live model from epoch 9 onward; the gap widens late in training as cosine LR anneals but SGD noise persists.

**New best (lower is better):**

| Metric | Value | vs PR #1436 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **92.3452** | −4.14 (−4.3%) |
| `test_avg/mae_surf_p` | **81.6297** | −4.70 (−5.4%) |

**Per-split test (best-val checkpoint, epoch 17):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 95.2950 |
| `test_geom_camber_rc` | 91.9270 |
| `test_geom_camber_cruise` | 58.7160 |
| `test_re_rand` | 80.5810 |
| **test_avg** | **81.6297** |

- **EMA-vs-live diagnostic:** epoch 17 live model test=104.70 vs EMA test=81.63 — EMA shadow is +28% better than live weights at same step
- **Config change:** `copy.deepcopy(model)` EMA shadow with `requires_grad=False`; updated after each `optimizer.step()` on fp32 master weights; val+test eval use `ema_model`
- **W&B run:** `gdfynh7o`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`

## 2026-05-12 21:10 — PR #1436: fern Huber + bf16 (compound winner)

Merged. Smooth L1 / Huber loss (β=1.0) replaces MSE in both training and `evaluate_split`. Stacked on top of the alphonse bf16 baseline; effects compounded as predicted — Huber's loss-shape alignment with the MAE metric (linear tails for high-Re extreme p samples) + bf16's epoch budget (~18 vs ~14 fp32).

**New best (lower is better):**

| Metric | Value | vs PR #1419 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **96.4863** | −12.81 (−11.7%) |
| `test_avg/mae_surf_p` | **86.3326** | −11.33 (−11.6%) |

**Per-split val (epoch 16, best checkpoint):**

| Split | mae_surf_p |
|-------|----------:|
| `val_single_in_dist` | 112.8995 |
| `val_geom_camber_rc` | 106.9168 |
| `val_geom_camber_cruise` | 75.1834 |
| `val_re_rand` | 90.9454 |
| **val_avg** | **96.4863** |

**Per-split test (best-val checkpoint):**

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|----------:|------------:|------------:|----------:|
| `test_single_in_dist` | 101.2155 | 1.4049 | 0.6030 | 108.6379 |
| `test_geom_camber_rc` | 95.6042 | 1.9262 | 0.8326 | 106.1176 |
| `test_geom_camber_cruise` | 64.2155 | 1.0321 | 0.4469 | 63.5676 |
| `test_re_rand` | 84.2951 | 1.3881 | 0.6406 | 85.9693 |
| **test_avg** | **86.3326** | **1.4378** | **0.6308** | **91.0731** |

- **Config change:** `sq_err = F.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')` replaces `sq_err = (pred - y_norm) ** 2` in two locations (training inner loop and `evaluate_split`).
- **W&B run:** `kmwsz3i4`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
- All 4 test splits improved (vs alphonse): in_dist −12.75, camber_rc −10.10, camber_cruise −9.16, re_rand −13.32.

## 2026-05-12 20:00 — PR #1419: alphonse bf16 autocast (round-1 winner)

Merged. bf16 mixed-precision training (`torch.amp.autocast(dtype=torch.bfloat16)`) + scoring NaN workaround in `evaluate_split`. Both changes are now in the advisor branch and will propagate to all subsequent student PRs.

**New best (lower is better):**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.2937** |
| `test_avg/mae_surf_p` | **97.6659** |

**Per-split val (epoch 18, best checkpoint):**

| Split | mae_surf_p |
|-------|----------:|
| `val_single_in_dist` | 133.2714 |
| `val_geom_camber_rc` | 115.3895 |
| `val_geom_camber_cruise` | 87.8295 |
| `val_re_rand` | 100.6844 |
| **val_avg** | **109.2937** |

**Per-split test (best-val checkpoint):**

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|----------:|------------:|------------:|----------:|
| `test_single_in_dist` | 113.9645 | 1.5436 | 0.7415 | 120.6592 |
| `test_geom_camber_rc` | 105.7068 | 2.3467 | 0.9479 | 109.4459 |
| `test_geom_camber_cruise` | 73.3736 | 1.1906 | 0.5263 | 74.9999 |
| `test_re_rand` | 97.6189 | 1.6668 | 0.7685 | 100.6900 |
| **test_avg** | **97.6659** | **1.6869** | **0.7460** | **101.4488** |

- **Config change:** bf16 autocast wraps forward + loss; optimizer and eval in fp32. ~101 s/epoch → 18 epochs in 30 min vs ~11-12 epochs fp32.
- **Scoring fix:** `evaluate_split` now pre-masks non-finite GT samples and applies `nan_to_num(y)` before `accumulate_batch`, eliminating `NaN*0=NaN` from `.test_geom_camber_cruise_gt/000020.pt`.
- **W&B run:** `4hy79j91`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (bf16 autocast and NaN workaround are now in the merged train.py; no extra flags needed)

## 2026-05-13 02:00 — PR #1672: nezuko linear LR warmup 1 epoch v2

**New best — 5th compound improvement**

- **val_avg/mae_surf_p:** 85.0926 (↓ from 85.9197, −0.96%)
- **test_avg/mae_surf_p:** 75.5171 (↓ from 76.5495, −1.35%)

**Per-split test (all four improved):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 87.1000 |
| `test_geom_camber_rc` | 84.5765 |
| `test_geom_camber_cruise` | 55.4971 |
| `test_re_rand` | 74.8950 |

- **Config:** EMA decay=0.999, Huber β=0.5, bf16 autocast, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, dropout=0.0, LR warmup 1 epoch (start_factor=0.2→1.0 over 375 steps, T_max=10875)
- **Epochs:** 17 in 30 min (~110 s/epoch)
- **EMA−Live gap:** −9.87 at epoch 17 (EMA −9.87 vs baseline −10.49)
- **W&B run:** `1hn6ur4l`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (warmup is now merged into train.py defaults; no extra flags needed)

## 2026-05-13 02:10 — PR #1763: edward torch.compile

**New best — 6th compound improvement (massive throughput win)**

- **val_avg/mae_surf_p:** 71.4371 (↓ from 85.0926, −16.06%)
- **test_avg/mae_surf_p:** 62.5927 (↓ from 75.5171, −17.11%)

**Per-split test (all four improved dramatically):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 70.4261 |
| `test_geom_camber_rc` | 74.0859 |
| `test_geom_camber_cruise` | 44.5085 |
| `test_re_rand` | 61.3503 |

- **Config:** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, dropout=0.0, **torch.compile(model, dynamic=True, mode='default')**
- **Epochs:** **29 in 30.7 min** (~63 s/epoch steady state, +12 s compile warmup on epoch 1)
- **Speedup:** −44% per-epoch wall time vs no-compile; +12 epochs in budget (+71%)
- **Peak GPU memory:** 23.8 GB / 96 GB
- **EMA-vs-live gap:** −1.0 at epoch 29 (EMA 71.44, live 70.55 — both healthy)
- **W&B run:** `o6k5dj4g`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (torch.compile is now applied to the live model by default in train.py; dynamic=True handles variable mesh sizes; no extra flags needed)
- **Confounder noted:** `--epochs 30` makes cosine T_max=30 (vs implicit baseline T_max=50). Part of the gain may be from a more aggressive cosine schedule. Throughput component is clean either way.
