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
