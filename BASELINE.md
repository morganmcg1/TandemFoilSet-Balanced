# TandemFoilSet — Advisor Baseline

**Branch:** `icml-appendix-charlie-pai2i-24h-r4`
**Round:** charlie-pai2i-24h-r4 (24h budget, 8 students × 1 GPU)
**Primary metric:** `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface pressure MAE across 4 val splits
**Test metric:** `test_avg/mae_surf_p` (computed at end of every run from the best val checkpoint)

## Current best (this branch)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p`              | _pending — first round results will establish baseline_ | — |
| `val_single_in_dist/mae_surf_p`   | _pending_ | — |
| `val_geom_camber_rc/mae_surf_p`   | _pending_ | — |
| `val_geom_camber_cruise/mae_surf_p` | _pending_ | — |
| `val_re_rand/mae_surf_p`          | _pending_ | — |
| `test_avg/mae_surf_p`             | _pending_ | — |

## Baseline configuration

The branch inherits the unmodified `train.py`. Configuration:

- **Model:** `Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)` (~1M params)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** CosineAnnealingLR(T_max=epochs)
- **Batch:** 4
- **surf_weight:** 10.0
- **Epochs:** 50 (cap) / `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap
- **Sampler:** WeightedRandomSampler with domain-balanced weights
- **Splits dir:** `/mnt/new-pvc/datasets/tandemfoil/splits_v2`

### Reproduce command

```bash
cd target && python train.py --agent <student> --experiment_name "<student>/baseline"
```

## Notes for round-1 PRs

- No prior committed metrics on this branch — your run is informative regardless. Report `val_avg/mae_surf_p`, the 4 per-split surface-p MAEs, and `test_avg/mae_surf_p` from the best checkpoint.
- The H8 (EMA) PR is designed to also log unmodified-baseline live-weights metrics each epoch as a side product, so the first round will collectively establish the baseline.
- Update this file once we have a confirmed baseline number from a clean run on this branch.
