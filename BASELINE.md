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

No completed PR has established numerical baselines on this branch yet. Round 1 hypotheses each include a stock control run or compare against published Transolver behavior.
