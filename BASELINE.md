# Baseline — icml-appendix-willow-pai2i-24h-r5

This launch starts fresh on `icml-appendix-willow-pai2i-24h-r5` (branched from `icml-appendix-willow`). No prior runs from this launch have been measured yet; the first round of experiments will both probe individual moves AND establish a confirmed baseline number.

## Current baseline configuration (head of advisor branch)

Model: **Transolver** (~1.5M params)
- `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `space_dim=2`, `fun_dim=22` (X_DIM=24 minus the 2 spatial dims)

Training:
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Loss: MSE in normalized space, `total = vol_loss + surf_weight * surf_loss`
- No mixed precision, no grad clipping
- `epochs=50` (max), capped by `SENPAI_TIMEOUT_MINUTES=30`

Per-run limits enforced by the harness:
- `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap)
- `SENPAI_MAX_EPOCHS=50` (hard epoch cap)
- 1 GPU per student, 96 GB VRAM

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing metric is `test_avg/mae_surf_p`, computed at the end of training using the best-val checkpoint.

## Baseline metrics

Pending — no run on this branch has completed yet. The first finishing experiment in round 1 establishes the initial baseline figure for this track. Subsequent rounds update this file with the running best.
