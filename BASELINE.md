# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-15

## Current best

No winners yet on this branch. The baseline is the unmodified Transolver from `train.py` at the head of the branch.

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | _to be established_ | Transolver default |
| `test_avg/mae_surf_p` | _to be established_ | Transolver default |

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)`
- Loss: `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

## Reproduce

```bash
cd target/ && python train.py --wandb_name baseline --agent <student>
```

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
