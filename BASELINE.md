# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-15

## Current best

No winners merged yet on this branch. The first reproducible baseline run was
re-run by `willowpai2i48h2-askeladd` inside PR #3176 (the baseline-w1 reference arm).

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **136.8873** | run `07efagec` (best @ epoch 14) |
| `test_avg/mae_surf_p` | NaN (cruise GT contains inf — scoring.py bug) | run `07efagec` |
| `test_avg/mae_surf_p` (3 valid splits) | 137.6945 | run `07efagec` |

Per-split validation (best @ epoch 14):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 151.8490 |
| val_geom_camber_rc | 173.9127 |
| val_geom_camber_cruise | 101.4053 |
| val_re_rand | 120.3820 |

Per-split test (best ckpt):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 136.5218 |
| test_geom_camber_rc | 157.5912 |
| test_geom_camber_cruise | NaN (data/scoring.py bug — `inf * 0 = NaN`) |
| test_re_rand | 118.9706 |

W&B run: `07efagec` — `baseline-w1-ref`, wandb_group `pressure-channel-weight`.

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)`
- Loss: `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`, uniform channel weighting
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30` (wall clock typically binds first; ~14 epochs land before timeout)

## Reproduce

```bash
cd target/ && python train.py --wandb_name baseline --agent <student>
```

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
