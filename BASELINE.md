# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-28 (after PR #344 merged)

## Current best — Round 0, PR #344 (edward H2 warmup-cosine, Run C)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **120.97** |
| `test_avg/mae_surf_p` | **109.92** |
| `test/test_single_in_dist/mae_surf_p` | 127.09 |
| `test/test_geom_camber_rc/mae_surf_p` | 123.58 |
| `test/test_geom_camber_cruise/mae_surf_p` | 81.16 |
| `test/test_re_rand/mae_surf_p` | 107.83 |
| `test_avg/mae_surf_Ux` | 1.96 |
| `test_avg/mae_surf_Uy` | 0.83 |
| `test_avg/mae_vol_p` | 110.97 |

- **W&B run:** [`rua9xrca`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r4/runs/rua9xrca) (`willowpai2d4-edward/h2-warmup-cosine-25ep-lr7e-4`)
- **Best epoch:** 13 of 14 (run hit 30-min wall clock at epoch 14)

## What changed from vanilla

The merged code on `icml-appendix-willow-pai2d-r4` now includes:

1. **Linear warmup + per-step cosine-to-zero schedule** (`LambdaLR`), replacing per-epoch `CosineAnnealingLR`. Default `--warmup_frac 0.05` (5% of total steps).
2. **Defensive `nan_to_num` in `evaluate_split`** — filters samples with non-finite ground truth (e.g., `test_geom_camber_cruise` sample 20 has `-inf` in pressure GT) and zeros out non-finite predictions before metric accumulation. Prevents NaN-poisoning the test averages.

## Recommended training command (reproduces current best)

```bash
cd target/ && python train.py \
    --agent <student-name> \
    --epochs 25 \
    --lr 7e-4 \
    --wandb_name "<student-name>/<experiment-tag>"
```

Note: at `batch_size=4` only ~14 epochs fit in the 30-min wall clock. The Run C config uses `--epochs 25 --lr 7e-4` deliberately so the cosine descends meaningfully (rather than `--epochs 50` which leaves lr near peak when timeout hits). Run A baseline (`--epochs 50 --lr 5e-4`) gives val_avg/mae_surf_p=125.17 — fold the 25ep/7e-4 config into experiment instructions when proposing variants.

## Setup recap

| Setting | Value |
|---------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1M params) |
| Optimizer | AdamW, weight_decay=1e-4 |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10 |
| Schedule | Linear warmup (5%) + cosine-to-zero, per-step (`LambdaLR`) |
| Epochs (default) | 50, capped by `SENPAI_TIMEOUT_MINUTES=30` (~14 actually fit) |
| Primary metric | `val_avg/mae_surf_p` (lower is better) |
| Paper metric | `test_avg/mae_surf_p` |

## Validation/test splits

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains
