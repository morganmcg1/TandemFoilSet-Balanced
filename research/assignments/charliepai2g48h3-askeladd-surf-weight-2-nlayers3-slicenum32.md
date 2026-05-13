# surf_weight=2 on n_layers=3+slice_num=32: bracket vol-gradient axis from below

## Hypothesis

Bracket the vol-gradient mechanism axis from far below default on current best stack. Pairs with fern's sw=5 (PR #2245) — together establish the sw curve at n_layers=3.

sw axis status: sw=2 never fully tested (#2109 closed stale); sw=5 untested at this depth; sw=10 baseline; sw=15 neutral.

If sw=2 wins (and fern's sw=5 wins): direction monotone, consider sw=1 next.
If sw=2 loses but sw=5 wins: optimum between, bracket converges.

## Instructions

Change ONLY `--surf_weight` from 10 to 2. Same config as PR #2107 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name surf-weight-2-nlayers3-slicenum32 \
  --epochs 27 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 2 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test `mae_surf_p` vs baseline (39.143/33.571)
2. **Per-split `mae_vol_p`** — should improve substantially at sw=2
3. Per-epoch wall-clock, best epoch
4. Parameter count, peak memory

## Baseline (PR #2107)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.405 | 35.977 |
| geom_camber_rc | 51.895 | 47.136 |
| geom_camber_cruise | 22.756 | 19.101 |
| re_rand | 41.517 | 32.070 |
| **avg** | **39.143** | **33.571** |
