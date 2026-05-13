# surf_weight=5 on n_layers=3+slice_num=32: vol-gradient mechanism on new best stack

## Hypothesis

Vol-gradient mechanism (sw lower → L1 gradient shifts to volume → richer volume features → better surface via shared encoder) was a major historic win (sw=5: −9.0% val at n_layers=6, PR #1836). Never tested on n_layers=3 stack.

sw axis bracket so far on compact stacks: sw=10 baseline, sw=15 neutral (above), sw=5 untested (below).

Pairs with askeladd's sw=2 (PR #2248) — together bracket the mechanism cleanly.

## Instructions

Change ONLY `--surf_weight` from 10 to 5. Same config as PR #2107 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name surf-weight-5-nlayers3-slicenum32 \
  --epochs 27 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 5 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test `mae_surf_p` vs baseline (39.143/33.571)
2. Per-split `mae_vol_p` (vol-gradient mechanism diagnostic)
3. Per-epoch wall-clock, best epoch, total
4. Parameter count, peak memory

## Baseline (PR #2107)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.405 | 35.977 |
| geom_camber_rc | 51.895 | 47.136 |
| geom_camber_cruise | 22.756 | 19.101 |
| re_rand | 41.517 | 32.070 |
| **avg** | **39.143** | **33.571** |
