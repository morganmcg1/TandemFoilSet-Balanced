# lr=1.5e-4 on n_layers=3+slice_num=12+epochs=36: LR at new baseline stack

## Hypothesis

Current baseline (PR #2351) uses lr=1e-4 at slice_num=12+epochs=36 (T_max=36). Tests lr=1.5e-4 at the NEW baseline stack. Distinct from thorfinn's in-flight test (#2353 at slice_num=24).

With T_max=36 and more cosine room, higher LR may allow faster early descent while the longer annealing tail still converges well. Previous lr=2e-4 at slice_num=24 lost (+4.4%); lr=1.5e-4 is more conservative.

## Instructions

Single flag change: `--lr 1.5e-4`. Use `--epochs 36` to match new baseline.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name lr-1p5e-4-nlayers3-slicenum12 \
  --epochs 36 \
  --lr 1.5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 12
```

## Reporting

1. Per-split val/test mae_surf_p vs NEW baseline (val=35.969 / test=30.265)
2. Per-split mae_vol_p
3. Train loss epochs 1-5 — faster early descent or instability?
4. Best epoch, per-epoch wall-clock (~50.3s expected), peak memory

## Baseline (PR #2351)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.308 | 33.241 |
| geom_camber_rc | 49.521 | 43.631 |
| geom_camber_cruise | 19.576 | 15.969 |
| re_rand | 38.470 | 28.220 |
| **avg** | **35.969** | **30.265** |
