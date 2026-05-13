# lr=1.5e-4 on n_layers=3+slice_num=24+epochs=33: LR retest at new baseline stack

## Hypothesis

fern (#2301) is testing lr=1.5e-4 on the OLD stack (n_layers=3+slice_num=32+epochs=30). But the baseline has now shifted to n_layers=3+slice_num=24+epochs=33. Even if fern's result is negative at slice_num=32, the LR optimum may differ on the new stack because:
- slice_num=24 changes the per-epoch compute profile
- epochs=33 with T_max=33 cosine has a different LR decay trajectory than epochs=30

This test answers: is lr=1.5e-4 beneficial specifically on the current best configuration (n_layers=3+slice_num=24)?

Single flag change: `--lr 1.5e-4`. Same config as PR #2229 otherwise.

## Instructions

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr-1p5e-4-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 1.5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24
```

## Reporting

1. Per-split val/test mae_surf_p vs NEW baseline (val=37.366 / test=31.371)
2. Per-split mae_vol_p
3. **Train loss epochs 1-5** (does lr=1.5e-4 cause early instability?)
4. Best epoch (earlier peak than lr=1e-4? or still final?)
5. Per-epoch wall-clock (~53.7s expected), parameter count, peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
