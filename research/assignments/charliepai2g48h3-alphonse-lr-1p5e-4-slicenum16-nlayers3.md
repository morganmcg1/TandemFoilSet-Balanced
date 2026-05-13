# lr=1.5e-4 × slice_num=16 on n_layers=3+epochs=36: compound LR+partition

## Hypothesis

Compound test: lr=1.5e-4 × slice_num=16 on the current best stack (n_layers=3+epochs=36).

Two independent signals:
1. **slice_num=16 just won** (PR #2348, val 35.969 → 35.548, −1.17%). Current best partition.
2. **lr=1.5e-4 showed positive signal** at slice_num=24 (thorfinn #2353 won vs old baseline). fern (#2409) testing same LR at slice_num=12; this arm tests it at slice_num=16.

Estimated compound gain: ~35.0–35.3 if both effects are additive.

## Instructions

Two flag changes from baseline: `--lr 1.5e-4 --slice_num 16`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name lr-1p5e-4-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1.5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs NEW baseline (val=35.548 / test=30.345)
2. Per-split mae_vol_p
3. Train loss epochs 1–5 — faster early descent or instability vs lr=1e-4?
4. Best epoch, total wall-clock, peak memory

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
