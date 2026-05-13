# slice_num=8 on n_layers=3: continue partition sweep to new floor

## Hypothesis

The partition-sweep mechanism continues. Trajectory so far:
- slice_num=32 → val=39.143 (PR #2107)
- slice_num=24 → val=37.366 (PR #2229)
- slice_num=12 → val=35.969 (PR #2351) ← current baseline, still descending at best_epoch=36/36

Tests slice_num=8: estimated ~44-47s/epoch → ~38-40 epochs in 30-min budget.

**Capacity risk:** If geom_camber_rc val mae_vol_p rises sharply vs baseline (~54.2), signal capacity collapse and report early.

## Instructions

Single flag change: `--slice_num 8` with `--epochs 38`. Wall-clock gate: reduce epochs if epoch 1 >47s.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name slicenum8-nlayers3 \
  --epochs 38 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 8
```

## Reporting

1. Epoch 1 wall-clock (expected 44-47s)
2. Per-split val/test mae_surf_p vs baseline (val=35.969 / test=30.265)
3. Per-split mae_vol_p — geom_camber_rc canary
4. Best epoch, total wall-clock
5. Parameter count (~513K), peak memory

## Baseline (PR #2351)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.308 | 33.241 |
| geom_camber_rc | 49.521 | 43.631 |
| geom_camber_cruise | 19.576 | 15.969 |
| re_rand | 38.470 | 28.220 |
| **avg** | **35.969** | **30.265** |
