# slice_num=12 on n_layers=3: floor probe for PhysicsAttention partition axis

## Hypothesis

PR #2229 (slice_num=24) and the broader partition sweep show per-epoch time scales roughly linearly with slice_num. At slice_num=12:
- Estimate ~46s/epoch → ~39 epochs in 30 min cap

But the question is whether PhysicsAttention can meaningfully operate at only 12 slices. Each "slice" is a learned partition of the input — too few slices means the attention lacks the granularity to differentiate local physics. This is a **floor probe**: we expect one of:
1. slice_num=12 wins → mechanism still not saturated, continue down
2. slice_num=12 neutral/loses → slice_num=24 is the representational floor

**This is a diagnostic run** — even if it loses, the loss tells us the exact floor of the slice axis.

## Instructions

Single flag change: `--slice_num 12`. Adjust epochs to fit budget using epoch 1 wall-clock (expect ~46s/epoch → use `--epochs 38`; if ep1 > 47s, reduce to `--epochs 36`).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name slicenum12-nlayers3 \
  --epochs 38 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 12
```

## Reporting

1. **Epoch 1 wall-clock** — critical for floor calibration
2. Per-split val/test mae_surf_p vs NEW baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p — watch for degradation on geom_camber_rc (hardest OOD split)
4. Per-epoch timing last 5 epochs, best epoch, total
5. Parameter count (~514K expected, slice_num doesn't change params), peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
