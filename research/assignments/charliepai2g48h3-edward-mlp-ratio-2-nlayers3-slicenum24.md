# mlp_ratio=2 on n_layers=3+slice_num=24+epochs=33: test if lighter FFN helps at compact depth

## Hypothesis

PR #2278 showed mlp_ratio=6 **hurts** at n_layers=3 (+5.4% val). Edward's own analysis: "at n_layers=3, attention/aggregation is the bottleneck, not per-token MLP capacity." If that's right, **reducing** mlp_ratio should:
- Reallocate parameters away from FFN toward the rest of the model
- Slightly lower per-epoch time (smaller FFN matmuls) → potentially 1-2 more epochs in budget
- Not regress: if the bottleneck is attention, the FFN is oversized already at mlp_ratio=4

This is a code change to train.py line 435: `mlp_ratio=4` → `mlp_ratio=2`.

Historical context: mlp_ratio=2 was tested previously (DEAD ENDS: "+9.95%" worse) but that was on a different, deeper stack. At n_layers=3+slice_num=24, the capacity dynamics differ.

## Instructions

Modify `train.py` **line 435**: change `mlp_ratio=4` → `mlp_ratio=2`.

Use the new baseline stack (n_layers=3+slice_num=24+epochs=33):

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name mlp-ratio-2-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24
```

## Reporting

1. **Epoch 1 wall-clock** — expect slightly faster than ~53.7s (smaller FFN)
2. Per-split val/test mae_surf_p vs NEW baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Best epoch (does earlier/faster convergence appear vs mlp_ratio=4?)
5. Parameter count (expected ~480K, lighter FFN), peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
