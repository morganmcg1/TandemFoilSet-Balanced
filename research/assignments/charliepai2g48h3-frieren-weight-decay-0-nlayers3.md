# weight_decay=0 on n_layers=3+slice_num=32+epochs=30: test if compact model needs regularization

## Hypothesis

WD axis was historically "flat" 1e-4 to 3e-2 (PR #1925) but WD=0 was never tested on the new compact stack (515K params, smallest in project history). At best_epoch=30/30 still descending under WD=1e-4, the model is underfitting — regularization may be slowing gradient updates unnecessarily.

If WD=0 wins: compound stack is underfit; future low-WD experiments become viable.
If WD=0 loses: WD=1e-4 is doing real regularization work; floor identified.

## Instructions

Single flag change: `--weight_decay 0`. Same config as PR #2228 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name weight-decay-0-nlayers3 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 0 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (38.270/32.470)
2. Per-split mae_vol_p
3. **Train loss trajectory** across all 30 epochs (diagnostic for overfit)
4. Per-epoch wall-clock, best epoch
5. Parameter count, peak memory

## Baseline (PR #2228)

| Split | val | test |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | 52.042 | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |
