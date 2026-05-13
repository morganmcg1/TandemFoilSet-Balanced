# lr=1.2e-4 on n_layers=3+slice_num=16+epochs=36: fine-grained LR probe

## Hypothesis

Fine-grained LR probe between confirmed trough (1e-4) and confirmed ceiling (1.5e-4 +7.3%) at slice_num=16. LR optimum shifts down with partition size — smaller slice → more conservative LR needed. Tests whether trough sits slightly above 1e-4.

## Instructions

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name lr-1p2e-4-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1.2e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

Wall-clock gate: if epoch 1 >52s, reduce to --epochs 34.

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
