# slice_num=18 on n_layers=3+epochs=36: partition gap 20→16

## Hypothesis

Fills the partition gap between slice_num=20 (val=36.854) and slice_num=16 (val=35.548, current baseline). Non-monotone trough confirmed at 16 (20>16<12). Testing 18 to pin the exact local optimum.

## Instructions

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name slicenum18-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 18
```

Note: if epoch 1 wall-clock >50s, reduce epochs to 34 to stay within 30-min cap.

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
