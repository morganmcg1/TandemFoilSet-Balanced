# lr=5e-5 on n_layers=3+slice_num=16+epochs=36: lower LR bound

## Hypothesis

Tests the lower LR bound at the new best stack (slice_num=16+n_layers=3+epochs=36). LR axis at slice_num=16: lr=5e-5 (this), lr=1e-4 (baseline), lr=1.5e-4 (alphonse #2431 in flight), lr=2e-4 (closed, +4.4% loss). With T_max=36 cosine, lower lr means smaller sign steps but gentler trajectory — may help OOD generalization.

## Instructions

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr-5e-5-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
