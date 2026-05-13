# n_layers=4+slice_num=16+epochs=27: depth-up vs budget trade-off

## Hypothesis

Tests depth-up at the new best partition. Counterpart to frieren's depth-down (n_layers=2) experiment. Per-epoch estimate at n_layers=4+slice_num=16: ~65-67s → 27 epochs in 30 min. Tests whether extra depth capacity outweighs reduced epoch budget.

## Instructions

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name nlayers4-slicenum16-epochs30 \
  --epochs 27 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 16
```

Wall-clock gate: if epoch 1 ≤60s use 29 epochs; 61-67s use 27 epochs; >67s use 25 epochs.

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
