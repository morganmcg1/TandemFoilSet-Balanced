# n_layers=2+slice_num=16+epochs=46: depth reduction for more epoch budget

## Hypothesis

Applies the dominant epoch-budget mechanism: reduce n_layers=3→2 → faster per-epoch → more epochs (46 vs 36) → more cosine annealing room → better convergence. Estimated per-epoch at n_layers=2: ~35–40s → 46 epochs = ~29.5 min. n_layers=3→2 step mirrors the n_layers=4→3 win that was the biggest single step in this campaign (−8.58%, PR #2107).

Capacity risk: n_layers=2 has ~342K params vs 513K. May be under-capacity at slice_num=16 if attention specialization needs depth. But compact-depth direction has consistently won when budget is limiting.

## Instructions

Two flag changes: `--n_layers 2 --epochs 46`. Wall-clock gate: if epoch 1 >42s, reduce to `--epochs 40`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
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
