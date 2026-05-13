# Linear warmup (2 ep) + cosine on n_layers=3: retest mechanism at long budget

## Hypothesis

Lion warmup was rejected at T_max=12 (PR #1790) because the 2-epoch cost = 17% of budget. At T_max=30, 2-epoch warmup = 6.7% of budget — much less disruptive. Lion's sign-update is sensitive to early noise; warmup might let the model find a better basin.

PR #2228 showed strong late-stage descent (41.89→38.27 in epochs 23-30). Early epochs spent at full LR=1e-4 with Lion's aggressive sign updates may have left optimization quality on the table.

## Instructions

Code change at train.py line 445: replace single cosine with SequentialLR(LinearLR + CosineAnnealingLR). See PR body for exact code.

Use: epochs=30, n_layers=3, slice_num=32, surf_weight=10, lr=1e-4, weight_decay=1e-4, batch_size=4. No --n_head.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name warmup-2-epochs-cosine-nlayers3 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (38.270/32.470)
2. Per-split mae_vol_p
3. LR trajectory across epochs 1-5 (confirm warmup working)
4. Per-epoch wall-clock, best epoch, total
5. Parameter count, peak memory

## Baseline (PR #2228)

| Split | val | test |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | 52.042 | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |
