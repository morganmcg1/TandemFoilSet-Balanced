# n_head=2 on n_layers=3+slice_num=12+epochs=36: attention-head axis at new baseline stack

## Hypothesis

Tests n_head=2 at the NEW baseline stack (slice_num=12+n_layers=3+epochs=36). edward #2383 is testing n_head=2 at OLD slice_num=24 — this is the same axis at current best config.

head_dim=64 (n_head=2) vs head_dim=32 (n_head=4) doubles per-head representational capacity. If n_head=2 wins at slice_num=12: head_dim is load-bearing at compact stack.

**Compound EV:** If both lr=1.5e-4 (fern #2409) AND n_head=2 win independently, the compound test becomes next priority (~35.0 estimated).

## Instructions

Single flag change: `--n_head 2`. Use `--epochs 36`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name nhead2-nlayers3-slicenum12 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 12 \
  --n_head 2
```

## Reporting

1. Epoch 1 wall-clock (~50.3s expected)
2. Per-split val/test mae_surf_p vs NEW baseline (val=35.969 / test=30.265)
3. Per-split mae_vol_p
4. Best epoch, parameter count, peak memory

## Baseline (PR #2351)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.308 | 33.241 |
| geom_camber_rc | 49.521 | 43.631 |
| geom_camber_cruise | 19.576 | 15.969 |
| re_rand | 38.470 | 28.220 |
| **avg** | **35.969** | **30.265** |
