# n_head=1 on n_layers=3+slice_num=24+epochs=33: single-head attention at compact depth

## Hypothesis

At compact stack (n_layers=3, slice_num=24), PhysicsAttention may not benefit from multi-head splitting. Tests the extreme: **n_head=1** (single head, head_dim=128). Edward (#2383) tests n_head=2 (head_dim=64) — these two together bracket the attention-head axis.

**Why single-head might win:** At n_layers=3, each attention layer must work harder per head. A single head with head_dim=128 maintains richer per-slice representations than 4 heads at head_dim=32. If heads specialize across layers, n_head=2 wins. If monolithic attention is better, n_head=1 wins.

## Instructions

Single flag change: `--n_head 1`. Same config as PR #2229 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name nhead1-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24 \
  --n_head 1
```

## Reporting

1. Epoch 1 wall-clock — confirm no overhead from n_head=1
2. Per-split val/test mae_surf_p vs baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Best epoch, parameter count (~514K), peak memory
5. Which split is worst-hit? (expect geom_camber_rc if multi-head specialization matters)

## Baseline (PR #2229)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
