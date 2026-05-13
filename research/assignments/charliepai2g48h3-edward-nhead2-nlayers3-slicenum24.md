# n_head=2 on n_layers=3+slice_num=24+epochs=33: test richer per-head attention at compact depth

## Hypothesis

At compact stack (n_layers=3, slice_num=24), **attention heads may be redundant**. The current default is n_head=4, giving head_dim = n_hidden / n_head = 128 / 4 = 32. Reducing to n_head=2 doubles the per-head representation space (head_dim=64), which is more standard for transformers of this scale.

**Key reasoning:**
- At n_layers=3 (compact depth), each layer's attention must work harder per head
- head_dim=32 is quite small — the Q/K/V projections have limited expressiveness
- head_dim=64 (n_head=2) matches standard small-transformer practice
- No change in total hidden dim (128) → per-epoch wall-clock and memory footprint essentially unchanged

If n_head=2 wins: attention head dimension is load-bearing at compact depth, head_dim=64 is better.
If n_head=2 loses: 4-head parallelism (specialization) matters more than per-head expressiveness, n_head=4 confirmed optimal.

## Instructions

Single flag change: `--n_head 2`. No other changes from PR #2229 baseline.

**Wall-clock check:** n_head change shouldn't affect per-epoch time significantly. Verify epoch 1 ≈ 53-54s; if >56s reduce epochs to 31.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name nhead2-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24 \
  --n_head 2
```

## Reporting

1. **Epoch 1 wall-clock** — confirm no per-epoch overhead from head change
2. Per-split val/test mae_surf_p vs baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Best epoch (same as final, or earlier convergence with richer heads?)
5. Parameter count (should be ~514K, head change doesn't alter param count), peak memory

## Baseline (PR #2229)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
