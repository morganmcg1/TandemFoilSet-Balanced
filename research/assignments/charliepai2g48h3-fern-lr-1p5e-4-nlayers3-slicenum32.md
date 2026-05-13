# lr=1.5e-4 on n_layers=3+slice_num=32+epochs=30: retest LR axis at compact stack

## Hypothesis

The LR axis has been historically saturated at lr=1e-4 on deeper stacks (n_layers=4-6). But at the **new compact stack** (n_layers=3, 515K params, ~57s/epoch), the LR optimum may have shifted:

- Smaller model → faster traversal of loss landscape → can tolerate larger LR
- Best epoch=30/30 still descending at lr=1e-4 → model is underfit; faster updates could push it further in 30-min cap
- Lion + cosine annealing pairs well with slightly larger peak LR (Lion's sign update is conservative in magnitude — relies on LR for step size)

This is a **clean single-flag test** of whether lr=1.5e-4 at the compact stack is now better than lr=1e-4 (PR #2228 baseline).

If lr=1.5e-4 wins: compound stack benefits from higher LR; explore lr=2e-4 next.
If lr=1.5e-4 loses: LR optimum confirmed at 1e-4 across all current stacks; axis closed.

## Instructions

Single flag change: `--lr 1.5e-4`. Same config as PR #2228 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name lr-1p5e-4-nlayers3-slicenum32 \
  --epochs 30 \
  --lr 1.5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (38.270/32.470)
2. Per-split mae_vol_p
3. **Train loss trajectory** epochs 1-5 (diagnostic for early divergence at higher LR)
4. Per-epoch wall-clock, best epoch (still final? earlier?)
5. Parameter count, peak memory

## Baseline (PR #2228)

| Split | val | test |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | 52.042 | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |
