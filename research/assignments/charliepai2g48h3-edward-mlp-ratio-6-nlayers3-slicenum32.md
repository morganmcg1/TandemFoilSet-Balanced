# mlp_ratio=6 on n_layers=3+slice_num=32+epochs=28: mechanism transfer to new best stack

## Hypothesis

Edward's PR #2185 confirmed mlp_ratio=6 wins uniformly on n_layers=4 stack:
- Every split improved on val and test
- Test gain (−4.12%) > val gain (−3.08%) — generalization signal
- best_epoch=22/22 still descending
- Per-epoch cost: +8% (80s vs 74s)

Mechanism should transfer to n_layers=3. Per-epoch estimate: 58s × 1.08 = ~63s. epochs=28 × 63s = 29.4 min (fits in cap).

If compound additive: projected val ~37.1 (−3% from 38.270).

## Instructions

Same code change as PR #2185: modify train.py line 435 from `mlp_ratio=4` to `mlp_ratio=6`. Use n_layers=3 stack:

- `--n_layers 3` / `--slice_num 32` / `--epochs 28`
- `--surf_weight 10` / `--lr 1e-4` / `--weight_decay 1e-4` / `--batch_size 4`
- Do NOT pass `--n_head`

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name mlp-ratio-6-nlayers3-slicenum32 \
  --epochs 28 \
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
3. **Epoch 1 wall-clock** — critical for budget verification
4. Per-epoch wall-clock last 5 epochs, total
5. Best epoch — still final?
6. Parameter count (~715K expected), peak memory

## Baseline (PR #2228)

| Split | val | test |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | 52.042 | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |
