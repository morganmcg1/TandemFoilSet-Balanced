# SENPAI Research Results

## 2026-05-15 14:10 — PR #3188: H10: Increase slice_num from 64 to 128

- Branch: `thorfinn/slice-num-128`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: Doubling physics-state slice tokens from 64→128 gives finer flow-regime discretization without changing hidden width or depth.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **134.7389** |
| val_single_in_dist | 159.8405 |
| val_geom_camber_rc | 149.3953 |
| val_geom_camber_cruise | 109.1693 |
| val_re_rand | 120.5507 |

| Split | test mae_surf_p |
|-------|-----------------|
| test_single_in_dist | 132.6239 |
| test_geom_camber_rc | 132.9377 |
| test_geom_camber_cruise | NaN (data corruption — see below) |
| test_re_rand | 119.2658 |
| **test 3-split avg (excl. cruise)** | **128.2758** |

W&B run: `912m0995` · Group: `slice_num_128`

### Run details
- Epochs: **11/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 11 — val still falling steeply (162 → 134 in final epoch; not converged)
- Peak VRAM: 54.5 GB (well within 96 GB; slice-attention 128×128 is negligible vs node ops)

### Infrastructure bug discovered
`.test_geom_camber_cruise_gt/000020.pt` has 761 `inf` values in `y[:,2]` (pressure). The masked-arithmetic `inf * 0 = NaN` propagates into the accumulator — poisoning `test_geom_camber_cruise/mae_surf_p` for **all students**. Val metrics unaffected (all val gt is clean). **Fix**: defensive `y_finite` masking in `train.py:evaluate_split` assigned to thorfinn (PR relative-mse-bugfix).

### Analysis
- No concurrent slice_num=64 baseline yet. Other round-1 students effectively provide the reference.
- VRAM cost of 128 vs 64 is negligible.
- Merged as Round-1 reference — establishes first measured val_avg/mae_surf_p on this advisor branch.
