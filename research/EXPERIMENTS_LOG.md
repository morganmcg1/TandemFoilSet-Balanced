# SENPAI Research Results

## 2026-05-15 15:45 — PR #3162: H9: Raise surf_weight 10→25 ✗ CLOSED

- Branch: `askeladd/surf-weight-25`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Raising surf_weight from 10 to 25 emphasizes the surface (the scored region) in the gradient, should improve val_avg/mae_surf_p.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **133.4123** |
| val_single_in_dist | 163.71 |
| val_geom_camber_rc | 194.32 |
| val_geom_camber_cruise | 103.60 |
| val_re_rand | 125.67 |

| Split | test mae_surf_p (patched scoring) |
|-------|----------------------------------|
| test_single_in_dist | 134.42 |
| test_geom_camber_rc | 141.56 |
| test_geom_camber_cruise | 92.36 (via local patched scoring) |
| test_re_rand | 120.00 |
| **test_avg/mae_surf_p** | **122.0843** |

W&B run: `hkka77kg` · Group: `surf_weight_sweep`

### Run details
- Epochs: **14/50** (30-min wall-clock cap; best at epoch 13)
- Noisy trajectory: 133.63 (ep11) → 142 (ep12) → 133.41 (ep13) → 146.83 (ep14, cut)
- Peak VRAM: 42.1 GB / 96 GB

### Analysis
- 133.41 does NOT beat the new Huber baseline (112.90). **Closed**.
- The hypothesis was tested against the wrong baseline (MSE loss). With Huber loss already providing MAE-aligned gradients, the marginal benefit of surface emphasis is smaller than expected.
- Loss-metric alignment (Huber) dominates surface weighting at the same compute budget.
- Askeladd also produced an excellent independent bug report on the cruise NaN scoring issue (now being fixed in thorfinn PR #3309) — same root cause as alphonse identified.

### Suggested follow-ups (taken into round 2)
- The surf_weight knob is still worth testing on top of the Huber base (separate from askeladd's follow-up).
- Askeladd assigned PR #3317: cosine T_max tuning to match actual epoch budget — directly addresses the LR-not-annealing observation.

## 2026-05-15 14:30 — PR #3159: H1: Huber loss (delta=0.1) — NEW BASELINE ✓ MERGED

- Branch: `alphonse/huber-loss-aligned`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Replace MSE loss with Huber(delta=0.1) to align training objective with the MAE evaluation metric. At delta=0.1 in normalized space, residuals above 0.1 are in the L1 (MAE-equivalent) regime, creating direct gradient alignment with the scoring metric.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **112.9001** |
| val_single_in_dist | 134.4612 |
| val_geom_camber_rc | 143.4094 |
| val_geom_camber_cruise | 75.8516 |
| val_re_rand | 97.8785 |

| Split | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|-------|-----------------|-----------------|-----------------|
| test_single_in_dist | 120.1970 | 1.4079 | 0.5594 |
| test_geom_camber_rc | 134.3200 | 2.2348 | 0.7179 |
| test_geom_camber_cruise | NaN (data corruption) | 0.9322 | 0.4473 |
| test_re_rand | 92.7597 | 1.3172 | 0.5779 |
| **test 3-split avg (excl. cruise)** | **115.7589** | 1.4730 | 0.5756 |

W&B run: `bpczoejx` · Group: `huber_loss_delta01`

### Run details
- Epochs: **14/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 14 — val still falling (248 → 113 over run; healthy monotonic decrease)
- Peak VRAM: 42.1 GB (well within 96 GB budget)

### Analysis
- **Clear winner**: 112.9 vs 134.7 (thorfinn's slice_num=128), improvement of ~16%.
- MAE alignment works: Huber loss directly creates gradient alignment with the scoring metric. The model learns to minimize mean absolute error rather than mean squared error, which is exactly what's being measured.
- **LR schedule mismatch**: T_max=50 with only 14 epochs completed means LR was still at ~82% of peak (≈0.00041) when training stopped. The cosine schedule never annealed. This is the biggest remaining optimization opportunity — the model is undertrained relative to schedule.
- **Delta regime**: With trained residuals O(0.05–0.2) at epoch 14, many residuals are still below delta=0.1 and in the L2 regime. Smaller delta (0.05 or 0.01) would push more residuals into L1, potentially improving MAE alignment further.
- Per-split pattern: cruise val best (75.85), then re_rand (97.88), while single_in_dist (134.46) and geom_camber_rc (143.41) remain hardest — high-Re raceCar samples dominate absolute error.

### Student suggested follow-ups
1. Tune T_max to actual epoch budget (~14-15 epochs)
2. Smaller Huber delta (0.05, 0.01) or pure L1 to push fully into MAE-aligned regime
3. Per-channel loss weighting (emphasize pressure channel)
4. Patch the cruise-gt NaN bug (separate PR, affects all test metrics)

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
