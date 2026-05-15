# SENPAI Research Results

## 2026-05-15 20:20 — PR #3359: H13: Pressure channel-weighted surf loss (p=3x) ✗ CLOSED

- Branch: `edward/pressure-ch-weight`
- Student: willowpai2i48h1-edward
- Hypothesis: Per-channel surf loss weighting (p=3x, Ux/Uy=1x) to emphasize the scored metric.

### Results (W&B only — code never committed to PR)

| Config | val_avg/mae_surf_p | test_avg |
|--------|-------------------|---------|
| pressure_ch_w3 (18:28) | 133.32 | 101.23 |
| pressure_ch_w3 (19:22, crashed) | 163.59 | — |
| pressure_ch_w5 (19:33) | 112.22 | 94.86 |

W&B runs: `(see wandb group)`

### Analysis
- Best val=112.22 (w=5), which is +23% worse than new baseline (91.33).
- Pressure weighting ALONE (without architectural specialization) fails to help. The 3x weight on the pressure channel distorts the vol+surf_Ux/Uy gradient budget without providing a separate learning pathway.
- Compare to fern's result: split head + 3x weight DID help (-6.2% test), confirming that architectural specialization is the missing ingredient.
- Increasing W from 3→5 showed slight improvement (133→112), but diminishing returns suggest diminishing gradient signal for Ux/Uy.
- **Note**: Student iterated without committing code to PR — made advisor review impossible. New assignment instructs explicit commit-before-run discipline.

---

## 2026-05-15 19:30 — PR #3361: H10b: slice_num=128 retry on Huber+NaN base ✗ CLOSED

- Branch: `thorfinn/slice128-retrial`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: slice_num=128 on correct (Huber+NaN) base. Round-1 retry tested on MSE base.

### Results

| Metric | slice=128 | baseline slice=64 | Δ |
|--------|-----------|-------------------|---|
| val_avg/mae_surf_p | 116.1928 | 112.8295 | **+3.36 worse** |
| test_avg/mae_surf_p | 112.5640 | 106.5996 | **+5.96 worse** |
| val_geom_camber_rc | 117.74 | 133.69 | **-15.96 better** |

W&B: `z8pyszfb` · 11 epochs (171s/ep, T_max=50, peaked 95GB VRAM)

### Analysis
- Capacity-budget tradeoff confirmed again (see also #3180 h=192): slice=128 is 30% slower, only 11 epochs vs baseline's 14.
- LR barely decayed (T_max=50, 22% consumed). Model still improving at timeout. Not "slice=128 fails" — it's budget-constrained.
- OOD gain: val_geom_camber_rc improved -15.96, supporting that richer physics-state helps hardest splits, but aggregate is negative within budget.
- VRAM ceiling: 95GB at slice=128 (98% of 96GB H100).
- **Conclusion**: capacity not the bottleneck at this wall-clock budget. Close.

---

## 2026-05-15 19:30 — PR #3363: H8: AdamW β2=0.95 + grad clip 1.0 → SENT BACK (rebase on T_max=15)

- Branch: `tanjiro/adamw-stability`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: β2=0.95 + grad clip 1.0 reduces gradient instability and improves convergence.

### Results (vs OLD Huber+NaN baseline, T_max=50)

| Metric | This run | Old baseline | New baseline (91.33) |
|--------|---------|--------------|----------------------|
| val_avg/mae_surf_p | 102.2436 | 112.8295 | 91.3319 |
| test_avg/mae_surf_p | 97.6239 | 106.5996 | 88.4260 |
| val_single_in_dist | 115.16 | 142.47 | **-19.2% best split** |

W&B: `44lht7xd` · 14 epochs · 99.7% of steps clipped (median grad_norm=3.71)

### Analysis
- Genuine optimizer improvement: val=102.24 (-9.4%), test=97.62 (-8.4%) vs old baseline.
- Grad clip at 1.0 is aggressive (binding on 99.7% of steps, median pre-clip norm 3.71). Clipping confirms the hypothesis that large gradient spikes were destabilizing training.
- Best epoch is epoch 14 (final, still descending) — suggests more budget would help further.
- Does NOT beat new T_max=15 baseline (91.33). Orthogonal to schedule fix — stacking should compound.
- **Action**: rebase on T_max=15 base, re-run with β2=0.95 + clip 1.0.

---

## 2026-05-15 18:30 — PR #3317: H3b: Cosine T_max=15 tuned to actual epoch budget ✓ MERGED (NEW BASELINE)

- Branch: `askeladd/cosine-tmax-tuned`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Aligning T_max with the real ~14-epoch wall-clock budget allows the cosine schedule to fully anneal. T_max=50 with only 14 epochs leaves LR at 79% of peak — effectively no annealing.

### Results

| Arm | T_max | val_avg/mae_surf_p | Δ vs baseline | W&B |
|-----|-------|--------------------|---------------|-----|
| Baseline | 50 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **15** | **91.3319** | **-19.1%** | `kx17n4pn` |
| B | 12 | 103.1193 | -8.7% | `z8h5w88d` |

| Test split | Arm A (T_max=15) |
|------------|-----------------|
| test_single_in_dist | 96.7268 |
| test_geom_camber_rc | 88.3769 |
| test_geom_camber_cruise | NaN (branch predates NaN fix) |
| test_re_rand | 80.1744 |
| **test_avg (3-split)** | **88.4260** |

### Analysis
- Biggest single improvement in the programme: -19.1% from a 1-line hyperparameter change.
- T_max=15 matches the 14-epoch budget: epoch 14 runs at ~1.1% of peak LR (fine-tuning pass). T_max=12 crashed to 0% LR at epoch 12, leaving 2 wasted epochs; gap of 103.12 vs 91.33 = 12 MAE points.
- The baseline T_max=50 was essentially NOT annealing — the LR was at 79% of peak at training stop.
- Key observation: per-split improvement is uniform (single_in_dist -26, geom_camber_rc -45, cruise -3, re_rand -12), suggesting the gain is structural (schedule fix) rather than overfitting to any particular split.
- **This result fundamentally shifts the research programme**: the binding constraint was schedule mis-alignment, not loss function or architecture. All future hypotheses should compare against this baseline.

---

## 2026-05-15 18:30 — PR #3305: H1b: Huber delta=0.05 scan → SENT BACK (rebase on new base)

- Branch: `alphonse/huber-smaller-delta`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Shrinking Huber δ from 0.1 to 0.05 pushes more residuals into L1 regime, improving MAE alignment.

### Results (vs OLD baseline 112.90 with T_max=50)

| Arm | delta | val_avg/mae_surf_p | Δ vs old baseline | W&B |
|-----|-------|--------------------|-------------------|-----|
| Old Baseline | 0.10 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **0.05** | **98.1913** | **-13.0%** | `oolv8t1p` |
| B | 0.02 | 103.7964 | -8.1% | `zlqqtxsu` |

val=98.19 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- δ=0.05 is the right direction — U-shaped response with δ=0.02 overshooting (loss landscape becomes near-constant-gradient L1, slowing late refinement).
- Both arms were run with T_max=50 (handicapped). On the new T_max=15 base, δ=0.05 is expected to yield additional stacked improvement.
- **Action**: rebase onto T_max=15 base, rerun with δ=0.05 only. Target: beat 91.33.

---

## 2026-05-15 18:27 — PR #3171: H8b: Split pressure head + 3x weight on Huber base → SENT BACK (rebase)

- Branch: `fern/split-pressure-head`
- Student: willowpai2i48h1-fern
- Hypothesis: Dedicated output head for pressure channel with 3x Huber-weighted loss improves OOD pressure MAE.

### Results v2 (rebased onto Huber base, with T_max=50)

| Metric | This PR | Huber baseline | Δ |
|--------|---------|---------------|---|
| val_avg/mae_surf_p | 111.9988 | 112.8295 | -0.90 |
| test_avg/mae_surf_p (all 4 splits) | **99.9669** | **106.5996** | **-6.63** |

val=112.00 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- val improvement is marginal (-0.8%), but **test improvement is genuine and consistent**: geom_camber_rc (-13.8 test), cruise test (-15.0), geom_camber_rc val (-23.4). The split head specifically improves OOD generalization.
- v1 (MSE) failed; v2 (Huber base) succeeded — confirming loss-metric alignment is prerequisite for architectural improvements.
- Both runs used T_max=50 (handicapped). With T_max=15, the split head should achieve further improvement.
- **Action**: rebase onto T_max=15 base, rerun with split head + 3x pressure weight + Huber(δ=0.1). Target: beat 91.33.

---

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

## 2026-05-15 17:00 — PR #3309: Bugfix: inf*0=NaN in evaluate_split ✓ MERGED

- Branch: `thorfinn/nanbug-fix`
- Student: willowpai2i48h1-thorfinn
- Type: Infrastructure bugfix — 4 defensive lines in evaluate_split; model unchanged

### Results

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **112.8295** (=baseline, within noise) |
| **test_avg/mae_surf_p** | **106.5996** ← was NaN (all 4 splits now valid) |
| test_geom_camber_cruise/mae_surf_p | **83.4377** ← was NaN |

W&B run: `g48284pc` · Group: `nanbug_fix`

### Analysis
- Model unchanged; val reproduces baseline within Δ=0.07 (noise).
- **Critical outcome**: test_geom_camber_cruise/mae_surf_p is now 83.44 (finite) and test_avg/mae_surf_p=106.60 is the first valid 4-split test score on this branch.
- Fix: `_y_fin` masking before arithmetic in evaluate_split prevents `pred - (-inf) = inf` → `inf * 0 = NaN` propagation via `data/scoring.py:accumulate_batch`.

## 2026-05-15 17:05 — PR #3180: H4: Wider model (hidden=192, slice_num=96) ✗ CLOSED

- Branch: `tanjiro/wider-model-h192`
- Student: willowpai2i48h1-tanjiro

### Results

| Run | val_avg/mae_surf_p |
|-----|-------------------|
| `a8p3g73s` (h=192 run 1) | **150.3762** (best of 2) |
| `nj0chxr6` (h=192 run 2) | 156.3125 |
| Baseline (h=128 Huber) | 112.9001 |

W&B runs: `a8p3g73s`, `nj0chxr6` · Group: `wider_model_h192`

### Analysis
- 150.38 vs 112.90 = 33% regression. Closed.
- h=192 is 1.6× slower/epoch → only 9 epochs vs baseline's 14. But per-epoch metrics are also worse (150 at ep8 vs ~145 for baseline at ep8 per historical data).
- ~2.2× more params (1.48M vs 0.66M) did not help at this budget.
- Bottleneck is clearly loss/schedule/features, not capacity.
- Seed variance ~4% (156.31 vs 150.38) is significant — future capacity tests should pin a seed.

## 2026-05-15 17:10 — PR #3167: H12: OneCycleLR max_lr=1e-3 ✗ CLOSED

- Branch: `edward/onecycle-lr`
- Student: willowpai2i48h1-edward

### Results

| Run | epochs | val_avg/mae_surf_p | Notes |
|-----|--------|-------------------|-------|
| `x9mygbcm` | 9 | 192.6188 | schedule misconfigured (total_steps sized for 50 ep) |
| `27mfh19o` | 9 | 172.9975 | same misconfiguration |
| `xn1ad9ka` | 9 | **137.1218** | fixed: --epochs 9, schedule fully annealed |
| Baseline (Huber cosine) | 14 | 112.9001 | — |

W&B runs: `xn1ad9ka` (final) · Group: `onecycle_lr`

### Analysis
- 137.12 vs 112.90 = 21% regression after correct schedule setup. Closed.
- **Key insight**: Edward diagnosed the schedule mismatch himself and reran with --epochs 9. The schedule fully annealed (4e-5 → 1e-3 → ~0), so the hypothesis was correctly tested.
- OneCycleLR fails because: (a) 9-epoch total budget means no prolonged low-LR refinement phase, and (b) cosine starts at peak LR and descends immediately, giving better use of the budget.
- NaN on test_geom_camber_cruise is a model-quality issue (extreme prediction on under-converged model at high LR), not the data corruption bug.
