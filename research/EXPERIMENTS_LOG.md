# SENPAI Research Results

## 2026-05-15 20:25 — PR #3393: Per-channel surface pressure weighting (surf_p_weight_extra=4)

- **Branch**: `charliepai2i24h3-thorfinn/surf-p-channel-weight`
- **Hypothesis**: Concentrate the extra surface gradient on the pressure channel (dim 2) only, leaving Ux/Uy surface losses and the vol loss unchanged. Effective pressure-channel weight = `surf_weight * (1 + surf_p_weight_extra) = 10 * 5 = 50`. Predicted 2-6% improvement, especially on val_single_in_dist (the hardest pressure split).
- **Outcome**: **SENT BACK — mechanism works but balance wrong; trying surf_p_weight_extra=2.0**

| Metric | Baseline | This run | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 117.66 | **117.95** | +0.28 (neutral wash) |
| `val_single_in_dist/mae_surf_p` | 147.77 | **132.70** | **-15.07 (big win)** |
| `val_geom_camber_rc/mae_surf_p` | 125.08 | 131.68 | +6.60 (regression) |
| `val_geom_camber_cruise/mae_surf_p` | 88.98 | 99.62 | +10.64 (regression on easiest split) |
| `val_re_rand/mae_surf_p` | 108.81 | 107.80 | -1.01 (slight gain) |
| `test_avg/mae_surf_p` (partial, 3 finite splits) | ~107.6 | 114.51 | +6.91 |
| Epochs run | 13/50 | 14/50 (still descending) | — |
| Peak VRAM | 42.11 GB | 42.11 GB | — |
| Artifact | — | `models/model-surf_p_weight_extra_4-20260515-192333/metrics.jsonl` | — |

**Analysis**: This is the most informative result so far. The gain mechanism works exactly as predicted — pressure-channel gradient redirection produces a 15-point improvement on val_single_in_dist (the hardest pressure split). But the gain is offset by losses on the easier splits (cruise +10.64, rc +6.60). At `extra=4`, the optimizer redistributes capacity from low-magnitude pressure splits toward high-magnitude ones; with a fixed-capacity model this is roughly zero-sum across splits when averaged.

Implementation is correct (vol metrics largely unchanged, Ux/Uy surface metrics unchanged in spirit, only pressure surface metrics moved). The PR is qualitatively different from #3303 in the right way — it doesn't break everything, just redistributes.

**Decision: send back with `surf_p_weight_extra=2.0`**. The PR's original decision rule pointed to 8.0 next, but the per-split data shows that doubling the redirection will make the redistribution worse, not better. The optimum lives *below* 4, not above. Asked thorfinn to try 2.0 (and 1.0 as a companion if GPU available) to find the curve shape.

Adjacent direction: this is the per-channel analogue of alphonse's #3177 (per-sample-scale-norm). Together they triangulate whether the right lever for hard-pressure splits is per-channel or per-sample.

---

## 2026-05-15 18:26 — PR #3303: Increase surf_weight from 10 to 50 with Huber baseline

- **Branch**: `charliepai2i24h3-thorfinn/surf-weight-50`
- **Hypothesis**: Increasing `surf_weight` from 10 to 50 (5× surface node emphasis) would focus gradient more on the boundary-layer region and improve surface pressure prediction across all splits.
- **Outcome**: **CLOSED — negative result (3.5% regression), hypothesis falsified at this magnitude**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **121.79** (best epoch 13; baseline 117.66) |
| `val_single_in_dist/mae_surf_p` | 155.27 (+7.50 vs baseline) |
| `val_geom_camber_rc/mae_surf_p` | 124.28 (−0.80 vs baseline — only split improved) |
| `val_geom_camber_cruise/mae_surf_p` | 93.70 (+4.72 vs baseline) |
| `val_re_rand/mae_surf_p` | 113.90 (+5.09 vs baseline) |
| `test_avg/mae_surf_p` (partial, 3 splits) | 123.88 |
| Epochs run | 13/50 (30-min cap) |
| Peak VRAM | 42.11 GB |
| Artifact | `models/model-surf_weight_50_huber-20260515-172709/metrics.jsonl` |

**Analysis**: The 5× surface-node scaling (uniform over Ux, Uy, p) degraded 3 of 4 splits, including the predicted-best (`val_geom_camber_cruise` went from 88.98 → 93.70). The only marginal improvement was `val_geom_camber_rc` (−0.80 pts). Model also produced NaN in `test_geom_camber_cruise` surface pressure predictions — a prediction-side instability triggered by the aggressive gradient, separate from the existing GT-side NaN bug.

Root cause: scalar `surf_weight` boosts all 3 surface channels together. With a fixed-capacity model, this is mostly zero-sum across surface channels vs volume. The gradient to the ranking channel (pressure) only receives 1/3 of the additional surface budget; the rest goes to Ux/Uy surface losses which were already well-optimized.

→ Closed. thorfinn reassigned to **per-channel surface pressure weighting** (PR #3393): concentrate the extra gradient specifically on the pressure channel (dim 2), leaving Ux/Uy surface losses unchanged. This is qualitatively different from the uniform-scalar approach.

---

## 2026-05-15 16:25 — PR #3238: Dual surface/volume output heads in final TransolverBlock

- **Branch**: `charliepai2i24h3-fern/dual-branch-heads`
- **Hypothesis**: Replace the single output MLP in the final TransolverBlock with two parallel MLPs (one for surface nodes, one for volume nodes), gated by `is_surface`. Predicted 4–10% improvement.
- **Outcome**: **SENT BACK — not apples-to-apples (used MSE; baseline uses Huber)**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 124.52 (best epoch 14, +5.8% over baseline) |
| `val_single_in_dist/mae_surf_p` | 147.74 |
| `val_geom_camber_rc/mae_surf_p` | 133.47 |
| `val_geom_camber_cruise/mae_surf_p` | 99.46 |
| `val_re_rand/mae_surf_p` | 117.41 |
| `test_avg/mae_surf_p` | 113.41 (finite — only test result without NaN!) |
| Epochs run | 14/50 (30-min cap) |
| Peak VRAM | 43.13 GB |
| Params | 0.68M (+17K vs single head) |
| Artifact | `models/model-charliepai2i24h3-fern-dual_branch_heads-20260515-143202/metrics.jsonl` |

**Analysis**: 5.8% above baseline, but the comparison is not clean — this run used MSE loss (the old default), while the new baseline uses Huber(δ=1.0). The architecture+MSE combination is plausibly less optimal than architecture+Huber. Also notable: val curve had a sharp ep13→ep14 drop (183.21 → 124.52), suggesting the last-epoch number may be noisier than the trajectory implies. Sent back for rebase + re-run with the merged Huber loss.

The test_avg = 113.41 (finite) is interesting — fern's run is the only one so far with a non-NaN test number. Either the dual-branch architecture happens to produce a finite output for the bad sample, or fern's predictions for `test_geom_camber_cruise/000020.pt` happen to be inf themselves (giving `(inf - inf).abs() = NaN` only in scoring.py's per-element step). Investigating after the rebase.

---

## 2026-05-15 15:25 — PR #3237: Huber loss (delta=1.0) to cap high-Re gradient outliers

- **Branch**: `charliepai2i24h3-edward/huber-loss`
- **Hypothesis**: Replace MSE with Huber(δ=1.0) in normalized target space. MSE squares residuals, biasing gradients toward high-Re samples with extreme values. Huber is quadratic for |r|<δ and linear beyond, capping outlier influence. Predicted 2–6% improvement on val_avg/mae_surf_p.
- **Outcome**: **MERGED — new baseline**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **117.66** (best epoch 13) |
| `val_single_in_dist/mae_surf_p` | 147.77 |
| `val_geom_camber_rc/mae_surf_p` | 125.08 |
| `val_geom_camber_cruise/mae_surf_p` | 88.98 |
| `val_re_rand/mae_surf_p` | 108.81 |
| `test_avg/mae_surf_p` | NaN (scoring.py bug — see below) |
| `test_avg/mae_surf_p` (clean, 3 splits) | ~107.6 |
| Epochs run | 14/50 (30-min cap) |
| Peak VRAM | 42.11 GB |
| Params | 0.66M |
| Artifact | `models/model-huber_loss_d1-20260515-130807/metrics.jsonl` |

**Analysis**: First completed experiment and new branch baseline. Huber loss trains stably and the val curve was still descending at epoch 13 (timeout). The per-split ordering is sensible: `val_geom_camber_cruise` (cruise tandem, moderate-Re) benefits most (88.98), while `val_single_in_dist` (raceCar single, high-Re range) is hardest (147.77). This is consistent with Huber's bounded gradient giving proportionally more signal to moderate-Re samples that MSE would underweight.

The 2-line change (MSE → Huber + config field) is now the lowest-complexity lever confirmed working. All future experiments build on this.

---

## 2026-05-15 14:43 — PR #3242: Reynolds-number curriculum (low-Re first 50% of epochs)

- **Branch**: `charliepai2i24h3-thorfinn/re-curriculum`
- **Hypothesis**: Sort training samples ascending by Re for first 50% of epochs (low-Re/laminar first), then switch to standard WeightedRandomSampler. Predicted 3–8% improvement, especially val_re_rand.
- **Outcome**: **CLOSED — 60% regression**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 188.22 (best epoch 14) |
| `val_single_in_dist/mae_surf_p` | 254.55 |
| `val_geom_camber_rc/mae_surf_p` | 206.90 |
| `val_geom_camber_cruise/mae_surf_p` | 129.78 |
| `val_re_rand/mae_surf_p` | 161.64 |
| Epochs run | 14/50 (30-min cap) |
| Artifact | `models/model-charliepai2i24h3-thorfinn-re-curriculum-frac-0.5-20260515-140645/metrics.jsonl` |

**Analysis**: The design was untestable within the 30-min budget. With `curriculum_frac=0.5` and `SENPAI_MAX_EPOCHS=50`, the switch point was epoch 25 — but the run stopped at epoch 14 so only the pure curriculum phase ran. Re-sorted (ascending) training without domain balance severely hurt `val_single_in_dist` (254.55) and `val_geom_camber_rc` (206.90). The WeightedRandomSampler exists precisely to prevent this domain imbalance. The curriculum concept remains interesting for a future test but needs `curriculum_frac ≤ 0.2` to test the combined design within the current budget.

---

## 2026-05-15 14:39 — PR #3235: Add local-Re feature (Re x |x|) as surface-gated 25th input dim

- **Branch**: `charliepai2i24h3-askeladd/local-re-feature`
- **Hypothesis**: Add `Re_x = log1p(Re * |x_chord|)` gated to surface nodes as 25th input feature. This makes the boundary-layer development signal spatially varying rather than global. Predicted 4–10% improvement, especially val_geom_camber.
- **Outcome**: **SENT BACK — promising direction, needs Huber loss on top**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 124.27 (best epoch 14, still improving) |
| `val_single_in_dist/mae_surf_p` | 156.90 |
| `val_geom_camber_rc/mae_surf_p` | 133.28 |
| `val_geom_camber_cruise/mae_surf_p` | 93.74 |
| `val_re_rand/mae_surf_p` | 113.15 |
| `test_avg/mae_surf_p` | NaN (scoring.py bug) |
| Epochs run | 14/50 (30-min cap) |
| Artifact | `models/model-charliepai2i24h3-askeladd-local-re-feature-20260515-140549/metrics.jsonl` |

**Analysis**: 124.27 vs. Huber-loss baseline 117.66 = 5.6% worse, but the trajectory was still descending. The gap is likely explained by using MSE (not Huber) — the Huber improvement is not included here. Feedback: rebase on advisor branch, add Huber(δ=1.0) which is now the default, and also try signed arc-length (saf, dims 2-3) as the surface coordinate instead of |x_chord| (dim 0).

---

## Branch-wide bug: NaN propagation in test_avg/mae_surf_p

**Reported by**: charliepai2i24h3-askeladd (PR #3235) and charliepai2i24h3-edward (PR #3237)

`test_geom_camber_cruise/000020.pt` contains 761 `inf` values in ground-truth `y` (likely a CFD artifact). `data/scoring.py::accumulate_batch` tries to skip these via `y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)`, but implements the skip by zeroing `surf_mask` *after* `err = (pred_orig.double() - y.double()).abs()` is computed. `inf - finite = inf`, and `inf × 0 = NaN` in IEEE-754, so the NaN propagates into the per-channel accumulator, making `test_avg/mae_surf_p = NaN` for all experiments.

**Impact**: All experiments on this branch report `test_avg/mae_surf_p = NaN`. The val_avg/mae_surf_p ranking is unaffected (no bad samples in val splits). The clean estimate for 3-split test_avg_surf_p is: take the mean of the finite 3 splits.

**Workaround**: Rank by `val_avg/mae_surf_p`. For paper-facing test numbers, use the 3-split mean or apply `nan_to_num` to `err` before the masked sum in `data/scoring.py`. `data/scoring.py` is marked read-only; this requires a coordinator-side fix or a read-only waiver.
