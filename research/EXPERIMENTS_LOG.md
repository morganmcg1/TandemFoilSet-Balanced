# SENPAI Research Results

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
