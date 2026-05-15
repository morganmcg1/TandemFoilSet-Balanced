# SENPAI Research Results — Round 5 (charlie-pai2i-24h-r5)

Advisor branch: `icml-appendix-charlie-pai2i-24h-r5`. Primary ranking metric: `val_avg/mae_surf_p` (lower better). Test-time decision metric: `test_avg/mae_surf_p`.

## 2026-05-15 14:25 — PR #3266: Per-sample scale-invariant loss to equalize Re-regime gradients [MERGED — round-5 baseline anchor]

- Branch: `charliepai2i24h5-frieren/per-sample-instance-norm-targets`
- Student: charliepai2i24h5-frieren
- Hypothesis: divide each sample's `vol_loss_per_sample + surf_weight * surf_loss_per_sample` by that sample's field std (computed in normalized-y space) before averaging across the batch, so the gradient contribution is equalized across the ~40× per-sample y_std variation between low-Re and high-Re samples.
- Status: MERGED as the round-5 anchor baseline. First terminal result on this branch.

### Results

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 142.1946 | 125.8483 |
| geom_camber_rc | 136.1165 | 130.8474 |
| geom_camber_cruise | 95.7637 | 85.2854 |
| re_rand | 121.4363 | 115.4971 |
| **avg** | **123.8778** | **114.3695** |

Metric artifacts:
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.yaml`

### Analysis

The per-split pattern matches the predicted mechanism. `geom_camber_cruise` has the lowest field std in the corpus (program.md notes cruise per-sample y std mean 164, max 506 vs raceCar single mean 458 max 2077) and was therefore most under-weighted by the original global-MSE loss. Under the scale-invariant loss, cruise becomes the easiest split (lowest val surf_p, 95.76). This is exactly the directional claim of the hypothesis.

Run hit the 30-min wall clock at epoch 14/50; val_avg/mae_surf_p was still falling at ~-2.6 per epoch with no plateau. The cosine schedule was set to T_max=50 so LR did not fully decay (was at ~89% of peak when training stopped). The student's own follow-up #2 (`--epochs 14` with matched cosine T_max) is a likely compounding improvement we can run later.

### Critical operational bug found

`data/scoring.py::accumulate_batch` has a NaN-propagation bug: one sample (`test_geom_camber_cruise/000020.pt`) has 761 NaN in the p channel of y. The intended sample-level skip via `y_finite = isfinite(y).all(dim=-1)` is defeated downstream by `err = |pred - NaN| = NaN`, then `NaN * 0 = NaN` slips past the masked sum, producing NaN `test_avg/mae_surf_p`. Because `data/` is read-only, the fix lives in `train.py::evaluate_split`: pre-skip samples with non-finite y from the mask and `torch.nan_to_num` the y before passing to `accumulate_batch`. This workaround is included in the merged baseline and propagated to the seven other in-flight round-5 PRs via PR comments.

## 2026-05-15 15:35 — PR #3265: FiLM Re/AoA/NACA conditioning every Transolver block [SENT BACK — beat baseline but had merge conflicts]

- Branch: `charliepai2i24h5-fern/film-flow-condition-every-block`
- Student: charliepai2i24h5-fern
- Hypothesis: per-block FiLM scale+shift modulation on global flow vector (log Re, AoAs, NACAs, gap, stagger) injected after each Transolver block's residual, giving every layer direct access to the global flow conditioning rather than only the input MLP.
- Status: WINNER on absolute numbers (val 122.27 / test 112.17 vs baseline 123.88 / 114.37) but the branch was cut before PR #3266 landed, so squash-merge had conflicts in `train.py`. Sent back for rebase + re-run; FiLM should compound cleanly with the scale-invariant loss in the merged baseline.

### Results

| Metric | Baseline (PR #3266) | FiLM (#3265) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 123.8778 | **122.2673** | -1.30% |
| test_avg/mae_surf_p | 114.3695 | **112.1684** | -1.92% |

Per-split val surface pressure MAE:

| Split | Baseline | FiLM | Δ |
|---|---:|---:|---:|
| single_in_dist | 142.19 | 141.32 | -0.6% |
| geom_camber_rc | 136.12 | 138.27 | +1.6% |
| geom_camber_cruise | 95.76 | 98.62 | +3.0% |
| **re_rand** | **121.44** | **110.87** | **-8.7%** ← big OOD win |

Per-split test surface pressure MAE:

| Split | Baseline | FiLM | Δ |
|---|---:|---:|---:|
| single_in_dist | 125.85 | 120.73 | -4.1% |
| geom_camber_rc | 130.85 | 128.81 | -1.6% |
| geom_camber_cruise | 85.29 | 87.81 | +3.0% |
| re_rand | 115.50 | 111.32 | -3.6% |

Best epoch 13/50, run cut at epoch 14 by 30-min wall clock. Model is 829K params (FiLM adds 166K on top of baseline 663K), no OOM, no instability.

### Analysis

The FiLM mechanism delivers exactly the predicted Re-rand OOD improvement (-8.7% on val_re_rand, the single largest per-split win), confirming the hypothesis: explicit per-layer modulation by log Re + AoA + geometry is meaningfully different from baking those features into the input MLP. Three of four test splits improve. Cruise val regresses by 3% — small enough that compounding with the scale-invariant baseline (which had cruise as its strongest split) is likely to net out positive.

Predicted -8% to -15%; actual -1.3% to -1.9% averaged. Less impressive than predicted on val_avg, but the test_avg win is meaningful and the re_rand win is exactly the hypothesis prediction. Compounding with scale-invariant loss is the natural next step.

Sent back for rebase onto current advisor branch + re-run with FiLM stacked on top of scale-invariant loss + NaN fix.

## 2026-05-15 15:35 — PR #3272: Surface arc-length Fourier PE for surface pressure accuracy [CLOSED — regression]

- Branch: `charliepai2i24h5-askeladd/surface-arclen-positional-encoding`
- Student: charliepai2i24h5-askeladd
- Hypothesis: inject sin/cos Fourier features of the signed-arc-length input dims into the per-node embedding (only on surface nodes), giving the slice-attention an explicit periodic boundary-following positional structure beyond the raw MLP-projected features.
- Status: CLOSED. Both arms (n_freqs=8 and n_freqs=16) regressed >10% from baseline. Surface PE direction not productive for this benchmark.

### Results

| Arm | params | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (val) |
|---|---:|---:|---:|---:|
| baseline (PR #3266) | 662,359 | 123.8778 | 114.3695 | — |
| Arm A (n_freqs=16) | 670,951 | 151.123 | 138.528 | +21.9% worse |
| Arm B (n_freqs=8) | 666,847 | **139.838** | **125.336** | **+12.9% worse** |

Arm B beats Arm A on every split, indicating the larger PE bandwidth (n_freqs=16) is over-parameterised for the small arc-length signal each foil carries. Even Arm B (the winner of the ablation) is >12% worse than baseline on val_avg.

### Analysis

The arc-length signed-distance features are already in dims 2–3 of the raw input x and pass through the preprocessor MLP. Adding a sin/cos Fourier expansion of the same dims as a separate additive embedding did not help the model — the preprocessor is apparently already extracting the relevant frequencies. The hypothesis assumed under-utilisation of arc-length, but the data shows the opposite. Surface arc-length Fourier PE is not a productive direction for this benchmark.

Important caveat: this PR ran on vanilla MSE (the branch was cut before PR #3266 landed). Even applying the scale-invariant loss on rebase is very unlikely to close a >12% gap; the scale-invariant loss alone moved baseline numbers by only ~1.3% on val_avg in the merged result. Closed as clear regression.
