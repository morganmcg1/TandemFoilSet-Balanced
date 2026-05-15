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
