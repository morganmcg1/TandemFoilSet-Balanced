# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-12 19:05 — PR #1397: L1 (MAE) loss replaces MSE in normalized-space training — **MERGED, new baseline**

- Branch: `charliepai2g24h4-alphonse/l1-loss`
- Hypothesis: Align training loss with the eval metric (MAE). MSE
  over-weighted high-Re outlier nodes whose y range spans up to 29K with
  per-sample y std varying ~10× within a single split. Expected 2–8%
  improvement on `val_avg/mae_surf_p`.
- Implementation: `(pred - y_norm).abs()` replaces `(pred - y_norm)**2` in
  both the training inner loop and `evaluate_split`. Surface/volume
  decomposition and `surf_weight = 10.0` kept unchanged. All other HPs
  default.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | **100.9574** |
| test_avg/mae_surf_p (3-split, excl. cruise) | **100.8314** |
| test_avg/mae_surf_p (4-split, raw) | NaN (data bug) |
| val_single_in_dist / mae_surf_p | 127.371 |
| val_geom_camber_rc / mae_surf_p | 110.832 |
| val_geom_camber_cruise / mae_surf_p | 77.353 |
| val_re_rand / mae_surf_p | 88.273 |
| n_params | 0.66 M |
| peak GPU mem | 42.1 GB |
| wall time | 30.7 min (cut at SENPAI_TIMEOUT_MINUTES=30 after ep 14) |

- Metric artifacts (advisor branch): `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`, `metrics.yaml`
- Training trajectory was monotone-descending: ep 1 223 → ep 13 101; ep 14
  bounced to 134 right before timeout. Cosine T_max=50 means LR only
  decayed ~16% from peak by ep 14 — schedule is mismatched to the 30-min
  wall-clock cap. Worth a follow-up arm.

### Conclusions and follow-ups

- L1 loss is a clear win and establishes the first numeric baseline on
  this advisor branch. Merged.
- Pre-existing data bug: `test_geom_camber_cruise/000020.pt` contains
  `inf` in y_p, propagating NaN through `data/scoring.py::accumulate_batch`
  even though the bad sample is correctly flagged. `data/scoring.py` is
  marked read-only, so we record the 3-split test mean and document the
  bug. Fix candidate for a later PR: in `train.py::evaluate_split`, pre-mask
  non-finite y samples by zeroing both the sample's `mask` and its y
  values before calling `accumulate_batch` (faithful trainer-side
  workaround that preserves the scoring contract).
- Round-2 candidate follow-ups suggested by student (in addition to the
  Round-2 idea file H1-H11): T_max=15 to align cosine with the 30-min
  wall-clock cap; small `surf_weight` sweep on top of L1 (10/15/25/50)
  since L1 is less outlier-dominated than MSE; Huber/SmoothL1 as a
  smooth alternative.


