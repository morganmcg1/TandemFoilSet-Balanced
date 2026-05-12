# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-12 19:10 — PR #1423: Enable unified_pos=True with ref=8 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/unified-pos`
- Hypothesis: Switch `unified_pos=False → True, ref=8` — add learned ref-grid
  positional features (Gaussian-RBF over an 8×8 grid in the (x, z) plane,
  repeat-interleaved to fill `ref**3 = 512`) before the preprocess MLP.
- Student noted real implementation concerns: `ref**3 = 512` packing inflates 2D
  features 8× (only 64 distinct grid cells); grid bounds were adjusted to
  `[-7, 7]` to match the actual data range. Proposed multi-scale RBFs and
  asymmetric per-axis grid bounds as round-2 follow-ups.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 118.605 |
| test_avg/mae_surf_p (4-split, NaN-safe) | 109.159 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +17.5% **worse** |

- Note: trained on MSE base (branched before L1 merge), so this is MSE+unified_pos
  rather than L1+unified_pos. Comparison is contaminated. Closed without rebase
  because (a) the absolute number is ~17% worse than L1, (b) re-running would
  consume 30 min for a hypothesis whose own author flagged implementation
  concerns, and (c) higher-EV ideas are queued. Multi-scale RBF variant may
  resurface later.
- Student also flagged the pre-existing `data/scoring.py` NaN-propagation bug
  (same one alphonse flagged in #1397) and committed a clean workaround in
  `evaluate_split`. We're propagating that workaround into all subsequent
  round-2 assignments.

## 2026-05-12 19:08 — PR #1403: Bump surf_weight 10 → 30 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-askeladd/surf-weight-30`
- Hypothesis: Increase `surf_weight` from 10 → 30 to focus optimizer pressure on
  the surface field that drives the primary metric.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 12/14) | 133.386 |
| test_avg/mae_surf_p (4-split, NaN-safe re-eval) | 120.962 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +32.1% **worse** |

- Trained on MSE base (branched before L1 merge), so this is MSE+surf_weight=30.
  Under L1 (less outlier-sensitive than MSE) the optimal `surf_weight` is unlikely
  to be larger than the default 10. Closed without rebase: re-running would burn
  30 min on a single-value HP sweep when L1 already wins by 30%+. A proper
  L1+surf_weight sweep (10/15/25/50) is a small follow-up worth considering only
  if other levers stop moving.
- Student also flagged the pre-existing `data/scoring.py` NaN bug and produced an
  independent NaN-safe re-evaluation script. Confirmed root cause.

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


