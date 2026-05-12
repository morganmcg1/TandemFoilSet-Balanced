# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-12 19:55 — Stale-WIP closures: 5 PRs branched off pre-L1 MSE base

Five round-1 PRs (#1407 wider/deeper, #1411 slice_num=128, #1417 lr-warmup=1e-3,
#1420 EMA weights, #1425 SwiGLU FFN) were assigned at 17:52 UTC, before L1 loss
(PR #1397) merged at 19:05. Student pods were stalled on GH API rate limits
through 19:50 and never started training. Closing because any result on those
branches would be measured against pre-L1 MSE base and not directly comparable
to the new L1 baseline. All five hypotheses remain valid avenues to revive
in a later round; they are deprioritized for round 2 in favour of architecture
and loss-formulation ideas from `RESEARCH_IDEAS_2026-05-12_round2.md`.

## 2026-05-12 19:50 — PR #1530: Per-channel L1 loss with pressure x3 weight — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/channel-weight-p3`
- Hypothesis: H4 from round-2 list. In L1 loss, multiply pressure channel by
  P_WEIGHT=3.0 to steer gradient flow toward the ranking metric `mae_surf_p`.
  Predicted 2-6% improvement on `val_avg/mae_surf_p`.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 102.184 | 100.957 | **+1.22% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.696 | 100.831 | -0.13% |
| test_avg/mae_surf_p (**4-split, NaN-safe, new finite ref**) | **92.465** | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 126.233 / 112.645 / 77.502 / 92.356 | 127.371 / 110.832 / 77.353 / 88.273 | mostly noise except +4.6% on re_rand |

- Effective combined surface-pressure weight became `surf_weight × P_WEIGHT = 30`,
  the same regime as closed PR #1403 (`surf_weight=30`), which also regressed.
  Student diagnosed this directly. The 3× upweight is too aggressive on top
  of L1's already-amplified surface gradients.
- **Lasting deliverable:** the NaN-safe pre-filter in `train.py::evaluate_split`
  works as designed and produced the first finite 4-split test mean on this
  branch (92.465). Pre-filter pattern is now bundled into every round-2 PR
  assignment so subsequent runs land a comparable 4-split test reference.
- Per-channel surface MAE at best val: surf_Ux=1.43, surf_Uy=0.69, surf_p=102.18;
  vol_Ux=4.81, vol_Uy=2.22, vol_p=103.80. Predicted Ux/Uy uptick in exchange for
  p drop did NOT materialize — we got a p regression instead.
- Suggested follow-ups (lower P_WEIGHT, combined surf_weight+P_WEIGHT sweep)
  are deferred until higher-EV round-2 levers are explored.

## 2026-05-12 19:48 — PR #1529: Gradient clipping (max_norm=1.0) — **CLOSED, much worse than L1**

- Branch: `charliepai2g24h4-askeladd/grad-clip-1.0`
- Hypothesis: H5 from round-2 list. Add `clip_grad_norm_(max_norm=1.0)` to
  reduce variance from gradient spikes on variable mesh sizes / high-Re samples.
  Predicted 1-4% improvement on `val_avg/mae_surf_p` via smoother convergence.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 11/14) | 106.401 | 100.957 | **+5.4% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 103.364 | 100.831 | +2.5% |
| test_avg/mae_surf_p (**4-split, NaN-safe, finite ref**) | **94.846** | NaN | — |

- Student logged per-epoch gradient norms (min 10, mean 47, max 245) and clip%
  per epoch (100% in every epoch). `max_norm=1.0` is far below the natural
  pre-clip norm of 10-245, so every step was rescaled by 0.02-0.10× — the
  model effectively trained at 1-5% of the configured LR throughout, which
  is too slow to converge inside 14 epochs.
- The diagnosis is exemplary post-hoc analysis and exactly the kind of
  per-epoch instrumentation we want from every arm.
- **Lasting deliverable:** NaN-safe pre-filter in `evaluate_split` (identical
  to tanjiro's #1530 fix). 4-split test mean (94.846) is reproducible and
  finite. Workaround is now standard in all round-2 assignments.
- Suggested follow-ups (`max_norm ∈ {10, 25, 50}`, AGC) deferred — higher-EV
  round-2 hypotheses have priority. If architecture/loss levers stall, we
  will return to AGC.

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


