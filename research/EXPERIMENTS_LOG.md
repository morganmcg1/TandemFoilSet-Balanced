# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-12 21:00 — PR #1555: Remove `in_project_fx` (Transolver++ tied projection) — **REQUEST CHANGES** (sent back to thorfinn for n_hidden=144 follow-up)

- Branch: `charliepai2g24h4-thorfinn/remove-in-project-fx`
- Hypothesis: H3 from round-2 list. Remove redundant `in_project_fx` from
  `PhysicsAttention`, re-using `x_mid` as the value source in the slice-pooling
  einsum (Transolver++, arXiv 2502.02414). Acts as a structural prior + frees VRAM.
- Run was on L1+stoch-depth base (post-#1552), so direct apples-to-apples vs current baseline.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 99.898 | 98.353 | **+1.57% (slightly worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 89.532 | 87.995 | +1.75% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 129.395 / 108.141 / 72.436 / 89.619 | 119.16 / 111.09 / 73.32 / 89.84 | **+8.60% / -2.65% / -1.21% / -0.25%** |
| n_params | 579,799 | 662,359 | **-12.5%** |
| Peak GPU memory | 39.63 GB | 42.11 GB | **-5.9%** |
| Wall time/epoch | ~125 s | ~123 s | ~unchanged |

- **Pattern is a classic capacity-vs-regularization tradeoff.** The three OOD-flavored
  splits (camber_rc, camber_cruise, re_rand) all improve modestly; single_in_dist
  regresses by +8.6%, pulling val_avg net negative. The tied projection acts as a
  structural regularizer that helps OOD but underfits the in-distribution mode.
  Efficiency gains are real: -12.5% params and -5.9% VRAM at identical wall time.
- **Action: sent back with re-tune spec** — keep the tied projection, but reinvest
  the freed parameter budget (~83k params) and VRAM headroom by widening
  `n_hidden=128 → 144`. This redistributes capacity across all weights rather than
  concentrating it in a single redundant projection. Expected: single_in_dist
  recovers toward 119, OOD gains preserved → net improvement vs 98.353. Student
  must rebase onto current HEAD to include the merged stoch-depth code.

## 2026-05-12 21:00 — PR #1514: Ada-Temp v2 (shared-across-heads Δτ) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/ada-temp` (v2 force-push)
- Hypothesis: v2 follow-up to test alphonse's own diagnosis that extra per-head
  Δτ capacity hurt cross-regime transfer. v2 uses `Linear(dim, 1)` (shared-heads).
- Run was on L1-only base (pre-#1552), so compared against 100.957 not 98.353.

| Metric | v2 | L1 baseline (#1397) | v1 (per-head) | Current baseline (#1552) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 104.366 | 100.957 | 101.770 | 98.353 |
| Δ vs L1 baseline | **+3.4% (worse)** | — | +0.81% | -2.58% |
| Δ vs current best | **+6.1% (worse)** | +2.65% | +3.47% | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 122.77 / 114.35 / 85.47 / 94.88 | 127.37 / 110.83 / 77.35 / 88.27 | 118.02 / 114.13 / 78.35 / 96.58 | 119.16 / 111.09 / 73.32 / 89.84 |
| Δ vs L1 per-split | -4.60 / +3.51 / **+8.12** / +6.60 | — | -9.35 / +3.30 / +1.00 / +8.31 | — |
| test_avg/mae_surf_p (4-split, NaN-safe) | 93.936 | NaN | NaN | 87.995 |

- **Both Ada-Temp variants are now exhausted.** v1 (per-head) regressed by +0.81%;
  v2 (shared-heads) regresses harder by +3.4% on val_avg.
- **The capacity-overfit hypothesis is partially contradicted.** Shared-heads
  narrowed v1's val_re_rand regression (+8.31 → +6.60) and partially preserved
  v1's val_single_in_dist gain (-9.35 → -4.60). But v2 introduced a new
  large regression on val_geom_camber_cruise (+1.00 → +8.12), which v1 didn't
  have. Removing per-head freedom collapses head specialization on the cruise
  regime that needed it most.
- **Action: CLOSED.** The NaN-safe pre-filter from this PR was independently
  preserved via #1552 (now standard in baseline). Slice-collapse is also being
  attacked via a different mechanism in #1553 (Gumbel-Softmax, WIP under nezuko).
  Alphonse's suggested follow-up (Eidetic Slice Embedding) goes on the
  wave-3 candidate pile for later revival if Gumbel-Softmax doesn't pan out.

## 2026-05-12 21:00 — PR #1547: Kendall uncertainty weighting — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/kendall-uncertainty`
- Hypothesis: H6 from round-2 list. Replace hand-tuned `surf_weight=10` with
  learnable per-task log-sigmas (Kendall et al., CVPR 2018) so the surf/vol
  balance becomes data-driven.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | 103.544 | 98.353 | **+5.28% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 94.524 | 87.995 | **+7.42% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 132.989 / 115.872 / 73.899 / 91.417 | 119.16 / 111.09 / 73.32 / 89.84 | +11.6% / +4.3% / +0.8% / +1.8% |
| Learned `log_sigma_surf` / `log_sigma_vol` | -0.288 / -0.079 | — | — |
| **Effective `surf_weight`** | **1.518** | 10.0 | -85% lower than hand-tuned |

- **Key diagnostic finding: the Kendall MLE objective is fundamentally misaligned
  with the physical evaluation metric.** Learned sigmas converged to
  effective_surf_weight=1.518, ~7× lower than the hand-tuned value of 10 that
  the baseline uses. Cross-referencing closed PRs #1403 (surf_weight=30, +5.1%
  worse) and #1530 (effective surf×P_WEIGHT=30, +1.22% worse), the empirical
  optimum for surf_weight is at or near 10, and learnable per-task likelihood
  pulls it the wrong way.
- **Lesson: learnable loss-balance objectives must align with the physical
  eval metric, not just calibrated likelihoods.** This rules out the entire
  family of MLE-style balance learning (Kendall, GradNorm, dynamic weight
  averaging) unless they're constrained to optimize the evaluation surrogate
  directly.
- **Action: CLOSED.** Clean negative result. No reasonable variant of the
  Kendall objective recovers the gap; the objective is the problem, not the
  parameterization.

## 2026-05-12 21:00 — PR #1545: Asymmetric Q/K slice projections (LinearNO) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/asymmetric-qk`
- Hypothesis: H2 from round-2 list. Independent V and K slice projections in
  PhysicsAttention (LinearNO-style) — separate the slice-assignment basis from
  the value basis to enable richer slice tokens.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 10) | 116.940 | 98.353 | **+18.90% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 105.058 | 87.995 | **+19.39% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 141.506 / 132.312 / 86.757 / 107.185 | 119.16 / 111.09 / 73.32 / 89.84 | +18.7% / +19.1% / +18.3% / +19.3% |
| n_params | 672,919 | 662,999 | +9,920 (+1.5%) |
| Epochs reached in 30-min cap | **10** | 15 | **-33%** |

- **Compute-bound failure mode.** The mechanism is empirically active (block-3
  slice cos-sim = 0.097 confirms slice divergence), but the extra `in_project_slice_k`
  projection adds ~40% wall-clock cost per epoch. Run terminated at epoch 10
  vs the baseline's 15 — same compute budget, fewer effective gradient steps.
- The trajectory was still descending at termination but needed ~17 additional
  MAE points of improvement to match baseline, which is implausible in the
  remaining 5 epochs even with monotonic descent.
- **Structural lesson: architectural changes that add >10% per-step compute
  are unviable in our 30-min training regime, even when the mechanism is
  theoretically sound.** Future architectural changes must be parameter-additions,
  not compute-additions, OR be paired with a complementary efficiency-saving
  (e.g., the tied-projection direction that thorfinn is iterating on in #1555).
- **Action: CLOSED.** Direction is dead within current budget constraints;
  asymmetric Q/K could only be re-attempted at higher budget or paired with a
  compute-saving change.

## 2026-05-12 20:52 — PR #1552: Stochastic depth (drop_rate=0.1, linear schedule) — **MERGED, new baseline**

- Branch: `charliepai2g24h4-frieren/stoch-depth-0.1`
- Hypothesis: H8 from round-2 list. Add stochastic depth (Huang et al., ECCV 2016)
  with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
  Implicit ensemble of shallower networks for OOD regularization. No-op at eval.
  Predicted 1-3% improvement on `val_avg/mae_surf_p`, primarily via OOD geometry splits.
- Also includes the NaN-safe pre-filter in `evaluate_split` (standardized in every
  round-2 PR after #1530/#1529 independently discovered it).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **98.353** | 100.957 | **-2.58% (improvement)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **87.995** | NaN (data bug) | **first finite 4-split ref** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 96.579 | 100.831 | -4.22% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 119.159 / 111.093 / 73.323 / 89.837 | 127.371 / 110.832 / 77.353 / 88.273 | **-6.45% / +0.24% / -5.21% / +1.77%** |
| Per-split test: single_in_dist / camber_rc / camber_cruise / re_rand | 104.953 / 101.883 / 62.243 / 82.901 | — | new finite ref |

- **The hypothesis held, but the OOD-specific framing was only half-supported.**
  Predicted gains were on OOD geometry splits (camber_rc, camber_cruise).
  Observed: camber_cruise -5.21% (large), camber_rc +0.24% (flat),
  single_in_dist -6.45% (largest gain), re_rand +1.77% (small regression).
  Student's reading: single_in_dist was the worst split at baseline despite
  being in-distribution, so it had the most regularization headroom.
  Stoch-depth's implicit ensemble flattens split-specific overfit modes
  regardless of the OOD axis.
- **Training dynamics:** val trace is noisier than L1 baseline (epoch 13: 105.69
  → epoch 14: 113.91 → epoch 15: 98.35 = new best). Bernoulli-block-drop noise
  injects variance into val. Best epoch landed at the wall-clock cap; more
  training time would likely extend the gain. The L1 baseline plateaued earlier
  at the same wall-clock budget, so stoch-depth is also getting more out of
  each minute of training.
- **Cosmetic NaN caveat:** loss/surf_loss aggregates for `test_geom_camber_cruise`
  still show NaN/Inf in `metrics.yaml` because the normalized-space loss path
  runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
  All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary ranking
  metric is clean. Out of scope; one-line follow-up.
- **Decision: MERGED.** First post-L1 architectural improvement; -2.58% on the
  primary metric and establishes the first finite 4-split test reference
  (87.995). Stoch-depth is now part of the canonical config; all subsequent
  wave-1 PRs in flight will be compared to this stronger baseline.
- **Suggested follow-ups (student):**
  1. Run longer — not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).
  2. Sweep `drop_rate` ∈ {0.05, 0.15, 0.20} — 0.05 might be Pareto-better given
     val_re_rand +1.77% suggests slight over-regularization; 0.15-0.20 might
     bite harder on val_geom_camber_rc which barely moved.
  3. Combine with `dropout` inside PhysicsAttention/MLP at 0.05 — standard
     ViT recipe, may compound with stoch-depth.
  4. Loss-NaN cosmetic fix — pre-filter finite samples before `y_norm` is
     formed so the normalized-space loss aggregates report finite numbers
     for `test_geom_camber_cruise`.

## 2026-05-12 20:02 — PR #1514: Ada-Temp per-point adaptive slice temperature — **REQUEST CHANGES** (sent back to alphonse for v2)

- Branch: `charliepai2g24h4-alphonse/ada-temp`
- Hypothesis: H1 from round-2 list. Replace scalar `self.temperature` with
  `τᵢ = τ₀ + Linear(dim, heads)(xᵢ)`, zero-init the projection so the model
  starts identical to baseline (Transolver++, arXiv 2502.02414).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | 101.770 | 100.957 | +0.81% (slightly worse) |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.825 | 100.831 | -0.007 (effectively flat) |
| test_avg/mae_surf_p (4-split) | NaN (no NaN-safe fix in v1) | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 118.023 / 114.128 / 78.348 / 96.582 | 127.371 / 110.832 / 77.353 / 88.273 | **-9.3 / +3.3 / +1.0 / +8.3** |

- **Per-split signal is the key story.** Ada-Temp helps single-foil in-distribution
  by ~9.3 (~7.3% gain) but regresses on val_re_rand by ~8.3 (~9.4% loss). The
  geometry-camber splits drift slightly worse. Net val_avg is essentially flat
  (slight regression) and test 3-split mean is statistically indistinguishable.
- **Implementation contribution worth recording**: alphonse identified that
  `Transolver.__init__` calls `self.apply(self._init_weights)` *after* `temp_proj`
  is zero-initialized, and `_init_weights` re-initializes every `nn.Linear` with
  `trunc_normal_(std=0.02)`. This silently breaks the "Δτ = 0 at step 0" invariant.
  Fix: re-zero loop after `self.apply(...)`. Without the fix an earlier run
  diverged from baseline from epoch 1. The committed run is the corrected version.
- **Diagnosis (student): extra per-head Δτ capacity hurts cross-regime transfer**
  inside a 30-min wall-clock budget. Single-foil in-dist benefits from sharper
  slice attention; tandem-flow OOD distributions cannot afford the extra
  capacity that lets the temperature head co-adapt to training-set spurious cues.
- **Action: sent back with v2 spec** — drop `temp_proj` from `Linear(dim, heads)`
  to `Linear(dim, 1)` (shared-across-heads Δτ), which cuts Ada-Temp's added
  capacity by ~75% (2,580 → 645 params). Direct test of the student's own
  capacity-overfit hypothesis. Also adds the NaN-safe pre-filter so v2 will
  report a finite 4-split test mean. Student suggested 4 follow-ups; v2 picks
  #2 (shared-across-heads), with #3 (last-blocks-only) as a stack-on if v2
  partially works and #4 (combine with Eidetic Slice Embedding) as a
  wave-3 candidate if v2 fails. Suggestion #1 (length-budgeted retest)
  is not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).

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


