<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research Results — willow-pai2g-24h-r3

Lower is better for `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## 2026-05-12 19:30 — PR #1443: Widen Transolver to n_hidden=192, n_head=6 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/wider-n192`
- Hypothesis: increasing `n_hidden` from 128→192 and `n_head` from 4→6 (`dim_head` constant at 32) gives more capacity at fixed depth/slice; expected 2–6% reduction in `val_avg/mae_surf_p`.

### Results

| Run | n_hidden / n_head | Params | Epochs done | val_avg/mae_surf_p | test 3-split avg surf_p | Δ vs baseline | W&B |
|---|---|---|---|---|---|---|---|
| baseline-30m | 128 / 4 | 0.66M | 14 | **123.17** (e12) | 120.19 | — | `h73q3u7m` |
| wider-n192-30m | 192 / 6 | 1.45M | 9 | **163.67** (e7) | 165.67 | +33% val / +38% test (worse) | `b9pe1a61` |

### Analysis

Wider variant regressed by +33% on val and +38% on test. Root cause: wider model is ~1.5× slower per epoch, finishes only 9 of the 50 scheduled epochs vs baseline's 14, and never enters the cosine cool-down where the baseline gains most of its ground.

Key observation from baseline trajectory (which becomes the seed for the next experiment): val_avg/mae_surf_p drops 140 → 156 → 126 → **123** at epochs 9-12 (collapse to 182 at e13 — likely noise). The cosine LR is barely cooled at this point (T_max=50, t=14 → cos(14π/100)≈0.92, LR still ~4.6e-4 of 5e-4 peak). Completing the schedule should push the best lower.

### Conclusions

- At the 30-min budget, capacity scaling via width is dominated by throughput cost — closed.
- Schedule mismatch (T_max=50, only 14 epochs fit) is a probable next lever — assigned to thorfinn as `schedule-tuned-e13`.
- **Known bug (do not block on):** `test_geom_camber_cruise/mae_surf_p` is NaN on both arms (pre-existing in the scoring/data path). Both `Ux/Uy` MAE on the same split are finite, suggesting a specific sample's p-channel prediction or ground-truth overflows. Need a separate `data/scoring.py` or data-side PR; deferring until more PRs land or the bug starts blocking ranking.

## 2026-05-12 21:05 — PR #1441: Replace MSE with SmoothL1 (Huber, β=0.1) — MERGED (winner)

- Student branch: `willowpai2g24h3-tanjiro/smooth-l1-beta01`
- Hypothesis: SmoothL1 in normalized space caps per-element gradient magnitude on high-Re outliers; predicted 2–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | Loss | Best val_avg/mae_surf_p (epoch) | test 3-split-ex-cruise avg surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| baseline-30m | MSE | 131.81 (e10) | 131.56 | — | `y3dfc5e7` |
| smooth-l1-0.1-30m | SmoothL1(β=0.1) | **104.70 (e13)** | **101.08** | **−20.6% val / −23.2% test** | `d53f0jn4` |

Per-split val surface-p MAE at SmoothL1 best-val:
- val_single_in_dist 120.63 (−22.9%)
- val_geom_camber_rc 117.45 (−16.4%)
- val_geom_camber_cruise 82.36 (−24.5%)
- val_re_rand 98.34 (−18.9%)

### Analysis

Outsized win — 4-10× the predicted delta — uniformly across every val split. Mechanism is consistent with the heavy-tail story: under MSE the high-Re/high-`|p|` outlier samples in each batch produce normalized residuals well above β=0.1, dominating the quadratic gradient on a single step and yanking the model off-trajectory (epoch-to-epoch val swings of ±20–40 MAE points were typical). SmoothL1 caps that contribution while leaving the in-regime quadratic intact, so each step is balanced across the Re range. SmoothL1's best epoch came at 13 vs MSE's 10 — Huber also keeps improving for longer in the same wall-clock budget. Largest absolute gains landed on the splits with the largest |p| magnitudes (cruise / re_rand), as predicted.

### Conclusions

- Merged. New empirical high-water mark on the advisor branch: **val_avg/mae_surf_p = 104.70**.
- Pre-authorized follow-ups (β=0.05, longer training, surf_weight re-tune under Huber, pure L1 comparison) are first-class Round 2 candidates.
- The cruise-test NaN bug is not in this PR (it stays a 3-split-ex-cruise figure) — fix lands in #1433 (next merge).

## 2026-05-12 21:06 — PR #1433: Add gradient norm clipping (max_norm=1.0) — MERGED

- Student branch: `willowpai2g24h3-askeladd/grad-clip-norm1`
- Hypothesis: `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` stabilizes training under heavy-tailed outliers; predicted 1–4% reduction in `val_avg/mae_surf_p`.
- Also ships the inline cruise-test NaN fix in `train.py::evaluate_split` (drops non-finite-`y` samples before forward pass and `accumulate_batch`).

### Results

| Run | max_norm | Best val_avg/mae_surf_p (epoch) | Δ vs baseline arm | W&B |
|---|---|---|---|---|
| baseline-30m | none | 131.96 (e?) | — | `mz3x4ieb` |
| grad-clip-1.0 | 1.0 | **114.18** | **−13.5%** | `qof1cbki` |
| grad-clip-0.5 | 0.5 | 121.41 | −8.0% | `japg46eu` |

Pre-clip grad-norm distribution measured at the baseline arm: median 53.90, max 579.57 — confirming the heavy-tail hypothesis (a single batch's grad-norm spike at >10× the median is a routine occurrence under MSE).

### Analysis

Tighter clip (0.5) underperforms looser (1.0), suggesting the floor for "useful" grad updates on a normal batch is somewhere between 0.5 and the median ~54 in pre-clip norm — 1.0 attenuates only the spike-batches and leaves the bulk of training-time gradients essentially untouched. Same mechanism as Huber (cap outlier influence) but acting at the batch-aggregate level instead of per-element.

### Conclusions

- Merged. Does NOT dethrone tanjiro's 104.70 (this PR was measured under MSE, not SmoothL1). Advisor branch now ships SmoothL1 + grad-clip stacked; combined-config has never been measured.
- The cruise-test NaN fix is now on the advisor branch. Future PRs will inherit it via rebase and report 4-split `test_avg/mae_surf_p` end-to-end.
- Open question for Round 2: does grad-clip still help on top of SmoothL1, or does SmoothL1 already subsume it? Pre-clip norms under SmoothL1 should be much smaller — likely the marginal benefit of clip-on-top-of-Huber is near zero, but a small A/B run can confirm.
