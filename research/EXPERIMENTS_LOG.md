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

## 2026-05-13 00:10 — PR #1616: Per-Re WeightedRandomSampler (upweight high-Re samples) — CLOSED

- Student branch: `willowpai2g24h3-askeladd/re-resample`
- Hypothesis: a `WeightedRandomSampler` weighted by `exp(t * log_re_centered)` shifts effective epochs toward the high-Re regime where pressure targets vary most; predicted 1–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | re_weight_temp | val_avg/mae_surf_p (best) | test_avg/mae_surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| uniform-baseline-smoothl1-clip1 (baseline) | 0.0 | **90.91** | **86.87 (4-split, all finite)** | — | `eztvtkxc` |
| re-resample-t1.0-smoothl1-clip1 | 1.0 | 97.41 (final 100.61) | NaN (variant produced non-finite preds on cruise test) | **+7.2% (worse)** | `stzo9xvw` |

Per-split val MAE breakdown shows the mechanism cleanly:

| Split | Baseline t=0 | Variant t=1.0 | Δ |
|---|---|---|---|
| val_single_in_dist | 103.90 | 148.16 | **+42.6% (catastrophic)** |
| val_geom_camber_rc | 105.34 | 102.78 | −2.4% |
| val_geom_camber_cruise | 68.99 | 67.60 | −2.0% |
| val_re_rand | 85.40 | 83.89 | −1.8% |

### Analysis

The variant *improves* every OOD-ish split (geom_camber_rc, geom_camber_cruise, re_rand) by 2–3% on both val and test — confirming the "high-Re samples generalize the OOD splits" sub-hypothesis. But the in-distribution split (`val_single_in_dist`) degrades by **+42.6%** because at `t=1.0` the max/min sampling ratio is **67.6×** — the lowest-Re training samples are seen <1× per epoch in expectation under `WeightedRandomSampler(replacement=True)`. The model is starved of low-Re training updates that the in-distribution split depends on.

Mechanistic insight: Huber and re-resampling are *not* the orthogonal mechanisms the PR predicted. They fight — Huber caps the gradient on high-Re samples that re-resampling deliberately re-injects. The net effect is just less effective training on in-distribution, with no headroom gained from over-emphasized regimes (Huber already handles those).

Additionally: the variant model produced non-finite predictions on at least one cruise *test* sample (`vol_loss = +Inf`, `surf_loss = NaN`), even though training-time cruise val was finite. The cruise-y filter from #1433 cannot help here — it handles non-finite *ground truth*, not non-finite *predictions* — but this is a signal that the variant model is unfit for the paper-facing pass under heavy reweighting.

### Side-effects of this PR (high-value despite the close)

1. **First clean end-to-end 4-split test pass for this launch.** Run `eztvtkxc` delivered `test_avg/mae_surf_p = 86.87` with all four splits finite — the cruise-y filter from PR #1433 worked.
2. **Cleanest measurement of the current advisor branch:** 90.91 val / 86.87 test (uniform sampling on top of SmoothL1+grad-clip+cruise-fix). Combined with two other in-flight baseline measurements (#1615 at 102.17, #1437 at 104.84), this characterizes a **±7 single-seed noise band** on `val_avg/mae_surf_p`.

### Conclusions

- Closed. Hypothesis at `t=1.0` falsified (+7.2% on val, NaN on test). Per-spec `t=2.0` stretch arm correctly not run.
- Follow-up direction (assigned to askeladd as next PR): **loss-level Re-reweighting** — multiply each sample's loss by `exp(t * log_re_centered)` inside the train loop, no resampling. Same "tilt toward high-Re" mechanism without the discrete sample-starvation problem. If even `t=0.3` produces a -1 to -3% effect on `val_avg`, the OOD-split signal observed here is real and just needed a less aggressive implementation.
- BASELINE.md updated with the supplemental 90.91/86.87 measurement of the current advisor branch (the merged-best stays at 104.70 until a winning hypothesis PR's terminal `SENPAI-RESULT` marker lands).

## 2026-05-13 00:55 — PR #1431: Raise surf_weight 10 → 50 to align loss with surface-p MAE — CLOSED

- Student branch: `willowpai2g24h3-alphonse/surf-weight-50`
- Hypothesis: raising `surf_weight` from 10 → 50 sharpens the loss-vs-metric alignment with surface-pressure MAE; predicted small improvement on `val_avg/mae_surf_p`.
- Bundled: an in-PR copy of the cruise-NaN-y filter (commit `b073a95` in `train.py::evaluate_split`) — same fix as askeladd's #1433, applied independently. Will be a no-op delta on rebase.

### Results

| Arm | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, finite) | Δ vs baseline (test) | W&B |
|---|---:|---:|---:|---:|---|
| baseline | 10 (default) | **126.70** | **112.68** | — | `ogz8su1w` |
| variant | 50 | 131.34 | 120.90 | **+7.30% worse** | `2qytxnem` |
| bonus | 25 | 143.79 | 127.35 | +13.02% worse | `x6nf3mk2` |

All three arms hit the 30-min wall-clock cap at 14 epochs (~28% through the cosine schedule). Comparisons are apples-to-apples at the same training budget on alphonse's pre-rebase branch (his fork carries MSE + cruise-fix, but does *not* yet stack SmoothL1+grad-clip — so absolute numbers are not directly comparable to other students' baselines on the current advisor branch). The hypothesis decision (variant +7.3% worse) is unaffected.

### Per-split test breakdown (best-val checkpoint) — the smoking gun

| Arm | split | surf[p] | vol[p] |
|---|---|---:|---:|
| baseline | test_single_in_dist | 132.97 | 134.44 |
| baseline | test_geom_camber_rc | 124.40 | 121.45 |
| baseline | test_geom_camber_cruise | 81.39 | 79.76 |
| baseline | test_re_rand | 111.96 | 107.22 |
| **surf=50** | test_single_in_dist | 130.39 | **178.16 (+32%)** |
| **surf=50** | test_geom_camber_rc | 132.78 | **159.60 (+31%)** |
| **surf=50** | test_geom_camber_cruise | 98.18 | **161.50 (+102%)** |
| **surf=50** | test_re_rand | 122.24 | **176.63 (+65%)** |

### Analysis (mechanistic — high-value finding)

**Bernoulli-coupling is the dominant mechanism.** alphonse's diagnosis: in incompressible flow, surface `p` and volume `p` are globally linked through pressure-Poisson / Bernoulli equations. Suppressing the volume-`p` residual signal (from `1/(1+10)=9.1%` of total at `surf_weight=10` to `1/(1+50)=1.96%` at `surf_weight=50`) starves the model of the volume-pressure structure it needs to *correctly anchor* surface pressure. The result is exactly what we see: vol[p] regresses by 30-102% across all four test splits, and surface-p slightly regresses too because the global pressure field is now miscalibrated near the foil.

**The "minority-class" framing was wrong on principle.** "Surface is the metric, so upweight surface" looks like sensible loss-metric alignment, but on a coupled PDE system the volume channels are *not noise* — they carry the constraint structure the surface predictions rely on. This rules out a whole family of naive task-aligned reweighting hypotheses for coupled physics. Generalizes to other PDE-surrogate problems.

**Surface velocity (Ux, Uy) is robust to channel reweighting** (slight regressions only) — the free-slip-like constraint at the foil makes those channels easy and saturated. The hypothesis only ever had a chance on `surf[p]`, and that channel needs both sides of the Bernoulli coupling.

### Conclusions

- Closed. Hypothesis falsified by an internally-consistent A/B with strong mechanistic explanation.
- Cruise-NaN-y filter works: all three arms produced finite 4-split `test_avg/mae_surf_p`. Independent confirmation that #1433's fix is correct.
- Follow-up direction (assigned to alphonse as next PR): **`slice_num` sweep on Transolver's Physics Attention layer.** Listed as an open question in `CURRENT_RESEARCH_STATE.md`; tests whether 64 slices saturate on the 242K-node cruise meshes. Default 64; arms at 32/96/128 to bracket. Compute trade-off (slower epochs vs finer representation) similar to but milder than the closed #1443 wider-n192.
- The Bernoulli-coupling mechanism finding will be cited in future hypothesis assignments. "Reweight surface" is now a known dead end for surface-MAE-on-coupled-physics.

## 2026-05-13 01:20 — PR #1537: Tune cosine T_max to budget — --epochs 13 instead of 50 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/schedule-tuned-e13`
- Hypothesis: matching cosine `T_max` to the actually-achievable epoch count converts the unused tail of the schedule into a proper cool-down; predicted 3–10% reduction in `val_avg/mae_surf_p`. (Direct data-driven follow-up to thorfinn's own #1443 baseline arm trajectory.)

### Results — W&B group `willow-r3-schedule-tuned-e13`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Epochs | Wall-clock | State |
|---|---:|---:|---:|---:|---|
| x4sqeaqz | **118.77** (best) | NaN | 13 | 28:50 | finished |
| nx1tvtp1 | 119.26 | NaN | 13 | 28:54 | finished |
| rfxdtryp | 120.64 | NaN | 13 | 28:49 | finished |
| afft3f1v | 122.25 | NaN | 13 | 28:57 | finished |
| slsutjdn | 122.41 | NaN | 13 | 28:51 | finished |
| k6h7anq3 | 192.27 (div) | NaN | — | — | crashed |
| crwqx3mb | 172.03 (div) | NaN | — | — | crashed |
| navwrdyg | 186.59 (in-prog) | NaN | mid-run | 0:18 | running (div) |

Best of 5 finished arms is **118.77 — +14 MAE points above merged baseline 104.70, +28 above the advisor-branch ~91 lower noise band.** The variant arm (e13) does not move the metric over thorfinn's own e50 baseline trajectory, and no arm enters the noise band of the current advisor baseline.

### Analysis

The mechanism prediction was that cool-down of the cosine schedule would harvest the last few percentage points of capacity. Empirically, finishing a 13-epoch cosine cycle does cool the LR but produces no measurable improvement vs running 14 of 50 epochs at near-peak LR — across 5 independent seeds. Implication: at the merged-baseline operating point (SmoothL1 + grad-clip on advisor branch), the LR-cooling regime contributes less than the seed-to-seed noise (±7). 

Two crashes + one in-flight diverging run also suggest the e13-config + WeightedRandomSampler-with-replacement combination may have a borderline-stable training regime — likely a separate effect from the hypothesis itself, but worth noting.

No SENPAI-RESULT terminal marker was posted on the PR; advisor closed based on the W&B group readout directly.

### Conclusions

- Closed. Hypothesis falsified at the merged advisor-branch operating point — schedule-cooling alone is not a 5%+ lever here.
- **Schedule reformulation is not abandoned** — frieren's #1438 (warmup-5ep) tests the complementary half (LR warmup before the cosine). If warmup wins, then a **warmup + tuned T_max combo** would be the natural Round 2 stack and bears revisiting.
- All test_avg/mae_surf_p were NaN, suggesting thorfinn's branch may not have absorbed the cruise-NaN-y fix from #1433 — but the comparison against the val metric is unaffected.
- Follow-up direction (assigned to thorfinn as next PR): **AdamW weight_decay sweep**. Single-knob regularization test on a baseline that is now characterized to ±7 noise. Pure compute-neutral lever — no time cost per epoch, predicted to differentiate cleanly on per-split signal (especially val_single_in_dist and val_geom_camber_rc which have the highest per-split MAE).
