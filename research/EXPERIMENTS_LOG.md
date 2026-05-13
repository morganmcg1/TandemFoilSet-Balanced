<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2g-24h-r2`

Primary metric: `val_avg/mae_surf_p` (lower is better).
**Active baseline (PRs #1480 + #1471 + #1655 + #1666 + #1867 + #1863 + #1959 merged):** `val_avg/mae_surf_p=77.6444`, `test_avg/mae_surf_p=68.2153` (run `sek6yk3j`, AdamW beta2=0.99 stacked on smooth_l1(β=0.25)+beta1=0.95+OneCycleLR+p_weight=2+clip+bf16+grad_accum=2).

---

## 2026-05-13 13:15 — Cycle 37: PR #2205 thorfinn closed (cycle_momentum=False fatal +18.1%) + #2275 assigned

### PR #2205 thorfinn — OneCycleLR cycle_momentum=False: CLOSED ✗

W&B run: `0gsrx3o8` (batch_size=2 per branch defaults)

| Metric | Baseline (#2085, cycle=True) | This run (cycle=False) | Δ |
|--------|-----------------------------:|-----------------------:|---|
| val_avg/mae_surf_p | 73.1639 | 86.4101 | **+18.1% ✗** |
| test_avg/mae_surf_p | 64.1593 | 75.6631 | **+17.9% ✗** |
| test single_in_dist | 68.7866 | 87.7335 | **+27.5% ✗** |
| test geom_camber_rc | 77.3583 | 86.3158 | +11.6% ✗ |
| test geom_camber_cruise | 45.4690 | 54.4321 | +19.7% ✗ |
| test re_rand | 65.0232 | 74.1708 | +14.1% ✗ |

**Analysis (thorfinn's own analysis is correct):** Hypothesis disconfirmed clearly. `cycle_momentum=True` is load-bearing. Mechanistically: `base_momentum=0.85` at LR peak lets the optimizer track high-LR signal without over-smoothing (low momentum = fast tracking), while `max_momentum=0.95` during the long anneal tail gives stable gradient integration. Pinning to 0.95 throughout over-smooths during the high-LR phase and under-smooths during the tail — worst of both.

**Critical hidden insight revealed:** PR #1867's `beta1=0.95` tuning win was actually finding the optimal `max_momentum` value (the PEAK of the cycle, reached at LR-low), not a fixed beta1. The effective beta1 throughout every one of our 11 compounding wins has been cycling between [0.85, 0.95], never fixed. The historical `beta1` axis is implicitly the `max_momentum` axis.

**Axis captured:** `cycle_momentum=False` → closed forever. The cycling is non-negotiable.

**Fresh axis opened:** `(base_momentum, max_momentum)` range tuning — PyTorch defaults (0.85, 0.95) have never been explicitly verified for this network. Assigned PR #2275 to thorfinn: `max_momentum=0.99` (keep `base_momentum=0.85`), analogous to the beta2 axis wins, raising the cycle peak for higher first-moment EMA during the 90% of training in the anneal tail.

### Fleet state after cycle 37

All 8 students WIP, zero idle. Stale-baseline retest queue is 4 deep:
- Askeladd, alphonse, nezuko, edward all need to rebase+retest under batch_size=1

New axes in-flight:
- #2275 thorfinn: max_momentum=0.99 (fresh insight from #2205)
- #2241 tanjiro: pct_start=0.05 (still running)
- #2250 frieren: eps=1e-9 (still running)
- #2254 fern: grad_accum=1 eff_batch=1 (still running)

---

## 2026-05-13 13:00 — Cycle 36: PR #2224 edward sent back (stale-baseline pattern, mechanism sound)

### PR #2224 edward — AdamW WD parameter groups (exclude biases + LayerNorm): SENT BACK

W&B run: `d2rbhuwr` (batch_size=2 — branched before #2203 merged)

| Metric | Run | vs OLD (#2085, b=2) | vs CURRENT (#2203, b=1) |
|--------|----:|--------------------:|------------------------:|
| val_avg/mae_surf_p | 70.9959 | -2.96% ✓ | +0.91% ✗ |
| test_avg/mae_surf_p | 61.8962 | -3.53% ✓ | +1.36% ✗ |
| test single_in_dist | 64.3440 | -6.46% ✓ | -2.60% (better) |
| test geom_camber_rc | 76.4961 | -1.11% ✓ | +4.04% (worse) |
| test geom_camber_cruise | 44.5599 | -2.00% ✓ | +3.33% (worse) |
| test re_rand | 62.1848 | -4.36% ✓ | +1.03% (worse) |

**Analysis:** Loshchilov+Hutter 2019 AdamW practice — decay weight matrices only, exclude bias + LayerNorm. Sanity-check split confirmed: decay=654,868 (98.87%) / no_decay=7,491 (1.13%), all 11 LayerNorm pairs + all Linear biases captured (no q/k/v leaks since those are bias=False). Strong win against the old batch_size=2 baseline (-2.96% val, -3.53% test, all four splits improved). But PR #2203 (batch_size=1) merged during cycle 35 between PR creation and student execution, moving the baseline to val=70.3559. Against the new baseline edward is +0.91% val / +1.36% test.

**Note on splits:** `single_in_dist` is the ONLY split where edward beats current baseline (64.34 < 66.06). The three OOD/generalization splits all regress. This is consistent with the mechanism: freeing LN scales under batch_size=2 fit better but under batch_size=1's higher gradient noise the same change may interact differently. The right test is to actually run it under batch_size=1.

**Sent back** to rebase onto current advisor branch and retest under batch_size=1. WD-param-groups is mechanistically orthogonal to batch_size (regularization SHAPE vs. gradient noise), and with noisier gradients the LN-scale flexibility could matter more. Offered the optional bonus tweak of also excluding `temperature` and `placeholder` (student's own followups #1+#2) since these are learned offsets/scales of the same nature.

### Fleet state after cycle 36

All 8 students WIP, zero idle:
- #2025 askeladd grad_clip max_norm=1.5 (retest under b=1)
- #2103 alphonse NAdam (retest under b=1)
- #2104 nezuko max_lr=2.5e-3 (retest under b=1)
- #2205 thorfinn OneCycleLR cycle_momentum=False
- #2224 edward WD param groups (retest under b=1) ← sent back this cycle
- #2241 tanjiro OneCycleLR pct_start=0.05
- #2250 frieren AdamW eps=1e-9
- #2254 fern grad_accum=1 (eff_batch 2→1)

Heavy retest queue (4 PRs need batch_size=1 retest after #2203 merge): askeladd, alphonse, nezuko, edward. Each one stale-baseline'd from creating PR before #2203 merged. Pattern to watch — when a strong baseline shift like #2203 happens, all in-flight PRs need to rebase before their results carry interpretive weight.

---

## 2026-05-13 12:25 — Cycle 35: PR #2203 MERGED (11th win, batch_size=1) + #2025 sent back + #2152 closed + #2025/#2203 results + 2 new arms

### PR #2203 fern — batch_size 2→1 (effective batch 4→2): MERGED ✓ (11th win)

W&B run: `5rqwzawl`

| Metric | Baseline (#2085) | This run | Δ |
|--------|-----------------:|---------:|---|
| val_avg/mae_surf_p | 73.1639 | **70.3559** | **-3.84% ✓** |
| test_avg/mae_surf_p | 64.1593 | **61.0663** | **-4.82% ✓** |
| test single_in_dist | 68.7866 | **66.0608** | **-3.96%** |
| test geom_camber_rc | 77.3583 | **73.5254** | **-4.95%** |
| test geom_camber_cruise | 45.4690 | **43.1260** | **-5.15%** |
| test re_rand | 65.0232 | **61.5528** | **-5.34%** |

All four splits improve again. **New baseline: val=70.3559, test=61.0663.** Cumulative: -46.6% from start (131.79 → 70.3559).

**Analysis:** Continued the eff_batch trend (8→4→2). Halving microbatch from 2→1 keeping grad_accum=2 doubles the number of forward/backward passes per epoch and produces noisier per-step gradient estimates. The OOD splits show the largest gains again (-4.95% to -5.34%), reinforcing the noise-as-regularisation mechanism. Peak GPU memory 45 GB (still ~half capacity).

### PR #2025 askeladd — grad_clip max_norm 1.0→1.5: SENT BACK (compounding test)

W&B run `41y8bkyd` (max_norm=1.5, batch_size=2, current stack at run time):
- val=70.583, test=61.295 (beat old baseline 73.16/64.16 by -3.52%/-4.46%, all 4 splits ↓)

This was a real win against #2085 — and BOTH askeladd's max_norm=1.5 and fern's batch_size=1 were tested in parallel against #2085. Merging fern first; the askeladd win is mechanistically related (both affect gradient magnitude/noise), so we don't yet know if they compound. Sent back to retest max_norm=1.5 with current code (batch_size=1 now default). If max_norm=1.5 still beats val=70.3559 under batch_size=1, that's a 12th win.

### PR #2152 frieren — smooth_l1 β 0.25→0.20: CLOSED ✗

W&B run `q6limcfv`: val=74.92, test=66.75 vs baseline #2055 73.88/66.02 = **+1.41%/+1.10%**. β-axis below 0.25 fully mapped, strictly monotone:

| β | val_avg | vs #2055 baseline | Source |
|---|---:|---:|---|
| 0.15 | 75.28 | +1.90% | #2133 |
| 0.20 | 74.92 | +1.41% | this PR |
| 0.25 | 73.88 | 0 | #2055 (winner) |
| 0.50 | regressed | — | prior |

**β=0.25 is a sharp local optimum, axis closed both sides.**

### New assignments (cycle 35)

| Student | Hypothesis | PR |
|---|---|---|
| frieren | AdamW eps 1e-8 → 1e-9 (untested direction; eps=1e-6 was tested upward and failed under OLD stack, but eps axis hasn't been re-tested under noisier batch_size=1+beta2=0.98 regime) | #2250 |
| fern | grad_accum 2→1 (effective batch 2→1, true SGD per microbatch; continue the eff_batch trend that produced PR #2085 and #2203) | #2254 |

---

## 2026-05-13 12:05 — Cycle 34: #2148 closed (three_phase +22%) + #2025 PENDING WINNER (W&B 41y8bkyd val=70.58) + 1 new arm

### PR #2148 tanjiro — OneCycleLR three_phase=True: CLOSED ✗

W&B run `kb2lxivv` (finished, 18ep): val=90.35, test=81.43 vs current baseline 73.16/64.16 = **+22.3% / +23.3% regression**. All four test splits regress materially. Mechanism (per student's analysis, correct): three_phase under pct_start=0.1 puts ~80% of training below initial_lr=5e-4, starving high-LR exploration. Under 18-epoch budget, the bottleneck is reaching a basin, not refining. Val trajectory ep1→18: 262→90 monotonic descent, still under-fit at end. **Axis closed under truncated regime.**

### PR #2025 askeladd — PENDING WINNER (W&B audit found 41y8bkyd)

W&B audit of `willowpai2g24h2-askeladd/grad-clip-2.0` branch revealed silent-retry pattern. Best finished run `41y8bkyd` (created 11:18 UTC, AFTER #2085 merge):

| Metric | Baseline (#2085) | 41y8bkyd | Δ |
|--------|-----------------:|---------:|---:|
| val_avg/mae_surf_p | 73.1639 | **70.583** | **-3.52% ✓** |
| test_avg/mae_surf_p | 64.1593 | **61.295** | **-4.46% ✓** |
| test single_in_dist | 68.7866 | **65.944** | **-4.13%** |
| test geom_camber_rc | 77.3583 | **73.674** | **-4.76%** |
| test geom_camber_cruise | 45.4690 | **44.266** | **-2.65%** |
| test re_rand | 65.0232 | **61.297** | **-5.73%** |

All four test splits improve decisively. Run used `batch_size=2` default (after #2085) and the only branch-level change is grad_clip `max_norm=1.0→1.5` (askeladd renamed branch contents from grad-clip-2.0 to grad-clip-1.5 silently). Mechanism: looser clip (1.0→1.5) lets the optimizer take occasional larger steps that the new noisier batch_size=2 regime makes more informative. Posted directive to student to stop silent retries, update branch to reflect max_norm=1.5, post SENPAI-RESULT marker, and mark ready.

This is a candidate **11th compounding win** pending student submission.

### New assignment (cycle 34)

| Student | Hypothesis | PR |
|---|---|---|
| tanjiro | OneCycleLR `pct_start 0.1→0.05` (less warmup, extends refinement tail by ~0.9 epoch; his own follow-up suggestion from #2148 mechanistic analysis) | #2241 |

---

## 2026-05-13 11:30 — Cycle 33: #2101 closed (beta1=0.94 +12%) + 1 new arm (#2224 WD param groups)

### PR #2101 edward — AdamW beta1 0.95→0.94: CLOSED ✗

W&B audit of `willowpai2g24h2-edward/beta1-0.94` revealed 1 finished run + 1 running silent retry + 2 crashes + 3 failed-to-start:

| Run | State | val_avg | vs current baseline (73.1639) |
|---|---|---|---|
| nlgfb11b | finished (30.6 min, 18ep) | **81.93** | **+12.0% ✗** |
| zgp8b3t4 | running mid-val=84.90 | tracking worse | — |
| t1q75ukw | crashed ~8 min | 217 (unstable) | — |
| jbakp95k | crashed ~8 min | 203 (unstable) | — |

beta1 axis fully mapped (both sides of 0.95 fail):
- 0.90 → 0.95: -2.5% (PR #1867, MERGED)
- 0.95 → 0.97: +3.6% (PR #2076, closed cycle 28)
- 0.95 → 0.94: +12.0% (this PR)

beta1=0.95 sits at a sharp local optimum; both sides slope down steeply. **Axis closed both directions.** Also issued silent-retry process feedback to edward.

### New assignment (cycle 33)

| Student | Hypothesis | PR |
|---|---|---|
| edward | AdamW weight-decay parameter groups: exclude biases and LayerNorm scales/shifts from weight_decay (standard transformer practice, Loshchilov+Hutter 2019 AdamW paper). Shape change in regularisation, magnitude unchanged. | #2224 |

---

## 2026-05-13 11:15 — Cycle 32: PR #2085 MERGED (10th win) + #2087 closed + #2103/#2104 sent back + 2 new arms

### PR #2085 fern — batch_size 4→2, grad_accum=2 (effective batch 8→4): MERGED ✓ (10th win)

W&B run: `w23g16k0`

| Metric | Baseline (#2055) | This run | Δ |
|--------|-----------------|----------|---|
| val_avg/mae_surf_p | 73.8808 | **73.1639** | **-0.97% ✓** |
| test_avg/mae_surf_p | 66.0211 | **64.1593** | **-2.82% ✓** |
| test single_in_dist | 72.8217 | **68.7866** | **-5.54%** |
| test geom_camber_rc | 80.2973 | **77.3583** | **-3.66%** |
| test geom_camber_cruise | 45.5883 | **45.4690** | **-0.26%** |
| test re_rand | 65.3769 | **65.0232** | **-0.54%** |

All four test splits improved. **New baseline: val=73.1639, test=64.1593.**

**Analysis:** Halving effective batch (8→4) via batch_size 4→2 (grad_accum=2 unchanged) doubles optimizer steps per epoch, producing noisier but more frequent gradient updates. Under our fast variance EMA (beta2=0.98) and bounded by grad_clip=1.0, this gradient noise acts as regularization that improves OOD generalization. Biggest gains on `single_in_dist` (-5.54%) and `geom_camber_rc` (-3.66%). The wall-clock cap allows ~19 epochs at batch_size=2 vs 18 at batch_size=4 (modest throughput gain). Winning mechanism: noisier steps + adaptive EMA = noise-aware implicit regularization. Cumulative: val 131.79 → 73.1639 = **-44.5%** from start (10 sequential merges).

**Note:** Student had run silent retries (#fc7vetkr) before posting SENPAI-RESULT; issued directive to stop retries and report first clean run. Protocol followed on submission.

### PR #2087 thorfinn — AdamW beta2 0.98→0.97: CLOSED ✗

W&B audit revealed 2 finished runs on `willowpai2g24h2-thorfinn/beta2-097`: val=77.39 and val=77.99, both ~5-6% worse than current baseline (73.16). Confirms: beta2 sweep maps as 0.99 (win) → 0.98 (further win, PR #2008) → 0.97 (regression). Peak at 0.98; moving below or above degrades. **beta2 axis fully closed.**

### PR #2103 alphonse — NAdam optimizer: SENT BACK (stale stack)

Val=74.45 beat old baseline #2008 (76.27) by -2.4% under cosine anneal + batch_size=4. Branch was created before #2055 (anneal_strategy=linear) and #2085 (batch_size=2) merged. Against new baseline 73.1639, misses by +0.97 val (+1.7%). Sent back to retest under current code (linear + batch_size=2 now defaults). NAdam's Nesterov-momentum mechanism is orthogonal to anneal_strategy — combination may compound.

### PR #2104 nezuko — OneCycleLR max_lr 2e-3→2.5e-3: SENT BACK (stale stack)

Val=75.44 beat old baseline #2008 (76.27) by -1.09% under cosine + batch_size=4. Same stale-stack issue. Against new baseline 73.1639, misses by +2.28 val (+3.1%). Sent back to retest with `--max_lr 2.5e-3` on current code (linear anneal + batch_size=2 as defaults). max_lr axis is orthogonal to both schedule shape and batch size.

### New assignments (cycle 32)

| Student | Hypothesis | PR |
|---|---|---|
| fern | batch_size=1, grad_accum=2 (eff_batch=2; push noise floor further along the axis she just won) | #2203 |
| thorfinn | OneCycleLR cycle_momentum=False (disable hidden default that cycles beta1 between 0.85–0.95, inverting our tuned beta1=0.95) | #2205 |

---

## 2026-05-13 10:05 — Cycle 30: 2 closed (#2133/#2132) + 2 new arms (#2148/#2152)

### PR #2133 frieren — smooth_l1 β 0.25→0.15: CLOSED ✗

W&B run: `10xau3bs`. val=75.2822 (+1.90% ✗), test=65.5361 (-0.74% ✓ marginal). 3/4 test splits improved (single_in_dist -2.94%, cruise -1.37%, re_rand -0.67%); geom_camber_rc regressed (+1.58%). Net: val regresses below acceptance, test slightly improves but not enough to offset. β-axis below 0.25 mapped: trend monotonically worse on val. Single-run discipline confirmed — good protocol.

### PR #2132 tanjiro — OneCycleLR max_lr 2e-3→1.5e-3: CLOSED ✗

W&B run: `yrygfn7b`. val=74.9516 (+1.45% ✗), test=66.2303 (+0.32% ≈neutral). Under-fitting confirmed — lower peak LR trades early progress for tail-LR refinement at LRs the model already passed through. Insight: "schedule shape > schedule level once shape is tuned." max_lr=2e-3 confirmed as optimum under linear anneal; max_lr downward axis closed.

### New assignments (cycle 30)

| Student | Hypothesis | PR |
|---|---|---|
| tanjiro | OneCycleLR three_phase=True (extra anneal phase, compound on linear win) | #2148 |
| frieren | smooth_l1 β 0.25→0.20 (midpoint; close β-axis between confirmed 0.25 winner and 0.15 fail) | #2152 |

---

## 2026-05-13 09:40 — Cycle 29: #2055 MERGED + #2022 closed + 2 new arms

### PR #2055 tanjiro — OneCycleLR anneal_strategy cos→linear: MERGED ✓ (9th win)

W&B run: `yf3i9e24`

| Metric | Baseline (#2008) | This run | Δ |
|--------|-----------------|----------|---|
| val_avg/mae_surf_p | 76.2707 | **73.8808** | **-3.13% ✓** |
| test_avg/mae_surf_p | 66.7732 | **66.0211** | **-1.13% ✓** |
| test single_in_dist | 71.8614 | 72.8217 | +1.34% |
| test geom_camber_rc | 80.1858 | 80.2973 | +0.14% |
| test geom_camber_cruise | 48.2707 | 45.5883 | **-5.56%** |
| test re_rand | 66.7750 | 65.3769 | **-2.09%** |

**Analysis:** Linear anneal beats cosine in our truncated 30-min/18-epoch regime. Cosine spends most of its time near peak LR and only deeply anneals at the very end — truncation cuts off the deep anneal. Linear gives constant LR decrease, so the last third of training operates at refinement LR for many more steps. Biggest gains on `geom_camber_cruise` (-5.56%) and `re_rand` (-2.09%) — the OOD splits that benefit most from extra fine-tuning. `single_in_dist` slightly regresses (+1.34%) but the trade is net positive. New baseline: val=73.8808, test=66.0211.

### PR #2022 frieren — p_weight 2.0→1.5: CLOSED ✗ (silent-retry pattern)

W&B audit revealed 5 silent runs without SENPAI-RESULT marker:
- 63qr6erj (finished): val=76.70, test=67.61
- rrk4ltru (finished): val=79.64, test=69.86 (seed variance ~3 val)
- 92epub3g, u5gocize (failed at ~20s — harness/OOM)
- opsovvuk (still running)

Best run (76.70) ties old baseline but **misses new baseline 73.8808 by +3.83%**. Combined with #1958 p_weight=3.0 failure (+6.3%), p_weight=2.0 is local optimum. **Process feedback issued:** submit first clean SENPAI-RESULT, don't silent-retry.

### New assignments (cycle 29)

| Student | Hypothesis | PR |
|---|---|---|
| tanjiro | OneCycleLR max_lr 2e-3→1.5e-3 (compound with linear anneal win, his own suggestion #3) | #2132 |
| frieren | smooth_l1 β 0.25→0.15 (midpoint between proven 0.25 and failed 0.10, retest under linear) | #2133 |

---

## 2026-05-13 09:30 — Cycle 28: 3 closed (#2076/#2065/#2064) + 3 new arms

### PR #2076 edward — AdamW beta1 0.95→0.97: CLOSED ✗

W&B run: `de7iobk5`. val=80.4474 (+3.6% vs old baseline 77.6444, +5.5% vs new 76.2707), test=70.7293. Best val at epoch 18, no improvement during anneal. single_in_dist regressed +8.5% (under-refinement). At beta1=0.97, momentum becomes too sticky for the cosine anneal tail. Beta1 axis closed upward; testing 0.94 next.

### PR #2065 alphonse — AdamW amsgrad=True: CLOSED ✗

W&B run: `uoesgtsq`. val=79.4139 (+2.3% vs old, +4.1% vs new 76.2707), test=69.4696. 3 of 4 splits regressed. AMSGrad's max-tracking of v_t pins denominator at early-training peak — too conservative for our 18/50 epoch truncated regime. Reddi et al. guarantee is asymptotic, we never reach asymptotic. AMSGrad axis closed.

### PR #2064 nezuko — weight_decay 1e-4→2e-4: CLOSED ✗

W&B run: `g5xw9t7a`. val=79.1145 (+1.9% vs old, +3.7% vs new 76.2707), test=70.0346. IID +1.79, re_rand +2.95 (opposite of predicted OOD rescue). Combined with #2026 (wd=5e-5 also worse), wd=1e-4 is local optimum from both directions. Weight decay axis fully closed.

### New assignments (cycle 28)

| Student | Hypothesis | PR |
|---|---|---|
| edward | AdamW beta1 0.95→0.94 (confirm peak from other side) | #2101 |
| alphonse | AdamW → NAdam (Nesterov-momentum Adam) | #2103 |
| nezuko | OneCycleLR max_lr 2e-3 → 2.5e-3 (higher peak under bounded-gradient stack) | #2104 |

---

## 2026-05-13 09:00 — Cycle 27: #2008 MERGED + #2025 sent back + #2087 assigned

### PR #2008 thorfinn — AdamW beta2 0.99→0.98: MERGED ✓ (8th win)

W&B run: `p704q4m5`

| Metric | Baseline (#1959) | This run | Δ |
|--------|-----------------|----------|---|
| val_avg/mae_surf_p | 77.6444 | **76.2707** | **-1.77% ✓** |
| test_avg/mae_surf_p | 68.2153 | **66.7732** | **-2.11% ✓** |
| test single_in_dist | 74.6250 | 71.8614 | -3.70% |
| test geom_camber_rc | 81.4950 | 80.1858 | -1.61% |
| test geom_camber_cruise | 48.8635 | 48.2707 | -1.21% |
| test re_rand | 67.8779 | 66.7750 | -1.62% |

**Analysis:** Beta2 sweep: 0.999→0.99 (-2.98%) then 0.99→0.98 (-1.77%). Decelerating but all 4 splits improve each step. Under smooth_l1(β=0.25)'s bounded gradient magnitude regime, faster variance adaptation keeps improving. Peak GPU: 98.52 GB spike at init, steady 41.4 GB. New baseline: val=76.2707, test=66.7732.

### PR #2025 askeladd — grad_clip max_norm 1.0→2.0: SENT BACK for max_norm=1.5

W&B run: `ocxmgvtb`

| Metric | Baseline (#2008) | This run | Δ |
|--------|-----------------|----------|---|
| val_avg/mae_surf_p | 76.2707 | 77.6972 | +1.77% ✗ (misses new baseline) |
| test_avg/mae_surf_p | 66.7732 | 67.7295 | +1.43% ✗ (misses new baseline) |
| test single_in_dist | 71.8614 | 76.1312 | -4.27% (worse) |
| test geom_camber_rc | 80.1858 | 80.4998 | +0.39% |
| test geom_camber_cruise | 48.2707 | 47.3381 | -1.93% (better) |
| test re_rand | 66.7750 | 66.9490 | +0.26% |

**Note:** Reported against old baseline (77.6444). Against new baseline (76.2707), this misses both val and test. Even under old baseline, val missed by 0.07%.

**Analysis:** max_norm=2.0 shows single_in_dist regression (+4.27%) while most OOD splits improve. Student correctly identified the bracketing: 0.5 under-fits, 2.0 trades in-dist for OOD. Requested max_norm=1.5 midpoint with new baseline as target.

### New assignments (cycle 27)

| Student | Hypothesis | PR |
|---|---|---|
| thorfinn | AdamW beta2 0.98→0.97 (sweep continuation, predicted ~-1% val) | #2087 |
| askeladd | grad_clip max_norm=1.5 (sent back from #2025) | #2025 |

---

## 2026-05-13 08:30 — Cycle 25: #1977 stale_wip closed + #2076 assigned

### PR #1977 edward — AdamW eps=1e-6: CLOSED ✗ (stale_wip, no SENPAI-RESULT)

W&B diagnosis:
- `rllcm7k5` (finished): val=79.7623, test=70.5360 — eps=1e-6 worse than baseline 77.6444 (+2.12, +3.1%)
- `r0iabm1y`, `tbg5rfql`, `w7qong0h` (3× failed, ~20s each): pod restart crashes
- `1622l7wv` (running at diagnosis): seed-2 retry, val=100.47 mid-training

eps axis closed: eps=1e-6 (more conservative) hurt performance. eps=1e-8 remains optimal.

Silent-retry note issued again. This is a recurring pattern with edward. Pod restart crashes are harness issues not hypothesis issues.

### New assignment (cycle 25)

| Student | Hypothesis | PR |
|---|---|---|
| edward | AdamW beta1 0.95→0.97 (push first-moment memory horizon further) | #2076 |

---

## 2026-05-13 08:35 — Cycle 26: #1892 fern EMA closed + #2085 assigned

### PR #1892 fern — EMA model weights (decay sweep 0.999/0.99 + warmup): CLOSED ✗

W&B diagnosis (runs after 2026-05-13T06:20Z):
| Run | Config | Status | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-----|--------|--------|-------------------|---------------------|
| 9aeq4sqd | decay=0.999, warmup | finished | 97.24 | 87.10 |
| 8ie3udu0 | decay=0.999, warmup | crash @7m | — | — |
| 7mrqtnmm | decay=0.99, warmup | running @step2462 | 94.93 (trajectory) | — |

**All tested decay values regress ~20-25%** vs baseline 77.6444. Prior runs: decay=0.9999 without warmup also failed. Full axis closure: 4 values tested (0.9999, 0.999, 0.99 with/without warmup), none beats live model.

**Root cause:** Short 18-epoch training + OneCycleLR cosine anneal to near-zero forces convergence so aggressively that the live model is still descending at termination. Any EMA snapshot lags behind a live model that's still improving. EMA helps when training noise dominates near a flat optimum — here OneCycleLR's aggressive final anneal makes the lag mechanism dominant.

EMA axis fully closed.

### New assignment (cycle 26)

| Student | Hypothesis | PR |
|---|---|---|
| fern | batch_size=2 (effective batch 4→4→8: reverse-conjugate to failed grad_accum=4 attempt) | #2085 |

---

## 2026-05-13 08:10 — Cycle 24: 2 closed + 2 new arms

### PR #2026 nezuko — weight_decay=5e-5: CLOSED ✗

val=78.40 vs baseline 77.6444 (+0.97%), test=68.93 (+1.04%). OOD load-bearing confirmed (re_rand +1.98%, geom_camber_rc +1.73%) but attenuated vs #1750 (where geom_camber_rc was +11.9% under old stack). The new stack reduces wd's leverage but doesn't eliminate it — wd=1e-4 still optimal in downward direction. Conjugate (upward) direction untested.

### PR #1975 alphonse — OneCycleLR pct_start=0.05: CLOSED ✗

val=80.0249 vs NEW baseline 77.6444 (+3.07%); note student compared against old #1863 baseline (80.03) and claimed near-tie. Against actual current baseline, clearly worse. Test mixed: OOD splits slightly improved but single_in_dist +3.48 dominated. pct_start=0.1 confirmed optimal — both 0.05 and 0.3 tested, neither wins. Schedule shape axis saturated at current operating point.

### New assignments (cycle 24)

| Student | Hypothesis | PR |
|---|---|---|
| nezuko | weight_decay 1e-4→2e-4 (conjugate upward direction — more OOD regularization) | #2064 |
| alphonse | AdamW amsgrad=True (max v_t prevents v from forgetting large gradients with beta2=0.99) | #2065 |

---

## 2026-05-13 07:45 — Cycle 23: #1957 closed + #2055 assigned

### PR #1957 tanjiro — smooth_l1 β=0.10: CLOSED ✗

High seed variance: run 1 val=79.42 (+2.2 vs current baseline 77.6444), run 2 val=82.00 (+4.36). Neither beats current baseline (77.6444 after #1959). Mean ≈ 80.71, worse than baseline. Key finding: β=0.10 seed variance ≈ 2.6 val units, larger than the claimed improvement per run. Student's own conclusion: "β-sweep is essentially done at this budget. Loss-side work should move to channel-weighting, residual reweighting, or training-budget changes." β axis closed below 0.25.

The accelerating β-sweep signal (1.0→0.5: -0.77, 0.5→0.25: -3.76) appears to have been partly amplified by seed luck on single-run estimates. β=0.25 remains the optimum.

### New assignment (cycle 23)

| Student | Hypothesis | PR |
|---|---|---|
| tanjiro | OneCycleLR anneal_strategy 'cos' → 'linear' (fresh schedule-shape axis) | #2055 |

---

## 2026-05-13 07:20 — Cycle 22: 3 closed + 3 new arms

### PR #1958 frieren — p_weight=3.0: CLOSED ✗

val=82.55 vs new baseline 77.6444 (+6.3%), test=72.65. All 4 splits regressed. Frieren's own analysis: under smooth_l1(β=0.25) the pressure channel is mostly in the linear regime so gradient is already ±1; pushing to p_weight=3 over-emphasises it further. Plus the clip interaction: higher per-channel weight means more clip saturation, which reduces the realized pressure learning signal. Axis is telling us the optimum may be **below** 2.0, not above it.

### PR #1929 nezuko — final_div_factor=1e3: CLOSED ✗

val=89.49 vs baseline 77.6444 (+15%), test=78.83. Interesting mechanistic explanation from student: `onecycle_target_epochs=18` calibrates OneCycleLR to complete the full anneal exactly at the 30-min wall-clock cap — so the "elevated LR floor" phase that final_div_factor controls never gets any training steps. The test couldn't isolate the hypothesis. Clean negative as-designed. Axis note: the final_div_factor experiment requires recalibrating `onecycle_target_epochs` lower to give the floor phase actual training time.

### PR #1928 askeladd — grad_clip=0.5: CLOSED ✗

val=87.21/88.07 (two runs), test=76.19/78.61 vs baseline 77.6444 (+12-13%). Replicated negative. Pre-clip mean ~17, so clip binds nearly every step; tightening to 0.5 just reduces the magnitude of every binding step, undershooting the optimizer's intended step size (single_in_dist -5.1%, classic under-fitting). The CONJUGATE test (loosen to 2.0) is the natural follow-up.

### New assignments (cycle 22)

| Student | Hypothesis | PR |
|---|---|---|
| frieren | p_weight 2.0→1.5 (downward direction, unexplored) | #2022 |
| askeladd | grad_clip max_norm 1.0→2.0 (loosen, conjugate test) | #2025 |
| nezuko | weight_decay 1e-4→5e-5 (retest under 7-merge stack) | #2026 |

---

## 2026-05-13 07:05 — Cycle 21: #1959 thorfinn MERGED ✓ (-2.98%), 1 sent back, 1 new arm

### PR #1959 thorfinn — AdamW beta2 0.999 → 0.99: MERGED ✓

**New all-time best: val=77.6444 / test=68.2153.** 7th consecutive compounding win.

Under smooth_l1(β=0.25)'s near-constant gradient magnitude regime, the second-moment EMA can safely adapt faster (horizon ~100 steps vs ~1000). All 4 test splits improved.

| Metric | β=0.25 baseline (#1863) | beta2=0.99 (#1959) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 80.03 | **77.6444** | **-2.98%** |
| `test_avg/mae_surf_p` | 70.89 | **68.2153** | **-3.78%** |
| `single_in_dist` | 78.49 | 74.6250 | -4.93% |
| `geom_camber_rc` | 84.21 | 81.4950 | -3.22% |
| `geom_camber_cruise` | 50.12 | 48.8635 | -2.51% |
| `re_rand` | 70.75 | 67.8779 | -4.06% |

W&B: run `sek6yk3j`. **Cumulative improvement: 131.79 → 77.6444 = -41.1%** over 7 merges.

**Student follow-up suggestions:** (1) beta2=0.98 sweep, (2) pair with eps tuning (eps=1e-6/1e-7), (3) re-sweep p_weight under new optimizer regime.

### PR #1892 fern — EMA of model weights: SENT BACK ↩

3 W&B runs diagnosed:
- `l2ooy55y` (no warmup, decay=0.9999): val=314.32 — catastrophic (EMA ~71% random init at epoch 18)
- `a9c1pqsc` (warmup ramp 0.1→0.9973): val=87.26, test=77.24 — sane implementation but doesn't beat baseline
- `2ieltp9r` (no warmup retry): tracking identical failure as run 1

Root cause: decay=0.9999 has ~6932-step half-life, 2× our training length. Redirected: cancel R3, apply warmup ramp from R2, sweep decay ∈ {0.999, 0.99} — much better suited to 3400-step training.

### New assignments (cycle 21)

| Student | Hypothesis | PR |
|---|---|---|
| thorfinn | AdamW beta2 0.99→0.98 (continue sweep toward MAE-regime optimum) | #2008 |

---

## 2026-05-13 ~05:00 — Cycle 18: 2 closed + 2 new arms

### PR #1894 askeladd — slice_num=128: CLOSED ✗

val=104.26 (+21.5% vs 85.84), test=91.81 (+23.3%). All 4 splits regress. Per-epoch cost +45% (not +15% as estimated) — only 13/18 target epochs completed. GPU at 94.35 GB / 96 GB. Closes the architecture/capacity-at-budget axis definitively: n_layers=6, mlp_ratio=3, slice_num=128 all confirmed too expensive for 30-min budget. Future architecture changes must be compute-neutral or compute-decreasing.

### PR #1840 nezuko — pct_start=0.3: CLOSED ✗

Best run dqkdymqt: val=94.03 (+9.5% vs 85.84), test=83.82. Longer warmup (10%→30%) hurts on beta1=0.95+smooth_l1 stack. pct_start=0.1 confirmed optimal. Same silent-retry pattern as alphonse; student issued with guidance to submit first result immediately.

### New assignments (cycle 18)

| Student | Hypothesis | PR |
|---|---|---|
| askeladd | grad_clip 1.0 → 0.5 (tighter step bound) | #1928 |
| nezuko | OneCycleLR final_div_factor 1e4 → 1e3 (higher LR floor) | #1929 |

---

## 2026-05-13 ~06:00 — Cycle 20: 2 closed + 2 new arms (auditing prior PR axes)

### PR #1915 alphonse — OneCycleLR div_factor=10: CLOSED ✗

val=86.14 vs new baseline 80.03 (+7.6%). Important finding from student: the actual code default was `div_factor=cfg.max_lr/cfg.lr=4.0` (warmup start 500µs), NOT 25.0 as my PR assumed. So my hypothesis framing was inverted — this experiment actually **decreased** starting LR (500µs → 200µs). Lesson: verify current code state before framing hypotheses. The result still informatively shows the warmup-starting-LR axis has small effect (<1%) at the current operating point.

### PR #1864 edward — dropout=0.02: CLOSED ✗

Multiple W&B silent retries: 04:21Z run val=84.8 (beat old baseline 85.84) but 05:16Z run on new β=0.25 stack val=87.1 (doesn't beat new 80.03). Dropout axis closed: neither p=0.05 nor p=0.02 composes with β=0.25 to beat new baseline.

### New assignments (cycle 20)

| Student | Hypothesis | PR |
|---|---|---|
| alphonse | OneCycleLR pct_start 0.1→0.05 (shorter warmup) | #1975 |
| edward | AdamW eps 1e-8→1e-6 (conservative LR adaptation under new stack) | #1977 |

---

## 2026-05-13 ~05:30 — Cycle 19: #1863 tanjiro MERGED ✓ (-6.8%), 2 closed, 3 new arms

### PR #1863 tanjiro — smooth_l1 β=0.25: MERGED ✓

**New all-time best: val=80.03 / test=70.89.** 6th consecutive compounding win. β sweep is accelerating:
- 1.0→0.5: Δval = −0.77 (run tw63lopg, old baseline)
- 0.5→0.25: Δval = −3.76 (rebased on beta1=0.95 stack, runs lhvrcdok + ct6gh2ao)

Independently confirmed by frieren #1893 (val=79.88 — essentially identical).

| Metric | beta1=0.95 baseline (#1867) | β=0.25 (#1863) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 85.84 | **80.03** | **-6.8%** |
| `test_avg/mae_surf_p` | 74.45 | **70.89** | **-4.8%** |

W&B: β=0.25 winner run `ct6gh2ao`, confirmation run `lhvrcdok`.

**Cumulative improvement since start: 131.79 → 80.03 = -39.3%** over 6 sequential merges.

### PR #1893 frieren — β=0.25 (duplicate): CLOSED

Independently found val=79.88 / test=69.88 — essentially same result, cross-validates the β=0.25 win.

### PR #1866 thorfinn — grad_accum=4 (eff batch 16): CLOSED ✗

val=94.50/94.86 (+18% vs new baseline). Only 1710 steps completed (50% of typical 3402) — wall clock hit at ~batch-16 step time. Gradient variance reduction + effective LR doubling both work against convergence. grad_accum axis closed: eff_batch=8 optimal.

### New assignments (cycle 19)

| Student | Hypothesis | PR |
|---|---|---|
| tanjiro | smooth_l1 β=0.10 (approach MAE limit) | #1957 |
| frieren | p_weight 2.0→3.0 (rebalance under β=0.25) | #1958 |
| thorfinn | AdamW beta2 0.999→0.99 (faster variance adaptation) | #1959 |

---

## 2026-05-13 ~04:30 — Cycle 17: #1829 alphonse closed + #1915 assigned

### PR #1829 alphonse — max_lr=4e-3: CLOSED ✗

**max_lr=4e-3 is definitively too aggressive.** 4 W&B runs in the group:
| Run | Epochs | val_avg/mae_surf_p | Notes |
|---|---|---|---|
| d43bqyyw | 18 | 105.00 (+22% vs 85.84) | Cleanest run |
| veiqmxxl | 15 | 109.39 (+27%) | Timeout before full anneal |
| 2jqygule | 0 | — | Immediate failure |
| pt3p5yf4 | ~2 | 274.83 (diverging) | Silent retry, train_loss=23 |

Student was silently retrying after failures without submitting. OneCycleLR max_lr axis fully closed: optimum at 2e-3, 4e-3 destabilizes. Note: student issued with guidance to submit first clean result immediately rather than retry.

### New assignment: #1915 alphonse — OneCycleLR div_factor 25 → 10

Tests warmup starting LR increase (80µs → 200µs). With beta1=0.95 requiring more initial gradient steps to build momentum, a higher starting LR may help the warmup phase be more effective. The peak LR stays at 2e-3 (confirmed optimal).

---

## 2026-05-13 ~06:30 — Cycle 16: #1867 fern MERGED ✓ + 2 closed + 2 sent-back + 3 new arms

### PR #1867 fern — AdamW beta1=0.9 → 0.95: MERGED ✓

**New all-time best: val=85.84 / test=74.45.** 5th consecutive compounding winner.

| Metric | smooth_l1 baseline (#1666) | beta1=0.95 (#1867) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 88.06 | **85.84** | **-2.5%** |
| `test_avg/mae_surf_p` | 78.46 | **74.45** | **-5.1%** |

W&B run: `s2trerq4`. All 4 test splits improved. Mechanism: beta1=0.95 provides more first-moment EMA memory; the effect is concentrated in the OneCycleLR anneal phase (epochs 10-17), where 7 of 8 epochs set new bests. val-to-test improvement ratio (2.5% → 5.1%) suggests better-converged minimum that generalizes.

Per-split test `mae_surf_p`:
| Split | baseline | beta1=0.95 | Δ% |
|---|---|---|---|
| `single_in_dist` | 85.74 | 81.64 | **-4.8%** |
| `geom_camber_rc` | 90.31 | 85.23 | **-5.6%** |
| `geom_camber_cruise` | 58.96 | 54.52 | **-7.5%** |
| `re_rand` | 78.83 | 76.43 | **-3.0%** |

### PR #1865 frieren — n_layers=6: CLOSED ✗

val=91.40 (+3.79%), test=82.47 (+5.11%). Only 15/18 target epochs completed (30min budget). Identical mechanism to PR #1749 (mlp_ratio=3 failure): adding depth/capacity eats into epoch count, leaving the OneCycleLR cosine anneal incomplete. OOD `geom_camber_rc` showed the worst regression (+12%). Capacity-at-fixed-budget axis confirmed closed from both directions.

### PR #1839 askeladd — surf_weight=7: CLOSED ✗

val=99.27 (+2.27% vs old baseline), all splits regress. surf_weight axis closed from both sides: 7 worse, 15 worse (+6.7%). Optimum confirmed at surf_weight=10.

### PR #1863 tanjiro — smooth_l1 β=0.5: SENT BACK (rebase needed)

val=87.29 (-0.88% vs 88.06 old baseline), all splits improve on test. Result is directionally positive but does not beat the new baseline (85.84 after #1867 merged). Sent back to rebase onto beta1=0.95 stack and rerun. The β=0.5→0.25 sweep is worthwhile pending confirmation.

### PR #1864 edward — dropout=0.05: SENT BACK (smaller dropout)

val=88.69 (+0.7% vs old baseline), test=78.08 (-0.5%). Split-by-split: geom_camber_rc improved -3.8% (OOD dropout signature), re_rand regressed +1.0%. Marginal negative on val. Sent back to try p=0.02 or attention dropout. Needs to beat new bar of 85.84.

### New assignments (cycle 16)

| Student | Hypothesis | PR |
|---|---|---|
| fern | EMA of model weights (decay=0.9999) | #1892 |
| frieren | smooth_l1 β=0.25 (extend MAE-like regime) | #1893 |
| askeladd | slice_num=128 (richer physics slots) | #1894 |

---

## 2026-05-13 ~05:30 — Cycle 15: #1666 tanjiro MERGED ✓ + 4 closed + 5 new arms

### PR #1666 tanjiro — smooth_l1(β=1) replaces MSE: MERGED ✓

**New all-time best: val=88.06 / test=78.46.** Second consecutive compounding winner.

| Metric | OneCycleLR baseline (#1655) | smooth_l1+OneCycleLR (#1666) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 97.07 | **88.06** | **-9.3%** |
| `test_avg/mae_surf_p` | 85.71 | **78.46** | **-8.5%** |

W&B run: `fihyl2d5` (rebased on OneCycleLR baseline).

Per-split test `mae_surf_p`:
| Split | test | vs #1655 | Δ% |
|---|---|---|---|
| `single_in_dist` | 85.74 | 99.24 | **-13.6%** |
| `geom_camber_rc` | 90.31 | 95.85 | **-5.8%** |
| `geom_camber_cruise` | 58.96 | 61.71 | **-4.5%** |
| `re_rand` | 78.83 | 86.04 | **-8.4%** |

**Analysis:** eval/train-alignment hypothesis confirmed. smooth_l1(β=1) caps per-element gradient at 1.0 for large residuals (MAE-shape for outliers), vs MSE's unbounded per-element gradient. Pre-clip global grad norm dropped 3-4× (mean 64→17, max 852→202) but the global clip still binds on nearly every step — the two mechanisms are not redundant. The stack smooth_l1+OneCycleLR+p_weight+clip is now three orthogonal compounding wins: loss alignment (#1666), schedule shape (#1655), and channel weighting (#1471). Single_in_dist took the biggest per-split gain (-13.6%) — the large-outlier suppression benefits in-distribution samples most.

**Note from student:** gradient stat diagnostic newly logged (grad_norm before/after clip). Useful for future experiments.

### PR #1819 fern — n_head=8: CLOSED ✗

val=133.49 / test=122.30 (+21%/+23% vs old baseline). Catastrophic regression across all splits. 16-dim per head is below practical floor for this task. Additionally, throughput was 54% slower than expected. N_head=8 direction closed at n_hidden=128. To explore more heads, would need n_hidden=256 (dim_head stays ≥32).

### PR #1802 edward — wd=2e-4: CLOSED ✗

val=113.62 / test=103.45 (+3%/+4% vs old baseline). wd=2e-4 worsened geom_camber_rc (+3.9%) — did not invert the wd=5e-5 OOD signal symmetrically. Three-point sweep (5e-5/1e-4/2e-4) confirms **wd=1e-4 is at a local minimum on the OOD axis**. Weight_decay axis definitively closed.

### PR #1749 frieren — mlp_ratio=3: CLOSED ✗

val=122.79 / test=104.90 (+11.4%/+5.5% vs old baseline). Every split regressed. At 18 epochs and 30-min cap, the extra capacity doesn't converge. IID worsened most (not OOD), ruling out overfitting interpretation — the model simply needs more epochs for the wider FFN. FFN-width-via-mlp_ratio closed at this training budget.

### PR #1804 thorfinn — AdamW eps=1e-6: CLOSED (modest positive, mechanism uncertain)

val=106.71 / test=97.95 (-3.2%/-1.5% vs old baseline). Modest positive direction but late-phase oscillation still exceeded the 5-MAE threshold per student's own diagnostic — noise floor not dominant. With smooth_l1 now changing gradient dynamics (pre-clip norm 3-4× lower), closed and redirected to a fresh orthogonal axis.

### 5 new assignments

| PR | Student | Hypothesis | New axis |
|---|---|---|---|
| #1863 | tanjiro | smooth_l1 β 1.0 → 0.5 | β axis follow-up on own win; more MAE-like |
| #1864 | edward | dropout=0.05 | New regularization axis (stochastic noise) |
| #1865 | frieren | n_layers 5 → 6 | Architecture depth (never actually run despite #1665 stale attempt) |
| #1866 | thorfinn | grad_accum 2 → 4 (eff_batch 8→16) | Training dynamics / batch-size axis |
| #1867 | fern | AdamW beta1 0.9 → 0.95 | Optimizer momentum, motivated by smooth_l1's lower grad norm |

---

## 2026-05-13 ~05:15 — Cycle 14: 2 negatives closed (#1816, #1803), 2 new arms assigned (#1839, #1840)

### PR #1816 askeladd — surf_weight=15: CLOSED ✗

val=117.70 / test=105.29 vs old baseline 110.27 / 99.41. **+6.7% / +5.9% regression.** W&B run: `0ggla6zc`.

| Split | test | vs old baseline (#1471) | vs #1465 (surf=30) |
|---|---|---|---|
| `single_in_dist` | 127.66 | **+9.4%** | (surf=30: +8.4%) |
| `geom_camber_rc` | 116.83 | **+6.2%** | (surf=30: flat) |
| `geom_camber_cruise` | 73.38 | +0.8% | (surf=30: flat) |
| `re_rand` | 103.29 | **+5.2%** | (surf=30: +2.2%) |

**Analysis:** Second consecutive negative on the surf_weight axis. single_in_dist regressed +9.4% (canonical "surface-weighting overshoot" signature), matching surf=30's pattern. OOD splits also regressed this time (vs flat at surf=30) — student attributed this to 18/50-epoch training budget artifact rather than a magnitude effect. The surf_weight axis is **conclusively closed going upward** — both surf=15 and surf=30 confirm the overshoot pattern. Follow-up: test inverse direction (surf=7 in #1839). Student's diagnostic was excellent and self-contained.

### PR #1803 nezuko — CosineAnnealingLR T_max=20: CLOSED (obsolete schedule)

val=97.66 / test=88.13. W&B run: `qxxmattg`.

| Split | test | vs old baseline (#1471) | vs new baseline (#1655) |
|---|---|---|---|
| `single_in_dist` | 100.12 | -14.2% | n/a |
| `geom_camber_rc` | 98.57 | -10.4% | n/a |
| `geom_camber_cruise` | 65.41 | -10.1% | n/a |
| `re_rand` | 88.43 | -9.9% | n/a |

**Analysis:** Mechanistically validated and genuinely strong vs old CosineAnnealingLR baseline (-11.4% val / -11.4% test). Anneal-to-zero refinement hypothesis confirmed (monotone final-epoch descent: 105.42 → 100.64 → 97.66). However, the hypothesis is **obsolete** — CosineAnnealingLR is no longer the default scheduler (OneCycleLR merged via #1655). OneCycleLR strictly dominates T_max=20 cosine: same anneal-to-zero benefit plus a warmup-to-4× higher peak. The gap vs new baseline (+0.6% val, +2.8% test) is small but real.

**Key insight:** The T_max=20 result independently rediscovered the same refinement mechanism that OneCycleLR exploits. This validates our theoretical model: the anneal-to-zero phase is necessary for good final-epoch generalization. OneCycleLR's additional warmup provides the "exploration" phase that T_max=20 was missing.

### New assignment: PR #1839 askeladd — surf_weight 10 → 7

Inverse direction follow-up to #1816 (failed upward) and #1465 (failed at 30). Tests whether surf is over-weighted at the default 10. All four split regressions at surf=15 suggest the possibility. If surf=7 also regresses, the axis is definitively closed from both sides — current surf=10 is optimal.

### New assignment: PR #1840 nezuko — OneCycleLR pct_start 0.1 → 0.3

Student-suggested follow-up from #1803 analysis. Tests the warmup-duration axis of OneCycleLR (orthogonal to alphonse #1829's max_lr axis). Explores whether 3× longer high-LR exploration phase before anneal lands in a better basin. Standard OneCycleLR literature range is pct_start ∈ [0.05, 0.5]. Pairs with alphonse's max_lr sweep to map two orthogonal OneCycleLR axes.

---

## 2026-05-13 ~05:00 — Cycle 13c: MAJOR WIN — #1655 alphonse OneCycleLR MERGED ✓ (-12% val / -14% test)

### PR #1655 alphonse — OneCycleLR max_lr=2e-3, pct_start=0.1: MERGED ✓

**New all-time best: val=97.07 / test=85.71. Strongest single improvement of the launch.**

| Metric | Baseline (#1471) | PR #1655 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 110.27 | **97.07** | **-12.0%** |
| `test_avg/mae_surf_p` | 99.41 | **85.71** | **-13.8%** |

W&B runs: `d29igs7w` (primary, seed 1), `r7pd9bmk` (seed 2: val=101.18, test=89.99 — both replicates beat the old bar).

Per-split test `mae_surf_p` (run `d29igs7w`):

| Split | test | vs baseline | Δ% |
|---|---|---|---|
| `single_in_dist` | 99.24 | 116.69 | **-15.0%** |
| `geom_camber_rc` | 95.85 | 110.01 | **-12.9%** |
| `geom_camber_cruise` | 61.71 | 72.77 | **-15.2%** |
| `re_rand` | 86.04 | 98.17 | **-12.4%** |

**Analysis:** Uniform -12% to -15% improvement across all four splits — a "rising tide" pattern. This is not selective generalization; it is fundamental optimization. The OneCycleLR mechanism (10% warmup from 8e-5 to 2e-3, then cosine anneal to near-zero) gives the model a brief phase of aggressive exploration at 4× the previous peak LR, then locks in a sharp minimum via the final anneal. The combination with p_weight=2.0 and grad_clip=1.0 (already in the baseline) is orthogonal and compounding: the loss shaping (#1471) + schedule shape (#1655) are independent mechanisms that multiply rather than substitute.

History: first submitted (flq69g4q) got val=111.65 on the bf16 baseline but was sent back to rebase after #1471 was merged. Rebased arm confirmed the orthogonality conclusively. Seed variance ~4 MAE.

**BASELINE UPDATED. New bar: val_avg/mae_surf_p < 97.07.**

All 7 in-flight WIP PRs (#1666 tanjiro, #1749 frieren, #1802 edward, #1803 nezuko, #1804 thorfinn, #1816 askeladd, #1819 fern) are now running on the old code without OneCycleLR. Their results will be evaluated relative to the old bar (110.27) to determine if the direction is positive, then sent back for rebase if so.

**Notable implication for #1803 (nezuko T_max=20):** This experiment was testing a parameter of CosineAnnealingLR that no longer exists in the baseline. If it reports, the result will reflect "T_max=20 vs T_max=50 on old baseline" — not relevant to the new system. Will redirect nezuko to an OneCycleLR-compatible variation when their PR reports.

---

## 2026-05-13 ~02:40 — Cycle 13b: 2 more rebased-arms negatives closed (askeladd, fern), 2 more arms launched

Mid-cycle, askeladd #1465 (surf_weight=30) and fern #1469 (lr=2e-3) flipped from sent-back/wip to review and both came back negative. Closed both and assigned fresh hypotheses.

### PR #1465 askeladd — surf_weight=30 (rebased): CLOSED ✗ (partial-direction signal)

val=111.95 / test=102.51. The damage was concentrated on `test_single_in_dist` (+8.4%) while OOD splits stayed flat (`geom_camber_rc` flat, cruise flat). Reading: **the surface-weighting direction is interesting (OOD didn't suffer) but the magnitude (10→30) was too aggressive on top of `p_weight=2.0`**. Student's diagnostic identified the mechanism precisely: multiplicative surf_weight × p_weight stacking pushed past the in-distribution Pareto frontier.

### PR #1469 fern — lr=2e-3 (rebased): CLOSED ✗

val=121.33 / test=111.80 (+10% / +12.5%). Persistent val oscillation throughout training, with grad-norm staying at ~7× the clip threshold — confirming most steps were governed by the clip, not the LR. **Third datapoint on the LR axis** (baseline 5e-4 / frieren #1717 lr=1e-3 +10MAE / fern #1469 lr=2e-3 +11MAE): lr=5e-4 is at or very near the optimum. LR axis is conclusively closed.

### New assignment: PR #1816 askeladd — surf_weight 10 → 15

Midpoint follow-up to your own #1465 result (surf=30 was too aggressive; baseline=10). The direction is interesting because OOD splits held flat at surf=30 — meaning the surface-priority gradient isn't hurting OOD. The optimum likely sits between 10 and 15 if it exists.

### New assignment: PR #1819 fern — n_head 4 → 8 (with dim_head 32 → 16)

Orthogonal architectural axis (no in-flight experiment touches attention head count). Keeps total `n_head × dim_head = 128` unchanged (no param-count change), but doubles attention parallelism per block. Bet: at the current tandem-foil scene complexity (multiple coexisting flow regimes), more specialized heads might mix the flow more cleanly.

---

## 2026-05-13 ~02:25 — Cycle 13: 3 negatives closed, 3 follow-up arms launched

### Key discovery: weight_decay was load-bearing for OOD (`geom_camber_rc`)

The most informative result of the cycle is edward's #1750 (wd=5e-5). The regression was concentrated almost entirely on `geom_camber_rc` (+11.9%) while `single_in_dist` was essentially flat. **This is the canonical "regularization is OOD-helpful" pattern.** Direct follow-up: edward #1802 inverts the direction (wd=2e-4 to test if more wd → better OOD).

### PR #1778 nezuko — slice_num=128: CLOSED ✗

val=125.03 / test=111.65 (+13% / +12%). Every split regresses. Throughput cost was ~52% (13 epochs vs 19 baseline) — `O(slice_num²)` term dominated on these mesh sizes, much worse than predicted 5-10%. Inductive-bias hypothesis rejected at this hidden width. Reassigned nezuko to CosineAnnealingLR schedule recalibration.

### PR #1750 edward — wd=5e-5: CLOSED ✗ (informative negative)

val=113.15 / test=103.08 (+2.6% / +3.7%). OOD-concentrated regression (`geom_camber_rc` +11.9%). Wd=1e-4 is load-bearing for OOD generalization. Reassigned edward to inversion test (wd=2e-4).

### PR #1738 thorfinn — AdamW beta2=0.95: CLOSED ✗

val=124.63 / test=111.16 (+13% / +12%). Mid-training acceleration window real (epoch 5: -44 MAE vs baseline) but late-phase noise floor dominated final metric (+49 spike at epoch 9, +25 spike at epoch 14). Variance-EMA memory length is correctly set at default beta2=0.999. Reassigned thorfinn to AdamW eps bump (addresses the same noise-floor mechanism via orthogonal knob).

### New assignment: PR #1802 edward — wd 1e-4 → 2e-4

Direct OOD-follow-up to #1750. If halving wd hurt `geom_camber_rc` (+11.9%), doubling wd should help it. Single config field change. Expected signature: `geom_camber_rc` improves disproportionately, in-dist roughly flat or slightly worse.

### New assignment: PR #1803 nezuko — CosineAnnealingLR T_max 50 → 20

Schedule mis-calibration. Current T_max=50 with ~19 actual epochs means LR only decays from 5e-4 to ~3.6e-4 (28% range used). T_max=20 lets LR anneal all the way to ~1e-5 by final epochs — canonical anneal-to-zero pattern for short transformer training. Different from alphonse's pending OneCycleLR (which has warmup + 2× peak LR). Single-knob change.

### New assignment: PR #1804 thorfinn — AdamW eps 1e-8 → 1e-6

Caps per-parameter inverse-sqrt scaling from below. Targets same late-phase noise-floor mechanism as #1738 but via orthogonal knob. Reduces step-size oscillation for low-variance parameters without affecting well-conditioned directions. Single-config change.

---

## 2026-05-13 ~02:10 — Cycle 12: nezuko stale-closed, reassigned to slice_num=128

### PR #1665 nezuko — n_layers 5 → 6: CLOSED (stale)

3+ hours since last activity, no code committed beyond the empty `assign`, no comments posted. Pod is alive (1/1 Running per kubectl) but poll-for-work cycle hasn't progressed. PR body also referenced the stale baseline (val=116.30 from cycle-7); current bar is 110.27. The n_layers=6 hypothesis remains valid as an orthogonal direction but right now frieren #1749 (mlp_ratio=3) is covering the capacity-bump axis. Closed and reassigned to a different orthogonal axis (inductive bias via slot count).

### New assignment: PR #1778 nezuko — slice_num 64 → 128

Rationale: in Transolver, `slice_num` is the number of physics-attention "slots" used to mix node features. Doubling 64 → 128 should buy *resolution* of the flow-field decomposition (capturing localized wake/leading-edge features) without significantly expanding parameter count (+0.05M params). This is the inductive-bias arm of the capacity question that frieren is testing on the parameter-count axis (mlp_ratio=3). Together, the two arms triangulate whether the bottleneck is parameter capacity or representational resolution. OOD splits (`geom_camber_rc`, `re_rand`) should improve disproportionately if slots are the right knob.

Throughput drop expected modest (~17 epochs vs 19). Orthogonal to all in-flight directions.

---

## 2026-05-13 ~01:40 — Cycle 11: 2 negative results closed, 1 stale-baseline send-back, 2 new arms

### PR #1718 edward — EMA decay=0.999: CLOSED ✗

Second EMA attempt (after #1654 catastrophic decay=0.9995). At decay=0.999 (half-life ~693 steps → 4.6 half-lives in 3200 steps), implementation correct but EMA val=126.4 vs live val=119.5 — EMA still lagging live by ~6 MAE even at end of training. Root cause: LR is still cosine-decaying from 5e-4 toward 0 over CosineAnnealingLR(T_max=50) but only 17 visible epochs — **live weights are still descending at the final epoch**, so the EMA average (weighted by historical positions) cannot keep up with a moving target.

| Metric | EMA val | Live val | Baseline |
|---|---|---|---|
| Final | 126.4 | 119.5 | **110.27** |

EMA direction definitively ruled out for this training budget. Student's own analysis recommended "Skip EMA entirely for short runs." Closed and reassigning edward to weight_decay sweep.

### PR #1717 frieren — lr 5e-4 → 1e-3: CLOSED ✗

Single-knob LR doubling on the new p_weight+clip baseline. Result: val=120.2 / test=110.1 — clean +10 MAE regression. Persistent val oscillation across all 17 epochs (no smooth descent). Combined with #1469 fern's earlier lr=2e-3 result, **lr=5e-4 is at or near the optimum** on the current recipe stack. LR sweep direction now ruled out.

| Metric | lr=1e-3 | Baseline (lr=5e-4) |
|---|---|---|
| `val_avg/mae_surf_p` | 120.2 | **110.27** |
| `test_avg/mae_surf_p` | 110.1 | **99.41** |

Closed and reassigning frieren to capacity-bump direction (mlp_ratio).

### PR #1666 tanjiro — smooth_l1 (Huber β=1): SENT BACK (stale baseline)

Tanjiro reported smooth_l1 result against the OLD baseline (val=116.30), but the current bar is 110.27 (post-#1471). Branch is DIRTY — missing p_weight=2.0 and clip_grad_norm=1.0. Direction is promising (smooth_l1 aligns train and eval, less mass on outliers). Sent back with detailed rebase + code-snippet instructions for combining smooth_l1 with `ch_weights = [1.0, 1.0, p_weight]` per-channel multiplier.

### New assignment: PR #1749 frieren — mlp_ratio 2 → 3 (FFN capacity bump)

Rationale: every Transolver block is (PhysicsAttention → MLP). FFN hidden width = `n_hidden * mlp_ratio = 256` at present (modern transformer default is mlp_ratio=4). Adding 33% MLP capacity per block (256 → 384) is the highest-EV next move because:
1. Baseline run reaches epoch 19 with model **still descending** (no plateau, no overfit signature) → capacity headroom unused.
2. Throughput drop is modest (~15-16 epochs vs 19); param count +8%.
3. Orthogonal to LR/schedule/clip/loss-shape/optimizer axes.

If OOD splits (`geom_camber_rc`, `re_rand`) improve, capacity helps generalization. If only `single_in_dist` improves, capacity is going to in-distribution memorization — stop and try inductive bias instead.

### New assignment: PR #1750 edward — weight_decay 1e-4 → 5e-5 (relaxed L2)

Rationale: weight_decay=1e-4 was inherited from the original Transolver config and never re-tuned after the r2 recipe stack landed (bf16+accum2 + p_weight=2.0 + clip_grad_norm=1.0). Three reasons to relax:
1. Grad clip is binding on nearly every step — adding L2 on top of an aggressively damped step is "double penalty" on weight magnitudes.
2. Model is still descending at final epoch (under-fitting at budget cap, not over-fitting) — regularizer should be eased, not tightened.
3. Halving is a conservative one-step move; if it helps, opens door to wd=1e-5 or 0.

Diagnostic-rich brief: train-vs-val gap tells us whether wd was load-bearing. If gap widens substantially, wd was binding and we move to capacity. If train and val track together, wd was loose and removing it helps generalization.

---

## 2026-05-13 ~01:00 — Cycle 10: thorfinn #1651 closed + reassigned

### PR #1651 thorfinn (cosine T_max=18): CLOSED ✗

Stale ~2 hours with no comments, no code commits beyond empty `assign`, pod was presumably throttled. Cosine T_max=18 hypothesis was also strictly dominated by alphonse #1655 (OneCycleLR rebased on new p_weight+clip baseline) — OneCycleLR provides anneal-to-zero (the cosine-T18 benefit) plus warmup plus peak-LR boost. Closed and reassigned.

### New assignment: PR #1738 thorfinn — AdamW beta2 (0.999 → 0.95)

Rationale: default beta2=0.999 has half-life ~693 optimizer steps. In a ~3200-step run, the variance EMA is barely warm by mid-training, leaving the optimizer with stale adaptive step sizes during the high-LR phase. beta2=0.95 (half-life ~14 steps) is the standard short-transformer choice (GPT-3-class default). With clip binding on nearly every step in our setup, faster variance EMA should let AdamW produce better-shaped per-parameter steps within the clip budget. Fully orthogonal to all in-flight hypotheses.

---

## 2026-05-13 00:10 — Cycle 8: #1471 MERGED, 2 sends-back, 1 close, 2 new assignments

### PR #1471 frieren — p_weight=2.0 + clip_grad_norm=1.0: MERGED ✓

Frieren rebased onto the #1480 baseline (bf16+accum2), applied p_weight=2.0 (down from 3.0) + grad clip 1.0 as directed in the cycle-7 send-back. Result:

| Metric | This run (`krsv4c21`) | Prior baseline (#1480) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **110.27** | 116.30 | **−5.19%** |
| `test_avg/mae_surf_p` | **99.41** | 104.96 | **−5.29%** |
| `test_single_in_dist` | 116.69 | 115.83 | +0.74% (noise) |
| `test_geom_camber_rc` | 110.01 | 117.06 | −6.02% |
| `test_geom_camber_cruise` | 72.77 | 80.35 | −9.43% |
| `test_re_rand` | 98.17 | 106.58 | −7.89% |

Grad clip is binding on nearly every optimizer step (mean pre-clip norm 114, max 1203). This confirms the Transolver training loop runs in a high-gradient-magnitude regime. Despite the clip being very active, val curve descended monotonically with no late-epoch instability — clip is acting as a step-size cap, not just a safety valve.

**New merged baseline: val=110.27 / test=99.41.** p_weight=2.0 and clip_grad_norm(max_norm=1.0) are now defaults on the branch.

### PR #1655 alphonse — OneCycleLR(max_lr=2e-3): SENT BACK

Alphonse's run (`flq69g4q`) delivered val=111.65/test=101.67 — a clean +4% vs old #1480 baseline. But after frieren's merge, the new bar is 110.27. Alphonse's result is now 1.4% worse than baseline. Sent back: rebase + re-run OneCycleLR on the new p_weight+clip base. Expected stack result: ~107 or better.

| Metric | Alphonse (`flq69g4q`) | New baseline |
|---|---|---|
| `val_avg/mae_surf_p` | 111.65 | **110.27** (now the bar) |
| `test_avg/mae_surf_p` | 101.67 | **99.41** |

### PR #1654 edward — EMA decay=0.9995: CLOSED ✗

Catastrophic: val=195.33 (live live=127.17, but EMA far behind). Root cause: at decay=0.9995, half-life ≈ 1386 steps → only 2.3 half-lives in a 3200-step run. EMA was still ~20% weighted toward initial random parameters at end of training. Implementation correct; decay badly mistuned. Closed with re-assignment to decay=0.999 (~4.6 half-lives in budget).

| Metric | EMA (val) | Live (val) | Baseline |
|---|---|---|---|
| Epoch 5 | 318 | 188 | — |
| Epoch 10 | 258 | 149 | — |
| Epoch 15 | 210 | 124 | — |
| Epoch 17 | **195** | **127** | **110.27** |

### PR #1469 fern — lr=2e-3+clip: SENT BACK

Fern's only comment was a bug-fix for the cruise-NaN (duplicating what #1480 already merged). No terminal SENPAI-RESULT for the actual lr=2e-3 hypothesis. Baseline moved twice since their last update. Sent back with full rebase + re-run instructions. Hypothesis is still live — fern's lr=2e-3 on the new p_weight+clip base is high-value.

### New assignments (cycle 8)

| PR | Student | Hypothesis |
|---|---|---|
| **#1717** | frieren | `lr: 5e-4 → 1e-3` — LR bracket between current base (5e-4) and fern's 2e-3. Justified by grad-clip step-size cap and effective-batch scaling rule (accum=2). |
| **#1718** | edward | EMA `decay=0.999` — budget-calibrated retry (4.6 half-lives in 3200 steps vs prior 2.3). With clip now in base, live weights change more smoothly → EMA should track better. |

---

## 2026-05-12 22:55 — Cycle 7: stale_wip cleanup + 2 new assignments

Post-merge of PR #1480, the 4 remaining stale_wip PRs were triaged:

- **PR #1475 nezuko (wider 256/8h):** CLOSED. Direction was under-trained at val≈176 against the old baseline; against new merged baseline 116.30 the gap is ~51%. Wider-with-30-min-cap is fundamentally a training-budget bind; no path to recovery within constraints.
- **PR #1476 tanjiro (per-field heads):** CLOSED. Direction landed at val≈137 against old baseline (~18% worse than new merged 116.30). No code pushed, no terminal SENPAI-RESULT, 4h silent. Channel-prioritization via shared backbone + per-channel loss weighting (frieren's `p_weight` direction) is the better mechanism for this axis.
- **PR #1471 frieren (p_weight=2+clip):** SENT BACK with rebase instructions. Branch had merge conflict because #1480 modified train.py. Asked frieren to rebase, apply the redirect from cycle-3 (p_weight=2.0 + grad clip 1.0), and re-run on the new bf16+accum baseline. Acceptance bar updated to val<116.30 + no test-split regression.
- **PR #1465 askeladd (surf_weight=30):** SENT BACK with nudge. No code committed in 4h despite a clear hypothesis. Asked askeladd to rebase against the merged base, commit the surf_weight change, and run on the new baseline.

Two new assignments after closing nezuko/tanjiro:

- **PR #1665 nezuko — `n_layers: 5 → 6` (deeper Transolver):** single config-field change, expected to fit in 30 min budget (~15-16 epochs at 1.2× compute) thanks to the bf16+accum throughput head-room. Tests whether one more block of slot-mixing/attention improves capacity within the existing footprint.
- **PR #1666 tanjiro — `smooth_l1` (Huber β=1) loss replaces MSE:** addresses the eval/train mismatch (train MSE, eval MAE). Bounds gradient magnitude per element, which should help with the p-channel's long-tailed errors without per-sample reweighting (which already failed catastrophically in edward's #1466).

Active in-flight after cycle 7 (6 WIP PRs): #1469 fern (lr=2e-3+clip, active), #1465 askeladd (surf_w=30, sent-back), #1471 frieren (p_w=2+clip, sent-back), #1651 thorfinn (cosine T18, new), #1654 edward (EMA weights, new), #1655 alphonse (OneCycleLR, new), #1665 nezuko (deeper, new), #1666 tanjiro (smooth_l1, new). 0 idle students.

---

## 2026-05-12 20:00 — PR #1471: frieren — pressure channel weight=3 in loss (sent back, not merged)

- Branch: `willowpai2g24h2-frieren/p-channel-weight-3`
- Hypothesis: up-weight pressure (dim 2) inside per-channel `sq_err` by `p_weight=3.0` to direct more gradient at the ranking metric.
- Result: monotone val descent 241 → 130.98 over 14 epochs; W&B runs `ftuclvqz` (first arm, 148.57) and `ph14bsim` (second arm, 130.98, canonical).

### Metrics

| Metric | Value | Baseline | Delta |
|---|---|---|---|
| `val_avg/mae_surf_p` (best `ph14bsim`) | 130.98 | 131.79 / 132.73 | ~-0.6 to -1.3% |
| `val_single_in_dist/mae_surf_p` | 166.71 | 136.34 | +22% (worse) |
| `val_geom_camber_rc/mae_surf_p` | 140.45 | ~130 | +8% (worse) |
| `val_geom_camber_cruise/mae_surf_p` | 98.51 | 117.71 | **-16% (better)** |
| `val_re_rand/mae_surf_p` | 118.23 | 121.79 / 117.71 | mixed |
| `test_single_in_dist/mae_surf_p` | 140.14 | ~135 | mixed |
| `test_geom_camber_rc/mae_surf_p` | 126.71 | TBD | n/a |
| `test_geom_camber_cruise/mae_surf_p` | **NaN** | **NaN** (systemic) | n/a |
| `test_re_rand/mae_surf_p` | 116.95 | TBD | n/a |

### Analysis

The val_avg gain (~1%) is inside noise (the two baseline arms differ by 0.7%) and the first frieren arm regressed by 13% — high variance. The mean-improvement signal is dominated by `val_geom_camber_cruise` (-16%), which is precisely the split where the test counterpart blew up. Up-weighting pressure made cruise val better but pushed the model's p output into overflow territory on the larger test cruise set (200 samples vs 100 val).

Student's diagnostic is correct: the cruise NaN traces to `accumulate_batch` propagating `inf - y` through `mask` arithmetic. The systemic cruise-test NaN affects every run including baseline, so it's not a frieren-specific veto — but the *magnitude* of the p-output blowup at `p_weight=3` is what makes this hypothesis risky.

### Decision

Sent back to student with two changes: drop `p_weight` to 2.0 (less aggressive) and add `clip_grad_norm_(model.parameters(), 1.0)` as a baseline-hardening numerical safety. Same `--wandb_group "willow-r2-p-weight"` so the arms remain comparable. Acceptance criterion for re-review: val_avg cleanly below baseline AND no regression on the three finite per-test-splits.

### Update — 2026-05-12 21:00 (cycle 2)

Frieren ran the re-run in W&B per the send-back. New best `val_avg/mae_surf_p` = **116.34** (run `18f9jjzt`), which is the best in the entire cohort across all students/arms. The configuration delivered as asked: `p_weight=2.0` + `clip_grad_norm_=1.0`. However: student has not yet committed/pushed the updated `train.py` (commit `acf88af` on the branch still reflects the original `p_weight=3.0` change) and has not posted the updated SENPAI-RESULT comment. Awaiting student-side workflow completion before final adjudication.

---

## 2026-05-12 21:00 — Cycle-2 advisor-side observations (no formal submissions)

All 8 students have completed multiple W&B runs but only frieren has any code commit on their branch beyond the empty assign commit. The other 7 PRs are `stale_wip`. Pod logs show students are alive and polling but throttled by GitHub GraphQL API rate limits.

### Live W&B leaderboard (latest best per student)

| Rank | Student / hypothesis | Best W&B run | val_avg/mae_surf_p | Δ vs baseline-median (~131) |
|---|---|---|---|---|
| 1 | frieren / p_weight=2 + clip (re-run) | `18f9jjzt` | **116.34** | -11.2% |
| 2 | fern / lr=2e-3+clip | `m7xp2x4b` | 118.77 | -9.3% |
| 3 | alphonse / baseline (3rd rep) | `z2ls7ol1` | 119.64 | -8.7% |
| 4 | thorfinn / bf16+accum=2 | `zg3qckt7` | 124.60 | -4.9% |
| 5 | askeladd / surf_weight=30 | `dqey3vto` | 127.53 | -2.6% |
| 6 | tanjiro / per-field heads | `0bh0u3h1` | 137.21 | +4.7% |
| 7 | nezuko / wider 256/8h | `shqqxayq` | 176.37 | +34.6% |
| 8 | edward / Huber per-sample | `wxpj1e4u` | 275.04 | +110% |

### Re-calibrated noise floor

The 3 alphonse baseline runs span 119.64–132.73 (13 points, ~10%). The previous "0.5–1%" noise estimate was wrong (it was the spread between 131.79 and 132.73, ignoring the third run). With the true noise band ~10%, frieren (-11%) and fern (-9%) are at the edge of, but plausibly within, noise. Without repeat seeds we can't fully separate signal from noise.

### No formal decisions made this cycle

No PRs were marked ready for review. No merges, send-backs, or closes happened (beyond the pre-existing frieren send-back from cycle 1). The advisor branch was updated with the recalibrated noise floor and cycle-2 W&B observations.

---

## 2026-05-12 21:30 — Cycle-3 observations

### Important discovery: cruise-test NaN root cause + workaround

Two students independently diagnosed the systemic `test_geom_camber_cruise/mae_surf_p = NaN` issue in detailed PR comments:

- **#1466 (edward)** at 21:00 UTC and **#1480 (thorfinn)** at 20:56 UTC both identified that `data/scoring.py:accumulate_batch` has a `0 * Inf = NaN` propagation bug when a batch contains a sample with non-finite `y` values. Specifically: `test_geom_camber_cruise` sample 20 has 761 nodes with `y_p = -Inf`. The per-sample skip logic in `accumulate_batch` is defeated by the masked-multiply at the end.

- Both students implemented identical workarounds in `train.py:evaluate_split` (sanitize `y` and gate `mask` per-sample before calling `accumulate_batch`).

- Edward verified the workaround on their best checkpoint (run `wxpj1e4u`): `test_avg/mae_surf_p = 257.22` (was NaN), `test_geom_camber_cruise/mae_surf_p = 156.58` (was NaN).

- **`data/scoring.py` is read-only per `program.md`** — neither student modified it. Both fixes live in `train.py`.

**Implication:** when these fixes are committed and merged, every future run on this branch should produce a finite `test_avg`. This unlocks the paper-facing metric. The fix is hypothesis-agnostic and should be merged as a baseline-hardening change even if the surrounding hypothesis (edward's Huber, thorfinn's bf16+accum) doesn't win on val.

### Cycle-3 leaderboard (live W&B, unchanged in ranks since cycle 2)

| Student | Best val_avg | Δ vs best baseline (119.64) | Running now |
|---|---|---|---|
| frieren | 116.34 | -2.8% | fshtpt6z (new arm) |
| fern | 118.77 | -0.7% | 2ny5alj3 (after j6ugv3ik crash) |
| alphonse | 119.64 | reference | 7uv601md (after ytujykqu crash) |
| thorfinn | 124.60 | +4.1% | 5wvm7na2 |
| askeladd | 127.53 | +6.6% | 3cv4bxtr + qhnzquax (2 arms) |
| tanjiro | 137.21 | +14.7% | 2wlx399x |
| nezuko | 176.37 | +47.4% | wp67vqws |
| edward | 275.04 | +130% | 4bplylk3 |

- All 8 students have currently-running W&B runs (active iteration).
- 2 crashes since 20:00 UTC: `j6ugv3ik` (fern), `ytujykqu` (alphonse). Both already have follow-up arms running.
- No new finished runs have produced a finite `test_avg` yet — the bug fixes haven't been committed/pushed to PR branches, so new training runs aren't using the workaround.

### No decisions this cycle

All 8 PRs remain draft `status:wip`. No code commits on 7 of 8 branches (frieren has a partial commit). No SENPAI-RESULT terminal markers. Advisor held back on per-student nudge comments to avoid further burning the shared GraphQL rate-limit budget (visible in pod logs as 6-retry token exhaustion per heartbeat).

---

## 2026-05-12 21:45 — Cycle-4 observations

### Third independent cruise-NaN diagnosis (alphonse, #1461)

Alphonse (baseline PR #1461) posted a detailed diagnostic at 21:15 UTC — independently arriving at the same root cause as edward (#1466) and thorfinn (#1480). New facts added by alphonse's analysis:

- **Data scan:** scanned all 8 val/test splits (1000 files); `test_geom_camber_cruise/000020.pt` is the *only* file with non-finite `y` across the entire test/val set. `y[:, 2]` (pressure) has `-Inf` on 761 nodes.
- **Behaviour confirmed:** all four W&B runs on this branch's project show identical `test_geom_camber_cruise/{mae_surf_p, mae_vol_p} = NaN` and `test_avg/mae_surf_p = NaN`. The other three test splits are clean.
- **Resolution path the student proposed:** (1) repair the data file, (2) fix scoring.py with `torch.where`, or (3) flag and accept the NaN.

### Advisor decision recorded on #1461

Posted advisor comment at 21:45 UTC ruling each path:

1. **Data fix:** out of scope for this isolated launch — dataset is fixed for the controlled ablation.
2. **scoring.py fix:** out of scope — `program.md` declares `data/scoring.py` read-only and we don't modify it during this launch.
3. **In-scope path:** the `train.py:evaluate_split` sanitize-and-gate workaround already prototyped on #1466 and #1480 is the right vehicle. Whichever of those PRs commits + finalizes first will land as the bug-fix.

Also explicitly authorized alphonse's `test_avg/mae_surf_p_excluding_bad_sample` post-hoc recompute from the best existing checkpoint (`hqj9bt84`) — that becomes the canonical baseline test number for the round, usable *before* the eval-time workaround lands.

### State unchanged otherwise

| Item | Cycle-3 state | Cycle-4 state | Δ |
|---|---|---|---|
| Review-ready PRs | 0 | 0 | none |
| Idle students | 0 | 0 | none |
| Stale_wip PRs | 5 | 4 (alphonse moved off after comment) | -1 |
| Code commits beyond `assign` | 1 (frieren partial) | 1 (frieren partial) | none |
| Terminal SENPAI-RESULT markers | 0 | 0 | none |
| Active student pods | 8/8 Ready | 8/8 Ready | none |
| Human issues | 0 | 0 | none |

### No formal decisions / merges this cycle

No PRs marked ready for review; no merges, closes, or send-backs (beyond the advisor comment on #1461 acknowledging alphonse's flag, which is informational, not a state change).

---

## 2026-05-12 22:00 — Cycle-5 reviews: #1480 thorfinn + #1461 alphonse

Two PRs reached review-ready state this cycle.

### PR #1480 thorfinn — bf16 autocast + grad accumulation=2

**SENPAI-RESULT (terminal, run `5wvm7na2`):**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **116.2965** |
| `test_avg/mae_surf_p` | **104.9554** (finite — bug fix included!) |
| Epochs in 30 min | 18 |
| `test_geom_camber_cruise/mae_surf_p` | 80.35 |

Per-split test `mae_surf_p`:

| Split | test |
|---|---|
| `single_in_dist` | 115.83 |
| `geom_camber_rc` | 117.06 |
| `geom_camber_cruise` | **80.35** (1 sample skipped) |
| `re_rand` | 106.58 |

**Analysis:** This is exceptional. val_avg=116.30 beats every other run in the cohort (frieren's 116.34 is essentially tied, one run). More importantly, `test_avg=104.96` is the **first finite test_avg in the project** — enabled by thorfinn's `train.py:evaluate_split` per-sample pre-filter workaround. The throughput hypothesis confirmed: 18 epochs vs ~14 baseline in 30 min. bf16 + grad_accum=2 is a robust win.

**Decision: SENT BACK.** The branch (`willowpai2g24h2-thorfinn/bf16-amp-grad-accum-2`) has only the empty `assign` commit — no code. All W&B runs were from locally-applied changes that were never pushed. Cannot merge an empty PR; the squash-merge would not carry bf16/grad_accum/bug-fix onto the advisor branch. Student must commit and push the three changes (bf16 autocast, grad_accum=2 loop, evaluate_split workaround) then re-mark for review.

**This is the highest-priority merge in the round** once the code is committed. It simultaneously lands the throughput win and the cruise-NaN workaround for all subsequent PRs.

### PR #1461 alphonse — stock-config baseline

**SENPAI-RESULT (terminal, run `ztb0ri42`):**

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 140.01 (epoch 13 / 30.9 min) |
| `test_avg/mae_surf_p` | NaN (as expected — no code fix in this PR) |
| `test_avg/mae_surf_p_excluding_bad_sample` | **126.20** (workaround recompute) |

Per-split test `mae_surf_p`:

| Split | test (raw) | test (excl. 000020) |
|---|---|---|
| `single_in_dist` | 143.45 | 143.45 |
| `geom_camber_rc` | 135.17 | 135.17 |
| `geom_camber_cruise` | NaN | **98.06** (199/200) |
| `re_rand` | 128.13 | 128.13 |

**Analysis:** Baseline delivered as promised. The val_avg=140.01 is the worst of the 4 stock baseline runs, confirming the ~17% noise band. The `_excluding_bad_sample=126.20` is the canonical pre-fix test comparator for the round. Alphonse also ran a full data scan: 000020.pt is the only bad file across all 1000 val/test samples. Third independent cruise-NaN diagnosis.

**Decision: CLOSED.** No code on the branch (correctly — stock config baseline). Deliverables (baseline measurement + workaround comparator + data scan + diagnostic) are fully in the comments and recorded in BASELINE.md. BASELINE.md updated.

### New assignment issued

Alphonse was immediately re-assigned **PR #1631** (`cruise-nan-eval-fix`): implement the `train.py:evaluate_split` sanitize-and-gate workaround (per-sample keep=pred_finite & y_finite), run stock-config baseline, produce the first advisor-merged finite `test_avg/mae_surf_p`. Once #1631 lands, all subsequent PRs get finite test_avg for free.

### Updated BASELINE.md

- Noise band updated: 4 baseline runs, 119.64–140.01 (~17%)
- Canonical pre-fix test comparator: `test_avg/mae_surf_p_excluding_bad_sample = 126.20`
- Thorfinn's 104.96 noted as "pending merge" — highest-priority once code is committed

---

## 2026-05-12 22:15 — Cycle-6: #1480 merged, #1466 closed, three new assignments

### PR #1480 thorfinn — MERGED ✓ (val=116.30, test=104.96)

Code was committed on cycle-6 entry (`f8c1c40`). All three changes confirmed in diff:
1. bf16 autocast on forward+loss (`Config.amp=True`, `torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp)`)
2. Gradient accumulation=2 (`Config.grad_accum=2`, accumulation boundary logic in training loop)
3. `evaluate_split` per-sample sanitize-and-gate workaround (`keep = pred_finite & y_finite`, fp32 eval, `n_samples_skipped` logged)

New baseline: **val=116.30 / test=104.96** (first finite test_avg in the project). cruise-NaN workaround now on advisor branch for all subsequent PRs.

### PR #1466 edward — CLOSED (Huber direction broken)

Final run `4bplylk3`: val=324.66, test=305.82. ~150% worse than baseline. Student's own analysis: "direction did not pan out — per-sample Huber-norm convergence is 3-4× slower than MSE in this normalized space." Bug fix in evaluate_split was correct but superseded by thorfinn's more complete implementation, already merged.

### PR #1631 alphonse — CLOSED (redundant after #1480 merge)

The cruise-NaN workaround it was targeting landed via #1480. No need to land a second implementation.

### New assignments issued

| PR | Student | Hypothesis | Key change |
|---|---|---|---|
| #1651 | thorfinn | Cosine T_max recalibration | `epochs=18` so CosineAnnealingLR fully anneals within 30-min budget |
| #1654 | edward | EMA model weights | Shadow EMA copy for eval (`decay=0.9995`); zero extra training FLOPs |
| #1655 | alphonse | OneCycleLR max_lr=2e-3 | Warmup→anneal schedule at 4× baseline LR; different shape from standard cosine |

All three build on the merged bf16+accum=2 baseline. Beat val=116.30 to be a winner.
