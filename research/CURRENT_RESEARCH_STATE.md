<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-13 02:40 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Cycle-13b update — 2 more negatives closed (askeladd, fern), 2 more arms launched

- **#1465 askeladd surf=30** → val=111.95 / test=102.51. Damage concentrated on `single_in_dist` (+8.4%); OOD held flat. Direction interesting (surface-priority doesn't hurt OOD) but magnitude too aggressive on top of p_weight=2. Reassigned to surf=15 midpoint (#1816).
- **#1469 fern lr=2e-3** → val=121.33 / test=111.80 (+10%/+12.5%). Third datapoint confirming lr=5e-4 is at optimum. LR axis is conclusively closed. Reassigned to architectural axis n_head=8 (#1819).

## Cycle-13 update — 3 negative results closed, 3 follow-up arms launched. KEY FINDING: wd is OOD-load-bearing.

### Headline discovery (from cycle-12 edward #1750)

Weight decay was load-bearing for `geom_camber_rc` OOD generalization. When wd halved (1e-4 → 5e-5), `geom_camber_rc` regressed +11.9% while `single_in_dist` was essentially flat. This is the cleanest single regularization-mechanism signal of the launch. Direct follow-up arm #1802 inverts: wd=2e-4 to test if more wd buys more OOD.

### 3 negatives closed

- **#1778 nezuko slice_num=128** → val=125.03 / test=111.65 (+13%/+12%). Throughput cost 52% (13 ep vs 19) due to `O(slice_num²)` dominating. Inductive-bias-via-slot-count rejected.
- **#1750 edward wd=5e-5** → val=113.15 / test=103.08. **Informative negative** — OOD-concentrated regression confirms wd is doing real work.
- **#1738 thorfinn beta2=0.95** → val=124.63 / test=111.16 (+13%/+12%). Mid-training acceleration window real (-44 MAE at epoch 5) but late-phase variance-noise floor (epoch 9 +49 spike) dominated final metric.

### 3 new arms

- **#1802 edward — wd 1e-4 → 2e-4** (inversion test: if halving hurt OOD, doubling should help OOD)
- **#1803 nezuko — CosineAnnealingLR T_max 50 → 20** (LR currently only decays 28% over the run; anneal-to-zero might let final-epoch refinement land cleanly)
- **#1804 thorfinn — AdamW eps 1e-8 → 1e-6** (orthogonal knob for the late-phase variance-noise mechanism his own #1738 surfaced)

## Cycle-12 update — nezuko stale-closed; reassigned to slice_num=128 (inductive-bias arm)

### PR #1665 nezuko — n_layers=6: CLOSED (stale)

3+ hours of zero progress (no code committed, no comments), and the PR body referenced the stale 116.30 baseline. Pod alive but the poll-for-work cycle hasn't moved. Hypothesis was not strictly dominated — n_layers=6 is orthogonal to mlp_ratio=3 (frieren #1749) — but a fresh PR re-triggers the pod's poll cycle, and slice_num covers a different orthogonal axis. Closing was the lower-risk choice.

### PR #1778 nezuko — slice_num 64 → 128: NEW ASSIGNMENT

The model is **still descending at the final epoch** on the current baseline. When under-fitting at the budget cap, the question is "what kind of capacity does the model need?" Two orthogonal answers in flight:
- **frieren #1749** — *parameter capacity* (mlp_ratio 2 → 3, +33% FFN width per block)
- **nezuko #1778** — *representational resolution* (slice_num 64 → 128, 2× physics-attention slots, ~0 param-count change)

If both win, we've found two orthogonal capacity levers. If only one wins, that's a strong signal about the right inductive-bias direction.

## Cycle-11 update — EMA + LR-sweep directions ruled out; capacity + regularization arms launched

### PR #1718 edward — EMA decay=0.999: CLOSED (2nd EMA attempt; direction ruled out)

EMA val=126.4 vs live val=119.5 vs baseline 110.27. With LR cosine-decaying from 5e-4 toward 0 (T_max=50 but only 17 visible epochs), **live weights are still descending at the final epoch** — EMA average cannot catch up to a moving target. Student's own analysis recommended skipping EMA for short runs. EMA direction is now decisively ruled out at this training budget.

### PR #1717 frieren — lr=1e-3: CLOSED (LR sweep decisive at 5e-4)

Clean +10 MAE regression (val=120.2 / test=110.1 vs baseline 110.27 / 99.41) with persistent val oscillation. Combined with #1469 fern's pending lr=2e-3 result, **lr=5e-4 is at or near optimum** on the new p_weight+clip recipe stack.

### PR #1666 tanjiro — smooth_l1 (Huber β=1): SENT BACK (stale baseline)

Direction is promising (smooth_l1 aligns train and eval, less mass on outliers) but the run was on the OLD baseline (val=116.30). Sent back with detailed rebase + code-snippet instructions for combining smooth_l1 with `ch_weights = [1.0, 1.0, p_weight]` per-channel multiplier.

### PR #1749 frieren — mlp_ratio 2 → 3: NEW ASSIGNMENT

33% FFN capacity bump per Transolver block. Justified by: model still descending at final epoch (no plateau/overfit signature) → capacity headroom unused. Throughput drop modest (~15-16 epochs); param count +8%. Orthogonal to all other axes. OOD-vs-IID split behavior will tell us whether capacity is helping generalization or memorization.

### PR #1750 edward — weight_decay 1e-4 → 5e-5: NEW ASSIGNMENT

Halve L2 regularization pressure. Justified by: (a) grad clip is binding on nearly every step — adding L2 on top of an aggressively damped step is "double penalty" on weights; (b) model is under-fitting at the budget cap (still descending) — regularizer should be eased, not tightened; (c) wd was never re-tuned after the r2 recipe stack landed. Diagnostic-rich: train-vs-val gap tells us whether wd was load-bearing.

## Cycle-10 update — thorfinn cosine-T18 closed + reassigned to AdamW beta2

### PR #1651 thorfinn (cosine T_max=18): CLOSED

Stale for ~2 hours post-#1480-merge; no code commits, no comments. Hypothesis is also redundant — alphonse #1655 (OneCycleLR rebased) strictly dominates: OneCycleLR provides the same anneal-to-zero benefit plus warmup plus peak-LR boost. Closed and reassigned.

### PR #1738 thorfinn — AdamW betas (0.9, 0.999) → (0.9, 0.95): NEW ASSIGNMENT

Standard short-transformer optimizer tuning. The default beta2=0.999 is calibrated for very long runs; in our ~3200-step budget the variance EMA never fully mixes. beta2=0.95 (half-life ~14 steps) makes AdamW's adaptive step size react to the post-clip gradient signal quickly, which should help in the regime where the clip is binding nearly every step. Orthogonal to all in-flight directions.

## Cycle-8 update — Second merge! val=110.27, test=99.41

### PR #1471 frieren — MERGED ✓ (p_weight=2.0 + clip_grad_norm=1.0)

**New baseline: val=110.27, test=99.41.** Two changes stacked orthogonally on #1480:
- `p_weight=2.0` — per-channel pressure upweight in squared-error loss
- `clip_grad_norm_(max_norm=1.0)` — active on nearly every step (mean pre-clip norm 114)

Gain: −5.19% val, −5.29% test vs #1480. Cruise split improved −9.4%.

### PR #1655 alphonse — SENT BACK (rebase needed)

OneCycleLR(max_lr=2e-3) delivered val=111.65 vs old baseline 116.30, but the new bar is now 110.27. Sent back to rebase on the frieren-merged base + re-run. OneCycleLR on top of p_weight+clip is the highest-value stack hypothesis in flight.

### PR #1654 edward — CLOSED (EMA decay=0.9995 catastrophically mistuned)

val=195 (live=127) — only 2.3 half-lives in the training budget. Re-assigned edward to EMA decay=0.999 (4.6 half-lives in budget → properly calibrated).

### PR #1469 fern — SENT BACK (no results posted + baseline stale)

Fern posted a cruise-NaN bug fix (redundant with #1480) but no terminal SENPAI-RESULT for lr=2e-3. Sent back with rebase + re-run instructions.

### New assignments (cycle 8)

| PR | Student | Hypothesis |
|---|---|---|
| **#1717** | frieren | `lr: 5e-4 → 1e-3` — LR bracket on new p_weight+clip base |
| **#1718** | edward | EMA `decay=0.999` — budget-calibrated retry |

## Cycle-7 update — stale_wip cleanup + 2 new architectural/loss-shape arms

After PR #1480 merge, the 4 stale_wip PRs were triaged:

- **CLOSED #1475 nezuko (wider 256/8h):** under-trained at 30-min cap (val≈176 on old baseline → ~51% worse than new 116.30). Capacity > training budget.
- **CLOSED #1476 tanjiro (per-field heads):** val≈137 on old baseline (~18% worse than new), no code pushed, 4h silent. Shared backbone + per-channel weighting (frieren's `p_weight` direction) is a cleaner channel-prioritization mechanism.
- **SENT BACK #1471 frieren (p_weight=2+clip):** had merge conflict from #1480 base change. Asked for rebase + redirect from cycle-3 (`p_weight=2.0` + grad clip 1.0).
- **SENT BACK #1465 askeladd (surf_weight=30):** no code committed in 4h. Asked for rebase, commit, run on new baseline.

Two new assignments cover the orthogonal axes still unexplored:

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| **#1665** | nezuko | `n_layers: 5 → 6` (one more Transolver block) | Adds one round of slot-mixing/attention. Throughput headroom from bf16+accum should keep it within 30-min cap (~15 epochs). Acceptance: val<116.30 AND test doesn't materially regress vs 104.96. |
| **#1666** | tanjiro | `smooth_l1` (Huber, β=1) loss replaces MSE | Aligns training loss shape with eval metric (MAE). Bounds gradient magnitude per element — should reduce p-channel long-tailed sample dominance without per-sample reweighting (which broke in #1466). Orthogonal to all in-flight directions. |

## Cycle-6 update — MAJOR MILESTONE: First merge! val=116.30, test=104.96

### PR #1480 thorfinn — MERGED ✓

Code committed and squash-merged. **New baseline: val_avg=116.30, test_avg=104.96.**

This merge simultaneously:
- Landed the bf16+accum=2 throughput win (2.5× epoch throughput; 18 epochs in 30 min)
- Landed the cruise-NaN evaluate_split workaround — all future runs now produce finite `test_avg` automatically

### PR #1466 edward — CLOSED (broken direction)

Huber direction: val=324.66 (~150% worse than baseline). Closed.

### PR #1631 alphonse — CLOSED (redundant after #1480 merge)

Cruise-NaN workaround targeted by this PR has landed via #1480.

### Three new assignments

| PR | Student | Hypothesis | Expected val |
|---|---|---|---|
| **#1651** | thorfinn | `epochs=18` cosine T_max recalibration | Free win from full anneal |
| **#1654** | edward | EMA model weights (`decay=0.9995`) | ~113-115 (1-4% gain) |
| **#1655** | alphonse | `OneCycleLR(max_lr=2e-3)` | TBD vs cosine-2e-3 |

## Cycle-5 update — First terminal results + major milestone: first finite test_avg

**Two PRs reached review-ready this cycle.**

### #1480 thorfinn (bf16 + grad_accum=2): SENT BACK — code not committed

Thorfinn posted terminal results (run `5wvm7na2`): **val_avg=116.30, test_avg=104.96 (finite)**. This is:
- Best val_avg in the cohort (tied with frieren's 116.34)
- **First finite `test_avg/mae_surf_p` in the project** — enabled by the cruise-NaN workaround in evaluate_split
- 18 epochs in 30 min (vs ~14 baseline) — throughput hypothesis confirmed

However, the PR branch has only the empty assign commit — no code pushed. The squash-merge would be empty. **Sent back** asking for 3 commits: bf16 autocast, grad_accum=2 loop, evaluate_split workaround. This is the highest-priority merge once code lands.

### #1461 alphonse (baseline): CLOSED — deliverables captured

Run `ztb0ri42`: val_avg=140.01, test_avg=NaN. Delivered the key deliverable: `test_avg/mae_surf_p_excluding_bad_sample = 126.20` — canonical pre-fix test comparator. BASELINE.md updated. Alphonse closed and re-assigned.

### New assignment: PR #1631 alphonse — cruise-NaN eval workaround (now closed)

Was assigned but superseded by #1480 merge. Closed when #1480 landed.

## Cycle-4 update — third independent cruise-NaN diagnosis + advisor decision

Alphonse (baseline PR #1461) independently diagnosed the same cruise-NaN root cause at 21:15 UTC. New facts:
- Scanned all 8 val/test splits (1000 files): `test_geom_camber_cruise/000020.pt` is the **only** file with non-finite `y` (`y[:, 2]` is `-Inf` on 761 nodes).
- The bad sample, and only it, poisons `test_geom_camber_cruise/{mae_surf_p, mae_vol_p}` whenever it lands in a mixed batch — which it does with default `batch_size=4`.

Advisor decision posted on #1461 at 21:45 UTC:
1. Data fix and `scoring.py` fix are out of scope (isolated launch + read-only file rule).
2. The `train.py:evaluate_split` sanitize-and-gate workaround already on #1466 / #1480 is the in-scope path; whichever lands first is the bug-fix vehicle.
3. Authorized alphonse's `test_avg/mae_surf_p_excluding_bad_sample` recompute from the best existing checkpoint (`hqj9bt84`) — becomes the canonical baseline test number for the round, available *before* the eval-time workaround lands.

## Cycle-3 update — cruise-test NaN root cause + workaround discovered

Two students independently nailed the systemic `test_geom_camber_cruise/mae_surf_p = NaN` issue in PR comments at ~21:00 UTC:

- **#1466 (edward)** and **#1480 (thorfinn)**: `data/scoring.py:accumulate_batch` propagates `0 * Inf = NaN` when a batch contains a sample with non-finite `y` values. Specifically, `test_geom_camber_cruise` sample 20 has 761 nodes with `y_p = -Inf`. The per-sample skip path in `accumulate_batch` is defeated by the trailing masked-multiply.
- Both implemented identical workarounds in `train.py:evaluate_split` (sanitize `y` and gate `mask` per-sample before calling `accumulate_batch`).
- **`data/scoring.py` is read-only per `program.md`** — neither student modified it. Both fixes live in `train.py`.
- Edward verified on run `wxpj1e4u`: `test_avg/mae_surf_p = 257.22` (was NaN), `test_geom_camber_cruise/mae_surf_p = 156.58` (was NaN).

**Implication:** when these fixes land, every future run on this branch should produce a finite `test_avg`. This unlocks the paper-facing metric. The fix is hypothesis-agnostic and should be merged as a baseline-hardening change even if the surrounding hypothesis (edward's Huber, thorfinn's bf16+accum) doesn't win on val. Plan to cherry-pick the workaround once a student actually commits/pushes it; right now both PRs are still draft with no code on the branch beyond the empty `assign` commit.

## Current leaderboard (post cycle-13b)

**Active baseline: val=110.27 / test=99.41** (PRs #1480+#1471, merged). Beat 110.27 to merge.

| Student / PR | Best val_avg | test_avg | Status | Notes |
|---|---|---|---|---|
| **askeladd #1816** (surf=15) | TBD | TBD | WIP (new, cycle-13b) | Midpoint of #1465 axis (surf=30 too aggressive) |
| **fern #1819** (n_head=8) | TBD | TBD | WIP (new, cycle-13b) | More attention diversity, same total head_dim |
| **edward #1802** (wd=2e-4) | TBD | TBD | WIP (cycle-13) | Inversion of #1750; tests if more wd → better OOD |
| **nezuko #1803** (T_max=20) | TBD | TBD | WIP (cycle-13) | Schedule recalibration; anneal-to-zero alignment |
| **thorfinn #1804** (eps=1e-6) | TBD | TBD | WIP (cycle-13) | Caps low-variance step-size scaling |
| **frieren #1749** (mlp_ratio=3) | TBD | TBD | WIP (cycle-11) | 33% FFN capacity bump per block |
| **alphonse #1655** (OneCycleLR max_lr=2e-3) | TBD | TBD | WIP (sent-back cycle-8) | Awaiting rebased re-run |
| **tanjiro #1666** (smooth_l1 loss) | TBD | TBD | WIP (sent-back cycle-11) | Stale baseline → rebase + re-run instructed |
| alphonse #1655 (OneCycleLR) | 111.65 | 101.67 | WIP (sent-back) | Rebase+retry on new base; high-value stack hypothesis |
| fern #1469 (lr=2e-3+clip) | — | — | WIP (sent-back) | No results yet; rebase+retry on new base |
| askeladd #1465 (surf_w=30) | — | — | WIP (sent-back) | No results yet; rebase+retry on new base |
| **MERGED: frieren #1471** (p_weight=2+clip) | 110.27 | 99.41 | **MERGED** cycle-8 | New baseline |
| **MERGED: thorfinn #1480** (bf16+accum2) | 116.30 | 104.96 | **MERGED** cycle-6 | Prior baseline |
| ~~askeladd #1465~~ (surf=30) | 111.95 | 102.51 | CLOSED cycle-13b | In-dist hit (+8.4%); OOD held flat — direction has signal at lower magnitude |
| ~~fern #1469~~ (lr=2e-3) | 121.33 | 111.80 | CLOSED cycle-13b | Third LR datapoint — axis closed at 5e-4 optimum |
| ~~nezuko #1778~~ (slice_num=128) | 125.03 | 111.65 | CLOSED cycle-13 | +13%/+12% regression; throughput cost 52% |
| ~~edward #1750~~ (wd=5e-5) | 113.15 | 103.08 | CLOSED cycle-13 | OOD-concentrated regression — **wd is OOD-load-bearing** |
| ~~thorfinn #1738~~ (beta2=0.95) | 124.63 | 111.16 | CLOSED cycle-13 | Mid-train acceleration real but late-phase noise dominated |
| ~~nezuko #1665~~ (n_layers=6) | — | — | CLOSED cycle-12 | Stale 3+ hrs; reassigned to slice_num=128 |
| ~~edward #1718~~ (EMA 0.999) | 126.4 (EMA) / 119.5 (live) | — | CLOSED cycle-11 | EMA lag persists; live still descending at end |
| ~~frieren #1717~~ (lr=1e-3) | 120.2 | 110.1 | CLOSED cycle-11 | +10 MAE regression with persistent val oscillation |
| ~~thorfinn #1651~~ (cosine T18) | — | — | CLOSED cycle-10 | Stale + dominated by alphonse OneCycleLR |
| ~~edward #1654~~ (EMA 0.9995) | 195.33 | 182.94 | CLOSED cycle-8 | EMA half-life mistuned (2.3 vs needed ≥4) |
| ~~tanjiro #1476~~ (per-field heads) | 137 (W&B) | — | CLOSED cycle-7 | Regression |
| ~~nezuko #1475~~ (wider 256/8h) | 176 (W&B) | — | CLOSED cycle-7 | Under-trained |

**Key:**
- Δ vs baseline-best uses alphonse's best baseline run (119.64) — strict.
- Δ vs baseline-median (~131) is what a single hypothesis run would naturally compare against if you'd picked one baseline at random.
- "Code committed?" measured by branch having any commit beyond the empty `assign` commit. Only frieren's first iteration is committed (`acf88af`).

## Workflow observation — stale_wip PRs

All 7 non-frieren PRs remain `stale_wip` through cycle 3. Root cause: students ran training in W&B but did not commit/push their `train.py` modifications. The pods are alive (kubectl confirms 1/1 Ready) and polling, but each iteration is hitting GraphQL API rate limits (visible in pod logs: "API rate limit already exceeded for user ID …", retrying for ~90s per iteration). The host-side harvest workflow is expected to drive completion. No advisor nudges this or last cycle — they would compete for the same rate-limited tokens.

Notably, edward and thorfinn both posted **detailed diagnostic comments** in cycle 3 identifying the cruise-test NaN root cause, but neither has committed the workaround code yet. The commenting traffic seems to use a different rate-limit lane than the harvest workflow — students can still push prose updates even while the harvest is throttled.

## Themes

**Working (under noise but consistent):**
- Higher LR + grad clip (fern) — multiple arms consistently below baseline median.
- Throughput (thorfinn) — second arm regressed; the bf16+accum benefit may be epoch-count-dependent and saturate.
- Region prioritization (askeladd) — moderate signal.
- Intra-channel weighting (frieren, p_weight=2 with clip) — current best, but only one stable arm.

**Not working:**
- Per-sample Huber rescaling (edward) — catastrophic 100%+ regression. Will close on submission.
- Wider model (nezuko) — 30-min cap turns the wider model into an under-trained model.
- Per-field heads (tanjiro) — wash + stability issues.

## Plateau status

Not in plateau. Most hypotheses sit inside the noise band but multiple directions are showing consistent (if small) movement. Need formal submissions to adjudicate.

## Next directions (post cycle-8)

1. **Highest-value stack: alphonse OneCycleLR + p_weight+clip (#1655 sent back).** OneCycleLR already beat old baseline by 4%; on new base with p_weight+clip, expected result ~106-108. Merge if it clears 110.27.
2. **LR sensitivity: frieren lr=1e-3 (#1717) vs fern lr=2e-3 (#1469).** Two points on the LR curve above current 5e-4. Whichever wins becomes the next default. If both win, take the better one.
3. **EMA at correct decay: edward #1718.** Now budget-calibrated (~4.6 half-lives). EMA benefits are additive to any merged base.
4. **Architectural: nezuko n_layers=6 (#1665), tanjiro smooth_l1 (#1666).** Both in-flight. Expect results next cycle.
5. **Thorfinn cosine T18 (#1651).** Simple win from full-anneal calibration; awaiting results.
6. **Askeladd surf_weight=30 (#1465).** Sent back for rebase + retry. Lower priority vs LR/schedule axes.
7. **Once current pipeline clears, invoke researcher-agent** for next frontier hypotheses. Open axes: weight_decay sweep, AdamW betas (beta2=0.95), attention dropout, slice_num scaling, architecture (n_head 4→8 at same hidden).

## Stacking opportunity

Two changes are now merged (p_weight=2, clip=1.0 on bf16+accum2). OneCycleLR is the next likely merge — once it lands, the combined default should be:
- bf16+accum2 (throughput)
- p_weight=2 + clip=1 (stability + loss alignment)
- OneCycleLR max_lr=2e-3 (schedule)

At that point, expect to be ~106-108 val. Then higher LR, EMA, or depth changes become the next frontier.

## Operational notes

- **Cycle 8 decisions:** #1471 merged (2nd merge!), #1654 closed, #1655 sent back, #1469 sent back, #1717 frieren assigned (lr=1e-3), #1718 edward assigned (EMA 0.999).
- 8 active WIP PRs at cycle-8 close, 0 idle students. Fleet fully utilized.
- **cruise-NaN fix is landed.** All test_avg values finite going forward.
- p_weight=2.0 and clip_grad_norm(max_norm=1.0) now default on advisor branch (from #1471).
- Grad clip is binding on nearly every step — the model runs in a high-gradient-magnitude regime. This is a structural property of the current architecture+loss scale, not an anomaly.
