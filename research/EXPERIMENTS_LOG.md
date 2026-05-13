<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2g-24h-r2`

Primary metric: `val_avg/mae_surf_p` (lower is better).
**Active baseline (PRs #1480 + #1471 merged):** `val_avg/mae_surf_p=110.27`, `test_avg/mae_surf_p=99.41` (run `krsv4c21`, p_weight=2.0+clip_grad_norm=1.0 stacked on bf16+grad_accum=2).

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
