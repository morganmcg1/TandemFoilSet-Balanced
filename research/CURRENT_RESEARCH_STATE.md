# SENPAI Research State

- **Date:** 2026-05-17 18:30
- **Launch:** willow-pai2i-48h-r1 (round 32 — Lookahead-Lion era; **⚠️ PLATEAU PROTOCOL ACTIVE**; **PROGRAMME ALL-TIME BEST val=45.7284 / test=44.5079 SEED=0** (PR #4402); **5-seed canonical val=46.83±0.41 SEM / test=45.49±0.40 SEM (PAPER-READY)**; **⚠️ MAJOR FINDING (round-31): Lion WD has been an fp32 NO-OP at wd ≤ 1e-4 for the ENTIRE programme** (tanjiro #4456 bit-identical proof); **23 closures since merge, ZERO improvements**; **⚠️ FIRST POSSIBLE WIN: alphonse #4521 wd=3e-3 single-seed val=45.49 (Δ −0.24) — SENT BACK for 4-seed replication**; **α-axis FULLY RESOLVED; LR-axis FULLY RESOLVED; warm restarts BUDGET-INFEASIBLE; SWA/EMA doesn't compose with Lookahead**; **8 experiments in flight: wd=1e-3 + wd=3e-3-seed-replication + outer momentum + focal-loss + Huber + SWA-last-4 + LLRD + joint (k=7,β2=0.9957)**)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## ⚠️ MAJOR FINDING — round 31 (tanjiro PR #4456): Lion WD is fp32 NO-OP at wd ≤ 1e-4

Tanjiro's WD=5e-5 probe returned **BIT-IDENTICAL** to #4402 across val and all 4 test splits to 10 decimal places. Diagnosis: Lion's WD step `p.data.mul_(1.0 - lr * wd)` at `lr_lion = cfg.lr/3 = 1.667e-4` and `wd = 1e-4` gives `lr*wd ≈ 1.67e-8`, which is BELOW fp32's half-ulp threshold (~2.98e-8) at 1.0 — so `(1.0 - lr*wd) → 1.0` in fp32 and the update is a no-op.

**Every Lion+Lookahead winner in BASELINE.md (#4123, #4269, #4373, #4402) has been running with effective wd=0.** The single-seed headline test=44.5079 was achieved WITHOUT weight decay. Implicit regularization (Lookahead averaging + β2=0.995 m-buffer + cosine LR) is doing all the work.

| nominal wd | lr*wd | vs fp32 half-ulp 2.98e-8 | effective? |
|---|---|---|---|
| 1e-5 → 1e-4 | 1.67e-9 → 1.67e-8 | <1× | NO (no-op) |
| 2e-4 | 3.33e-8 | 1.12× | barely (confirmed #4482 +0.07 within noise) |
| 1e-3 | 1.67e-7 | 5.6× | active (tanjiro #4518) |
| 3e-3 | 5.00e-7 | 16.8× | active (alphonse #4521) |
| 1e-2 | 1.67e-6 | 56× | active (frieren #4523) |

**Paper implication:** Lookahead-Lion-β2=0.995-k=6 result must be reported as "without explicit weight decay" — the result is intrinsic to the smoothing compound, not WD-aided.

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=6/α=0.7) + Lion β2=0.995 + triple-stack (PROGRAMME BEST)** | **45.7284** | **44.5079** | PR #4402, W&B `ejacndhj`, seed=0 | Merged 2026-05-17 11:00; super-additive compound |

Win threshold: **val < 45.7284**. Prior best: val=46.8383 (PR #4373, k=5).

### 5-seed canonical of NEW programme best (COMPLETE — round-31; PAPER-READY)

| Seed | val_avg | test_avg | W&B | best_ep | Source |
|---|---|---|---|---|---|
| 0 | **45.7284** | **44.5079** | `ejacndhj` | 17 | PR #4402 MERGED |
| 1 | 48.1367 | 46.3894 | `yqdvl3nr` | 17 | PR #4428 closed (round-26) |
| 2 | 46.9893 | 45.9393 | `0sqdsm53` | 17 | PR #4429 closed (round-26) |
| 3 | 46.2238 | 44.5338 | `gklk517h` | 17 | PR #4457 closed (round-28) |
| 4 | 47.0672 | 46.0682 | (frieren) | 17 | PR #4498 closed (round-31) |
| **mean** | **46.8291** | **45.4877** | — | — | |
| **σ̂_sample (ddof=1)** | **0.9176** | **0.8979** | — | — | |
| **σ̂_mean (SEM)** | **0.4105** | **0.4015** | — | — | |

### ⚠️ CRITICAL paper-facing finding (round-28 finalized round-31)

**PR #4402's headline test=44.5079 is at −1.09σ_sample of the 5-seed test distribution** (44.51 vs μ=45.49, σ̂_sample=0.90). The population-level test claim is **45.49 ± 0.40 (SEM)**, not the single-seed merged baseline. The single-seed headline is a lucky-draw of seed=0.

**For the paper, report BOTH:**
1. Single-seed merged baseline: val 45.7284 / test 44.5079 (PR #4402, seed=0)
2. n=5 canonical: val 46.83 ± 0.41 SEM / test 45.49 ± 0.40 SEM

The single-seed merge stands per single-seed merge rule, but reviewers will want the population claim. The single-seed result reflects what's possible with this compound; the canonical reflects expected performance.

**Future winner candidates must beat 45.49 test mean with ≥0.9pt margin (≥1σ̂_sample)**, not just the seed=0 best of 44.51.

### Variance comparison: NEW best vs OLD best

**σ̂_sample at NEW best (k=6+β2=0.995) is 1.05 val — 2.3× wider than σ̂=0.46 at OLD best (k=5+β2=0.995).** Mean improvement k=5 → k=6 (3-seed) = 0.15 val on val; pooled σ̂_diff ≈ 0.74; z ≈ 0.20 → NOT statistically significant. Mechanism: at k=6 the inner trajectory takes one extra step before each Lookahead pull, amplifying seed-dependent trajectory direction. Longer m-buffer (β2=0.995) does NOT fully compensate at k=6 the way it does at k=5.

### 3-seed canonical of PREVIOUS best k=5+β2=0.995 (COMPLETE — round-25)

| Seed | val_avg | test_avg | W&B | Status |
|---|---|---|---|---|
| 0 | 46.8383 | 45.3196 | `3k6hob38` | PR #4373 merged |
| 1 | 47.6478 | 46.6685 | `jxahw2bk` | PR #4386 closed |
| 2 | 46.8264 | 45.4651 | `hqel4ej1` | PR #4385 closed |
| **mean ± σ̂** | **47.10 ± 0.46** | **45.82 ± 0.74** | — | PAPER-READY |

Val σ̂=0.46 is tighter than OLD baseline σ̂=0.80 — β2=0.995 reduces per-seed val variance.

## SUPER-ADDITIVE COMPOUND DECOMPOSITION

| Intervention | Δ val | Cumulative val | Mechanism |
|---|---|---|---|
| AdamW baseline (T_max=17 + GeGLU + β2=0.95) | — | ~56.0 | triple-stack foundation |
| AdamW → Lion (all previous) | ~−8.0 | ~48.0 | sign-based, eliminate gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | — | 47.97 | slow-weight averaging (PR #4123) |
| α=0.5 → α=0.7 (k=5, β2=0.99) | −0.38 | 47.59 | stronger basin-averaging pull (PR #4269) |
| β2=0.99 → β2=0.995 (k=5, α=0.7) | −0.75 | 46.84 | within-basin gradient smoothing, m-buffer half-life 69→138 steps (PR #4373) |
| k=5 → k=6 (β2=0.995, α=0.7) | **−1.11** | **45.73** | **k-bowl shift + super-additive compound** (PR #4402) |

**Super-additivity at k=6:** k alone gave −0.45 val at β2=0.99; combined with β2=0.995 gives −1.11 (additive would predict −0.45 → combined −0.75+0.45=0.75 = predicted −0.45, actual −1.11 = super-additive by 0.66). Mechanism: β2=0.995 m-buffer lets inner trajectory settle coherently before k=6's longer outer step — the two mechanisms amplify each other at complementary timescales.

## α-frontier — FULLY MAPPED at k=6+β2=0.995 (round-31; α=0.70 RESOLVED optimal)

| α | k | β2 | val | Status |
|---|---|---|---|---|
| 0.5 | 5 | 0.99 | 47.97 | superseded |
| **0.7** | **5** | **0.99** | **47.59** | merged #4269 |
| **0.65** | **5** | **0.995** | **46.6514** | closed #4415 (round-27; α-LEFT shift at k=5, not new best) |
| **0.7** | **5** | **0.995** | **46.84** | merged #4373 |
| **0.60** | **6** | **0.995** | **46.36** | closed #4475 (round-31) — α-LEFT confirmed regress at k=6 |
| **0.65** | **6** | **0.995** | **46.93** | closed #4472 (round-27) — α-LEFT does NOT carry from k=5 |
| **0.7** | **6** | **0.995** | **45.73** | **merged #4402 — CURRENT BEST** |
| **0.75** | **6** | **0.995** | **45.8003** | closed #4430 (round-27; α-RIGHT bowl flat) |
| 0.8 | 5 | 0.99 | 48.25 | closed #4343 |

**α-bowl at k=6+β2=0.995 is FULLY RESOLVED.** Shallow bowl with slight RIGHT-tilt; α=0.70 optimal. α-LEFT shift discovered at k=5 (nezuko #4415) does NOT generalize to k=6 — at k=6 the inner-step pre-smoothing already coherent enough that less slow-weight pull is over-damped.

### α-schedule mechanism — FALSIFIED forward direction (round-31, alphonse #4496)

α COSINE SCHEDULE 0.5 → 0.7 lands within population at val=46.96 / test=45.97. Reaches α=0.7 exactly at LR-cosine endpoint (step 6375 ≈ epoch 17) — by then inner Lion takes near-zero-LR steps and both schedules converge. Mid-training, the schedule-run has WORSE val at every checkpoint than static α=0.7 — early-α=0.5 *reduces* slow-weight pull rather than improving exploration. Mechanism story FALSIFIED for forward direction. **Reverse direction (0.7 → 0.5) assigned to nezuko #4524 as symmetric mechanism check.**

## β2-frontier — FULLY MAPPED at k=5; re-mapping at k=6

At k=5:
| β2 | val | m-buffer half-life | Status |
|---|---|---|---|
| 0.95 | 54.62 | 14 steps | catastrophic (PR #4264) |
| 0.99 | 47.59 | 69 steps | prior baseline |
| **0.995** | **46.84** | **138 steps** | **winner at k=5** |
| 0.997 | 46.80 (val flat, test +0.49) | 230 steps | closed #4384 — NOT a win |
| 0.999 | 52.04 | 692 steps | catastrophic (PR #4356) |

At k=6 (currently known only seed=0):
| β2 | val | Status |
|---|---|---|
| 0.99 | 47.14 | closed #4371 |
| **0.993** | **TBD** | **fern #4473 in flight (round 27)** — LEFT bracket-tighter |
| **0.994** | **TBD** | **frieren #4431 in flight (round 25)** |
| **0.995** | **45.73** | **merged #4402 — CURRENT BEST** |
| 0.997 | 47.24 | closed #4427 (round-27; β2-bowl NARROWED at k=6, not shifted) |

**β2-bowl narrowed at k=6** (vs k=5): β2=0.997 went from val-flat regression (closed #4384 at k=5) to +1.51 val regression at k=6.

## k-frontier — mapped at β2=0.99; being mapped at β2=0.995

At β2=0.99, α=0.7:
| k | α/k | val | Status |
|---|---|---|---|
| 4 | 0.175 | ~48.30 | closed #4355 |
| 5 | 0.140 | 47.59 | merged #4269 → superseded |
| **6** | **0.117** | **47.14** | closed #4371 (beats k=5 at β2=0.99) |
| 7 | 0.100 | 48.48 | closed #4310 (at α=0.5/β2=0.99) |

At β2=0.995, α=0.7:
| k | α/k | val | Status |
|---|---|---|---|
| 5 | 0.140 | 46.84 | merged #4373 → superseded |
| **6** | **0.117** | **45.73** | **merged #4402 — CURRENT BEST** |
| 7 | 0.10 | 46.42 | closed #4426 (round-26) — k-bowl right-flank FIRM at k=6 |

**k-bowl bounded:** k=6 is the unique optimum at α=0.7 + β2=0.995. The super-additivity is LOCAL to k=6, not a broad "longer sync interval works under longer m-buffer" claim.

## Lion ACTIVE WD bowl — currently UNMAPPED (all prior probes were fp32 no-ops; round-31 dispatch begins mapping)

Per tanjiro's #4456 finding, every Lion WD probe at wd ≤ 1e-4 in this programme was a fp32 no-op — the WD bowl has NEVER been mapped at meaningful regularization strength. Round-31 dispatches 3 parallel ACTIVE probes:

| wd | lr*wd | activity factor | PR | Student | Predicted |
|---|---|---|---|---|---|
| 1e-3 | 1.67e-7 | 5.6× half-ulp | **#4518** | tanjiro | Mild active reg — may help generalization |
| 3e-3 | 5.00e-7 | 16.8× half-ulp | **#4521** | alphonse | Mid; classic transformer default |
| 1e-2 | 1.67e-6 | 56× half-ulp | **#4523** | frieren | High; tests if over-reg regresses |
| 2e-4 | 3.33e-8 | 1.12× half-ulp | #4482 closed | thorfinn (was) | +0.07 within noise (barely active) |
| ≤ 1e-4 | ≤ 1.67e-8 | ≤ 0.56× | #4123–#4402 | all winners | NO-OP (effective wd=0) |

If any of {1e-3, 3e-3, 1e-2} beats #4402, the entire programme has a new compound. If all regress, the smoothing compound genuinely doesn't want explicit WD — that itself is a paper-grade finding.

## Active WIP experiments (round 32)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4562 | frieren | **Layer-wise LR decay (LLRD) decay=0.95 per Transolver block** | NEW (round 32) | Bold per-block training dynamics |
| #4547 | nezuko | **Huber loss δ=1.0** — loss-curvature reformulation (Lion m-buffer hard-example concentration) | NEW (round 32) | Bold loss-mechanism swing |
| #4546 | thorfinn | **SWA uniform mean of slow_weights at ep {14,15,16,17}** — variance reduction in converged regime | NEW (round 32) | Bold variance-reduction (distinct from fern #4500 EMA) |
| #4537 | fern | **Focal-loss per-node hardness weighting γ=1.0** on surface MAE | Running (round 31) | Bold loss reformulation |
| #4536 | askeladd | **Lookahead OUTER momentum β_outer=0.5** on slow-step direction | Running (round 31) | Bold optimizer mechanism |
| **#4521** | **alphonse** | **--weight_decay 3e-3 SEED REPLICATION (seeds 1,2,3,4)** | **SENT BACK (round 32)** | **⚠️ FIRST POSSIBLE WIN since #4402 — verify reproducibility** |
| #4518 | tanjiro | **--weight_decay 1e-3** — Lion ACTIVE WD probe LOW (their own finding) | Running (round 31) | First proper WD-bowl mapping LOW |
| #4506 | edward | (k=7, β2=0.9957, α=0.7) JOINT shift — k×(1−β2)≈0.03 invariant | Running (round 30) | Bold joint-shift mechanism |

**All 8 students active. Zero idle. Single-arm policy in force, except alphonse #4521 doing seed-replication (4 sequential runs ~2h total). 8 axes in flight: LLRD + Huber + SWA + focal-loss + outer-momentum + wd=3e-3 replication + wd=1e-3 + joint k×β2.**

### ⚠️ Round-29 strategic theme: PLATEAU PROTOCOL activated

**Trigger:** 11 closures since #4402 merge, ZERO improvements. All micro-probes around (k=6, β2=0.995, α=0.7) regressing within or near σ̂_sample=1.05 val. The compound is a **precisely-tuned sharp pin**, not a flat plateau.

**Pivot:** Bolder mechanism-grounded swings, not micro-probes.

- **α COSINE SCHEDULE (alphonse #4496):** Schedule α from 0.5 (early) → 0.7 (late). Mechanism: gradient coherence varies with training stage; static α=0.7 over-pulls early when inner trajectory is noisy.
- **EMA on Lookahead slow_weights (fern #4500):** SWA-style EMA decay=0.999 on slow_weights themselves, starting epoch 8. Mechanism: slow_weights still have sync-to-sync variance; averaging reduces it.
- **LR=6e-4 MID (askeladd #4497):** completes LR bowl mapping (LEFT 4e-4 closed +0.95, RIGHT 7e-4 in flight). Safe bracket-completion probe.
- **seed=4 canonical (frieren #4498):** strengthens paper SEM from n=4 (0.48 test) to n=5 (~0.43 test). Low-risk paper-strengthening.

If alphonse or fern wins, multi-seed it immediately. If both regress, escalate to: (1) per-region loss weighting, (2) Tiger/ScheduleFree optimizers, (3) cosine warm restarts, (4) data augmentation.

## Round-32 active-WD outcomes (in flight + 1 sent-back + 1 closed)

⚠️ **FIRST POSSIBLE WIN since #4402**: alphonse #4521 at wd=3e-3 returned val=45.4891 (Δ −0.24 vs #4402) on seed=0. Single-seed improvement is within σ̂ (26% of σ̂_sample=0.92) and test slightly regressed (+0.13). **Sent back for 4-seed replication (seeds 1,2,3,4)** — if combined 5-seed val mean < 46.50 AND test mean < 45.20, this is the new baseline.

- **#4523 frieren (--weight_decay 1e-2)** CLOSED NEUTRAL: val=46.71 / test=45.57, both within 1 SEM of canonical mean. wd=1e-2 starting to over-regularize. Frieren reassigned to **LLRD decay=0.95 per Transolver block (#4562)**.
- **#4521 alphonse (--weight_decay 3e-3)** SENT BACK: val=45.49 (best single-seed in 22 closures) but within noise; needs multi-seed verification. 4-seed replication assigned.
- **#4518 tanjiro (--weight_decay 1e-3)** STILL IN FLIGHT: low end of WD bowl.

Single-seed WD bowl shape so far: 0 → 1e-3 (TBD) → 3e-3 (best, val=45.49) → 1e-2 (over-reg, val=46.71). Mid-range (3e-3) is the sweet spot — pending tanjiro's 1e-3 result and alphonse's seed-replication.

## Round-32 closures (4 closures)

- **#4497 askeladd (cfg.lr=6e-4 + k=6 + β2=0.995 + α=0.7)** CLOSED: val=46.49 (+0.76, ~0.72σ̂). **LR-bowl at NEW compound FULLY MAPPED** (4e-4: +0.95, 5e-4: floor, 6e-4: +0.76, 7e-4: +2.40). Asymmetric — flat LEFT/MID, sharp RIGHT. Askeladd reassigned to **Lookahead OUTER momentum β_outer=0.5 (#4536)**.
- **#4500 fern (EMA on Lookahead slow_weights, decay=0.999 from ep 8)** CLOSED CATASTROPHIC: val=57.29 (+11.56). Mechanism failure (your post-mortem): EMA at decay=0.999 + only 562 updates → 57% weight on epoch-8 init snapshot → temporal lag, not variance reduction. SWA/EMA does NOT compose with Lookahead at fixed-T_max; slow_weights ARE the variance-reduced parameters. Fern reassigned to **focal-loss per-node weighting γ=1.0 (#4537)**.
- **#4524 nezuko (Lookahead α REVERSE COSINE 0.7 → 0.5)** CLOSED: val=46.78 (+1.05, ~1.14σ̂). **Both α-schedule directions now falsified** (forward +1.23, reverse +1.05); reverse 0.18 better than forward (slight time-budget asymmetry) but both ~1σ̂ worse than static. **α=0.70 is a SHARP optimum at k=6+β2=0.995, robust against scheduling perturbations.** Paper-grade negative result. Nezuko reassigned to **Huber loss δ=1.0 (#4547)**.
- **#4525 thorfinn (Cosine warm restarts 2 cycles T_0=8)** CLOSED CATASTROPHIC: val=49.41 (+3.68, ~4σ̂). Mechanism failure: cycle 1 never converged (lowest val=67.7 at ep 7); each LR restart caused +20-27 val regression. **At 17-epoch budget, single-cosine T_max=17 is OPTIMAL — multi-cycle scheduling INFEASIBLE.** Thorfinn reassigned to **SWA uniform mean of slow_weights at ep {14,15,16,17} (#4546)**.

## Round-31 closures (5 closures — including MAJOR FINDING)

- **#4456 tanjiro (--weight_decay 5e-5)** CLOSED with full credit: val/test BIT-IDENTICAL to #4402 to 10 decimals. **MAJOR FINDING: Lion WD is fp32 NO-OP at wd ≤ 1e-4 — entire programme has been running with effective wd=0.** Tanjiro reassigned to wd=1e-3 (1st active WD probe, #4518).
- **#4496 alphonse (α COSINE SCHEDULE 0.5 → 0.7)** CLOSED: val=46.96 (+1.23), test=45.97 (+1.46). Lands within population. α-schedule FORWARD direction FALSIFIED — converges to same fixed point at LR endpoint, worse mid-training. Alphonse reassigned to wd=3e-3 (Lion active WD MID, #4521).
- **#4498 frieren (seed=4 canonical)** CLOSED: val=47.07, test=46.07. n=5 canonical PAPER-READY: **val 46.83 ± 0.41 SEM / test 45.49 ± 0.40 SEM**. Frieren reassigned to wd=1e-2 (Lion active WD RIGHT edge, #4523).
- **#4475 nezuko (α=0.60 + k=6 + β2=0.995)** CLOSED: val=46.36 (+0.63, within 0.53σ̂), test=45.38 (+0.87, within 0.90σ̂). α-bowl at k=6+β2=0.995 FULLY MAPPED — α=0.70 optimum. Nezuko reassigned to REVERSE α COSINE SCHEDULE 0.7 → 0.5 (#4524).
- **#4482 thorfinn (--lion_wd 2e-4)** CLOSED: val=45.80 (+0.07, within 0.07σ̂), test=45.01 (+0.50). FIRST barely-active WD value (lr*wd 1.12× half-ulp); cumulative ~2.1e-4 fractional drift over 6375 steps. Thorfinn reassigned to COSINE WARM RESTARTS (2 cycles T_max=8 + T_max=9, #4525).

## Round-30 closures (1 closure)

- **#4432 edward (cfg.lr=7e-4 + k=6 + β2=0.995)** CLOSED on W&B data (stale_wip, no SENPAI-RESULT comment posted): val=48.125 (+2.40), test=46.603 (+2.10). LR-bowl RIGHT side SHARPER than LEFT — smoothed compound prefers DAMPED steps. Heartbeat re-launch pattern (3 finished + 1 failed + 1 running) — operational discipline flagged. Edward reassigned to JOINT (k=7, β2=0.9957) probe (#4506).

## Round-29 closures (4 closures — PLATEAU PROTOCOL trigger)

- **#4472 alphonse (α=0.65 + k=6 + β2=0.995)** CLOSED: val=46.93 (+1.20). α-LEFT does NOT carry from k=5 to k=6.
- **#4473 fern (β2=0.993 + k=6 + α=0.7)** CLOSED: val=45.75 (+0.02, FLAT — likely seed noise). β2-bowl LEFT effectively flat.
- **#4455 askeladd (cfg.lr=4e-4)** CLOSED: val=46.68 (+0.95). LR-LEFT mild regression.
- **#4431 frieren (β2=0.994)** CLOSED: val=47.12 (+1.39). β2-bowl sharper on left at k=6 than at k=5.

## Round-28 closures (1 closure)

- **#4457 thorfinn (seed=3 canonical k=6+β2=0.995+α=0.7)** CLOSED: val=46.22 / test=44.53. n=4 canonical PAPER-READY at val 46.77±0.52 SEM / test 45.34±0.48 SEM. **CRITICAL: PR #4402's headline test=44.51 is at −0.83σ of population — single-seed merge is lucky-draw, not expected outcome.** Thorfinn reassigned to WD RIGHT bracket #4482.

## Round-27 closures (3 closures)

- **#4430 alphonse (α=0.75 + k=6 + β2=0.995)** CLOSED: val=45.80 (+0.07 within σ̂=1.20 noise). α-RIGHT side at NEW compound flat — α=0.70 confirmed RIGHT-edge optimum.
- **#4427 fern (β2=0.997 + k=6 + α=0.7)** CLOSED: val=47.24 (+1.51). β2-bowl NARROWED at k=6 (not shifted right); super-additive win is sweet-spot not trend.
- **#4415 nezuko (α=0.65 + k=5 + β2=0.995)** CLOSED: val=46.65 (vs OLD k=5 winner: Δ=−0.19; vs CURRENT best #4402: +0.92). α-LEFT shift mechanism confirmed at k=5, but not a new programme best. Drives the round-27 α-LEFT probe at k=6.

## Round-26 closures (3 closures)

- **#4429 thorfinn (seed=2 k=6+β2=0.995)** CLOSED: val=46.99 / test=45.95. Near distribution center; completes 3-seed canonical.
- **#4428 tanjiro (seed=1 k=6+β2=0.995)** CLOSED: val=48.14 (+2.41 vs seed=0). VARIANCE FINDING: σ̂_val=1.20 at k=6 vs 0.46 at k=5 (2.6× wider). Mean improvement (k=5→k=6) NOT statistically significant.
- **#4426 askeladd (k=7+β2=0.995+α=0.7)** CLOSED: val=46.42 (+0.69 vs k=6). k-bowl right-flank FIRM at k=6 under α=0.7+β2=0.995. Super-additivity is LOCAL to k=6.

## Round-25 closures (7 closures; 1 merge)

- **#4402 askeladd (k=6 + β2=0.995 + α=0.7) MERGED — NEW PROGRAMME BEST** val=45.7284 / test=44.5079. SUPER-ADDITIVE by 0.66 beyond additive prediction. All 4 test splits improve. Peak VRAM 101 GB (fine). Askeladd reassigned to k=7 probe (#4426).
- **#4386 tanjiro (seed=1 k=5+β2=0.995)** CLOSED: val=47.6478 / test=46.6685, +0.81 vs seed=0 (within 1.27σ). Completes 3-seed table of OLD best; mean 47.10 ± 0.46.
- **#4385 thorfinn (seed=2 k=5+β2=0.995)** CLOSED: val=46.8264 (TIES seed=0 within 0.012). Tightest seed-lock in programme.
- **#4384 fern (β2=0.997 at k=5)** CLOSED: val=46.80 (flat), test=45.81 (+0.49 regression). β2 right-edge at k=5 CLOSED; 0.995 is optimum at k=5.
- **#4376 alphonse (β1=0.88)** CLOSED: val=48.21 (+0.62). β1 bowl floor confirmed at 0.90; landscape SHARPER under α=0.7. Branch DIRTY (needs-rebase) but regression → closed without rebase.
- **#4375 frieren (slice_num=32 + α=0.7)** CLOSED: val=48.39 (+0.80). α-slice interaction confirmed; α-optimal is slice_num-dependent.
- **#4374 edward (h=192)** CLOSED: val=50.43, best_ep=14 (budget-cut, +35-40% per-epoch). ALL 5 ARCHITECTURAL UP-ARMS CLOSED.

## Round-24 closures (1 closure)

- **#4345 nezuko (α=0.7 seed=2, β2=0.99)** CLOSED with corrected canonical `qohoymnk`. OLD baseline 3-seed (β2=0.99): val 47.77±0.80 / test 46.51±0.46.

## Round-23 closures (1 closure)

- **#4371 askeladd (k=6 + α=0.7, β2=0.99)** CLOSED: val=47.14 (k-bowl shift to k=6 under α=0.7).

## Round-22 closures (3 closures + 1 merge)

- #4373 fern MERGED: β2=0.995+α=0.7 → val=46.84 (prior programme best)
- #4356 tanjiro β2=0.999 CLOSED: +5.20 catastrophic
- #4355 thorfinn k=4 CLOSED: +0.71

## Architectural Portfolio — ALL 5 UP-ARMS CLOSED (budget-bound)

| Arm | PR | Δ val vs old baseline | best_ep | Overhead |
|---|---|---|---|---|
| ~~Attention heads~~ | #4304 | +9.11 | 12 | +36% |
| ~~Slices (up)~~ | #4323 | +5.51 | 14 | +18% |
| ~~Depth~~ | #4294 | +3.01 | 14 | +21% |
| ~~FFN width~~ | #4286 | +1.04 | 16 | +5% |
| ~~Hidden dim (h=192)~~ | #4374 | +2.84 | 14 | **+35-40% (larger than estimated)** |

**ALL UP ARMS DEAD.** Capacity-up regresses proportionally to per-epoch cost under 30-min wall-clock. Slice-DOWN (#4325, val=47.78) and all HP knobs are more productive.

## Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null |
| LR=1e-3 | Divergence |
| Head+embed LR boost | All null/worse |
| T_max < 17 | Suboptimal |
| RMSNorm | −5.18 regression |
| slice_num=128 | −10.92 regression |
| slice_num=96 | +5.51 regression |
| heads=8 (30-min budget) | +8.73 regression |
| h=192 (30-min budget) | +2.84 regression (budget-cut) |
| clip_norm=1.0 | −3.68 regression |
| Warmup before cosine | Worse |
| SWA variants | Regression or NO-OP |
| β1=0.95 + β2=0.95 | +2.86 regression |
| Lookahead k=8 | +2.87 regression |
| Lookahead α=0.3 | +4.36 regression |
| Lion β2=0.95 | +6.65 regression |
| Lion β2=0.999 | +5.20 regression |
| Lion β2=0.997 (k=5) | val flat, test +0.49 regression |
| Lion cfg.lr=3e-4 | +1.44 regression |
| Lion β1=0.95 | +3.03 regression |
| Lookahead α=0.8 (k=5) | +0.66 regression |
| Lookahead k=4 (α=0.7) | +0.71 regression |
| β1=0.88 (α=0.7) | +0.62 regression |
| slice_num=32 + α=0.7 | +0.80 regression (α-slice interaction) |
| depth=6 (budget-cut) | +3.01 regression |
| mlp_ratio=3 | +1.04 regression |

## Next research directions

### Priority 1 (IN FLIGHT round 25)

- **k=7 + β2=0.995 + α=0.7** (askeladd #4426) — k-bowl right-flank; predicted 45.3-46.3 range
- **β2=0.997 + k=6 + α=0.7** (fern #4427) — β2-bowl right-edge at new k; may shift vs k=5 result
- **β2=0.994 + k=6 + α=0.7** (frieren #4431) — β2-bowl left-edge at new k; brackets bowl from below
- **α=0.75 + k=6 + β2=0.995** (alphonse #4430) — α-bowl right at new compound
- **cfg.lr=7e-4 + k=6 + β2=0.995** (edward #4432) — LR re-tune at double-smooth; tail risk new winner
- **3-seed canonical** (tanjiro #4428, thorfinn #4429) — paper-facing variance for new best
- **α=0.65 + k=5 + β2=0.995** (nezuko #4415) — α left-side at old k; still informative for paper

### Priority 2 (next idle-round; gated on round-27 outcomes)

- **--lion_wd 2e-4 + k=6 + β2=0.995** — WD RIGHT bracket (if WD-LEFT #4456 regresses, probe RIGHT)
- **cfg.lr=6e-4 + k=6 + β2=0.995** — LR MID bracket (if both LR-LEFT 4e-4 and LR-RIGHT 7e-4 regress, search inside)
- **α=0.65 + k=5 + β2=0.995 SEED REPLICATES** — confirm nezuko's α-LEFT win isn't seed luck (σ̂=0.46 noise)
- **(α<0.7, k=6, β2=0.995) × seed=2/3** — if alphonse or nezuko wins at α=0.65 or α=0.60, immediately seed-canonical it
- **k=7 + β2=0.997 + α=0.7** — (k, β2) joint shift up the right diagonal (if k-bowl shifts with β2)
- **Warmup epochs=1 at new compound** — if seed σ̂=1.20 traces to early-trajectory instability, warmup may damp it
- **lr-schedule: cosine + 0.1 floor** — keep slow weights moving in last few epochs; gentle (no T_max change)

### Priority 3 (Plateau-protocol escalation if round-25 saturates)

- **Loss reformulation:** physics-informed continuity-equation constraint, per-region loss weighting
- **ScheduleFree-Lion** — variance-reduction optimizer class that may compose with Lookahead differently
- **cosine restart at epoch 17** — extend training with slow-weight pull intact past T_max
- **Tiger optimizer** (Chen 2023 Lion successor) — different sign-update dynamics

## Key mechanistic findings for paper

1. **k-bowl shifts under (α, β2) regime:** at β2=0.99, k=5 optimal; at β2=0.995, k=6 optimal. The k-optimum tracks the β2/m-buffer timescale — when the m-buffer smooths more aggressively, the Lookahead sync interval can be longer.
2. **Super-additive compound:** k=6 + β2=0.995 is super-additive by 0.66 val beyond prediction. The two smoothers amplify each other at complementary timescales (m-buffer EMA vs Lookahead sync interval).
3. **β2 bowl narrow at each k:** at k=5, bowl is at 0.995 (right-edge closed); at k=6, bowl position TBD but expected near 0.995.
4. **β1=0.90 is robust:** β1 landscape is sharper under α=0.7 — faster pull amplifies β1 sensitivity. β1=0.90 is a stable local optimum regardless of α.
5. **ALL architectural up-arms dead at 30-min budget:** capacity-up regresses proportionally to per-epoch overhead. The wall-clock constraint creates a hard budget incompatibility for larger models.
6. **β2=0.995 tightens seed-noise on val:** σ̂=0.46 at β2=0.995 vs 0.80 at β2=0.99. Smoother m-buffer reduces initialization-dependent variance.
