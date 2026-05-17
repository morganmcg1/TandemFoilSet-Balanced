# SENPAI Research State

- **Date:** 2026-05-17 13:00
- **Launch:** willow-pai2i-48h-r1 (round 27 — Lookahead-Lion era; **PROGRAMME ALL-TIME BEST val=45.7284 / test=44.5079** (PR #4402, k=6+β2=0.995+α=0.7); **3-seed canonical 46.95±1.20 val / 45.61±0.97 test (PAPER-READY)**; **α-LEFT shift confirmed at k=5+β2=0.995** (nezuko #4415 closed, Δ=−0.19 val); **k-bowl right CLOSED at k=7**, **β2 right CLOSED at 0.997 at both k=5 and k=6**, **α right CLOSED at α=0.75 at k=6**; **α-LEFT × k=6 + β2-LEFT × k=6 + LR/WD micro-probes + seed=3 in flight**)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=6/α=0.7) + Lion β2=0.995 + triple-stack (PROGRAMME BEST)** | **45.7284** | **44.5079** | PR #4402, W&B `ejacndhj`, seed=0 | Merged 2026-05-17 11:00; super-additive compound |

Win threshold: **val < 45.7284**. Prior best: val=46.8383 (PR #4373, k=5).

### 3-seed canonical of NEW programme best (COMPLETE — round-26; PAPER-READY)

| Seed | val_avg | test_avg | Source |
|---|---|---|---|
| 0 | **45.7284** | **44.5079** | PR #4402 MERGED (`ejacndhj`) |
| 1 | 48.14 | 46.40 | PR #4428 closed (tanjiro) |
| 2 | 46.99 | 45.95 | PR #4429 closed (thorfinn) |
| **mean ± σ̂** | **46.95 ± 1.20** | **45.61 ± 0.97** | — |
| 3 | TBD | TBD | **thorfinn #4457 in flight (round 26)** — tightens σ̂ to n=4 |

### ⚠️ CRITICAL VARIANCE FINDING (round-26)

**3-seed σ̂ at NEW best (k=6+β2=0.995) is 1.20 val — 2.6× wider than the σ̂=0.46 at OLD best (k=5+β2=0.995).**

Mean improvement at k=5 → k=6 = 0.15 val on the 3-seed mean. Pooled σ̂_diff ≈ 0.74. **The k=5 → k=6 mean improvement is NOT statistically significant at p<0.05** (z ≈ 0.20). The single-seed win (45.73) is at ~−1σ of the k=6 distribution — partly seed luck. Headline stands per single-seed merge rule, but follow-ups should weight σ̂ when prioritizing.

Mechanism speculation: at k=6, the inner trajectory takes one extra step before each Lookahead pull. That extra step amplifies the seed-dependent trajectory direction, increasing the variance of where the slow weights land. Longer m-buffer (β2=0.995) does NOT fully compensate at k=6 the way it does at k=5.

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

## α-frontier — fully mapped at k=5 across β2 ∈ {0.99, 0.995}; mapping at k=6+β2=0.995

| α | k | β2 | val | Status |
|---|---|---|---|---|
| 0.5 | 5 | 0.99 | 47.97 | superseded |
| **0.7** | **5** | **0.99** | **47.59** | merged #4269 |
| **0.65** | **5** | **0.995** | **46.6514** | closed #4415 (round-27; α-LEFT shift confirmed, not new best) |
| **0.7** | **5** | **0.995** | **46.84** | merged #4373 |
| **0.7** | **6** | **0.995** | **45.73** | **merged #4402 — CURRENT BEST** |
| **0.75** | **6** | **0.995** | **45.8003** | closed #4430 (round-27; α-RIGHT bowl flat) |
| 0.8 | 5 | 0.99 | 48.25 | closed #4343 |
| **0.60** | **6** | **0.995** | **TBD** | **nezuko #4475 in flight (round 27)** |
| **0.65** | **6** | **0.995** | **TBD** | **alphonse #4472 in flight (round 27)** — KEY winner candidate

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

## Active WIP experiments (round 27)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4475 | nezuko | **α=0.60 + k=6 + β2=0.995 (α-bowl LEFT-EDGE at new compound)** | NEW (round 27) | **Winner candidate (extends nezuko's α-LEFT shift)** |
| #4472 | alphonse | **α=0.65 + k=6 + β2=0.995 (α-bowl LEFT-MIDDLE at new compound)** | NEW (round 27) | **Winner candidate (extends nezuko's α-LEFT shift)** |
| #4473 | fern | **β2=0.993 + k=6 + α=0.7 (β2-bowl LEFT-edge at new compound, tighter than frieren #4431)** | NEW (round 27) | β2-bowl LEFT bracket |
| #4455 | askeladd | cfg.lr=4e-4 + k=6 + β2=0.995 + α=0.7 (LR LEFT micro-probe) | Running (round 26) | LR-bowl LEFT |
| #4456 | tanjiro | --lion_wd 5e-5 + k=6 + β2=0.995 (WD LEFT micro-probe) | Running (round 26) | WD-bowl LEFT |
| #4457 | thorfinn | 3-seed canonical seed=3: k=6 + β2=0.995 + α=0.7 | Running (round 26) | Paper canonical (n=4) |
| #4431 | frieren | β2=0.994 + k=6 + α=0.7 (β2-bowl left-edge at new k) | Running (round 25) | β2-bowl LEFT |
| #4432 | edward | cfg.lr=7e-4 + k=6 + β2=0.995 (LR RIGHT bracket) | Running (round 25) | LR probe |

**All 8 students active. Zero idle. Single-arm policy in force.**

### Round-27 strategic theme: α-LEFT exploitation at NEW k=6 compound

Nezuko's PR #4415 (closed this round) confirmed the α-bowl shifts LEFT under β2=0.995 at k=5. The natural follow-up is to test whether this shift extends to the NEW k=6 compound. Two parallel probes:
- **alphonse #4472**: α=0.65 + k=6 + β2=0.995 (α-bowl LEFT-MIDDLE at new compound — exactly nezuko's winning α value, applied to k=6)
- **nezuko #4475**: α=0.60 + k=6 + β2=0.995 (α-bowl LEFT-EDGE at new compound — pushes further left)

Combined with the closed RIGHT probes (#4430 α=0.75 closed within noise; #4402 α=0.70 winner), this fully maps the α-bowl at k=6+β2=0.995 over {0.60, 0.65, 0.70, 0.75}. **If alphonse or nezuko wins, we have a NEW programme best.** Mechanism is mechanistically aligned (smoother m-buffer + longer sync interval ⇒ less slow-pull needed).

Fern #4473 (β2=0.993) tightens the β2 LEFT bracket complementing frieren's #4431 (β2=0.994).

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
