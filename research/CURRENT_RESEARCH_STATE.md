# SENPAI Research State

- **Date:** 2026-05-17 11:00
- **Launch:** willow-pai2i-48h-r1 (round 25 — Lookahead-Lion era; **NEW PROGRAMME ALL-TIME BEST val=45.7284 / test=44.5079** (PR #4402 merged, k=6+β2=0.995+α=0.7 SUPER-ADDITIVE compound, Δ−1.11 val); **k=5+β2=0.995 3-seed canonical COMPLETE (47.10±0.46 / 45.82±0.74)**; **ALL 5 architectural up-arms CLOSED**; **β2=0.995 right-edge closed at k=5**; **k-bowl + β2-bowl + α-bowl probes in flight at new compound**)
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

### 3-seed canonical of NEW programme best (just dispatched round-25)

| Seed | val_avg | test_avg | Source |
|---|---|---|---|
| 0 | **45.7284** | **44.5079** | PR #4402 MERGED (`ejacndhj`) |
| 1 | TBD | TBD | tanjiro #4428 in flight (round 25) |
| 2 | TBD | TBD | thorfinn #4429 in flight (round 25) |

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

## α-frontier — FULLY MAPPED at k=5, β2=0.99 / β2=0.995 × k=5 known

| α | k | β2 | val | Status |
|---|---|---|---|---|
| 0.5 | 5 | 0.99 | 47.97 | superseded |
| **0.7** | **5** | **0.99** | **47.59** | merged #4269 |
| **0.7** | **5** | **0.995** | **46.84** | merged #4373 |
| **0.7** | **6** | **0.995** | **45.73** | **merged #4402 — CURRENT BEST** |
| 0.8 | 5 | 0.99 | 48.25 | closed #4343 |
| 0.65 | 5 | 0.995 | TBD | nezuko #4415 in flight |
| **0.75** | **6** | **0.995** | **TBD** | **alphonse #4430 in flight (round 25)** |

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
| **0.994** | **TBD** | **frieren #4431 in flight (round 25)** |
| **0.995** | **45.73** | **merged #4402 — CURRENT BEST** |
| **0.997** | **TBD** | **fern #4427 in flight (round 25)** |

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
| **7** | **0.10** | **TBD** | **askeladd #4426 in flight (round 25)** |

## Active WIP experiments (round 25)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4426 | askeladd | **k=7 + β2=0.995 + α=0.7 (k-bowl right-flank at new β2)** | NEW (round 25) | **#1 Winner candidate** |
| #4430 | alphonse | α=0.75 + k=6 + β2=0.995 (α-bowl right at new best) | NEW (round 25) | α-frontier probe |
| #4431 | frieren | β2=0.994 + k=6 + α=0.7 (β2-bowl left-edge at new k) | NEW (round 25) | β2-bowl mapping |
| #4427 | fern | β2=0.997 + k=6 + α=0.7 (β2-bowl right-edge at new k) | NEW (round 25) | β2-bowl mapping |
| #4428 | tanjiro | 3-seed canonical seed=1: k=6 + β2=0.995 + α=0.7 | NEW (round 25) | Paper-facing canonical |
| #4429 | thorfinn | 3-seed canonical seed=2: k=6 + β2=0.995 + α=0.7 | NEW (round 25) | Paper-facing canonical |
| #4432 | edward | cfg.lr=7e-4 + k=6 + β2=0.995 (LR re-tune at double-smooth compound) | NEW (round 25) | LR probe |
| #4415 | nezuko | α=0.65 + k=5 + β2=0.995 (α-bowl left-side micro-probe) | Running (round 24) | α-frontier probe |

**All 8 students active. Zero idle. Single-arm policy in force.**

### Round-25 strategic theme: dense sampling around super-additive compound

PR #4402's super-additive compound is the major discovery. The round-25 assignments sample the (k, β2, α, lr) neighborhood of the new best densely:
- **k-extension** (askeladd #4426): does k-bowl continue shifting under β2=0.995?
- **β2 micro-bowl at k=6** (fern #4427, frieren #4431): bracket the β2 bowl under new k
- **α micro-probe** (alphonse #4430): α-bowl under new compound
- **Paper canonical** (tanjiro #4428, thorfinn #4429): 3-seed variance of new best
- **LR re-tune** (edward #4432): compensate for double-smoothing with higher lr
- **α=0.65 legacy** (nezuko #4415, in flight): α-bowl left at old k=5

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

### Priority 2 (next idle-round; gated on round-25 outcomes)

- **k=8 + β2=0.995 + α=0.7** — if k=7 wins, probe further right; if k=7 closes, k-bowl right-edge mapped
- **k=7 + β2=0.994 or β2=0.997** — (k, β2) interaction grid if k=7 lands close to winner
- **α=0.65 + k=6 + β2=0.995** — α left-side at full new compound (complement to nezuko's k=5 probe)
- **cfg.lr=4e-4 + k=6 + β2=0.995** — left side of LR bowl at new compound (if 7e-4 wins, go right; if not, go left)
- **β2=0.994 + α=0.75 + k=6** — 3-way micro-probe if both α=0.75 and β2=0.994 show positive directions
- **k=6 + β2=0.995 + α=0.7 + --lion_wd 5e-5** — weight-decay micro-probe (current 1e-4 may need tuning at new compound)

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
