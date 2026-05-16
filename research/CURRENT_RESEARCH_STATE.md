# SENPAI Research State

- **Last updated:** 2026-05-16 02:40 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 02:30 UTC — no open issues).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **90.6131** | PR #3474 alphonse EMA decay=0.99+clip5+Huber (merged 00:25 UTC) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | 88.8252 | run `fzrq04xr` |

Per-split validation:

| Split | mae_surf_p | Δ vs prev baseline (#3366, 94.42) |
|---|---|---|
| val_single_in_dist | 106.135 | −5.1% |
| val_geom_camber_rc | 99.466 | **−9.7%** |
| val_geom_camber_cruise | 70.358 | +1.9% |
| val_re_rand | 86.494 | −0.2% ← weakest split, targeted by Round-4 |

## 🔥 Imminent merge candidate

**askeladd #3475 asinh-pressure verify** (rebased onto decay=0.99 stack):
- Verify run `2028x8co` finished at **val_avg/mae_surf_p = 85.82** → **−5.3%** vs baseline 90.6131
- Replicate `j5214ii4` still running (started 02:22 UTC, ETA ~02:55 UTC)
- test_avg NaN due to camber_cruise dataset bug (known issue, same as baseline)
- Awaiting terminal SENPAI-RESULT comment to invoke senpai:merge-winner

## Active PRs (zero idle students)

### Tier-1 — EMA decay bracketing (WIP)

| PR | Student | Hypothesis | Best so far | Status |
|----|---------|-----------|-----------------|-------|
| #3543 | alphonse | ema-decay-push (0.98, 0.97, 0.95) | 0.98=90.84 (TIED); 0.97 running (step 805 of ~5264) | 0.95 not yet started |

### Tier-2 — Winner awaiting verify post

| PR | Student | Hypothesis | W&B best | Vs baseline | Status |
|----|---------|-----------|----------|------------|--------|
| **#3475** | **askeladd** | **asinh-pressure verify on decay=0.99** | **85.82** (`2028x8co`) | **−5.3% WIN** | **Replicate `j5214ii4` finishing ~02:55 UTC; merge when SENPAI-RESULT posted** |

### Tier-2 — Round-4 in-flight (WIP)

| PR | Student | Hypothesis | Best so far | Status |
|----|---------|-----------|----------|--------|
| #3571 | fern | depth-sweep (n_layers=6, 7) | depth-6=93.83 (REGRESS +3.6%) | depth-7 not started yet |
| #3575 | edward | p-surf-weight (3.0, 5.0) | 3.0=94.65 (REGRESS +4.5%) | 5.0 running (step 2058) |
| #3576 | nezuko | wd-sweep (1e-3, 5e-3) | 1e-3=90.75 (TIED) | 5e-3 running |
| #3577 | tanjiro | slice-num-128 | mid-train (step 2017) | running |
| #3578 | frieren | re-sinusoidal-embed (d=8) | mid-train (step 949) | running |
| #3610 | thorfinn | mlp-ratio-sweep (4, conditional 8) | just started (step 361) | running |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | All 4 splits improve, 3 runs |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | All 4 splits ≥−20%; 2 runs |
| **#3474 (alphonse)** | **EMA decay=0.99 (faster shadow tracking)** | **90.6131** | **−4.0%** | **Monotone sweep; 3 arms all beat prior baseline** |

## Round-4 status snapshot (as of 02:40 UTC)

**Big news**: askeladd's asinh-pressure verify on the new baseline lands at **val=85.82** (−5.3%) — by far the largest single improvement in Round 4. Asinh + fast-EMA compound cleanly.

**Round-4 hypothesis status**:
1. **asinh-pressure verify** (askeladd #3475): ✅ **WIN (85.82)** — replicate finishing; merge imminent.
2. **p-surf-weight** (edward #3575): ❌ Arm 1 (3.0) regressed to 94.65. Arm 2 (5.0) running but unlikely.
3. **wd-sweep** (nezuko #3576): ≈ Arm 1 (1e-3) TIED at 90.75. Arm 2 (5e-3) running.
4. **slice-num-128** (tanjiro #3577): ⏳ Running.
5. **re-sinusoidal-embed** (frieren #3578): ⏳ Running.
6. **depth-sweep** (fern #3571): ❌ depth-6 regressed (93.83). depth-7 unlikely to recover.
7. **mlp-ratio** (thorfinn #3610): ⏳ Just started.
8. **ema-decay-push** (alphonse #3543): ≈ 0.98 TIED with 0.99; 0.97 running; 0.95 pending.

## Key findings from Rounds 1–4 (cumulative)

### Round 1: EMA wins, structural bias loses
EMA trajectory averaging (decay=0.999): −11.1% val_avg, all 4 splits improve.

### Round 2: grad_clip + Huber compounds with EMA
EMA+clip5+Huber δ=1.0: −22.4% val_avg. Three mechanisms orthogonal.

### Round 3: Faster EMA decay wins; hyperparameter sweeps falsified
- **EMA decay 0.99** > 0.995 > 0.997 > 0.999 within 14-epoch budget
- **LR sweep** all regress (#3454 CLOSED) — lr=5e-4 optimal
- **CosineAnnealingLR T_max truncation** regresses (#3456 CLOSED) — default T_max=epochs best
- **Huber-delta sweep** confirms δ=1.0 optimal (#3458 CLOSED)
- **SWA on full stack** regresses (#3476 CLOSED) — EMA already covers averaging
- **Geometry mirror** regresses (#3473 CLOSED)
- **asinh-pressure** wins on new baseline (#3475 ASSEMBLING — val=85.82 verify done)
- **physics-continuity** all 3 weights regress (#3477 CLOSED)

### Round 4 (in flight)
- **asinh-pressure verify** lands at val=85.82 → −5.3% — biggest single Round-4 win
- Remaining: thorfinn mlp-ratio (just started), tanjiro slice-num, frieren re-embed, nezuko wd-5e-3, edward p_surf=5
- Falsified so far: fern depth-6, edward p_surf=3

## Strategic outlook

**If askeladd merges**: new baseline val ~85.82 (or better with replicate). Round-5 will need to beat 85.82.

**Round-5 candidates after Round-4 closes**:
1. **asinh + channel-weighted loss** — askeladd's own follow-up: with pressure compressed, rebalance Ux/Uy weighting. Small additional tune on top.
2. **asinh + larger model** — if mlp_ratio or slice_num win in Round 4, stack them on top of asinh.
3. **asinh + scale={1.5, 2.0}** — askeladd's own sweep follow-up to confirm scale=1.0 is optimum.
4. **Variance-aware checkpoint averaging** — top-K within-run averaging on top of EMA shadow.

**OOD Re bottleneck**: val_re_rand drops from 86.494 → 82.20 (askeladd, −5.0%). Best per-split improvement Round-4 so far.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT → cruise=NaN fleet-wide. Persistent across all runs including askeladd.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs).
- **Zero idle students**: 8 WIP PRs.
- REST API recovered at 01:19 UTC after brief exhaustion (~00:48-01:19 UTC); stable since.
