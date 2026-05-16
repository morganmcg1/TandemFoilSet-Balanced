# SENPAI Research State

- **Last updated:** 2026-05-16 01:00 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 00:30 UTC — no open issues).

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
| val_re_rand | 86.494 | −0.2% |

## Active PRs (zero idle students)

### Tier-1 — hyperparameter sweeps (all WIP, training active)

| PR | Student | Hypothesis | W&B best so far | Vs new baseline 90.61 |
|----|---------|-----------|-----------------|----------------------|
| #3454 | edward | lr-sweep (1e-3, 2e-3, 5e-3) | 93.47 (lr=1e-3); lr=2e-3 running | +3.2% so far |
| #3456 | nezuko | cosine T_max=14, T_max=9 | 96.04 (T_max=14 only; T_max=9 NOT YET RUN) | +5.9% so far |
| #3458 | tanjiro | huber-delta sweep (0.5, 1.0, 2.0, 0.0) | 93.91 (δ=1.0); δ=0.0 running | +3.6% so far |

### Tier-2 — orthogonal mechanisms (WIP)

| PR | Student | Hypothesis | W&B best | Vs new baseline | Status |
|----|---------|-----------|----------|-----------------|--------|
| **#3475** | **askeladd** | **asinh-pressure (scale=1.0)** | **88.67** (old base) | **−2.1% WIN** | **Sent back to WIP — rebase onto new main + re-verify single arm on ema_decay=0.99 baseline; merge on receipt** |
| #3476 | frieren | swa-on-full-stack start=6 | 96.00 | +5.9% REGRESS | start=4 arm still pending |
| #3477 | thorfinn | physics-continuity | 98.66 (w=0.01); w=0.1 running | +8.9% so far | Await w=0.1 + w=0.5 completion |
| #3571 | fern | depth-sweep (n_layers=6, then 7) | — just assigned | — | After her geom-aug closed (99.79 regression) |

### Tier-3 — follow-up on confirmed direction (EMA decay bracketing)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3543 | alphonse | ema-decay-push (0.98, 0.97, 0.95) | WIP |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | All 4 splits improve, 3 runs |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | All 4 splits ≥−20%; 2 runs |
| **#3474 (alphonse)** | **EMA decay=0.99 (faster shadow tracking)** | **90.6131** | **−4.0%** | **Monotone sweep; 3 arms all beat prior baseline** |

## Key findings from Rounds 1–3 (so far)

### Round 1: EMA wins, structural bias loses
EMA trajectory averaging (decay=0.999): −11.1% val_avg, all 4 splits improve. Loss-redirection family all failed (structural pattern).

### Round 2: grad_clip + Huber compounds with EMA
EMA+clip5+Huber δ=1.0: −22.4% val_avg. Three mechanisms orthogonal (EMA smoothing, grad clip, Huber outlier control).

### Round 3 (in progress): Faster EMA decay wins too
- EMA decay 0.99 > 0.995 > 0.997 > 0.999 within the 14-epoch budget
- Mechanism: shorter half-life (69 steps vs 693) lets shadow track late-training improvements without lagging
- Shadow still helps at ema_lag_rel=2% — denoises last few steps, not a long-window average
- **Trend still monotone at 0.99 — optimum not yet bracketed from below**

### Early Round-3 signal from Tier-2:
- **asinh-pressure scale=1.0 (askeladd): val_avg=88.67** on OLD baseline — beats new 90.61 by 2.1%. Sent back to rebase + verify on new ema_decay=0.99 stack.
- Geometry mirror (fern): 99.79 — CLOSED (regression). Vertical mirror augmentation too aggressive at p=0.5; tandem mirror not implemented.
- SWA on full stack (frieren): 96.00 — regresses vs new baseline. SWA's 14-epoch window too short.
- Physics continuity (thorfinn): 98.66 (w=0.01); w=0.1 still running.

## Round 3 goal and next priorities

**Primary goal:** push `val_avg/mae_surf_p` below 88 (combining multiple improvements).

**Most likely next merge(s):**
1. **#3475 askeladd asinh-pressure** (88.67, needs terminal result posted) → merge immediately on receipt
2. **#3543 alphonse ema-decay-push** — if 0.98 or 0.97 beats 90.61, compound with asinh for a double-win

**Strategic outlook:**
- If asinh (88.67) and EMA decay-push both win, combining them (arinh on 0.99 config = compound) could push below 86.
- Tier-1 Tier-2 hypothesis coverage: LR (edward), T_max (nezuko), Huber-delta (tanjiro) all showing ≤3-6% regression vs new baseline — any of these would be secondary wins at best.
- After Round-3 completes: consider (a) LR+faster-EMA compound, (b) asinh+new-EMA compound, (c) geometry-augmentation with corrected indices if fern identified an index error.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT pressure → cruise=NaN fleet-wide.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs); all runs hit cap.
- **Zero idle students**: alphonse just assigned #3543; 9 WIP PRs total (alphonse + 8 others).
- REST API was rate-limited earlier (~21:46-22:05 UTC) but GraphQL fallback worked. Both healthy now.
