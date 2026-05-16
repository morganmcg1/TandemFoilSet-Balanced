# SENPAI Research State

- **Last updated:** 2026-05-16 01:25 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 01:20 UTC — no open issues).

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

## Active PRs (zero idle students)

### Tier-1 — EMA decay bracketing (WIP)

| PR | Student | Hypothesis | W&B best so far | Notes |
|----|---------|-----------|-----------------|-------|
| #3543 | alphonse | ema-decay-push (0.98, 0.97, 0.95) | 90.839 (0.98 ≈ tied); 0.97 running | Optimum may be at 0.99 |

### Tier-2 — Winner awaiting verify (WIP)

| PR | Student | Hypothesis | W&B best | Vs baseline | Status |
|----|---------|-----------|----------|------------|--------|
| **#3475** | **askeladd** | **asinh-pressure (scale=1.0)** | **88.67** (old ema_decay=0.999 base) | **−2.1% WIN** | **Sent back — rebase + single arm verify on ema_decay=0.99; merge on receipt** |

### Tier-2 — Mechanism sweeps (WIP)

| PR | Student | Hypothesis | W&B best | Vs baseline | Status |
|----|---------|-----------|----------|------------|--------|
| #3477 | thorfinn | physics-continuity (w=0.01, 0.1, 0.5) | 98.62 (w=0.1); w=0.5 pending | +8.9% REGRESS | w=0.5 still to run |
| #3571 | fern | depth-sweep (n_layers=6, then 7) | — just assigned | — | First run pending |

### Tier-3 — Round-4 fresh hypotheses (newly assigned)

| PR | Student | Hypothesis | Axis | Arms |
|----|---------|-----------|------|------|
| #3575 | edward | p-surf-weight (3.0, 5.0) | Loss: per-channel pressure upweight | 2 |
| #3576 | nezuko | wd-sweep (1e-3, 5e-3) | Regularization: L2 weight norm | 2 |
| #3577 | tanjiro | slice-num-128 | Architecture: PhysicsAttention tokens 64→128 | 1+1 cond. |
| #3578 | frieren | re-sinusoidal-embed (d=8) | Feature: Re scalar → sinusoidal encoding | 1 |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | All 4 splits improve, 3 runs |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | All 4 splits ≥−20%; 2 runs |
| **#3474 (alphonse)** | **EMA decay=0.99 (faster shadow tracking)** | **90.6131** | **−4.0%** | **Monotone sweep; 3 arms all beat prior baseline** |

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
- **asinh-pressure**: 88.67 winner on old base — verify pending (#3475 WIP)
- **physics-continuity**: w=0.01/0.1 both regress; w=0.5 pending (#3477 WIP)

### EMA decay bracketing (in progress #3543)
- ema_decay=0.99 (baseline): 90.613
- ema_decay=0.98: 90.839 (≈TIED — optimum may be at 0.99)
- ema_decay=0.97: running; 0.95: pending

### Round 4 hypotheses (newly launched)
All 4 target orthogonal axes unexplored so far:
1. **p-surf-weight** (edward #3575): loss gradient alignment with primary metric
2. **wd-sweep** (nezuko #3576): L2 regularization for OOD Re generalization
3. **slice-num-128** (tanjiro #3577): architecture capacity via token count
4. **re-sinusoidal-embed** (frieren #3578): spectral Re-regime encoding for val_re_rand

## Strategic outlook

**Top merge candidate**: askeladd #3475 asinh-pressure (88.67 −2.1%) — needs verify arm on ema_decay=0.99 stack. Expected to compound with EMA.

**If asinh compounds with ema-decay-push**: combining both on the ema_decay=0.99 stack could push below 86 if 0.97 beats 0.99.

**Architecture axis**: fern (n_layers=6) + tanjiro (slice_num=128) are independent capacity dimensions — could compound if both win.

**OOD Re bottleneck**: val_re_rand (86.494) only −0.2% over 3 rounds. frieren (Re-sinusoidal) + nezuko (wd) both directly target this.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT → cruise=NaN fleet-wide.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs).
- **Zero idle students**: 8 WIP PRs total.
- REST API recovered at 01:19 UTC after brief exhaustion (~00:48-01:19 UTC).
