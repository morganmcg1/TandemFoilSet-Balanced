# SENPAI Research State

- **As of:** 2026-05-13 00:22 (edward #1730 layers-6 assigned; all 8 students active; tanjiro+fern+askeladd actively training)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **NEW BEST:** val=110.59 / test=102.28 (edward #1374 Huber loss on merged recipe). Huber stacks cleanly on T_max=15 — a −4.22 val improvement confirmed on seed=42. Cross-seed σ ≈ 3.5 val (alphonse #1685). Stack-target: Huber + EMA + (T_max=18 or grad-clip).

## Merged recipe (current advisor base)

Seven merges (after surf_weight=20 effectively reverted):

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59 NEW BEST

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, loss=F.smooth_l1_loss(beta=1.0)`

## Themes

1. **Robust loss functions — MERGED.** Huber loss (#1374, edward) merged — val=110.59 new best (−3.7% vs T_max=15). Stack-target: now add EMA on top of Huber. Log-cosh (fern #1635) still in flight as orthogonal alt.
2. **EMA weight averaging — HIGHEST PRIORITY.** Askeladd #1540 (EMA decay=0.999) needs rebase onto new HEAD (now includes Huber). Branch was CONFLICTING; student picked up work at 00:00 UTC. Seeded rerun on Huber recipe expected to stack cleanly. Prior result val=121.16 unseeded/pre-Huber — expected improvement on this HEAD.
3. **LR schedule.** T_max=15 merged (val=114.81, superseded by Huber). T_max=18 follow-up (nezuko #1695) running — may give another 1-2 pts on top of Huber.
4. **Architecture / capacity.** hidden256 (tanjiro #1575) training at GPU 100%. Edward #1730 (layers-6, n_layers 5→6) assigned — orthogonal depth test. Together: 2×2 design {5,6 layers} × {128,256 hidden}. Key diagnostic: single_in_dist (127.85) is worse than OOD cruise (95.72) — suggests depth-capacity bottleneck on training distribution.
5. **Gradient regularization.** Gradient clipping max_norm=1.0 (frieren #1696) in flight.
6. **Positional encoding.** Global pos norm (thorfinn #1576) — sent back for seeded rerun on new HEAD.
7. **Cross-seed σ calibration.** σ on pre-Huber recipe: ≈3.5 val / 0.5 test (alphonse #1685). Alphonse #1714 will measure σ on Huber recipe specifically.

## Round-2 leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **Huber loss (edward #1374)** | **110.59** | **102.28** | merged + Huber(beta=1.0) + seed=42 | **MERGED — CURRENT BEST** |
| T_max=15 + merged (nezuko #1542) | 114.81 | 104.68 | merged + T_max=15, seed=42 | MERGED → superseded |
| Seeded baseline (alphonse #1577) | 116.43 | 108.87 | merged + seed=42 | MERGED → superseded |
| EMA decay=0.999 (askeladd #1540) | 121.16 | 108.69 | default + EMA (unseeded, OLD config) | Rebase onto Huber HEAD pending |
| Global pos norm (thorfinn #1576) | 123.56 | 109.71 | merged + global_norm (unseeded) | Sent back for seeded rerun |
| Default + scoring-fix (#1512) | 123.99 | 110.97 | default only | Superseded |
| wd5e-4 (frieren #1394) | 135.35 | NaN | default + wd5e-4 (unseeded) | **CLOSED — regression** |

## Active student assignments

### Priority: rebase onto Huber HEAD + seeded rerun
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — Picked up PR at 00:00 UTC; rebasing onto Huber HEAD. EMA on top of Huber is the primary stacking test; expected to push val below 108.

### Round-2 in flight
- **PR #1575 — `hidden256-bf16` (tanjiro)** — **WIP (training)** — GPU at 100% since 23:58. First capacity test on merged recipe.
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — **WIP (sent back)** — seeded rerun on Huber HEAD required
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP (training, GPU 98%)** — log-cosh alt to Huber
- **PR #1695 — `tmax-18` (nezuko)** — **WIP** — T_max=15→18 schedule refinement
- **PR #1696 — `grad-clip-1.0` (frieren)** — **WIP** — gradient clipping max_norm=1.0
- **PR #1714 — `huber-seed7-variance` (alphonse)** — **WIP** — σ calibration on Huber recipe
- **PR #1730 — `layers-6` (edward)** — **WIP** — n_layers=5→6 depth capacity test

## Closed / dead ends
- wd5e-4 (#1394): regression on val_avg (+9.2%); mixed per-split signal not net positive
- surf_weight=20 on merged recipe (#1570): regression confirmed; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound (superseded by bf16)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)

## Highest-priority stacking target

**Merged recipe + T_max=15 + Huber + EMA (all seeded)**

Huber (#1374) merged at val=110.59. The gate is now askeladd's EMA rebase onto the Huber HEAD. EMA + Huber should push below 108. Simultaneously: T_max=18 (nezuko #1695) and grad-clip (frieren #1696) as orthogonal levers.
