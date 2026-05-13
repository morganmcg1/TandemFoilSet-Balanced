# SENPAI Research State

- **As of:** 2026-05-13 01:27 (grad-clip merged val=96.78 new best; T_max=18 on Huber base sent back for grad-clip HEAD rerun; EMA on Huber sent back for grad-clip HEAD rerun; all 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **NEW BEST:** val=96.78 / test=86.56 (frieren #1696 grad-clip max_norm=1.0 on merged recipe). Grad-clip delivers a stunning 15.7% improvement — the mechanism is normalized gradient descent (every step clipped since norms 30–1000 >> max_norm). Stack-target: EMA + grad-clip (askeladd rebasing) expected sub-90.

## Merged recipe (current advisor base)

Eight merges (after surf_weight=20 effectively reverted):

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59
8. **#1696** (grad-clip max_norm=1.0) — val=96.78 NEW BEST

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Themes

1. **Gradient clipping — MERGED (HIGHEST SINGLE LEVER).** grad-clip max_norm=1.0 (frieren #1696) merged — val=96.78 new best (−15.7% vs T_max=15 baseline, −12.5% vs Huber). Mechanism: every step clipped (norms 30–1000 >> 1.0) → normalized gradient descent. Follow-up: frieren #1759 max_norm=0.5.
2. **EMA weight averaging — HIGHEST PRIORITY (next stack).** Askeladd #1540 delivered val=99.60 on Huber+EMA (no grad-clip). Sent back to rebase onto grad-clip HEAD. EMA+grad-clip expected sub-90. This is the critical next merge candidate.
3. **Robust loss — MERGED.** Huber loss (#1374) merged at val=110.59, now superseded by grad-clip. Current HEAD has both Huber + grad-clip (untested combination). Log-cosh (fern #1635) in flight as orthogonal comparison.
4. **LR schedule.** T_max=18 (nezuko #1695) Huber-base result val=109.43 / test=101.08 (Δ−1.16 vs Huber, within σ but per-split signal clean); sent back to rerun on grad-clip HEAD.
5. **Architecture / capacity.** hidden256 CLOSED (wall-clock-bound). Edward #1730 (layers-6, n_layers 5→6) in flight — depth test on new HEAD.
6. **Loss weighting.** Tanjiro #1762 (surf_weight=5.0) assigned — tests if surf weight needs rebalancing given changed gradient dynamics.
7. **Positional encoding.** Global pos norm (thorfinn #1576) — sent back for seeded rerun.
8. **Cross-seed σ.** Alphonse #1714 (Huber-seed7) in flight — σ on Huber baseline pre-grad-clip. Will need repeat on grad-clip HEAD after first confirmations.

## Round-2 leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **grad-clip max_norm=1.0 (frieren #1696)** | **96.78** | **86.56** | merged + Huber + grad-clip + seed=42 | **MERGED — CURRENT BEST** |
| EMA (askeladd #1540) 2nd run | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Sent back for grad-clip HEAD rerun |
| T_max=18 (nezuko #1695) | 109.43 | 101.08 | Huber + T_max=18 (no grad-clip) | Sent back for grad-clip HEAD rerun |
| Huber loss (edward #1374) | 110.59 | 102.28 | merged + Huber(beta=1.0) + seed=42 | MERGED → superseded |
| T_max=15 (nezuko #1542) | 114.81 | 104.68 | merged + T_max=15, seed=42 | MERGED → superseded |
| Seeded baseline (alphonse #1577) | 116.43 | 108.87 | merged + seed=42 | MERGED → superseded |
| EMA (askeladd #1540) 1st run | 121.16 | 108.69 | default + EMA (unseeded, old config) | Superseded |
| hidden256 (tanjiro #1575) | 150.77 | 136.31 | merged + hidden256 | **CLOSED — wall-clock-bound** |
| Global pos norm (thorfinn #1576) | 123.56 | 109.71 | merged + global_norm (unseeded) | Sent back for seeded rerun |
| wd5e-4 (frieren #1394) | 135.35 | NaN | default + wd5e-4 (unseeded) | **CLOSED — regression** |

## Active student assignments

### Priority: rebase onto grad-clip HEAD + seeded rerun
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — Second run val=99.60 on Huber+EMA (no grad-clip); sent back. Rebasing onto Huber+grad-clip HEAD. EMA+grad-clip is the primary stacking test; expected sub-90 val.

### Round-2 in flight
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — **WIP (sent back)** — seeded rerun on grad-clip HEAD; pod active, training in-flight
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP** — log-cosh alt to Huber (pre-grad-clip HEAD, result pending); pod active at 100% GPU
- **PR #1695 — `tmax-18` (nezuko)** — **WIP (sent back)** — Huber-base result val=109.43; rebasing onto grad-clip HEAD
- **PR #1714 — `huber-seed7-variance` (alphonse)** — **WIP** — σ calibration on Huber recipe
- **PR #1730 — `layers-6` (edward)** — **WIP** — n_layers=5→6 depth test on grad-clip HEAD
- **PR #1759 — `max-norm-0.5` (frieren)** — **WIP** — tighter clip 1.0→0.5 on Huber+grad-clip baseline
- **PR #1762 — `surf-weight-5` (tanjiro)** — **WIP** — surf_weight=10→5 on Huber+grad-clip baseline

## Closed / dead ends
- wd5e-4 (#1394): regression on val_avg (+9.2%)
- surf_weight=20 on merged recipe (#1570): regression; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound (12 epochs, val=150.77)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)

## Highest-priority stacking target

**Merged recipe + Huber + grad-clip + EMA (all seeded)**

grad-clip (#1696) merged at val=96.78. Gate is askeladd's EMA rebase onto Huber+grad-clip HEAD. EMA+grad-clip expected sub-90. Simultaneously: max_norm=0.5 (frieren #1759) and surf_weight=5.0 (tanjiro #1762) as orthogonal lever tests.
