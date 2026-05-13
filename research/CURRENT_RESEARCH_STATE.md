# SENPAI Research State

- **As of:** 2026-05-13 02:05 (edward layers-6 val=98.24 + fern log-cosh val=104.31 sent back for grad-clip HEAD reruns; thorfinn #1576 closed; thorfinn new assignment #1812 lr-warmup; all 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=96.78 / test=86.56 (frieren #1696 grad-clip max_norm=1.0). Four lever tests now rebasing onto grad-clip HEAD simultaneously: depth (edward #1730 layers-6, val=98.24 on Huber base Δ−11.2%), log-cosh loss (fern #1635, val=104.31 on Huber base Δ−5.7%), EMA (askeladd #1540, val=99.60 on Huber+EMA Δ−9.9%), T_max=18 (nezuko #1695, val=109.43). Most promising expected improvement: layers-6 stack targeting sub-90.

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
2. **EMA weight averaging — HIGH PRIORITY (next stack).** Askeladd #1540 delivered val=99.60 on Huber+EMA (no grad-clip). Sent back to rebase onto grad-clip HEAD. EMA+grad-clip expected sub-90. This is the critical next merge candidate.
3. **Depth / capacity.** Edward #1730 (layers-6) returned val=98.24 on Huber base (Δ−11.2% from Huber baseline). Sent back for grad-clip HEAD rerun — expected proportional improvement would target sub-88. Most promising architectural lever on the board.
4. **Robust loss — MERGED.** Huber loss (#1374) merged at val=110.59. Fern #1635 log-cosh returned val=104.31 (Δ−5.7% vs Huber, better than Huber). Sent back for grad-clip HEAD rerun (log-cosh REPLACES Huber in that rerun).
5. **LR schedule.** T_max=18 (nezuko #1695) returned val=109.43 on Huber base; sent back for grad-clip HEAD rerun. Thorfinn #1812 lr-warmup (1ep linear + cosine) now assigned — tests epoch-1 noise suppression.
6. **Loss weighting.** Tanjiro #1762 (surf_weight=5.0) — in flight on grad-clip HEAD.
7. **Positional encoding.** Global pos norm (thorfinn #1576) — CLOSED. Within σ noise on grad-clip HEAD; mechanism absorbed by grad-clip + Huber.
8. **Gradient clip sensitivity.** Frieren #1759 max_norm=0.5 — in flight on grad-clip HEAD.
9. **Cross-seed σ.** Alphonse #1714 (Huber-seed7) in flight — σ calibration on Huber recipe.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **grad-clip max_norm=1.0 (frieren #1696)** | **96.78** | **86.56** | merged + Huber + grad-clip + seed=42 | **MERGED — CURRENT BEST** |
| EMA (askeladd #1540) 2nd run | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Sent back for grad-clip HEAD rerun |
| layers-6 (edward #1730) | 98.24 | 88.35 | Huber + layers=6 (no grad-clip) | Sent back for grad-clip HEAD rerun |
| log-cosh (fern #1635) | 104.31 | 95.10 | merged recipe + log-cosh (no grad-clip) | Sent back for grad-clip HEAD rerun (replaces Huber) |
| T_max=18 (nezuko #1695) | 109.43 | 101.08 | Huber + T_max=18 (no grad-clip) | Sent back for grad-clip HEAD rerun |
| Huber loss (edward #1374) | 110.59 | 102.28 | merged + Huber(beta=1.0) + seed=42 | MERGED → superseded |
| T_max=15 (nezuko #1542) | 114.81 | 104.68 | merged + T_max=15, seed=42 | MERGED → superseded |
| Seeded baseline (alphonse #1577) | 116.43 | 108.87 | merged + seed=42 | MERGED → superseded |
| EMA (askeladd #1540) 1st run | 121.16 | 108.69 | default + EMA (unseeded, old config) | Superseded |
| hidden256 (tanjiro #1575) | 150.77 | 136.31 | merged + hidden256 | **CLOSED — wall-clock-bound** |
| Global pos norm seeded (thorfinn #1576) | 98.41 | 87.51 | grad-clip HEAD + global pos norm | **CLOSED — within σ noise** |
| wd5e-4 (frieren #1394) | 135.35 | NaN | default + wd5e-4 (unseeded) | **CLOSED — regression** |

## Active student assignments

### Priority: rebase onto grad-clip HEAD + seeded rerun
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — val=99.60 on Huber+EMA; sent back. Rebasing onto Huber+grad-clip HEAD. EMA+grad-clip is the primary stacking test; expected sub-90 val.
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP (sent back)** — val=104.31 on merged recipe + log-cosh (no grad-clip). Rebasing onto grad-clip HEAD to replace Huber with log-cosh; target <96.78.
- **PR #1695 — `tmax-18` (nezuko)** — **WIP (sent back)** — val=109.43 on Huber base; rebasing onto grad-clip HEAD. Target <96.78.
- **PR #1730 — `layers-6` (edward)** — **WIP (sent back)** — val=98.24 on Huber base (Δ−11.2%). Rebasing onto grad-clip HEAD. Most promising next merge; target sub-90.

### In flight on grad-clip HEAD
- **PR #1714 — `huber-seed7-variance` (alphonse)** — **WIP** — σ calibration on Huber recipe
- **PR #1759 — `max-norm-0.5` (frieren)** — **WIP** — tighter clip 1.0→0.5 on Huber+grad-clip baseline
- **PR #1762 — `surf-weight-5` (tanjiro)** — **WIP** — surf_weight=10→5 on Huber+grad-clip baseline
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP (new)** — 1-epoch linear warmup + cosine T_max=14; epoch-1 noise suppression on grad-clip HEAD

## Closed / dead ends
- wd5e-4 (#1394): regression on val_avg (+9.2%)
- surf_weight=20 on merged recipe (#1570): regression; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound (12 epochs, val=150.77)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)
- global-pos-norm seeded (#1576): within σ noise on grad-clip HEAD; mechanism absorbed by clipping + Huber

## Highest-priority stacking target

**Merged recipe + Huber + grad-clip + layers-6 (edward rerun)**

grad-clip (#1696) merged at val=96.78. Four levers currently rebasing onto this HEAD: layers-6 (highest expected impact, proportional target ~85–88 val), log-cosh replacement loss, EMA averaging, T_max=18 schedule. Layers-6 + grad-clip stack is the most exciting hypothesis on the board — if it delivers close to the Huber-base proportional improvement, it would give a new best in the low-80s. EMA on grad-clip HEAD is the second-highest priority (expected sub-90).
