# SENPAI Research State

- **As of:** 2026-05-13 03:05 (T_max=18 merged val=84.67 new best; 10 effective merges; 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=84.67 / test=74.94 (nezuko #1695 T_max=18). Schedule alignment — matching the cosine period to the true 18-epoch wall-clock budget — was the latest big win. Loss shape (log-cosh) and tighter clipping (max_norm=0.5) confirmed dead ends under grad-clip+surf_weight=5 recipe.

**Sub-80 val is the next milestone.** Expected from layers-6 (edward, in flight) and EMA (askeladd, in rebase). Multiple schedule and architecture probes in flight.

## Merged recipe (current advisor base — 10 effective merges)

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59
8. **#1696** (grad-clip max_norm=1.0) — val=96.78
9. **#1762** (surf_weight=5.0) — val=90.58
10. **#1695** (T_max=18) — val=84.67 **NEW BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=18, eta_min=0.0), AdamW, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results (closed this round)

- **max_norm=0.5** (frieren #1759): +0.43 val regression on surf_weight=5 HEAD. Under normalized gradient descent, tighter clipping adds no incremental regularization beyond max_norm=1.0 when norms are already 2-5. Dead end on max_norm axis below 1.0.
- **log-cosh loss** (fern #1635): +0.61 val regression on surf_weight=5 HEAD. Grad-clip's global norm normalization removes the tail-shape benefit of log-cosh vs Huber — both reduce to MSE near zero. Loss-shape axis confirmed closed under this recipe.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0 (frieren #1696) — val=96.78. Mechanism: normalized gradient descent.
2. **Loss weighting — MERGED.** surf_weight=5 (tanjiro #1762) — val=90.58 −6.4%.
3. **LR schedule alignment — MERGED.** T_max=18 (nezuko #1695) — val=84.67 −6.5%. Best achievable epoch count, cosine reaches minimum at end of training.
4. **EMA weight averaging.** Askeladd #1540 rebasing onto current HEAD. EMA+current recipe expected sub-80.
5. **Depth / capacity.** Edward #1730 (layers-6) rerunning on current HEAD. Alphonse #1834 (layers-7) in flight.
6. **Selective gradient clipping.** Frieren #1851 (max_norm=3.0) — probe whether looser clipping (some steps unclipped) helps under low-gradient regime.
7. **Attention inductive bias.** Nezuko #1853 (n_head=8) — zero-param doubling of attention heads.
8. **Schedule floor.** Fern #1855 (eta_min=5e-5) — non-zero LR floor to activate final-epoch steps.
9. **LR schedule.** Thorfinn #1812 (lr-warmup-1ep) — notified of new T_max=18 baseline; cosine portion should target T_max=17.
10. **Surface weight sweep.** Tanjiro #1832 (surf_weight=3) — notified of new baseline (84.67).

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **T_max=18 (nezuko #1695)** | **84.67** | **74.94** | merged + T_max=18 | **MERGED — CURRENT BEST** |
| surf_weight=5 (tanjiro #1762) | 90.58 | 80.00 | merged + surf_weight=5 | MERGED → superseded |
| grad-clip max_norm=1.0 (#1696) | 96.78 | 86.56 | merged + grad-clip | MERGED → superseded |
| max_norm=0.5 (frieren #1759) rerun | 91.01 | 80.20 | surf_weight=5 + max_norm=0.5 | **CLOSED — regression** |
| log-cosh (fern #1635) rerun | 91.19 | 81.72 | surf_weight=5 + log-cosh | **CLOSED — regression** |
| EMA (askeladd #1540) | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Rebasing onto current HEAD |
| layers-6 (edward #1730) | 98.24 | 88.35 | Huber + layers=6 (no grad-clip) | Rerunning on current HEAD |
| Huber loss (#1374) | 110.59 | 102.28 | merged + Huber | MERGED → superseded |
| Global pos norm seeded (#1576) | 98.41 | 87.51 | grad-clip HEAD + global pos norm | **CLOSED — within σ noise** |

## Active student assignments (all 8)

### Priority: rebase onto current HEAD + seeded rerun
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — Still needs rebase onto current HEAD (cc7f555). EMA + full recipe is the primary stacking test; expected sub-80.
- **PR #1730 — `layers-6` (edward)** — **WIP** — Rerunning on grad-clip HEAD; n_layers=5→6. Notified of new baseline (84.67). Evaluate on completion; if <84.67 merge, else rebase to T_max=18.
- **PR #1834 — `layers-7` (alphonse)** — **WIP** — Depth probe n_layers 5→7. Notified of new baseline (84.67).

### New assignments (fresh hypotheses on current HEAD)
- **PR #1851 — `max-norm-3-selective` (frieren)** — **WIP (new)** — max_norm=3.0 selective clipping on current recipe
- **PR #1853 — `n-head-8` (nezuko)** — **WIP (new)** — n_head=4→8 zero-param inductive bias probe
- **PR #1855 — `eta-min-5e-5` (fern)** — **WIP (new)** — eta_min=0.0→5e-5 non-zero cosine LR floor

### In flight on current HEAD
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP** — 1-epoch warmup + cosine; notified T_max should be 17 for cosine portion
- **PR #1832 — `surf-weight-3` (tanjiro)** — **WIP** — surf_weight=5→3; notified of new baseline (84.67)

## Closed / dead ends
- max_norm=0.5 (#1759): regression vs surf_weight=5 HEAD — normalized steps already optimal at max_norm=1.0
- log-cosh (#1635): regression vs surf_weight=5 HEAD — grad-clip removes tail-shape benefit
- wd5e-4 (#1394): regression
- surf_weight=20 (#1570): regression; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)
- global-pos-norm seeded (#1576): within σ noise on grad-clip HEAD
- huber-seed7-variance (#1714): informational σ calibration (σ≈8.5 on Huber recipe, N=2)

## Highest-priority stacking target

**Current recipe + layers-6 (edward confirmation run on grad-clip+surf_weight=5+T_max=18 HEAD)**

After T_max=18 merge at 84.67, the next expected big gain is layers-6: Huber-base delta was −11.2% (98.24→87.0 est on grad-clip base). Stacking on T_max=18 targets sub-80. EMA (askeladd) is the second-highest expected stacking gain. The 3 new assignments (max_norm=3, n_head=8, eta_min) are lighter probes that could compound on top.
