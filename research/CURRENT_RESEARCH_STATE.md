# SENPAI Research State

- **As of:** 2026-05-13 02:20 (surf_weight=5 merged val=90.58 new best; σ on Huber recipe ≈8.5 (N=2); 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=90.58 / test=80.00 (tanjiro #1762 surf_weight=5.0). Key insight: grad-clip's normalized gradient descent eliminates the need for heavy surface weighting — lower surf_weight better balances surface-vs-volume residuals, improving all 4 splits. Pre-clip grad norms dropped 10-20× under surf_weight=5.

**Sub-90 target is in reach.** Multiple levers in flight on top of current best: depth (layers-6 edward rerun, layers-7 alphonse), schedule (lr-warmup thorfinn), loss (log-cosh fern rerun), EMA (askeladd rebase).

## Merged recipe (current advisor base — 9 effective merges)

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59
8. **#1696** (grad-clip max_norm=1.0) — val=96.78
9. **#1762** (surf_weight=5.0) — val=90.58 **NEW BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Themes

1. **Gradient clipping — MERGED (HIGHEST SINGLE LEVER).** max_norm=1.0 (frieren #1696) merged — val=96.78. Mechanism: normalized gradient descent. Follow-up: frieren #1759 max_norm=0.5 (on new baseline, rerunning).
2. **Loss weighting — MERGED (SECOND HIGHEST LEVER).** surf_weight=5 (tanjiro #1762) merged — val=90.58 −6.4%. All 4 splits improved. Pre-clip norms dropped 10-20×. Follow-up: surf_weight=3 (tanjiro #1832) continues sweep.
3. **EMA weight averaging.** Askeladd #1540 rebasing onto current HEAD. EMA+grad-clip+surf_weight=5 expected sub-85 val.
4. **Depth / capacity.** Edward #1730 (layers-6) rerunning on grad-clip HEAD (expected sub-85). Alphonse #1834 (layers-7) forward probe. Both depth tests in-flight simultaneously.
5. **Robust loss.** Fern #1635 (log-cosh) rerunning on grad-clip HEAD replacing Huber; target <90.58.
6. **LR schedule.** Nezuko #1695 (T_max=18) rebasing. Thorfinn #1812 (lr-warmup-1ep) in flight.
7. **Gradient clip sensitivity.** Frieren #1759 (max_norm=0.5) sent back for rerun on new surf_weight=5 baseline.
8. **Cross-seed σ.** Alphonse #1714 closed: σ ≈ 8.5 val on Huber recipe (N=2, loose); grad-clip likely reduces σ via deterministic normalized steps. σ on current recipe still unknown — deferred until more confirmations land.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **surf_weight=5 (tanjiro #1762)** | **90.58** | **80.00** | merged + Huber + grad-clip + surf_weight=5 | **MERGED — CURRENT BEST** |
| grad-clip max_norm=1.0 (#1696) | 96.78 | 86.56 | merged + Huber + grad-clip + surf_weight=10 | MERGED → superseded |
| max_norm=0.5 (frieren #1759) | 95.53 | 85.37 | Huber + grad-clip + surf_weight=10 | Sent back (rerun on surf_weight=5 HEAD) |
| EMA (askeladd #1540) 2nd run | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Rebasing onto current HEAD |
| layers-6 (edward #1730) | 98.24 | 88.35 | Huber + layers=6 (no grad-clip) | Rebasing onto grad-clip HEAD; target <90.58 |
| log-cosh (fern #1635) | 104.31 | 95.10 | merged + log-cosh (no grad-clip) | Rebasing onto current HEAD (replaces Huber) |
| T_max=18 (nezuko #1695) | 109.43 | 101.08 | Huber + T_max=18 (no grad-clip) | Rebasing onto current HEAD |
| Huber loss (#1374) | 110.59 | 102.28 | merged + Huber | MERGED → superseded |
| Global pos norm seeded (#1576) | 98.41 | 87.51 | grad-clip HEAD + global pos norm | **CLOSED — within σ noise** |
| hidden256 (#1575) | 150.77 | 136.31 | merged + hidden256 | **CLOSED — wall-clock-bound** |
| wd5e-4 (#1394) | 135.35 | NaN | default + wd5e-4 | **CLOSED — regression** |

## Active student assignments (all 8)

### Priority: rebase onto current HEAD + seeded rerun
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — Rebasing onto Huber+grad-clip+surf_weight=5 HEAD. EMA+current recipe is primary stacking test; expected sub-85.
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP (sent back)** — Rebasing onto current HEAD; replaces Huber with log-cosh; target <90.58.
- **PR #1695 — `tmax-18` (nezuko)** — **WIP (sent back)** — Rebasing onto current HEAD; T_max=18 on full recipe; target <90.58.
- **PR #1730 — `layers-6` (edward)** — **WIP (sent back)** — Rebasing onto grad-clip HEAD; n_layers=5→6; highest expected single gain; target sub-88.
- **PR #1759 — `max-norm-0.5` (frieren)** — **WIP (sent back)** — Rebasing onto surf_weight=5 HEAD; max_norm=1.0→0.5; target <90.58.

### In flight on current HEAD
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP** — 1-epoch linear warmup + cosine T_max=14; epoch-1 noise suppression
- **PR #1832 — `surf-weight-3` (tanjiro)** — **WIP (new)** — Continue surf_weight sweep: 5→3; find optimum
- **PR #1834 — `layers-7` (alphonse)** — **WIP (new)** — Depth probe n_layers 5→7 (forward bet ahead of layers-6 confirmation)

## Closed / dead ends
- wd5e-4 (#1394): regression
- surf_weight=20 (#1570): regression; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)
- global-pos-norm seeded (#1576): within σ noise on grad-clip HEAD
- huber-seed7-variance (#1714): closed — informational σ calibration

## Highest-priority stacking target

**Current recipe + layers-6 (edward confirmation run)**

surf_weight=5 merged at val=90.58. Next most-expected improvement: layers-6 on current HEAD (proportional extrapolation gives ~80.5 val if the Huber-base Δ−11.2% holds). EMA is the second-highest priority (askeladd rebasing). The 5 simultaneous rebases mean results will cluster — next advisory round may have 3-5 review-ready PRs at once.
