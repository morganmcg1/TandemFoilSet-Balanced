# SENPAI Research State

- **As of:** 2026-05-13 04:35 (depth axis fully bracketed via layers-7; alphonse assigned lr-3e-4 #1914; 11 effective merges; 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=83.95 / test=74.70 (fern #1855 eta_min=5e-5). Non-zero LR floor keeps final-epoch gradient steps meaningful under normalized gradient descent. Depth axis now fully bracketed and closed (n_layers 5→6→7: monotonic regression under global grad-clip).

**Sub-80 val is the next milestone.** EMA (askeladd, in rebase) is the highest-priority stacking test. Optimizer (β1/β2), attention (n_head=8), architecture (slice_num=128), LR tuning (lr=3e-4, eta_min=1e-4), and warmup probes all in flight.

## Merged recipe (current advisor base — 11 effective merges)

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59
8. **#1696** (grad-clip max_norm=1.0) — val=96.78
9. **#1762** (surf_weight=5.0) — val=90.58
10. **#1695** (T_max=18) — val=84.67
11. **#1855** (eta_min=5e-5) — val=83.95 **CURRENT BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=18, eta_min=5e-5), AdamW, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 0.5→91.01, **1.0→84.67** (optimum), 3.0→86.30. Fully normalized gradient descent is optimal.
- **surf_weight axis CLOSED**: 3→90.88, **5→optimum**, 10→96.78, 20→127.94.
- **depth axis CLOSED (under grad-clip)**: n_layers=5→83.95, 6→93.97, 7→96.81. Monotonically worse. Global clipping causes gradient dilution per layer + wall-clock penalty (18→15→13 epochs for n_layers 5→6→7).
- **loss shape axis CLOSED**: log-cosh regression confirms grad-clip removes tail-shape benefit.
- **eta_min OPEN**: 5e-5 merged. Bracketing with 1e-4 (fern #1901) in flight.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0 (frieren #1696) — val=96.78. Mechanism: normalized gradient descent.
2. **Loss weighting — MERGED.** surf_weight=5 (tanjiro #1762) — val=90.58 −6.4%.
3. **LR schedule alignment — MERGED.** T_max=18 (nezuko #1695) — val=84.67 −6.5%.
4. **LR floor — MERGED.** eta_min=5e-5 (fern #1855) — val=83.95 −0.85%. Non-zero floor activates final-epoch steps.
5. **EMA weight averaging.** Askeladd #1540 rebasing onto current HEAD. EMA+current recipe expected sub-80.
6. **AdamW β2 tuning.** Frieren #1886 (β2=0.999→0.98) — faster variance adaptation.
7. **AdamW β1 tuning.** Tanjiro #1888 (β1=0.9→0.95) — smoother momentum.
8. **Attention heads.** Nezuko #1853 (n_head=8) — zero-param inductive bias probe.
9. **LR floor bracket.** Fern #1901 (eta_min=1e-4) — bracket higher than merged 5e-5.
10. **Physics slice routing.** Edward #1902 (slice_num=128) — finer Transolver routing, zero-param.
11. **LR warmup.** Thorfinn #1812 (lr-warmup-1ep) — 1-epoch warmup + cosine.
12. **Peak LR reduction.** Alphonse #1914 (lr=5e-4→3e-4) — smaller steps, tighter basin convergence.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **eta_min=5e-5 (fern #1855)** | **83.95** | **74.70** | merged + eta_min=5e-5 | **MERGED — CURRENT BEST** |
| T_max=18 (nezuko #1695) | 84.67 | 74.94 | merged + T_max=18 | MERGED → superseded |
| surf_weight=5 (tanjiro #1762) | 90.58 | 80.00 | merged + surf_weight=5 | MERGED → superseded |
| grad-clip max_norm=1.0 (#1696) | 96.78 | 86.56 | merged + grad-clip | MERGED → superseded |
| max_norm=0.5 (frieren #1759) rerun | 91.01 | 80.20 | surf_weight=5 + max_norm=0.5 | **CLOSED — regression** |
| log-cosh (fern #1635) rerun | 91.19 | 81.72 | surf_weight=5 + log-cosh | **CLOSED — regression** |
| layers-7 (alphonse #1834) | 96.81 | 87.41 | current HEAD + n_layers=7 | **CLOSED — +15.3% regression; depth axis closed** |
| layers-6 (edward #1730) | 93.97 | 83.05 | grad-clip+surf5+T15 + layers=6 | **CLOSED — +3.7% regression; depth axis closed** |
| EMA (askeladd #1540) | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Rebasing onto current HEAD |
| Huber loss (#1374) | 110.59 | 102.28 | merged + Huber | MERGED → superseded |

## Active student assignments (all 8)

### Priority: rebase onto current HEAD
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Rebased; EMA + full recipe is primary stacking test; expected sub-80.

### New assignments (fresh hypotheses on current HEAD)
- **PR #1886 — `adamw-beta2-0.98` (frieren)** — **WIP** — β2=0.999→0.98 faster variance adaptation
- **PR #1888 — `adamw-beta1-0.95` (tanjiro)** — **WIP** — β1=0.9→0.95 smoother momentum
- **PR #1853 — `n-head-8` (nezuko)** — **WIP** — n_head=4→8 zero-param inductive bias probe
- **PR #1901 — `eta-min-1e-4` (fern)** — **WIP** — bracket eta_min higher: 5e-5→1e-4
- **PR #1902 — `slice-num-128` (edward)** — **WIP** — double Transolver physics slices 64→128
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP** — 1-epoch warmup + cosine
- **PR #1914 — `lr-3e-4` (alphonse)** — **WIP (new)** — lower peak LR 5e-4→3e-4; smaller steps, tighter basin

## Closed / dead ends
- max_norm axis CLOSED: 0.5 (+0.43), **1.0 (optimum)**, 3.0 (+1.63) — fully bracketed
- surf_weight axis CLOSED: 3 (+0.30), **5 (optimum)**, 10 (+6.20), 20 (+37) — fully bracketed
- depth axis CLOSED: 5 (optimum), 6 (+9.6%), 7 (+15.3%) — monotonically worse under global grad-clip
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

**EMA (askeladd #1540)** — rebased onto current HEAD. EMA on earlier recipe was best-in-class; expected to push val below 80. All other in-flight probes (optimizer betas, attention heads, slice routing, LR tuning, warmup) are lighter tests that compound if they hit.
