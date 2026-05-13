# SENPAI Research State

- **As of:** 2026-05-13 05:00 (AdamW betas axis fully bracketed and closed; tanjiro #1923 wd-1e-5 + frieren #1919 mlp-ratio-4 assigned; 11 effective merges; 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=83.95 / test=74.70 (fern #1855 eta_min=5e-5). Optimizer beta tuning closed (both β1 and β2 regress from defaults). Depth axis closed (5 optimum, 6/7 regress). Bracketing exhausted on conventional hyperparameter axes; remaining levers are capacity (mlp_ratio, slice_num), regularization (wd, EMA), schedule (warmup, eta_min, lr_peak), and architectural (attention heads).

**Sub-80 val is the next milestone.** EMA (askeladd, in rebase) remains the highest expected stacking gain. 7 in-flight probes covering capacity, regularization, schedule, and attention.

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

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=18, eta_min=5e-5), AdamW(0.9, 0.999), loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 0.5→91.01, **1.0→84.67** (optimum), 3.0→86.30.
- **surf_weight axis CLOSED**: 3→90.88, **5→optimum**, 10→96.78, 20→127.94.
- **depth axis CLOSED**: 5→83.95, 6→93.97, 7→96.81. Monotonic regression under global grad-clip.
- **AdamW β1 axis CLOSED**: 0.9 (optimum), 0.95 (+4.3% regression).
- **AdamW β2 axis CLOSED**: 0.999 (optimum), 0.98 (+1.5% regression).
- **loss shape axis CLOSED**: log-cosh regression — grad-clip removes tail-shape benefit.
- **eta_min OPEN**: 5e-5 merged. Bracketing with 1e-4 (fern #1901) in flight.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0 (frieren #1696) — val=96.78.
2. **Loss weighting — MERGED.** surf_weight=5 (tanjiro #1762) — val=90.58.
3. **LR schedule alignment — MERGED.** T_max=18 (nezuko #1695) — val=84.67.
4. **LR floor — MERGED.** eta_min=5e-5 (fern #1855) — val=83.95.
5. **EMA weight averaging.** Askeladd #1540 rebasing onto current HEAD. Expected sub-80.
6. **Attention heads.** Nezuko #1853 (n_head=8) — zero-param inductive bias probe.
7. **LR floor bracket.** Fern #1901 (eta_min=1e-4) — bracket higher than merged 5e-5.
8. **Physics slice routing.** Edward #1902 (slice_num=128) — finer Transolver routing, zero-param.
9. **LR warmup.** Thorfinn #1812 (lr-warmup-1ep) — 1-epoch warmup + cosine.
10. **Peak LR reduction.** Alphonse #1914 (lr=5e-4→3e-4) — smaller steps, tighter basin.
11. **FFN capacity.** Frieren #1919 (mlp_ratio=2→4) — double per-block FFN width.
12. **Regularization reduction.** Tanjiro #1923 (wd=1e-4→1e-5) — less weight decay for small model.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **eta_min=5e-5 (fern #1855)** | **83.95** | **74.70** | merged + eta_min=5e-5 | **MERGED — CURRENT BEST** |
| T_max=18 (nezuko #1695) | 84.67 | 74.94 | merged + T_max=18 | MERGED → superseded |
| β2=0.98 (frieren #1886) | 85.94 | 76.16 | T_max=18 base + β2=0.98 | **CLOSED — +1.5% regression** |
| β1=0.95 (tanjiro #1888) | 88.32 | 78.58 | T_max=18 base + β1=0.95 | **CLOSED — +4.3% regression** |
| surf_weight=5 (tanjiro #1762) | 90.58 | 80.00 | merged + surf_weight=5 | MERGED → superseded |
| grad-clip max_norm=1.0 (#1696) | 96.78 | 86.56 | merged + grad-clip | MERGED → superseded |
| layers-7 (alphonse #1834) | 96.81 | 87.41 | current HEAD + n_layers=7 | **CLOSED — depth axis closed** |
| layers-6 (edward #1730) | 93.97 | 83.05 | grad-clip+surf5+T15 + layers=6 | **CLOSED — depth axis closed** |
| EMA (askeladd #1540) | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Rebasing onto current HEAD |

## Active student assignments (all 8)

### Priority: rebase onto current HEAD
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Rebased; EMA + full recipe is primary stacking test; expected sub-80.

### Capacity / architecture probes
- **PR #1919 — `mlp-ratio-4` (frieren)** — **WIP (new)** — mlp_ratio=2→4 per-block FFN doubling; ~+30% params at same depth
- **PR #1902 — `slice-num-128` (edward)** — **WIP** — Transolver physics slices 64→128, zero-param
- **PR #1853 — `n-head-8` (nezuko)** — **WIP** — n_head=4→8 zero-param inductive bias probe

### Regularization / generalization probes
- **PR #1923 — `wd-1e-5` (tanjiro)** — **WIP (new)** — weight decay 1e-4→1e-5; less regularization

### Schedule / LR tuning probes
- **PR #1901 — `eta-min-1e-4` (fern)** — **WIP** — bracket eta_min higher: 5e-5→1e-4
- **PR #1914 — `lr-3e-4` (alphonse)** — **WIP** — lower peak LR 5e-4→3e-4
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP** — 1-epoch warmup + cosine

## Closed / dead ends
- max_norm axis CLOSED: bracketed at 0.5/1.0/3.0
- surf_weight axis CLOSED: bracketed at 3/5/10/20
- depth axis CLOSED: 5/6/7 monotonically worse under global grad-clip
- AdamW β1 axis CLOSED: 0.9 optimum, 0.95 regresses
- AdamW β2 axis CLOSED: 0.999 optimum, 0.98 regresses
- log-cosh (#1635): regression under grad-clip
- wd5e-4 (#1394, pre-recipe): regression — not directly applicable
- surf_weight=20 (#1570): rolled back
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)
- global-pos-norm seeded (#1576): within σ noise
- huber-seed7-variance (#1714): informational σ calibration (σ≈8.5)

## Highest-priority stacking target

**EMA (askeladd #1540)** — rebased onto current HEAD. EMA on the older recipe was best-in-class; expected to push val below 80 on the full stack. All other 7 in-flight probes are independent levers that compound if they hit.
