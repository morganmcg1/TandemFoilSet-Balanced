# SENPAI Research State

- **As of:** 2026-05-13 05:20 (slice_num axis closed at 64; thorfinn warmup sent back for stacking; edward assigned ref-16; 11 effective merges; 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=83.95 / test=74.70 (fern #1855 eta_min=5e-5). Most conventional hyperparameter axes now bracketed: max_norm, surf_weight, depth, AdamW betas, slice_num. Remaining levers concentrate on EMA, capacity-within-block (mlp_ratio), regularization (wd), schedule shape (warmup, lr peak, eta_min), attention heads, and position encoding granularity.

**Sub-80 val is the next milestone.** EMA (askeladd, in rebase) remains the highest expected stacking gain.

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
- **slice_num axis CLOSED**: 64 optimum (128 +19% wall-clock-bound; going below 64 would lose capacity).
- **loss shape axis CLOSED**: log-cosh regression — grad-clip removes tail-shape benefit.
- **eta_min OPEN**: 5e-5 merged. Bracketing with 1e-4 (fern #1901) in flight.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0 (frieren #1696) — val=96.78.
2. **Loss weighting — MERGED.** surf_weight=5 (tanjiro #1762) — val=90.58.
3. **LR schedule alignment — MERGED.** T_max=18 (nezuko #1695) — val=84.67.
4. **LR floor — MERGED.** eta_min=5e-5 (fern #1855) — val=83.95.
5. **EMA weight averaging.** Askeladd #1540 rebasing onto current HEAD. Expected sub-80.
6. **Attention heads.** Nezuko #1853 (n_head=8) — zero-param inductive bias probe; pod stale at 03:05 update.
7. **LR floor bracket.** Fern #1901 (eta_min=1e-4) — bracket higher than merged 5e-5.
8. **LR warmup (rerun).** Thorfinn #1812 (lr-warmup-1ep) — first run hinted at improvement (val=83.64 vs old baseline) but on different schedule; sent back to stack on current HEAD with eta_min=5e-5 preserved.
9. **Peak LR reduction.** Alphonse #1914 (lr=5e-4→3e-4) — smaller steps, tighter basin.
10. **FFN capacity.** Frieren #1919 (mlp_ratio=2→4) — double per-block FFN width.
11. **Regularization reduction.** Tanjiro #1923 (wd=1e-4→1e-5) — less weight decay for small model.
12. **Position encoding granularity.** Edward #1943 (ref=8→16) — zero-param, finer unified_pos.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **eta_min=5e-5 (fern #1855)** | **83.95** | **74.70** | merged + eta_min=5e-5 | **MERGED — CURRENT BEST** |
| lr-warmup-1ep (thorfinn #1812) | 83.64 | 74.65 | T_max=17 cosine + 1ep warmup, eta_min=0.0 | **SENT BACK — within noise; needs stacking on current HEAD** |
| T_max=18 (nezuko #1695) | 84.67 | 74.94 | merged + T_max=18 | MERGED → superseded |
| β2=0.98 (frieren #1886) | 85.94 | 76.16 | T_max=18 base + β2=0.98 | CLOSED |
| β1=0.95 (tanjiro #1888) | 88.32 | 78.58 | T_max=18 base + β1=0.95 | CLOSED |
| surf_weight=5 (tanjiro #1762) | 90.58 | 80.00 | merged + surf_weight=5 | MERGED → superseded |
| layers-7 (alphonse #1834) | 96.81 | 87.41 | current HEAD + n_layers=7 | CLOSED |
| slice_num=128 (edward #1902) | 99.86 | 90.41 | current HEAD + slice_num=128 | **CLOSED — wall-clock bound** |
| EMA (askeladd #1540) | 99.60 | 91.15 | Huber + EMA (no grad-clip) | Rebasing onto current HEAD |

## Active student assignments (all 8)

### Priority: rebase onto current HEAD
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Rebased; EMA + full recipe is primary stacking test; expected sub-80.
- **PR #1812 — `lr-warmup-1ep` (thorfinn)** — **WIP (rerun)** — Sent back: rebase + add eta_min=5e-5 to cosine portion for apples-to-apples comparison.

### Capacity / architecture probes
- **PR #1919 — `mlp-ratio-4` (frieren)** — **WIP** — mlp_ratio=2→4 per-block FFN doubling.
- **PR #1943 — `ref-16` (edward)** — **WIP (new)** — unified_pos ref=8→16, finer position normalization.
- **PR #1853 — `n-head-8` (nezuko)** — **WIP (stale)** — n_head=4→8 attention head probe. No comments since 03:05; pod may be running or stuck.

### Regularization / generalization probes
- **PR #1923 — `wd-1e-5` (tanjiro)** — **WIP** — weight decay 1e-4→1e-5; less regularization.

### Schedule / LR tuning probes
- **PR #1901 — `eta-min-1e-4` (fern)** — **WIP** — bracket eta_min higher: 5e-5→1e-4.
- **PR #1914 — `lr-3e-4` (alphonse)** — **WIP** — lower peak LR 5e-4→3e-4.

## Closed / dead ends
- max_norm axis CLOSED: bracketed at 0.5/1.0/3.0
- surf_weight axis CLOSED: bracketed at 3/5/10/20
- depth axis CLOSED: 5/6/7 monotonically worse under global grad-clip
- AdamW β1 axis CLOSED: 0.9 optimum, 0.95 regresses
- AdamW β2 axis CLOSED: 0.999 optimum, 0.98 regresses
- slice_num axis CLOSED: 64 optimum, 128 wall-clock bound
- log-cosh (#1635): regression under grad-clip
- wd5e-4 (#1394, pre-recipe): regression — not directly applicable
- surf_weight=20 (#1570): rolled back
- hidden192 (#1406): wall-clock-bound
- hidden256 (#1575): wall-clock-bound
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)
- global-pos-norm seeded (#1576): within σ noise
- huber-seed7-variance (#1714): informational σ calibration (σ≈8.5)

## Highest-priority stacking targets

1. **EMA (askeladd #1540)** — rebased onto current HEAD. Highest expected single gain. Expected sub-80.
2. **lr-warmup rerun (thorfinn #1812)** — first run hinted improvement; awaiting apples-to-apples confirmation on current HEAD.
3. **All 6 other probes** — capacity/reg/schedule/attention/position levers in flight; each could compound if it hits.
