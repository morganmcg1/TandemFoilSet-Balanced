# SENPAI Research State

- **As of:** 2026-05-13 06:05 (warmup #1812 merged val=82.56 new best; lr axis lower-bound closed; alphonse #1972 batch-size-2, thorfinn #1968 lr-7e-4; 12 effective merges; 8 students active)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=82.56 / test=74.13 (thorfinn #1812 lr-warmup-1ep). Warmup is the 12th merge — it damps epoch-1 AdamW momentum corruption and positions the model better for the final cosine descent. Sub-80 val remains the next milestone.

## Merged recipe (current advisor base — 12 effective merges)

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99
2. **#1513** (bf16 autocast) — 24% per-epoch speedup
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577
5. **#1577** (seed=42 + surf_weight=10 rollback) — val=116.43
6. **#1542** (T_max=15) — val=114.81
7. **#1374** (Huber loss beta=1.0) — val=110.59
8. **#1696** (grad-clip max_norm=1.0) — val=96.78
9. **#1762** (surf_weight=5.0) — val=90.58
10. **#1695** (T_max=18) — val=84.67
11. **#1855** (eta_min=5e-5) — val=83.95
12. **#1812** (lr-warmup-1ep) — val=82.56 **CURRENT BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=4, SequentialLR(LinearLR(1ep warmup, 5e-6→5e-4) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 0.5/1.0/3.0 → 1.0 optimum.
- **surf_weight axis CLOSED**: 3/5/10/20 → 5 optimum.
- **depth axis CLOSED**: 5/6/7 → 5 optimum. Gradient dilution + wall-clock penalty under grad-clip.
- **AdamW β1 axis CLOSED**: 0.9 optimum, 0.95 regresses.
- **AdamW β2 axis CLOSED**: 0.999 optimum, 0.98 regresses.
- **slice_num axis CLOSED**: 64 optimum (128 wall-clock bound).
- **lr lower-bound CLOSED**: 3e-4 dominated by 5e-4 (+8%). 5e-4 optimum on lower side.
- **loss shape axis CLOSED**: log-cosh regression under grad-clip.
- **eta_min OPEN**: 5e-5 merged. Bracketing with 1e-4 (fern #1901) in flight.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0.
2. **Loss weighting — MERGED.** surf_weight=5.
3. **LR schedule alignment — MERGED.** T_max=18.
4. **LR floor — MERGED.** eta_min=5e-5.
5. **LR warmup — MERGED.** 1-epoch warmup (#1812) — val=82.56.
6. **EMA weight averaging.** Askeladd #1540. Expected sub-80. Highest priority.
7. **FFN capacity.** Frieren #1919 (mlp_ratio=4) — in flight.
8. **Attention heads.** Nezuko #1853 (n_head=8) — training completed; awaiting results post.
9. **LR floor bracket.** Fern #1901 (eta_min=1e-4) — in flight.
10. **Position encoding granularity.** Edward #1943 (ref=16) — in flight.
11. **Regularization.** Tanjiro #1923 (wd=1e-5) — in flight.
12. **LR upper bracket.** Thorfinn #1968 (lr=7e-4 + warmup) — NEW. Natural upper-bracket.
13. **Batch size.** Alphonse #1972 (batch_size=4→2) — NEW. 2x optimizer steps/epoch.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **lr-warmup-1ep (thorfinn #1812)** | **82.56** | **74.13** | **MERGED — CURRENT BEST** |
| eta_min=5e-5 (fern #1855) | 83.95 | 74.70 | MERGED → superseded |
| lr-warmup-1ep run 1 (no eta_min) | 83.64 | 74.65 | STALE — superseded by rerun |
| T_max=18 (nezuko #1695) | 84.67 | 74.94 | MERGED → superseded |
| β2=0.98 (frieren #1886) | 85.94 | 76.16 | CLOSED |
| β1=0.95 (tanjiro #1888) | 88.32 | 78.58 | CLOSED |
| lr=3e-4 (alphonse #1914) | 90.67 | 81.94 | CLOSED — +8% regression |
| EMA (askeladd #1540) | 99.60 | 91.15 | Rebasing onto current HEAD — highest priority |

## Active student assignments (all 8)

### Priority: rebase / highest stacking potential
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — EMA + full recipe expected sub-80.

### Capacity / architecture probes
- **PR #1919 — `mlp-ratio-4` (frieren)** — **WIP** — mlp_ratio=2→4 per-block FFN.
- **PR #1853 — `n-head-8` (nezuko)** — **WIP** — training completed ~05:34 UTC, pod had GraphQL rate limit; results expected soon.
- **PR #1943 — `ref-16` (edward)** — **WIP** — unified_pos ref=8→16.

### LR / schedule probes
- **PR #1968 — `lr-7e-4` (thorfinn)** — **WIP (new)** — lr=5e-4→7e-4 with warmup; upper LR bracket.
- **PR #1901 — `eta-min-1e-4` (fern)** — **WIP** — bracket eta_min=5e-5→1e-4.

### Regularization / training dynamics probes
- **PR #1923 — `wd-1e-5` (tanjiro)** — **WIP** — wd=1e-4→1e-5.
- **PR #1972 — `batch-size-2` (alphonse)** — **WIP (new)** — batch_size=4→2, 2x optimizer steps/epoch.

## Closed / dead ends (complete list)
- max_norm: bracketed at 0.5/1.0/3.0
- surf_weight: bracketed at 3/5/10/20
- depth: 5/6/7 monotonically worse (grad-clip gradient dilution)
- AdamW β1: 0.9 optimum, 0.95 regresses
- AdamW β2: 0.999 optimum, 0.98 regresses
- slice_num: 64 optimum, 128 wall-clock bound
- lr lower: 3e-4 dominated by 5e-4
- log-cosh: regression under grad-clip
- hidden192/256: wall-clock-bound
- lr1e3-warmup (pre-recipe): warmup consumed budget
- global-pos-norm: within σ noise
- huber-seed7-variance: informational (σ≈8.5)

## Highest-priority stacking target

**EMA (askeladd #1540)** — expected single-run gain > 5 val points on top of current 82.56. All other probes are supporting levers. With warmup merged, nezuko's n_head=8 about to post, and 6 fresh in-flight probes, the lab is in active discovery phase.
