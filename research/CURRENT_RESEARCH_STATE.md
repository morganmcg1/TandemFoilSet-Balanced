# SENPAI Research State

- **As of:** 2026-05-12 23:45 (all 8 students active; #1374, #1540, #1575 blocked by GraphQL rate limit, will resume after ~23:49 UTC reset)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **NEW BEST:** val=114.81 / test=104.68 (nezuko #1542 T_max=15 on merged recipe). The schedule fix stacks cleanly with prior merges. Stack-target: Huber + EMA on top of (merged + seed=42 + T_max=15).

## Merged recipe (current advisor base)

Six merges (after surf_weight=20 effectively reverted):

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD
4. **#1369** (surf_weight=10→20) — REVERTED via #1577 rollback (confirmed regression)
5. **#1577** (seed=42 + surf_weight=10 rollback) — determinism + val=116.43
6. **#1542** (T_max=15) — val=114.81 NEW BEST

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW`

## Themes

1. **Robust loss functions — HIGHEST PRIORITY.** Huber loss (#1374, edward, val=112.06 unseeded default) is the strongest single lever discovered. Edward's rebase + seeded rerun on the now-merged-T_max=15 recipe is the critical gate; expected to land sub-110.
2. **LR schedule.** T_max=15 (nezuko #1542) merged — val=114.81 new best. Open: T_max=18 to match achievable epoch count exactly (nezuko follow-up); could give another 1-2 pts.
3. **EMA weight averaging.** EMA decay=0.999 (askeladd #1540, val=121.16 unseeded default). Branch still CONFLICTING; rebase + seeded rerun pending — student has not yet picked up advisor feedback (likely GraphQL-rate-limited polling).
4. **Architecture / capacity.** hidden256 (tanjiro #1575) in flight — first fair capacity test at merged recipe.
5. **Positional encoding.** Global pos norm (thorfinn #1576) — sent back for seeded rerun.
6. **Across-seed variance.** Alphonse #1685 (seed=7) — calibrates σ for the new recipe.
7. **Robust loss alternatives.** Log-cosh (fern #1635) — orthogonal alt to Huber.
8. **OOD regularization.** wd5e-4 closed as dead end; gradient clipping assigned to frieren as more principled regularizer.

## Round-2 leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **T_max=15 + merged (nezuko #1542)** | **114.81** | **104.68** | merged + T_max=15 (unseeded, surf_weight=20→10 via merge) | **MERGED — CURRENT BEST** |
| Huber loss (edward #1374) | 112.06 | 107.52 (3-split) | default + Huber (unseeded) | Branch now MERGEABLE; awaiting seeded rerun on merged HEAD |
| Seeded baseline (alphonse #1577) | 116.43 | 108.87 | merged + seed=42 | MERGED |
| EMA decay=0.999 (askeladd #1540) | 121.16 | 108.69 | default + EMA (unseeded) | Branch CONFLICTING; rebase + seeded rerun pending |
| Global pos norm (thorfinn #1576) | 123.56 | 109.71 | merged + global_norm (unseeded) | Sent back |
| Default + scoring-fix (#1512) | 123.99 | 110.97 | default only | Superseded |
| wd5e-4 (frieren #1394) | 135.35 | NaN | default + wd5e-4 (unseeded, pre-merge) | **CLOSED — regression** |

## Active student assignments

### Priority: rebase + seeded rerun on merged recipe
- **PR #1374 — `huber-loss` (edward)** — **WIP (rebase)** — STRONGEST LEVER val=112.06 unseeded; branch now MERGEABLE against current HEAD but result is on OLD config; seeded rerun on new HEAD (with T_max=15) is the critical next merge.
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — Branch still CONFLICTING; advisor rebase request from 20:56 UTC, no commits since 20:53 — likely GraphQL-rate-limited polling.

### Round-2 in flight
- **PR #1575 — `hidden256-bf16` (tanjiro)** — **WIP** — capacity test
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — **WIP (sent back)** — seeded rerun required
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP** — log-cosh alt to Huber
- **PR #1685 — `seed7-variance` (alphonse)** — **WIP** — cross-seed σ check
- **PR #1695 — `tmax-18` (nezuko)** — **WIP** — follow-up to #1542, T_max=15→18 to match achievable epoch count
- **PR #1696 — `grad-clip-1.0` (frieren)** — **WIP** — gradient clipping max_norm=1.0 as principled regularizer

## Closed / dead ends
- wd5e-4 (#1394): regression on val_avg (+9.2%); mixed per-split signal not net positive
- surf_weight=20 on merged recipe (#1570): regression confirmed; rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound (superseded by bf16)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)

## Highest-priority stacking target

**Merged recipe + T_max=15 + Huber + EMA (all seeded)**

After #1542 (T_max=15) merge, the gate is edward's Huber rebase. Expected to land sub-110 val_avg.
