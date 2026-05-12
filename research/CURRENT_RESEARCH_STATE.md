# SENPAI Research State

- **As of:** 2026-05-12 22:05 (Huber loss val=112.06 is new best by large margin; surf_weight=20 stack confirmed dead end; 3 rebases in flight; log-cosh assigned to fern; frieren training ~22:30)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **Round 2 key finding:** Huber loss (-9.6% vs baseline, val=112.06) is the strongest lever found; robust loss functions are the primary research direction. Three rebases in flight to confirm levers on merged recipe.

## Merged recipe (current advisor base)

Four PRs merged into the advisor branch:
1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD, val cruise=91.85
4. **#1369** (surf_weight=10→20) — merged within noise; NOW KNOWN to be regression on merged recipe

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=20.0, batch_size=4, CosineAnnealingLR(T_max=50), AdamW`

**⚠️ Merged recipe may be regression vs default:** fern #1570 showed val=127.86 on merged recipe vs 123.99 default. surf_weight=20 is likely the culprit (confirmed dead end in #1570 and #1533). Alphonse #1577 (seeded baseline) will confirm. May consider rolling back #1369 if confirmed.

## Themes

1. **Robust loss functions — HIGHEST PRIORITY.** Huber loss (#1374, edward, val=112.06, -9.6%) is the strongest single-lever result. Default MSE over-weights high-Re samples; tail-capping losses redistribute gradient budget. Next: log-cosh alternative (fern #1635), Huber beta sweep, Huber+EMA, Huber+T_max=15 stacks.
2. **LR schedule mismatch.** T_max=15 (nezuko #1542) confirmed schedule fix (val=121.83 on default). Rebase + rerun on merged recipe in progress. Target: T_max=15 + Huber stack.
3. **EMA weight averaging.** EMA (askeladd #1540, val=121.16 on default) is confirmed second-best lever. Rebase + rerun on merged recipe in progress. Target: EMA + Huber + T_max=15 full stack.
4. **Weight decay for OOD.** wd5e-4 (frieren #1394) now training post-rate-limit (~22:30 finish). Independent from loss/schedule levers.
5. **Merged recipe regression concern.** #1570 showed the merged recipe (surf_weight=20) doesn't stack with unified_pos+bf16. Mapped: surf_weight axis has no benefit beyond default 10.

## Round-2 leaderboard (on default config, unseeded)

| Lever | val_avg | test_avg | Config |
|---|---|---|---|
| **Huber loss (edward #1374)** | **112.06** | 107.52 (3-split) | default + Huber |
| EMA decay=0.999 (askeladd #1540) | 121.16 | 108.69 | default + EMA |
| Cosine T_max=15 (nezuko #1542) | 121.83 | 110.50 | default + T_max=15 |
| Default + scoring-fix (fern #1512) | 123.99 | 110.97 | default only |
| Merged recipe + surf_weight=20 (fern #1570) | 127.86 | 119.28 | merged = regression |

## Active student assignments

### Priority: rebase + rerun on merged recipe
- **PR #1374 — `huber-loss` (edward)** — **WIP (rebase)** — STRONGEST LEVER val=112.06; rebase train.py onto merged recipe, rerun, submit. This is highest priority merge candidate.
- **PR #1540 — `ema-weights` (askeladd)** — **WIP (rebase)** — val=121.16; rebase + rerun on merged recipe (94GB VRAM training in flight earlier, result pending).
- **PR #1542 — `cosine-trunc-t15` (nezuko)** — **WIP (rebase)** — val=121.83; rebase onto merged recipe + rerun.

### Round-2 in flight (no rebase needed)
- **PR #1394 — `wd5e-4` (frieren)** — **WIP** — training in progress ~22:30 finish, GPU 100%
- **PR #1577 — `seed42-baseline` (alphonse)** — **WIP** — seeded baseline for merged recipe
- **PR #1575 — `hidden256-bf16` (tanjiro)** — **WIP** — first fair capacity test at merged recipe
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — **WIP** — corpus-level pos normalization

### New assignments (round 2/3)
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP** — log-cosh as alt to Huber (no beta hyperparameter); orthogonal to edward's rerun

## Closed / dead ends
- surf_weight=20 on merged recipe (#1570): regression confirmed (+3% val, +7% test)
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound (superseded by bf16)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)

## Highest-priority stacking target

**Merged recipe + Huber + T_max=15 + EMA**

All three are orthogonal mechanisms (loss form / lr schedule / weight averaging) and all showed strong individual results. The combined model could push val_avg well below 110. Edward's rerun on merged recipe is the critical gate.
