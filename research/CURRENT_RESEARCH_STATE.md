# SENPAI Research State

- **As of:** 2026-05-12 23:05 (seeded baseline merged: val=116.43 new best; surf_weight=20 rolled back; 3 rebases in flight)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **Key finding:** Seeded baseline (alphonse #1577) on the 3-merge recipe (unified_pos + bf16 + scoring-fix, surf_weight=10) achieves val=116.43 — new best on branch. Robust loss (Huber) remains the strongest single lever at val=112.06 on default config; rebase + seeded rerun on merged recipe is the critical gate.

## Merged recipe (current advisor base)

Six PRs merged into the advisor branch (surf_weight=20 reverted):

1. **#1512** (`data/scoring.py` NaN fix) — baseline = 123.99 (default config)
2. **#1513** (bf16 autocast) — 24% per-epoch speedup, ~18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8) — best cruise OOD, val cruise=91.85
4. **#1369** (surf_weight=10→20) — EFFECTIVELY REVERTED via #1577 (confirmed regression: #1570 val=127.86)
5. **#1577** (seed=42 + surf_weight=10 rollback) — NEW BEST val=116.43; deterministic seeding now standard

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=50), AdamW`

**⚠️ All new experiments MUST use seed=42 for comparability.** Unseeded results cannot be compared against the 116.43 baseline.

## Themes

1. **Robust loss functions — HIGHEST PRIORITY.** Huber loss (#1374, edward, val=112.06 unseeded default, -9.6%) is the strongest single-lever result. After seeding, the stacked target is: seeded merged recipe + Huber → expected sub-112 val. Edward's rebase (#1374) is the critical gate. Log-cosh alternative is in flight with fern (#1635).
2. **LR schedule mismatch.** T_max=15 (nezuko #1542) confirmed schedule fix (val=121.83 unseeded default). Rebase + seeded rerun in progress.
3. **EMA weight averaging.** EMA decay=0.999 (askeladd #1540, val=121.16 unseeded default). Rebase completed (branch now MERGEABLE), awaiting seeded training rerun.
4. **Weight decay for OOD.** wd5e-4 (frieren #1394) in flight — tests whether stronger weight decay helps camber-OOD splits.
5. **Positional encoding.** Global pos norm (thorfinn #1576) shows test improvement (-7.41 vs prior unified_pos) but needs seeded rerun to confirm. Sent back.
6. **Across-seed variance.** Unknown for the new recipe. seed=7 rerun assigned to alphonse to calibrate σ.

## Round-2 leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Config | Status |
|---|---|---|---|---|
| **Huber loss (edward #1374)** | **112.06** | 107.52 (3-split) | default + Huber (unseeded) | Rebase in flight |
| **Seeded baseline (alphonse #1577)** | **116.43** | **108.87** | merged recipe + seed=42 | **MERGED — CURRENT BEST** |
| EMA decay=0.999 (askeladd #1540) | 121.16 | 108.69 | default + EMA (unseeded) | Rebase done, seeded rerun pending |
| Cosine T_max=15 (nezuko #1542) | 121.83 | 110.50 | default + T_max=15 (unseeded) | Rebase in flight |
| Global pos norm (thorfinn #1576) | 123.56 | 109.71 | merged + global_norm (unseeded) | Sent back for seeded rerun |
| Default + scoring-fix (#1512) | 123.99 | 110.97 | default only | Superseded |
| Merged recipe + surf_weight=20 (#1570) | 127.86 | 119.28 | regression | Closed |

## Active student assignments

### Priority: rebase + seeded rerun on merged recipe
- **PR #1374 — `huber-loss` (edward)** — **WIP (rebase)** — STRONGEST LEVER val=112.06 unseeded; seeded rerun highest priority.
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Branch rebased (MERGEABLE); awaiting seeded training rerun on merged recipe.
- **PR #1542 — `cosine-trunc-t15` (nezuko)** — **WIP (rebase)** — val=121.83 unseeded; rebase + seeded rerun in progress.

### Round-2 in flight
- **PR #1394 — `wd5e-4` (frieren)** — **WIP** — weight decay OOD test, training
- **PR #1575 — `hidden256-bf16` (tanjiro)** — **WIP** — capacity test at merged recipe
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — **WIP (sent back)** — seeded rerun required
- **PR #1635 — `log-cosh-loss` (fern)** — **WIP** — log-cosh alt to Huber
- **PR TBD — `seed7-variance` (alphonse)** — **WIP** — cross-seed σ check on seeded recipe (seed=7)

## Closed / dead ends
- surf_weight=20 on merged recipe (#1570): regression confirmed (+3% val, +7% test); rolled back
- surf_weight=3x (#1533): +25% regression
- hidden192 (#1406): wall-clock-bound (superseded by bf16)
- lr1e3-warmup (#1376): warmup consumed budget (19% regression)

## Highest-priority stacking target

**Merged recipe + Huber + T_max=15 + EMA (all seeded, seed=42)**

Expected to push val well below 110. Edward's Huber seeded rerun is the gate.
