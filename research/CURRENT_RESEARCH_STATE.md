# SENPAI Research State

- **As of:** 2026-05-13 10:15 (MERGED #2012 edward loss-beta-0-5 val=66.32 NEW BEST; CLOSED #2089 fern wd-2e-4 flat; SENT BACK #2125 tanjiro β2=0.95 for rerun on new HEAD; assigned edward #2162 t-max-20-bs1, fern #2164 loss-beta-0-25-bs1, tanjiro #2125 β2 rerun; 15 effective merges)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=66.32 / test=59.68 (edward #2012 loss-beta-0-5 at bs=1). This is the 15th merge — batch_size=1 + smooth_l1_loss(beta=0.5).

**Key diagnostics:**
- Val still actively descending at epoch 21 (FINAL). T_max=17 schedule ends at epoch 18; epochs 19–21 are in cosine restart upswing (lr rising). Model still improving — **undertrained**.
- rc split (val=81.07) is the hardest OOD split; flat to beta change (−0.04). Other splits all benefited −4 to −7 val pts.
- Sub-65 val milestone is ~1.3 val pts away.

**Sub-65 val is the next milestone.** T_max alignment (edward #2162) and beta bracket below 0.5 (fern #2164) are the highest-confidence next steps.

## Merged recipe (current advisor base — 15 effective merges)

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
12. **#1812** (lr-warmup-1ep) — val=82.56
13. **#1972** (batch_size=4→2) — val=76.24
14. **#2036** (batch_size=2→1) — val=70.30
15. **#2012** (smooth_l1 beta=1.0→0.5) — val=66.32 **CURRENT BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=1, SequentialLR(LinearLR(1ep warmup, 5e-6→5e-4) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), loss=F.smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 1.0 optimum.
- **surf_weight axis CLOSED**: 5 optimum.
- **depth axis CLOSED**: 5 optimum.
- **AdamW β1 axis CLOSED**: 0.9 optimum.
- **AdamW β2 axis SEMI-CLOSED**: 0.999 best so far; β2=0.95 val-improves OLD baseline but test-regresses; rerunning on new HEAD (beta=0.5 stack). β2=0.99 also in rerun arm.
- **slice_num axis FULLY CLOSED**: 32 ≈ 64 (tie) ≪ 128 (worse). 64 confirmed optimum.
- **lr lower-bound CLOSED**: 3e-4 dominated. 5e-4 optimum lower-side.
- **loss shape axis OPEN**: Huber → beta=1.0 → beta=0.5 (current best). beta=0.25 (fern #2164) and T_max alignment (edward #2162) testing.
- **eta_min axis CLOSED**: 5e-5 optimum.
- **ref axis CLOSED**: ref=8 optimum.
- **mlp_ratio upper CLOSED**: mlp_ratio=4 regresses. mlp_ratio=1 (frieren #1992) still WIP.
- **wd axis FULLY CLOSED**: wd=1e-4 optimum (1e-5 worse; 2e-4 ≈ tie within noise — both directions closed).
- **warmup length CLOSED**: 1-epoch optimum.
- **n_head axis FULLY CLOSED**: n_head=8 (+16.7%) and n_head=2 (+1.9%) both worse. n_head=4 unimodal optimum.
- **CAWR schedules CLOSED** for 30-min budget: catastrophic (#1990 val=100.46).

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0.
2. **Loss weighting — MERGED.** surf_weight=5.
3. **LR schedule alignment — MERGED.** T_max=18.
4. **LR floor — MERGED.** eta_min=5e-5.
5. **LR warmup — MERGED.** 1-epoch warmup (#1812) — val=82.56.
6. **Batch size — MERGED.** batch_size=1 (#2036) — val=70.30 (−7.78%). bs axis fully closed at 1.
7. **Loss beta reduction — MERGED.** beta=0.5 (#2012) — val=66.32 (−5.66%). Loss-beta axis open below 0.5.
8. **T_max alignment for bs=1.** Edward #2162 — T_max=17→20 to match ~21 epochs/30min at bs=1.
9. **Loss beta bracket below 0.5.** Fern #2164 — beta=0.5→0.25. Next step in L1 approach.
10. **AdamW β2 bracket.** Tanjiro #2125 (rerun) — β2=0.95 and β2=0.99 on new HEAD with beta=0.5.
11. **EMA weight averaging.** Askeladd #1540. Actively training on bs=1 baseline.
12. **LR tuning at bs=1.** Alphonse #2106 (lr-4e-4-bs1) — WIP.
13. **FFN capacity.** Frieren #1992 (mlp-ratio-1) — WIP rerun.
14. **OneCycleLR schedule.** Nezuko #2014 — WIP rerun.
15. **LR upper bracket at bs=1.** Thorfinn #1968 (lr-7e-4 rerun) — WIP.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **loss-beta-0-5 + bs=1 (edward #2012)** | **66.32** | **59.68** | **MERGED — CURRENT BEST** |
| batch-size-1 (alphonse #2036) | 70.30 | 61.39 | MERGED → superseded |
| adamw-beta2-0-95 (tanjiro #2125) | 69.74 | 62.37 | SENT BACK — val improved OLD baseline, test regressed; rerun on new HEAD |
| batch-size-2 (alphonse #1972) | 76.24 | 66.85 | MERGED → superseded |
| loss-beta-0-5 on bs=4 (edward #2012 first run) | 81.21 | 72.52 | SENT BACK — beat old, not new baseline |
| mlp-ratio-1 (frieren #1992) | 81.91 | 73.12 | SENT BACK — beat old, not new baseline |
| lr-warmup-1ep (thorfinn #1812) | 82.56 | 74.13 | MERGED → superseded |
| lr-7e-4 (thorfinn #1968) | 79.77 | 72.06 | SENT BACK — beat old, not new baseline |

## Active student assignments (all 8)

### Schedule alignment (undertrained model — highest priority)
- **PR #2162 — `t-max-20-bs1` (edward)** — **WIP (new)** — Align T_max with actual bs=1 budget (~21 ep/30min). One-line change: T_max=17→20. Descending epoch 21 result is the clearest signal in the programme.

### Loss beta bracket
- **PR #2164 — `loss-beta-0-25-bs1` (fern)** — **WIP (new)** — Narrow beta=0.5→0.25. Continues L1 approach. Closes axis below current optimum.

### AdamW β2 bracket
- **PR #2125 — `adamw-beta2-0-95` (tanjiro)** — **WIP (sent back)** — Rerun on new HEAD (beta=0.5). Two arms: β2=0.95 and β2=0.99.

### LR tuning
- **PR #2106 — `lr-4e-4-bs1` (alphonse)** — **WIP** — LR downward bracket at bs=1.
- **PR #1968 — `lr-7e-4` (thorfinn)** — **WIP** — LR upper bracket at bs=1 (rerunning).

### Schedule shape
- **PR #2014 — `onecycle-lr` (nezuko)** — **WIP** — Evaluate vs bs=1 baseline.

### EMA + capacity
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Training at ~99% GPU. Highest priority stacking candidate.
- **PR #1992 — `mlp-ratio-1` (frieren)** — **WIP** — Evaluate vs bs=1 baseline.

## Closed / dead ends (complete list)
- max_norm: 0.5/1.0/3.0 → 1.0
- surf_weight: 3/5/10/20 → 5
- depth: 5/6/7 → 5
- AdamW β1: 0.9
- slice_num: 64 (32≈64≪128)
- lr lower: 3e-4 dominated
- log-cosh loss
- hidden192/256
- ref: 8 optimum (16 worse)
- mlp_ratio upper: 4 worse
- eta_min: 5e-5 optimum
- wd: 1e-4 optimum (1e-5 worse; 2e-4 ≈ tie — FULLY CLOSED)
- CAWR schedules in 30-min budget: catastrophic
- warmup length: 1ep optimum
- n_head: 4 unimodal optimum (FULL BRACKET)

## Highest-priority stacking targets

1. **T_max alignment (edward #2162)** — Model was still descending at epoch 21 (final). T_max=20 aligns schedule minimum with actual budget. Highest confidence immediate win.
2. **EMA (askeladd #1540)** — Actively training. Independent mechanism from all current merges. Expected sub-65.
3. **Loss beta bracket (fern #2164)** — beta=0.25 could improve further if L1 approach continues.

## Next frontier after current round

- Sub-65 val achievable via T_max alignment + EMA stacking
- Loss beta axis: bracket below 0.5 (beta=0.25, pure L1) to find minimum
- Architectural VRAM exploitation: VRAM at 8.5/98GB — n_hidden=192 could be tested
- β2 axis: clarify whether intermediate β2 (0.97, 0.99) helps at bs=1
- Extended schedule: if T_max=20 wins, can we push further with modified budget
