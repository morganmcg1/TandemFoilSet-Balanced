# SENPAI Research State

- **As of:** 2026-05-13 08:15 (CLOSED #1993 tanjiro n-head-2 — n_head axis fully closed at 4; SENT BACK #2012 edward loss-beta-0-5 (beat OLD 82.56 not NEW 76.24); askeladd #1540 EMA retraining on new HEAD GPU=99%; assigning tanjiro fresh hypothesis; 13 effective merges)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=76.24 / test=66.85 (alphonse #1972 batch-size-2). Batch_size=2 is the 13th merge — it doubles optimizer steps/epoch at the same wall-clock, providing 2x gradient refinement per unit time without increased compute cost.

**Key diagnostic:** val still actively descending at final epoch (ep17→18 delta=2.44 in prior warmup run; bs=2 got 19 epochs and best at ep18). Model is undertrained. Batch_size=2 directly addresses this by 2x more parameter updates per wall-clock minute.

**Sub-70 val is the next milestone.** EMA (askeladd #1540) expected to push further. VRAM headroom at bs=2 (17GB of 98GB) enables bigger architectures too.

## Merged recipe (current advisor base — 13 effective merges)

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
13. **#1972** (batch_size=4→2) — val=76.24 **CURRENT BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, seed=42, batch_size=2, SequentialLR(LinearLR(1ep warmup, 5e-6→5e-4) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 1.0 optimum.
- **surf_weight axis CLOSED**: 5 optimum.
- **depth axis CLOSED**: 5 optimum.
- **AdamW β1 axis CLOSED**: 0.9 optimum.
- **AdamW β2 axis CLOSED**: 0.999 optimum.
- **slice_num axis CLOSED**: 64 optimum.
- **lr lower-bound CLOSED**: 3e-4 dominated. 5e-4 optimum lower-side.
- **loss shape axis CLOSED**: log-cosh regression. Huber beta=1.0 merged. beta=0.5 (edward #2012) testing sub-axis.
- **eta_min axis CLOSED**: 5e-5 optimum (0 → 84.67, 5e-5 → 83.95, 1e-4 → 85.06).
- **ref axis CLOSED**: ref=8 optimum.
- **mlp_ratio upper CLOSED**: mlp_ratio=4 regresses. mlp_ratio=1 (frieren #1992) testing lower.
- **wd axis CLOSED**: wd=1e-4 optimum.
- **warmup length CLOSED**: 1-epoch optimum.
- **n_head axis FULLY CLOSED**: n_head=8 (#1853, val=96.33) and n_head=2 (#1993, val=83.78) both worse. n_head=4 unimodal optimum.

## Themes

1. **Gradient clipping — MERGED.** max_norm=1.0.
2. **Loss weighting — MERGED.** surf_weight=5.
3. **LR schedule alignment — MERGED.** T_max=18.
4. **LR floor — MERGED.** eta_min=5e-5.
5. **LR warmup — MERGED.** 1-epoch warmup (#1812) — val=82.56.
6. **Batch size — MERGED.** batch_size=2 (#1972) — val=76.24 (+7.65% improvement).
7. **EMA weight averaging.** Askeladd #1540. Actively training on new HEAD. Expected sub-70.
8. **LR upper bracket.** Thorfinn #1968 (lr=7e-4, sent back) — rerunning with bs=2 baseline.
9. **Batch size lower bracket.** Alphonse #2036 (batch-size-1) — NEW. Closes bs axis.
10. **LR schedule restart.** Fern #1990 (cawr-t0-9) — WIP on old bs=4 HEAD; needs rebase evaluation.
11. **FFN capacity downward bracket.** Frieren #1992 (mlp-ratio-1) — WIP on old HEAD.
12. **Loss shape sub-axis.** Edward #2012 (loss-beta-0-5) — WIP.
13. **OneCycleLR schedule.** Nezuko #2014 (onecycle-lr, max_lr=8e-4) — WIP.
14. **Attention head lower bracket.** Tanjiro #1993 (n-head-2) — CLOSED val=83.78 worse than 4. n_head axis fully bracketed.

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **batch-size-2 (alphonse #1972)** | **76.24** | **66.85** | **MERGED — CURRENT BEST** |
| loss-beta-0-5 (edward #2012) | 81.21 | 72.52 | SENT BACK — beat old, not new baseline |
| lr-warmup-1ep (thorfinn #1812) | 82.56 | 74.13 | MERGED → superseded |
| lr-7e-4 (thorfinn #1968) | 79.77 | 72.06 | SENT BACK — beat old, not new baseline |
| n-head-2 (tanjiro #1993) | 83.78 | 73.71 | CLOSED — n_head=4 unimodal optimum |
| eta_min=5e-5 (fern #1855) | 83.95 | 74.70 | MERGED → superseded |
| warmup-2ep (edward #1991) | 83.35 | 75.06 | CLOSED — warmup saturates at 1ep |
| T_max=18 (nezuko #1695) | 84.67 | 74.94 | MERGED → superseded |
| eta_min=1e-4 (fern #1901) | 85.06 | 76.41 | CLOSED — eta_min 5e-5 optimum |
| n_head=8 (nezuko #1853) | 96.33 | 86.97 | CLOSED — n_head=4 unimodal optimum |
| EMA (askeladd #1540) | stale | — | Actively training on new HEAD — highest priority |

## Active student assignments (all 8)

### Priority: EMA + stacking on new baseline
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Training at 99% GPU. Results likely on old HEAD. Will evaluate vs new 76.24.

### Batch size bracket
- **PR #2036 — `batch-size-1` (alphonse)** — **WIP (new)** — extreme batch bracket.

### LR + schedule probes (running on new bs=2 HEAD)
- **PR #1968 — `lr-7e-4` (thorfinn)** — **WIP (sent back)** — rerunning with bs=2 + lr=7e-4.
- **PR #1990 — `cawr-t0-9` (fern)** — **WIP** — was on bs=4; evaluate vs new baseline.
- **PR #2014 — `onecycle-lr` (nezuko)** — **WIP** — was on bs=4; evaluate vs new baseline.

### Capacity / architecture / loss probes
- **PR #1992 — `mlp-ratio-1` (frieren)** — **WIP** — was on bs=4; evaluate vs new baseline.
- **PR #2012 — `loss-beta-0-5` (edward)** — **WIP** — was on bs=4; evaluate vs new baseline.
- **PR #2073 — `slice-num-32` (tanjiro)** — **WIP (new)** — lower bracket of slice_num axis on bs=2 baseline.

## Closed / dead ends (complete list)
- max_norm: 0.5/1.0/3.0 → 1.0
- surf_weight: 3/5/10/20 → 5
- depth: 5/6/7 → 5
- AdamW β1: 0.9
- AdamW β2: 0.999
- slice_num: 64
- lr lower: 3e-4 dominated
- log-cosh loss
- hidden192/256
- ref: 8 optimum (16 worse)
- mlp_ratio upper: 4 worse (1 in flight)
- eta_min: 3-pt bracket, 5e-5 optimum
- wd: 1e-4 optimum (1e-5 worse)
- warmup length: 1ep optimum (2ep worse)
- n_head: 4 unimodal optimum (2 worse +1.48%, 8 worse +16.7% — FULL BRACKET)

## Highest-priority stacking target

**EMA (askeladd #1540)** — actively training. On old bs=4 HEAD, results will arrive shortly. Even if on old HEAD, if it shows meaningful val improvement, we'll evaluate whether to send back for rerun on new bs=2 HEAD.

## Next frontier after current round

With val=76.24 (sub-80 milestone achieved!), next targets:
- Sub-70 val with EMA + bs=2 stacking
- LR exploration: lr=7e-4 + bs=2 (thorfinn rerun)
- Schedule: CAWR, OneCycleLR on new baseline
- Capacity: VRAM freed by bs=2 → could test n_hidden=192 again (wall-clock concern may ease)
