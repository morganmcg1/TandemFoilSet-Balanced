# SENPAI Research State

- **Last updated:** 2026-05-13 ~00:05 (merged #1684 T_max-aligned-14 at −11.3% → new baseline 84.562; assigned frieren #1707 per-sample-loss-norm; notified all in-flight students to use --epochs 14)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 84.562`** — PR #1684 (T_max alignment --epochs 14), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=103.231, val_rc=95.256, val_cruise=60.589, val_re_rand=79.170.  
Test: test_avg=74.947 (test_single=91.146, test_rc=87.193, test_cruise=50.942, test_re_rand=70.508).

> ⚠️ All in-flight PRs except #1707 were assigned before #1684 merged — they're running with --epochs 20 (T_max=20, schedule-misaligned). Their results will appear ~11% worse than if run with --epochs 14. Each has been notified to use --epochs 14 on re-runs. If a result beats 84.562, merge. If it loses by <12%, consider re-run with --epochs 14 before closing.

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- **84.562** (#1684 T_max=14 alignment) — **current**

See `BASELINE.md` for full per-split details.

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1663 | alphonse | `smooth-l1-full-stack` | Full-stack validation run | --epochs 20 ⚠️ | WIP — notified to re-run with --epochs 14 |
| #1682 | tanjiro | `pure-l1-loss` | Pure F.l1_loss (remove β quadratic regime) | --epochs 20 ⚠️ | WIP — notified to re-run with --epochs 14 |
| #1707 | frieren | `per-sample-loss-norm` | Per-sample loss norm by pressure std | **--epochs 14** ✓ | WIP — just assigned |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — notified to re-run with --epochs 14 |
| #1658 | askeladd | `swa-ep10-14` | SWA averaging epochs 10–14 | --epochs 20 ⚠️ | WIP — notified; SWA needs --epochs 14 re-config |
| #1659 | nezuko | `slice-96-stable` | slice_num 64→96 | --epochs 20 ⚠️ | WIP — notified to re-run with --epochs 14 |
| #1435 | thorfinn | `unified-pos-ref8` | Unified pos encoding ref=8 (conflict, stale) | --epochs 20 ⚠️ | WIP — notified rebase x3 + --epochs 14 |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — notified rebase x2 + --epochs 14 |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse (under-trained)
- #1432 tanjiro wall-distance-rebased: +12.8% worse vs current (negative stacking)
- #1597 frieren depth-6-layers: +36% worse vs current (capacity not bottleneck)

## Current research focus

1. **Per-sample loss normalization (frieren #1707)** — fresh loss-axis experiment using the new 84.562 baseline with correct --epochs 14.
2. **Full-stack validation (alphonse #1663)** — confirms combined stack metric. May re-run with --epochs 14.
3. **Pure L1 loss (tanjiro #1682)** — tests β→0 limit; result needed to decide β sweep direction.
4. **Fourier RFF encoding (fern #1657)** — high OOD potential; may need --epochs 14 re-run.
5. **SWA epochs 10–14 (askeladd #1658)** — weight averaging. SWA epochs need re-config for --epochs 14 schedule.
6. **slice_num=96 (nezuko #1659)** — finer attention tokens.
7. **Thorfinn #1435 / Edward #1421** — stale WIP awaiting rebases.

## Key research insights so far

- **Loss shape wins big:** Smooth L1 (β=0.1) → −22.3% from #1418 baseline. L1 regime aligns with MAE eval criterion.
- **Channel weighting [1,1,3]:** −9.5% standalone. Now part of canonical stack.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — the schedule was being truncated at ~37% LR. MUST use --epochs 14 for all future experiments.
- **NaN-skip now canonical:** test_avg/mae_surf_p is clean 4-split.
- **Architecture/capacity axes exhausted:** depth-6 regressed. slice_num=64 optimal at this budget. Width-scaling killed by 30-min cap.
- **Wall-distance negative stacking:** helps in isolation but hurts under CW+warmup regime.

## Next research directions (when new slots open)

1. **β sweep for Smooth L1** — after tanjiro #1682 (pure-L1) result settles the lower bound question.
2. **OneCycleLR schedule** — aggressive LR shaping (peak LR at 30% of training, not 14%).
3. **Data augmentation** — foil geometry mirroring, pressure field transformations.
4. **Gradient accumulation** — larger effective batch within 30-min cap.
5. **Loss: relative pressure error** — normalize error by per-sample pressure range.
6. **Longer warmup test** — 4-epoch warmup vs current 2-epoch.

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by #1684 completing 14/14 in 31.7 min)
- **Canonical schedule: --epochs 14, T_max=14 (or T_max=12 post-warmup if code subtracts warmup)**
- T_max=20 (OLD) → LR at epoch 14 is ~37% of peak — WRONG, DO NOT USE
