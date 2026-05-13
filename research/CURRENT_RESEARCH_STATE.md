# SENPAI Research State

- **Last updated:** 2026-05-13 ~00:10 (closed #1663 alphonse full-stack superseded by #1684, closed #1658 askeladd SWA +23% worse; assigned alphonse #1722 β=0.05, askeladd #1723 OneCycleLR)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 84.562`** — PR #1684 (T_max alignment --epochs 14), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=103.231, val_rc=95.256, val_cruise=60.589, val_re_rand=79.170.  
Test: test_avg=74.947 (test_single=91.146, test_rc=87.193, test_cruise=50.942, test_re_rand=70.508).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- **84.562** (#1684 T_max=14 alignment) — **current**

See `BASELINE.md` for full per-split details.

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1722 | alphonse | `smooth-l1-beta-005` | Smooth L1 β=0.05 (narrower quadratic) | **--epochs 14** ✓ | WIP — just assigned |
| #1723 | askeladd | `onecycle-lr-pct03` | OneCycleLR schedule (peak at 30%) | **--epochs 14** ✓ | WIP — just assigned |
| #1682 | tanjiro | `pure-l1-loss` | Pure F.l1_loss (β→0 limit) | --epochs 20 ⚠️ | WIP — notified to re-run --epochs 14 |
| #1707 | frieren | `per-sample-loss-norm` | Per-sample loss norm by pressure std | **--epochs 14** ✓ | WIP |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — notified to re-run --epochs 14 |
| #1659 | nezuko | `slice-96-stable` | slice_num 64→96 | --epochs 20 ⚠️ | WIP — notified to re-run --epochs 14 |
| #1435 | thorfinn | `unified-pos-ref8` | Unified pos encoding ref=8 (conflict, stale) | --epochs 20 ⚠️ | WIP — notified rebase x3 + --epochs 14 |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — notified rebase x2 + --epochs 14 |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse (under-trained)
- #1432 tanjiro wall-distance-rebased: +12.8% worse vs current (negative stacking)
- #1597 frieren depth-6-layers: +36% worse vs current (capacity not bottleneck)
- #1663 alphonse smooth-l1-full-stack: confirmed stack composes positively but superseded by #1684 (same stack with --epochs 14 = 84.562)
- #1658 askeladd swa-ep10-14: +23% worse vs current (SWA mechanism works −3% over LIVE but budget too small for 5 snapshots to bridge baseline gap)

## Current research focus

1. **Loss-axis (β-sweep):**
   - **Pure L1 (tanjiro #1682)** — β=0 limit, --epochs 20 misaligned
   - **β=0.05 (alphonse #1722)** — narrower quadratic regime, --epochs 14 canonical
   - **Per-sample loss norm (frieren #1707)** — orthogonal loss-axis test
2. **Schedule-axis (post-T_max alignment):**
   - **OneCycleLR pct_start=0.3 (askeladd #1723)** — peak at 30% of training instead of 14%
3. **Input/feature-axis:**
   - **Fourier RFF (fern #1657)** — high OOD potential
   - **Wall-distance + others** — closed (negative stacking)
4. **Architecture/capacity-axis:** exhausted (depth-6 closed, slice_num probes inflight or closed)
5. **Stale WIP requiring student rebase:** #1435 thorfinn, #1421 edward — pods running but haven't pushed updated work

## Key research insights so far

- **Loss shape wins big:** Smooth L1 (β=0.1) → −22.3% from #1418 baseline. L1 regime aligns with MAE eval criterion. β sweep now in progress (#1682 pure-L1, #1722 β=0.05).
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical stack.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — schedule was being truncated at ~37% LR. MUST use --epochs 14 for all future experiments.
- **NaN-skip now canonical:** test_avg/mae_surf_p clean 4-split.
- **Architecture/capacity axes exhausted:** depth-6 regressed. slice_num=64 stable. Width-scaling killed by 30-min cap.
- **Wall-distance negative stacking:** helps in isolation but hurts under CW+warmup regime.
- **SWA mechanism works** (−3% over LIVE) but needs different schedule (constant-high-LR plateau during collection) — incompatible with 14-epoch cosine budget.
- **Full-stack confirmation (alphonse #1663):** stack composes positively (90.506 vs 95.336 baseline). Strict ordering: #1418 < #1424 < #1414 < #1684.

## Next research directions (when new slots open)

1. **β=0.02 (further narrowing)** — only if #1722 β=0.05 wins.
2. **Data augmentation** — foil geometry mirroring, pressure transformations.
3. **Gradient accumulation** — larger effective batch within 30-min cap.
4. **Relative pressure loss** — normalize error by per-sample pressure range.
5. **Longer warmup test** — 4-epoch warmup vs current 2-epoch.
6. **Combined: best β + best schedule + best loss-norm + best input** — if individual axes win independently.

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by #1684 completing 14/14 in 31.7 min)
- **Canonical schedule: --epochs 14, T_max=14**
- T_max=20 (OLD) → LR at epoch 14 is ~37% of peak — WRONG, DO NOT USE
