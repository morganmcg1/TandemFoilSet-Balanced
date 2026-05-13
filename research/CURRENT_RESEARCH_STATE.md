# SENPAI Research State

- **Last updated:** 2026-05-13 ~01:45 (closed #1707 frieren per-sample-loss-norm +12.2% dead end; closed #1659 nezuko slice-96 +27.9% dead end; assigned frieren #1776 warmup-4-epochs, nezuko #1777 asinh-pressure-gain-1)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 83.230`** — PR #1682 (pure-L1 loss + T_max=14 alignment), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=99.310, val_rc=95.316, val_cruise=61.818, val_re_rand=76.477.  
Test: test_avg=73.513 (test_single=88.714, test_rc=83.649, test_cruise=50.535, test_re_rand=71.156).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- **83.230** (#1682 pure-L1 loss) — **current**

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 (in normalized space)
- AdamW lr=7e-4, 2-epoch linear warmup, CosineAnnealingLR(T_max=14), grad_clip=1.0
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1723 | askeladd | `onecycle-lr-pct03` | OneCycleLR (peak at 30%) — REBASE NEEDED | --epochs 14 ✓ | SENT BACK — rebase onto pure-L1 HEAD + rerun |
| #1744 | tanjiro | `grad-accum-4` | Gradient accumulation (effective batch 4→16) | **--epochs 14** ✓ | WIP — in progress |
| #1722 | alphonse | `smooth-l1-beta-005` | Smooth L1 β=0.05 (stale, pure-L1 won) | **--epochs 14** ✓ | WIP — on Smooth L1 (stale); result still informative for β direction |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — notified to re-run --epochs 14 + pure-L1 base |
| #1776 | frieren | `warmup-4-epochs` | Longer warmup: 2→4 epochs (LR peak at 29% vs 14%) | **--epochs 14** ✓ | WIP — just assigned |
| #1777 | nezuko | `asinh-pressure-gain-1` | Asinh value compression on pressure target (heavy-tail handling) | **--epochs 14** ✓ | WIP — just assigned |
| #1435 | thorfinn | `unified-pos-ref8` | Unified pos encoding ref=8 (stale, rebase x4) | --epochs 20 ⚠️ | WIP — needs rebase |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — needs rebase |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse (under-trained)
- #1432 tanjiro wall-distance-rebased: +12.8% worse vs current (negative stacking)
- #1597 frieren depth-6-layers: +36% worse vs current (capacity not bottleneck)
- #1663 alphonse smooth-l1-full-stack: confirmed stack, superseded by #1684 canonical
- #1658 askeladd swa-ep10-14: +23% worse vs current (mechanism works, budget too small)
- #1707 frieren per-sample-loss-norm: +14.0% worse — clamp threshold pathology (min ~5e-4 << clamp 1e-6)
- #1659 nezuko slice-96-stable: +30% worse — token-capacity dilution, capacity axis exhausted

## Current research focus

1. **Schedule axis (highest EV):**
   - **OneCycleLR rebase (askeladd #1723)** — sent back; rebase onto pure-L1 HEAD, re-run. Expected stacked ~82.0.
   - **Gradient accumulation (tanjiro #1744)** — effective batch 4→16, orthogonal probe.
   - **4-epoch warmup (frieren #1776)** — push LR peak from epoch 2/14 (14%) to epoch 4/14 (29%). Companion question to askeladd's pct_start probe.

2. **Value-level / loss tail axis:**
   - **Asinh pressure compression (nezuko #1777)** — compress heavy-tailed pressure targets before L1 loss; decompress for metric reporting. Tests whether peak suction pressure at high Re is hurting gradient signal on bulk surface.

3. **Loss axis (stale, pure-L1 is now canonical):**
   - **Alphonse #1722 β=0.05** — running on Smooth L1; result tells us if β=0.05 < β=0.1. Informative for β direction but won't match canonical pure-L1 config.

4. **Input/feature axis:**
   - **Fern #1657 RFF** — high OOD potential; --epochs 20 stale. Needs rerun at --epochs 14 + pure-L1.

5. **Stale WIP:** #1435 thorfinn (x4 rebase notifications), #1421 edward (x3).

## Key research insights so far

- **Loss shape wins:** pure-L1 → tightest surrogate for MAE criterion. β ladder: MSE > Smooth L1 β=0.1 > pure-L1. Monotone improvement as β → 0.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE.
- **T_max alignment (--epochs 14):** −11.3% — critical! All future work must use --epochs 14.
- **OneCycleLR:** −1.38% additional on top of cosine/T_max alignment. Orthogonal to loss; rebase+restack with pure-L1 in progress.
- **Architecture/capacity axes: EXHAUSTED.** depth-6, slice_num=96/128, hidden=192, mlp_ratio=4 all regressive on this dataset at this epoch budget. Do not assign more capacity experiments.
- **Per-sample loss weighting:** naive std-normalization pathological (2000× weight ratio). Bounded clamp (min=0.5, max=2.0) is the corrected version but on pure-L1 base may be lower priority.
- **SWA/EMA:** don't fit 14-epoch budget cleanly.
- **Wall-distance:** negative stacking under new regime.

## Next research directions (when new slots open)

1. **OneCycleLR + pure-L1 stacked confirm** (askeladd rebase/rerun — in progress)
2. **4-epoch warmup** (frieren #1776 — in progress)
3. **Asinh pressure target** (nezuko #1777 — in progress) — if wins, sweep ASINH_GAIN ∈ {0.5, 2.0}
4. **pct_start sweep** — OneCycleLR pct_start ∈ {0.2, 0.4} if askeladd's base result confirms
5. **Data augmentation** — foil geometry mirroring (y-coord flip, Uy sign flip, AoA sign flip) or Gaussian input noise
6. **Relative pressure loss** — per-sample range normalization (range >> std; naturally bounded above zero)
7. **EMA on pure-L1 base** — #1517 was neutral on Smooth L1; worth retesting on pure-L1 base

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by #1684 completing 14/14 in 31.7 min)
- **Canonical schedule: --epochs 14, T_max=14 (cosine) or OneCycleLR(epochs=14)**
