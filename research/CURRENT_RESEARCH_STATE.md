# SENPAI Research State

- **Last updated:** 2026-05-13 ~06:10 (closed #1941 nezuko asinh-all-channels +2.75% dead end; assigned nezuko #1970 drop-path-0.1)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 74.2082`** — PR #1895 (lr=1.5e-3 + asinh+warmup-4 stack), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=83.733, val_rc=91.690, val_cruise=50.392, val_re_rand=71.018.  
Test: test_avg=65.1123 (test_single=75.443, test_rc=82.056, test_cruise=41.545, test_re_rand=61.405).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- 80.7014 (#1776 4-epoch warmup)
- 79.8623 (#1777 asinh pressure compression GAIN=1.0)
- 77.1419 (#1814 lr=1e-3 + asinh super-additive stack)
- **74.2082** (#1895 lr=1.5e-3 ceiling probe) — **current**

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 (on asinh-compressed targets in normalized space)
- **Asinh pressure compression**: `compress_pressure(y_norm)` / `decompress_pressure(y_c)` with ASINH_GAIN=1.0 (pressure channel only)
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10), grad_clip=1.0
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1942 | alphonse | `lr-2e-3` | LR ceiling: 1.5e-3→2e-3 (ceiling still open per #1895) | **--epochs 14** ✓ | WIP — just assigned |
| #1970 | nezuko | `drop-path-0.1` | Stochastic Depth (DropPath) on Transolver residual branches | **--epochs 14** ✓ | WIP — just assigned |
| #1813 | frieren | `warmup-5-epochs` | Warmup 4→5 epochs (bracket above winner) | **--epochs 14** ✓ | WIP — in progress (notified of 77.14 target; now needs 74.21) |
| #1815 | askeladd | `node-dropout-0.9` | Mesh node dropout 0.9 (rebasing onto asinh base) | **--epochs 14** ✓ | WIP — rebasing |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (smooth-near-zero L1) | **--epochs 14** ✓ | WIP — in progress |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (L2 regularization) | **--epochs 14** ✓ | WIP — in progress |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — stale |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — stale |

### Closed as dead ends (this round)
- #1911 nezuko warmup-3-epochs: +1.56% vs 77.1419 (3-epoch ramp too steep at lr=1e-3)
- #1941 nezuko asinh-all-channels: +2.75% vs 74.2082 (mechanism pressure-specific; velocity channels are Gaussian, not heavy-tailed)
- #1835 nezuko asinh-gain-0.5: +1.96% vs 79.8623 (asymmetric axis — GAIN<1 erodes bulk-redistribution)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse
- #1432 tanjiro wall-distance-rebased: +12.8% worse
- #1597 frieren depth-6-layers: +36% worse
- #1663 alphonse smooth-l1-full-stack: superseded
- #1658 askeladd swa-ep10-14: +23% worse (budget too small)
- #1707 frieren per-sample-loss-norm: +14.0% worse (clamp pathology)
- #1659 nezuko slice-96-stable: +30% worse (capacity axis exhausted)
- #1744 tanjiro grad-accum-4: +14.9% worse (update-count starved, no LR scaling)
- #1723 askeladd OneCycleLR rebased: tied/+0.37% (schedule shape saturated vs pure-L1)
- #1722 alphonse β=0.05: +0.47% worse (β axis complete: β=0 pure-L1 is minimum)
- #1435 thorfinn unified-pos-ref8: STALE (x5 rebase, architecture axis exhausted)

## Current research focus

1. **LR axis — ceiling still open:**
   - **#1942 alphonse lr-2e-3**: #1895 showed best_epoch=final, largest epoch-14 drop (−7.38 units), cosine still productive at cutoff → lr ceiling still open. Probing 2e-3. Key risk: epoch-5 spike will be larger; monitor pred_abs_max for runaway vs bouncy-bounded.

2. **Regularization axis — stochastic depth:**
   - **#1970 nezuko drop-path-0.1**: DropPath with linear schedule [0.0, 0.025, 0.05, 0.075, 0.1] across 5 Transolver layers. Parameter-free ensemble regularizer; expected to help val_rc and val_re_rand. Mechanistically orthogonal to LR, asinh, node-dropout, loss shape.

3. **Schedule axis — warmup-duration sweep:**
   - **#1813 frieren warmup-5-epochs**: bracket above the winning 4-epoch. ⚠️ Frieren needs to be notified of new 74.2082 target (was tracking 77.14).

4. **Data augmentation axis:**
   - **#1815 askeladd node-dropout-0.9**: rebasing onto asinh base to confirm stacking.

5. **Loss smoothness axis:**
   - **#1817 tanjiro charbonnier-eps-1e-3**: smooth-near-zero L1 on asinh-compressed targets.

6. **Regularization axis:**
   - **#1820 thorfinn weight-decay-5e-3**: moderate L2 on asinh+lr=1.5e-3 base.

## Key research insights so far

- **Loss shape wins (exhausted):** pure-L1 is the global minimum of Smooth-L1 family for MAE criterion.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — critical!
- **4-epoch warmup:** −3.04% on pure-L1 base. Canonical.
- **Asinh pressure compression (GAIN=1.0):** −1.04% on warmup-4 base. Bulk-redistribution mechanism. Cruise gains most. **PRESSURE-SPECIFIC** — high kurtosis of suction-peak tail; velocity channels Gaussian (asinh-all-channels regressed +2.75%).
- **lr=1e-3 + asinh SUPER-ADDITIVE:** −4.41% vs old base (2.8× sum-of-parts). Mechanism: asinh stabilizes high-LR gradient → escape local minima on hard splits while holding cruise.
- **lr=1.5e-3 + asinh:** −3.80% further gain (77.14 → 74.21). Epoch-5 peak-LR spike re-emerges but bounded and recoverable. LR ceiling NOT closed.
- **Warmup duration axis CONVERGING:** warmup=3 (too steep), warmup=4 (canonical winner), warmup=5 (in flight). Best_epoch=final at all tested LRs → cosine still productive.
- **val_rc RESISTANT SPLIT:** gains −0.86% at lr=1.5e-3, −2.54% at lr=1e-3 vs 5%+ on other splits. Architecture or data limitation for multi-foil configurations.
- **Architecture/capacity axes: EXHAUSTED.** All depth/width/slice-num variations regress.

## Next research directions (when new slots open)

1. **lr=3e-3** (if lr=2e-3 still improves — push LR ceiling until instability dominates)
2. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip — 2× effective data, high EV but implementation-complex)
3. **Gradient accumulation + linear LR scaling** (ACCUM_STEPS=2 with lr=3e-3 — effective BS doubles)
4. **Longer training (--epochs 18)** if LR ceiling closes (best_epoch=final multiple times is strong signal for more budget)
5. **val_rc targeted probe** (dedicated experiment for the most resistant split — geometry augmentation, separate loss weight sweep)

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by multiple runs)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
