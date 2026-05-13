# SENPAI Research State

- **Last updated:** 2026-05-13 ~02:10 (merged #1776 frieren warmup-4-epochs -3.04% → new baseline 80.7014; closed #1744 tanjiro grad-accum +14.9%, #1723 askeladd OneCycleLR rebased tied, #1722 alphonse β=0.05 +0.47%, #1435 thorfinn stale; assigned #1813 frieren warmup-5, #1814 alphonse lr-1e-3, #1815 askeladd node-dropout, #1817 tanjiro charbonnier, #1820 thorfinn wd-5e-3)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 80.7014`** — PR #1776 (4-epoch warmup + pure-L1 + T_max=14 alignment), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=97.712, val_rc=94.420, val_cruise=55.330, val_re_rand=75.344.  
Test: test_avg=71.9145 (test_single=89.684, test_rc=84.215, test_cruise=46.094, test_re_rand=67.666).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- **80.7014** (#1776 4-epoch warmup) — **current**

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 (in normalized space)
- AdamW lr=7e-4, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10), grad_clip=1.0
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1813 | frieren | `warmup-5-epochs` | Warmup 4→5 epochs (localize optimum) | **--epochs 14** ✓ | WIP — just assigned |
| #1814 | alphonse | `lr-1e-3-warmup4` | Peak LR 7e-4→1e-3 (can warmup-4 allow higher LR?) | **--epochs 14** ✓ | WIP — just assigned |
| #1815 | askeladd | `node-dropout-0.9` | Mesh node dropout 0.9 (vol nodes only, data augmentation) | **--epochs 14** ✓ | WIP — just assigned |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (smooth-near-zero L1) | **--epochs 14** ✓ | WIP — just assigned |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (L2 regularization) | **--epochs 14** ✓ | WIP — just assigned |
| #1777 | nezuko | `asinh-pressure-gain-1` | Asinh value compression on pressure target | **--epochs 14** ✓ | WIP — in progress (notified of 80.70 target) |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — stale (notified of 80.70 target) |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — stale |

### Closed as dead ends (this round)
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

1. **Schedule axis — warmup-duration sweep:**
   - **#1813 frieren warmup-5-epochs**: push from 36% to 43% peak position. Bracket above the winning #1776.

2. **LR axis — high-LR probe with warmup buffer:**
   - **#1814 alphonse lr-1e-3**: tests if 4-epoch warmup buffers allow safely pushing peak LR from 7e-4 to 1e-3.

3. **Data augmentation axis (fresh, untested):**
   - **#1815 askeladd node-dropout-0.9**: randomly drop 10% of volume mesh nodes per training batch. No surface nodes dropped. Tests OOD generalization improvement via regularized mesh coverage.

4. **Loss smoothness axis:**
   - **#1817 tanjiro charbonnier-eps-1e-3**: smooth-near-zero L1 variant. Reduces gradient pressure on near-zero residuals in late epochs. Near-zero regime only (eps=1e-3).

5. **Regularization axis (fresh):**
   - **#1820 thorfinn weight-decay-5e-3**: AdamW weight decay 1e-4→5e-3. Tests if moderate L2 regularization helps on small dataset (1499 samples, 662K params).

6. **Value-level / loss tail axis:**
   - **#1777 nezuko asinh-pressure-gain-1**: asinh value compression on pressure channel. Targets heavy-tailed pressure distribution at high Re.

## Key research insights so far

- **Loss shape wins (exhausted):** pure-L1 is the global minimum of Smooth-L1 family for MAE criterion. β ladder: MSE > β=0.1 > β=0.05 > β=0 monotone. Do not assign more β variants.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — critical!
- **4-epoch warmup:** −3.04% on pure-L1 base. Longer low-LR warmup stabilizes initialization. val_cruise −10.5% dominant gain.
- **OneCycleLR:** −1.38% on Smooth L1, but tied vs pure-L1. Schedule-shape axis saturated; warmup-duration axis is the residual signal.
- **Architecture/capacity axes: EXHAUSTED.** All depth/width/slice-num variations regress.
- **Gradient accumulation (4×, no LR scaling):** update-count-bounded. Not viable at 14-epoch budget without linear LR scaling.
- **SWA/EMA:** don't fit 14-epoch budget cleanly.
- **Wall-distance:** negative stacking.

## Next research directions (when new slots open)

1. **warmup_epochs=3** (bracket below winning 4-epoch, to fully localize the optimum)
2. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip — complex but high EV)
3. **Relative pressure loss** (per-sample range normalization — bounded alternative to #1707's failed std-normalization)
4. **Gradient accumulation + linear LR scaling** (ACCUM_STEPS=2 with lr=1.4e-3 or ACCUM_STEPS=4 with lr=2.8e-3) — tanjiro's suggestion; fixes the update-count-starving pathology from #1744

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by multiple runs)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
