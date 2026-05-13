# SENPAI Research State

- **Last updated:** 2026-05-13 ~08:00 (merged #2004 nezuko adamw-beta2-0.99 −0.29% new best 73.9964; assigned nezuko #TBD β2=0.95)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 73.9964`** — PR #2004 (β2=0.99 + lr=1.5e-3 + asinh+warmup-4 stack), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=85.100, val_rc=89.815, val_cruise=50.761, val_re_rand=70.309.  
Test: test_avg=64.4437 (test_single=76.764, test_rc=78.036, test_cruise=41.463, test_re_rand=61.511).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- 80.7014 (#1776 4-epoch warmup)
- 79.8623 (#1777 asinh pressure compression GAIN=1.0)
- 77.1419 (#1814 lr=1e-3 + asinh super-additive stack)
- 74.2082 (#1895 lr=1.5e-3 ceiling probe)
- **73.9964** (#2004 adamw-β2=0.99) — **current**

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 (on asinh-compressed targets in normalized space)
- **Asinh pressure compression**: `compress_pressure(y_norm)` / `decompress_pressure(y_c)` with ASINH_GAIN=1.0 (pressure channel only)
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10), grad_clip=1.0, **betas=(0.9, 0.99)**
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #2045 | alphonse | `lr-1.75e-3` | LR midpoint probe: binary-search ceiling between 1.5e-3 (winner) and 2e-3 (dead-end) | **--epochs 14** ✓ | WIP — just assigned |
| nezuko TBD | nezuko | `adamw-beta2-0.95` | β2 axis: probe 0.95 (monotone test toward RoFormer/DeiT) | **--epochs 14** ✓ | WIP — being assigned |
| #1813 | frieren | `warmup-5-epochs` | Warmup 4→5 epochs (bracket above winner) | **--epochs 14** ✓ | WIP — needs new 73.9964 target |
| #1815 | askeladd | `node-dropout-0.9` | Mesh node dropout 0.9 (rebasing onto asinh base) | **--epochs 14** ✓ | WIP — rebasing |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (smooth-near-zero L1) | **--epochs 14** ✓ | WIP — in progress |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (L2 regularization) | **--epochs 14** ✓ | WIP — in progress |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — stale |
| #1421 | edward | `surf-weight-25` | Surface weight 10→25 (stale) | --epochs 20 ⚠️ | WIP — stale |

### Merged PRs (this round — new bests)
- #2004 nezuko adamw-beta2-0.99: **−0.29%** val (74.2082 → 73.9964), test_rc **−4.9%** breakthrough on resistant split
- #1895 alphonse lr-1.5e-3: **−3.80%** (77.1419 → 74.2082) — previous best

### Closed as dead ends (this round)
- #1942 alphonse lr-2e-3: +2.99% vs 74.2082 (stable but optimization quality degraded; LR ceiling between 1.5e-3 and 2e-3; binary-search → #2045 lr-1.75e-3)
- #1911 nezuko warmup-3-epochs: +1.56% vs 77.1419 (3-epoch ramp too steep at lr=1e-3)
- #1970 nezuko drop-path-0.1: +6.99% vs 74.2082 (capacity-reducing regularizer incompatible with 14-epoch budget; DropPath needs 100s+ epochs)
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

1. **LR axis — binary search in progress:**
   - **#2045 alphonse lr-1.75e-3**: midpoint between confirmed winner 1.5e-3 and confirmed dead-end 2e-3. Confirms ceiling location or finds marginal gain.

2. **Optimizer β2 axis — probe β2=0.95:**
   - β2=0.99 won by −0.29% val, −1.03% test with test_rc −4.9% breakthrough. Nezuko's follow-up #1: β2=0.95 (RoFormer/DeiT recipe). Monotone test: does faster adaptation continue to help?

3. **Schedule axis — warmup-duration sweep:**
   - **#1813 frieren warmup-5-epochs**: bracket above the winning 4-epoch. ⚠️ Frieren needs to be notified of new **73.9964** target (was tracking 74.2082).

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
- **Asinh pressure compression (GAIN=1.0):** −1.04% on warmup-4 base. Bulk-redistribution mechanism. Cruise gains most. PRESSURE-SPECIFIC — velocity channels Gaussian.
- **lr=1e-3 + asinh SUPER-ADDITIVE:** −4.41% vs old base (2.8× sum-of-parts).
- **lr=1.5e-3 + asinh:** −3.80% further gain (77.14 → 74.21). LR ceiling NOT closed.
- **β2=0.99:** −0.29% val, −1.03% test. Epoch-5 spike collapsed (+20.4 → +3.2 units). Test_rc breakthrough −4.9%. **β2 axis: probe 0.95 next.**
- **Warmup duration axis CONVERGING:** warmup=3 (too steep), warmup=4 (canonical winner), warmup=5 (in flight).
- **14-epoch-budget constraint on regularizers:** Capacity-reducing regularizers (DropPath, SWA) require 100s+ epochs. Only convergence-preserving mechanisms safe.
- **val_rc RESISTANT SPLIT:** was −0.86% at lr=1.5e-3; β2=0.99 now shows −2.04% val_rc and −4.9% test_rc breakthrough. β2 adaptation rate interacts with multi-foil geometry generalization.
- **Architecture/capacity axes: EXHAUSTED.** All depth/width/slice-num variations regress.

## Next research directions (when new slots open)

1. **β2=0.98** (DeBERTa V3 choice — midpoint between 0.99 winner and 0.95 probe)
2. **β1=0.95 + β2=0.99** (DeiT-style combined; faster momentum decay to smooth warmup→peak transition)
3. **lr=1.75e-3 + β2=0.95 stack** (if both win — compound optimizer-hyperparameter stack)
4. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip — 2× effective data, high EV but implementation-complex)
5. **Longer training (--epochs 18)** if best_epoch=final continues (cosine still productive signal strong)
6. **val_rc targeted probe** (dedicated experiment for the most resistant split)

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by multiple runs)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
