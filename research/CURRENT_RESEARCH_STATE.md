# SENPAI Research State

- **Last updated:** 2026-05-13 ~04:00 (merged #1814 alphonse lr-1e-3+asinh −3.40% → new baseline 77.1419; assigned alphonse #1836 lr-1.5e-3)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 77.1419`** — PR #1814 (lr=1e-3 + 4-epoch warmup on asinh-compressed targets), epoch 14/14, 0.66M param Transolver.

Per-split: val_single=89.672, val_rc=92.482, val_cruise=54.093, val_re_rand=72.321.  
Test: test_avg=67.6796 (test_single=78.491, test_rc=83.212, test_cruise=44.225, test_re_rand=64.791).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- 80.7014 (#1776 4-epoch warmup)
- 79.8623 (#1777 asinh pressure compression GAIN=1.0)
- **77.1419** (#1814 lr=1e-3 + asinh super-additive stack) — **current**

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 (on asinh-compressed targets in normalized space)
- **Asinh pressure compression**: `compress_pressure(y_norm)` / `decompress_pressure(y_c)` with ASINH_GAIN=1.0
- AdamW **lr=1e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10), grad_clip=1.0
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #1836 | alphonse | `lr-1.5e-3` | LR ceiling probe: 1e-3→1.5e-3 on asinh+warmup4 | **--epochs 14** ✓ | WIP — just assigned |
| #1813 | frieren | `warmup-5-epochs` | Warmup 4→5 epochs (localize optimum) | **--epochs 14** ✓ | WIP — in progress (notified of 77.14 target) |
| #1815 | askeladd | `node-dropout-0.9` | Mesh node dropout 0.9 (rebasing onto asinh base) | **--epochs 14** ✓ | WIP — rebasing to confirm stacking |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (smooth-near-zero L1) | **--epochs 14** ✓ | WIP — in progress |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (L2 regularization) | **--epochs 14** ✓ | WIP — in progress |
| #1835 | nezuko | `asinh-gain-0.5` | ASINH_GAIN 1.0→0.5 (wider linear region, milder compression) | **--epochs 14** ✓ | WIP — in progress |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF (space_dim 2→64) | --epochs 20 ⚠️ | WIP — stale (notified of 77.14 target) |
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

1. **LR axis — ceiling probe:**
   - **#1836 alphonse lr-1.5e-3**: probe LR ceiling above 1e-3. Strict monotone descent at 1e-3 (no peak-LR spike with asinh) suggests headroom. Next ceiling bracket.

2. **Schedule axis — warmup-duration sweep:**
   - **#1813 frieren warmup-5-epochs**: bracket above the winning 4-epoch (43% peak position). Helps localize warmup optimum.

3. **Data augmentation axis:**
   - **#1815 askeladd node-dropout-0.9**: rebasing onto asinh base to confirm stacking. Pre-asinh: val_single −1.97%, val_re_rand −1.96% (memorization regularization).

4. **Loss smoothness axis:**
   - **#1817 tanjiro charbonnier-eps-1e-3**: smooth-near-zero L1 on asinh-compressed targets.

5. **Regularization axis:**
   - **#1820 thorfinn weight-decay-5e-3**: moderate L2 on asinh+lr=1e-3 base.

6. **ASINH_GAIN sweep:**
   - **#1835 nezuko asinh-gain-0.5**: GAIN 1.0→0.5 (milder compression). But note: now that lr=1e-3 is canonical, this is testing GAIN effect on the new base.

## Key research insights so far

- **Loss shape wins (exhausted):** pure-L1 is the global minimum of Smooth-L1 family for MAE criterion.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — critical!
- **4-epoch warmup:** −3.04% on pure-L1 base. Canonical.
- **Asinh pressure compression (GAIN=1.0):** −1.04% on warmup-4 base. Bulk-redistribution mechanism. Cruise gains most.
- **lr=1e-3 + asinh SUPER-ADDITIVE:** −4.41% vs old base (2.8× sum-of-parts). Mechanism: asinh stabilizes high-LR gradient → escape local minima on hard splits while holding cruise. val_single −7.99% largest single-split gain. Epoch-5 peak-LR spike GONE with asinh active.
- **Architecture/capacity axes: EXHAUSTED.** All depth/width/slice-num variations regress.
- **Gradient accumulation (4×, no LR scaling):** update-count-bounded.
- **SWA/EMA:** don't fit 14-epoch budget cleanly.
- **Schedule-shape axis:** saturated (OneCycleLR tied vs pure-L1).

## Next research directions (when new slots open)

1. **lr=2e-3** (if lr=1.5e-3 wins, push ceiling further)
2. **warmup_epochs=3** (bracket below 4, fully localize warmup optimum)
3. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip — high EV but complex)
4. **Gradient accumulation + linear LR scaling** (ACCUM_STEPS=2 with lr=2e-3 — enables effectively longer training while respecting the update-count constraint)
5. **Asinh on Ux/Uy channels** (bulk-redistribution may help velocity channels too)

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params)
- 30-min cap: **14 epochs max** (confirmed by multiple runs)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
