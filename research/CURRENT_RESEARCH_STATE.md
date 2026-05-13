# SENPAI Research State

- **Last updated:** 2026-05-13 ~13:25 — PR #2260 (grad_clip=0.5) MERGED new best 65.2170; PRs #2238 (trainable-B), #2265 (lr-1.25e-3) CLOSED; assigned nezuko #2291 clip=0.25, alphonse #2293 wd=1e-3, fern #2292 n-head-8
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 65.2170`** — PR #2260 (grad_clip=0.5 on RFF+asinh+warmup-4+lr=1.5e-3+β2=0.99 canonical stack), epoch 14/14.

Per-split: val_single=73.7639, val_rc=79.4389, val_cruise=42.8481, val_re_rand=64.8172.  
Test: test_avg=56.4581 (test_single=64.4538, test_rc=71.6744, test_cruise=35.2533, test_re_rand=54.4508).

**Historical trajectory (this launch):**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- 80.7014 (#1776 4-epoch warmup)
- 79.8623 (#1777 asinh GAIN=1.0)
- 77.1419 (#1814 lr=1e-3 + asinh super-additive)
- 74.2082 (#1895 lr=1.5e-3)
- 73.9964 (#2004 adamw-β2=0.99)
- 65.3304 (#1657 RFF σ=3.0 +64-dim pos encoding) — LARGEST SINGLE JUMP −11.71%
- **65.2170** (#2260 grad_clip=0.5) — **current** (−0.17% val, −0.85% test; eliminates epoch-5 spike)

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 in asinh-compressed target space
- **Asinh pressure compression**: ASINH_GAIN=1.0 (pressure channel only)
- **RFF positional encoding**: σ=3.0, 64-dim [cos, sin] of (x,z), fixed B seeded 42, preprocess MLP input 24→86
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10, eta_min=0), **grad_clip=0.5**, **betas=(0.9, 0.99)**, wd=1e-4, ε=1e-8
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Status |
|----|---------|------|------|---|
| #2291 | nezuko | `grad-clip-0p25` | Clip bracket: 0.5→0.25; completes the clip axis monotone test | **WIP — just assigned** |
| #2293 | alphonse | `wd-1e-3` | Weight decay 1e-4→1e-3 (10×); untested axis on new stack | **WIP — just assigned** |
| #2292 | fern | `n-head-8` | Transolver n_head 4→8; same params/FLOPs, finer-grained attention | **WIP — just assigned** |
| #2257 | frieren | `foil-mirror-aug` | Z-axis foil reflection augmentation; doubles effective training data | **WIP — in training** |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 50× (1e-4→5e-3); nudged to rebase onto clip=0.5 HEAD | **WIP — nudged** |
| #1421 | edward | `surf-weight-25` | surf_weight 10→25; nudged to rebase onto clip=0.5 HEAD | **WIP — nudged** |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss alternative; rebasing on RFF+clip=0.5 base | **WIP — rebasing** |
| #1815 | askeladd | `node-dropout-0.9` | Node dropout p=0.9; rebasing/rerunning on RFF+clip=0.5 base | **WIP — rebasing** |

## Closed axes (exhausted)

| Axis | Status | Key result |
|---|---|---|
| LR | **CLOSED** at lr=1.5e-3 | Both 1.25e-3 (−2%) and 2e-3 (−13%) regress; optimal on both pre-RFF and RFF bases |
| σ-bandwidth | **CLOSED** at σ=3.0 | σ=1.0 (−6.4%), σ=3.0 (−11.71%), σ=5.0 (+2.75%) — non-monotone |
| RFF anisotropy | **CLOSED** isotropic | σ_z=1.5 gives +0.51% val regression (split tradeoff) |
| RFF capacity (n_features) | **CLOSED** at d=32 | n=64 gave +1.78% regression; per-split signature: cruise ↓, rc/single ↑ |
| Trainable RFF B | **CLOSED** | +1.88% regression; fixed B prior better at 14-epoch budget |
| eta_min | **CLOSED** at 0 | eta_min=1e-4 gives +1.77% (raises entire cosine curve, overshoots cruise) |
| warmup duration (RFF) | **CLOSED** at 4 epochs | warmup=5 +0.85% on RFF (doesn't stack from pre-RFF) |
| AdamW ε | **CLOSED** on RFF | ε=1e-6 tie; spike is gradient-magnitude not ε-driven |
| β2 | **CLOSED** at 0.99 | β2=0.95 +1.12% (non-monotone, 0.99 sweet spot) |

## Key research insights

- **Loss shape wins (exhausted):** pure-L1 is global minimum for MAE criterion. Canonical.
- **RFF σ=3.0 BREAKTHROUGH:** −11.71% val / −11.65% test. Largest single improvement. ALL 4 splits improve. σ axis closed.
- **REPEATED per-split signature across RFF axis mods** (σ=5.0, anisotropic, n_features=64): more complex RFF → cruise improves, rc/single regress. Cruise is sensitive to optimization quality; rc/single need SIMPLER, more stable feature representation.
- **RFF base has SHARPER curvature near optimum** (5× worse lr regression than pre-RFF). Any perturbation adding noise regresses more. Optimal hyperparameters are TIGHTER on RFF base.
- **Epoch-5 spike (RFF base): +91 units** — gradient-magnitude driven (86-dim input × peak LR). clip=0.5 eliminates it → descends instead. clip=1.0 was insufficient (spike still occurred).
- **Grad_clip axis open**: clip=0.5 gives −0.17% val / −0.85% test; test improvement much larger → reduced generalization error. Next: clip=0.25.
- **Split hardness hierarchy:** val_rc > val_single > val_re_rand > val_cruise for OOD difficulty. Cruise responds to almost any optimization improvement; rc/single are the hard OOD targets.
- **best_epoch=14/14 consistently** — model never fully converges in 30-min budget. Optimization quality matters more than architecture at this stage.
- **Pre-RFF validated improvements may NOT stack** — must re-validate each axis independently on the RFF base.

## Next research directions (priority order)

1. **Clip bracket completion** (#2291 nezuko clip=0.25): monotone test on clip axis; if positive, probe 0.125
2. **Weight decay tuning** (#2293 alphonse wd=1e-3 / #1820 thorfinn wd=5e-3): wd=1e-4 never re-optimized; standard transformer wd is 1e-2 to 1e-3; OOD generalization target
3. **Attention architecture** (#2292 fern n_head=8): finer-grained attention; no param/FLOP cost; targets rc/single geometry OOD
4. **Data augmentation** (#2257 frieren foil-mirror-aug): z-axis reflection doubles effective data; orthogonal to optimization axes
5. **surf_weight tuning** (#1421 edward surf-weight=25): −2.95% pre-RFF, worth retesting on RFF+clip=0.5 base
6. **Alternative loss functions** (#1817 tanjiro charbonnier): still rebasing, value unclear on RFF base
7. **Slice_num architecture** (untested): probe 64→96 if n_head=8 closes architecture axis
8. **RFF on surface normals** (untested): add (n_x, n_z) to RFF encoding, target rc/single geometry splits

## Epoch budget arithmetic

- Epoch time: ~131–133s (678K params, RFF overhead ~1s/epoch)
- 30-min cap: **14 epochs max** (confirmed)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 after warmup**
- best_epoch=14/14 consistently → model not converged; optimization quality is the binding constraint
