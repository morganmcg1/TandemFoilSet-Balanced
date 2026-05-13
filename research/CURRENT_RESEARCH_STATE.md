# SENPAI Research State

- **Last updated:** 2026-05-13 ~15:10 — Closed #1820 thorfinn wd-5e-3 (extrapolated dead; reassigned to #2373 beta1-0.95); sent #1815 askeladd back for rerun on current baseline (terminal result was on 12h-stale pre-RFF base). 7 in-flight PRs, zero idle students.
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
| #2364 | alphonse | `tmax-14` | CosineAnnealing T_max 10→14; final LR no longer hits 0; addresses best_epoch=14/14 non-convergence | **WIP — just assigned** |
| #2365 | frieren | `chan-weights-5` | channel_weights [1,1,3]→[1,1,5]; pushes 71% (vs 60%) of training signal onto pressure channel | **WIP — just assigned** |
| #2366 | tanjiro | `asinh-gain-2` | ASINH_GAIN 1.0→2.0; tighter pressure-outlier compression; orthogonal to loss-function axis | **WIP — just assigned** |
| #2345 | nezuko | `batch-size-2` | bsz 4→2; gradient-noise regularization, 2× opt steps/epoch; targets OOD via flatter minima | **WIP — training** |
| #2346 | fern | `slice-num-96` | Transolver slice_num 64→96 (+50% physics slices); +5K params; risk: ~35min for 14 epochs | **WIP — training** |
| #2373 | thorfinn | `beta1-0.95` | AdamW β1 0.9→0.95; symmetry with β2=0.99 (closed sweet spot) suggests untested β1 may also have sweet spot >0.9 | **WIP — just assigned** |
| #1421 | edward | `surf-only-channel-weight` | **PROMISING**: val=64.2691 vs 65.2170 baseline on PRE-clip=0.5 HEAD; sent back for rerun on clip=0.5 stack | **WIP — rerun requested** |
| #1815 | askeladd | `node-dropout-0.9` | Node dropout p=0.9; ON OLD BASE val=79.8056 (−1.11% vs 80.7014); sent back for rerun on current canonical (RFF+clip=0.5 stack) | **WIP — rerun on new base** |

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
| grad_clip | **CLOSED** at 0.5 | 0.25 gives +0.65% val; clipping 100% saturated at peak-LR window; halving clip halves effective LR |
| n_head | **CLOSED** at 4 | n=8 gave +23.4% val / +24.6% test; **CRITICAL implementation quirk**: Transolver q/k/v scale as `dim_head²`, so n_head=8 actually LOSES 16.6K params |
| weight_decay | **CLOSED** at 1e-4 | wd=1e-3 gives +1.27% val; **epoch-5 spike RETURNS** even with clip=0.5 (val_avg ep4=186 → ep5=253); wd amplifies effective gradient at peak LR |
| foil-mirror-aug | **CLOSED** | +19.97% val / +21.97% test catastrophic; z=0 is NOT a valid symmetry for tandem-foil dataset (asymmetric flow direction → mirrored samples mis-labeled) |

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

1. **Schedule/LR-tail probe** (#2364 alphonse tmax-14): IN FLIGHT — addresses best_epoch=14/14 non-convergence by keeping LR>0 at end
2. **Loss-weighting probe** (#2365 frieren chan-weights-5): IN FLIGHT — push pressure weighting 60%→71% of training signal
3. **Pressure compression probe** (#2366 tanjiro asinh-gain-2): IN FLIGHT — tighten asinh GAIN, attenuate outlier influence
4. **Batch noise / flat-minima** (#2345 nezuko batch-size-2): IN FLIGHT
5. **Slice_num architecture** (#2346 fern slice-num-96): IN FLIGHT
6. **surf_weight / channel structure** (#1421 edward surf-only): rerun on clip=0.5 stack
7. **RFF on surface normals** (untested): add (n_x, n_z) channels to RFF positional encoding; targets rc/single geometry splits — high-value, more involved
8. **AdamW β1** (#2373 thorfinn beta1-0.95): IN FLIGHT — β1=0.9 untested, sweet spot may be above 0.9 by symmetry with β2
9. **Warmup duration variations** (untested combos): warmup_epochs=2 paired with T_max=12 to maintain endpoint while extending peak-LR window
10. **mlp_ratio probe** (untested): 2→4 doubles FFN width; +~150K params; tests if FFN is the bottleneck on convergence
11. **n_layers depth probe** (untested): 5→6 deeper representation; +~140K params; some compute risk
12. **RFF on surface normals** (untested): add (n_x, n_z) channels to RFF positional encoding; targets rc/single geometry splits

## Epoch budget arithmetic

- Epoch time: ~131–133s (678K params, RFF overhead ~1s/epoch)
- 30-min cap: **14 epochs max** (confirmed)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 after warmup**
- best_epoch=14/14 consistently → model not converged; optimization quality is the binding constraint
