# SENPAI Research State

- **Last updated:** 2026-05-13 ~15:30 — Closed #2364 alphonse tmax-14 (+11.71%, T_max axis closed at 10; hard splits need low-LR fine-tuning). Sent #2366 tanjiro asinh-gain-2 back for batch=2 rerun (val=64.0586 on batch=4 was −1.78% but new baseline is 63.1086). Assigned alphonse #2406 lr-1.25e-3 retest on batch=2 base.
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 63.1086`** — PR #2345 (batch_size=2 on grad_clip=0.5+RFF+asinh+warmup-4+lr=1.5e-3+β2=0.99 canonical stack), epoch 14/14.

Per-split: val_single=69.1032, val_rc=77.8463, val_cruise=41.6722, val_re_rand=63.8128.  
Test: test_avg=54.9824 (test_single=62.4700, test_rc=69.6733, test_cruise=34.7817, test_re_rand=53.0047).

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
- 65.2170 (#2260 grad_clip=0.5) (−0.17% val, −0.85% test; eliminates epoch-5 spike)
- **63.1086** (#2345 batch_size=2) — **current** (−3.23% val, −2.62% test; val_single −6.32%)

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 in asinh-compressed target space
- **Asinh pressure compression**: ASINH_GAIN=1.0 (pressure channel only)
- **RFF positional encoding**: σ=3.0, 64-dim [cos, sin] of (x,z), fixed B seeded 42, preprocess MLP input 24→86
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10, eta_min=0), **grad_clip=0.5**, **betas=(0.9, 0.99)**, wd=1e-4, ε=1e-8
- **batch_size=2** (canonical from #2345), surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Status |
|----|---------|------|------|---|
| PR | Student | Slug | Axis | Status |
|----|---------|------|------|---|
| #2406 | alphonse | `lr-1p25e-3-bsz2` | lr 1.5e-3→1.25e-3 retest on batch=2 base; LR axis was closed pre-batch=2, may reopen | **WIP — just assigned** |
| #2387 | nezuko | `batch-size-1` | bsz 2→1 (true SGD); probe if batch axis continues; may timeout at ep13-14 | **WIP — training** |
| #2388 | fern | `warmup-2-tmax-12` | warmup 4→2 + T_max 10→12; same cosine endpoint, more peak-LR time; batch_size=2 canonical | **WIP — training** |
| #2366 | tanjiro | `asinh-gain-2` (rerun bsz=2) | ASINH_GAIN 1.0→2.0; per-split signature matched hypothesis exactly on bsz=4 (val_single −2.91%, val_rc −2.50%, val_cruise tied); rerun with bsz=2 to test stacking | **WIP — rerun on bsz=2** |
| #2365 | frieren | `chan-weights-5` | channel_weights [1,1,3]→[1,1,5]; **NOTE: used batch=4, evaluate vs 63.1086** | **WIP — likely finishing** |
| #2373 | thorfinn | `beta1-0.95` | AdamW β1 0.9→0.95; untested axis; **NOTE: used batch=4, evaluate vs 63.1086** | **WIP — training** |
| #1421 | edward | `surf-only-channel-weight` | val=64.2691 on PRE-clip=0.5 HEAD; sent back for rerun on current canonical | **WIP — rerun requested** |
| #1815 | askeladd | `node-dropout-0.9` | on OLD base val=79.8056 (−1.11% vs 80.7014); sent back for rerun on current stack | **WIP — rerun on new base** |

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
| slice_num | **CLOSED** at 64 | slice_num=96 gives +11.0% val (epoch-12, timeout); all splits regress uniformly; no physics-slice benefit; per-epoch overhead is the liability |
| batch_size | **OPEN — 2 is current best** | batch=2 gives −3.23% val (NEW BEST 63.1086); val_single −6.32%; batch=1 probe in flight (#2387) |
| T_max (schedule extension) | **CLOSED** at 10 | T_max=14 gives +11.71% val (val_rc +19.9% worst); model under-converged at epoch 14 (lr=4.25e-4, still dropping 5.7/epoch); **cosine-to-zero tail does critical fine-tuning** — truncating it costs more than mid-LR-time gains; hard splits depend critically on low-LR fine-tuning quality |

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

1. **Batch-axis continuation** (#2387 nezuko batch-size-1): IN FLIGHT — true SGD probe; timeout-limited to ~13 epochs
2. **Warmup/schedule** (#2388 fern warmup-2-tmax-12): IN FLIGHT — more peak-LR time, same cosine endpoint; batch=2 canonical
3. **Schedule tail** (#2364 alphonse tmax-14): IN FLIGHT (batch=4, evaluate vs 63.1086)
4. **Channel weighting** (#2365 frieren chan-weights-5): IN FLIGHT (batch=4)
5. **Pressure compression** (#2366 tanjiro asinh-gain-2): IN FLIGHT (batch=4)
6. **Momentum** (#2373 thorfinn beta1-0.95): IN FLIGHT (batch=4)
7. **surf_weight / channel structure** (#1421 edward surf-only): rerun on current stack
8. **Node dropout** (#1815 askeladd node-dropout-0.9): rerun on current stack
9. **mlp_ratio probe** (untested): 2→4 doubles FFN width; +~150K params; compute-risky (~35min) but capacity axis important
10. **n_layers depth probe** (untested): 5→6; +~140K params; similar compute risk
11. **warmup_epochs=3** (untested midpoint): split the difference between 2 and 4 once fern's warmup=2 result is in
12. **RFF on surface normals** (untested): add (n_x, n_z) channels to RFF encoding; higher-complexity but high-value for rc/single OOD

## Epoch budget arithmetic

- Epoch time: ~131–133s (678K params, RFF overhead ~1s/epoch)
- 30-min cap: **14 epochs max** (confirmed)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 after warmup**
- best_epoch=14/14 consistently → model not converged; optimization quality is the binding constraint
