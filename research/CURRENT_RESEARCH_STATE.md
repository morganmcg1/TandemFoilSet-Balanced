# SENPAI Research State
- 2026-04-29 (branch: icml-appendix-charlie-pai2f-r4)
- No human researcher team directives received yet.

## Current r4 Baseline

| Metric | Value | PR |
|--------|-------|----|
| **val_avg/mae_surf_p** | **88.421** | #1243 (fern, n_hidden=192 + AMP bfloat16, epoch 15/15) |

Per-split val breakdown (PR #1243):
- val_single_in_dist: 105.684
- val_geom_camber_rc: 103.384
- val_geom_camber_cruise: 61.552
- val_re_rand: 83.065

Test results (PR #1243):
- test_single_in_dist: 91.001
- test_geom_camber_rc: 90.916
- test_geom_camber_cruise: NaN (corrupt sample — 761 +Inf values in ground-truth pressure)
- test_re_rand: 75.778

**IMPORTANT ARCHITECTURAL NOTE:** The 75.750 result (PR #1197) was achieved on the r1 branch, which included RFF positional encoding + SwiGLU FFN. The r4 codebase does NOT currently have these features:
- `Transolver.preprocess`: plain `MLP(fun_dim + space_dim, n_hidden*2, n_hidden, n_layers=0)` — **no RFF**
- `TransolverBlock.mlp`: vanilla GELU MLP — **no SwiGLU**
- Target for r4 WIP students: **< 88.421** (not 75.750)
- To approach 75.750, RFF + SwiGLU must be ported to r4 (tanjiro's PR #1193 covers RFF; SwiGLU needs explicit assignment)

## r4 Improvement Chain

1. PR #1201 (fern): CosineAnnealingLR T_max=15 + LR 1e-3 → **92.170**
2. PR #1243 (fern): n_hidden=192 + AMP bfloat16 → **88.421** (current r4 baseline)

The r1 branch history (75.750) is a separate track; PRs #1201 and #1243 are the only merged improvements on r4.

## Active Experiments (Round 5, r4)

| PR    | Student    | Status | Hypothesis |
|-------|------------|--------|-----------|
| #1271 | fern       | WIP    | Port SwiGLU FFN to r4 (second half of r1 architectural gap) |
| #1249 | edward     | WIP    | Curvature-weighted surface loss (alpha=5, beta=20) — up-weight LE/TE via raw saf feature; orthogonal to per-sample 1/σ |
| #1193 | tanjiro    | WIP    | Random Fourier Features for multi-scale node positional encoding (n_rff=16, rff_scale=10.0) — **also patches architectural gap vs r1** |
| #1137 | nezuko     | WIP    | Scale Transolver to n_hidden=256, n_layers=8 for high-Re splits |
| #1117 | thorfinn   | WIP    | Re-conditioned output scale head for magnitude adaptation |
| #1111 | askeladd   | WIP    | Layer-wise LR decay for geometry-stable representations |
| #1110 | alphonse   | WIP    | Log-modulus transform on pressure channel loss |
| #1284 | frieren    | WIP    | slice_num=128 with AMP+n_hidden=192 (attention granularity; previously memory-blocked pre-AMP) |

All students should target **val_avg/mae_surf_p < 88.421** (r4 true baseline). The 75.750 figure from PR #1197 was on r1 branch with RFF+SwiGLU and is not the r4 target.

## Recently Merged (r4)

- PR #1243 (fern): n_hidden=192 + AMP bfloat16 → 88.421 (**current r4 baseline**, -4.1% vs 92.170)
- PR #1201 (fern): CosineAnnealingLR T_max=15 + LR 1e-3 → 92.170
- PR #1187 (fern): Gradient clipping (max_norm=1.0) + LR 8e-4 → 102.080
- PR #1128 (edward): Per-sample Re-adaptive loss 1/σ → 124.727
- PR #1112 (edward): Attention dropout=0.1 → 129.531

## Recently Closed

- PR #1114 (frieren, curriculum+sweep surf_weight): CLOSED — sw=4 (92.544) lost to sw=10 control (89.734) on the AMP+n_hidden=160+lr=1e-3+T_max=15 recipe. Old sw=4-5 optimum was a recipe artifact (slow LR cooling); does not transfer. surf_weight=10 confirmed as default for current and future recipes until RFF+SwiGLU land.
- PR #1230 (fern, gradient accumulation bs=4 + lr=1e-3): CLOSED — 101.013 (+33.4% worse). Batch-size direction conclusively closed.
- PR #1213 (fern, batch_size=8 + linear LR 2e-3): CLOSED — 118.098 (+56% worse). Linear LR scaling overshoots small-batch regime.
- PR #1186 (edward, sw5+per-sample loss rerun): CLOSED — 87.924/90.97 vs old baseline → +16.1% regression on r4 recipe.
- PR #1235 (thorfinn-r1, deeper FiLM 3-layer residual): CLOSED — slower convergence at budget, OOD splits regress.

## Key Findings

1. **n_hidden=192 + AMP bfloat16** (PR #1243, r4): 92.170→88.421 (-4.1%); VRAM 44.0 GB of 96 GB; ~124s/epoch; 1.47M params; room for further scaling
2. **AMP bfloat16 + n_hidden=160** (PR #1197, r1 track): best known absolute result 75.750; NOT r4 codebase
3. **CosineAnnealingLR T_max=15** matched to budget: 102.080→92.170; T_max mismatch was critical
4. **Gradient clipping + LR 8e-4**: 124.727→102.080 (18.2% improvement)
5. **Per-sample Re-adaptive loss** (1/σ weighting): 129.531→124.727
6. **Attention dropout=0.1**: established 129.531 at round 4 start
7. **surf_weight optimum at sw=4-5 (non-monotonic)**: full sweep sw=3→134.91, sw=4→126.49, sw=5→126.93, sw=6→132.90
8. **VRAM headroom**: peak 44.0 GB of 96 GB — ~52 GB unused; room for larger n_hidden or n_layers
9. **Training budget**: ~15 epochs of 50 configured with AMP; convergence speed and throughput are primary levers
10. **Batch-size direction conclusively closed**: bs=8 + linear LR scaling (#1213, +56%) and bs=4 grad-accum effective bs=8 (#1230, +33%) both regress
11. **Architectural gap with r1**: r4 is missing RFF positional encoding + SwiGLU FFN — these are the primary missing features to recover the r1 75.750 performance on r4 codebase; tanjiro's #1193 covers RFF

## Key Dataset Observations

- Cruise split (61.552 val) dramatically easier than raceCar splits (~83-105): multi-domain difficulty imbalance
- Per-sample pressure std varies by order of magnitude within a split (high-Re drives extremes)
- VRAM: peaked 44.0 GB of 96 GB available — substantial headroom for scaling
- Timeout: ~30 min wall-clock → ~15 epochs with AMP bfloat16; LR schedule / convergence speed is major lever
- test_geom_camber_cruise/000020.pt has 761 +Inf values in ground-truth pressure (scoring returns NaN for this split without +Inf masking)

## Current Research Focus

**Target:** TandemFoilSet CFD surrogate — predict (Ux, Uy, p) at every mesh node.
**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 val splits (lower is better).
**Model:** Transolver with physics-aware attention over irregular meshes.
**Round 5/5. Best r4 result: PR #1243 (fern) at val_avg/mae_surf_p=88.421.**

Key strategic priorities:
1. **Port RFF positional encoding to r4** (tanjiro #1193 WIP — highest priority; directly patches architectural gap)
2. **Port SwiGLU FFN to r4** (fern #1271 WIP — second half of r1 architectural gap)
3. **Surface loss improvements** (edward #1249 — curvature-weighted LE/TE)
4. **Architecture scaling on capacity axis** (nezuko #1137 — n_hidden=256/n_layers=8)
5. **Architecture scaling on attention-granularity axis** (frieren — slice_num=128 with AMP+n_hidden=192; previously memory-blocked)
6. **LR scheduling improvements** (askeladd #1111 — layer-wise LR decay)
7. **Loss formulation** (alphonse #1110 — log-modulus pressure transform)
8. **Output head specialization** (thorfinn #1117 — re-conditioned output scale head)

## Potential Next Research Directions (Post Round 5)

1. **RFF + SwiGLU combined**: After both individual ports (#1193, #1271) confirm, test them together on r4 to see if they reproduce the r1 75.750 result
2. **Larger slice_num with AMP** (now in flight as frieren's new assignment): slice_num=128 was closed pre-AMP; revisit on n_hidden=192+AMP recipe
3. **Warm restart LR** (CosineAnnealingWarmRestarts, T_0=7): 2 full cycles within ~15-epoch budget
4. **Divergence-free penalty**: Approximate ∇·u=0 constraint penalty (lambda=0.01) — physics-informed regularization
5. **Per-channel output head**: Three separate 1-output linear layers for Ux, Uy, p — reduces channel interference
6. **Multi-seed bracketing** for the top-3 contenders post-RFF/SwiGLU — to firm up the magnitude estimates of small wins
7. **Test split NaN-guard fix** (evaluate_split predictions-side check for non-finite preds — flagged 4× by frieren, currently only ground-truth side is guarded)
8. **surf_weight re-sweep on the unified r4+RFF+SwiGLU recipe**: Once architecture is unified, the optimum may shift again — a tight {6, 8, 10, 12, 16} sweep would be the right shape
