# SENPAI Research State

- **Last updated**: 2026-05-14 ~05:30 UTC (Wave 20 / Iter-14 CLOSED 8/8 — second consecutive full washout; Wave 21 / Iter-15: ASSIGNING 8 corrected-mechanism PRs H94-H101)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received on this branch.

## Current best baseline (PR #2648 merged — linear attn-temp anneal √3→√2, 21st compound win)

- `val_avg/mae_surf_p` = **55.1595** (e12; full stack: ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + LayerScale γ no-WD 10× lr init=0.1 + LR warmup + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth + **fixed attention scale = √2 × default** + **linear anneal τ: √3→√2 over 12 epochs**)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **48.3010**
- Per-split val: single_in_dist=60.851 / camber_rc=68.657 / camber_cruise=35.762 / re_rand=55.368
- Per-split test: single_in_dist=54.849 / camber_rc=61.620 / camber_cruise=29.653 / re_rand=47.082
- **Compound progress**: 100.957 → **55.1595** = **−45.37% over 21 merges**
- **n_params**: **892,637** (unchanged — `attn_sharpening_factor` is a buffer, not a parameter)

## Current research focus

**Wave 21 — SECOND PLATEAU-PROTOCOL ESCALATION: CORRECTED MECHANISMS**. Wave 19 (7/7 closed) and Wave 20 (8/8 closed) are two consecutive full washouts. Wave 20's first escalation (model-class changes, loss reformulation, curriculum, SSL) all failed, but each failure revealed *why* it failed — not just that it did. Wave 21 applies corrected implementations of each Wave 20 hypothesis.

**KEY FINDING FROM WAVE 20**: Every Wave 20 failure was an implementation-level problem, not a mechanism-level closure:
- Relative L1: train/eval metric mismatch (gradient normalized, eval absolute)
- Camber curriculum: easy-first downweighted the hard OOD target (camber_rc); anti-curriculum was better
- Camber-cond LN: additive correction on frozen LN requires unlearning; not feasible in 12 epochs
- Sobolev surface loss: mesh irregularity causes ds_min ≈ 0 → gradient explosion; student self-corrected λ but still unstable
- GeoMPNN: per-forward KNN graph construction = only ~3 effective training epochs; model never trained
- Masked node SSL: masked input geometry (wrong signal); should mask output pressure targets
- SE(2)-equivariant: rigid equivariance eliminates AoA direction information; d_v=2 too small
- Bernoulli physics loss: physical-space gradients 100-1000× larger than normalized MAE

**Wave 21 strategy** — 8 corrected hypotheses H94-H101, each a targeted fix of a Wave 20 failure:

### Corrected implementations (7 hypotheses)
- **H94 hard-first camber upsampling**: fixed 2×/3× upsampling of camber_rc samples (alphonse / #pending). Corrects H87's easy-first direction.
- **H95 pressure-node mask SSL**: mask output pressure targets, reconstruct as aux supervised task (edward / #pending). Corrects H91's input-masking error.
- **H96 Re-cond feature scale**: single scalar Re-conditional gain multiplier, identity-initialized (nezuko / #pending). Corrects H88's additive-correction-on-frozen-LN failure.
- **H97 arc-length Sobolev loss**: equispaced arc-length resampling before dp/ds computation (fern / #pending). Corrects H89's ds_min instability.
- **H98 static KNN GNN correction**: precomputed graph + 1-layer GNN correction layer on top of Transolver (tanjiro / #pending). Corrects H90's per-forward-pass graph construction overhead.
- **H99 normalized Bernoulli λ=1e-5**: Bernoulli residual in normalized space at λ=1e-5 (frieren / #pending). Corrects H93's physical-space gradient amplification.
- **H100 AoA-decomposed attention**: (cos α, sin α) encoding + 2-head directional cross-attention (thorfinn / #pending). Corrects H92's AoA-information loss from SE(2) constraint.

### New hypothesis (1)
- **H101 adaptive surf-weight schedule**: schedule surf_ch_weight [1,1,1]→[0.5,0.5,2.0] over epochs 1-4 (askeladd / #pending). Novel application of #2648's scheduling-beats-fixed-value precedent to the channel weight axis.

**OOD targets**: camber_rc=68.657 (worst split, +12.8% gap vs in-dist) and re_rand=55.368 remain the primary improvement targets.

## Wave 21 / Iter-15 active threads (assigning)

| Student | Slug | Hypothesis | Family |
|---------|------|---------|--------|
| alphonse | hard-first-camber-upsampling | Fixed 2×/3× upsampling of camber_rc | Curriculum B |
| edward | pressure-node-mask-ssl | Mask output pressure targets, reconstruct aux | SSL B |
| nezuko | re-cond-feature-scale | Scalar Re-conditional gain, identity init | Conditioning B |
| fern | arclength-sobolev-loss | Equispaced arc-length dp/ds computation | Loss B2 |
| tanjiro | static-knn-gnn-correction | Precomputed KNN + 1-layer GNN correction | Model Class B |
| frieren | normalized-bernoulli-1e5 | Bernoulli in normalized space at λ=1e-5 | Loss C2 |
| thorfinn | aoa-decomposed-attention | (cos α, sin α) + directional cross-attention | Equivariant B |
| askeladd | adaptive-surf-weight-schedule | Schedule surf_ch_weight [1,1,1]→[0.5,0.5,2] | Loss D |

## Permanently closed axes (do not re-test)

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum |
| Cosine T_max | T_max=14 (per-batch) | #2308: T_max=12 +3.24% |
| LR warmup | epoch-1 linear ramp | Merged |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L (fixed) | L=6 dyadic | L=8 plateau; now learned |
| LayerScale init | γ_l=0.1 | #2475 19th win; sweep fully closed |
| Surf-ch-weight (fixed) | [0.5,0.5,2.0] | 4× p:v ratio optimum — note: schedule variant H101 now testing |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | #2720 slice_num=96 +13.00%; re-confirmed |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal |
| Gate | ReGLU (ReLU) | SiLU<GELU<ReLU<AbsGLU<SqReLU; ReLU optimum |
| n_hidden | 128 | #2371: quadratic scaling, compute-bound |
| inner_dim | 288 | #2386: 320 over-fits; 256 under-fits |
| OOD domain upsampling | equal weights | #2391: extrapolation gap, not density gap — NOTE: H94 tests *hard-split upsampling*, not uniform OOD upsampling |
| EMA, dropout, coord-jitter | off | Compound improvements prove these wrong |
| Fourier variants (separate xy, hybrid, equilibrium init) | learned L=6 dyadic | All closed in Wave 14–16 |
| FiLM conditioning | not used | #2453: helps ID, hurts all OOD |
| SWA | off | #2476: model still descending at cutoff |
| RMSNorm QK-norm | not used | Both F.normalize and RMSNorm variants closed |
| LayerScale decoupled/asymmetric | symmetric init=0.1 | #2414/#2510 closed; trajectory > endpoint |
| Y-axis reflection | not used | #2514: tandem foil NOT y-symmetric |
| **Attention-temperature axis FULLY CLOSED** | fixed √2 × linear anneal √3→√2 over 12 epochs starting epoch 1 | #2519 + #2648 won; #2574/#2655/#2714/#2715/#2716/#2717/#2718 all closed |
| Additive-scale / optimizer-config (8 closures) | n/a | #2517/#2518/#2511/#2515/#2513/#2576 all mechanism-overlap with #2519 |
| FOMA input noise | n/a | #2649 closed; too weak in 12-epoch budget |
| Sparse MoE out_project (4-expert) | n/a | #2651 closed; under-trains in 12 epochs |
| Thin-airfoil Cp_TAT aux loss | n/a | #2652 closed; NACA 4-digit too crude for tandem-foil |
| Spectral HF penalty on slice tokens | n/a | #2653 closed; slice tokens are dispatch reps, not spatial fields |
| SDF geometry features | n/a | #2654 closed; conflicts with FourierCoordEnc |
| Slice-token mixup | n/a | #2575 catastrophic; dispatch structure destroyed |
| Geo aux head (slice-token camber+Re prediction) | n/a | #2719 closed +4.67%; output-side aux interferes |
| Laplacian eigenvector PE from KNN graph | n/a | #2656 closed +10.18%; spectral PE interferes with Fourier coords |
| Relative L1 loss | n/a | #2767 closed +10.3%; train/eval metric mismatch; corrected direction → H101 |
| Camber easy-first curriculum | n/a | #2769 closed +10.3%/+5.9%; downweights hard OOD target; corrected → H94 hard-first |
| Camber-cond LayerNorm (additive) | n/a | #2771 closed ~+5%; additive correction on frozen LN fails in budget; corrected → H96 |
| Sobolev surface loss (raw mesh) | n/a | #2774 closed +11.9%; ds_min instability; corrected → H97 arc-length resampled |
| GeoMPNN full model replacement | n/a | #2777 closed +209%; per-forward KNN too slow (3 epochs); corrected → H98 static graph |
| Masked input-geometry SSL | n/a | #2781 closed ~+12%; wrong task signal; corrected → H95 pressure-target masking |
| SE(2)-equivariant attention | n/a | #2783 closed ~+8%; loses AoA direction; corrected → H100 AoA-decomposed |
| Bernoulli physics loss (physical-space) | n/a | #2785 closed ~+14%; physical-space gradient amplification; corrected → H99 normalized |

## Third escalation plan (if Wave 21 washes)

If Wave 21 produces another full washout (0/8 wins), the third escalation will be:
1. **Full model-class replacement**: FNO (Fourier Neural Operator), DeepONet, or large GNN — complete Transolver replacement, not hybrid
2. **Data augmentation within camber families**: geometry-conditional mixing of samples within camber neighborhoods to synthesize OOD training examples
3. **Formal NAS / joint hyperparameter search**: systematic grid or Bayesian search over the remaining high-dimensional hyperparameter space (depth × width × heads × LR × schedule)
4. **Physics-informed architecture**: mesh-free collocation methods, PINN variants with explicit boundary conditions

## Prioritized future ideas (queued after Wave 21)

1. **If H94 (hard-first upsampling) wins**: try different upsampling ratios (4×, 5×), or make the ratio an annealed curriculum (high early, taper to baseline)
2. **If H95 (pressure SSL) wins**: try higher masking ratios (30%, 40%), contrastive SSL variants across airfoil geometries
3. **If H96 (Re-cond scale) wins**: extend to camber-conditional, full-NACA-conditional variants; try application at intermediate blocks
4. **If H97 (arc-length Sobolev) wins**: try higher λ values, combined Sobolev + L1 loss scheduling
5. **If H98 (static KNN GNN) wins**: add more GNN layers, try different aggregation (attention-weighted vs mean)
6. **If H99 (normalized Bernoulli) wins**: layer in incompressibility constraint, momentum balance, full PDE residual in normalized space
7. **If H100 (AoA-decomposed attn) wins**: enrich direction encoding (more Fourier frequencies of α), apply at multiple block positions
8. **If H101 (surf-weight schedule) wins**: try different schedule shapes (cosine, quadratic), different endpoint ratios
