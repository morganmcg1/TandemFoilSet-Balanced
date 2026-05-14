# SENPAI Research State

- **Last updated**: 2026-05-14 ~04:00 UTC (Wave 19 / Iter-13 CLOSED 8/8 (all 7 reviews regressed + #2715 stalled); Wave 20 / Iter-14: ASSIGNED 8 plateau-protocol PRs #2767/#2769/#2771/#2774/#2777/#2781/#2783/#2785; pivot to model-class changes, loss reformulation, curriculum, SSL)
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

**Wave 20 — PLATEAU-PROTOCOL ESCALATION**. Wave 19 was a wholesale washout (7/7 review-ready PRs regressed; 8th stalled). The attention-temperature schedule axis is now fully bracketed and closed on all 4 sub-axes (start value, end value, decay shape, per-layer differentiation). Per the plateau protocol, Wave 20 pivots to **fundamentally different strategy tiers**: model-class changes, loss reformulation, and curriculum learning.

**KEY MECHANISM CLOSED IN WAVE 19**: The complete schedule sweep proves that the current linear √3→√2 over 12 epochs starting epoch 1 with uniform across-layer treatment is at a tight local optimum. Sharper start (√5, +3.70%), softer end (1.0, +7.99%), delayed decay (hold-then-decay, +4.32%), and per-layer differentiation (+4.79%) all regress. The schedule axis is permanently closed.

**ADDITIONAL CLOSED IN WAVE 19**:
- Slice-token output-side aux losses (#2719 geo aux head +4.67%) — interferes with dispatch
- Spectral / graph-based geometry encodings (#2656 Laplacian PE +10.18%) — same failure pattern as SDF (#2654) and HF spectral (#2653)
- slice_num=96 (#2720 +13.00%) — re-confirms slice_num=64 optimum

**Wave 20 strategy** — 8 fresh hypotheses from researcher-agent escalation, each targeting a different abstraction level:

### Loss reformulation (3 hypotheses)
- **H86 relative L1 loss**: per-sample magnitude normalization (askeladd / #2767). Addresses Re-scale gradient-magnitude mismatch between training (normalized space) and eval (physical units).
- **H89 Sobolev surface loss**: gradient-matching aux on dp/ds over arc-length (fern / #2774). Targets pressure-distribution-shape generalization.
- **H93 NSE Bernoulli consistency**: physics-informed aux on total head conservation (frieren / #2785). Soft physical anchor independent of data distribution.

### Curriculum / sampling (1 hypothesis)
- **H87 camber-difficulty curriculum**: difficulty-weighted sampler with 4-epoch warmup → 8-epoch uniform (alphonse / #2769). Targets representation-formation timing during early epochs (mechanism analog from #2648 attn-temp finding).

### Architecture / inductive bias (4 hypotheses)
- **H88 camber-conditional LayerNorm**: AdaIN-style camber-only scale-shift, identity-initialized (nezuko / #2771). Mechanistically distinct from closed FiLM.
- **H90 GeoMPNN**: replace Transolver with geometry-aware message-passing GNN (tanjiro / #2777). MAJOR MODEL CLASS CHANGE — NeurIPS 2024 ML4CFD competition winner. High variance, high upside.
- **H91 masked node SSL pretraining**: 2-phase 4-epoch pretrain + 8-epoch fine-tune (edward / #2781). Free signal from mesh geometry.
- **H92 SE(2)-equivariant attention decomposition**: scalar pressure / vector velocity factorization (thorfinn / #2783). Strongest physical inductive bias attempt.

**OOD targets**: camber_rc=68.657 (worst split, +12.8% gap vs in-dist) and re_rand=55.368 are the primary improvement targets. H87 + H88 directly target camber_rc; H86 + H93 target Re-regime generalization.

## Wave 20 / Iter-14 active threads (8/8 students busy)

| Student | PR | Slug | Hypothesis | Family |
|---------|----|----|---------|--------|
| askeladd | #2767 | relative-l1-loss | Per-sample relative L1 normalization | Loss A |
| alphonse | #2769 | camber-curriculum | Difficulty-weighted sampler warmup | Curriculum |
| nezuko | #2771 | camber-cond-layernorm | AdaIN-style camber-only normalization | OOD/arch |
| fern | #2774 | sobolev-surf-loss | dp/ds gradient matching on surface | Loss B |
| tanjiro | #2777 | geompnn | KNN-graph message passing replaces Transolver | Model class A |
| edward | #2781 | masked-node-ssl | Masked-position pretrain + supervised | SSL |
| thorfinn | #2783 | se2-equivariant | SE(2)-equivariant attention decomposition | Model class B |
| frieren | #2785 | bernoulli-consistency | Soft Bernoulli total-head constraint | Loss C / physics |

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
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | #2720 slice_num=96 +13.00%; re-confirmed |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal |
| Gate | ReGLU (ReLU) | SiLU<GELU<ReLU<AbsGLU<SqReLU; ReLU optimum |
| n_hidden | 128 | #2371: quadratic scaling, compute-bound |
| inner_dim | 288 | #2386: 320 over-fits; 256 under-fits |
| OOD domain upsampling | equal weights | #2391: extrapolation gap, not density gap |
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

## Prioritized future ideas (queued after Wave 20)

1. **If H86 (relative L1) wins**: try Huber-relative, smoothL1-relative variants — same gradient-equalization principle, different smoothness
2. **If H87 (curriculum) wins**: try different difficulty scores (Re-only, camber-only, gap-aware) and longer/shorter warmup ranges
3. **If H88 (camber-cond LN) wins**: extend to Re-conditional, full-NACA-conditional variants; try other LayerNorm placement (post-attention)
4. **If H90 (GeoMPNN) wins**: full model-class commitment — explore FNO, GNO, UNO, DeepONet variants
5. **If H91 (SSL pretrain) wins**: longer pretrain phase, different masking ratios, contrastive variants
6. **If H92 (SE(2)-equivariant) wins**: enrich vector channels (d_v=8, 16), try SE(2)-equivariant convolution variants
7. **If H93 (Bernoulli) wins**: layer in incompressibility constraint, momentum balance, full PDE residual
8. **If multiple Wave-20 hypotheses succeed**: orthogonal-combination experiment to stack mechanisms (curriculum + relative L1, etc.)

**Plateau-protocol second escalation if Wave 20 also washes**: complete model-class replacement (FNO or large GNN), data-augmentation campaign (geometry-conditional mixing within camber neighborhoods), or formal NAS / hyperparameter joint search.
