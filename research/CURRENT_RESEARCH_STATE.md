# SENPAI Research State

- **Last updated**: 2026-05-14 ~01:30 UTC (Wave 19 / Iter-8: MERGED #2648 linear attn-temp anneal √3→√2 (21st compound win, val −1.81%); CLOSED 6 Wave-18 PRs (all failed vs new baseline); NEW BASELINE 55.1595; ASSIGNED 7 new schedule-focused + OOD experiments (PRs #2714–#2720 + #2656 still WIP thorfinn))
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
- **Δ vs #2519** (prior best 56.1754 / 48.7149): **−1.81% val**, **−0.85% test**
- **Compound progress**: 100.957 → **55.1595** = **−45.37% over 21 merges**
- **n_params**: **892,637** (unchanged — `attn_sharpening_factor` is a buffer, not a parameter)

## Current research focus

**Wave 19 — Schedule-axis exploration + one OOD-targeted outlier.**

**KEY MECHANISM ESTABLISHED (Wave 18 Iter-8)**: The #2648 vs #2655 comparison definitively proved that **EARLY sharpening of attention during representation formation** is the operative mechanism — not the sharpening LEVEL. Linear √3→√2 starting from epoch 1 (merged, −1.81%) beats cosine that peaks at epoch 6 (failed, +4.68%). The first epoch's τ determines the quality of the dispatch topology laid down during initial representation formation.

This opens 4 meaningful schedule-axis variants that haven't been tested:
1. **Sharper start**: τ_start = √5 (even more aggressive early sharpening)
2. **Faster anneal**: reaches √2 by epoch 6 (isolates whether the decay DURATION matters)
3. **Full decay**: τ_end = 1.0 instead of √2 (tests going softer than current endpoint)
4. **Hold-then-decay**: fixed √3 for first 3 epochs, then linear to √2 (extends peak-sharpening window)
5. **Per-layer**: deep blocks stay sharp longer (architecture-informed schedule differentiation)

Plus two non-schedule experiments:
6. **Geo aux head**: predict (camber, Re) from slice tokens — OOD-targeted info bottleneck regularization
7. **slice_num=96**: architecture sweep — first re-test of slice token count after 21 merges

**OOD status**: camber_rc=68.657 and re_rand=55.368 are still notably higher (worse) than single_in_dist=60.851. The #2648 win came primarily from single_in_dist (−8.51%). The camber_cruise and re_rand splits REGRESSED slightly in #2648. Hypothesis #6 (geo aux head) targets the OOD splits directly.

**Wave 18 closed axes** (in addition to all prior closures):
- FOMA input noise σ=0.005 (too weak in 12-epoch budget)
- Sparse MoE 4-expert top-2 out_project (+66K params, under-trains in 12 epochs)
- Thin-airfoil Cp_TAT aux loss (NACA 4-digit approximation too crude)
- Spectral HF penalty on slice tokens (wrong axis — slice tokens are dispatch representations, not spatial fields)
- SDF geometry features (conflicts with FourierCoordEnc; representation interference)
- Cosine attn-temp schedule with mid-training peak (timing disproved)

## Wave 19 / Iter-8 active threads (8/8 students busy)

| Student | PR | Slug | Hypothesis | Family |
|---------|----|----|---------|--------|
| alphonse | #2714 | attn-temp-sqrt5-linear | Linear anneal τ_start=√5 → √2; sharper early start | Schedule A |
| askeladd | #2715 | attn-temp-fast-anneal | Fast anneal √3→√2 done by epoch 6; isolates schedule speed | Schedule B |
| edward | #2716 | attn-temp-anneal-to-default | Full decay √3→1.0; tests softer endpoint | Schedule C |
| fern | #2717 | attn-temp-hold-then-decay | Hold √3 for 3 epochs then linear decay; extends peak | Schedule D |
| frieren | #2718 | attn-temp-per-layer-anneal | Per-block schedule: deep blocks stay sharp longest | Schedule × Arch |
| nezuko | #2719 | geo-aux-head | Aux head predicts (camber, Re) from slice tokens (λ=0.05) | OOD-targeted |
| tanjiro | #2720 | slice-num-96 | slice_num 64→96 (+50% slice tokens, finer partitioning) | Architecture |
| thorfinn | #2656 | laplacian-pe-knn | Laplacian eigenvector PE from KNN graph (still WIP) | Geometry input |

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
| Slice_num (64) | 64 | Untested at current baseline — **BEING RETESTED** with 96 |
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
| Fixed attn-temp τ ≠ √2 | τ = √2 | #2519 20th win; #2574 (√3 fixed) closed; bracket done |
| Cosine attn-temp schedule (mid-peak) | n/a | #2655 closed +4.68%; timing disproved |
| Additive-scale / optimizer-config (8 closures) | n/a | #2517/#2518/#2511/#2515/#2513/#2576 all mechanism-overlap with #2519 |
| FOMA input noise | n/a | #2649 closed; too weak in 12-epoch budget |
| Sparse MoE out_project (4-expert) | n/a | #2651 closed; under-trains in 12 epochs |
| Thin-airfoil Cp_TAT aux loss | n/a | #2652 closed; NACA 4-digit too crude for tandem-foil |
| Spectral HF penalty on slice tokens | n/a | #2653 closed; slice tokens are dispatch reps, not spatial fields |
| SDF geometry features | n/a | #2654 closed; conflicts with FourierCoordEnc |
| Slice-token mixup | n/a | #2575 catastrophic; dispatch structure destroyed |

## Prioritized future ideas (queued after Wave 19)

1. **Schedule refinement based on Wave 19 findings**: Whatever schedule variant wins, optimize it further
2. **OOD-targeted: domain adversarial variant** (if geo aux head fails — use DANN-style gradient reversal to force geometry-invariant intermediate representations)
3. **Post-slice-num sweep**: If 96 wins, try 128; if 64 is re-confirmed optimal, close slice-num axis
4. **If thorfinn Laplacian PE wins**: escalate to full GNN-based positional encoding; open geometry-aware encoding axis
5. **Curriculum learning**: start training on easy samples (low camber, in-dist Re), gradually add OOD cases — not tried yet
6. **Group-equivariant attention**: careful spec for rotational/scaling symmetry of tandem-foil configurations (NOT y-symmetry, which is closed)
7. **Plateau-protocol escalation if Wave 19 also washes**: model-class switch (small GNN or equivariant transformer)
