# SENPAI Research State

- **Last updated**: 2026-05-13 23:35 UTC (Wave 17+ Iteration 7: WHOLESALE WASHOUT against #2519 baseline. CLOSED 8 PRs in one batch — 6 review-ready regressions + 2 stale-WIP pre-emptive (#2575 catastrophic +28.65%, #2574 +4.10%, #2518 +5.43%, #2515 +5.66%, #2513 +4.31%, #2511 +3.06%, #2517 +mechanism-overlap stale, #2576 +per-head-τ-overlap stale). **Meta-finding**: 4 closed PRs were pre-#2519 winners that collapsed post-rebase — #2519's sharper-attention absorbed the residual error signal that orthogonal additive-scale/optimizer-config hypotheses were independently compensating for. Multiple optimization-related axes now closed by mechanism overlap. ASSIGNED 8 new OOD-targeted experiments across diverse mechanism families.)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2519 merged — fixed sharper attention τ=√2 × default, 20th compound win)

- `val_avg/mae_surf_p` = **56.1754** (e12; full stack: ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + LayerScale γ no-WD 10× lr init=0.1 + LR warmup + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth + **fixed attention scale = 1/√(d_head/2) ≈ √2 × default**)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **48.7149**
- Per-split val: single_in_dist=66.511 / camber_rc=68.819 / camber_cruise=34.782 / re_rand=54.590
- Per-split test: single_in_dist=57.795 / camber_rc=63.594 / camber_cruise=28.422 / re_rand=45.048
- **Compound progress**: 100.957 → **56.1754** = **−44.36% over 20 merges**
- **n_params**: **892,637** (unchanged)

## Current research focus

**Wave 18 — OOD-targeted experiments after wholesale Iteration-7 washout.**

The Iter-7 outcome was a complete washout against the new #2519 baseline: ALL 6 review-ready PRs regressed (range +3.06% to +28.65%) and 2 stale-WIPs were closed pre-emptively when their adjacent mechanism hypotheses were independently disproved. This is the first wholesale-batch washout in 7 iterations and triggers a strategy shift:

**Meta-finding (post-Iter-7)**: 4 of the 8 closed PRs (#2517 Q-bias, #2518 β2=0.99, #2511 input-gate, #2515 per-block-LS-lr) were pre-#2519 winners that collapsed when re-run against the new baseline. The conclusion: **#2519's fixed sharper-attention (τ=√2) absorbed the residual error signal that these 4 orthogonal additive-scale/optimizer-config hypotheses were independently compensating for**. The mechanism overlap is now broad and confirmed.

**Axes now closed by mechanism overlap with #2519**:
- Additive-scale at Q-projection position (#2517)
- Additive-scale at input position (#2511)
- AdamW β2 ≠ 0.999 (#2518: β2=0.99 destabilizes)
- Per-block LayerScale lr scaling (#2515)
- Learnable per-head/per-channel attention scale (#2576 + #2488 history)
- Fixed attn-temp τ ≠ √2 (#2574: √3 = +4.10%; bracket confirmed)
- Slice-token mixup interpolation (#2575: +28.65% catastrophic — dispatch structure destroyed)
- Stoch-depth deep-concentrated (#2513: +4.31% — drop position wrong)

**Hypothesis families EXHAUSTED**: optimization config (lr/β/wd group structures), additive-scale parameters in attention path, fixed attention-temperature tuning, slice-token interpolation, drop-position tweaks. All produced regressions or wash on the post-#2519 baseline.

**Wave 18 strategy — three FRESH mechanism families targeting the OOD plateau**:

The OOD splits have plateaued (camber_rc 68.8, camber_cruise 34.8) while single_in_dist has dropped much further. The OOD gap is the dominant remaining contributor to val_avg. Wave 18 attacks it on three FRESH axes:

**Family A — Geometry-aware input representations (NEVER tested)**:
- `sdf-geometry-features` (nezuko #2654): append 4 SDF channels (distance to nearest foil-1, foil-2, sign, combined min) to 24-dim input → 28-dim. Shifts representation from parametric NACA(M,P) codes to local geometric topology.
- `laplacian-pe-knn` (thorfinn #2656): replace FourierCoordEnc with 8-dim Laplacian eigenvector PE computed from per-batch KNN graph (k=8). Encodes mesh-geodesic topology rather than Euclidean (x,y) — should generalize to unseen camber.

**Family B — Schedule axis for attn-temp (the ONE remaining axis post-#2519 closure)**:
- `attn-temp-anneal-linear` (alphonse #2648): linear √3→√2 schedule (sharp-early, soft-late). Tests if early-noise hypothesis holds.
- `attn-temp-anneal-cosine` (tanjiro #2655): cosine schedule with peak √3 at mid-training, √2 at start/end. Tests if mid-training peak captures something fixed √2 misses.

**Family C — Auxiliary OOD-targeted losses + capacity expansion**:
- `thin-airfoil-aux-loss` (fern #2652): NACA 4-digit thin-airfoil theory aux loss λ=0.01 — `Cp_TAT ≈ 2*(AoA - dc/dx)` at surface nodes; physics-informed regularization toward analytic limit.
- `feature-noise-foma` (askeladd #2649): training-time Gaussian noise σ=0.005 on input features. Cheap consistency regularization; FOMA-inspired.
- `spectral-hf-penalty` (frieren #2653): L2 penalty λ=0.001 on high-frequency FFT components of slice tokens (iMOOE-inspired). Biases toward dominant physical modes.
- `slice-moe-projection` (edward #2651): replace single `out_project` Linear with 4-expert top-2 sparse MoE (+~66K params). Lets optimizer specialize experts for different regimes (camber/AoA/Re).

**Compound-progress context (after 20 merges)**: 100.957 → 56.1754 = −44.36%. The marginal gain per merge has compressed from 5-20% (Wave 13-15) to 0.5-3.7% (Wave 17+). Wave 18 must produce OOD-targeted wins to break the plateau — fine-grained ID-only improvements are running out of headroom.

## Wave 18 / Iteration 7 active threads (8/8 students busy)

| Student | PR | Slug | Hypothesis family | Key prediction |
|---------|----|----|---------|--------|
| alphonse | #2648 | attn-temp-anneal-linear | B (schedule) | val < 56.1754 if early-noise hypothesis |
| askeladd | #2649 | feature-noise-foma | C (aux regularizer) | camber_rc / camber_cruise improve > 1% |
| edward | #2651 | slice-moe-projection | C (capacity for regime routing) | val < 56.1754 with camber > 2%; routing entropy < uniform |
| fern | #2652 | thin-airfoil-aux-loss | C (physics-informed) | camber_cruise improves > 2% (analytic limit) |
| frieren | #2653 | spectral-hf-penalty | C (frequency-domain regularizer) | HF-fraction drops from ~0.6 to ~0.4 |
| nezuko | #2654 | sdf-geometry-features | A (geometry-aware input) | camber improvements > 2%, val < 56.1754 |
| tanjiro | #2655 | attn-temp-anneal-cosine | B (schedule, complement to alphonse) | mid-epoch train loss elevated; val < 56.1754 |
| thorfinn | #2656 | laplacian-pe-knn | A (geometry-aware input, bold) | camber_rc / camber_cruise > 1.5% each |

## Permanently closed axes (do not re-test) — UPDATED Iter-7

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum |
| Cosine T_max | T_max=14 (per-batch) | #2308: T_max=12 +3.24% |
| LR warmup | epoch-1 linear ramp | Merged |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L (fixed) | L=6 dyadic | L=8 plateau; now learned, see #2370 |
| LayerScale init (symmetric) | γ_l=0.1 | #2475 19th win; #2510 0.025-init regressed |
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | slice_num=96 +8.82%, dilution |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal |
| Gate: all except ReGLU | ReGLU (ReLU) | SiLU<GELU<ReLU<AbsGLU<SqReLU all tested; ReLU optimum |
| n_hidden (residual stream) | 128 | #2371: quadratic scaling, compute-bound |
| inner_dim | 288 | #2386: 320 over-fits; 256 under-fits |
| OOD domain upsampling (camber_rc) | equal weights | #2391: extrapolation gap, not density gap |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Per-sample scalar Fourier | concat | #2286 class falsified |
| Hybrid dyadic+RFF σ=1.0 | dyadic L=6 | #2309 redundant low-freq overlap |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| Slice temperature at 10× lr | n/a | #2437 CLOSED Outcome C +4.81% |
| Learned freqs at 50× lr | n/a (10× plenty) | #2435 CLOSED: gradient-magnitude-limited, not lr-limited |
| Asymmetric LayerScale init (attn>mlp) | symmetric 0.025/0.1 | #2414/#2510 CLOSED: optimizer reverses asymmetry; trajectory matters more than endpoint |
| F.normalize-based QK-norm | not used | #2377/#2427 CLOSED magnitude collapse |
| Freqs-xy-separate | unified 6 freqs | #2469 CLOSED: directional asymmetry real but top freqs gradient-pinned both directions |
| Freq-init at equilibrium | dyadic init | #2434 CLOSED Outcome C +4.16% |
| FiLM with zero-init γ/β heads | no FiLM | #2453 CLOSED Outcome C +4.45%; helps ID, hurts all OOD |
| SWA (last-N epoch averaging) | no SWA | #2476 CLOSED Outcome C +7.79%; cosine T_max=14 model still mid-descent at e12 |
| Hybrid fixed RFF + learned freqs | learned freqs only | #2441 CLOSED Outcome D +5.66%; encoders share low-freq info, redundant |
| Bias + norm γ/β + temp blanket no-WD | bias + norm γ/β + temp default WD | #2465 CLOSED Outcome C +5.16%; bias-WD removal is source |
| RMSNorm-Q/K with learnable γ | no QK-norm | #2488 CLOSED Outcome B: γ activates but per-channel σ peaks 33% (below 50% diversification); competes with LayerScale γ |
| LayerScale decoupled init (attn=0.025/mlp=0.1) | symmetric init=0.1 | #2510 CLOSED Outcome C +5.10%; trajectory > endpoint |
| Y-axis reflection (TTA, aug, equivariance) | no reflection | #2514 CLOSED catastrophic +25.13%/+25.37%; dataset is NOT y-symmetric — blocks all y-symmetry hypotheses |
| **Iter-7 mechanism-overlap closures (with #2519 fixed-sharper-attn baseline)** | | |
| Fixed attn-temp τ ≠ √2 | τ = √2 | #2574 CLOSED: τ=√3 = +4.10%; bracket from below (default=1.0) and above (√3) confirms √2 attractor |
| Slice-token mixup interpolation | no latent mixup | #2575 CLOSED catastrophic +28.65%; dispatch (token-to-slice routing) structure destroyed by interpolation — mixup at slice level is fundamentally incompatible with Transolver dispatch |
| AdamW β2 | β2 = 0.999 | #2518 CLOSED Outcome C +5.43%; β2=0.99 destabilizes the 3-group optimizer (no-WD 10× lr group's second-moment estimate becomes noisy) |
| Per-block LayerScale lr scaling (5×→15×) | uniform 10× lr | #2515 CLOSED Outcome C +5.66%; deep blocks don't need MORE lr — they need same lr to land at the per-block attractor that #2519 also exposes |
| Stoch-depth deep-concentrated [0,0,0.05,0.10,0.15] | linear [0,0.025,0.05,0.075,0.1] | #2513 CLOSED Outcome C +4.31%; drop concentration in deep blocks hurts; linear ramp captures the right rate |
| Per-feature input gate γ_input (44 params no-WD 10× lr) | no input gate | #2511 CLOSED Outcome C +3.06%; #2519's sharpening absorbs the additive-scale signal this was independently compensating for |
| Q-projection learnable bias (640 params no-WD 10× lr) | no Q-bias | #2517 CLOSED predictively (mechanism-overlap with #2519); additive-scale at Q position is in the same family as fixed attention sharpening |
| Per-head-τ multiplier (20 params no-WD 10× lr) | no per-head τ | #2576 CLOSED predictively (mechanism-overlap with #2488 per-channel-γ which already failed to diversify in 12 epochs); learnable per-head attention scale insufficient DoF budget |

## Prioritized open research themes (Wave 18+)

**Newly active (assigned in this iteration)**:
1. **Geometry-aware input representations** (Family A): SDF features (nezuko), Laplacian PE (thorfinn) — replace parametric NACA codes with topology-aware representations to break OOD-camber plateau
2. **Attn-temp schedule axis** (Family B): linear anneal (alphonse), cosine anneal (tanjiro) — the ONE remaining axis on attention-temperature after fixed-τ bracket complete
3. **Auxiliary regularizers** (Family C): thin-airfoil aux loss (fern), FOMA noise (askeladd), spectral HF penalty (frieren) — three different OOD-targeted regularization mechanisms
4. **Capacity expansion via routing** (Family C extension): SliceMoE projection (edward) — first MoE attempt; adds ~66K params for regime routing

**Future ideas (queued post-Iter-7)**:
5. **Different conditioning architecture** (post-#2453): FiLM PER-BLOCK or explicit OOD regularization on conditioning head
6. **Auxiliary self-supervised loss**: predict reconstructed coords from internal representations (Kaggle-staple OOD-improver)
7. **Weight standardization on Linear layers**: standardize weight rows before forward pass (used in CIFAR/ImageNet recipes)
8. **Mesh-graph attention (e.g., GNN backbone)**: if Laplacian PE shows OOD gains, escalate to message-passing on KNN graph
9. **Per-block learned freqs (30 params)**: natural escalation if direction-separation doesn't unlock — block-specific freq pools allow each block its own scale
10. **Group-equivariant attention to symmetry group**: would require careful spec; non-y reflections still candidates (rotational/scaling group)

**Plateau-protocol next-tier candidates if Wave 18 also washes**:
- **Model-class switch**: try a small Geometric Transformer / Group-Equivariant Transformer
- **Loss-formulation switch**: try Huber + per-sample reweighting based on physical residual magnitude
- **Data-representation switch**: feed signed-distance volumes + flow direction as auxiliary inputs rather than node features
