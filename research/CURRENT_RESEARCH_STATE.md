# SENPAI Research State
- 2026-04-27 (initial — pai-2 cluster, icml-appendix-charlie-pai2-r3)
- No directives from human researcher team yet
- Baseline: Not yet established on pai-2. Round 1 sweeps will establish the vanilla anchor and test differentiated hypotheses.

## Context

This is the `icml-appendix-charlie-pai2-r3` track — a fresh TandemFoilSet research track on the pai-2 cluster, running in parallel with r2 and r4.

The kagent_v_students track (PRs #3–#39) produced real results on a different cluster:
- Best result achieved: nl=3, sn=8, nh=2 compound → val_avg/mae_surf_p ≈ 49.44 (from vanilla baseline of ~131.99)
- Key winning ingredients: L1 loss, surf_weight=1, AMP+grad_accum=4, Fourier PE σ=0.7, SwiGLU FFN, n_head=2, slice_num=8 or 16, n_layers=3
- These are NOT implemented in the current target/train.py — students must implement them

## Differentiated Strategy for r3

- r2 track: vanilla hyperparameter sweeps (LR, surf_weight, width, depth, etc.)
- r4 track: replicating proven winners from kagent_v_students + vanilla baseline
- **r3 track (this track)**: 1 vanilla anchor + 7 novel/differentiated hypotheses:
  - L1 loss + surf_weight=1 (most proven single change)
  - Asinh target transform (compress high-Re extremes)
  - Gradient clipping with L1+sw=1
  - nl=3, sn=8 compound (the prior best architectural config)
  - Weight decay sweep for OOD robustness
  - Per-channel pressure upweighting
  - LR warmup before cosine decay

## Current Research Focus

Round 1 goals:
1. Establish vanilla anchor on pai-2 (actual numbers on this hardware)
2. Rapidly test the most proven changes from the prior track
3. Identify the most promising directions for Round 2 compounding

**8 active Round 1 experiments:**

| Student | Hypothesis |
|---------|------------|
| charliepai2r3-alphonse | Vanilla baseline anchor — establish exact pai-2 baseline |
| charliepai2r3-askeladd | L1 loss + surf_weight=1 compound (most proven single change from prior track) |
| charliepai2r3-edward | Asinh target transform — compress high-Re extremes in target space |
| charliepai2r3-fern | Gradient clipping sweep {0.5, 1.0, 2.0} with L1 + sw=1 |
| charliepai2r3-frieren | n_layers=3, slice_num=8 compound (the prior best architectural config) |
| charliepai2r3-nezuko | Weight decay sweep {1e-3, 5e-3, 1e-2} — stronger regularization for OOD robustness |
| charliepai2r3-tanjiro | Per-channel pressure upweighting: separate vol+surf loss for p vs {Ux, Uy} |
| charliepai2r3-thorfinn | Warmup LR: 5-epoch linear warmup then cosine (vs. raw cosine from epoch 0) |

## Potential Next Research Directions

After Round 1 results arrive:

### Physics-informed approaches
1. **Boundary layer thickness proxy**: Wall-normal distance as an extra feature
2. **Wake signature features**: Add wake-angle and downstream velocity-deficit estimates
3. **Reynolds stress tensor features**: Estimate anisotropy tensor components from Re and geometry
4. **Pressure coefficient normalization**: Normalize pressure output by dynamic pressure (½ρU²) per sample

### Architecture directions
5. **Cross-attention between surface and volume nodes**
6. **Multi-scale slice pooling**: Parallel paths with sn=8 and sn=32, merge then decode
7. **n_head=1 single-head attention**: Prior kagent work found single-head outperforms multi-head
8. **Full compound**: n_head=1 × n_layers=3 × slice_num=8

### Loss/training directions
9. **Focal-style surface loss**: Upweight nodes where the model currently has highest surface error
10. **Sobolev loss**: Add gradient consistency penalty for smoother pressure fields
11. **EMA checkpoint averaging**: Exponential moving average for more stable best checkpoint
12. **Full proven recipe**: L1+sw=1+gradient_clipping+nl=3+sn=8+AMP+grad_accum=4
