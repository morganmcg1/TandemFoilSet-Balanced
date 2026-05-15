# SENPAI Research State

- **Updated:** 2026-05-15 12:35 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch yet. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current research focus

**Round 1 (round-trip baseline + first knob sweep).** Establish numerical
baseline and probe the four highest-leverage knobs in parallel:

1. **Loss / metric alignment** (MSE → SmoothL1 / Huber; higher surf_weight;
   per-channel pressure boost).
2. **Mesh-resolution capacity** (slice_num 64 → 128; mlp_ratio 2 → 4).
3. **Throughput** (bf16 autocast for more iterations per epoch).
4. **Optimization stability** (LR warmup; possibly grad clipping).

Each round 1 PR changes exactly one knob from the baseline config. We assign
one student to a clean baseline reproduction so subsequent comparisons have a
ground-truth number on this hardware.

## Potential next research directions

After round 1 numbers settle, candidate themes:

- **Loss compounding**: stack the winning round 1 loss formulation with the
  winning capacity setting; explore per-domain reweighting.
- **Stronger positional encoding**: `unified_pos=True` with various ref grids,
  Fourier / RFF on (x, z) input dims.
- **Decoder ergonomics**: per-channel output heads; surface-specific decoder
  trunk; surface-only auxiliary loss.
- **Regularization for OOD**: EMA / SWA of weights, dropout sweep, stochastic
  depth on transformer blocks.
- **Capacity / scaling**: n_hidden 128 → 192/256, n_layers 5 → 6/7, slice_num
  → 256 (paired with mlp_ratio).
- **Optimizer**: Lion (manual implementation if needed since PyTorch lacks a
  native Lion in older versions), Sophia, Adan; LR schedule variants.
- **Sampling / augmentation**: mesh-node subsampling during training, AoA-
  symmetric augmentation if physically valid for symmetric foils, per-domain
  curriculum.
- **Physical priors**: divergence-free penalty on (Ux, Uy), boundary-layer-
  aware weighting (sample weighting by distance to nearest surface node).

## Plateau plan (if it happens)

If 5+ consecutive rounds show no improvement, escalate per the program-level
plateau protocol: from knob tuning → architecture → loss reformulation → data
representation. Use the researcher-agent to mine new literature and try bigger
swings.
