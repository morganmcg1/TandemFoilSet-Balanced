# SENPAI Research State

- **Date:** 2026-04-27 22:30 UTC
- **Advisor branch:** `icml-appendix-willow-pai2d-r4`
- **Most recent human-team direction:** none received yet on this advisor branch
- **Active research focus:** Round 0 — establish first round of improvements over the vanilla Transolver baseline on TandemFoilSet. Primary metric `val_avg/mae_surf_p`; paper metric `test_avg/mae_surf_p`. Eight students assigned to non-overlapping hypothesis families covering loss reformulation, training-recipe fixes, target/feature engineering, throughput, augmentation, position encoding, and architecture levers.

## Current themes

1. **Scale-aware losses** — exposed weakness: per-sample y-std varies ~40x across the corpus, so a global `y_std` makes high-Re samples dominate gradients. H1 (per-sample y-std normalization) and H3 (Huber on surface) attack this directly.
2. **Training-recipe debt** — no warmup, no AMP/compile, T_max mismatched to actual run length, fp32 only. H2 (warmup + corrected cosine) and H6 (bf16 + compile + larger batch) close these gaps.
3. **Geometry-OOD support** — held-out front-foil cambers test extrapolation to unseen NACA. H5 (random Fourier features) and H7 (z-mirror augmentation) add geometric capacity and free training data.
4. **Architectural capacity** — H8 (slice_num scaling) increases the number of physics-attention partitions for >200K-node meshes.
5. **Surface specialization** — H4 (surface-only norm + signed distance feature) lets the network treat surface and volume distributions separately and adds a boundary-layer-aware geometric input.

## Round 0 assignments (one per student)

| PR | Student | Hypothesis | Bucket | Predicted Δ |
|----|---------|------------|--------|-------------|
| #342 | alphonse | H1: per-sample y-std loss normalization | Loss reformulation | -8% to -18% |
| #343 | askeladd | H6: bf16 + torch.compile + larger batch | Throughput | -3% to -9% |
| #344 | edward | H2: warmup + corrected cosine schedule | Optimization | -3% to -7% |
| #345 | fern | H4: surface-only norm + distance feature | Target/feature engineering | -4% to -10% |
| #346 | frieren | H7: z-mirror augmentation with sign flips | Regularization/sampling | -3% to -8% |
| #347 | nezuko | H5: random Fourier features on (x, z) | Position handling | -2% to -8% |
| #348 | tanjiro | H3: Smooth L1 (Huber) on surface pressure | Loss reformulation | -2% to -6% |
| #349 | thorfinn | H8: slice_num scaling matrix (128/256) | Architecture | -2% to -7% |

## Held in reserve for next round

- H9: pressure-gradient penalty along surface (∇_s p smoothness)
- H10: surf_weight ramp curriculum (5 → 30)
- H11: Re-conditional FiLM modulation between blocks
- H12: EMA of model weights for evaluation

## Potential next research directions

- Combine the round-0 winner with H10 (surf_weight ramp) and H12 (EMA) — both cheap compounding levers.
- After throughput lands (H6), revisit larger architectures with H8 follow-ups and H11 (FiLM).
- If geometry-OOD splits remain stubborn after rounds 0/1, escalate to graph/edge-aware mesh modules or coordinate-network heads.
- If Re-OOD splits remain stubborn, explore Re-conditional separate models or hierarchical heads.
