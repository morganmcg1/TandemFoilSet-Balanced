<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-04-28 (updated)
- **Most recent research direction from human researcher team:** None (no GitHub Issues received)
- **Track:** icml-appendix-charlie-pai2e-r5
- **Current baseline:** `val_avg/mae_surf_p` = **97.4483** (PR #798, L1 loss, epoch 14)

---

## Current Research Focus and Themes

The round just started. The foundational L1 loss alignment (PR #798, −24.4% improvement) was merged as the very first experiment. All 8 students are now running their first round of experiments testing orthogonal directions on top of L1.

### Active WIP Experiments (Round 2+)

| PR | Student | Hypothesis | Theme |
|----|---------|------------|-------|
| #799 | askeladd | Lion optimizer (sign-based updates vs AdamW) | Optimizer |
| #801 | edward | EMA model averaging (decay=0.995) | Regularization |
| #806 | thorfinn | FiLM domain conditioning (single/rcTandem/cruise) | Architecture |
| #817 | alphonse | surf_weight sweep {10, 15, 20, 25, 30} for L1 | Hyperparameter |
| #822 | nezuko | SmoothL1/Huber loss (beta sweep 0.1/0.3/1.0) | Loss function |
| #823 | tanjiro | asinh pressure target transform (scale sweep) | Target transform |
| #824 | frieren | Gradient clipping + weight decay tuning (max_norm sweep) | Optimization stability |
| #852 | fern | Per-channel L1 loss weighting: amplify pressure in surf_loss | Loss function |
| #857 | askeladd (2nd slot) | Drop-path stochastic depth regularization sweep | Regularization |

---

## Key Context

- **Training budget is tight**: baseline runs hit the 30-min wall-clock timeout at epoch 14/50. Models are NOT converging — training curves still trending down at epoch 14. This is a critical constraint: any experiment that can improve sample-per-second throughput (bf16, batch_size, data loading) has extra leverage here.
- **NaN issue in test_geom_camber_cruise**: `test_avg/mae_surf_p` is partially NaN due to a pre-existing `data/scoring.py` bug where `0 * NaN = NaN` propagates through masked sums. Val metric is reliable; need to resolve this before paper-facing test numbers.
- **surf_weight=20 was tuned for MSE**: the L1 switch likely shifts the optimal surf_weight. The alphonse sweep should resolve this.
- **Domain imbalance**: 3 physical regimes (raceCar single, raceCar tandem, cruise) are equally weighted via the sampler, but cruise has much smaller magnitudes. L1 evened this out somewhat versus MSE.

---

## Potential Next Research Directions

### High Priority (likely high-value)
1. **Weighted surface pressure loss** — explicitly upweight the pressure channel within the surface loss beyond the current `surf_weight` scalar. Surface p is the primary metric but Ux/Uy still dilute gradients.
2. **Larger model capacity** — n_hidden=256, n_layers=6, or n_head=8. The baseline is quite small (128 hidden, 5 layers). With 96GB VRAM there's headroom to go bigger.
3. **Cosine warmup schedule** — the baseline LR schedule has no warmup; adding a 3-epoch linear warmup with eta_min=5e-5 could stabilize early L1 training which has sharper gradients near zero.
4. **Per-channel normalization tuning** — `y_std` normalization is shared; channel-specific scaling might better balance Ux/Uy/p loss contributions.
5. **Batch size increase** — current batch_size=4 is low; bf16 could enable batch_size=8, doubling throughput and getting more epochs in the timeout window.

### Medium Priority (architecture exploration)
6. **FiLM conditioning** (in-flight, PR #806) — condition model on domain ID (single vs rcTandem vs cruise) to specialize internal representations.
7. **Slice-aware attention** — vary slice_num (currently 64) to find better slice granularity for capturing foil surface detail.
8. **Residual connections in preprocessing MLP** — additional MLP layers with residual skip might enrich feature embeddings for the complex 24-dim input space.
9. **Graph-based foil 1/2 distinction** — currently the model can't distinguish foil 1 from foil 2 surface nodes; encoding this could help tandem predictions.
10. **Attention over surface-only nodes** — dedicated attention pathway for the sparser surface node set, complementing volumetric attention.

### Bold/Risky Ideas
11. **Physics-informed loss** — add a continuity equation residual term (div(U)=0 for incompressible flow) as an auxiliary loss on volume predictions.
12. **Multi-scale Transolver** — process coarse/fine mesh scales with different attention modules, merging predictions.
13. **Test-time normalization** — normalize per-sample rather than using global stats, which might reduce Re-regime bias.
14. **Geometric augmentation** — small random perturbations to NACA parameters at training time for robustness.
15. **NaN-fix in scoring.py** — separate fix to replace `0 * NaN` with `torch.where(mask, err, 0)`, enabling clean test_geom_camber_cruise pressure metrics.

---

## Summary

9 students are actively running experiments covering optimizer, regularization, architecture, and loss-tuning angles simultaneously. Next round should focus on model capacity scaling (larger hidden dims, more layers) and cosine LR warmup once these results are in. The next advisor review will rank results and compound winners.
