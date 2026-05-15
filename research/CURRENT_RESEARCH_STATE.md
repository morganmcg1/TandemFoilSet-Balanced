# SENPAI Research State

- **Date:** 2026-05-15 (Round 5, 48h launch, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current research focus

Fresh start on `icml-appendix-willow-pai2i-48h-r5`. Baseline is Transolver (~1.5M params) trained on TandemFoilSet with MSE loss, `surf_weight=10`, AdamW + cosine LR. No prior merged experiments on this branch.

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 splits (in-dist, camber-OOD×2, Re-OOD). Test equivalent: `test_avg/mae_surf_p`.

**Binding constraint per run:** SENPAI_TIMEOUT_MINUTES=30.0, SENPAI_MAX_EPOCHS=50, 1 GPU per student.

## Round 1 — Active PRs (opened 2026-05-15)

Eight first-round hypotheses cover independent axes:

| PR | Student | Hypothesis | Expected delta |
|----|---------|------------|----------------|
| #3098 | alphonse | SmoothL1/Huber loss for heavy-tailed targets | -3 to -8% |
| #3100 | askeladd | Transolver scale-up (n_hidden 128→192/256) | -5 to -15% |
| #3103 | edward | Slice-num scaling (64→128/192 physics tokens) | -2 to -6% |
| #3105 | fern | Linear warmup + cosine LR (step-level) | -3 to -7% |
| #3109 | frieren | bf16 + bigger batch (bs=4→8/16) | -2 to -5% |
| #3114 | nezuko | Gradient clipping + EMA weights | -2 to -5% |
| #3118 | tanjiro | Per-channel surface loss (bias toward p) | -3 to -8% |
| #3123 | thorfinn | Random Fourier positional features on (x,z) | -3 to -7% |

Each PR includes an internal baseline arm (arm A) for within-PR comparison.

## Potential next research directions (Round 2+)

Once Round 1 results are in, compound the winners. Priority order:

1. **Combine the top two winners** — architecture scale-up + best loss variant, or scale-up + EMA.
2. **Larger architecture grid** — push n_hidden up to 320/512 if 192→256 helps significantly.
3. **Relative error loss** — `|pred - true| / (|true| + ε)` to handle cross-Re heteroskedasticity.
4. **Sobolev loss** — penalize gradient of predictions on surface nodes (physics-regularized).
5. **Surface-node-only training split** — ablate whether predicting volume nodes at all helps pressure.
6. **Domain-conditional normalization** — separate y_mean/y_std per domain group (raceCar vs cruise).
7. **Attention head dimensionality sweep** — dim_head=64 vs 96 vs 128 at fixed param budget.
8. **Stochastic depth** — drop layers during training for implicit ensembling.
9. **Multi-scale slice decomposition** — coarse + fine slice levels per attention block.
10. **Larger batch + more epochs** — if bf16 frees enough VRAM for bs=16, try --epochs 50 with full cosine decay in budget.
