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

Informed by literature scan (`research/RESEARCH_IDEAS_2026-05-15_initial.md`).

**Compound winners from Round 1.** First priority: combine the best loss + architecture + optimization winners into a single PR (Round 2A).

**New mechanisms not covered in Round 1:**

1. **FiLM conditioning on log(Re)** (AeroDiT 2024, arXiv:2412.17394). Directly addresses `val_re_rand` weakness — add `(gamma, beta) = MLP(log_re_global)` modulation in each Transolver block. Higher effort (~50 LOC) but uniquely targets cross-Re generalization.
2. **Per-sample relative L2 loss** (FNO/GNOT standard). Divide each sample's loss by `y.std(dim=1)` to equalize gradient magnitude across the Re spectrum. Distinct mechanism from Huber.
3. **1st-Order SAM** (Kaddour 2024, arXiv:2411.01714). Flat minima → better OOD on 3 of 4 val splits. Cost: 2x forward passes — must budget epochs against 30-min wall clock.
4. **AoA reflection symmetry augmentation** (NeuralFoil 2025, arXiv:2503.16323). Doubles RaceCar single-foil samples via horizontal mirror. Gotcha: dsdf features (dims 4-11) are distance-based — verify symmetry handling for those before flipping.
5. **STRING 2D RoPE** (Schenck 2025, arXiv:2502.02562, ICML spotlight). Relative spatial encoding in PhysicsAttention. Higher implementation risk — defer to Round 3 if simpler ideas plateau.

**Other levers if plateau hits:**

- **Larger architecture grid** — push n_hidden up to 320/512 if 192→256 helps significantly.
- **Sobolev loss** — penalize gradient of predictions on surface nodes (physics-regularized).
- **Domain-conditional normalization** — separate y_mean/y_std per domain group.
- **DPOT-style denoising** (arXiv:2403.03542) — diffusion-style training paradigm.
- **Stochastic depth** — drop layers during training for implicit ensembling.
- **Multi-scale slice decomposition** — coarse + fine slice levels per attention block.
