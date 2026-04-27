# SENPAI Research State — willow-pai2c-r3

- **Date / time:** 2026-04-27
- **Advisor branch:** `icml-appendix-willow-pai2c-r3`
- **Target:** TandemFoilSet — minimize `val_avg/mae_surf_p` (mean surface pressure MAE across 4 splits)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-r3`

## Most-recent direction from human researcher team
None recorded yet (no human-tagged GitHub Issues open at boot). Will continue polling.

## Current research focus
First round on a fresh advisor branch. No `BASELINE.md` yet — round-1 PRs each include a baseline trial and an intervention trial in the same W&B group, so the round establishes both the baseline reference and 8 candidate improvements simultaneously.

The Transolver baseline (5L × 128h × 4 heads × slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, batch=4, surf_weight=10, AdamW + cosine, 50 epochs, 30-min cap) is well-tuned but untested on this branch. Round 1 attacks 8 orthogonal mechanisms so wins compose.

## Round 1 — in flight (8 PRs, 1 per student)

| # | Student | PR | Mechanism | Predicted Δ on `val_avg/mae_surf_p` |
|---|---------|-----|-----------|-------|
| 1 | alphonse | #230 | Width capacity: `n_hidden` 128 → 256 | -5% to -15% |
| 2 | askeladd | #232 | Multi-task uncertainty weighting (Kendall log-σ × 5) | -3% to -8% |
| 3 | edward | #233 | Surface-token cross-attention decoder head | -7% to -15% |
| 4 | fern | #234 | Slice-num scaling: 64 → 128 physics tokens | -3% to -8% |
| 5 | frieren | #235 | EMA weight averaging (decay = 0.999) | -2% to -5% |
| 6 | nezuko | #236 | Re-conditioned per-sample `y_std` (closed-form, 6 bins) | -5% to -12% |
| 7 | tanjiro | #237 | Re/AoA/gap FiLM conditioning per block | -3% to -8% (esp. `val_re_rand`) |
| 8 | thorfinn | #238 | Depth scaling: `n_layers` 5 → 8 | -3% to -8% |

Each PR has a paired baseline trial in the same W&B group, so the round produces 8 baseline runs (variance estimate) + 8 interventions.

## Potential next research directions (round 2+)

**Heavy-tail / output parameterization**
- Predict pressure as `(sign, log-magnitude)` decomposition (mu-law-like).
- Predict residual from a simple analytic potential-flow prior.
- Sliced-Wasserstein loss on per-sample pressure distribution.

**Architecture / decoders**
- Mixture-of-Experts decoder gated by Re + foil regime.
- Foil-1 / foil-2 / volume token-type embeddings.
- Hard top-k slice attention (sparse) → enables `slice_num=256+`.

**Loss / regularization**
- Energy / divergence physical regularizer on velocity.
- Geometry-aware contrastive pretext (mask + reconstruct wake nodes).
- PCGrad gradient projection on top of uncertainty weighting.

**Optimization**
- SWA on top of EMA in the last 30% of training.
- Layer-wise LR decay.
- Lion / Adan optimizer A/B.

**Curriculum / data**
- Re curriculum: low-Re early, expand to full range.
- x-flip augmentation on cruise samples (reflection-symmetric).
- Test-time augmentation with x-flip on cruise.

**Composability** (if multiple round-1 winners merge)
- Width 256 + slice_num 128 + depth 8 stacked.
- Width 256 + EMA + uncertainty weighting.
- Re-conditioned rescaling + FiLM (orthogonal — both target Re generalization).

The full menu is in `research/RESEARCH_IDEAS_2026-04-27_round2.md`.
