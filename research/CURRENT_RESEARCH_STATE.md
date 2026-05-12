# SENPAI Research State

- **Date:** 2026-05-12
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current research focus (round 1)

No prior PRs on this branch — round 1 is a broad, well-grounded sweep over orthogonal levers that each promise measurable signal within a 30-min training run. Themes:

- **Optimization** — learning rate / warmup, weight decay, mixed precision, batch size
- **Loss formulation** — surface vs volume weighting, Huber/robust loss, per-channel rebalancing
- **Architecture** — Transolver depth / width / slice_num
- **Data** — augmentation that respects per-domain physical symmetries
- **Training strategy** — gradient accumulation, EMA, dropout

The aim of round 1 is to populate the BASELINE.md leaderboard with credible reference points and identify which lever is most worth pushing in round 2.

## Potential follow-up directions

- Compound the round-1 winners (loss + optimizer + architecture changes are often orthogonal)
- Surface-aware decoder / dual-head architecture (separate volume and surface decoders)
- Re-conditioning: explicit Re-aware embeddings, log-Re aware positional encoding
- Spectral / Fourier neural operator hybrids
- Per-domain auxiliary heads / domain adversarial features
- Curriculum on mesh size or Re regime
- Larger-batch / cross-mesh contrastive pretraining
- Test-time augmentation (TTA) using physical symmetries

Update this file as round 1 results come in. Treat it as a living document.
