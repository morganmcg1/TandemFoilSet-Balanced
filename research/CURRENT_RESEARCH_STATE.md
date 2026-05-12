# SENPAI Research State

- **Date:** 2026-05-12
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Round 1 — in flight

Broad sweep over orthogonal levers. All 8 students assigned, no PRs yet review-ready. Each PR targets `icml-appendix-willow-pai2g-48h-r5`.

| Student | PR | Hypothesis | Lever |
|---------|----|-----------|-------|
| alphonse | #1419 | bf16 autocast (mixed precision) | Optimisation / throughput |
| askeladd | #1427 | `surf_weight=30` (3×) | Loss weighting |
| edward | #1430 | `lr=1e-3` + 5% linear warmup + cosine | Optimisation |
| fern | #1436 | Smooth L1 (Huber) loss | Loss form |
| frieren | #1442 | Wider Transolver `n_hidden=192` | Architecture (capacity) |
| nezuko | #1445 | Per-channel surface weighting `(0.5, 0.5, 2.0)` | Loss / metric alignment |
| tanjiro | #1447 | `batch_size=8` (2×) | Gradient noise / throughput |
| thorfinn | #1451 | `slice_num=128` (2×) | Architecture (attention partitioning) |

## Potential round-2 directions (post-results)

- Compound winners of round 1 (loss + optimiser + architecture often orthogonal)
- Surface-aware decoder / dual-head architecture (separate volume and surface heads)
- Re-conditioning: explicit Re-aware embeddings, log-Re aware positional encoding
- Spectral / Fourier neural operator hybrids
- Per-domain auxiliary heads or domain adversarial features
- Curriculum on mesh size or Re regime
- Larger-batch / cross-mesh contrastive pretraining
- Test-time augmentation (TTA) using physical symmetries
- EMA of model weights for eval (~1-3% free gain in many cases)

Researcher-agent literature review is in flight; will land at `research/RESEARCH_IDEAS_2026-05-12_round1.md` and feed round-2 planning.

This is a living document — refresh as round 1 lands and round 2 planning starts.
