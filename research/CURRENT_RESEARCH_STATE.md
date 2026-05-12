# SENPAI Research State

- **Date:** 2026-05-12
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — compound winner)

**PR #1436 — fern Huber + bf16** (merged 2026-05-12, stacked on top of #1419):
- `val_avg/mae_surf_p = 96.4863` (epoch 16; vs 109.29 alphonse bf16 → −11.7%)
- `test_avg/mae_surf_p = 86.3326` (vs 97.67 → −11.6%)
- Config: Huber loss (β=1.0) + bf16 autocast + NaN scoring fix, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- ~18 epochs / 30 min (bf16 = ~100 s/epoch)

**Round-1 underlying baseline (also merged):** PR #1419 alphonse bf16: val=109.29, test=97.67.

**Compounding observed**: bf16 (epoch budget) + Huber (loss-shape) stack to −12 MAE improvement on val. Per-split improvements (−9 to −13 MAE) are uniform across all 4 splits, so the gains aren't from one split alone.

## Active round-2 experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1544 | `mlp_ratio=4` (2× MLP width) | Architecture (capacity) | WIP | based on bf16-only baseline |
| askeladd | #1427 | `surf_weight=30` (3×) | Loss weighting | WIP (stale, ~2h) | NaN-fix rerun pending |
| edward | #1546 | `n_layers=8` (Transolver paper default) | Architecture (depth) | WIP | based on bf16-only baseline |
| fern | #1606 | EMA of model weights (decay=0.999) | Weight averaging | WIP | round-3, on top of Huber+bf16 |
| frieren | #1442 | Wider `n_hidden=192` | Architecture (width) | WIP (rebasing) | based on bf16-only baseline |
| nezuko | #1445 | Per-channel surf weights `(0.5, 0.5, 2.0)` | Loss / metric alignment | WIP (stale, ~2h) | |
| tanjiro | #1534 | Gradient clipping `max_norm=1.0` | Gradient stability | WIP (rebasing) | based on bf16-only baseline |
| thorfinn | #1550 | `slice_num=96` (clean test at bs=4+bf16) | Architecture (attention) | WIP | based on bf16-only baseline |

**Important**: round-2 experiments above were started against the **bf16-only** baseline (109.29 val). With Huber now merged (96.49 val), any of these PRs that don't reach val<96.49 will need to either rebase + retest on Huber+bf16 or be closed if they're clearly orthogonal-but-non-additive.

## Key observations from round 1

1. **bf16 is the dominant lever** — 18 epochs/30 min vs 11-14 for fp32. Under our hard wall-clock cap, per-epoch speed is the strongest convergence driver. Now merged.
2. **Huber loss shows remarkable signal** — fern achieved val=109.45 at fp32 14 epochs, nearly matching bf16 at 18 epochs. This implies ~4 epochs of effective speedup from loss-shape alignment with MAE metric. Huber + bf16 expected to compound to ~95-105.
3. **Gradient norms are massive without clipping** — tanjiro's grad-clip diagnostic: 5250/5250 steps clipped at `max_norm=1.0`, max pre-clip norm = 837. AdamW's per-parameter scaling alone is not constraining raw gradient magnitudes; the clip acts as full gradient normalization, not a safety net. Trajectory smoothing was visible (val swings ±20 vs ±50 unclipped). Tanjiro's fp32 val=111.91 nearly matched bf16 baseline.
4. **batch_size scaling doesn't help** — bs=8 halved optimizer steps/min without reducing per-epoch time (dataloader bottleneck). Closed.
5. **slice_num=128 confounded by bs=2 OOM** — can't interpret. slice_num=96 (thorfinn #1550) tests this cleanly at bs=4.
6. **n_hidden=192 confounded by bs=2 OOM** (shared-GPU contention) — frieren #1442 sent back to retest at bs=4 on bf16 baseline.
7. **LR=1e-3 + cosine T_max mismatch** — schedule sized to 30 epochs ran only 14; under-annealed. Closed. edward retesting n_layers=8 (depth is a cleaner lever at bf16 baseline).
8. **val_single_in_dist is consistently the hardest split** (~133-175 MAE across runs) — not an overfitting artifact, more likely extreme-Re / extreme-p samples. OOD camber_cruise is consistently easiest (73-106 MAE).

## Round-2 candidate pool (from `research/RESEARCH_IDEAS_2026-05-12_round1.md`)

H1 (gradient clipping → tanjiro), H3 (n_layers=8 → edward), H4 (slice_num=96 → thorfinn), H5 (mlp_ratio=4 → alphonse) all assigned. Remaining:

- **H8 — dropout 0.1**: regularisation for the 3-of-4 OOD val splits. Assign to next idle student.

## Broader follow-up directions (post-round-2)

- Compound winners of round 1 (loss + optimiser + architecture often orthogonal)
- Surface-aware decoder / dual-head architecture (separate volume and surface heads)
- Re-conditioning: explicit Re-aware embeddings, log-Re aware positional encoding
- Spectral / Fourier neural operator hybrids
- Per-domain auxiliary heads or domain adversarial features
- Curriculum on mesh size or Re regime
- Test-time augmentation (TTA) using physical symmetries
- EMA of model weights for eval

This is a living document — refresh as round 1 lands and round 2 planning starts.
