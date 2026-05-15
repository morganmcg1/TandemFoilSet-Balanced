# SENPAI Research State

- **Date:** 2026-05-15 12:35
- **Launch:** willow-pai2i-48h-r1 (round 1, 48h horizon)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Students (8):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn (1 GPU each)
- **Budget per run:** 30 min wall clock, 50 epochs max
- **Latest direction from human team:** None (no open Issues for this launch)

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) — the equal-weight mean surface pressure MAE across the 4 val/test splits on TandemFoilSet. Lower is better.

## Round-1 strategy
No prior results on this branch — round 1 explores the recipe-level lever set in parallel across 8 students. Pick orthogonal axes likely to compound:

1. **Loss reformulation** — MSE → MAE-aligned (Huber / L1) so training and the eval metric agree.
2. **Surface emphasis** — raise `surf_weight` so the metric we score gets more gradient.
3. **Optimizer schedule** — warmup + tuned LR; current plain cosine starts at peak LR.
4. **Capacity bump** — modest width increase (128 → 192/256 hidden) within the 30-min budget.
5. **Positional encoding** — turn on `unified_pos`; raw (x,z) coords are likely under-exploited.
6. **Spectral / Fourier features** — multi-scale positional embedding to capture flow length scales.
7. **Channel rebalancing** — emphasize pressure channel since it's the scoring channel.
8. **Gradient stability** — AdamW betas + clip + slightly larger batch to denoise gradients.

## Themes / next directions to consider once round-1 returns
- Train-time symmetry augmentation (horizontal flip with sign flips on AoA, Ux) — needs care because foils have camber.
- Test-time augmentation (TTA) using same symmetry.
- Physics-informed regularisers (divergence-free for volume `(Ux, Uy)`).
- Domain-specific output heads (raceCar single vs raceCar tandem vs cruise tandem).
- Better surface attention (cross-attention from surface nodes to volume slice tokens).
- Per-domain or per-channel normalization (current stats are global; pressure ranges differ by domain).
- Spectral / FNO-style operator blocks alongside Transolver attention.
- Mesh-aware sub-sampling for training (large meshes dominate compute).
- Multi-scale prediction (predict residual on top of a coarse prediction).

## Baseline (post round-1 first merge — PR #3188)
- **PR #3188 merged:** slice_num 64→128 (thorfinn)
- Transolver: 5 layers, hidden=128, heads=4, **slice_num=128**, mlp_ratio=2.
- AdamW lr=5e-4, weight_decay=1e-4, batch=4, cosine T_max=50.
- Loss = vol_MSE + 10·surf_MSE on normalized targets.
- **val_avg/mae_surf_p = 134.7389** (epoch 11/50, 30-min cap, not converged)
- ~173 s/epoch → ~11 epochs in 30-min budget at batch=4, 1 GPU.

## Known infrastructure issue
`.test_geom_camber_cruise_gt/000020.pt` has 761 inf values in pressure channel.  
→ `test_geom_camber_cruise/mae_surf_p` = NaN for ALL students. Val unaffected.  
Fix: defensive `y_finite` masking in train.py assigned to thorfinn (relative-mse-bugfix).
