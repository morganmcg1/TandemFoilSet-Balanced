# SENPAI Research State

- **Date:** 2026-05-15 (initial advisor invocation of `icml-appendix-willow-pai2i-48h-r4`)
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` for validation, `test_avg/mae_surf_p` for paper-facing numbers (equal-weight mean surface-pressure MAE across 4 splits — `in_dist`, `geom_camber_rc`, `geom_camber_cruise`, `re_rand`).

## Most recent research direction from human researcher team
No GitHub Issues open for this track. Proceeding from the program contract in `target/program.md` only.

## Current research focus and themes

This is the inaugural round on this branch. The baseline is the Transolver in `train.py` as-is (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; AdamW lr=5e-4, bs=4, surf_weight=10, MSE loss, cosine annealing over 50 epochs, no warmup, no grad clipping, no augmentation).

Round 1 goal: a broad sweep across the orthogonal levers most likely to move `val_avg/mae_surf_p`. Each PR isolates one change so we can attribute deltas cleanly and stack improvements in later rounds.

## Round 1 assignments (all in-flight as of 2026-05-15)

| # | Student | Hypothesis | Why it matters |
|---|---------|-----------|----------------|
| #3089 | alphonse | L1 / smooth-L1 loss in normalized space | Targets directly align with MAE metric; reduces high-Re outlier dominance |
| #3090 | askeladd | Width: n_hidden 128→192 (+follow-up 256) | Baseline is below Transolver paper's standard width; capacity lever |
| #3091 | edward | LR warmup + grad clip + higher peak LR | Stability primitives unlock higher LR; common transformer recipe |
| #3092 | fern | More physics-attention slices: slice_num 64→128 (+192) | Raises resolution of physics decomposition for variable mesh sizes |
| #3093 | frieren | bf16 autocast + batch_size 4→8 | Speed + better gradient estimate via larger batch |
| #3095 | nezuko | surf_weight 10→30 + per-channel p-weighting | Direct loss-side push on the primary metric |
| #3096 | tanjiro | x-axis symmetry augmentation (Ux/AoA/stagger flip) | Free dataset doubling; expected big OOD gains on geom_camber tracks |
| #3097 | thorfinn | Depth: n_layers 5→8 + stochastic depth (DropPath 0.1) | More refinement iterations with regularization |

## Potential next research directions

Once Round 1 lands, the orthogonal levers are:

1. **Compose winners.** If L1 + wider + warmup all win, stack them in a single PR before exploring novel directions.
2. **Output head redesign.** Separate per-channel decoder heads (Ux, Uy, p have very different value ranges and behaviour) — break the shared output trunk.
3. **Position encoding upgrades.** Fourier features on `(x, z)`; or `unified_pos=True` (already in Transolver) with sweep on `ref`.
4. **Physics-aware loss terms.** Divergence-free residual penalty on Ux/Uy interior; surface-normal pressure gradient consistency.
5. **Curriculum / hard-example mining.** Re-stratified batching that oversamples high-Re or surface-rich samples.
6. **Bigger architectural rethinks.** Mesh-graph neural operator hybrids; multiscale / hierarchical attention; spectral convs in slice space.
7. **EMA model weights** for stabilization at the best-val checkpoint.
8. **Test-time augmentation** using the same x-flip symmetry — should compound with the train-time aug.
9. **Sharpness-aware minimization (SAM)** for OOD generalization on the geom_camber tracks.

## Notes & open questions

- Researcher-agent run dispatched (writing to `/workspace/senpai/research/RESEARCH_IDEAS_2026-05-15_initial.md`) for the next round of hypotheses; constrained to literature only (no cross-track senpai history).
- No prior runs on this advisor branch — Round 1 also establishes the actual baseline numbers via the per-arm runs each student will produce (each PR includes an optional baseline reproduce command).
- All 8 students dispatched; pods all healthy as of dispatch time.
