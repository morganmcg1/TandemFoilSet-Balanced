# SENPAI Research State

- 2026-05-15 12:33 — initial advisor invocation, launch `willow-pai2i-24h-r1`
- No directives from the human researcher team yet on this launch.

## Current research focus

Round 1 on a fresh advisor branch (`icml-appendix-willow-pai2i-24h-r1`) — we
have no merged improvements yet and no measured baseline numbers for this
exact track. The Transolver baseline in `train.py` is small (~570K params),
trained with MSE loss + `surf_weight=10` for 50 epochs, AdamW + cosine schedule,
batch size 4. Each training run is capped at 30 min wall-clock.

The primary ranking metric is `val_avg/mae_surf_p` — surface pressure MAE
averaged over four validation splits (`val_single_in_dist`,
`val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Surface pressure
matters most because: (a) it's the headline paper number, and (b) targets span
an order of magnitude across high-Re vs low-Re samples, with the surface-pressure
extremes dominating absolute error.

## Round-1 hypothesis matrix (8 students, orthogonal axes)

The plan covers six orthogonal levers so winners can compound after merging:

1. **Capacity for slice attention** (`alphonse`) — slice_num 64 → 128/256.
   PhysicsAttention's slice tokens are the only place where global mesh
   structure is captured; default of 64 may be undercount for 200K-node meshes.
2. **Surface-weighting strength** (`askeladd`) — surf_weight 10 → 30/80.
   Primary metric is surface-only, so an aggressive surface weighting may
   directly trade volume MAE for surface MAE gains.
3. **Robust loss vs MSE** (`edward`) — Charbonnier (smooth L1) replacing MSE.
   Per-sample y std varies 10× within a split; MSE is dominated by high-Re
   outliers. Charbonnier should balance regimes.
4. **Depth** (`fern`) — n_layers 5 → 8/10. Mesh sizes are large; deeper stacks
   may help the model integrate multi-scale flow features.
5. **Width** (`frieren`) — n_hidden 128 → 192/256 with proportional heads.
   The model is small for 96GB VRAM; widening is cheap.
6. **Surface-pressure-targeted loss** (`nezuko`) — per-channel/location loss
   weights that explicitly upweight surface p. Directly targets the metric.
7. **LR schedule** (`tanjiro`) — 3-epoch linear warmup + cosine to small min,
   peak LR sweep. Transolver training without warmup is often unstable in the
   first few epochs.
8. **EMA weights** (`thorfinn`) — exponential moving average of model
   parameters for evaluation. Free improvement in many surrogate / vision
   tasks, and orthogonal to everything else.

## Potential next-round directions (post round-1 review)

- Compound the round-1 winners (each is orthogonal by design).
- Loss/data augmentation: chord rotation / NACA-symmetric flip for raceCar
  domains.
- Curriculum: order training by mesh size or Re to stabilize early epochs.
- Architectural: Fourier features for position channels, GELU → SwiGLU, GeGLU
  in MLPs, longer slice MLPs.
- Output-side: separate readout heads for surface vs volume, dedicated
  per-channel heads.
- Test-time augmentation: average predictions over k augmented forward passes.
- Mesh-aware encodings: signed-distance field, surface normals, KNN positional
  encoding.

## Operational notes

- Per-run cap: 30 min wall-clock, 50 epochs (hard, set via `SENPAI_TIMEOUT_MINUTES`,
  `SENPAI_MAX_EPOCHS`).
- All experiments live on advisor branch `icml-appendix-willow-pai2i-24h-r1`.
- 8 GPUs available (1 per student), so sweeps of 2-3 arms within one student's
  budget are realistic.
