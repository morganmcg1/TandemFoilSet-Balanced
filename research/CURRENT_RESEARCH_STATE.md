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

## Round-1 hypothesis matrix (8 students, orthogonal axes — all in flight)

The plan covers orthogonal levers so winners can compound after merging:

| PR | Student | Lever | Sweep |
|----|---------|-------|-------|
| #3138 | alphonse  | PhysicsAttention slice count          | `slice_num` ∈ {64, 128, 256} |
| #3142 | askeladd  | Surface-loss weight                   | `surf_weight` ∈ {10, 30, 80} |
| #3143 | edward    | Robust loss (Charbonnier vs MSE)      | `loss_fn` ∈ {MSE, Charbonnier ε=1e-3} |
| #3145 | fern      | Depth                                 | `n_layers` ∈ {5, 8, 10} |
| #3148 | frieren   | Width (proportional heads)            | `n_hidden` ∈ {128, 192, 256} |
| #3149 | nezuko    | Per-channel surface-loss weighting    | surf channel weights {[1,1,1], [1,1,4], [1,1,10]} |
| #3150 | tanjiro   | Warmup + cosine schedule              | peak `lr` ∈ {5e-4, 1e-3, 1.5e-3} (3-epoch warmup) |
| #3151 | thorfinn  | EMA model weights for eval            | `ema_decay` ∈ {0, 0.999, 0.9999} |

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
