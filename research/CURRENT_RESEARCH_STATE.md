# SENPAI Research State

- **Updated:** 2026-05-15
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No human directives received yet on this launch. Tracking the GitHub Issues feed.

## Current research focus

Round 1 of a fresh research track on TandemFoilSet. The primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val tracks: `val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Paper-facing metric is `test_avg/mae_surf_p` (best-val checkpoint, four test tracks).

The launch is isolated — only inspect/cite/borrow from this branch and the assigned student PR branches. No history transfer from other tracks.

### Round 1 portfolio (single-axis probes)

Eight orthogonal one-knob changes designed to map out the local neighborhood of the baseline. Each PR isolates one knob so we can attribute deltas to the change.

| Student | Axis | Change |
|---|---|---|
| alphonse | compute / wall-clock | bf16 autocast on forward + backward |
| askeladd | optimizer schedule | `lr=1e-3` + 2-epoch linear warmup + cosine to 0 |
| edward | architecture depth | `n_layers=5 → 7` |
| fern | loss weighting | `surf_weight=10 → 25` |
| frieren | attention slicing | `slice_num=64 → 128` |
| nezuko | loss formulation | Huber (smooth L1, β=1.0) on volume term |
| tanjiro | stability | grad clipping `max_norm=1.0` |
| thorfinn | regularization | `dropout=0.05` in Transolver blocks |

Coverage by category:
- **Compute / throughput**: bf16 (alphonse) — directly trades precision for epochs/min
- **Optimizer**: warmup+cosine at higher peak (askeladd), grad clip (tanjiro)
- **Architecture**: depth (edward), attention slicing (frieren), dropout regularization (thorfinn)
- **Loss**: surf weighting (fern), Huber on volume (nezuko)

Notably absent from round 1 (deliberate — keep round 1 readable): wider model, larger batch size, OneCycleLR, EMA, SWA, NACA/AoA conditioning, mesh-aware features. These are reserved as next-step candidates depending on what round 1 reveals.

## Potential next research directions

After round 1 results land:

1. **Compound wins** — if multiple round-1 levers help, the next round should stack the top 2-3 (e.g., bf16 + grad-clip + higher LR with warmup).
2. **Wider model + bf16** — n_hidden 128 → 192, conditional on bf16 unlocking enough wall clock.
3. **Per-channel surface weights** — the primary metric is surface pressure specifically; pull surface-`p` harder than surface-`{Ux, Uy}`.
4. **EMA over last quarter of training** — cheap variance reduction for OOD checkpoint selection.
5. **OneCycleLR** — alternative to warmup+cosine, known short-budget winner; only worth it if cosine baseline plateaus.
6. **Architecture extensions** — global-scalar conditioning (FiLM on log Re, AoA, NACA, gap, stagger), since dims 13-23 of `x` are constant per-sample and currently broadcast through expensive per-node MLPs.
7. **Loss reformulation on `p` specifically** — log-magnitude scaling or per-sample y-std-aware loss to handle the 10x cross-domain magnitude variation.
8. **Data augmentation** — vertical flip / AoA negation pairs, mesh subsampling for cheap epochs.

## Dataset notes carried into round 1

- Four val tracks have **very different y magnitudes** (`val_single_in_dist` extremes ~10x `val_geom_camber_cruise`). Equal-weight averaging means the in-dist track dominates `val_avg/mae_surf_p` unless predictions on it are very accurate.
- High-Re samples drive the extremes within every split.
- The training set is **balanced across the three physical domains** via `WeightedRandomSampler`; raceCar single would otherwise dominate.
