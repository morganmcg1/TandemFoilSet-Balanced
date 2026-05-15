# Dataset Analysis — TandemFoilSet-Balanced

## Scale and structure
- 2699 total samples = 1499 train / 400 val (4×100) / 800 test (4×200).
- Variable mesh sizes: 74K–242K nodes per sample. Cruise meshes (~210K nodes) are roughly 2.5× larger than raceCar single (~85K).
- 24 input features per node; 3 output channels (Ux, Uy, p).

## Three physical domains (balanced via WeightedRandomSampler)
| Domain | Train | Mean nodes | Re range | AoA |
|--------|-------|-----------|----------|-----|
| RaceCar single | 599 | ~85K | 100K–5M | -10° to 0° |
| RaceCar tandem | 457 | ~127K | 1M–5M | -10° to 0° |
| Cruise tandem | 443 | ~210K | 110K–5M | -5° to +6° |

Without rebalancing the train sampler, raceCar single would dominate (40%).

## Four validation/test tracks
| Track | Test of |
|-------|---------|
| `*_single_in_dist` | Random holdout from single-foil — sanity check, easiest |
| `*_geom_camber_rc` | Unseen front-foil camber M=6–8 (raceCar). Geometry interpolation. |
| `*_geom_camber_cruise` | Unseen front-foil camber M=2–4 (cruise). Geometry interpolation. |
| `*_re_rand` | Stratified Re holdout across all tandem domains. Cross-regime test. |

The two `geom_camber` splits hit the hardest axis: front-foil M is fully held out. The model must interpolate NACA geometry it has never seen.

## Target magnitudes vary by ~order-of-magnitude
| Split | y range | Avg per-sample y std | Max per-sample y std |
|-------|---------|---------------------|---------------------|
| val_single_in_dist (raceCar) | (-29K, +2.7K) | 458 | 2077 |
| val_geom_camber_rc (raceCar tandem P2) | (-10K, +2.2K) | 377 | 1237 |
| val_geom_camber_cruise (cruise P2) | (-7.6K, +2.6K) | 164 | 506 |

Implications:
- High-Re samples produce extreme |p| values that can dominate the MSE gradient. A robust loss (Huber/SmoothL1) is well motivated.
- Cruise tandem has the smallest magnitudes — its per-sample contribution to MAE is small in absolute terms, but it has the largest meshes, so wall clock is highest.
- Normalization is done globally (single y_mean/y_std across all domains) — within-batch magnitude variance is still large after normalization.

## Surface vs volume node ratio
Surfaces are a small fraction of total nodes (the foil contour vs the field), so `surf_weight=10` is needed to bring surface gradient norm onto the volume scale during MSE training. The primary metric is surface pressure, so further bias here is well motivated.

## Feature notes
- Dim 12 (`is_surface`) is the same boolean used in scoring — the model knows where surfaces are.
- Dims 18–23 encode tandem-only info (foil-2 AoA, NACA, gap, stagger). They are 0 for single-foil samples — the model can infer single vs tandem from these.
- Dims 4–11 (`dsdf`, distance-based shape descriptor) carry geometric information beyond raw NACA codes.
- Single-foil "specials" cohort encodes non-NACA foils as (0,0,0) in dims 15–17. File 3 (raceCar P3) also contains 5 non-NACA airfoils.

## Edge cases / risk factors
- Variable mesh sizes mean batch padding fraction varies. Batch size 4 over a mix of 85K and 242K samples → much of the batch is padding for the small samples. A bigger batch makes this worse.
- Sample weights balance domains but do not balance Re within a domain. High-Re samples may still dominate gradients.
- `pad_collate` zero-pads — these positions are masked from the loss correctly in the baseline, but any custom loss must respect `mask`.

## Levers ranked by expected impact (subjective, pre-experiment)
1. **Loss reformulation** — Huber/SmoothL1 + per-channel weighting (pressure-heavy). MSE in this magnitude range is suboptimal.
2. **Surface bias** — `surf_weight` ramp from 10 → 25–50. Primary metric is surface pressure.
3. **Capacity scale-up** — n_hidden 128 → 192 or 256, more slices, more layers. Baseline is small.
4. **Position encoding** — Fourier features for (x, z). Helps coordinate-based models on irregular meshes.
5. **Schedule** — Linear warmup + cosine, possibly 100 epochs.
6. **Augmentation** — Reflection symmetry over the chord line (especially in raceCar where AoA is negative).
7. **Geometry-aware features** — Signed distance, per-node curvature, normals.
8. **Alternative architectures** — FNO, GNN (MeshGraphNet), GINO, hierarchical attention.
