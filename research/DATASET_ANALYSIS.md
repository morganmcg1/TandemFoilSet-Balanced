# TandemFoilSet — Dataset Analysis

## Headline facts
- 1499 training samples across three physical domains, balanced via `WeightedRandomSampler`.
  - RaceCar single — 599 samples, mean ~85K mesh nodes, single inverted airfoil with ground effect, Re 100K–5M, AoA −10°…0°.
  - RaceCar tandem — 457 samples, mean ~127K nodes, dual inverted foils (Parts 1+3), Re 1M–5M, AoA −10°…0°.
  - Cruise tandem — 443 samples, mean ~210K nodes, tandem freestream foils (Parts 1+3), Re 110K–5M, AoA −5°…+6°.
- 4 validation tracks × 100 samples; 4 test tracks × 200 samples. Test ground truth is stored in hidden `.test_*_gt/` dirs and joined in by `load_test_data`.
- Padded mesh sizes can reach 242K nodes — VRAM-bound on the largest samples.

## Inputs (24-dim node features)
| Dims | Feature |
|------|---------|
| 0–1   | Node position (x, z) — already in geometric units; not normalized to a small range by `stats` (these are spatial coords used by the model). |
| 2–3   | Signed arc-length features (`saf`) |
| 4–11  | Distance-based shape descriptor (`dsdf`) — 8 scalars |
| 12    | Surface indicator (also passed to the trainer separately as `is_surface`) |
| 13    | `log(Re)` — single scalar, span ~100K → 5M ≈ log range 11.5 → 15.4 |
| 14    | AoA foil 1 (radians) |
| 15–17 | NACA foil 1 — (camber M, position P, thickness T), each in [0,1] |
| 18    | AoA foil 2 (radians) — 0 for single-foil |
| 19–21 | NACA foil 2 — 0 for single-foil |
| 22    | Gap between foils — 0 for single-foil |
| 23    | Stagger between foils — 0 for single-foil |

All 24 features are mean/std normalized via `stats["x_mean"]`, `stats["x_std"]` before being fed to the model.

## Targets (y, 3-dim)
- Channel 0: `Ux`
- Channel 1: `Uy` (note: actually `Uz` in physical coords; channel is z-velocity in physics, named Uy in code)
- Channel 2: `p` — kinematic pressure (p/ρ, m²/s²)

Targets are normalized for loss: `y_norm = (y - y_mean) / y_std`. Model emits predictions in normalized space; denormalized via `pred * y_std + y_mean` for MAE.

## Target magnitudes (single-foil val holdouts)
| Split | Re range | y range | Avg y-std | Max y-std |
|-------|----------|---------|-----------|-----------|
| `val_single_in_dist` | 104K–5M | (−29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` | 1.0M–5M | (−10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` | 122K–5M | (−7,648, +2,648) | 164 | 506 |

Per-sample y-std varies by an order of magnitude WITHIN any single domain. High-Re cases drive the extremes; low-Re cases live in the small-magnitude regime. The model must serve both.

## Validation/test splits
| Track | Tests |
|-------|-------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity: random holdout from single-foil. |
| `val_geom_camber_rc` / `test_geom_camber_rc` | Geometry interp on raceCar (front foil camber M=6–8 held out). |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Geometry interp on cruise (front foil camber M=2–4 held out). |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout, cross-regime. |

Primary metric (lower is better): `val_avg/mae_surf_p` for checkpoint selection, `test_avg/mae_surf_p` for paper-facing comparison. Both are the equal-weight mean of surface-pressure MAE across the 4 splits, in physical units.

Per-split diagnostics: `{split}/mae_surf_{Ux,Uy,p}` and `{split}/mae_vol_{Ux,Uy,p}`.

## Physics caveats
- 2D incompressible flow over overset meshes; 3 zones (coarse background + 2 fine foil zones, collapsed to a single `is_surface` boolean).
- The dataset does NOT distinguish foil 1 from foil 2 in `is_surface`; the model must infer the tandem geometry from x[18:24] (foil 2 AoA, NACA, gap, stagger).
- Foils are NOT mirror-symmetric — they have camber. Naïve horizontal flips are not a valid augmentation unless camber-position dim is also flipped.
- Cruise vs raceCar AoA ranges differ; cruise allows positive AoA, raceCar is always inverted (negative loading).

## Implications for hypothesis design
1. **Surface pressure is THE metric.** Surface nodes are a small fraction of total mesh — the volume loss has many more samples and can dominate gradients. Raising `surf_weight` should compound.
2. **MAE-aligned losses** (Huber, L1) likely beat MSE because the metric is MAE.
3. **Per-channel weighting** — three channels of y have different physical ranges; normalization equalizes them but if the metric is dominated by `p`, weighting p in the loss could help.
4. **Per-sample y-std varies by 10×.** Loss normalization by per-sample y-std (relative MSE/L1) may help low-Re samples contribute more.
5. **Positional features (dims 0–1) are raw coordinates** — Fourier features could help model multi-scale flow structure.
6. **High-Re drives the tails** — robust losses (Huber, log-cosh) may stabilize training against high-magnitude samples.
