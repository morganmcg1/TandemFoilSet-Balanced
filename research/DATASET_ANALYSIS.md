# TandemFoilSet — Dataset Analysis

Round: `charlie-pai2h-48h-r3` (started 2026-05-15).

## Source of truth

Pre-materialized samples on PVC: `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`. Each `.pt` is `{x: [N,24], y: [N,3], is_surface: [N]}`. Variable mesh sizes 74K–242K nodes per sample; padded with zeros in `pad_collate` and a `mask` boolean returned alongside.

## Sample counts

| Subset | n |
|---|---|
| Train | 1499 |
| Val (4 × 100) | 400 |
| Test (4 × 200) | 800 |
| Total | 2699 |

Train is split across three physical domains, equally weighted at training time via `WeightedRandomSampler` over `sample_weights`:

| Domain | n_train | Avg nodes | Re | AoA | Ground BC |
|---|---|---|---|---|---|
| raceCar single | 599 | ~85K | 100K–5M | -10°–0° | yes (inverted) |
| raceCar tandem (P1+P3) | 457 | ~127K | 1M–5M | -10°–0° | yes |
| Cruise tandem (P1+P3) | 443 | ~210K | 110K–5M | -5°–+6° | no (freestream) |

## Validation / test tracks (4 each)

Each val split is 100 samples; each test split is 200 samples. They probe different generalization axes:

| Track | Holdout axis | Test stresses |
|---|---|---|
| `*_single_in_dist` | Random hold-out of raceCar single | Sanity / in-distribution capacity |
| `*_geom_camber_rc` | raceCar tandem M=6-8 (P2 fully held out) | Geometry interpolation, race-car regime |
| `*_geom_camber_cruise` | cruise tandem M=2-4 (P2 fully held out) | Geometry interpolation, cruise regime |
| `*_re_rand` | Stratified Re across P1+P3+P4+P6 | Cross-regime Re generalization |

**Primary metric:** `val_avg/mae_surf_p` = mean over the four splits of per-split surface pressure MAE (denormalized). `mae_surf_p(S) = Σ|p_pred − p_true| / n_surf_nodes` over all valid surface nodes in the split (float64, per-sample skip if y has non-finite values). Test metric mirrors this.

## Input feature layout (x has 24 dimensions)

| Dim | Feature | Notes |
|---|---|---|
| 0-1 | Node position (x, z) | physical coords, normalized via stats.json |
| 2-3 | Signed arc-length (`saf`) | along foil boundary |
| 4-11 | Distance-based shape descriptor (`dsdf`) | 8-dim per-node geometric encoding |
| 12 | `is_surface` | duplicated from the separate boolean (informational) |
| 13 | `log(Re)` | flow-conditioning |
| 14 | AoA foil 1 (rad) | |
| 15-17 | NACA foil 1 (M, P, T) | normalized to [0, 1]; (0,0,0) for non-NACA specials |
| 18 | AoA foil 2 (rad) | 0 for single-foil |
| 19-21 | NACA foil 2 (M, P, T) | 0,0,0 for single-foil |
| 22 | Gap between foils | 0 for single-foil |
| 23 | Stagger between foils | 0 for single-foil |

All `x` features are normalized in `train.py` by `(x - stats.x_mean) / stats.x_std` before entering the model.

## Target characteristics

`y[:, 0:3]` = `(Ux, Uy, p)` in physical units (`Ux, Uy` in m/s; `p` is kinematic pressure `p/ρ` in m²/s²). Normalized by `(y - y_mean) / y_std` for loss; predictions in normalized space; denormalized for MAE.

Per `program.md` ranges from the single-file val holdouts:

| Source split | Re | y range | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|---|
| `val_single_in_dist` (raceCar single) | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` (raceCar tandem P2) | 1M–5M | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` (cruise tandem P2) | 122K–5M | (-7,648, +2,648) | 164 | 506 |

**Critical implication.** Within every split, high-Re samples drive the extremes — per-sample y std varies by ~10× even inside one domain. The loss surface is therefore dominated by high-Re samples unless re-weighted. This is a recurring lever for the round.

## Surface node sparsity

Surface nodes are a small fraction of each mesh (the foil boundary), but are weighted ×10 in the training loss (`surf_weight=10.0`). The MAE metric averages over all surface nodes in the split — so cruise meshes (~210K total nodes) contribute many more surface nodes than raceCar-single meshes (~85K), and the `val_avg/mae_surf_p` mean across splits absorbs that. Improvements concentrated on one mesh size do not necessarily transfer.

## Open questions for experimentation

1. Does the loss heterogeneity (per-sample y-std spans an order of magnitude) hurt low-Re predictions? Per-sample standardization or relative loss would address this directly.
2. Is `n_hidden=128, n_layers=5` ≈0.5M params under-parameterized for a 1.5K-sample regression problem? With 96 GB VRAM the entire baseline runs in well under 10 GB — large headroom to scale.
3. Are positional encodings (`space_dim=2` raw coords + 1 linear preprocess) strong enough? Fourier features, SIREN, or `unified_pos=True` may give the network higher-frequency capacity.
4. Does the surface-emphasis structure (single `surf_weight=10` scalar) leave performance on the table? A separate surface head, surface-aware attention bias, or surface-only auxiliary loss are unexplored.
5. Geometry-OOD performance (`val_geom_camber_*`) likely depends on how well the NACA codes + `dsdf` features generalize. Augmenting the geometric encoding could lift those splits selectively.
6. Re-OOD performance (`val_re_rand`) is a function of how cleanly Re is encoded. `log(Re)` (dim 13) is a single scalar — Fourier-encoding Re or AoA might help.

These open questions seed the first batch of hypotheses for this round.

## Editable surface

- `train.py` is the primary editable file. The model contract `x → {"preds": [B, N, 3]}` in normalized space must be preserved.
- `pyproject.toml` can take new dependencies in the same PR that uses them.
- `data/loader.py`, `data/scoring.py`, splits, and the manifest are **read-only**. New samplers, feature transforms, and per-sample weighting must live in `train.py`.
