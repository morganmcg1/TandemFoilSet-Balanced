# Dataset Analysis — TandemFoilSet

Source of truth: `target/program.md`, `target/data/SPLITS.md`, `target/data/loader.py`,
`target/data/scoring.py`, and the materialized splits at
`/mnt/new-pvc/datasets/tandemfoil/splits_v2/`.

## Geometry / topology

- Variable mesh size per sample: **74K to 242K nodes**. Padded via `pad_collate`
  with a `mask` tensor that the loss / metrics MUST respect.
- 2D CFD over an overset mesh (coarse background + dense per-foil zones).
- Boundary semantics collapsed to a single `is_surface` boolean per node; the
  model has no native foil-1 / foil-2 distinction — that information is in
  features 14–23.

## Input features (x ∈ R^24)

| Dims | Feature |
|------|---------|
| 0–1   | Node position (x, z) |
| 2–3   | Signed arc length (saf) |
| 4–11  | Distance-based shape descriptor (dsdf) |
| 12    | Is-surface flag |
| 13    | log(Re) |
| 14    | AoA foil 1 (rad) |
| 15–17 | NACA foil 1 (M, P, T), normalized to [0,1] |
| 18    | AoA foil 2 (rad, 0 if single) |
| 19–21 | NACA foil 2 (zeros if single) |
| 22    | Gap (0 if single) |
| 23    | Stagger (0 if single) |

`stats.json` provides `x_mean, x_std, y_mean, y_std` — normalization is done
inside `train.py`, never in the loader.

## Targets (y ∈ R^3)

`Ux, Uy, p` in physical units. Model predicts in normalized
`(y - y_mean) / y_std` space; `data.scoring.accumulate_batch` denormalizes
before MAE.

## Splits and counts

| Split | Train | Val | Test |
|-------|-------|-----|------|
| RaceCar single | 599 | — | — |
| RaceCar tandem | 457 | — | — |
| Cruise tandem | 443 | — | — |
| `val_single_in_dist` / `test_single_in_dist` | — | 100 | 200 |
| `val_geom_camber_rc` / `test_geom_camber_rc` | — | 100 | 200 |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | — | 100 | 200 |
| `val_re_rand` / `test_re_rand` | — | 100 | 200 |

Training uses a `WeightedRandomSampler` (`sample_weights`) to balance the three
training domains. Without it, RaceCar single (~599 samples, ~85K nodes)
dominates the gradient.

## Reynolds and AoA regimes

- **Re**: ~100K to ~5M across the corpus, no intentional OOD Re slice.
  `val_re_rand` is a stratified holdout across tandem domains, not a
  Re extrapolation.
- **AoA**: raceCar -10° to 0° (inverted, negative loading);
  cruise -5° to +6° (per foil for tandem).
- **Gap / stagger**: gap ~[-0.8, 1.6], stagger ~[0.0, 2.0]; zero for single.

## NACA partitioning (held-out cambers)

- **raceCar tandem**: P1 M=2–5, P2 M=6–8 (**val/test held out**), P3 M=9 + 5 non-NACA specials.
- **cruise tandem**: P1 M=0–2, P2 M=2–4 (**val/test held out**), P3 M=4–6.

This is the structural reason `val_geom_camber_rc` and `val_geom_camber_cruise`
are the hardest tracks: the model never sees those cambers during training and
must interpolate over the NACA parameter space.

## Target distribution (heavy-tailed)

From `program.md`:

| Source split | Re range | y range | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|---|
| `val_single_in_dist` | 104K–5M | -29,136 → +2,692 | 458 | 2,077 |
| `val_geom_camber_rc` | 1.0M–5M | -10,312 → +2,228 | 377 | 1,237 |
| `val_geom_camber_cruise` | 122K–5M | -7,648 → +2,648 | 164 | 506 |

Implications for modelling:

- Per-sample target std varies by an order of magnitude even inside one domain.
- High-Re samples produce the extremes and dominate squared error gradients.
- Surface pressure values can be far larger than typical, so MAE on surface p
  is highly sensitive to which Re regime gets fit well.
- A heavy-tailed-aware loss (Huber, log-sinh, relative MAE) is a natural lever.
- Per-channel weighting / decoupled heads can prevent velocity gradients from
  drowning out pressure gradients during early training.

## What is hard

1. **Pressure surface MAE** (the primary metric) cares only about a small
   subset of nodes (boundary nodes). Loss is dominated by volume nodes
   unless `surf_weight` is high enough.
2. **Camber OOD splits** (raceCar M=6–8, cruise M=2–4) test geometry
   interpolation. Augmentation over NACA parameters or shape descriptors
   is the natural intervention.
3. **Re generalization** (`val_re_rand`). FiLM-style Re conditioning or
   explicit Re-binned multi-task heads can help.
4. **Mesh size variance** (74K–242K nodes). Memory and time scale with
   the largest sample in each batch.
