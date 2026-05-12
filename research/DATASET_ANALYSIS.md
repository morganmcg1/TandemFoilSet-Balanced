# TandemFoilSet Dataset Analysis

(Notes distilled from `program.md` and `data/SPLITS.md` for the willow-r4 advisor.)

## Shape & magnitude

- 1499 training samples on `/mnt/new-pvc/datasets/tandemfoil/splits_v2/` split across three physical domains, **equally weighted** in training via `WeightedRandomSampler`:
  - RaceCar single (~85K mesh nodes, 599 samples)
  - RaceCar tandem (~127K nodes, 457 samples)
  - Cruise tandem (~210K nodes, 443 samples)
- Mesh size varies 74K-242K nodes — `pad_collate` pads to the largest sample per batch. The `mask` tensor is critical; everything downstream must respect it.
- Inputs `x` are 24-d (geometry + flow + foil descriptors); targets `y` are `[Ux, Uy, p]` in physical units.
- Normalization stats live in `stats.json`; `train.py` normalizes inputs and targets before loss, denormalizes for MAE.

## Target dynamic range (key challenge)

| Split | Re range | y range | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|---|
| val_single_in_dist | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| val_geom_camber_rc | 1.0M–5M | (-10,312, +2,228) | 377 | 1,237 |
| val_geom_camber_cruise | 122K–5M | (-7,648, +2,648) | 164 | 506 |

**Implication:** within every split, high-Re samples drive the extremes; per-sample y std varies by an order of magnitude. MSE in normalized space then de-normalized to physical MAE means high-Re samples dominate the loss; loss reformulations that downweight extreme outliers (Huber, relative-MAE) are worth trying.

## Four validation tracks

| Track | Stress |
|---|---|
| `val_single_in_dist` | Sanity — random holdout, single-foil |
| `val_geom_camber_rc` | Unseen front-foil camber M=6–8 (raceCar) |
| `val_geom_camber_cruise` | Unseen front-foil camber M=2–4 (cruise) |
| `val_re_rand` | Stratified Re holdout, all tandem domains |

A change that helps `val_single_in_dist` but not the geom/Re tracks is **overfitting**, not progress. The primary metric averages all four equal-weight; ablate per-track when splits disagree.

## Critical contract bits

- `data/loader.py` and `data/scoring.py` are read-only.
- Model input: `{"x": [B, N, 24]}` in **normalized** space.
- Model output: `{"preds": [B, N, 3]}` in **normalized** space.
- `data/scoring.py` denormalizes via `pred * y_std + y_mean` before MAE.
- VRAM cap is 96 GB. Max-mesh batches at fp32 can be heavy — bfloat16 should comfortably fit a 1.5–2x bigger model.
