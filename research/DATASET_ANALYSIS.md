# TandemFoilSet — Dataset Analysis

Source of truth for this analysis is `target/program.md` and `target/data/SPLITS.md`. Numbers below are reproduced from those documents; if they diverge, the source docs win.

## Composition

| Set | Samples | % |
|---|---|---|
| Train | 1499 | 55.5% |
| Val (4 × 100) | 400 | 14.8% |
| Test (4 × 200) | 800 | 29.6% |
| **Total** | 2699 | 100% |

## Files and their roles

| File | Name | Train | Val | Test | NACA camber M | Re | AoA | Ground |
|---|---|---|---|---|---|---|---|---|
| 0 | raceCar single | 599 | 100 | 200 | M=2-9 + 5 specials | 100K-5M | -10° to 0° | Yes |
| 1 | raceCar tandem P1 | ~225 | ~25 | ~50 | M=2-5 | 1M-5M | -10° to 0° | Yes |
| 2 | raceCar tandem P2 | **0** | 100 | 200 | **M=6-8 (full holdout)** | 1M-5M | -10° to 0° | Yes |
| 3 | raceCar tandem P3 | ~225 | ~25 | ~50 | M=9 + 5 non-NACA specials | 1M-5M | -10° to 0° | Yes |
| 4 | cruise tandem P1 | ~225 | ~25 | ~50 | M=0-2 | 110K-5M | -5° to +6° | No |
| 5 | cruise tandem P2 | **0** | 100 | 200 | **M=2-4 (full holdout)** | 110K-5M | -5° to +6° | No |
| 6 | cruise tandem P3 | ~225 | ~25 | ~50 | M=4-6 | 110K-5M | -5° to +6° | No |

## Generalization axes

Four val/test tracks each test one axis:

1. **`val_single_in_dist` / `test_single_in_dist`** — sanity, random holdout from File 0.
2. **`val_geom_camber_rc` / `test_geom_camber_rc`** — front-foil NACA camber **M=6-8** (raceCar). Train sees M=2-5, M=9, specials only.
3. **`val_geom_camber_cruise` / `test_geom_camber_cruise`** — front-foil NACA camber **M=2-4** (cruise). Train sees M=0-2, M=4-6.
4. **`val_re_rand` / `test_re_rand`** — stratified Re holdout (every 4th sample after sorting by Re) across the four tandem training files.

These are *non-overlapping axes*: a winner has to generalize across geometry AND Reynolds simultaneously. Single-split hacks are penalized in the equal-weight mean.

## Mesh size

Variable, 74K to 242K nodes per sample. Mean ~85K (single), ~127K (raceCar tandem), ~210K (cruise tandem). Batched with zero-padding + boolean `mask`; predictions over padding must be excluded from loss and metrics.

## Input features (x, 24 dims)

| Dim | Feature |
|---|---|
| 0-1 | Node position (x, z), normalized |
| 2-3 | Signed arc-length feature (`saf`) |
| 4-11 | Distance-based shape descriptor (`dsdf`) |
| 12 | is-surface boolean |
| 13 | `log(Re)` |
| 14 | AoA foil 1 (radians) |
| 15-17 | NACA foil 1: camber M, position P, thickness T (normalized to [0,1]) |
| 18 | AoA foil 2 (radians, 0 for single-foil) |
| 19-21 | NACA foil 2 (0,0,0 for single-foil) |
| 22 | Gap between foils (0 for single-foil) |
| 23 | Stagger between foils (0 for single-foil) |

## Targets (y, 3 dims)

| Channel | Field | Units |
|---|---|---|
| 0 | Ux | velocity x-component |
| 1 | Uy | velocity z-component |
| 2 | p | kinematic pressure (p/ρ, m²/s²) |

## Dynamic range — the key training signal challenge

Per-sample y std varies dramatically *within* a split (from `program.md`):

| Source split | Re range | y range (min, max) | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|---|
| `val_single_in_dist` (raceCar single) | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` (raceCar tandem P2) | 1.0M–5M | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` (cruise tandem P2) | 122K–5M | (-7,648, +2,648) | 164 | 506 |

A single sample at Re=5M can have y values ~10× larger than one at Re=100K. Under uniform MSE, the high-Re sample's gradient dominates by ~100×. The model therefore over-fits to the high-Re regime and under-fits low-Re cruise. **This is the central design issue driving Round 1 loss-reformulation hypotheses.**

## Domain balancing

Counts: 599 raceCar single, 457 raceCar tandem (P1+P3), 443 cruise tandem (P1+P3+P6).

The `WeightedRandomSampler` from `load_data` upweights cruise and tandem to equal the single-foil count over an epoch — otherwise single would dominate by 1.35×. The sampler is already enabled in the baseline `train.py`.

## Surface vs volume nodes

`is_surface` is True on **either** foil surface (foil 1 OR foil 2 — the model does not distinguish them; use dims 18-23 to infer tandem-vs-single). The scoring metric is **surface-only pressure MAE**, accumulated globally over surface nodes within a split. Volume nodes still receive supervision via `vol_loss` (MSE, weight 1.0) but are not in the primary metric.

## Normalization contract

- `stats.json` provides `x_mean, x_std, y_mean, y_std` — single global stats over the entire dataset.
- Inputs are normalized: `x_norm = (x - x_mean) / x_std`.
- Targets normalized for loss: `y_norm = (y - y_mean) / y_std`.
- Model output is in normalized space: `pred ∈ [B, N, 3]` matches `y_norm`.
- `data/scoring.py` denormalizes: `pred_orig = pred * y_std + y_mean`, then takes MAE in physical units.

**Do not change the model's I/O contract.** All loss reformulations operate on `pred` and `y_norm` in normalized space.

## Mesh structure (physics context)

Each sample is an overset mesh with up to 3 zones (coarse background + 2 dense foil zones). Boundary IDs were collapsed into `is_surface` during preprocessing. The model does not distinguish foil 1 from foil 2 — use the conditioning features (dims 14-23) to learn that distinction.

raceCar has ground effect (slip-wall BC) with inverted foils; cruise is freestream on all boundaries — these BCs are reflected in the y values but not exposed as separate features. AoA and `is_surface` plus the NACA + gap/stagger features must carry the relevant signal.
