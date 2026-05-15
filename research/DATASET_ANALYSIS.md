# TandemFoilSet Dataset Analysis

## Source

[TandemFoilSet (OpenReview)](https://openreview.net/forum?id=4Z0P4Nbosn) — 2D overset-mesh CFD simulations of tandem NACA airfoils. Pre-processed samples live on PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`.

## Training data summary

| Domain | Samples | Mean mesh nodes | Physics description |
|--------|---------|-----------------|---------------------|
| RaceCar single | 599 | ~85K | Single inverted airfoil, ground effect, Re 100K–5M, AoA -10° to 0° |
| RaceCar tandem | 457 | ~127K | Dual inverted foils (P1+P3), Re 1M–5M, AoA -10° to 0° |
| Cruise tandem | 443 | ~210K | Tandem freestream foils (P1+P3), Re 110K–5M, AoA -5° to +6° |

**Total training samples:** 1,499

## Input feature space (24-dim x)

| Dims | Feature | Notes |
|------|---------|-------|
| 0–1 | Node position (x, z) | Physical coordinates |
| 2–3 | Signed arc-length (saf) | Along-surface parameterization |
| 4–11 | Distance-based shape descriptor (dsdf) | 8-dim foil geometry encoding |
| 12 | Is-surface flag | Bool: foil boundary vs interior |
| 13 | log(Re) | Reynolds number feature |
| 14 | AoA foil 1 (radians) | Angle of attack |
| 15–17 | NACA foil 1 (M, P, T) | Normalized [0,1] |
| 18 | AoA foil 2 (radians, 0 for single) | |
| 19–21 | NACA foil 2 (0,0,0 for single) | |
| 22 | Gap between foils (0 for single) | ~[-0.8, 1.6] |
| 23 | Stagger between foils (0 for single) | ~[0.0, 2.0] |

## Output targets (3-dim y)

| Channel | Description | Physical units |
|---------|-------------|----------------|
| 0 | Ux | m/s velocity x |
| 1 | Uy | m/s velocity z |
| 2 | p | m²/s² kinematic pressure |

## Target magnitude key facts

| Split | Re range | y range | Avg per-sample y std | Max per-sample y std |
|-------|----------|---------|---------------------|----------------------|
| val_single_in_dist | 104K–5M | (-29136, +2692) | 458 | 2,077 |
| val_geom_camber_rc | 1.0M–5M | (-10312, +2228) | 377 | 1,237 |
| val_geom_camber_cruise | 122K–5M | (-7648, +2648) | 164 | 506 |

**Key insight:** Per-sample y-std varies by ~10x within each split (low-Re vs high-Re). This heteroskedasticity is the primary reason robust losses (Huber) and Re-conditioning may outperform plain MSE.

## Validation / test splits (100 / 200 samples each)

| Track | What it tests |
|-------|--------------|
| val/test_single_in_dist | In-distribution sanity: random holdout from raceCar single |
| val/test_geom_camber_rc | OOD geometry: unseen front foil camber M=6-8 (raceCar tandem P2) |
| val/test_geom_camber_cruise | OOD geometry: unseen front foil camber M=2-4 (cruise tandem P2) |
| val/test_re_rand | OOD regime: stratified Re holdout across all tandem domains |

## Primary metric

`val_avg/mae_surf_p` = equal-weight mean surface pressure MAE across all 4 val splits.  
**Lower is better.** Surface nodes only; non-finite samples skipped; computed in float64.

## Key mesh facts

- Variable mesh size: 74K to 242K nodes per sample
- Padded to N_max per batch; `mask` tensor separates real from padding
- `is_surface` marks foil boundary nodes (models must handle surface vs. volume differently)
- Three overset zones per sample (background + up to 2 foil-dense zones)

## Physical insights for model design

1. **Pressure is highest near leading edge and in the foil-foil gap** — any model that captures these sharp spatial gradients earns on the primary metric.
2. **Log(Re) encodes a 50× dynamic range** — the model needs to adapt predictions between laminar-near (Re=100K) and turbulent-near (Re=5M) regimes.
3. **Tandem interaction is encoded only via gap/stagger dims** — single-foil samples have zeros, so the model must learn to activate tandem-interaction pathways conditionally.
4. **OOD splits test interpolation in NACA space** — camber M=6-8 and M=2-4 are held out from tandem training. Models that rely too heavily on shape-descriptor features over physical geometry will struggle.
