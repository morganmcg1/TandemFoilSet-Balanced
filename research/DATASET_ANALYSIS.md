# TandemFoilSet — Dataset Analysis

Reference for advisor and student decisions on hypothesis design. Cross-check `target/program.md`, `data/SPLITS.md`, and `splits_v2/stats.json` / `meta.json` if these notes get stale.

## Headline numbers

- **Train samples:** 1499 (across 3 balanced domains via `WeightedRandomSampler`)
- **Val splits:** 4 × 100 samples
- **Test splits:** 4 × 200 samples
- **Total train nodes:** ~201.5M (mean 134K/sample)
- **Mesh sizes:** 74K to 242K nodes per sample (3.3× variance, padded with mask)
- **Input dims:** 24 (positions ×2, saf ×2, dsdf ×8, is_surface, log_Re, AoA1, NACA1 ×3, AoA2, NACA2 ×3, gap, stagger)
- **Target dims:** 3 (Ux, Uy, p)

## Target distribution

| Channel | y_mean | y_std | Notes |
|---------|--------|-------|-------|
| Ux | 30.18 | 21.78 | Streamwise velocity, typical magnitude ~30 m/s |
| Uy | -0.48 | 9.74 | Cross-stream velocity, near-zero mean |
| **p** | **-129.22** | **679.45** | Kinematic pressure, **~30× the std of Ux** |

> **Implication:** in normalized space, an MSE loss treats all three channels equally. But the metric (`mae_surf_p`) lives in physical space, where the `y_std=679` multiplier means tiny normalized errors blow up to large physical pressure errors. **Train signal must over-emphasize p.**
>
> Also, max per-sample y std reaches 2077 in `val_single_in_dist` — pressure has heavy tails, so MSE is gradient-dominated by a small fraction of high-Re extreme samples. **L1 / Huber on pressure surfaces is well-aligned with the metric.**

## Domain breakdown

| Domain | Train | Mean nodes | Re range | AoA range |
|--------|-------|-----------|----------|-----------|
| RaceCar single | 599 | ~85K | 100K–5M | -10° to 0° |
| RaceCar tandem | 457 | ~127K | 1M–5M | -10° to 0° |
| Cruise tandem | 443 | ~210K | 110K–5M | -5° to +6° |

Domains are equally weighted in training via the `sample_weights` returned from `load_data` — otherwise raceCar single (the largest cohort) would dominate.

## Validation tracks (what each generalization axis tests)

| Track | What's held out | Generalization tested |
|-------|----------------|----------------------|
| `val_single_in_dist` / `test_single_in_dist` | Random subset of single-foil | In-distribution sanity |
| `val_geom_camber_rc` / `test_geom_camber_rc` | RaceCar tandem with front-foil M=6-8 | Geometry interpolation (raceCar) |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Cruise tandem with front-foil M=2-4 | Geometry interpolation (cruise) |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem | Cross-Re generalization |

The two `val_geom_camber_*` tracks are the **strongest generalization tests** — front-foil camber values that the model literally never sees. Strong feature engineering (Fourier Re, FiLM, boundary-layer features) and regularization (dropout, stochastic depth) should help most here.

## Physics-derived feature opportunities

Existing inputs cover global state (Re, AoA, NACA, gap, stagger) and per-node geometry (position, saf, dsdf). What's *missing* but easy to derive:

| Derived feature | Formula | Physics motivation |
|----------------|---------|--------------------|
| Local boundary-layer Re | `log(Re) + log(\|saf\|)` ≈ log(Re·x/L) | Determines local boundary-layer state — primary driver of surface pressure |
| Adverse pressure gradient indicator | grad(`saf`) along surface | Boundary-layer separation precursor; useful surface-only feature |
| Stagnation-point distance | `\|saf\|` minimum on surface | Predicts stagnation-pressure region |
| Per-node sin/cos of orientation | from `dsdf` gradients | Surface-normal directionality |

**Round-1 thorfinn PR (#762)** tests the first of these.

## Padding / mask discipline

- `pad_collate` zero-pads every batch to the largest mesh in the batch.
- `mask` tensor: `True` for real nodes, `False` for padding.
- Model output includes predictions for padding positions — these **must** be masked out of loss and metrics.
- The baseline `train.py` and `data/scoring.py` handle this correctly; any custom loss/pooling/aggregation must respect `mask`.
- This is a frequent footgun for new architectural changes (e.g., custom pooling, kNN attention, cross-attention conditioning).

## Constraints to respect when designing experiments

- **VRAM:** 96 GB. Largest meshes are 242K nodes × hidden dim. Can fit much larger models than baseline at batch_size=2.
- **Wall-clock:** `SENPAI_TIMEOUT_MINUTES` (default 30 min) per run. Bigger models train fewer epochs; cosine annealing schedule must reach a reasonable LR even if early-stopped.
- **Epochs:** `SENPAI_MAX_EPOCHS` (default 50) cap. With ~375 batches/epoch at bs=4, epoch time roughly determines whether 50 epochs is achievable.
- **Read-only data loaders.** Any new sampling strategy or feature transform goes in `train.py`, not `data/`.
- **No new packages** outside `pyproject.toml` without including the dep in the same PR.

## Open dataset questions (for future analysis)

- Do high-Re samples (Re > 1M) genuinely dominate the validation MAE? If yes, robust losses (Huber, focal) gain more.
- Are the OOD camber tracks `geom_camber_rc` (M=6-8) and `geom_camber_cruise` (M=2-4) symmetric in difficulty, or is one harder than the other? The cruise track has higher mesh resolution (mean 210K vs ~127K).
- How much of the surface pressure error comes from a small set of "hard" samples vs. distributed across the corpus? If concentrated, oversampling those samples at training time is high-EV.
