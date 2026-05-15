# TandemFoilSet — Dataset Analysis

Living analysis of the dataset that informs hypothesis design on the Charlie
local-metrics arm (`icml-appendix-charlie-pai2i-48h-r1`).

## Problem

Predict full velocity `(Ux, Uy)` and pressure `p` field at every mesh node of a
2D CFD simulation over tandem (and single) airfoil geometries. Inputs are
geometry + flow descriptors at each node; outputs are the three field channels.

Primary ranking metric: **`val_avg/mae_surf_p`** (lower = better) — equal-weight
mean of surface-pressure MAE across four splits, computed in physical
(denormalized) units in float64.

Test analogue: **`test_avg/mae_surf_p`** computed from the best-val checkpoint.

## Splits

| Track | Train? | Val | Test | What it probes |
|-------|:------:|:---:|:----:|----------------|
| `single_in_dist` | yes | 100 | 200 | Random holdout, sanity / in-distribution |
| `geom_camber_rc` | no  | 100 | 200 | **OOD**: unseen front-foil camber M=6-8 (raceCar tandem) |
| `geom_camber_cruise` | no | 100 | 200 | **OOD**: unseen front-foil camber M=2-4 (cruise tandem) |
| `re_rand` | no | 100 | 200 | **OOD**: stratified Re holdout across tandem domains |

Three of four tracks are OOD. **OOD generalization, not in-distribution
overfitting, is the limiting factor** for the metric.

## Domains (training)

| Domain | n_train | Mean nodes | Flow regime |
|--------|--------:|-----------:|-------------|
| RaceCar single | 599 | ~85K  | Single inverted foil + ground, Re 100K–5M, AoA -10°–0° |
| RaceCar tandem | 457 | ~127K | Dual inverted foils, Re 1M–5M, AoA -10°–0° |
| Cruise tandem  | 443 | ~210K | Tandem freestream, Re 110K–5M, AoA -5°–+6° |

Domains are **equally weighted** in training via `WeightedRandomSampler`. Note
that cruise meshes are ~2.5× denser than raceCar single, so node-seconds per
epoch lean cruise even with equal sample weight.

## Input features (24 dims)

- Dims 0-1: position `(x, z)` — critical for spatial inductive bias
- Dims 2-3: signed arc-length `saf`
- Dims 4-11: distance-based shape descriptor `dsdf`
- Dim 12: `is_surface` (boolean) — same as `is_surface` we mask metrics with
- Dim 13: `log(Re)` — single flow scalar
- Dims 14-17: foil 1 AoA + NACA (camber, position, thickness)
- Dims 18-21: foil 2 AoA + NACA (0 on single-foil)
- Dim 22: gap (0 on single-foil)
- Dim 23: stagger (0 on single-foil)

## Target characteristics

Targets are normalized by `(y_mean, y_std)` from `stats.json` outside the model.
Loss is MSE in normalized space; MAE is computed in physical space.

| Split | y range | mean per-sample y std | max per-sample y std |
|-------|---------|-----------------------|----------------------|
| `val_single_in_dist` | (-29136, +2692) | 458 | 2077 |
| `val_geom_camber_rc` | (-10312, +2228) | 377 | 1237 |
| `val_geom_camber_cruise` | (-7648, +2648) | 164 | 506 |

High-Re samples dominate the magnitude tails; per-sample y std varies by
**~10×** within each domain. This means:
- A single-channel MAE in physical units is **dominated by the worst (most
  extreme) samples**.
- Loss in normalized space treats them uniformly; MAE in physical units does
  not. Anything that handles extreme-magnitude samples disproportionately well
  (e.g. L1-style loss, robust loss, EMA, per-sample reweighting) can move the
  metric.

## Loss vs metric mismatch (major)

- **Loss**: `MSE(pred_norm, y_norm)` aggregated over volume + surface (with
  `surf_weight=10` on surface MSE).
- **Metric**: `MAE` (physical units) aggregated globally over surface nodes,
  then averaged over splits.

Mismatches:

1. **MSE vs MAE**. MSE is L2-optimal (predicts mean), MAE is L1-optimal
   (predicts median). For extreme-value distributions like the high-Re
   tail, MSE overweights large errors and can push the model toward smooth
   centroidal predictions that are robust to outliers but suboptimal for
   per-node accuracy. Switching to SmoothL1 / Huber / L1 in the loss
   should directly help.
2. **Surf vs vol weighting**. `surf_weight=10` is already a heavy thumb on
   the scale, but the metric is **only** surface pressure (1/3 channels on
   ~few % of nodes). Larger surf_weight (25-100) may help.
3. **Per-channel**. Loss treats Ux, Uy, p equally; metric is only p.
   Per-channel weighting that emphasizes p will more directly target the
   metric.
4. **Per-domain**. Loss averages within batch; metric averages across splits.
   If one OOD split dominates, addressing the worst split helps the avg.

## Mesh / batching realities

- Mesh sizes 74K–242K nodes; padded to N_max per batch via `pad_collate`.
- Batch=4 → padded tensor up to `[4, ~210K, 24]` for cruise-heavy batches.
- `mask` must be applied everywhere (loss, metrics, attention if customized).
- Model has 1-2M params; VRAM bottleneck is intermediate activations, not
  weights. 96 GB allows substantial scaling on width/depth or slice_num.
- Transolver's slice attention complexity is `O(N · slice_num · d_head)` for
  the slice projection plus `O(slice_num² · d_head)` for self-attention on
  slice tokens — both linear in N. Increasing slice_num from 64 → 128 should
  roughly halve the slice-token "compression ratio" without quadratically
  blowing up cost.

## Constraints summary

- **Per-run wall clock**: 30 min (hard cap from `SENPAI_TIMEOUT_MINUTES`).
- **Per-run epochs**: 50 (hard cap from `SENPAI_MAX_EPOCHS`).
- **VRAM**: 96 GB.
- **No new packages outside `pyproject.toml`** without adding them in the same
  PR. PyTorch-only optimizers (AdamW, Adam, Adagrad, RAdam, etc.) and built-in
  schedules are fair game.
- **Read-only**: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`,
  `data/generate_manifest.py`, `data/split_manifest.json`, `data/SPLITS.md`.
  All experiment code goes in `train.py` (or new `train_*.py` if needed).
- **Local JSONL metrics only**: NO remote experiment tracking (Weave / W&B etc).
  Metrics committed under `models/<experiment>/metrics.jsonl`.

## Implications for hypothesis design

- Prioritize **loss / metric alignment** (L1 / Huber, per-channel weighting,
  higher surf_weight).
- Prioritize **OOD-friendly inductive bias** (stronger pos encoding, EMA /
  Polyak avg, dropout, regularization).
- Prioritize **mesh-resolution-aware capacity** (slice_num, mlp_ratio, depth)
  over raw width since VRAM is abundant but compute time is tight.
- De-prioritize **architecture rewrites** in early rounds — knobs first, then
  bigger structural moves.
- Mixed precision (`bf16` autocast) buys throughput → more epochs / better
  convergence within the 30-min cap.
