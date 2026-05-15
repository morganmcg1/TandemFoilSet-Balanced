# TandemFoilSet Dataset Analysis

A working notebook of dataset facts that should inform every experiment design.

## Sample counts

| Split type | Count | Notes |
|------------|-------|-------|
| Train       | 1499 | 3 physical domains, balanced via WeightedRandomSampler |
| Val (each of 4 tracks) | 100 |
| Test (each of 4 tracks) | 200 |

### Train domain split

| Domain | Train samples | Mesh nodes (mean) | Re | AoA |
|--------|---------------|-------------------|----|-----|
| RaceCar single | 599 | ~85K  | 100K–5M | -10° to 0° |
| RaceCar tandem | 457 | ~127K | 1M–5M | -10° to 0° |
| Cruise | 443 | ~210K | 110K–5M | -5° to +6° |

Sampling: domain-balanced via `WeightedRandomSampler` with weights ∝ 1/group_size. Otherwise raceCar single dominates by sample count.

## Per-sample y std (from `program.md`)

| Source split | Re | y range (min,max) | Avg y std | Max y std |
|--------------|----|-------------------|-----------|-----------|
| `val_single_in_dist` (raceCar single) | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` (raceCar tandem P2) | 1M–5M | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` (cruise tandem P2) | 122K–5M | (-7,648, +2,648) | 164 | 506 |

**Implication 1.** Per-sample y std spans ~12× across domains and ~13× within a single
domain (high-Re vs low-Re). Even after global y-normalization, high-Re samples produce
normalized y of much larger magnitude than low-Re ones. MSE loss is quadratic in
prediction error → high-Re samples dominate gradients.

**Implication 2.** The 4 val tracks have heterogeneous magnitudes too: averaging
`mae_surf_p` across the 4 tracks with equal weight assigns relatively higher
"per-unit-error" cost to the cruise track than to the raceCar single track.

## Input features (24 dims)

| Dims | Feature | Notes |
|------|---------|-------|
| 0–1   | (x, z) position | Variable across nodes within a sample |
| 2–3   | Signed arc-length (saf) | Geometry-related |
| 4–11  | Distance-based shape descriptor (dsdf) | 8-D geometric encoding |
| 12    | `is_surface` (0/1) | Already in input — model can condition on surface |
| 13    | `log(Re)` | **Global — same for all nodes in a sample** |
| 14    | AoA foil 1 (rad) | Global |
| 15–17 | NACA foil 1 (camber M, position P, thickness T) | Global, normalized to [0,1] |
| 18    | AoA foil 2 (rad, 0 for single-foil) | Global |
| 19–21 | NACA foil 2 (0,0,0 for single-foil) | Global |
| 22    | Gap (0 for single-foil) | Global, tandem-only |
| 23    | Stagger (0 for single-foil) | Global, tandem-only |

**Implication 3.** Dims 13–23 are constant within a sample. The Transolver
preprocess MLP sees them at every node, which is wasteful conditioning — they could
be encoded once and broadcast. This is the lever for the GALE-style geometry
conditioning.

## OOD splits

| Track | What's held out |
|-------|------------------|
| `val_single_in_dist` / `test_single_in_dist` | random holdout from single-foil — sanity |
| `val_geom_camber_rc` / `test_geom_camber_rc` | unseen front foil camber M=6-8 in raceCar tandem |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | unseen front foil camber M=2-4 in cruise tandem |
| `val_re_rand` / `test_re_rand` | stratified Re holdout across all tandem domains |

**Implication 4.** Two of the 4 val tracks specifically test *interpolation
to unseen front-foil camber*. Geometry conditioning that survives the depth of
the network has high leverage on these tracks.

**Implication 5.** `val_re_rand` is a Re-stratified holdout across *all* tandem
domains. This is a different generalization axis from camber. A method that
helps one OOD axis may regress on the other; flag any split-level disagreement.

## Mesh structure

Variable mesh sizes 74K to 242K nodes per sample. Padded with `pad_collate`. The
`mask` tensor distinguishes valid nodes from padding — **all loss and metric
code must respect the mask.**

```
┌─────────────────────────────────────────────────┐
│  Zone 0 — coarse background (full domain)       │
│       ┌──────────────┐   ┌──────────────┐       │
│       │  Zone 1      │   │  Zone 2      │       │
│       │  (dense)     │   │  (dense)     │       │
│       │   foil 1     │   │   foil 2     │       │
│       └──────────────┘   └──────────────┘       │
└─────────────────────────────────────────────────┘
```

Boundary IDs collapsed to single `is_surface` boolean across both foils — model
does not see foil 1 vs foil 2 distinction at node level. Distinction is recovered
from global features (foil 2 AoA, NACA, gap, stagger).

## Output targets (3 dims)

| Channel | Description |
|---------|-------------|
| 0 | `Ux` — velocity x-component |
| 1 | `Uy` — velocity z-component |
| 2 | `p` — kinematic pressure (p/ρ, m²/s²) |

**Implication 6.** Primary metric is **surface pressure only** (channel 2 on
surface nodes). All other prediction signal is auxiliary. Loss formulations that
emphasize pressure on surface should help, provided they do not destabilize the
shared-representation training of the other channels.

## Workflow / contract reminders

- `data/scoring.py`: float64 MAE accumulation, per-sample skip for non-finite GT,
  global node-level aggregation. **Read-only.**
- `data/loader.py`: `SplitDataset`, `TestDataset`, `load_data`, `load_test_data`,
  `pad_collate`. **Read-only — don't change loader contract.**
- Stats are in `splits_v2/stats.json`; mean/std for input/output normalization.
- Best checkpoint selection: lowest `val_avg/mae_surf_p`. Evaluated on test at
  training end. Saved under `models/<experiment>/checkpoint.pt` with
  `metrics.yaml` and `metrics.jsonl`.

## Hypothesis-design heuristics for this dataset

1. **Surface-pressure-aligned objective beats volume-loss-dominated objective** (per Implication 6).
2. **Magnitude-aware loss handling helps** (per Implication 1) — log compression, Huber, per-domain renorm, gradient clipping.
3. **Persistent geometry conditioning** is structurally underused (per Implication 3).
4. **Domain-balanced sampling alone is insufficient** for sample-level magnitude balance (per Implication 1) — Re-stratified upweighting can help.
5. **Watch all 4 val tracks separately** — a single-track improvement that regresses another may net to zero on `val_avg`.
6. **Compounding small improvements** is the right strategy; each round's winner becomes the baseline for the next.
