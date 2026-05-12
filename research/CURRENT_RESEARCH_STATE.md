# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-12 (round-1 assignments out)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None yet on this branch; instructions
  scoped to the launch (treat experiments as isolated, no W&B logging,
  `SENPAI_TIMEOUT_MINUTES=30` cap per training execution).

## Round-5 research focus

We start clean: 8 idle students, 1 GPU each, and no prior round-5 winners
recorded on this branch. The aim of round 5 is to **stake out the highest-value
single-axis levers on the Transolver baseline** for `val_avg/mae_surf_p` so the
fleet can lock in a real baseline number and identify which levers compound.

## Round-1 fleet status

### Merged winners
| PR | Student | Hypothesis | val_avg/mae_surf_p | Notes |
|---|---|---|---|---|
| #1444 ✓ | charliepai2g48h5-thorfinn | MSE → Smooth-L1 (Huber, β=1.0) | **110.76** | Epoch 14 of 50 (30-min cap); still improving |

**Round-5 baseline floor: val_avg/mae_surf_p = 110.7608**

> Note: `test_avg/mae_surf_p` is NaN for all PRs in round 5 due to a data
> corruption bug in `test_geom_camber_cruise/000020.pt` interacting with
> `data/scoring.py`'s masking logic (`0 × Inf = NaN`). Round-5 ranking is
> by `val_avg/mae_surf_p` only.

### In-flight (WIP)
| PR | Student | Hypothesis | Theme |
|---|---|---|---|
| #1375 | charliepai2g48h5-alphonse | `surf_weight` 10 → 30 | Loss balance |
| #1388 | charliepai2g48h5-askeladd | 5-epoch warmup + `lr` 5e-4 → 1e-3 | Schedule |
| #1398 | charliepai2g48h5-edward | `n_hidden` 128 → 192 | Width |
| #1413 | charliepai2g48h5-fern | `n_layers` 5 → 7 | Depth |
| #1422 | charliepai2g48h5-frieren | `slice_num` 64 → 128 | Slice granularity |
| #1428 | charliepai2g48h5-nezuko | Per-channel weights [1,1,3] favoring pressure | Loss channel |
| #1439 | charliepai2g48h5-tanjiro | `batch_size` 4 → 8 | Gradient variance |

### thorfinn — next assignment
Next experiment: **bf16 mixed precision (bfloat16 AMP)** in `train.py`. The
model reached epoch 14 of 50 at the 30-min cap. If AMP halves step time we get
~28 epochs, which is substantial additional convergence under the same budget.
Also includes the `evaluate_split` scoring-fix workaround (filter bad sample
before calling `accumulate_batch`) to recover finite `test_avg/mae_surf_p`.

Constraints shape what we can sensibly try in 30-minute training executions:

- A single execution is wall-clock-bound; counterfactuals must show signal
  within the first ~30 wall-clock minutes (≈ 15-30 epochs for the default
  Transolver on this dataset, depending on mesh sizes).
- Reference architecture (n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2)
  is well-shaped for fast iteration — moderate scale-ups still fit on 96 GB.
- Test-time metric `test_avg/mae_surf_p` always evaluated at the end from the
  best val checkpoint — every PR must report both val + test averages, and the
  per-split breakdown, with finite values across all four val and four test
  splits.

## Themes seeded into round-5 round-1 assignments

1. **Loss reformulation** (cheap, high expected value):
   - Surface weight tuning (raise `surf_weight` to bias gradient toward
     surface pressure, the ranking quantity).
   - Per-channel reweighting (overweight pressure since it dominates ranking).
   - Swap MSE → Smooth-L1 in normalized space (closer to the MAE objective,
     more robust on the order-of-magnitude span of normalized y at extreme Re).

2. **Optimization schedule** (cheap, high expected value):
   - Add linear warmup + raise peak lr (5e-4 → 1e-3 with 5-epoch warmup), to
     compensate for the small batch and accelerate early convergence under a
     short wall-clock budget.

3. **Capacity / model topology** (moderate cost, moderate expected value):
   - Wider Transolver (`n_hidden` 128 → 192).
   - Deeper Transolver (`n_layers` 5 → 7).
   - Larger slice count (`slice_num` 64 → 128) — more physics-aware slices.

4. **Effective batch size** (cheap, moderate expected value):
   - Double batch via gradient accumulation, equivalent throughput per step
     but lower-variance gradient estimates.

## Potential next research directions (round 2+ once baseline is in place)

- **Compound the winners.** If two orthogonal axes win independently (e.g.
  surf_weight + lr/warmup), stack them in a single follow-up PR.
- **Rotational / reflection augmentation** on coordinates and AoA — cheap data
  augmentation grounded in the physics (the airfoil flow is well-defined under
  signed rotation of geometry + AoA).
- **Spectral / Fourier feature lifting on (x, z)** before the preprocess MLP
  — helps high-frequency surface features that drive pressure.
- **Boundary-aware attention masking or extra surface-aware blocks at the
  end of the encoder** — pressure peaks live at the foil surface; biasing
  attention toward surface slices could help.
- **Re-aware feature normalization or log-magnitude head** — high-Re samples
  drive extremes; explicit Re-conditioning of the output scale may reduce
  loss-channel coupling.
- **Auxiliary divergence or gradient penalty losses** — incompressible flow
  divergence-free constraint as a physics regularizer.

This document is a living summary — update after each PR cycle to reflect
which themes compounded and which directions have been ruled out.
