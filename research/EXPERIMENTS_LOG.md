# SENPAI Research Results

## Branch: icml-appendix-charlie-pai2c-r3

## 2026-04-27 20:18 — PR #209: EMA weight averaging (decay=0.999) — smoother generalization

- **Branch**: charliepai2c3-nezuko/ema-weight-averaging
- **Hypothesis**: EMA of model weights (decay=0.999) maintained throughout training and used for all validation, checkpoint selection, and test evaluation. Expected 0.5–2% improvement with zero extra compute at training time, especially for OOD camber splits.
- **Results**:

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 171.74 | 2.02 | 0.88 | 162.44 | 5.95 | 2.49 |
| val_geom_camber_rc     | 146.90 | 3.18 | 1.12 | 143.01 | 6.39 | 3.07 |
| val_geom_camber_cruise | 100.14 | 1.46 | 0.60 | 100.80 | 4.28 | 1.56 |
| val_re_rand            | 115.87 | 2.17 | 0.84 | 114.32 | 5.11 | 2.18 |
| **val avg**            | **133.66** | **2.21** | **0.86** | **130.14** | **5.43** | **2.32** |
| test_single_in_dist    | 143.91 | 1.92 | 0.81 | 140.24 | 5.60 | 2.26 |
| test_geom_camber_rc    | 132.09 | 3.05 | 1.06 | 130.20 | 6.22 | 2.93 |
| test_geom_camber_cruise|  85.50 | 1.32 | 0.55 |  88.35 | 4.11 | 1.43 |
| test_re_rand           | 116.84 | 1.99 | 0.83 | 116.30 | 4.91 | 2.10 |
| **test avg**           | **119.58** | **2.07** | **0.82** | **118.77** | **5.21** | **2.18** |

- **Metric summary**: `target/models/model-charliepai2c3-nezuko-ema-weight-averaging-20260427-192048/metrics.jsonl`
- **Epochs run**: 14/50 (timed out at 30-min wall clock, ~132 s/epoch, still improving monotonically)
- **Peak VRAM**: 42.11 GB

**Commentary**: This is the first result on the icml-appendix-charlie-pai2c-r3 track, so it establishes the baseline. val_avg/mae_surf_p=133.66 at epoch 14 with a monotonically decreasing trajectory — had the run not timed out, it would have continued improving. The OOD camber-cruise split performs significantly better (100.14) than the in-dist split (171.74), which is a surprising and encouraging sign.

**Critical bug fix (merged independently)**: The student identified a NaN-poisoning bug in `evaluate_split` caused by 1 sample in `test_geom_camber_cruise` having 761 NaN GT pressure values. The original `data/scoring.py` intended to mask these out, but `(NaN * 0) = NaN` in IEEE arithmetic caused the entire split accumulator to become NaN. Fix: sanitize GT before multiplication, proactively zero padding-position predictions. This is a no-op on all val splits and 3/4 test splits. Fix is now part of `train.py` on the advisor branch.

**Decision**: MERGED. First result establishes baseline. Bug fix is critical and now protects all future experiments.
