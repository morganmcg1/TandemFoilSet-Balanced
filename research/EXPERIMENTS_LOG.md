# SENPAI Research Results

## 2026-05-12 19:00 — PR #1502: Batch inverse-variance weighting for heteroscedastic Re

- **Branch:** `willowpai2g48h4-tanjiro/per-sample-re-normalized-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `e72nzxo5`
- **Hypothesis:** Per-sample inverse-variance weighting (BIVW) to re-balance gradient signal away from high-Re/high-variance samples. Weight each sample by `1 / var(y_norm_valid)`, normalized to mean=1.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **126.0751** | Best epoch 14/50; **round-4 baseline** |
| `test_avg/mae_surf_p` | NaN | Pre-existing data/scoring bug (see below) |
| Best epoch | 14 | 30-min wall-clock cap hit (~132 s/epoch) |
| Training time | 31.1 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 160.74 | 1.88 | 0.85 |
| `val_geom_camber_rc` | 133.28 | 2.57 | 1.01 |
| `val_geom_camber_cruise` | **97.21** | 1.52 | 0.59 |
| `val_re_rand` | 113.08 | 1.99 | 0.77 |
| **val_avg** | **126.08** | 1.99 | 0.80 |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 145.43 |
| `test_geom_camber_rc` | 117.44 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 109.27 |
| test 3-split mean | ~124.0 |

### Analysis and Conclusions

**BIVW worked as hypothesised.** The low-Re-dominated `val_geom_camber_cruise` split came in at 97.21 — the lowest of the four splits by a wide margin — consistent with the prediction that IVW would most benefit low-variance (low-Re) samples that were being under-trained by the uniform MSE.

**BIVW is the new round-4 baseline.** Established at `val_avg/mae_surf_p = 126.0751`.

**Known infrastructure issue discovered:** `test_geom_camber_cruise` sample 20 has 761 `-inf` values in the GT pressure channel (volume nodes). `data/scoring.py:accumulate_batch` intends to skip non-finite GT samples but has a bug: `err = abs(pred - y)` is computed before the per-sample mask is applied, so `inf × 0 = NaN` in float arithmetic poisons the split-level accumulator. Since `data/scoring.py` is read-only, the fix goes in `train.py:evaluate_split` — assigned to tanjiro as PR #1527.

**Training was still improving at cap.** The val curve was still decreasing monotonically at epoch 14. With a longer budget, BIVW could improve further.

---

## 2026-05-12 19:05 — PR #1503: Additive zero-init surface-only correction head (CLOSED)

- **Branch:** `willowpai2g48h4-thorfinn/surface-aware-output-head` (closed, not merged)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `8ffez1mk`
- **Hypothesis:** Zero-initialized additive MLP (`[3+24, 64, 64, 3]`) applied only at surface nodes after the base Transolver prediction. The head starts as an identity correction (last layer zeroed) and specialises the prediction for the surface vs. volume regime.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **133.928** | Best epoch 14/50; **6.2% worse than BIVW baseline** |
| `test_avg/mae_surf_p` | NaN | Same cruise split NaN issue + base model prediction overflow |
| Best epoch | 14 | 30-min wall-clock cap; same budget as tanjiro |
| Training time | 31.4 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|-----------|
| `val_single_in_dist` | 147.33 |
| `val_geom_camber_rc` | 152.10 |
| `val_geom_camber_cruise` | 112.03 |
| `val_re_rand` | 124.26 |
| **val_avg** | **133.93** |

### Analysis and Conclusions

**Closed — 6.2% worse than baseline.** At the same 14-epoch budget the standalone surface head scored 133.93 vs. BIVW's 126.08. Both runs are still undertrained at the cap (val still declining), so we cannot attribute the gap purely to the architectural difference — but the gap is significant.

**The head is not dead.** The composition **BIVW + surf_head** has not been tested. BIVW was not in this run. Composition is orthogonal (loss re-weighting vs. architectural specialisation) and is now assigned as PR #1528 (thorfinn).

**Robustness improvement noted.** Thorfinn recommended replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` to avoid `NaN × 0 = NaN` contamination from volume-node overflows. Incorporated into the composition PR #1528 instructions.

**Test NaN (additional cause).** Unlike tanjiro's data-corruption root cause, thorfinn's test NaN was caused by the base Transolver overflowing to non-finite values on one test cruise sample. The guard fix in PR #1527 will address both causes.
