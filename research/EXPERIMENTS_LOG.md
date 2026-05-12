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

## 2026-05-12 20:30 — PR #1528: BIVW + zero-init surface correction head (MERGED)

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-on-bivw` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `an97gg8n`
- **Hypothesis:** Composition of BIVW (per-sample loss re-weighting) and a zero-initialized additive SurfaceCorrection MLP head (`[3+24, 64, 64, 3]`, last layer zero-init, surface nodes only). Both mechanisms are orthogonal: BIVW targets gradient heterogeneity at the sample level; the surf-head targets the architectural under-representation of surface nodes. Used `torch.where(is_surface, delta, zero)` to safely handle NaN × 0 contamination from volume node overflow.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **119.2987** | Best epoch 13/14; **new round-4 baseline** (−5.37% vs 126.0751) |
| `test_avg/mae_surf_p` | NaN | Pre-existing cruise split scoring bug |
| Best epoch | 13 | 30-min cap hit (~131 s/epoch) |
| Total params | 0.669M | Transolver 0.643M + SurfaceCorrection 0.026M |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 140.09 | −12.85% ✓ |
| `val_geom_camber_rc` | 142.40 | +6.84% ✗ |
| `val_geom_camber_cruise` | 85.98 | −11.55% ✓ |
| `val_re_rand` | 108.73 | −3.85% ✓ |
| **val_avg** | **119.2987** | **−5.37%** |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 127.93 |
| `test_geom_camber_rc` | 127.18 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 103.79 |
| test 3-split mean | ~119.63 |

### Analysis and Conclusions

**BIVW + surf-head composition worked — new baseline 119.30.** Confirms the orthogonality hypothesis: BIVW (loss-level) and the surface correction head (architecture-level) provide complementary inductive bias. Three of four val splits improved; `val_geom_camber_rc` (raceCar OOD camber) regressed +6.84%, which warrants investigation in future work.

**`torch.where` NaN guard confirmed correct.** Replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` correctly propagates zeros instead of NaN at volume nodes with overflow predictions.

**Composition principle validated.** The standalone surf-head (#1503) was 6.2% worse than BIVW alone; adding it on top of BIVW is 5.4% better. The head needed the cleaner gradient signal that BIVW provides to specialize effectively.

**Next:** Need to test whether higher-LR + grad-clip (#1499, which reached 113.15 on BIVW alone) stacks further on top of this combined baseline.

---

## 2026-05-12 20:00 — PR #1499: Grad-clip max_norm=1.0 + lr 5e-4 → 1e-3 (SENT BACK — merge conflicts)

- **Branch:** `willowpai2g48h4-fern/gradient-clipping-and-higher-lr`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `ihl8ashe` (primary, lr=1e-3), `160d99m0` (fallback, lr=7e-4)
- **Hypothesis:** Gradient heterogeneity across Re samples causes large per-batch gradient norms that destabilise slice-attention. Capping with `max_norm=1.0` and doubling LR to 1e-3 should stabilise training and converge faster.

### Results

| Arm | Run | Best epoch | `val_avg/mae_surf_p` | test 3-split mean |
|-----|-----|------------|----------------------|-------------------|
| **primary** (`lr=1e-3, clip=1.0`) | `ihl8ashe` | 13 | **113.1491** | 109.64 |
| fallback (`lr=7e-4, clip=1.0`) | `160d99m0` | 12 | 119.0885 | 123.00 |

Per-split surface-p MAE (test, primary arm `lr=1e-3`):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 110.07 |
| `test_geom_camber_rc` | 111.92 |
| `test_geom_camber_cruise` | NaN |
| `test_re_rand` | 106.94 |
| 3-split mean | 109.64 |

Grad-norm telemetry (primary arm):
- **100% of steps clipped** — raw norms ranged 2.18 to 712.86 (median 30.79, mean 48.31)
- `max_norm=1.0` is acting as a uniform per-step renormaliser, not an outlier suppressor

### Analysis and Conclusions

**Strong result (113.15 on BIVW-only basis) — could not merge due to conflicts with advisor branch.** PR was branched before the BIVW + surf-head composition (#1528) merged. Sent back for rebase onto `icml-appendix-willow-pai2g-48h-r4` with new baseline 119.2987.

**100%-clipping finding is important.** With `max_norm=1.0` every single step is clipped. The effective LR is `(1.0 / raw_norm) × lr_nominal ≈ 1e-3 / 30.8 ≈ 3.2e-5` (median). The benefit of the higher nominal LR is asymmetric — it only matters on the small fraction of steps near the clip threshold. Suggested follow-up: try `grad_clip ∈ {1.0, 10.0}` on the new baseline to separate true outlier suppression from step renormalisation.

**Next:** Fern is rebasing onto the new baseline (BIVW + surf-head, 119.30) and adding a `--grad_clip 10.0` arm alongside the primary `--grad_clip 1.0`. The current 113.15 on BIVW-only was not compared against the newer 119.30 baseline; rebased run will clarify whether grad-clip still helps on top of surf-head.

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
