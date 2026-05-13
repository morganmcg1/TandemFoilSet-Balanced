# TandemFoilSet Baseline Metrics

Primary metric: `val_avg/mae_surf_p` — lower is better.
Paper metric: `test_avg/mae_surf_p` — lower is better.

---

## 2026-05-12 14:00 — PR #1502: Batch inverse-variance weighting for heteroscedastic Re

- **Branch:** `icml-appendix-willow-pai2g-48h-r4` (merged)
- **W&B run:** `e72nzxo5`
- **Best epoch:** 14 / 50 configured (hit 30-min wall-clock cap)
- **val_avg/mae_surf_p:** `126.0751` ← **current best**
- **test_avg/mae_surf_p:** `NaN` (pre-existing data/scoring bug: test_geom_camber_cruise sample 20 has 761 -inf in GT p-channel; `0×inf=NaN` in accumulate_batch poisons the split-average)

### Per-split val surface-p MAE (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 160.7360 | 1.8779 | 0.8524 |
| `val_geom_camber_rc` | 133.2787 | 2.5736 | 1.0051 |
| `val_geom_camber_cruise` | 97.2075 | 1.5158 | 0.5869 |
| `val_re_rand` | 113.0781 | 1.9906 | 0.7708 |
| **val_avg** | **126.0751** | 1.9895 | 0.8038 |

### Per-split test surface-p MAE (best checkpoint — 3 of 4 clean)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 145.4262 |
| `test_geom_camber_rc` | 117.4369 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 109.2676 |
| **test_avg** | **NaN** (cruise split corrupts mean) |

Mean of 3 valid test splits: **~124.0** (indicative only).

### Reproduce

```bash
cd target && python train.py \
    --wandb_group per-sample-re-normalized-loss \
    --wandb_name bivw-mean1-clamp1e-4 \
    --agent willowpai2g48h4-tanjiro
```

### Notes

- BIVW weights each sample by `1 / var(y_norm_valid)`, normalized to mean=1.
  This re-balances gradient signal away from high-Re/high-variance samples.
- Test NaN is a known infrastructure issue, not a model quality issue.
  Until fixed: report all four individual test split numbers; compute manual
  3-split mean excluding cruise as a surrogate paper metric.

---

## 2026-05-12 20:30 — PR #1528: BIVW + zero-init surface correction head composition

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-on-bivw` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **W&B run:** `an97gg8n`
- **Best epoch:** 13 / 14 run (hit 30-min wall-clock cap)
- **val_avg/mae_surf_p:** `119.2987` ← **current best** (−5.37% vs prior 126.0751)
- **test_avg/mae_surf_p:** `NaN` (same pre-existing cruise split bug)

### Per-split val surface-p MAE (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 140.09 | — | — |
| `val_geom_camber_rc` | 142.40 | — | — |
| `val_geom_camber_cruise` | 85.98 | — | — |
| `val_re_rand` | 108.73 | — | — |
| **val_avg** | **119.2987** | — | — |

### Per-split test surface-p MAE (best checkpoint — 3 of 4 clean)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 127.93 |
| `test_geom_camber_rc` | 127.18 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 103.79 |
| **test_avg** | **NaN** (cruise split corrupts mean) |

Mean of 3 valid test splits: **~119.63** (indicative only).

### Reproduce

```bash
cd target && python train.py \
    --wandb_group surf-head-on-bivw \
    --wandb_name bivw-surf-head-zeroinit \
    --agent willowpai2g48h4-thorfinn
```

### Notes

- Composition: BIVW loss weighting (sample-level) + zero-init additive SurfaceCorrection MLP head (architectural specialisation).
- SurfaceCorrection head: `[3+24, 64, 64, 3]`, last layer zeroed at init, applied only at surface nodes.
- Regression on `val_geom_camber_rc` (+6.84%); three other splits improved substantially.
- Total params: 0.669M (Transolver 0.643M + SurfaceCorrection 0.026M).
- **All future PRs must beat `val_avg/mae_surf_p < 119.2987` to merge.**
- Test NaN infrastructure fixed in PR #1527 (merged) — `evaluate_split` now `nan_to_num`-guards both `pred_orig` and `y` before `accumulate_batch`, and passes an explicit `_y_ok` finite-sample mask. From PR #1527 forward, expect all four test split `mae_surf_p` values to be finite.
- Indicative test_avg from tanjiro's BIVW-only PR #1527 run (`dg5xbm6g`, no surf-head): `test_avg/mae_surf_p = 119.7792` with `test_geom_camber_cruise = 81.42`. Actual test_avg for BIVW+surf-head+fix combo pending next merged run.

---

## 2026-05-12 22:00 — PR #1558: Huber (SmoothL1) surface loss, delta=0.5

- **Branch:** `willowpai2g48h4-thorfinn/smooth-l1-surface-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **W&B run:** `2w7nverc` (winning arm, delta=0.5); `3goyvktl` (delta=1.0 secondary)
- **Best epoch:** 14 / 14 completed (hit 30-min wall-clock cap; still improving)
- **val_avg/mae_surf_p:** `98.1642` ← **current best** (−17.72% vs prior 119.2987)
- **test_avg/mae_surf_p:** `NaN` (cruise split pre-existing bug); **test 3-split mean: 98.7537** (−17.45% vs ~119.63)

### Per-split val surface-p MAE (best checkpoint, delta=0.5)

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 123.14 | 140.09 → **−12.1%** ✓ |
| `val_geom_camber_rc` | 107.24 | 142.40 → **−24.7%** ✓ (OOD regression fully reversed) |
| `val_geom_camber_cruise` | 73.28 | 85.98 → **−14.8%** ✓ |
| `val_re_rand` | 88.99 | 108.73 → **−18.2%** ✓ |
| **val_avg** | **98.1642** | **−17.72%** |

### Per-split test surface-p MAE (3 of 4 clean, delta=0.5)

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `test_single_in_dist` | 111.92 | 127.93 → **−12.5%** |
| `test_geom_camber_rc` | 98.91 | 127.18 → **−22.2%** |
| `test_geom_camber_cruise` | NaN | (pre-existing cruise bug) |
| `test_re_rand` | 85.43 | 103.79 → **−17.7%** |
| **test 3-split mean** | **98.7537** | **−17.45%** |

### Reproduce

```bash
cd target && python train.py \
    --huber_delta 0.5 \
    --wandb_group smooth-l1-surface-loss \
    --wandb_name huber-delta-0.5 \
    --agent willowpai2g48h4-thorfinn
```

### Notes

- Huber loss (SmoothL1, delta=0.5) on surface, MSE on volume. Applied to all 3 surface channels jointly.
- `delta=0.5` wins because most surface residuals in normalised space (~O(0.3–1.5)) fall in the L1 regime, giving constant-magnitude gradients that directly minimise MAE.
- `delta=1.0` gives only −1.3% (barely above noise floor) — too much residual in quadratic regime.
- Reverses val_geom_camber_rc OOD regression from PR #1528 (+6.84% → −24.7%): Huber suppresses the large-residual surf-head pull toward OOD outlier nodes.
- Synergy with BIVW: BIVW removes between-sample gradient inflation; Huber removes within-sample per-node gradient inflation — orthogonal channels that compound.
- **All future PRs must beat `val_avg/mae_surf_p < 98.1642` to merge.**
- Test cruise NaN is unchanged; use 3-split mean as surrogate paper metric.

---

## 2026-05-13 05:30 — PR #1795: Decoupled LR for surf_head vs encoder

- **val_avg/mae_surf_p: 97.9914** (best checkpoint epoch 11)
- **test_avg/mae_surf_p: 88.5311** (4-split, cruise now finite post #1527)
- **test 3-split mean (excl cruise): 99.5856** (apples-to-apples vs prior 98.7537; slight +0.85 regression on 3-split)
- **W&B run:** `eg1rhrzg` (surf_head_lr=5e-3, arm 3)
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --huber_delta 0.5 \
      --surf_head_lr 5e-3 \
      --wandb_group decoupled-lr-surf-head \
      --wandb_name surf-head-lr-5e-3 \
      --agent willowpai2g48h4-thorfinn
  ```

### Per-split val surface-p MAE (best checkpoint epoch 11)

| Split | mae_surf_p | vs prior baseline (98.1642) |
|-------|-----------|------------------------------|
| `val_single_in_dist` | 120.31 | 123.14 → **−2.3%** ✓ |
| `val_geom_camber_rc` | 115.98 | 107.24 → +8.2% ✗ |
| `val_geom_camber_cruise` | 66.04 | 73.28 → **−9.9%** ✓ |
| `val_re_rand` | 89.64 | 88.99 → +0.7% ≈ |
| **val_avg** | **97.9914** | **−0.18% ✓** |

### Per-split test surface-p MAE

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 112.27 |
| `test_geom_camber_rc` | 104.81 |
| `test_geom_camber_cruise` | 55.36 (finite, NaN guard active) |
| `test_re_rand` | 81.69 |
| **test 4-split mean** | **88.5311** |
| **test 3-split mean (no cruise)** | **99.5856** |

### Notes

- surf_head uses `surf_head_lr=5e-3` (10× encoder LR of 5e-4) via a separate AdamW param group.
- Encoder LR unchanged at 5e-4. Both groups share weight_decay=1e-4.
- Monotonic trend across sweep: arm 1 (1e-3, +15.9%) → arm 2 (3e-3, +6.6%) → arm 3 (5e-3, −0.18%). Margin not yet exhausted at 5e-3.
- Late training oscillation at 5e-3: best epoch 11 (97.99), oscillates to ~108-113 before settling to 99.85 at epoch 14.
- Test 3-split slightly regresses (+0.85) vs val improvement (−0.17); cruise (finite now) dramatically improves (73→55).
- **All future PRs must beat `val_avg/mae_surf_p < 97.9914` to merge.**

---

## 2026-05-13 09:00 — PR #2031: Weight decay re-tune {1e-4 → 5e-4}

- **val_avg/mae_surf_p: 93.6198** (best checkpoint epoch 14, still improving at wall-clock cap) — **−4.46% vs 97.9914**
- **test_avg/mae_surf_p: 83.8825** — **−5.26% vs 88.5311**
- **W&B run:** `u3q47f4s` (weight_decay=5e-4 arm)
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --huber_delta 0.5 \
      --surf_head_lr 5e-3 \
      --weight_decay 5e-4 \
      --wandb_group weight-decay-sweep \
      --wandb_name weight-decay-5e-4 \
      --agent willowpai2g48h4-fern
  ```

### Per-split val surface-p MAE (best checkpoint epoch 14)

| Split | mae_surf_p | vs prior baseline (97.9914) |
|-------|-----------|------------------------------|
| `val_single_in_dist` | 109.61 | 120.31 → **−8.9%** ✓ |
| `val_geom_camber_rc` | 118.05 | 115.98 → +1.8% ≈ |
| `val_geom_camber_cruise` | 61.07 | 66.04 → **−7.5%** ✓ |
| `val_re_rand` | 85.75 | 89.64 → **−4.3%** ✓ |
| **val_avg** | **93.6198** | **−4.46% ✓** |

### Per-split test surface-p MAE (best checkpoint)

| Split | mae_surf_p | vs prior baseline (88.5311) |
|-------|-----------|------------------------------|
| `test_single_in_dist` | 101.87 | 112.27 → **−9.3%** ✓ |
| `test_geom_camber_rc` | 105.24 | 104.81 → +0.4% ≈ |
| `test_geom_camber_cruise` | 51.43 | 55.36 → **−7.1%** ✓ |
| `test_re_rand` | 76.99 | 81.69 → **−5.8%** ✓ |
| **test 4-split mean** | **83.8825** | **−5.26% ✓** |
| **test 3-split mean (no cruise)** | **94.7000** | |

### Notes

- `weight_decay=5e-4` (5× prior `1e-4`). Stacks on the full Huber δ=0.5 + decoupled surf_head_lr=5e-3 recipe.
- Best epoch=14 means trajectory was still descending at the wall-clock cap. Cosine LR `T_max=50` means we've only traversed 28% of the cycle; true plateau not reached.
- **Test gain (−5.26%) stronger than val gain (−4.46%)** — not val-overfitting; the regularization gain generalizes.
- Wins on 3 of 4 val/test splits cleanly; flat on `geom_camber_rc` (the OOD camber holdout). The biggest wins are on splits where the surf_head's 10× LR was pushing hardest (`single_in_dist`, `cruise`).
- Late-epoch trajectory note: WD=5e-4 had a transient epoch-12 spike (152.99) followed by recovery to 93.62 at epoch 14 — same late-epoch oscillation pattern observed across other baselines, but the post-spike recovery breached a new minimum.
- **All future PRs must beat `val_avg/mae_surf_p < 93.6198` to merge.**
- **Hyperparameter staleness principle confirmed:** the −4.46% gain from a single 5× WD bump validates that the optimizer hyperparameters inherited pre-Huber/pre-decoupled-LR are systematically suspect. Re-evaluation of other axes (encoder_lr, β2, etc.) on the new baseline is warranted.

---

## 2026-05-13 11:00 — PR #2091: torch.compile throughput unlock (21 epochs in 30 min)

- **val_avg/mae_surf_p: 89.7197** (best checkpoint epoch 18) — **−4.16% vs 93.6198**
- **test_avg/mae_surf_p: 79.3167** — **−5.44% vs 83.8825**
- **W&B run:** `fvlekakd` (mode=default, dynamic=True)
- **Training config:** weight_decay=1e-4 (old default — branch predates PR #2031 merge), huber_delta=0.5, surf_head_lr=5e-3
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --huber_delta 0.5 \
      --surf_head_lr 5e-3 \
      --weight_decay 1e-4 \
      --use_torch_compile \
      --compile_mode default \
      --wandb_group torch-compile \
      --wandb_name torch-compile-default \
      --agent willowpai2g48h4-frieren
  ```
- **NOTE:** This result was achieved with weight_decay=1e-4 (the old pre-#2031 default). Composing with weight_decay=5e-4 is the highest-priority next experiment for frieren.

### Per-split val surface-p MAE (best checkpoint epoch 18)

| Split | mae_surf_p | vs prior baseline (93.6198) |
|-------|-----------|------------------------------|
| `val_single_in_dist` | 114.92 | 109.61 → +4.9% ↑ |
| `val_geom_camber_rc` | 108.66 | 118.05 → **−8.0%** ✓ |
| `val_geom_camber_cruise` | 55.45 | 61.07 → **−9.2%** ✓ |
| `val_re_rand` | 79.85 | 85.75 → **−6.9%** ✓ |
| **val_avg** | **89.7197** | **−4.16% ✓** |

### Per-split test surface-p MAE (best checkpoint)

| Split | mae_surf_p | vs prior baseline (83.8825) |
|-------|-----------|------------------------------|
| `test_single_in_dist` | 104.29 | 101.87 → +2.4% ↑ |
| `test_geom_camber_rc` | 96.30 | 105.24 → **−8.5%** ✓ |
| `test_geom_camber_cruise` | 46.12 | 51.43 → **−10.3%** ✓ |
| `test_re_rand` | 70.55 | 76.99 → **−8.4%** ✓ |
| **test 4-split mean** | **79.3167** | **−5.44% ✓** |

### Notes

- `torch.compile(mode="default", dynamic=True)` — single symbolic-shape graph, no per-shape recompile chaos (only 2 frames compiled despite 74K–242K node mesh sizes). Compile warmup: 8.8s (negligible).
- **Throughput:** 21 epochs in 30 min vs prior 14 (1.43× per-epoch speedup, 50% more epochs in same budget).
- `mode="reduce-overhead"` OOM'd at epoch 1, batch 219 (CUDA Graph private pool accumulation on variable shapes). Incompatible without upstream bucketed-padding.
- Also enables: `torch.set_float32_matmul_precision("high")` (TF32 matmul), `cudnn.benchmark=True`.
- **val_single_in_dist slightly regresses** (+4.9% vs prior baseline) while OOD geometry splits dramatically improve (geom_camber_rc −8%, cruise −9.2%). This trade-off likely reflects weight_decay=1e-4 (old default) being under-regularized at 21 epochs — in-distribution split may be benefiting less from longer training than OOD.
- **Highest-priority follow-up:** compose torch.compile + weight_decay=5e-4 (which is now the default on this branch). Expected to recover the in-distribution split regression while preserving the OOD gains.
- **All future PRs must beat `val_avg/mae_surf_p < 89.7197` to merge (superseded — see PR #2178 below).**

---

## 2026-05-13 13:25 — PR #2178: Compose torch.compile + weight_decay=3e-4 (new SOTA)

- **val_avg/mae_surf_p: 87.0144** (best checkpoint epoch 21) — **−3.01% vs 89.7197**
- **test_avg/mae_surf_p: 78.9539** — **−0.46% vs 79.3167**
- **W&B run:** `7r9t0jab` (WD=3e-4, winning arm)
- **Reproduce:**
  ```bash
  cd target && python train.py \
      --huber_delta 0.5 \
      --surf_head_lr 5e-3 \
      --weight_decay 3e-4 \
      --use_torch_compile \
      --compile_mode default \
      --wandb_group compile-wd-compose \
      --wandb_name compile-wd-3e-4 \
      --agent willowpai2g48h4-frieren
  ```

### Per-split val surface-p MAE (best checkpoint epoch 21)

| Split | mae_surf_p | vs prior baseline (89.7197) |
|-------|-----------|------------------------------|
| `val_single_in_dist` | 106.99 | 114.92 → **−6.9%** ✓ |
| `val_geom_camber_rc` | 104.00 | 108.66 → **−4.3%** ✓ |
| `val_geom_camber_cruise` | 57.33 | 55.45 → +3.4% ↑ |
| `val_re_rand` | 79.74 | 79.85 → −0.1% ≈ |
| **val_avg** | **87.0144** | **−3.01% ✓** |

### Per-split test surface-p MAE (best checkpoint)

| Split | mae_surf_p | vs prior baseline (79.3167) |
|-------|-----------|------------------------------|
| `test_single_in_dist` | 95.70 | 104.29 → **−8.2%** ✓ |
| `test_geom_camber_rc` | 97.92 | 96.30 → +1.7% ↑ |
| `test_geom_camber_cruise` | 48.06 | 46.12 → +4.2% ↑ |
| `test_re_rand` | 74.14 | 70.55 → +5.1% ↑ |
| **test 4-split mean** | **78.9539** | **−0.46% ✓** |

### Notes

- WD=3e-4 (not 5e-4) is the optimal under the 21-epoch compile budget. WD=5e-4 over-regularizes at 21 epochs — the e12 spike AMPLIFIES (+27%) under WD=5e-4 but is DAMPED under WD=3e-4 (smooth descent e10→e12).
- WD=5e-4 arm (run `b1p4li7l`) regressed +1.17% val / +3.10% test — do not use.
- In-distribution split fully recovered: val_single_in_dist 114.92 → 106.99 (−6.9%), reversing the +4.9% regression from PR #2091.
- cruise OOD gives back slightly (+3.4% val, +4.2% test) under higher WD; WD=1e-4 preferred on cruise specifically.
- Val gain (−3.01%) is notably larger than test gain (−0.46%) — regularization helps in-dist val substantially but OOD test splits are mixed.
- Both arms at 21 epochs in ~30.7 min (same throughput as PR #2091).
- **Critical lesson:** WD axis is budget-dependent. WD=5e-4 was optimal at 14 epochs (PR #2031); WD=3e-4 is optimal at 21 epochs. PRs in-flight using WD=5e-4 must now beat **87.0144** to merit merge.
- **All future PRs must beat `val_avg/mae_surf_p < 87.0144` to merge.**
