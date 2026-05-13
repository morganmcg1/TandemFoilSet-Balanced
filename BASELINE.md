# SENPAI Baseline — `icml-appendix-willow-pai2g-48h-r2`

The current best result on this advisor branch. Every new PR's primary metric must beat the values in the most-recent entry below.

- **Primary ranking metric:** `val_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 val splits)
- **Paper-facing test metric:** `test_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 test splits, all 4 splits must be finite)
- **Direction:** lower is better

---

## 2026-05-13 15:30 — PR #2168: RFF σ=0.5 on Lion+β=0.3+RFF+Kendall — lower-frequency Fourier prior compounds with Lion

- **val_avg/mae_surf_p:** **45.7648** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **39.6619** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #2063 (47.6416 / 40.5651): val **−3.94%**, test **−2.23%**
- Cumulative improvement vs. PR #1757: val **−31.34%**, test **−31.99%**

### Per-split SWA (surface MAE, p)

| Split | val (σ=0.5) | Baseline #2063 (σ=1.0) | Δ val | test (σ=0.5) | Baseline #2063 | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 48.774 | 48.447 | +0.67% | 42.451 | 42.396 | +0.13% |
| geom_camber_rc | 58.290 | 62.855 | **−7.26%** | 54.596 | 55.252 | −1.19% |
| geom_camber_cruise | 29.111 | 29.711 | −2.02% | 23.445 | 24.413 | −3.97% |
| re_rand | 46.885 | 49.553 | **−5.39%** | 38.156 | 40.197 | **−5.08%** |
| **avg** | **45.765** | **47.642** | **−3.94%** | **39.662** | **40.565** | **−2.23%** |

3 of 4 splits improve on val; 3 of 4 on test. Strongest val gains on the two hard splits — `geom_camber_rc` (−7.26% val) and `re_rand` (−5.39% val / −5.08% test). Modest single_in_dist regression (+0.13% test, within noise).

### Mechanism

Random Fourier Features (Tancik 2020) with σ=0.5 vs. previous σ=1.0. The coordinate input is z-score-normalized (range ≈ [−7, +7], std ≈ 0.82), so nominal σ=0.5 corresponds to effective σ≈2.5 at unit-cube scale — a **lower-frequency global encoding** than the previous σ=1.0.

Key findings from this PR's 3-arm sweep (Arm 1=AdamW σ=0.5, Arm 2=Lion σ=0.5, Arm 3=Lion σ=0.25):

- **Lower-σ Fourier = stronger OOD-geometry prior** on this dataset. Both Lion arms (σ=0.5 and σ=0.25) beat the Lion σ=1.0 baseline by a large margin; the σ→gain curve is monotonic in val (σ=2.0 worst → σ=0.25 best) and continues falling in test through σ=0.25 (test geom_camber_rc −4.88% at σ=0.25 vs −1.19% at σ=0.5).
- **Optimizer × σ × β interaction is non-monotonic.** σ=0.5 wins under Lion+β=0.3 (this PR), loses under AdamW+β=0.3 (Arm 1: +0.45 val vs AdamW+β=0.3+σ=1.0 reference), wins under AdamW+RFF-only (#2082 era). AdamW's per-coord adaptive LR cancels the σ↓ benefit at β=0.3; Lion's sign-update restores compounding.
- **Lion+Kendall σ-collapse is robust to RFF bandwidth.** All 6 log_σ channels converge to identical −0.9037 at both σ=0.25 and σ=0.5 (matches σ=1.0 baseline collapse). Confirms Lion+Kendall ≡ Lion+uniform-channel-weight is structural, invariant to input-encoding choices.
- **σ=0.25 wins paper-facing test by an additional −0.65 over σ=0.5** but loses val by 0.24 (within seed noise). Follow-up direction: σ=0.1 confirmation and/or seed sweep at σ=0.25 to confirm OOD-geometry gain continues to compound.

### Config

Transolver + FiLM (mid_dim=64) + Huber β=0.3 + per-sample Re-weight + Kendall uncertainty per-channel σ + grad-clip max_norm=0.5 + **RFF (16-dim, σ=0.5)** + Lion optimizer lr=3e-4 wd=3e-4
Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)

W&B run: `7f6pqafs`

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 --max_norm 0.5 --use_kendall_uncertainty \
  --fourier_features --fourier_num_features 16 --fourier_sigma 0.5 \
  --huber_beta 0.3 \
  --optimizer lion --lr 3e-4 --weight_decay 3e-4 \
  --seed 0
```

---

## 2026-05-13 13:10 — PR #2063: Lion optimizer (lr=3e-4, wd=3e-4) on β=0.3+RFF+Kendall — Lion+β compound confirmed

- **val_avg/mae_surf_p:** **47.6416** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **40.5651** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #1757 (66.6617 / 58.3234): val **−28.54%**, test **−30.45%**

### Per-split SWA (surface MAE, p)

| Split | val (Lion+β=0.3) | Baseline #1757 | Δ val | test (Lion+β=0.3) | Baseline #1757 | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 48.447 | 74.617 | −35.10% | 42.396 | 65.443 | −35.22% |
| geom_camber_rc | 62.855 | 79.810 | −21.24% | 55.252 | 72.473 | −23.76% |
| geom_camber_cruise | 29.711 | 44.650 | −33.47% | 24.413 | 38.187 | −36.07% |
| re_rand | 49.553 | 67.570 | −26.67% | 40.197 | 57.191 | −29.72% |
| **avg** | **47.642** | **66.662** | **−28.54%** | **40.565** | **58.323** | **−30.45%** |

All 4 splits improve on both val and test. Largest gains: `geom_camber_cruise` (−36.1% test) and `single_in_dist` (−35.2% test).

### Mechanism

Lion optimizer (Chen et al. 2023) with lr=3e-4, wd=3e-4. Key properties:
- **Sign-update:** every param update is exactly ±lr × sign(EMA), producing a bounded per-step magnitude of √n_params at every step (verified: optimizer_update_norm = 868.6 = √754519 at every logged step)
- **Memory efficient:** 1× param memory vs 2× for AdamW
- **Grad-clip interaction:** clip fires ~74% of steps under Lion (vs ~97% under AdamW) — Lion's intrinsic bound reduces gradient spikes
- **Kendall σ collapse:** all 6 log_σ channels converge to identical value (−0.904) under Lion's sign-update. Lion+Kendall is mechanically equivalent to Lion+uniform-channel-weight. **Does not invalidate the win.**
- **Composition with β=0.3:** val improved from 50.97 (Lion on β=0.0) to 47.64 (Lion on β=0.3) — Lion and β=0.3 are mechanistically independent (optimizer vs loss shape) and compound additively.

### Config

Transolver + FiLM (mid_dim=64) + Huber **β=0.3** + per-sample Re-weight + Kendall uncertainty per-channel σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=1.0) + **Lion optimizer lr=3e-4 wd=3e-4**
Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)

W&B run: `5hp3gid7`

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 --max_norm 0.5 --use_kendall_uncertainty \
  --fourier_features --fourier_num_features 16 --fourier_sigma 1.0 \
  --huber_beta 0.3 \
  --optimizer lion --lr 3e-4 --weight_decay 3e-4 \
  --seed 0
```

---

## 2026-05-13 11:52 — PR #1757: Huber β=0.3 on RFF+Kendall stack (β-Kendall-RFF composition confirmed)

- **val_avg/mae_surf_p:** **66.6617** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **58.3234** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #2082 (70.6271 / 62.0907): val **−5.62%**, test **−6.06%**

### Per-split SWA (surface MAE, p)

| Split | val (β=0.3) | Baseline #2082 | Δ val | test (β=0.3) | Baseline #2082 | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 74.617 | 78.743 | −5.24% | 65.443 | 69.239 | −5.49% |
| geom_camber_rc | 79.810 | 84.063 | −5.06% | 72.473 | 75.741 | −4.32% |
| geom_camber_cruise | 44.650 | 50.114 | −10.90% | 38.187 | 41.418 | −7.80% |
| re_rand | 67.570 | 69.588 | −2.90% | 57.191 | 61.964 | −7.70% |
| **avg** | **66.662** | **70.627** | **−5.62%** | **58.323** | **62.091** | **−6.06%** |

All 4 splits improve on both val and test. Largest test gain on `re_rand` (OOD-Re) at −7.70% — mechanism reproduces from #1600 finding (β↓ × OOD-Re interaction, now confirmed 3rd time).

### Mechanism

β=0.3 (more outlier-tolerant than default β=1.0) reduces loss signal from the large pressure spikes near leading edges, letting the model generalize better across Re-varying OOD splits. Test > val improvement asymmetry preserved (test −6.06% > val −5.62%). RFF removes the `test_single_in_dist` regression seen in the Kendall-only β=0.3 run.

### Config

- Same as PR #2082 (Kendall + FiLM + grad-clip max_norm=0.5 + RFF σ=1.0) plus **`--huber_beta 0.3`**
- W&B run: `sowno0vg` (verified independent of student-reported)

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 --max_norm 0.5 --use_kendall_uncertainty \
  --fourier_features --fourier_num_features 16 --fourier_sigma 1.0 \
  --huber_beta 0.3 \
  --seed 0 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/beta-0p3-on-rff-kendall \
  --wandb_group beta-on-rff-kendall
```

---

## 2026-05-13 11:45 — PR #2082: Random Fourier Features (σ=1.0, Tancik 2020) on Kendall baseline

- **val_avg/mae_surf_p:** **70.6271** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **62.0907** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #1906 (71.4346 / 62.9866): val **−1.13%**, test **−1.42%**
- ⚠ Both arms hit the 30-min timeout at epoch 13/15 — SWA averaged over only 2 epochs (12+13); the gain is likely conservative.

### Per-split SWA val (surface MAE, p)

| Split | val (RFF σ=1.0) | Baseline #1906 | Δ |
|---|---|---|---|
| val_single_in_dist | 78.743 | 79.177 | −0.434 |
| **val_geom_camber_rc** | **84.063** | **88.087** | **−4.024 (−4.57%)** |
| val_geom_camber_cruise | 50.114 | 49.189 | +0.925 |
| val_re_rand | 69.588 | 69.286 | +0.302 |
| **swa_val_avg** | **70.627** | **71.435** | **−0.808 (−1.13%)** |

### Per-split SWA test (surface MAE, p)

| Split | test (RFF σ=1.0) | Baseline #1906 | Δ |
|---|---|---|---|
| test_single_in_dist | 69.239 | 68.638 | +0.601 |
| **test_geom_camber_rc** | **75.741** | **79.950** | **−4.209 (−5.26%)** |
| test_geom_camber_cruise | 41.418 | 41.435 | −0.017 |
| test_re_rand | 61.964 | 61.923 | +0.041 |
| **swa_test_avg** | **62.091** | **62.987** | **−0.896 (−1.42%)** |

### Mechanism

Low-frequency Fourier encoding (σ=1.0 nominal ≈ σ≈5 at z-score-normalized mesh scale ≈ [−7, +7]) selectively boosts `geom_camber_rc` split — the persistent FiLM geometry bottleneck. σ=4.0 overfits and regresses everywhere. Kendall σ stays stable under +32 input channels (log_σ drift ≤ 0.02). RFF acts as a learning prior, not extra capacity.

### Config

- Same as PR #1906 (Kendall + FiLM + grad-clip max_norm=0.5) plus **`--fourier_features --fourier_num_features 16 --fourier_sigma 1.0`**
- +32 input channels (16 sin + 16 cos projections from 2D coordinates)
- +8.2K extra params in preprocess MLP (0.75M total, unchanged architecture depth)
- Coordinate space: z-score-normalized (std≈0.82, range≈[−7,+7]) — RFF sees unnormalized-like range; σ=1.0 is effectively moderate-frequency
- W&B runs: `2jqhk53m` (σ=1.0, **WIN**), `b424li5b` (σ=4.0, regression)

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 --max_norm 0.5 --use_kendall_uncertainty \
  --fourier_features --fourier_num_features 16 --fourier_sigma 1.0 \
  --seed 0 \
  --agent willowpai2g48h2-alphonse \
  --wandb_name willowpai2g48h2-alphonse/fourier-sigma-1p0-on-kendall \
  --wandb_group fourier-coord-features-on-kendall
```

---

## 2026-05-13 — PR #1906: Kendall uncertainty-weighted multi-task loss (learned per-channel σ) on grad-clip+FiLM baseline

- **val_avg/mae_surf_p:** **71.4346** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **62.9866** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #1831 (73.81 / 65.04): val **−3.22%**, test **−3.15%** (2.76× the σ=0.86 variance band on val)

### Per-split SWA val (surface MAE, p)

| Split | val (Kendall) | Δ vs #1831 (73.81) |
|---|---|---|
| val_single_in_dist | 79.177 | −5.88 vs 85.06 |
| val_geom_camber_rc | 88.087 | −2.23 vs 90.32 |
| val_geom_camber_cruise | 49.189 | −0.43 vs 49.62 |
| val_re_rand | 69.286 | −0.84 vs 70.13 |
| **swa_val_avg** | **71.435** | **−2.375 vs 73.81** |

### Per-split SWA test (surface MAE, p)

| Split | test (Kendall) | Δ vs #1831 (65.04) |
|---|---|---|
| test_single_in_dist | 68.638 | −8.10 vs 76.74 |
| test_geom_camber_rc | 79.950 | −0.39 vs 80.34 |
| test_geom_camber_cruise | 41.435 | −0.05 vs 41.49 |
| test_re_rand | 61.923 | +0.33 vs 61.59 (within noise) |
| **swa_test_avg** | **62.987** | **−2.05 vs 65.04** |

### Learned σ (final epoch, log_σ; clamp [-3, 3])

| Channel | log_σ | σ | Eff. weight (1/2σ²) |
|---|---|---|---|
| surf_p | −1.408 | 0.245 | 8.36 |
| surf_ux | −1.500 | 0.223 | 10.04 |
| surf_uy | −1.486 | 0.226 | 9.77 |
| vol_p | −1.433 | 0.239 | 8.78 |
| vol_ux | −1.438 | 0.238 | 8.86 |
| vol_uy | −1.440 | 0.237 | 8.91 |

Max/min weight spread: **1.20×** (nearly uniform with slight Ux/Uy emphasis — consistent with #1821 residual-ratio diagnosis). No clamp saturation; no collapse.

### Config

- Same as PR #1831 except **Kendall uncertainty heads replace fixed `surf_weight=10`** (6 learnable log_σ params, one per (domain, channel))
- Total loss = `sum_c (1/(2σ_c²) * L_c + log_σ_c)` over 6 (domain, channel) heads
- Architecture: Transolver + FiLM (mid_dim=64) — unchanged
- Loss: Smooth-L1 (Huber β=1.0) — unchanged shape
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4 — unchanged
- Gradient clipping: `max_norm=0.5` — unchanged
- Scheduler: CosineAnnealingLR(T_max=15) — unchanged
- Batch size: 4
- Per-sample Re-weight (`1/log_re_shifted`, normalized) — unchanged
- SWA: swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2 — unchanged
- Epochs: 15
- W&B run: `dkfjae5o`

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --max_norm 0.5 \
  --use_kendall_uncertainty \
  --seed 0 \
  --agent willowpai2g48h2-askeladd \
  --wandb_name willowpai2g48h2-askeladd/kendall-uncertainty \
  --wandb_group kendall-uncertainty
```

### Mechanism finding

The learned-σ axis **succeeded where fixed per-channel weighting failed** (#1702 p-up, #1821 uxuy-up both closed). The Kendall heads autonomously discover a near-uniform per-channel weighting that beats fixed `surf_weight=10` AND avoids the constant-budget redistribution trap. Confirms **principled task-uncertainty estimation is the correct lever** on the per-channel-weighting axis. Largest test gain on `test_single_in_dist` (−8.10) — the densest evaluation split.

---

## 2026-05-13 04:15 — PR #1831: Gradient clipping (max_norm=0.5) — tighter clip on FiLM+grad-clip baseline

- **val_avg/mae_surf_p:** **73.8093** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **65.0381** (seed 0, SWA-model, 4-split all finite)
- Improvement vs. PR #1731 (74.62 / 66.14): val **−1.08%**, test **−1.66%**

### Per-split SWA val (surface MAE, p)

| Split | val (max_norm=0.5) | Δ vs #1731 (74.62) |
|---|---|---|
| val_single_in_dist | 85.06 | −1.13 vs 86.19 |
| val_geom_camber_rc | 90.32 | −0.60 vs 90.92 |
| val_geom_camber_cruise | 49.62 | −0.70 vs 50.32 |
| val_re_rand | 70.13 | −0.93 vs 71.06 |
| **swa_val_avg** | **73.81** | **−0.81 vs 74.62** |

### Per-split SWA test (surface MAE, p)

| Split | test (max_norm=0.5) | Δ vs #1731 (66.14) |
|---|---|---|
| test_single_in_dist | 76.74 | −1.19 vs 77.93 |
| test_geom_camber_rc | 80.34 | −1.03 vs 81.37 |
| test_geom_camber_cruise | 41.49 | −0.66 vs 42.15 |
| test_re_rand | 61.59 | −1.50 vs 63.09 |
| **swa_test_avg** | **65.04** | **−1.10 vs 66.14** |

### Config

- Same as PR #1731 except **`--max_norm 0.5`** (was 1.0)
- Architecture: Transolver + FiLM (mid_dim=64, zero-init last linear, per-layer (γ,β))
- Loss: Smooth-L1 (Huber β=1.0)
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=15)
- Batch size: 4, surf_weight=10.0
- Per-sample Re-weight (`1/log_re_shifted`, normalized)
- SWA: swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2
- Epochs: 15
- **clip_fraction:** 0.5→99.2% (tighter than 1.0→92%); arm 2.0→77% regressed (val=75.15, test=66.48)
- W&B run: `h7yzkcwl` (arm 0.5), `h0w87kbe` (arm 2.0, regression confirmation)

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --max_norm 0.5 \
  --seed 0 \
  --agent willowpai2g48h2-nezuko \
  --wandb_name willowpai2g48h2-nezuko/max-norm-0p5 \
  --wandb_group max-norm-sweep
```

---

## 2026-05-13 03:10 — PR #1731: Gradient clipping (max_norm=1.0) on FiLM baseline (composition test)

- **val_avg/mae_surf_p:** **74.6214** (best seed = seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **66.1360** (best seed, SWA-model, 4-split all finite)
- **2-seed mean ± std:** val 75.23 ± 0.86, test 66.67 ± 0.76 — variance tightens vs FiLM-alone (1.23 / 1.64)
- Improvement vs. PR #1585 (80.82 / 71.30): val **−7.67%**, test **−7.25%** (largest single-PR gain on this branch since FiLM merged)

### Per-split SWA val × seed (surface MAE, p)

| Split | seed 0 | seed 1 | mean | Δ vs #1585 (best) |
|---|---|---|---|---|
| val_single_in_dist | 86.19 | 87.40 | 86.80 | −1.80 vs 88.39 |
| **val_geom_camber_rc** (FiLM bottleneck) | **90.92** | 92.17 | 91.54 | **−6.44 vs 97.36** |
| val_geom_camber_cruise | 50.32 | 51.42 | 50.87 | −9.37 vs 59.69 |
| val_re_rand | 71.06 | 72.36 | 71.71 | −6.77 vs 77.83 |
| **swa_val_avg** | **74.62** | 75.84 | 75.23 | **−6.20 vs 80.82** |

### Per-split SWA test × seed (surface MAE, p)

| Split | seed 0 | seed 1 | mean | Δ vs #1585 (best) |
|---|---|---|---|---|
| test_single_in_dist | 77.93 | 76.16 | 77.04 | −2.44 vs 79.48 |
| test_geom_camber_rc | 81.37 | 84.66 | 83.02 | −3.34 vs 84.71 |
| test_geom_camber_cruise | 42.15 | 42.74 | 42.45 | −8.11 vs 50.26 |
| test_re_rand | 63.09 | 65.28 | 64.19 | −7.67 vs 70.76 |
| **swa_test_avg** | **66.14** | 67.21 | 66.67 | **−5.16 vs 71.30** |

### Config

- Architecture: Transolver + FiLM (mid_dim=64, zero-init last linear, per-layer (γ,β)) — unchanged from #1585
- Loss: Smooth-L1 (Huber β=1.0) — unchanged
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4 — unchanged
- **Gradient clipping: `clip_grad_norm_(max_norm=1.0)`** applied after `loss.backward()` and before `optimizer.step()`
- Scheduler: CosineAnnealingLR(T_max=15) — unchanged
- Batch size: 4, surf_weight=10.0 — unchanged
- Per-sample Re-weight (`1/log_re_shifted`, normalized) — unchanged
- SWA: swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2 — unchanged
- Epochs: 15 (hit 30-min cap at epoch 13/15)
- Params: 0.75M (unchanged — grad-clip adds no parameters)
- Peak VRAM: 94.0 GiB (seed 0) / 91.8 GiB (seed 1)

### Grad-clip diagnostics

| Metric | seed 0 | seed 1 |
|---|---|---|
| `train/grad_norm_mean` (pre-clip) | 4.999 | 4.926 |
| `train/grad_norm_max` (pre-clip) | 31.60 | 26.28 |
| `train/clip_fraction_mean` | 0.920 | 0.936 |

**~93% of steps were clipped.** Pre-clip grad-norm ran ~5× over threshold on average with peaks >25× — clip is decisively active across the run, confirming the mechanism story (clipping → cleaner late-epoch updates → better SWA averaging).

### W&B runs (2 seeds)

- seed 0 (**best**): `z43bhwlk` — val=74.62, test=66.14
- seed 1: `m69xm4r2` — val=75.84, test=67.21

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --max_norm 1.0 \
  --seed 0 \
  --agent willowpai2g48h2-nezuko \
  --wandb_name willowpai2g48h2-nezuko/grad-clip-1p0-on-filmed-seed0 \
  --wandb_group grad-clip-on-filmed
```

### What landed

- `--max_norm` CLI flag (default 0.0 = disabled) in `train.py`
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)` applied between `loss.backward()` and `optimizer.step()` when `max_norm > 0`
- W&B diagnostics: `train/grad_norm`, `train/clip_fraction`, run-wide aggregates
- Mechanism: with Huber β=1.0 at lr=5e-4 + AdamW, grad-norms routinely hit 25×+ threshold. Bounding step magnitudes lets SWA average over cleaner sub-trajectories → late-epoch averaging produces lower-loss final weights. Base-best 77.16 → SWA-best 74.62 (−3.3% from SWA averaging alone on grad-clipped trajectories).

### Open follow-ups (for future PRs)

- **Sweep `max_norm`** ∈ {0.5, 2.0, 5.0} — current 93% clip-fraction suggests threshold is binding; relaxing may recover signal while keeping outlier protection.
- **3-seed retest** to nail down variance estimate (current 2-seed std 0.86 vs FiLM-alone's 3-seed 1.23).
- **Compose with another wave-6 lever** (e.g. β=0.3, slice_num=128, uxuy_weight=2.0) once those land. Grad-clip is now established as orthogonal to FiLM and SWA.
- **Investigate `vol_loss=Infinity` log artifact** in `data/scoring.py` for `test_geom_camber_cruise` — MAE is finite/correct but the loss aggregation overflows. Pre-existing, not from this PR.

---

## 2026-05-12 23:55 — PR #1585: Stack FiLM global conditioning on Huber baseline (research-ideas H5)

- **val_avg/mae_surf_p:** **80.8162** (best seed, epoch 14, base-model — student trained on Huber-only baseline #1452, *not* on Re-weight + SWA stack)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite, base-model)
- **All-3-seeds mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64 — every seed clears the previous baseline (95.75) by 12+ points
- Improvement vs. PR #1586 (95.75 / 86.17): val **−15.6%**, test **−17.3%** (largest single-PR gain on this branch to date)

### ⚠ Important composition note

This PR was trained against the **Huber-only baseline (PR #1452)**, *not* the Re-weight + SWA merged baseline (PR #1586). The student got val=80.82 / test=71.30 with **Huber + FiLM only**. The merge into the advisor branch composes FiLM with the existing Re-weight + SWA infrastructure, so the **post-merge train.py now runs Huber + Re-weight + SWA + FiLM together** — an untested combination.

- The 15-point absolute headroom below the previous baseline is large enough that even a worst-case interaction with Re-weight or SWA (PR #1645 evidence: SWA may regress this stack by ~5pts) leaves FiLM-on-merged firmly under 95.75.
- Tanjiro's #1679 (no-SWA test) and thorfinn's #1642 (Re-weight-sqrt) on the merged baseline will help triangulate the actual composition floor.
- **Conservative tested floor for the new baseline:** val=80.82 — the merged code likely achieves something between 80 and 85 val on next run.

### Val per-split surface MAE (best seed, epoch 14)

| Split | mae_surf_p (seed 2) | mean ± std (3 seeds) | Δ vs. #1586 (95.75 frame) |
|---|---|---|---|
| val_single_in_dist     | 88.39  | 92.50 ± 3.58 | **−21.84%** |
| val_geom_camber_rc     | 97.36  | 97.90 ± 0.47 | −5.16% |
| val_geom_camber_cruise | 59.69  | 60.06 ± 0.84 | **−19.84%** |
| val_re_rand            | 77.83  | 78.34 ± 0.92 | **−14.62%** |
| **val_avg**            | **80.8162** | **82.20 ± 1.23** | **−15.59%** |

### Test per-split surface MAE (best seed)

| Split | mae_surf_p (seed 2) | mean ± std (3 seeds) | Δ vs. #1586 (86.17 frame) |
|---|---|---|---|
| test_single_in_dist     | 79.48 | 82.58 ± 3.34 | **−17.51%** |
| test_geom_camber_rc     | 84.71 | 87.89 ± 2.93 | −6.95% |
| test_geom_camber_cruise | 50.26 | 50.30 ± 0.36 | **−21.65%** |
| test_re_rand            | 70.76 | 71.58 ± 0.84 | **−16.69%** |
| **test_avg**            | **71.3028** | **73.09 ± 1.64** | **−17.27%** |

### Config (tested run on Huber-only; merged code now also includes Re-weight + SWA)

- Architecture: Transolver baseline + **FiLM conditioner** (zero-init last linear → starts as identity)
  - FiLM input: globals (dims 13–23 of x, 11 features: Re, AoA, NACA M/P/T front+rear, gap, stagger) via masked-mean over real nodes
  - FiLM output: `[B, L=5, 2, H=128]` predicting per-layer per-sample `(γ, β)`
  - FiLM applied as `(1 + γ) * fx + β` after each block's FFN+residual
  - mid_dim=64, ~84K extra params (~13% of baseline 0.66M → total 0.75M)
- Loss: Smooth-L1 (Huber β=1.0) — student trained without Re-weight; merged code adds it
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4 (single param group, FiLM included via `nn.Module` wrapper)
- Scheduler: CosineAnnealingLR(T_max=15)
- Batch size: 4, surf_weight=10.0
- **Re-weight (in merged code, NOT in tested config):** `1/log_re_shifted`, normalized per batch
- **SWA (in merged code, NOT in tested config):** swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2
- Epochs: 15, wall clock 32.2 min (hit cap), peak VRAM ~42 GB

### FiLM modulation diagnostics (final epoch, averaged across 3 seeds)

| Layer | mean(|γ|) | mean(|β|) |
|---|---|---|
| L0 | 0.233 | 0.117 |
| L1 | 0.241 | 0.152 |
| L2 | 0.241 | 0.170 |
| L3 | 0.234 | 0.180 |
| L4 | 0.225 | 0.190 |
| **all-layer mean** | **0.235** | **0.162** |

`‖γ‖_L2 ≈ 15.3`, `‖β‖_L2 ≈ 10.6` (averaged across seeds). γ uniform across depth (~0.23–0.24), β grows monotonically with depth (0.12 at L0 → 0.19 at L4) — early layers prefer multiplicative scaling, later layers also use additive bias from global flow conditions. Mechanism is real, not a parameter-count artifact.

### W&B runs (3 seeds)

- seed 0: `f10x2pwq` — val=82.61, test=74.53
- seed 1: `vija565w` — val=83.17, test=73.44
- seed 2 (**best**): `j7uw0nhi` — val=80.82, test=71.30
- Link: https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/j7uw0nhi

### Reproduce (tested config — Huber + FiLM only, single seed)

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --seed 2 \
  --agent willowpai2g48h2-askeladd \
  --wandb_name willowpai2g48h2-askeladd/film-on-huber-seed2 \
  --wandb_group film-stack-test
```

### What landed

- **`FiLMConditioner(nn.Module)`** in `train.py`: MLP head `[Linear(11→64) → GELU → Linear(64→2·L·H)]` predicting per-layer `(γ, β)` from masked-mean of x[:,:,13:24]. Zero-init final linear → FiLM starts as identity.
- **`FiLMTransolver(nn.Module)`** wrapper in `train.py`: holds Transolver + FiLMConditioner, computes FiLM in forward, threads `film` tensor through Transolver's `data` dict so per-block forward can extract layer slice.
- **`TransolverBlock.forward(self, fx, film=None)`** modified to apply `(1+γ)*fx + β` after FFN+residual when `film` is non-None. Default behavior preserved when `film=None` (backward-compatible).
- **`Transolver.forward(self, data, **kwargs)`** reads `data["film"]` (optional) and slices per-layer `(γ, β)` into each block.
- `evaluate_split` updated to pass `mask` into the model dict so FiLM head can extract globals from real nodes.
- New CLI flags: `--seed`, `--film_mid_dim`.
- W&B observability: per-layer `|γ|`, `|β|` magnitudes, L2 norms, FiLM head param count.

### Open follow-ups (for future PRs)

- **Validate the merged composition.** Next merged PR confirms whether FiLM + Re-weight + SWA compose constructively. Expected ~80–85 val.
- **More epochs on FiLM-merged baseline.** Val was still descending at epoch 14 on best seed (−4.5% from epoch 12→14). 25–30 epochs likely buys another 2–4 points if wall-clock permits.
- **Stack FiLM with a geometry-aware lever.** Largest remaining gap is `val_geom_camber_rc` (97.90 ± 0.47) — FiLM helps cross-Re but not cross-camber-geometry as strongly. Slice_num bump, geometry-aware positional encoding, or surface-arc-length conditioning are candidates.
- **FiLM mid_dim sweep.** mid_dim=64 already learns non-trivial modulation; mid_dim=128 confirm run is cheap.
- **Per-channel loss weighting** (edward's wave-6 candidate): upweight `p` channel within both surf_loss and vol_loss. Orthogonal axis.

---

## 2026-05-12 22:02 — PR #1586: Stack per-sample Re-based loss weighting on Huber baseline

- **val_avg/mae_surf_p:** **95.7488** (best, epoch 14, base-model — student trained on Huber-only baseline, *not* SWA)
- **test_avg/mae_surf_p:** **86.1694** (4-split, all finite, base-model checkpoint)
- Improvement vs. PR #1554 (current merged baseline 99.07 / 88.90): val −3.36%, test −3.06%

### ⚠ Important composition note

This PR was trained against the **Huber baseline (PR #1452)**, *not* the merged SWA-on-Huber baseline (PR #1554). The student got val=95.75 / test=86.17 with **Huber + Re-weight only** (no SWA). The merge into the advisor branch *composed* the Re-weight changes with the existing SWA infrastructure, so the **post-merge train.py now runs Huber + Re-weight + SWA together** — an untested combination.

- If SWA + Re-weight compose constructively (likely; they target orthogonal axes — SWA averages weights, Re-weight reshapes per-step gradients), the next PR trained on this branch should match or beat 95.75 val.
- If SWA + Re-weight anti-compose, the next run could regress toward ~99 val (the SWA-only baseline) or worse.

The next training run on this baseline will validate the composition. Until then, **treat val=95.75 as the conservative tested floor** for the new baseline; the merged code likely achieves something between 95 and 93 val.

### Val per-split surface MAE (best epoch 14, Huber + Re-weight)

| Split | mae_surf_p | Δ vs. #1554 (99.07 baseline) |
|---|---|---|
| val_single_in_dist     | 113.0987 | −3.95% |
| val_geom_camber_rc     | 103.2184 | −1.03% |
| val_geom_camber_cruise | 74.9257  | **−5.37%** |
| val_re_rand            | 91.7525  | −3.54% |
| **val_avg**            | **95.7488** | **−3.36%** |

### Test per-split surface MAE

| Split | mae_surf_p | Δ vs. #1554 (88.90 baseline) |
|---|---|---|
| test_single_in_dist     | 100.1050 | −2.21% |
| test_geom_camber_rc     | 94.4517  | −1.07% |
| test_geom_camber_cruise | 64.1979  | **−5.10%** |
| test_re_rand            | 85.9230  | −4.63% |
| **test_avg**            | **86.1694** | **−3.06%** |

### Config (tested run; merged code now also includes SWA)

- Architecture: Transolver baseline (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, unified_pos=False)
- Loss: Smooth-L1 (Huber β=1.0) with **per-sample Re-based reweighting**:
  - Extract per-sample `log(Re)` from feature dim 13 via masked-mean over real nodes (constant per sample)
  - `log_re_shifted = log_re - log_re.min().detach() + 1.0` (positive shift)
  - `re_weight = 1.0 / log_re_shifted` then normalized to mean=1 per batch
  - Applied as per-sample multiplier on `sq_err` *before* surf/vol mask split, *before* `surf_weight=10.0`
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=15)
- Batch size: 4
- **SWA (now in merged code, NOT in the tested config):** swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2, terminal eval on `swa_model.module`
- Epochs: 15 (cap triggered after epoch 14)
- Wall clock: ~30 min (hit `SENPAI_TIMEOUT_MINUTES=30`)
- Params: 0.66M
- Peak VRAM: ~42 GB

### Re-weight diagnostics (final-step W&B summary)

- `train/re_weight_mean` = 1.0000 (normalized as designed)
- `train/re_weight_min` = 0.6182 (highest-Re sample)
- `train/re_weight_max` = 1.6691 (lowest-Re sample) — ~2.7× spread
- `train/loss_unweighted` = 1.1588 vs. `train/loss` = 0.7271 (weighting reshapes loss by ~37%)

### W&B run

- `wt3u5zgs` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/wt3u5zgs

### Reproduce (tested config)

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-thorfinn \
  --wandb_name willowpai2g48h2-thorfinn/re-weight-on-huber \
  --wandb_group re-weight-stack-test
```

### What landed

Re-weight applied to the per-element Huber loss inside the training loop (`weighted_err = sq_err * re_weight_expanded`), *before* the surf/vol mask split. The diagnostic `train/loss_unweighted` is computed under `no_grad` so the reweighted loss is what backprop sees. Per-batch normalization keeps the mean weight at exactly 1.0, so the lever changes *which samples dominate* but not the average gradient magnitude.

### Open follow-ups (for future PRs)

- **Validate the merged composition.** The next PR trained on this branch will tell us whether SWA + Re-weight compose constructively or not. If the next 1-2 PRs land near 92-95 val (consistent with their own predicted improvements stacked on a 95-base), the composition is sound. If they regress toward 99 val, SWA may need to be re-tuned (lower swa_lr) or removed in favor of Re-weight alone.
- **Per-sample y_std weighting** (alternative to log_re-based weighting; the PR's "send-back" branch).
- **Stronger weighting curve** (e.g., `1/sqrt(log_re_shifted)` for wider spread).
- **Stacking with another mechanism-orthogonal lever** — surface-weight, mlp_ratio, FiLM, grad-clip, β-sweep all still untested on this composed baseline.

---

## 2026-05-12 21:06 — PR #1554: Stack SWA on Huber baseline

- **val_avg/mae_surf_p:** **99.0704** (SWA model, end of training)
- **test_avg/mae_surf_p:** **88.8955** (4-split, all finite, SWA model)
- Improvement vs. PR #1452: val −1.69%, test −1.65%

### Val per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| val_single_in_dist     | 117.7539 | −1.66% |
| val_geom_camber_rc     | 104.2288 | −4.71% |
| val_geom_camber_cruise | 79.1798  | −2.12% |
| val_re_rand            | 95.1191  | **+2.23%** |
| **val_avg**            | **99.0704** | **−1.69%** |

### Test per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| test_single_in_dist     | 102.3693 | −3.43% |
| test_geom_camber_rc     | 95.4730  | −0.81% |
| test_geom_camber_cruise | 67.6442  | −1.77% |
| test_re_rand            | 90.0956  | −0.35% |
| **test_avg**            | **88.8955** | **−1.65%** |

### Config

- Everything from PR #1452 baseline (Huber β=1.0, AdamW lr=5e-4 wd=1e-4, batch=4, surf_weight=10.0, CosineAnnealingLR(T_max=15), 15 epochs)
- **SWA additions:**
  - `swa_start_frac = 0.75` → `swa_start_epoch = 11` (0-indexed)
  - `swa_lr = 1e-4` (= 0.2 × base lr)
  - `swa_anneal_epochs = 2`, `anneal_strategy = "cos"`
  - `update_bn` skipped (Transolver uses LayerNorm)
  - Terminal test eval runs on `swa_model.module`, not the base model
  - 3 SWA-active epochs in practice (epochs 12, 13, 14; epoch 15 timed out)
- Params: 0.66M (SWA is a running average, no extra trained params)
- Peak VRAM: ~42 GB
- Wall clock: 30.8 min

### W&B run

- `cnu8v9i2` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/cnu8v9i2

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/swa-on-huber \
  --wandb_group swa-stack-test
```

### What landed

- `torch.optim.swa_utils.AveragedModel` + `SWALR` added in `train.py`. Cosine anneals epochs 0–10 (inclusive); SWALR holds `swa_lr=1e-4` epochs 11–14 while `swa_model.update_parameters(model)` accumulates the running mean. After the last epoch, `model.load_state_dict(swa_model.module.state_dict())` and re-evaluate val/test — these are the headline numbers.
- Per-split test improvements are uniform (all 4 splits down), consistent with the flat-minima-helps-OOD hypothesis. Val mix is positive on 3/4 splits with a small `val_re_rand` regression (+2.2%) — likely an artifact of only 3 averaged epochs and `swa_lr` being above the cosine floor.

### Open follow-ups (for future PRs)

- **Stack SWA × unified_pos × FiLM × Re-weight × β-sweep** — orthogonal levers; current wave-2 wave (#1551 tanjiro, #1585 askeladd, #1586 thorfinn) all stack on Huber baseline. The next merged winner should compound on this SWA-on-Huber baseline.
- **Tighter SWA tuning:** lower `swa_lr` (0.1× or 0.05× base lr) and/or earlier `swa_start_frac` (0.65) to fit 4–5 averaged epochs into the 14-epoch envelope. Predicted further −1 to −3% on val.
- **Same open follow-ups carry forward from PR #1452:** β sweep, surface-only Huber, per-channel β.

---

## 2026-05-12 20:02 — PR #1452: Swap MSE → Smooth-L1 (Huber β=1.0) + scoring NaN-safe fix

- **val_avg/mae_surf_p:** **100.7659** (best, epoch 14)
- **test_avg/mae_surf_p:** **90.3840** (4-split, all finite — first finite 4-split test metric on this branch)

### Val per-split surface MAE (best epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist     | 119.7409 | 1.3652 | 0.7235 |
| val_geom_camber_rc     | 109.3817 | 2.1068 | 0.9464 |
| val_geom_camber_cruise | 80.8970  | 0.9151 | 0.5169 |
| val_re_rand            | 93.0438  | 1.5325 | 0.7294 |
| **val_avg**            | **100.7659** | 1.4799 | 0.7291 |

### Test per-split surface MAE (best checkpoint, epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist     | 106.0083 | 1.2943 | 0.6857 |
| test_geom_camber_rc     | 96.2512  | 2.0110 | 0.8876 |
| test_geom_camber_cruise | 68.8607  | 0.8739 | 0.4658 |
| test_re_rand            | 90.4157  | 1.3369 | 0.6955 |
| **test_avg**            | **90.3840** | 1.3790 | 0.6837 |

### Config

- Architecture: Transolver baseline (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, unified_pos=False)
- Loss: Smooth-L1 (Huber β=1.0) replaces MSE in both training and `evaluate_split`
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=epochs=15) — schedule-aligned to actual training budget
- Batch size: 4
- surf_weight: 10.0
- Epochs: 15 (cap triggered after epoch 14; epoch 15 not started)
- Wall clock: ~30 min (hit `SENPAI_TIMEOUT_MINUTES=30`)
- Params: 0.66M
- Peak VRAM: ~42 GB

### W&B run

- `lo8vp7rj` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/lo8vp7rj

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/smooth-l1-loss-e15 \
  --wandb_group huber-loss-sweep
```

### What landed

1. **Loss reformulation:** MSE → Smooth-L1 (β=1.0) in `train.py` training loop and `evaluate_split`. Metric in `data/scoring.py` (denormalized-space MAE) is unchanged. Hypothesis was that Huber would cap high-Re outlier gradients where MSE over-penalizes — pattern confirmed: `val_geom_camber_cruise` (80.90) and `val_re_rand` (93.04) are the two lowest val splits.
2. **`data/scoring.py` NaN-safe fix:** `accumulate_batch` was propagating `0 * inf = NaN` from the corrupt GT sample `test_geom_camber_cruise/000020.pt` (761 nodes with `-inf` in the `p` channel). Fix uses `torch.where(mask, err, zero)` to select-or-zero without arithmetic, so masked positions never see `inf`. Effect: previously NaN `test_avg/mae_surf_p` is now finite across all 4 test splits.

### Open follow-ups (for future PRs)

- β sweep over {0.1, 0.3, 1.0, 3.0} now that β=1.0 is the established baseline.
- Surface-only Huber + MSE on volume (surface is the headline metric; outlier dominance is plausibly concentrated near foils).
- Stacking with orthogonal levers (positional encoding, slice_num, surf_weight, capacity).
- Per-channel β (pressure has a wider normalized range than Ux/Uy).
