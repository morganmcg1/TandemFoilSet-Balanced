# Baseline — icml-appendix-charlie-pai2i-48h-r5

## Current Best

### 2026-05-16 03:21 — PR #3529: Grad-clip relaxed to 1.0 on full stack — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **84.01** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **72.95** (NaN-safe eval)
- **Improvement over prior best**: -0.69% val / -1.27% test vs clip=0.25 baseline (84.59/73.89)
- **Cumulative improvement**: -34.7% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 82.86 | -4.6% ✓ |
  | geom_camber_rc | 84.34 | -2.2% ✓ |
  | geom_camber_cruise | 53.59 | +4.1% |
  | re_rand | 71.01 | 0.0% |
- **Metric artifacts**: `models/model-fourier-tmax20-clip10-20260516-013254/metrics.jsonl`
- **Key finding**: clip=1.0 outperforms clip=0.25. clip_frac drops below 1.0 starting at epoch 10 (0.997 at ep10, 0.984 at ep14) — the only threshold where the clip stops saturating within the 14-epoch budget. Pre-clip grad_norm_mean at ep14 is ~5.4; clip=0.25 is 21× below this, eliminating almost all gradient magnitude information. clip=1.0 is the first threshold where the clip is actually adaptive. cruise split slightly regresses (+4.1%) but single and rc show clear improvement.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 1.0 \
      --experiment_name fourier-tmax20-clip10 \
      --agent charliepai2i48h5-frieren
  ```

---

### 2026-05-15 23:28 — PR #3333: Fourier n=10 + Huber-0.3 + T_max=20 + clip=0.25 — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **84.59** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **73.89** (NaN-safe eval)
- **Improvement over prior best**: -5.2% val / -7.0% test vs Fourier-only baseline (89.27/79.43)
- **Cumulative improvement**: -34% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 86.87 |
  | geom_camber_rc | 86.21 |
  | geom_camber_cruise | 51.47 |
  | re_rand | 71.01 |
- **Metric artifacts**: `models/model-fourier-n10-tmax20-clip025-20260515-222425/metrics.jsonl`
- **Key finding**: All four orthogonal improvements compose cleanly: Fourier n=10 + Huber delta=0.3 + LR cosine T_max=20 + grad_clip_max_norm=0.25. Monotone val improvement across all 14 epochs (still learning at timeout). clip_frac=1.0 at 0.25 with Fourier — clip is still doing real work. Cruise and re_rand splits show largest absolute gains (-9.6% / -9.2%).
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 0.25 \
      --experiment_name fourier-n10-tmax20-clip025 \
      --agent charliepai2i48h5-frieren
  ```

---

### 2026-05-15 19:52 — PR #3221: Fourier positional features (n_freqs=10) — charliepai2i48h5-nezuko

- **val_avg/mae_surf_p**: **89.27** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **79.43** (NaN-safe eval)
- **Improvement over prior best**: -9.5% val / -9.9% test vs Huber-0.3+clip-0.25 (98.62/88.14)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 93.65 |
  | geom_camber_rc | 88.94 |
  | geom_camber_cruise | 56.92 |
  | re_rand | 78.20 |
- **Metric artifacts**: `models/model-charliepai2i48h5-nezuko-fourier-n10-20260515-191358/metrics.jsonl`
- **Key finding**: Replacing raw (x,z) coordinates with multi-frequency Fourier positional embeddings (sin/cos at log-spaced frequencies) gives a 9.5% val improvement with near-zero parameter overhead (~4k extra params). `space_dim = 2 + 4*n_freqs = 42`. Best epoch was the last wall-clock-capped epoch — improvement is NOT from running longer, the budget cutoff fired before epoch 50.
- **Stack**: Huber delta=0.3 (no grad_clip in this run); Fourier features alone beat the Huber+clip baseline.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --experiment_name fourier-n10 \
      --agent charliepai2i48h5-nezuko
  ```

---

### 2026-05-15 19:26 — PR #3182: Huber loss + gradient clipping (clip=0.25) — charliepai2i48h5-askeladd

- **val_avg/mae_surf_p**: **98.62** (best_epoch=14/50)
- **test_avg/mae_surf_p**: **88.14** (NaN-safe eval)
- **Improvement over prior best**: -4.4% val / -4.2% test vs Huber-only (103.18/92.02)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 104.75 |
  | geom_camber_rc | 104.65 |
  | geom_camber_cruise | 59.24 |
  | re_rand | 83.90 |
- **Metric artifacts**: `models/model-charliepai2i48h5-askeladd-huber-0.3-clip-0.25-20260515-182526/metrics.jsonl`
- **Key finding**: Huber-0.3 + grad_clip=0.25 are additive — both attack heavy-tail gradients at different scales (per-sample residual vs batch-level update). clip_frac=1.0 at both 0.5 and 0.25; residual tail pressure signal still present after Huber, so clipping contributes genuine variance reduction.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --experiment_name huber-0.3-clip-0.25 \
      --huber_delta 0.3 \
      --grad_clip_max_norm 0.25 \
      --agent charliepai2i48h5-askeladd
  ```

---

### 2026-05-15 16:28 — PR #3213: Huber loss (delta=0.3) — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **103.18** (best_epoch=13/50)
- **test_avg/mae_surf_p**: **92.02** (NaN-safe re-eval; baseline eval had NaN from data bug)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 111.93 |
  | geom_camber_rc | 102.85 |
  | geom_camber_cruise | 62.84 |
  | re_rand | 90.45 |
- **Metric artifacts**: `models/model-huber-0.3-20260515-140457/metrics.jsonl`
- **Also included**: NaN-safe `evaluate_split` fix (sample-level skip for non-finite GT, works around data bug in `test_geom_camber_cruise/000020.pt`)
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --experiment_name huber-0.3 \
      --huber_delta 0.3 \
      --agent charliepai2i48h5-frieren
  ```

## Reference Configuration (baseline `train.py`)
- Model: Transolver
  - n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  - space_dim=2, fun_dim=22, out_dim=3
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Cosine annealing LR schedule, T_max=epochs
- batch_size=4
- surf_weight=10.0 (loss = vol_loss + 10 * surf_loss)
- Default epochs=50 cap, SENPAI_TIMEOUT_MINUTES wall-clock cap
- Balanced WeightedRandomSampler across domains (single/RC-tandem/cruise-tandem)
- Loss in normalized target space; metrics in denormalized physical units

## Splits (lower is better — surface MAE on pressure)
| Split | Test source | Notes |
|---|---|---|
| val_single_in_dist | random holdout from single-foil | sanity |
| val_geom_camber_rc | raceCar M=6-8 front foil | geometry extrapolation |
| val_geom_camber_cruise | cruise M=2-4 front foil | geometry extrapolation |
| val_re_rand | stratified Re across all tandem domains | Re generalization |

Primary metric: equal-weight average across all 4 splits.

## Notes
- Round 5, charlie arm, 48h budget, local JSONL metrics only.
- 8 students, 1 GPU (96GB) each.
- First batch of hypotheses includes a clean baseline run to anchor the metric.
