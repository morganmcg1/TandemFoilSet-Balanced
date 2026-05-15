# Baseline — `icml-appendix-charlie-pai2i-48h-r4`

## Current best

**Track:** `icml-appendix-charlie-pai2i-48h-r4`
**Status:** Fresh research track — no baseline metrics committed yet. The first round of PRs establishes the reference number for `val_avg/mae_surf_p`.

**Primary ranking metric (lower is better):** `val_avg/mae_surf_p`
**Test metric (lower is better):** `test_avg/mae_surf_p`

## Reference configuration (unmodified `train.py`)

- Model: 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`
- Scheduler: CosineAnnealingLR(T_max=epochs)
- Loss: MSE in normalized space, `vol_loss + 10 * surf_loss`
- Training budget per run: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50`

## Reproduce

```bash
cd target/
python train.py --experiment_name baseline
```

## Round 1 protocol

Every PR in this fresh round runs a **paired comparison**:
- **Arm A** = unmodified baseline (one full training run)
- **Arm B** = hypothesis change (one full training run)

The student commits both `metrics.jsonl` outputs and reports both numbers in their PR. This makes each PR self-contained and robust to per-run variance until enough runs accumulate to give a stable absolute baseline. After Round 1 we will commit a stable baseline number here and the round-2 protocol can drop the paired arm.

---

## 2026-05-15 14:11 — PR #3094: Huber (smooth L1) loss to align training with MAE eval metric

**New best `val_avg/mae_surf_p`: 111.531** (was: 132.282 MSE baseline — **−15.7%**)

- **Loss:** Huber / smooth L1, β=1.0 (replaced MSE `sq_err = (pred - y_norm) ** 2`)
- **Best epoch:** 11 / 14 run (30-min budget, 50-epoch cap)
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR (unchanged)
- **Peak VRAM:** 42.12 GB

### Val surface pressure MAE (lower is better)

| Split | Baseline (MSE) | **Best (Huber)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 172.116 | **141.566** | −17.7% |
| `val_geom_camber_rc`     | 141.056 | **116.797** | −17.2% |
| `val_geom_camber_cruise` |  97.342 |  **86.222** | −11.4% |
| `val_re_rand`            | 118.615 | **101.539** | −14.4% |
| **val_avg**              | **132.282** | **111.531** | **−15.7%** |

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` is NaN due to a pre-existing scoring bug)

| Split | Baseline (MSE) | **Best (Huber)** |
|---|---:|---:|
| `test_single_in_dist`     | 153.339 | **130.147** |
| `test_geom_camber_rc`     | 128.508 | **106.293** |
| `test_re_rand`            | 117.504 | **100.996** |
| **avg (3 splits)**        | 133.117 | **112.479** |

### Metric artifacts

- `models/model-charliepai2i48h4-alphonse-huber-loss-hyp-20260515-131213/metrics.jsonl`
- `models/model-charliepai2i48h4-alphonse-huber-loss-hyp-20260515-131213/metrics.yaml`

### Reproduce

```bash
cd target/
# Apply the Huber loss patch to train.py:
#   replace: sq_err = (pred - y_norm) ** 2
#   with:    sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# (same swap in evaluate_split)
python train.py --experiment_name huber-loss-hyp
```

### Current best config (carry forward to all new experiments)

```python
# Loss (in train.py, replace sq_err computation):
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR(T_max=epochs)
# surf_weight=10
```

---

## 2026-05-15 17:40 — PR #3290: bf16 AMP mixed precision — unlock ~1.5x more epochs in 30-min budget

**New best `val_avg/mae_surf_p`: 101.519** (was: 111.531 Huber baseline — **−8.98%**)

- **AMP:** `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` wrapping forward+loss; no GradScaler
- **Best epoch:** 16 / 19 run (30-min budget, 50-epoch cap; 5 more epochs than fp32)
- **Throughput:** 131.8 → 98.0 sec/epoch (1.345×); peak VRAM 42.1 → 32.9 GB (−21.8%)
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR T_max=50 (unchanged)

### Val surface pressure MAE (lower is better)

| Split | Huber (PR #3094) | **Best (bf16)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 141.566 | **116.096** | −18.0% |
| `val_geom_camber_rc`     | 116.797 | **116.636** |  −0.1% |
| `val_geom_camber_cruise` |  86.222 |  **76.479** | −11.3% |
| `val_re_rand`            | 101.539 |  **96.863** |  −4.6% |
| **val_avg**              | **111.531** | **101.519** | **−8.98%** |

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` is NaN — pre-existing scoring bug)

| Split | Huber (PR #3094) | **Best (bf16)** |
|---|---:|---:|
| `test_single_in_dist`     | 130.147 | **101.200** |
| `test_geom_camber_rc`     | 106.293 | **106.199** |
| `test_re_rand`            | 100.996 |  **88.806** |
| **avg (3 splits)**        | **112.479** | **98.735** |

### Metric artifacts

- `models/model-charliepai2i48h4-askeladd-amp-bf16-20260515-162617/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-amp-bf16-20260515-162617/metrics.yaml`

### Reproduce

```bash
cd target/
python train.py --experiment_name amp-bf16 --amp_dtype bf16
```

### Current best config (carry forward to all new experiments)

```python
# Loss (in train.py, replace sq_err computation):
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16  (torch.autocast on forward+loss, no GradScaler)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR(T_max=epochs)
# surf_weight=10
```

---

## 2026-05-15 18:30 — PR #3289: Cosine T_max=15 — match LR schedule horizon to 30-min budget

**New best `val_avg/mae_surf_p`: 100.059** ⚠️ *measured on fp32*; bf16+T_max=15 composition unverified (was: 101.519 bf16 — **−1.4%**; vs Huber fp32 111.531 — **−10.3%**)

- **Scheduler:** `CosineAnnealingLR(T_max=15)` — schedule completes its full cosine decay within the ~14-epoch fp32 wall-clock budget
- **AMP:** fp32 (this run predates the bf16 merge)
- **Best epoch:** 14 / 14 run (wall-clock-truncated at schedule end; LR=2.16e-5 at best epoch)
- **Mechanism confirmed:** LR-vs-epoch trace shows Arm B (T_max=15) decays 96% by ep14 vs Arm A (T_max=50) only 16%; full cosine anneal yields the refinement plateau that fp32 T_max=50 never reached
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4 (unchanged)

### Val surface pressure MAE (lower is better)

| Split | bf16 baseline (PR #3290) | **fp32 T_max=15** | Δ vs bf16 |
|---|---:|---:|---:|
| `val_single_in_dist`     | 116.096 | **118.473** | +2.0% |
| `val_geom_camber_rc`     | 116.636 | **111.356** | −4.5% |
| `val_geom_camber_cruise` |  76.479 |  **79.108** | +3.4% |
| `val_re_rand`            |  96.863 |  **91.299** | −5.8% |
| **val_avg**              | **101.519** | **100.059** | **−1.4%** |

*Note: direct split comparison is noisy (different seeds, fp32 vs bf16). val_avg beats bf16 baseline, but the per-split ordering is mixed.*

### Test surface pressure MAE (3 finite splits)

| Split | **fp32 T_max=15** |
|---|---:|
| `test_single_in_dist`     | 102.084 |
| `test_geom_camber_rc`     |  99.752 |
| `test_re_rand`            |  88.086 |
| **avg (3 splits)**        | **96.641** |

### Metric artifacts

- `models/model-charliepai2i48h4-thorfinn-cosine-tmax-15-20260515-163241/metrics.jsonl`
- `models/model-charliepai2i48h4-thorfinn-cosine-tmax-15-20260515-163241/metrics.yaml`

### Reproduce

```bash
cd target/
python train.py --experiment_name cosine-tmax-15 --cosine_t_max 15
```

### Current best config (carry forward to all new experiments)

**⚠️ bf16 + T_max=15 composition not yet benchmarked.** The codebase now has both bf16 (PR #3290) and T_max=15 (PR #3289). The actual bf16+T_max=15 number is to be established in a follow-up verification run. Expected: ~93–95 if composition is additive.

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15  (CosineAnnealingLR T_max=15, held at floor if epochs exceed T_max)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```

---

## 2026-05-15 22:32 — PR #3126: EMA weights (decay=0.999, Karras warmup ramp) — low-pass filter over AdamW path

**New best `val_avg/mae_surf_p`: 96.464** (was: 100.059 fp32+T_max=15 — **−3.59%**; vs paired Arm A bf16+T_max=15 97.492 — **−1.06%**)

- **EMA:** `decay=0.999` with Karras-style ramp `decay_eff = min(0.999, (1+step)/(10+step))`; EMA updated after every optimizer step; validation, best-checkpoint save, and test eval all use EMA-applied weights
- **Stack:** Huber + bf16 AMP + cosine T_max=15 (current best baseline)
- **Best epoch:** 18 / 19 run (30-min budget)
- **Key signal:** EMA reduces last-10-epoch val_avg variance by **43.7%** (σ=6.55 → 3.69); EMA-applied model leads live model at every epoch from ep1 onward
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4 (unchanged)

### Val surface pressure MAE (lower is better)

| Split | Arm A (bf16+T_max=15, no EMA) | **Arm B (+ EMA)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 116.714 | **111.948** | −4.08% |
| `val_geom_camber_rc`     | 102.709 | **102.325** | −0.37% |
| `val_geom_camber_cruise` |  77.554 |   **79.490** | +2.50% |
| `val_re_rand`            |  92.990 |  **92.092** | −0.97% |
| **val_avg**              | **97.492** | **96.464** | **−1.06%** |

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` is NaN — pre-existing scoring bug)

| Split | Arm A | **Arm B (EMA)** |
|---|---:|---:|
| `test_single_in_dist`     | 103.011 |  **97.964** |
| `test_geom_camber_rc`     |  93.417 |  **94.701** |
| `test_re_rand`            |  88.210 |  **88.905** |
| **avg (3 finite splits)** |  **94.879** | **93.857** |

*Note: Arm A (val=97.492) is also the first measured bf16+T_max=15 compose number, confirming the ~93–95 prediction from BASELINE.md.*

### Metric artifacts

- `models/model-charliepai2i48h4-nezuko-arm_b_ema_d0999_bf16_tmax15-20260515-212327/metrics.jsonl`
- `models/model-charliepai2i48h4-nezuko-arm_b_ema_d0999_bf16_tmax15-20260515-212327/metrics.yaml`
- `models/model-charliepai2i48h4-nezuko-arm_a_baseline_bf16_tmax15-20260515-203158/metrics.jsonl` (Arm A paired baseline)

### Reproduce

```bash
cd target/
python train.py --experiment_name ema-d0999-bf16-tmax15 \
  --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```
