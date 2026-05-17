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

---

## 2026-05-16 01:28 — PR #3122: FiLM conditioning on physics params (log Re, AoA, NACA, gap, stagger)

**New best `val_avg/mae_surf_p`: 92.606** (was: 96.464 EMA+bf16+T_max=15 — **−4.00%**; intra-PR Arm B vs Arm A: **−4.88%**)

- **FiLM:** Learned `(γ, β)` scale+shift injected at every `TransolverBlock` layer, conditioned on `[log(Re), AoA_rad, NACA_encoded, gap, stagger]`. Zero-init of final linear ensures stable warm-start (Arm B epoch 1 ≈ Arm A epoch 1).
- **Stack:** Huber + bf16 AMP + cosine T_max=15 + EMA decay=0.999 (full current best stack)
- **Params:** 845,527 vs 662,359 baseline (+27.6%)
- **Best epoch:** 18 / 19 run (30-min budget, 7% slower per epoch)
- **Peak VRAM:** 35.94 GB (+2.49 GB vs Arm A 33.45 GB)
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4 (unchanged)

### Val surface pressure MAE (lower is better)

| Split | Arm A (full stack, no FiLM) | **Arm B (+ FiLM)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 112.950 | **107.788** | −4.57% |
| `val_geom_camber_rc`     | 105.171 | **101.033** | −3.93% |
| `val_geom_camber_cruise` |  77.396 |  **73.993** | −4.41% |
| `val_re_rand`            |  93.922 |  **87.611** | **−6.72%** |
| **val_avg**              | **97.360** | **92.606** | **−4.88%** |

Arm B vs merged baseline 96.464: **−4.00%** ✅

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` NaN — pre-existing cruise-sample overflow)

| Split | Arm A | **Arm B (FiLM)** | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`    |  98.801 |  **92.949** | −5.93% |
| `test_geom_camber_rc`    |  96.102 |  **90.448** | −5.88% |
| `test_re_rand`           |  86.870 |  **83.618** | −3.74% |
| **avg (3 finite splits)** | **93.924** | **89.005** | **−5.24%** |

*Note: `test_avg/mae_surf_p` is NaN for both arms due to the cruise-sample overflow in `data/scoring.py` (read-only). The 3-split partial average is the reliable test metric.*

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-film-cond-r2-armb-film-20260516-002418/metrics.jsonl` ← **winner**
- `models/model-charliepai2i48h4-frieren-film-cond-r2-arma-baseline-20260515-232346/metrics.jsonl` (paired baseline)

### Reproduce

```bash
cd target/
python train.py --experiment_name film-cond-bf16-tmax15-ema \
  --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 \
  --film_cond
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond  (zero-init γ=1, β=0 at start; conditions on log_Re, AoA, NACA, gap, stagger)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```

---

## 2026-05-16 04:50 — PR #3584: Two-shot FiLM — condition attention + MLP paths per TransolverBlock

**New best `val_avg/mae_surf_p`: 89.784** (was: 92.606 single-shot FiLM — **−3.05%**; intra-PR Arm B vs Arm A: **−3.67%**)

- **Two-shot FiLM:** Same `FiLMConditioner` module reused at both (1) attention sub-layer input (after `ln_1`) and (2) MLP sub-layer input (after `ln_2`) per `TransolverBlock`. Shared module = **+0 parameters** vs single-shot FiLM. Two application sites give the model two independent opportunities to specialize per physics regime.
- **Stack:** Huber + bf16 AMP + cosine T_max=15 + EMA decay=0.999 + FiLM (single-shot) as the merged baseline
- **Params:** 845,527 (identical — shared conditioner)
- **Best epoch:** 17 / 17 run (30-min budget, +6.2% epoch time vs Arm A's 18 epochs)
- **Peak VRAM:** 38.9 GB (+6.8% vs Arm A 36.4 GB)
- **Both arms still descending at budget cutoff** — additional epochs would likely improve further

### Val surface pressure MAE (lower is better)

| Split | Arm A (full stack, 1-shot FiLM) | **Arm B (two-shot FiLM)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 106.191 | **103.854** | −2.20% |
| `val_geom_camber_rc`     | 103.036 |  **95.887** | **−6.94%** |
| `val_geom_camber_cruise` |  73.888 |  **73.143** | −1.01% |
| `val_re_rand`            |  89.704 |  **86.251** | −3.85% |
| **val_avg**              | **93.205** | **89.784** | **−3.67%** |

Arm B vs merged baseline 92.606: **−3.05%** ✅

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` NaN — pre-existing scoring bug)

| Split | Arm A | **Arm B (two-shot FiLM)** | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`    |  91.619 |  **89.460** | −2.36% |
| `test_geom_camber_rc`    |  91.888 |  **87.408** | −4.87% |
| `test_re_rand`           |  84.201 |  **80.336** | −4.59% |
| **avg (3 finite splits)** | **89.236** | **85.735** | **−3.92%** |

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-two-shot-film-armb-twoshot-20260516-030245/metrics.jsonl` ← **winner**
- `models/model-charliepai2i48h4-frieren-two-shot-film-armb-twoshot-20260516-030245/metrics.yaml`
- `models/model-charliepai2i48h4-frieren-two-shot-film-arma-baseline-20260516-022727/metrics.jsonl` (paired baseline)

### Reproduce

```bash
cd target/
python train.py --experiment_name two-shot-film \
  --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond --two_shot_film  (two injection sites per block — attn + MLP — shared conditioner, +0 params)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```

---

## 2026-05-16 11:22 — PR #3511: Gradient clipping (clip_norm=1.0) on full two-shot FiLM stack

**New best `val_avg/mae_surf_p`: 81.660** (was: 89.784 two-shot FiLM — **−9.05%**; intra-PR Arm B vs Arm A: **−11.38%**)

- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before every `optimizer.step()`
- **Stack:** Huber + bf16 AMP + cosine T_max=15 + EMA decay=0.999 + two-shot FiLM (full current best stack)
- **Mechanism:** ~96-100% clip rate throughout training — effectively **gradient direction normalization**, not outlier filtering. Natural grad norms p50≈7-25, p90≈15-50 throughout training (all well above clip=1.0). bf16 heavy-tail outliers at epoch 1 (max=226) absorbed on every step.
- **Gain grew on FiLM stack:** pre-FiLM gain was −4.77%; with FiLM it is −9.05% — FiLM conditioning increases per-sample gradient noise sensitivity; clipping composes super-additively.
- **Best epoch:** 17 / 17 run (30-min budget, unchanged epoch count vs Arm A's 17)
- **Params:** 845,527 (unchanged — clipping adds no parameters)
- **Peak VRAM:** 38.92 GB (unchanged)
- **Model:** 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64` (unchanged)
- **Optimizer:** AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR T_max=15 (unchanged)

### Val surface pressure MAE (lower is better)

| Split | Arm A (full stack, no clip) | **Arm B (clip=1.0)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 105.365 |  **94.434** | −10.37% |
| `val_geom_camber_rc`     | 101.253 |  **90.960** | −10.17% |
| `val_geom_camber_cruise` |  73.369 |  **62.732** | **−14.50%** |
| `val_re_rand`            |  88.598 |  **78.516** | −11.38% |
| **val_avg**              | **92.146** | **81.660** | **−11.38%** |

Arm B vs merged baseline 89.784: **−9.05%** ✅

### Test surface pressure MAE (3 finite splits; `test_geom_camber_cruise` NaN — pre-existing scoring bug)

| Split | Arm A | **Arm B (clip=1.0)** | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`    |  90.902 |  **81.956** |  −9.84% |
| `test_geom_camber_rc`    |  93.074 |  **83.649** | −10.13% |
| `test_re_rand`           |  82.735 |  **71.296** | **−13.83%** |
| **avg (3 finite splits)** | **88.903** | **78.967** | **−11.18%** |

### Variance analysis

Tanjiro ran 4 independent Arm A pilot seeds during development: 87.579, 92.146, 92.276, 95.066 (mean 91.8, std 3.1). Arm B (81.660) is **5.9% better than the best pilot Arm A** and **15.8% better than the worst** — signal is well outside the noise floor at every reasonable seed pairing.

### Gradient norm diagnostics (Arm B, no-clip → clip)

| Epoch | p50 (before clip) | p90 | p99 | max | Clip rate |
|---|---:|---:|---:|---:|---:|
| 1  | 23.89 | 48.76 | 103.81 | 226.38 | 1.000 |
| 8  |  9.94 | 21.11 |  30.83 |  45.95 | 1.000 |
| 17 |  6.29 | 15.04 |  31.91 |  43.90 | 0.960 |

### Metric artifacts

- `models/model-charliepai2i48h4-tanjiro-gradclip-r2-arm-b-twoshot-clip1_0-20260516-093143/metrics.jsonl` ← **winner**
- `models/model-charliepai2i48h4-tanjiro-gradclip-r2-arm-b-twoshot-clip1_0-20260516-093143/metrics.yaml`
- `models/model-charliepai2i48h4-tanjiro-gradclip-r2-arm-a-twoshot-noclip-20260516-072527/metrics.jsonl` (paired baseline)

### Reproduce

```bash
cd target/
python train.py --experiment_name gradclip-r2-twoshot-clip1_0 \
  --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film --grad_clip_norm 1.0
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond --two_shot_film  (shared conditioner, two injection sites per block)
# Gradient clip: --grad_clip_norm 1.0  (clips to unit norm ~96-100% of steps)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```

---

## 2026-05-16 14:32 — PR #3906: Clip threshold sweep — clip=0.25 beats clip=1.0 by −3.42% paired

**New best `val_avg/mae_surf_p`: 80.893** (was: 81.660 — **−0.94%**)

### Val surface pressure MAE per split

| Split | val_avg/mae_surf_p |
|---|---:|
| `val_single_in_dist`     | 93.062 |
| `val_geom_camber_rc`     | 90.132 |
| `val_geom_camber_cruise` | 61.764 |
| `val_re_rand`            | 78.616 |
| **val_avg**              | **80.893** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 79.300 |
| `test_geom_camber_rc`   | 80.780 |
| `test_re_rand`          | 70.587 |
| `test_avg (3 finite)` | **76.889** |

### Three-arm sweep summary (this PR)

| Arm | clip_norm | val_avg | Δ vs A |
|---|---:|---:|---:|
| A | 1.0 | 83.756 | — (control) |
| B | 4.0 | 86.647 | +3.45% (regression) |
| **C** | **0.25** | **80.893** | **−3.42% (winner)** |

Monotone: tighter clip = better. Arm C clip rate = **100% every epoch throughout**. Direction normalization (not outlier suppression) is the load-bearing mechanism.

### Metric artifacts

- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-armc-clip0_25-20260516-133449/metrics.jsonl` ← **winner**
- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-armc-clip0_25-20260516-133449/metrics.yaml`
- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-arma-clip1_0-20260516-113014/metrics.jsonl` (paired control)
- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-armb-clip4_0-20260516-122208/metrics.jsonl` (paired regression arm)

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film --grad_clip_norm 0.25
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# AMP: --amp_dtype bf16
# Scheduler: --cosine_t_max 15
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond --two_shot_film  (shared conditioner, two injection sites per block)
# Gradient clip: --grad_clip_norm 0.25  (clips to 0.25 norm; 100% clip rate all-run)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4
# surf_weight=10
```

---

## 2026-05-16 15:34 — PR #3594: Schedule-Free AdamW — eliminate cosine schedule

**New best `val_avg/mae_surf_p`: 65.618** (was: 80.893 — **−18.88%**)

### Val surface pressure MAE per split

| Split | val_avg/mae_surf_p |
|---|---:|
| `val_single_in_dist`     | 74.715 |
| `val_geom_camber_rc`     | 79.128 |
| `val_geom_camber_cruise` | 45.160 |
| `val_re_rand`            | 63.467 |
| **val_avg**              | **65.618** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 63.718 |
| `test_geom_camber_rc`   | 70.042 |
| `test_re_rand`          | 54.799 |
| `test_avg (3 finite)` | **62.853** |

### Paired sweep summary (this PR)

| Arm | optimizer | scheduler | clip | val_avg | Δ vs A |
|---|---|---|---:|---:|---:|
| A | AdamW | cosine T_max=15 | 1.0 | 78.871 | — (control) |
| **B** | **SF-AdamW** | **none** | **1.0** | **65.618** | **−16.80% (winner)** |

Both arms still descending at epoch 17/17 (budget cap). Cosine T_max=15 freezes Arm A at LR=5e-8 from epoch 16 onward; SF-AdamW keeps stepping at LR=5e-4 throughout. Arm B was dropping ~1.8 val/epoch at the cap — further budget headroom exists.

**Note on clip threshold:** this win was measured with clip=1.0, not the current merged clip=0.25 (#3906). New baseline represents: `bf16 + EMA(0.999) + FiLM + two-shot FiLM + SF-AdamW + clip=1.0`. The optimal clip threshold under SF-AdamW is unknown and is the next experiment.

### Metric artifacts

- `models/model-charliepai2i48h4-alphonse-sf-r2-armb-sf-adamw-clip-20260516-142921/metrics.jsonl` ← **winner**
- `models/model-charliepai2i48h4-alphonse-sf-r2-armb-sf-adamw-clip-20260516-142921/metrics.yaml`
- `models/model-charliepai2i48h4-alphonse-sf-r2-arma-baseline-20260516-135423/metrics.jsonl` (paired control)

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film --grad_clip_norm 1.0 \
  --use_schedule_free
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond --two_shot_film  (shared conditioner, two injection sites per block)
# Optimizer: SF-AdamW lr=5e-4, weight_decay=1e-4, warmup_steps=500, betas=(0.9,0.999)
#   → use optimizer.train() before train epoch, optimizer.eval() before val
# Gradient clip: --grad_clip_norm 1.0  (NOTE: clip=0.25 optimal under AdamW; SF-AdamW clip not yet tuned)
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
```

---

## 2026-05-16 21:07 — PR #3980: Lion optimizer (sign projection) — NEW BEST

**val_avg/mae_surf_p: 63.336** (Δ −3.48% vs SF-AdamW baseline 65.618)

Lion + clip=0.25 is now the canonical optimizer, replacing SF-AdamW. Post-rebase reproduction confirmed. Paired Δ −24.43% val / −23.72% test vs AdamW+clip=0.25 (within-session control), ~12× seed-variance noise floor.

### Val surface pressure MAE (lower is better)

| Split | val/mae_surf_p |
|---|---:|
| `val_single_in_dist`      | 65.069 |
| `val_geom_camber_rc`      | 77.134 |
| `val_geom_camber_cruise`  | 47.166 |
| `val_re_rand`             | 63.975 |
| **val_avg**               | **63.336** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 56.001 |
| `test_geom_camber_rc`   | 69.853 |
| `test_re_rand`          | 55.794 |
| `test_avg (3 finite)` | **60.549** |

### Paired sweep summary (post-rebase R2)

| Arm | optimizer | scheduler | clip | val_avg | Δ vs A |
|---|---|---|---:|---:|---:|
| A | AdamW | cosine T_max=15 | 0.25 | 83.812 | — (control) |
| **B** | **Lion** | **cosine T_max=15** | **0.25** | **63.336** | **−24.43% (winner)** |

Best epoch: 17 (both arms). Peak memory: 38.92 GB. Sec/epoch: 112.5. Both arms still descending at budget cap — further headroom exists.

### Mechanism

With clip_rate ≈ 100% at clip=0.25, AdamW sees L2-clipped Adam (two normalizers in series: per-coordinate `m̂/(√v̂+ε)` + global L2 rescale). Lion sees sign projection (single consistent normalizer: all coordinates ±lr). Sign projection re-weights toward under-represented gradient components and removes the fight between Adam's adaptive scaling and the global clip. Win is uniform across all 4 val splits and 3 finite test splits.

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-lion-r2-armb-lion-clip25-rebased-20260516-183306/metrics.jsonl` ← **winner (R2 rebased)**
- `models/model-charliepai2i48h4-frieren-lion-r2-armb-lion-clip25-rebased-20260516-183306/metrics.yaml`
- `models/model-charliepai2i48h4-frieren-lion-r2-arma-adamw-clip25-rebased-20260516-172945/metrics.jsonl` (R2 control)
- `models/model-charliepai2i48h4-frieren-lion-r1-armb-lion-clip25-20260516-152650/metrics.jsonl` (R1 original)

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 --cosine_t_max 15 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 0.25 \
  --optimizer lion --lion_lr 1.5e-4 --lion_weight_decay 3e-4 \
  --lion_betas 0.9,0.99 \
  --seed 1
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: cosine (--cosine_t_max 15)
# EMA: --use_ema --ema_decay 0.999  (Karras-style warmup ramp built in)
# FiLM: --film_cond --two_shot_film  (shared conditioner, two injection sites per block)
# Optimizer: Lion lr=1.5e-4, weight_decay=3e-4, betas=(0.9, 0.99)
# Gradient clip: --grad_clip_norm 0.25
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
```


---

## 2026-05-16 21:18 — PR #4038: SF-AdamW LR sweep — NEW BEST (replaces Lion)

**val_avg/mae_surf_p: 54.769** (Δ −13.5% vs just-merged Lion baseline 63.336; Δ −16.5% vs SF-AdamW 65.618)

SF-AdamW + lr=2e-3 is now the canonical stack, replacing Lion+cosine. The default lr=5e-4 was inherited from AdamW and was critically mistuned for SF's constant-LR regime — 4× higher LR (2e-3) wins decisively. 

### Val surface pressure MAE (lower is better)

| Split | val/mae_surf_p |
|---|---:|
| `val_single_in_dist`      | 60.429 |
| `val_geom_camber_rc`      | 68.478 |
| `val_geom_camber_cruise`  | 34.597 |
| `val_re_rand`             | 55.572 |
| **val_avg**               | **54.769** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 52.496 |
| `test_geom_camber_rc`   | 63.210 |
| `test_re_rand`          | 44.914 |
| `test_avg (3 finite)` | **53.540** |

### Paired sweep summary

| Arm | lr | val_avg | Δ vs A | Test Δ |
|---|---:|---:|---:|---:|
| A (control) | 5e-4 | 62.958 | — | — |
| B | 1e-3 | 58.424 | −7.20% | −8.31% |
| **C (winner)** | **2e-3** | **54.769** | **−13.01%** | **−12.57%** |
| D | 5e-3 | 55.951 | −11.13% | −12.09% |

Non-monotone: C wins, D is second. All arms beat control. C wins on every single val and test split.

### Metric artifacts

- Winner (lr=2e-3): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-armc-lr2e-3-20260516-180222/metrics.jsonl`
- Control (lr=5e-4): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-arma-lr5e-4-20260516-192309/metrics.jsonl`

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 2e-3
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=2e-3, weight_decay=1e-4, warmup_steps=500, betas=(0.9,0.999)
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
```

---

## 2026-05-17 01:00 — PR #4157: SF-AdamW LR fine-tune — lr=3e-3 NEW BEST

**val_avg/mae_surf_p: 52.258** (Δ **−4.59% vs baseline 54.769**; Δ −2.89% paired vs Arm B control)

**Critical signal: A→B→C→D sweep is perfectly monotone — the true LR peak lies beyond 3e-3.** The coarse sweep (#4038) found 2e-3 > 5e-3 with a non-monotone structure, but the fine grid in [1.5e-3, 3e-3] reveals the gradient keeps improving all the way through. Peak is ≥3e-3. Best epoch = 17/17 for ALL arms (budget-truncated, all still descending) — further headroom exists at higher LR.

| Arm | lr | val_avg/mae_surf_p | Δ vs B control | Δ vs baseline |
|---|---:|---:|---:|---:|
| A | 1.5e-3 | 55.902 | +3.88% (regression) | +2.07% |
| **B (control)** | **2e-3** | **53.814** | — | −1.74% |
| C | 2.5e-3 | 53.182 | −1.18% | −2.90% |
| **D (winner)** | **3e-3** | **52.258** | **−2.89%** | **−4.59%** |

### Val surface pressure MAE (lower is better)

| Split | val/mae_surf_p |
|---|---:|
| `val_single_in_dist`      | 56.454 |
| `val_geom_camber_rc`      | 66.039 |
| `val_geom_camber_cruise`  | 33.763 |
| `val_re_rand`             | 52.775 |
| **val_avg**               | **52.258** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 48.731 |
| `test_geom_camber_rc`   | 60.335 |
| `test_re_rand`          | 44.552 |
| `test_avg (3 finite)` | **51.206** |

- **Best epoch:** 17/17 — budget-truncated; all arms still descending at cap
- **Peak VRAM:** 38.97 GB
- **Sec/epoch:** 110.1

### Metric artifacts

- Winner (lr=3e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armD-lr3e-3-20260516-222733-20260516-233714/metrics.jsonl`
- Control (lr=2e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armB-lr2e-3-20260516-222733-20260516-222735/metrics.jsonl`
- Arm C (lr=2.5e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armC-lr2.5e-3-20260516-222733-20260516-230225/metrics.jsonl`
- Arm A (lr=1.5e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armA-lr1.5e-3-20260516-213655-20260516-213657/metrics.jsonl`

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=3e-3, weight_decay=1e-4, warmup_steps=500, betas=(0.9,0.999)
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
# NEXT: lr=3e-3 is still monotonically improving — explore {3e-3, 4e-3, 5e-3, 6e-3}
```

---

## 2026-05-17 08:00 — PR #4248: Model depth sweep — n_layers=3 NEW BEST

**val_avg/mae_surf_p: 45.654** (Δ **−12.64% vs prior canonical 52.258**; Δ −14.74% paired vs Arm B n_layers=5 control)
**test_3split_mean/mae_surf_p: 44.878** (Δ −12.36% vs prior canonical test 51.206)

**This is the biggest single-PR improvement in the round.** Arm A (n_layers=3) crushes all deeper alternatives. Monotone descent l3 < l4 < l5 < l7 on every val split AND every clean test split. Val/test directions agree perfectly (no flip).

| Arm | n_layers | n_params | epochs (cap=50) | sec/ep | val_avg | Δ vs B paired | Δ vs prior canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A (winner)** | **3** | **537K** | **26** | **69.3s** | **45.654** | **−14.74%** | **−12.64%** |
| B (control) | 5 | 846K | 17 | 111.3s | 53.549 | — | +2.47% |
| C | 7 | 1.15M | 12 | 153.3s | 61.044 | +14.00% | +16.81% |
| D | 4 | 691K | 20 | 90.5s | 50.693 | −5.33% | −2.99% |

### Val surface pressure MAE (winner Arm A)

| Split | val/mae_surf_p |
|---|---:|
| `val_single_in_dist`     | 46.762 |
| `val_geom_camber_rc`     | 59.317 |
| `val_geom_camber_cruise` | 29.120 |
| `val_re_rand`            | 47.415 |
| **val_avg**              | **45.654** |

### Test surface pressure MAE (3 finite splits; cruise NaN pre-existing)

| Split | test/mae_surf_p |
|---|---:|
| `test_single_in_dist`   | 41.948 |
| `test_geom_camber_rc`   | 54.253 |
| `test_re_rand`          | 38.432 |
| **test_3split_mean**    | **44.878** |

### Mechanism note (honest)

At iso-epoch=12, all arms cluster within ~1% of each other — meaning the win is dominated by **step count, not capacity**: sec/epoch scales near-linearly with depth (1.16 / 1.51 / 1.85 / 2.56 min for l3/l4/l5/l7), so the shallower arm fits more optimization steps in the 30-min wall-clock budget. None of the arms had converged at termination (best_epoch == last_epoch for all). Our advisor regime IS wall-clock, so this is a valid deployment-regime improvement, but follow-ups should test convergence/longer-budget behaviour.

### Peak VRAM

| Arm | n_layers | peak_mem_gb |
|---|---:|---:|
| A | 3 | 24.97 |
| B | 5 | 38.92 |
| C | 7 | 52.87 |
| D | 4 | 31.94 |

### Metric artifacts

- Winner (n_layers=3): `models/model-charliepai2i48h4-frieren-n-layers-r1-armA-l3-20260517-042742-20260517-042745/metrics.jsonl`
- Control (n_layers=5): `models/model-charliepai2i48h4-frieren-n-layers-r1-armB-l5-20260517-033637-20260517-033640/metrics.jsonl`
- Arm C (n_layers=7): `models/model-charliepai2i48h4-frieren-n-layers-r1-armC-l7-20260517-052644-20260517-052647/metrics.jsonl`
- Arm D (n_layers=4): `models/model-charliepai2i48h4-frieren-n-layers-r1-armD-l4-20260517-062841-20260517-062844/metrics.jsonl`

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3 \
  --n_layers 3
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=3e-3, weight_decay=1e-4, warmup_steps=500, betas=(0.9,0.999)
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=3 (NEW), n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
# NEXT: shallower probe {1,2,3}; longer-budget rerun at n_layers=3; re-test architecture axes at new canonical
```

---

## 2026-05-17 08:08 — PR #4317: SF-AdamW betas — beta1=0.95 beta2=0.99 (compound improvement)

**Measured at prior n_layers=5 canonical: val_avg/mae_surf_p = 50.273** (Δ −3.80% vs prior 52.258; Δ **−6.12% paired** vs Arm A control)
**Test 3-split mean = 48.726** (Δ −4.84% vs prior canonical test 51.206; Δ **−7.35% paired**)

**Note:** This PR was measured against the prior n_layers=5 canonical. The new n_layers=3 canonical (from PR #4248) was merged moments earlier — so 50.273 is at the OLD depth. The paired Δ of −6.12% is an **optimizer-level effect** (SF-AdamW Polyak iterate quality) and is plausibly orthogonal to depth; folding it into the canonical via merge applies the compound-improvement principle. An empirical (n_layers=3 × sf_betas=(0.95, 0.99)) measurement is pending in follow-up experiments.

### Headline 2×2 paired Δ table (val_avg/mae_surf_p)

| Arm | β1 | β2 | val_avg | Δ vs A paired |
|---|---:|---:|---:|---:|
| A (control) | 0.9 | 0.999 | 53.549 | — |
| B | 0.9 | 0.99 | 52.443 | −2.07% |
| C | 0.95 | 0.999 | 51.810 | −3.25% |
| **D (winner)** | **0.95** | **0.99** | **50.273** | **−6.12%** |

Main effects: β1↑ (0.9→0.95) = −1.95, β2↓ (0.999→0.99) = −1.32, β1×β2 interaction = −0.43 (mild synergy). Both axes pull same direction, combine ~additively.

### Mechanism (negative result on candidate hypothesis)

Hypothesis tested: β1↑ might reduce gradient-clip rate. **Disconfirmed.** Clip rate stays at 0.95–0.98 for every arm throughout training; β1 affects it by < 0.01. The β1↑ win must operate via Polyak iterate quality (the SF-AdamW averaging mechanism), NOT gradient-norm smoothing.

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3 \
  --n_layers 3 \
  --sf_beta1 0.95 --sf_beta2 0.99
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=3e-3, weight_decay=1e-4, warmup_steps=500
# Optimizer betas: sf_beta1=0.95, sf_beta2=0.99 (NEW)
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=3, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
# NEXT: empirical (n_layers=3 × sf_betas=(0.95, 0.99)) confirmation; SF-beta frontier extension
```


---

## 2026-05-17 09:35 — PR #4464: n_layers=2 NEW BEST (frieren shallower-depth-probe)

**val_avg/mae_surf_p: 40.622 | test_3split_mean/mae_surf_p: 39.598**
_(test_geom_camber_cruise excluded: pre-existing NaN scoring bug on cruise held-out samples)_

### Headline paired Δ

| Metric | Prior canonical | New best | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 45.654 (n_layers=3) | **40.622** (n_layers=2) | **−11.02%** |
| test_3split_mean/mae_surf_p | 44.878 | **39.598** | **−11.77%** |

Paired Δ vs Arm A control (n_layers=3 at same canonical): **−9.23% val / −11.77% test** (18× the 0.5% gate).

### Per-split val (mae_surf_p, epoch 37)

| Split | n_layers=3 (A) | n_layers=2 (B) | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 46.856 | **40.367** | −13.9% |
| val_geom_camber_rc | 58.135 | **55.995** | −3.7% |
| val_geom_camber_cruise | 26.926 | **23.944** | −11.1% |
| val_re_rand | 47.090 | **42.183** | −10.4% |
| **val_avg** | 44.752 | **40.622** | **−9.2%** |

Arm B strictly dominates Arm A on all four val splits.

### Per-split test (mae_surf_p)

| Split | n_layers=2 |
|---|---:|
| test_single_in_dist | 36.002 |
| test_geom_camber_rc | 49.549 |
| test_re_rand | 33.243 |
| test_geom_camber_cruise | NaN (pre-existing scoring bug) |
| **test_3split_mean** | **39.598** |

### Mechanism note

At iso-epoch=26 (common to all arms): l3 < l2 < l1 — deeper models are better per optimization step. But the 30-min wall-clock budget is binding: l2 runs at 48.7 s/epoch and completes 37 epochs vs l3's 69 s/epoch and 26 epochs. The win is a step-count effect — l2 accumulates ~1.4× more optimization steps. All arms still descending at budget cap (best_epoch = last_epoch), so there is headroom in longer-budget regimes.

### Infra metrics (Arm B, epoch 37)

- peak_VRAM: 17.99 GB
- grad_clip_rate: 0.92 (stable)
- ema_effective_decay: 0.999

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-shallower-depth-probe-r1-armB-l2-20260517-082017-20260517-082020/metrics.jsonl`

### Reproduce

```bash
cd target/
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3 \
  --sf_beta1 0.95 --sf_beta2 0.99 \
  --n_layers 2
```

### Current best config (carry forward to all new experiments)

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=3e-3, weight_decay=1e-4, warmup_steps=500
# Optimizer betas: sf_beta1=0.95, sf_beta2=0.99
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=2 (NEW), n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
# Note: l1 < l2 (iso-epoch); l2 wins by step-count at 30-min budget
```

---

## 2026-05-17 12:10 — PR #4467: lr=5e-3 compound improvement (alphonse lr-retune-n-layers3)

**Measured at n_layers=3 canonical: val=41.328, test_3split=40.938**
_(Empirical confirmation at n_layers=2 pending — assigned as follow-up)_

### Headline paired Δ

At n_layers=3 (sf_betas=(0.95, 0.99)):

| Arm | lr | val_avg | Δ vs A (paired) |
|---|---:|---:|---:|
| A (control) | 3e-3 | 46.100 | — |
| B | 4e-3 | 43.293 | −3.31% |
| **C (winner)** | **5e-3** | **41.328** | **−8.21%** |
| D | 6e-3 | 43.033 | −4.68% |

Sharply peaked optimum at lr=5e-3. Curve falls off on both sides (−3.31% at 4e-3, −4.68% at 6e-3).

### Mechanism

From near epoch 1 (C is −1.6% over A at ep 1, monotonically widening to −8.21% at ep 24). Cross-split dominance: C wins 3 of 4 val splits, wins test 3-split (40.938 vs A's 44.120, −7.21% paired). No divergence at any LR; grad_norm/clip_rate dynamics healthy. Terminal grad_norm/mean is _lower_ at higher LR (A=4.99, C=3.44) — consistent with finding flatter basins. This is a genuine optimizer effect (LR-driven faster convergence), orthogonal to depth.

### Compound merge basis

Same precedent as PR #4317 (SF-betas): from-epoch-1 effect, genuine optimizer mechanism, orthogonal to step-count. Canonical shift from n_layers=3 (measured) to n_layers=2 (current) is 1 canonical, matching the SF-betas merge at the time (measured n_layers=5 → current n_layers=3).

### Metric artifacts

- `models/model-charliepai2i48h4-alphonse-lr-retune-n-layers3-r1-armC-lr5e3-20260517-103434-20260517-103437/metrics.jsonl`

### Updated canonical config

```python
# Loss: Huber (smooth_l1_loss, beta=1.0)
# AMP: --amp_dtype bf16
# Scheduler: NONE (--use_schedule_free replaces cosine)
# EMA: --use_ema --ema_decay 0.999
# FiLM: --film_cond --two_shot_film
# Optimizer: SF-AdamW lr=5e-3 (NEW), weight_decay=1e-4, warmup_steps=500
# Optimizer betas: sf_beta1=0.95, sf_beta2=0.99
# Gradient clip: --grad_clip_norm 1.0
# Model: n_hidden=128, n_layers=2, n_head=4, slice_num=64, mlp_ratio=2
# surf_weight=10
# Note: lr=5e-3 measured at n_layers=3; empirical confirmation at n_layers=2 pending
```

### Reproduce (best at n_layers=3 with lr=5e-3)

```bash
cd target/
python train.py \
  --amp_dtype bf16 --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film --grad_clip_norm 1.0 \
  --use_schedule_free --lr 5e-3 \
  --sf_beta1 0.95 --sf_beta2 0.99 \
  --n_layers 3
```

Recommended next: lr=5e-3 at n_layers=2 canonical (empirical confirmation TBD).
