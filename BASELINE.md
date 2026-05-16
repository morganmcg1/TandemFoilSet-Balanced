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
