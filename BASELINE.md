# Baseline Metrics

## Current Baseline — PR #2011 (film-re-attention)

**val_avg/mae_surf_p = 28.8762** (epoch 28 of 28; 30-min cap) — **-1.17% vs previous 29.2179**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- **+ ReScaleHead** (3-channel, 163 params): learned Re→scale MLP applied to Transolver output
- **+ per-channel loss weights**: `ch_weights=[1.0, 1.0, 5.0]` applied as linear multiplier post-Huber on per-element Huber output (p_channel_weight=5.0)
- **+ ReFiLM**: FiLM Re-conditioning inside PhysicsAttention slice logits (+4,624 params, ~0.7% overhead). Shared module across all 5 blocks/4 heads; zero-init (γ=0, β=0); gate opens to |γ|max=0.70, |β|max=0.62 by epoch 28.
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**
- **`torch.compile(mode="default", dynamic=True)`**
- **bf16 AMP**
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1)+rel-L2 with p post-Huber weight=5 in numerator; denominator unweighted
- 28 epochs in ~30 min, peak GPU 27.79 GB (+3.9 GB vs prior baseline from FiLM intermediates)
- Cumulative: **-75.3%** vs initial 117.17

**Per-split val at best epoch (28):**

| Split | mae_surf_p | vs baseline (#1614) |
|-------|-----------|---------|
| val_single_in_dist | **28.6013** | +0.039 (+0.14%) |
| val_geom_camber_rc | **41.9483** | −0.741 (−1.73%) |
| val_geom_camber_cruise | **14.1462** | +0.375 (+2.72%) |
| val_re_rand | **30.8090** | −1.041 (−3.27%) |
| **val_avg** | **28.8762** | **−0.342 (−1.17%)** |

**Per-split test at best epoch (28):**

| Split | mae_surf_p | vs baseline (#1614) |
|-------|-----------|---------|
| test_single_in_dist | **29.5300** | −0.605 (−2.01%) |
| test_geom_camber_rc | **37.0266** | −1.913 (−4.91%) |
| test_geom_camber_cruise | **11.0171** | +0.170 (+1.57%) |
| test_re_rand | **22.4230** | −0.065 (−0.29%) |
| **test_avg** | **24.9992** | **−0.603 (−2.36%)** |

**Mechanism**: Re-conditioning of PhysicsAttention slice-logits via FiLM (γ(Re), β(Re)). With zero-init gates, the module trains from identity and opens monotonically — mean slice entropy drops 33% (4.153→2.759), confirming the model genuinely uses different slice subsets per Re value. Gains concentrate on Re-variable splits (re_rand −1.04 val) and hard OOD geometry (geom_camber_rc test −1.91). Flat/mildly negative on fixed-Re splits (single_in_dist, geom_camber_cruise).

**Artifact**: `models/model-charliepai2g24h1-fern-film-re-attention-20260513-072042/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 SENPAI_MAX_EPOCHS=50 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# Stack: SOAP + bf16 + torch.compile(default, dynamic) + CosineAnnealingLR(T_max=28, eta_min=1e-5)
# + ReScaleHead(hidden=32, out_channels=3) + p_channel_weight=5 (post-Huber)
# + ReFiLM(Re) on slice logits (shared, zero-init)
```

**Key insight**: ReFiLM adds a 4,624-param Re-conditioned FiLM gate to slice selection inside PhysicsAttention. The gate opens monotonically during training (zero-init ensures stable early dynamics), enabling Re-dependent attention slice specialisation. Orthogonal to ReScaleHead (output rescaling) and p_channel_weight (loss reweighting). Best==last epoch (28) — schedule still potentially binding.

---

## Previous Baseline — PR #1614 (per-channel-loss-weights)

**val_avg/mae_surf_p = 29.2179** (epoch 29 of 29; 30-min cap) — **-2.11% vs previous 29.8463**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- **+ ReScaleHead** (3-channel, 163 params): learned Re→scale MLP applied to Transolver output
- **+ per-channel loss weights**: `ch_weights=[1.0, 1.0, 5.0]` applied as linear multiplier **post-Huber** on per-element Huber output (p_channel_weight=5.0)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**
- **`torch.compile(mode="default", dynamic=True)`**
- **bf16 AMP**
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1)+rel-L2 with p post-Huber weight=5 in numerator; denominator unweighted
- 29 epochs in ~30 min, peak GPU 23.91 GB
- Cumulative: **-75.1%** vs initial 117.17

**Per-split val at best epoch (29):**

| Split | mae_surf_p | vs prev (#1599) |
|-------|-----------|---------|
| val_single_in_dist | **28.5620** | −1.638 (−5.43%) |
| val_geom_camber_rc | **42.6891** | −0.421 (−0.97%) |
| val_geom_camber_cruise | **13.7711** | −0.769 (−5.29%) |
| val_re_rand | **31.8496** | +0.310 (+0.99%) |
| **val_avg** | **29.2179** | **−0.6284 (−2.11%)** |

**Per-split test at best epoch (29):**

| Split | mae_surf_p | vs prev (#1599) |
|-------|-----------|---------|
| test_single_in_dist | **30.1346** | +0.045 (+0.15%) |
| test_geom_camber_rc | **38.9393** | −0.471 (−1.19%) |
| test_geom_camber_cruise | **10.8473** | −0.893 (−7.60%) |
| test_re_rand | **22.4885** | −0.672 (−2.90%) |
| **test_avg** | **25.6024** | **−0.4981 (−1.91%)** |

**Per-channel trade-off (val avg @ best epoch):**

| Channel | with p_weight=5 | baseline (#1599) | Δ |
|---------|----------------|-----------------|---|
| mae_surf_p | **29.218** | 29.846 | −2.1% |
| mae_surf_Ux | 0.440 | 0.376 | +17.1% |
| mae_surf_Uy | 0.248 | 0.224 | +10.7% |

**Artifact**: `models/model-charliepai2g24h1-edward-per-channel-loss-weights-p5-20260513-052434/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# Stack: SOAP + bf16 + torch.compile(default, dynamic) + CosineAnnealingLR(T_max=28, eta_min=1e-5)
# + ReScaleHead(hidden=32, out_channels=3) + p_channel_weight=5 (post-Huber)
```

**Key insight**: Upweighting the pressure channel by 5× post-Huber shifts ~7× more gradient mass to p vs each velocity channel. Numerator-only weighting preserves the denominator's cross-sample scaling, giving exactly 5× amplification across all Huber regimes (no variable amplification as l2_frac grows). Stable training, 3/4 val splits and all 4 test splits improve. Model still improving at ep 29 (last epoch) — convergence-limited.

---

## Previous Baseline — PR #1599 (re-conditioned-scaling)

**val_avg/mae_surf_p = 29.8463** (epoch 27 of 29 completed in 30-min cap) — **-1.95% vs previous 30.4412**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- **+ ReScaleHead**: 163-param learned Re→scale MLP (hidden=32, out_channels=3), softplus activation, identity init (softplus(0.541)≈1.0)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`) — params include model + rescale_head
- **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**
- **`torch.compile(mode="default", dynamic=True)`** (Transolver only; ReScaleHead uncompiled)
- **bf16 AMP** (wraps both Transolver + ReScaleHead forward)
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- 29 epochs in ~30 min, peak GPU 24 GB
- Cumulative: **-74.5%** vs initial 117.17

**Per-split val at best epoch (27):**

| Split | mae_surf_p | vs prev (30.4412) |
|-------|-----------|---------|
| val_single_in_dist | **30.20** | −4.07 (−11.9%) |
| val_geom_camber_rc | **43.11** | +1.68 (+4.1%) |
| val_geom_camber_cruise | **14.54** | +0.50 (+3.6%) |
| val_re_rand | **31.54** | −0.48 (−1.5%) |
| **val_avg** | **29.8463** | **−0.59 (−1.95%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev (26.1013) |
|-------|-----------|---------|
| test_single_in_dist | **30.09** | −2.87 (−8.7%) |
| test_geom_camber_rc | **39.41** | +1.51 (+4.0%) |
| test_geom_camber_cruise | **11.74** | +0.36 (+3.2%) |
| test_re_rand | **23.16** | +1.00 (+4.5%) |
| **test_avg** | **26.1005** | **−0.0008 (≈0%)** |

**ReScaleHead diagnostics (best epoch 27):**

| Channel | scale mean | scale std | corr(scale, log Re) |
|---------|-----------|----------|---------------------|
| Ux | 1.180 | 0.058 | +0.637 |
| Uy | 1.111 | 0.262 | +0.936 |
| p | 1.308 | 0.527 | +0.858 |

**Artifact**: `models/model-charliepai2g24h1-fern-re-conditioned-scaling-20260513-035742/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# Stack: SOAP + bf16 AMP + torch.compile(default, dynamic) + CosineAnnealingLR(T_max=28, eta_min=1e-5)
# + ReScaleHead(hidden=32, out_channels=3) applied to Transolver output with log(Re) conditioning
```

**Key insight**: ReScaleHead reliably improves `single_in_dist` (−4.07 val) by separating shape learning from Re-scale calibration, but mildly regresses OOD-shape splits (rc, cruise). Val average still wins by 0.59 points because in-dist gain dominates. Mechanism confirmed in 3 of 3 runs: Uy/p show strong corr(scale, log Re) ~0.86–0.94; Ux is unused (freestream-dominated). Compound size has shrunk vs prior base because the stronger SOAP+torch.compile backbone already implicitly learns Re-scale.

---

## Previous Baseline — PR #1794 (torch-compile)

**val_avg/mae_surf_p = 30.4412** (epoch 30 of 30 completed in 30-min cap) — **-17.5% vs previous 36.8778**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`** ← updated for 30-epoch budget
- **`torch.compile(mode="default", dynamic=True)`** ← key addition (+76% throughput)
- **bf16 AMP enabled**
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- **30 epochs in ~30 min** (vs 17 previously, +76% throughput from torch.compile)
- Peak GPU 24 GB (down from 33 GB — compile reduces memory fragmentation)
- `dynamic=True` required because `pad_collate` produces variable-shape tensors (reduce-overhead would trigger recompilation storms)

**Per-split val at best epoch (30):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| val_single_in_dist | **34.27** | −8.65 (−20.2%) |
| val_geom_camber_rc | **41.43** | −6.35 (−13.3%) |
| val_geom_camber_cruise | **14.04** | −4.56 (−24.5%) |
| val_re_rand | **32.02** | −6.19 (−16.2%) |
| **val_avg** | **30.4412** | **−6.44 (−17.5%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| test_single_in_dist | **32.96** | −9.19 (−21.8%) |
| test_geom_camber_rc | **37.90** | −4.79 (−11.2%) |
| test_geom_camber_cruise | **11.38** | −3.88 (−25.4%) |
| test_re_rand | **22.16** | −5.37 (−19.5%) |
| **test_avg** | **26.1013** | **−5.80 (−18.2%)** |

**Artifact**: `models/model-charliepai2g24h1-alphonse-torch-compile-20260513-021531/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP + bf16 + torch.compile(mode="default", dynamic=True) + CosineAnnealingLR(T_max=28, eta_min=1e-5) now defaults
```

**Key insight**: `torch.compile` delivers +76% throughput (+47% more epochs per 30-min run), enabling 30 epochs vs 17. All 8 splits improved. Model was still improving at ep 30 (best was ep 30). `mode="default"` with `dynamic=True` is the correct setting — pad_collate produces variable-length tensors that cause recompilation storms under `reduce-overhead`. Peak memory dropped 33→24 GB. Cumulative gain now **-74.0%** vs initial 117.17 baseline.

---

## Previous Baseline — PR #1456 (bf16-amp + cosine-eta-min)

**val_avg/mae_surf_p = 36.8778** (epoch 16 of 17 completed in 30-min cap) — **-7.51% vs previous 39.8693**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=17, eta_min=1e-5)`** ← key addition (was T_max=14)
- **bf16 AMP enabled** ← key addition (was fp32)
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- **17 epochs in ~30 min** (vs 13 previously, ~+29% throughput from bf16)
- Peak GPU 32.98 GB (room for larger batch/model)

**Per-split val at best epoch (16):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| val_single_in_dist | **42.92** | −4.89 (−10.2%) |
| val_geom_camber_rc | **47.78** | −4.50 (−8.6%) |
| val_geom_camber_cruise | **18.60** | −2.29 (−11.0%) |
| val_re_rand | 38.21 | −0.28 (−0.7%) |
| **val_avg** | **36.8778** | **−2.99 (−7.51%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| test_single_in_dist | **42.15** | −3.80 (−8.3%) |
| test_geom_camber_rc | **42.69** | −3.64 (−7.9%) |
| test_geom_camber_cruise | **15.26** | −1.98 (−11.5%) |
| test_re_rand | **27.53** | −3.84 (−12.2%) |
| **test_avg** | **31.9058** | **−3.32 (−9.42%)** |

**Convergence trace**: 172.42 → 161.06 → 135.82 → 106.21 → 88.47 → 79.09 → 76.91 → 72.14 → 61.82 → 58.93 → 52.57 → 51.18 → 43.74 → 39.71 → 38.44 → **36.88** → 36.97 (ep 16 best; ep 17 drifts back +0.09 at LR floor).

**Grad clip / norm trace**: clip_frac smoothly decays 0.98 → 0.34 across 17 epochs. `huber_l2_frac` rises 0.42 → 0.86 — Huber actively capping outliers throughout.

**Artifact**: `models/model-charliepai2g24h1-alphonse-bf16-amp-cosine-eta-min-20260513-005955/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP + bf16 + CosineAnnealingLR(T_max=17, eta_min=1e-5) are now defaults on this branch
```

**Key insight**: bf16 AMP gives ~+29% throughput at zero quality cost. T_max=17 aligns the cosine tail with the new 17-epoch budget. ALL 8 splits (4 val + 4 test) improved — the broad win signals the model genuinely benefits from more epochs, not just a per-split tuning. eta_min=1e-5 keeps the late epoch usable. Peak memory only 33/96 GB — substantial headroom for larger batch or model.

---

## Previous Baseline — PR #1630 (cosine-eta-min)

**val_avg/mae_surf_p = 39.8693** (epoch 13 / 13 completed in 30-min cap) — **-5.97% vs previous 42.4015**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- **`CosineAnnealingLR(T_max=14, eta_min=1e-5)`** ← key addition (was `eta_min=0`)
- `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals
- ~13 epochs in ~30 min; peak GPU 42.15 GB

**Per-split val at best epoch (13):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| val_single_in_dist | 47.81 | +1.72 (worse) |
| val_geom_camber_rc | **52.28** | −3.70 |
| val_geom_camber_cruise | **20.89** | −3.43 |
| val_re_rand | **38.49** | −4.73 |
| **val_avg** | **39.8693** | **−2.53 (−5.97%)** |

**Test (all 4 splits):**

| Split | mae_surf_p | vs prev |
|-------|-----------|---------|
| test_single_in_dist | 45.95 | +4.19 (worse) |
| test_geom_camber_rc | **46.33** | −1.77 |
| test_geom_camber_cruise | **17.24** | −2.55 |
| test_re_rand | **31.37** | −4.60 |
| **test_avg** | **35.2214** | **−1.18 (−3.24%)** |

**LR trace (epoch 13)**: LR at ep 13 ≈ 5.90e-5 (vs 4.95e-5 without eta_min floor — +19% relative LR at the critical final epoch). Every earlier epoch is essentially identical to eta_min=0.

**Convergence trace**: 167.84 → 134.09 → 107.90 → 97.98 → 84.20 → 81.79 → 76.84 → 62.82 → 52.34 → 50.44 → 45.42 → 42.63 → **39.87** (monotone descent, still falling).

**Artifact**: `models/model-charliepai2g24h1-tanjiro-cosine-eta-min-20260512-231540/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP + CosineAnnealingLR(T_max=14, eta_min=1e-5) are now defaults on this branch
```

**Key insight**: `eta_min=1e-5` prevents the cosine schedule from reaching near-zero at epoch 13 (the run's last/best epoch). The +19% relative LR boost at the terminal epoch is enough to squeeze 3 additional OOD improvement without any other change. Single-line, zero-risk compounding on SOAP. Val still monotone descending at ep 13.

---

## Previous Baseline — PR #1613 (soap-optimizer)

**val_avg/mae_surf_p = 42.4015** (epoch 13 / 13 completed in 30-min cap) — **-52.6% vs previous 89.3940**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: **SOAP** (`precondition_frequency=10, max_precond_dim=256`, `lr=1e-3, wd=1e-4`)
- `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`
- Loss: Huber(δ=0.1) on relative-L2 normalized residuals (inherited)
- ~13 epochs in ~30 min (slight SOAP overhead vs 14 epoch baseline)
- SOAP vendored as `soap.py` (upstream commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`)

**Per-split val at best epoch (13):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 46.09 |
| val_geom_camber_rc | 55.98 |
| val_geom_camber_cruise | **24.32** |
| val_re_rand | 43.22 |
| **val_avg** | **42.4015** |

**Test (all 4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 41.76 |
| test_geom_camber_rc | 48.10 |
| test_geom_camber_cruise | 19.79 |
| test_re_rand | 35.97 |
| **test_avg** | **36.4017** |

**Convergence trace**: 163.69 → 107.26 → 83.20 → 75.45 → 72.44 → 58.88 → 52.97 → 49.79 → 44.55 → 42.40 (still falling at ep 13).

**Grad norm trace**: 38.87 → 27.04 → 19.66 → 14.39 → 9.16. SOAP's Kronecker-factored preconditioner is producing 4.2× gradient norm reduction. Clip frac: 1.000 through ep 10, then 0.997 → 0.987 → 0.984.

**Artifact**: `models/model-charliepai2g24h1-thorfinn-soap-optimizer-20260512-220030/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# SOAP optimizer, Huber+rel-L2, lr=1e-3, T_max=14, grad_clip=1.0 are now defaults on this branch
```

**Key insight**: SOAP's Kronecker-factored quasi-Newton preconditioner transforms this problem. The 4.2× grad norm reduction means each step is much better conditioned — the optimizer is following the loss surface curvature rather than a noisy first-order gradient. This is the largest single improvement in the programme (+52.6%). Val is still falling at ep 13 — more epochs (bf16-amp) would compound significantly.

---

## Previous Baseline — PR #1473 (huber-relative-l2-compound)

**val_avg/mae_surf_p = 89.3940** (epoch 14 / 14 completed in 30-min cap) — **-0.24% vs previous 89.6121**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`
- `batch_size=4`, `surf_weight=10.0`, **Huber(δ=0.1) applied to relative-L2 normalized residuals**
- ~14 epochs in ~30 min
- Loss: `huber_relative_l2` — Huber on per-sample energy-normalized residuals (δ=0.1 in normalized space)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 109.01 |
| val_geom_camber_rc | 101.19 |
| val_geom_camber_cruise | **66.36** |
| val_re_rand | 81.02 |
| **val_avg** | **89.3940** |

**Test (all 4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 98.51 |
| test_geom_camber_rc | 88.12 |
| test_geom_camber_cruise | 54.80 |
| test_re_rand | 76.97 |
| **test_avg** | **79.5993** |

**Artifact**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-211810/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# Huber(delta=0.1) on relative-L2, lr=1e-3, T_max=14, grad_clip=1.0 now defaults on this branch
```

**Key insight**: Huber(δ=0.1) in normalized residual space compounds cleanly with relative-L2. The L2-fraction trajectory (33%→63%) shows Huber remains genuinely active throughout training — the delta=0.1 in normalized space is well-placed for intra-sample outlier capping without collapsing to MSE early. Grad clip_frac dropped from 1.0 to 0.075 by epoch 14 (vs ~0.98 on rel-L2-only) — the compound loss is significantly smoother. Val still falling at epoch 14.

---

## Previous Baseline — PR #1460 (relative-l2-loss)

**val_avg/mae_surf_p = 89.6121** (epoch 14 / 14 completed in 30-min cap) — **-7.20% vs previous 96.5587**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`
- `batch_size=4`, `surf_weight=10.0`, **per-sample relative L2 loss** (`||pred-y||²/||y||²`)
- ~131s/epoch; 14 epochs in ~30 min
- Loss: per-sample relative L2 in normalized space (replaces MSE)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 109.07 |
| val_geom_camber_rc | 97.99 |
| val_geom_camber_cruise | **67.09** |
| val_re_rand | 84.29 |
| **val_avg** | **89.6121** |

**Test (4 splits):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 91.14 |
| test_geom_camber_rc | 85.89 |
| test_geom_camber_cruise | 56.35 |
| test_re_rand | 79.18 |
| **test_avg** | **78.14** |

**Artifact**: `models/model-charliepai2g24h1-fern-relative-l2-loss-20260512-200551/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# loss=relative_l2, lr=1e-3, T_max=14, grad_clip=1.0 are now the default Config values on this branch
```

**Key insight**: Relative L2 loss (`||pred-y||²/||y||²`) normalizes by sample energy, automatically down-weighting high-energy (extreme-value) samples and up-weighting low-energy ones. This is a better inductive bias than MSE for flows with large Re variation — the loss landscape is flatter and more homogeneous across splits. Val still falling at epoch 14 (95.94 → 93.35 → 89.61 in last 3 epochs); more epochs would help.

**Gradient diagnostic**: clip_frac fell to 0.984 at ep 14 (was 1.0 throughout on MSE baseline) — relative-L2 is producing smaller raw gradient norms. The loss surface is smoother.

---

## Previous Baseline — PR #1518 (higher-lr-cosine-14)

**val_avg/mae_surf_p = 96.5587** (epoch 14 / 14 completed in 30-min cap) — **-17.6% vs previous 117.17**

- Architecture: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params)
- Optimizer: `AdamW(lr=1e-3, wd=1e-4)`, **`CosineAnnealingLR(T_max=14)`** ← key addition (was T_max=50)
- `batch_size=4`, `surf_weight=10.0`, `grad_clip=1.0`, MSE loss in normalized space
- ~131s/epoch; 14 epochs in ~30 min
- NaN scoring fix included: y-sanitization in `train.py:evaluate_split` (test_avg now 4-split clean)

**Per-split val at best epoch (14):**

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 108.58 |
| val_geom_camber_rc | 110.59 |
| val_geom_camber_cruise | **74.35** |
| val_re_rand | 92.71 |
| **val_avg** | **96.5587** |

**Test (all 4 splits, NaN-free — scoring fix now in branch):**

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 94.97 |
| test_geom_camber_rc | 99.77 |
| test_geom_camber_cruise | 61.86 |
| test_re_rand | 86.88 |
| **test_avg** | **85.87** |

**Artifact**: `models/model-charliepai2g24h1-thorfinn-higher-lr-cosine-14-20260512-191045/metrics.jsonl`

**Reproduce**:
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py \
  --agent <name> --experiment_name <name> --epochs 50
# lr=1e-3, T_max=14, grad_clip=1.0 are now the default Config values on this branch
```

**Key insight**: With `grad_clip=1.0` bounding the effective step size, pushing lr from 5e-4 to 1e-3 gave faster convergence. Reducing T_max from 50 to 14 made the model actually reach its low-LR fine-tuning phase within the 14-epoch wall-clock budget — the model was still improving at epoch 14. Val still falling at epoch 14 (100.34 → 98.66 → 96.56); a slightly longer or flatter cosine tail may yield additional gains.

**Key diagnostic**: Val crossed the old 117.17 baseline at epoch 10, reaching 96.56 by epoch 14. Pre-clip norms still 23–66 / 288–740 max. Clipping fires on ~100% of batches.

---

## Previous Baseline — PR #1479 (grad-clip-1)

**val_avg/mae_surf_p = 117.17** (epoch 13 / 14 completed in 30-min cap)

- `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=50)`, `grad_clip=1.0`
- **Artifact**: `models/model-charliepai2g24h1-thorfinn-grad-clip-1-20260512-180544/metrics.jsonl`

---

## Update Log

| Date | PR | val_avg/mae_surf_p | test_avg | Notes |
|------|----|--------------------|---------|-------|
| 2026-05-13 | #2011 | **28.8762** | **24.9992** | film-re-attention (ReFiLM on slice logits); 28 epochs / 30 min; -1.17% |
| 2026-05-13 | #1614 | **29.2179** | **25.6024** | per-channel-loss-weights p=5 post-Huber; 29 epochs / 30 min; -2.11% |
| 2026-05-13 | #1599 | **29.8463** | **26.1005** | re-conditioned-scaling (ReScaleHead 3ch); 29 epochs / 30 min; -1.95% |
| 2026-05-13 | #1794 | **30.4412** | **26.1013** | torch.compile(default,dynamic=True); 30 epochs / 30 min; -17.5% |
| 2026-05-13 | #1456 | **36.8778** | **31.9058** | bf16-amp + cosine-eta-min T_max=17; 17 epochs / 30 min; -7.51% |
| 2026-05-13 | #1630 | **39.8693** | **35.2214** | cosine-eta-min (eta_min=1e-5); 13 epochs / 30 min; -5.97% |
| 2026-05-12 | #1613 | **42.4015** | **36.4017** | SOAP optimizer; 13 epochs / 30 min; -52.6% |
| 2026-05-12 | #1460 | **89.6121** | **78.14** | relative-l2-loss; 14 epochs / 30 min; -7.20% |
| 2026-05-12 | #1473 | **89.3940** | **79.5993** | huber-relative-l2-compound; 14 epochs / 30 min; -0.24% |
| 2026-05-12 | #1518 | **96.5587** | **85.87** (4-split) | Round 2 winner. lr=1e-3, T_max=14; 14 epochs / 30 min |
| 2026-05-12 | #1479 | 117.17 | 116.17 (3-split) | Round 1 winner. grad_clip=1.0; 14 epochs / 30 min |
