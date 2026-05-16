# Baseline Metrics — icml-appendix-charlie-pai2i-24h-r2

## Current best

### 2026-05-16 06:27 — PR #3643: n_head 4→2 (head_dim 24→48) — wider per-head subspace

- **val_avg/mae_surf_p:** 70.9252 (best @ epoch 14 / 14; **−6.16% vs previous best**)
- **test_avg/mae_surf_p:** 61.9143 (**−7.23% vs previous best**)
- **Per-split val mae_surf_p:** All four splits improved (curve still descending at E14)
- **Per-split test mae_surf_p:** All four splits improved
- **Changes:** n_head 4→2 (head_dim 24→48). Wider per-head subspace at constant n_hidden=96.
- **n_params:** 509,389 (+7.93% vs PR #3654's 471,959 — Q/K/V projections scale with dim_head)
- **Wall-clock:** 27.97 min total (~119.9 s/epoch; all 14 epochs completed, **faster** than baseline)
- **Peak VRAM:** 39.13 GB (vs baseline 46.92 GB — head reduction saves attention memory)
- **Metric artifacts:** `models/model-n-head-2-head-dim-48-20260516-053417/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name n-head-2-head-dim-48 --agent charliepai2i24h2-askeladd --epochs 14`
- **Delta vs previous best (#3654):** −4.65 val_avg/mae_surf_p (75.578 → 70.925); −4.83 test (66.740 → 61.914). Every val and test split improved. Last epoch drop −6.7% suggests further headroom with longer training.

### 2026-05-16 05:29 — PR #3654: SwiGLU full mlp_ratio=2 (hidden_inner 128→192, superseded)

- **val_avg/mae_surf_p:** 75.5776 (best @ epoch 13 / 14; −3.61% vs previous best)
- **test_avg/mae_surf_p:** 66.7399 (−2.39% vs previous best)
- **Per-split val mae_surf_p:** single 86.290 | geom_rc 89.762 | geom_cruise 55.041 | re_rand 71.218
- **Per-split test mae_surf_p:** single 77.887 | geom_rc 77.254 | geom_cruise 47.216 | re_rand 64.602
- **Changes:** SwiGLU hidden_inner 128→192 (+50% FFN params per block).
- **n_params:** 471,959
- **Metric artifacts:** `models/model-swiglu-full-mlpratio2-20260516-042703/metrics.{jsonl,yaml}`

### 2026-05-16 03:30 — PR #3608: SwiGLU FFN (param-matched replacement for GELU, superseded)

- **val_avg/mae_surf_p:** 78.407 (best @ epoch 13 / 14; −18.2% vs previous best)
- **test_avg/mae_surf_p:** 68.375 (−20.1% vs previous best)
- **Per-split val mae_surf_p:** single 94.301 | geom_rc 89.780 | geom_cruise 56.169 | re_rand 73.379
- **Per-split test mae_surf_p:** single 83.095 | geom_rc 79.596 | geom_cruise 45.973 | re_rand 64.837
- **Changes:** GELU FFN replaced by SwiGLU `W2(SiLU(W1(x)) ⊙ V(x))` with hidden_inner=128 (param-matched). All 5 Transolver blocks use SwiGLU.
- **n_params:** 379,799
- **Metric artifacts:** `models/model-swiglu-ffn-20260516-022733/metrics.{jsonl,yaml}`

### 2026-05-16 00:35 — PR #3314: AdamW weight_decay 1e-4 → 3e-4 on decay group (single-axis, superseded)

- **val_avg/mae_surf_p:** 95.808 (best @ epoch 14; all 14 epochs completed)
- **test_avg/mae_surf_p:** 85.578
- **Per-split val mae_surf_p:** single 110.886 | geom_rc 105.776 | geom_cruise 76.060 | re_rand 90.510
- **Per-split test mae_surf_p:** single 97.804 | geom_rc 94.519 | geom_cruise 64.863 | re_rand 85.126
- **Changes:** weight_decay 1e-4→3e-4 on decay group only (LN/bias/1D still no-decay)
- **Wall-clock:** 32.0 min (~137 s/epoch, all 14 epochs completed within cap)
- **Peak VRAM:** 40.96 GB
- **Metric artifacts:** `models/model-weight-decay-3e-4-rebased-20260515-232904/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name weight-decay-3e-4-rebased --agent charliepai2i24h2-fern --epochs 14`
- **Delta vs previous best (#3377):** -0.89% val_avg/mae_surf_p (96.667 → 95.808); +0.14% test (85.454 → 85.578, within noise)

### 2026-05-15 22:58 — PR #3377: Scale n_hidden 128→96 (single-axis width sweep, superseded)

- **val_avg/mae_surf_p:** 96.667 (best @ epoch 14; all 14 epochs completed)
- **test_avg/mae_surf_p:** 85.454
- **Per-split val mae_surf_p:** single 116.665 | geom_rc 105.516 | geom_cruise 73.065 | re_rand 91.421
- **Per-split test mae_surf_p:** single 99.939 | geom_rc 95.608 | geom_cruise 61.246 | re_rand 85.023
- **Changes:** n_hidden 128→96 (single-axis; capacity -43%, 0.381M vs 0.655M params)
- **Wall-clock:** 31.9 min (~136.7 s/epoch, all 14 epochs completed within cap)
- **Peak VRAM:** 40.97 GB
- **Metric artifacts:** `models/model-n-hidden-96-rebased-20260515-213114/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name n-hidden-96-rebased --agent charliepai2i24h2-thorfinn --epochs 14`
- **Delta vs previous best (#3399):** -1.12% val_avg/mae_surf_p (97.757 → 96.667); -1.08% test (86.388 → 85.454)

### 2026-05-15 21:15 — PR #3399: Scale slice_num 64→96 (superseded)

- **val_avg/mae_surf_p:** 97.757 (best @ epoch 12; 12 epochs completed under 30-min cap, still descending at cutoff)
- **test_avg/mae_surf_p:** 86.388
- **Per-split val mae_surf_p:** single 115.495 | geom_rc 110.451 | geom_cruise 75.398 | re_rand 89.685
- **Per-split test mae_surf_p:** single 101.647 | geom_rc 94.849 | geom_cruise 64.297 | re_rand 84.757
- **Changes:** slice_num 64→96 (single-axis)
- **Loss/optimizer/schedule:** carried forward from PR #3294 stack (warmup+cosine 14ep, lr=7e-4, AdamW selective decay, grad-clip, NaN guard)
- **Wall-clock:** 30.2 min (~151 s/epoch, fits 12 of 14 scheduled epochs)
- **Metric artifacts:** `models/model-slice-num-96-20260515-192511/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name slice-num-96 --agent charliepai2i24h2-edward --epochs 14`
- **Delta vs previous best (#3294):** -3.03% val_avg/mae_surf_p (100.811 → 97.757)

### 2026-05-15 17:30 — PR #3294: Warmup + cosine over 14 epochs, lr=7e-4 (superseded)

- **val_avg/mae_surf_p:** 100.811 (best @ epoch 14; 14 epochs completed under 30-min cap)
- **test_avg/mae_surf_p:** NaN (pre-existing infra issue on test_geom_camber_cruise); 3-clean-split mean 99.15
- **Per-split val mae_surf_p:** single 118.74 | geom_rc 107.10 | geom_cruise 81.97 | re_rand 95.43
- **Per-split test mae_surf_p:** single 109.88 | geom_rc 95.54 | geom_cruise NaN | re_rand 92.02
- **Changes:** SequentialLR (LinearLR warmup over 2 ep + CosineAnnealingLR T_max=12) + lr 5e-4→7e-4 + epochs=14 budget-matched
- **Loss:** SmoothL1 (Huber, β=1.0) — carried forward from PR #3208
- **Optimizer:** AdamW selective decay (weight_decay=1e-4) + grad-clip max_norm=1.0 — carried forward from PR #3276
- **Metric artifacts:** `models/model-warmup-cosine-14ep-20260515-162249/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name warmup-cosine-14ep --agent charliepai2i24h2-tanjiro --epochs 14`
- **Delta vs previous best (#3276):** -8.08% val_avg/mae_surf_p (109.681 → 100.811)

### 2026-05-15 15:30 — PR #3276: Gradient clip + AdamW selective decay (+ test NaN guard) (superseded)

- **val_avg/mae_surf_p:** 109.681 (best @ epoch 14; 14 epochs completed under 30-min cap)
- **test_avg/mae_surf_p:** 97.315 (finite for first time — NaN guard fixed test_geom_camber_cruise)
- **Per-split val mae_surf_p:** single 148.09 | geom_rc 114.87 | geom_cruise 78.85 | re_rand 96.91
- **Per-split test mae_surf_p:** single 123.24 | geom_rc 104.76 | geom_cruise 68.48 | re_rand 92.79
- **Changes:** torch.nn.utils.clip_grad_norm\_(max_norm=1.0) + AdamW selective decay (LN/bias/1D no-decay) + NaN sample guard in evaluate_split
- **Optimizer groups:** decay=49 groups (0.655M params), no_decay=62 groups (0.008M params)
- **Loss:** SmoothL1 (Huber, β=1.0) — carried forward from PR #3208
- **Metric artifacts:** `models/model-grad-clip-selective-decay-20260515-142950/metrics.{jsonl,yaml}`
- **Reproduce:** `cd target && python train.py --experiment_name grad-clip-selective-decay --agent fern --epochs 50`
- **Delta vs PR #3208 baseline:** -5.94% val_avg/mae_surf_p (116.61 → 109.68)

### 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss (superseded)

- **val_avg/mae_surf_p:** 116.611 (best @ epoch 13; 14 epochs completed under 30-min cap)
- **Metric artifacts:** `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}`

## Reference model config
```python
model_config = dict(
    space_dim=2,
    fun_dim=22,      # X_DIM (24) - 2 position dims
    out_dim=3,
    n_hidden=96,      # updated by PR #3377
    n_layers=5,
    n_head=4,
    slice_num=96,    # updated by PR #3399
    mlp_ratio=2,
    # FFN is now SwiGLU with hidden_inner=128 (updated by PR #3608)
)
```
~380k params.

## Reference training config (current baseline stack — PR #3208 + #3276 + #3294 + #3399)
- AdamW: lr=7e-4, weight_decay=3e-4 (decay group only, updated by PR #3314), no-decay group for LN/bias/1D
- grad-clip: clip_grad_norm_(max_norm=1.0)
- batch_size=4
- surf_weight=10.0 (additional surface loss weight in normalized space)
- epochs=14 (budget-matched; wall-clock cap ~30 min → ~12 epochs at ~151 s/epoch with slice_num=96)
- Scheduler: SequentialLR (LinearLR warmup start_factor=1e-3 over 2 ep, then CosineAnnealingLR T_max=12)
- Loss: SmoothL1 (Huber, β=1.0) in normalized target space; vol_loss + surf_weight * surf_loss
- NaN guard in evaluate_split (pre-sanitize y, mask bad samples before scoring)
- Balanced domain sampler (WeightedRandomSampler) over 1499 train samples

## Diagnostic targets (per split, surface MAE for p)
We track per-split surface pressure MAE separately so that geometry vs Re axes are visible:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

The same four exist for test splits. We report all four plus the equal-weight average.
