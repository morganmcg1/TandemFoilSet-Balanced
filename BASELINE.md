# Baseline Metrics — icml-appendix-charlie-pai2i-48h-r2

All baselines on `val_avg/mae_surf_p` (lower is better). Test metric is `test_avg/mae_surf_p`.

---

## 2026-05-15 13:50 — PR #3119: Round-1 baseline (epochs=80 config, ~14 effective epochs)

**Student:** charliepai2i48h2-thorfinn  
**Change:** epochs 50→80 (cosine T_max=80). 30-min timeout cap hit at epoch 14 — schedule change had no effective impact. This run establishes the round-1 absolute reference.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **135.0153** |
| val_single_in_dist/mae_surf_p | 167.5711 |
| val_geom_camber_rc/mae_surf_p | 148.3005 |
| val_geom_camber_cruise/mae_surf_p | 103.9843 |
| val_re_rand/mae_surf_p | 120.2051 |
| **test_avg/mae_surf_p** | **NaN** (pre-existing scoring bug; 3-split proxy = 132.4954) |
| test_single_in_dist/mae_surf_p | 145.30 |
| test_geom_camber_rc/mae_surf_p | 133.81 |
| test_geom_camber_cruise/mae_surf_p | NaN (1 GT sample has NaN p) |
| test_re_rand/mae_surf_p | 118.37 |
| Best epoch | 12 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU, trunc_normal_(std=0.02)  
**Optimizer:** AdamW lr=5e-4, weight_decay=1e-4, CosineAnnealingLR(T_max=80, effective T_max≈50)  
**Loss:** vol_loss + 10·surf_loss (uniform per-channel)  
**Batch:** 4

**Scoring bug note:** `test_geom_camber_cruise/mae_surf_p` is NaN due to sample 20 of that split having NaN p in GT (accumulate_batch NaN×0=NaN propagation). A one-line fix via `torch.nan_to_num` in `evaluate_split` in train.py will resolve this; it is tracked as PR bug-fix work.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --epochs 50
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **135.0153**~~ — superseded by PR #3101 below.

---

## 2026-05-15 16:26 — PR #3101: surf_weight 10→30 (3× surface loss emphasis)

**Student:** charliepai2i48h2-askeladd  
**Change:** `Config.surf_weight: 10.0 → 30.0` — single-line change aligning training loss with primary metric.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **127.4122** |
| val_single_in_dist/mae_surf_p | 152.8215 |
| val_geom_camber_rc/mae_surf_p | 134.8461 |
| val_geom_camber_cruise/mae_surf_p | 102.5963 |
| val_re_rand/mae_surf_p | 119.3849 |
| **test_avg/mae_surf_p** | **NaN** (scoring bug; corrected 4-split avg = 116.83) |
| test_single_in_dist/mae_surf_p | 136.68 |
| test_geom_camber_rc/mae_surf_p | 123.03 |
| test_geom_camber_cruise/mae_surf_p | NaN → 88.12 (corrected via test_metrics_corrected.json) |
| test_re_rand/mae_surf_p | 119.51 |
| Best epoch | 14 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** AdamW lr=5e-4, weight_decay=1e-4, CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + **30·surf_loss** (was 10) — uniform per-channel  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-askeladd-surf-weight-30-20260515-124531/metrics.jsonl`

**Note:** test_avg/mae_surf_p is NaN for this run (pre-merge; bug-fix PR #3274 now merged). Use test_metrics_corrected.json for the corrected 4-split test avg (116.83).

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 30
```

> ~~**Beat this:** val below **127.4122**~~ — superseded by PR #3293 below.

---

## 2026-05-15 17:25 — PR #3293: Lion optimizer replacing AdamW (lr=1.7e-4, wd=3e-4)

**Student:** charliepai2i48h2-nezuko  
**Change:** Replace AdamW with Lion (sign-based update, single momentum buffer). lr=5e-4→1.7e-4, wd=1e-4→3e-4. All other settings baseline (including surf_weight=10 during the run; merged config now has surf_weight=30 from PR #3101 compounded automatically).

| Metric | Value (measured at surf_weight=10) |
|--------|------------------------------------|
| **val_avg/mae_surf_p** | **117.5014** |
| val_single_in_dist/mae_surf_p | 137.2384 |
| val_geom_camber_rc/mae_surf_p | 124.4193 |
| val_geom_camber_cruise/mae_surf_p | 97.3192 |
| val_re_rand/mae_surf_p | 111.0287 |
| **test_avg/mae_surf_p** | **NaN** (pre-merge; #3274 fix now live) |
| test_single_in_dist/mae_surf_p | 121.5755 |
| test_geom_camber_rc/mae_surf_p | 117.3048 |
| test_geom_camber_cruise/mae_surf_p | NaN (scoring bug — now fixed) |
| test_re_rand/mae_surf_p | 108.2318 |
| test_avg (3-split proxy) | 115.7040 |
| Best epoch | 9 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** **Lion** lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + **30·surf_loss** (surf_weight=30 from PR #3101, compounded in merged HEAD)  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-nezuko-lion-optimizer-20260515-153522/metrics.jsonl`

**Note:** val=117.50 was measured with surf_weight=10; merged HEAD has surf_weight=30 + Lion (unmeasured combined). Next experiments should compare against 117.50 and also produce a clean Lion+surf30 measurement.

**Reproduce (merged config — Lion + surf_weight=30):**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** val below **117.5014**~~ — superseded by PR #3357 below.

---

## 2026-05-15 18:40 — PR #3357: asinh loss transform for surface pressure

**Student:** charliepai2i48h2-tanjiro  
**Change:** Apply `torch.asinh()` to pressure channel z-scores in the training loss only (not evaluation). Compresses heavy-tail gradient signal on pressure; Ux/Uy loss unchanged. Evaluation MAE remains in physical units — comparison is fair.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **84.9819** |
| val_single_in_dist/mae_surf_p | 108.0437 |
| val_geom_camber_rc/mae_surf_p | 90.6260 |
| val_geom_camber_cruise/mae_surf_p | 62.6771 |
| val_re_rand/mae_surf_p | 78.5807 |
| **test_avg/mae_surf_p** | **76.1441** |
| test_single_in_dist/mae_surf_p | 94.3654 |
| test_geom_camber_rc/mae_surf_p | 82.8340 |
| test_geom_camber_cruise/mae_surf_p | 53.5469 |
| test_re_rand/mae_surf_p | 73.8302 |
| Best epoch | 14 (timeout-bound; val still descending) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with **asinh(z) transform on pressure channel z-scores**  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-tanjiro-asinh-pressure-loss-20260515-173315/metrics.jsonl`

**Change (7 lines in train.py):**
```python
# asinh loss compression on pressure channel (index 2) only.
sq_err_uxuy = (pred[..., :2] - y_norm[..., :2]) ** 2
sq_err_p = (torch.asinh(pred[..., 2:3]) - torch.asinh(y_norm[..., 2:3])) ** 2
sq_err = torch.cat([sq_err_uxuy, sq_err_p], dim=-1)
```

**Note:** −27.7% on val (84.98 vs 117.50), −34% on test (76.14 vs 115.70 proxy). All 4 splits improved 20-36%. Val curve still descending at epoch 14 timeout — 84.98 is a loose upper bound on what this loss can do. Stronger gains on OOD splits (geom_camber_rc −27%, re_rand −29%) than in-dist (single_in_dist −21%).

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **84.9819**~~ — superseded by PR #3382 below.

---

## 2026-05-15 21:34 — PR #3382: EMA weights (decay=0.999) on asinh baseline

**Student:** charliepai2i48h2-askeladd  
**Change:** EMA shadow (decay=0.999) applied at every val and test pass on top of Lion + surf_weight=30 + asinh baseline. Live weights unchanged during training; EMA shadow loaded for all evaluation passes.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **83.1874** |
| val_single_in_dist/mae_surf_p | 99.9507 |
| val_geom_camber_rc/mae_surf_p | 94.1543 |
| val_geom_camber_cruise/mae_surf_p | 60.2646 |
| val_re_rand/mae_surf_p | 78.3801 |
| **test_avg/mae_surf_p** | **74.5193** |
| test_single_in_dist/mae_surf_p | 88.9512 |
| test_geom_camber_rc/mae_surf_p | 85.8766 |
| test_geom_camber_cruise/mae_surf_p | 50.9533 |
| test_re_rand/mae_surf_p | 72.2961 |
| Best epoch | 14 (timeout-bound; every epoch a new best, shadow still converging) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with asinh(z) on pressure channel  
**EMA:** decay=0.999, applied at val/test passes only — live weights unchanged  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-askeladd-ema-weights-decay-0999-rebased-20260515-203115/metrics.jsonl`

**Note:** −2.11% on val (83.19 vs 84.98), −2.13% on test (74.52 vs 76.14). All 4 val splits improved. Smoothness diagnostic: 0 sign flips in val curve (vs 8 for Lion baseline), every epoch a new best — EMA shadow still catching up at timeout. Mechanisms compose cleanly: asinh smooths loss landscape, EMA then smooths parameter trajectory on top.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **83.1874**~~ — superseded by PR #3384 below.

---

## 2026-05-15 23:37 — PR #3384: Gradient clipping (max_norm=1.0) on full EMA+asinh stack

**Student:** charliepai2i48h2-fern  
**Change:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` inserted between `loss.backward()` and `optimizer.step()` in the training loop. Rebased on full stack: Lion lr=1.7e-4, wd=3e-4, surf_weight=30, asinh pressure-loss, EMA(0.999).

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **70.2479** |
| val_single_in_dist/mae_surf_p | 81.5014 |
| val_geom_camber_rc/mae_surf_p | 82.7986 |
| val_geom_camber_cruise/mae_surf_p | 49.2201 |
| val_re_rand/mae_surf_p | 67.4712 |
| **test_avg/mae_surf_p** | **62.0765** |
| test_single_in_dist/mae_surf_p | 73.2488 |
| test_geom_camber_rc/mae_surf_p | 73.2597 |
| test_geom_camber_cruise/mae_surf_p | 41.3097 |
| test_re_rand/mae_surf_p | 60.4878 |
| Best epoch | 14 (timeout-bound; val still descending) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with asinh(z) on pressure channel  
**EMA:** decay=0.999, applied at val/test passes  
**Gradient clipping:** max_norm=1.0 (inserted before optimizer.step())  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-fern-lion-gradclip-1.0-rebased-20260515-222530/metrics.jsonl`

**Note:** −15.6% on val (70.25 vs 83.19), −16.7% on test (62.08 vs 74.52). All 4 splits improved 12-19%. Mechanisms compose cleanly: asinh compresses loss-level heavy tails, EMA smooths parameter trajectory, grad-clip caps per-step gradient norm. Post-asinh pre-clip norms still 25-180 (100% of steps clip) — the three mechanisms target different points in the gradient pipeline. Val curve still descending at epoch 14 timeout.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> **Beat this:** submit a PR improving `val_avg/mae_surf_p` below **70.2479** with a terminal `SENPAI-RESULT` marker.
