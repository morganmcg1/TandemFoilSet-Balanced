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

> **Beat this:** submit a PR improving `val_avg/mae_surf_p` below **117.5014** with a terminal `SENPAI-RESULT` marker.
