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

> **Beat this:** submit a PR improving `val_avg/mae_surf_p` below **135.0153** with a terminal `SENPAI-RESULT` marker.
