# TandemFoilSet Baseline Metrics

## 2026-04-29 15:45 — PR #1201: CosineAnnealingLR T_max=15 + LR 1e-3 for matched convergence

**Student:** charliepai2f4-fern
**Branch:** charliepai2f4-fern/cosine-tmax-match-lr1e3
**Model:** Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.1) + gradient clipping (max_norm=1.0) + lr=1e-3 + CosineAnnealingLR(T_max=15, eta_min=1e-6)
**Best epoch:** 14 / 50 (training timed out at ~14 epochs)
**Metric summary:** `models/model-cosine-tmax15-lr1e3-20260429-145857/metrics.yaml`

### Primary metric (lower is better)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **92.170** |

### Per-split surface pressure MAE (val)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 113.457 |
| val_geom_camber_rc | 107.113 |
| val_geom_camber_cruise | 64.413 |
| val_re_rand | 83.696 |

### Test split surface pressure MAE

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 98.184 |
| test_geom_camber_rc | 97.509 |
| test_geom_camber_cruise | NaN (pre-existing corrupt sample) |
| test_re_rand | 79.432 |

### Key method details

- Matched T_max=15 to actual training budget (~14 epochs), ensuring full cosine decay within wall-clock limit
- Peak LR raised from 8e-4 to 1e-3 (25% increase), stable with gradient clipping max_norm=1.0
- eta_min=1e-6 (explicit near-zero floor for cosine schedule)
- surf_weight=10, dropout=0.1, batch_size=4, AdamW, weight_decay=1e-4
- 9.7% improvement on val_avg/mae_surf_p over prior baseline (PR #1187: 102.080 → 92.170)
- Peak memory: 42.74 GB

### Reproduce

```bash
cd target/ && python train.py \
  --experiment_name cosine-tmax15-lr1e3 \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --dropout 0.1 --lr 1e-3 --surf_weight 10 --batch_size 4 \
  --agent charliepai2f4-fern
```
(Note: T_max=15, eta_min=1e-6 are set as defaults in train.py Config after this merge.)

---

## 2026-04-29 14:38 — PR #1187: Gradient clipping + raised LR 8e-4 for faster convergence

**Student:** charliepai2f4-fern
**Branch:** charliepai2f4-fern/gradient-clip-lr8e4
**Model:** Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.1) + gradient clipping (max_norm=1.0) + lr=8e-4
**Best epoch:** 14 / 50 (training timed out at ~14 epochs)
**Metric summary:** `models/model-gradient-clip-lr8e4-20260429-135924/metrics.yaml`

### Primary metric (lower is better)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **102.080** |

### Per-split surface pressure MAE (val)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 124.600 |
| val_geom_camber_rc | 105.819 |
| val_geom_camber_cruise | 82.739 |
| val_re_rand | 95.161 |

### Test split surface pressure MAE

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 106.286 |
| test_geom_camber_rc | 97.570 |
| test_geom_camber_cruise | NaN (pre-existing corrupt sample) |
| test_re_rand | 90.364 |

### Key method details

- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` and before `optimizer.step()`
- LR raised from 5e-4 to 8e-4 (60% increase), stable with gradient clipping
- surf_weight=10, dropout=0.1, batch_size=4, CosineAnnealingLR(T_max=50), AdamW, weight_decay=1e-4
- 18.2% improvement on val_avg/mae_surf_p over prior baseline (PR #1128: 124.727 → 102.080)
- Peak memory: 42.75 GB

### Reproduce

```bash
cd target/ && python train.py \
  --experiment_name gradient-clip-lr8e4 \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --dropout 0.1 --lr 8e-4 --surf_weight 10 --batch_size 4 \
  --agent charliepai2f4-fern
```

---

## 2026-04-29 — PR #1128: Per-sample Re-adaptive loss normalization for pressure scale

**Student:** charliepai2f4-edward
**Branch:** charliepai2f4-edward/per-sample-re-adaptive-loss
**Model:** Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.1) + per-sample GT-std inverse weighting (1/σ, clamp=0.2, mean-1 norm)
**Best epoch:** 13 / 50 (training timed out at ~14 epochs)
**Metric summary:** `models/model-per-sample-re-adaptive-loss-invstd-20260429-130050/metrics.yaml`

### Primary metric (lower is better)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **124.727** |

### Per-split surface pressure MAE (val)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 153.200 |
| val_geom_camber_rc | 133.070 |
| val_geom_camber_cruise | 96.830 |
| val_re_rand | 115.800 |

### Test split surface pressure MAE

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 138.800 |
| test_geom_camber_rc | 124.400 |
| test_geom_camber_cruise | NaN (pre-existing corrupt sample) |
| test_re_rand | 114.690 |

### Key method details

- Per-sample GT surface pressure std (from `y_norm[:, :, 2]` surface nodes) used to weight sq_err
- Weighting form: `1/σ` (not `1/σ²`) with `clamp(min=0.2)` and mean-1 batch normalization
- `surf_weight=10` unchanged; per-sample equalization operates at sample level
- Run 4 in sequence: pred-std diverged (Runs 1-2), GT-std with 1/σ² also failed (Run 3), 1/σ succeeded

### Reproduce

```bash
cd target/ && python train.py \
  --experiment_name per-sample-re-adaptive-loss-invstd \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --dropout 0.1 --lr 5e-4 --surf_weight 10 --batch_size 4 \
  --agent charliepai2f4-edward
```

---

## 2026-04-29 — PR #1112: Attention dropout=0.1 for OOD slice regularization

**Student:** charliepai2f4-edward  
**Branch:** charliepai2f4-edward/attention-dropout  
**Model:** Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.1)  
**Best epoch:** 13 / 50 (training timed out at ~14 epochs)  
**Metric summary:** `models/model-attention-dropout-20260429-100507/metrics.yaml`

### Primary metric (lower is better)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **129.531** |

### Per-split surface pressure MAE (val)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 159.429 |
| val_geom_camber_rc | 155.559 |
| val_geom_camber_cruise | 92.955 |
| val_re_rand | 110.181 |

### Per-split surface MAE (val, all channels)

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p |
|-------|-------------|-------------|------------|
| val_single_in_dist | 1.832 | 0.847 | 159.429 |
| val_geom_camber_rc | 3.366 | 1.140 | 155.559 |
| val_geom_camber_cruise | 1.489 | 0.594 | 92.955 |
| val_re_rand | 2.028 | 0.859 | 110.181 |

### Test split surface pressure MAE

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 139.912 |
| test_geom_camber_rc | 138.778 |
| test_geom_camber_cruise | NaN (corrupt sample 000020.pt — inf values in ground truth) |
| test_re_rand | 110.986 |
| **test_avg/mae_surf_p** | **NaN** (due to corrupt sample) |

Note: `test_geom_camber_cruise/000020.pt` has 761 inf values in the ground-truth `y` tensor, causing `scoring.py` to produce NaN for the cruise test split. Valid test splits (3 of 4) show consistent generalization.

### Reproduce

```bash
cd target/ && python train.py \
  --experiment_name attention-dropout \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --dropout 0.1 --lr 5e-4 --surf_weight 10 --batch_size 4
```
