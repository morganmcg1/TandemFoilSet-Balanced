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
