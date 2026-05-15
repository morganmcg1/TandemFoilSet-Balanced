# TandemFoilSet Baseline — branch `icml-appendix-charlie-pai2i-24h-r3`

This branch tracks the best-merged result on `icml-appendix-charlie-pai2i-24h-r3`.

## Baseline configuration

The starting point is the Transolver baseline in `train.py`:

- **Model**: Transolver with PhysicsAttention over slice-tokens
  - `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `space_dim=2`, `fun_dim=22`, `out_dim=3`
- **Optimizer**: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- **Schedule**: `CosineAnnealingLR(T_max=epochs)`
- **Batch**: `batch_size=4`
- **Loss**: `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`, both MSE in normalized target space
- **Sampling**: `WeightedRandomSampler` with domain-balancing weights
- **Run budget**: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30` (hard upper bounds)

## Primary metrics

- Validation ranking: **`val_avg/mae_surf_p`** (equal-weight mean surface pressure MAE across the four validation splits)
- Paper-facing test: **`test_avg/mae_surf_p`** (same metric across the four test splits, evaluated at the best-val checkpoint)
- Lower is better

## Reproduce

```bash
cd target/
python train.py --experiment_name baseline_transolver --agent baseline
```

This produces `models/model-baseline-<stamp>/metrics.jsonl` with per-epoch val metrics and a final test record.

## Best result

### 2026-05-15 15:30 — PR #3237: Huber loss (delta=1.0) to cap high-Re gradient outliers

**Winner**: edward (`charliepai2i24h3-edward/huber-loss`)

- **`val_avg/mae_surf_p` = 117.6594** (best epoch 13 / 14 run, still improving at timeout)
- **`test_avg/mae_surf_p` = NaN** (scoring.py NaN bug — one sample in test_geom_camber_cruise has inf in GT)
  - Clean estimate (3 finite test splits): ~107.6
- **Per-split val metrics (epoch 13 best checkpoint)**:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 147.77 | 1.620 | 0.855 |
| val_geom_camber_rc | 125.08 | 2.199 | 1.030 |
| val_geom_camber_cruise | 88.98 | 1.609 | 0.625 |
| val_re_rand | 108.81 | 1.983 | 0.813 |
| **val_avg** | **117.66** | 1.853 | 0.831 |

- **Change**: 2-line swap: `sq_err = F.huber_loss(pred, y_norm, reduction='none', delta=cfg.huber_delta)` + `huber_delta: float = 1.0` config field. All other hyperparameters unchanged.
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params)
- **Peak VRAM**: 42.11 GB
- **Metric artifacts**: `models/model-huber_loss_d1-20260515-130807/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name huber_loss_d1 --agent charliepai2i24h3-edward`

> **Note**: `test_avg/mae_surf_p` is NaN for all experiments due to a bug in `data/scoring.py` — sample 20 of `test_geom_camber_cruise` has `inf` in GT, which propagates through the masked sum (`inf * 0 = NaN`). All ranking uses `val_avg/mae_surf_p` until this is resolved. See EXPERIMENTS_LOG.md for details.

---

### 2026-05-15 22:45 — PR #3300: BF16 mixed-precision to get more epochs in 30-min budget

**Winner**: edward (`charliepai2i24h3-edward/bf16-mixed-precision`)

- **`val_avg/mae_surf_p` = 97.5474** (best epoch 17 / 19 run, hit 30-min cap)
- **`test_avg/mae_surf_p` = NaN** (scoring.py NaN bug — test_geom_camber_cruise/000020.pt inf in GT)
  - Clean estimate (3 finite test splits): **93.99** (vs ~107.6 prior clean estimate)
- **Per-split val metrics (epoch 17 best checkpoint)**:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 114.41 | 1.387 | 0.674 |
| val_geom_camber_rc | 104.96 | 2.060 | 0.851 |
| val_geom_camber_cruise | 79.72 | 1.135 | 0.531 |
| val_re_rand | 91.09 | 1.532 | 0.678 |
| **val_avg** | **97.55** | 1.529 | 0.684 |

- **Delta vs prior baseline**: −20.11 (−17.1%) on val_avg/mae_surf_p. All 4 splits improve.
- **Change**: Add `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around training forward+loss and validation forward. No GradScaler needed (BF16 has same exponent range as FP32).
- **Budget gain**: Reached epoch 19 vs 14 at baseline (+5 epochs in same 30-min cap). s/epoch: ~98 vs ~128 (1.3x throughput).
- **Peak VRAM**: 32.95 GB (vs 42.11 GB, −22%)
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params). No model changes.
- **Metric artifacts**: `models/model-bf16_huber-20260515-212744/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name bf16_huber --agent charliepai2i24h3-edward`
