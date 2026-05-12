# SENPAI Baseline — `icml-appendix-willow-pai2g-48h-r2`

The current best result on this advisor branch. Every new PR's primary metric must beat the values in the most-recent entry below.

- **Primary ranking metric:** `val_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 val splits)
- **Paper-facing test metric:** `test_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 test splits, all 4 splits must be finite)
- **Direction:** lower is better

---

## 2026-05-12 20:02 — PR #1452: Swap MSE → Smooth-L1 (Huber β=1.0) + scoring NaN-safe fix

- **val_avg/mae_surf_p:** **100.7659** (best, epoch 14)
- **test_avg/mae_surf_p:** **90.3840** (4-split, all finite — first finite 4-split test metric on this branch)

### Val per-split surface MAE (best epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist     | 119.7409 | 1.3652 | 0.7235 |
| val_geom_camber_rc     | 109.3817 | 2.1068 | 0.9464 |
| val_geom_camber_cruise | 80.8970  | 0.9151 | 0.5169 |
| val_re_rand            | 93.0438  | 1.5325 | 0.7294 |
| **val_avg**            | **100.7659** | 1.4799 | 0.7291 |

### Test per-split surface MAE (best checkpoint, epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist     | 106.0083 | 1.2943 | 0.6857 |
| test_geom_camber_rc     | 96.2512  | 2.0110 | 0.8876 |
| test_geom_camber_cruise | 68.8607  | 0.8739 | 0.4658 |
| test_re_rand            | 90.4157  | 1.3369 | 0.6955 |
| **test_avg**            | **90.3840** | 1.3790 | 0.6837 |

### Config

- Architecture: Transolver baseline (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, unified_pos=False)
- Loss: Smooth-L1 (Huber β=1.0) replaces MSE in both training and `evaluate_split`
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=epochs=15) — schedule-aligned to actual training budget
- Batch size: 4
- surf_weight: 10.0
- Epochs: 15 (cap triggered after epoch 14; epoch 15 not started)
- Wall clock: ~30 min (hit `SENPAI_TIMEOUT_MINUTES=30`)
- Params: 0.66M
- Peak VRAM: ~42 GB

### W&B run

- `lo8vp7rj` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/lo8vp7rj

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/smooth-l1-loss-e15 \
  --wandb_group huber-loss-sweep
```

### What landed

1. **Loss reformulation:** MSE → Smooth-L1 (β=1.0) in `train.py` training loop and `evaluate_split`. Metric in `data/scoring.py` (denormalized-space MAE) is unchanged. Hypothesis was that Huber would cap high-Re outlier gradients where MSE over-penalizes — pattern confirmed: `val_geom_camber_cruise` (80.90) and `val_re_rand` (93.04) are the two lowest val splits.
2. **`data/scoring.py` NaN-safe fix:** `accumulate_batch` was propagating `0 * inf = NaN` from the corrupt GT sample `test_geom_camber_cruise/000020.pt` (761 nodes with `-inf` in the `p` channel). Fix uses `torch.where(mask, err, zero)` to select-or-zero without arithmetic, so masked positions never see `inf`. Effect: previously NaN `test_avg/mae_surf_p` is now finite across all 4 test splits.

### Open follow-ups (for future PRs)

- β sweep over {0.1, 0.3, 1.0, 3.0} now that β=1.0 is the established baseline.
- Surface-only Huber + MSE on volume (surface is the headline metric; outlier dominance is plausibly concentrated near foils).
- Stacking with orthogonal levers (positional encoding, slice_num, surf_weight, capacity).
- Per-channel β (pressure has a wider normalized range than Ux/Uy).
