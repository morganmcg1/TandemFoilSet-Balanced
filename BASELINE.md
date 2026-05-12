# SENPAI Baseline — `icml-appendix-willow-pai2g-48h-r2`

The current best result on this advisor branch. Every new PR's primary metric must beat the values in the most-recent entry below.

- **Primary ranking metric:** `val_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 val splits)
- **Paper-facing test metric:** `test_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 test splits, all 4 splits must be finite)
- **Direction:** lower is better

---

## 2026-05-12 21:06 — PR #1554: Stack SWA on Huber baseline

- **val_avg/mae_surf_p:** **99.0704** (SWA model, end of training)
- **test_avg/mae_surf_p:** **88.8955** (4-split, all finite, SWA model)
- Improvement vs. PR #1452: val −1.69%, test −1.65%

### Val per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| val_single_in_dist     | 117.7539 | −1.66% |
| val_geom_camber_rc     | 104.2288 | −4.71% |
| val_geom_camber_cruise | 79.1798  | −2.12% |
| val_re_rand            | 95.1191  | **+2.23%** |
| **val_avg**            | **99.0704** | **−1.69%** |

### Test per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| test_single_in_dist     | 102.3693 | −3.43% |
| test_geom_camber_rc     | 95.4730  | −0.81% |
| test_geom_camber_cruise | 67.6442  | −1.77% |
| test_re_rand            | 90.0956  | −0.35% |
| **test_avg**            | **88.8955** | **−1.65%** |

### Config

- Everything from PR #1452 baseline (Huber β=1.0, AdamW lr=5e-4 wd=1e-4, batch=4, surf_weight=10.0, CosineAnnealingLR(T_max=15), 15 epochs)
- **SWA additions:**
  - `swa_start_frac = 0.75` → `swa_start_epoch = 11` (0-indexed)
  - `swa_lr = 1e-4` (= 0.2 × base lr)
  - `swa_anneal_epochs = 2`, `anneal_strategy = "cos"`
  - `update_bn` skipped (Transolver uses LayerNorm)
  - Terminal test eval runs on `swa_model.module`, not the base model
  - 3 SWA-active epochs in practice (epochs 12, 13, 14; epoch 15 timed out)
- Params: 0.66M (SWA is a running average, no extra trained params)
- Peak VRAM: ~42 GB
- Wall clock: 30.8 min

### W&B run

- `cnu8v9i2` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/cnu8v9i2

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/swa-on-huber \
  --wandb_group swa-stack-test
```

### What landed

- `torch.optim.swa_utils.AveragedModel` + `SWALR` added in `train.py`. Cosine anneals epochs 0–10 (inclusive); SWALR holds `swa_lr=1e-4` epochs 11–14 while `swa_model.update_parameters(model)` accumulates the running mean. After the last epoch, `model.load_state_dict(swa_model.module.state_dict())` and re-evaluate val/test — these are the headline numbers.
- Per-split test improvements are uniform (all 4 splits down), consistent with the flat-minima-helps-OOD hypothesis. Val mix is positive on 3/4 splits with a small `val_re_rand` regression (+2.2%) — likely an artifact of only 3 averaged epochs and `swa_lr` being above the cosine floor.

### Open follow-ups (for future PRs)

- **Stack SWA × unified_pos × FiLM × Re-weight × β-sweep** — orthogonal levers; current wave-2 wave (#1551 tanjiro, #1585 askeladd, #1586 thorfinn) all stack on Huber baseline. The next merged winner should compound on this SWA-on-Huber baseline.
- **Tighter SWA tuning:** lower `swa_lr` (0.1× or 0.05× base lr) and/or earlier `swa_start_frac` (0.65) to fit 4–5 averaged epochs into the 14-epoch envelope. Predicted further −1 to −3% on val.
- **Same open follow-ups carry forward from PR #1452:** β sweep, surface-only Huber, per-channel β.

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
