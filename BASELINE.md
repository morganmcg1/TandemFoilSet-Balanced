# Baseline — icml-appendix-willow-pai2i-24h-r5

This launch starts fresh on `icml-appendix-willow-pai2i-24h-r5` (branched from `icml-appendix-willow`). No prior runs from this launch have been measured yet; the first round of experiments will both probe individual moves AND establish a confirmed baseline number.

## Current baseline configuration (head of advisor branch)

Model: **Transolver** (~1.5M params)
- `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `space_dim=2`, `fun_dim=22` (X_DIM=24 minus the 2 spatial dims)

Training:
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingWarmRestarts(T_0=5, T_mult=2)` with per-batch `scheduler.step()`
- Loss: vol_loss (MSE) + surf_weight × surf_loss (L1/MAE), `total = vol_loss + surf_weight * surf_loss`
- Grad clip: `clip_grad_norm_(max_norm=1.0)`
- No mixed precision, no grad clipping
- `epochs=50` (max), capped by `SENPAI_TIMEOUT_MINUTES=30`

Per-run limits enforced by the harness:
- `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap)
- `SENPAI_MAX_EPOCHS=50` (hard epoch cap)
- 1 GPU per student, 96 GB VRAM

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing metric is `test_avg/mae_surf_p`, computed at the end of training using the best-val checkpoint.

## Baseline metrics

### 2026-05-15 21:30 — PR #3434: L1 surface loss (vol MSE + surf L1)

**New best: `val_avg/mae_surf_p = 90.04`** — single arm, -8.84 pp (-8.94%) vs warm-restarts baseline 98.88

| Split | val mae_surf_p (run `tcci4fzk`) | Δ vs prior baseline |
|---|---|---|
| val_single_in_dist | 108.95 | −7.41 |
| val_geom_camber_rc | 97.70 | −10.70 |
| val_geom_camber_cruise | 70.40 | −7.51 |
| val_re_rand | 83.11 | −9.76 |
| **val_avg** | **90.04** | **−8.84** |
| test avg (3-split excl. cruise) | 87.78 | −7.04 |

- **W&B run:** `tcci4fzk`
- **Epochs:** 14 / 50 (30-min wall-clock cap; still improving at epoch 14 — last two val_avg: 98.80 → 90.04)
- **Peak VRAM:** 42.1 GB (unchanged from prior baseline)
- **Change vs prior baseline:** replaced `sq_err` for `surf_loss` with `abs_err = (pred - y_norm).abs()` — vol_loss remains MSE, surf_loss is now L1/MAE
- **Why it works:** L1 minimizer = conditional median, the MAE-optimal estimator. Grad clip (`max_norm=1.0`) already normalizes step sizes, so L1 vs L2 convergence speed is similar (L1 only 1 epoch slower at epoch 1, then led all the way). Largest gain on `val_geom_camber_rc` (-10.70 pp) — heavy-tailed OOD error distribution where L2 chases outliers, L1 distributes gradient more evenly.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-edward \
    --wandb_group willow-pai2i-24h-r5-round3 \
    --wandb_name edward-l1-surf-loss
  ```

---

### 2026-05-15 20:25 — PR #3320: CosineAnnealingWarmRestarts T_0=5 T_mult=2

**New best: `val_avg/mae_surf_p = 98.88`** — Replicated across 3 independent runs (mean 100.67, spread ~3 pp)

| Split | val mae_surf_p (best run `oeo67jf2`) | 3-run mean |
|---|---|---|
| val_single_in_dist | 116.36 | 118.50 |
| val_geom_camber_rc | 108.40 | 110.54 |
| val_geom_camber_cruise | 77.91 | 79.25 |
| val_re_rand | 92.87 | 94.39 |
| **val_avg** | **98.88** | **100.67** |

- **test_avg/mae_surf_p:** NaN (cruise test split bad sample #3292; 3-split avg excl. cruise ≈ **94.82** best run, **96.71** mean across 3 runs)
- **W&B runs:** `oeo67jf2` (★ best), `79m50be7`, `iyhrbvuq` — group `willow-pai2i-24h-r5-round2`
- **Epochs:** 14 / 50 (30 min wall-clock cap; warm-restart cycle boundaries at epochs 5, 10, 20)
- **Peak VRAM:** 42.1 GB (unchanged from baseline)
- **Change vs prior baseline:** replaced `CosineAnnealingLR(T_max=MAX_EPOCHS)` with `CosineAnnealingWarmRestarts(T_0=5, eta_min=0, T_mult=2)` and `scheduler.step()` after each batch (not epoch)
- **Why it works:** Restarts at epochs 5, 10, 20 within the 14-epoch budget give multiple escape-from-local-minima opportunities. Improvement and low variance are both consistent with warm restarts breaking out of early-convergence plateaus.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-nezuko \
    --wandb_group willow-pai2i-24h-r5-round2 \
    --wandb_name nezuko-warm-restarts
  ```

---

### 2026-05-15 15:40 — PR #3157: Grad clipping max_norm=1.0

**New best: `val_avg/mae_surf_p = 117.16`**

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 138.19 |
| val_geom_camber_rc | 137.91 |
| val_geom_camber_cruise | 85.86 |
| val_re_rand | 106.68 |
| **val_avg** | **117.16** |

- **test_avg/mae_surf_p:** NaN (cruise test split has a bad sample — see GH issue #3292; 3-split avg excl. cruise ≈ 116.40)
- **W&B run:** `cfp7lnaq`
- **Epochs:** 14 / 50 (hit 30 min wall-clock cap)
- **Peak VRAM:** 42.1 GB
- **Change vs head:** added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-tanjiro \
    --wandb_group willow-pai2i-24h-r5-round1 \
    --wandb_name tanjiro-gradclip-1p0
  ```
- **Note:** max_norm=1.0 fired on 100% of steps (median pre-clip grad norm ≈ 45.7). Effective LR is ~45× lower than the nominal 5e-4. Next experiment will probe max_norm=10 to allow spike-only clipping.
