# Baseline metrics (willow-pai2g-24h-r4)

**Branch:** `icml-appendix-willow-pai2g-24h-r4`
**Run cap:** `SENPAI_TIMEOUT_MINUTES=30` per training run, hard.

## Baseline config (`train.py` defaults)

| | |
|---|---|
| Optimizer | AdamW |
| LR | 5e-4 |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Epochs | 50 (capped at 30 min wall) |
| Scheduler | CosineAnnealingLR(T_max=epochs) |
| Loss | MSE, `vol_loss + 10 * surf_loss` |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |

## Primary ranking metric

`test_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four test splits. **Lower is better.**

Validation analogue (used for checkpoint selection): `val_avg/mae_surf_p`.

## Current best

### 2026-05-12 20:XX — PR #1415: bf16 mixed precision + grad_clip (on top of slice_num=128 + scoring fix)

- **val_avg/mae_surf_p:** **98.7664** (best epoch 18 of 18 completed) ✓ NEW BASELINE
- **test_avg/mae_surf_p:** NaN at submit-time (bf16-induced `inf` pred on one cruise test node). After PR #1521 scoring fix merged: future runs/re-evals should report finite test (posinf zeroed in `nan_to_num`).
- **Test 3-split mean (excl. cruise):** 97.12
- **Per-split val surface MAE (best epoch 18):**
  - `val_single_in_dist`: p=108.76, Ux=1.67, Uy=0.68
  - `val_geom_camber_rc`: p=115.38, Ux=2.31, Uy=0.90
  - `val_geom_camber_cruise`: p=78.21, Ux=0.98, Uy=0.53
  - `val_re_rand`: p=92.71, Ux=1.61, Uy=0.72
- **Per-split test (raw):** test_single_in_dist=98.36, test_geom_camber_rc=104.62, test_re_rand=88.39, test_geom_camber_cruise=NaN (bf16 inf)
- **W&B run:** `ojdeyn8r`
- **Peak GPU:** 32.9 GB | **Sec/epoch:** ~99s | **Epochs:** 18/50 (30-min cap, still descending)
- **Model diff vs prior baseline (slice_num=128 + scoring fix):**
  - bf16 autocast in train forward + grad_clip_norm=1.0
  - bf16 autocast in eval forward (suspected source of cruise inf — future work should test fp32 eval)
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-thorfinn-bf16 --agent willowpai2g24h4-thorfinn
  ```

**Note on test_avg:** The bf16 eval autocast caused one `pred` node to overflow on `test_geom_camber_cruise`. The merged PR #1521 scoring fix now zeros that out, but reduces the affected channel's MAE slightly (the overflowing node now contributes 0 instead of being properly skipped). Follow-up to switch eval to fp32 is the natural next step.

**Next target:** beat val_avg/mae_surf_p = 98.7664

---

### Previous baselines

#### 2026-05-12 19:XX — PR #1396: Double Transolver slice tokens (slice_num 64 → 128)

- **val_avg/mae_surf_p:** 146.2510 (epoch 9 of 11 completed)
- **test_avg/mae_surf_p:** NaN ⚠️ — GT NaN in `test_geom_camber_cruise` sample 20 leaks through `err * mask` in `data/scoring.py:49`. Bug-fix PR pending; val number is valid.
- **Per-split val surface MAE (best epoch 9):**
  - `val_single_in_dist`: p=175.68, Ux=—, Uy=—
  - `val_geom_camber_rc`: p=158.18
  - `val_geom_camber_cruise`: p=115.62
  - `val_re_rand`: p=135.53
- **Test (3-split excl. cruise):** 147.07
- **W&B run:** `5qh8pj8v`
- **Peak GPU:** 54.5 GB | **Sec/epoch:** ~172s | **Epochs:** 11/50 (30-min cap)
- **Model diff vs original baseline:** `slice_num=128` (was 64); all other config unchanged.
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-frieren-slice128 --agent willowpai2g24h4-frieren
  ```

_(Previous baseline — superseded by #1415)_
