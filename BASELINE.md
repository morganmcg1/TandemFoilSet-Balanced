# Baseline — `icml-appendix-willow-pai2i-48h-r4`

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better). For paper-facing comparison use `test_avg/mae_surf_p` (NaN until scoring bug fixed — see note below).

---

## 2026-05-15 14:38 — PR #3091: LR warmup + gradient clipping + lr=1e-3

- **val_avg/mae_surf_p: 109.4166** (best epoch 14/15, W&B run `0ez1sqmi`)
- **test_avg/mae_surf_p: NaN** — scoring bug in `data/scoring.py`: `0.0 * NaN = NaN` on accumulation of 1 non-finite cruise test sample; 3-split workaround = **107.4694** (excl. `test_geom_camber_cruise`)

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 119.58 | 111.04 |
| geom_camber_rc | 119.40 | 110.20 |
| geom_camber_cruise | 88.57 | **NaN (bug)** |
| re_rand | 110.12 | 101.17 |
| **avg** | **109.42** | — |

- **Model config:** Transolver `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~3M params)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs (0.1→1.0), then cosine to 0 over remaining epochs
- **Grad clip:** max_norm=1.0 (logs `train/grad_norm`)
- **Batch:** 4, surf_weight=10.0, MSE loss
- **Budget:** 30-min wall clock → 15 epochs of configured 50 (T_max=50, so cosine barely anneals — see note)

**Reproduce command:**
```bash
cd target/ && python train.py --epochs 50 --lr 1e-3 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```

> **Note on cosine schedule:** `T_max=50` with 30-min budget means only ~14-15 epochs run and LR anneals only ~15% of its range. Experiments targeting proper LR annealing should pass `--epochs 10` (or whatever matches the realistic epoch count) to get T_max=10 so cosine fully decays within the budget.

> **Note on test_geom_camber_cruise NaN:** `data/scoring.py`'s `accumulate_batch` computes `err * surf_mask` where `err` includes NaN from non-finite ground truth at 1 sample; `0 * NaN = NaN` propagates to the sum. Fix is `err.nan_to_num(0.0)` in `accumulate_batch` or clamping `y` before passing to it. This is tracked in edward's follow-up PR. Until fixed, compare on `val_avg/mae_surf_p` (clean) and 3-split test workaround.
