# In-track baseline — `icml-appendix-willow-pai2d-r3`

Lower is better on **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four val splits) — this is the primary ranking metric. Paper-facing number is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-28 00:30 — PR #320: Linear warmup + higher peak LR (5e-4 → 1e-3, 2-epoch warmup)

- **Best val avg surface MAE:** `val_avg/mae_surf_p = 115.8379` (epoch 14)
- **Per-split val MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `val_single_in_dist` | 131.0594 |
  | `val_geom_camber_rc` | 129.5697 |
  | `val_geom_camber_cruise` | 92.5489 |
  | `val_re_rand` | 110.1734 |
  | **val_avg** | **115.8379** |

- **Per-split test MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `test_single_in_dist` | 111.75 |
  | `test_geom_camber_rc` | 117.86 |
  | `test_geom_camber_cruise` | **NaN** (pre-existing bug — model emits non-finite predictions on at least one sample) |
  | `test_re_rand` | 108.73 |
  | **test_avg** | NaN (because of `test_geom_camber_cruise`) |
  | mean of 3 valid test splits | 112.78 |

- **W&B run:** `w3mjq2ua` in group `lr-warmup-sweep` (project `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`)
- **Reproduce:**
  ```bash
  cd target/
  python train.py --peak_lr 1e-3 --warmup_epochs 2 --epochs 50 \
      --wandb_group lr-warmup-sweep --wandb_name lr-1e-3-w2-r1 \
      --agent willowpai2d3-nezuko
  ```
- **Notes:**
  - All Round-1 sweep runs hit the 30-min `SENPAI_TIMEOUT_MINUTES` at ~epoch 14 of 50; cosine never fully annealed. Comparisons across PRs in this round are at this same truncated budget.
  - The test_avg NaN is due to non-finite predictions on at least one sample in `test_geom_camber_cruise` — same NaN at all three peak_lr values, so it's lr-independent. **Open issue** — flagged for follow-up.
  - Hyperparameter snapshot at this baseline: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`.
