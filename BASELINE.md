# Baseline — icml-appendix-charlie-pai2g-48h-r3

No prior experiments have been run on this advisor branch yet. The reference
configuration is the stock `train.py` on this branch:

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | MSE in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution (`SENPAI_TIMEOUT_MINUTES=30`) |

**Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE
across the four val splits). Lower is better. Test metric: `test_avg/mae_surf_p`
computed from the best-val checkpoint.

**Best known metric on this branch:** none yet — first round of experiments is
in flight (PRs assigned 2026-05-12).
