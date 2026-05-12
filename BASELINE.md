# BASELINE — icml-appendix-charlie-pai2g-24h-r2

Fresh research track. No prior experiments merged on this branch yet.

## Reference baseline (target/train.py defaults)

The first winning PR establishes the empirical baseline. Until then, the **reference baseline** for hypothesis comparison is the unmodified Transolver in `train.py`:

| Field | Value |
|---|---|
| Model | Transolver |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| params | ~1.4M |
| optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs (config) | 50 |
| timeout (env) | SENPAI_TIMEOUT_MINUTES=30 |
| scheduler | CosineAnnealingLR(T_max=MAX_EPOCHS) |
| sampler | WeightedRandomSampler (domain-balanced) |
| loss | MSE in normalized space: vol_loss + surf_weight * surf_loss |
| precision | fp32 |
| amp | none |
| grad_clip | none |

Primary metric: `val_avg/mae_surf_p` (equal-weight surface pressure MAE across 4 val splits). Test-time metric for paper-facing comparisons: `test_avg/mae_surf_p` from the best validation checkpoint.

No measured value on this branch yet. The first hypothesis that produces terminal `SENPAI-RESULT` with a primary metric value sets the floor.

## Update protocol

When a PR is merged and beats the current floor, replace the value here with `{val_metric, test_metric, pr_number}`.
</content>
</invoke>