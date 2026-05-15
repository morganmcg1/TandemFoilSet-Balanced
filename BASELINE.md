# BASELINE — icml-appendix-charlie-pai2i-24h-r1

Fresh advisor track for the Charlie local-metrics arm (24h training budget per run).
No measured baseline metrics on this branch yet — the first terminal `SENPAI-RESULT`
that ships will fix the reference.

## Reference configuration (Transolver, current `train.py`)

| Component | Value |
|---|---|
| Architecture | Transolver with physics-aware attention |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| Optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs (default) | 50 |
| schedule | CosineAnnealingLR (T_max=epochs) |
| sampler | WeightedRandomSampler (3-domain balanced) |
| loss | MSE on normalized targets, `vol_loss + surf_weight * surf_loss` |

## Metrics contract

- **Primary ranking (val)**: `val_avg/mae_surf_p` — equal-weight surface pressure MAE across the four val splits.
- **Paper-facing (test)**: `test_avg/mae_surf_p` — same aggregation over the four test splits, computed from the best-val checkpoint.
- **Per-split diagnostics**: `{split}/mae_{surf,vol}_{Ux,Uy,p}` and `{split}/{vol,surf,total}_loss`.
- Direction: **lower is better**.

## How students should report

Students commit `models/<experiment>/metrics.jsonl` and a terminal `SENPAI-RESULT`
marker in the PR with both `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## Compounding

Once a PR beats baseline and is merged into `icml-appendix-charlie-pai2i-24h-r1`,
update this file with the new best `val_avg/mae_surf_p` / `test_avg/mae_surf_p`
and PR number; subsequent assignments will use the updated reference.
