# TandemFoilSet Baseline — branch `icml-appendix-charlie-pai2i-24h-r3`

This branch is a fresh launch in the Charlie local-metrics arm. No experiments have been merged yet on this branch.

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

_None yet._ First winning PR will populate this section.
