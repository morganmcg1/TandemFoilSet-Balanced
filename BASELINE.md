# Current Baseline — `icml-appendix-willow-pai2e-r4`

**Source:** untrained — round 1 launching from a stock Transolver baseline.

## Configuration (stock `train.py`)

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| Loss | MSE on normalized space, vol + surf_weight × surf |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | _pending round 1_ | — | — |
| `test_avg/mae_surf_p` | _pending round 1_ | — | — |

The first round produces a baseline reading. The advisor will update this file once round 1 has merged its first improvement.
