# BASELINE — icml-appendix-charlie-pai2d-r3

The reference baseline is the unmodified `train.py` Transolver configuration.
This advisor branch starts round 3 with no measured baseline yet — the first
round 3 experiments will establish concrete reference numbers for `val_avg/mae_surf_p`
and `test_avg/mae_surf_p`, after which this file will be updated with the best
measured metrics.

## Reference configuration

Defaults from `train.py`:

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` | 2 |
| `fun_dim` | 22 (X_DIM - 2) |
| `out_dim` | 3 (Ux, Uy, p) |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, MSE in normalized space |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Epochs | 50 |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Reproduce the baseline

```bash
cd target/
python train.py --epochs 50 --experiment_name baseline_ref
```

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)

## Round 3 measured baseline

TBD — populated when the first round 3 experiment with comparable knobs reports
metrics. Until then, students should:

1. Report `val_avg/mae_surf_p` per epoch in `models/<experiment>/metrics.jsonl`.
2. Run final test evaluation (do not pass `--skip_test`) so `test_avg/mae_surf_p`
   is recorded.
3. Compare against peers in this round — the best `val_avg/mae_surf_p` becomes
   the new baseline once that PR merges.
