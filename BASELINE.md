# Baseline — willow-pai2d-r1

This is the first round on `icml-appendix-willow-pai2d-r1`. No experiment runs
have been recorded on this branch yet, so the *de facto* baseline is the
default Transolver configuration in `train.py`:

## Default config

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- Loss: per-channel-equal MSE in normalized space, with surface vs. volume
  split via `surf_weight`
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22`, `out_dim=3`)

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --wandb_group baseline-willow-r1
```

## Primary ranking metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four
validation splits. Test-time number reported alongside is `test_avg/mae_surf_p`.

## Round-1 baseline reference run

Round-1 includes one explicit baseline run (assigned to `alphonse`) so that
later interventions have a clean reference to compare against. Once that run
completes, this file will be updated with the measured numbers and W&B run
id, and any improvements over it will be merged into the advisor branch.
