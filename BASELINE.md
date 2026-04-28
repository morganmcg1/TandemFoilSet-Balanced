# BASELINE — TandemFoilSet (icml-appendix-willow-pai2e-r1)

Track: `icml-appendix-willow-pai2e-r1`. Round 1 — first results in.

## Implicit baseline

The current implicit baseline is the **unmodified `train.py`** with default config:

- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- AdamW: `lr=5e-4, weight_decay=1e-4`, `CosineAnnealingLR(T_max=epochs)`
- Training: `batch_size=4, surf_weight=10, epochs=50`, MSE loss in normalized space
- Sampler: `WeightedRandomSampler` balancing 3 train domains
- 24-dim node features, 3-dim outputs `(Ux, Uy, p)`
- Reproduce: `python train.py --agent baseline --wandb_name baseline --wandb_group baseline`

## Primary metric

`val_avg/mae_surf_p` for checkpoint selection; `test_avg/mae_surf_p` for paper-facing comparison.
Both are equal-weight means of surface-pressure MAE across the four splits, computed in original
denormalized target space.

## Best so far

| PR   | W&B run    | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes                                |
|------|------------|---------------------|---------------------|--------------------------------------|
| **#773** | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | **119.35** | **108.79** | EMA decay=0.99, epoch 13, **MERGED ✓** |

**Note:** A clean unmodified-baseline run is still in flight (PR #846, willowpai2e1-edward). The EMA run beats the un-averaged live model at the same epoch (124.15) by 3.9% on val_avg.

## Per-split test metrics (current best — PR #773, EMA decay=0.99)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | 122.60         |
| test_geom_camber_rc        | 121.49         |
| test_geom_camber_cruise    |  81.38         |
| test_re_rand               | 109.69         |

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-fern \
    --wandb_group ema-decay-sweep --wandb_name ema-decay0.99 \
    --ema_decay 0.99
```
