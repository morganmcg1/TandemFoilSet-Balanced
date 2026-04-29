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
| **#881** | [jej4y8gt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jej4y8gt) | **85.23** | **76.64** | Huber δ=0.1 + EMA=0.99, no clip/warmup, epoch 14, **MERGED ✓** |
| #775 | [h22uwyy3](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/h22uwyy3) | 96.54 | 85.33 | warmup=0 + clip=0.5 + Huber δ=0.5 + EMA=0.99, **MERGED ✓** |
| #769 | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) | 102.86 | 94.83 | Huber δ=0.5, no clip, no EMA, **MERGED ✓** |
| #773 | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | 119.35 | 108.79 | EMA decay=0.99, no Huber, **MERGED ✓** |
| #846 (ref) | [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | Unmodified default @ 14 ep — canonical reference |

**Accumulated gains vs unmodified default (140.95/128.32):**
- EMA alone (PR #773): −15.4% val / −15.3% test
- Huber δ=0.5 alone (PR #769): −27.0% val / −26.2% test
- clip=0.5 + warmup=0 + Huber δ=0.5 + EMA=0.99 (PR #775): −31.5% val / −33.5% test
- **Huber δ=0.1 + EMA=0.99 (PR #881): −39.5% val / −40.3% test** — *new best*

**Note on δ=0.1 vs full 4-way stack:** PR #881 (δ=0.1 + EMA, no clip) beats PR #775 (δ=0.5 + EMA + clip + warmup=0) by 11.7% val. This does NOT mean clip/warmup hurt — it means δ=0.1 is the dominant lever. Whether δ=0.1 + clip + warmup=0 + EMA (5-way stack) improves further is an open question assigned to alphonse (PR in flight).

## Per-split test metrics (current best — PR #881, Huber δ=0.1 + EMA=0.99)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | 94.31          |
| test_geom_camber_rc        | 86.19          |
| test_geom_camber_cruise    | **53.08**      |
| test_re_rand               | 72.97          |

Biggest gain: cruise −30.3% vs Huber-δ=0.5-alone (76.12→53.08).

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-alphonse \
    --wandb_group huber-ema-stack --wandb_name huber0.1-ema0.99 \
    --huber_delta 0.1 --ema_decay 0.99
```

**Minimum required flags for all future experiments:**
```
--huber_delta 0.1 --ema_decay 0.99
```
Whether clip+warmup=0 helps on top of δ=0.1 is under active investigation (5-way stack test in flight).
