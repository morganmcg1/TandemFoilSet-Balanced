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
| **#862** | [jsat9zk5](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jsat9zk5) | **82.64** | **73.02** | slice=32 + 4-way stack (δ=0.5+EMA+clip+w0), epoch 16, **MERGED ✓** |
| #881 | [jej4y8gt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jej4y8gt) | 85.23 | 76.64 | Huber δ=0.1 + EMA=0.99, no clip/warmup, **MERGED ✓** |
| #775 | [h22uwyy3](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/h22uwyy3) | 96.54 | 85.33 | warmup=0 + clip=0.5 + Huber δ=0.5 + EMA=0.99, **MERGED ✓** |
| #769 | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) | 102.86 | 94.83 | Huber δ=0.5, no clip, no EMA, **MERGED ✓** |
| #773 | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | 119.35 | 108.79 | EMA decay=0.99, no Huber, **MERGED ✓** |
| #846 (ref) | [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | Unmodified default @ 14 ep — canonical reference |

**Accumulated gains vs unmodified default (140.95/128.32):**
- EMA alone (PR #773): −15.4% val / −15.3% test
- Huber δ=0.5 alone (PR #769): −27.0% val / −26.2% test
- clip=0.5 + warmup=0 + Huber δ=0.5 + EMA=0.99 (PR #775): −31.5% val / −33.5% test
- Huber δ=0.1 + EMA=0.99 (PR #881): −39.5% val / −40.3% test
- **slice=32 + 4-way stack (PR #862): −41.4% val / −43.1% test** — *new best*

**Note on parallel branches:** PR #862 (val=82.64) is on δ=0.5 + clip + EMA + slice=32. PR #881 (val=85.23) is on δ=0.1 + EMA only (no clip, no slice change). The slice=32 + δ=0.1 combination is untested — frieren's follow-up will probe it.

## Per-split test metrics (current best — PR #862, slice=32 + 4-way stack)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | 87.34          |
| test_geom_camber_rc        | 83.28          |
| test_geom_camber_cruise    | **50.90**      |
| test_re_rand               | 70.56          |

Biggest gain: cruise 50.90 (vs 53.08 PR #881, vs 76.12 Huber-δ=0.5-alone).

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-frieren \
    --wandb_group slice-scan-v2 --wandb_name slice32-fullstack \
    --warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99 \
    --slice_num 32
```

**Minimum required flags for all future experiments:**
```
--huber_delta 0.1 --ema_decay 0.99    (or --huber_delta 0.5 --clip_norm 0.5 --warmup_epochs 0 --ema_decay 0.99 if slice=32)
--slice_num 32                        (clear architectural win on full stack)
```

**Two parallel best stacks:**
1. **PR #862:** slice=32 + δ=0.5 + clip=0.5 + warmup=0 + EMA=0.99 → val=82.64 / test=73.02
2. **PR #881:** slice=64 + δ=0.1 + EMA=0.99 (no clip/warmup) → val=85.23 / test=76.64

The fully-merged combination (slice=32 + δ=0.1 + ...) is being tested next.
