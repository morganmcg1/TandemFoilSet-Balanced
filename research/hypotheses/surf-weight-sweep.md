# Hypothesis: surf-weight-sweep (nezuko)

## Hypothesis
Sweep `surf_weight` values on the Huber baseline to find the optimal surface:volume
loss balance. Currently `surf_weight=10` was tuned for the MSE baseline. With Huber
loss, the effective gradient contribution from high-magnitude surface errors is capped
at δ (=2.0), which changes the functional gradient norm of the surface term. The
optimal balance between vol_loss and surf_loss may have shifted with Huber.

- `surf_weight=5`: reduce surface emphasis — reduces potential overfit on surface near the Huber elbow
- `surf_weight=20`: increase surface emphasis — amplifies the signal for the primary metric

Both arms run pure CLI change, no code modification.

**Predicted improvement:** −2 to −6 on val_avg/mae_surf_p vs 107.46, depending on
which direction the optimum lies.

## Instructions

No code changes needed. Run two arms via CLI flags:

### Arm 1: surf_weight=5 (reduce surface emphasis)
```bash
cd target/ && python train.py \
    --surf_weight 5.0 \
    --wandb_group surf-weight-sweep \
    --wandb_name surf-weight-5 \
    --agent willowpai2i24h3-nezuko
```

### Arm 2: surf_weight=20 (increase surface emphasis)
```bash
cd target/ && python train.py \
    --surf_weight 20.0 \
    --wandb_group surf-weight-sweep \
    --wandb_name surf-weight-20 \
    --agent willowpai2i24h3-nezuko
```

Compare both arms against the baseline (surf_weight=10) and report all 4 val splits
for the best run. Include per-split val metrics in the terminal SENPAI-RESULT to help
understand whether surface emphasis affects specific difficulty levels differently.

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0, surf_weight=10)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
