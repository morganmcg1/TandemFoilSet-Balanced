# Hypothesis: huber-delta-tuning (tanjiro)

## Hypothesis
Sweep Huber delta values around the current default (δ=2.0) to find the optimal
gradient clipping threshold. The δ parameter controls the transition point between
quadratic (MSE-like, for small errors) and linear (MAE-like, for large errors) behavior.

- **δ=0.5**: very tight clipping — almost L1 everywhere. Risk: underfits fine-scale
  low-Re structure that needs quadratic precision.
- **δ=1.0**: moderate clip — frieren's own suggestion (expected best based on
  the high-Re tail hypothesis).
- **δ=3.0**: looser — closer to MSE than baseline δ=2.0. Useful if we're currently
  over-clipping and the 2.0 threshold is too aggressive.

All arms are pure CLI changes: `--huber_delta <value>`. No code modifications.

**Predicted improvement:** δ=1.0 predicted to improve by −2 to −6 vs 107.46.

## Instructions

No code changes needed. Run three arms via CLI:

### Arm 1: δ=1.0 (tighter, most promising per round-3 frieren suggestion)
```bash
cd target/ && python train.py \
    --huber_delta 1.0 \
    --wandb_group huber-delta-tuning \
    --wandb_name huber-delta-1p0 \
    --agent willowpai2i24h3-tanjiro
```

### Arm 2: δ=0.5 (very tight)
```bash
cd target/ && python train.py \
    --huber_delta 0.5 \
    --wandb_group huber-delta-tuning \
    --wandb_name huber-delta-0p5 \
    --agent willowpai2i24h3-tanjiro
```

### Arm 3: δ=3.0 (looser)
```bash
cd target/ && python train.py \
    --huber_delta 3.0 \
    --wandb_group huber-delta-tuning \
    --wandb_name huber-delta-3p0 \
    --agent willowpai2i24h3-tanjiro
```

Run all three. Report the full per-split val table for the best arm and include all
run IDs in the terminal SENPAI-RESULT. We're building a sensitivity curve for δ — even
if none beat baseline, the curve is valuable for design of future experiments.

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
