## Hypothesis

**H78: Test β₂=0.999 (slower EMA) on top of H73 baseline (Lion + slice_num=96).**

H68 (just closed, slice=64 + Lion + RMSNorm + lr=1e-4) found β₂=0.999 beats β₂=0.95 by **3 pts val_avg**. H73 used β₂=0.99 (default H58 spec). The H68 signal suggests slower EMA forgetting helps Lion.

Mechanistic prior: Lion's β₂ controls the slow exponential moving average of gradients used to compute momentum. Slower β₂ (closer to 1) means longer-horizon gradient memory → smoother optimization trajectory. With slice_num=96's wider gradient surface, more averaging may further stabilize updates.

Two arms:

- **Arm A: β₂=0.999** — direct port of H68's winning lever to H73 baseline.
- **Arm B: β₂=0.995** — interpolates between H73's 0.99 and H68's 0.999. May find the sweet spot.

**Predicted:**
- Arm A: ~41-43 val_avg (1-2 pt improvement if H68 signal transfers)
- Arm B: ~42-43 val_avg (intermediate; less aggressive change from H73)

**Risk:** β₂=0.999 with very few epochs (15 wall-cut) may not benefit from longer EMA horizon — there isn't enough training time for the slower EMA to "warm up" its averaged gradient. If Arm A regresses by >1 pt, that's evidence the budget is too short.

## Instructions

Both arms differ only in `--beta2`. No code changes expected.

```bash
# Arm A — β₂=0.999
cd target/ && python train.py --epochs 50 \
  --experiment_name h78-arm-a-beta2-0999 \
  --agent charliepai2i48h3-thorfinn \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.999 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — β₂=0.995
cd target/ && python train.py --epochs 50 \
  --experiment_name h78-arm-b-beta2-0995 \
  --agent charliepai2i48h3-thorfinn \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.995 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- Per-epoch val_avg trajectory — does the slower EMA delay the loss descent in early epochs (because the EMA is unfilled)?
- Best epoch, mean s/epoch, peak GPU memory
- **β₂ sensitivity:** combine with H73 (β₂=0.99) to plot val_avg vs β₂ (0.99, 0.995, 0.999)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 + β₂=0.99 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + **β=(0.9, 0.99)** + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15.

H68 reference (just closed, slice=64+RMSNorm+Lion lr=1e-4): β₂=0.999 wins by 3 pts over β₂=0.95.

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
