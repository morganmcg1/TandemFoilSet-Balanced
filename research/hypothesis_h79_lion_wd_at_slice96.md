## Hypothesis

**H79: Retune Lion wd at the H73 baseline (Lion lr=3e-4 + slice_num=96).**

H71 (just closed, slice=64 + Lion + RMSNorm + lr=1e-4) found wd=1e-4 beats wd=5e-4 by **4 pts val_avg**. H73 used wd=1e-3 (the H58 spec, 10× larger than H71's winner) and still won at 42.98. The interaction between wd and slice_num is unclear:
- Lower wd allows weights to grow → may help under-fit-from-budget (which we are)
- Higher wd regularizes → may help overfit (which we may not be at val=42.98)

Two arms:

- **Arm A: wd=1e-4** — direct port of H71's winning lever to H73 baseline. 10× lower than H73's wd=1e-3.
- **Arm B: wd=5e-5** — even lower (matches the original AdamW H38 winning wd). Tests whether the slice=96 + Lion regime needs less regularization than H73's default.

**Predicted:**
- Arm A: ~41-44 val_avg (uncertain — Lion's decoupled wd scales differently than AdamW, and wd=1e-3 was specifically tuned with Lion at slice=64; lowering may help or hurt)
- Arm B: ~41-44 val_avg (similar; explores the lower bound)

**Risk:** With wall-cut at ep 15/50, the model is likely still in the under-fit regime — so lowering wd may help by allowing more weight magnitude. But H58 at slice=64 also wall-cut and used wd=1e-3 (matching H73), so the H58 → H73 transfer is the established pattern.

## Instructions

Both arms differ only in `--weight_decay`. No code changes expected.

```bash
# Arm A — wd=1e-4
cd target/ && python train.py --epochs 50 \
  --experiment_name h79-arm-a-wd1e4 \
  --agent charliepai2i48h3-nezuko \
  --optimizer lion --lr 3e-4 --weight_decay 1e-4 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — wd=5e-5
cd target/ && python train.py --epochs 50 \
  --experiment_name h79-arm-b-wd5e5 \
  --agent charliepai2i48h3-nezuko \
  --optimizer lion --lr 3e-4 --weight_decay 5e-5 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- Weight magnitude metrics if available (L2 norm of parameter tensors) at end of training — verify lower wd → larger weights
- Best epoch, mean s/epoch, peak GPU memory
- Per-epoch val_avg trajectory
- **wd sensitivity:** combine with H73 (wd=1e-3) to plot val_avg vs wd (5e-5, 1e-4, 1e-3)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 + wd=1e-3 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + lr=3e-4 + **wd=1e-3** + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15.

H71 reference (just closed, slice=64+RMSNorm+Lion lr=1e-4): wd=1e-4 wins by 4 pts over wd=5e-4. (But H73 won at wd=1e-3 — different baseline, different lr.)

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
