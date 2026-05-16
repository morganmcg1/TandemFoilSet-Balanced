## Hypothesis

**H77: Test n_head=4 (and n_head=3) on top of H73 baseline (Lion + slice_num=96).**

H70 (just closed, slice=64 + Lion + RMSNorm) found n_head=4 beats n_head=2 by **1.1 pts val_avg** (H59 baseline 56.9 → n_head=2 45.66; n_head=4 was a similar baseline = 45.56). H73 used n_head=2 and won at 42.98. The n_head=4 lever has NOT been tested at slice_num=96.

Mechanistic prior: more attention heads = finer-grained query/key subspaces = more diverse spatial filtering. With slice_num=96 (50% wider attention bottleneck), the per-head dim is currently n_hidden / n_head = 128 / 2 = 64; with n_head=4, per-head dim drops to 32. The fundamental trade-off is **head count vs. per-head expressiveness**.

Two arms:

- **Arm A: n_head=4** — direct test of H70's winning lever at H73 baseline. Per-head dim = 32.
- **Arm B: n_head=3** — intermediate; tests if the optimum is at 3 (per-head dim = ~43) or further. n_head=3 may not divide n_hidden=128 evenly — student should verify train.py allows it or report.

**Predicted:**
- Arm A: ~41-43 val_avg (1-2 pt improvement if H70 signal transfers; small risk it doesn't because slice=96 already provides spatial selectivity that competed with multi-head)
- Arm B: ~42-44 val_avg (intermediate; may interpolate between 2 and 4)

**Risk:** If slice_num=96 already saturates the spatial-selectivity capacity at n_head=2, adding more heads may regress. H70's win at n_head=4 was specifically at slice=64 — the win may not transfer.

## Instructions

Both arms differ only in `--n_head`. No code changes expected.

```bash
# Arm A — n_head=4
cd target/ && python train.py --epochs 50 \
  --experiment_name h77-arm-a-nhead4 \
  --agent charliepai2i48h3-frieren \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 4 --clip_grad_norm 1.0

# Arm B — n_head=3
cd target/ && python train.py --epochs 50 \
  --experiment_name h77-arm-b-nhead3 \
  --agent charliepai2i48h3-frieren \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 3 --clip_grad_norm 1.0
```

If `--n_head 3` is rejected (e.g. n_hidden=128 must be divisible by n_head), report and replace Arm B with `--n_head 8` (per-head dim = 16) instead.

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- n_params for both arms (n_head affects parameter count for the per-head Q/K/V projections)
- Mean s/epoch (different n_head changes attention compute) and peak GPU memory
- Per-epoch val_avg trajectory
- **n_head sensitivity:** combine with H73's n_head=2 result to plot val_avg vs n_head (2, 3, 4)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 + n_head=2 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + **n_head=2** + clip_grad_norm=1.0 + LayerNorm + T_max=15. n_params=864,907.

H70 reference (just closed, slice=64+RMSNorm+Lion lr=1e-4): n_head=4 wins by 1.1 pts over n_head=1.

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
