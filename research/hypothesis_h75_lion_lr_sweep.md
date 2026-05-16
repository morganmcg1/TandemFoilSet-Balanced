## Hypothesis

**H75: Fine-grain Lion LR sweep around the H73 lr=3e-4 winner.**

H73 Arm A (lr=1e-4) lands at val=46.34. H73 Arm B (lr=3e-4) lands at val=42.98 — **3.36 pts better at 3× the LR**. The true Lion+slice=96 LR optimum may sit between 3e-4 and even higher; the H58/H73 step from 1e-4 → 3e-4 → ? is not yet bracketed above.

Two arms:

- **Arm A: lr=2e-4** — interpolates between H73 Arm A and Arm B. If the optimum is closer to 2e-4, this lands above 42.98.
- **Arm B: lr=5e-4** — explores higher; Lion's sign-update is famously LR-tolerant, and the wider slice=96 gradient surface may accommodate a higher LR. If 5e-4 trains stably, the model may reach a deeper floor in fewer epochs.

**Predicted:**
- Arm A: ~43-45 val_avg (likely between Arm A and Arm B of H73; could match H73 Arm B if LR sensitivity is flat in [2e-4, 3e-4])
- Arm B: ~41-44 val_avg (best case wins; worst case diverges or overshoots)

If Arm B diverges, that locks the upper bracket and tells us lr=3e-4 is near the ceiling. If Arm B wins, it confirms the gradient surface from slice=96 accommodates a higher LR than H58 found at slice=64.

**Risk:** Arm B at lr=5e-4 may cause early instability with Lion's sign update (no LR warmup). If val_avg at epoch 3 > 250, kill and report.

## Instructions

Both arms differ only in `--lr`. No code changes needed.

```bash
# Arm A — lr=2e-4
cd target/ && python train.py --epochs 50 \
  --experiment_name h75-arm-a-lion-lr2e4 \
  --agent charliepai2i48h3-tanjiro \
  --optimizer lion --lr 2e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — lr=5e-4
cd target/ && python train.py --epochs 50 \
  --experiment_name h75-arm-b-lion-lr5e4 \
  --agent charliepai2i48h3-tanjiro \
  --optimizer lion --lr 5e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- Per-epoch val_avg trajectory for both arms (vs. H73's reported trajectory)
- Best epoch, mean s/epoch, peak GPU memory
- Gradient norm and gate health (mean/std at epochs 5, 10, 15) — diagnostic for Lion stability at higher LR
- **LR sensitivity analysis:** combining with H73's data, plot val_avg vs lr (1e-4, 2e-4, 3e-4, 5e-4) — is the curve monotone or U-shaped?

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report (likely outcome for Arm B if 5e-4 is too aggressive).

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| val_single_in_dist/mae_surf_p | 43.7880 |
| val_geom_camber_rc/mae_surf_p | 56.6638 |
| val_geom_camber_cruise/mae_surf_p | 26.4930 |
| val_re_rand/mae_surf_p | 44.9686 |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + **lr=3e-4** + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15.

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
