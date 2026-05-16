## Hypothesis

**H81: Retest RMSNorm under Lion + slice_num=96 on the H73 baseline.**

H72 found that RMSNorm + slice_num=96 **anti-compounds under AdamW** (+1.58 pts regression vs predicted −0.35, val=57.58 vs H66's 56.75). However, H73 uses **Lion** + LayerNorm at slice=96 and wins at val=42.98. The H72 anti-compound was specifically under AdamW — and the mechanistic explanation is that H59's RMSNorm win at slice=64 was partially from kernel-op speedup giving more steps per epoch, not purely from normalization quality.

Under Lion, the optimization dynamics are fundamentally different (sign-update removes gradient magnitude, so per-parameter step sizes are set by LR directly, not by gradient scaling). This changes the normalization interaction:
- **AdamW + LayerNorm**: standard path — bias/scale absorb gradient magnitude variation
- **Lion + LayerNorm**: sign-update already removes magnitude; LayerNorm bias/scale may be redundant weight
- **Lion + RMSNorm**: sign-update + no learned bias = cleaner gradient signal at the input to each layer

Two arms:

- **Arm A: RMSNorm + lr=3e-4** — direct swap of LayerNorm → RMSNorm on H73 baseline.
- **Arm B: RMSNorm + lr=2e-4** — lower LR hedge; if RMSNorm changes the effective gradient scale, the optimizer may prefer a slightly lower step size.

**Predicted:**
- Arm A: ~41-44 val_avg (uncertain — could win if Lion+RMSNorm compound, or tie/regress if H72's anti-compound mechanism also applies under Lion)
- Arm B: ~42-45 val_avg (hedging LR)
- **If Arm A < 42.98**: RMSNorm helps under Lion too. This is the new lever to stack.
- **If Arm A ≈ 43-44**: Normalization is neutral under Lion; LayerNorm preference is genuine.
- **If Arm A > 44**: Anti-compound confirmed under Lion as well.

**Risk:** Anti-compound may persist even under Lion (H72's mechanism was not fully isolated to AdamW). This is a hypothesis, not a certainty. Even a neutral result (same as H73) is informative — it closes the normalization question.

## Instructions

Arm A and B differ only in `--norm_type` and `--lr`. No code changes expected (RMSNorm and LayerNorm are both in the codebase from H59/H72).

```bash
# Arm A — RMSNorm at lr=3e-4 (H73 baseline + norm_type swap)
cd target/ && python train.py --epochs 50 \
  --experiment_name h81-arm-a-rmsnorm-lr3e4 \
  --agent charliepai2i48h3-alphonse \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --norm_type rmsnorm

# Arm B — RMSNorm at lr=2e-4 (hedged LR)
cd target/ && python train.py --epochs 50 \
  --experiment_name h81-arm-b-rmsnorm-lr2e4 \
  --agent charliepai2i48h3-alphonse \
  --optimizer lion --lr 2e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --norm_type rmsnorm
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- Per-epoch val_avg trajectory — compare to H73's baseline trajectory
- Mean s/epoch (RMSNorm removes bias computation — slight speedup expected)
- Best epoch, peak GPU memory
- **Normalization verdict:** state whether RMSNorm helps, hurts, or is neutral under Lion+slice=96. Compare to H72 (AdamW+RMSNorm+slice=96, val=57.58) and H73 (Lion+LayerNorm+slice=96, val=42.98).

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 + LayerNorm (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| val_single_in_dist/mae_surf_p | 43.7880 |
| val_geom_camber_rc/mae_surf_p | 56.6638 |
| val_geom_camber_cruise/mae_surf_p | 26.4930 |
| val_re_rand/mae_surf_p | 44.9686 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **41.5455** |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + **norm_type=layernorm** + T_max=15.

**Reference (anti-compound under AdamW):** H72 (AdamW+RMSNorm+slice=96): val=57.58 vs H66 (AdamW+LayerNorm+slice=96): val=56.75 → RMSNorm regressed under AdamW at slice=96.

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
