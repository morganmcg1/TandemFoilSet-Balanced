# Baseline — TandemFoilSet (icml-appendix-willow-pai2g-48h-r3)

This is round 1 of a fresh launch. No baseline metrics are recorded yet.

The reference baseline is the as-is `train.py` on this branch:

## Reference config

- Optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
- LR schedule: CosineAnnealingLR(T_max=epochs)
- Batch size: 4
- Loss: vol_mse + surf_weight * surf_mse, surf_weight=10.0, normalized-space MSE
- Epochs: 50 (capped at `SENPAI_TIMEOUT_MINUTES=30` wall clock)
- Model: Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1M params)

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing number is `test_avg/mae_surf_p` (also lower is better), computed at the end of every run from the best-val checkpoint.

## Status

Current baseline: **PR #1505 (Huber/SmoothL1 surface loss, β=0.5)** stacked on top of #1504 (mask-aware PhysicsAttention). All subsequent PRs should compare against #1505's metrics.

## 2026-05-12 21:52 — PR #1504: Mask padded nodes in PhysicsAttention slice softmax

- **`val_avg/mae_surf_p`:** 119.450 (best-val checkpoint, `hg135fap`)
- **`test_avg/mae_surf_p`:** 109.669
- **Per-split val (best-val):** single_in_dist=140.20, geom_camber_rc=133.10, geom_camber_cruise=93.08, re_rand=111.42
- **Per-split test:** single_in_dist=123.97, geom_camber_rc=121.92, geom_camber_cruise=81.06, re_rand=111.73
- **W&B runs:** `hg135fap` (submitted), `xqrz8bjw` (seed-2: val=128.97, test=117.62)
- **Implementation note:** mask is applied to `slice_weights` **after** the slice softmax (`slice_weights * mask[:,None,:,None]`), not before — applying `-inf` before softmax over `slice_num` would produce NaN. Both seeds train cleanly with finite metrics on all four test splits, including `geom_camber_cruise` which was returning None on every other unmasked round-1 run.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-alphonse \
      --wandb_name "willowpai2g48h3-alphonse/mask-aware-physics-attn" \
      --wandb_group mask-aware-physics-attn
  ```

## 2026-05-13 00:00 — PR #1505: Huber/SmoothL1 surface loss (β=0.5)

- **`val_avg/mae_surf_p`:** 113.794 (best-val checkpoint, `ikjxaaze`)
- **`test_avg/mae_surf_p`:** 101.782
- **Per-split val (best-val):** [from W&B `ikjxaaze` epoch 13 — see PR comment for exact numbers]
- **Per-split test:** single_in_dist=118.85, geom_camber_rc=111.21, geom_camber_cruise=75.21, re_rand=101.87
- **W&B run:** `ikjxaaze` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r3/runs/ikjxaaze)
- **Implementation note:** Surface loss only — `F.smooth_l1_loss(beta=0.5, reduction="none")`. Volume term remains MSE. Eval `evaluate_split` also uses Huber for surf. MAE accumulators unchanged.
- **Delta vs PR #1504:** val −4.74% (119.45 → 113.79), test −7.19% (109.67 → 101.78). Test gain exceeded predicted ceiling (−8%) on `geom_camber_rc` (−8.78%) and `re_rand` (−8.83%), consistent with Huber suppressing high-error outliers and improving OOD generalization.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-askeladd \
      --wandb_name "willowpai2g48h3-askeladd/huber-surf-beta0p5-postmerge" \
      --wandb_group huber-surf
  ```
