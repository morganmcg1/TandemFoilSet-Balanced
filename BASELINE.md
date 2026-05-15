# Baseline — icml-appendix-willow-pai2i-48h-r3

## Current best (as of 2026-05-15 16:00)

Two round-1 winners merged: **Huber loss** (PR #3155, fern, −18.1%) and **LR warmup + peak 1e-3** (PR #3147, askeladd, −8.9%). New canonical baseline on the merged branch combines both changes.

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **110.83** (run `3nivkqy0`, fern Huber, epoch 13)

**Per-val-split surface pressure MAE (Huber variant, run 3nivkqy0):**
- `val_single_in_dist/mae_surf_p` = 132.06
- `val_geom_camber_rc/mae_surf_p` = 124.13
- `val_geom_camber_cruise/mae_surf_p` = 82.72
- `val_re_rand/mae_surf_p` = 104.40

**Test (paper-facing, from run 3nivkqy0):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **109.75**
  - `test_single_in_dist/mae_surf_p` = 118.65
  - `test_geom_camber_rc/mae_surf_p` = 111.97
  - `test_re_rand/mae_surf_p` = 98.64
  - `test_geom_camber_cruise/mae_surf_p` = **NaN** (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- AdamW lr=1e-3, warmup_epochs=3 (LinearLR) then CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, Huber loss (SmoothL1 beta=1.0) `vol_loss + 10*surf_loss`
- `param count = 0.66M`

**Note:** The LR warmup run `gyl9qikv` was tested on the pre-Huber baseline. The Huber+LR-warmup combined gain has not yet been directly measured — it should compound (−8.9% + −18.1% are from largely orthogonal changes). Next round validates this directly.

---

## Previous best: round-3 launch baseline (PR #3140, run xehwt9bi)

Round-3 baseline established by `willowpai2i48h3-alphonse` in PR #3140 (closed; baseline arm xehwt9bi).

## Best baseline metrics (run xehwt9bi, epoch 13/14 under 30-min cap)

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **135.30** (epoch 13)

**Per-val-split surface pressure MAE:**
- `val_single_in_dist/mae_surf_p` = 167.88
- `val_geom_camber_rc/mae_surf_p` = 135.88
- `val_geom_camber_cruise/mae_surf_p` = 109.77
- `val_re_rand/mae_surf_p` = 127.66

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **135.54**
  - `test_single_in_dist/mae_surf_p` = 141.05
  - `test_geom_camber_rc/mae_surf_p` = 134.38
  - `test_re_rand/mae_surf_p` = 131.18
  - `test_geom_camber_cruise/mae_surf_p` = **NaN** (pre-existing bug — see CURRENT_RESEARCH_STATE.md)

**Compute:**
- Wall-clock per epoch ≈ 131.8 s
- Best checkpoint reached at epoch 13 within the 30-min cap (14 epochs completed)
- Peak VRAM ≈ 95% of 96 GB

**Config:**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- AdamW lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs CosineAnnealingLR, MSE loss `vol_loss + 10*surf_loss`
- `param count = 0.66M`

W&B group: [`capacity-width-heads`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/capacity-width-heads) (baseline run `xehwt9bi`).

## Comparison protocol for round-1 PRs

All round-1 PRs are dual-arm (baseline + variant in same `--wandb_group`), so each PR has its own internal A/B. Use **within-group deltas** as the primary comparison since baseline-arm metrics fluctuate slightly between dual-arm runs (best_epoch may differ under the wall-clock cap).

For aggregate ranking across PRs, use the variant arm's `val_avg/mae_surf_p` (primary) and `test_avg/mae_surf_p_excl_cruise` (paper-facing proxy) against the canonical baseline above as a sanity check.

## Cruise-test NaN bug (pre-existing)

`test_geom_camber_cruise/mae_surf_p` returns NaN on the baseline arm and any unchanged model, due to a non-finite pressure prediction on at least one test sample whose squared error propagates Inf through `data/scoring.py:accumulate_batch`. Validation cruise is finite — only test cruise is affected.

Until fixed, **all PRs should report `test_avg` excluding cruise** (3-split mean: single-in-dist + geom-camber-rc + re-rand). See CURRENT_RESEARCH_STATE.md for the tracked fix.
