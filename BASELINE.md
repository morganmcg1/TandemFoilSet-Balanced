# Baseline — icml-appendix-willow-pai2i-48h-r3

## Current best (as of 2026-05-16 12:00) — PR #3612: Cauchy loss c=1.0

Eight winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + **Cauchy loss c=1.0** (PR #3612, edward, **−3.67% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **52.494** (run `mep5yevo`, edward variant-cauchy-c1-ema99-freq5, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **51.220** (−3.06% vs previous 52.837)
  - `test_single_in_dist/mae_surf_p` = 58.170
  - `test_geom_camber_rc/mae_surf_p` = 53.490
  - `test_re_rand/mae_surf_p` = 42.010
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Cauchy loss (cauchy_c=1.0)** replaces Huber; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- Wall-clock: ~32.6 min / arm (hit 30-min cap; best epoch 14)
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --lr 1e-3 --warmup_epochs 3 \
  --cauchy_c 1.0 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --wandb_group cauchy-ema-decay99 \
  --wandb_run_name variant-cauchy-c1-ema99-freq5
```

---

## Previous best (as of 2026-05-16 10:05) — PR #3316: Huber beta=0.5

Seven winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + **Huber beta=0.5** (PR #3316, fern, **−6.05% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **54.494** (run `9acc7fff`, fern variant-delta0.5, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **52.837** (−6.83% vs previous 56.713)
  - `test_single_in_dist/mae_surf_p` = 60.705
  - `test_geom_camber_rc/mae_surf_p` = 55.273
  - `test_re_rand/mae_surf_p` = 42.534
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (SmoothL1 beta=0.5)** `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step, used for all validation/test evaluation)
- Wall-clock: ~32.3 min / arm (hit 30-min cap; best epoch 14)
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.5 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --wandb_group huber-delta-sweep-ema-soap \
  --wandb_run_name variant-delta0.5
```

---

## Previous best (as of 2026-05-16 05:57) — PR #3591: EMA decay=0.99

Six winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + **EMA decay=0.99** (PR #3591, nezuko, **−3.85% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **58.005** (run `1xy36vpn`, nezuko variant-decay0.99, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **56.713** (−4.32% vs previous 59.27)
  - `test_single_in_dist/mae_surf_p` = 65.929
  - `test_geom_camber_rc/mae_surf_p` = 58.354
  - `test_re_rand/mae_surf_p` = 45.857
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Note on runs:** All 3 sweep arms ran with `precondition_frequency=10` (pre-PR-#3495). Post-merge canonical uses `precondition_frequency=5` (default after PR #3495). Expected compound val with freq=5 + decay=0.99 is approximately 56-57.

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, Huber loss (SmoothL1 beta=1.0) `vol_loss + 10*surf_loss`
- **EMA of model weights** (**ema_decay=0.99**, updated each training step, used for all validation/test evaluation)
- Wall-clock: ~136 s/epoch
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --lr 1e-3 --warmup_epochs 3 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --wandb_group ema-decay-sweep \
  --wandb_run_name variant-decay0.99
```

---

## Previous best (as of 2026-05-16 05:05) — PR #3495: SOAP precond_freq=5

Five winners merged: **Huber loss** (PR #3155, −18.1%) + **LR warmup 1e-3** (PR #3147, −8.9%) + **SOAP optimizer** (PR #3283, −31.7%) + **EMA of model weights decay=0.999** (PR #3430, −18.8%) + **SOAP precondition_frequency=5** (PR #3495, askeladd, **−1.78% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **60.33** (run `94f3r1yb`, askeladd freq=5 EMA, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **59.27** (−2.70% vs previous 60.92)
  - `test_single_in_dist/mae_surf_p` = 69.39
  - `test_geom_camber_rc/mae_surf_p` = 60.65
  - `test_re_rand/mae_surf_p` = 47.78
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Sanity:** baseline-freq10-ema (run `uu4nll7s`) reproduced canonical exactly: val=61.43, test=60.92 with seed=42 — confirms EMA+SOAP stack is fully reproducible.

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (**precondition_frequency=5**) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, Huber loss (SmoothL1 beta=1.0) `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.999, updated each training step, used for all validation/test evaluation)
- Wall-clock: ~138 s/epoch (+1.0% vs freq=10; negligible overhead)
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.999 \
  --wandb_group precond-freq-ema-soap \
  --wandb_name baseline-freq5-ema
```

---

## Previous best (as of 2026-05-15 22:30) — PR #3283: SOAP optimizer

Three round-1/2 winners merged: **Huber loss** (PR #3155, fern, −18.1%) + **LR warmup + peak 1e-3** (PR #3147, askeladd, −8.9%) + **SOAP optimizer** (PR #3283, alphonse, **−31.7% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **75.70** (run `vbvixri5`, alphonse SOAP, best epoch)

**Per-val-split surface pressure MAE (SOAP run vbvixri5):**
- `val_single_in_dist/mae_surf_p` ≈ n/a (avg reported; per-split not separately confirmed)
- `val_re_rand/mae_surf_p` — strong OOD generalization (re_rand test=66.21)

**Test (paper-facing, from run vbvixri5):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **75.39**
  - `test_single_in_dist/mae_surf_p` = 69.65
  - `test_geom_camber_rc/mae_surf_p` = 90.30
  - `test_re_rand/mae_surf_p` = 66.21
  - `test_geom_camber_cruise/mae_surf_p` = **NaN** (pre-existing bug, excluded)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (Shampoo eigenbasis + Adam updates, precondition_frequency=10) lr=1e-3, warmup_epochs=3 (LinearLR) then CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, Huber loss (SmoothL1 beta=1.0) `vol_loss + 10*surf_loss`
- Wall-clock: ~135.7 s/epoch (+2.9% vs AdamW)
- `param count = 0.66M`

---

## Previous best (as of 2026-05-15 16:00) — Huber + LR warmup

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
