# Baseline — icml-appendix-willow-pai2i-48h-r3

## Current best (as of 2026-05-17 03:10) — PR #4263: Cosine T_max=25 (schedule aligned to bf16 epoch budget)

Fourteen winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + Huber beta=0.1 (PR #3868, −3.77%) + Lookahead k=5 (PR #3947, −4.14%) + Gradient clipping max_norm=1.0 (PR #3497, −2.72%) + Huber beta=0.01 (PR #4037, −2.51%) + bfloat16 autocast (PR #3975, −9.74%) + **cosine T_max=25** (PR #4263, tanjiro, **−8.47% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **37.9354** (run `ymqw3n5m`, tanjiro arm3-tmax25, best epoch 17)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **39.0519** (−9.64% vs previous 43.2173)
  - `test_single_in_dist/mae_surf_p` = 40.7102
  - `test_geom_camber_rc/mae_surf_p` = 45.1351
  - `test_re_rand/mae_surf_p` = 31.3105
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR(**T_max=22**, i.e. `--cosine_t_max 25`), weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.01, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- **Lookahead (k=5, alpha=0.5)** wrapping SOAP
- **Gradient clipping (max_norm=1.0)** applied before optimizer.step()
- **bfloat16 autocast** (`--use_bf16`): forward + loss in bf16, backward + optimizer in fp32; no GradScaler
- **cosine_t_max=25** (`--cosine_t_max 25`): cosine phase = 25 − 3 warmup = 22 effective epochs; LR at epoch 17 ≈ 2.9e-4 (not zero — partial cooldown is better than full)
- Wall-clock: ~30 min cap, **best epoch 17**; epoch_time ~107s
- Peak VRAM: **33.0 GB**
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0 \
  --use_bf16 \
  --cosine_t_max 25
```

**Mechanism note:** With bf16 giving 17 epochs and the previous T_max=50, the LR at best epoch was ~80% of peak — the cosine cooldown phase was effectively disabled. T_max=25 gives a 22-epoch cosine window; at epoch 17 the LR is ~2.9e-4 (29% of peak), providing genuine refinement. T_max=17 (full budget match) goes all the way to zero by epoch 17, which is slightly too aggressive: Arm 3 (T_max=25) beats Arm 2 (T_max=17) on val (37.93 vs 38.32), suggesting the optimal end-point is a non-zero LR floor rather than fully cooled.

---

## Previous best (as of 2026-05-17 00:25) — PR #3975: bfloat16 autocast (+3 epochs in wall-clock cap)

Thirteen winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + Huber beta=0.1 (PR #3868, −3.77%) + Lookahead k=5 (PR #3947, −4.14%) + Gradient clipping max_norm=1.0 (PR #3497, −2.72%) + Huber beta=0.01 (PR #4037, −2.51%) + **bfloat16 autocast** (PR #3975, askeladd, **−9.74% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **41.4446** (run `cwlrnp3b`, askeladd variant-bf16-canonical, best epoch 17)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **43.2173** (−4.19% vs previous 45.1094)
  - `test_single_in_dist/mae_surf_p` = 45.9176
  - `test_geom_camber_rc/mae_surf_p` = 49.1937
  - `test_re_rand/mae_surf_p` = 34.5406
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.01, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- **Lookahead (k=5, alpha=0.5)** wrapping SOAP
- **Gradient clipping (max_norm=1.0)** applied before optimizer.step()
- **bfloat16 autocast** (`--use_bf16`): forward + loss in bf16, backward + optimizer in fp32; no GradScaler
- Wall-clock: ~30 min cap, **best epoch 17** (vs 14 without bf16); epoch_time ~107s vs 138s (1.285× speedup)
- Peak VRAM: **33.0 GB** (down from 42.1 GB; −21.6%)
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0 \
  --use_bf16
```

**Mechanism note:** bf16 autocast is quality-neutral at matched epoch (mean Δ +0.74 val over 14 epochs, within hardware drift window). Gain is pure throughput: 137.82 s/epoch → 107.25 s/epoch (+28.5% compute) → 3 extra epochs (14→17) in the 30-min wall-clock cap. SOAP eigendecomposition, Lookahead slow-weight buffers, and grad_clip stay in fp32. VRAM reduction (42.1 → 33.0 GB) unlocks headroom for batch size sweep and wider Transolver architecture.

---

## Previous best (as of 2026-05-16 20:35) — PR #4037: Huber beta=0.01 (lower bound)

Twelve winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + Huber beta=0.1 (PR #3868, −3.77%) + Lookahead k=5 (PR #3947, −4.14%) + Gradient clipping max_norm=1.0 (PR #3497, −2.72%) + **Huber beta=0.01** (PR #4037, fern, **−2.51% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **45.9199** (run `ysoma18c`, fern variant-beta001-lookahead-gradclip, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **45.1094** (−2.49% vs previous 46.2590)
  - `test_single_in_dist/mae_surf_p` = 50.89
  - `test_geom_camber_rc/mae_surf_p` = 48.61
  - `test_re_rand/mae_surf_p` = 35.83
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.01, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- **Lookahead (k=5, alpha=0.5)** wrapping SOAP
- **Gradient clipping (max_norm=1.0)** applied before optimizer.step()
- Wall-clock: ~35 min / arm (hit 30-min cap; best epoch 14, ~5,264 steps)
- Peak VRAM: 42.1 GB
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.01 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0
```

**Mechanism note:** Huber β=0.01 is near-pure L1 loss (MAE) — at this threshold, >99% of residuals above |0.01| contribute linear (not quadratic) gradient signal. Gradient clipping + L1-dominant loss combination produces a highly regularized update direction. The β=0.1→0.01 step compresses the quadratic zone by 10×, sharpening focus on large-residual geometry (OOD Re, complex camber cases).

---

## Previous best (as of 2026-05-16 19:20) — PR #3497: Gradient clipping (max_norm=1.0)

Eleven winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + Huber beta=0.1 (PR #3868, −3.77%) + Lookahead k=5 (PR #3947, −4.14%) + **Gradient clipping max_norm=1.0** (PR #3497, tanjiro, **−2.72% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **47.1000** (run `epby4q4n`, tanjiro variant-clip1-huber01-look-v2, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **46.2590** (−3.23% vs previous 47.8034)
  - `test_single_in_dist/mae_surf_p` = 50.9813
  - `test_geom_camber_rc/mae_surf_p` = 50.7476
  - `test_re_rand/mae_surf_p` = 37.0480
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.1, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- **Lookahead (k=5, alpha=0.5)** wrapping SOAP
- **Gradient clipping (max_norm=1.0)** applied before optimizer.step()
- Wall-clock: ~35 min / arm (hit 30-min cap; best epoch 14, 5,250 steps)
- Peak VRAM: 42.1 GB
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.1 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --grad_clip 1.0 \
  --wandb_group grad-clip-huber01-look \
  --wandb_run_name variant-clip1-huber01-look-v2
```

**Mechanism note:** Huber β=0.1 produces explosive pre-clip grad_norm (p50=112, max=730) vs Cauchy (p50=17). L1-dominant gradients have huge dynamic range — clip=1.0 renormalizes 100% of steps. SOAP preconditioner cares about gradient direction (relative curvature), not absolute magnitude, so aggressive scaling doesn't hurt and stabilizes per-step updates.

---

## Previous best (as of 2026-05-16 17:30) — PR #3947: Lookahead k=5 on SOAP (precondition_frequency=5)

Ten winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + Huber beta=0.1 (PR #3868, −3.77%) + **Lookahead k=5** (PR #3947, alphonse, **−4.14% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **48.4191** (run `yi5ektgs`, alphonse variant-lookahead-k5-freq5-huber01, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **47.8034** (−4.10% vs previous 49.8493)
  - `test_single_in_dist/mae_surf_p` = 54.0308
  - `test_geom_camber_rc/mae_surf_p` = 50.6174
  - `test_re_rand/mae_surf_p` = 38.7619
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.1, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- **Lookahead (k=5, alpha=0.5)** wrapping SOAP — k=5 aligns with precondition_frequency=5
- Wall-clock: ~32.3 min / arm (hit 30-min cap; best epoch 14)
- Peak VRAM: 42.1 GB
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.1 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --wandb_group lookahead-soap-huber01 \
  --wandb_run_name variant-lookahead-k5-freq5-huber01
```

**Note on hardware variance:** alphonse's baseline-no-lookahead arm (identical config to PR #3868) reproduced at val=48.823 vs BASELINE.md 50.5133 — a ~1.7 val drift between GPU machines (likely SOAP eigendecomposition non-determinism). The within-PR Lookahead delta (−0.83% val / −1.01% test) is hardware-controlled and robust. All future students should compare against this BASELINE.md number but note that absolute values may drift ±1-2 val between pods.

---

## Previous best (as of 2026-05-16 14:55) — PR #3868: Huber beta=0.1

Nine winners merged: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + SOAP optimizer (PR #3283, −31.7%) + SOAP precond_freq=5 (PR #3495, −1.78%) + EMA model weights decay=0.999 (PR #3430, −18.8%) + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy loss c=1.0 (PR #3612, −3.67%) + **Huber beta=0.1** (PR #3868, fern, **−3.77% vs previous canonical**).

**Primary ranking metric:**
- `val_avg/mae_surf_p` = **50.5133** (run `3yejzgk1`, fern variant-beta010, best epoch 14)

**Test (paper-facing):**
- `test_avg/mae_surf_p_excl_cruise` (3-split mean) = **49.8493** (−2.68% vs previous 51.220)
  - `test_single_in_dist/mae_surf_p` = 56.7466
  - `test_geom_camber_rc/mae_surf_p` = 53.1346
  - `test_re_rand/mae_surf_p` = 39.6666
  - `test_geom_camber_cruise/mae_surf_p` = NaN (pre-existing bug)

**Config (post-merge):**
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0
- **SOAP optimizer** (precondition_frequency=5) lr=1e-3, warmup_epochs=3 (LinearLR) → CosineAnnealingLR, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, **Huber loss (huber_beta=0.1, cauchy_c=0.0)**; `vol_loss + 10*surf_loss`
- **EMA of model weights** (ema_decay=0.99, updated each training step)
- Wall-clock: ~32 min / arm (hit 30-min cap; best epoch 14)
- `param count = 0.66M`

**Reproduce:**
```bash
cd target/ && python train.py \
  --optimizer soap \
  --precondition_frequency 5 \
  --lr 1e-3 --warmup_epochs 3 \
  --huber_beta 0.1 \
  --surf_weight 10.0 --seed 42 \
  --ema_decay 0.99 \
  --wandb_group huber-beta-finer-sweep \
  --wandb_run_name variant-beta010
```

---

## Previous best (as of 2026-05-16 12:00) — PR #3612: Cauchy loss c=1.0

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
  --precondition_frequency 5 \
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
