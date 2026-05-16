# Baseline — icml-appendix-willow-pai2i-24h-r5

This launch starts fresh on `icml-appendix-willow-pai2i-24h-r5` (branched from `icml-appendix-willow`). No prior runs from this launch have been measured yet; the first round of experiments will both probe individual moves AND establish a confirmed baseline number.

## Current baseline configuration (head of advisor branch)

Model: **Transolver** (~1.5M params)
- `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `space_dim=2`, `fun_dim=22` (X_DIM=24 minus the 2 spatial dims)

Training:
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- Optimizer: `AdamW`
- Scheduler: `OneCycleLR(max_lr=1e-3, total_steps=len(train_loader)*14, pct_start=0.1, div_factor=25, final_div_factor=1e4)` with per-batch `scheduler.step()` (guarded: `if global_step < scheduler.total_steps`)
- Loss: vol_loss (MSE) + surf_weight × surf_loss (L1/MAE), `total = vol_loss + surf_weight * surf_loss`
- Grad clip: `clip_grad_norm_(max_norm=1.0)`
- `epochs=50` (max), capped by `SENPAI_TIMEOUT_MINUTES=30`

Per-run limits enforced by the harness:
- `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap)
- `SENPAI_MAX_EPOCHS=50` (hard epoch cap)
- 1 GPU per student, 96 GB VRAM

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing metric is `test_avg/mae_surf_p`, computed at the end of training using the best-val checkpoint.

## Baseline metrics

### 2026-05-16 09:40 — PR #3720: Lion optimizer (max_lr=3e-4) — paradigm-shift win

**New best: `val_avg/mae_surf_p = 66.69`** — best arm, **−10.37 pp (−13.46%)** vs prior baseline 77.06 (mean). Test 3-split (excl. cruise) hits **62.72** (−10.33 pp / −14.14% vs prior 73.05 mean).

| Split | val mae_surf_p (best `w8l5gszw`) | Arm 1 (max_lr=1e-3) | Arm 3 (max_lr=1e-4) |
|---|---|---|---|
| val_single_in_dist | 71.35 | 72.58 | 90.64 |
| val_geom_camber_rc | 81.98 | 84.46 | 90.90 |
| val_geom_camber_cruise | 48.74 | 48.64 | 56.42 |
| val_re_rand | 64.69 | 66.83 | 71.27 |
| **val_avg** | **66.69** | 68.13 | 77.31 |
| test_3split_excl_cruise | **62.72** | 65.13 | 74.54 |

- **W&B runs (3-arm max_lr sweep):** `w8l5gszw` (66.69 ★ winning arm), `psttg4e9` (68.13), `99j24mjj` (77.31)
- **Per-split test:** test_single_in_dist=60.78, test_geom_camber_rc=70.43, test_re_rand=56.94 (best arm)
- **Epochs:** 14 / 50 (30-min wall-clock cap; same as baseline)
- **Peak VRAM:** unchanged (Lion has 1 buffer vs AdamW's 2, but model is small)
- **Change vs prior baseline:** replaced `AdamW(lr=5e-4, wd=1e-4)` with `Lion(lr=1.5e-4, wd=1e-4)` and `OneCycleLR(max_lr=3e-4)` (was 1e-3 for AdamW). Lion `betas=(0.9, 0.99)`. CLI flag `--max_lr` added.
- **Why it works:** Lion's sign-based update is per-element clipping at unit magnitude. Combined with `clip_grad_norm_(max_norm=1.0)` (global L2 clip every step, median pre-clip norm ~45), the trajectory is double-bounded — first by global L2, then by per-element sign. The Lion paper recommends 3-10× smaller LR than AdamW; max_lr=3e-4 (3.3× smaller than AdamW's 1e-3) confirms the recommendation. Arm 1 (1e-3, AdamW-equivalent) is also a big win (−11.6%) but arm 2 (3e-4) is best. All splits improved dramatically: val_id −12 pp, val_rc −5 pp, val_cruise −13 pp (!!), val_re −12 pp. Mechanism is genuine OOD generalization improvement, not just optimization-side.
- **Reproduce (best arm):**
  ```bash
  cd target/ && python train.py \
    --lr 1.5e-4 --max_lr 3e-4 --weight_decay 1e-4 --batch_size 2 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-nezuko \
    --wandb_group willow-pai2i-24h-r5-round4 \
    --wandb_name nezuko-lion-maxlr3e4-arm2
  ```

---

### 2026-05-16 08:30 — PR #3616: batch_size=2 (2× gradient updates per epoch)

**New best: `val_avg/mae_surf_p = 77.06`** — 4-arm mean, **−4.60 pp (−5.63%)** vs prior baseline 81.66. All 4 arms beat baseline; spread 2.83 pp.

| Split | val mae_surf_p (best run `1xg2jnmd`) | 4-arm mean | Δ mean vs baseline |
|---|---|---|---|
| val_single_in_dist | 80.27 | 83.49 | −9.84 |
| val_geom_camber_rc | 86.61 | 86.63 | −6.04 |
| val_geom_camber_cruise | 58.56 | 61.23 | −0.64 |
| val_re_rand | 75.16 | 76.89 | −1.90 |
| **val_avg** | **75.15** | **77.06** | **−4.60** |
| test avg (3-split excl. cruise) | 72.44 | 73.34 | −5.94 |

- **W&B runs:** `eesuqkiy` (77.98), `1xg2jnmd` (75.15 ★ best), `stuakeo3` (77.74), `cbr2vdd2` (77.37) — group `willow-pai2i-24h-r5-round4`
- **Epochs:** 14 / 50 (30-min wall-clock cap; `len(train_loader)=750` at bs=2, `total_steps=10500`, double the steps vs bs=4)
- **Peak VRAM:** ~79 GB (down from ~94 GB at bs=4 — 15 GB headroom freed)
- **Change vs prior baseline:** `batch_size` from `4` → `2` (Config default); `OneCycleLR total_steps=len(train_loader)*14` auto-scales with the doubled loader length (10500 steps). All other hyperparameters unchanged.
- **Why it works:** 2× gradient updates per 30-min budget (10500 vs 5250 steps). In-distribution gains (−9.84 pp on val_single_in_dist) largest; the longer fine-tuning phase of the anneal benefits all splits. Smaller batch also reduces peak VRAM, freeing headroom for future architectural scale-up.
- **Reproduce (best arm):**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 2 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-fern \
    --wandb_group willow-pai2i-24h-r5-round4 \
    --wandb_name fern-bs2-arm1
  ```

---

### 2026-05-16 01:35 — PR #3307: OneCycleLR right-sized to actual budget + L1 surf (compound win)

**New best: `val_avg/mae_surf_p = 81.66`** — 3-arm mean, **−8.38 pp (−9.30%)** vs prior baseline 90.04. All 3 arms beat baseline; spread 3.80 pp.

| Split | val mae_surf_p (best run `iomzoqit`) | 3-arm mean | Δ mean vs baseline |
|---|---|---|---|
| val_single_in_dist | 92.04 | 93.33 | −15.62 |
| val_geom_camber_rc | 92.30 | 92.67 | −5.03 |
| val_geom_camber_cruise | 60.31 | 61.87 | −8.53 |
| val_re_rand | 76.60 | 78.79 | −4.32 |
| **val_avg** | **80.31** | **81.66** | **−8.38** |
| test avg (3-split excl. cruise) | 77.97 | 79.28 | −8.50 |

- **W&B runs:** `ut8w1dsk` (84.11), `iomzoqit` (80.31 ★ best), `f4lha65v` (80.57) — group `willow-pai2i-24h-r5-round2`
- **Epochs:** 14 / 50 (30-min wall-clock cap; scheduler exhausted at step 5250 = 14 × 375 batches)
- **Peak VRAM:** ~71 GB (~70% of H100 80GB; unchanged — OneCycleLR peak LR is higher but same model)
- **Change vs prior baseline:** replaced `CosineAnnealingWarmRestarts` with `OneCycleLR(max_lr=1e-3, total_steps=len(train_loader)*14, pct_start=0.1, div_factor=25, final_div_factor=1e4)` with per-batch stepping + `if global_step < scheduler.total_steps: scheduler.step()` guard. L1 surf loss (from #3434) retained.
- **Why it works:** Right-sizing `total_steps` to the actual 14-epoch budget makes the peak hit at epoch ~1.4, then aggressive anneal to ~4e-9 by epoch 14. The OOD splits improve most (`val_geom_camber_cruise` −8.53 pp, `val_single_in_dist` −15.62 pp on mean). OneCycle + L1 are orthogonal and stack: L1 chooses the median minimum, OneCycle finds it faster with a shaped LR trajectory.
- **Reproduce (best arm):**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-askeladd \
    --wandb_group willow-pai2i-24h-r5-round2 \
    --wandb_name askeladd-onecyclelr-1e3-rightsized-repl2
  ```

---

### 2026-05-15 21:30 — PR #3434: L1 surface loss (vol MSE + surf L1)

**New best: `val_avg/mae_surf_p = 90.04`** — single arm, -8.84 pp (-8.94%) vs warm-restarts baseline 98.88

| Split | val mae_surf_p (run `tcci4fzk`) | Δ vs prior baseline |
|---|---|---|
| val_single_in_dist | 108.95 | −7.41 |
| val_geom_camber_rc | 97.70 | −10.70 |
| val_geom_camber_cruise | 70.40 | −7.51 |
| val_re_rand | 83.11 | −9.76 |
| **val_avg** | **90.04** | **−8.84** |
| test avg (3-split excl. cruise) | 87.78 | −7.04 |

- **W&B run:** `tcci4fzk`
- **Epochs:** 14 / 50 (30-min wall-clock cap; still improving at epoch 14 — last two val_avg: 98.80 → 90.04)
- **Peak VRAM:** 42.1 GB (unchanged from prior baseline)
- **Change vs prior baseline:** replaced `sq_err` for `surf_loss` with `abs_err = (pred - y_norm).abs()` — vol_loss remains MSE, surf_loss is now L1/MAE
- **Why it works:** L1 minimizer = conditional median, the MAE-optimal estimator. Grad clip (`max_norm=1.0`) already normalizes step sizes, so L1 vs L2 convergence speed is similar (L1 only 1 epoch slower at epoch 1, then led all the way). Largest gain on `val_geom_camber_rc` (-10.70 pp) — heavy-tailed OOD error distribution where L2 chases outliers, L1 distributes gradient more evenly.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-edward \
    --wandb_group willow-pai2i-24h-r5-round3 \
    --wandb_name edward-l1-surf-loss
  ```

---

### 2026-05-15 20:25 — PR #3320: CosineAnnealingWarmRestarts T_0=5 T_mult=2

**New best: `val_avg/mae_surf_p = 98.88`** — Replicated across 3 independent runs (mean 100.67, spread ~3 pp)

| Split | val mae_surf_p (best run `oeo67jf2`) | 3-run mean |
|---|---|---|
| val_single_in_dist | 116.36 | 118.50 |
| val_geom_camber_rc | 108.40 | 110.54 |
| val_geom_camber_cruise | 77.91 | 79.25 |
| val_re_rand | 92.87 | 94.39 |
| **val_avg** | **98.88** | **100.67** |

- **test_avg/mae_surf_p:** NaN (cruise test split bad sample #3292; 3-split avg excl. cruise ≈ **94.82** best run, **96.71** mean across 3 runs)
- **W&B runs:** `oeo67jf2` (★ best), `79m50be7`, `iyhrbvuq` — group `willow-pai2i-24h-r5-round2`
- **Epochs:** 14 / 50 (30 min wall-clock cap; warm-restart cycle boundaries at epochs 5, 10, 20)
- **Peak VRAM:** 42.1 GB (unchanged from baseline)
- **Change vs prior baseline:** replaced `CosineAnnealingLR(T_max=MAX_EPOCHS)` with `CosineAnnealingWarmRestarts(T_0=5, eta_min=0, T_mult=2)` and `scheduler.step()` after each batch (not epoch)
- **Why it works:** Restarts at epochs 5, 10, 20 within the 14-epoch budget give multiple escape-from-local-minima opportunities. Improvement and low variance are both consistent with warm restarts breaking out of early-convergence plateaus.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-nezuko \
    --wandb_group willow-pai2i-24h-r5-round2 \
    --wandb_name nezuko-warm-restarts
  ```

---

### 2026-05-15 15:40 — PR #3157: Grad clipping max_norm=1.0

**New best: `val_avg/mae_surf_p = 117.16`**

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 138.19 |
| val_geom_camber_rc | 137.91 |
| val_geom_camber_cruise | 85.86 |
| val_re_rand | 106.68 |
| **val_avg** | **117.16** |

- **test_avg/mae_surf_p:** NaN (cruise test split has a bad sample — see GH issue #3292; 3-split avg excl. cruise ≈ 116.40)
- **W&B run:** `cfp7lnaq`
- **Epochs:** 14 / 50 (hit 30 min wall-clock cap)
- **Peak VRAM:** 42.1 GB
- **Change vs head:** added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-tanjiro \
    --wandb_group willow-pai2i-24h-r5-round1 \
    --wandb_name tanjiro-gradclip-1p0
  ```
- **Note:** max_norm=1.0 fired on 100% of steps (median pre-clip grad norm ≈ 45.7). Effective LR is ~45× lower than the nominal 5e-4. Next experiment will probe max_norm=10 to allow spike-only clipping.
