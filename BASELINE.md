# Baseline metrics (willow-pai2g-24h-r4)

**Branch:** `icml-appendix-willow-pai2g-24h-r4`
**Run cap:** `SENPAI_TIMEOUT_MINUTES=30` per training run, hard.

## Baseline config (`train.py` defaults)

| | |
|---|---|
| Optimizer | AdamW |
| LR / Scheduler | OneCycleLR(max_lr=1.5e-3, total_steps=29×steps/ep, pct_start=0.05, div_factor=10, final_div_factor=1e3) |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Epochs | 50 (capped at 30 min wall); eval_every_n_epochs=3 |
| Loss | MSE, `vol_loss + 10 * surf_loss` |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=128, mlp_ratio=2 |
| Compile | torch.compile(model, mode="default", dynamic=True) |
| Precision | bf16 autocast train, **fp32 eval (no autocast)**, grad_clip_norm=1.0 |

## Primary ranking metric

`test_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four test splits. **Lower is better.**

Validation analogue (used for checkpoint selection): `val_avg/mae_surf_p`.

## Current best

### 2026-05-13 03:00 — PR #1719: OneCycleLR pct_start=0.05 — compositional OOD-camber win on new max_lr baseline

- **val_avg/mae_surf_p:** **66.1352** (best epoch 27/29) ✓ NEW BASELINE
- **test_avg/mae_surf_p (4-split, fp32 eval):** **56.8971**
- **Per-split val (best epoch 27):**
  - `val_single_in_dist`: 73.33 (was 73.78, −0.6% — already saturated by max_lr=1.5e-3)
  - `val_geom_camber_rc`: 79.02 (was 80.71, −2.1%)
  - `val_geom_camber_cruise`: 47.51 (was 51.72, **−8.2% — biggest mover, all from pct_start**)
  - `val_re_rand`: 64.68 (was 68.10, −5.0%)
- **Per-split test:**
  - `test_single_in_dist`: ~60.5
  - `test_geom_camber_rc`: ~70.8
  - `test_geom_camber_cruise`: ~38.5
  - `test_re_rand`: ~57.8
- **W&B run:** `vfkbmgnp`
- **vs prior baseline (#1716):** val −2.45 (−3.57%), test −3.46 (−5.72%) — **test gain > val gain → strong generalization**
- **Peak GPU:** ~49 GB | **Sec/epoch:** ~62s | **Epochs:** 29 (best=27)
- **Model diff vs PR #1716:** `pct_start=0.1 → 0.05` in OneCycleLR constructor + W&B config (2 lines)
- **LR shape:** Peak LR (1.5e-3) reached at epoch 1.5 instead of 2.9. ~8 additional epochs in the deep-decay regime (LR < 1e-4).
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-nezuko-onecycle-pctstart-005 --agent willowpai2g24h4-nezuko
  ```

**Key insight (mechanistic — compositional, not redundant):** `max_lr=1.5e-3` (#1716) and `pct_start=0.05` (#1719) address ORTHOGONAL failure modes:
- High `max_lr` accelerates the in-dist basin descent (saturated at val_single_in_dist −8.8% in #1716; this PR adds only −0.6% there).
- Low `pct_start` extends the deep-decay tail. OOD splits (geom_camber_rc, geom_camber_cruise) are NOT LR-saturated — they're starved for refinement steps at low LR. pct_start=0.05 gives them ~28% more deep-decay epochs and unlocks val_geom_camber_cruise −8.2% / val_re_rand −5.0%.

**⚠️ Multi-seed calibration (PR #1874, 2026-05-13 05:01):** The single-seed gain above was variance-inflated. Seed=0 sat at the −1.15σ lucky tail across ALL metrics. 3-seed mean (seeds 0, 1, 2):

| Metric | seed=0 (best) | seed=1 | seed=2 | **3-seed mean ± std** |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 66.1352 | 70.5405 | 69.9678 | **68.88 ± 2.40** |
| test_avg/mae_surf_p | 56.8971 | 61.1382 | 60.7911 | **59.61 ± 2.36** |

Real OOD/in-dist trade-off: OOD splits (geom_camber_rc, geom_camber_cruise, re_rand) improve ~2-3.5% in mean; in-dist regresses +3.8% test / +4.5% val. The headline val_geom_camber_cruise −8.2% was actually ~−3.3% in the 3-seed mean. **Standing rule: future single-seed wins below 6% must be confirmed with ≥2 seeds before paper framing.**

**Next target:** beat val_avg/mae_surf_p = 66.1352 / test_avg/mae_surf_p = 56.8971 (single-run best) — or beat **3-seed mean 68.88 / 59.61** for paper-publishable claim.

---

### 2026-05-13 05:00 — PR #1874: 2-seed confirmation + seed flag (methodology)

- **val_avg/mae_surf_p:** N/A (confirmation run, not metric win)
- **test_avg/mae_surf_p:** N/A
- **W&B runs:** seed=1 `roajxtd5`, seed=2 `2tnq94du` (seed=0 reference: `vfkbmgnp`)
- **Code change:** `seed: int = 0` added to `Config`; `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed` seeded after `sp.parse(Config)`. Seed logged to W&B.
- **Reproduce (seed=N):**
  ```bash
  cd target
  python train.py --seed N --wandb_name willow-r4-confirm-seed-N --agent willowpai2g24h4-thorfinn
  ```

---

### 2026-05-13 01:30 — PR #1716: OneCycleLR max_lr=1.5e-3 — in-dist squeeze + LR ceiling probe

- **val_avg/mae_surf_p:** **68.5843** (best epoch 27/29 — last 3 within 0.03, converged tail) ✓ NEW BASELINE
- **test_avg/mae_surf_p (4-split, fp32 eval):** **60.3521**
- **Per-split val (best epoch 27):**
  - `val_single_in_dist`: 73.78 (was 80.87, −8.8% — biggest mover)
  - `val_geom_camber_rc`: 80.71 (was 80.80, ≈unchanged)
  - `val_geom_camber_cruise`: 51.72 (was 51.75, ≈unchanged)
  - `val_re_rand`: 68.10 (was 70.36, −3.2%)
- **Per-split test:**
  - `test_single_in_dist`: 63.5400
  - `test_geom_camber_rc`: 74.6399
  - `test_geom_camber_cruise`: 42.1353
  - `test_re_rand`: 61.0934
- **W&B run:** `dvk0201k`
- **vs prior baseline (#1556):** val −2.36 (−3.3%), test −1.48 (−2.4%)
- **Peak GPU:** 48.80 GB | **Sec/epoch:** ~62.1s | **Epochs:** 29 (best=27, last 3 within 0.03 — converged)
- **Model diff vs PR #1556:** `max_lr=1e-3 → 1.5e-3` in OneCycleLR constructor + W&B config (2 lines)
- **LR sweep:** 1.5e-4 → 1.5e-3 (peak) → 1.5e-7 (final), 10904 steps, no instability
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-alphonse-onecycle-maxlr1p5e3 --agent willowpai2g24h4-alphonse
  ```

**Key insight:** Higher peak LR accelerates convergence into the in-distribution basin (val_single_in_dist −8.8%). OOD splits (geom_camber_rc/cruise) are LR-saturated and don't move — they require a different lever (augmentation, regularization, geometry features). Best epoch shifts 29→27, indicating the schedule fully consumed its descent budget. Next step: push to max_lr=2e-3 to test the LR ceiling, and separately probe the OOD bottleneck.

**Superseded** by PR #1719 (composition with pct_start=0.05).

---

### 2026-05-13 01:05 — PR #1556: fp32 eval + eval_every_n_epochs=3 — paper-faithful test_avg (composes onto OneCycleLR)

- **val_avg/mae_surf_p:** **70.9449** (carried forward — unchanged, val cruise was always finite under bf16) ✓ NEW BASELINE STACK
- **test_avg/mae_surf_p (4-split, fp32 eval):** to be re-measured on next post-merge run — frieren's iter2 produced **64.3287** on the pre-OneCycleLR scheduler (Cosine); the fp32-eval contribution is orthogonal and should compose additively with OneCycleLR. Future PRs will re-establish the faithful number on this stack.
- **Cruise test fidelity:** `test_geom_camber_cruise/mae_surf_p` = **43.71** (frieren iter2, finite — no bf16 inf, no nan_to_num zeroing)
- **W&B run (orthogonal-merge reference, not OneCycleLR stack):** `uwk17oc0` (CosineAnnealingLR + fp32-eval-gate, 30 epochs / 30 min)
- **Peak GPU:** 29.8 GB | **Sec/epoch:** ~60.6s (mean across train + 10 eval epochs) | **Epochs:** 30/30 cap
- **Model diff vs prior baseline (#1404, OneCycleLR):**
  - Removed `torch.amp.autocast(bfloat16)` from `evaluate_split` — eval now runs in fp32
  - Added `eval_every_n_epochs: int = 3` to Config with final-epoch guard
  - `should_eval = (epoch % cfg.eval_every_n_epochs == 0) or (epoch == MAX_EPOCHS - 1)`
  - Best-checkpoint selection gated by `should_eval`
- **Reproduce:**
  ```bash
  cd target
  python train.py --agent willowpai2g24h4-frieren --wandb_name "willowpai2g24h4-frieren/fp32-eval-n3"
  ```

**Key insight:** This is a metric-FIDELITY improvement, not a metric-VALUE improvement. The val 70.94 stays valid (val cruise was always finite under bf16). The test_avg of 61.83 (under bf16 eval, cruise zeroed) is no longer the right comparison — future runs producing test_avg under fp32 eval will be higher in absolute terms but more honest. Cruise contribution recovers from artificial 42.96 (zeroed) to faithful ~43.71.

**Note on next target:** The first post-merge run on this stack will establish the new faithful test_avg/mae_surf_p baseline. Until then, beat val_avg/mae_surf_p = **70.9449** and recognize that test_avg numbers from this merge forward are not directly comparable to the pre-merge `61.8276` value.

**Next target:** beat val_avg/mae_surf_p = 70.9449 (test_avg/mae_surf_p baseline to be re-established by next experiment on this stack)

---

### 2026-05-13 00:00 — PR #1404: OneCycleLR (max_lr=1e-3, SCHEDULER_EPOCHS=29, per-batch step) — schedule shape win

- **val_avg/mae_surf_p:** **70.9449** (best epoch 29 of 29 — still descending at cutoff!) ✓ NEW BASELINE
- **test_avg/mae_surf_p (4-split, bf16 eval):** **61.8276** (cruise biased low by nan_to_num zeroing)
- **Test 3-split mean (excl. cruise):** 68.12
- **Per-split val surface MAE (best epoch 29):**
  - `val_single_in_dist`: p=80.87, Ux=0.971, Uy=0.518
  - `val_geom_camber_rc`: p=80.80, Ux=1.610, Uy=0.683
  - `val_geom_camber_cruise`: p=51.75, Ux=0.596, Uy=0.379
  - `val_re_rand`: p=70.36, Ux=1.109, Uy=0.530
- **Per-split test:** test_single_in_dist=70.98, test_geom_camber_rc=72.75, test_re_rand=60.63, test_geom_camber_cruise=42.96 (biased low)
- **W&B run:** `wd9na4r7`
- **Peak GPU:** 50.97 GB | **Sec/epoch:** ~62.5s | **Epochs:** 29/50 (30-min cap, best epoch=29 = last)
- **vs prior baseline (#1373):** val −4.90 (−6.5%), test −5.48 (−8.1%) — large, consistent across all 4 splits
- **Model diff vs prior baseline (#1373, lr=1e-3 + warmup + cosine):**
  - Replaced `SequentialLR([LinearLR(warmup=3), CosineAnnealingLR(T_max=47)])` with `OneCycleLR(max_lr=1e-3, total_steps=29×steps_per_epoch, pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=1e3)`
  - `scheduler.step()` moved from per-epoch to per-batch (inside training loop)
  - LR range: 1e-4 → 1e-3 (peak) → 1e-7 (final) — decay tail fires within the 30-min cap
- **Reproduce:**
  ```bash
  cd target
  python train.py --agent willowpai2g24h4-nezuko --wandb_name "willowpai2g24h4-nezuko/onecycle-lr-max1e3-pct0.1-28ep-rebased"
  ```

**Key insight:** Best epoch is the last (29) — model still descending monotonically at cutoff. The OneCycleLR per-batch stepping gives 10875 LR updates (vs 29 for cosine) and fully fires the decay tail to LR≈1e-7. The schedule shape advantage over warmup+cosine is real and large.

**Note on test_avg:** cruise bf16 vol_loss=inf (nan_to_num zeroed), same caveat as prior baselines. Frieren #1556 fp32-eval follow-up will recover faithful 4-split test_avg.

**Next target:** beat val_avg/mae_surf_p = 70.9449 / test_avg/mae_surf_p = 61.8276

---

### 2026-05-12 23:XX — PR #1373: lr=1e-3 + 3-epoch linear warmup + cosine (on top of compile + bf16 + slice_num=128)

- **val_avg/mae_surf_p:** **75.8473** (best epoch 27 of 29 completed) ✓ NEW BASELINE
- **test_avg/mae_surf_p (4-split):** **67.3037** (all 4 splits clean; cruise biased low by nan_to_num zeroing)
- **Test 3-split mean (excl. cruise):** 74.80
- **Per-split val surface MAE (best epoch 27):**
  - `val_single_in_dist`: p=86.08, Ux=1.099, Uy=0.581
  - `val_geom_camber_rc`: p=91.75, Ux=1.795, Uy=0.744
  - `val_geom_camber_cruise`: p=52.57, Ux=0.669, Uy=0.398
  - `val_re_rand`: p=72.99, Ux=1.138, Uy=0.558
- **Per-split test:** test_single_in_dist=76.74, test_geom_camber_rc=83.08, test_re_rand=64.59, test_geom_camber_cruise=44.80
- **W&B run:** `waeuuqkw`
- **Peak GPU:** 48.60 GB | **Sec/epoch:** ~61.2s | **Epochs:** 29/50 (30-min cap, epoch 27 best)
- **Model diff vs prior baseline (torch.compile + bf16 + slice_num=128):**
  - `lr: 5e-4 → 1e-3`
  - `SequentialLR([LinearLR(start_factor=0.1, total_iters=3), CosineAnnealingLR(T_max=47)], milestones=[3])` replaces single `CosineAnnealingLR(T_max=50)`
- **vs prior baseline (#1584):** val −0.58 (−0.76%), test −1.46 (−2.12%)
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-alphonse-lr1e3-warmup-compile --agent willowpai2g24h4-alphonse
  ```

**Note on test_avg:** test_geom_camber_cruise result (44.80) reflects nan_to_num zeroing of bf16-overflow cruise predictions — biased low. Faithful fp32-eval follow-up in progress via #1556.

**Next target:** beat val_avg/mae_surf_p = 75.8473 / test_avg/mae_surf_p = 67.3037

---

### 2026-05-12 21:56 — PR #1584: torch.compile(model, dynamic=True) — free throughput (on top of bf16 + slice_num=128)

- **val_avg/mae_surf_p:** **76.4310** (best epoch 27 of 29 completed) ✓ NEW BASELINE
- **test_avg/mae_surf_p (4-split, bf16 eval):** 68.7604 (cruise node posinf zeroed by scoring fix — biased low; fp32 eval follow-up in progress via #1556)
- **Test 3-split mean (excl. cruise):** 74.84
- **Per-split val surface MAE (best epoch 27):**
  - `val_single_in_dist`: p=99.46, Ux=1.158, Uy=0.632
  - `val_geom_camber_rc`: p=93.02, Ux=1.700, Uy=0.750
  - `val_geom_camber_cruise`: p=51.21, Ux=0.698, Uy=0.422
  - `val_re_rand`: p=73.14, Ux=1.191, Uy=0.582
- **Per-split test (bf16 eval):** test_single_in_dist=74.40, test_geom_camber_rc=82.96, test_re_rand=67.16, test_geom_camber_cruise=50.53 (bf16 inf zeroed by scoring fix)
- **W&B run:** `t0zwgi1n`
- **Peak GPU:** 50.78 GB | **Sec/epoch:** ~62.6s (median, epochs 3+) | **Epochs:** 29/50 (30-min cap, epoch 27 best)
- **Model diff vs prior baseline (bf16 + slice_num=128):**
  - `model = torch.compile(model, mode="default", dynamic=True)` immediately after `.to(device)`
  - Everything else identical: bf16 autocast in train forward, bf16 autocast in eval forward, grad_clip_norm=1.0, slice_num=128, T_max=50
- **Throughput:** 1.58× speedup (compile amortized over 29 epochs; 1-time ~12s overhead in epoch 1 only)
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-thorfinn-compile --agent willowpai2g24h4-thorfinn
  ```

**Note on test_avg:** bf16 eval still in place (same cruise overflow zeroing as PR #1415). Frieren's #1556 (fp32 eval every N epochs) will recover the faithful 4-split test_avg on top of this baseline.

**Next target:** beat val_avg/mae_surf_p = 76.4310

---

### 2026-05-12 20:XX — PR #1415: bf16 mixed precision + grad_clip (on top of slice_num=128 + scoring fix)

- **val_avg/mae_surf_p:** **98.7664** (best epoch 18 of 18 completed) ✓ NEW BASELINE
- **test_avg/mae_surf_p:** NaN at submit-time (bf16-induced `inf` pred on one cruise test node). After PR #1521 scoring fix merged: future runs/re-evals should report finite test (posinf zeroed in `nan_to_num`).
- **Test 3-split mean (excl. cruise):** 97.12
- **Per-split val surface MAE (best epoch 18):**
  - `val_single_in_dist`: p=108.76, Ux=1.67, Uy=0.68
  - `val_geom_camber_rc`: p=115.38, Ux=2.31, Uy=0.90
  - `val_geom_camber_cruise`: p=78.21, Ux=0.98, Uy=0.53
  - `val_re_rand`: p=92.71, Ux=1.61, Uy=0.72
- **Per-split test (raw):** test_single_in_dist=98.36, test_geom_camber_rc=104.62, test_re_rand=88.39, test_geom_camber_cruise=NaN (bf16 inf)
- **W&B run:** `ojdeyn8r`
- **Peak GPU:** 32.9 GB | **Sec/epoch:** ~99s | **Epochs:** 18/50 (30-min cap, still descending)
- **Model diff vs prior baseline (slice_num=128 + scoring fix):**
  - bf16 autocast in train forward + grad_clip_norm=1.0
  - bf16 autocast in eval forward (suspected source of cruise inf — future work should test fp32 eval)
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-thorfinn-bf16 --agent willowpai2g24h4-thorfinn
  ```

**Note on test_avg:** The bf16 eval autocast caused one `pred` node to overflow on `test_geom_camber_cruise`. The merged PR #1521 scoring fix now zeros that out, but reduces the affected channel's MAE slightly (the overflowing node now contributes 0 instead of being properly skipped). Follow-up to switch eval to fp32 is the natural next step.

**Next target:** beat val_avg/mae_surf_p = 98.7664

---

### Previous baselines

#### 2026-05-12 19:XX — PR #1396: Double Transolver slice tokens (slice_num 64 → 128)

- **val_avg/mae_surf_p:** 146.2510 (epoch 9 of 11 completed)
- **test_avg/mae_surf_p:** NaN ⚠️ — GT NaN in `test_geom_camber_cruise` sample 20 leaks through `err * mask` in `data/scoring.py:49`. Bug-fix PR pending; val number is valid.
- **Per-split val surface MAE (best epoch 9):**
  - `val_single_in_dist`: p=175.68, Ux=—, Uy=—
  - `val_geom_camber_rc`: p=158.18
  - `val_geom_camber_cruise`: p=115.62
  - `val_re_rand`: p=135.53
- **Test (3-split excl. cruise):** 147.07
- **W&B run:** `5qh8pj8v`
- **Peak GPU:** 54.5 GB | **Sec/epoch:** ~172s | **Epochs:** 11/50 (30-min cap)
- **Model diff vs original baseline:** `slice_num=128` (was 64); all other config unchanged.
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-frieren-slice128 --agent willowpai2g24h4-frieren
  ```

_(Previous baseline — superseded by #1415)_
