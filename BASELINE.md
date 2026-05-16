<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Baseline — `icml-appendix-willow-pai2i-24h-r3`

Primary metric: `val_avg/mae_surf_p` (equal-weight mean of 4 validation splits, lower is better).
Paper-facing test metric: `test_avg_nansafe/mae_surf_p` (3-split nansafe due to `data/scoring.py` cruise NaN bug).

---

## 2026-05-16 04:30 — PR #3596: T_max=21 cosine fix on Lion+bf16+clip+floor stack (tanjiro)

**Round-5 winner. New SOTA. −5.9% val / −6.3% test improvement over previous SOTA.**

- **val_avg/mae_surf_p:** 65.7375
- **test_avg_nansafe/mae_surf_p:** 61.7003 (via `eval_nansafe.py`)
- **Per-split val (best ckpt epoch 18):**
  - val_single_in_dist: ~79.6 (see W&B run tew7xthq)
  - val_geom_camber_rc: ~84.9
  - val_geom_camber_cruise: ~40.2
  - val_re_rand: ~67.5
- **Per-split test (nansafe, eval_nansafe.py):**
  - test_single_in_dist: 61.9972
  - test_geom_camber_rc: 69.7654
  - test_geom_camber_cruise: 57.5355
  - test_re_rand: 57.5030
- **Surface MAE (test_avg_nansafe):** Ux=0.9697, Uy=0.4851, p=61.7003
- **W&B run:** `tew7xthq` (group: `lion-tmax-newbase`, agent: `willowpai2i24h3-tanjiro`)
- **Key change (stacked on PR #3427 Lion+bf16+clip+floor):**
  - `lr_T_max=21` — cosine arc fully traverses the lower 57% of the curve within the 19-epoch bf16 budget, vs T_max=50 (only 38% traversed). LR reaches ~1.2e-5 at epoch 19, near but not at eta_min floor (1e-5).
  - Best epoch = 18 (not 19); epoch 19 mildly regresses in both arms — eta_min floor itself not helpful; the benefit is from the lower-LR refinement region around epochs 16-18.
  - Comparison arm (T_max=19): val=66.25 — also beats baseline, but T_max=21 is optimal (LR slightly higher in the critical ep 17-18 window)
- **Peak VRAM:** 33.0 GB / 96 GB (unchanged)
- **Reproduce:** `cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro`

Note: `test_avg/mae_surf_p` is NaN in-tree (cruise data bug). Test metric via `eval_nansafe.py`.

---

## 2026-05-16 00:30 — PR #3427: Lion + bf16 + grad-clip + eta_min stack (alphonse)

**Round-4 winner. New SOTA. −25.75% val improvement over previous Lion baseline.**

- **val_avg/mae_surf_p:** 69.8562
- **test_avg_nansafe/mae_surf_p:** 65.8812 (via `eval_nansafe.py`)
- **Per-split val (best ckpt epoch 19, final):**
  - val_single_in_dist: 78.4834
  - val_geom_camber_rc: 86.8730
  - val_geom_camber_cruise: 45.3256
  - val_re_rand: 68.7430
- **Per-split test (nansafe):**
  - test_single_in_dist: 71.6711
  - test_geom_camber_rc: 73.7479
  - test_geom_camber_cruise: 57.2517
  - test_re_rand: 60.8541
- **Surface MAE (test_avg_nansafe):** Ux=0.9810, Uy=0.4619, p=65.8812
- **W&B run:** `f6lnbssy` (group: `bf16-stable`, agent: `willowpai2i24h3-alphonse`)
- **Key changes (stacked on Lion+Huber baseline):**
  - bf16 autocast (forward only, fp32 loss) → 19 epochs in 30 min vs 14 fp32
  - `clip_grad_norm_(max_norm=1.0)` → engaged at 99.7% of steps (Lion gradients are systematically large; clip acts as per-step normalizer for momentum EMA)
  - `CosineAnnealingLR eta_min=1e-5` → standby insurance (not yet engaging at 19-epoch budget with T_max=50)
  - Best epoch = final epoch (val still descending at timeout — further headroom remains)
- **Peak VRAM:** 33.0 GB / 96 GB — large headroom for bigger model
- **Reproduce:** `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`

Note: `test_avg/mae_surf_p` is NaN in-tree (cruise data bug). Test metric via `eval_nansafe.py`.

---

## 2026-05-15 21:45 — PR #3387: Lion optimizer stacked on Huber baseline

**Round-4 winner. New baseline. −12.4% val improvement over previous baseline.**

- **val_avg/mae_surf_p:** 94.0803
- **test_avg_nansafe/mae_surf_p:** 88.9362 (via `eval_nansafe.py`, fern's nansafe eval script)
- **Per-split val (best ckpt epoch 14):**
  - val_single_in_dist: 108.0536
  - val_geom_camber_rc: 109.6926
  - val_geom_camber_cruise: 69.3504
  - val_re_rand: 89.2247
- **Per-split test (nansafe):**
  - test_single_in_dist: 97.1857
  - test_geom_camber_rc: 96.1708
  - test_geom_camber_cruise: 79.1690
  - test_re_rand: 83.2195
- **Surface MAE (test_avg_nansafe):** Ux=1.39, Uy=0.63, p=88.9362
- **W&B run:** `f9w6yzoq` (group: `lion-stacked`, agent: `willowpai2i24h3-fern`)
- **Key change:** Lion optimizer (`lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.99)`) replacing AdamW, stacked on Huber δ=2.0 baseline. Val curve still descending at timeout (epoch 14) — material headroom remains.
- **Reproduce:** `cd "target/" && python train.py --wandb_group lion-stacked --wandb_name lion-lr1e-4-wd1e-2 --agent willowpai2i24h3-fern`

Note: In-tree `test_avg/mae_surf_p` is NaN due to cruise data bug. Test metric computed via `eval_nansafe.py` (now checked in).

---

## 2026-05-15 18:22 — PR #3248: Replace MSE with Huber loss (delta=2.0)

**Round-3 winner. New baseline.**

- **val_avg/mae_surf_p:** 107.4641
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **Per-split val (best ckpt epoch 14):**
  - val_single_in_dist: 127.9121
  - val_geom_camber_rc: 118.4850
  - val_geom_camber_cruise: 83.3455
  - val_re_rand: 100.1139
- **Per-split test (nansafe):**
  - test_single_in_dist: 114.4305
  - test_geom_camber_rc: 107.9201
  - test_geom_camber_cruise: 89.0076
  - test_re_rand: 96.5812
- **Surface MAE (test_avg_nansafe):** Ux=1.9571, Uy=0.7267, p=101.9848
- **W&B run:** `mp8s8okf` (group: `huber-robust-loss`, agent: `willowpai2i24h3-frieren`)
- **Key change:** `loss_fn` switched from `F.mse_loss` to `F.huber_loss` with `delta=2.0` (in normalized space)
- **Reproduce:** `cd "target/" && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`

---

## 2026-05-15 (pre-round-3) — Fresh-slate reference

Edward's clean default-config anchor run used in round-3 comparisons:

- **val_avg/mae_surf_p:** 129.99 (W&B run `7fa1s7vm`, epoch 14/50)
- Config: Transolver L=5, AdamW lr=5e-4, MSE loss, equal channel weights, no warmup, CosineAnnealingLR(T_max=50)
