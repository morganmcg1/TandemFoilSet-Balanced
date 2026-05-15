<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Baseline — `icml-appendix-willow-pai2i-24h-r3`

Primary metric: `val_avg/mae_surf_p` (equal-weight mean of 4 validation splits, lower is better).
Paper-facing test metric: `test_avg_nansafe/mae_surf_p` (3-split nansafe due to `data/scoring.py` cruise NaN bug).

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
