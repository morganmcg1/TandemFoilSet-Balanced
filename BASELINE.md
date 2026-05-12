# Baseline metrics (willow-pai2g-24h-r4)

**Branch:** `icml-appendix-willow-pai2g-24h-r4`
**Run cap:** `SENPAI_TIMEOUT_MINUTES=30` per training run, hard.

## Baseline config (`train.py` defaults)

| | |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 (+ 3-epoch linear warmup) |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Epochs | 50 (capped at 30 min wall) |
| Scheduler | SequentialLR: LinearLR(3 ep) → CosineAnnealingLR(T_max=47) |
| Loss | MSE, `vol_loss + 10 * surf_loss` |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=128, mlp_ratio=2 |
| Compile | torch.compile(model, mode="default", dynamic=True) |
| Precision | bf16 autocast train+eval, grad_clip_norm=1.0 |

## Primary ranking metric

`test_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four test splits. **Lower is better.**

Validation analogue (used for checkpoint selection): `val_avg/mae_surf_p`.

## Current best

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
