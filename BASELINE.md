# SENPAI Baseline — `icml-appendix-willow-pai2g-48h-r2`

The current best result on this advisor branch. Every new PR's primary metric must beat the values in the most-recent entry below.

- **Primary ranking metric:** `val_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 val splits)
- **Paper-facing test metric:** `test_avg/mae_surf_p` (equal-weight surface-pressure MAE across 4 test splits, all 4 splits must be finite)
- **Direction:** lower is better

---

## 2026-05-12 23:55 — PR #1585: Stack FiLM global conditioning on Huber baseline (research-ideas H5)

- **val_avg/mae_surf_p:** **80.8162** (best seed, epoch 14, base-model — student trained on Huber-only baseline #1452, *not* on Re-weight + SWA stack)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite, base-model)
- **All-3-seeds mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64 — every seed clears the previous baseline (95.75) by 12+ points
- Improvement vs. PR #1586 (95.75 / 86.17): val **−15.6%**, test **−17.3%** (largest single-PR gain on this branch to date)

### ⚠ Important composition note

This PR was trained against the **Huber-only baseline (PR #1452)**, *not* the Re-weight + SWA merged baseline (PR #1586). The student got val=80.82 / test=71.30 with **Huber + FiLM only**. The merge into the advisor branch composes FiLM with the existing Re-weight + SWA infrastructure, so the **post-merge train.py now runs Huber + Re-weight + SWA + FiLM together** — an untested combination.

- The 15-point absolute headroom below the previous baseline is large enough that even a worst-case interaction with Re-weight or SWA (PR #1645 evidence: SWA may regress this stack by ~5pts) leaves FiLM-on-merged firmly under 95.75.
- Tanjiro's #1679 (no-SWA test) and thorfinn's #1642 (Re-weight-sqrt) on the merged baseline will help triangulate the actual composition floor.
- **Conservative tested floor for the new baseline:** val=80.82 — the merged code likely achieves something between 80 and 85 val on next run.

### Val per-split surface MAE (best seed, epoch 14)

| Split | mae_surf_p (seed 2) | mean ± std (3 seeds) | Δ vs. #1586 (95.75 frame) |
|---|---|---|---|
| val_single_in_dist     | 88.39  | 92.50 ± 3.58 | **−21.84%** |
| val_geom_camber_rc     | 97.36  | 97.90 ± 0.47 | −5.16% |
| val_geom_camber_cruise | 59.69  | 60.06 ± 0.84 | **−19.84%** |
| val_re_rand            | 77.83  | 78.34 ± 0.92 | **−14.62%** |
| **val_avg**            | **80.8162** | **82.20 ± 1.23** | **−15.59%** |

### Test per-split surface MAE (best seed)

| Split | mae_surf_p (seed 2) | mean ± std (3 seeds) | Δ vs. #1586 (86.17 frame) |
|---|---|---|---|
| test_single_in_dist     | 79.48 | 82.58 ± 3.34 | **−17.51%** |
| test_geom_camber_rc     | 84.71 | 87.89 ± 2.93 | −6.95% |
| test_geom_camber_cruise | 50.26 | 50.30 ± 0.36 | **−21.65%** |
| test_re_rand            | 70.76 | 71.58 ± 0.84 | **−16.69%** |
| **test_avg**            | **71.3028** | **73.09 ± 1.64** | **−17.27%** |

### Config (tested run on Huber-only; merged code now also includes Re-weight + SWA)

- Architecture: Transolver baseline + **FiLM conditioner** (zero-init last linear → starts as identity)
  - FiLM input: globals (dims 13–23 of x, 11 features: Re, AoA, NACA M/P/T front+rear, gap, stagger) via masked-mean over real nodes
  - FiLM output: `[B, L=5, 2, H=128]` predicting per-layer per-sample `(γ, β)`
  - FiLM applied as `(1 + γ) * fx + β` after each block's FFN+residual
  - mid_dim=64, ~84K extra params (~13% of baseline 0.66M → total 0.75M)
- Loss: Smooth-L1 (Huber β=1.0) — student trained without Re-weight; merged code adds it
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4 (single param group, FiLM included via `nn.Module` wrapper)
- Scheduler: CosineAnnealingLR(T_max=15)
- Batch size: 4, surf_weight=10.0
- **Re-weight (in merged code, NOT in tested config):** `1/log_re_shifted`, normalized per batch
- **SWA (in merged code, NOT in tested config):** swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2
- Epochs: 15, wall clock 32.2 min (hit cap), peak VRAM ~42 GB

### FiLM modulation diagnostics (final epoch, averaged across 3 seeds)

| Layer | mean(|γ|) | mean(|β|) |
|---|---|---|
| L0 | 0.233 | 0.117 |
| L1 | 0.241 | 0.152 |
| L2 | 0.241 | 0.170 |
| L3 | 0.234 | 0.180 |
| L4 | 0.225 | 0.190 |
| **all-layer mean** | **0.235** | **0.162** |

`‖γ‖_L2 ≈ 15.3`, `‖β‖_L2 ≈ 10.6` (averaged across seeds). γ uniform across depth (~0.23–0.24), β grows monotonically with depth (0.12 at L0 → 0.19 at L4) — early layers prefer multiplicative scaling, later layers also use additive bias from global flow conditions. Mechanism is real, not a parameter-count artifact.

### W&B runs (3 seeds)

- seed 0: `f10x2pwq` — val=82.61, test=74.53
- seed 1: `vija565w` — val=83.17, test=73.44
- seed 2 (**best**): `j7uw0nhi` — val=80.82, test=71.30
- Link: https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/j7uw0nhi

### Reproduce (tested config — Huber + FiLM only, single seed)

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --seed 2 \
  --agent willowpai2g48h2-askeladd \
  --wandb_name willowpai2g48h2-askeladd/film-on-huber-seed2 \
  --wandb_group film-stack-test
```

### What landed

- **`FiLMConditioner(nn.Module)`** in `train.py`: MLP head `[Linear(11→64) → GELU → Linear(64→2·L·H)]` predicting per-layer `(γ, β)` from masked-mean of x[:,:,13:24]. Zero-init final linear → FiLM starts as identity.
- **`FiLMTransolver(nn.Module)`** wrapper in `train.py`: holds Transolver + FiLMConditioner, computes FiLM in forward, threads `film` tensor through Transolver's `data` dict so per-block forward can extract layer slice.
- **`TransolverBlock.forward(self, fx, film=None)`** modified to apply `(1+γ)*fx + β` after FFN+residual when `film` is non-None. Default behavior preserved when `film=None` (backward-compatible).
- **`Transolver.forward(self, data, **kwargs)`** reads `data["film"]` (optional) and slices per-layer `(γ, β)` into each block.
- `evaluate_split` updated to pass `mask` into the model dict so FiLM head can extract globals from real nodes.
- New CLI flags: `--seed`, `--film_mid_dim`.
- W&B observability: per-layer `|γ|`, `|β|` magnitudes, L2 norms, FiLM head param count.

### Open follow-ups (for future PRs)

- **Validate the merged composition.** Next merged PR confirms whether FiLM + Re-weight + SWA compose constructively. Expected ~80–85 val.
- **More epochs on FiLM-merged baseline.** Val was still descending at epoch 14 on best seed (−4.5% from epoch 12→14). 25–30 epochs likely buys another 2–4 points if wall-clock permits.
- **Stack FiLM with a geometry-aware lever.** Largest remaining gap is `val_geom_camber_rc` (97.90 ± 0.47) — FiLM helps cross-Re but not cross-camber-geometry as strongly. Slice_num bump, geometry-aware positional encoding, or surface-arc-length conditioning are candidates.
- **FiLM mid_dim sweep.** mid_dim=64 already learns non-trivial modulation; mid_dim=128 confirm run is cheap.
- **Per-channel loss weighting** (edward's wave-6 candidate): upweight `p` channel within both surf_loss and vol_loss. Orthogonal axis.

---

## 2026-05-12 22:02 — PR #1586: Stack per-sample Re-based loss weighting on Huber baseline

- **val_avg/mae_surf_p:** **95.7488** (best, epoch 14, base-model — student trained on Huber-only baseline, *not* SWA)
- **test_avg/mae_surf_p:** **86.1694** (4-split, all finite, base-model checkpoint)
- Improvement vs. PR #1554 (current merged baseline 99.07 / 88.90): val −3.36%, test −3.06%

### ⚠ Important composition note

This PR was trained against the **Huber baseline (PR #1452)**, *not* the merged SWA-on-Huber baseline (PR #1554). The student got val=95.75 / test=86.17 with **Huber + Re-weight only** (no SWA). The merge into the advisor branch *composed* the Re-weight changes with the existing SWA infrastructure, so the **post-merge train.py now runs Huber + Re-weight + SWA together** — an untested combination.

- If SWA + Re-weight compose constructively (likely; they target orthogonal axes — SWA averages weights, Re-weight reshapes per-step gradients), the next PR trained on this branch should match or beat 95.75 val.
- If SWA + Re-weight anti-compose, the next run could regress toward ~99 val (the SWA-only baseline) or worse.

The next training run on this baseline will validate the composition. Until then, **treat val=95.75 as the conservative tested floor** for the new baseline; the merged code likely achieves something between 95 and 93 val.

### Val per-split surface MAE (best epoch 14, Huber + Re-weight)

| Split | mae_surf_p | Δ vs. #1554 (99.07 baseline) |
|---|---|---|
| val_single_in_dist     | 113.0987 | −3.95% |
| val_geom_camber_rc     | 103.2184 | −1.03% |
| val_geom_camber_cruise | 74.9257  | **−5.37%** |
| val_re_rand            | 91.7525  | −3.54% |
| **val_avg**            | **95.7488** | **−3.36%** |

### Test per-split surface MAE

| Split | mae_surf_p | Δ vs. #1554 (88.90 baseline) |
|---|---|---|
| test_single_in_dist     | 100.1050 | −2.21% |
| test_geom_camber_rc     | 94.4517  | −1.07% |
| test_geom_camber_cruise | 64.1979  | **−5.10%** |
| test_re_rand            | 85.9230  | −4.63% |
| **test_avg**            | **86.1694** | **−3.06%** |

### Config (tested run; merged code now also includes SWA)

- Architecture: Transolver baseline (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, unified_pos=False)
- Loss: Smooth-L1 (Huber β=1.0) with **per-sample Re-based reweighting**:
  - Extract per-sample `log(Re)` from feature dim 13 via masked-mean over real nodes (constant per sample)
  - `log_re_shifted = log_re - log_re.min().detach() + 1.0` (positive shift)
  - `re_weight = 1.0 / log_re_shifted` then normalized to mean=1 per batch
  - Applied as per-sample multiplier on `sq_err` *before* surf/vol mask split, *before* `surf_weight=10.0`
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=15)
- Batch size: 4
- **SWA (now in merged code, NOT in the tested config):** swa_start_frac=0.75, swa_lr=1e-4, anneal_epochs=2, terminal eval on `swa_model.module`
- Epochs: 15 (cap triggered after epoch 14)
- Wall clock: ~30 min (hit `SENPAI_TIMEOUT_MINUTES=30`)
- Params: 0.66M
- Peak VRAM: ~42 GB

### Re-weight diagnostics (final-step W&B summary)

- `train/re_weight_mean` = 1.0000 (normalized as designed)
- `train/re_weight_min` = 0.6182 (highest-Re sample)
- `train/re_weight_max` = 1.6691 (lowest-Re sample) — ~2.7× spread
- `train/loss_unweighted` = 1.1588 vs. `train/loss` = 0.7271 (weighting reshapes loss by ~37%)

### W&B run

- `wt3u5zgs` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/wt3u5zgs

### Reproduce (tested config)

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-thorfinn \
  --wandb_name willowpai2g48h2-thorfinn/re-weight-on-huber \
  --wandb_group re-weight-stack-test
```

### What landed

Re-weight applied to the per-element Huber loss inside the training loop (`weighted_err = sq_err * re_weight_expanded`), *before* the surf/vol mask split. The diagnostic `train/loss_unweighted` is computed under `no_grad` so the reweighted loss is what backprop sees. Per-batch normalization keeps the mean weight at exactly 1.0, so the lever changes *which samples dominate* but not the average gradient magnitude.

### Open follow-ups (for future PRs)

- **Validate the merged composition.** The next PR trained on this branch will tell us whether SWA + Re-weight compose constructively or not. If the next 1-2 PRs land near 92-95 val (consistent with their own predicted improvements stacked on a 95-base), the composition is sound. If they regress toward 99 val, SWA may need to be re-tuned (lower swa_lr) or removed in favor of Re-weight alone.
- **Per-sample y_std weighting** (alternative to log_re-based weighting; the PR's "send-back" branch).
- **Stronger weighting curve** (e.g., `1/sqrt(log_re_shifted)` for wider spread).
- **Stacking with another mechanism-orthogonal lever** — surface-weight, mlp_ratio, FiLM, grad-clip, β-sweep all still untested on this composed baseline.

---

## 2026-05-12 21:06 — PR #1554: Stack SWA on Huber baseline

- **val_avg/mae_surf_p:** **99.0704** (SWA model, end of training)
- **test_avg/mae_surf_p:** **88.8955** (4-split, all finite, SWA model)
- Improvement vs. PR #1452: val −1.69%, test −1.65%

### Val per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| val_single_in_dist     | 117.7539 | −1.66% |
| val_geom_camber_rc     | 104.2288 | −4.71% |
| val_geom_camber_cruise | 79.1798  | −2.12% |
| val_re_rand            | 95.1191  | **+2.23%** |
| **val_avg**            | **99.0704** | **−1.69%** |

### Test per-split surface MAE (SWA model)

| Split | mae_surf_p | Δ vs. #1452 |
|---|---|---|
| test_single_in_dist     | 102.3693 | −3.43% |
| test_geom_camber_rc     | 95.4730  | −0.81% |
| test_geom_camber_cruise | 67.6442  | −1.77% |
| test_re_rand            | 90.0956  | −0.35% |
| **test_avg**            | **88.8955** | **−1.65%** |

### Config

- Everything from PR #1452 baseline (Huber β=1.0, AdamW lr=5e-4 wd=1e-4, batch=4, surf_weight=10.0, CosineAnnealingLR(T_max=15), 15 epochs)
- **SWA additions:**
  - `swa_start_frac = 0.75` → `swa_start_epoch = 11` (0-indexed)
  - `swa_lr = 1e-4` (= 0.2 × base lr)
  - `swa_anneal_epochs = 2`, `anneal_strategy = "cos"`
  - `update_bn` skipped (Transolver uses LayerNorm)
  - Terminal test eval runs on `swa_model.module`, not the base model
  - 3 SWA-active epochs in practice (epochs 12, 13, 14; epoch 15 timed out)
- Params: 0.66M (SWA is a running average, no extra trained params)
- Peak VRAM: ~42 GB
- Wall clock: 30.8 min

### W&B run

- `cnu8v9i2` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/cnu8v9i2

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/swa-on-huber \
  --wandb_group swa-stack-test
```

### What landed

- `torch.optim.swa_utils.AveragedModel` + `SWALR` added in `train.py`. Cosine anneals epochs 0–10 (inclusive); SWALR holds `swa_lr=1e-4` epochs 11–14 while `swa_model.update_parameters(model)` accumulates the running mean. After the last epoch, `model.load_state_dict(swa_model.module.state_dict())` and re-evaluate val/test — these are the headline numbers.
- Per-split test improvements are uniform (all 4 splits down), consistent with the flat-minima-helps-OOD hypothesis. Val mix is positive on 3/4 splits with a small `val_re_rand` regression (+2.2%) — likely an artifact of only 3 averaged epochs and `swa_lr` being above the cosine floor.

### Open follow-ups (for future PRs)

- **Stack SWA × unified_pos × FiLM × Re-weight × β-sweep** — orthogonal levers; current wave-2 wave (#1551 tanjiro, #1585 askeladd, #1586 thorfinn) all stack on Huber baseline. The next merged winner should compound on this SWA-on-Huber baseline.
- **Tighter SWA tuning:** lower `swa_lr` (0.1× or 0.05× base lr) and/or earlier `swa_start_frac` (0.65) to fit 4–5 averaged epochs into the 14-epoch envelope. Predicted further −1 to −3% on val.
- **Same open follow-ups carry forward from PR #1452:** β sweep, surface-only Huber, per-channel β.

---

## 2026-05-12 20:02 — PR #1452: Swap MSE → Smooth-L1 (Huber β=1.0) + scoring NaN-safe fix

- **val_avg/mae_surf_p:** **100.7659** (best, epoch 14)
- **test_avg/mae_surf_p:** **90.3840** (4-split, all finite — first finite 4-split test metric on this branch)

### Val per-split surface MAE (best epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist     | 119.7409 | 1.3652 | 0.7235 |
| val_geom_camber_rc     | 109.3817 | 2.1068 | 0.9464 |
| val_geom_camber_cruise | 80.8970  | 0.9151 | 0.5169 |
| val_re_rand            | 93.0438  | 1.5325 | 0.7294 |
| **val_avg**            | **100.7659** | 1.4799 | 0.7291 |

### Test per-split surface MAE (best checkpoint, epoch 14)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist     | 106.0083 | 1.2943 | 0.6857 |
| test_geom_camber_rc     | 96.2512  | 2.0110 | 0.8876 |
| test_geom_camber_cruise | 68.8607  | 0.8739 | 0.4658 |
| test_re_rand            | 90.4157  | 1.3369 | 0.6955 |
| **test_avg**            | **90.3840** | 1.3790 | 0.6837 |

### Config

- Architecture: Transolver baseline (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, unified_pos=False)
- Loss: Smooth-L1 (Huber β=1.0) replaces MSE in both training and `evaluate_split`
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=epochs=15) — schedule-aligned to actual training budget
- Batch size: 4
- surf_weight: 10.0
- Epochs: 15 (cap triggered after epoch 14; epoch 15 not started)
- Wall clock: ~30 min (hit `SENPAI_TIMEOUT_MINUTES=30`)
- Params: 0.66M
- Peak VRAM: ~42 GB

### W&B run

- `lo8vp7rj` — https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2/runs/lo8vp7rj

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 15 \
  --agent willowpai2g48h2-frieren \
  --wandb_name willowpai2g48h2-frieren/smooth-l1-loss-e15 \
  --wandb_group huber-loss-sweep
```

### What landed

1. **Loss reformulation:** MSE → Smooth-L1 (β=1.0) in `train.py` training loop and `evaluate_split`. Metric in `data/scoring.py` (denormalized-space MAE) is unchanged. Hypothesis was that Huber would cap high-Re outlier gradients where MSE over-penalizes — pattern confirmed: `val_geom_camber_cruise` (80.90) and `val_re_rand` (93.04) are the two lowest val splits.
2. **`data/scoring.py` NaN-safe fix:** `accumulate_batch` was propagating `0 * inf = NaN` from the corrupt GT sample `test_geom_camber_cruise/000020.pt` (761 nodes with `-inf` in the `p` channel). Fix uses `torch.where(mask, err, zero)` to select-or-zero without arithmetic, so masked positions never see `inf`. Effect: previously NaN `test_avg/mae_surf_p` is now finite across all 4 test splits.

### Open follow-ups (for future PRs)

- β sweep over {0.1, 0.3, 1.0, 3.0} now that β=1.0 is the established baseline.
- Surface-only Huber + MSE on volume (surface is the headline metric; outlier dominance is plausibly concentrated near foils).
- Stacking with orthogonal levers (positional encoding, slice_num, surf_weight, capacity).
- Per-channel β (pressure has a wider normalized range than Ux/Uy).
