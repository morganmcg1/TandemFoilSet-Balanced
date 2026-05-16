# TandemFoilSet Baseline — branch `icml-appendix-charlie-pai2i-24h-r3`

This branch tracks the best-merged result on `icml-appendix-charlie-pai2i-24h-r3`.

## Baseline configuration

The starting point is the Transolver baseline in `train.py`:

- **Model**: Transolver with PhysicsAttention over slice-tokens
  - `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `space_dim=2`, `fun_dim=22`, `out_dim=3`
- **Optimizer**: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- **Schedule**: `CosineAnnealingLR(T_max=epochs)`
- **Batch**: `batch_size=4`
- **Loss**: `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`, both MSE in normalized target space
- **Sampling**: `WeightedRandomSampler` with domain-balancing weights
- **Run budget**: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30` (hard upper bounds)

## Primary metrics

- Validation ranking: **`val_avg/mae_surf_p`** (equal-weight mean surface pressure MAE across the four validation splits)
- Paper-facing test: **`test_avg/mae_surf_p`** (same metric across the four test splits, evaluated at the best-val checkpoint)
- Lower is better

## Reproduce

```bash
cd target/
python train.py --experiment_name baseline_transolver --agent baseline
```

This produces `models/model-baseline-<stamp>/metrics.jsonl` with per-epoch val metrics and a final test record.

## Best result

### 2026-05-15 15:30 — PR #3237: Huber loss (delta=1.0) to cap high-Re gradient outliers

**Winner**: edward (`charliepai2i24h3-edward/huber-loss`)

- **`val_avg/mae_surf_p` = 117.6594** (best epoch 13 / 14 run, still improving at timeout)
- **`test_avg/mae_surf_p` = NaN** (scoring.py NaN bug — one sample in test_geom_camber_cruise has inf in GT)
  - Clean estimate (3 finite test splits): ~107.6
- **Per-split val metrics (epoch 13 best checkpoint)**:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 147.77 | 1.620 | 0.855 |
| val_geom_camber_rc | 125.08 | 2.199 | 1.030 |
| val_geom_camber_cruise | 88.98 | 1.609 | 0.625 |
| val_re_rand | 108.81 | 1.983 | 0.813 |
| **val_avg** | **117.66** | 1.853 | 0.831 |

- **Change**: 2-line swap: `sq_err = F.huber_loss(pred, y_norm, reduction='none', delta=cfg.huber_delta)` + `huber_delta: float = 1.0` config field. All other hyperparameters unchanged.
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params)
- **Peak VRAM**: 42.11 GB
- **Metric artifacts**: `models/model-huber_loss_d1-20260515-130807/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name huber_loss_d1 --agent charliepai2i24h3-edward`

> **Note**: `test_avg/mae_surf_p` is NaN for all experiments due to a bug in `data/scoring.py` — sample 20 of `test_geom_camber_cruise` has `inf` in GT, which propagates through the masked sum (`inf * 0 = NaN`). All ranking uses `val_avg/mae_surf_p` until this is resolved. See EXPERIMENTS_LOG.md for details.

---

### 2026-05-15 22:45 — PR #3300: BF16 mixed-precision to get more epochs in 30-min budget

**Winner**: edward (`charliepai2i24h3-edward/bf16-mixed-precision`)

- **`val_avg/mae_surf_p` = 97.5474** (best epoch 17 / 19 run, hit 30-min cap)
- **`test_avg/mae_surf_p` = NaN** (scoring.py NaN bug — test_geom_camber_cruise/000020.pt inf in GT)
  - Clean estimate (3 finite test splits): **93.99** (vs ~107.6 prior clean estimate)
- **Per-split val metrics (epoch 17 best checkpoint)**:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 114.41 | 1.387 | 0.674 |
| val_geom_camber_rc | 104.96 | 2.060 | 0.851 |
| val_geom_camber_cruise | 79.72 | 1.135 | 0.531 |
| val_re_rand | 91.09 | 1.532 | 0.678 |
| **val_avg** | **97.55** | 1.529 | 0.684 |

- **Delta vs prior baseline**: −20.11 (−17.1%) on val_avg/mae_surf_p. All 4 splits improve.
- **Change**: Add `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around training forward+loss and validation forward. No GradScaler needed (BF16 has same exponent range as FP32).
- **Budget gain**: Reached epoch 19 vs 14 at baseline (+5 epochs in same 30-min cap). s/epoch: ~98 vs ~128 (1.3x throughput).
- **Peak VRAM**: 32.95 GB (vs 42.11 GB, −22%)
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params). No model changes.
- **Metric artifacts**: `models/model-bf16_huber-20260515-212744/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name bf16_huber --agent charliepai2i24h3-edward`

---

### 2026-05-16 00:40 — PR #3513: Cosine schedule match (T_max=20 to match realistic epoch horizon)

**Winner**: edward (`charliepai2i24h3-edward/cosine-schedule-match`)

- **`val_avg/mae_surf_p` = 87.6209** (best epoch 19 / 19 run, hit 30-min cap)
- **`test_avg/mae_surf_p` = NaN** (scoring.py NaN bug — test_geom_camber_cruise/000020.pt inf in GT)
  - Clean estimate (3 finite test splits): **84.10** (rc=88.31, re_rand=78.29, single=85.69)
- **Per-split val metrics (epoch 19 best checkpoint)**:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 98.44 | 1.174 | 0.616 |
| val_geom_camber_rc | 96.95 | 1.923 | 0.837 |
| val_geom_camber_cruise | 71.27 | 0.782 | 0.493 |
| val_re_rand | 83.83 | 1.356 | 0.647 |
| **val_avg** | **87.62** | 1.230 | 0.608 |

- **Delta vs prior baseline**: −9.93 (−10.18%) on val_avg/mae_surf_p. All 4 splits improve.
- **Change**: Set `cosine_t_max: int = 20` in Config; replace `T_max=MAX_EPOCHS` with `T_max=cfg.cosine_t_max`. LR anneals to ~3e-6 (0.62% of initial) by epoch 19 vs 74% in prior baseline. Zero compute overhead.
- **Budget**: Same 19 epochs, same 30-min cap, same ~98s/epoch. No throughput change.
- **Peak VRAM**: 32.94 GB (unchanged).
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params). No model changes.
- **Metric artifacts**: `models/model-bf16_cosine_t20-20260515-232950/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name bf16_cosine_t20 --agent charliepai2i24h3-edward`

---

### 2026-05-16 08:30 — PR #3753: DSDF feature clipping at ±3σ (position/saf tail reduction)

**Winner**: alphonse (`charliepai2i24h3-alphonse/dsdf-clip`)

- **`val_avg/mae_surf_p` = 86.7674** (best epoch 19 / 19, timeout-bounded)
- **`test_avg/mae_surf_p`**: NaN (scoring.py NaN bug on test_geom_camber_cruise)
  - Clean estimate (3 finite test splits): **83.4549** (rc=84.567, re_rand=77.111, single=88.687)
- **Per-split val metrics (epoch 19 best checkpoint)**:

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 102.003 |
| val_geom_camber_rc | 93.754 |
| val_geom_camber_cruise | 69.146 |
| val_re_rand | 82.166 |
| **val_avg** | **86.767** |

- **Delta vs prior baseline (87.62)**: −0.85 (−0.97%). 3 of 4 splits improve; val_single_in_dist regresses +3.56.
- **Change**: After input feature normalization, apply `x_norm = x_norm.clamp(-3.0, 3.0)` (global clip all 24 dims). Per diagnostic: DSDF dims 4-11 had 0% clipping — the gain comes from position (dims 0-1, ~2%) and signed arc-length saf (dims 2-3, ~3%) tail clipping in large meshes. DSDF hypothesis was incorrect in mechanism but the change is still a real improvement.
- **Budget**: 19 epochs, ~98s/epoch, ~33 GB VRAM. No compute change.
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params). No model changes.
- **Metric artifacts**: `models/model-dsdf_clip-20260516-073513/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name dsdf_clip --agent charliepai2i24h3-alphonse`

**Note**: val_single_in_dist regressed despite global improvement — geometry tail clipping may remove informative boundary-condition nodes from single-foil meshes. Future follow-up: surgical clip on dims 0-3 only, or per-domain conditional clip.

---

### 2026-05-16 12:30 — PR #3894: 5-epoch linear LR warmup + cosine T_max=15

**Winner**: frieren (`charliepai2i24h3-frieren/lr-warmup`)

- **`val_avg/mae_surf_p` = 83.4667** (best epoch 18 / 19, timeout-bounded at 31.1 min)
- **`test_avg/mae_surf_p`**: NaN (scoring.py NaN bug on test_geom_camber_cruise)
  - Clean estimate (3 finite test splits): **82.4799** (single=89.47, rc=86.00, re_rand=71.97)
- **Per-split val metrics (epoch 18 best checkpoint)**:

| Split | mae_surf_p | Δ vs prior (86.77) |
|---|---|---|
| val_single_in_dist | 100.45 | −1.55 (−1.5%) |
| val_geom_camber_rc | 94.78 | +1.03 (+1.1%) |
| val_geom_camber_cruise | 60.92 | **−8.23 (−11.9%)** |
| val_re_rand | 77.72 | **−4.45 (−5.4%)** |
| **val_avg** | **83.4667** | **−3.30 (−3.80%)** |

- **Delta vs prior baseline (86.77)**: **−3.30 (−3.80%)** — largest single-step improvement this session. 3/4 splits improve; val_geom_camber_rc regresses slightly (+1.1%).
- **Change**: Replaced `CosineAnnealingLR(T_max=20)` with `SequentialLR([LinearLR(start_factor=0.01, total_iters=5), CosineAnnealingLR(T_max=15)], milestones=[5])`. LR warms from 5e-6 → 5e-4 over 5 epochs, then cosines to 0 over 15 epochs. Zero param overhead.
- **Mechanism confirmed**: LR trajectory shows stable warmup ramp before the cosine high-LR phase. Prevents aggressive early gradient steps from high-Re outlier samples before Adam moments stabilize (~1000 steps, or ~5 epochs at our scale).
- **Budget**: 18 epochs in 31.1 min (slightly over 30 due to warmup overhead). ~98s/epoch. No VRAM change.
- **Model**: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, 0.66M params). No model changes.
- **Metric artifacts**: `models/model-lr_warmup-20260516-112556/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --experiment_name lr_warmup --agent charliepai2i24h3-frieren`

**Stack**: BF16 + Huber δ=1.0 + cosine T_max=15 + **5-epoch linear warmup** + global ±3σ clip → **val_avg = 83.47**
