# TandemFoilSet Baseline

## 2026-05-12 20:10 — PR #1391: BF16 + batch 8: more epochs within 30-min cap via AMP

**Changes merged:** bf16 autocast on training forward+loss, `batch_size=8`, `lr=7e-4` (√2 scaled), fp32 eval kept; scoring bug workaround in `evaluate_split` for `test_geom_camber_cruise/000020` (761 inf values in ground-truth pressure y — skip non-finite-y samples before accumulation).

### Primary metrics (best val checkpoint, epoch 17)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **133.7491** |
| **test_avg/mae_surf_p** | **121.2830** |

### Per-split test MAE (surface pressure)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| test_single_in_dist | 166.1911 | 2.0391 | 0.9130 | 171.1126 |
| test_geom_camber_rc | 136.1980 | 3.1547 | 1.1068 | 133.5701 |
| test_geom_camber_cruise | 78.5697 | 1.2584 | 0.5572 | 78.2832 |
| test_re_rand | 104.1732 | 1.7889 | 0.8165 | 103.6310 |

### Run info

- **W&B run:** `s8kl6dza` — group `bf16-batch-8`
- **Epochs:** 17 / 50 (30-min timeout, ~107 s/epoch)
- **Peak GPU memory:** 65.9 GB
- **Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.67M params)

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 50 \
  --agent <student> \
  --wandb_name <run-name> \
  --wandb_group <group>
```

*(Config defaults in `train.py` now include `lr=7e-4`, `batch_size=8`, bf16 autocast, and the scoring-bug workaround — no extra flags needed.)*

## 2026-05-12 22:06 — PR #1591: Cosine schedule aligned to 30-min budget: epochs=18

**Changes merged:** `epochs: int = 18` (was 50) in `Config` dataclass — aligns cosine T_max to the realistic 30-min budget. The merged baseline ran 17 epochs with final LR ≈ 6.2e-4 (barely decayed); this change lets cosine reach ~5e-6 final LR, giving the model the low-LR weight-space refinement phase it was missing. One-line diff in `train.py`, zero-overhead change.

### Primary metrics (best val checkpoint, epoch 15 of 17)

| Metric | Value | Δ vs prev |
|---|---|---|
| **val_avg/mae_surf_p** | **125.3551** | −6.27% |
| **test_avg/mae_surf_p** | **111.9787** | **−7.67%** |

### Per-split test MAE (surface pressure)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 148.79 | — | — |
| test_geom_camber_rc | 117.15 | — | — |
| test_geom_camber_cruise | 77.85 | — | — |
| test_re_rand | 104.13 | — | — |

### Run info

- **W&B run:** `h7w6skh8` — group `cosine-aligned-epochs`
- **Epochs:** 17 / 18 (30-min timeout, ~106 s/epoch)
- **Final LR:** 5.32e-6 (full cosine decay confirmed)
- **Peak GPU memory:** 82.68 GB
- **Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.67M params)

### Reproduce

```bash
cd "target/" && python train.py \
  --agent <student> \
  --wandb_name <run-name> \
  --wandb_group <group>
```

*(No `--epochs` flag needed — default is now 18. All other defaults: `lr=7e-4`, `batch_size=8`, bf16 autocast, scoring-bug workaround.)*

## 2026-05-13 01:20 — PR #1361: Wider hidden n_hidden 128→192 (batch_size=4 fallback)

**Changes merged:** `n_hidden=192` (was 128) in `model_config` in `train.py`. n_hidden=192 + bs=8 + bf16 OOMs at ~94 GB; batch_size must be set to 4 at runtime. All other config at schedule-aligned defaults.

**Key finding:** Width × schedule alignment compounds. Trial-4 (un-aligned T_max=50 schedule) gave −4.93% test; trial-5 on the schedule-aligned baseline gives −10.97% test. The schedule fix is a force-multiplier for capacity.

### Primary metrics (best val checkpoint, epoch 15 of 16, 3-seed mean)

| Metric | Value | Δ vs prev |
|---|---|---|
| **val_avg/mae_surf_p** | **111.32 ± 2.87** (mean ± std, n=3) | −11.51% |
| **test_avg/mae_surf_p** | **99.69 ± 3.16** (mean ± std, n=3) | **−10.97%** |

Best single-seed test: **96.19** (W&B `jvphwc6p`). Worst single-seed: **102.30** — both beat baseline.

### Per-split test MAE (surface pressure, 3-seed mean)

| Split | mae_surf_p (mean) | Δ vs prev baseline |
|---|---|---|
| test_single_in_dist | 116.57 | −21.6% |
| test_geom_camber_rc | 108.61 | −7.3% |
| test_geom_camber_cruise | 74.18 | −4.7% |
| test_re_rand | 99.41 | −4.5% |

### Run info

- **W&B runs (3 seeds):** `jvphwc6p`, `dcfy4v1z`, `9skp8i3k` — group `wider-hidden-192`
- **Epochs:** 15–16 / 18 (30-min timeout, ~126 s/epoch at bs=4)
- **Peak GPU memory:** ~30–40 GB estimated (bs=4 + bf16 + n_hidden=192; n_hidden=192 + bs=8 OOMs at ~94 GB)
- **Model config:** n_hidden=**192**, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1.47M params)

### Reproduce

```bash
cd "target/" && python train.py \
  --batch_size 4 \
  --agent <student> \
  --wandb_name <run-name> \
  --wandb_group <group>
```

*(Note: `--batch_size 4` is required — n_hidden=192 + bs=8 + bf16 OOMs at ~94 GB. All other defaults: lr=7e-4, epochs=18, bf16 autocast, scoring-bug workaround.)*
