# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16

## Current best — PR #3474: EMA decay=0.99 + grad_clip=5 + Huber δ=1.0 (alphonse)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **90.6131** | run `fzrq04xr` (best @ epoch 14) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | **88.8252** | run `fzrq04xr` |

Per-split validation (best @ epoch 14):

| Split | mae_surf_p | Δ vs prev baseline (#3366, 94.4199) |
|---|---|---|
| val_single_in_dist | 106.135 | **−5.1%** |
| val_geom_camber_rc | 99.466 | **−9.7%** |
| val_geom_camber_cruise | 70.358 | +1.9% |
| val_re_rand | 86.494 | −0.2% |

Per-split test (best ckpt):

| Split | mae_surf_p | Δ vs prev baseline |
|---|---|---|
| test_single_in_dist | 95.735 | −4.1% |
| test_geom_camber_rc | 89.726 | −6.8% |
| test_geom_camber_cruise | NaN (data/scoring.py bug — `inf * 0 = NaN`) | — |
| test_re_rand | 81.015 | −0.0% |

Sweep summary (all arms beat prior baseline 94.4199):

| Arm | ema_decay | W&B run | val_avg | Δ vs #3366 | test 3-split |
|---|---|---|---|---|---|
| A | 0.997 | `ml7l5jck` | 91.990 | −2.6% | 88.322 |
| B | 0.995 | `y5xumcvw` | 91.205 | −3.4% | 88.177 |
| **C (best)** | **0.99** | **`fzrq04xr`** | **90.613** | **−4.0%** | **88.825** |

All arms hit wall-clock cap at epoch 14 (monotone improvement — headroom remains).
Merged from PR #3474, student `willowpai2i48h2-alphonse`.

## Current best configuration

EMA (fast decay) + gradient clipping + Huber loss:
- **`ema_decay = 0.99`** ← updated from 0.999 (PR #3474)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")` replaces MSE element-wise loss
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`
- All other settings unchanged from prior config

**Key mechanistic finding:** At the 14-epoch wall-clock budget, decay=0.99 (half-life ~69 steps) outperforms decay=0.999 (half-life ~693 steps) because the shorter-lag shadow tracks recent parameter improvements in the late-training phase rather than lagging behind them. The shadow still smooths the last ~70 optimizer steps, which is enough to suppress per-step noise while reflecting the current basin. Improvement is monotone: 0.997 > 0.995 > 0.99 > 0.999 within this budget.

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~14 epochs)
- Loss: `F.huber_loss(delta=1.0)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

## Reproduce

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 1.0 \
  --ema_decay 0.99 \
  --wandb_group ema-decay0.99-clip5-huber \
  --wandb_name ema-decay0.99-clip5-huber \
  --agent <student>
```

## History

| Date | PR | val_avg/mae_surf_p | Δ | Notes |
|---|---|---|---|---|
| 2026-05-15 (seed) | ref run `07efagec` | 136.8873 | — | askeladd baseline-w1 reference arm |
| 2026-05-15 17:30 | #3186 fern EMA | 121.6850 | −11.10% | All 4 val splits improve; 3 reproducible runs |
| 2026-05-15 20:40 | #3366 fern EMA+clip+Huber | 94.4199 | −22.4% | All 4 val splits ≥−20%; 2 reproducible runs; val still monotone at epoch 14 |
| **2026-05-16 00:25** | **#3474 alphonse EMA decay=0.99** | **90.6131** | **−4.0%** | **All 3 arms beat baseline; monotone in decay direction; 3 runs; val monotone at ep14** |

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
