# Baseline — `icml-appendix-charlie-pai2g-48h-r5`

This branch is the **Charlie no-W&B logging ablation, round 5 (charlie-pai2g-48h-r5)**.

Experiment metrics are written to local JSONL only (`models/<experiment>/metrics.jsonl`).
**Do not** add or query W&B / wandb experiment logging for this arm.

## Primary ranking metric

- **Validation:** `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE
  across the four val tracks (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` from the best-val checkpoint.

> ✅ **Round-5 scoring bug fixed (merged via PR #1532):** `test_geom_camber_cruise/000020.pt`
> contains ±Inf values in the `p` channel. The `train.py:evaluate_split` workaround
> (batch-level `y_finite_mask` filter before `accumulate_batch`) is now on the
> advisor branch. All subsequent PRs must include this fix on their branch and
> should report **finite `test_avg/mae_surf_p`**. Round-5 ranking remains
> `val_avg/mae_surf_p` as the primary metric.

## Reference configuration (train.py defaults)

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
)
optimizer = AdamW; scheduler = CosineAnnealingLR(T_max=epochs)
```

Each training execution is hard-capped by `SENPAI_TIMEOUT_MINUTES=30` (wall clock).
`--epochs 50` is an upper bound; runs typically reach 12-16 epochs under the
30-min cap at the default model size.

## Current best (val)

| Metric | Value | PR | Config | Notes |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **101.1212** | #1532 | bf16 AMP + scoring fix; else Smooth-L1 defaults | epoch 17 of 19; still improving at timeout |
| `test_avg/mae_surf_p` | **91.5013** | #1532 | — | first finite test avg on this branch |

All subsequent PRs must beat `val_avg/mae_surf_p < 101.1212` to be merged.

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 17 (wall-clock-bound at 30 min; model still improving at epoch 19)
- **Epochs reached:** 19 (~25% faster than fp32: ~98 s/epoch vs ~131 s)
- **Peak GPU memory:** 32.95 GB (well under 96 GB limit)

| Split | val mae_surf_p | Δ vs #1444 |
|---|---|---|
| `val_single_in_dist` | 120.0176 | -15.14 |
| `val_geom_camber_rc` | 107.0980 | -21.98 |
| `val_geom_camber_cruise` | 82.8425 | +5.14 |
| `val_re_rand` | 94.5268 | -6.57 |
| **val_avg** | **101.1212** | **-9.64** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 105.4434 |
| `test_geom_camber_rc` | 99.9931 |
| `test_geom_camber_cruise` | 69.2841 |
| `test_re_rand` | 91.2844 |
| **test_avg** | **91.5013** |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/bf16-amp-scoring-fix" \
      --epochs 50
  ```
  (bf16 AMP via `torch.autocast` + scoring workaround — see PR #1532 diff)

---

## 2026-05-12 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 14 (wall-clock-bound at 30 min; model still improving)
- **Peak GPU memory:** 42.1 GB
- **Time per epoch:** ~131 s

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 135.16 | 1.719 | 0.769 |
| `val_geom_camber_rc` | 129.08 | 2.104 | 0.988 |
| `val_geom_camber_cruise` | 77.70 | 1.047 | 0.555 |
| `val_re_rand` | 101.10 | 1.607 | 0.740 |
| **val_avg** | **110.76** | — | — |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/smooth-l1-loss" \
      --epochs 50
  ```
  (plus the Smooth-L1 substitution in `train.py` — see PR #1444 diff)

## Reproduce command (reference defaults)

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name "<student>/<short-description>" \
    --epochs 50
```

Commit `models/<experiment>/metrics.jsonl` and `metrics.yaml` with the PR and
quote the key values in the PR results comment plus the
`SENPAI-RESULT` terminal marker.
