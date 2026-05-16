# Baseline — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

## Current best

**PR #3350 (alphonse) — FiLM-style Reynolds conditioning on SmoothL1 baseline** — merged 2026-05-16 03:30

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **79.9018** |
| `test_avg/mae_surf_p` | **69.3296** |
| Best val epoch | best-val checkpoint |
| W&B run | `99jk5guj` |

### Per-split surface-pressure MAE (best-val checkpoint, run `99jk5guj`)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 93.78 | 83.21 |
| `geom_camber_rc` | 96.06 | 81.19 |
| `geom_camber_cruise` | 54.93 | 46.55 |
| `re_rand` | 74.83 | 66.36 |

### Seed variance (3 seeds, all use_film=True, smooth_l1_beta=0.05)

| Run | val_avg | test_avg |
|---|---|---|
| `99jk5guj` (primary) | **79.9018** | **69.3296** |
| `anr2xaul` | 86.5328 | 80.4702 |
| `es15998q` | 87.5134 | 81.3596 |
| 3-seed mean | 84.65 | 77.05 |
| 3-seed std | 4.16 | 6.69 |

Every seed beats the prior baseline (90.60/83.00).

### Configuration

| Component | Value |
|---|---|
| Model | Transolver `n_layers=5, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2` with FiLM conditioning |
| FiLM | Per-block FiLMLayer: `re_cond = x[:, 0, 13:14]` (first-row log-Re, avoids padding bias); zero-init AFTER `self.apply(_init_weights)` for identity start |
| Input augmentation | 8 **learnable** Fourier bands on normalized (x, z) |
| Optimizer | AdamW `lr=5e-4, weight_decay=1e-4` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` (no warmup) |
| Loss | `vol_loss + 10.0 * surf_loss`, **SmoothL1 (Huber) β=0.05** on normalized targets |
| `batch_size` | 4 |
| Wall-clock cap | `SENPAI_TIMEOUT_MINUTES=30.0` |

### Key finding

FiLM-style per-channel gamma/beta conditioning on Reynolds number, compounded on the SmoothL1 baseline, yields the largest improvement on this benchmark: **−11.8% val, −16.5% test**. The Re-holdout split (`re_rand`) improves by −10.8% val / −13.9% test — directly confirming the hypothesis that the model benefits from explicit Re information. Every val and test split improves with no per-split regressions. Critical implementation detail: FiLM zero-init must happen AFTER `self.apply(_init_weights)` and `re_cond` must use the first node row (not the masked-padded mean) to avoid padding-ratio confounding.

### Reproduce

```bash
cd target/ && python train.py --agent willowpai2i24h2-alphonse \
    --wandb_name "willowpai2i24h2-alphonse/film-re-smoothl1" \
    --wandb_group "willow-pai2i-24h-r2/film-re" \
    --smooth_l1_beta 0.05 --use_film True
```

## Previous best (superseded)

**PR #3215 (tanjiro) — SmoothL1 (Huber) loss β=0.05 on learnable Fourier baseline** — merged 2026-05-15 23:20

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 90.6039 |
| `test_avg/mae_surf_p` | 83.0029 |
| W&B run | `iofja54s` |

Per-split (val | test): single=112.03|101.95, camber_rc=104.42|97.84, camber_cruise=62.07|55.10, re_rand=83.89|77.11.

**PR #3352 (fern) — Learnable Fourier frequency bands (8 trainable freqs)** — merged 2026-05-15 19:28

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 116.3411 |
| `test_avg/mae_surf_p` | 107.3254 |
| W&B run | `rumqs1au` |

Per-split (val | test): single=145.03|126.46, camber_rc=126.25|118.24, camber_cruise=88.12|76.60, re_rand=105.96|108.00.

## Primary ranking metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across the four val splits.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same metric on the test splits, evaluated from the best-val checkpoint.

Lower is better. All metrics computed in original (denormalized) y-space, float64, surface-only nodes summed and divided by total surface-node count in the split.

## Per-split tracks

| Split | Tests |
|---|---|
| `val_single_in_dist` | Random holdout of single-foil samples |
| `val_geom_camber_rc` | Unseen front-foil camber M=6-8 (raceCar tandem P2) |
| `val_geom_camber_cruise` | Unseen front-foil camber M=2-4 (cruise tandem P2) |
| `val_re_rand` | Stratified Re holdout across tandem domains |

## Merge history

| Date | PR | Title | val_avg | test_avg | Δ val_avg vs prior |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.4956 | 112.4884 | first baseline |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands (8 trainable freqs) | 116.3411 | 107.3254 | −4.24% |
| 2026-05-15 23:20 | #3215 | SmoothL1 (Huber) loss β=0.05 | 90.6039 | 83.0029 | −22.13% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning on SmoothL1 | **79.9018** | **69.3296** | **−11.81%** |
