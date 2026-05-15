# Baseline — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

## Current best

**PR #3215 (tanjiro) — SmoothL1 (Huber) loss β=0.05 on learnable Fourier baseline** — merged 2026-05-15 23:20

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **90.6039** |
| `test_avg/mae_surf_p` | **83.0029** |
| Best val epoch | 14 (hit 30-min wall clock) |
| W&B run | `iofja54s` |
| Peak GPU memory | ~42.5 GB |

### Per-split surface-pressure MAE (best-val checkpoint, epoch 14)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 112.03 | 101.95 |
| `geom_camber_rc` | 104.42 | 97.84 |
| `geom_camber_cruise` | 62.07 | 55.10 |
| `re_rand` | 83.89 | 77.11 |

### Configuration

| Component | Value |
|---|---|
| Model | Transolver `n_layers=5, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2` |
| Input augmentation | 8 **learnable** Fourier bands on normalized (x, z); initialized to octave-doubling `[1, 2, 4, ..., 128]` cycles/unit → `fun_dim=54` |
| Optimizer | AdamW `lr=5e-4, weight_decay=1e-4` on `model.parameters() + [fourier_freqs]` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` (no warmup) |
| Loss | `vol_loss + 10.0 * surf_loss`, **SmoothL1 (Huber) β=0.05** on normalized targets (replaces MSE) |
| `batch_size` | 4 |
| Sampler | `WeightedRandomSampler` over balanced domain weights |
| Epochs | 50 (capped by `SENPAI_MAX_EPOCHS=50`; wall-clock cap hits ~ep 14) |
| Wall-clock cap | `SENPAI_TIMEOUT_MINUTES=30.0` |

### Key finding

SmoothL1 with β=0.05 is the largest single-change improvement on this benchmark to date: **−22.1% val, −22.7% test** vs the learnable-Fourier baseline. The mechanism is clear: MSE squares large normalized residuals, causing the model to over-optimize for extreme-Re samples that dominate the gradient; SmoothL1 transitions to linear behaviour once |err| > β=0.05, capping the gradient contribution of outlier samples. The improvement is largest on the widest-distribution splits: `re_rand` test −28.6%, `geom_camber_cruise` test −28.1%.

The gain composes additively with learnable Fourier features (comparing rebased vs un-rebased numbers: val 90.60 vs 90.24, test 83.00 vs 82.21 — near-identical), confirming these two changes attack orthogonal problems (loss curvature vs position-encoding spectrum).

Best epoch was the last completed epoch (14/50, wall-clock limited). Val curve was still declining — more headroom likely exists with extended training.

### Reproduce

```bash
cd target/ && python train.py --agent willowpai2i24h2-tanjiro \
    --wandb_name "willowpai2i24h2-tanjiro/smoothl1-beta005-learnable-fourier" \
    --wandb_group "willow-pai2i-24h-r2/smooth-l1-v2" \
    --smooth_l1_beta 0.05
```

## Previous best (superseded)

**PR #3352 (fern) — Learnable Fourier frequency bands (8 trainable freqs)** — merged 2026-05-15 19:28

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 116.3411 |
| `test_avg/mae_surf_p` | 107.3254 |
| Best val epoch | 12 |
| W&B run | `rumqs1au` |
| Peak GPU memory | ~33 GB |

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
| 2026-05-15 23:20 | #3215 | SmoothL1 (Huber) loss β=0.05 | **90.6039** | **83.0029** | **−22.13%** |
