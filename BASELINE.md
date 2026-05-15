# Baseline — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

## Current best

**PR #3352 (fern) — Learnable Fourier frequency bands (8 trainable freqs)** — merged 2026-05-15 19:28

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **116.3411** |
| `test_avg/mae_surf_p` | **107.3254** |
| Best val epoch | 12 (hit 30-min wall clock) |
| W&B run | `rumqs1au` |
| Peak GPU memory | ~33 GB |

### Per-split surface-pressure MAE (best-val checkpoint, epoch 12)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 145.03 | 126.46 |
| `geom_camber_rc` | 126.25 | 118.24 |
| `geom_camber_cruise` | 88.12 | 76.60 |
| `re_rand` | 105.96 | 108.00 |

### Configuration

| Component | Value |
|---|---|
| Model | Transolver `n_layers=5, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2` |
| Input augmentation | 8 **learnable** Fourier bands on normalized (x, z); initialized to octave-doubling `[1, 2, 4, ..., 128]` cycles/unit → `fun_dim=54` |
| Optimizer | AdamW `lr=5e-4, weight_decay=1e-4` on `model.parameters() + [fourier_freqs]` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` (no warmup) |
| Loss | `vol_loss + 10.0 * surf_loss`, MSE on normalized targets |
| `batch_size` | 4 |
| Sampler | `WeightedRandomSampler` over balanced domain weights |
| Epochs | 50 (capped by `SENPAI_MAX_EPOCHS=50`; wall-clock cap hits ~ep 12–14) |
| Wall-clock cap | `SENPAI_TIMEOUT_MINUTES=30.0` |

### Key finding

The 8 learned frequencies barely moved from their octave-doubling initialization (max drift 2.47% on freq_0: 1.000 → 1.025; all others < 1%). The improvement over fixed Fourier (−4.24% val, −4.59% test) is attributed to the extra gradient signal through the frequency parameters during optimization rather than discovery of a qualitatively different frequency basis. The octave-doubling init is empirically near-optimal for this dataset.

Largest per-split gains vs PR #3200: `geom_camber_rc` −9.0% val / −11.3% test; `geom_camber_cruise` −5.8% val / −7.8% test. `single_in_dist` regressed +3.7% — net improvement driven by OOD geometry splits.

### Reproduce

```bash
cd target/ && python train.py --agent willowpai2i24h2-fern \
    --wandb_name "willowpai2i24h2-fern/learnable-fourier-freqs-8bands" \
    --wandb_group "willow-pai2i-24h-r2/learnable-fourier"
```

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
