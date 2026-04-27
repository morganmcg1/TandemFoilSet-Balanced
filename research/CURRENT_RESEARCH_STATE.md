# SENPAI Research State

- **Date**: 2026-04-27 20:20 UTC
- **Most recent research direction from human researcher team**: None (infrastructure issue reported in #257 — GitHub label-index regression affecting student PR polling; no methodology directives)
- **Current research focus**: Round 1 experiments on TandemFoilSet CFD surrogate. First result in — EMA baseline at val_avg/mae_surf_p=133.66 (epoch 14/50, not converged). 7 students still WIP. Vanilla baseline (#193) still running.

## Current Baseline

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p | **133.66** | PR #209, epoch 14/50, EMA decay=0.999 |
| test_avg/mae_surf_p | **119.58** | NaN-corrected (bug fix now in train.py) |

**Critical**: The NaN-poisoning bug fix is now merged. All future experiments benefit automatically.

**Caveat**: The EMA baseline is not converged (14/50 epochs, monotonically improving). A true vanilla baseline (PR #193, alphonse) is still running. Once alphonse's results arrive, we will know whether EMA itself helps vs vanilla.

## Active Round 1 Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #193 | charliepai2c3-alphonse | Vanilla baseline anchor | WIP |
| #198 | charliepai2c3-askeladd | L1 loss with surf_weight=1 | WIP |
| #200 | charliepai2c3-edward | surf_weight sweep: 20 and 50 | WIP |
| #203 | charliepai2c3-fern | Wider Transolver n_hidden=256 | WIP |
| #207 | charliepai2c3-frieren | LR warmup + cosine to 1e-3 | WIP |
| #209 | charliepai2c3-nezuko | EMA weight averaging (decay=0.999) | MERGED — baseline |
| #214 | charliepai2c3-tanjiro | Per-channel pressure up-weighting (3x on p) | WIP |
| #219 | charliepai2c3-thorfinn | Per-channel decoder heads | WIP |

## Current Research Themes

1. **Loss formulation**: L1 vs L2 (#198), per-channel pressure up-weighting (#214)
2. **Hyperparameter tuning**: surf_weight sweep 10→20,50 (#200), LR warmup + cosine to 1e-3 (#207)
3. **Architecture exploration**: Wider model n_hidden=256 (#203), per-channel decoder heads (#219)
4. **Regularization**: EMA weight averaging (MERGED — now baseline)
5. **Reference**: Vanilla baseline anchor (#193) — will reveal EMA's true contribution

## Key Observations So Far

- val_geom_camber_cruise (OOD cruise camber) performs **best** at 100.14, while val_single_in_dist (in-dist) is worst at 171.74. This is counter-intuitive — investigate whether this is a data artifact or the EMA helping OOD more than in-dist.
- Mean epoch time: ~132 s → only ~14 epochs in 30-min window at baseline architecture. Any technique that reduces per-epoch time would be very valuable.
- VRAM at baseline EMA: 42 GB (out of 96 GB) — significant headroom for larger models or bigger batches.

## Potential Next Research Directions

**High priority (likely to beat baseline):**
- asinh/signed-log transform on p channel — compress heavy-tailed pressure values to better match MAE objective
- Surface-only loss (vol_weight=0) — focus all gradient signal on surface pressure
- Batch size increase (8 or 16) — more VRAM headroom exists, larger batches stabilize gradients
- Higher LR with EMA (1e-3) + cosine — EMA smooths gradient noise, may allow more aggressive LR
- Deeper model: n_layers=8 with n_hidden=128 — depth rather than width

**Architecture:**
- FiLM conditioning: inject Re + NACA params as global conditioning per layer
- Fourier/sinusoidal positional encoding for spatial coords (dims 0-1)
- Huber loss (smooth L1) — compromise between MSE stability and L1-MAE alignment
- SwiGLU activations in Transolver MLP blocks

**OOD focus:**
- Domain-adversarial training: penalize camber-split predictions that diverge from mean
- Data augmentation: random Re perturbation during training
- Contrastive geometry embedding: learn geometry-invariant flow features

## Key Constraints

- VRAM: 96 GB per GPU, meshes up to 242K nodes
- Timeout: ~30 min wall clock → ~14 epochs at baseline speed
- Epochs cap: controlled by SENPAI_MAX_EPOCHS env var
- Data loaders are read-only (only train.py is editable)
- Primary metric: val_avg/mae_surf_p (lower is better)
