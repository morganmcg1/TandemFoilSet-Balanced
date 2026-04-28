# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #578 (charliepai2d3-thorfinn) — **L1 + 12-freq spatial FF +
EMA(0.997) + matched cosine + lr=7.5e-4 + grad clipping (max_norm=1.0)
+ decoupled head LR (2× on `mlp2` + `ln_3`)**. Run on pre-#572 advisor
(no aux log-p, no max_norm=5.0 yet — see caveat).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **75.78** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **66.27** |
| Per-epoch wallclock | ~132 s |
| Peak GPU memory (batch=4) | 42.51 GB |
| Wallclock total | ~32 min |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 84.61 |
| val_geom_camber_rc     | 85.83 |
| val_geom_camber_cruise | 58.09 |
| val_re_rand            | 74.58 |
| **val_avg**            | **75.78** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 72.57 |
| test_geom_camber_rc     | 75.92 |
| test_geom_camber_cruise | 49.56 |
| test_re_rand            | 67.02 |
| **test_avg**            | **66.27** |

**Critical caveat**: PR #578 was branched off pre-#572 advisor (no
aux log-p, max_norm=1.0). The post-merge advisor includes:
- aux log-p (weight=0.25) from PR #572
- max_norm=5.0 from PR #596
- decoupled head LR (2x) from this PR (#578)

The actual joint config (all stacked) is **untested** but expected
to land below 75.78 since PR #572 (+aux log-p) and PR #596
(+max_norm=5.0) both individually showed val improvements on their
assigned baselines.

**Mechanistic insight from PR #578**: the largest gains were on
`val_single_in_dist` (−7.18%) and `val_geom_camber_rc` (−5.45%),
opposite of the original prediction (which expected OOD-camber-cruise
to gain most). The askeladd #489 finding ("OOD-camber wants higher
LR") was incomplete — the actual story is **the head fits in-dist
patterns slowly under the conservative 7.5e-4 backbone LR**, and
giving the head 2× lets it converge in matched-cosine epochs without
dragging the backbone faster. `val_geom_camber_cruise` mildly
regressed (+3.44%) but remains the smallest-magnitude split.

**Recommended reproduce command**:

```bash
cd target/
python train.py --epochs 14 --lr 7.5e-4 --experiment_name baseline_ref
```

The post-merge advisor `train.py` has L1 + 12-freq FF + EMA(0.999) +
grad clipping baked in. The two CLI flags supply matched cosine
(`--epochs 14`) and the bumped peak LR (`lr=7.5e-4`).

## Round 3 progress

| Round | val | test | Lever | Δ vs prior |
|-------|----:|-----:|-------|--:|
| Pre-r3 | TBD | — | — | — |
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine `--epochs 14` (CLI) | −1.06% / −0.33% |
| PR #447 |  82.97 |  73.58 | + EMA(0.999) | **−8.7% / −9.0%** |
| PR #461 |  80.28 |  70.92 | + lr=7.5e-4 (CLI) | −3.2% / −3.6% |
| PR #462 |  80.06 |  70.04 | + grad clipping max_norm=1.0 | −0.27% / −1.24% |
| PR #506 |  78.80 |  69.13 | + NUM_FOURIER_FREQS=12 | −1.57% / −1.30% |
| PR #534 |  78.60 |  67.77 | + EMA_DECAY=0.997 (schedule × EMA fix) | −0.25% / −1.97% |
| PR #572 |  77.78 |  67.71 | + aux log-p loss (weight=0.25) | −1.06% / −0.09% |
| PR #596 |  77.01 |  67.78 | + max_norm=5.0 (loosened clip) | −0.99% / +0.10% |
| **PR #578 (current)** | **75.78** | **66.27** | **+ decoupled head LR (2×)** | **−1.60% / −2.23%** |

**Cumulative round-3 improvement: −43.9% on val, −46.2% on test.**

## Round-3 proven levers (cumulative — seven stacked levers)

1. **L1 surface loss** (PR #280)
2. **8→12-freq spatial Fourier features** (PR #400 → PR #506)
3. **Matched cosine `--epochs 14`** (PR #389, CLI)
4. **EMA-of-weights, decay=0.999** (PR #447)
5. **Peak LR `lr=7.5e-4`** (PR #461, CLI)
6. **Gradient clipping max_norm=1.0** (PR #462)
7. **NUM_FOURIER_FREQS=12** (PR #506) — refinement of lever #2.
8. **EMA_DECAY=0.997** (PR #534) — schedule × EMA interference fix.
9. **Auxiliary log-pressure loss (weight=0.25)** (PR #572) — per-split tradeoff lever.
10. **Loosened gradient clipping (max_norm=5.0)** (PR #596) — clip × LR joint optimum shift.
11. **Decoupled head LR (2× on `mlp2`+`ln_3`)** (PR #578) — head adapts faster than backbone.

The advisor `train.py` bakes in 1, 2, 4, 6, 7, 8, 9, 10, 11 by
default. Levers 3 and 5 are CLI flags (`--epochs 14 --lr 7.5e-4`).

## Compose pattern map (round-3 finding, comprehensive)

Round-3 PRs revealed multiple compose patterns:

| compose pattern | with FF/EMA | examples | result |
|----------------|---------|----------|:--|
| Distributional / trajectory averaging | additive | matched cosine + lr=7.5e-4 (#461), grad clipping (#462), FF freq bump (#506) | merged |
| Magnitude-based regulariser, small dose | additive | wd=5e-4 standalone (#469) | partial — saturates on full stack (#500) |
| Magnitude-based regulariser, large dose | destructive on rc-camber | wd=1e-3 (#437), beta2=0.95 (#446) | closed |
| Loss-shape regulariser | overlaps with EMA | L1-volume × EMA (#492) | closed |
| LR overshoot | regression | lr=1e-3 × EMA (#489) | closed |
| Direction-only-update cliff | under-convergence | max_norm=0.5 (#499), DropPath 0.1 (#501) | closed |
| Schedule × averaging interference | OOD regression | matched cosine × EMA (#476) | closed |
| Saturated regularisation overlap | no marginal value | wd=5e-4 × full stack (#500) | closed |
| Input encoding on already-rich features | net-flat / regression | log(Re) FF (#432) | closed |

**Round-5 assignment heuristic**:
- Prefer levers that are **distributional**, **trajectory-averaging**,
  or **mechanistically different** from existing regularisers.
- Magnitude-based regularisers (wd, beta2) compose with FF only at
  small doses; large doses interfere on rc-camber.
- Schedule × averaging × magnitude-regulariser interactions are
  non-trivial; the canonical 6-lever-with-EMA stack hides
  significant interference (matched cosine × EMA per PR #476).
- Per-split signal is load-bearing for compose decisions.

## Reference (unmodified Transolver) configuration

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `fun_dim` | `X_DIM - 2 + 4 * NUM_FOURIER_FREQS` = 22 + 48 = **70** (FF=12) |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, **MSE volume + L1 surface** |
| Input encoding | raw 24-d `x` + 12-frequency Fourier of `(x, z)` |
| Weight averaging | **EMA(decay=0.997)** at every step, swap for val/test eval |
| Gradient clipping | **`clip_grad_norm_(max_norm=5.0)`** before optimiser step |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Default LR | **5e-4** (override with `--lr 7.5e-4` for the round-3 best config) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
