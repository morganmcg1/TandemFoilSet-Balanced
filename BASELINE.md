# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-15)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `unified_pos=False` |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.5)** in normalized space, `surf_weight=10.0` (PR #3280 — tuned from beta=1.0 of #3111) |
| EMA   | **Polyak averaging, decay=0.999**, evaluated at val/test time (PR #3285) |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #3280, SmoothL1 beta=0.5, best epoch 14)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **98.45** |
| `test_avg/mae_surf_p` | **87.63** |
| `test/test_geom_camber_rc/mae_surf_p` | 94.91 |
| `test/test_re_rand/mae_surf_p` | 86.17 |
| `test/test_single_in_dist/mae_surf_p` | 106.01 |
| `test/test_geom_camber_cruise/mae_surf_p` | 63.44 |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|------------|
| `val_single_in_dist`     | 119.70 |
| `val_geom_camber_rc`     | 108.17 |
| `val_geom_camber_cruise` |  74.09 |
| `val_re_rand`            |  91.84 |
| **avg** | **98.45** |

Artifact: `models/model-charliepai2i48h1-askeladd-smooth-l1-beta05-20260515-173606/metrics.jsonl`

Reproduce:

```bash
cd target/
python train.py --experiment_name smooth-l1-beta05-repro --agent <name>
# (SmoothL1 beta=0.5 + EMA-0.999 + NaN-safe scoring all in train.py on icml-appendix-charlie-pai2i-48h-r1)
```

### Note on val variance

Single-seed variance is ±5-10 pts on `val_avg/mae_surf_p`. Improvements
smaller than ~5% may be within noise — confirm with a re-run if uncertain.

### Calibration-only baseline (PR #3107, default config MSE)

For reference — this is the un-improved baseline, not the current winner:

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | 143.52 (epoch 11, 14 epochs run) |
| `test_avg/mae_surf_p` (NaN-safe recompute) | 130.34 |

Split: `val_single=181.35, val_rc=163.47, val_cruise=105.77, val_re_rand=123.49`

## Primary ranking metric

`val_avg/mae_surf_p` for checkpoint selection; `test_avg/mae_surf_p` for
paper-facing ranking. Lower is better. Equal-weight mean of surface-pressure
MAE across the four val/test splits in physical (denormalized) units.

## How this file is updated

After every merged winner, the advisor:
1. Replaces the "Current best" block with the new PR's `val_avg/mae_surf_p`
   and `test_avg/mae_surf_p` (and the per-split surface-p MAE table).
2. Appends a one-line entry under "History" with PR #, hypothesis tag, and the
   new score.

## History

| Date | PR | Hypothesis | val_avg/mae_surf_p | Δ |
|------|----|------------|--------------------|---|
| 2026-05-15 | #3107 | baseline (MSE, default config) | 143.52 | — (calibration) |
| 2026-05-15 | #3111 | SmoothL1 loss (Huber beta=1.0) | 115.17 | -19.7% |
| 2026-05-15 | #3279 | NaN-safe scoring (infra, also re-rolls val seed) | 108.47 | -5.8% (stochastic) |
| 2026-05-15 | #3285 | EMA model weights, decay=0.999 | 104.52 | -3.6% |
| 2026-05-15 | #3280 | SmoothL1 beta=0.5 (tuned from 1.0) | **98.45** | **-5.81%** |
