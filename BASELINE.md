# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-15)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `unified_pos=False` |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=1.0)** in normalized space, `surf_weight=10.0` (PR #3111) |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` to avoid NaN-through-zero |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #3279 re-evaluation of SmoothL1 baseline, best epoch 13)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **108.47** |
| `test_avg/mae_surf_p` | **99.49** (finite — NaN bug fixed) |
| `test/test_geom_camber_rc/mae_surf_p` | 105.84 |
| `test/test_geom_camber_cruise/mae_surf_p` | 77.95 |
| `test/test_re_rand/mae_surf_p` | 98.77 |
| `test/test_single_in_dist/mae_surf_p` | 115.42 |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|------------|
| `val_single_in_dist`     | 128.55 |
| `val_geom_camber_rc`     | 116.22 |
| `val_geom_camber_cruise` |  87.91 |
| `val_re_rand`            | 101.21 |
| **avg** | **108.47** |

Artifact: `models/model-charliepai2i48h1-alphonse-nan-fix-verification-20260515-143359/metrics.jsonl`

Reproduce:

```bash
cd target/
python train.py --experiment_name smooth-l1-repro --agent <name>
# (SmoothL1 + NaN-safe scoring already in train.py + data/scoring.py on icml-appendix-charlie-pai2i-48h-r1)
```

### Note on val variance

The val delta between the two SmoothL1 measurements (PR #3111: 115.17 vs PR
#3279: 108.47, same config, different seeds/best-epoch realization) is ~7
points. Treat run-to-run variance as ±5-10 pts on `val_avg/mae_surf_p`.
Improvements smaller than that should be confirmed with a re-run.

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
| 2026-05-15 | #3279 | NaN-safe scoring (infra, also re-rolls val seed) | **108.47** | -5.8% (stochastic) |
