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
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #3111, best epoch 13, 14 epochs total)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **115.17** |
| `test_avg/mae_surf_p` | NaN (scoring.py bug — see note) |
| `test/test_geom_camber_rc/mae_surf_p` | 111.81 |
| `test/test_re_rand/mae_surf_p` | 101.43 |
| `test/test_single_in_dist/mae_surf_p` | 125.70 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN (bad sample — see note) |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|------------|
| `val_single_in_dist`     | 144.61 |
| `val_geom_camber_rc`     | 124.04 |
| `val_geom_camber_cruise` |  89.33 |
| `val_re_rand`            | 102.70 |
| **avg** | **115.17** |

Artifact: `models/model-smooth-l1-loss-20260515-124521/metrics.jsonl`

Reproduce:

```bash
cd target/
python train.py --experiment_name smooth-l1-loss --agent <name>
# (SmoothL1 change is already in train.py on icml-appendix-charlie-pai2i-48h-r1)
```

### Calibration-only baseline (PR #3107, default config MSE)

For reference — this is the un-improved baseline, not the current winner:

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | 143.52 (epoch 11, 14 epochs run) |
| `test_avg/mae_surf_p` (NaN-safe recompute) | 130.34 |

Split: `val_single=181.35, val_rc=163.47, val_cruise=105.77, val_re_rand=123.49`

### Scoring bug note

`test_geom_camber_cruise/000020.pt` has non-finite pressure on 761 volume nodes.
`data/scoring.py`'s NaN-skip relies on masking but `NaN * 0 = NaN` in IEEE 754,
so the accumulator is poisoned even after the mask. **Val splits are clean.**
Use `val_avg/mae_surf_p` as the primary ranking metric; `test_avg/mae_surf_p`
will be NaN until the bug is fixed. The 3 clean test splits are still usable
as secondary comparisons.

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
| 2026-05-15 | #3111 | SmoothL1 loss (Huber beta=1.0) | **115.17** | **-19.7%** |
