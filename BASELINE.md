# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=16`, `mlp_ratio=2`, `unified_pos=False` |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.999**, evaluated at val/test time (PR #3285) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #3602, slice_num=16, single-seed, best epoch 18)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **84.44** |
| `test_avg/mae_surf_p` | **74.75** |
| `test/test_single_in_dist/mae_surf_p` | 88.51 |
| `test/test_geom_camber_rc/mae_surf_p` | 83.91 |
| `test/test_geom_camber_cruise/mae_surf_p` | 53.62 |
| `test/test_re_rand/mae_surf_p` | 72.94 |

Per-split val surface-p MAE at best checkpoint (single seed):

| Split | mae_surf_p | Δ vs prev |
|-------|------------|-----------|
| `val_single_in_dist`     | 100.09 | -7.43% |
| `val_geom_camber_rc`     |  94.49 | -9.38% |
| `val_geom_camber_cruise` |  63.60 | -4.75% |
| `val_re_rand`            |  79.60 | -4.27% |
| **avg** | **84.44** | **-6.78%** |

Artifact: `models/model-charliepai2i48h1-askeladd-slice-num-16-20260516-032444/metrics.jsonl`

Note: single-seed merge; all 4 val splits improved. Best epoch 18 = final epoch (compute-bound, not capacity-bound). Per-epoch time 105.5s (-8.4% vs slice_num=32), peak GPU mem 35.27 GB. Val trajectory was monotonically improving at cap — consistent with slice_num=32's behavior.

Why it works: O(K²) attention cost drops 64→32→16: from 4096 → 1024 → 256 op-pairs. Each slice now covers ~1250 mesh nodes. Model is not capacity-limited at this resolution; the implicit regularization + extra training epochs (18 vs 16 vs 14) continues to compound. Expressiveness ceiling is below slice_num=16 for this mesh density.

Slice_num progression: 64→96.17, 32→90.58 (-5.81%), 16→84.44 (-6.78%). Next probe: slice_num=8.

Reproduce:

```bash
cd target/
python train.py --experiment_name slice-num-16-repro --agent <name>
# (slice_num=16 + SmoothL1 beta=0.25 + EMA-0.999 + dropout=0.1 all in train.py on icml-appendix-charlie-pai2i-48h-r1)
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
| 2026-05-15 | #3280 | SmoothL1 beta=0.5 (tuned from 1.0) | 98.45 | -5.81% |
| 2026-05-15 | #3400 | SmoothL1 beta=0.25 (2-seed mean; beta lever saturated) | 97.15 | -1.32% |
| 2026-05-15 | #3402 | dropout=0.1 in PhysicsAttention (8/8 split consistency) | 96.17 | -1.01% |
| 2026-05-16 | #3533 | slice_num=64→32 (halve slice-attention cost, +2 epochs, implicit reg) | 90.58 | -5.81% |
| 2026-05-16 | #3602 | slice_num=32→16 (continue halving, +2 epochs to 18, still compute-bound) | **84.44** | **-6.78%** |
