# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=12`, `mlp_ratio=1`, `unified_pos=False`, **FiLM-on-Re head** |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #4004, FiLM-on-Re, single-seed, best epoch 18)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **71.46** |
| `test_avg/mae_surf_p` | **62.53** |
| `test/test_single_in_dist/mae_surf_p` | 72.86 |
| `test/test_geom_camber_rc/mae_surf_p` | 74.86 |
| `test/test_geom_camber_cruise/mae_surf_p` | 41.88 |
| `test/test_re_rand/mae_surf_p` | 60.52 |

Per-split val surface-p MAE at best checkpoint (single seed):

| Split | mae_surf_p | Δ vs prev |
|-------|------------|-----------|
| `val_single_in_dist`     |  83.22 | -9.9% |
| `val_geom_camber_rc`     |  81.69 | -9.6% |
| `val_geom_camber_cruise` |  50.61 | -14.2% |
| `val_re_rand`            |  70.32 | -5.5% |
| **avg** | **71.46** | **-9.6%** |

Artifact: `models/model-film-re-20260516-145147/metrics.jsonl`

Note: landmark win — largest single-PR improvement since the loss-function rounds. val -9.6%, test -10.4%, ALL 4 val and 4 test splits improved. val_single_in_dist (structural gap split): 92.38 → 83.22 (-9.9%). Training still monotonically descending at epoch 18 — 30-min cap is the hard constraint, not overfitting. Peak GPU mem 32.17 GB, +165K params from film_head.

Why it works: FiLM (Feature-wise Linear Modulation, Perez 2017) conditions every Transolver block on log(Re) via learned `(γ, β) = MLP(log_Re)`, applied as `(1+γ)·fx + β` before each block. Re is THE fundamental dimensionless parameter in fluid mechanics (sets viscous vs. inertial regime, boundary layer behavior, separation). Treating it as one of 24 input channels buries this signal; FiLM gives it first-class status in every layer. Identity init (γ=0, β=0 at epoch 0) means the model starts equivalent to baseline and freely learns to use conditioning. The global improvement across all 4 splits (not just val_single_in_dist as originally hypothesized) confirms that Re conditioning benefits every flow regime.

**+6.3% sec/epoch overhead** (102s vs 96s) due to per-block affine ops, costing -1 epoch (18 vs 19). Still a massive net win because the per-epoch improvement rate dominates.

Reproduce:

```bash
cd target/
python train.py --experiment_name film-re-repro --agent <name>
# FiLM-on-Re: film_head = MLP(1 → 128 → 1280) conditioning each of 5 Transolver blocks
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
| 2026-05-16 | #3602 | slice_num=32→16 (continue halving, +2 epochs to 18, still compute-bound) | 84.44 | -6.78% |
| 2026-05-16 | #3601 | EMA decay 0.999→0.998 (tighter window, confirmed on slice_num=16 base) | 81.16 | -3.88% |
| 2026-05-16 | #3783 | EMA decay 0.998→0.997 (probe looser; diminishing returns) | 80.88 | -0.34% |
| 2026-05-16 | #3950 | slice_num 16→12 (triangulate; tie within noise) | 80.60 | -0.34% |
| 2026-05-16 | #3982 | mlp_ratio 2→1 (halve FFN width, +1 epoch from compute saving) | 79.05 | -1.92% |
| 2026-05-16 | #4004 | FiLM-on-Re: condition each Transolver block on log(Re) scalar | **71.46** | **-9.6%** |
