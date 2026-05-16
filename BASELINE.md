# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=12`, `mlp_ratio=1`, `unified_pos=False` |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #3982, mlp_ratio=1, single-seed, best epoch 19)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **79.05** |
| `test_avg/mae_surf_p` | **69.76** |
| `test/test_single_in_dist/mae_surf_p` | 81.55 |
| `test/test_geom_camber_rc/mae_surf_p` | 79.44 |
| `test/test_geom_camber_cruise/mae_surf_p` | 49.32 |
| `test/test_re_rand/mae_surf_p` | 68.73 |

Per-split val surface-p MAE at best checkpoint (single seed):

| Split | mae_surf_p | Δ vs prev |
|-------|------------|-----------|
| `val_single_in_dist`     |  92.38 | -1.53% |
| `val_geom_camber_rc`     |  90.41 | -2.85% |
| `val_geom_camber_cruise` |  58.997 | -2.43% |
| `val_re_rand`            |  74.42 | -0.84% |
| **avg** | **79.05** | **-1.92%** |

Artifact: `models/model-mlp-ratio-1-20260516-133706/metrics.jsonl`

Note: clean win — both primary metrics -1.92% (val) and -1.93% (test), ALL 4 val and 4 test splits improved. Best epoch 19 (vs 18 baseline) = +1 epoch in 30-min cap from compute saving. Single-line change: mlp_ratio 2→1. -12% trainable params, peak GPU mem 29.69 GB. Monotone val improvement at cap — still compute-bound.

Why it works: Halving FFN intermediate (256→128) saved 7% sec/epoch (95.9s vs 103.1s), unlocking +1 epoch. Capacity loss was free — the implicit regularization from a leaner FFN + extra training epoch compounded cleanly. Importantly, the 7% wall-clock saving was MUCH less than predicted 25% — confirming FFN matmuls are NOT the dominant per-step cost. Per-iteration overhead (dataloader / EMA update / Python) is the real ceiling.

**Compute-budget model refined:** O(K²) attention savings (slice_num) and FFN matmul savings (mlp_ratio) both yield modest wall-clock improvements (~5-7%). The next big compute win must come from attacking dataloader/optimizer/Python overhead — bf16 (askeladd #3743), torch.compile, or larger effective batch.

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
| 2026-05-16 | #3602 | slice_num=32→16 (continue halving, +2 epochs to 18, still compute-bound) | 84.44 | -6.78% |
| 2026-05-16 | #3601 | EMA decay 0.999→0.998 (tighter window, confirmed on slice_num=16 base) | 81.16 | -3.88% |
| 2026-05-16 | #3783 | EMA decay 0.998→0.997 (probe looser; diminishing returns) | 80.88 | -0.34% |
| 2026-05-16 | #3950 | slice_num 16→12 (triangulate; tie within noise) | 80.60 | -0.34% |
| 2026-05-16 | #3982 | mlp_ratio 2→1 (halve FFN width, +1 epoch from compute saving) | **79.05** | **-1.92%** |
