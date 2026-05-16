# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=12`, `mlp_ratio=1`, `unified_pos=False`, **FiLM head on [log_Re, AoA0, AoA1]** |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Precision | **bf16 autocast** (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) — PR #4064 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #4064, bf16 autocast on FiLM-Re+AoA, single-seed, best epoch 25)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **59.08** |
| `test_avg/mae_surf_p` | **51.29** |
| `test/test_single_in_dist/mae_surf_p` | 60.89 |
| `test/test_geom_camber_rc/mae_surf_p` | 63.00 |
| `test/test_geom_camber_cruise/mae_surf_p` | 32.91 |
| `test/test_re_rand/mae_surf_p` | 48.38 |

Per-split val surface-p MAE at best checkpoint (single seed, epoch 25):

| Split | mae_surf_p | Δ vs prev (68.80) |
|-------|------------|-----------|
| `val_single_in_dist`     |  69.49 | -13.8% |
| `val_geom_camber_rc`     |  68.90 | -14.1% |
| `val_geom_camber_cruise` |  40.32 | -15.7% |
| `val_re_rand`            |  57.60 | -13.4% |
| **avg** | **59.08** | **-14.1%** |

Artifact: `models/model-bf16-on-film-20260516-182629/metrics.jsonl`

Note: bf16 autocast delivers -27% sec/epoch (102s → 74s), enabling 25 epochs in the 30-min cap vs 18. The model was **still monotonically descending at epoch 25 (final)** — another compute savings compounding would push further. All 4 val and 4 test splits improved 13-16%. Zero NaN/Inf issues throughout; autocast keeps reduction ops in fp32 for stability.

Why it works: bf16 halves activation storage in attention/MLP intermediates, cutting both compute and memory bandwidth. The 7 extra epochs each produced ~1.4 pts/epoch improvement. The FiLM affines and LayerNorms remain numerically stable because PyTorch autocast keeps `sum` and `smooth_l1_loss` reductions in fp32.

**bf16 compute impact:** sec/epoch 102 → 74 (-27%), VRAM 32.2 → 23.5 GB (-27%). Headroom freed for larger batch or deeper model in subsequent experiments.

Reproduce:
```bash
cd target/
python train.py --experiment_name bf16-repro --agent <name>
# bf16: torch.autocast(device_type='cuda', dtype=torch.bfloat16) wraps forward+loss
```

Reproduce:

```bash
cd target/
python train.py --experiment_name film-re-aoa-repro --agent <name>
# FiLM: film_head = MLP(3 → 128 → 1280), cond = x[:, 0, [13, 14, 18]] (log_Re, AoA0, AoA1)
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
| 2026-05-16 | #4004 | FiLM-on-Re: condition each Transolver block on log(Re) scalar | 71.46 | -9.6% |
| 2026-05-16 | #4018 | FiLM-Re+AoA: expand conditioning to [log_Re, AoA0, AoA1] | 68.80 | -3.7% |
| 2026-05-16 | #4064 | bf16 autocast: -27% sec/epoch, 18→25 epochs in 30-min cap | **59.08** | **-14.1%** |
