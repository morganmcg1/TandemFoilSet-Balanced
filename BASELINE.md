# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=12`, `mlp_ratio=1`, `unified_pos=False`, **FiLM head on [log_Re, AoA0, AoA1]**, **GEGLU FFN (PR #4105)** |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Precision | **bf16 autocast** (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) — PR #4064 |
| FFN | **GEGLU gating** in Transolver block MLP (`FFN(x) = W2(GELU(W1a(x)) * W1b(x))`) — PR #4105 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #4105, GEGLU FFN on bf16 FiLM-Re+AoA, single-seed, best epoch 23)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **50.57** |
| `test_avg/mae_surf_p` | **43.94** |
| `test/test_single_in_dist/mae_surf_p` | 49.90 |
| `test/test_geom_camber_rc/mae_surf_p` | 56.89 |
| `test/test_geom_camber_cruise/mae_surf_p` | 26.45 |
| `test/test_re_rand/mae_surf_p` | 42.52 |

Per-split val surface-p MAE at best checkpoint (single seed, epoch 23):

| Split | mae_surf_p | Δ vs prev (59.08) |
|-------|------------|-----------|
| `val_single_in_dist`     |  56.18 | -19.2% |
| `val_geom_camber_rc`     |  63.01 |  -8.5% |
| `val_geom_camber_cruise` |  32.57 | -19.2% |
| `val_re_rand`            |  50.52 | -12.3% |
| **avg** | **50.57** | **-14.4%** |

Artifact: `models/model-charliepai2i48h1-frieren-geglu-ffn-on-bf16-20260516-194450/metrics.jsonl`

Note: GEGLU replaces the standard `Linear → GELU → Linear` FFN with gated `W2(GELU(W1a(x)) * W1b(x))` (Shazeer 2020 arXiv:2002.05202). The gate enables per-position, per-channel feature selection — the model learns "this channel matters at this kind of mesh point" rather than activating uniformly. n_params 492K → 737K (+50%, all in FFN stack). Per-epoch cost +6% (74.4s → 78.9s). Trained 23 epochs in 30-min cap vs baseline's 25 — losing 2 epochs but gaining ~1.7 pts per remaining epoch and still descending hard at terminal epoch. Largest gains on `val_single_in_dist` and `val_geom_camber_cruise` (-19.2% each), exactly the regimes where pressure magnitudes vary most. Zero NaN/Inf throughout.

Why it works: surface-pressure distributions vary dramatically by regime (high-Re vs low-Re samples differ by an order of magnitude in y-std; surface vs interior nodes have wildly different statistics). The gating projection lets the FFN gate features rather than uniformly transforming them, providing implicit "experts" within each block. The mechanism is orthogonal to bf16 (precision/bandwidth), FiLM (broadcast-scalar conditioning), and EMA (averaging), so it compounds cleanly with prior wins.

**GEGLU compute impact:** sec/epoch 74.4 → 78.9 (+6%), VRAM ~23.5 → 25.7 GB (+9%). Still well under the 96 GB cap; headroom remains for further experiments.

Reproduce:
```bash
cd target/
python train.py --experiment_name geglu-repro --agent <name>
# GEGLU: PhysicsAttention.mlp = MLP_GEGLU(n_hidden, n_hidden*mlp_ratio, n_hidden)
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
| 2026-05-16 | #4064 | bf16 autocast: -27% sec/epoch, 18→25 epochs in 30-min cap | 59.08 | -14.1% |
| 2026-05-16 | #4105 | GEGLU FFN: gating projection replaces vanilla MLP, all 4+4 splits improved 9-19% | **50.57** | **-14.4%** |
