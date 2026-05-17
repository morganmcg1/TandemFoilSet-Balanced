# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-17)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, **`slice_num=8` (PR #4107)**, **`mlp_ratio=2` (PR #4282)**, `unified_pos=False`, **FiLM head on [log_Re, AoA0, AoA1]**, **GEGLU FFN (PR #4105)** |
| FFN width | **`mlp_ratio=2` effective** (PR #4282 fixed dead-code bug: `GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=int(hidden_dim * mlp_ratio))`) — inner GEGLU projection now 256-d instead of 128-d |
| Compile | **`torch.compile(model, dynamic=True, mode="default")`** (PR #4069) — fuses FiLM affine + GEGLU gate + QKV projections; `dynamic=True` required for pad_collate variable-length batches |
| Optim | **Schedule-Free AdamW** `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — PR #4071; NO LR scheduler |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783); EMA built before compile so `ema_model.module` is uncompiled |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Precision | **bf16 autocast** (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) — PR #4064 |
| FFN | **GEGLU gating** in Transolver block MLP (`FFN(x) = W2(GELU(W1a(x)) * W1b(x))`) with **inner_dim=256** (mlp_ratio=2) — PR #4105 + PR #4282 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run; use `load_target = getattr(model, "_orig_mod", model)` to load state dict after compile |

## Current best metrics (PR #4282, mlp_ratio=2 fix on compile+bf16+GEGLU+SF+slice=8, single-seed, best epoch 37)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **36.13** |
| `test_avg/mae_surf_p` | **31.97** |
| `test/test_single_in_dist/mae_surf_p` | 36.53 |
| `test/test_geom_camber_rc/mae_surf_p` | 44.62 |
| `test/test_geom_camber_cruise/mae_surf_p` | 17.23 |
| `test/test_re_rand/mae_surf_p` | 29.50 |

Per-split val surface-p MAE at best checkpoint (single seed, epoch 37):

| Split | mae_surf_p | Δ vs prev (37.31) |
|-------|------------|-----------|
| `val_single_in_dist`     |  36.67 | -1.4% |
| `val_geom_camber_rc`     |  48.15 | -4.7% |
| `val_geom_camber_cruise` |  21.37 | -0.5% |
| `val_re_rand`            |  38.34 | -4.4% |
| **avg** | **36.13** | **-3.2%** |

Artifact: `models/model-mlp-ratio-2-with-geglu-on-compile-stack-20260517-015040/metrics.jsonl`

**mlp_ratio=2 fix (dead-code bug fix + actual capacity increase):**
- n_params: 736,831 → **983,871** (+33.6% — GEGLU inner dim 128→256, weights in w1/w2)
- Per-epoch wall-clock: 42.4s → **47.76s** (+12.6% — expected from larger FFN)
- Epochs in 30-min cap: 42 → **37** (−5 epochs — cost of extra capacity)
- Peak VRAM: 18.88 GB → **22.61 GB** (+19.7% — larger activations from 2× FFN width)
- All 4 val splits improved: single −1.4%, rc −4.7%, cruise −0.5%, re_rand −4.4%
- Val still descending at epoch 37; more compute would push further

**The dead-code bug:** Previous PR #3982 (mlp_ratio 2→1) and all subsequent tests were
running `GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=hidden_dim)` regardless of mlp_ratio
config value — `mlp_ratio` was passed to `TransolverBlock.__init__` but never forwarded.
Fix in PR #4282: `GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=int(hidden_dim * mlp_ratio))`.
This means the previous "mlp_ratio=1 closed" finding was invalid — the axis was never properly
tested. mlp_ratio=2 now represents a genuine open axis with 36.13 as the new optimum.

**Key implementation note (GEGLUBlock instantiation):**
```python
# In TransolverBlock.__init__:
self.mlp = GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=int(hidden_dim * mlp_ratio))
# mlp_ratio=2: inner_dim=256; mlp_ratio=1: inner_dim=128
```

**Key implementation notes:**
```python
# After EMA wrap, before training loop:
try:
    model = torch.compile(model, dynamic=True, mode="default")
except Exception as e:
    print(f"torch.compile: FAILED, falling back to eager — {e}")

# On checkpoint reload for test eval:
load_target = getattr(model, "_orig_mod", model)
load_target.load_state_dict(torch.load(model_path, ...))

# SF mode switches still required:
optimizer.train()   # before each training step
optimizer.eval()    # before val/test (including after checkpoint load)
```

Reproduce:
```bash
cd target/
pip install schedulefree
python train.py --experiment_name mlp-ratio-2-repro --agent <name>
```

### Note on val variance

Single-seed variance is ±5-10 pts on `val_avg/mae_surf_p`. Improvements
smaller than ~5% may be within noise — confirm with a re-run if uncertain.

**Total improvement from calibration baseline:** 143.52 → 36.13 = **-74.8%**

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
| 2026-05-16 | #4105 | GEGLU FFN: gating projection replaces vanilla MLP, all 4+4 splits improved 9-19% | 50.57 | -14.4% |
| 2026-05-16 | #4071 | Schedule-Free AdamW: eliminates cosine T_max fragility, all 8 splits improved 7-18% | 45.07 | -10.9% |
| 2026-05-16 | #4107 | slice_num 12→8 on bf16+GEGLU+SF: -8.7% sec/epoch → +2 epochs, 3/4 splits improved, rc-split flip from regress to win | 43.82 | -2.78% |
| 2026-05-17 | #4069 | torch.compile(dynamic=True) on bf16+GEGLU+SF+slice=8: -41.3% sec/epoch (72→42s), 25→42 epochs, all 8 splits improved | 37.31 | -14.9% |
| 2026-05-17 | #4282 | mlp_ratio=2 fix (dead-code bug: GEGLUBlock now uses int(hidden_dim*mlp_ratio)); +33.6% params, 983k, all 4 splits improved | **36.13** | **-3.2%** |
