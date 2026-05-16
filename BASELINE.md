# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Current best configuration (merged as of 2026-05-16)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, **`slice_num=8` (PR #4107)**, `mlp_ratio=1`, `unified_pos=False`, **FiLM head on [log_Re, AoA0, AoA1]**, **GEGLU FFN (PR #4105)** |
| Optim | **Schedule-Free AdamW** `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — PR #4071; NO LR scheduler |
| Loss  | **SmoothL1 (Huber, beta=0.25)** in normalized space, `surf_weight=10.0` (PR #3400) |
| EMA   | **Polyak averaging, decay=0.997**, evaluated at val/test time (PR #3783) |
| Dropout | **dropout=0.1** in PhysicsAttention (attn + to_out) — PR #3402 |
| Precision | **bf16 autocast** (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) — PR #4064 |
| FFN | **GEGLU gating** in Transolver block MLP (`FFN(x) = W2(GELU(W1a(x)) * W1b(x))`) — PR #4105 |
| Scoring | NaN-safe accumulators (PR #3279) — `torch.where` instead of `mask * err` |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val EMA checkpoint evaluated on 4 test splits at end of run |

## Current best metrics (PR #4107, slice_num 12→8 on bf16+GEGLU+SF, single-seed, best epoch 25)

**Beat this to be a winner.**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` **(primary)** | **43.82** |
| `test_avg/mae_surf_p` | **38.05** |
| `test/test_single_in_dist/mae_surf_p` | 42.38 |
| `test/test_geom_camber_rc/mae_surf_p` | 50.51 |
| `test/test_geom_camber_cruise/mae_surf_p` | 22.71 |
| `test/test_re_rand/mae_surf_p` | 36.58 |

Per-split val surface-p MAE at best checkpoint (single seed, epoch 25):

| Split | mae_surf_p | Δ vs prev (45.07) |
|-------|------------|-----------|
| `val_single_in_dist`     |  47.39 | -2.87% |
| `val_geom_camber_rc`     |  55.44 | -5.35% |
| `val_geom_camber_cruise` |  26.97 | +0.94% (tied) |
| `val_re_rand`            |  45.50 | -1.55% |
| **avg** | **43.82** | **-2.78%** |

Artifact: `models/model-slice-num-8-on-bf16-geglu-20260516-215247/metrics.jsonl`

Note: slice_num=8 reduces the slice-attention projection cost (O(N·S²)) by 33% relative to slice=12. The halving trajectory (64→32→16→12→8) has gained on every step. On bf16+GEGLU+SF, the per-epoch saving (-8.7%, 79.2→72.3s) translates to +2 epochs (23→25) of training; SF keeps full LR on those extra epochs. Critical validation: the rc-split that *regressed* on the bf16-only baseline (+3.93% test) now *improves* on the full stack (-2.09% test) — GEGLU + SF give enough capacity headroom that the slice budget reduction is tolerated even on the hardest OOD geometry. Run still descending at epoch 25 (-0.71 pts/epoch at terminal).

slice_num halving trajectory (every step a merge):
- 64→32 (PR #3533): -5.81%
- 32→16 (PR #3602): -6.78%
- 16→12 (PR #3950): -0.34%
- **12→8 (PR #4107): -2.78%**

12→8 out-improved 16→12 — suggests optimum is below 8 (slice_num=6 is the next probe).

Note: Schedule-Free AdamW (Defazio et al. 2024, arXiv:2405.15682) replaces cosine annealing with an optimizer that maintains its own iterate average — no `T_max` required. The key insight: with `T_max=50` and only 23 effective epochs, cosine LR at epoch 23 is at ~59% of peak (i.e. `0.5*(1+cos(23/50*π))*lr ≈ 0.59*lr`), so the last ~10 epochs of the GEGLU baseline were making under-powered gradient steps. Schedule-Free keeps the effective LR at full strength right up to the timeout, giving late epochs ~1.5-2× larger updates. Zero compute overhead: 79.2 s/epoch vs 78.9 s baseline (+0.4%); peak VRAM 25.96 GB (+0.3% vs GEGLU baseline's 25.7 GB). All 8 val+test splits improved. EMA (decay=0.997) and SF coexist cleanly — SF operates on the optimizer iterate, EMA on shadow model parameters, no competition. Run was still descending at epoch 23 (-0.79 pts/epoch at terminal), so there remains headroom.

Why it works: the cosine T_max fragility grows with each compute-saving win (bf16, n_layers, compile each silently shrink effective epochs vs the T_max budget). Schedule-Free eliminates this hidden coupling by construction — the optimizer is portable across compute budgets. The LR schedule was the hidden bottleneck limiting late-epoch gradient quality.

**SF key implementation detail:** call `optimizer.train()` before each training step and `optimizer.eval()` before val/test evaluation; omitting these switches silently evaluates at the wrong iterate and negates the win.

Reproduce:
```bash
cd target/
pip install schedulefree
python train.py --experiment_name sf-repro --agent <name>
# SF: optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=5e-4, weight_decay=1e-4, warmup_steps=200)
# Remove cosine scheduler entirely. Add optimizer.train()/optimizer.eval() around train/val steps.
```

### Note on val variance

Single-seed variance is ±5-10 pts on `val_avg/mae_surf_p`. Improvements
smaller than ~5% may be within noise — confirm with a re-run if uncertain.

**Total improvement from calibration baseline:** 143.52 → 43.82 = **-69.5%**

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
| 2026-05-16 | #4107 | slice_num 12→8 on bf16+GEGLU+SF: -8.7% sec/epoch → +2 epochs, 3/4 splits improved, rc-split flip from regress to win | **43.82** | **-2.78%** |
