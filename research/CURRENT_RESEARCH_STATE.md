# SENPAI Research State

- 2026-05-12 19:00
- No human researcher directives (no open issues)
- Round 5 of the Charlie no-W&B logging ablation arm (SENPAI_TIMEOUT_MINUTES=30, local JSONL only)

## Round 5 update — what we now know

### Compute budget reality

The baseline Transolver completes **~14 epochs in 30 minutes** (~130 s/epoch on 1× RTX PRO 6000 Blackwell with FP32, batch=4, n_hidden=128, slice_num=64). This is *much* less than the 50-epoch default in `train.py`, and it makes two things newly important:

1. **Cosine LR schedule (T_max=50) barely decays** in 14 epochs. LR at epoch 14 is still ~82% of peak. Any hypothesis that depends on the LR valley (e.g., SWA) needs `--epochs` matched to the budget.
2. **Multi-epoch hypotheses** (deeper model, mixed precision-induced speedup, SWA, EMA of weights) become more valuable than at the assumed 50-epoch budget.

### Informal baseline floor

The implicit baseline from PR #1463 (SWA never engaged → effectively surf_weight=10 baseline over 14 epochs):

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (epoch 14) | **125.20** |
| test_avg/mae_surf_p | NaN (cruise pressure overflow) |

This is **not a merged baseline** — but it's the only round-5 number we have for comparison.

### Test-time numerical instability (baseline pathology)

Both #1459 and #1463 baseline-equivalent runs hit `mae_surf_p = NaN` on `test_geom_camber_cruise` because the model occasionally outputs Inf/NaN pressure predictions, and `data/scoring.py` does not skip non-finite predictions (only non-finite GT). Since `data/` is read-only, the fix lives in `train.py`:
- `torch.nan_to_num(pred_orig, nan=0.0, posinf=1e6, neginf=-1e6)` in `evaluate_split` before scoring.
- Pin a seed for reproducibility.

Both fixes are now part of #1463's revised assignment (askeladd's rerun).

## Current state of PRs

### Closed
- **#1459 alphonse / surf_weight=20:** 8.4% regression on val vs implicit baseline. Closed.

### WIP awaiting first results
- **#1463 askeladd / SWA, revised:** SWA_START_EPOCH=8, --epochs 14, seed pinned, nan_to_num guard.
- **#1470 edward / instance-norm loss**
- **#1474 fern / per-channel p-weighting (3× on p)**
- **#1478 frieren / wider model n_hidden=192**
- **#1481 nezuko / slice_num=128**
- **#1483 tanjiro / gradient clipping max_norm=1.0**
- **#1487 thorfinn / surface skip branch**

### Idle students (about to be re-assigned)
- charliepai2g24h5-alphonse — being assigned H10 (warmup + cosine matched to 14 epochs)
- charliepai2g24h5-askeladd — has #1463 WIP (no new assignment needed)

## Current research focus

1. **Establish a clean, reproducible round-5 baseline floor with valid test number.** PR #1463 rerun should produce this.
2. **Test orthogonal architecture and loss changes** within the 30-min budget to find what compounds.
3. **Address the test pressure NaN universally** by promoting the train.py-side `nan_to_num` guard into every subsequent experiment as part of its hardening.

## Potential next research themes

When students go idle, queue:

1. **H11 — batch=8 + BF16 mixed precision:** doubles effective throughput, could enable 20–25 epochs in 30 min instead of 14. Highest expected gain given the budget reality.
2. **H12 — Transolver++ local adaptive correction:** moderate engineering, highest expected accuracy gain from the literature.
3. **H9 deferred — n_layers=7 with --epochs matched.** Skip if H11 wins (BF16 may be a precondition).
4. **Compose winning changes from round 5.** If #1474 (per-channel) and #1483 (grad clip) both win, the next PR is "p-weight + grad clip + nan_to_num + seed" as the new starting point.
5. **EMA of weights** as a cheaper alternative to SWA — applies gradient-by-gradient, no warmup-to-valley requirement.
6. **Per-domain loss weighting:** if val_re_rand or val_geom_camber_cruise behave differently, weight their effective loss contribution explicitly via the WeightedRandomSampler weights.

## Pinned principles for round 5

- Treat the 30-min cap as a hard constraint. Design experiments with achievable epochs (≤ 14 at baseline cost) in mind.
- Pin seeds in every new PR (`torch.manual_seed(42)` plus numpy/random/cudnn deterministic).
- Add `torch.nan_to_num` guard on predictions in `evaluate_split` until a baseline-clean test number is established.
- Do not merge any PR with NaN in `test_avg/mae_surf_p` — even if `val_avg/mae_surf_p` is improved — per program.md fidelity rule.
