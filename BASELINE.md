# BASELINE — icml-appendix-charlie-pai2i-24h-r1

Active advisor branch baseline. Updated after each merged winner. All
val/test MAE numbers below come from the committed `models/<experiment>/metrics.jsonl`
on the listed PR.

## Current best — PR #3863 (charliepai2i24h1-fern / grad-clip-1p0)

- **val_avg/mae_surf_p**: **89.5987** (best at epoch 18; 18 of 18 epochs realized)
- **test_avg/mae_surf_p**: **78.4928** (NaN-safe 4-split)
- **Per-val-split mae_surf_p** (best epoch 18):
  - val_single_in_dist: 104.31 (−5.90% vs #3127)
  - val_geom_camber_rc: 101.16 (−2.52% vs #3127)
  - val_geom_camber_cruise: 69.32 (−8.60% vs #3127)
  - val_re_rand: 83.60 (−6.51% vs #3127)
- **Per-test-split mae_surf_p** (from best-val EMA checkpoint):
  - test_single_in_dist: 90.78
  - test_geom_camber_rc: 87.74
  - test_geom_camber_cruise: 57.75
  - test_re_rand: 77.69
- **n_params**: 662,359 (unchanged)
- **peak_memory_gb**: 32.95 (unchanged)
- **per_epoch_wall_time**: ~99 s/epoch (unchanged — clip adds negligible cost)
- **epochs_realized**: 18 of 18 (full cosine annealing)
- **Metric artifacts**: `models/model-grad-clip-1p0-20260516-102827/metrics.jsonl`, `models/model-grad-clip-1p0-20260516-102827/metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --experiment_name grad-clip-1p0 --agent charliepai2i24h1-fern --epochs 18 --grad_clip_norm 1.0`

### Gradient clipping win note

PR #3863 adds `clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()`. The diagnostic result: raw gradient norms are 25–300× the threshold (grad_clip_frac=1.0 on every batch throughout training), meaning this is NOT occasional spike protection — it is global gradient-direction rescaling (normalized-gradient AdamW). The source of large raw norms is bf16 autocast + surf_weight=25.0 composition. Active recipe now has **grad_clip_norm=1.0** as standard; new config field `grad_clip_norm: float` with default 0.0 in Config (0 = no clip; 1.0 = current standard).

## Previous best — PR #3127 (charliepai2i24h1-askeladd / smoothl1-rebased)

- **val_avg/mae_surf_p**: **94.9723** (best at epoch 18; 18 of 18 epochs realized — full cosine anneal)
- **test_avg/mae_surf_p**: **85.0372** (NaN-safe 4-split — first sub-90 test result on this track)
- **Per-val-split mae_surf_p** (best epoch 18):
  - val_single_in_dist: 110.85 (−17.0% vs #3478)
  - val_geom_camber_rc: 103.78 (−14.5% vs #3478)
  - val_geom_camber_cruise: 75.84 (−14.7% vs #3478)
  - val_re_rand: 89.42 (−13.3% vs #3478)
- **Per-test-split mae_surf_p** (from best-val EMA checkpoint):
  - test_single_in_dist: 97.18
  - test_geom_camber_rc: 93.68
  - test_geom_camber_cruise: 64.13
  - test_re_rand: 85.16
- **n_params**: 662,359 (unchanged — pure loss-axis change)
- **peak_memory_gb**: 32.95 (unchanged)
- **per_epoch_wall_time**: ~98 s/epoch (unchanged)
- **epochs_realized**: 18 of 18 (full cosine annealing)
- **Metric artifacts**: `models/model-charliepai2i24h1-askeladd-smoothl1-rebased-20260516-052919/metrics.jsonl`, `models/model-charliepai2i24h1-askeladd-smoothl1-rebased-20260516-052919/metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --experiment_name smoothl1-rebased --agent charliepai2i24h1-askeladd --epochs 18`

### SmoothL1 win note

PR #3127 replaces MSE with `F.smooth_l1_loss(..., beta=1.0, reduction='none')` in both the training loop and `evaluate_split`. Zero compute overhead — same wall time, same memory, same n_params. The gain is −15.0% on val_avg across all 4 splits uniformly (13–17%). SmoothL1's bounded-gradient L1 tail better matches the `mae_surf_p` evaluation metric and de-emphasizes outlier samples. Also includes a per-sample `y_finite` filter in `evaluate_split` for NaN-safe test eval (complements the `data/scoring.py` fix from #3378). Active recipe now has **SmoothL1 loss** as the standard.

## Previous best — PR #3478 (charliepai2i24h1-edward / narrow-bf16)

- **val_avg/mae_surf_p**: 111.7473 (epoch 18 of 18)
- **test_avg/mae_surf_p**: 99.3066 (NaN-safe 4-split)
- **Per-val-split**: single_in_dist=133.64, geom_camber_rc=121.33, geom_camber_cruise=88.92, re_rand=103.10
- **Reproduce**: `cd target/ && python train.py --experiment_name narrow-bf16-aligned --agent charliepai2i24h1-edward --epochs 18`

### Architecture revert note

PR #3478 reverts #3130 (wider trunk n_hidden 192→128, n_head 6→4) and adds bf16 autocast. The decisive result: narrow+bf16 realizes 18 epochs vs the wider trunk's 8-9 epochs at the same 30-min budget. The quality difference was schedule completion, not raw capacity. The active advisor recipe is now **n_hidden=128, n_head=4, bf16 autocast, EMA decay=0.999, surf_weight=25.0, T_max=18, epochs=18**.

## Active model configuration

| Component | Value |
|---|---|
| Architecture | Transolver with physics-aware attention |
| n_hidden | **128** (reverted from 192 by #3478) |
| n_layers | 5 |
| n_head | **4** (reverted from 6 by #3478) |
| slice_num | 64 |
| mlp_ratio | 2 |
| **Precision** | **bf16 autocast** on forward + loss + eval (master weights / grads / EMA stay fp32) |
| **EMA decay** | **0.999** (from #3137) — applied to eval/test/checkpoint |
| Optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| **surf_weight** | **25.0** (from #3136) |
| epochs (configured) | **18** (budget-aligned: ~98 s/ep × 18 ≈ 29.4 min) |
| schedule | CosineAnnealingLR (T_max=**18**) |
| sampler | WeightedRandomSampler (3-domain balanced) |
| **loss** | **SmoothL1** (`F.smooth_l1_loss, beta=1.0`) on normalized targets, `vol_loss + surf_weight * surf_loss` (from #3127) |

## Metrics contract

- **Primary ranking (val)**: `val_avg/mae_surf_p` — equal-weight surface pressure MAE across the four val splits.
- **Paper-facing (test)**: `test_avg/mae_surf_p` — same aggregation over the four test splits, computed from the best-val checkpoint.
- **Per-split diagnostics**: `{split}/mae_{surf,vol}_{Ux,Uy,p}` and `{split}/{vol,surf,total}_loss`.
- Direction: **lower is better**.

## Known issues / systemic constraints

1. **Schedule misalignment under 30-min cap.** `SENPAI_TIMEOUT_MINUTES=30` lets ~9-15 epochs complete per run; cosine `T_max=50` only anneals ~18-30% of its schedule. The model is evaluated near peak LR rather than after a low-LR fine-tune. For a paper-quality absolute number we will need a separate round with `--epochs ≈ realized_budget` and `T_max=epochs` so the schedule actually anneals. Thorfinn's send-back of #3144 includes this recipe.
2. **Cruise-test pressure NaN — FIXED in #3378.** Root cause was `data/scoring.py::accumulate_batch` computing `err = (pred - y).abs()` BEFORE masking, then multiplying by `surf_mask/vol_mask`. IEEE-754 `Inf * 0 = NaN` caused the pressure accumulator to go NaN whenever GT contained Inf (cruise test sample 20 has 761 `Inf` values in GT `p`). Fix: replace element-wise product with `torch.where(mask, err, 0)` — never reads `err` where mask=False, so Inf never enters the sum. Per-sample-skip semantics (`y_finite` filter) preserved. Unit test in `tests/test_scoring_nan_safe.py` confirms the failure mode and the fix. All PRs from #3378 onwards report `test_avg/mae_surf_p` as a finite 4-split mean directly from `metrics.jsonl`.
   - **Residual issue (out of scope for #3378)**: `train.py`'s eval-loss aggregation (computes MSE separately, not via `accumulate_batch`) still hits the same Inf×0=NaN pattern. Shows as `loss=nan` in per-split eval prints for `test_geom_camber_cruise` but does NOT affect the paper-facing MAE metric. Queue a separate one-spot patch with its own waiver if/when it matters.

## How students should report

Students commit `models/<experiment>/metrics.jsonl` and a terminal `SENPAI-RESULT`
marker in the PR with both `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.
For the test number, run the NaN-safe re-eval (tanjiro's pattern from #3141)
and report the 3-split finite mean alongside the raw value.

## Compounding

When a PR beats this baseline, it gets merged and this file is updated with
the new best metrics. Future hypotheses are then layered on top of the
new advisor configuration.

## Merge history

| Date | PR | Student | Change | val_avg/mae_surf_p |
|---|---|---|---|---|
| 2026-05-15 12:44 | #3130 | edward | Wider: n_hidden 128→192, n_head 4→6 | 166.5037 |
| 2026-05-15 14:24 | #3137 | nezuko | EMA decay=0.999 on eval/test/ckpt | 129.4217 |
| 2026-05-15 14:35 | #3136 | frieren | surf_weight 10→25 | **126.3241** |
| 2026-05-15 20:30 | #3378 | thorfinn | data/scoring.py NaN-safe (`err*mask` → `where(mask, err, 0)`) — system fix, val unchanged | 126.3241 |
| 2026-05-16 | #3478 | edward | Narrow+bf16: n_hidden 192→128, n_head 6→4, bf16 autocast, epochs 14→18 full anneal | **111.7473** |
| 2026-05-16 | #3127 | askeladd | SmoothL1 loss (Huber beta=1.0) replacing MSE — pure loss-axis, zero compute cost | **94.9723** |
