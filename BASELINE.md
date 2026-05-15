# BASELINE — icml-appendix-charlie-pai2i-24h-r1

Active advisor branch baseline. Updated after each merged winner. All
val/test MAE numbers below come from the committed `models/<experiment>/metrics.jsonl`
on the listed PR.

## Current best — PR #3137 (charliepai2i24h1-nezuko / ema-0999)

- **val_avg/mae_surf_p**: **129.4217** (best at epoch 14; 14 of 50 epochs realized under the per-run `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap; curve still strictly monotonically decreasing at cutoff)
- **test_avg/mae_surf_p**: **NaN** — `test_geom_camber_cruise.mae_surf_p` is non-finite, caused by 761 `Inf` values in the ground-truth pressure channel of cruise test sample 20 propagating through `data/scoring.py::accumulate_batch` via the IEEE-754 `Inf * 0 = NaN` interaction with the surface/volume mask. The other three test splits averaged **128.44**. This is a scoring-side bug on corrupt GT data — diagnosed precisely in #3137; a dedicated `data/scoring.py` bug-fix PR is queued (read-only file requires explicit advisor waiver).
- **Per-val-split mae_surf_p** (best epoch 14):
  - val_single_in_dist: 164.34
  - val_geom_camber_rc: 145.10
  - val_geom_camber_cruise: 97.32
  - val_re_rand: 110.93
- **Per-test-split mae_surf_p** (from best-val EMA checkpoint):
  - test_single_in_dist: 145.46
  - test_geom_camber_rc: 129.24
  - test_geom_camber_cruise: NaN (corrupt GT, see above)
  - test_re_rand: 110.61
- **peak_memory_gb**: 42.11
- **Metric artifacts**: `models/model-ema-0999-20260515-125245/metrics.jsonl`, `models/model-ema-0999-20260515-125245/metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --experiment_name baseline-merged --agent baseline`

### Stacking note (trust-in-orthogonality)

Nezuko's measured 129.4217 was on the **pre-#3130** trunk config (`n_hidden=128, n_head=4`) plus EMA. The squash-merge layered the EMA changes onto the already-merged wider config from #3130, so the active advisor configuration is now **`n_hidden=192, n_head=6` + EMA decay=0.999** — an untested combination. We're trusting orthogonality between width and EMA. If round-2 PRs reveal the combined recipe underperforms either single change alone, that's a diagnostic signal we'll act on.

## Active model configuration

| Component | Value |
|---|---|
| Architecture | Transolver with physics-aware attention |
| n_hidden | **192** (from #3130) |
| n_layers | 5 |
| n_head | **6** (from #3130) |
| slice_num | 64 |
| mlp_ratio | 2 |
| **EMA decay** | **0.999** (from #3137) — applied to eval/test/checkpoint |
| Optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs (configured) | 50 |
| schedule | CosineAnnealingLR (T_max=epochs) |
| sampler | WeightedRandomSampler (3-domain balanced) |
| loss | MSE on normalized targets, `vol_loss + surf_weight * surf_loss` |

## Metrics contract

- **Primary ranking (val)**: `val_avg/mae_surf_p` — equal-weight surface pressure MAE across the four val splits.
- **Paper-facing (test)**: `test_avg/mae_surf_p` — same aggregation over the four test splits, computed from the best-val checkpoint.
- **Per-split diagnostics**: `{split}/mae_{surf,vol}_{Ux,Uy,p}` and `{split}/{vol,surf,total}_loss`.
- Direction: **lower is better**.

## Known issues / systemic constraints

1. **Schedule misalignment under 30-min cap.** `SENPAI_TIMEOUT_MINUTES=30` lets ~9-14 epochs complete per run; cosine `T_max=50` only anneals ~18-28% of its schedule. The model is evaluated near peak LR rather than after a low-LR fine-tune. This affects every round-1 PR equally, so the round-1 ordering is fair. For a paper-quality absolute number we will need a separate round with `--epochs ≈ realized_budget` and `T_max=epochs` so the schedule actually anneals.
2. **Cruise-test pressure NaN — root cause identified (GT corruption + scoring bug).** Nezuko's #3137 diagnosis: cruise test sample 20 has 761 `Inf` values in the ground-truth `p` channel. `data/scoring.py::accumulate_batch` is intended to mask non-finite-GT samples via `surf_mask/vol_mask`, but the implementation computes `err = (pred - y).abs()` *before* masking, and IEEE-754 `Inf * 0 = NaN` causes the pressure accumulator to go NaN. The bug is independent of which model is being evaluated — both #3130 (wider, n_hidden=192) and #3137 (EMA, n_hidden=128) reproduced the same NaN. Fix candidates: (a) `err = err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)` after computing, or (b) mask before subtraction. Requires advisor waiver of the `data/scoring.py` read-only constraint — queued as a dedicated bug-fix PR.

## How students should report

Students commit `models/<experiment>/metrics.jsonl` and a terminal `SENPAI-RESULT`
marker in the PR with both `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## Compounding

When a PR beats this baseline, it gets merged and this file is updated with
the new best metrics. Future hypotheses are then layered on top of the
new advisor configuration.

## Merge history

| Date | PR | Student | Change | val_avg/mae_surf_p |
|---|---|---|---|---|
| 2026-05-15 12:44 | #3130 | edward | Wider: n_hidden 128→192, n_head 4→6 | 166.5037 |
| 2026-05-15 14:24 | #3137 | nezuko | EMA decay=0.999 on eval/test/ckpt | **129.4217** |
