# BASELINE — icml-appendix-charlie-pai2i-24h-r1

Active advisor branch baseline. Updated after each merged winner. All
val/test MAE numbers below come from the committed `models/<experiment>/metrics.jsonl`
on the listed PR.

## Current best — PR #3130 (charliepai2i24h1-edward / wider-h192-h6)

- **val_avg/mae_surf_p**: **166.5037** (best at epoch 8; 9 of 50 epochs realized under the per-run `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap)
- **test_avg/mae_surf_p**: **NaN** — `test_geom_camber_cruise.mae_surf_p` is non-finite due to a single sample's pressure prediction overflow. The other three test splits averaged 166.58. Eliminating this NaN is a primary target for the next experiment.
- **Per-val-split mae_surf_p** (best epoch 8):
  - val_single_in_dist: 209.69
  - val_geom_camber_rc: 177.40
  - val_geom_camber_cruise: 126.99
  - val_re_rand: 151.93
- **Per-test-split mae_surf_p** (from best-val checkpoint):
  - test_single_in_dist: 184.18
  - test_geom_camber_rc: 169.65
  - test_geom_camber_cruise: NaN
  - test_re_rand: 145.91
- **n_params**: 1,447,521 (1.45 M)
- **peak_memory_gb**: 63.0
- **Metric artifacts**: `models/model-charliepai2i24h1-edward-wider-h192-h6-20260515-124423/metrics.jsonl`, `models/model-charliepai2i24h1-edward-wider-h192-h6-20260515-124423/metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --experiment_name baseline-merged --agent baseline`

## Active model configuration

| Component | Value |
|---|---|
| Architecture | Transolver with physics-aware attention |
| n_hidden | **192** (was 128 pre-#3130) |
| n_layers | 5 |
| n_head | **6** (was 4 pre-#3130) |
| slice_num | 64 |
| mlp_ratio | 2 |
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

1. **Schedule misalignment under 30-min cap.** `SENPAI_TIMEOUT_MINUTES=30` lets ~9 epochs complete per run; cosine `T_max=50` only anneals ~18% of its schedule. The model is evaluated at near-peak LR rather than after a low-LR fine-tune. This affects every round-1 PR equally, so the round-1 ordering is fair. To get a paper-quality absolute number we will need a separate round with `--epochs ≈ 10` and `T_max=epochs` so the schedule actually anneals.
2. **Cruise-test pressure NaN.** At least one cruise test sample produces a non-finite pressure prediction under the current model. Goes through the unguarded `accumulate_batch` and propagates NaN to the whole-split average. Fix candidates: (a) per-sample Re-conditioned output scaler (ReScaler), (b) bounded pressure head (tanh-scaled), (c) advisor-side guard in scoring. The ReScaler is being tested in edward's next PR.

## How students should report

Students commit `models/<experiment>/metrics.jsonl` and a terminal `SENPAI-RESULT`
marker in the PR with both `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## Compounding

When a PR beats this baseline, it gets merged and this file is updated with
the new best metrics. Future hypotheses are then layered on top of the
new advisor configuration.
