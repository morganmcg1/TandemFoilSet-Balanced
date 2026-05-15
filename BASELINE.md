# Round-5 Baseline — `icml-appendix-charlie-pai2i-24h-r5`

Primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four validation splits). Test-time decision metric is `test_avg/mae_surf_p`. Lower is better.

## 2026-05-15 14:25 — PR #3266: Per-sample scale-invariant loss to equalize Re-regime gradients

Round-5 anchor baseline. First merged result on this branch; subsequent PRs in this round should beat these numbers. Also lands a critical NaN-propagation workaround in `train.py::evaluate_split` (the `data/scoring.py` accumulator has `NaN * 0 = NaN` slipping past the sample-skip mask when the cruise test set contains the one bad sample with 761 NaN in p; we sanitize upstream because `data/` is read-only).

**Primary**
- `val_avg/mae_surf_p` = **123.8778** (best epoch 14 / 50, run cut by 30-min wall clock)
- `test_avg/mae_surf_p` = **114.3695**

**Per-split surface pressure MAE**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 142.1946 | 125.8483 |
| geom_camber_rc | 136.1165 | 130.8474 |
| geom_camber_cruise | 95.7637 | 85.2854 |
| re_rand | 121.4363 | 115.4971 |
| **avg** | **123.8778** | **114.3695** |

**Surface MAE per channel (validation, averaged over four splits)**
- Ux ≈ 1.86, Uy ≈ 0.83, p ≈ 123.88

**Model config**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.0
- 662 359 parameters
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant `(vol_loss_per_sample + surf_weight * surf_loss_per_sample) / per_sample_field_scale(y_norm, mask)` averaged over batch
- `evaluate_split` sanitizes non-finite ground-truth before passing to `accumulate_batch`

**Metric artifacts**
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.yaml`
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --experiment_name "round5_baseline_repro" \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 14 epochs in 30 min on this hardware.)
