# Round-5 Baseline — `icml-appendix-charlie-pai2i-24h-r5`

Primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four validation splits). Test-time decision metric is `test_avg/mae_surf_p`. Lower is better.

## 2026-05-15 16:23 — PR #3281: EMA model weights for checkpoint selection and test eval (current best)

Stacks on top of PR #3266's scale-invariant loss + NaN workaround. Maintains a shadow EMA copy of the model with decay=0.999 (≈1000-step averaging window ≈ 2.7 epochs at batch_size=4), updated after every `optimizer.step()`. Validation, checkpoint selection, and final test eval all run from the EMA weights rather than the raw final-iterate model. The flat-minimum effect of weight averaging compounds with the undercooked training horizon (14/50 epochs by wall clock): gains concentrate strongly on the OOD splits.

**Primary**
- `val_avg/mae_surf_p` = **114.1704** (best epoch 14 / 50, run cut by 30-min wall clock; -7.84% vs PR #3266)
- `test_avg/mae_surf_p` = **102.0813** (-10.74% vs PR #3266)

**Per-split surface pressure MAE**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 138.4778 | 121.8061 |
| geom_camber_rc | 130.8363 | 115.6647 |
| geom_camber_cruise | 84.6023 | 71.5962 |
| re_rand | 102.7651 | 99.2581 |
| **avg** | **114.1704** | **102.0813** |

**Model config**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.0
- 662 359 parameters (raw model); EMA holds a shadow copy (no training overhead, ~0.4 GB extra VRAM)
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant (inherited from #3266)
- EMA: `update_ema(ema_model, model, decay=0.999)` after every `optimizer.step()`; `evaluate_split` runs `ema_model`; checkpoint = `ema_model.state_dict()`

**Metric artifacts**
- `models/model-charliepai2i24h5-frieren-ema_weights_checkpoint-20260515-153007/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-ema_weights_checkpoint-20260515-153007/metrics.yaml`
- `models/model-charliepai2i24h5-frieren-ema_weights_checkpoint-20260515-153007/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --experiment_name "round5_baseline_repro" \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 14 epochs in 30 min on this hardware.)

---

## 2026-05-15 14:25 — PR #3266: Per-sample scale-invariant loss to equalize Re-regime gradients (superseded by #3281)

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
