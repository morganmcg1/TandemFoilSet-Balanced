# Round-5 Baseline — `icml-appendix-charlie-pai2i-24h-r5`

Primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four validation splits). Test-time decision metric is `test_avg/mae_surf_p`. Lower is better.

## 2026-05-15 20:26 — PR #3265: FiLM per-block global-condition modulation (current best)

Stacks on top of PR #3337's surf-L1 aux + PR #3281's EMA + PR #3266's scale-invariant loss. Adds `FiLMConditioner` modules (shared SiLU MLP → per-block `(scale, shift)`) that inject the global flow condition vector (log Re, AoAs, NACAs, gap, stagger from `x[:, 0, 13:24]`) via `fx = fx * (1 + scale_i) + shift_i` after each of the 5 Transolver block residuals. This gives every layer direct access to the global regime state rather than relying solely on the input MLP to encode it. Compounds with EMA and scale-invariant loss: FiLM targets regime-conditioning (architectural) while EMA targets iterate averaging and scale-inv targets gradient magnitude equalization (loss-level) — orthogonal mechanisms.

> **Note on tested code state:** This run was validated on `origin/icml-appendix-charlie-pai2i-24h-r5` at tip `1211ee5` (EMA + scale-inv loss + NaN fix, pre-surf-L1). The merged code also includes PR #3337's surf-L1 aux, which compounds with FiLM. Expected compound val_avg (FiLM + surf-L1 + EMA + scale-inv) is ~97–100 based on both mechanisms being orthogonal. The next validated run against the full merged code will establish the confirmed compound metric.

**Primary** (FiLM validated on #3281 baseline, rebased run at tip `1211ee5`)
- `val_avg/mae_surf_p` = **103.0171** (best epoch 14 / 50, run cut by 30-min wall clock; **-9.77% vs PR #3281**)
- `test_avg/mae_surf_p` = **92.1617** (**-9.74% vs PR #3281**)

**Per-split surface pressure MAE (val, rebased FiLM run)**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 122.1931 | 109.6200 |
| geom_camber_rc | 120.7163 | 107.8300 |
| geom_camber_cruise | 76.8948 | 62.4400 |
| re_rand | 92.2644 | 88.7600 |
| **avg** | **103.0171** | **92.1617** |

**Largest per-split gain:** val_re_rand −24.0% vs #3266 baseline; val_geom_camber_cruise −22.0% vs pre-rebase FiLM.

**Model config (architecture change):**
- FiLM: shared SiLU MLP with `[cond_dim=11, 64, 2*n_hidden*n_layers]` — generates per-block `(scale_i, shift_i)` pairs
- Params: 829,015 (+166K vs single-head baseline 662K)
- Peak VRAM: 44.6 GB (vs 42.1 GB baseline)
- All other hypers unchanged from #3281: n_hidden=128, n_layers=5, n_head=4, slice_num=64, lr=5e-4, wd=1e-4, bs=4, EMA decay=0.999, surf_weight=10.0

**Metric artifacts**
- `models/model-charliepai2i24h5-fern-film_flow_condition_every_block_rebased-20260515-184444/metrics.jsonl` (rebased run — use this)
- `models/model-charliepai2i24h5-fern-film_flow_condition_every_block-20260515-133304/metrics.jsonl` (original pre-rebase, historical only)

**Reproduce**
```bash
cd target/ && python train.py \
    --experiment_name "round5_baseline_repro_pr3265" \
    --epochs 50
```

## 2026-05-15 19:31 — PR #3337: Surface-pressure L1 auxiliary loss (previous baseline)

Stacks on top of PR #3281's EMA + PR #3266's scale-invariant loss + NaN workaround. Adds a single auxiliary L1 term on the pressure channel of surface nodes (in normalized-y space) at weight w=1.0 of the existing per-sample scale-invariant MSE loss. The L1 aggregation is pooled `Σ|err| / n_surf`, identical in shape to the eval-time `mae_surf_p` metric — so the gradient direction is the sign-of-error that directly minimizes MAE. Compounds cleanly with EMA (loss-side vs parameter-trajectory mechanisms) and with the scale-invariant loss (the L1 term is added post-normalization).

**Primary**
- `val_avg/mae_surf_p` = **106.8550** (best epoch 14 / 50, run cut by 30-min wall clock; **-6.41% vs PR #3281**)
- `test_avg/mae_surf_p` = **96.8671** (**-5.11% vs PR #3281**)

**Per-split surface pressure MAE**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 127.8497 | 115.6571 |
| geom_camber_rc | 121.1106 | 108.6600 |
| geom_camber_cruise | 81.3908 | 68.2457 |
| re_rand | 97.0689 | 94.9055 |
| **avg** | **106.8550** | **96.8671** |

**Model config** (unchanged from #3281)
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- 662 359 parameters; EMA shadow copy at decay=0.999
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant MSE + `surf_p_l1_weight=1.0 * surf_p_l1` (new aux term)

**Cumulative round-5 improvement:** -13.74% val_avg (123.88 → 106.86) and -15.30% test_avg (114.37 → 96.87) over the pre-round-5 baseline.

**Metric artifacts**
- `models/model-charliepai2i24h5-frieren-surf_p_l1_aux_w1.0-20260515-172842/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-surf_p_l1_aux_w1.0-20260515-172842/metrics.yaml`
- `models/model-charliepai2i24h5-frieren-surf_p_l1_aux_w1.0-20260515-172842/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --experiment_name "round5_baseline_repro_pr3337" \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```

## 2026-05-15 16:23 — PR #3281: EMA model weights for checkpoint selection and test eval (previous baseline)

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
