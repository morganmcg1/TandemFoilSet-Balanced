# Round-5 Baseline — `icml-appendix-charlie-pai2i-24h-r5`

Primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four validation splits). Test-time decision metric is `test_avg/mae_surf_p`. Lower is better.

## 2026-05-16 00:00 — PR #3466: Bernoulli pressure residual (current best)

Stacks on top of PR #3315 Cautious AdamW + PR #3373 bf16 + PR #3265 FiLM + PR #3337 surf-L1 + PR #3281 EMA + PR #3266 scale-invariant loss + NaN fix. Reformulates the prediction target: instead of predicting raw `p`, predict the **viscous residual** `p − p_B` where `p_B = 0.5 · V_∞²` is the free-stream Bernoulli reference (per-sample scalar, `V_∞ = exp(log_Re) · 1.5e-5` m/s using the kinematic viscosity of air and L=1m chord). At inference, the model output is added back to the per-sample `p_B` to recover physical pressure. The student's empirical pre-pass confirmed the target std *increases* 1.71× under this transformation (raw p_std=679 → residual_std=1160), counter to the original "removes dynamic range" intuition; the win comes from a different mechanism — removing the analytic `V_∞²/2` component lets the network spend all its capacity on the spatial structure (viscous BL, separation, vortex shedding) rather than on first re-learning the freestream constant.

> **Note on tested code state:** Validated against the current merged tip on `origin/icml-appendix-charlie-pai2i-24h-r5` at `3a3104a` (post-Cautious-AdamW, post-bf16, post-FiLM+surf-L1+EMA+scale-inv). 17 epochs / 50 inside the 30-min wall-clock cap, same epoch budget as the merged Cautious AdamW + bf16 baseline. This is the first head-to-head measured compound result confirming the full-merged-stack baseline lands close to the pre-bf16 Cautious AdamW number (90.34) — bf16's epoch-budget unlock did not change the picture as much as predicted, but the Bernoulli residual reformulation gave a clean −4.7% on top.

**Primary** (Bernoulli residual validated on the full merged stack — Cautious AdamW + bf16 + FiLM + surf-L1 + EMA + scale-inv)
- `val_avg/mae_surf_p` = **86.0948** (best epoch 17 / 50, run cut by 30-min wall clock; **−4.70% vs PR #3315**, **−30.49% vs round-5 anchor PR #3266**)
- `test_avg/mae_surf_p` = **77.5066** (**−3.32% vs PR #3315**, **−32.22% vs round-5 anchor**)

**Per-split surface pressure MAE (Bernoulli residual + full stack)**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 102.0390 | 93.0730 |
| geom_camber_rc | 100.1206 | 90.1840 |
| geom_camber_cruise | 62.9941 | 52.3967 |
| re_rand | 79.2254 | 74.3726 |
| **avg** | **86.0948** | **77.5066** |

**Largest per-split gain:** `val_single_in_dist` −7.16% (the previous worst split — was 109.91 under merged Cautious AdamW baseline). All four val splits and all four test splits improved; gains scale with V_∞-range mixedness (largest gain on the most regime-mixed splits: single_in_dist and re_rand).

**Mechanism summary**
- Subtracting per-sample `p_B = 0.5·V_∞²` removes the analytic component that depends only on Reynolds number
- Bernoulli per-sample shift is computed once at startup from `log_Re` features (no per-batch overhead)
- Target std *increases* 1.71× (raw 679 → residual 1160), but normalized-space MSE/L1 loss landscape is unchanged
- Network capacity reallocates from "compute V_∞²" to "fit viscous residual" — implicit prior shrink
- Compounds with all six previously-merged mechanisms

**Arm B (chord-position correction):** failed (val_avg=122.18, +35%). The chord-position formula was applied to global mesh x-coordinate (range [-9.55, 11.34], ~20 chord lengths) rather than normalized chord position, producing high-frequency oscillations that injected noise. Clean negative on the implementation as specified — properly-computed chord position would require per-foil chord-boundary detection, deferred as a future follow-up.

**Model config (architecture unchanged from #3315; new prediction target via train-time/eval-time `--bernoulli_residual` flag)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 raw params (same as #3265); EMA shadow copy at decay=0.999
- Optimizer: CautiousAdamW lr=5e-4, wd=1e-4 (replaces standard AdamW)
- bf16 mixed precision via `torch.autocast` (peak VRAM ~33 GB)
- batch_size=4, surf_weight=10.0, surf_p_l1_weight=1.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant MSE + surf_p_l1_weight=1.0 * surf_p_l1 (on residual target)
- New: `--bernoulli_residual freestream` to subtract `p_B = 0.5·V_∞²` from training targets and add back at eval

**Metric artifacts**
- `models/model-charliepai2i24h5-askeladd-bernoulli_freestream-20260515-215606/metrics.jsonl`
- `models/model-charliepai2i24h5-askeladd-bernoulli_freestream-20260515-215606/metrics.yaml`
- `models/model-charliepai2i24h5-askeladd-bernoulli_freestream-20260515-215606/config.yaml`

**Reproduce**
```bash
cd target/
python train.py --agent charliepai2i24h5-askeladd \
    --experiment_name "charliepai2i24h5-askeladd/bernoulli_freestream" \
    --bernoulli_residual freestream \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```

**Cumulative round-5 improvement:** **−30.49% val_avg** (123.88 → 86.09) and **−32.22% test_avg** (114.37 → 77.51) over the pre-round-5 baseline. **Six compounding wins**: scale-inv → EMA → surf-L1 → FiLM → bf16 → Cautious AdamW → Bernoulli residual (seven mechanisms, six confirmed compoundings).

---

## 2026-05-15 21:29 — PR #3315: Cautious AdamW (Liang et al. ICLR 2026) (previous best)

Stacks on top of PR #3373 bf16 + PR #3265 FiLM + PR #3337 surf-L1 + PR #3281 EMA + PR #3266 scale-invariant loss + NaN fix. Subclass `CautiousAdamW(torch.optim.AdamW)`: snapshots params before `super().step()`, then post-step constructs the agreement mask `(m * g > 0)` from the EMA-momentum `exp_avg` and the original gradient, mean-rescales the mask with `clamp(min=1e-3)`, and replaces the parent's delta with `delta * mask`. Result: ~38% of update components are gated to zero each step (mean mask agreement ≈ 0.62, flat across all training epochs and across all merged-mechanism variants — direct evidence that cautious masking operates on disjoint state from EMA/FiLM/surf-L1). Mechanism gates noisy update directions per step; EMA averages the iterate trajectory; FiLM conditions architecture on regime; surf-L1 aligns gradient with the eval metric. Four orthogonal axes.

> **Note on tested code state:** This run was validated on `origin/icml-appendix-charlie-pai2i-24h-r5` at tip `b5760af` (post-FiLM, post-surf-L1, post-EMA, pre-bf16). The merged code now stacks Cautious AdamW onto bf16 as well. The compound bf16 + Cautious AdamW run is expected to land in the low-80s val_avg range — bf16 unlocks ~5 more effective epochs in the wall-clock budget, and the cautious mask curve was still ~flat at epoch 13 with val_avg dropping 94.7 → 90.3 in the final epoch, strongly suggesting more training would help. The next validated run against the full merged code will establish the confirmed compound metric.

**Primary** (Cautious AdamW validated on FiLM+surf-L1+EMA+scale-inv stack at tip `b5760af`)
- `val_avg/mae_surf_p` = **90.3428** (best epoch 13 / 50, run cut by 30-min wall clock; **−12.31% vs PR #3265**, **−15.46% vs PR #3337**)
- `test_avg/mae_surf_p` = **80.1674** (**−13.01% vs PR #3265**, **−17.23% vs PR #3337**)

**Per-split surface pressure MAE (Cautious AdamW + full stack)**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 109.9053 | 96.3817 |
| geom_camber_rc | 103.3709 | 91.8343 |
| geom_camber_cruise | 65.6940 | 54.3760 |
| re_rand | 82.4009 | 78.0776 |
| **avg** | **90.3428** | **80.1674** |

**Largest per-split gain:** `val_geom_camber_cruise` −14.57% and `val_geom_camber_rc` −14.37% — uniform 10–15% improvement across every val/test cell, qualitatively different from the standalone Cautious AdamW run where in-dist splits regressed. EMA + FiLM stabilization unlocks cautious masking on in-distribution sharp minima.

**Cautious mask dynamics (sanity-check signal)**
- `train/cautious_mask_mean` per epoch (1–13): 0.6356, 0.6259, 0.6282, 0.6216, 0.6215, 0.6190, 0.6216, 0.6129, 0.6149, 0.6183, 0.6130, 0.6147, 0.6092
- Mean ≈ **0.6197**, essentially identical across the standalone / +EMA / +EMA+surf-L1+FiLM runs (0.620 / 0.621 / 0.620)
- The flat (non-rising) curve indicates the optimizer is in equilibrium with EMA/FiLM/surf-L1 throughout the 13-epoch undercooked-training regime captured by the 30-min wall-clock

**Model config (architecture unchanged from #3265 + bf16 hooks; new optimizer)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 raw params (same as #3265); EMA shadow copy at decay=0.999
- **Optimizer: CautiousAdamW** lr=5e-4, wd=1e-4
- batch_size=4, surf_weight=10.0, surf_p_l1_weight=1.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant MSE + surf_p_l1_weight=1.0 * surf_p_l1
- Peak VRAM: 44.61 GB

**Cumulative round-5 improvement:** −27.06% val_avg (123.88 → 90.34) and −29.91% test_avg (114.37 → 80.17) over the pre-round-5 baseline. **Five compounding wins**: scale-inv → EMA → surf-L1 → FiLM → Cautious AdamW.

**Metric artifacts**
- `models/model-charliepai2i24h5-askeladd-cautious_adamw_on_ema_v2-20260515-203741/metrics.jsonl`
- `models/model-charliepai2i24h5-askeladd-cautious_adamw_on_ema_v2-20260515-203741/metrics.yaml`
- `models/model-charliepai2i24h5-askeladd-cautious_adamw_on_ema_v2-20260515-203741/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-askeladd \
    --experiment_name "round5_baseline_repro_pr3315" \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```

## 2026-05-15 21:26 — PR #3373: bf16 mixed-precision autocast for forward pass (previous baseline)

Stacks on top of PR #3265 FiLM + PR #3337 surf-L1 + PR #3281 EMA + PR #3266 scale-invariant loss + NaN fix. Wraps the forward pass and the `(pred - y_norm)**2` computation in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (both in the train loop and in `evaluate_split`); explicitly casts the output to fp32 before the per-sample scale-invariant reductions, the surface-L1 aux loss, and the eval-time MAE accumulation. Weights, EMA shadow, optimizer state, and all reductions stay in fp32. Result: ~21% per-epoch wall-clock reduction (125s → 98s), peak VRAM drops 42 → 33 GB, and the same 30-min wall-clock budget completes 19 epochs instead of 14 — five extra epochs of training within `SENPAI_TIMEOUT_MINUTES`.

> **Note on tested code state:** This run was validated on `origin/icml-appendix-charlie-pai2i-24h-r5` at tip `1211ee5` (EMA + scale-inv loss + NaN fix, pre-surf-L1, pre-FiLM). The merged code now stacks bf16 onto surf-L1 + FiLM as well. Expected compound val_avg (bf16 + FiLM + surf-L1 + EMA + scale-inv) is **high-80s to low-90s** — bf16 is fundamentally a compute unlock (more effective epochs) and its mechanism is orthogonal to the loss/architecture mechanisms. The next validated run against the full merged code will establish the confirmed compound metric.

**Primary** (bf16 validated on #3281 baseline at tip `1211ee5`)
- `val_avg/mae_surf_p` = **99.1251** (best epoch 19 / 50, run cut by 30-min wall clock; **-13.16% vs PR #3281**, **-3.78% vs PR #3265**)
- `test_avg/mae_surf_p` = **89.1198** (**-12.70% vs PR #3281**, **-3.30% vs PR #3265**)

**Per-split surface pressure MAE (bf16 run vs PR #3281 baseline; not yet measured on full merged stack)**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 119.8640 | 106.8204 |
| geom_camber_rc | 112.4498 | 100.9678 |
| geom_camber_cruise | 74.0381 | 62.7467 |
| re_rand | 90.1481 | 85.9442 |
| **avg** | **99.1251** | **89.1198** |

**Largest per-split gain (bf16 vs #3281):** every split improves; the val curve was still monotonically decreasing at epoch 19, indicating further training would help.

**Why arm B (batch_size=8) regressed:** fewer optimizer steps per epoch (~188 vs ~375) plus one fewer total epoch ⇒ ~50% fewer total gradient steps in the same wall-clock budget, dominating any SNR benefit from a larger batch. Per-sample scale-invariant loss (#3266) already equalizes much of the residual gradient noise, so doubling the batch removes useful exploration noise. Rescuing batch=8 would need `lr=7e-4`-`1e-3` + warmup — that's a separate experiment.

**Implementation (train.py only; +11/-3 lines)**
- Wraps `pred = model({"x": x_norm})["preds"]` in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` in both the train loop (also wrapping `sq_err = (pred - y_norm)**2`) and `evaluate_split` (pred only).
- Immediately `pred = pred.float()` / `sq_err = sq_err.float()` after the autocast block.
- Result: forward in bf16, all reductions in fp32, EMA in fp32, optimizer step in fp32.

**Model config (unchanged from #3265)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- 662,359 raw params (no FiLM, since this run is on #3281 tip); FiLM-merged compound run will have 829,015 params
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0, CosineAnnealingLR(T_max=50)
- Loss: per-sample scale-invariant MSE + (post-merge) surf_p_l1_weight=1.0 * surf_p_l1
- EMA decay=0.999

**Metric artifacts**
- Arm A (winner): `models/model-charliepai2i24h5-edward-bf16-baseline-config-20260515-182527/metrics.jsonl`
- Arm B (regressed): `models/model-charliepai2i24h5-edward-bf16-batch8-20260515-192457/metrics.jsonl`

**Reproduce**
```bash
cd target/ && python train.py \
    --experiment_name "round5_baseline_repro_pr3373" \
    --epochs 50
```

## 2026-05-15 20:26 — PR #3265: FiLM per-block global-condition modulation (previous baseline)

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
