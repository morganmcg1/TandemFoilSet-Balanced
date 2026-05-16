# Round-5 Baseline — `icml-appendix-charlie-pai2i-24h-r5`

Primary ranking metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four validation splits). Test-time decision metric is `test_avg/mae_surf_p`. Lower is better.

## 2026-05-16 10:35 — PR #3809: Gradient clipping — clip=0.5 + lr=1e-3 + compile (current best)

Stacks on top of PR #3666 (lr=1e-3 + compile) plus a new `--grad_clip_norm` CLI flag and `torch.nn.utils.clip_grad_norm_` call between `loss.backward()` and `optimizer.step()`. **The single biggest win of round 5** — Arm B (clip_norm=0.5) beats current baseline by −10.31% val_avg. The unclipped gradient distribution had been silently far higher than expected: pre-clip mean ‖g‖ = 7.4, worst-step ‖g‖ = 200–273. Clipping fires on **~100% of steps in both arms** at lr=1e-3, with the tighter 0.5 threshold acting as a stronger implicit regularizer and yielding a uniform improvement across all 16 val/test cells. The Cautious mask mean (0.624) is invariant under clipping (clipping preserves gradient direction, only rescales magnitude — the sign-agreement test is unaffected). Clipping and Cautious compose cleanly: clipping bounds magnitude, Cautious gates direction.

**Important stack note:** This baseline reverts LR from 1.5e-3 → 1e-3. Frieren's run was on lr=1e-3 (assigned in Loop 19 before #3771 merged). The natural follow-up is the compound **lr=1.5e-3 + clip=0.5** — if both mechanisms compose, the next win could be substantially larger. Assigned to frieren as #3920 (TBD).

**Primary** (Arm B: grad_clip_norm=0.5 + lr=1e-3 + compile, n_hidden=128)
- `val_avg/mae_surf_p` = **45.4964** (best epoch 31 / 50, run cut by 30-min wall clock; **−10.31% vs PR #3771**, **−63.27% vs round-5 anchor**)
- `test_avg/mae_surf_p` = **38.3732** (**−13.48% vs PR #3771**, **−66.45% vs round-5 anchor**)
- **Cumulative round-5 improvement:** −63.27% val_avg (123.88 → 45.50), −66.45% test_avg (114.37 → 38.37). **Thirteen compounding wins.**

**Per-split surface pressure MAE (Arm B — grad_clip=0.5 at lr=1e-3)**

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs PR #3771 | Δ test vs PR #3771 |
|---|---:|---:|---:|---:|
| single_in_dist | 45.415 | 39.907 | −12.59% | −14.89% |
| geom_camber_rc | 62.109 | 54.158 | −4.55% | −8.36% |
| geom_camber_cruise | 28.200 | 22.447 | −14.50% | −17.66% |
| re_rand | 46.261 | 36.981 | −12.39% | −16.24% |
| **avg** | **45.4964** | **38.3732** | **−10.31%** | **−13.48%** |

All 16 val/test cells improve. Largest gains on test_geom_camber_cruise (−17.66%) and test_re_rand (−16.24%). camber_rc improvement is the smallest (−4.55% val) — consistent with the camber_rc capacity-saturation finding from #3463.

**Arm A (clip_norm=1.0):** val_avg=47.1697 (−6.93% vs 50.70). Also a winner but clip=0.5 is decisively better. Tells us the optimum is below 1.0; the next refinement (clip=0.25 vs 0.1) is the natural follow-up.

**Mechanism — why grad-norm clipping helps this much:**
The unclipped gradient distribution had outlier steps with ‖g‖ up to 273 (vs the mean of 7.4). Without clipping, these outliers translate to ~37× larger parameter updates than typical, "punching" the model into bad regions. Cautious AdamW absorbed these in aggregate (no NaN, no training divergence), but generalization suffered: the model carries inherited bias from those outlier updates. Clipping caps the per-step update norm at a global budget; the model navigates the loss landscape with smaller, more consistent steps, which the merged stack's implicit-regularization mechanisms (FiLM, scale-inv loss, Cautious) handle better.

**Model config**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 parameters; EMA shadow at decay=0.999; torch_compile_active=True (default mode, dynamic=True)
- **CautiousAdamW lr=1e-3**, wd=1e-4, **grad_clip_norm=0.5** (new flag added in this PR)
- CosineAnnealingLR(T_max=25, eta_min=1e-4, eta_min_factor=0.10)
- bf16 AMP, surf_weight=10.0, surf_p_l1_weight=1.0
- bernoulli_residual=False
- Peak VRAM: 24.39 GB (zero memory overhead from clipping)
- Avg s/epoch: 58.1 (within noise of compile baseline)
- Cautious mask mean: 0.6244 (invariant under clipping — clipping preserves direction)
- Pre-clip gradient norm: mean 7.578, range [3.686, 21.253] per epoch, worst-step 199.48

**Metric artifacts**
- `models/model-charliepai2i24h5-frieren-gradclip0p5_lr1e3_compile-20260516-092643/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-gradclip0p5_lr1e3_compile-20260516-092643/metrics.yaml`
- `models/model-charliepai2i24h5-frieren-gradclip0p5_lr1e3_compile-20260516-092643/config.yaml`
- (Arm A): `models/model-charliepai2i24h5-frieren-gradclip1p0_lr1e3_compile-20260516-083415/metrics.jsonl`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-frieren \
    --experiment_name "baseline_repro_gradclip0p5_lr1e3_compile" \
    --torch_compile \
    --lr 1e-3 --grad_clip_norm 0.5 \
    --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 32 epochs in 30 min.)

---

## 2026-05-16 09:45 — PR #3771: LR continuation — lr=1.5e-3 + compile (previous best)

Stacks on top of full round-5 merged baseline: scale-inv loss + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + T_max=25/eta_min_factor=0.10 + torch.compile + lr=1e-3 (PR #3666 stack). Pure optimizer-hyperparameter change: lr 1e-3 → 1.5e-3 on n_hidden=128. Arm A wins decisively on all 8 val/test cells. Arm B (lr=2e-3) regresses (val_avg 53.41) — not from epoch-1 instability (both arms had clean ~370 epoch-1 val, fully neutralised by compile + TF32), but from the late-schedule eta_min floor (2e-4) being too high to settle. The LR-vs-val_avg curve has an interior optimum at lr=1.5e-3 in the 32-epoch budget. Arm B already rebounding at epoch 32 (53.49 vs 53.41 at epoch 31), confirming the floor effect.

**Note:** This baseline returns to n_hidden=128 from #3463's n_hidden=192. The thorfinn run's lr=1.5e-3 win at n=128 (val_avg=50.70) is **decisively better than #3463's n=192 + lr=1e-3** (val_avg=53.19). This indicates lr and capacity axes are not strictly orthogonal — pure LR was the dominant lever. The compound n=192 + lr=1.5e-3 has not been tested and is the natural next experiment.

**Primary** (Arm A: lr=1.5e-3 + compile, full merged stack at n_hidden=128)
- `val_avg/mae_surf_p` = **50.7001** (best epoch 32 / 50, run cut by 30-min wall clock; **−4.68% vs PR #3463**, **−59.07% vs round-5 anchor**)
- `test_avg/mae_surf_p` = **44.3493** (**−6.77% vs PR #3463**, **−61.21% vs round-5 anchor**)
- **Cumulative round-5 improvement:** −59.07% val_avg (123.88 → 50.70), −61.21% test_avg (114.37 → 44.35). **Twelve compounding wins.**

**Per-split surface pressure MAE (Arm A — lr=1.5e-3, n_hidden=128)**

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs PR #3463 | Δ test vs PR #3463 |
|---|---:|---:|---:|---:|
| single_in_dist | 51.954 | 46.889 | −6.17% | −9.37% |
| geom_camber_rc | 65.066 | 59.097 | −7.33% | −9.23% |
| geom_camber_cruise | 32.981 | 27.259 | −2.35% | +0.09% |
| re_rand | 52.799 | 44.153 | −1.15% | −4.43% |
| **avg** | **50.7001** | **44.3493** | **−4.68%** | **−6.77%** |

7 of 8 cells improve. The single test_geom_camber_cruise is essentially neutral (+0.09%). Notably, camber_rc (which REGRESSED at n=192 + lr=1e-3) improves cleanly here (−7.33% val / −9.23% test). High LR is the correct knob for camber_rc.

**Arm B (lr=2e-3 + compile):** val_avg=53.4114, test_avg=48.0565. Regresses on 3 of 8 cells (val/test_geom_camber_rc, test_re_rand). The eta_min floor at 2e-4 prevents settling. Past-optimum, not unstable.

**Mechanism — epoch-1 stability holds:**
Both arms epoch-1 val_avg ≈ 370 — within 1% of the lr=1e-3 baseline. TF32 + compile fully neutralises the catastrophic-perturbation regime through at least lr=2e-3. The threshold (if any) is above the tested range.

**Model config**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 parameters; EMA shadow at decay=0.999; torch_compile_active=True (default mode, dynamic=True)
- **CautiousAdamW lr=1.5e-3**, wd=1e-4 (key change from lr=1e-3)
- CosineAnnealingLR(T_max=25, eta_min=1.5e-4, eta_min_factor=0.10)
- bf16 AMP, surf_weight=10.0, surf_p_l1_weight=1.0
- bernoulli_residual=False
- Peak VRAM: 24.38 GB (identical to lr=1e-3 baseline — LR change has zero memory footprint)
- Avg s/epoch: 57.7
- Cautious mask mean: 0.609 (invariant — matches all prior measurements 0.61 ± 0.01)

**Metric artifacts**
- `models/model-charliepai2i24h5-thorfinn-lr1p5e3_compile-20260516-073113/metrics.jsonl`
- `models/model-charliepai2i24h5-thorfinn-lr1p5e3_compile-20260516-073113/metrics.yaml`
- `models/model-charliepai2i24h5-thorfinn-lr1p5e3_compile-20260516-073113/config.yaml`
- (Arm B): `models/model-charliepai2i24h5-thorfinn-lr2e3_compile-20260516-083024/metrics.jsonl`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-thorfinn \
    --experiment_name "baseline_repro_lr1p5e3_compile" \
    --torch_compile \
    --lr 1.5e-3 --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 32 epochs in 30.8 min.)

---

## 2026-05-16 09:15 — PR #3463: Capacity revisit — n_hidden=192 + lr=1e-3 + compile (previous best)

Stacks on top of PR #3666 (lr=1e-3) which is on top of the full round-5 merged baseline: scale-inv loss + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + T_max=25/eta_min_factor=0.10 + torch.compile. Pure capacity change: n_hidden 128 → 192 (1.84M vs 829K params, 2.22× params). Win is real but sub-multiplicative: lr=1e-3 already captured most of the camber_rc capacity headroom, so the compound is smaller than the independent-axis prediction (~39% of predicted gain). The cruise split keeps benefiting from width at any LR. Descent still active at −0.97/epoch at cutoff with LR already at eta_min — the win is wall-clock-bound, not convergence-bound.

**Primary** (n_hidden=192 + lr=1e-3 + compile, full merged stack)
- `val_avg/mae_surf_p` = **53.1915** (best epoch 24 / 50, run cut by 30-min wall clock; **−1.60% vs PR #3666**, **−57.07% vs round-5 anchor**)
- `test_avg/mae_surf_p` = **47.5701** (**−1.19% vs PR #3666**, **−58.40% vs round-5 anchor**)
- **Cumulative round-5 improvement:** −57.07% val_avg (123.88 → 53.19), −58.40% test_avg (114.37 → 47.57). **Eleven compounding wins.**

**Per-split surface pressure MAE (n_hidden=192 + lr=1e-3 + compile)**

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs PR #3666 | Δ test vs PR #3666 |
|---|---:|---:|---:|---:|
| single_in_dist | 55.368 | 51.739 | −1.70% | −2.81% |
| geom_camber_rc | 70.212 | 65.107 | **+2.01%** | **+2.33%** |
| geom_camber_cruise | 33.776 | 27.234 | −6.23% | −8.88% |
| re_rand | 53.411 | 46.201 | −2.97% | **+0.82%** |
| **avg** | **53.1915** | **47.5701** | **−1.60%** | **−1.19%** |

5/8 cells improve, 3 regress. Win concentrated on `geom_camber_cruise` (−6.23% val / −8.88% test). `geom_camber_rc` regresses (+2.0% val / +2.3% test) — lr=1e-3 already drove this split into a regime where added width cannot help; they compete on this OOD-geometry split. `re_rand` test marginally regresses (+0.82%). val_avg and test_avg both beat the gate.

**Model config (n_hidden widened from 128 → 192)**
- Transolver — n_hidden=192, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- **1,843,393 parameters** (vs 829K at n_hidden=128); EMA shadow at decay=0.999; torch_compile_active=True (default mode, dynamic=True)
- CautiousAdamW lr=1e-3, wd=1e-4
- CosineAnnealingLR(T_max=25, eta_min=1e-4, eta_min_factor=0.10)
- bf16 AMP, surf_weight=10.0, surf_p_l1_weight=1.0
- bernoulli_residual=False
- **Peak VRAM: 33.84 GB** (+9.46 GB vs lr=1e-3+n128 baseline; well within 96 GB device)
- Avg s/epoch: 75.5; 24 epochs completed in 30.2 min
- Cautious mask mean: 0.6105 (invariant — matches 0.617 n=128 baseline, 0.614 prior n=192 results)

**Metric artifacts**
- `models/model-charliepai2i24h5-edward-capacity_n192_lr1e3_compile-20260516-072856/metrics.jsonl`
- `models/model-charliepai2i24h5-edward-capacity_n192_lr1e3_compile-20260516-072856/metrics.yaml`
- `models/model-charliepai2i24h5-edward-capacity_n192_lr1e3_compile-20260516-072856/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-edward \
    --experiment_name "baseline_repro_capacity_n192_lr1e3_compile" \
    --torch_compile \
    --n_hidden 192 \
    --lr 1e-3 --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 24 epochs in 30.2 min.)

---

## 2026-05-16 06:42 — PR #3666: Peak LR sweep — lr=1e-3 + compile (previous best)

Stacks on top of full round-5 merged baseline: scale-inv loss + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + T_max=25/eta_min_factor=0.10 + torch.compile. A pure optimizer-hyperparameter change: lr 5e-4 → 1e-3 on the compile-enabled stack. The key finding is that the epoch-1 instability seen at lr=1e-3 without compile (PR #3581: val_avg spiked to ~800, unrecoverable in 17 epochs) **entirely disappears** with compile + `torch.set_float32_matmul_precision("high")` TF32 mode — epoch 1 at lr=1e-3 lands at 367, essentially identical to the compile baseline's ~365. With no recovery deficit, the higher LR finds a strictly better minimum across all 32 epochs: Arm B (lr=1e-3) sits below Arm A (lr=7e-4) from epoch 1 to 32. Cautious mask remains invariant (~0.61) across both arms and all LR levels — masking does not gate the high-LR updates. Both arms still descending at ~−0.5/epoch at cutoff; the 32-epoch budget remains undercooked.

**Primary** (Arm B: lr=1e-3 + compile, full merged stack)
- `val_avg/mae_surf_p` = **54.0564** (best epoch 32 / 50, run cut by 30-min wall clock; **−11.68% vs PR #3582**, **−56.37% vs round-5 anchor**)
- `test_avg/mae_surf_p` = **48.1422** (**−10.86% vs PR #3582**, **−57.90% vs round-5 anchor**)
- **Cumulative round-5 improvement:** −56.37% val_avg (123.88 → 54.06), −57.90% test_avg (114.37 → 48.14). **Ten compounding wins.**

**Per-split surface pressure MAE (Arm B — lr=1e-3 + compile)**

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs PR #3582 | Δ test vs PR #3582 |
|---|---:|---:|---:|---:|
| single_in_dist | 56.328 | 53.236 | −13.92% | −10.05% |
| geom_camber_rc | 68.829 | 63.621 | −10.35% | −9.85% |
| geom_camber_cruise | 36.021 | 29.889 | −13.18% | −12.45% |
| re_rand | 55.047 | 45.823 | −9.92% | −12.10% |
| **avg** | **54.0564** | **48.1422** | **−11.68%** | **−10.86%** |

All 8 val/test cells improve. Uniform gains across OOD-geometry (camber: −10–13%) and in-distribution (−14%) and Re-OOD (−10–12%) splits.

**Arm A (lr=7e-4 + compile):** val_avg=58.3825 (−4.61% vs baseline) — also a winner, but Arm B wins decisively.

**Mechanism — why compile + TF32 kills the epoch-1 spike:**
The thesis from PR #3581 was that lr=1e-3 caused catastrophic parameter perturbations in the first optimizer step, producing unrecoverable early-epoch spikes. With compile + `torch.set_float32_matmul_precision("high")`, the TF32 matmul numerical characteristics change the effective Hessian in the first step, apparently landing in a better initial regime. This is an empirical observation — a clean test (compile + "highest" precision) was suggested by thorfinn as a future probe, but is not a blocking issue.

**Model config (architecture unchanged from PR #3582)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 parameters; EMA shadow at decay=0.999; torch_compile_active=True (default mode, dynamic=True)
- **CautiousAdamW lr=1e-3**, wd=1e-4 (key change from lr=5e-4)
- CosineAnnealingLR(T_max=25, eta_min=5e-5, eta_min_factor=0.10)
- bf16 AMP, surf_weight=10.0, surf_p_l1_weight=1.0
- bernoulli_residual=False (per compile-baseline stack)
- Peak VRAM: 24.38 GB (identical to compile baseline — LR change has no memory footprint)
- Avg s/epoch: 57.7 (Arm B), 58.4 (Arm A)

**Metric artifacts**
- `models/model-charliepai2i24h5-thorfinn-lr1e3_compile-20260516-052709/metrics.jsonl`
- `models/model-charliepai2i24h5-thorfinn-lr1e3_compile-20260516-052709/metrics.yaml`
- `models/model-charliepai2i24h5-thorfinn-lr1e3_compile-20260516-052709/config.yaml`
- (Arm A): `models/model-charliepai2i24h5-thorfinn-lr7e4_compile-20260516-042552/metrics.jsonl`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-thorfinn \
    --experiment_name "baseline_repro_lr1e3_compile" \
    --torch_compile \
    --lr 1e-3 --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 32 epochs in 30 min with compile + lr=1e-3.)

---

## 2026-05-16 04:21 — PR #3582: torch.compile() for more effective epochs (previous best)

Stacks on top of PR #3465 T_max=25 alignment + PR #3466 Cautious AdamW + PR #3373 bf16 + PR #3265 FiLM + PR #3337 surf-L1 + PR #3281 EMA + PR #3266 scale-invariant loss + NaN fix. Wraps the Transolver in `torch.compile(mode="default", dynamic=True)` to reduce per-forward overhead. **Result: 1.88× per-epoch speedup (108s → 57s), unlocking 32 effective epochs within the 30-min wall-clock cap vs 17 at prior baseline.** The −18.83% val improvement comes entirely from the additional 15 epochs — at matched epoch number the compile arm is ~3% worse than the pre-compile baseline (small fp32 accumulation-order differences from kernel fusion). Model still descending at −0.7/epoch at the epoch-32 cutoff; schedule realignment is the next axis.

> **Note on verified stack state:** Both compile arms ran with `bernoulli_residual=False` (confirmed via config.yaml). Investigation revealed the prior baseline (`thorfinn-tmax25_em10` at commit `3a3104a`) also ran without bernoulli — the field was absent from that branch's Config dataclass. The actual merged stack is a **7-mechanism stack**: scale-inv + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + T_max=25 + compile. Bernoulli residual claim from PR #3466 is suspect; enabling it properly is a pending cleanup item.

**Primary** (Arm A: `torch.compile(mode="default", dynamic=True)`, full merged stack + compile)
- `val_avg/mae_surf_p` = **61.2023** (best epoch 32 / 50, run cut by 30-min wall clock)
- `test_avg/mae_surf_p` = **54.0076**
- **Cumulative round-5 improvement:** −50.59% val_avg (123.88 → 61.20), −52.78% test_avg (114.37 → 54.01)

**Per-split surface pressure MAE (Arm A — winner)**

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs prior (75.40) | Δ test vs prior (65.86) |
|---|---:|---:|---:|---:|
| single_in_dist | 65.4371 | 59.1840 | −22.9% | −19.9% |
| geom_camber_rc | 76.7732 | 70.5756 | −10.6% | −8.6% |
| geom_camber_cruise | 41.4908 | 34.1383 | −28.0% | −26.8% |
| re_rand | 61.1080 | 52.1313 | −16.5% | −20.6% |
| **avg** | **61.2023** | **54.0076** | **−18.83%** | **−18.00%** |

All 8 val/test cells improve. Biggest gains: `geom_camber_cruise` (−28% val) and `single_in_dist` (−22.9% val).

**Throughput and VRAM**

| | Baseline (no compile) | Arm A (default) | Arm B (reduce-overhead) |
|---|---:|---:|---:|
| Steady-state s/epoch | 108.0 | **57.3** | 57.4 |
| Effective epochs (30min) | 17 | **32** | 32 |
| Peak VRAM | ~35.5 GB | **24.4 GB** | 26.4 GB |

**Model config (same as prior baseline, with compile wrapper)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- 829,015 parameters (note: prior BASELINE.md listed 662,359 — likely pre-FiLM count; actual merged-stack model is 829K)
- CautiousAdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0
- CosineAnnealingLR(T_max=25, eta_min_factor=0.10, eta_min=5e-5)
- bf16 AMP, EMA decay=0.999, surf_p_l1_weight=1.0
- bernoulli_residual=False (see note above — verify + enable pending)
- **torch_compile=True, torch_compile_mode="default", dynamic=True**
- `dynamic=True` handles variable padded mesh sizes (74K–242K nodes) without recompiling per shape
- EMA deepcopy BEFORE compile so EMA shadow holds raw `Transolver` (avoids compiled module deepcopy issues)
- `torch.set_float32_matmul_precision("high")` enables TF32 when compile is active
- Cautious mask mean ≈ 0.617 (stable throughout, invariant to LR per PR #3581 close)

**Metric artifacts**
- `models/model-charliepai2i24h5-fern-torch_compile_default-20260516-015149/metrics.jsonl`
- `models/model-charliepai2i24h5-fern-torch_compile_default-20260516-015149/metrics.yaml`
- `models/model-charliepai2i24h5-fern-torch_compile_default-20260516-015149/config.yaml`

**Reproduce**
```bash
cd target/ && python train.py \
    --agent charliepai2i24h5-fern \
    --experiment_name "baseline_repro_compile" \
    --torch_compile \
    --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 \
    --epochs 50
```
(Wall clock capped by `SENPAI_TIMEOUT_MINUTES`; run hit 32 epochs in 30 min with compile enabled.)

---

## 2026-05-16 02:15 — PR #3465: Cosine schedule T_max alignment (current best)

Stacks on top of PR #3466 Bernoulli residual + PR #3315 Cautious AdamW + PR #3373 bf16 + PR #3265 FiLM + PR #3337 surf-L1 + PR #3281 EMA + PR #3266 scale-invariant loss + NaN fix. Changes `CosineAnnealingLR` from `T_max=50` (default, predated bf16 epoch unlock) to `T_max=25` to match the actual training epoch budget with bf16 (~17–19 epochs). Also sets `eta_min_factor=0.10` (min LR = `0.1 × 5e-4 = 5e-5`) so the cosine curve still has gradient signal at its floor — whereas the previous default `eta_min=0` caused the LR to zero out before the run hit wall clock. **This is a pure schedule fix: the cosine curve now actually decays through a useful range within the wall-clock window instead of lingering at near-zero LR for half its scheduled life.** Mechanism is orthogonal to all 7 previously-merged axes (optimizer, precision, conditioning, loss, target reformulation, EMA averaging).

> **Note on tested code state:** Validated at commit `3a3104a` — the post-Bernoulli-residual tip of the merged advisor branch. The full 8-mechanism compound result: scale-inv + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + Bernoulli residual + T_max=25/eta_min_factor=0.10. **Arm A (T_max=19, eta_min_factor=0.05)** also beats prior baseline at val_avg=80.12, confirming that any T_max alignment shorter than 50 improves — but Arm B's longer window with a maintained LR floor wins.

**Primary** (Arm B: T_max=25, eta_min_factor=0.10, full merged stack)
- `val_avg/mae_surf_p` = **75.4040** (best epoch 17 / 50, run cut by 30-min wall clock; **−12.42% vs PR #3466**, **−39.15% vs round-5 anchor PR #3266**)
- `test_avg/mae_surf_p` = **65.8592** (**−15.04% vs PR #3466**, **−42.45% vs round-5 anchor**)

**Per-split surface pressure MAE (Arm B — T_max=25, eta_min_factor=0.10)**

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 84.8767 | 73.8438 |
| geom_camber_rc | 85.9009 | 77.2500 |
| geom_camber_cruise | 57.6499 | 46.6446 |
| re_rand | 73.1886 | 65.6982 |
| **avg** | **75.4040** | **65.8592** |

**Per-split deltas vs PR #3466 Bernoulli baseline (86.09 val):**
- `single_in_dist` val: 102.04 → 84.88 (−16.8%), test: 93.07 → 73.84 (−20.7%)
- `geom_camber_rc` val: 100.12 → 85.90 (−14.2%), test: 90.18 → 77.25 (−14.3%)
- `geom_camber_cruise` val: 62.99 → 57.65 (−8.5%), test: 52.40 → 46.64 (−11.0%)
- `re_rand` val: 79.23 → 73.19 (−7.6%), test: 74.37 → 65.70 (−11.7%)
All 8 cells improved — no regressions. Biggest gain on `single_in_dist` and `geom_camber_rc`.

**Arm A (T_max=19, eta_min_factor=0.05):** val_avg=80.1207 (−6.95% vs #3466) — also beats prior baseline but Arm B wins.

**Why Arm B won over Arm A:** T_max=19 set decay tail at exactly the wall-clock epoch count, causing LR to reach `eta_min` right at the cutoff — the last 2–3 epochs had near-zero gradient signal. T_max=25 extends the cosine curve slightly beyond the budget, so LR at epoch 17 was still 1.79e-4 (vs a near-zero for T_max=19 with `eta_min=0`). Combined with `eta_min_factor=0.10` maintaining a non-zero floor, Arm B keeps gradient signal alive at cutoff.

**Cautious mask dynamics:** `train/cautious_mask_mean` flat at 0.617 across all 17 epochs — schedule realignment did not perturb the cautious gating equilibrium.

**Model config (architecture unchanged; schedule-only change)**
- Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; FiLM per-block conditioning
- 829,015 raw params; EMA shadow copy at decay=0.999
- Optimizer: CautiousAdamW lr=5e-4, wd=1e-4
- bf16 mixed precision (peak VRAM ~35.46 GB)
- batch_size=4, surf_weight=10.0, surf_p_l1_weight=1.0
- **CosineAnnealingLR(T_max=25, eta_min=5e-5)** — the key change
- Loss: per-sample scale-invariant MSE + surf_p_l1_weight=1.0 * surf_p_l1 (on Bernoulli residual target)

**Metric artifacts (Arm B)**
- `models/model-charliepai2i24h5-thorfinn-tmax25_em10-20260515-232319/metrics.jsonl`
- `models/model-charliepai2i24h5-thorfinn-tmax25_em10-20260515-232319/metrics.yaml`
- `models/model-charliepai2i24h5-thorfinn-tmax25_em10-20260515-232319/config.yaml`

**Reproduce**
```bash
cd target/
python train.py --agent charliepai2i24h5-thorfinn \
    --experiment_name "charliepai2i24h5-thorfinn/tmax25_em10" \
    --t_max 25 --eta_min_factor 0.10 \
    --surf_p_l1_weight 1.0 --bernoulli_residual freestream \
    --epochs 50
```

**Cumulative round-5 improvement:** **−39.15% val_avg** (123.88 → 75.40) and **−42.45% test_avg** (114.37 → 65.86) vs the pre-round-5 floor. **Eight compounding wins in sequence**: scale-inv → EMA → surf-L1 → FiLM → bf16 → Cautious AdamW → Bernoulli residual → T_max schedule alignment.

---

## 2026-05-16 00:00 — PR #3466: Bernoulli pressure residual (previous best)

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
