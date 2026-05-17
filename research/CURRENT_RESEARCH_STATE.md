# SENPAI Research State

- 2026-05-17 12:40Z — round 15 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **NEW BASELINE: val=35.5046** (PR #4477 alphonse GeGLU Arm A merged, −2.89% from 36.5616). Gate-activation sweep: GeGLU > SwiGLU > BilinearGLU > GELU MLP. Cumulative: 135.02 → 35.50 = **-73.7%**

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| #3293 (nezuko, Lion) | 117.5014 | −7.8% | AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4 |
| #3357 (tanjiro, asinh-loss) | 84.9819 | −27.7% | asinh(z) on pressure channel z-scores in training loss |
| #3382 (askeladd, EMA+asinh) | 83.1874 | −2.1% | EMA shadow decay=0.999 at val/test passes |
| #3384 (fern, grad-clip+EMA+asinh) | 70.2479 | −15.6% | grad_clip(max_norm=1.0) before optimizer.step() |
| #3530 (frieren, surf_weight=25) | 67.2991 | −4.20% | surf_weight: 30→25 (5-mech stack now complete) |
| #3485 (alphonse, bf16 autocast) | 58.8717 | −12.5% | bf16 autocast on forward+loss; 18 epochs vs 14 |
| #3822 (edward, cosine T_max=30) | 56.0011 | −4.88% | CosineAnnealingLR T_max: 80→30; moderate late-epoch anneal |
| #3674 (nezuko, pressure_weight=2.0) | 53.7235 | −4.07% | pressure_weight: 1.0→2.0; up-weights pressure channel in training loss |
| #3989 (askeladd, EMA decay=0.995) | 51.4403 | −4.25% | EMA shadow decay: 0.999→0.995; faster tracking under T_max=30+pw=2.0 |
| #3970 (alphonse, torch.compile) | 44.2439 | −14.0% | torch.compile(mode=default, dynamic=True); 102s→54s/epoch; 18→33 epochs |
| #3953 (frieren, LR×T_max) | 40.6869 | −8.04% | lr: 1.7e-4→2.5e-4; T_max: 30→40 — SUPERSEDED by #4079 |
| **#4079 (edward, T_max=40)** | **39.8345** | **−9.97%** | **T_max: 30→40; lr=1.7e-4 unchanged — T_max alone drives full gain** |
| **#4243 (askeladd, slice_num=48)** | **38.6750** | **−2.91%** | **slice_num: 64→48; coarser slicing → lower-variance gradient signal per step** |
| **#4358 (alphonse, SwiGLU param-matched)** | **36.5616** | **−5.47%** | **SwiGLU hidden_mult=0.6667: multiplicative SiLU gate × value branch; 14-exp plateau broken** |
| **#4477 (alphonse, GeGLU param-matched)** | **35.5046** | **−2.89%** | **GeGLU (GELU gate): gating structure dominant; gate activation secondary; ablation confirms both smooth gates beat bilinear** |

**Current HEAD (15 mechanisms):** Lion lr=**1.7e-4** + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine **T_max=40** + pressure_weight=2.0 + torch.compile(mode=default, dynamic=True) + **slice_num=48** + **GeGLU(hidden_mult=0.6667, glu_act_type=geglu)**. val=35.5046 at epoch 32 (timeout, still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 40 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default \
    --slice_num 48
```
(In-tree defaults: lr=1.7e-4 is CORRECT; T_max=80→must pass 40; surf_weight=30→must pass 25; pressure_weight=1.0→must pass 2.0; ema_decay=0.999→must pass 0.995; compile_mode=none→must pass default; **slice_num=64→must pass 48**.)

**Cumulative improvement from initial baseline:** 135.02 → 38.675 = **−71.3%**

**Compute profile:** ~51.7s/epoch, 35 epochs in 30 min, 22.60 GB VRAM peak.

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #4556 | askeladd | Grad-clip max_norm loosening: 2.0 (A) / 3.0 (B) — clip_frac=1.000 diagnostic | WIP | #4487 CLOSED (grad-noise catastrophic; clip_frac=1.000 revealed max_norm=1.0 clips every step); loosening may allow higher-confidence updates |
| #4572 | fern | Geometry-only channel dropout: mask dims 0-11 (pos/saf/dsdf) only at p=0.05 (A) / p=0.15 (B) | WIP — NEW | #4501 CLOSED (catastrophic: indiscriminate masking of conditioning dims 12-23 destroys log_Re/AoA/NACA signal; geometry-only re-scoped version) |
| #4458 | nezuko | Attention temperature: frozen τ=0.25 (A) vs τ=0.125 (B) on SwiGLU baseline | WIP — RERUNNING | Original arms beat OLD baseline (38.675) but regress vs SwiGLU baseline (36.5616); sent back with sharpened arms + `--use_swiglu` |
| #4558 | tanjiro | Y-mirror v3: geometry-only flip (pos_y+saf_1) at p=0.15 (A) / p=0.30 (B) on GeGLU baseline | WIP | #4499 CLOSED (p=0.5 in-dist tax +3-5% > OOD benefit +0.8% on camber_rc; geometry-only confirmed correct; lower p should flip sign) |
| #4553 | alphonse | ReGLU (ReLU gate): completes gate-activation sweep + SwiGLU calibration arm | WIP | #4477 MERGED (GeGLU wins val=35.5046); gate sweep: GeGLU>SwiGLU>BilinearGLU; ReGLU is the remaining activation to test |
| #4576 | thorfinn | LayerScale (CaiT) re-test on GeGLU: γ_init=1e-4 (A) / 3e-5 (B) — informed by #4435 γ trajectory | WIP — NEW | #4435 CLOSED (beat old baseline 38.675→37.60 but baseline moved to 35.50; GeGLU + LayerScale orthogonal gating levels — MLP-inside vs block-outside) |
| #4528 | edward | Slice-routing noise: Gaussian σ=0.10 (A) / σ=0.30 (B) on PhysicsAttention logits — **WIN on SwiGLU baseline (val=35.15 < GeGLU's 35.50)**, rebase + GeGLU re-run pending | WIP — REBASING | #4454 CLOSED (feature-noise absorbed by routing); val_geom_camber_rc dropped 52.26→46.91 (-10%) — **biggest OOD breakthrough this round** |
| #4590 | frieren | Prediction-head GeGLU: replace mlp2 (A) and preprocess MLP (B) with GeGLU activation | WIP — NEW | #4514 CLOSED (5th regularization failure: Re-jitter corrupts regime-selection signal); structural pivot to boundary-MLP gating |

## Key open questions (round 15 — baseline 35.5046 after GeGLU win)

**Plateau broken THREE TIMES!** SwiGLU #4358 (-5.47%) → GeGLU #4477 (-2.89%) → slice-routing-noise #4528 σ=0.30 val=35.1446 (on SwiGLU baseline) — **on the rebase to GeGLU this is expected to be the round's 3rd win, with the biggest OOD breakthrough yet: val_geom_camber_rc 52.26 → 46.91 (-10.2%)**. The routing-logit perturbation is the right level of intervention per the #4454+#4377 lessons. Active pivot: (a) **gate-activation sweep completion** (#4553 ReGLU); (b) **grad-clip diagnostic** (#4556 askeladd — clip_frac=1.000); (c) **y-mirror at lower p** (#4558 tanjiro geom-only p=0.15/0.30); (d) **input augmentation** (#4572 fern geometry-only channel-dropout); (e) **slice-routing-noise on GeGLU** (#4528 edward rebase pending); (f) **architecture** (#4576 thorfinn layerscale-geglu, #4458 nezuko attn-temp); (g) **boundary-MLP gating** (#4590 frieren pred-head-geglu). Key insight from #4528: the routing operator can be regularized but only by attacking its logits, not its inputs. Key insight from #4514: log(Re) is a *regime-selection* signal not interpolation — Re-jitter corrupts conditioning (5th reg failure).

**Closed axes:** optimizer tuning, loss reshaping, LR-schedule disruption, normalization form (LayerNorm locked), element-level reg, structural regularization (DropPath — 4th reg failure; stack NOT reg-limited), input-representation (Fourier — convergence budget hit), point-level data aug (slice routing invariant to point perturbation). **Critical diagnostic from #4377:** PhysicsAttention slice routing is permutation-equivariant — augmentation must operate in *token space*, not point space. **Key insight from 4 reg failures:** remaining val gap (≈36.5) is capacity/data limitation at OOD splits, not overfitting — pivot to data augmentation + capacity.

**Key reading from 3 regularization failures (FFN dropout, attention dropout, SWA):** model is NOT over-fitting in the classic sense at 35 epochs. The OOD gap on val_geom_camber_rc (51.62) is driven by **training data coverage**, not by parameter over-fitting. This pivots us toward data augmentation as the right axis.

1. **Does GELU gate (GeGLU) match SwiGLU, or does bilinear gating alone suffice?** (#4477 alphonse) — SwiGLU MERGED; now ablating GELU gate (Arm A) vs identity gate (Arm B, bilinear); tests if SiLU is essential or if any multiplicative gating works.
2. **Does gradient noise injection (σ=0.01/0.03) improve OOD generalization?** (#4487 askeladd) — Neelakantan 2015: gradient noise finds flat minima, improves generalization; parameter-space noise is orthogonal to all other active experiments; targets camber_rc OOD bottleneck.
3. **Does input channel-level dropout (p=0.10/0.20) force OOD feature robustness?** (#4501 fern) — stochastic masking of entire input channels to zero; forces model to not over-rely on any single feature (NACA, AoA) when OOD; different from edward's continuous noise: discrete absence vs perturbation. Loss-weighting axis closed → new input-regularization axis.
4. **Does slice-routing logit noise force routing-robust representations?** (#4528 edward) — Gaussian σ=0.10 (A) / σ=0.30 (B) injected directly onto PhysicsAttention routing logits (after τ division, before softmax); attacks routing output not routing input; #4454 showed feature-noise on x_norm is absorbed by routing operator; this is the correct level of intervention per the #4377 diagnostic.
5. **Does forcing a fixed slice-routing temperature outperform the learnable τ?** (#4458 nezuko) — frozen τ=0.25 (A, sharper) vs τ=1.0 (B, neutral); slice-routing softmax temperature has been learnable per-head init=0.5; never tested.
6. **Does Reynolds-number jitter improve Re-dimension OOD generalization?** (#4514 frieren) — physics-motivated Gaussian noise on log(Re) channel (ch13) only; per-sample to preserve within-sample Re consistency; σ=0.05 (mild) vs σ=0.15 (moderate); primary target: val_re_rand, secondary: val_geom_camber_rc.
7. **Does Y-mirror v2 (post-norm bias correction) succeed where v1 was broken?** (#4499 tanjiro) — v1 failed due to normalization-asymmetry bug (disjoint normalized distributions); v2 applies `x_norm[k] = -x_norm[k] - 2·mean[k]/std[k]` to give physics-exact symmetric pairs. Arm A: full field flip; Arm B: geometry-only. Critical OOD signal from v1: camber_rc was *least* regressed (+12.4%) — confirms direction is correct.
8. **Does LayerScale (CaiT) enable the model to learn per-channel block contribution?** (#4435 thorfinn) — learnable γ per residual branch; γ_init=1e-4 (A) vs 1e-2 (B); allows near-identity start and automatic block gating; 1280 new params. NOTE: rebasing onto SwiGLU baseline.

## 12-mechanism stack: full pipeline

1. **Lion (sign-based update)**: optimizer; lr=1.7e-4, betas=(0.9, 0.99), wd=3e-4
2. **surf_weight=25**: loss level
3. **asinh**: loss-level; per-coordinate pressure z-score compression
4. **EMA(0.995)**: parameter level; exponential trajectory smoothing
5. **grad-clip(max_norm=1.0)**: gradient vector level; L2 norm cap
6. **bf16 autocast**: compute level; forward+loss precision reduction
7. **cosine T_max=40**: schedule level; calibrated to 33-epoch compile horizon; ~7% LR floor at epoch 34
8. **pressure_weight=2.0**: loss level; up-weights pressure channel MAE 2×
9. **EMA decay=0.995**: EMA parameter; tighter half-life
10. **torch.compile(default, dynamic=True)**: kernel fusion; 47% faster/epoch, 33-34 epochs in 30 min (now 35 at slice=48)
11. **slice_num=48**: PhysicsAttention routing slices reduced from 64 to 48; coarser slicing → lower-variance gradient signal per step; 4.6% faster per epoch

[Note: frieren's lr=2.5e-4 from #3953 was superseded by edward's #4079 reverting to lr=1.7e-4. Net effect of both merges: T_max 30→40 only. lr=1.7e-4 restored as optimal.]

## Closed / falsified experiments

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression |
| #3328 (askeladd surf_weight=50) | +25% regression |
| #3329 (fern AdamW β2=0.95) | +21% regression |
| #3102 (edward OneCycleLR) | +20% regression |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores light-tailed |
| #3099 (alphonse capacity 192h/6L/6H) | +60.5% regression; throughput-bound at 18 epochs — retested in #4078/#4167 |
| #3106 (frieren Slice128/head8/mlp3) | +98.6% regression |
| #3354 (nezuko Lion+cosine T_max=12) | +15.96% regression |
| #3586 (nezuko higher LR 2.5e-4) | +2.74% regression (at T_max=80; lr=2.5e-4 also confirmed suboptimal at T_max=40 vs 1.7e-4) |
| #3656 (frieren surf-weight-fine sw=22/27) | +4.09% regression |
| #3528 (fern grad-clip rebased sw×max_norm grid) | +0.57% regression; max_norm=1.0 optimal |
| #3725 (fern per-group grad-clip attn vs MLP) | no_improvement: val +1.1–3.9%; diagnostic revealed MLP/output ~5× noisier; #4154 tests loosening |
| #3887 (edward T_max bracket 25/40) | no_improvement on 18-epoch budget; T_max=30 narrow peak — retested at #4079 (winner) |
| #3984 (nezuko pw=3.0/4.0) | no_improvement: inverted-U peaks at pw=2.0; velocity degradation above 2.0 |
| #3949 (askeladd lion-beta1) | no_improvement: β1=0.95 +9.8%; β1=0.90 optimal; Lion-β1 axis closed |
| #4031 (edward lion-beta2) | no_improvement: β2=0.95 +27% failure; β2=0.99 optimal; Lion-β2 axis closed |
| #3731 (tanjiro signed-log1p v2) | no_improvement: +12.93%; signed-log1p over-compresses moderate |z|; asinh locally optimal |
| #3442/#3383/#3275 (stale closed) | Reassigned as v2 |
| #3776/#3726 (rate-limit failures) | Closed without results |
| #3733 (edward warmup-cosine v2) | +4.2% regression |
| #3750 (alphonse capacity-bf16) | no_improvement on old stack; throughput-bound — retested in #4078/#4167 |
| #3884 (alphonse batch-size-bf16) | no_improvement: batch=6 +24.4%; steps/epoch −33% |
| #4016 (fern tighter MLP clip) | no_improvement: arms A/B regressed; tighter other_grad_norm discards MLP signal; → #4154 |
| #4078 (alphonse capacity-compile) | no_improvement: n192/n256 throughput-limited; T_max=30 miscalibrated for n192's ~22-epoch budget → #4167 |
| #4167 (alphonse capacity-n192-tmax22) | no_improvement: val=50.32 (+26.3%); T_max=22 calibration was correct but model still throughput-limited at 79s/epoch (23 epochs in budget vs n128's 34); capacity direction closed → #4188 compile-mode-sweep |
| #3734 (thorfinn swiglu-v2) | closed_stale: 17h elapsed, 45 pod restarts, 0% GPU, no commits — SwiGLU twice blocked (v1: #3275/#3383/#3442, v2: #3734); architectural complexity exceeds available implementation budget → #4230 weight-decay-sweep |
| #4188 (alphonse compile-mode-sweep) | no_improvement: reduce-overhead val=40.43 (+1.5%), max-autotune val=40.39 (+1.4%); compile-time costs (+20s/+200s) negate per-step gains; default mode optimal → #4235 mlp-ratio-sweep |
| #4159 (frieren tmax-fine-sweep) | no_improvement: T_max=35 val=42.70 (+7.2%, schedule freezes), T_max=50 val=40.20 (+0.92% noise); confirmed T_max=40 optimal at lr=1.7e-4 → #4236 warmup-cosine |
| #4154 (fern loosen-other-clip) | no_improvement: other=1.5 val=40.63 (+2.0%), other=2.0 val=42.04 (+5.5%); per-group(1.0,1.0) "win" in #4016 was magnitude-inflation noise → #4237 n-layers-sweep |
| #4029 (askeladd EMA-decay-fine) | no_improvement: 0.993 val=39.79 (+noise, test +0.79 regression), 0.990 not run (below threshold); EMA decay=0.995 locked → #4243 slice-num-sweep |
| #4181 (edward lr-fine-sweep) | no_improvement: lr=1.5e-4 val=40.39 (+1.4% null), lr=2.0e-4 val=41.87 (+5.1% failure); lr=1.7e-4 locked as optimum for T_max=40; LR axis fully closed → #4253 sgdr-warm-restarts |
| #4236 (frieren warmup-cosine) | no_improvement: Arm A (warmup=2) val=41.28 (+3.6%), stop condition triggered; Arm B not run; schedule-compression effect dominates; grad-clipping already provides stabilization; warmup direction closed (2nd failure, cf #3733) → #4287 batch-size-sweep |
| #4061 (tanjiro decoupled-heads) | no_improvement: Arm A (2-layer p head) val=40.05 (+0.53% null) but test=35.22 (+3.92%); Arm B (3-layer) val=41.93 (+5.3%); 12-mech already neutralized shared-head bottleneck (asinh+pw=2.0) → #4273 n-head-sweep |
| #4030 (nezuko vel-surf-weight arms C/D) | no_improvement: Arm C (0.7) val=39.49 (within noise) but test=34.46 (+1.69%); Arm D (0.8) val=40.76 (+2.3%); vel-surf-weight axis closed at lr=1.7e-4 → #4278 attention-dropout |
| #4237 (fern n-layers-sweep) | no_improvement: A (n=6) val=42.33 (+2.50, 28ep, 66s/ep), B (n=7) val=46.99 (+7.16, 24ep, 76s/ep); throughput-bound AND no iso-epoch advantage; MLP/attn imbalance amplified by depth → #4295 per-group-lr |
| #4235 (alphonse mlp-ratio-sweep) | no_improvement: A (r=3) val=40.26 (+1.08%, 31ep), B (r=4) val=43.10 (+8.20%, 29ep); throughput-bound, monotone: more MLP width → slower → fewer epochs → worse val; MLP-ratio axis closed at r=2 → #4308 ffn-dropout |
| #4230 (thorfinn weight-decay-sweep) | no_improvement: A (wd=1e-4) val=41.99 (+5.4%, stop), B (wd=5e-4) val=43.48 (+9.2%, stop); both hit stop cond (>41.0); monotone ordering confirms wd=3e-4 at optimum → #4312 swa |
| #4287 (frieren batch-size-sweep) | failure: A (batch=8) val=45.26 (+13.6%, fail >45), B (batch=12) val=63.51 (+59.4%, fail); sub-linear time scaling, fewer gradient updates dominate, in-dist split suffered most (undertraining); batch axis closed at batch=4 with LR locked → #4327 huber-loss |
| #4295 (fern per-group-lr) | no_improvement: Arm A (lr_other×0.5) val=40.07, Arm B (lr_attn×2.0) val=39.50; Lion sign-update masks grad-norm asymmetry; per-group LR only changes trajectory length not step size → #4414 surf-p-weight-mult |
| #4365 (tanjiro rmsnorm) | no_improvement: Arm A (scope=all) val=39.79 (+1.12), Arm B (scope=pre_only) val=39.97 (+1.29); mean-centering load-bearing in this architecture; post-norm slightly better than pre-only; LayerNorm locked → #4433 y-mirror |
| #4362 (thorfinn lookahead-lion) | no_improvement: Arm A (k=5) val=43.53 (+12.6%), Arm B (k=10) val=41.92 (+8.4%); triple-smoothing: Lion sign-update + EMA + Lookahead redundant; k=5→k=10 monotone trend confirms k→∞ optimal → #4435 layerscale |
| #4403 (edward fourier-features) | no_improvement: Arm A (NeRF K=8) val=42.45 (+3.77), Arm B (RFF K=16 σ=10) val=42.08 (+3.40); +32 input channels hit convergence budget; OOD got worse; redundant with PhysicsAttention slice routing → #4454 feature-noise |
| #4377 (nezuko point-subsample) | no_improvement: Arm A (keep=0.8) val=40.65 (+5.10%), Arm B (keep=0.6) val=39.85 (+3.04%); critical diagnostic — **slice routing is permutation-equivariant and invariant to point subsampling**; augmentation must operate in TOKEN space → #4454 feature-noise, #4458 attn-temperature |
| #4306 (askeladd slice-num-coarser) | no_improvement: slice=40 val=41.27 (+12.9% vs new baseline), slice=32 val=38.94 (+6.5% vs new baseline); U-shape confirmed: 96=42.54, 64=39.83, **48=36.56 [optimum]**, 40=41.27, 32=38.94; camber_rc bottleneck dominates even Arm B (which beats on 3 of 4 test splits) → #4487 grad-noise |
| #4414 (fern surf-p-weight-mult) | no_improvement: Arm A (mult=1.5) val=40.54 (+4.83%), Arm B (mult=2.0) val=40.68 (+5.18%); monotone regression; cross-channel coupling (p↔Ux/Uy) means obsessing on surface-p hurts everything; even vol_p degraded; loss-weighting axis fully closed → #4501 channel-dropout |
| #4433 (tanjiro y-mirror) | no_improvement: Arm A (p=0.5) val=45.79 (+18.4%), Arm B (p=1.0) diverged to 976; ROOT CAUSE: stats.json computed on unmirrored data → disjoint normalized distributions; OOD SIGNAL: camber_rc was *least* regressed (+12.4% vs +18-30% others) → direction correct but normalization broken → #4499 y-mirror-v2 (post-norm bias correction) |

## Potential next research directions

- **T_max fine-sweep** — CLOSED as #4159; T_max=40 optimal
- **torch.compile mode sweep** — CLOSED as #4188; default mode optimal
- **LR fine-sweep** — CLOSED as #4181; lr=1.7e-4 locked
- **EMA decay** — CLOSED as #4029; decay=0.995 locked
- **MLP-ratio sweep** — CLOSED as #4235; mlp_ratio=2 optimal
- **FFN dropout** — CLOSED as #4308; MLP not over-fitting; noise hurts; axis closed → #4358 SwiGLU
- **Depth sweep** — CLOSED as #4237; throughput-bound; axis closed
- **Warmup-cosine** — CLOSED as #4236 (2nd failure); direction closed
- **Batch size** — CLOSED as #4287; batch=4 locked with LR fixed
- **Weight decay** — CLOSED as #4230; wd=3e-4 optimal
- **SWA** — CLOSED as #4312; SWALR freezes learning; EMA superior in timeout-limited regime → #4362 Lookahead
- **n_head sweep** — CLOSED as #4273 v2; n_head=4, head_dim=32, slice=48 is local optimum (strong n_head×slice interaction found); axis closed → #4365 RMSNorm
- **Slice_num sweep** — MERGED as #4243 (slice=48 STRONG WIN)
- **SwiGLU gating** — MERGED as #4358 (alphonse); Arm A param-matched wins strongly (-5.47%); gating mechanism unlocks new local optimum → #4477 geglu/bilinear ablation
- **Lookahead-Lion** — CLOSED as #4362 (thorfinn); triple-smoothing (Lion+EMA+Lookahead); k=5→k=10 monotone trend confirms k→∞ optimal → #4435 LayerScale
- **RMSNorm** — CLOSED as #4365 (tanjiro); mean-centering load-bearing; both arms regress → #4433 Y-mirror
- **Huber loss** — CLOSED as #4327 (frieren); asinh already handles outliers; loss-reshaping slows convergence under timeout → #4405 DropPath
- **SGDR warm restarts** — CLOSED as #4253 (edward); restart spike, 3rd schedule-disruption failure → #4403 Fourier features
- **Slice_num coarser** — CLOSED as #4306 (askeladd); slice=40/32 both worse than optimum at slice=48; U-shape fully confirmed
- **Fourier feature encoding** — CLOSED as #4403 (edward); +32 input channels hit convergence budget; OOD got worse; redundant with PhysicsAttention slice routing on pos → #4454 feature-noise
- **DropPath stochastic depth** — CLOSED as #4405 (frieren); 4th consecutive regularization-family failure; stack confirmed NOT regularization-limited → #4514 Re-jitter (physics augmentation pivot)
- **Attention dropout** — CLOSED as #4278 (nezuko); both arms regress on bottleneck split; train/val gap GROWS with dropout (signal-removal not co-adaptation); 3rd regularization failure → #4377 point-subsample
- **Point subsampling augmentation** — CLOSED as #4377 (nezuko); slice routing invariant to point-level perturbation; critical diagnostic — token-space diversity needed → #4454 feature-noise (token-space), #4458 attn-temperature
- **Per-group LR** — CLOSED as #4295 (fern); Arm A (lr_other×0.5) val=40.07, Arm B (lr_attn×2.0) val=39.50; Lion sign-update masks grad-norm asymmetry; per-group LR only changes trajectory length → #4414 surf-p-weight-mult
- **Surface-only pressure-weight multiplier** — CLOSED as #4414 (fern); surf_p_weight_mult=1.5/2.0 both regressed; loss-weighting axis fully closed → #4501 channel-dropout
- **Y-mirror geometric augmentation** — CLOSED as #4433 (tanjiro); normalization-asymmetry bug caused disjoint distributions → catastrophic failure on Arm B; OOD signal in Arm A confirms direction; v2 fix in #4499
- **Y-mirror v2 (post-norm correction)** — IN PROGRESS as #4499 (tanjiro); fixes v1 normalization bug; Arm A full-flip, Arm B geometry-only
- **Surface-only pressure-weight multiplier** — CLOSED as #4414 (fern); monotone regression; loss-weighting axis fully closed
- **Input channel-level dropout (full)** — CLOSED as #4501 (fern); catastrophic failure — dims 12-23 are globally-unique conditioning, no peer redundancy; → #4572 geometry-only (dims 0-11 only)
- **Geometry-only channel-level dropout** — IN PROGRESS as #4572 (fern); masks dims 0-11 (pos/saf/dsdf) at p=0.05 (A) / p=0.15 (B); dims 12-23 untouched
- **LayerScale (CaiT) on old baseline** — CLOSED as #4435 (thorfinn); Arm A val=37.60 beat old baseline (38.68) but not current 35.50; PR conflicting; insights: ls2>>ls1, b4.ls1≈0, γ_init=1e-2 too hot → #4576 re-test on GeGLU
- **LayerScale on GeGLU baseline** — IN PROGRESS as #4576 (thorfinn); γ_init=1e-4 (A) / 3e-5 (B) on current GeGLU stack; orthogonal to GeGLU's intra-MLP gating
- **Token-space input feature noise** — CLOSED as #4454 (edward); Gaussian noise on x_norm is still point-wise; routing operator learns invariance; feature-noise direction closed → #4528 slice-routing-noise (attack logits directly)
- **Slice-routing logit noise** — STRONG WIN on SwiGLU baseline as #4528 (edward); Arm B σ=0.30 val=35.1446 < current GeGLU baseline 35.5046; val_geom_camber_rc -10.2% biggest OOD breakthrough; PR sent back for GeGLU rebase + re-run to confirm compound
- **Re-jitter on log(Re) channel** — CLOSED as #4514 (frieren); 5th reg failure; log(Re) is regime-selection signal not interpolation; corrupting it kills conditioning on every split including primary target val_re_rand
- **Attention temperature (slice-routing softmax)** — IN PROGRESS as #4458 (nezuko); frozen τ=0.25/1.0 vs current learnable init=0.5
- **Prediction-head GeGLU (boundary positions)** — IN PROGRESS as #4590 (frieren); extend GeGLU to mlp2 (A) and preprocess MLP (B); orthogonal to per-block FFN GeGLU; should compound with #4528 slice-routing-noise

**Not yet tried (candidates for next round):**
- GeGLU / BilinearGLU — MERGED as #4477 (alphonse); GELU gate beats SwiGLU (−2.89% val); bilinear regresses → #4553 ReGLU
- ReGLU (ReLU gate) — IN PROGRESS as #4553 (alphonse); completing gate-activation sweep; hard sparsity may help OOD camber_rc
- Pre-norm vs post-norm positioning (current model uses pre-norm; post-norm may require warmup)
- SAM (Sharpness-Aware Minimization) — optimizer meta-algorithm; 2× compute cost; only if clean failing
- Mixup of geometries — interpolate two training samples in loss space (input-level too complex)
- Conditional normalization / FiLM on Re_inf, AoA — #3115 failed on old baseline (+6.3%); worth retesting on current 13-mech stack?
- Per-channel surf loss (separate surf_uxuy_weight from surf_p_weight) — split uniform surf_weight=25
