# SENPAI Research State

- 2026-05-17 06:10Z — round 15 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=38.6750** (PR #4243 askeladd slice_num=48 merged, −2.91% from 39.83)

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

**Current HEAD (13 mechanisms):** Lion lr=**1.7e-4** + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine **T_max=40** + pressure_weight=2.0 + torch.compile(mode=default, dynamic=True) + **slice_num=48**. val=38.675 at epoch 35 (timeout-bound, val still descending).

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
| #4358 | alphonse | SwiGLU activation: gated MLP (SiLU gate) vs GELU on slice=48 stack | WIP | #4308 closed (ffn-dropout regresses); activation-form untested; Arm A param-matched (hidden×2/3), Arm B full hidden |
| #4295 | fern | Per-group LR: lr_attn_mult vs lr_other_mult to rebalance MLP/attn updates | WIP | #4237 closed; depth throughput-bound; MLP/attn 5× grad-norm imbalance (PR #4154) still unaddressed |
| #4306 | askeladd | slice_num coarser: 40 (Arm A), 32 (Arm B) — below new baseline slice=48 | WIP (stale — was rate-limited; GPU active again) | #4243 MERGED (slice=48 strong win); continue coarser direction; trend clear |
| #4377 | nezuko | Point subsampling augmentation: keep_rate=0.8 (Arm A), 0.6 (Arm B) on non-surface points | WIP | #4278 closed (attn-dropout regresses); pivot to DATA augmentation; targets val_geom_camber_rc (51.62) coverage |
| #4362 | thorfinn | Lookahead-Lion: slow/fast weight anchoring (k=5 Arm A, k=10 Arm B) on slice=48 | WIP | #4312 closed (SWA fails); Lookahead preserves cosine schedule, adds trust-region anchoring |
| #4365 | tanjiro | RMSNorm vs LayerNorm: modern normalization on slice=48 stack | WIP | #4273 closed (n_head mechanisms don't compound with slice=48); norm form untested |
| #4403 | edward | Fourier feature encoding of mesh pos coords: NeRF octaves K=8 vs RFF K=16 σ=10 | WIP — NEW | #4253 closed (SGDR 3rd schedule-disruption failure); INPUT REPRESENTATION axis first test; spectral bias hypothesis |
| #4405 | frieren | DropPath stochastic depth: p_max=0.10 (Arm A), 0.20 (Arm B) — block-level vs element-wise | WIP — NEW | #4327 closed (huber regresses); STRUCTURAL REGULARIZATION axis — drops entire blocks not activations |

## Key open questions (round 15 — new baseline 38.675, slice_num=48 — ACTIVE ESCALATION)

**Escalation status:** 9 consecutive no_improvement results since slice=48 merged (now also +#4327 huber-loss, +#4253 SGDR; plus #4278 attn-dropout, #4308 ffn-dropout, #4312 SWA, #4273 n_head v2, + earlier #4235 mlp-ratio, #4230 weight-decay, #4287 batch-size). Loss-function reshaping and LR-schedule disruption are now fully closed directions. Escalating to: (a) architectural changes (SwiGLU, RMSNorm, Lookahead in-flight); (b) **data augmentation** (point subsampling); (c) **input representation** (Fourier features); (d) **structural regularization** (DropPath — block-level vs element-wise).

**Key reading from 3 regularization failures (FFN dropout, attention dropout, SWA):** model is NOT over-fitting in the classic sense at 35 epochs. The OOD gap on val_geom_camber_rc (51.62) is driven by **training data coverage**, not by parameter over-fitting. This pivots us toward data augmentation as the right axis.

1. **Does SwiGLU gating improve over GELU in this timeout-limited PDE regime?** (#4358 alphonse) — modern transformer best practice, first clean test of activation function; Arm A param-matched (hidden×2/3), Arm B full-hidden gating.
2. **Does Lookahead-wrapped Lion find better minima than plain Lion + EMA?** (#4362 thorfinn) — trust-region anchoring at step level (k=5 or k=10); preserves cosine schedule unlike SWA; hypothesis: Lion sign-noise + Lookahead slow weights compound.
3. **Does RMSNorm outperform LayerNorm in this bf16, small-data regime?** (#4365 tanjiro) — simpler norm without mean centering; bf16-safer; used in LLaMA, Mistral, T5v1.1; normalization form completely untested axis.
4. **Do even coarser slices (32, 40) continue the slice_num winning trend?** (#4306 askeladd) — slice=48 strong win vs slice=64/96; monotone coarser trend may continue.
5. **Does per-group LR rebalancing address the MLP/attn update magnitude imbalance?** (#4295 fern) — Arm A reins in MLP (lr_other×0.5), Arm B boosts attn (lr_attn×2.0); Lion sign-update may mask the imbalance.
6. **Does point subsampling augmentation improve OOD generalization?** (#4377 nezuko) — drop 20% (A) or 40% (B) of non-surface points per training batch; tests data-coverage hypothesis directly; surface points always preserved.
7. **Does Fourier feature encoding unlock high-frequency spatial signal for surface pressure?** (#4403 edward) — NeRF octaves K=8 vs RFF K=16 σ=10; removes spectral bias of raw coord inputs; may disproportionately help val_geom_camber_rc OOD split.
8. **Does DropPath (block-level dropout) succeed where element-wise dropout failed?** (#4405 frieren) — drops entire residual branches at p=0.10/0.20; stronger generalization pressure with less per-step noise; linear schedule 0→p_max over 5 layers.

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
- **SwiGLU gating** — IN PROGRESS as #4358 (alphonse); Arm A param-matched, Arm B full hidden; first clean implementation
- **Lookahead-Lion** — IN PROGRESS as #4362 (thorfinn); k=5 or k=10; trust-region anchoring
- **RMSNorm** — IN PROGRESS as #4365 (tanjiro); Arm A all norms, Arm B pre-block only
- **Huber loss** — CLOSED as #4327 (frieren); asinh already handles outliers; loss-reshaping slows convergence under timeout → #4405 DropPath
- **SGDR warm restarts** — CLOSED as #4253 (edward); restart spike, 3rd schedule-disruption failure → #4403 Fourier features
- **Slice_num coarser** — IN PROGRESS as #4306 (askeladd); slice=40/32
- **Fourier feature encoding** — IN PROGRESS as #4403 (edward); NeRF octaves K=8 vs RFF K=16 σ=10; first input-representation experiment
- **DropPath stochastic depth** — IN PROGRESS as #4405 (frieren); p_max=0.10/0.20; first block-level regularization attempt
- **Attention dropout** — CLOSED as #4278 (nezuko); both arms regress on bottleneck split; train/val gap GROWS with dropout (signal-removal not co-adaptation); 3rd regularization failure → #4377 point-subsample
- **Point subsampling augmentation** — IN PROGRESS as #4377 (nezuko); keep_rate=0.8 (A), 0.6 (B); first data-augmentation experiment in the program
- **Per-group LR** — IN PROGRESS as #4295 (fern)

**Not yet tried (candidates for next round):**
- Y-mirror geometric augmentation — full physics-aware mirror of geometry + targets; targets camber_rc bottleneck (more complex than point-subsample but more effective)
- Mixup of geometries — interpolate two training samples
- SAM (Sharpness-Aware Minimization) — if Lookahead fails, SAM is next optimizer meta-algorithm
- DropPath (stochastic depth) — different mechanism from FFN/attn dropout; drops entire blocks
- GeGLU (GELU gate variant of SwiGLU) — if SwiGLU fails
- Pre-norm vs post-norm positioning (current model norm ordering unchecked)
- Conditional normalization / FiLM on Re_inf, AoA — physical conditioning axis untouched
- Per-channel surf loss (separate surf_uxuy_weight from surf_p_weight) — split current uniform surf_weight=25
