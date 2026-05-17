# SENPAI Research State

- 2026-05-16 22:00Z — round 15 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=39.8345** (PR #4079 edward T_max=40 at lr=1.7e-4 merged, −9.97% from 44.24)

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

**Current HEAD (12 mechanisms):** Lion lr=**1.7e-4** + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine **T_max=40** + pressure_weight=2.0 + torch.compile(mode=default, dynamic=True). val=39.83 at epoch 34 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 40 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default
```
(In-tree defaults: lr=1.7e-4 is CORRECT — do NOT pass --lr 2.5e-4; T_max=80→must pass 40; surf_weight=30→must pass 25; pressure_weight=1.0→must pass 2.0; ema_decay=0.999→must pass 0.995; compile_mode=none→must pass default.)

**Cumulative improvement from initial baseline:** 135.02 → 39.83 = **−70.5%**

**Compute profile:** ~54s/epoch, 33-34 epochs in 30 min, 23.84 GB VRAM peak.

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #4235 | alphonse | MLP-ratio sweep: 3 and 4 (vs current 2) on 12-mech stack | WIP | #4188 closed; plateau protocol → architectural angle; MLPs are the largest param group |
| #4287 | frieren | Batch size sweep: 8 (Arm A), 12 (Arm B) vs current 4 on 12-mech stack | WIP — NEW | #4236 closed; warmup refuted; 23.84/80 GB VRAM headroom → test batch dynamics (9 vs 4-5 vs 3 steps/epoch) |
| #4237 | fern | Depth sweep: n_layers=6, 7 (vs current 5) on 12-mech stack | WIP | #4154 closed; depth untouched since round 1; width was throughput-bound (#4167) |
| #4253 | edward | SGDR warm restarts: T_0=17 (Arm A), T_0=12 (Arm B) on 12-mech stack | WIP — NEW | #4181 closed (LR axis locked at 1.7e-4); all arms show val descending at timeout → test mid-training LR restart |
| #4243 | askeladd | slice_num sweep: 48 (Arm A), 96 (Arm B) vs current 64 on 12-mech stack | WIP — NEW | #4029 closed (EMA decay 0.993/0.990 within noise); slice_num cleanest untested architectural axis |
| #4278 | nezuko | Attention dropout: p=0.05 (Arm A), p=0.10 (Arm B) in PhysicsAttention | WIP — NEW | #4030 closed; vel-surf-weight null at lr=1.7e-4; val_geom_camber_rc=53.15 dominant gap → test attention regularization |
| #4273 | tanjiro | n_head sweep: 2 (Arm A), 8 (Arm B) vs current 4 on 12-mech stack | WIP — NEW | #4061 closed; decoupled-heads null (shared head not bottleneck); pure n_head is uncharted (#3106 compounded 3 changes) |
| #4230 | thorfinn | Weight decay sweep: wd=1e-4, 5e-4 bracketing 3e-4 | WIP | #3734 SwiGLU closed (17h stale, blocked); wd untouched since #3293 (Lion change) |

## Key open questions (round 15 — new baseline 39.83 — PLATEAU PROTOCOL ACTIVE)

**Plateau signal:** 5+ recent no_improvement experiments in a row (#4188, #4154, #4159, #4167, #4078, plus partial #4030). Per CLAUDE.md plateau protocol, escalating from pure hyperparameter sweeps to architectural changes.

1. **Does a mid-training LR reset (SGDR) escape the timeout-bound attractor?** (#4253 edward) — val descends in every arm at ep34; T_0=17 gives a second high-LR phase; T_0=12 gives nearly a full second cycle.
2. **Does velocity surface down-weighting (0.7/0.8) at lr=1.7e-4 beat baseline?** (#4030 nezuko) — Arm B at lr=2.5e-4 showed test=33.72 (promising); rerun at correct LR needed.
3. **Does weight decay re-tuning unlock more headroom?** (#4230 thorfinn) — wd untouched at 3e-4 since Lion switch in #3293.
4. **Does MLP-ratio expansion (3, 4) beat baseline?** (#4235 alphonse) — capacity along widest parameter group; throughput penalty ~10-20% per arm.
5. **Does larger batch_size (8, 12) unlock better convergence at fixed LR?** (#4287 frieren) — 30% VRAM utilization (23.84/80 GB headroom); batch=4 gives only ~9 steps/epoch; larger batch could improve gradient quality and potentially throughput.
6. **Does depth scaling (n_layers=6, 7) work where width scaling didn't?** (#4237 fern) — depth untouched since round 1; per-block slice attention scales linearly with layers.
7. **Does attention dropout (p=0.05, 0.10) reduce OOD over-fitting on geom_camber_rc?** (#4278 nezuko) — dominant val error is rc-camber split (53.15); model has zero attention regularization; small-data regime (36 training geometries).
8. **Does slice_num=48 or 96 improve on the current 64?** (#4243 askeladd) — cleanest untested architectural axis; prior #3106 was a three-way compound; pure slice sweep is new.
9. **Does n_head=2 or 8 outperform current n_head=4?** (#4273 tanjiro) — head_dim directly controls attention subspace rank per block; pure n_head sweep is new ground (prior #3106 compounded 3 changes).

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
10. **torch.compile(default, dynamic=True)**: kernel fusion; 47% faster/epoch, 33-34 epochs in 30 min

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

## Potential next research directions

- **T_max fine-sweep** — CLOSED as #4159 (frieren); T_max=35 regressed (+7.2%), T_max=50 null (+0.92%); T_max=40 optimal
- **torch.compile mode sweep** — CLOSED as #4188 (alphonse); both modes regress; default optimal
- **LR fine-sweep at T_max=40** — CLOSED as #4181 (edward); lr=1.5e-4 null (+1.4%), lr=2.0e-4 failure (+5.1%); lr=1.7e-4 locked
- **EMA decay re-sweep** — CLOSED as #4029 (askeladd); 0.993 within noise + test regression; decay=0.995 locked
- **MLP-ratio sweep** — IN PROGRESS as #4235 (alphonse); mlp_ratio=3, 4 (vs current 2)
- **Depth sweep** — IN PROGRESS as #4237 (fern); n_layers=6, 7 (vs current 5)
- **Warmup-cosine** — CLOSED as #4236 (frieren); Arm A (warmup=2) val=41.28 (+3.6%), stop triggered; schedule-compression effect dominates; direction closed (2nd failure)
- **Batch size sweep** — IN PROGRESS as #4287 (frieren); batch_size=8 and 12 vs current 4; VRAM headroom at 30% suggests potential throughput gain
- **SGDR warm restarts** — IN PROGRESS as #4253 (edward); T_0=17 and T_0=12 — mid-training LR reset to escape timeout-bound attractor
- **Slice_num sweep** — IN PROGRESS as #4243 (askeladd); slice_num=48, 96 (vs current 64) — cleanest untested architectural axis
- **Velocity surface down-weighting** — CLOSED as #4030 (nezuko); null at lr=1.7e-4 (Arm C val within noise but test +1.69%); axis closed
- **Channel-decoupled heads** — CLOSED as #4061 (tanjiro); null (12-mech already neutralized shared-head bottleneck); axis closed
- **Attention dropout** — IN PROGRESS as #4278 (nezuko); p=0.05, 0.10 in PhysicsAttention; targets OOD gap at val_geom_camber_rc
- **n_head sweep** — IN PROGRESS as #4273 (tanjiro); n_head=2 (head_dim=64) and n_head=8 (head_dim=16) vs current 4
- **Weight decay sweep** — IN PROGRESS as #4230 (thorfinn); wd ∈ {1e-4, 5e-4} bracketing 3e-4
- **SwiGLU gating** — CLOSED as #3734 (thorfinn); blocked on implementation, 3 attempts failed
