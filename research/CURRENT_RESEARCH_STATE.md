# SENPAI Research State

- 2026-05-16 18:00Z — round 14 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=44.2439** (PR #3970 torch.compile merged, −14.0% from 51.44)

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
| **#3970 (alphonse, torch.compile)** | **44.2439** | **−14.0%** | **torch.compile(mode=default, dynamic=True); 102s→54s/epoch; 18→33 epochs** |

**Current HEAD (10 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + pressure_weight=2.0 + **torch.compile(mode=default, dynamic=True)**. val=44.24 at epoch 33 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default
```
(In-tree defaults: T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999, compile_mode="" — must pass all five explicitly.)

**Cumulative improvement from initial baseline:** 135.02 → 44.24 = **−67.2%**

**Compute profile (after compile):** ~54s/epoch, 33 epochs in 30 min, 23.84 GB VRAM peak.

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #4078 | alphonse | Capacity scale-up on compile stack: n_hidden=192/256 — WIP NEW | WIP — NEW | #3750 showed capacity throughput-blocked at 18 epochs; compile gives ~22 epochs for n192; 9 GB freed VRAM |
| #4079 | edward | T_max calibration for 33-epoch compile horizon: T_max=33/40 | WIP — NEW | T_max=30 was tuned for 18 epochs; on 33-epoch budget, LR reaches 0 at ep30 and oscillates for 3 wasted epochs |
| #4029 | askeladd | EMA decay fine sweep: 0.993 and 0.990 on compile stack | WIP | #3989 monotone 0.999→0.995; gap at noise floor; now with compile (33 epochs) — notified |
| #4030 | nezuko | Velocity surface down-weighting: surf_ux/uy=0.5,0.7 with pw=2.0 | WIP | Compile stack (33 epochs) — notified |
| #4016 | fern | Tighter MLP/output grad-clip (other=0.5, 0.3) on full 10-mech stack | WIP (draft) | Updated to 10-mech stack; must rebase from advisor branch |
| #4061 | tanjiro | Channel-decoupled output heads: split velocity (Ux,Uy) from pressure (p) | WIP — NEW | Now targeting compile baseline 44.24; notified |
| #3953 | frieren | LR × T_max coupling: lr=2.1e-4 and 2.5e-4 under T_max=30 | WIP | Notified of compile stack; T_max/lr coupling may shift with 33-epoch budget |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Notified of compile stack; must add --compile_mode default |

## Key open questions (round 14 — new baseline 44.24)

1. **Does capacity scale-up compose with compile?** (#4078 alphonse) — 9 GB freed VRAM + 54s/epoch opens capacity; n192 failed at 18 epochs but should get ~22 epochs with compile
2. **Should T_max be recalibrated from 30 to 33+ on the compile stack?** (#4079 edward) — T_max=30 causes LR→0 at ep30; epochs 31-33 wasted; T_max=33/40 may fully utilize the 33-epoch budget
3. **Is EMA decay still monotone past 0.995 on the compile stack?** (#4029 askeladd) — 33-epoch context changes; fine sweep with compile now
4. **Can velocity surface down-weighting free gradient budget for pressure?** (#4030 nezuko) — surf_ux/uy=0.5 on 33-epoch compile stack
5. **Is MLP gradient over-clipped (single-clip=1.0)?** (#4016 fern) — other_gn ~18 >> attn_gn ~3.5; tighter MLP clip on compile stack
6. **Do channel-decoupled output heads improve pressure specialization?** (#4061 tanjiro) — now vs compile baseline 44.24
7. **Does lr_init shift under compile+T_max=30+pw=2.0?** (#3953 frieren) — T_max/lr coupling on 33-epoch budget
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — stale; needs compile stack update

## 10-mechanism stack: full pipeline

1. **Lion (sign-based update)**: optimizer; single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level; balance surface vs volume loss
3. **asinh**: loss-level; per-coordinate pressure z-score compression
4. **EMA(0.995)**: parameter level; exponential trajectory smoothing with faster decay under T_max=30
5. **grad-clip(max_norm=1.0)**: gradient vector level; L2 norm cap
6. **bf16 autocast**: compute level; forward+loss precision reduction (~23% faster/epoch base)
7. **cosine T_max=30**: schedule level; moderate late-epoch annealing
8. **pressure_weight=2.0**: loss level; up-weights pressure channel MAE 2×
9. **EMA decay=0.995**: EMA parameter; tighter half-life under steep annealing
10. **torch.compile(default, dynamic=True)**: kernel fusion; 47% faster/epoch (102s→54s), 18→33 epochs, −9 GB VRAM

## Closed / falsified experiments

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression |
| #3328 (askeladd surf_weight=50) | +25% regression |
| #3329 (fern AdamW β2=0.95) | +21% regression |
| #3102 (edward OneCycleLR) | +20% regression |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores light-tailed |
| #3099 (alphonse capacity 192h/6L/6H) | +60.5% regression; throughput-bound at 18 epochs — now retested with compile (#4078) |
| #3106 (frieren Slice128/head8/mlp3) | +98.6% regression |
| #3354 (nezuko Lion+cosine T_max=12) | +15.96% regression |
| #3586 (nezuko higher LR 2.5e-4) | +2.74% regression |
| #3656 (frieren surf-weight-fine sw=22/27) | +4.09% regression |
| #3528 (fern grad-clip rebased sw×max_norm grid) | +0.57% regression; max_norm=1.0 optimal |
| #3725 (fern per-group grad-clip attn vs MLP) | no_improvement: val +1.1–3.9%; diagnostic revealed MLP/output ~5× noisier; #4016 tests tighter MLP clip |
| #3887 (edward T_max bracket 25/40) | no_improvement on 18-epoch budget; T_max=30 narrow peak — retested for 33-epoch budget (#4079) |
| #3984 (nezuko pw=3.0/4.0) | no_improvement: inverted-U peaks at pw=2.0; velocity degradation above 2.0 |
| #3949 (askeladd lion-beta1) | no_improvement: β1=0.95 +9.8%; β1=0.90 optimal; Lion-β1 axis closed |
| #4031 (edward lion-beta2) | no_improvement: β2=0.95 +27% failure; β2=0.99 optimal; Lion-β2 axis closed |
| #3731 (tanjiro signed-log1p v2) | no_improvement: +12.93%; signed-log1p over-compresses moderate |z|; asinh locally optimal |
| #3442/#3383/#3275 (stale closed) | Reassigned as v2 |
| #3776/#3726 (rate-limit failures) | Closed without results |
| #3733 (edward warmup-cosine v2) | +4.2% regression |
| #3750 (alphonse capacity-bf16) | no_improvement on old stack; throughput-bound — retested with compile (#4078) |
| #3884 (alphonse batch-size-bf16) | no_improvement: batch=6 +24.4%; steps/epoch −33% |

## Potential next research directions

- **T_max fine-tuning for compile** — IN PROGRESS as #4079 (edward); T_max=33/40 for 33-epoch budget
- **Capacity scale-up with compile** — IN PROGRESS as #4078 (alphonse); n192/n256 on 10-mech stack
- **EMA fine sweep on compile stack** — IN PROGRESS as #4029 (askeladd); 0.993/0.990
- **Velocity surface down-weighting** — IN PROGRESS as #4030 (nezuko)
- **Tighter MLP clip on compile stack** — IN PROGRESS as #4016 (fern)
- **Channel-decoupled heads** — IN PROGRESS as #4061 (tanjiro)
- **SwiGLU gating** — IN PROGRESS as #3734 (thorfinn); stale
- **LR × T_max coupling** — IN PROGRESS as #3953 (frieren)
- **Weight decay sweep**: wd ∈ {1e-4, 2e-4, 5e-4} — fixed at 3e-4 since PR #3293; untested under 10-mech compile stack
- **Lion β1 < 0.90**: β1=0.85/0.88 — only tested above 0.90 (β1=0.95 failed); below untested
- **WeightedRandomSampler dynamic reweighting**: inverse-error resampling after epoch 1 — static sampler currently
- **Deeper capacity**: n_layers=6/7 on compile stack — separate from width scaling in #4078
- **Slice count sweep**: slice_num ∈ {32, 48, 96} — fixed at 64; PhysicsAttention slice routing granularity
