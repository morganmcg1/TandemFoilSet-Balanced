# SENPAI Research State

- 2026-05-16 21:35Z — round 15 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=40.6869** (PR #3953 frieren LR×T_max re-calibration merged, −8.04% from 44.24)

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
| **#3953 (frieren, LR×T_max)** | **40.6869** | **−8.04%** | **lr: 1.7e-4→2.5e-4; T_max: 30→40; re-calibrated for 33-epoch compile horizon** |

**Current HEAD (11 mechanisms):** Lion lr=**2.5e-4** + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine **T_max=40** + pressure_weight=2.0 + torch.compile(mode=default, dynamic=True). val=40.69 at epoch 33 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 40 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default \
    --lr 2.5e-4
```
(In-tree defaults: lr=1.7e-4, T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999, compile_mode="" — must pass all six explicitly.)

**Cumulative improvement from initial baseline:** 135.02 → 40.69 = **−69.8%**

**Compute profile:** ~54-55s/epoch, 33 epochs in 30 min, 23.84 GB VRAM peak.

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #4078 | alphonse | Capacity scale-up on compile stack: n_hidden=192/256 | WIP | Notified of new baseline 40.69; was testing at T_max=30+lr=1.7e-4 (old 10-mech); training pods completed ~21:00Z |
| #4079 | edward | T_max calibration: T_max=33/40 at lr=1.7e-4 (pure T_max sweep) | WIP | KEY: isolates T_max contribution from frieren's joint win; training completed ~21:00Z |
| #4029 | askeladd | EMA decay fine sweep: 0.993 and 0.990 on compile stack | WIP | Notified of new baseline 40.69; training completed ~21:00Z |
| #4030 | nezuko | Velocity surface down-weighting: surf_ux/uy=0.5,0.7 with pw=2.0 | WIP | Notified of new baseline 40.69; pod restarted at 20:41Z |
| #4154 | fern | Per-group grad-clip looser: other_grad_norm=1.5, 2.0 on 11-mech stack | WIP — NEW | Sanity arm from #4016 showed 1.95% borderline win; testing inverse direction |
| #4061 | tanjiro | Channel-decoupled output heads: split velocity (Ux,Uy) from pressure (p) | WIP | Notified of new baseline 40.69; pod restarted at 20:41Z |
| #4159 | frieren | T_max fine-sweep at lr=2.5e-4: T_max=35 and 50 | WIP — NEW | Testing whether T_max=40 is the optimum for lr=2.5e-4 |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Notified of new baseline 40.69; pod restarted at 20:49Z |

## Key open questions (round 15 — new baseline 40.69)

1. **Is the LR gain from frieren isolated or joint with T_max?** (#4079 edward) — T_max=33/40 at lr=1.7e-4; pure T_max contribution to the 40.69 result. Results expected any time.
2. **Is T_max=40 optimal for lr=2.5e-4?** (#4159 frieren) — T_max=35/50 bracket at fixed lr=2.5e-4.
3. **Does capacity scale-up compose with the new 11-mech stack?** (#4078 alphonse) — n192/n256 with compile, but tested on OLD 10-mech (lr=1.7e-4, T_max=30); results still instructive.
4. **Is EMA decay still monotone past 0.995 on the compile stack?** (#4029 askeladd) — 0.993/0.990; tested on OLD 10-mech baseline.
5. **Can velocity surface down-weighting free gradient budget for pressure?** (#4030 nezuko) — surf_ux/uy=0.5,0.7.
6. **Do per-group clip (looser MLP budget) improve on the 11-mech stack?** (#4154 fern) — other_grad_norm=1.5/2.0.
7. **Do channel-decoupled output heads improve pressure specialization?** (#4061 tanjiro) — targets 11-mech compile baseline 40.69.
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — stale; needs 11-mech stack update.

## 11-mechanism stack: full pipeline

1. **Lion (sign-based update)**: optimizer; single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level; balance surface vs volume loss
3. **asinh**: loss-level; per-coordinate pressure z-score compression
4. **EMA(0.995)**: parameter level; exponential trajectory smoothing with faster decay
5. **grad-clip(max_norm=1.0)**: gradient vector level; L2 norm cap
6. **bf16 autocast**: compute level; forward+loss precision reduction (~23% faster/epoch base)
7. **cosine T_max=40**: schedule level; gentler annealing calibrated to 33-epoch horizon; ~10% LR floor at epoch 33
8. **pressure_weight=2.0**: loss level; up-weights pressure channel MAE 2×
9. **EMA decay=0.995**: EMA parameter; tighter half-life under steep annealing
10. **torch.compile(default, dynamic=True)**: kernel fusion; 47% faster/epoch (102s→54s), 18→33 epochs, −9 GB VRAM
11. **lr=2.5e-4**: LR re-calibration; higher lr_init matched to 33-epoch budget with T_max=40

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
| #3586 (nezuko higher LR 2.5e-4) | +2.74% regression (at T_max=80; now a winner at T_max=40) |
| #3656 (frieren surf-weight-fine sw=22/27) | +4.09% regression |
| #3528 (fern grad-clip rebased sw×max_norm grid) | +0.57% regression; max_norm=1.0 optimal |
| #3725 (fern per-group grad-clip attn vs MLP) | no_improvement: val +1.1–3.9%; diagnostic revealed MLP/output ~5× noisier; #4154 tests loosening |
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
| #4016 (fern tighter MLP clip) | no_improvement: arms A/B regressed; tighter other_grad_norm discards MLP signal; sanity 1.95% borderline → follow-up #4154 |

## Potential next research directions

- **T_max fine-sweep at lr=2.5e-4** — IN PROGRESS as #4159 (frieren); T_max=35/50 bracket
- **LR fine-sweep at T_max=40** — lr ∈ {2.2e-4, 2.7e-4, 3.0e-4}; Arm A vs Arm B gap (0.32 pts) suggests optimum may sit near 2.5e-4 but worth confirming
- **Capacity scale-up with new 11-mech stack** — n_hidden=192 with lr=2.5e-4+T_max=40; separate from #4078 which used old stack
- **EMA decay re-sweep on 11-mech stack** — IN PROGRESS as #4029 (askeladd); 0.993/0.990 tested on OLD stack
- **Velocity surface down-weighting** — IN PROGRESS as #4030 (nezuko)
- **Per-group clip loosening** — IN PROGRESS as #4154 (fern); other_grad_norm=1.5/2.0
- **Channel-decoupled heads** — IN PROGRESS as #4061 (tanjiro)
- **SwiGLU gating** — IN PROGRESS as #3734 (thorfinn); stale
- **Weight decay sweep**: wd ∈ {1e-4, 2e-4, 5e-4} — fixed at 3e-4 since PR #3293; untested under 11-mech stack
- **Lion β1 < 0.90**: β1=0.85/0.88 — only tested above 0.90; below untested
- **WeightedRandomSampler dynamic reweighting**: inverse-error resampling after epoch 1 — static sampler currently
- **Deeper capacity**: n_layers=6/7 on compile stack — separate from width scaling
- **Slice count sweep**: slice_num ∈ {32, 48, 96} — fixed at 64; PhysicsAttention slice routing granularity
- **Weight initialization**: trunc_normal_(std=0.02) is the default; alternative: scaled initialization (e.g., 1/sqrt(n_layers))
