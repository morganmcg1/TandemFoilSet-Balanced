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
| #4167 | alphonse | Capacity n192 + T_max=22 (calibrated to n192 epoch budget) | WIP — NEW | #4078 confirmed per-epoch n192 benefit; T_max=30 miscalibrated for ~22 epoch n192 budget |
| #4159 | frieren | T_max fine-sweep: T_max=35 and 50 at lr=1.7e-4 (updated from 2.5e-4) | WIP — NEW | Edward proved 1.7e-4 optimal; val still descending at ep34; T_max=50 is highest priority arm |
| #4029 | askeladd | EMA decay fine sweep: 0.993 and 0.990 on compile stack | WIP | Notified of new baseline 39.83 |
| #4030 | nezuko | Velocity surface down-weighting: surf_ux/uy=0.5,0.7 with pw=2.0 | WIP | Notified of new baseline 39.83 |
| #4154 | fern | Per-group grad-clip looser: other_grad_norm=1.5, 2.0 | WIP | Notified of new baseline 39.83; uses lr=1.7e-4 (correct) |
| #4061 | tanjiro | Channel-decoupled output heads: split velocity from pressure | WIP | Notified of new baseline 39.83 |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Notified of new baseline 39.83 |

## Key open questions (round 15 — new baseline 39.83)

1. **Is T_max=40 the optimum, or should it go higher?** (#4159 frieren) — val still descending at epoch 34 with lr=1.25e-5 (~7% of init); T_max=35/50 bracket at lr=1.7e-4. T_max=50 is the highest-priority arm.
2. **Does capacity n192 with calibrated T_max=22 compete?** (#4167 alphonse) — per-epoch benefit confirmed in #4078; the fix is T_max calibration.
3. **Is EMA decay 0.993/0.990 better under T_max=40?** (#4029 askeladd) — EMA optimal decay may shift under gentler annealing.
4. **Can velocity surface down-weighting free gradient budget?** (#4030 nezuko)
5. **Do per-group clip (looser MLP budget) improve on the 12-mech stack?** (#4154 fern)
6. **Do channel-decoupled output heads improve pressure specialization?** (#4061 tanjiro)
7. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn)

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

## Potential next research directions

- **T_max fine-sweep** — IN PROGRESS as #4159 (frieren); T_max=35/50 at lr=1.7e-4. T_max=50 is highest priority.
- **Capacity n192 + calibrated T_max** — IN PROGRESS as #4167 (alphonse); T_max=22 for n192's 22-epoch budget
- **LR fine-sweep at T_max=40** — lr ∈ {1.5e-4, 2.0e-4} to confirm 1.7e-4 is the precise optimum
- **EMA decay re-sweep on 12-mech stack** — IN PROGRESS as #4029 (askeladd); 0.993/0.990
- **Velocity surface down-weighting** — IN PROGRESS as #4030 (nezuko)
- **Per-group clip loosening** — IN PROGRESS as #4154 (fern); other_grad_norm=1.5/2.0
- **Channel-decoupled heads** — IN PROGRESS as #4061 (tanjiro)
- **SwiGLU gating** — IN PROGRESS as #3734 (thorfinn); stale
- **Weight decay sweep**: wd ∈ {1e-4, 2e-4, 5e-4} — fixed at 3e-4 since PR #3293; untested under 12-mech stack
- **Deeper capacity**: n_layers=6/7 on compile stack — separate from width scaling
- **Slice count sweep**: slice_num ∈ {32, 48, 96} — fixed at 64; PhysicsAttention slice routing granularity
- **Longer training** — val still descending at ep34; T_max=50/60 may extract more
