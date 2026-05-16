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
| #4181 | edward | LR fine-sweep at T_max=40: lr=1.5e-4 (Arm A), 2.0e-4 (Arm B) | WIP — NEW | #4079 only has two LR data points (1.7e-4 vs 2.5e-4); bracket needed to confirm optimum |
| #4188 | alphonse | torch.compile mode sweep: reduce-overhead and max-autotune at n128 | WIP — NEW | #4167 closed: n192 throughput-limited; compile mode may cut epoch time and reopen capacity direction |
| #4159 | frieren | T_max fine-sweep: T_max=35 and 50 at lr=1.7e-4 (updated from 2.5e-4) | WIP | Edward proved 1.7e-4 optimal; val still descending at ep34; T_max=50 is highest priority arm |
| #4029 | askeladd | EMA decay fine sweep: 0.993 and 0.990 on compile stack | WIP | Notified of new baseline 39.83 |
| #4030 | nezuko | Velocity surface down-weighting (re-test at lr=1.7e-4): Arm C (ux=uy=0.7), Arm D (ux=uy=0.8) | WIP — SENT BACK | Initial arms ran at lr=2.5e-4: Arm B (0.7) hit val=40.19, test=33.72 (test beats current baseline 33.89); need lr=1.7e-4 re-test for clean comparison |
| #4154 | fern | Per-group grad-clip looser: other_grad_norm=1.5, 2.0 | WIP | Notified of new baseline 39.83; uses lr=1.7e-4 (correct) |
| #4061 | tanjiro | Channel-decoupled output heads: split velocity from pressure | WIP — STALE (6h, 0% GPU) | Nudged for status; minimal-viable implementation hint posted |
| #4230 | thorfinn | Weight decay sweep: wd=1e-4, 5e-4 bracketing 3e-4 | WIP — NEW | #3734 SwiGLU closed (17h stale, blocked); wd untouched since #3293 (Lion change) |

## Key open questions (round 15 — new baseline 39.83)

1. **Is T_max=40 the optimum, or should it go higher?** (#4159 frieren) — val still descending at epoch 34 with lr=1.25e-5 (~7% of init); T_max=35/50 bracket at lr=1.7e-4. T_max=50 is the highest-priority arm.
2. **Is lr=1.7e-4 the precise optimum at T_max=40?** (#4181 edward) — only two data points (1.7e-4 vs 2.5e-4); bracket with 1.5e-4 and 2.0e-4 to characterize the LR landscape under T_max=40.
3. **Can compile_mode=reduce-overhead or max-autotune cut per-epoch time?** (#4188 alphonse) — n192 is throughput-limited at 79s/epoch; if we can cut n128's 54s/epoch, we gain epochs at the proven optimum (val still descending at ep34) and may reopen the capacity direction.
4. **Is EMA decay 0.993/0.990 better under T_max=40?** (#4029 askeladd) — EMA optimal decay may shift under gentler annealing.
5. **Can velocity surface down-weighting free gradient budget?** (#4030 nezuko) — Arm B (ux=uy=0.7) at lr=2.5e-4 hit test=33.72 (beats baseline test 33.89); val=40.19 fails at lr=2.5e-4. Re-running Arms C/D (0.7 and 0.8) at lr=1.7e-4 for clean comparison.
6. **Do per-group clip (looser MLP budget) improve on the 12-mech stack?** (#4154 fern)
7. **Do channel-decoupled output heads improve pressure specialization?** (#4061 tanjiro)
8. **Does weight decay re-tuning unlock more headroom?** (#4230 thorfinn) — wd untouched at 3e-4 since Lion switch in #3293; bracket with 1e-4 and 5e-4.

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

## Potential next research directions

- **T_max fine-sweep** — IN PROGRESS as #4159 (frieren); T_max=35/50 at lr=1.7e-4. T_max=50 is highest priority.
- **torch.compile mode sweep** — IN PROGRESS as #4188 (alphonse); reduce-overhead and max-autotune at n128 baseline
- **LR fine-sweep at T_max=40** — IN PROGRESS as #4181 (edward); lr ∈ {1.5e-4, 2.0e-4} bracketing 1.7e-4
- **EMA decay re-sweep on 12-mech stack** — IN PROGRESS as #4029 (askeladd); 0.993/0.990
- **Velocity surface down-weighting** — IN PROGRESS as #4030 (nezuko)
- **Per-group clip loosening** — IN PROGRESS as #4154 (fern); other_grad_norm=1.5/2.0
- **Channel-decoupled heads** — IN PROGRESS as #4061 (tanjiro)
- **SwiGLU gating** — CLOSED as #3734 (thorfinn 17h stale, blocked on implementation)
- **Weight decay sweep**: IN PROGRESS as #4230 (thorfinn); wd ∈ {1e-4, 5e-4} bracketing 3e-4 on 12-mech stack
- **Deeper capacity**: n_layers=6/7 on compile stack — separate from width scaling
- **Slice count sweep**: slice_num ∈ {32, 48, 96} — fixed at 64; PhysicsAttention slice routing granularity
- **Longer training** — val still descending at ep34; T_max=50/60 may extract more
