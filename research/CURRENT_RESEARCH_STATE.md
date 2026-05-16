# SENPAI Research State

- 2026-05-16 17:30Z — round 13 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=51.4403** (PR #3989 EMA decay=0.995 merged, −4.25% from 53.72)

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
| **#3989 (askeladd, EMA decay=0.995)** | **51.4403** | **−4.25%** | **EMA shadow decay: 0.999→0.995; faster tracking under T_max=30+pw=2.0** |

**Current HEAD (9 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + pressure_weight=2.0. val=51.44 at epoch 18 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0 \
    --ema_decay 0.995
```
(In-tree defaults: T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999 — must pass all four explicitly.)

**Cumulative improvement from initial baseline:** 135.02 → 51.44 = **−61.9%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #4029 | askeladd | EMA decay fine sweep: 0.993 and 0.990 on 9-mech stack | WIP — NEW | #3989 confirmed monotone 0.999→0.997→0.995; gap at noise floor; curve may continue |
| #4030 | nezuko | Velocity surface down-weighting: surf_ux/uy=0.5,0.7 with pw=2.0 | WIP — NEW | #3984 diagnostic: pw>2.0 breaks velocity representation; down-weighting vel surface frees gradient budget differently |
| #4031 | edward | Lion β2 sweep: 0.95 and 0.98 vs default 0.99 | WIP — NEW | Untested axis; under T_max=30+pw=2.0+EMA=0.995 the optimal gradient memory window may have shifted |
| #3970 | alphonse | torch.compile throughput: default and reduce-overhead modes | WIP — STALE | #3884+#3750 confirmed throughput-bound; compile buys per-epoch time → more steps; notified of 51.44 target |
| #3953 | frieren | LR × T_max coupling: lr=2.1e-4 and 2.5e-4 under T_max=30 | WIP | Effective avg-LR shifted 60%; optimal lr_init may have moved; notified of 53.72 target |
| #4016 | fern | Tighter MLP/output grad-clip (other=0.5, 0.3) on 8-mech stack | WIP — NEW | #3725 diagnostic: MLP~5× noisier than attn; tighten dominant group; pw=2.0 elevates output-head grad further |
| #4061 | tanjiro | Channel-decoupled output heads: split velocity (Ux,Uy) from pressure (p) | WIP — NEW | Pressure has different statistics; dedicated head can specialize; orthogonal to all current mechanisms |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Long overdue; notified of 51.44 target |

## Key open questions (round 13 — new baseline 51.44)

1. **Is EMA decay still monotone past 0.995?** (#4029 askeladd) — #3989 showed 0.999→0.997→0.995 monotone but at noise-floor gaps; testing 0.993/0.990 to find the floor
2. **Can velocity surface down-weighting free gradient budget for pressure?** (#4030 nezuko) — #3984 showed pw>2.0 breaks shared representation; down-weighting vel surface (ux=0.5, uy=0.5) is a gentler reallocation that keeps vol loss carrying velocity signal
3. **Is Lion β2=0.99 optimal under the 9-mech stack?** (#4031 edward) — β2 controls momentum buffer half-life (~69 steps); under T_max=30+pw=2.0+EMA=0.995, shorter memory (β2=0.95, ~14 steps) may track the shifting loss landscape better
4. **Does torch.compile buy meaningful per-epoch speedup?** (#3970 alphonse) — throughput-bound; 15-25% faster epochs → 22-24 epochs in same wall-clock
5. **Does lr_init shift under T_max=30+pw=2.0+ema=0.995?** (#3953 frieren) — lr=2.1/2.5e-4; prior lr=2.5e-4 failure was on T_max=80 stack
6. **Is `other` (MLP/output) gradient over-clipped by the existing single-clip(1.0)?** (#4016 fern) — other_gn mean ~18 >> attn_gn mean ~3.5; tightening other below 1.0 tests if dominant MLP signal benefits from stronger clipping
7. **Do channel-decoupled output heads improve specialization?** (#4061 tanjiro) — pressure has heavy-tailed/2× weighted statistics distinct from velocity; dedicated head can specialize, freeing the shared backbone from a 3-channel multi-task constraint
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — input-dependent gating

## 9-mechanism stack: gradient pipeline analysis

Nine mechanisms targeting DIFFERENT points:
1. **Lion (sign-based update)**: optimizer level — single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level — balance surface vs volume loss
3. **asinh**: loss-level — per-coordinate pressure z-score compression
4. **EMA(0.995)**: parameter level — exponential trajectory smoothing; faster decay under T_max=30
5. **grad-clip**: gradient vector level — L2 norm cap at 1.0
6. **bf16 autocast**: compute level — forward+loss precision reduction; 23% faster/epoch
7. **cosine T_max=30**: schedule level — moderate late-epoch annealing (40% of initial LR at epoch 18)
8. **pressure_weight=2.0**: loss level — up-weights pressure channel MAE 2× in training loss; constructively stacks with asinh
9. **EMA decay=0.995** ← **NEW** — tighter half-life (~138 steps vs ~693 at 0.999); better tracks recent-weights regime under steep annealing

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression |
| #3328 (askeladd surf_weight=50) | +25% regression; instability above sw=30 |
| #3329 (fern AdamW β2=0.95) | +21% regression |
| #3102 (edward OneCycleLR) | +20% regression |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores light-tailed |
| #3099 (alphonse capacity 192h/6L/6H) | +60.5% regression; wall-clock budget dominates |
| #3106 (frieren Slice128/head8/mlp3) | +98.6% regression; same wall-clock penalty |
| #3354 (nezuko Lion+cosine T_max=12) | +15.96% regression; curve budget-limited not LR-limited |
| #3586 (nezuko higher LR 2.5e-4) | +2.74% regression; lr=1.7e-4 near-optimal |
| #3656 (frieren surf-weight-fine sw=22/27) | +4.09% regression; curve flat in [20,25] |
| #3528 (fern grad-clip rebased sw×max_norm grid) | +0.57% regression; max_norm=1.0 optimal |
| #3725 (fern per-group grad-clip attn vs MLP) | no_improvement: val +1.1–3.9% vs 56.00; diagnostic revealed MLP/output ~5× noisier than attention; #4016 tests tighter MLP clip |
| #3887 (edward T_max bracket 25/40) | no_improvement: T_max=25 +5.07%, T_max=40 +2.86%; T_max=30 is narrow peak at 40% final LR; closed |
| #3984 (nezuko pw=3.0/4.0) | no_improvement: pw=3.0 +2.38%, pw=4.0 +5.73%; inverted-U curve peaks at pw=2.0; velocity degradation breaks shared representation above 2.0; closed |
| #3442 (tanjiro signed-log1p) | Closed stale; reassigned as #3731 |
| #3383 (edward warmup-cosine) | Closed stale; reassigned as #3733 |
| #3275 (thorfinn SwiGLU) | Closed stale; reassigned as #3734 |
| #3470 (askeladd EMA-decay) | Closed prematurely; reassigned as #3776 |
| #3733 (edward warmup-cosine v2) | +4.2% regression (val=61.33 vs 58.87) |
| #3750 (alphonse capacity-bf16) | no_improvement: n144 val=59.85 (+1.66%), n_layers=6 val=64.57 (+9.7%); throughput-bound |
| #3884 (alphonse batch-size-bf16) | no_improvement: batch=6 val=69.68 (+24.4%); steps/epoch −33%; throughput-bound at batch=4 |
| #3949 (askeladd lion-beta1) | no_improvement: β1=0.95 val=61.49 (+9.8%); β1=0.90 optimal |
| #3731 (tanjiro signed-log1p v2) | no_improvement: val=58.09 (+12.93% vs 51.44); signed-log1p over-compresses moderate |z| range where pressure gradient signal lives; asinh confirmed locally optimal; pressure-transform axis exhausted |
| #3776 (askeladd EMA-decay-bf16-v2) | Closed without results; rate-limit cascade failure |
| #3726 (frieren Lion-wd-sweep) | Closed without results; rate-limit cascade failure |

## Potential next research directions

- ~~EMA decay on 8-mech stack~~ — MERGED #3989 (ema=0.995, −4.25%)
- ~~EMA decay fine sweep~~ — IN PROGRESS as #4029 (askeladd); 0.993/0.990
- ~~Pressure weight fine sweep~~ — CLOSED #3984 (pw=3.0/4.0 both regress; pw=2.0 is unique peak)
- ~~T_max refinement~~ — CLOSED #3887 (T_max=25/40 both regress; T_max=30 is narrow peak)
- **Velocity surface down-weighting** — IN PROGRESS as #4030 (nezuko); surf_ux/uy=0.5,0.7
- **Lion β2 sweep** — IN PROGRESS as #4031 (edward); β2=0.95/0.98
- **torch.compile throughput** — IN PROGRESS as #3970 (alphonse); stale
- **LR × T_max coupling** — IN PROGRESS as #3953 (frieren); lr=2.1/2.5e-4
- **Tighter MLP clip** — IN PROGRESS as #4016 (fern); other=0.5/0.3
- ~~Signed log1p~~ — CLOSED #3731 (val +12.93%; signed-log1p over-compresses moderate |z|; asinh locally optimal; pressure-transform axis exhausted)
- **SwiGLU gating** — IN PROGRESS as #3734 (thorfinn); stale
- **Channel-decoupled output heads** — IN PROGRESS as #4061 (tanjiro); split velocity (Ux,Uy) and pressure (p) into separate output MLPs
- **WeightedRandomSampler dynamic reweighting**: inverse-error resampling after epoch 1 — static domain-balanced sampler already present; dynamic per-sample reweighting is the next step
- **Lion β1 < 0.90**: β1=0.85/0.88 — more reactive momentum; untested below 0.90 (was only tested above)
- **Weight decay sweep**: wd ∈ {1e-4, 2e-4, 5e-4} — fixed at 3e-4 since PR #3293; untested under new 9-mech stack
