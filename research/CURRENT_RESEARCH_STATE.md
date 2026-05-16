# SENPAI Research State

- 2026-05-16 15:45Z — round 12 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=53.7235** (PR #3674 pressure_weight=2.0 merged, −4.07% from 56.00)

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
| **#3674 (nezuko, pressure_weight=2.0)** | **53.7235** | **−4.07%** | **pressure_weight: 1.0→2.0; up-weights pressure channel in training loss** |

**Current HEAD (8 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.999) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + pressure_weight=2.0. val=53.72 at epoch 18 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0
```
(In-tree defaults: T_max=80, surf_weight=30, pressure_weight=1.0 — must pass all three explicitly.)

**Cumulative improvement from initial baseline:** 135.02 → 53.72 = **−60.2%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3989 | askeladd | EMA decay on 8-mech stack: 0.997 and 0.995 vs 0.999 | WIP — NEW | Overdue test; T_max=30+pw=2.0 changed convergence horizon; #3470/#3776 closed rate-limited |
| #3984 | nezuko | Pressure weight sweep: pw=3.0 and 4.0 above the pw=2.0 winner | WIP — NEW | pw monotone in {0.5, 1.0, 2.0}; optimum may not be at 2.0; mapping the curve upward |
| #3970 | alphonse | torch.compile throughput: default and reduce-overhead modes | WIP | #3884+#3750 confirmed throughput-bound; compile buys per-epoch time → more steps |
| #3953 | frieren | LR × T_max coupling: lr=2.1e-4 and 2.5e-4 under T_max=30 | WIP | Effective avg-LR shifted 60%; optimal lr_init may have moved; notified of 53.72 target |
| #3887 | edward | Cosine T_max refinement: bracket T_max=30 with T_max=25 and 40 | WIP — STALE | Pod recovering from rate-limit; notified of 53.72 target |
| #4016 | fern | Tighter MLP/output grad-clip (other=0.5, 0.3) on 8-mech stack | WIP — NEW | #3725 diagnostic: MLP~5× noisier than attn; tighten dominant group; pw=2.0 elevates output-head grad further |
| #3731 | tanjiro | Signed log1p on pressure: direct asinh competitor (v2) | WIP — STALE | Still rate-limit blocked; notified of 53.72 target |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Pod recovering from rate-limit; notified of 53.72 target |

## Key open questions (round 12 — new baseline 56.00)

1. **Is pw=2.0 the pressure weight optimum or does the curve continue?** (#3984 nezuko) — pw monotone in {0.5, 1.0, 2.0}; sweeping 3.0 and 4.0 to find the peak
2. **Does EMA decay < 0.999 win on the 8-mech stack?** (#3989 askeladd) — 0.997/0.995 won pre-bf16; T_max=30+pw=2.0 changed convergence horizon; overdue test
3. **Does torch.compile buy meaningful per-epoch speedup?** (#3970 alphonse) — #3884+#3750 confirmed throughput-bound; compile is direct attack
4. **Does lr_init shift under T_max=30?** (#3953 frieren) — lr=2.1/2.5e-4 under T_max=30+pw=2.0; prior lr=2.5e-4 failure was on T_max=80 stack
5. **Is T_max=30 optimal or is there headroom?** (#3887 edward) — T_max=25 and 40 bracket the winner
6. **Is `other` (MLP/output) gradient over-clipped by the existing single-clip(1.0)?** (#4016 fern) — #3725 revealed MLP gradients ~5× larger than attention (other_gn mean ~18); tightening to 0.5/0.3 tests if stronger clipping of the dominant group helps on the 8-mech stack
7. **Does signed log1p beat asinh?** (#3731 tanjiro) — more aggressive tail compression
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — input-dependent gating

## 7-mechanism stack: gradient pipeline analysis

Eight mechanisms targeting DIFFERENT points:
1. **Lion (sign-based update)**: optimizer level — single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level — balance surface vs volume loss
3. **asinh**: loss-level — per-coordinate pressure z-score compression
4. **EMA**: parameter level — exponential trajectory smoothing
5. **grad-clip**: gradient vector level — L2 norm cap at 1.0
6. **bf16 autocast**: compute level — forward+loss precision reduction; 23% faster/epoch
7. **cosine T_max=30**: schedule level — moderate late-epoch annealing (40% of initial LR at epoch 18)
8. **pressure_weight=2.0**: loss level — up-weights pressure channel MAE 2× in training loss; constructively stacks with asinh

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
| #3725 (fern per-group grad-clip attn vs MLP) | no_improvement: val +1.1–3.9% vs 56.00; diagnostic revealed MLP/output ~5× noisier than attention; direction inverted from hypothesis; #4016 tests tighter MLP clip below 1.0 on 8-mech stack |
| #3442 (tanjiro signed-log1p) | Closed stale; reassigned as #3731 |
| #3383 (edward warmup-cosine) | Closed stale; reassigned as #3733 |
| #3275 (thorfinn SwiGLU) | Closed stale; reassigned as #3734 |
| #3470 (askeladd EMA-decay) | Closed prematurely; reassigned as #3776 |
| #3733 (edward warmup-cosine v2) | +4.2% regression (val=61.33 vs 58.87); warmup shifted curve right without compensating gain |
| #3750 (alphonse capacity-bf16) | no_improvement: Arm A (n144) val=59.85 (+1.66%), Arm B (n_layers=6) val=64.57 (+9.7%); per-epoch time +14-20% drops epoch count 18→15-16; throughput-bound |
| #3884 (alphonse batch-size-bf16) | no_improvement: Arm A (batch=6) val=69.68 (+24.4%); steps/epoch −33% (250 vs 374); per-epoch time flat (+3s); throughput fully saturated at batch=4 — step-count is binding |
| #3949 (askeladd lion-beta1) | no_improvement: Arm A (β1=0.95) val=61.49 (+9.8%); stop condition triggered; β1=0.90 appears optimal at LR=1.7e-4/T_max=30/bs=4; β1<0.90 remains untested |
| #3776 (askeladd EMA-decay-bf16-v2) | Closed at 11:35Z without terminal results; force-push reset; likely rate-limit cascade failure preventing pod commits |
| #3726 (frieren Lion-wd-sweep) | Closed at 11:33Z without terminal results; force-push reset; same rate-limit cascade issue |

## Potential next research directions

- ~~Finer surf_weight sweep~~ — CLOSED
- ~~Higher LR sweep (T_max=80 stack)~~ — CLOSED (#3586)
- ~~Capacity (n_hidden=144, n_layers=6)~~ — CLOSED by #3750; retestable with longer budget
- ~~Cosine T_max alignment~~ — MERGED (#3822)
- ~~EMA decay re-tune on bf16/T_max=30 stack~~ — CLOSED #3776 (no results due to rate-limit; direction was convergence-horizon hypothesis; may revisit)
- ~~Lion weight decay sweep~~ — CLOSED #3726 (no results due to rate-limit; may revisit on 7-mech stack)
- ~~Batch size scaling~~ — CLOSED #3884 (batch=6 +24.4%; step-count binding; bf16 compute-saturated at batch=4)
- **torch.compile throughput** — IN PROGRESS as #3970 (alphonse); direct attack on throughput-bound constraint
- **Per-channel pressure weight** — MERGED #3674 (pw=2.0, −4.07%); now fine-sweeping upward as #3984
- **Pressure weight fine sweep** — IN PROGRESS as #3984 (nezuko); pw=3.0/4.0; monotone in {0.5,1.0,2.0}
- **EMA decay on 8-mech stack** — IN PROGRESS as #3989 (askeladd); overdue test with T_max=30+pw=2.0
- **T_max refinement** — IN PROGRESS as #3887 (edward)
- ~~Lion β1 momentum sweep~~ — CLOSED #3949 (β1=0.95 +9.8%; default 0.90 optimal; β1<0.90 direction untested)
- **LR × T_max coupling** — IN PROGRESS as #3953 (frieren); lr=2.1/2.5e-4 under T_max=30
- **WeightedRandomSampler dynamic reweighting**: inverse-error resampling after epoch 1 — note: static domain-balanced sampler already present; dynamic per-sample reweighting is the next step
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p (pressure has very different statistics)
- **Lion β2 sweep**: β2=0.95 or 0.98 vs current 0.99 — separate axis from β1
