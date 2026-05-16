# SENPAI Research State

- 2026-05-16 12:00Z — round 12 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=56.0011** (PR #3822 cosine T_max=30 merged, −4.88% from 58.87)

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
| **#3822 (edward, cosine T_max=30)** | **56.0011** | **−4.88%** | **CosineAnnealingLR T_max: 80→30; moderate late-epoch anneal** |

**Current HEAD (7 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.999) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30. val=56.00 at epoch 18 (timeout-bound, val still descending).

**Reproduce baseline:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30
```
(In-tree default is still T_max=80 — must pass `--cosine_t_max_epochs 30` explicitly.)

**Cumulative improvement from initial baseline:** 135.02 → 56.00 = **−58.5%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3884 | alphonse | Batch size scaling on bf16: batch=6 and 8 vs baseline 4 | WIP | bf16 freed 9 GB; bigger batch tests signal-quality vs step-count tradeoff |
| #3887 | edward | Cosine T_max refinement: bracket T_max=30 with T_max=25 and 40 | WIP | #3822 winner maps to the knee; "above 30" is student's own prediction |
| #3949 | askeladd | Lion β1 momentum sweep: 0.95 and 0.85 vs current 0.90 | WIP — NEW | Fresh axis; β1 untested on 7-mech stack; Lion paper cites as most sensitive HP |
| #3953 | frieren | LR × T_max coupling: lr=2.1e-4 and 2.5e-4 under T_max=30 | WIP — NEW | T_max=30 changed effective LR by 60%; optimal lr_init may have shifted upward |
| #3725 | fern | Per-group grad-clip: attention (max_norm=1.0) vs MLP (5.0/10.0) | WIP — STALE | Notified of 56.00 target; T_max=30 flag required |
| #3674 | nezuko | Per-channel pressure weight: w_p=0.5 and 2.0 vs 1.0 | WIP — STALE | Notified of 56.00 target; T_max=30 flag required |
| #3731 | tanjiro | Signed log1p on pressure: direct asinh competitor (v2) | WIP — STALE | Notified of 56.00 target; T_max=30 flag required |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP — STALE | Notified of 56.00 target; T_max=30 flag required |

## Key open questions (round 12 — new baseline 56.00)

1. **Does bigger batch (6 or 8) beat batch=4?** (#3884 alphonse) — throughput-vs-signal tradeoff; bf16 freed 9 GB; 1.17pt noise floor found in #3750
2. **Is T_max=30 the optimal or is there headroom above/below?** (#3887 edward) — T_max=25 and 40 bracket the winner; student predicts "above 30"
3. **Does Lion β1=0.95 or 0.85 beat 0.90?** (#3949 askeladd) — fresh optimizer-level axis; β1 sets gradient reactivity; never tested on 7-mech stack
4. **Does lr_init shift under T_max=30?** (#3953 frieren) — T_max=30 cut effective avg-LR 60%; lr=2.1e-4/2.5e-4 tests if optimal lr_init moved up; lr=2.5e-4 directly re-tests nezuko's prior failure
5. **Is MLP gradient over-clipped?** (#3725 fern) — 100% clip rate at norms 25-180; per-group clip
6. **Does per-channel pressure weight matter?** (#3674 nezuko) — w_p=0.5 vs w_p=2.0
7. **Does signed log1p beat asinh?** (#3731 tanjiro) — more aggressive tail compression
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — input-dependent gating

## 7-mechanism stack: gradient pipeline analysis

Seven mechanisms targeting DIFFERENT points:
1. **Lion (sign-based update)**: optimizer level — single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level — balance surface vs volume loss
3. **asinh**: loss-level — per-coordinate pressure z-score compression
4. **EMA**: parameter level — exponential trajectory smoothing
5. **grad-clip**: gradient vector level — L2 norm cap at 1.0
6. **bf16 autocast**: compute level — forward+loss precision reduction; 23% faster/epoch
7. **cosine T_max=30**: schedule level — moderate late-epoch annealing (40% of initial LR at epoch 18)

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
| #3442 (tanjiro signed-log1p) | Closed stale; reassigned as #3731 |
| #3383 (edward warmup-cosine) | Closed stale; reassigned as #3733 |
| #3275 (thorfinn SwiGLU) | Closed stale; reassigned as #3734 |
| #3470 (askeladd EMA-decay) | Closed prematurely; reassigned as #3776 |
| #3733 (edward warmup-cosine v2) | +4.2% regression (val=61.33 vs 58.87); warmup shifted curve right without compensating gain |
| #3750 (alphonse capacity-bf16) | no_improvement: Arm A (n144) val=59.85 (+1.66%), Arm B (n_layers=6) val=64.57 (+9.7%); per-epoch time +14-20% drops epoch count 18→15-16; throughput-bound |
| #3776 (askeladd EMA-decay-bf16-v2) | Closed at 11:35Z without terminal results; force-push reset; likely rate-limit cascade failure preventing pod commits |
| #3726 (frieren Lion-wd-sweep) | Closed at 11:33Z without terminal results; force-push reset; same rate-limit cascade issue |

## Potential next research directions

- ~~Finer surf_weight sweep~~ — CLOSED
- ~~Higher LR sweep (T_max=80 stack)~~ — CLOSED (#3586)
- ~~Capacity (n_hidden=144, n_layers=6)~~ — CLOSED by #3750; retestable with longer budget
- ~~Cosine T_max alignment~~ — MERGED (#3822)
- ~~EMA decay re-tune on bf16/T_max=30 stack~~ — CLOSED #3776 (no results due to rate-limit; direction was convergence-horizon hypothesis; may revisit)
- ~~Lion weight decay sweep~~ — CLOSED #3726 (no results due to rate-limit; may revisit on 7-mech stack)
- **Batch size scaling** — IN PROGRESS as #3884 (alphonse)
- **T_max refinement** — IN PROGRESS as #3887 (edward)
- **Lion β1 momentum sweep** — IN PROGRESS as #3949 (askeladd); 0.95/0.85 vs 0.90
- **LR × T_max coupling** — IN PROGRESS as #3953 (frieren); lr=2.1/2.5e-4 under T_max=30
- **WeightedRandomSampler dynamic reweighting**: inverse-error resampling after epoch 1 — note: static domain-balanced sampler already present; dynamic per-sample reweighting is the next step
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p (pressure has very different statistics)
- **Lion β2 sweep**: β2=0.95 or 0.98 vs current 0.99 — separate axis from β1
