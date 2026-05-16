# SENPAI Research State

- 2026-05-16 — round 12 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team
- **New baseline: val=58.8717** (PR #3485 bf16 autocast merged, −12.5% from 67.30)

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
| **#3485 (alphonse, bf16 autocast)** | **58.8717** | **−12.5%** | **bf16 autocast on forward+loss; 18 epochs vs 14** |

**Current HEAD (6 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.999) + grad_clip(max_norm=1.0) + bf16 autocast. val=58.87 at epoch 18 (timeout-bound, val still descending).

**Cumulative improvement from initial baseline:** 135.02 → 58.87 = **−56.4%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3750 | alphonse | Capacity expansion on bf16: n_hidden=144 vs n_layers=6 | WIP — NEW | bf16 freed 9 GB VRAM; original #3099 capacity fail was throughput-bound |
| #3725 | fern | Per-group grad-clip: attention (max_norm=1.0) vs MLP (5.0/10.0) | WIP | Needs rebase for bf16; beat 58.87 or 67.30 if pre-bf16 |
| #3726 | frieren | Lion weight decay sweep: wd=1e-3 and 3e-3 (Lion paper rec.) | WIP | Needs rebase for bf16; beat 58.87 or 67.30 if pre-bf16 |
| #3674 | nezuko | Per-channel pressure weight: w_p=0.5 and 2.0 vs 1.0 | WIP — STALE | Rebase advised; beat 58.87 with bf16 or 67.30 without |
| #3731 | tanjiro | Signed log1p on pressure: direct asinh competitor (v2) | WIP | Needs rebase for bf16; beat 58.87 or 67.30 if pre-bf16 |
| #3776 | askeladd | EMA decay ablation on bf16 stack: 0.997/0.995 (v2) | WIP — NEW | #3470 closed prematurely; reassigned on bf16 stack; pre-bf16 0.997→62.21, 0.995→62.67 |
| #3822 | edward | Cosine schedule T_max alignment: 20 and 30 vs current 80 | WIP — NEW | Late-schedule complement to closed warmup #3733; current LR at 0.852× initial at epoch 18 |
| #3734 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs (v2) | WIP | Needs rebase for bf16; beat 58.87 or 67.30 if pre-bf16 |

## Key open questions (round 12 — new baseline 58.87)

1. **Does moderate capacity expand on bf16 win?** (#3750 alphonse) — 9 GB freed VRAM opens n_hidden=144 or n_layers=6; original #3099 failed due to fp32 epoch budget starvation
2. **Is MLP gradient over-clipped?** (#3725 fern) — 100% clip rate at norms 25-180; per-group clip tests if attention drives the aggregate while MLP is squeezed
3. **Is Lion weight decay too low?** (#3726 frieren) — Lion paper recommends 3-10× higher wd; current 3e-4 is AdamW-style; wd=1e-3 and 3e-3 untested
4. **Does per-channel pressure weight matter?** (#3674 nezuko) — w_p=0.5 vs w_p=2.0; pressure channel rebalancing
5. **Does signed log1p beat asinh?** (#3731 tanjiro) — more aggressive tail compression; direct competitor to the winning mechanism
6. **Does EMA decay=0.997 or 0.995 still beat 0.999 on bf16?** (#3776 askeladd v2) — convergence-horizon hypothesis: 4 extra epochs may neutralize the pre-bf16 advantage of faster decays
7. **Does T_max alignment unlock late-epoch refinement?** (#3822 edward) — current cosine essentially constant (LR 0.85× initial at epoch 18); T_max=20/30 tests if Lion is bouncing at minimum
8. **Does SwiGLU gating improve OOD generalization?** (#3734 thorfinn) — input-dependent gating for surface vs volume node specialization

## 5-mechanism stack: gradient pipeline analysis

Five mechanisms targeting DIFFERENT points:
1. **Lion (sign-based update)**: optimizer level — single momentum buffer, fixed-magnitude steps
2. **surf_weight=25**: loss level — balance of surface vs volume loss; optimal knee identified at 25 (lower flattens re_rand/cruise gains)
3. **asinh**: loss-level (per-coordinate pressure z-score compression, ~3-5× norm reduction)
4. **EMA**: parameter level (exponential trajectory smoothing, 0 sign flips vs 8 for Lion)
5. **grad-clip**: gradient vector level (L2 norm cap, still 100% clip rate post-asinh at mean 25-180)

They compose cleanly because they're orthogonal. The surf_weight reduction from 30→25 is explained by mechanisms 3+4+5 already implicitly reducing extreme-pressure gradient signal.

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression |
| #3328 (askeladd surf_weight=50) | +25% regression; instability above sw=30 |
| #3329 (fern AdamW β2=0.95) | +21% regression |
| #3102 (edward OneCycleLR) | +20% regression |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores light-tailed |
| #3099 (alphonse capacity 192h/6L/6H) | +60.5% regression; wall-clock budget dominates |
| #3106 (frieren Slice128/head8/mlp3) | +98.6% regression; same wall-clock penalty, 7 epochs vs 14 |
| #3354 (nezuko Lion+cosine T_max=12) | +15.96% regression (81.45 vs 70.25); val still descending at epoch 12, curve is budget-limited not LR-limited |
| #3586 (nezuko higher LR 2.5e-4) | +2.74% regression (69.14 vs 67.30); lr=1.7e-4 is near-optimal for 5-mech stack; closed without running Arm B |
| #3656 (frieren surf-weight-fine sw=22/27) | +4.09% regression (sw=22 val=70.04); interpolation hypothesis falsified; curve is flat in [20,25], gain is fully captured at sw=25 |
| #3528 (fern grad-clip rebased sw×max_norm grid) | +0.57% regression (val=67.68 vs 67.30); max_norm=1.0 is optimal; sub-additive composition confirmed: asinh+EMA+grad-clip share gradient bandwidth |
| #3442 (tanjiro signed-log1p) | Closed stale — branch stuck on pre-asinh base (Round-7) despite 4 rebase-guidance comments; reassigned as #3731 on fresh HEAD |
| #3383 (edward warmup-cosine) | Closed stale — branch stuck on pre-asinh base (Round-4) despite 5 rebase-guidance comments; reassigned as #3733 on fresh HEAD |
| #3275 (thorfinn SwiGLU) | Closed stale — branch stuck at initial baseline (Round-0) despite 6 rebase-guidance comments; reassigned as #3734 on fresh HEAD |
| #3470 (askeladd EMA-decay) | Closed prematurely — pre-bf16 arms ran but didn't beat new 58.87 bf16 baseline (0.997→62.21, 0.995→62.67, 0.990→64.32); student rebased and started bf16 re-runs but PR was already closed; reassigned as #3776 on bf16 stack |
| #3733 (edward warmup-cosine v2) | +4.2% regression on bf16 stack (val=61.33 vs 58.87); 2-epoch warmup shifted val curve right by ~1 epoch in fixed budget without compensating stability gain; Lion+EMA+grad-clip already absorb early instability; reassigned as #3822 cosine-tmax-align (late-schedule complement) |

## Potential next research directions

- ~~Finer surf_weight sweep around 25~~ — CLOSED: sw=22 regressed +4.09%; curve is flat in [20, 25]. sw=25 is at the plateau.
- ~~Higher LR sweep~~ — CLOSED: lr=2.5e-4 regressed +2.74%; lr=1.7e-4 is near-optimal for 5-mech stack
- **bf16 × capacity**: if bf16 cuts per-epoch time to 90s, a moderate capacity bump (n_hidden=160) could reach 18+ epochs within budget
- **Per-group grad-clip**: attention projection and output head dominate the aggregate norm; per-group clipping lets us tune aggressiveness separately
- **WeightedRandomSampler**: inverse-error reweighting after epoch 1 — fundamentally different mechanism from all 5 current wins
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p
- ~~Cosine schedule T_max alignment~~ — IN PROGRESS as #3822 (edward, T_max=20 and 30)
