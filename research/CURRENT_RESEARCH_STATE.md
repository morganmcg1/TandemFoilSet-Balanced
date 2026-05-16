# SENPAI Research State

- 2026-05-16 — round 12 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| #3293 (nezuko, Lion) | 117.5014 | −7.8% | AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4 |
| #3357 (tanjiro, asinh-loss) | 84.9819 | −27.7% | asinh(z) on pressure channel z-scores in training loss |
| #3382 (askeladd, EMA+asinh) | 83.1874 | −2.1% | EMA shadow decay=0.999 at val/test passes |
| #3384 (fern, grad-clip+EMA+asinh) | 70.2479 | −15.6% | grad_clip(max_norm=1.0) before optimizer.step() |
| **#3530 (frieren, surf_weight=25)** | **67.2991** | **−4.20%** | **surf_weight: 30→25 (5-mech stack now complete)** |

**Current HEAD (5 mechanisms):** Lion + surf_weight=25 + asinh pressure-loss + EMA(0.999) + grad_clip(max_norm=1.0). val=67.30 at epoch 14 (timeout-bound, val still descending).

**Cumulative improvement from initial baseline:** 135.02 → 67.30 = **−50.2%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3725 | fern | Per-group grad-clip: attention (max_norm=1.0) vs MLP (5.0/10.0) | WIP — NEW | Sub-additivity finding from #3528; test if MLP is over-clipped |
| #3726 | frieren | Lion weight decay sweep: wd=1e-3 and 3e-3 (Lion paper rec.) | WIP — NEW | Lion paper: 3-10× higher wd than AdamW; current 3e-4 never tested |
| #3674 | nezuko | Per-channel pressure weight: w_p=0.5 and 2.0 vs 1.0 | WIP — NEW | On 5-mech baseline 67.30; #3586 closed (lr=2.5e-4 regressed +2.74%) |
| #3442 | tanjiro | signed log1p on pressure (stronger compression) | WIP — REBASING | Rebase guided; target 67.30 |
| #3470 | askeladd | EMA decay ablation: 0.997/0.995/0.990 | WIP — REBASING | Rebased on grad-clip; target 67.30 |
| #3485 | alphonse | bf16 autocast: faster forward → more epochs | WIP — REBASING | Rebase guided; target 67.30 |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP — REBASING | Rebase guided; target 67.30 |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP — REBASING | Rebase guided; target 67.30 |

## Key open questions (round 12)

1. **Is MLP gradient over-clipped?** (#3725 fern) — 100% clip rate at aggregate norms 25-180; per-group clip tests if attention drives the aggregate while MLP is unnecessarily squeezed. Natural successor to #3528 sub-additivity finding.
2. **Is Lion weight decay too low?** (#3726 frieren) — Lion paper recommends 3-10× higher wd than AdamW; current wd=3e-4 matches AdamW convention, not Lion. wd=1e-3 and 3e-3 not yet tested.
3. **Does per-channel pressure weight matter?** (#3674 nezuko) — w_p=0.5 (free velocity gradient) vs w_p=2.0 (focus on metric channel). Both w_p=1.0 equilibrium and LR tested; lr=1.7e-4 is near-optimal (#3586 finding)
4. **Does signed log1p beat asinh?** (#3442 tanjiro) — now needs to beat 67.30
5. **Does EMA decay=0.995 converge faster within budget?** (#3470 askeladd)
6. **Does bf16 give meaningful epoch count boost?** (#3485 alphonse) — key to unlocking capacity and schedule changes
7. **Does warmup-cosine help?** (#3383 edward) — on the 5-mech stack
8. **Does SwiGLU activate gating help?** (#3275 thorfinn) — doesn't increase per-epoch cost

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

## Potential next research directions

- ~~Finer surf_weight sweep around 25~~ — CLOSED: sw=22 regressed +4.09%; curve is flat in [20, 25]. sw=25 is at the plateau.
- ~~Higher LR sweep~~ — CLOSED: lr=2.5e-4 regressed +2.74%; lr=1.7e-4 is near-optimal for 5-mech stack
- **bf16 × capacity**: if bf16 cuts per-epoch time to 90s, a moderate capacity bump (n_hidden=160) could reach 18+ epochs within budget
- **Per-group grad-clip**: attention projection and output head dominate the aggregate norm; per-group clipping lets us tune aggressiveness separately
- **WeightedRandomSampler**: inverse-error reweighting after epoch 1 — fundamentally different mechanism from all 5 current wins
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p
- **Cosine schedule T_max alignment**: T_max=14 (matching wall-clock cap) — currently the cosine schedule barely anneals in 14 epochs (LR at 0.962×initial at epoch 14)
