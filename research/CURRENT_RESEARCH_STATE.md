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
| **#3384 (fern, grad-clip+EMA+asinh)** | **70.2479** | **−15.6%** | **grad_clip(max_norm=1.0) before optimizer.step()** |

**Current HEAD (4 mechanisms):** Lion + surf_weight=30 + NaN-safe eval + asinh pressure-loss + EMA(0.999) + grad_clip(max_norm=1.0). val=70.25 at epoch 14 (timeout-bound, val still descending).

**Cumulative improvement from initial baseline:** 135.02 → 70.25 = **−48.0%**

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3528 | fern | Grad-clip threshold sweep (2.0/5.0/10.0 vs 1.0) | WIP — NEW | On 4-mech baseline 70.25 |
| #3530 | frieren | surf_weight ablation (25/20 vs 30) on 4-mech stack | WIP — NEW | On 4-mech baseline 70.25 |
| #3442 | tanjiro | signed log1p on pressure (stronger compression) | WIP | Notified of 70.25 target |
| #3470 | askeladd | EMA decay ablation: 0.997/0.995/0.990 | WIP | Notified of 70.25 target |
| #3485 | alphonse | bf16 autocast: faster forward → more epochs | WIP | Notified of 70.25 target |
| #3586 | nezuko | Higher Lion LR: 2.5e-4/3.4e-4 vs 1.7e-4 on 4-mech stack | WIP — NEW | On 4-mech baseline 70.25; #3354 closed, key insight: curve still descending at epoch 14 |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP | Notified of 70.25 target |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP | Notified of 70.25 target |

## Key open questions (round 12)

1. **Can higher LR compress convergence within the budget?** (#3586 nezuko) — curve still descending at epoch 14; 1.5–2× LR multiplier on the denoised 4-mech stack should converge faster
2. **Is max_norm=1.0 over-clipping?** (#3528 fern) — post-asinh norms 25-180 mean; threshold of 5-10 may let bulk curvature signal through while still clipping spikes
3. **Is surf_weight=30 still optimal with 4 mechanisms?** (#3530 frieren) — asinh+clip both reduce effective pressure weighting; may need sw reduction to 20-25
4. **Does signed log1p beat asinh?** (#3442 tanjiro) — now needs to beat 70.25, harder target
5. **Does EMA decay=0.995 converge faster within budget?** (#3470 askeladd)
6. **Does bf16 give meaningful epoch count boost?** (#3485 alphonse) — key to unlocking capacity and schedule changes
7. **Does warmup-cosine help?** (#3383 edward) — on the 4-mech stack
8. **Does SwiGLU activate gating help?** (#3275 thorfinn) — doesn't increase per-epoch cost

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
| #3354 (nezuko Lion+cosine T_max=12) | +15.96% regression (81.45 vs 70.25); key insight: val still descending at epoch 12, curve is budget-limited not LR-limited |

## 4-mechanism stack: gradient pipeline analysis

Three mechanisms target DIFFERENT points:
1. **asinh**: loss-level (per-coordinate pressure z-score compression, ~3-5× norm reduction)
2. **EMA**: parameter level (exponential trajectory smoothing, 0 sign flips vs 8 for Lion)
3. **grad-clip**: gradient vector level (L2 norm cap, still 100% clip rate post-asinh at mean 25-180)

They compose cleanly because they're orthogonal. This is the theoretical grounding for why each subsequent mechanism can add value even after the prior one reduced noise at its own level.

## Key open questions (round 11 — superseded, see round 12 below)

1. **Is max_norm=1.0 over-clipping?** (#3528 fern) — post-asinh norms 25-180 mean; threshold of 5-10 may let bulk curvature signal through while still clipping spikes
2. **Is surf_weight=30 still optimal with 4 mechanisms?** (#3530 frieren) — asinh+clip both reduce effective pressure weighting; may need sw reduction to 20-25
3. **Does signed log1p beat asinh?** (#3442 tanjiro) — now needs to beat 70.25, harder target
4. **Does EMA decay=0.995 converge faster within budget?** (#3470 askeladd)
5. **Does bf16 give meaningful epoch count boost?** (#3485 alphonse) — key to unlocking capacity and schedule changes
6. **Does warmup-cosine help?** (#3383 edward) — on the 4-mech stack
7. **Does SwiGLU activate gating help?** (#3275 thorfinn) — doesn't increase per-epoch cost

## Potential next research directions

- **Grad-clip + schedule alignment**: T_max=14 (matching wall-clock cap) + optimal clip threshold — currently the cosine schedule barely anneals in 14 epochs (LR at 0.962×initial at epoch 14)
- **bf16 × capacity**: if bf16 cuts per-epoch time to 90s, a moderate capacity bump (n_hidden=160) could reach 18+ epochs within budget
- **Per-group grad-clip**: attention projection and output head dominate the aggregate norm; per-group clipping lets us tune aggressiveness separately
- **WeightedRandomSampler**: inverse-error reweighting after epoch 1 — fundamentally different mechanism from all 4 current wins, could be a 5th additive improvement
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p
- **Focal-loss style pressure weighting**: rather than asinh, apply a focal-loss γ to the pressure channel loss (reduces weight on easy/well-predicted samples)
