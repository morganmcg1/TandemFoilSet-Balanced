# SENPAI Research State

- 2026-05-15 18:50 — round 6 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| #3293 (nezuko, Lion) | 117.5014 | −7.8% | AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4 |
| **#3357 (tanjiro, asinh-loss)** | **84.9819** | **−27.7%** | **asinh(z) on pressure channel z-scores in training loss** |

**Current HEAD:** Lion + surf_weight=30 + NaN-safe eval + asinh pressure-loss. val=84.98 at epoch 14 (timeout-bound, curve still descending at cutoff).

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3411 | tanjiro | Extend asinh to Ux/Uy channels (all-channel) | WIP — NEW | On full asinh baseline 84.98 |
| #3354 | nezuko | Lion + cap-matched cosine (T_max=12) | WIP | Notified of 84.98 target |
| #3382 | askeladd | EMA weights (decay=0.999) for evaluation | WIP | Notified of 84.98 target |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP | Notified of 84.98 target |
| #3384 | fern | Gradient clipping (max_norm=1.0) with Lion | WIP | Notified of 84.98 target |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP | Notified of 84.98 target |
| #3106 | frieren | Slice128/head8/mlp3 + Lion lr=3.4e-4 rerun | WIP | Notified of 84.98 target |
| #3099 | alphonse | Capacity 192h/6L/6H + Lion lr=3.4e-4 rerun | WIP | Notified of 84.98 target |

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression; FiLM helps in-dist but hurts OOD |
| #3328 (askeladd surf_weight=50) | +25% regression; instability above sw=30 |
| #3329 (fern AdamW β2=0.95) | +21% regression; wrong smoothing direction for B=4 |
| #3102 (edward OneCycleLR) | +20% regression; schedule shape wrong for 14-epoch budget |

## Critical insight: asinh loss transform is a fundamental win

The -27.7% improvement from asinh is larger than the entire prior improvement history combined. The mechanism: heavy-tail pressure z-scores dominate the gradient under standard squared error, causing the model to over-optimize for rare extreme samples. asinh compresses these z-scores such that the optimization gradient is proportional to 1/|z| for extreme samples, making the loss landscape smoother and faster to converge.

Val curve was STILL DESCENDING at epoch 14 (timeout cap), meaning:
1. The optimization landscape is genuinely easier — convergence is faster
2. More budget (longer wall-clock) would yield further improvements
3. The new frontier is to compound this with architectural/optimizer improvements

## Key open questions (round 6)

1. **Extend asinh to Ux/Uy?** (tanjiro #3411) — if velocity also has heavy-tail z-scores, extending compression to all channels could yield further 3-10% improvement
2. **Does asinh interact with architecture scale-up?** (alphonse #3099, frieren #3106) — larger models + Lion lr compression + asinh may stack
3. **Does asinh interact with schedule shape?** (nezuko #3354, edward #3383) — with faster convergence from asinh, does T_max or warmup matter differently?
4. **Does asinh interact with optimization tricks?** (askeladd EMA, fern gradclip) — EMA may help less because val curve is already smoother; gradclip may help less because asinh reduces gradient magnitude variance

## Potential next research directions (post-round-6)

- **asinh tau parameter**: `asinh(z/tau)*tau` with learnable or annealed tau per channel — optimizes the compression knee
- **Longer run confirmation**: at current 84.98 (curve still descending at timeout), a 45-min run or cosine T_max=12 would push into the 70-80 range
- **surf_weight ablation with asinh**: the 30× weighting was set before asinh — if asinh already compresses pressure extremes, surf_weight may need reducing (askeladd's analysis suggested 30 was optimal vs 50, but on the old MSE loss)
- **WeightedRandomSampler toward hard samples**: inverse-error reweighting after epoch 1 — the mechanism is different from asinh and could stack
- **Channel-decoupled output heads**: separate mlp2 for Ux/Uy vs p — the pressure-specific inductive bias argument is even stronger now that we know pressure z-scores are the bottleneck
- **Geometry-aware features**: LE distance, signed normal distance — after FiLM failure, direct feature injection more promising
- **Learnable per-channel output scaling**: scale + offset per output channel, jointly learned
