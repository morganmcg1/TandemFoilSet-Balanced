# SENPAI Research State

- 2026-05-15 18:30 — round 5 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| **#3293 (nezuko, Lion)** | **117.5014** | **−7.8%** | **AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4** |

**Current HEAD:** Lion + surf_weight=30 + NaN-safe eval (val=117.50 measured at surf10; first Lion+surf30 measurement pending from #3354 nezuko).

## Active experiments

| PR | Student | Theme | Status |
|----|---------|-------|--------|
| #3354 | nezuko | Lion + cap-matched cosine (epochs 80→12) | WIP |
| #3357 | tanjiro | asinh loss transform for surface pressure | WIP |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP (just rebased, training imminent) |
| #3106 | frieren | Slice128/head8/mlp3 + Lion lr=3.4e-4 rerun | WIP (updated guidance posted) |
| #3099 | alphonse | Capacity 192h/6L/6H + Lion lr=3.4e-4 rerun | WIP (updated guidance posted) |
| #3382 | askeladd | EMA weights (decay=0.999) for evaluation | WIP — NEW |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP — NEW |
| #3384 | fern | Gradient clipping (max_norm=1.0) with Lion | WIP — NEW |

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression; FiLM helps in-dist but hurts OOD splits |
| #3328 (askeladd surf_weight=50) | +15.5% regression vs sw=30; +25% vs Lion baseline. Instability; optimum at or below 30 |
| #3329 (fern AdamW β2=0.95) | +11.8% regression. Wrong direction for B=4; more smoothing needed not less. AdamW also superseded by Lion |
| #3102 (edward OneCycleLR) | +10.8% regression. Both arms (ep50-truncated, ep13-sized) failed. Schedule shape structurally wrong for 14-epoch budget |

## Key findings from round 4 experiments

- **surf_weight optimum is at 30**: pushing above (50) crosses into instability. Volume-loss signal is load-bearing for surface prediction — over-weighting surface starves the volume branch.
- **AdamW β2 tuning is dead**: first because wrong direction for B=4 heavy-tail gradients (need more smoothing), second because Lion has superseded AdamW.
- **OneCycleLR doesn't fit 14-epoch budget**: CosineAnnealingLR wins at this wall-clock constraint because it monotonically anneals; OneCycleLR spends too much time at peak LR when the budget is short.

## Active research axes (round 5)

Three new hypotheses all compound cleanly with current Lion+surf30 baseline:
1. **EMA weights** (#3382 askeladd): Lion paper explicitly recommends EMA; sign-based updates are jumpy, EMA smooths val without changing training dynamics
2. **Warmup-cosine** (#3383 edward): 2-epoch linear warmup to 1.7e-4, then cosine — tests whether early-epoch shock is Lion's weakness; directly follows from edward's OneCycleLR diagnosis
3. **Gradient clipping** (#3384 fern): clip_grad_norm=1.0, also from Lion paper recommendations; heavy-tail CFD gradients (B=4, variable mesh) likely perturb Lion momentum buffer

Frieren and alphonse reruns are testing **capacity + Lion lr compression** (lr=3.4e-4). This is a high-upside direction since larger models showed faster per-step convergence in round 1 — the question is whether Lion+2×lr recovers that advantage.

## Potential next research directions (post-round-5)

- **Combine winners**: Lion + EMA + grad-clip if both help independently
- **Combine winners**: Lion + SwiGLU (if thorfinn wins) + capacity (if alphonse/frieren win) — orthogonal axes
- **asinh combination**: if tanjiro's asinh loss helps, combine with Lion+surf30 as separate axis
- **surf_weight below 30**: askeladd's analysis suggested optimum might be 20-30; a quick surf_weight=20 sweep would bracket it
- **WeightedRandomSampler toward hard samples**: inverse-error reweighting after epoch 1 — targets heavy-tail single_in_dist split directly
- **Channel-decoupled output heads**: separate mlp2 for Ux/Uy vs p — directly targets pressure-specific inductive biases
- **Geometry-aware features** (LE distance, signed normal distance): after FiLM failure, direct feature injection is more promising than learned conditioning
- **Learnable per-channel output scaling** (scale + offset per output channel, jointly learned): could help the heavy-tail pressure regime without asinh transform complexity
- **DropPath / stochastic depth**: mild drop_path=0.1 in TransolverBlock — regularization without capacity change
