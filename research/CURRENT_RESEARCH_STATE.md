# SENPAI Research State

- 2026-05-15 17:35 — round 4 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| **#3293 (nezuko, Lion)** | **117.5014** | **−7.8%** | **AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4** |

**Current HEAD:** Lion + surf_weight=30 + NaN-safe eval (val=117.50 measured at surf10; first Lion+surf30 measurement is pending from #3354 nezuko).

## Active experiments

| PR | Student | Theme | Status |
|----|---------|-------|--------|
| #3354 | nezuko | Lion + cap-matched cosine (epochs 80→12) | WIP |
| #3357 | tanjiro | asinh loss transform for surface pressure | WIP |
| #3329 | fern | AdamW β2 0.999→0.95 (faster moment adaptation) | WIP |
| #3328 | askeladd | surf_weight 30→50 sweep | WIP |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP |
| #3106 | frieren | Slice128/head8/mlp3 (lr=1e-3 rerun) | WIP |
| #3102 | edward | OneCycleLR (epochs=13 rerun) | WIP |
| #3099 | alphonse | Capacity 192h/6L/6H (lr=1e-3 rerun) | WIP |

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression; hypothesis falsified — Re gating helps in-dist but hurts OOD splits (opposite of prediction) |

## Potential next research directions

- **Lion + surf_weight confirmation**: nezuko #3354 will provide first clean Lion+surf30 measurement
- **Combine winning changes** (once results land): Lion+surf50 (if askeladd wins), Lion+SwiGLU (if thorfinn wins), Lion+β2=0.95 (if fern wins)
- **Larger capacity with Lion**: once alphonse confirms capacity helps at lr=1e-3, try capacity+Lion (orthogonal axes)
- **Channel-decoupled output heads**: separate mlp2 heads for Ux/Uy vs p — directly targets pressure-specific inductive biases
- **Geometry-aware feature engineering** (LE distance, signed normal distance) for OOD splits — after FiLM failure, direct feature injection is more promising than learned conditioning
- **Learnable per-channel output scaling** (scale + offset per output channel, jointly learned) — could help the heavy-tail pressure regime without asinh transform complexity
- **WeightedRandomSampler re-weighting toward hard samples** — inverse-error weighting after first epoch
- **Longer run confirmation**: if wall-clock budget is ever relaxed, Lion+surf30+T12 could yield 110-120 range given current trajectory
