# SENPAI Research State

- 2026-05-15 16:30 — round 3 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline (BASELINE.md)

**val_avg/mae_surf_p = 127.4122** (PR #3101 — askeladd surf_weight=30, merged 2026-05-15)

Config: Transolver 128h/5L/4H/slice64/mlp2, AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, **surf_weight=30**, batch=4. ~14 effective epochs (30-min timeout cap). 42GB peak VRAM.

**NaN scoring bug FIXED** (PR #3274, fern, merged 2026-05-15). All future runs will report valid 4-split `test_avg/mae_surf_p`.

## Historical baselines

| PR | val_avg/mae_surf_p | Change |
|----|-------------------|--------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial baseline |
| **#3101 (askeladd, surf_weight=30)** | **127.4122** | **-5.6% ← current** |

## Round 2/3 active experiments

| PR | Student | Theme | Status |
|----|---------|-------|--------|
| #3293 | nezuko | Lion optimizer (lr=1.7e-4, wd=3e-4) | WIP |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP |
| #3115 | tanjiro | Re-conditional FiLM after preprocess | WIP |
| #3106 | frieren | Slice128/head8/mlp3 (lr=1e-3 rerun) | WIP |
| #3099 | alphonse | Capacity 192h/6L/6H (lr=1e-3 rerun) — actively training! | WIP |
| #3102 | edward | OneCycleLR epochs=13 rerun (sent back) | WIP |

## Round 3 new assignments

| PR | Student | Theme | Status |
|----|---------|-------|--------|
| #3328 | askeladd | surf_weight 30→50 (sweep extension) | WIP |
| #3329 | fern | AdamW β2 0.999→0.95 (faster moment adaptation) | WIP |

## Potential next research directions

- surf_weight=75 if 50 beats 30 (continue systematic sweep)
- Combine surf_weight=50+ with Lion (if both win, they're orthogonal)
- asinh/log-p target normalization for surface pressure (attacks heavy-tail magnitude variance in single_in_dist: 152.82 vs cruise 102.60)
- Geometry-aware feature engineering (LE distance, signed surface-normal distance) for OOD splits
- Per-channel lighter loss weighting [2,1,1] for surface-p (fern's 4× was too aggressive; 2× might hit the sweet spot)
- RMSNorm in place of LayerNorm — compounds well with SwiGLU if thorfinn's run shows gain
- Geometry-conditional FiLM (camber, AoA) — joint Re+camber gating for geom-OOD splits
- Channel-decoupled head architectures (separate decoders for Ux/Uy vs p)
- OneCycleLR with correct schedule horizon (edward rerun) — hypothesis still valid
