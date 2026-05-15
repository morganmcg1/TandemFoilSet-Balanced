# SENPAI Research State

- 2026-05-15 — round 1/2 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline (BASELINE.md)

**val_avg/mae_surf_p = 135.0153** (PR #3119 — thorfinn, merged 2026-05-15)

Effective config: Transolver 128h/5L/4H/slice64/mlp2, AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, batch=4, surf_weight=10. ~14 effective epochs (30-min timeout cap). 42GB peak VRAM.

Known issues: `test_avg/mae_surf_p` is NaN due to a pre-existing bug in `data/scoring.py:accumulate_batch` (NaN×0=NaN propagation when 1 test_cruise GT sample has NaN p). Fix is tracked as a bug-fix PR for fern.

## Round 1 results

| PR | Student | Theme | val_avg/mae_surf_p | Decision |
|----|---------|-------|-------------------|----------|
| #3119 | thorfinn | epochs 50→80 | **135.0153** ← baseline | Merged |
| #3104 | fern | per-channel p 4× | 149.1018 (+10.4%) | Closed — regression |
| #3099 | alphonse | capacity 192h/6L/6H | WIP | — |
| #3101 | askeladd | surf_weight 30 | WIP | — |
| #3102 | edward | OneCycleLR max_lr=1e-3 | WIP | — |
| #3106 | frieren | slice_num 128, head 8 | WIP | — |
| #3110 | nezuko | batch 8, lr 8e-4 | WIP | — |
| #3115 | tanjiro | Re-FiLM (log Re) | WIP | — |

## Round 2 assignments (idle students after round-1 reviews)

| PR | Student | Theme |
|----|---------|-------|
| TBD | fern | Bug fix: NaN-safe evaluate_split in train.py |
| TBD | thorfinn | SwiGLU activation in TransolverBlock MLPs |

## Potential next research directions

- Best-checkpoint physical-space target transforms (asinh / log-p) if val gradients are dominated by extreme-Re samples.
- Multi-scale / hierarchical Transolver variants if capacity scale-up shows positive but limited returns.
- Geometry-aware feature engineering (LE distance, signed surface-normal distance) for the cruise/raceCar camber-OOD splits.
- Lion or AdamW-β2-tuning if optimization plateau emerges.
- SwiGLU / RMSNorm activation-and-norm swap if architecture is the bottleneck.
- Geometry-conditional FiLM (camber, AoA) and joint Re+camber gating for the geom-OOD splits.
- Channel-decoupled head architectures (separate decoders for Ux/Uy vs p).
