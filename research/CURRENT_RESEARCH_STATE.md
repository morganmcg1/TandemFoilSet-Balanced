# SENPAI Research State

- 2026-05-15 15:30 — round 2/3 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline (BASELINE.md)

**val_avg/mae_surf_p = 135.0153** (PR #3119 — thorfinn, merged 2026-05-15)

Effective config: Transolver 128h/5L/4H/slice64/mlp2, AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, batch=4, surf_weight=10. ~14 effective epochs (30-min timeout cap). 42GB peak VRAM.

Known issues: `test_avg/mae_surf_p` is NaN due to a pre-existing bug in `data/scoring.py:accumulate_batch` (NaN×0=NaN propagation when 1 test_cruise GT sample has NaN p). Fix is tracked as bug-fix PR #3274 (fern).

## Round 1 results

| PR | Student | Theme | val_avg/mae_surf_p | Decision |
|----|---------|-------|-------------------|----------|
| #3119 | thorfinn | epochs 50→80 | **135.0153** ← baseline | Merged |
| #3104 | fern | per-channel p 4× | 149.1018 (+10.4%) | Closed — regression |
| #3110 | nezuko | batch 8, lr 8e-4 | 150.4371 (+11.4%) | Closed — undertrained (fewer steps/cap) |
| #3099 | alphonse | capacity 192h/6L/6H | 169.99 at ep7 (sent back; rerun lr=1e-3) | Sent back |
| #3101 | askeladd | surf_weight 30 | **127.4122 (-5.6%)** ← new best | Pending rebase |
| #3102 | edward | OneCycleLR max_lr=1e-3 | WIP (rate-limited; resumed) | — |
| #3106 | frieren | slice_num 128, head 8 | 163.98 at ep6 (sent back; rerun lr=1e-3) | Sent back |
| #3115 | tanjiro | Re-FiLM (log Re) | WIP (rate-limited; resumed) | — |

## Round 2 assignments

| PR | Student | Theme | Status |
|----|---------|-------|--------|
| #3274 | fern | Bug fix: NaN-safe evaluate_split (nan_to_num before masked sum) | WIP |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP |
| #3293 | nezuko | Lion optimizer (lr=1.7e-4, wd=3e-4) replacing AdamW | WIP |

## Round 2 sent-back reruns

| PR | Student | Rerun guidance |
|----|---------|---------------|
| #3099 | alphonse | Capacity 192h/6L/6H; rerun with lr=1e-3 |
| #3106 | frieren | Slice128/head8/mlp3; rerun with lr=1e-3 |

## Promising pending result

**#3101 (askeladd) surf_weight=30**: val=127.41 (-5.6% vs baseline) — terminal SENPAI-RESULT posted. Blocked only by merge conflict; student pod has picked up the rebase request. **Merge when rebased.**

## Potential next research directions

- asinh / log-p target normalization for surface pressure (attacks heavy-tail magnitude variance in single_in_dist)
- AdamW β2 tuning (β2=0.95 is transformer best practice for noisy-gradient / small-batch regimes)
- Geometry-aware feature engineering (LE distance, signed surface-normal distance) for cruise/raceCar OOD splits
- Per-channel lighter loss weighting [1,1,2] for surface-p (fern's 4× was too aggressive; worth retrying lighter after NaN fix merges)
- Multi-scale / hierarchical Transolver variants if capacity scale-up (alphonse) shows positive but limited returns
- Geometry-conditional FiLM (camber, AoA) — joint Re+camber gating for geom-OOD splits (orthogonal to tanjiro's Re-only FiLM)
- Channel-decoupled head architectures (separate decoders for Ux/Uy vs p)
- RMSNorm in place of LayerNorm if SwiGLU (thorfinn) shows gain — they compound well
