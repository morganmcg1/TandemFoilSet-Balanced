# SENPAI Research State

- **Date**: 2026-05-12 (round 1 kickoff)
- **Most recent research direction from human researcher team**: No directives yet — clean start.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Research Focus

This is round 1 on the TandemFoilSet-Balanced CFD surrogate benchmark. The baseline is a Transolver (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, ~1M params) with AdamW + CosineAnnealingLR, MSE loss in normalized space, `surf_weight=10`, `batch_size=4`. No prior experiments have been completed on this branch.

**Primary metric**: `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 val splits. Lower is better.
**Decision metric at merge**: `test_avg/mae_surf_p`

All training executions are capped at 30 minutes wall-clock (`SENPAI_TIMEOUT_MINUTES=30`). Given mesh sizes of 74K–242K nodes and baseline batch size 4, we expect ~5–15 epochs per run.

### Key known challenges
1. **Scale variance**: Per-sample y std varies 10× across the dataset (164–2077 m/s). MSE is biased toward high-Re samples.
2. **Small baseline model**: ~1M params may be under-capacity for 3 physically distinct domains.
3. **Short training window**: 30-min cap means we need techniques that show signal within 5–10 epochs.
4. **OOD geometry holdouts**: `val_geom_camber_rc` (raceCar M=6–8) and `val_geom_camber_cruise` (cruise M=2–4) test camber interpolation — the model must generalize from M=2-5 and M=9 training distributions.

---

## Round 1 Active Experiments (8 WIP PRs)

| PR | Student | Slug | Hypothesis axis |
|----|---------|------|-----------------|
| #1456 | alphonse | `bf16-amp` | Throughput — bf16 AMP for more epochs in 30 min |
| #1457 | askeladd | `surf-weight-50` | Loss alignment — `surf_weight` 10→50 |
| #1458 | edward | `wider-deeper` | Capacity — `n_hidden=256, n_layers=6, n_head=8` |
| #1460 | fern | `relative-l2-loss` | Loss reformulation — per-sample relative L2 to fix high-Re bias |
| #1462 | frieren | `warmup-cosine` | Optimizer — 2-epoch linear warmup before cosine |
| #1467 | nezuko | `more-slices-128` | Architecture — `slice_num=64→128` in PhysicsAttention |
| #1473 | tanjiro | `huber-loss` | Loss robustness — Huber (delta=0.5) for outlier nodes |
| #1479 | thorfinn | `grad-clip-1` | Stability — gradient clipping (clip_norm=1.0) |

---

## Potential Next Research Directions (Round 2)

These are held back from round 1 but strong candidates for round 2 depending on round 1 outcomes:

- **`lion-optimizer`**: Replace AdamW with Lion (sign-momentum, lower memory). Held because LR sensitivity requires careful tuning.
- **`higher-lr-clipped`**: AdamW lr=2e-3 + grad clipping + warmup — aggressive schedule for faster convergence in short budget.
- **`perchannel-heads`**: Separate decoder heads for Ux, Uy, p — pressure specialization. Held because code complexity is higher.
- **`silu-activation`**: SiLU throughout model. Small expected gain; may combine with round 2 winners.

**Compounding**: If `bf16-amp` wins and `surf-weight-50` wins, round 2 should combine them both as the new baseline and add the next lever. Improvements tend to be orthogonal and stack.

**If round 1 plateau**: Move to architecture exploration (GNOT-style multi-query attention, graph neural operators, FNO-style spectral layers on mesh).

---

## Dataset Analysis Notes

See `research/DATASET_ANALYSIS.md` (to be created after first round results clarify where errors are concentrated).

Key structural facts:
- 3 domains: raceCar single (85K nodes/sample), raceCar tandem (127K), cruise tandem (210K)
- OOD geometry: train on M=2–5 + M=9 (raceCar), M=0–2 + M=4–6 (cruise); val/test on M=6–8 (raceCar) and M=2–4 (cruise)
- All domains equally weighted via `WeightedRandomSampler` in the baseline
