# SENPAI Research State

- **Date**: 2026-05-12 (round 1 partial results)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 117.17** — PR #1479 (grad-clip-1), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=50)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, MSE loss.

---

## Current Research Focus

**Round 1 finding**: The Transolver baseline was gradient-unstable without clipping (pre-clip norms 50–800, 100% clip rate). `grad_clip=1.0` is now mandatory infrastructure. All future experiments must build on the grad-clip baseline.

**Key open question**: Do loss reformulation (relative-L2, Huber), architecture scaling (wider-deeper), and LR schedule changes compound on top of the stable grad-clip baseline?

**Per-split profile** at baseline:
- Easiest: `val_geom_camber_cruise` (87.04) — cruise tandem, lower Re, smaller y magnitudes
- Hardest: `val_geom_camber_rc` (134.17) and `val_single_in_dist` (134.83) — raceCar regime, high Re, extreme p values

**Known bug**: `test_geom_camber_cruise` has a corrupted sample (±Inf in GT pressure). Fix: add y-sanitization wrapper in `train.py:evaluate_split` before calling `accumulate_batch`. Students are instructed to include this fix.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP | |
| #1457 | askeladd | `surf-weight-50` | WIP (v2, sent back) | Now: surf_weight=30 + grad-clip baseline |
| #1458 | edward | `wider-deeper` | WIP (v2, sent back) | Now: batch_size=4 + grad-clip baseline |
| #1460 | fern | `relative-l2-loss` | WIP | |
| #1462 | frieren | `warmup-cosine` | WIP (v2, sent back) | Now: 1-epoch warmup + grad-clip baseline |
| #1467 | nezuko | `more-slices-128` | WIP | |
| #1473 | tanjiro | `huber-loss` | WIP | |
| #1518 | thorfinn | `higher-lr-cosine-14` | WIP (new) | lr=1e-3, T_max=14, exploit grad-clip stability |

---

## Potential Next Research Directions

Round 2 candidates (pending round 1 final results):

- **Compound: bf16-amp + grad-clip + wider-deeper**: If bf16 gives more epochs, wider model should dominate. Best expected compound.
- **surf_weight sweep**: Test 25, 50, 75 once stable baseline is confirmed.
- **`lion-optimizer`**: Lower memory + potential convergence benefit; needs LR tuning.
- **Fourier/position encoding**: Improve geometric encoding for OOD camber generalization (the harder splits are raceCar geometry and single-foil, not cruise).
- **SiLU activation**: Cheap, orthogonal to all other changes.
- **EMA**: Smooth val metrics from noisy 14-epoch trajectories.
- **CosineAnnealingWarmRestarts**: May suit the 14-epoch budget better than standard cosine.

**Plateau protocol**: If 5 consecutive experiments fail to beat 117.17, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
