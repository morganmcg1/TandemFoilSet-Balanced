# SENPAI Research State
- 2026-04-29 11:00 (round: icml-appendix-charlie-pai2f-r3)
- Most recent research direction from human researcher team: None (no GitHub Issues)

## Current Best Baseline

**val_avg/mae_surf_p = 47.3987** (PR #1093, merged 2026-04-29)
Configuration: Lion + L1 + EMA(0.995) + bf16 + n_layers=1 + surf_weight=28 + cosine T_max=15 + clip_grad=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + 50 epochs

## Confirmed signals of round 3 (both pending rebase)

1. **PR #1104 — edward, FiLM global conditioning (Re/AoA/NACA)** — `val_avg/mae_surf_p = 42.3822` against reference 47.7385 (−11.2%) and against the new anchor 47.3987 (−10.6%). All four val splits improve. test_avg (post-fix rerun) = 35.8802 bf16 / 35.8504 fp32. Best epoch = 50 (still descending). DiT-style zero-init, scale+shift on both attn and MLP sublayers, conditioning read from node 0. Top priority of the round; sent back for rebase. Gate to merge: val_avg ≤ ~45.
2. **PR #1106 — frieren, Fourier positional encoding on (x, z)** — `val_avg/mae_surf_p = 45.3304` against reference 47.7385 (−5.05%) and against the new anchor 47.3987 (−4.36%). 3/4 splits improve, val_geom_camber_rc flat. test_avg/mae_surf_p = 38.1284. Best epoch = 50 (still descending). Sent back for rebase. Gate to merge: val_avg ≤ ~46.

Both PRs ship a fix for the same critical NaN-leak bug in `evaluate_split` (sample 20 of `test_geom_camber_cruise` has 761 inf ground-truth entries that contaminate bf16 test metrics via `inf * 0`); the fix needs to land regardless of which experiment lands first.

## Current Research Focus

charlie-pai2f round 3: build on the newly merged compound baseline (PR #1093). Best checkpoint was at final epoch (50) → training duration and schedule shape remain high-priority levers. Positional/geometric representation has emerged as the strongest signal so far and is the first follow-up family to deepen.

- **Primary metric**: `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface-pressure MAE across 4 val splits

## Active Experiments (8/8 students running)

| PR   | Student   | Hypothesis                                                        | Status |
|------|-----------|-------------------------------------------------------------------|--------|
| #1103 | askeladd  | `slice_num` sweep {32, 64, 128}                                  | wip |
| #1104 | edward    | FiLM global conditioning (Re/AoA/NACA via scale+shift on hidden states) — **REBASE** | wip (draft, sent back) |
| #1105 | fern      | Per-channel pressure weight W_p sweep {2, 3, 5}                  | wip |
| #1106 | frieren   | Fourier positional encoding on (x,z) — REBASE                    | wip (draft, sent back) |
| #1107 | nezuko    | EMA decay sweep {0.99, 0.995, 0.999}                             | wip |
| #1108 | tanjiro   | n_hidden width sweep {128, 192, 256}                             | wip |
| #1109 | thorfinn  | Boundary-layer proxy feature: log(Re×|saf|+ε)                   | wip |
| #1127 | alphonse  | Extended training: 75 epochs + cosine T_max=10 for convergence   | wip |

## Current Research Themes

1. **Physics conditioning** — FiLM global conditioning (edward, **leading signal at −10.6%**), boundary-layer features (thorfinn). Strongest current direction.
2. **Positional / geometric representation** — Fourier positional encoding (frieren, second-strongest signal at −4.4%). Natural follow-ups: frequency-count sweep, learnable Gaussian frequencies (Tancik et al. 2020), Fourier expansion of the 8-dim `dsdf` distance descriptor.
3. **Architecture capacity** — slice_num (askeladd), n_hidden width (tanjiro)
4. **Training dynamics** — EMA decay (nezuko), extended training/schedule shape (alphonse)
5. **Loss weighting** — Per-channel pressure weight W_p sweep (fern)

## Critical Cross-Cutting Bug

`evaluate_split` masked-sum pattern produces `inf * 0 = NaN` whenever a sample has any non-finite ground-truth entry. Confirmed culprit: sample 20 of `test_geom_camber_cruise` has 761 inf entries. Frieren's #1106 carries a guard that zeroes the bad sample's `y` and drops its nodes from `mask` before loss/error accumulation. Once #1106 (rebased) merges, all subsequent runs will produce clean `test_avg/mae_surf_p` without per-PR firefighting.

## Potential Next Research Directions

Once current round results are in:

1. **Fourier-feature follow-ups** (gated on #1106 landing): frequency-count sweep `L ∈ {4, 6, 8, 10}`, learnable random-Fourier-feature projection (Tancik et al. 2020), Fourier expansion on `dsdf` (dims 4–11).
2. **Stack winning ideas** — combine Fourier(x,z) with a plausible second winner (e.g., FiLM conditioning, longer cosine schedule, larger n_hidden) once both are in to test orthogonality.
3. **Multi-layer Transolver** — test n_layers=2 with n_hidden=128 vs n_layers=1 with n_hidden=256 (width vs depth tradeoff); Fourier widening the basin may shift this trade-off.
4. **Adaptive surf_weight** — dynamic weighting that ramps up surf_weight late in training to focus later epochs on surface pressure accuracy.
5. **Domain-aware attention** — explicit domain label (single/raceCar-tandem/cruise-tandem) injected as conditioning. `val_geom_camber_rc` continues to dominate the error budget and is the obvious target.
6. **Separate decoders per output channel** — specialized decoders for (Ux, Uy, p) so pressure can use a richer head without bloating velocity cost.
7. **Graph-based local attention** — k-NN local attention over mesh nodes to capture boundary-layer gradients.
8. **Curriculum learning** — order samples by difficulty (Re magnitude or domain complexity) to improve OOD generalization.
9. **Physics-informed loss terms** — soft continuity equation residual as auxiliary loss.
10. **Stochastic depth / DropPath** — regularization for n_layers>1 configurations.
11. **Mixed precision training improvements** — fp16 vs bf16 comparison, gradient scaling.
12. **Test-time ensembling** — average predictions from EMA checkpoints at different epochs.
13. **Warm restarts with decay** — CosineAnnealingWarmRestarts with T_mult>1 (T_mult=2) so each restart is longer than the last.
14. **Lower EMA decay on fresh training** — EMA 0.992 as follow-up if extended training (alphonse) shows continued improvement.
15. **Bigger swings if a plateau forms** — Fourier Neural Operator / Geometric Galerkin Transformer / graph-based local attention over mesh nodes.
