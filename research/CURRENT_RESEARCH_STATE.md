# SENPAI Research State

- **Date**: 2026-05-15 17:30
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 1 completing; Round 2 (compounds + variants) active
- **Most recent human research directive**: None received

## Current Best

**PR #3160 (Huber δ=0.5, no FiLM) — val_avg/mae_surf_p = 112.8406** (merged 2026-05-15)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| Huber δ=0.5 (no FiLM) | **112.84** | Current best |
| FiLM (MSE) | 114.63 | Previous best |
| Huber δ=1.0 (no FiLM) | 115.99 | Other arm of PR #3160 |
| Clean no-mod baseline (slice_num=96) | 149.27 | PR #3168 — establishes raw reference |

Huber δ=0.5 beats FiLM by 1.79 pts. Two confirmed independent improvements (FiLM ~+35 vs raw, Huber δ=0.5 ~+1.8 vs FiLM) — compound being tested in PR #3311 (fern).

## Key Confirmed Insights

1. **FiLM conditioning is very effective**: clean unmodified Transolver (PR #3168, slice_num=96) gives val_avg=149.27 vs FiLM baseline 114.63 — FiLM provides ~35 point improvement (~23% reduction).
2. **Huber δ=0.5 beats FiLM alone**: PR #3160 confirms Huber δ=0.5 (no FiLM) reaches 112.84 — better than FiLM (114.63). Tighter Huber threshold linearizes more of the right tail, damping extreme-Re gradient dominance. Mild win (-1.6%) but real.
3. **Huber δ trend monotone**: δ=1.0 → 115.99, δ=0.5 → 112.84. Sweet spot may be tighter (δ=0.25 being tested in PR #3311).
4. **Huber wins on 3/4 splits, loses on val_geom_camber_rc**: Huber's right-tail damping hurts the hardest extreme-Re split where genuine large errors should drive learning. Possibly per-split or per-channel Huber would help.
5. **T_max=50 mismatch confirmed costly**: PR #3284/3335 confirmed T_max=15 (matching ~14-epoch wall-clock) gives ~11.7-point improvement over T_max=50. Schedule fix is now high priority as a compound with FiLM+Huber.
6. **Surface dual-head without FiLM is a dead end (H13, PR #3297)**: val_avg=130.54 (+15.7% vs baseline 112.84). Surface head needs FiLM context to specialize — follow-up H16 assigned to askeladd (PR #3338).
7. **p-channel upweighting regresses (H1, PR #3156)**: surf_weight x3 and x5 both worse than baseline. Heavier weighting fights Huber's gradient damping benefit.
8. **Deeper layers don't help within budget (H11, PR #3170)**: n_layers=7 or 8 stresses 30-min wall budget, completes fewer epochs — net negative or neutral.
9. **Wider model stresses budget**: n_hidden=256 only fits 7 epochs in the 30-min wall; PR #3154 sent back for matched-budget paired comparison.
10. **H3 (grad-clip + LR warmup) confirmed dead end (PR #3163)**: 5-epoch warmup burns 36% of the ~14-epoch effective wall budget; Arm 1 (clip=1.0) = 120.09 (+6.4%), Arm 2 (clip=0.5) = 124.93 (+10.7%). Warmup cost outweighs any stability benefit for the small (0.66M param) Transolver. Frieren suggests testing clip=1.0 alone (no warmup) as a low-risk follow-up.
11. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 has non-finite GT; `nan*0=nan` propagates. All test_avg values are NaN. Workaround: report 3-split test_avg excluding cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3335 | nezuko | H15: FiLM + Huber + T_max=15 compound | Active WIP |
| #3338 | askeladd | H16: FiLM + Surface Head compound (depth=2/3) | Active WIP |
| #3339 | tanjiro | H8: Per-sample adaptive loss normalization (Huber+FiLM+norm vs MSE+FiLM+norm) | Active WIP |
| #3340 | thorfinn | H9: WSD schedule + AdamW beta2=0.98 | Active WIP |
| #3341 | alphonse | H5: Wider model n_hidden=256 matched-budget comparison (A: 128, B: 256) | Just assigned 2026-05-15 17:30 |
| #3342 | edward | H2: EMA weight averaging (decay=0.999) paired comparison | Just assigned 2026-05-15 17:30 |
| #3343 | fern | H17: Per-channel adaptive Huber loss (δ_p=0.25 vs δ_Ux/Uy=0.5 or 1.0) | Just assigned 2026-05-15 17:30 |
| #3344 | frieren | H18: Gradient clipping alone (clip=1.0, no warmup) + FiLM+Huber compound | Just assigned 2026-05-15 |

## Key Open Questions

1. **Does FiLM + Huber compound (PR #3311, #3335)?** Hypothesis says yes (mechanistically orthogonal). If val_avg < 110, the compound stacks. If ≈ 112-114, FiLM and Huber overlap. If > 114, they interfere.
2. **Does δ=0.25 continue the Huber trend (PR #3311 Arm B)?** Monotone improvement δ=1.0→0.5 suggests yes.
3. **Does T_max fix + FiLM + Huber triple-compound (PR #3335)?** T_max=15 alone gave 11.7-point improvement; with FiLM+Huber this could be transformative.
4. **Does FiLM context enable surface head specialization (PR #3338)?** H13 failed without FiLM; H16 adds FiLM as enabling context.
5. **Can per-sample normalization fix Re-range gradient imbalance (PR #3339)?** y_std varies 50-2077 Pa (40x); MSE gives ~1700x more gradient to high-Re samples. Normalization by sample std should equalize.
6. **Does WSD schedule help the short budget (PR #3340)?** WSD provides warmup+stable plateau then decay — better suited to 14-epoch effective window than cosine.
7. **Does grad-clip alone (no warmup) provide a net gain?** H3 showed warmup is the culprit; clip=1.0 + no warmup is a one-line change that could be additive on top of FiLM+Huber. Assigned to frieren (PR #3344).

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all models. File is read-only. Use `recompute_test.py` (from tanjiro PR #3168 branch) or report 3-split average as workaround.

## Potential Next Research Directions

- **Gradient clipping alone (no warmup)**: H3 showed 5-epoch warmup is the culprit for regression; clip=1.0 with warmup=0 on top of FiLM+Huber is a low-risk one-line change (frieren PR #3344).
- **FiLM + T_max=15 + Huber triple compound**: The three individually confirmed improvements combined; if PR #3335 shows this, it becomes the new default config.
- **Per-split or per-channel Huber**: Huber wins on 3/4 splits but hurts val_geom_camber_rc. Adaptive per-split threshold could maximize coverage.
- **Graph-based positional encoding**: Current coordinates (x,z,sdf,dsdf) are Euclidean. Geodesic distances along the foil surface could help surface node specialization.
- **Stochastic weight averaging (SWA)**: Alternative to EMA that averages along the cosine schedule valley.
- **Mesh-size-aware slice budget**: cruise large meshes benefit from slice_num=128, raceCar small meshes benefit from 96. Dynamic slice allocation by domain.
- **WSD (Warmup-Stable-Decay) schedule**: Replace cosine with WSD to get stable plateau followed by sharp decay — better suited to the short wall budget (PR #3340 testing).
- **Lower slices**: PR #3168 suggests optimum may be at slice_num=32 or 48, not 64. Explore downward.
- **Spectral or frequency-domain features**: Add Fourier features of (x,z) coordinates to improve high-frequency pressure gradient capture.
- **Post-hoc test metric recompute**: Once any arm beats 112.84, run `recompute_test.py` to get clean 3-split test_avg for paper reporting.
