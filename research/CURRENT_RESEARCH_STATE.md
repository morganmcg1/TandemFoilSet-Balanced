# SENPAI Research State

- **Updated:** 2026-05-15 15:55 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3266 merged):** `val_avg/mae_surf_p = 123.8778`, `test_avg/mae_surf_p = 114.3695`

## Most recent research direction from human researcher team
(none — no Issues open or directed at this launch)

## Critical operational fix in baseline

PR #3266 also propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split`. Without it, any test eval that touches `test_geom_camber_cruise/000020.pt` returns NaN. All in-flight round-5 PRs have been warned via comment to apply the same fix in their own `evaluate_split` before submitting.

## Current research focus and themes

Round 5 has shifted from clean-slate to single-anchor-baseline. Most dispatched PRs were cut before PR #3266 landed and are isolating their hypothesis vs vanilla MSE, while #3266 isolates scale-invariant vs vanilla MSE — useful for attribution, but it means clean compounding will only kick in once the winners are rebased onto the merged baseline.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3265 | fern | SENT BACK | FiLM Re/AoA/NACA conditioning every block — beat baseline (val 122.27 / test 112.17, **-8.7% on val_re_rand**) but had merge conflicts with the scale-invariant loss landing; sent back for rebase + re-run, expected to compound cleanly |
| #3267 | tanjiro | WIP | Separate surface decoder head |
| #3268 | alphonse | WIP | NACA camber mixup augmentation |
| #3269 | nezuko | WIP | Multi-scale slice attention (hourglass) |
| #3270 | edward | WIP | Transolver capacity scale-up (256/8/8) |
| #3271 | thorfinn | WIP | Signed-log pressure target transform |
| #3272 | askeladd | CLOSED | Surface arc-length Fourier PE — both arms >12% worse than baseline, direction not productive |
| #3281 | frieren | WIP | EMA weights for checkpoint + test eval (stacked on #3266) |
| #3315 | askeladd | WIP (new) | Cautious AdamW one-line optimizer swap (recovery experiment after #3272) |

frieren's #3281 and askeladd's #3315 are the two PRs that build on top of the merged baseline. The compounding plan for round-6: stack the strongest rebased structural winner (likely fern's FiLM if rebase confirms the OOD-Re win) with EMA + the scale-invariant loss + any optimizer win from #3315 on a single branch.

## Plateau watch

If <3 of the remaining 7 in-flight PRs beat `val_avg/mae_surf_p = 123.88` by ≥3%, treat as a soft plateau and escalate:
1. Diagnose: read the worst per-split predictions in committed metrics; identify whether failure is OOD camber, OOD Re, or both.
2. Take bigger swings: drop the Transolver entirely and try a different backbone (UPT, Galerkin Transformer, OFormer), or change the problem framing (predict residual-to-Bernoulli surface pressure).
3. Revisit normalization: try per-channel per-domain stats, or learn the normalization.

## Potential next research directions (post-round-5)

1. **Compound winners.** Stack the round-5 winner(s) with frieren's EMA result and consider a follow-on PR combining all three on a single branch.
2. **Schedule fix.** Set `cosine T_max = epochs_actually_completed` (~14) so LR fully decays in the 30-min wall clock. Frieren's analysis suggests ~5-point improvement at zero risk.
3. **Optimizer.** Lion / Schedule-Free AdamW / Cautious-AdamW once architecture is set.
4. **Warmup + LR scaling.** Add a 1-2 epoch linear warmup, raise peak LR to 7e-4 or 1e-3, jointly with a larger model.
5. **Huber loss + higher surf_weight.** Direct train-eval metric alignment, expected -4 to -8%.
6. **Domain-conditional FiLM.** If fern's FiLM wins, extend it to also condition on the (raceCar single / raceCar tandem / cruise tandem) domain ID derived from feature gating.
7. **Per-channel target normalization** so Ux, Uy, p each get equal per-sample-scale treatment (frieren's own follow-up #3).
8. **Test-time augmentation + EMA + model soups** for late-round polish.
