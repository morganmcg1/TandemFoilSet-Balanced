# SENPAI Research State

- **Updated:** 2026-05-15 14:35 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3266 merged):** `val_avg/mae_surf_p = 123.8778`, `test_avg/mae_surf_p = 114.3695`

## Most recent research direction from human researcher team
(none — no Issues open or directed at this launch)

## Critical operational fix in baseline

PR #3266 also propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split`. Without it, any test eval that touches `test_geom_camber_cruise/000020.pt` returns NaN. All seven in-flight round-5 PRs have been warned via comment to apply the same fix in their own `evaluate_split` before submitting.

## Current research focus and themes

Round 5 has shifted from clean-slate to single-anchor-baseline. The remaining seven dispatched PRs are testing orthogonal structural/loss hypotheses on top of vanilla Transolver; they will be re-anchored against PR #3266's numbers (val 123.88 / test 114.37) at review time, with the caveat that they did **not** include the scale-invariant loss change. That means most of them are isolating their hypothesis vs vanilla MSE, while #3266 isolates scale-invariant vs vanilla MSE — useful for attribution, but it means clean compounding will only kick in from the next round onwards.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3265 | fern | WIP | FiLM Re/AoA/NACA conditioning every block |
| #3267 | tanjiro | WIP | Separate surface decoder head |
| #3268 | alphonse | WIP | NACA camber mixup augmentation |
| #3269 | nezuko | WIP | Multi-scale slice attention (hourglass) |
| #3270 | edward | WIP | Transolver capacity scale-up (256/8/8) |
| #3271 | thorfinn | WIP | Signed-log pressure target transform |
| #3272 | askeladd | WIP | Surface arc-length Fourier PE |
| #3281 | frieren | WIP | EMA weights for checkpoint + test eval (stacked on #3266) |

frieren's #3281 is the only PR that builds on top of the merged baseline. Once the others come in, the natural next move is to stack the strongest winner with EMA + the scale-invariant loss for a compounding round-6 push.

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
