# SENPAI Research State

- **Updated:** 2026-05-15 13:30 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)

## Most recent research direction from human researcher team
(none — no Issues open or directed at this launch as of this update)

## Current research focus and themes

Clean-slate round 5 on the Charlie 24h pai2i track. Baseline is the default Transolver (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, AdamW lr=5e-4 wd=1e-4, batch=4, surf_weight=10, CosineAnnealing, 50 epochs). Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across 4 val splits) and the test-time deciding metric is `test_avg/mae_surf_p`.

This round attacks the problem from **eight orthogonal angles**, biased toward bold structural changes rather than further hyperparameter tuning, with each hypothesis targeting either (a) the primary surf-pressure objective, (b) the unseen-camber generalization splits, or (c) the cross-Re generalization split:

| PR | Student | Hypothesis | Predicted δ | Targets |
|----|---------|-----------|-------------|---------|
| #3265 | fern | FiLM Re/AoA/NACA conditioning every block | −8 to −15% | Re-rand + unseen-camber |
| #3266 | frieren | Per-sample scale-invariant loss | −6 to −14% | Re-rand |
| #3267 | tanjiro | Separate surface decoder head | −5 to −12% | All splits, surf_p focus |
| #3268 | alphonse | NACA camber mixup augmentation | −5 to −12% | Unseen-camber |
| #3269 | nezuko | Multi-scale slice attention (hourglass) | −6 to −12% | All splits |
| #3270 | edward | Transolver capacity scale-up (256/8/8) | −5 to −10% | All splits |
| #3271 | thorfinn | Signed-log pressure target transform | −4 to −10% | Re-rand |
| #3272 | askeladd | Surface arc-length Fourier PE | −4 to −9% | Surf_p focus |

Full ideas list with motivations and references: `research/RESEARCH_IDEAS_2026-05-15_12:43.md`.

## Potential next research directions (post-round-5)

If results compound, the next round should explore:

1. **Compound winners** — stack the best 2–3 wins from this round (e.g. FiLM + EMA + larger model) into a single PR to test orthogonality.
2. **Optimizer changes** — Lion / C-AdamW / Schedule-Free AdamW once architecture is settled. Lowest risk additions but largest deltas only kick in after structure is right.
3. **Warmup + extended cosine schedule** + gradient clipping — polishing move, expected −2 to −5% on top of any architectural win.
4. **EMA / SWA for checkpoint selection** — near-free win, expected −3 to −7% especially on OOD splits.
5. **Huber loss + higher surf_weight** — direct train-eval metric alignment, expected −4 to −8%.
6. **Output reparameterization** — per-channel separate heads, or predict residual-from-analytic-Bernoulli surface pressure.
7. **Domain-conditional FiLM** — extend FiLM to also condition on (raceCar single / raceCar tandem / cruise tandem) domain ID derived from feature gating.
8. **Self-distillation / model soups** — final round polish if compute permits.

## Plateau watch
This is round 5 of the charlie-pai2i series. If <2 of the 8 PRs beat baseline by ≥3% in this round, treat as plateau and escalate to first-principles re-examination: read worst per-split predictions, attempt physics-loss / Bernoulli residual modeling, or revisit data normalization choices.
