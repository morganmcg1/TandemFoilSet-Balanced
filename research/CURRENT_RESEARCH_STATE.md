# SENPAI Research State

- **Updated:** 2026-05-15 16:45 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3281 merged):** `val_avg/mae_surf_p = 114.1704`, `test_avg/mae_surf_p = 102.0813`
- **Previous baseline (PR #3266, now superseded):** `val_avg = 123.8778`, `test_avg = 114.3695`. Total round-5 improvement: -7.84% val, -10.74% test.

## Most recent research direction from human researcher team
(none — no Issues open or directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval that touches `test_geom_camber_cruise/000020.pt` returns NaN. PR #3281 preserves this fix. All in-flight round-5 PRs have been warned via comment to apply the same fix in their own `evaluate_split` before submitting; the three terminal results so far (frieren #3266 & #3281, thorfinn #3271, tanjiro #3267) all correctly applied it.

## Current research focus and themes

The merged baseline now stacks **scale-invariant loss (#3266) + EMA weights (#3281) + NaN-safe evaluate_split**. The OOD splits have compressed dramatically — val ranges from 84.6 (cruise) to 138.5 (single_in_dist) vs the pre-round-5 floor of ~140 across the board. The strongest remaining axes for improvement are:

1. **Schedule alignment**: cosine `T_max=50` is wildly mismatched with the ~14-epoch budget; the LR barely decays. Fixing this is almost-certain to help.
2. **Loss-metric alignment**: train uses scale-invariant MSE; eval uses surface-pressure MAE. Direct L1 on surface-pressure aux loss should help.
3. **OOD structural priors**: FiLM (#3265, sent back) already showed -8.7% on val_re_rand — once rebased onto the EMA baseline, the compound effect could be substantial.
4. **Specialization in heads**: tanjiro's surface head (#3267, sent back) gave -9.3% on val_single_in_dist within-PR. Rebase on EMA baseline could compound.
5. **Domain-aware capacity**: edward's scale-up (#3270) and nezuko's hourglass (#3269) tackle the question of whether the current 663K-param model is bottlenecked.
6. **Augmentation**: alphonse's NACA-camber mixup (#3268) targets the OOD-camber splits directly.
7. **Optimizer**: askeladd's Cautious AdamW (#3315) is a clean independent axis.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3265 | fern | SENT BACK | FiLM Re/AoA/NACA conditioning every block — beat #3266 baseline; needs rebase onto EMA baseline |
| #3267 | tanjiro | SENT BACK (today) | Separate surface decoder head — beat student's own baseline but not merged baseline; needs rebase onto EMA |
| #3268 | alphonse | WIP | NACA camber mixup augmentation |
| #3269 | nezuko | WIP | Multi-scale slice attention (hourglass) |
| #3270 | edward | WIP | Transolver capacity scale-up (256/8/8) |
| #3271 | thorfinn | CLOSED (today) | Signed-log pressure target transform — both arms >15% worse than baseline |
| #3272 | askeladd | CLOSED (last round) | Surface arc-length Fourier PE — both arms >12% worse than baseline |
| #3281 | frieren | MERGED (today) | EMA weights — current baseline anchor |
| #3315 | askeladd | WIP | Cautious AdamW one-line optimizer swap |
| #3337 | frieren | WIP (new) | Surface-pressure L1 aux loss — direct alignment with MAE eval metric |
| (pending) | thorfinn | TO BE ASSIGNED | Cosine T_max + warmup + LR=7e-4 (recovery experiment, blocked on GH rate limit) |

## Plateau watch

Two-step improvement already in round 5 (scale-invariant -1.3%, EMA -7.84% atop that). Still far from plateau. The val_single_in_dist split remains the worst at 138.5 (>30% above val_geom_camber_cruise at 84.6), suggesting headroom on the largest-dynamic-range split.

## Potential next research directions (post-merge of round-5 wins)

1. **Compound winners**. fern's FiLM rebase + EMA + tanjiro's surface head + scale-invariant loss — likely a single bundled PR for round 6.
2. **Per-channel target normalization** (frieren's own follow-up #3): each channel (Ux, Uy, p) gets its own per-sample-scale treatment rather than the joint y-std normalization currently in use.
3. **Sharpness-Aware Minimization (SAM)**: compounds with EMA on the flat-minimum axis; would need budget management to avoid epoch starvation.
4. **TTA at inference**: small AoA/Re perturbations averaged for test predictions — zero training cost.
5. **Per-domain loss reweighting**: the 3 problem domains (raceCar single, raceCar tandem, cruise tandem) have unequal sample counts; weight loss by inverse domain frequency.
6. **Gradient clipping**: cheap insurance against the heavy-tailed pressure values; expected small win.
7. **Domain-conditional FiLM**: once fern's FiLM rebase wins, extend with domain ID gating.

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
