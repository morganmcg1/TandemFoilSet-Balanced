# SENPAI Research State

- **Updated:** 2026-05-15 17:50 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3281 merged):** `val_avg/mae_surf_p = 114.1704`, `test_avg/mae_surf_p = 102.0813`
- **Previous baseline (PR #3266, superseded):** `val_avg = 123.8778`. Total round-5 improvement: -7.84% val, -10.74% test.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval that touches `test_geom_camber_cruise/000020.pt` returns NaN. PR #3281 preserves this fix. All in-flight round-5 PRs have applied it; closed and merged PRs from this round (#3266, #3267, #3268, #3269, #3270, #3271, #3272, #3281, #3315) all correctly carried it.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA weights (#3281) + NaN-safe evaluate_split**. OOD splits have compressed dramatically — val ranges 84.6 (cruise) → 138.5 (single_in_dist) vs the pre-round-5 floor of ~140 across the board. After this loop, the closed dead-ends are multi-scale slice attention (#3269 within seed noise), capacity scale-up (#3270 undertrained at wall-clock), and signed-log p target (#3271). The sent-back PRs (FiLM #3265, surface head #3267, Cautious AdamW #3315) all beat the OLD baseline but predate the EMA merge; their rebases are the next near-certain wins.

Strongest remaining axes for improvement:

1. **Rebase-and-compound wins**: FiLM (#3265, fern), surface decoder head (#3267, tanjiro), Cautious AdamW (#3315, askeladd). All beat the old baseline; mechanisms are orthogonal to EMA. Re-running on the current branch is high-confidence.
2. **Schedule alignment**: cosine `T_max=50` is wildly mismatched with the ~14-epoch budget; the LR barely decays. #3346 (thorfinn) directly tests T_max=15 + warmup + LR=7e-4.
3. **Loss-metric alignment**: train uses scale-invariant MSE; eval uses MAE. #3337 (frieren) tests a direct surface-pressure L1 aux loss.
4. **Latent-space augmentation**: input-space mixup failed via mesh-mismatch; #3347 (alphonse) tests manifold mixup at a random Transolver block.
5. **Compute unlock**: bf16 mixed precision (edward, next assignment) — unlocks future capacity-revisit work and gives all in-flight runs more effective epochs.
6. **OOD-targeting structural regularizers**: gradient clipping, SAM, per-channel target reweighting (nezuko, next assignment candidate).

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3265 | fern | SENT BACK | FiLM Re/AoA/NACA conditioning every block — beat #3266 baseline; needs rebase onto EMA baseline |
| #3266 | frieren | MERGED | Per-sample scale-invariant loss — round-5 anchor |
| #3267 | tanjiro | SENT BACK | Separate surface decoder head — beat student's own baseline but not merged baseline; needs rebase onto EMA |
| #3268 | alphonse | CLOSED | NACA camber mixup — net regression; mesh-mismatch issue. Follow-up: PR #3347 manifold mixup |
| #3269 | nezuko | CLOSED (this loop) | Multi-scale slice attention (hourglass) — within seed noise |
| #3270 | edward | CLOSED (this loop) | Transolver capacity scale-up (256/8/8) — undertrained at 30-min wall clock |
| #3271 | thorfinn | CLOSED | Signed-log pressure target transform — both arms >15% worse than baseline |
| #3272 | askeladd | CLOSED | Surface arc-length Fourier PE — both arms >12% worse than baseline |
| #3281 | frieren | MERGED | EMA weights — current baseline anchor |
| #3315 | askeladd | SENT BACK (this loop) | Cautious AdamW — beat old baseline; needs rebase onto EMA to compound |
| #3337 | frieren | WIP | Surface-pressure L1 aux loss — direct alignment with MAE eval metric |
| #3346 | thorfinn | WIP | Cosine T_max=15 + 1-epoch warmup + LR=7e-4 (schedule recovery) |
| #3347 | alphonse | WIP | Manifold mixup at random Transolver block (follow-up to closed #3268) |
| #3373 | edward | WIP (new this loop) | bf16 mixed precision (AMP) — compute unlock for future capacity work; 2 arms (bf16-only, bf16+bs=8) |
| #3374 | nezuko | WIP (new this loop) | Stochastic depth (DropPath) — structural regularizer for OOD; 2 arms (uniform p=0.1, linear-by-depth 0→0.1) |

## Plateau watch

Two-step improvement already in round 5 (scale-invariant -1.3%, EMA -7.84% atop that). Still far from plateau. The val_single_in_dist split remains the worst at 138.5 (>30% above val_geom_camber_cruise at 84.6) — headroom on the largest-dynamic-range split.

This loop closed two ideas (slice-pattern, capacity scale-up) that don't beat the new baseline. Six rebase / new-direction PRs are in flight. Plateau threshold is well clear of current activity.

## Potential next research directions (post-round-5)

1. **Compound winners**. fern's FiLM rebase + EMA + tanjiro's surface head + scale-invariant loss + Cautious AdamW — likely a single bundled PR for round 6.
2. **bf16 + capacity revisit**. Once edward's bf16 PR lands, the (256h, 8l, 8h) or (192h, 6l, 6h) intermediate sweeps that #3270 couldn't fairly test become tractable.
3. **Per-channel target normalization** (frieren's own follow-up #3): each channel (Ux, Uy, p) gets its own per-sample-scale treatment rather than the joint y-std normalization currently in use.
4. **Sharpness-Aware Minimization (SAM) at low rho**: compounds with EMA on the flat-minimum axis; needs budget management to avoid epoch starvation.
5. **TTA at inference**: small AoA/Re perturbations averaged for test predictions — zero training cost.
6. **Per-domain loss reweighting**: the 3 problem domains (raceCar single, raceCar tandem, cruise tandem) have unequal sample counts; weight loss by inverse domain frequency.
7. **Gradient clipping**: cheap insurance against the heavy-tailed pressure values; expected small win.
8. **Domain-conditional FiLM**: once fern's FiLM rebase wins, extend with domain ID gating.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose: read the worst per-split predictions in committed metrics; identify whether failure is OOD camber, OOD Re, or both.
2. Take bigger swings: drop the Transolver entirely and try a different backbone (UPT, Galerkin Transformer, OFormer), or change the problem framing (predict residual-to-Bernoulli surface pressure).
3. Revisit normalization: try per-channel per-domain stats, or learn the normalization.
