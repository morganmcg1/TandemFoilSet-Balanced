# SENPAI Research State

- **Updated:** 2026-05-15 19:55 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3337 merged):** `val_avg/mae_surf_p = 106.8550`, `test_avg/mae_surf_p = 96.8671`
- **Previous baseline (PR #3281, superseded):** `val_avg = 114.1704`. Cumulative round-5 improvement: **-13.74% val, -15.30% test** vs the pre-round-5 floor.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval that touches `test_geom_camber_cruise/000020.pt` returns NaN. PR #3281 and PR #3337 preserve this fix.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA weights (#3281) + surface-pressure L1 aux loss (#3337) + NaN-safe evaluate_split**. The new surface-L1 aux gave −6.41% val_avg and is the largest single-PR round-5 improvement. Per-split val from #3337: single_in_dist=127.85, geom_camber_rc=121.11, geom_camber_cruise=81.39, re_rand=97.07.

Strongest remaining axes for improvement:

1. **Optimizer/schedule**: cosine `T_max=50` is still wildly mismatched with the ~14-epoch wall-clock budget. Three competing fixes are now in flight: schedule alignment (#3346 thorfinn, T_max=15+warmup), Cautious AdamW (#3315 askeladd, gated updates), and Schedule-Free AdamW (#3425 tanjiro, polynomial-averaged iterate — replaces both AdamW and the cosine schedule). At most one of these will be the new merged baseline, but the picture they produce together will pin down which optimizer/schedule axis matters most.
2. **Loss refinement on the merged L1**: #3422 (frieren) tests Huber as a smoother replacement for the new merged surface-pressure L1 aux. Direct follow-up to frieren's own oscillation observation in #3337.
3. **Architecture / conditioning**: FiLM rebase (#3265 fern), manifold mixup (#3347 alphonse), stochastic depth (#3374 nezuko) — all orthogonal regularizers/conditioners on top of merged baseline.
4. **Compute unlock**: bf16 mixed precision (#3373 edward) → roughly halves per-epoch wall-clock, unlocks future capacity-revisit work and gives all in-flight runs more effective epochs within the 30-min cap.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss — round-5 anchor |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED (this loop) | Surface-pressure L1 aux loss — new round-5 baseline, −6.41% val |
| #3267 | tanjiro | CLOSED (this loop) | Separate surface decoder head (rebased) — +5.83% val regression vs merged baseline; head mechanism subsumed by EMA + scale-inv loss |
| #3265 | fern | WIP (rebase) | FiLM Re/AoA/NACA conditioning every block — rebase onto EMA baseline |
| #3315 | askeladd | WIP (rebase) | Cautious AdamW — rebase onto EMA to compound |
| #3346 | thorfinn | WIP | Cosine T_max=15 + 1-epoch warmup + LR=7e-4 (schedule recovery) |
| #3347 | alphonse | WIP | Manifold mixup at random Transolver block |
| #3373 | edward | WIP | bf16 mixed precision (AMP) — 2 arms (bf16-only, bf16+bs=8) |
| #3374 | nezuko | WIP | Stochastic depth (DropPath) — 2 arms (uniform p=0.1, linear-by-depth 0→0.1) |
| #3422 | frieren | WIP (new this loop) | Huber loss replacement for surface-pressure aux — 2 arms (δ=1.0, δ=0.5) |
| #3425 | tanjiro | WIP (new this loop) | Schedule-Free AdamW (drop-in) — 2 arms (lr=5e-4, lr=7e-4) |

## Plateau watch

Three-step improvement now in round 5 (scale-invariant −1.3%, EMA −7.84%, surface-L1 −6.41% atop EMA). Cumulative −13.74% val and −15.30% test on the unmoved pre-round-5 baseline. Still far from plateau. The val_single_in_dist split remains the worst at 127.85 (>57% above val_geom_camber_cruise at 81.39) — headroom on the largest-dynamic-range split.

This loop merged one winner (#3337) and closed one clean regression (#3267). Seven in-flight PRs cover optimizer (3), loss (1), architecture (3), and compute (1) axes; the next loop's expected actions are confirming rebase wins (#3265, #3315) and reviewing new-direction terminal results.

## Potential next research directions (post-round-5)

1. **Compound winners**. fern's FiLM rebase + tanjiro's SF-AdamW + nezuko's stochastic depth on top of the merged surface-L1 baseline — likely a single bundled "round 6 baseline" PR.
2. **bf16 + capacity revisit**. Once edward's bf16 (#3373) lands, the (256h, 8l, 8h) or (192h, 6l, 6h) intermediate sweeps become tractable at the 30-min wall-clock budget.
3. **SEMA (Switch EMA)** — feedback the EMA weights into the trainable model at epoch boundaries; compounds with the existing EMA on a different time-scale. RANK #2 in current research agenda.
4. **Per-Domain Target Normalization** — separate per-sample y-std normalization within each of the 3 source domains (raceCar single, raceCar tandem, cruise tandem); removes the residual scale mismatch that the global per-sample-scale loss leaves on the table. RANK #3.
5. **Per-channel target normalization** (frieren's own follow-up): each channel (Ux, Uy, p) gets its own per-sample-scale treatment rather than the joint y-std normalization.
6. **Bernoulli pressure residual** — predict `p − p_bernoulli(Re, AoA)` so the network only learns the deviation from the physics prediction; reduces dynamic-range problem on single_in_dist.
7. **F-SAM (Friendly Sharpness-Aware Minimization)** at low ρ — compounds with EMA on the flat-minimum axis; needs careful budget management.
8. **TTA at inference**: small AoA/Re perturbations averaged for test predictions — zero training cost.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose: read the worst per-split predictions in committed metrics; identify whether failure is OOD camber, OOD Re, or both.
2. Take bigger swings: drop the Transolver entirely and try a different backbone (UPT, Galerkin Transformer, OFormer), or change the problem framing (predict residual-to-Bernoulli surface pressure).
3. Revisit normalization: try per-channel per-domain stats, or learn the normalization.
