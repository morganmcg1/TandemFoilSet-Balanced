# SENPAI Research State

- **Updated:** 2026-05-15 20:35 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3265 merged):** `val_avg/mae_surf_p = 103.0171`, `test_avg/mae_surf_p = 92.1617`
  - Note: FiLM run validated on pre-surf-L1 code (#3281 baseline); merged code includes #3337 surf-L1 + FiLM. Expected compound val_avg ~97–100. Askeladd's next rebase (#3315) will establish the confirmed compound metric.
- **Cumulative round-5 improvement:** −16.9% val_avg (123.88 → 103.02) and −19.4% test_avg (114.37 → 92.16) vs pre-round-5 floor.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval touching `test_geom_camber_cruise/000020.pt` returns NaN. All merged PRs include this fix.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA weights (#3281) + surface-pressure L1 aux (#3337) + FiLM per-block conditioning (#3265) + NaN-safe evaluate_split**. Four-step improvement trajectory in round 5: −1.3% (scale-inv) → −7.84% (EMA) → −6.41% (surf-L1) → −9.77% (FiLM). Cumulative −16.9% val.

Key observations from merged results:
- FiLM largest win was `cruise` (−22%) and `re_rand` (−24%) — regime-conditioning axes. `single_in_dist` still the weakest split at 122.19 after FiLM.
- Surface-L1 aux is loss-metric alignment; FiLM is architecture conditioning; EMA is iterate averaging. All three mechanisms are orthogonal. Three-way compound is untested but theoretically clean.
- Manifold mixup (alphonse #3347) closed — mesh-correspondence problem is fundamental for variable-mesh CFD. Entire input-mixup/latent-mixup family is closed for now.
- Cautious AdamW (#3315 askeladd) is proven strong (−9.77% standalone on #3281) but needs one more rebase to include both #3337 and FiLM (#3265). Third-and-final rebase expected soon.

Strongest remaining axes:

1. **Optimizer + schedule**: Cautious AdamW (#3315 askeladd, rebase), Schedule-Free AdamW (#3425 tanjiro), Schedule alignment (#3346 thorfinn). All targeting the high-LR regime at epoch-14 wall-clock cap.
2. **EMA refinement**: SEMA (#3432 fern) — copy EMA weights back each epoch. Expected −1 to −3% at zero cost.
3. **Domain-specific normalization**: per-domain target norm (#3433 alphonse) — directly targets single_in_dist gap.
4. **Loss refinement**: Huber aux (#3422 frieren) — smoother gradient near zero for surface-pressure L1.
5. **Regularization**: Stochastic depth (#3374 nezuko), bf16 compute unlock (#3373 edward).

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED | Surface-pressure L1 aux loss (w=1.0) |
| #3265 | fern | MERGED (this loop) | FiLM per-block global conditioning — new best: val_avg 103.02 |
| #3347 | alphonse | CLOSED (this loop) | Manifold mixup — regression; mesh-correspondence fundamental problem |
| #3315 | askeladd | SENT BACK (this loop, 2nd rebase) | Cautious AdamW — needs rebase onto #3337+FiLM; predicted val_avg 96-100 |
| #3346 | thorfinn | WIP | Cosine T_max=15 + warmup + LR=7e-4 (schedule fix) |
| #3347 | alphonse | WIP | — closed |
| #3373 | edward | WIP | bf16 mixed precision (AMP) — 2 arms |
| #3374 | nezuko | WIP | Stochastic depth (DropPath) — 2 arms |
| #3422 | frieren | WIP | Huber loss for surf-pressure aux — 2 arms (δ=1.0, δ=0.5) |
| #3425 | tanjiro | WIP | Schedule-Free AdamW — 2 arms (lr=5e-4, lr=7e-4) |
| #3432 | fern | WIP (new this loop) | SEMA: copy EMA weights back each epoch — 2 arms (freq=1, freq=2) |
| #3433 | alphonse | WIP (new this loop) | Per-domain target normalization — 2 arms (hard labels, per-channel) |

## Plateau watch

Round 5 now shows 4 compounding wins totaling −16.9% val_avg. The val_single_in_dist split is the persistent outlier at 122.19 — #3433 directly targets this. Cautious AdamW (#3315 rebase) is the highest-probability near-term winner and could push val_avg into the mid-90s range. Schedule-Free AdamW (#3425) is independent and could also be decisive.

## Potential next research directions (post-current batch)

1. **Compound all winners**: FiLM + surf-L1 + Cautious AdamW + EMA + scale-inv is the natural round-6 bundle if askeladd's rebase confirms the expected ~97-100 val_avg.
2. **SEMA + SEMA-freq sweep**: if SEMA wins, sweep freq∈{1,2,3} and warmup∈{0,5} to find the optimal per-epoch copy schedule.
3. **bf16 + capacity revisit**: once edward's bf16 lands, (256h, 8l, 8h) or (192h, 6l, 6h) configurations become feasible.
4. **Bernoulli pressure residual prediction** (Idea #4): predict `p − p_Bernoulli(Re, AoA)` residual instead of raw pressure. High-risk, high-reward — reduces the dynamic range problem on single_in_dist.
5. **Domain-conditional FiLM**: extend the merged FiLM with domain-specific condition MLPs that also take domain ID; extends the FiLM hypothesis beyond shared-MLP.
6. **SEMA + Cautious AdamW stack**: if both win independently, test whether the flat-minimum SEMA starting point compounds with the noise-gated Cautious updates.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (Bernoulli), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
