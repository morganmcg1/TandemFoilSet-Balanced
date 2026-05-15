# SENPAI Research State

- **Updated:** 2026-05-15 23:30 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3315 merged):** `val_avg/mae_surf_p = 90.3428`, `test_avg/mae_surf_p = 80.1674`
  - Note: Cautious AdamW run validated on pre-bf16 code (tip `b5760af` with FiLM+surf-L1+EMA+scale-inv). bf16 (#3373) is now also merged; full compound (bf16 + Cautious AdamW + full stack) expected in low-80s val_avg. Next student results will establish confirmed compound metric.
- **Cumulative round-5 improvement:** −27.06% val_avg (123.88 → 90.34) and −29.91% test_avg (114.37 → 80.17) vs pre-round-5 floor. Five compounding wins in sequence.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval touching `test_geom_camber_cruise/000020.pt` returns NaN. All merged PRs include this fix.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA (#3281) + surface-pressure L1 aux (#3337) + FiLM per-block conditioning (#3265) + bf16 AMP (#3373) + Cautious AdamW (#3315) + NaN-safe evaluate_split**. Six-step improvement trajectory in round 5: −1.3% (scale-inv) → −7.84% (EMA) → −6.41% (surf-L1) → −9.77% (FiLM) → −13.16% (bf16, on #3281 baseline) → −12.31% (Cautious AdamW, on FiLM baseline). Cumulative −27.1% val.

Key observations from merged results:
- **Cautious AdamW mask agreement is invariant across all merged mechanisms** (~0.62 flat across 13 epochs, identical in standalone / +EMA / +EMA+surf-L1+FiLM runs). Direct empirical evidence that the six mechanisms operate on disjoint state.
- **Uniform per-split wins from Cautious AdamW on full stack** (10–15% across all 8 val/test cells) — qualitatively different from standalone cautious run where in-dist splits regressed. EMA+FiLM together stabilize the iterate enough for cautious gating to be a net positive everywhere.
- **Steep epoch-13 descent** (94.7 → 90.3 in final epoch) and flat mask curve signal training is still in cold-start; bf16's 5-6 extra epochs should compound directly.
- **single_in_dist is still the worst split** at val=109.91 after Cautious AdamW — large improvement from 122.19 (FiLM) to 109.91, but still 44 points worse than cruise (65.69). Per-domain norm (#3433) and Bernoulli residual (#3466) both target this.
- **Schedule mismatch confirmed** (thorfinn #3346 negative result, 3 seeds): warmup costs ~7% of budget, and the "always high" T_max=50 cosine truncated at epoch 14-19 is actually a feature not a bug — matches the undercooked regime. The T_max alignment (match bf16 budget → T_max=19, no warmup) is the cleanest remaining schedule fix.

Strongest remaining axes (in priority order):

1. **bf16 + Cautious AdamW compound**: the next WIP results will establish the actual measured compound baseline. All in-flight students will need to rebase onto the new merged code once their experiments complete.
2. **Schedule alignment** (#3465 thorfinn): T_max=19 matching bf16 epoch count — the most defensible schedule fix based on #3346 negative result.
3. **Capacity revisit** (#3463 edward): n_hidden=192/256 sweep, now tractable with 33 GB VRAM from bf16.
4. **Bernoulli pressure residual** (#3466 askeladd): high-novelty physics-informed target reformulation — predict viscous residual only, removing analytic dynamic range from the prediction target.
5. **EMA refinement**: SEMA (#3432 fern) — copy EMA weights back each epoch.
6. **Domain-specific normalization**: per-domain target norm (#3433 alphonse) — directly targets single_in_dist gap.
7. **Conditioning richness** (#3519 nezuko new this loop): Fourier-embedded FiLM — Tancik-style multi-frequency sin/cos encoding of the 11-dim flow vector before FiLM MLP; targets single_in_dist via richer angular resolution on AoA.
8. **Loss refinement**: Huber aux (#3422 frieren) — smoother gradient near zero for surface-pressure L1.
9. **Alternate optimizer head-to-head**: Schedule-Free AdamW (#3425 tanjiro) — needs rebase before head-to-head can complete; standalone 87.24 vs current best 90.34 says it's plausible.

**Closed in this loop**: #3374 stochastic depth — 3-seed robust negative (both arms regressed +7-14%); regularization too strong for 14-epoch cold-start regime.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED | Surface-pressure L1 aux loss (w=1.0) |
| #3265 | fern | MERGED | FiLM per-block global conditioning |
| #3373 | edward | MERGED | bf16 mixed precision — VRAM 42→33 GB, 14→19 effective epochs |
| #3315 | askeladd | MERGED (this loop) | Cautious AdamW — new best: val_avg 90.34 |
| #3347 | alphonse | CLOSED | Manifold mixup — mesh-correspondence problem |
| #3346 | thorfinn | CLOSED (this loop) | Cosine T_max=15 + warmup + LR=7e-4 — clear regression (3 seeds) |
| #3374 | nezuko | CLOSED (this loop) | Stochastic depth — 3-seed robust negative; both arms regressed +7-14% vs old #3281 baseline |
| #3422 | frieren | WIP | Huber loss for surf-pressure aux — 2 arms (δ=1.0, δ=0.5) |
| #3425 | tanjiro | WIP (rebase pending) | Schedule-Free AdamW — head-to-head vs Cautious AdamW after rebase |
| #3432 | fern | WIP | SEMA: copy EMA weights back each epoch — 2 arms (freq=1, freq=2) |
| #3433 | alphonse | WIP | Per-domain target normalization — 2 arms (hard labels, per-channel) |
| #3463 | edward | WIP | Capacity revisit with bf16: n_hidden=192, n_hidden=256 |
| #3465 | thorfinn | WIP | T_max alignment: T_max=19 no-warmup, T_max=25 — schedule fix |
| #3466 | askeladd | WIP | Bernoulli pressure residual — physics-informed target reformulation |
| #3519 | nezuko | WIP (new this loop) | Fourier-embedded FiLM conditioning — 2 arms (k=4 σ=1, k=8 σ=2) |

## Plateau watch

Round 5 now shows 6 compounding wins totaling −27.1% val_avg. No plateau signal. The val_single_in_dist split improved from 122.19 (post-FiLM) to 109.91 (post-Cautious-AdamW) — still the worst split, but no longer an outlier in % gap. Four WIPs (#3433, #3466, #3463, #3519) directly target this split. Schedule-Free AdamW (#3425) is the most similar to Cautious AdamW (alternate optimizer) and should be considered a confirm-or-supersede experiment after rebase.

Note: WIPs started pre-bf16 / pre-Cautious-AdamW (#3422, #3425, #3432, #3433) still need rebases when their terminal results come in. Cautious AdamW's mask invariance (flat at 0.62 regardless of merged mechanisms) suggests the rebase will compound cleanly for optimizer-orthogonal ideas (SEMA, per-domain norm, Huber). Schedule-Free AdamW (#3425) conflicts with Cautious AdamW at the optimizer level — if SF-AdamW wins on rebase, it should replace Cautious AdamW in a fresh comparison before merging.

**Operational note (this loop):** The three stale_wip flags on #3433/#3432/#3422 traced to a github API rate-limit storm at 22:10-23:20 UTC that interrupted the students' result-posting cycle after training completed (GPU dropped from 99-100% to 0% across all three pods coincident with the rate-limit timestamp). All three pods recovered from rate limit by iteration ~100 and are now back in heartbeat mode — they should detect their completed-but-unposted training and post results on the next iteration. No advisor action needed; will re-check at next loop.

## Potential next research directions (post-current batch)

1. **Compound all winners**: bf16 + FiLM + surf-L1 + Cautious AdamW + EMA + scale-inv → low-80s val_avg expected from first measured compound run.
2. **T_max alignment** (#3465): the schedule has never actually annealed within the wall-clock budget; matching T_max=19 to the bf16 epoch count is the most defensible schedule fix and could give another −2% to −4%.
3. **Capacity revisit** (#3463): n_hidden=192/256 — now the first fair test of model capacity since bf16 unlocked 9 GB VRAM headroom. If capacity wins, the next natural question is whether to push n_hidden=384 with gradient checkpointing.
4. **Bernoulli pressure residual** (#3466): if the viscous-residual reformulation works, it opens a physics-informed target family: Cp coefficient prediction, lift/drag residual, boundary-layer displacement thickness correction.
5. **Fourier-embedded FiLM** (#3519, this loop): if multi-frequency sin/cos conditioning helps, next steps are learnable Random Fourier Features (RFF) and spatial Fourier embedding of point coordinates.
6. **Cautious Lion + EMA + FiLM + surf-L1**: askeladd's suggestion. Liang et al. show the agreement trick generalizes to Lion. With the compound mechanism confirmed, Lion's signed-gradient updates + cautious masking would test whether optimizer-internals contribution can be further improved. (Conflicts with current Cautious AdamW; defer until SF-AdamW vs Cautious head-to-head is resolved.)
7. **Per-block mask-mean logging**: with FiLM also modulating per-block, per-block cautious mask logging would reveal whether gating fires more in FiLM-modulated layers — useful for architecture-mask co-design.
8. **torch.compile() on top of bf16** (edward's #3373 suggestion #3): pure speed-up, may unlock 22+ effective epochs. Risk: compatibility with CautiousAdamW custom optimizer step.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale/domain), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (Bernoulli, assigned), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
