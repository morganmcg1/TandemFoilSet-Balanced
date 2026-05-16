# SENPAI Research State

- **Updated:** 2026-05-16 03:20 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3465 merged):** `val_avg/mae_surf_p = 75.4040`, `test_avg/mae_surf_p = 65.8592`
  - Note: T_max=25 schedule alignment validated on the full merged stack (Cautious AdamW + bf16 + FiLM + surf-L1 + EMA + scale-inv + Bernoulli residual) at tip `f4ae741`. Arm B (T_max=25, eta_min_factor=0.10) beat prior best (86.09) by −12.42%, with all 8 val/test cells improving. Descent still active at −2.74/epoch at epoch 17 — model remains undercooked at wall-clock cutoff.
- **Cumulative round-5 improvement:** −39.15% val_avg (123.88 → 75.40) and −42.45% test_avg (114.37 → 65.86) vs pre-round-5 floor. **Eight compounding wins in sequence**.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval touching `test_geom_camber_cruise/000020.pt` returns NaN. All merged PRs include this fix.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA (#3281) + surface-pressure L1 aux (#3337) + FiLM per-block conditioning (#3265) + bf16 AMP (#3373) + Cautious AdamW (#3315) + Bernoulli pressure residual (#3466) + CosineAnnealingLR T_max=25 alignment (#3465) + NaN-safe evaluate_split**. Eight-step improvement trajectory in round 5: −1.3% (scale-inv) → −7.84% (EMA) → −6.41% (surf-L1) → −9.77% (FiLM) → −13.16% (bf16) → −12.31% (Cautious AdamW) → −4.70% (Bernoulli residual) → −12.42% (T_max schedule alignment). Cumulative −39.15% val.

Key observations from merged results:
- **T_max=25 alignment is the biggest single-loop win** — a pure schedule fix that unlocks the cosine decay to operate within the actual training window. LR at epoch 17 (cutoff) was 1.79e-4, still productive. With T_max=50, the cosine spent most of its decay budget *after* training ended.
- **Descent still active at cutoff** (−2.74/epoch at epoch 17). Best epoch = 17/50. The model is still in mid-descent. This is the key bottleneck: every experiment is hitting the wall-clock cap in the cold-start regime. The two fresh experiments (LR sweep, torch.compile) both directly target this.
- **single_in_dist is closing the gap** — val went from 122.19 (FiLM) → 109.91 (Cautious AdamW) → 102.04 (Bernoulli) → 84.88 (T_max). Still worst split at val=84.88 vs cruise at 57.65, but gap is narrowing significantly.
- **Cautious AdamW mask agreement is invariant across all merged mechanisms** (~0.62 flat, including at T_max=25). The schedule change did not interact with the optimizer gating.
- **Negative results from this loop (#3432 SEMA):** SEMA at freq={1,2} with EMA decay=0.999 causes 2–3 epoch resets (EMA window ~2.7 epochs), losing gradient progress. SEMA requires lower EMA decay or much longer training to work. Clear dead end for this run duration.

Strongest remaining axes (in priority order):

1. **More compute within wall-clock** (#3582 fern new): torch.compile() for 10–25% speedup → 2–4 more effective epochs → ~5–11 expected val improvement.
2. **Higher peak LR on aligned schedule** (#3581 thorfinn new): lr=7e-4 / lr=1e-3 with T_max=25 — descent active at cutoff, higher LR may accelerate it within the budget.
3. **Optimizer alternative head-to-head** (#3425 tanjiro): SF-AdamW must beat 75.40 after rebasing onto latest tip. Rebase comment posted. Decision: SF-AdamW (no cosine scheduler) may have a natural schedule alignment advantage.
4. **Cp normalization** (#3547 askeladd WIP): divide by p_B in addition to subtract — the natural physics extension of the Bernoulli win. Standard CFD non-dimensionalization. Baseline updated to 75.40.
5. **Spatial representation** (#3631 nezuko new): NeRF-style Fourier positional encoding on (x, y) and optionally dsdf before input MLP. Addresses OOD geometry weakness (#3519 showed cond-path Fourier hurts camber extrapolation; spatial-path is the correct target).
6. **EMA refinement** (#3545 alphonse WIP): EMA decay annealing — low decay early, high decay late, addresses cold-start.
7. **Test-time augmentation** (#3548 frieren WIP): AoA-jitter TTA — K-ensemble of inference with small AoA perturbations.
8. **Capacity revisit** (#3463 edward WIP, actively training): n_hidden=192/256 sweep, tractable with bf16 VRAM savings.

**Closed in this loop**: 
- #3432 (fern SEMA) — +22.8% to +33.4% regression; SEMA mechanism inverted under EMA decay=0.999 / short wall-clock horizon (copy = 2–3 epoch reset, not flat-region refinement).

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED | Surface-pressure L1 aux loss (w=1.0) |
| #3265 | fern | MERGED | FiLM per-block global conditioning |
| #3373 | edward | MERGED | bf16 mixed precision |
| #3315 | askeladd | MERGED | Cautious AdamW |
| #3466 | askeladd | MERGED | Bernoulli pressure residual — val_avg 86.09 |
| #3465 | thorfinn | MERGED (this loop) | T_max=25 schedule alignment — new best: val_avg 75.40 |
| #3347 | alphonse | CLOSED | Manifold mixup — mesh-correspondence problem |
| #3346 | thorfinn | CLOSED | Cosine T_max=15 + warmup + LR=7e-4 — clear regression (3 seeds) |
| #3374 | nezuko | CLOSED | Stochastic depth — 3-seed robust negative |
| #3422 | frieren | CLOSED | Huber loss for surf-pressure aux — both arms regressed +5-8% |
| #3433 | alphonse | CLOSED | Per-domain target normalization — both arms regressed +3-11% |
| #3432 | fern | CLOSED (this loop) | SEMA — both arms regressed +22-33%; EMA-lag reset mechanism |
| #3425 | tanjiro | WIP (rebase done, re-running on 75.40 baseline) | Schedule-Free AdamW — head-to-head vs Cautious AdamW |
| #3463 | edward | WIP (sent back this loop — strong n=192 win on old baseline, rebase + re-run on 75.40) | Capacity revisit: n_hidden=192 |
| #3519 | nezuko | CLOSED (this loop) | Fourier FiLM — +4.75% regression vs actual baseline; OOD geometry degraded |
| #3631 | nezuko | WIP (new this loop) | Fourier positional encoding on spatial coords (x,y,dsdf) — NeRF-style, targets OOD geometry |
| #3545 | alphonse | WIP | EMA decay annealing — 2 arms (linear5, cosine19) |
| #3547 | askeladd | WIP | Cp normalization — 2 arms (cp, halfcp) |
| #3548 | frieren | WIP | AoA-jitter TTA — 2 arms |
| #3581 | thorfinn | WIP (new this loop) | Peak LR sweep on T_max=25: lr=7e-4 / lr=1e-3 |
| #3582 | fern | WIP (new this loop) | torch.compile() for more effective epochs |

## Plateau watch

Round 5 now shows 8 compounding wins totaling −39.15% val_avg. No plateau signal. The val_single_in_dist split improved dramatically this loop (102.04 → 84.88, −16.8%), driven by the schedule alignment's unlock of more productive LR range. Model still undercooked at each wall-clock cutoff — throughput/speed improvements (torch.compile) and LR optimization (LR sweep) are the highest-leverage axes right now.

Note: WIPs (#3425, #3463, #3545, #3547, #3548) started before T_max alignment merged. Schedule alignment is additive to all of them *except* #3425 SF-AdamW (which removes CosineAnnealingLR entirely — SF-AdamW's internal schedule may provide equivalent or better alignment). Rebases should be clean for all non-optimizer WIPs.

## Potential next research directions (post-current batch)

1. **Proper chord-position Bernoulli** (askeladd #3466 follow-up): per-foil chord-boundary detection + chord-position-dependent Cp formula. Could unlock another −2% to −4%.
2. **Drop EMA, eval from SF-AdamW `optimizer.eval()` directly** (tanjiro #3425 suggested follow-up): two averaging mechanisms (SF-internal polynomial + external EMA-of-y) may compete. Clean ablation if SF-AdamW wins.
3. **Per-block mask-mean logging**: with FiLM modulating per-block, per-block cautious mask logging would reveal if gating fires more in FiLM-modulated layers.
4. **Mach-correction Bernoulli**: at high V_∞, p_B should include compressibility term. Small gain (~0.5%), but physics-correct.
5. **Lift/drag coefficient auxiliary head**: predict integrated lift and drag as side output with L1 loss against CFD-computed coefficients. Multi-task regularizer.
6. **Learned residual MLP head**: small MLP stacked on Transolver output to predict remaining error structure. Targets the still-large per-split MAE on single_in_dist.
7. **Cautious Lion + EMA**: Lion + cautious gating — signed updates + agreement filter. Defer until SF-AdamW vs Cautious head-to-head (#3425) is resolved.
8. **SEMA revisit with decay=0.99**: if EMA decay annealing (#3545) wins, a low-decay SEMA (decay=0.99, window≈100 steps, copy≈0.27-epoch spacing) avoids the 2–3 epoch reset problem.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale/domain), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (chord-Bernoulli, Cp, lift/drag aux), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
