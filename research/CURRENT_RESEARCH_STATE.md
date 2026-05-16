# SENPAI Research State

- **Updated:** 2026-05-16 00:40 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3466 merged):** `val_avg/mae_surf_p = 86.0948`, `test_avg/mae_surf_p = 77.5066`
  - Note: Bernoulli residual validated on the full merged stack (Cautious AdamW + bf16 + FiLM + surf-L1 + EMA + scale-inv) at tip `3a3104a`. First measured head-to-head against full-merged-stack — confirms compound stack lands ~90 val (slightly worse than pre-bf16 Cautious AdamW alone, since the Cautious AdamW number was on a different baseline), and Bernoulli residual gives another −4.7% on top.
- **Cumulative round-5 improvement:** −30.49% val_avg (123.88 → 86.09) and −32.22% test_avg (114.37 → 77.51) vs pre-round-5 floor. **Seven compounding wins in sequence**.

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval touching `test_geom_camber_cruise/000020.pt` returns NaN. All merged PRs include this fix.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA (#3281) + surface-pressure L1 aux (#3337) + FiLM per-block conditioning (#3265) + bf16 AMP (#3373) + Cautious AdamW (#3315) + Bernoulli pressure residual (#3466) + NaN-safe evaluate_split**. Seven-step improvement trajectory in round 5: −1.3% (scale-inv) → −7.84% (EMA) → −6.41% (surf-L1) → −9.77% (FiLM) → −13.16% (bf16) → −12.31% (Cautious AdamW) → −4.70% (Bernoulli residual). Cumulative −30.49% val.

Key observations from merged results:
- **Bernoulli residual win uniform across splits** (every val and test cell improved 1.8-7.2%); largest gain on `single_in_dist` (val: 109.91 → 102.04, −7.16%) — the previous worst split. The student's pre-pass showed residual std *increases* 1.71× under the transformation; the win comes from removing the analytic V_∞² component, not from std reduction.
- **Cautious AdamW mask agreement is invariant across all merged mechanisms** (~0.62 flat across 13 epochs). Direct empirical evidence that the seven mechanisms operate on disjoint state.
- **single_in_dist is still the worst split** at val=102.04 after Bernoulli — improvement from 122.19 (FiLM) → 109.91 (Cautious AdamW) → 102.04 (Bernoulli residual). Still ~39 points worse than `geom_camber_cruise` (62.99). Three new WIPs (#3519 Fourier FiLM, #3547 Cp normalization, #3548 AoA-jitter TTA) directly target this gap.
- **Cold-start regime persists.** Even with bf16's epoch unlock, best epoch was 17/50 for Bernoulli residual — training is still in mid-descent at the wall-clock cap. EMA decay-annealing (#3545) and capacity revisit (#3463) both target this.
- **Negative results from this loop (#3433 per-domain norm, #3422 Huber):** Per-domain output normalization makes single_in_dist *worse* (the regime-mismatch is in V_∞², addressed analytically by Bernoulli). Huber smooths the train-loss trajectory but loses signal — L1's constant gradient at small residuals was the real driver.

Strongest remaining axes (in priority order):

1. **Schedule alignment** (#3465 thorfinn): T_max=19 matching bf16 epoch count — most defensible schedule fix based on #3346 negative result.
2. **Capacity revisit** (#3463 edward): n_hidden=192/256 sweep, now tractable with 33 GB VRAM from bf16. Edward just started training.
3. **Optimizer alternative head-to-head** (#3425 tanjiro): Schedule-Free AdamW vs Cautious AdamW after rebase onto new merged baseline (86.09).
4. **Cp normalization** (#3547 askeladd new this loop): divide by p_B in addition to subtract — the natural physics extension of the Bernoulli win. Standard CFD non-dimensionalization.
5. **Conditioning richness** (#3519 nezuko): Fourier-embedded FiLM — Tancik-style multi-frequency sin/cos encoding of the 11-dim flow vector before FiLM MLP; targets single_in_dist via richer angular resolution on AoA.
6. **EMA refinement** (#3432 fern): SEMA — copy EMA weights back each epoch; (#3545 alphonse new this loop): EMA decay annealing — low decay early, high decay late, addresses cold-start.
7. **Test-time augmentation** (#3548 frieren new this loop): AoA-jitter TTA — K-ensemble of inference with small AoA perturbations. First eval-only experiment in round 5.

**Closed in this loop**: 
- #3433 (alphonse per-domain norm) — +3.2% to +11.5% regression; per-domain output normalization is the wrong direction (Bernoulli residual addresses the same issue analytically).
- #3422 (frieren Huber) — +5.4% to +7.7% regression; Huber's quadratic regime at small residuals damps the very signal L1 aux contributes.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED | Surface-pressure L1 aux loss (w=1.0) |
| #3265 | fern | MERGED | FiLM per-block global conditioning |
| #3373 | edward | MERGED | bf16 mixed precision — VRAM 42→33 GB, 14→19 effective epochs |
| #3315 | askeladd | MERGED | Cautious AdamW |
| #3466 | askeladd | MERGED (this loop) | Bernoulli pressure residual — new best: val_avg 86.09 |
| #3347 | alphonse | CLOSED | Manifold mixup — mesh-correspondence problem |
| #3346 | thorfinn | CLOSED | Cosine T_max=15 + warmup + LR=7e-4 — clear regression (3 seeds) |
| #3374 | nezuko | CLOSED | Stochastic depth — 3-seed robust negative |
| #3422 | frieren | CLOSED (this loop) | Huber loss for surf-pressure aux — both arms regressed +5-8% |
| #3433 | alphonse | CLOSED (this loop) | Per-domain target normalization — both arms regressed +3-11% |
| #3425 | tanjiro | WIP (rebase pending — new baseline 86.09) | Schedule-Free AdamW — head-to-head vs Cautious AdamW |
| #3432 | fern | WIP | SEMA: copy EMA weights back each epoch — 2 arms (freq=1, freq=2) |
| #3463 | edward | WIP (actively training) | Capacity revisit with bf16: n_hidden=192, n_hidden=256 |
| #3465 | thorfinn | WIP | T_max alignment: T_max=19 no-warmup, T_max=25 — schedule fix |
| #3519 | nezuko | WIP | Fourier-embedded FiLM conditioning — 2 arms (k=4 σ=1, k=8 σ=2) |
| #3545 | alphonse | WIP (new this loop) | EMA decay annealing — 2 arms (linear5, cosine19) |
| #3547 | askeladd | WIP (new this loop) | Cp normalization — 2 arms (cp, halfcp) |
| #3548 | frieren | WIP (new this loop) | AoA-jitter TTA — 2 arms (K=4 σ=0.1°, K=8 σ=0.05° + cond) |

## Plateau watch

Round 5 now shows 7 compounding wins totaling −30.49% val_avg. No plateau signal. The val_single_in_dist split improved from 122.19 (post-FiLM) → 109.91 (post-Cautious-AdamW) → 102.04 (post-Bernoulli) — still the worst split but closing the gap to OOD splits. Five WIPs (#3463, #3519, #3547, #3548, #3545) target this split or its cold-start training-time bottleneck.

Note: WIPs started pre-Bernoulli (#3425, #3432, #3463, #3465, #3519) still need rebases when their terminal results come in. The Bernoulli residual subtraction is an additive transformation in the data pipeline — orthogonal to optimizer (Cautious vs SF-AdamW), schedule (T_max), capacity (n_hidden), EMA dynamics (SEMA), and conditioning (Fourier FiLM). Rebases should compound cleanly for all six axes.

## Potential next research directions (post-current batch)

1. **Proper chord-position Bernoulli** (askeladd #3466 follow-up): per-foil chord-boundary detection + chord-position-dependent Cp formula. Arm B of #3466 failed on the implementation (used global mesh x-coord instead of normalized chord); fixing this could unlock another −2% to −4%.
2. **Cautious Lion + EMA + FiLM**: askeladd's earlier suggestion. Lion + cautious gating would test whether signed updates + agreement filter is better than AdamW + cautious. Defer until SF-AdamW vs Cautious head-to-head (#3425) is resolved.
3. **torch.compile() on top of bf16** (edward's #3373 follow-up #3): pure speed-up, may unlock 22+ effective epochs. Risk: compatibility with CautiousAdamW custom optimizer step + Bernoulli pre-subtract.
4. **Per-block mask-mean logging**: with FiLM also modulating per-block, per-block cautious mask logging would reveal whether gating fires more in FiLM-modulated layers — useful for architecture-mask co-design.
5. **Mach-correction Bernoulli**: at high V_∞, p_B should include a compressibility term `(1 + γ/2 · M²) ^ (γ/(γ-1))`. Would apply only at the highest-V_∞ samples (raceCar). Likely small gain (~0.5%), but physics-correct.
6. **Lift/drag coefficient auxiliary head**: predict integrated lift and drag as a side output, with L1 loss against the analytical-CFD-computed coefficients. Multi-task learning regularizer.
7. **Learned residual MLP head**: stack a small residual-prediction MLP on the Transolver output. Targets the still-large per-split MAE on single_in_dist.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale/domain), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (chord-Bernoulli, Cp, lift/drag aux), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
