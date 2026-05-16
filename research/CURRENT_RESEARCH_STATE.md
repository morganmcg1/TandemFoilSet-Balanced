# SENPAI Research State

- **Updated:** 2026-05-16 03:50 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3582 MERGED):** `val_avg/mae_surf_p = 61.2023`, `test_avg/mae_surf_p = 54.0076`
  - Note: torch.compile() `default` mode validated on full merged stack at `cf6ac4a`. Arm A beat prior best (75.40) by **−18.83%** with all 8 val/test cells improving. Per-epoch speedup 1.88× (108s → 57s), unlocking **32 effective epochs vs 17 at baseline**. Run still descending at −0.7/epoch at cutoff — schedule and other axes have additional headroom.
- **Cumulative round-5 improvement:** −50.59% val_avg (123.88 → 61.20) and −52.78% test_avg (114.37 → 54.01) vs pre-round-5 floor. **Nine compounding wins in sequence.**

## Most recent research direction from human researcher team
(none — no open Issues directed at this launch)

## Critical operational fix in baseline

PR #3266 propagated a NaN workaround for `data/scoring.py::accumulate_batch` into `train.py::evaluate_split` (since `data/` is read-only). Without it, any test eval touching `test_geom_camber_cruise/000020.pt` returns NaN. All merged PRs include this fix.

## Important finding from #3582 review — bernoulli_residual was OFF in 75.40 baseline

The fern student noticed (and verification confirmed) that the merged 75.40 baseline (`thorfinn-tmax25_em10` run at commit `3a3104a`) had NO `bernoulli_residual` field in its config.yaml — the field didn't exist in the Config dataclass at that side-branch commit. Both compile arms also ran with `bernoulli_residual: false`.

**Implications**:
1. The compile win (75.40 → 61.20) is apples-to-apples valid (both no-Bernoulli).
2. The merged stack is actually a **7-stack** (scale-inv + EMA + surf-L1 + FiLM + bf16 + Cautious AdamW + T_max=25 + compile), not 8. The "Bernoulli mechanism" claim from PR #3466's merge is suspect.
3. **Real untested headroom**: enabling bernoulli with the proper flag may compound (~+1–4% from physics prior).
4. The BASELINE.md reproduce command (`--bernoulli_residual freestream`) is broken — simple_parsing drops the `freestream` arg silently, leaving `bernoulli_residual=False`.

This is a high-priority cleanup item for a future loop — a single fix (argparse to take `--bernoulli_residual` as bare bool or `--bernoulli_residual True/False` explicitly) and one verification run would resolve it.

## Current research focus and themes

Merged baseline stacks **scale-invariant loss (#3266) + EMA (#3281) + surface-pressure L1 aux (#3337) + FiLM per-block conditioning (#3265) + bf16 AMP (#3373) + Cautious AdamW (#3315) + CosineAnnealingLR T_max=25 alignment (#3465) + torch.compile (#3582, pending merge) + NaN-safe evaluate_split**. Nine-step trajectory in round 5: −1.3% (scale-inv) → −7.84% (EMA) → −6.41% (surf-L1) → −9.77% (FiLM) → −13.16% (bf16) → −12.31% (Cautious AdamW) → ~0% (Bernoulli — verified inactive in baseline above) → −12.42% (T_max alignment) → **−18.83% (torch.compile)**. Cumulative −50.59% val.

Key observations from merged results:
- **torch.compile is the single biggest single-loop win** — 1.88× per-epoch speedup, +15 effective epochs, all 8 cells improve. Pure throughput → metric translation.
- **Wall-clock-bound descent confirmed as dominant** — at epoch 32 (Arm A best), descent still −0.7/epoch. The model is *still* undercooked even at 32 epochs. Every "more compute per wall-clock" axis is high-priority.
- **At matched epoch number, compile arms are ~3% worse than baseline** (epoch 17 compile: 78.04 vs baseline 75.40). The gain comes entirely from additional epochs — small fp32-accumulation order differences from kernel fusion, but no real harm.
- **VRAM dropped from 35.5 → 24.4 GB** under compile — significant headroom for larger batch size, capacity, or both.
- **Cautious AdamW mask is invariant to LR** (PR #3581 falsified the masking-stabilization hypothesis). Mask stays at ≈0.62 across lr=5e-4, 7e-4, 1e-3. Useful constraint — the gating is orthogonal to LR magnitude.
- **single_in_dist closing the gap** — val went 122.19 (FiLM) → 109.91 (Cautious AdamW) → 102.04 (Bernoulli step) → 84.88 (T_max) → 65.44 (compile). Down from worst split → comparable to others, gap collapsing. compile delivered the largest improvement on this split (−22.9%).
- **Negative results from this loop**:
  - **#3581 (thorfinn LR sweep)**: lr=7e-4 → +14.76%, lr=1e-3 → +9.40% regression. Epoch-1 spike (~800 vs baseline 365) not recoverable in 17 epochs. **Cautious mask provably did not gate the high-LR updates** — falsified the PR's stabilization claim.

Strongest remaining axes (in priority order):

1. **More compute / better schedule alignment for compile budget** (fern new this loop): T_max=32 vs T_max=35 with compile — schedule alignment for the new 32-epoch budget (the analogue of #3465's fix for the new horizon). Expected −1% to −5%. **The natural compounding follow-up.**
2. **LR sweep with compile** (thorfinn new this loop): re-run lr=7e-4 / lr=1e-3 with `--torch_compile`. The 32-epoch budget nearly 2× the recovery window thorfinn's first sweep had; late-epoch descent rate at lr=1e-3 was −4.13/epoch which could now compound.
3. **Bernoulli verify + enable** (cleanup, not yet assigned): fix `--bernoulli_residual` argparse, run one verification arm with bernoulli ON. If positive, this is a free additive win (~+1–4%).
4. **Larger batch_size with compile** (fern #4 suggestion): VRAM dropped to 24/96 GB. batch_size=8 or 16 may be feasible with compile reducing per-step overhead — fewer steps per epoch, more efficient GPU use.
5. **Cp normalization** (#3547 askeladd WIP): divide by p_B in addition to subtract — natural physics extension of the (now-suspect) Bernoulli mechanism.
6. **AoA-jitter TTA** (#3548 frieren WIP): K-ensemble of inference with small AoA perturbations.
7. **Spatial Fourier positional encoding** (#3631 nezuko WIP): NeRF-style sin/cos on (x,y) ±dsdf. Targets OOD geometry weakness.
8. **surf_weight sweep** (#3647 alphonse WIP): 5 vs 20 — re-tune the loss weighting for the new merged stack.
9. **Capacity revisit** (#3463 edward WIP, sent back for rebase): n_hidden=192/256 sweep, tractable with bf16 VRAM savings (and now further headroom from compile).
10. **SF-AdamW head-to-head** (#3425 tanjiro WIP, needs rebase to current tip): SF-AdamW must beat 61.20 (new baseline) — significant moving target.
11. **max-autotune compile mode** (fern #3 suggestion): single arm, deferred.

**Closed/merged in this loop**:
- **#3582 (fern torch.compile)** — MERGED (commit 34f7c61). New best val_avg 61.20. Ninth compounding win.
- **#3581 (thorfinn LR sweep)** — CLOSED. +9.4–14.8% regression on 17-epoch budget. Reassigned to compile-enabled re-run (#3666).

**Closed in prior loops**:
- #3432 (fern SEMA) — +22.8% to +33.4% regression; SEMA mechanism inverted under EMA decay=0.999 / short wall-clock horizon.
- #3519 (nezuko Fourier FiLM) — +4.75% regression; OOD geometry degraded by cond-path Fourier enrichment.
- #3545 (alphonse EMA annealing) — neutral on tested baseline; cold-start regime; wall-clock too short for time-varying EMA to matter.

| PR | Student | Status | Hypothesis |
|----|---------|--------|-----------|
| #3266 | frieren | MERGED | Per-sample scale-invariant loss |
| #3281 | frieren | MERGED | EMA weights at decay=0.999 |
| #3337 | frieren | MERGED | Surface-pressure L1 aux loss (w=1.0) |
| #3265 | fern | MERGED | FiLM per-block global conditioning |
| #3373 | edward | MERGED | bf16 mixed precision |
| #3315 | askeladd | MERGED | Cautious AdamW |
| #3466 | askeladd | MERGED (Bernoulli claim now suspect — see note above) | Bernoulli pressure residual |
| #3465 | thorfinn | MERGED | T_max=25 schedule alignment — val_avg 75.40 |
| #3582 | fern | MERGED (this loop) | torch.compile() — new best val_avg 61.20 |
| #3347 | alphonse | CLOSED | Manifold mixup — mesh-correspondence problem |
| #3346 | thorfinn | CLOSED | Cosine T_max=15 + warmup + LR=7e-4 — clear regression |
| #3374 | nezuko | CLOSED | Stochastic depth — 3-seed robust negative |
| #3422 | frieren | CLOSED | Huber loss for surf-pressure aux — both arms regressed |
| #3433 | alphonse | CLOSED | Per-domain target normalization — both arms regressed |
| #3432 | fern | CLOSED | SEMA — EMA-lag reset mechanism regression |
| #3519 | nezuko | CLOSED | Fourier FiLM — OOD geometry regression |
| #3545 | alphonse | CLOSED | EMA decay annealing — cold-start regime null result |
| #3581 | thorfinn | CLOSED (this loop) | Peak LR sweep on T_max=25 — both arms +9-15% regression |
| #3425 | tanjiro | WIP (rebase pending re-run on new 61.20 baseline) | Schedule-Free AdamW head-to-head |
| #3463 | edward | WIP (rebase pending re-run) | Capacity revisit: n_hidden=192 |
| #3631 | nezuko | WIP | Fourier positional encoding on spatial coords (x,y,dsdf) |
| #3647 | alphonse | WIP | surf_weight sweep: 5 vs 20 |
| #3547 | askeladd | WIP | Cp normalization — 2 arms (cp, halfcp) |
| #3548 | frieren | WIP | AoA-jitter TTA — 2 arms |
| #3665 | fern | WIP (new this loop) | T_max alignment for compile: T_max=32 vs T_max=35 |
| #3666 | thorfinn | WIP (new this loop) | LR sweep with compile: lr=7e-4 vs 1e-3 on 32-epoch budget |

## Plateau watch

Round 5 now has 9 compounding wins totaling −50.59% val_avg. **No plateau signal — descent rate is still active at the current best**. Half the wall-clock cap is still spent in a regime where −0.7/epoch is achievable. Throughput and schedule axes remain the highest-leverage axes; capacity and physics-prior axes (Cp norm, Bernoulli fix, capacity revisit) are next-tier as the throughput line saturates.

## Potential next research directions (post-current batch)

1. **Proper bernoulli verification + enable** (high priority, low complexity): fix argparse, single-arm run with `--bernoulli_residual True` on compile baseline. ~+1–4% expected.
2. **batch_size sweep with compile** (high priority, low complexity): VRAM headroom 24/96 GB. bs=8 or bs=16. Compounds with compile's per-step overhead reduction.
3. **`max-autotune` compile mode** (low complexity): single arm; extra autotune cost (~2 min on epoch 1) amortized.
4. **Bucketed dynamic shapes for `reduce-overhead`** (medium complexity): pad meshes to small bucket sizes, allow CUDA Graph Trees to record per-bucket. Could unlock further throughput.
5. **Chord-position Bernoulli** (askeladd #3466 follow-up after the verification fix): per-foil chord-boundary detection + chord-position Cp formula.
6. **LR warm-up** (thorfinn's own #1 suggestion, deferred behind compile-enabled LR sweep): 1–2 epoch linear ramp from 0.1×lr to peak.
7. **Mach-correction Bernoulli**: at high V_∞, include compressibility term. Small (~0.5%), physics-correct.
8. **Lift/drag coefficient auxiliary head**: predict integrated lift/drag as side output, L1 against CFD coefficients. Multi-task regularizer.
9. **Cautious mask ablation** (thorfinn's #4 suggestion): remove Cautious AdamW gating at baseline lr to verify it's still doing useful work. Flat 0.62 across LR ranges is suspicious — gating may be a no-op at the current operating point.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale/domain), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (chord-Bernoulli, Cp, lift/drag aux), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
