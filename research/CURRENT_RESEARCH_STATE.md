# SENPAI Research State

- **Updated:** 2026-05-16 09:15 UTC
- **Launch:** `charlie-pai2i-24h-r5` (round 5)
- **Advisor branch:** `icml-appendix-charlie-pai2i-24h-r5`
- **Target base branch:** `icml-appendix-charlie`
- **Metrics:** local JSONL only (no remote tracking on this branch)
- **Current round-5 baseline (PR #3463 MERGED):** `val_avg/mae_surf_p = 53.1915`, `test_avg/mae_surf_p = 47.5701`
  - Note: n_hidden=192 + lr=1e-3 + compile stack. Sub-multiplicative compounding: lr=1e-3 already saturated camber_rc headroom (geom_camber_rc regressed +2.0% vs n=128 baseline), so only 39% of predicted gain materialized. geom_camber_cruise continues to benefit from width (−6.23%). 5/8 cells improve. Still descending at −0.97/epoch at cutoff (epoch 24, LR already at eta_min). Cautious mask invariant at 0.6105.
- **Cumulative round-5 improvement:** −57.07% val_avg (123.88 → 53.19) and −58.40% test_avg (114.37 → 47.57) vs pre-round-5 floor. **Eleven compounding wins in sequence.**

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

1. **LR continuation** (#3771 thorfinn WIP, new loop 17): lr=1.5e-3 vs 2e-3 — continuing the monotonic sweep (5e-4→7e-4→1e-3 all improved). Key question: at what LR does TF32+compile fail to neutralize the epoch-1 spike?
2. **T_max=35 + lr=1e-3 compound** (#3665 fern WIP, sent back loop 17): schedule realignment compounds with LR win. fern's analysis predicted optimal T_max shifts under LR scaling. Expected −1–3% additional.
3. **n_hidden=192 + lr=1e-3** (#3463 edward WIP, sent back loop 17): capacity compound at new LR. Prior win was −4.09% at lr=5e-4; at lr=1e-3 the higher gradient signal should accelerate convergence on the 24-epoch budget.
4. **Bernoulli residual verify** (#3839 edward WIP, new loop 20): bernoulli=True on n=192 + lr=1e-3 + compile. Targets the geom_camber_rc regression from loop 20 — physics prior should help on OOD geometry.
5. **Gradient clipping sweep** (#3809 frieren WIP, new loop 19): clip_norm=1.0 vs 0.5. Orthogonal defensive measure.
6. **Weight decay sweep** (#3785 tanjiro WIP, new loop 18): wd=5e-5 vs wd=5e-4 — never swept on merged stack.
7. **slice_num sweep** (#3739 alphonse WIP, baseline-update sent loop 20): slice_num=96 vs 128 — physics-region granularity, OOD-geometry motivation.
8. **Cp normalization** (#3547 askeladd WIP, sent back for rebase loop 20): physics-motivated output normalization.
9. **Spatial Fourier positional encoding** (#3631 nezuko WIP, sent back for rebase loop 20): OOD geometry encoding.
10. **max-autotune compile mode** (fern suggestion, deferred): single arm, low complexity.

**Closed/merged/assigned in this loop (Loop 20)**:
- **#3463 (edward n_hidden=192 + lr=1e-3 + compile)** — MERGED (commit `f7c3b8e`). New best val_avg=53.1915 / test_avg=47.5701. Capacity + LR compounding is sub-multiplicative (camber_rc regresses +2% due to saturation between lr=1e-3 and n=192 on that split), but val_avg gate met (−1.60%). Eleven compounding wins.
- **#3739 (alphonse slice_num sweep)** — baseline-update comment posted (new target: 53.19). Label fixed from stale-wip → wip.
- **#3631 (nezuko Fourier)** — sent back for rebase (merge conflict, baseline moved 61.20 → 53.19). Must re-run on current n=192 + lr=1e-3 + compile full stack.
- **#3547 (askeladd Cp normalization)** — sent back for rebase (merge conflict, baseline moved 54.06 → 53.19). Must re-run on current full stack.
- **#3839 (edward Bernoulli verify)** — ASSIGNED (Loop 20). bernoulli_residual=True on n=192 + lr=1e-3 + compile. Long-deferred; never properly tested on any compile-era baseline. Strongest motivation on geom_camber_rc (the split that regressed +2% in loop 20) — Bernoulli correction accounts for free-stream pressure variation that the model has to infer from scratch without it.

**Closed/assigned in this loop (Loop 19)**:
- **#3694 (frieren Bernoulli verify + enable)** — SELF-CLOSED by student at 07:37 UTC after baseline-update comment posted in Loop 17. Same pattern as #3702 tanjiro: force-pushed and closed simultaneously, no results submitted. Treated as clean abandon. Two students self-closing after baseline shifts is now a recurring pattern — they appear to prefer fresh-axis assignments over re-running on a moved target.
- **#3809 (frieren gradient clipping sweep)** — ASSIGNED (Loop 19). grad_clip_norm=1.0 vs 0.5 on lr=1e-3 + compile baseline. Confirmed via grep that train.py has **no gradient clipping** — single-knob untested axis structurally orthogonal to LR/schedule/capacity/Bernoulli/slice_num/wd/Cp/Fourier in-flight runs. Defensive measure that compounds with thorfinn's #3771 high-LR continuation (clipping is the standard remedy if high-LR instability appears).
- Fresh baseline-update comments sent to **#3547 askeladd** and **#3631 nezuko** with the new 54.06 + lr=1e-3 target (their previous baseline-update comments from Loops 15/17 still pointed at the old 61.20 baseline).

**Closed/assigned in Loop 18**:
- **#3702 (tanjiro batch-size sweep)** — SELF-CLOSED by student at 07:35 UTC after baseline-update comment posted in Loop 17. Force-pushed simultaneously suggests an aborted run. No results submitted. Treated as a clean abandon; assigning fresh hypothesis on different axis.
- **#3785 (tanjiro weight_decay sweep)** — ASSIGNED (Loop 18). wd=5e-5 vs wd=5e-4 on lr=1e-3 + compile baseline. Structurally orthogonal to in-flight LR/schedule/capacity/Bernoulli compounds.
- Fresh baseline-update comments sent to **#3547 askeladd** and **#3631 nezuko** with the new 54.06 + lr=1e-3 target (both PRs had stale comments pointing at the old 61.20 baseline).

**Closed/merged in Loop 17**:
- **#3666 (thorfinn LR sweep with compile)** — MERGED (commit `92658ac`). New best val_avg=54.0564 / test_avg=48.1422. lr=1e-3 + compile. All 8 cells improve (−9.9% to −13.9% val). Epoch-1 val_avg=367 (no instability). Cautious mask 0.61 (invariant). Arm A (lr=7e-4) also a winner at 58.38 but Arm B wins decisively. Both arms still descending at epoch 32 cutoff. Ten compounding wins.
- **#3665 (fern T_max=35) — SENT BACK**: result 58.55 doesn't beat new 54.06. Re-run with lr=1e-3 + T_max=35 requested.
- **#3463 (edward n_hidden=192) — SENT BACK**: result 58.70 doesn't beat new 54.06. Re-run with lr=1e-3 + n_hidden=192 requested.
- **#3771 (thorfinn lr continuation) — ASSIGNED** (Loop 17). Testing lr=1.5e-3 vs 2e-3 continuation of monotonic LR sweep.

**Closed/merged in Loop 16**:
- **#3647 (alphonse surf_weight sweep)** — CLOSED. surf_weight=5 → +12.05% regression (val_avg 84.49), surf_weight=20 → +19.92% regression (val_avg 90.42), both vs OLD 75.40 baseline (=+38-48% vs current 61.20). All 8 val + 8 test cells regress for both arms. Clean representational-overfitting signal: train/surf_loss decreases for both arms while val MAE worsens. surf_weight=10 confirmed as a real local optimum on the full merged stack. Arm B also catastrophically degrades volume (+28.9% mae_vol_p), showing the shared backbone collapses globally when surface is over-weighted. Key finding: loss-weight ratio dynamic range that preserves shared backbone is narrow (~2–7× weighted-surf/vol); sw=10 lands at ~5.8×, which is the sweet spot. Also confirmed independently that bare `--bernoulli_residual` flag parses correctly via simple_parsing — alphonse's runs had bernoulli_residual=True and ran cleanly.
- **#3739 (alphonse slice_num sweep)** — ASSIGNED (Loop 16). Testing slice_num=96 vs 128 (default=64) on full compile stack. Mechanistic story: finer physics-region decomposition → more capacity for OOD geometry generalization on camber splits.

**Closed/merged in prior loops (Loop 14)**:
- **#3548 (frieren AoA-TTA)** — CLOSED. +51% regression vs 61.20. Key finding: FiLM-conditioned model is already AoA-smooth at σ=0.1deg (Arm A TTA vs no-TTA delta: 0.02%). Arm B's K-averaged Bernoulli term produces Jensen-biased upward offset. AoA-TTA is a dead end on this stack.
- **#3425 (tanjiro SF-AdamW)** — CLOSED after 3 rounds. Final result: val_avg=85.99 (+40.5% vs 61.20). Mechanism: SF-AdamW removes cosine annealing entirely (train/lr=5e-4 constant), bypassing the late-epoch polishing that made #3465 a 12.42% win. Cautious AdamW wins the optimizer slot.

**Closed/merged in Loop 13**:
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
| #3582 | fern | MERGED (loop 13) | torch.compile() — new best val_avg 61.20 |
| #3347 | alphonse | CLOSED | Manifold mixup — mesh-correspondence problem |
| #3346 | thorfinn | CLOSED | Cosine T_max=15 + warmup + LR=7e-4 — clear regression |
| #3374 | nezuko | CLOSED | Stochastic depth — 3-seed robust negative |
| #3422 | frieren | CLOSED | Huber loss for surf-pressure aux — both arms regressed |
| #3433 | alphonse | CLOSED | Per-domain target normalization — both arms regressed |
| #3432 | fern | CLOSED | SEMA — EMA-lag reset mechanism regression |
| #3519 | nezuko | CLOSED | Fourier FiLM — OOD geometry regression |
| #3545 | alphonse | CLOSED | EMA decay annealing — cold-start regime null result |
| #3581 | thorfinn | CLOSED (loop 13) | Peak LR sweep on T_max=25 — both arms +9-15% regression |
| #3425 | tanjiro | CLOSED (loop 14) | SF-AdamW head-to-head — 3 rounds, consistently +40%+ regression |
| #3548 | frieren | CLOSED (loop 14) | AoA-jitter TTA — FiLM already smooth, Jensen bias kills Arm B |
| #3463 | edward | MERGED (loop 20) | Capacity revisit: n_hidden=192 + lr=1e-3 + compile — new best val_avg 53.19 |
| #3631 | nezuko | WIP (sent back loop 20 — rebase + re-run on n=192+lr=1e-3+compile stack) | Fourier positional encoding on spatial coords (x,y,dsdf) |
| #3647 | alphonse | CLOSED (loop 16) | surf_weight sweep: 5 vs 20 — both arms +12-20% regression, sw=10 confirmed local optimum |
| #3739 | alphonse | WIP (new loop 16) | slice_num sweep: 96 vs 128 — physics-region granularity |
| #3547 | askeladd | WIP (sent back loop 20 — rebase + re-run on n=192+lr=1e-3+compile stack) | Cp normalization — 2 arms (cp, halfcp) |
| #3665 | fern | WIP (sent back loop 17 — rebase + lr=1e-3 + T_max=35) | T_max alignment for compile: T_max=35 + lr=1e-3 compound |
| #3666 | thorfinn | MERGED (loop 17) | LR sweep with compile: lr=1e-3 wins — val_avg 54.06 (now superseded) |
| #3771 | thorfinn | WIP (new loop 17) | LR continuation: lr=1.5e-3 vs 2e-3 |
| #3694 | frieren | CLOSED (loop 19, self-closed) | Bernoulli verify + enable — abandoned after baseline shift to 54.06 |
| #3702 | tanjiro | CLOSED (loop 18, self-closed) | Batch size sweep — abandoned after baseline shift to 54.06 |
| #3785 | tanjiro | WIP (new loop 18) | Weight decay sweep: wd=5e-5 vs wd=5e-4 |
| #3809 | frieren | WIP (new loop 19) | Gradient clipping sweep: clip_norm=1.0 vs 0.5 |
| #3839 | edward | WIP (new loop 20) | Bernoulli residual verify on n=192 + lr=1e-3 + compile |

## Plateau watch

Round 5 now has **11 compounding wins totaling −57.07% val_avg**. **No plateau signal — improvements continue to land every 1–2 loops.** The capacity (n=192) win is sub-multiplicative with LR (only 39% of predicted gain) — indicating the LR and capacity axes are not perfectly orthogonal. The camber_rc split now shows a regression (+2%) relative to the n=128 + lr=1e-3 baseline, which is the primary focus of the next round (Bernoulli, thorfinn LR continuation, fern T_max). The LR continuation sweep (lr=1.5e-3 vs 2e-3, thorfinn #3771) is the highest-variance pending test — if high-LR succeeds at n=192, it could be a much larger win than seen at n=128.

## Potential next research directions (post-current batch)

1. **`max-autotune` compile mode** (low complexity): single arm; extra autotune cost (~2 min on epoch 1) amortized. Fern's suggested follow-up from #3582.
2. **Bucketed dynamic shapes for `reduce-overhead`** (medium complexity): pad meshes to small bucket sizes, allow CUDA Graph Trees to record per-bucket. Could unlock further throughput.
3. **Chord-position Bernoulli** (askeladd follow-up after Bernoulli verify): per-foil chord-boundary correction Cp formula.
4. **LR warm-up** (thorfinn's own #1 suggestion, deferred behind compile-enabled LR sweep): 1–2 epoch linear ramp from 0.1×lr to peak.
5. **Mach-correction Bernoulli**: at high V_∞, include compressibility term. Small (~0.5%), physics-correct.
6. **Lift/drag coefficient auxiliary head**: predict integrated lift/drag as side output, L1 against CFD coefficients. Multi-task regularizer.
7. **Cautious mask ablation** (thorfinn's #4 suggestion): remove Cautious AdamW gating at baseline lr to verify it's still doing useful work. Flat 0.62 across LR ranges is suspicious — gating may be a no-op at the current operating point.
8. **AoA-TTA with larger sigma** (deferred from #3548): AoA σ=0.5–1deg might force the model to actually generalize across AoA rather than staying on the FiLM-smooth manifold. Not recommended near-term; mechanism is theoretically weak.

## If plateau hits (5 consecutive no-improvement)

1. Diagnose via per-split metrics: is failure concentrated on single_in_dist (scale/domain), geom_camber_rc (camber OOD), or re_rand (Re OOD)?
2. Take bigger swings: different backbone (UPT, Galerkin Transformer, OFormer), residual-prediction (chord-Bernoulli, Cp, lift/drag aux), learned normalization.
3. Architecture-level: graph-neural-network style message passing vs the Transolver slice attention — test whether the approximation is a bottleneck.
