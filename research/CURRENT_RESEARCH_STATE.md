# SENPAI Research State

- **As of:** 2026-05-13 (updated cycle 30)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 93.6198`** — PR #2031 (weight_decay=5e-4), merged 2026-05-13 cycle 30.
**Test 4-split mean: 83.8825** (test_avg/mae_surf_p).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | #1558 | Huber surface loss δ=0.5 | 98.1642 | −17.72% |
| 18 | #1795 | Decoupled LR surf_head (5e-3) | 97.9914 | −0.18% |
| 30 | **#2031** | **Weight decay 1e-4 → 5e-4** | **93.6198** | **−4.46%** |

## Current research focus

**Cycle 30.** Five stacked mechanisms now merged (BIVW + surf-head + Huber + decoupled surf_head LR + weight_decay=5e-4) giving 25.8% total improvement from cycle-2 baseline. New baseline 93.6198 establishes a fresh frontier — 6 fresh hypotheses now in flight targeting the next gain.

**Post-#2031 insights:**
- **Hyperparameter staleness is real**: WD=1e-4 survived BIVW, Huber, and decoupled-LR unquestioned. 5× increase to 5e-4 yielded −4.46% — largest optimizer-side gain since Huber. Every other hyperparameter (LR, β2, δ) should be audited for staleness.
- **Late-epoch oscillation characterization**: The e11→e12 spike pattern is steady-state, confirmed across β2 (#2015), warmup (#1949), gradient clipping (#2058), and wider head (#2057). The three closed in cycle 30 (#2058, #2057, #1949 previously) all failed to dampen it. Untested: denominator-floor (ε) and cosine-schedule (T_max).
- **surf_head is NOT the capacity bottleneck** (#2057): hidden_dim 64→128 gave +5.36% regression. The encoder is the limiter for OOD geometry.
- **Gradient clipping on surf_head is NOT the mechanism** (#2058): sh_grad_norm is 0.77× encoder norm, not larger. The spike is in the step magnitude (LR × m/√v), not the gradient. The untested levers: AdamW ε (denominator floor), cosine T_max (LR schedule), per-group step decay.
- **AdamW WD effective on surf_head at 10×LR**: with coupled WD, surf_head sees 10× effective shrinkage. Decoupled-WD experiment next.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 2013 | tanjiro | logcosh-surface-loss | WIP | C²-smooth logcosh vs Huber δ kink; arms scale={1.0, 0.5}; stacks on #2031 baseline |
| 2091 | frieren | torch-compile | WIP | torch.compile mode ∈ {default, reduce-overhead}; throughput unlock → 18+ epochs in 30 min |
| 2120 | fern | wd-deeper | WIP (NEW) | WD sweep {7e-4, 1e-3, 2e-3}: exploit #2031's winning trajectory |
| 2122 | edward | decoupled-wd | WIP (NEW) | Per-group WD: encoder vs surf_head (10× LR → 10× effective shrinkage at coupled WD) |
| 2123 | askeladd | cosine-tmax | WIP (NEW) | T_max sweep {15, 20, 25}: T_max=50 wastes 72% of cosine cycle at 14-ep wall-clock cap |
| 2124 | alphonse | surf-only-pw | WIP (NEW) | Surface-only pressure weight {0.5, 1.5}: NO mean-normalisation (avoids #1496 bug) |
| 2127 | thorfinn | surf-head-step-decay | WIP (NEW) | Step decay surf_head LR at e10 {×0.5, ×0.3}: directly targets late-epoch oscillation |
| 2128 | nezuko | adamw-eps | WIP (NEW) | AdamW ε sweep {1e-7, 1e-6}: denominator-floor stabilizer (orthogonal to β2/WD/clip) |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head** — confirmed (PR #1528, −5.4% additional).
3. **Huber surface loss delta=0.5** — confirmed (PR #1558, **−17.7%**). The largest gain yet.
4. **Per-channel BIVW** — **rejected** (PR #1580, +29.6% regression). Scalar BIVW's p-dominated coupling is beneficial — do not break it.
5. **Grad-clip + higher LR on Huber base** — **rejected** (PR #1499, +1.45% regression). Huber already compresses grad norms 5×; clip is redundant.
6. **Huber on volume loss** — **rejected** (PR #1650, best +8.7% regression). Volume loss is the encoder's supervisory signal; Huber removes scale information the encoder needs. Surface Huber + volume MSE remains the correct recipe.
7. **Smaller Huber delta** — **rejected** (PR #1627). δ=0.3 (+15.6%) and δ=0.2 (+17.2%) both regress. δ=0.5 is a narrow local optimum.
8. **surf_weight tuning on Huber baseline** — **rejected** (PR #1720, all arms +7-21% regression). sw=10 already optimal.
9. **Frozen p-variance stratified sampling** — **rejected** (PR #1746, +272% regression). Variance dynamic range is 8 OOM.
9a. **log(Re) quantile bucketing** — **rejected** (PR #1868, +8.4% regression). Structural no-op.
9b. **Re-curriculum via per-sample loss multiplier** — **rejected** (PR #1978, +16.87% regression). BIVW × Re-tail double-weights low-Re, cancels high-Re boost. Entire Re-reweighting direction closed.
10. **BF16/AMP** — **rejected** (PR #1572, +3.62% val / +30.09% val at n256). Precision-sensitive surface MAE: val_geom_camber_rc +11.33%. FP32 required.
10a. **torch.compile** — testing (#2091 frieren, WIP). FP32 preserved; pure throughput unlock.
11. **Wider MLP (ratio=4)** — **rejected** (PR #1498, +24.97%). Wall-clock-bound.
11a. **Slice_num=128** — **rejected** (PR #1501, +19.30%). Wall-clock-bound.
11b. **Shallower depth (n_layers=4)** — **rejected** (PR #1881, +8.39%). Underfitting. Depth=5/14ep is Pareto.
11c. **n_head 4→8** — **rejected** (PR #1924, +18.84%). Wall-clock-bound. Pareto frontier fully characterized.
11d. **Encoder LR re-tune** — **rejected** (PR #1974, +5.42% val, +3.18% test). Encoder LR=5e-4 is correct at the new WD=5e-4 baseline. Both 3e-4 and 7e-4 regress. Confirmed optimal.
12. **Warmup schedule** — **rejected** (PR #1497, +17.98%). Wall-clock-bound; 5 warmup epochs burn productive steps. Wrong axis for the steady-state oscillation.
13. **Pressure-channel emphasis** — **rejected** (PR #1496, +20.04% regression). Mean-normalisation inverted the experiment: pw=3 → normalised weights [0.6, 0.6, 1.8] → Ux/Uy down-weighted 40%. Follow-up: surface-only pressure weight WITHOUT mean-normalisation (#2124 alphonse, NEW).
14. **EMA model weights** — **rejected** (PR #1808, +7.8-16.2%). 14-epoch budget too short; model in descent phase.
15. **Decoupled LR for surf_head vs encoder** — **confirmed** (PR #1795, −0.18%). surf_head_lr=5e-3 is the winning arm. Fully characterized (#1949).
16. **Per-channel Huber delta** — **rejected** (PR #1922, +5.61% val, +3.70% test). Δ_p=0.5/δ_ux=δ_uy=2.0 arm produced worst result — larger δ on Ux/Uy flattens velocity mid-magnitude gradients. Global δ=0.5 is correct.
17. **Adaptive Huber δ** — **rejected** (PR #1950, +2.25%). EMA δ collapsed to floor (0.2) within 60 steps.
18. **SWA late-epoch averaging** — **rejected** (PR #1951, +3.33%). SWA mechanism worked but trajectory seeding worse.
19. **Stochastic Depth (DropPath)** — **rejected** (PR #1987, +3.31%). Oscillation smoothed away but +3-ep convergence slowdown dominated.
20. **AdamW β2 sweep** — **rejected** (PR #2015, +6.49%). β2=0.999 is a structural stabilizer — REQUIRED.
21. **Weight decay re-tune** — **confirmed** (PR #2031, **−4.46%** val / **−5.26%** test). WD 1e-4→5e-4. New baseline 93.6198.
22. **Decoupled LR + head warmup** — **rejected** (PR #1949, best arm +1.30%). LR axis fully exhausted at 5e-3.
23. **Wider surf_head** — **rejected** (PR #2057, +5.36% val). hidden_dim=128 regressed vs 64. Encoder is the capacity bottleneck, not the head.
24. **Per-group surf_head gradient clipping** — **rejected** (PR #2058, +10.39% val). sh_grad_norm is 0.77× encoder norm — gradient is NOT the problem. The update magnitude spike is m/√v driven, not ‖g‖ driven. Wrong mechanism.
25. **LogCosh surface loss** — testing (#2013 tanjiro, WIP). C²-smooth alternative to Huber kink.
26. **Deeper WD sweep {7e-4, 1e-3, 2e-3}** — testing (#2120 fern, NEW). Exploits #2031's win trajectory; checks if 5e-4 is a local optimum or inflection.
27. **Decoupled weight_decay per param group** — testing (#2122 edward, NEW). surf_head at 10× LR sees 10× effective WD shrinkage; decoupling may unlock further gains.
28. **Cosine T_max sweep {15, 20, 25}** — testing (#2123 askeladd, NEW). T_max=50 means only 28% of cosine cycle is traversed in 14 epochs — effectively a slowly-decaying constant LR.
29. **Surface-only pressure weight {0.5, 1.5}** — testing (#2124 alphonse, NEW). Sub-unit weight on surface pressure only, NO mean-normalisation (corrects #1496's bug).
30. **surf_head step decay at e10 {×0.5, ×0.3}** — testing (#2127 thorfinn, NEW). Step-decays surf_head LR in late training; per #2058 diagnostics, the update magnitude comes from the 10× LR, not gradient size.
31. **AdamW ε sweep {1e-7, 1e-6}** — testing (#2128 nezuko, NEW). Denominator-floor stabilizer; orthogonal to β2/WD/grad-clip; small-batch sampler creates low-v regimes where ε becomes the dominant denominator term.

## Key insights

**Cycle 11 (PR #1580):** Per-channel BIVW failed (+29.6%). Scalar BIVW is *implicitly* a p-variance-driven Re-curriculum. Decoupling removes the beneficial coupling. **Lesson:** scalar BIVW coupling across channels is load-bearing for `mae_surf_p`.

**Cycle 13 (PR #1650):** Huber on volume regressed all arms. Surface is the evaluated readout; volume is the encoder's supervisory signal. **Lesson:** surface Huber + volume MSE is the correct recipe.

**Cycle 14 (PR #1627):** Huber delta sweep (δ=0.2, 0.3) both regress. δ=0.5 is a narrow sweet spot. **Lesson:** do not re-sweep delta without changing other levers.

**Cycles 25-26 (PRs #2015, #1949):** β2=0.999 is a structural stabilizer, not a tunable lag. The balanced sampler draws from 3 domains with heterogeneous mesh sizes (74K-242K nodes) and y-magnitudes (164-458 std), producing high per-batch variance. **Lesson:** β2 must be preserved at 0.999 unless the sampler is simultaneously redesigned.

**Cycles 25-26 (PR #1949):** The late-epoch oscillation (e11 best, e12 spike, e14 recovery) is a *steady-state property* of the optimization landscape. Warmup, β2 (via #2015), gradient clipping (#2058), and wider head (#2057) all failed to dampen it. The spike is in the LR × m/√v step magnitude, not in gradient size. **Next: denominator (ε), LR decay schedule (T_max), per-group LR step decay.**

**Cycle 29 (PR #1572):** Surface MAE is precision-sensitive in a load-bearing way. BF16's 7-bit mantissa caused +11.33% on val_geom_camber_rc. FP32 required on surf_head + Huber loss. **Lesson:** any future precision relaxation must be matmul-only.

**Cycle 30 (PR #2031):** WD=1e-4 was stale since the BIVW era. Increasing 5× to 5e-4 gave −4.46% val / −5.26% test. Hyperparameter staleness is real. **Lesson:** revisit all stale hyperparameters systematically — cosine T_max (still at 50) is the next obvious candidate.

**Cycle 30 (PR #1974, encoder LR):** Encoder LR=5e-4 was confirmed optimal even at the new WD=5e-4 baseline. Both 3e-4 (underfitting) and 7e-4 (overshoot) regressed. **Lesson:** encoder LR and WD are roughly orthogonal — WD win did not shift the LR optimum.

**Cycle 30 (PR #2058, grad-clip analysis):** Per-student diagnostic logging (sh_grad_norm = 0.77× encoder) falsified the "large gradient" hypothesis for late-epoch oscillation. The spike is in the optimizer step (LR × m/√v), not the gradient. **Key implication:** stabilizers must target either the denominator (ε), the LR schedule (T_max, step decay), or the second moment (β2 — but REQUIRED at 0.999), not the gradient.

## Closed / rejected hypotheses (summary)

- #1503 (standalone surf-head, no BIVW) — worse
- #1500 (n_hidden=256) — budget failure
- #1580 (per-channel BIVW) — 29.6% regression
- #1499 (grad-clip + higher LR) — 1.45% regression
- #1650 (Huber on volume) — 8.7-19.9% regression
- #1627 (Huber delta 0.2/0.3) — 15.6-17.2% regression
- #1497 (warmup + cosine LR) — 17.98% regression, wall-clock-bound
- #1746 (frozen p-variance sampler) — 272% regression
- #1498 (wider MLP ratio=4) — 24.97% regression, wall-clock-bound
- #1501 (slice_num=128) — 19.30% regression, wall-clock-bound
- #1881 (n_layers=4) — 8.39% regression, underfitting
- #1720 (surf_weight sweep) — 7-21% regression
- #1808 (EMA weights) — 7.8-16.2% regression, budget too short
- #1868 (log(Re) quantile bucketing) — 8.4% regression, structural no-op
- #1924 (n_head=8) — 18.84% regression, wall-clock-bound
- #1950 (adaptive Huber δ) — 2.25% regression
- #1978 (Re-loss multiplier) — 16.87% regression, destructive BIVW interaction
- #1951 (SWA) — 3.33% regression
- #1987 (DropPath) — 3.31% regression, wall-clock-bound regularization
- #2015 (β2=0.95) — 6.49% regression, β2=0.999 REQUIRED
- #1949 (surf_head LR warmup) — 1.30% regression, LR axis exhausted
- #1572 (BF16) — 3.62-30.09% regression, precision-sensitive
- #2057 (wider surf_head) — 5.36% regression, encoder is the bottleneck
- #2058 (surf_head grad-clip) — 10.39% regression, wrong mechanism
- #1974 (encoder LR retune) — 5.42% regression, LR=5e-4 confirmed optimal
- #1922 (per-channel Huber delta) — 5.61% regression, global δ=0.5 is correct
- #1496 (pressure-channel emphasis) — 20.04% regression, mean-normalisation bug

## Potential next directions (after cycle 30 in-flight)

- **Cosine restart** (SGDR): single restart mid-training may escape the e12 local basin
- **Discriminative LR decay** (per-layer LR scaling): some encoder layers may be over-regularized
- **n_hidden=256 + FP32** (if torch.compile #2091 unlocks epochs): capacity unlock under budget-neutral throughput gain
- **Cross-channel attention in surf_head**: current head is channel-independent MLP; cross-channel mixing may unlock inter-channel geometry
- **Per-node loss weighting by local curvature/geometry**: leading/trailing edge nodes may dominate MAE
- **Loss-side sampler rebalance with bounded weighting**: tempered 1/var(p) (log-compressed) vs current raw scalar BIVW
