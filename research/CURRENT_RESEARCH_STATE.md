# SENPAI Research State

- **As of:** 2026-05-13 (updated cycle 42)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 87.0144`** — PR #2178 (torch.compile default mode, WD=3e-4, 21 epochs), merged 2026-05-13 cycle 42.
**Test 4-split mean: 78.9539** (test_avg/mae_surf_p).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | #1558 | Huber surface loss δ=0.5 | 98.1642 | −17.72% |
| 18 | #1795 | Decoupled LR surf_head (5e-3) | 97.9914 | −0.18% |
| 30 | #2031 | Weight decay 1e-4 → 5e-4 | 93.6198 | −4.46% |
| 33 | #2091 | torch.compile default (21 epochs vs 14) | 89.7197 | −4.16% |
| **42** | **#2178** | **WD re-tune 1e-4→3e-4 at 21 epochs** | **87.0144** | **−3.01%** |

## Current research focus

**Cycle 42.** Seven stacked mechanisms now merged (BIVW + surf-head + Huber + decoupled surf_head LR + WD re-tune to 3e-4 + torch.compile) giving 31.0% total improvement from cycle-2 baseline. **Critical finding from PR #2178: WD axis is BUDGET-DEPENDENT.** WD=5e-4 was optimal at 14 epochs; at 21 epochs it OVER-REGULARIZES and amplifies the e12 spike. WD=3e-4 damps the spike and reaches 87.0144.

**Key new insights from PR #2178:**
- **WD optimal shifts with epoch budget.** WD=5e-4 (14-epoch optimum) → WD=3e-4 (21-epoch optimum). All in-flight PRs using WD=5e-4 are impaired and must now beat **87.0144** (harder target). Results should still be informative for their respective hypotheses.
- **e12 spike may be a SYMPTOM of over-regularization.** WD=3e-4 damps the e12 spike (e10=112→e12=108, smooth) while winning; WD=5e-4 amplifies it (e10=106→e12=135, +27%). This partially challenges the cycle 34 reframing ("spike is beneficial") — or more precisely: the spike from optimal WD is different from the spike from over-regularized WD.
- **In-distribution split fully recovered.** val_single_in_dist regression from PR #2091 (+4.9%) reversed (−6.9% at WD=3e-4). OOD cruise gives back slightly (+3.4%) — cruise prefers lower WD.
- **Finer WD sweep around 3e-4 is the highest-priority next step** (frieren assigned #2284): test {2e-4, 2.5e-4, 4e-4} to find the precise 21-epoch optimum and check if further headroom exists.
- **Recipe still wall-clock-bound.** Both arms best_epoch=21 (still descending). Longer training could yield more gains.

**Post-#2031 and #2091 insights (still relevant):**
- **torch.compile(default, dynamic=True)** is a pure throughput unlock (1.43×). Only 2 compile frames despite 74K–242K node mesh range.
- **surf_head is NOT the capacity bottleneck** (#2057). Encoder is the limiter for OOD geometry.
- **Gradient clipping NOT the mechanism** (#2058). Spike is LR × m/√v (step magnitude), not gradient size. Denominator-floor (ε) ruled out (#2128 — surf_frac_below_eps=0 always).
- **AdamW WD effective on surf_head at 10×LR**: with coupled WD, surf_head sees 10× effective shrinkage. Decoupled-WD experiment next.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 2013 | tanjiro | logcosh-surface-loss | CLOSED | Both arms regressed (+3.51%, +14.18% vs 97.99). C² smoothness was a non-issue. Surface-loss family well-characterized as dead end. |
| 2189 | tanjiro | ema-21epoch | CLOSED | Both arms regressed (+9.07%/+16.46% val, +10.93%/+20.34% test). EMA init bias decay too slow; live model still descending at e21. EMA direction permanently closed. |
| 2091 | frieren | torch-compile | **MERGED** | torch.compile default mode; 21 epochs in 30 min; val 89.7197 / test 79.3167 — NEW BASELINE |
| 2178 | frieren | compile-wd-compose | **MERGED** | WD=3e-4+compile; val 87.0144/test 78.9539 — NEW BASELINE. WD=5e-4 regressed (+1.17%). |
| 2284 | frieren | finer-wd-sweep-21epoch | WIP (NEW) | Finer WD sweep {2e-4, 2.5e-4, 4e-4}: map WD curve at 21 epochs around 3e-4 optimum |
| 2120 | fern | wd-deeper | CLOSED | Arm 1 (WD=7e-4) regressed +18.85% val / +18.22% test. Branching rule halted Arms 2-3. WD=5e-4 is a SHARP peak. |
| 2153 | fern | wd-bracket | CLOSED | Both arms +15.43%/+12.60%. WD=5e-4 SHARP bilateral peak. Key: rc↔sid pull opposite WD directions; e14 breakthrough load-bearing (-13.5% → -2.4% at 4e-4). WD axis fully closed. |
| 2259 | fern | stratified-sampler | WIP (NEW) | Per-batch domain stratification (strict/rotated): tests sampler-variance → per-split asymmetry hypothesis; arms: 1+1+1+1weighted vs 2+1+1 rotating |
| 2122 | edward | decoupled-wd | CLOSED | Both arms regressed (+9.17%/+18.62%). Key insight: head NEEDS more shrinkage (10× coupled WD was protective). Head-down direction fully rejected. |
| 2232 | edward | head-up-wd | WIP (NEW) | Symmetric direction: surf_head_wd∈{1e-3,2e-3} vs encoder 5e-4. Head-up composes with torch.compile. |
| 2123 | askeladd | cosine-tmax | WIP (NEW) | T_max sweep {15, 20, 25}: T_max=50 wastes 72% of cosine cycle at 14-ep wall-clock cap |
| 2124 | alphonse | surf-only-pw | CLOSED | Both arms +11.85%/+21.85%. Monotonic regression across all k values. k=1.0 is sharp local minimum. Channel-weight axis fully closed. |
| 2227 | alphonse | cosine-restart | WIP (NEW) | SGDR cosine restart T_0=10/7: induces planned spike+recovery based on cycle 34 reframing that e12 spike is BENEFICIAL |
| 2127 | thorfinn | surf-head-step-decay | CLOSED | Both arms regressed (+7.05%, +2.83%). MECHANISM CONFIRMED (spike damped) but spike+recovery is BENEFICIAL — damping spike loses recovery. |
| 2188 | thorfinn | encoder-lr-boost | WIP (NEW) | Encoder LR boost at e15-18 (dual to head-LR damp); arms: ×2.0, ×3.0; composes w/ compile |
| 2128 | nezuko | adamw-eps | CLOSED | Both arms +13.13%/+23.08%. Decisive: surf_frac_below_eps=0 always. Eps cannot affect surf_head update shape. Denominator-floor mechanism ruled out. |
| 2201 | nezuko | beta2-long | WIP (NEW) | AdamW β2=0.9999/0.9995: symmetric untested direction from #2015 (β2=0.95 regressed); longer second-moment timescale |

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
10a. **torch.compile(default, dynamic=True)** — **confirmed** (PR #2091, **−4.16% val / −5.44% test**). 21 epochs in 30 min (1.43× speedup). reduce-overhead OOM'd. **New baseline 89.7197.**
10b. **torch.compile + WD re-tune to 3e-4** — **confirmed** (PR #2178, **−3.01% val / −0.46% test**). WD=3e-4 is optimal at 21 epochs. WD=5e-4 over-regularizes at 21 epochs (+1.17% regression). New baseline 87.0144.
10c. **Finer WD sweep {2e-4, 2.5e-4, 4e-4} at 21 epochs** — testing (#2284 frieren, NEW). Map the WD curve around the new 3e-4 optimum.
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
25. **LogCosh surface loss** — **rejected** (PR #2013, +3.51% Arm 1 / +14.18% Arm 2). C² smoothness was a non-issue (oscillation amplitude unchanged). Quadratic-regime gradient at typical residuals ~half of Huber — under-trained surf_head. **Surface-loss family is now well-characterized as a dead end** (#1558 δ=0.5 winner, #1627 δ-sweep, #1950 adaptive, #1922 per-channel δ, #2013 LogCosh).
25a. **EMA model weights re-screen at 21-epoch budget** — **rejected** (PR #2189, +9.07%/+16.46% val, +10.93%/+20.34% test). Both arms regressed. Mechanism: ema_init_bias decay too slow at 20 epochs (still 0.034 at e18); live model still descending at e21 (no plateau for EMA to help). Arm 2 live e14=87.07 matched new SOTA but EMA buried it. **EMA direction PERMANENTLY CLOSED.**
26. **Deeper WD sweep {7e-4, 1e-3, 2e-3}** — **rejected** (PR #2120, +18.85% val at 7e-4). Branching rule correctly halted Arms 2-3. **WD=5e-4 is a SHARP peak**, not a plateau — confirmed by uniform regression across all 4 splits and an attenuated e14 breakthrough.
26a. **WD bracket sweep {4e-4, 5.5e-4, 6e-4}** — **rejected** (PR #2153, +15.43%/+12.60%). WD=5e-4 SHARP bilateral peak confirmed. Critical finding: per-split asymmetry (rc wants more WD, sid wants less); e14 breakthrough is load-bearing (−13.5% at optimum vs −2.4% off-peak). Weight-norm growth is FLAT across WD range — WD acts on trajectory, not final parameter scale.
26b. **Stratified per-domain batch sampler** — testing (#2259 fern, NEW). Guarantees all 3 domains in every batch (strict: 1+1+1+1weighted; rotated: 2+1+1 cycling). Tests whether per-batch domain variance is the source of the rc↔sid asymmetry found in #2153.
27. **Decoupled weight_decay per param group (head-down)** — **rejected** (PR #2122, +9.17%/+18.62%). The 10× effective per-step shrinkage on surf_head from coupled WD was PROTECTIVE, not over-regularization. surf_head (tiny 3-layer MLP, 10× LR, noisy residual target) NEEDS more shrinkage. val_geom_camber_rc hit hardest by head_wd↓ (+14 pts) — OOD geometry split is most sensitive to head memorization.
27a. **Decoupled WD head-UP: surf_head_wd∈{1e-3,2e-3}** — testing (#2232 edward, NEW). Symmetric untested direction: raise head WD above encoder's 5e-4. Expected to help OOD splits by suppressing residual memorization further.
28. **Cosine T_max sweep {15, 20, 25}** — testing (#2123 askeladd, NEW). T_max=50 means only 28% of cosine cycle is traversed in 14 epochs — effectively a slowly-decaying constant LR.
29. **Surface-only pressure weight {0.5, 1.5}** — **rejected** (PR #2124, +11.85% / +21.85%). k=1.0 is a sharp local minimum in both directions. Velocity rebalancing mechanism did NOT fire (k=0.5 regressed velocity). Channel-weight axis fully closed across all 5 experiments (#1496 with mean-norm bug, #2124 without).
29a. **SGDR cosine restart T_0=10/7** — testing (#2227 alphonse, NEW). Induced planned spike+recovery based on cycle 34 reframing. If e12 spike is beneficial exploration, cosine restarts should amplify gains. Two arms: single restart at e10 vs periodic restarts every 7 epochs.
30. **surf_head step decay at e10 {×0.5, ×0.3}** — **rejected** (PR #2127, +7.05% / +2.83%). MECHANISM CONFIRMED (clean spike damping observed) but the spike+recovery is a beneficial training dynamic — damping the spike also damps the e14 deep minimum. **Reframing:** the e12 spike is an exploration burst, not pathology.
30a. **Encoder LR boost at e15-18 (dual to head-LR damp)** — testing (#2188 thorfinn, NEW). If the spike comes from encoder entering a new landscape where the head is pre-positioned "too forward," briefly speeding up the encoder during transition might preserve recovery while smoothing the misalignment.
31. **AdamW ε sweep {1e-7, 1e-6}** — **rejected** (PR #2128, +13.13% / +23.08%). Decisive diagnostic: surf_head `frac_below_eps = 0.0` for all epochs. Eps cannot affect surf_head updates because `sqrt(v) >> eps` always. **Denominator-floor mechanism is ruled out as a source of the late-epoch oscillation.** Eps axis is closed.
31a. **AdamW β2=0.9999/0.9995 (longer second-moment timescale)** — testing (#2201 nezuko, NEW). Symmetric untested direction from #2015 (β2=0.95 regressed +6.49%). β2=0.9999 averages v over ~10000-step timescale, a ~10× stronger low-pass filter than current β2=0.999. Tests whether variance-noise interpretation (help) or spike-as-signal interpretation (hurt) prevails. Composes with torch.compile + WD=5e-4.

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
- #2120 (deeper WD 7e-4) — 18.85% val / 18.22% test regression, WD=5e-4 is sharp peak not plateau
- #2127 (surf_head step decay) — 7.05% / 2.83% regression, but key reframing: e12 spike+recovery is BENEFICIAL
- #2013 (LogCosh surface loss) — 3.51% / 14.18% regression, C² smoothness was non-issue, surface-loss family closed
- #2128 (AdamW eps sweep 1e-7/1e-6) — +13.13% / +23.08% regression. surf_frac_below_eps=0 always: eps cannot affect surf_head update shape. Denominator-floor mechanism ruled out entirely.
- #2124 (surface-only pressure weight k=0.5/1.5) — +11.85% / +21.85% regression. k=1.0 is sharp local minimum in both directions. Velocity rebalancing mechanism absent. Channel-weight axis fully closed.
- #2122 (decoupled WD head-down) — +9.17%/+18.62% regression. Reversed hypothesis: head NEEDS 10× effective shrinkage per step (coupled WD is protective). val_geom_camber_rc hit hardest. Head-up direction (#2232) is the untested symmetric point.
- #2153 (WD bracket 4e-4/5.5e-4) — +15.43%/+12.60% regression vs 89.7197 baseline. Key insight: e14 breakthrough load-bearing, weight-norm FLAT. Note: this experiment was at 14-epoch budget; the WD axis at 21 epochs (post-compile) is distinct — WD=3e-4 wins there (#2178).
- #2178 Arm 1 (WD=5e-4 + compile) — +1.17% regression vs 89.7197. WD=5e-4 over-regularizes at 21 epochs, amplifies e12 spike. WD=3e-4 is the 21-epoch optimum.
- #2189 (EMA 21-epoch re-screen) — +9.07%/+16.46% regression. ema_init_bias decay too slow; live model still descending at e21. EMA direction permanently closed across two attempts (#1808 at 14ep, #2189 at 21ep).

## Potential next directions (after cycle 30 in-flight)

- **Re-screen wall-clock-bound rejections at 21 epochs**: EMA (#1808, was +7.8-16.2%), n_head=8 (#1924, was +18.84%), DropPath (#1987, was +3.31%). All were closed for being wall-clock-bound. With torch.compile giving +50% epoch budget, these should be re-evaluated.
- **Cosine restart** (SGDR): single restart mid-training may escape the e12 local basin
- **Discriminative LR decay** (per-layer LR scaling): some encoder layers may be over-regularized
- **n_hidden=256 + FP32** (if torch.compile #2091 unlocks epochs): capacity unlock under budget-neutral throughput gain
- **Cross-channel attention in surf_head**: current head is channel-independent MLP; cross-channel mixing may unlock inter-channel geometry
- **Per-node loss weighting by local curvature/geometry**: leading/trailing edge nodes may dominate MAE
- **Loss-side sampler rebalance with bounded weighting**: tempered 1/var(p) (log-compressed) vs current raw scalar BIVW
