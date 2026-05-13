# SENPAI Research State

- **Date:** 2026-05-13 19:00
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 15-compound stack; 15th compound winner)

**PR #2247 — frieren batch_size 4→2** (merged 2026-05-13 18:45):
- `val_avg/mae_surf_p = 46.6788` (↓ from 50.3812, **−7.35%**, −3.70 absolute)
- `test_avg/mae_surf_p = 39.7696` (↓ from 43.7187, **−9.04%**, −3.95 absolute)
- **All 4 test splits improve sharply.** camber_cruise −10.48% and re_rand −9.13% dominant OOD gains. in_dist −10.05%.
- Config: full 14-compound + **batch_size=2** (n_hidden=192, grad-clip=2.5, T_max=50, huber_beta=0.25)
- Clip rate: 94.70% (DROPPED from 98.93% — first measured saturation easing that wins; 2.06× opt-steps from 25,500 vs 12,375)
- **W&B run:** `t5xloer3`
- **Reproduce:** `cd target/ && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --batch_size 2 --epochs 50`
  (batch_size=2 must be specified; all other 14-compound settings baked into train.py from prior merges)
- **THROUGHPUT NOTE:** at clean throughput (53s/epoch), 34 epochs reached. Result is throughput-dependent — first two runs at GPU contention (109s/ep, 25 ep) landed at val ~56. Compound retests MUST specify `--batch_size 2`.

**Previous baseline — PR #2142** (Huber β=0.25, 14-compound): val=50.3812, test=43.7187.

**TRUE DIRECT MEASUREMENT THRESHOLD: PR #2247 at batch_size=2**: val < 46.6788, test < 39.7696. All new assignments default to `--n_hidden 192 --n_layers 3 --batch_size 2 --epochs 50`.

**CRITICAL: REPRODUCE COMMANDS MUST SPECIFY:** `--n_hidden 192 --n_layers 3 --batch_size 2 --epochs 50`. train.py defaults batch_size=4 which is now the OLD stack.

**Cumulative compounding (14 merges):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise ball at eval |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment in moderate-error band |
| PR #1672 nezuko warmup 1ep | 85.09 | 75.52 | LR warmup → AdamW 2nd-moment stabilization |
| PR #1763 edward torch.compile | 71.44 | 62.59 | 44% speedup → 29 vs 17 epochs in budget |
| PR #1875 frieren n_layers=3 | 69.45 | 61.19 | 35% further speedup + capacity-right-sizing |
| PR #1784 tanjiro grad-clip=10 | 65.98 | 57.07 | Soft-scaling regime damps gradient heavy tail |
| PR #1899 alphonse n_hidden=192 | 63.72 | 55.64 | Width reinvestment — compact+wide beats compact+narrow |
| PR #1930 tanjiro grad-clip=5 | 63.48 | 54.98 | Tighter threshold → 90% clip rate, 4.3× downscaling |
| PR #1953 alphonse T_max=50 + full stack | 55.76 | 48.10 | First full-stack measurement + schedule fix. All 4 splits −12% |
| PR #1982 tanjiro grad-clip=2.5 | 52.64 | 44.98 | Threshold scan step 3. Clip rate 98.9%, 7.1× downscaling. ALL splits improve |
| PR #2023 frieren n_hidden=224 | 53.25* | 46.60* | Width scaling step (measured at grad-clip=5.0 stack; superseded) |
| **PR #2142 fern Huber β=0.25** | **50.38** | **43.72** | Loss-shape tighter MAE alignment. 14th compound. All 4 splits improve. camber_cruise −6.81%, re_rand −4.09% dominant OOD gains |
| **PR #2247 frieren batch_size=2** | **46.68** | **39.77** | Opt-step density 2× (bs=4→2). 15th compound. All 4 splits −8-10%. Clip rate 98.93%→94.70% — FIRST measured saturation easing that wins |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #2219 | n_hidden=160 width-floor compound retest (14-compound stack with huber_beta=0.25) | Architecture (width, floor) | WIP-COMPOUND-RETEST | n_hidden=160 won −4.04% val / −2.99% test vs #1982 old baseline at n=192+huber=0.5 (W&B: `smqwihpd`). But baseline moved when fern #2142 merged: new baseline 50.38/43.72. Need compound retest: does n_hidden=160 still win on top of huber_beta=0.25 baked in? Sent back with reproduce command at new baseline targets. |
| askeladd | #2328 | grad-clip max_norm 2.5 → 3.0 (saturation-exit test, +20% threshold raise) | Optimization (clip threshold, raise) | WIP | #2231 CLOSED — lr=3e-4 FAILED (val 52.77 = +0.25% vs old / +4.74% vs new baseline #2142; test 45.99 = +2.26% / +5.20%; clip rate climbed to 99.38% — OPPOSITE of prediction; student's mechanism correction: raw gradient magnitudes are NOT LR-independent — lower LR ⇒ slower descent ⇒ more time in high-gradient regime ⇒ clip rate rises). LR axis now fully bracketed [3e-4 FAIL / 5e-4 OPT / 7.5e-4 FAIL]. This PR tests student's own follow-up suggestion: directly relax the clip threshold. Six different axes have failed under clip-saturation (n_hidden=224, T_max=80, lr=7.5e-4, mlp_ratio=3, β₂=0.95, lr=3e-4 — all increase clip rate to 99%+). Modest +20% raise from 2.5 → 3.0 interpolates toward known data point grad-clip=5.0 (worse than 2.5 at OLD stack but stack has substantially changed). Predicted clip rate drop 98.93% → 92-95%. If wins, would unlock retest of all 6 saturated axes. Outcomes: (A) val<50.38 win — saturation regime exited; (B) wash, push to 3.5; (C) val>52 fail — 2.5 fundamental, pivot to upstream mechanisms |
| edward | #2024 | EMA decay 0.999 → 0.998 COMPOUND RETEST at bs=2 | Optimization (EMA, post-clip averaging) | WIP-COMPOUND-RETEST | v4 on 14-compound stack (n_hidden=192, grad-clip=2.5, Huber β=0.25): val 49.5940 (−1.56%), test 43.1170 (−1.38%); all 4 splits improve; additive prediction matched exactly (predicted 49.6, got 49.59); clip rate 99.93%; EMA-live gap −3.57 (tightened from v3's −5.84 — live model less noisy under Huber=0.25); mechanism confirmed bias-like not noise-multiplicative (absolute val improvement constant at ~−0.79 across v3→v4 despite changing clip rate). Sent back because frieren #2247 merged (new baseline 46.68/39.77). Now retesting at bs=2 — prediction: val ≈ 45.89, test ≈ 39.17. Must add `--batch_size 2` and verify ema_decay=0.998 |
| fern | #2299 | Huber β=0.1 COMPOUND RETEST at bs=2 | Loss shape | WIP-COMPOUND-RETEST | WON on 14-compound stack: val 48.8982 (−2.94% vs old), test 42.1038 (−3.69%); monotone 4-win β scan confirmed (β=1.0→0.5→0.25→0.1 all win); gain consistent across all 4 splits, cruise −9.59% dominant; clip rate 99.992% confirming upstream-of-gradient bypass; grad_norm/mean only 31.71 (no explosion). Sent back because frieren #2247 merged (new baseline 46.68/39.77). Now retesting at bs=2 stack — additive prediction: val ≈ 45.35, test ≈ 38.29. Must add `--batch_size 2` flag. β=0.1 change in train.py must be verified (grep beta) |
| frieren | #2355 | batch_size=2 → 1 (4× opt-steps from baseline, continue winning density push) | Optimization (opt-step density, continued) | WIP | **#2247 MERGED** (15th compound, −7.35% val / −9.04% test, all 4 splits −8-17%; clip rate 98.93%→94.70% — FIRST measured saturation easing that won; canonical run t5xloer3 at 53s/ep, 34 epochs, 25,500 opt-steps). This PR pushes further: bs=1 = 1500 steps/epoch × ~25 epochs = ~37,500 total opt-steps (~3× baseline, ~1.5× #2247). THROUGHPUT IS KEY RISK — if epoch_time_s >80s abort and report. Clip rate prediction: ~90-92%. Outcomes: (A) val<46.68 win — density continues scaling; (B) wash at bs=1 OPT; (C) val>49 fail — throughput cliff or gradient-noise floor |
| nezuko | #2329 | AdamW eps 1e-8 → 1e-6 — denominator-stability test (100× floor raise) | Optimization (optimizer internal, denominator) | WIP | #2267 CLOSED — slice_num=48 FAILED (val 60.59 = +15.10% vs old / +20.27% vs new baseline #2142; test 52.61 = +16.97% / +20.34%; all 4 splits regress uniformly). Mechanism: compute leg disproved (epoch_time held at 52.96s — slice_num too small a lever at this model size), capacity leg materialized (under-converged + under-capacity). Slice axis now fully bracketed [48 FAIL / 64 OPT / 96 FAIL — third axis fully bracketed]. This PR tests untested optimizer-internal axis: AdamW eps acts on denominator √v_t + eps. Default 1e-8 is very small; 1e-6 is 100× larger floor. Acts as regularization on sparse-gradient parameters (small v_t). Mechanically DOWNSTREAM of clipping and INSIDE the optimizer denominator — orthogonal to all blocked axes. 1500-sample × 0.93M params makes many params low-update — eps=1e-6 prevents these from getting noisy oversized updates. β₂ has been tested (#2186 blocked by saturation) but eps is genuinely untested. Outcomes: (A) val<50.38 win — eps undertuned at 1e-8; (B) wash, close axis; (C) val>52 fail — eps=1e-8 confirmed OPT |
| tanjiro | #2305 | weight_decay 1e-4 → 3e-4 COMPOUND RETEST at bs=2 | Regularization (parameter scale) | WIP-COMPOUND-RETEST | WON on 14-compound stack: val 49.7804 (−1.19% vs old), test 43.1068 (−1.40%); OOD-positive pattern (re_rand −2.72%, camber_cruise −2.07%, camber_rc −1.38%, in_dist +0.07% neutral); clip rate 99.91%→99.86% (decoupled wd confirmed orthogonal to saturation — parameter shrinkage downstream of clipped gradient); EMA-live gap widened to +15.6% (wd-induced trajectory noise absorbed by EMA). Sent back because frieren #2247 merged (new baseline 46.68/39.77). Now retesting at bs=2 stack — additive prediction: val ≈ 46.12, test ≈ 39.21. Must add `--batch_size 2` flag |
| thorfinn | #2276 | n_layers 3 → 2 at n_hidden=192 — depth-axis compact push | Architecture (depth, compact+wide) | WIP | #2186 CLOSED — AdamW β₂=0.95 (val 54.80 = +4.10% / test 47.36 = +5.30% at 99.29% clip rate — FIFTH clip-saturation interaction; mechanism: β₂ shrinking variance-window 1000→20 steps amplified direction noise post-clip, optimizer-internal axis NOT bypassed by saturation as predicted; student diagnosis sharp — clipping renormalizes amplitude but passes direction faithfully so noisier directions land at 2.5/||g|| scale, averaging slower). This PR shifts to architecture axis: untested depth-down direction (n_layers=2). Compute lever: ~30-35% epoch reduction → ~43 epochs/budget vs 33. Composition-depth capacity-down vs throughput gain. Outcomes: (A) val<52.64 win — depth=2 sufficient + extended schedule completion; (B) wash; (C) val>53.5 fail — composition depth bracketed [2, 3(OPT)] |

**Baseline alert**: New baseline is PR #1953 (**val=55.7634, test=48.0960**). All future merges must beat this. WIP PRs running against the older PR #1930 baseline (val=63.48) MUST be rebased and retested with `--epochs 50` (T_max=50). The schedule fix alone is the dominant lift — any experiment without it cannot beat the new baseline.

**Critical mandate**: ALL student reproduce commands MUST now specify `--n_hidden 192 --n_layers 3 --epochs 50` (or `--epochs 80` to match the in-flight schedule push). The PR #1953 merge produced an empty diff — these CLI flags are how students access the 11-compound stack.

## Critical diagnostic: schedule truncation pattern — CONFIRMED MASSIVE

**RESOLVED by PR #1953**: T_max=50 vs T_max=30 (same wall-clock budget) delivered the dominant single-merge improvement in 11 cycles (−12% across all 4 test splits). The schedule axis is now confirmed as the highest-leverage remaining dimension.

**Current status:** model is epoch-saturated, not capacity-saturated. Val slope at #1953 termination was **−0.84/ep at epoch 30/30**. PR #2000 (alphonse T_max=80) is the immediate follow-up to test whether schedule can extract further gains.

**Retroactive implications:** all previously-closed experiments measured under T_max=30 may have hidden viability under the new T_max=50 protocol. Specifically: the asymmetric OOD/IID trade-off observed at grad-clip=5 (in_dist +1.00 vs OOD wins) might reverse under longer effective training.

## Closed hypotheses (all rounds)

### Loss / feature engineering
- **per-channel surface weights (0.5, 0.5, 2.0)** (#1445 v2, nezuko) — +1.4% worse. p already dominates.
- **SiLU activation** (#1648, edward) — +5.0% worse. GELU/lr=5e-4 is well-tuned.
- **surf_weight=30** (#1427 v2, askeladd) — +3.6% worse. Huber β=0.5 already does the work.
- **surf_weight=5** (#1743, askeladd) — +2.05% worse. surf_weight=10 is optimum on primary metric. Sweep bracketed: 5/10/30.

### Regularization / noise
- **Dropout=0.1, 0.05** (#1629 v2/v3, thorfinn) — both +2% worse. β=0.5 sharpened landscape; per-step Bernoulli noise becomes gradient corruption.
- **Gradient clipping max_norm=1.0** (#1534 v2, tanjiro) — +1.6% worse. 100% clipping = normalized SGD with ~22× downscaling. Asymmetric OOD-helps/IID-hurts split. Direction-normalization regime.
- **Lookahead optimizer k=5, α=0.5** (#1783, thorfinn) — +1.39% worse. Competes with EMA for trajectory-smoothing budget; EMA-live gap collapses from −10.5 to −1.6.
- **SGDR cosine warm restarts (T_0=10, T_mult=2)** (#1858, thorfinn) — val=72.99 (+2.17% vs OLD 71.44 baseline; +5.09% vs NEW 69.45 baseline). LR restart mechanism worked exactly as designed. Cycle 1 wasted on converge-then-reset; EMA-tax of ~10 epochs to wash cycle-2 high-LR noise > marginal exploration benefit.
- **Gradient accumulation steps=2 (effective bs=8)** (#1913, thorfinn) — val=75.64 (+8.9% vs old baseline; +18.7% vs current). Mechanism diagnosis was sharp: per-epoch wall time unchanged but opt-steps halved → cosine starves → model undertrained at +0.5/ep at the cap. EMA-live gap DID close to +1.88 (variance reduction effect IS present) but step-count deficit dominates at fixed --epochs. Fixed wall-clock retest mechanically valid but baseline shifted; closed.
- **Refined pattern**: 5 trajectory-shape interventions (dropout, grad-clip 1.0, surf_weight=30, Lookahead, SGDR) failed on β=0.5+EMA stack. But **grad-clip=10 in the soft-scaling regime succeeded** (PR #1784, −7.65%). The failure pattern wasn't about clipping/perturbing per se — it was about direction-normalization (clip 100%, ~22× scaling) and exploration-vs-EMA conflict. Soft proportional damping of the heavy gradient tail (72% clip, ~2.1× typical scaling) preserves bulk direction signal while suppressing outliers. Future smoothing experiments should target this regime, not the high-intensity end.

### Gradient stability / heavy-tail damping (NEW direction — opens after #1784)
- **Gradient clipping max_norm=10** (#1784, tanjiro) — **MERGED, −7.65% val, −6.74% test**. All 4 splits improve. Soft-scaling regime: clip rate 72.4%, typical step downscaling ~2.1×, p99 downscaling 9.2×.
- **Gradient clipping max_norm=5.0** (#1930, tanjiro) — **MERGED, −0.38% val, −1.18% test**. 3/4 splits improve (in_dist regresses +1.00 — first sign of asymmetric OOD/IID trade-off). Clip rate 90.1%, downscaling 4.3×. Predictions matched exactly.
- **Threshold scan in progress**: tanjiro #1982 max_norm=2.5 — regime shift test. (~97% clip, ~8-9× downscaling predicted). Two outcomes: continued gain or U-shape confirmation → bracket at 2.5–5.0.

### LR warmup
- **Warmup 1 epoch** (#1672 v2, nezuko) — MERGED, −0.96%. Mechanism: AdamW 2nd-moment stabilization, not EMA catch-up.
- **Warmup 2 epochs** (#1806, nezuko) — +3.57% worse. Hypothesis prediction directly falsified. Bracketed at 1 epoch.

### Loss shape / Huber β
- **Huber β=0.5** (#1689, fern) — MERGED, −6.96%. Direct MAE alignment in moderate-error band.
- **Huber β=0.25** (#1705, fern) — +9.31% worse. β sweep bracketed: 0.25 < 0.5 (BEST) > 1.0.

### Architecture (capacity-up — all fail)
- **n_layers=8** (#1546, edward) — +24.9% worse. 155 s/epoch; underfitting at budget.
- **mlp_ratio=4** (#1544, alphonse) — +5.3% worse. Over-parameterized for 1500-sample dataset.
- **slice_num=96** (#1550, thorfinn) — +10.4% worse. Confirmed in two runs.
- **n_hidden=192** (#1442 v2, frieren) — +12.5% worse. **On n_layers=5 stack — now retesting on n_layers=3 (#1899).**
- **Pattern**: 4/4 capacity-up fails on n_layers=5 compile stack. Capacity-up direction is closed for n_layers=5; open inversion on n_layers=3.

### Architecture (FFN capacity-down — LOSS)
- **mlp_ratio=1** (#1878, nezuko) — +10.5% vs current 10-compound baseline; +0.99% vs n_layers=3 baseline. **CLOSED.** FFN capacity is not the bottleneck at n_layers=3 + n_hidden=192. mlp_ratio=2 is the optimum for this 1500-sample dataset. Per-axis capacity-down matrix: depth-down WIN, FFN-down LOSS, slice-down in-flight (#1841).

### Attention configuration
- **n_head=8** (#1994, nezuko) — **CLOSED**. val=63.35 (vs new baseline 55.76, **+13.61% regression**). All 4 splits regress slightly uniformly (in_dist +0.054, camber_rc +0.549, camber_cruise +0.289, re_rand +0.279). Mechanism falsified: head_dim=24 is below the bottleneck threshold; PhysicsAttention slice geometry already partitions the input space, further sub-partitioning dilutes signal. n_head=4 (head_dim=48 at n_hidden=192) confirmed local optimum. Attention-head axis now bracketed.

### Architecture (capacity-down + width reinvestment — winning direction)
- **n_layers=3** (#1875 v2, frieren) — MERGED, −2.78% val. 35% throughput boost, param count 0.23× baseline. Val descending at final epoch.
- **n_hidden=192 × n_layers=3** (#1899, alphonse) — **MERGED, −8.25% val, −9.06% test. All 4 splits improve.** 0.93M params, 54 s/epoch. "Compact but wide" hypothesis confirmed: per-layer expressivity was the bottleneck at n_layers=3; wider layers compensate for reduced composition depth. Val still descending at ep 30 → schedule fix in progress.

### LR magnitude
- **lr=7e-4** (#1791, alphonse) — +0.42% worse vs pre-compile baseline. EMA half-life fixed; faster convergence + larger EMA step cancel. LR-magnitude sweep dead.

### Training efficiency
- **EMA without diagnostic pass** (#1626, fern) — within noise. Diagnostic overhead was small.
- **torch.compile(model, dynamic=True)** (#1763, edward) — MERGED, −16.06%. Dominant throughput lever.
- **EMA decay=0.9995** (#1669, edward) — catastrophic (+41 MAE). 3.7-epoch half-life can't settle in budget.

## Potential next directions

### Active experiments (current direct-measurement baseline: PR #1982 at n_hidden=192; val=52.64, test=44.98)

Fleet 8/8 WIP. Width axis bracketed at n_hidden=192 from above (#2066, #2068); alphonse #2219 testing the floor side.

1. **n_hidden=160 width-floor test** — #2219 (alphonse, COMPOUND RETEST at bs=2). Must add `--batch_size 2`.
2. **grad-clip max_norm 2.5 → 3.0 saturation-exit** — #2328 (askeladd). DIRECT TEST of clip-saturation. **Critical: must now add `--batch_size 2`** or will measure on old stack.
3. **EMA decay 0.999 → 0.998 compound retest** — #2024 (edward, COMPOUND RETEST at bs=2). Additive prediction val≈45.89, test≈39.17.
4. **Huber β=0.1 compound retest** — #2299 (fern, COMPOUND RETEST at bs=2). Additive prediction val≈45.35, test≈38.29.
5. **batch_size=1** — #2355 (frieren). Continue opt-step density push. 4× opt-steps from baseline. Throughput risk.
6. **AdamW eps 1e-8 → 1e-6** — #2329 (nezuko). **Must now add `--batch_size 2`** to measure on current stack.
7. **n_layers 3 → 2 depth-axis compact push** — #2276 (thorfinn). **Must now add `--batch_size 2`** to measure on current stack.
8. **weight_decay 1e-4 → 3e-4 compound retest** — #2305 (tanjiro, COMPOUND RETEST at bs=2). Additive prediction val≈46.12, test≈39.21.

**New baseline for all active/pending experiments:** val < 46.6788 / test < 39.7696 (PR #2247). ALL students MUST add `--batch_size 2` to their reproduce commands. Any run at bs=4 measures the old 14-compound stack.

**FLEET-WIDE CRITICAL NOTE:** The 15-compound stack includes batch_size=2 as a BAKED-IN ingredient. Students who were measuring at bs=4 are now measuring the wrong stack. Askeladd (#2328), nezuko (#2329), thorfinn (#2276), alphonse (#2219) may all need to restart with `--batch_size 2` added.

**Sub-optimal-width caveat:** PR #2024 was assigned at `--n_hidden 224` before tanjiro #2066 closed the width axis at 192. Result still informative IF the axis-effect overcomes the throughput penalty. Any winners should be retested at n_hidden=192 to confirm the lift is intrinsic to the axis change, not noise from sub-optimal stack.

### Closed (cycles 27-36)
- #2068 thorfinn n_hidden=256 (runtime-induced epoch deficit; width axis ceiling at 30-min budget)
- #2066 tanjiro n_hidden=224+grad-clip=2.5 compound (val 54.34 = +3.22% / test +4.92% — direct measurement, width axis bracketed at 192 from above)
- #2000 alphonse T_max=80 retest at grad-clip=2.5 (val 55.03 = +4.54% / test 48.05 = +6.84% — clip rate climbed to 99.44%, direction-normalization regime; schedule extension axis closed under current clip threshold)
- #2159 askeladd lr=7.5e-4 raise at grad-clip=2.5 (val 56.73 = +7.77% / test 49.19 = +9.37% on all 4 splits — clip rate climbed to 99.30%, LR amplitude axis blocked by clip saturation; third consecutive clip-saturation interaction)
- #2160 frieren weight_decay=1e-5 (10× REDUCTION from baseline 1e-4 per student catch; replicate-pair mean val 53.52 = +1.66% / mean test 46.38 = +3.11%; inter-replicate spread ±1.1 val pts > effect size — wash/loss on mean; clip rate eased 98.93%→96.67% confirming regularization axis is NOT blocked by saturation but net OOD-negative; 3/4 OOD splits regress)
- #2053 nezuko mlp_ratio=3 retest at n_hidden=224+grad-clip=2.5 (val 56.15 = +6.66% / test 48.53 = +7.91% at 99.57% clip rate — FOURTH clip-saturation interaction; mlp_ratio axis confirmed amplitude-mediated; mechanism real at grad-clip=5.0 v1 −1.70% won, destroyed at threshold=2.5; n_hidden=224 throughput penalty also contributes)
- #2186 thorfinn AdamW β₂=0.95 (val 54.80 = +4.10% / test 47.36 = +5.30% at 99.29% clip rate — FIFTH clip-saturation interaction; first OPTIMIZER-INTERNAL axis confirmed blocked; mechanism: β₂ shrinking variance-window 1000→20 steps amplifies direction noise post-clip — clipping renormalizes amplitude but passes direction faithfully so noisier directions land at fixed 2.5/||g|| magnitude with slower averaging; CONFIRMS clip saturation closes any axis that amplifies update-direction variance, not just amplitude axes)
- #2199 tanjiro --epochs 33 schedule alignment (val 55.10 = +4.66% vs old / +9.36% vs new baseline #2142; test 47.42 = +5.43% / +8.47%; all 4 splits regress with camber_rc worst at +7.19%; mechanism: model still descending at epoch 33 with slope −0.32/ep, forcing LR=0 at termination cuts useful gradient updates; EMA-live gap sign-flipped from baseline −8.32 to +0.41 — diagnostic that LR=0 stops live model progression so EMA catches up; SCHEDULE AXIS FULLY BRACKETED: T_max=33 fail / T_max=50 OPTIMUM #1982-#2142 / T_max=80 fail — U-shape, no remaining schedule headroom under current clip threshold)
- #2267 nezuko slice_num=48 (val 60.59 = +15.10% / test 52.61 = +16.97% vs OLD; vs NEW #2142 baseline +20.27%/+20.34%; all 4 splits regress uniformly; compute leg disproved — epoch_time held at 52.96s confirming slice_num is NOT the wall-time bottleneck at n_hidden=192/n_layers=3; capacity leg materialized — under-converged AND under-capacity at 25 epochs; clip rate climbed 98.9%→99.7% but 0.8pp shift can't explain 17% regression — pure capacity loss; SLICE AXIS FULLY BRACKETED [48 FAIL / 64 OPT / 96 FAIL at n_layers=5 old stack] — third axis to be closed both sides)
- #2231 askeladd lr=3e-4 (val 52.77 = +0.25% wash vs old / +4.74% fail vs new baseline #2142; test 45.99 = +2.26% / +5.20%; clip rate climbed 98.93%→99.38% OPPOSITE of prediction; student's mechanism correction: raw gradient norms NOT LR-independent — lower LR ⇒ slower descent ⇒ more time in high-gradient regime ⇒ MORE clipping not less; pattern v3 refinement: "any perturbation that delays convergence increases clip rate"; LR AXIS FULLY BRACKETED [3e-4 FAIL / 5e-4 OPT / 7.5e-4 FAIL] — fourth axis to be closed both sides; student suggested grad-clip threshold raise as direct next test — taking that suggestion in #2328)

### Clip-saturation interaction pattern (CRITICAL FINDING — SIX confirmed instances; pattern v3 refined to "delayed-convergence amplification")

| PR | Lever | Where it acts | grad-clip | clip rate | val Δ | Verdict |
|---|---|---|---|---:|---:|---|
| #2066 tanjiro | n_hidden=224 | input width (amplitude) | 2.5 | 99.31% | +3.22% | amplitude-axis blocked |
| #2000 alphonse | T_max=80 | LR schedule (amplitude) | 2.5 | 99.44% | +4.54% | direction-only SGD |
| #2159 askeladd | lr=7.5e-4 | LR magnitude (amplitude up) | 2.5 | 99.30% | +7.77% | amplitude over-saturation |
| #2053 nezuko | mlp_ratio=3 | FFN width (amplitude) | 2.5 | 99.57% | +6.66% | FFN amplitude blocked |
| #2186 thorfinn | β₂=0.95 | AdamW 2nd-moment (variance) | 2.5 | 99.29% | +4.10% | direction-variance amplification |
| #2231 askeladd | lr=3e-4 | LR magnitude (amplitude down) | 2.5 | 99.38% | +4.74% (vs new) | slow-descent amplification |

**Pattern v3 (refined from askeladd #2231 mechanism correction):** At grad-clip=2.5 + 98.93% baseline clip rate, clip-saturation closes:
> **Any axis perturbation that increases time-in-high-gradient-regime will increase clip rate and fail.** This includes:
> - Direct amplitude scaling (n_hidden, mlp_ratio): more params → more time before convergence
> - Schedule shifts (T_max=80 extends; T_max=33 truncates): both reshape descent rate
> - LR magnitude in EITHER direction: too high → unstable, too low → slow descent
> - Optimizer variance (β₂=0.95): noisier directions → slower per-direction averaging
> - Capacity-down (slice_num=48): under-fitting keeps gradient magnitudes high

**Mechanism unification:** clipping renormalizes amplitude (passes magnitude as 2.5/||g||) but passes direction faithfully. So axes that *delay convergence* see higher gradient norms throughout training, which means more clipping, more direction-only SGD, less amplitude information for adaptive optimization. The "blocked" set is therefore: **anything that slows trajectory through the high-gradient regime.** This is broader than pattern v2's "amplitude OR direction-variance" framing.

**Inverse-saturation hypothesis (NEW — testing in #2328):** if the clip threshold itself is binding, raising it from 2.5 → 3.0 should release amplitude information and exit saturation. Pattern v3 predicts that if 3.0 wins, the previously-blocked axes may re-open for retesting at the higher threshold.

**Mechanisms expected to bypass saturation (axes that REDUCE direction variance OR operate upstream of optimizer):**
- **DIRECT TEST: grad-clip threshold raise — 2.5 → 3.0** (askeladd #2328 — most direct test of saturation hypothesis, untested at current stack)
- Weight decay tighter — parameter scale, decoupled from gradient (tanjiro #2305 — testing 3e-4)
- Huber β tighter — loss curvature, upstream of gradient (fern #2299 — testing 0.1)
- EMA decay — post-clip averaging (edward #2024 compound retest)
- Width narrowing — frees epoch budget (alphonse #2219 compound retest)
- Opt-step density — more opt-steps per epoch (frieren #2247 batch_size=2)
- Depth-down — model topology, upstream of all gradient computation (thorfinn #2276 n_layers=2)
- **AdamW eps — denominator stability, downstream of clipping** (nezuko #2329 — untested optimizer-internal axis)

**Mechanisms now CONFIRMED blocked (failed under saturation):**
- LR raise (#2159) and LR lower (#2231)
- T_max extend (#2000) and T_max shorten (#2199)
- n_hidden raise (#2066, #2068)
- mlp_ratio raise (#2053)
- AdamW β₂ shorter window (#2186)
- slice_num lower (#2267) and slice_num higher at OLD stack (#1550)
- Weight decay LOOSER (#2160 — net OOD-negative)

### Schedule axis status (post-#2000 closure)

| Stack | T_max | Clip rate | val | Verdict |
|---|---|---:|---:|---|
| grad-clip=5.0 + T_max=50 (#1953) | 50 | ~73% | 55.76 | superseded |
| grad-clip=5.0 + T_max=80 (alphonse 1st run) | 80 | 93.86% | 54.51 | won −2.25% (now stale stack) |
| **grad-clip=2.5 + T_max=50 (#1982 → #2142)** | **50** | **98.93%** | **52.64 → 50.38** | **OPTIMUM** |
| grad-clip=2.5 + T_max=80 (#2000) | 80 | 99.44% | 55.03 | FAIL +4.54% |
| grad-clip=2.5 + T_max=33 (#2199) | 33 | TBD | 55.10 | FAIL +4.66% |

**SCHEDULE AXIS FULLY BRACKETED** — U-shape confirmed with T_max=50 in the trough. Both directions fail symmetrically (extension +4.54%, alignment +4.66%). The schedule lever requires headroom in the clip distribution — when grad-clip pushes the clip rate above ~99%, extending T_max doesn't translate into useful effective LR. And shortening T_max truncates the model's late-phase descent at LR=0, losing useful gradient updates. Direction-only SGD has no benefit from "keep LR high in mid-training" but ALSO requires "keep delivering gradient updates" — the U-shape closes both ends.

### Next after current round
- **If askeladd grad-clip=3.0 WINS** — retest ALL 6 previously-saturated axes at new threshold. n_hidden=224, T_max=80, lr=7.5e-4, mlp_ratio=3, β₂=0.95 may all re-open. This would be a huge follow-up program.
- **If askeladd grad-clip=3.0 LOSES** — pivot to upstream mechanisms: gradient noise injection, Lion/Adafactor optimizer family, RMSNorm vs LayerNorm
- **grad-clip max_norm 3.5** — if 3.0 wins, push further
- **AdamW eps=1e-9** — if nezuko's 1e-6 loses, test smaller eps (more precision)
- **AdamW eps=1e-5** — if nezuko's 1e-6 wins, scan further
- **Batch size=1 or batch_size=8** — bracket opt-step density axis (if frieren #2247 batch_size=2 wins or losses)
- **EMA decay 0.997 or 0.995** — if edward's 0.998 wins, continue scan
- **Huber β=0.05 or pure L1** — if fern's 0.1 wins, continue tightening
- **n_hidden=192 retest of any winner at sub-optimal width** — confirm intrinsic axis lift
- **Compound winners** — once individual axis wins land at n_hidden=192, test best-of-each compound

### Open questions raised by #2066 finding (status update)
- ~~Does T_max=epochs=33 (tanjiro #2199) win?~~ **ANSWERED — NO** (#2199 FAIL +4.66% val). Late-phase low-LR phase IS needed; LR=0 truncates useful descent.
- ~~Does T_max=80 + n_hidden=192 (alphonse #2000 retest) win?~~ **ANSWERED — NO** (#2000 FAIL +4.54%). Schedule extension also fails. **Both directions closed → schedule axis fully bracketed**, U-shape with T_max=50 in the trough.
- ~~Does lr=3e-4 lower-amplitude (askeladd #2231) win?~~ **ANSWERED — NO** (#2231 FAIL +4.74% vs new baseline). Lower LR INCREASES clip rate (slow-descent amplification). LR axis fully bracketed.
- Where is the n_hidden=160 floor? Untested under current stack at the new huber_beta=0.25 baseline (compound retest in flight, #2219).
- **NEW CRITICAL OPEN QUESTION:** Does grad-clip threshold raise 2.5→3.0 (askeladd #2328) exit the saturation regime? If YES, ALL 6 previously-saturated axes may re-open for retesting at the higher threshold. This is the highest-leverage open experiment in the fleet.

### Medium priority
- **n_hidden=160 × n_layers=3** — bracket width from below; is 192 the sweet spot or above it?
- **n_layers=2 × n_hidden=192** — push "compact+wide" further along the depth axis.
- **Grad-clip max_norm=15-20** — bracket the upper side of the threshold.
- **n_layers=3 + n_hidden=192 + slice_num=48** — compound width + capacity-down on slice axis.
- **batch_size=8** — GPU memory freed by smaller model (~46-50 GB for n_hidden=192).

### Longer horizon
- **Re-aware embeddings** — explicit log-Re positional encoding; re_rand split (54.10 at new baseline) is the clearest OOD gap.
- **Surface-aware dual-head decoder** — separate volume/surface heads; surface MAE is the metric.
- **Spectral / Fourier neural operator hybrid** — if attention-based approach plateaus.
- **Systematic retest under new combined stack** — all previously-closed capacity/LR experiments should be reconsidered under n_layers=3 + n_hidden=192 + grad-clip=10 + schedule-fixed.
