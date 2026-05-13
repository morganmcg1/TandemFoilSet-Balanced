# SENPAI Research State

- **Date:** 2026-05-13 15:40
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 13-compound stack; direct measurement: 12th compound winner)

**PR #1982 — tanjiro grad-clip max_norm=5.0→2.5 + T_max=50** (merged 2026-05-13 12:00):
- `val_avg/mae_surf_p = 52.6406` (↓ from 55.7634, **−5.60%**)
- `test_avg/mae_surf_p = 44.9791` (↓ from 48.0960, **−6.49%**)
- **All 4 splits improve massively.** in_dist regression from #1930 fully reversed.
- Config: full 11-compound + grad-clip=**2.5**, T_max=50 (n_hidden=192)
- Clip diagnostics: rate 98.9%, mean downscaling ~7.1×, norm_mean=17.85
- **W&B run:** `bb6o68xa`
- **Reproduce:** `cd target/ && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --epochs 50`
  (grad-clip=2.5 now baked into advisor branch train.py from #1982 merge)

**PR #2023 — frieren n_hidden 192→224** (merged 2026-05-13 12:05, NOW SUPERSEDED):
- Measured `val=53.2494` at grad-clip=5.0 stack (protocol-stale).
- **COMBINED STATE PROVEN WORSE THAN n_hidden=192**: tanjiro #2066 directly measured n_hidden=224 + grad-clip=2.5 + T_max=50 = `val=54.3382, test=47.1909` (+3.22%/+4.92% regression vs #1982). The "win" was an artifact of measuring under the OLD clip threshold.
- **Mechanism (confirmed by #2066, #2068, #2113):** throughput-induced epoch deficit. Wider/slower-per-epoch models hit the 30-min cap at lower epoch counts → cosine T_max=50 never fully decays → undertrained.
- **W&B run:** `80b6pnb9` (stale measurement)

**TRUE DIRECT MEASUREMENT THRESHOLD: PR #1982 at n_hidden=192**: val < 52.6406, test < 44.9791. All new assignments default to `--n_hidden 192`.

**CRITICAL: REPRODUCE COMMANDS MUST SPECIFY:** `--n_hidden 192 --n_layers 3 --epochs 50` minimum; `--n_hidden 224` for the wider model. train.py defaults are stale.

**Cumulative compounding (13 merges):**

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
| **PR #1982 tanjiro grad-clip=2.5** | **52.64** | **44.98** | Threshold scan step 3. Clip rate 98.9%, 7.1× downscaling. ALL splits improve, in_dist regression fully reversed |
| PR #2023 frieren n_hidden=224 | 53.25* | 46.60* | Width scaling step (measured at grad-clip=5.0 stack; *combined state unmeasured) |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #2219 | n_hidden=160 width-floor test (throughput-vs-capacity tradeoff at 30-min budget) | Architecture (width, floor) | WIP | #2000 CLOSED — T_max=80 retest at grad-clip=2.5 FAILED (val 55.03 = +4.54%, test 48.05 = +6.84% on all 4 splits — mechanism: clip-rate climbed to 99.44%, direction-normalization regime same as frieren #2067 max_norm=1.5 fail; schedule extension axis closed under current clip threshold). PR #2066 bracketed width axis at 192 from above (224/256 fail on epoch deficit); this PR tests symmetric question from below: does n_hidden=160 win by giving up some capacity for ~47 epochs in budget (vs 33 at 192)? Outcomes: (A) val<52.64 win — narrower+more-epochs is right tradeoff; (B) wash; (C) val>53.5 fail — 192 bracketed [160, 224] |
| askeladd | #2159 | Peak LR 5e-4 → 7.5e-4 (1.5×) — epoch-saturated escape via amplitude | Optimization (LR amplitude) | WIP | #2113 CLOSED (slice=96 val=57.79, +9.77% regression; 7% throughput penalty cut training to 27/50 epochs, slice axis bracketed at 64). Different mechanism from alphonse #2000 T_max=80 (schedule extension) — this tests amplitude scaling. Outcomes: (A) val<52.64 win, model was LR-limited; (B) wash, plateau region; (C) val>54 clip rate hits 100%, direction-norm fail |
| edward | #2024 | EMA decay 0.999 → 0.998 — retest on 13-compound stack | Optimization (EMA) | WIP-RETEST | v1 ran at grad-clip=5.0 (protocol-stale). val=54.22 beat #1953 (-2.77%) but +3.00% vs current #1982. 3/4 OOD test splits improved cleanly. EMA-live gap did NOT close (stayed −8.68 vs predicted ~0) but val/test improved — mechanism reframed: shorter half-life = better noise rejection, not closing live-gap visibility. Retest on n_hidden=224+grad-clip=2.5 stack |
| fern | #2142 | Huber β=0.5 → 0.25 — tighter MAE alignment on 13-compound stack | Loss shape | WIP | #1805 CLOSED (β anneal v3 val=53.92, +2.43% regression; mechanism: grad-clip=2.5 saturates 99.7-100% during high-β phase, removes curvature benefit). Natural follow-up to #1689 (β=1.0→0.5 won −7.4%). Constant β=0.25 has no high-β phase — tests steady-state loss-shape axis at current grad-clip regime |
| frieren | #2160 | Weight decay 0 → 1e-5 — untested regularization axis on 13-compound stack | Optimization (regularization) | WIP | #2094 CLOSED (max_norm=2.0 val=53.11, +0.9% regression; soft shoulder, 2.5 optimum holds. Threshold scan now fully bracketed: 10/5/2.5(WIN)/2.0(soft)/1.5(FAIL)/1.0). Completely different axis from grad-clip work. Tests whether explicit L2 closes train/val gap on 1500-sample dataset with 1.26M params. Outcomes: (A) val<52.64 win; (B) wash, no overfit; (C) val>54 wd too strong |
| nezuko | #2053 | mlp_ratio 2 → 3 — retest on current 13-compound stack | Architecture (FFN width) | WIP-RETEST | v1 ran at grad-clip=5.0 (protocol-stale by #1982). val=54.82 beat #1953 (−1.70%) but +4.13% vs current #1982. 3/4 OOD splits descended (camber_rc/cruise/re_rand all improve); in_dist neutral. Strong mechanism (FFN does more work at shallower n_layers=3) — needs retest at n_hidden=224 + grad-clip=2.5 |
| tanjiro | #2199 | --epochs 33 cosine-schedule alignment with realized 30-min budget | Schedule (alignment) | WIP | #2066 CLOSED with critical finding: n_hidden=224 + grad-clip=2.5 + T_max=50 = val 54.34/test 47.19 (+3.22%/+4.92% regression on all 4 splits). Mechanism: 30-min cap × T_max=50 cosine leaves cosine LR at 26-37% of base at termination — under-converged. This PR tests T_max=epochs=33 alignment: cosine fully decays by realized 33-epoch budget at n_hidden=192. Distinct from alphonse #2000 T_max=80 (opposite direction — extend cosine, keep LR high). Outcomes: (A) val<52.64 win, late-phase low-LR refinement matters; (B) wash; (C) val>53.5 fail, LR=0 final phase hurts |
| thorfinn | #2186 | AdamW betas (0.9, 0.999) → (0.9, 0.95) — beta_2 reduction | Optimization (optimizer adaptation) | WIP | #2068 CLOSED (n_hidden=256 val=54.57 = +3.67%/test 47.48 = +5.55% regression on all 4 splits — mechanism: throughput-induced epoch deficit, run cut at 27/50 epochs by 30-min timeout, T_max=50 schedule never completes decay. Width axis bracketed at 224 within 30-min budget: 192/224(OPT)/256(runtime-fail)). beta_2 reduction makes second-moment estimate react ~50× faster — untested optimizer-tuning axis distinct from LR/grad-clip/wd axes. Outcomes: (A) WIN faster adaptation tracks grad-clip 98.9% saturation regime; (B) FAIL variance too noisy → late-training instability |

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

1. **n_hidden=160 width-floor test** — #2219 (alphonse): symmetric to #2066/#2068 from below. Tests throughput-vs-capacity tradeoff at 30-min budget — ~47 epochs predicted vs 33 at n_hidden=192.
2. **Peak LR 5e-4 → 7.5e-4** — #2159 (askeladd): amplitude-scaling axis at n_hidden=224 (assigned before #2066 finding — may be at sub-optimal width).
3. **EMA decay 0.999 → 0.998 retest** — #2024 (edward): protocol-stale retest at n_hidden=224 stack (sub-optimal width but axis still informative).
4. **Huber β=0.25** — #2142 (fern): steady-state loss-shape axis at current grad-clip regime.
5. **Weight decay 0 → 1e-5** — #2160 (frieren): untested regularization axis at n_hidden=224 (sub-optimal width).
6. **mlp_ratio=3 retest** — #2053 (nezuko): protocol-stale retest at n_hidden=224 stack (sub-optimal width).
7. **AdamW betas (0.9, 0.95)** — #2186 (thorfinn): beta_2 reduction, untested optimizer-adaptation axis at n_hidden=224 (sub-optimal width).
8. **--epochs 33 schedule alignment** — #2199 (tanjiro): cosine fully decays by realized 33-epoch budget at n_hidden=192.

**Sub-optimal-width caveat:** PRs #2159/#2024/#2160/#2053/#2186 were assigned at `--n_hidden 224` before tanjiro #2066 closed the width axis at 192. Their results are still informative IF the axis-effect overcomes the throughput penalty. Any winners should be retested at n_hidden=192 to confirm the lift is intrinsic to the axis change, not noise from sub-optimal stack.

### Closed (cycles 27-29)
- #2068 thorfinn n_hidden=256 (runtime-induced epoch deficit; width axis ceiling at 30-min budget)
- #2066 tanjiro n_hidden=224+grad-clip=2.5 compound (val 54.34 = +3.22% / test +4.92% — direct measurement, width axis bracketed at 192 from above)
- #2000 alphonse T_max=80 retest at grad-clip=2.5 (val 55.03 = +4.54% / test 48.05 = +6.84% — clip rate climbed to 99.44%, direction-normalization regime; schedule extension axis closed under current clip threshold)

### Schedule axis status (post-#2000 closure)

| Stack | T_max | Clip rate | val | Verdict |
|---|---|---:|---:|---|
| grad-clip=5.0 + T_max=50 (#1953) | 50 | ~73% | 55.76 | superseded |
| grad-clip=5.0 + T_max=80 (alphonse 1st run) | 80 | 93.86% | 54.51 | won −2.25% (now stale stack) |
| **grad-clip=2.5 + T_max=50 (#1982)** | **50** | **98.93%** | **52.64** | **OPTIMUM** |
| grad-clip=2.5 + T_max=80 (#2000) | 80 | 99.44% | 55.03 | FAIL +4.54% |
| grad-clip=2.5 + T_max=33 (#2199 in flight) | 33 | TBD | TBD | tests opposite direction |

The schedule lever requires headroom in the clip distribution — when grad-clip pushes the clip rate above ~99%, extending T_max doesn't translate into useful effective LR. Direction-only SGD has no benefit from "keep LR high in mid-training."

### Next after current round
- **n_hidden=192 retest of any winner at sub-optimal width** — confirm intrinsic axis lift
- **AdamW eps=1e-8 → 1e-6** — denominator stability, untested
- **Batch size=2 at n_hidden=192** — more optimization steps per epoch (untested; #1913 used grad-accum which is different)
- **EMA decay 0.997 or 0.995** — if edward's 0.998 wins, continue scan
- **Huber β=0.1** — if fern's 0.25 wins, continue tightening
- **weight_decay=1e-4 or 3e-5** — if frieren's 1e-5 wins, continue scan
- **Compound winners** — once individual axis wins land at n_hidden=192, test best-of-each compound

### Open questions raised by #2066 finding
- Does T_max=epochs=33 (tanjiro #2199) win? If yes, **late-phase low-LR refinement** is the dominant unrealized lift.
- Does T_max=80 + n_hidden=192 (alphonse #2000 retest) win? If yes, **extending high-LR exploration** is dominant. These two are direct opposites — at most one wins.
- Where is the n_hidden=160 floor? Untested under current stack. Might be even better than 192 if throughput frees more epochs.

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
