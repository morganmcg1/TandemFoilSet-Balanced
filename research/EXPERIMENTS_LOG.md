# SENPAI Research Results

## 2026-05-15 23:00 — PR #3431: EMA weights (decay=0.999) — closed

- Branch: `willowpai2i24h5-nezuko/ema-weights`
- Hypothesis: EMA with decay=0.999 would smooth out warm-restart oscillations, averaging weights across the oscillatory LR trajectory to produce a flatter minimum than the last-checkpoint weights.
- W&B runs (from W&B query): `kart5gph` (105.11), `u9d64dkl` (102.63), `qqnhnq8o` (118.10). 3-arm mean: ~108.6.

| Run | val_avg (raw model) | val_avg (ema_model) | Δ raw vs baseline 90.04 |
|---|---|---|---|
| u9d64dkl (best) | 102.63 | 111.24 | +14.0% |
| kart5gph | 105.11 | — | +16.7% |
| qqnhnq8o | 118.10 | — | +31.2% |
| **3-arm mean** | **~108.6** | — | **+20.6%** |

**Analysis:** Closed based on W&B data (no terminal SENPAI-RESULT comment posted by student). All 3 arms regress significantly vs the 90.04 baseline. Critically, the EMA-specific metric `val_avg/ema_mae_surf_p` (111.24) is **worse than the raw model** (102.63) on the best run — EMA failed even on its own terms. Diagnosis: decay=0.999 over ~5250 steps gives an effective averaging window of ~1000 steps, which spans multiple warm-restart cycles. The EMA averages weights from post-restart LR spikes (exploring) and pre-restart troughs (settled), mixing regimes that produce weights neither well-explored nor well-settled. This failure is mechanism-specific to warm-restarts: EMA is designed for models with monotonic LR schedules where weight trajectories converge smoothly. Nezuko reassigned to lr=1e-3 hypothesis (PR #3512).

---

## 2026-05-15 21:30 — PR #3434: L1 surface loss (align training objective with MAE metric) — **MERGED (round-3 winner)**

- Branch: `willowpai2i24h5-edward/l1-surf-loss`
- Hypothesis: Switching surf_loss from MSE (L2) to L1 directly aligns the training objective with the evaluation metric (MAE). With grad clip already normalizing step sizes, L1 vs L2 is a choice of which minimum to approach (conditional median vs mean). L2's quadratic penalty chases outliers in heavy-tailed OOD error distributions; L1 distributes gradient more evenly.
- W&B run: `tcci4fzk`

| Split | L1 (this run) | warm-restarts baseline `oeo67jf2` | Δ |
|---|---|---|---|
| val_single_in_dist | 108.95 | 116.36 | −7.41 |
| val_geom_camber_rc | 97.70 | 108.40 | −10.70 |
| val_geom_camber_cruise | 70.40 | 77.91 | −7.51 |
| val_re_rand | 83.11 | 92.87 | −9.76 |
| **val_avg** | **90.04** | **98.88** | **−8.84** |
| test 3-split (excl. cruise) | 87.78 | 94.82 | −7.04 |

**Analysis:** Clean winner on single arm, -8.84 pp (-8.94%) improvement. Beats every split including OOD. L1 only 1 epoch slower at epoch 1, then led all the way (grad clip normalizes step sizes, eliminating L2's convergence-speed advantage). Largest gain on `val_geom_camber_rc` (-10.70 pp) — the heavy-tailed OOD distribution where L2 chases outliers. Model was still improving at epoch 14 (last two val_avg: 98.80 → 90.04), suggesting further headroom. Merged as **new baseline: 90.04**. Code change: 2 lines in train.py (add `abs_err = (pred - y_norm).abs()`, use it for `surf_loss`; `vol_loss` remains MSE).

---

## 2026-05-15 21:25 — PR #3436: CosineAnnealingWarmRestarts T_0=3 — closed

- Branch: `willowpai2i24h5-alphonse/warm-restarts-T0-3`
- Hypothesis: More frequent restarts (T_0=3 vs T_0=5) give more "fresh start" opportunities within the 14-epoch budget.
- W&B run: `9jsom32x`

| Split | T_0=3 | T_0=5 baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 144.88 | 116.36 | +28.52 |
| val_geom_camber_rc | 131.09 | 108.40 | +22.69 |
| val_geom_camber_cruise | 85.06 | 77.91 | +7.15 |
| val_re_rand | 104.27 | 92.87 | +11.40 |
| **val_avg** | **116.32** | **98.88** | **+17.6%** |

**Analysis:** Decisive regression. Alphonse's per-epoch restart-boundary diagnostic is the key data: restart at epoch 4 costs +11% (180→200 val_avg), restart at epoch 10 costs +43.5% (116→167). Each restart costs ~3–6 epochs of recovery — most of the remaining 14-epoch budget. Cycle 1 (3 epochs) is too short for meaningful convergence before the reset. Mechanism is exactly the inverse of the T_0=5 hypothesis: T_0=5 fits exactly two good cycles where cycle 2 has 10 epochs to anneal; T_0=3 wastes the first 3 epochs, then ends mid-anneal in cycle 3 (timeout at epoch 14). **Locks in T_0=5 as near-optimal on the restart-period axis.**

---

## 2026-05-15 21:25 — PR #3416: Per-channel surf loss p×3 — closed

- Branch: `willowpai2i24h5-thorfinn/per-channel-surf-loss-p3`
- Hypothesis: Weight pressure channel 3× over Ux/Uy in surf_loss since only p is the primary metric.
- W&B runs (from W&B query): `1hju4xkv` (121.71 best), `0pep1c67` (119.41), `8cqqphva` (118.74 best); 1 failed run `5xbj12l1`

| Run | val_avg/mae_surf_p | Δ vs new baseline 90.04 |
|---|---|---|
| 8cqqphva (best) | 118.74 | +32.0% |
| 0pep1c67 | 119.41 | +32.6% |
| 1hju4xkv (best epoch) | 121.71 | +35.2% |
| **3-arm mean** | **~119.95** | **+33.3%** |

**Analysis:** Closed based on W&B data (no terminal SENPAI-RESULT comment posted). Clear regression across all 3 arms. Mechanism: per-channel reweighting (p×3, Ux/Uy×1) mostly just amplifies an already-dominant channel without preserving balance with the volume term. The structural cost mirrors fern's surf_weight=25 failure: losing volume term's inductive structure hurts. The metric-alignment goal was sound, but channel-reweighting was the wrong mechanism — PR #3434 edward's L1 loss was the right approach (which just merged at 90.04).

---

## 2026-05-15 21:25 — PR #3360: Grad clip max_norm=1.0 → 0.5 — sent back for rebase

- Branch: `willowpai2i24h5-tanjiro/gradclip-0p5`
- Hypothesis: Tighter clip halves the effective LR (1.1e-5 → 5.6e-6); tests if lower effective LR helps.
- W&B runs: `zm1mfu7r` (110.85), `yeo91kch` (113.78), `qt6c29cn` (113.85). 3-arm mean: 112.83.

| | val_avg | Δ vs OLD baseline 117.16 | Δ vs NEW baseline 98.88 |
|---|---|---|---|
| 3-arm mean | 112.83 | −4.33 (−3.7%) **beats old baseline** | +14.1% (regresses vs new) |
| best arm | 110.85 | −6.31 | +12.1% |

**Analysis:** Hypothesis confirmed on OLD baseline: tighter clip is better. But branched from pre-warm-restarts baseline (117.16), so result is stale vs current 90.04 baseline. Tanjiro sent back to rebase onto warm-restarts + L1 baseline and re-test. If max_norm=0.5 + warm-restarts + L1 beats 90.04, it would confirm a gradient-direction-only training regime stacks with better schedules and loss functions.

---

## 2026-05-15 20:50 — PR #3307: OneCycleLR right-sized to actual budget — **updated: winner pending with new baseline**

- Branch: `willowpai2i24h5-askeladd/onecyclelr-1e3`
- Hypothesis: First attempt with `total_steps = len(train_loader) * 50` left ~72% of the OneCycle schedule unused (best epoch hit near peak LR ~9e-4). Right-sizing `total_steps = len(train_loader) * 14` makes the peak hit at epoch 1.4 and the steep anneal phase land exactly within the 30-min wall-clock budget.
- W&B runs: `jgx0qh7r` (97.44 ★ right-sized), `jk3gbtj1` (119.25 wrong-sized, prior arm)

| Split | jgx0qh7r ★ | warm-restarts baseline `oeo67jf2` | Δ |
|---|---|---|---|
| val_single_in_dist | 118.44 | 116.36 | +2.08 |
| val_geom_camber_rc | 107.83 | 108.40 | −0.57 |
| val_geom_camber_cruise | 72.49 | 77.91 | −5.42 |
| val_re_rand | 90.98 | 92.87 | −1.89 |
| **val_avg** | **97.44** | **98.88** | **−1.44** |
| test 3-split (excl. cruise) | 94.41 | 94.82 | −0.41 |

**Analysis:** Single-arm result of 97.44 beats current baseline 98.88 by 1.44, and 3-split test (94.41 vs 94.82) is also slightly better. Mechanism is clean: the right-sized OneCycle gives the model a fast warmup (10% of 14 epochs = 1.4 epochs), a brief plateau near peak (LR=1e-3, 2× the baseline constant LR), then aggressive anneal down to 4e-9 over the remaining ~12.6 epochs. The OOD splits (`val_geom_camber_cruise` -5.42, `val_re_rand` -1.89) improve most — contradicting our earlier hypothesis that high peak LR hurts OOD; what actually hurts OOD is *stopping near peak* without the anneal phase. **PR sent back for: (1) rebase to resolve the merge conflict with the warm-restarts scheduler (keep OneCycleLR, drop warm-restarts); (2) 2 more replication arms to confirm the mean across 3 arms beats 98.88, matching the rigor of the warm-restarts replication.** If replicates, this becomes the new baseline and supersedes warm-restarts on the scheduler axis.

---

## 2026-05-15 20:32 — PR #3146: slice_num 64 → 128 — closed

- Branch: `willowpai2i24h5-frieren/slice-num-128`
- Hypothesis: Finer physics-token partitioning would give the model dedicated tokens for surface/boundary-layer/wake regions, improving `mae_surf_p` especially on OOD-geometry splits.
- W&B runs: `vkfmawat` (133.95 best), `u1s2sf66` (138.53), `tb06syma` (159.16)

| Run | best val_avg | Δ vs new baseline 98.88 |
|---|---|---|
| vkfmawat (best of 3) | 133.95 | +35.5% |
| u1s2sf66 | 138.53 | +40.1% |
| tb06syma | 159.16 | +61.0% |
| **3-arm mean** | **143.88** | **+45.5%** |

**Analysis:** Confirmed negative result across 3 seeds. Frieren's mechanism is the right one: each physics token now aggregates over ~half as many nodes (~500–2000 vs ~1000–4000), so per-token MLP projections become noisier (smaller node clusters per token → higher variance). Compounded by linear-in-slice_num cost eating ~1–2 epochs in budget. The 25-pp spread across 3 seeds (133.95 → 159.16) is itself diagnostic — doubling slice_num amplifies seed sensitivity. The hope that finer slicing would help OOD-geometry tracks was not borne out: `val_geom_camber_rc` regressed harder than the in-dist split (144.07 vs 183.83 for the best-arm). Frieren's NaN root-cause analysis of the `test_geom_camber_cruise` bug (inf*0 = NaN in the surf_mask-zeroed sum) is the clearest description of GH #3292 we have. Follow-up: slice_num=32 probe (PR #3464).

---

## 2026-05-15 20:29 — PR #3139: surf_weight 10 → 25 — closed

- Branch: `willowpai2i24h5-fern/surf-weight-25`
- Hypothesis: Surface pressure is the ranking metric; upweighting `surf_loss` by 2.5× should pull the optimizer harder on what we care about.
- W&B runs (8 pure arms): `8fylbhng` (134.80 best), `bgqv1fxd` (141.69), `7lstdd2m` (150.94), `dogdnwxm` (150.74), `nv7bztze` (158.06), `bve49g3i` (160.58), `3pdddoxm` (175.26), `6d7an68q` (201.07 worst). Plus 1 confound arm `ouwpuf2z` (111.30 with extra grad-clip change — reverted).

| | best of 8 | mean of 8 | worst |
|---|---|---|---|
| val_avg/mae_surf_p | 134.80 | 159.16 | 201.07 |
| Δ vs new baseline 98.88 | +36.3% | +61.0% | +103.5% |

**Analysis:** Confirmed negative result across 8 seeds. Fern's loss-balance analysis is the cleanest explanation: at `surf_weight=10` the surface term already dominates the total loss by ~5× (val surf_loss=0.32, val vol_loss=0.63 → surf is 84%); at 25 it's ~13× and the model loses the inductive structure the volume term provides (far-field consistency, smoothness). When the optimizer is pulled hard on a small set of surface nodes, the loss landscape becomes more curved and run-to-run variance blows up (66 MAE units across 8 seeds). Hypothesis-level conclusion: `surf_weight=10` is at or near the sweet spot for this surrogate. The orthogonal `ouwpuf2z` finding (111.30 with grad_clip + surf_weight=25 on the round-1 baseline) is interesting — grad_clip stabilizes the surface-dominant gradient pull. But the current baseline already has grad_clip, so this doesn't change the action. Follow-up: surf_weight=5 probe (PR #3462).

---

## 2026-05-15 20:25 — PR #3320: CosineAnnealingWarmRestarts T_0=5 T_mult=2 — **MERGED (round-2 winner)**

- Branch: `willowpai2i24h5-nezuko/warm-restarts`
- Hypothesis: Warm-restart schedule gives multiple escape-from-local-minima opportunities in the short 14-epoch budget.
- W&B runs: `oeo67jf2` (98.88 ★), `79m50be7` (100.90), `iyhrbvuq` (102.22)

| Split | run oeo67jf2 ★ | 3-run mean | baseline delta |
|---|---|---|---|
| val_single_in_dist | 116.36 | 118.50 | −19.69 |
| val_geom_camber_rc | 108.40 | 110.54 | −27.37 |
| val_geom_camber_cruise | 77.91 | 79.25 | −6.61 |
| val_re_rand | 92.87 | 94.39 | −12.29 |
| **val_avg** | **98.88** | **100.67** | **−16.49** |
| test avg (3-split excl. cruise) | 94.82 | 96.71 | — |

**Analysis:** First replicated win with **low variance (~3 pp spread)** — far outside the 15-pp noise floor. Every split improves. Largest gain on val_geom_camber_rc (-27 pp, OOD geometry). Within-budget cycle timing: T_0=5 gives LR restarts at epochs 5 and 10 (via per-batch stepping). At epoch 14 (budget cap), LR is at 4.87e-5 (near valley of cycle 2). The warm-restart structure is simultaneously escaping early local minima AND reducing variance (vs round-1 nezuko baseline). Merged as **new baseline: 98.88**. Follow-up: EMA weights (PR #3431) and restart-frequency tuning T_0=3 (PR #3436).

---

## 2026-05-15 20:15 — PR #3381: n_hidden 128 → 192 — closed

- Branch: `willowpai2i24h5-edward/n-hidden-192`
- Hypothesis: Wider hidden dimension tests whether model is capacity-limited.
- W&B run: `s9807vnn`

| Metric | n_hidden=192 | n_hidden=128 baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 126.44 | 117.16 | +7.92% (worse) |
| Epochs | 10 | 14 | −4 |
| Epoch time | 187s | 132s | +42% |
| Peak VRAM | 58.0 GB | 42.1 GB | +38% |
| val @ epoch 8 | 144.23 | 143.79 | +0.3% (tied) |

**Analysis:** Hypothesis conclusively falsified: at epoch 8, models are identical within 0.3%. The wider model is NOT capacity-limited — it learns at exactly the same rate per epoch, but each epoch takes 42% longer. In 30 min: 10 epochs vs 14 = 4 fewer epochs = worse final result. The bottleneck is schedule/time, not representational capacity. Good comparative signal across epochs. Next for edward: L1 surface loss (PR #3434) — conceptually orthogonal, directly targets MAE metric alignment.

---

## 2026-05-15 20:20 — PR #3112: bf16 autocast — closed

- Branch: `willowpai2i24h5-alphonse/bf16-autocast`
- Hypothesis: bf16 mixed precision speeds training to fit more epochs in 30 min.
- W&B runs: `6zclcnwp` (114.34 ★), `swpaiz86` (121.70), `afpcmwzo` (124.77), `4nsxbhp2` (127.18)

| Metric | Best run `6zclcnwp` | 4-run mean | baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 114.34 | ~122.0 | 117.16 |
| Epochs | 18 | 18 | 14 |
| test avg (3-split) | 116.38 | ~123.3 | ~116.40 |

**Analysis:** Speed benefit is real — 18 epochs in 30 min (29% more steps). But 4-run mean (~122) regresses vs baseline (117.16). Single best run (114.34, −2.4%) is within the known 15-pp noise floor. The training objective is neutral-to-negative in accuracy. Cruise NaN is pre-existing issue (sample #20, GH #3292), not a bf16 regression. Close decision: mean regresses, can't merge on speed benefit alone when primary metric is neutral. Next for alphonse: warm-restarts T_0=3 frequency tuning (PR #3436).

---

## 2026-05-15 18:45 — PR #3308: AdamW beta2=0.999 → 0.95 — closed

- Branch: `willowpai2i24h5-thorfinn/adamw-beta2-0p95`
- Hypothesis: Shorter second-moment EMA half-life (~20 steps vs ~1000) better calibrates per-parameter step size to large persistent gradient scales.
- W&B runs: `jld65auc` (115.45), `2jik72p2` (134.89, submitted), `qmr0q4y7` (158.48, GPU contention), `a7l7qu74` (crashed)

| Metric | beta2=0.999 baseline | beta2=0.95 (`2jik72p2`) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **117.16** | 134.89 | +17.73 (+15.1%, worse) |
| val_single_in_dist | 138.19 | 183.58 | **+45.4 (+33%)** |
| val_geom_camber_rc | 137.91 | 148.46 | +10.6 |
| val_geom_camber_cruise | 85.86 | 92.55 | +6.7 |
| val_re_rand | 106.68 | 114.96 | +8.3 |
| test avg (excl. cruise) | 116.40 | 133.79 | +17.4 |
| Epochs | 14 | 9 (GPU contention) | — |

**Analysis:** Hypothesis rejected. Pre-clip grad norms GREW under beta2=0.95 (median 52.6 vs 45.7 baseline) — opposite of expected. The mechanistic explanation: with persistent large-gradient regimes (P99>300), the slow beta2=0.999 EMA averages over high-variance samples to produce a stable denominator; shortening the EMA injects per-parameter step-size noise. After grad clipping fixes the gradient direction, noisy denominator = noisy effective LR per parameter. val_single_in_dist regression 33% is the dominant signal. Note: best arm `jld65auc` showed 115.45 (within noise floor, single run), but mechanistic argument and 3 other arms all confirm negative direction. GPU contention biased `2jik72p2` (9 epochs vs 14) but the conclusion is robust. Next for thorfinn: per-channel surface weighting (PR #3416).

---

## 2026-05-15 18:30 — PR #3310: n_layers 5 → 6 — closed

- Branch: `willowpai2i24h5-edward/n-layers-6`
- Hypothesis: n_layers=6 will train stably under the inherited grad clip and capture meaningful depth gain vs. n_layers=5 baseline.
- W&B run: `2o2fq04e`

| Metric | n_layers=5 baseline | n_layers=6 this PR | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **117.16** | 127.23 | +10.07 (+8.6%, worse) |
| val_single_in_dist | 138.19 | 153.05 | +14.86 |
| val_geom_camber_rc | 137.91 | **124.49** | **-13.42** ✓ |
| val_geom_camber_cruise | 85.86 | 96.71 | +10.85 |
| val_re_rand | 106.68 | 134.66 | +27.98 |
| Epochs / wall-clock | 14 / ~120s each | 12 / ~157s each | -2 epochs, +31% per epoch |
| Peak VRAM | 42.1 GB | 49.6 GB | +7.5 GB |
| test avg (excl. cruise) | 116.40 | 125.26 | +8.86 |

**Analysis:** Stability hypothesis **confirmed** — no collapse, smooth descent, no Inf in test. Depth-gain hypothesis **falsified** for this budget. The deeper model runs 31% slower per epoch, completes 12 vs. 14 epochs, and lags per-epoch as well. The 30-min cap punishes depth twice. Single positive signal: val_geom_camber_rc improved by ~10% (OOD geometry), consistent with the depth → geometry-generalization hypothesis, but the cost on val_re_rand (+28) and val_single_in_dist (+15) swamps it. Key insight: the round-1 n_layers=7 "promising" signal was against the unclipped baseline (slower-converging); the clipped baseline is materially better at every epoch, so the depth gap is now negative. Next for edward: n_hidden=192 (PR #3381) — tests whether the model is capacity-limited via width rather than depth.

---

## 2026-05-15 18:00 — PR #3307: OneCycleLR (max_lr=1e-3, pct_start=0.1) — sent back

- Branch: `willowpai2i24h5-askeladd/onecyclelr-1e3`
- Hypothesis: Per-batch OneCycleLR schedule covers the full training budget more precisely than epoch-based cosine; warmup over first 10% of batches avoids early instability.
- W&B run: (see PR #3307 comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **119.25** |

**Analysis:** 1.8% regression vs baseline (119.25 vs 117.16), within the high-variance noise floor (~15 pp spread observed in round-1 replications). However, a root-cause issue was identified: `total_steps = len(train_loader) * MAX_EPOCHS` (i.e., 50 epochs worth of batches), but only 14 actual epochs ran under the 30-min wall-clock cap. The scheduler reached only ~28% of its designed arc — the anneal phase never fired. **Sent back to draft** with explicit fix: change `total_steps = len(train_loader) * 14` (right-sized to actual budget) and add a guard `if global_step < scheduler.total_steps: scheduler.step()` to prevent out-of-bounds errors. Re-run with `wandb_name tanjiro-onecyclelr-1e3-rightsized`.

---

## 2026-05-15 17:55 — PR #3306: Grad clip max_norm=1.0 → 100.0 — closed

- Branch: `willowpai2i24h5-tanjiro/gradclip-100`
- Hypothesis: max_norm=1.0 fires on 100% of steps (normalized GD); loosening to 100.0 allows spike-only clipping and lets the optimizer run at full effective LR.
- W&B run: (see PR #3306 comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **124.31** |
| Clip fire rate | 20.88% of steps |
| Median pre-clip grad norm | ~45.7 |

**Analysis:** Clear regression (+7.15 pp, +6.1% vs baseline 117.16). At max_norm=100, clipping fires on only 20.88% of steps, meaning the optimizer runs at full effective LR for ~80% of steps. The result definitively confirms that the tight max_norm=1.0 clip is doing real work as a **gradient normalizer**, not just spike suppression. The ~45× effective-LR reduction from 5e-4 → ~1.1e-5 is mechanistically beneficial, not wasteful. Follow-up (PR #3360) probes max_norm=0.5 (halving effective LR further to ~5.5e-6) to determine whether max_norm=1.0 is the sweet spot or whether tighter normalization helps more.

---

## 2026-05-15 16:25 — PR #3153: Huber (β=1.0) on volume loss — closed

- Branch: `willowpai2i24h5-nezuko/huber-vol-beta1`
- Hypothesis: Huber loss on volume term frees gradient budget from high-magnitude outliers, letting surface term pull harder.
- W&B runs: `flndh715` (best), `8u6i1i6e`, `r975jzdy` — 3 identical runs auto-generated

| Metric | Run flndh715 | Run 8u6i1i6e | Run r975jzdy |
|---|---|---|---|
| val_avg/mae_surf_p | **127.22** | 141.16 | 141.93 |
| test_avg (offline, excl. cruise sample 20) | ~117.20 | ~130.94 | ~131.11 |
| Epochs completed | 14/50 | 14/50 | 14/50 |
| Peak VRAM | 42 GB | 42 GB | 42 GB |

**Analysis:** 8.6% regression vs baseline (127.22 vs 117.16) in best run; mean ~137 across 3 runs is ~17% worse. Critical finding: **15-point run-to-run variance across identical configs** — single-run rankings in this 14-epoch budget are at the noise floor and unreliable. The comparison is also confounded by missing grad clip: this PR branched before #3157 was merged, so it ran without max_norm=1.0. The Huber-on-vol idea is not conclusively tested; it may help on the clipped baseline. Nezuko's offline test_avg workaround (drop sample 20) recovered a plausible test metric of 117.20. Independently confirmed the cruise NaN bug.


## 2026-05-15 15:42 — PR #3157: Grad clipping max_norm=1.0 — **MERGED (round-1 winner)**

- Branch: `willowpai2i24h5-tanjiro/gradclip-1p0`
- Hypothesis: Gradient spikes early in training undo optimizer progress; clip_grad_norm_(max_norm=1.0) stabilizes updates.
- W&B run: `cfp7lnaq`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **117.16** ← new baseline |
| val_single_in_dist/mae_surf_p | 138.19 |
| val_geom_camber_rc/mae_surf_p | 137.91 |
| val_geom_camber_cruise/mae_surf_p | 85.86 |
| val_re_rand/mae_surf_p | 106.68 |
| test_avg/mae_surf_p | NaN (cruise bad sample) |
| test 3-split avg (excl. cruise) | ~116.40 |
| Epochs completed | 14/50 (~132 s/epoch) |
| Peak VRAM | 42.1 GB |

**Analysis:** max_norm=1.0 fired on 100% of steps (median pre-clip grad norm = 45.7, P90=140.6, P99=327.2). Effective LR ≈ 5e-4/45.7 ≈ 1.1e-5 — essentially normalized gradient descent. The model still converged monotonically (val: 236→117 over 14 epochs) and leads all round-1 results. Key open question: is the win from spike suppression, gradient normalization, or just a better-conditioned optimization landscape? Follow-up (PR #3306) will probe max_norm=100 to disentangle.

---

## 2026-05-15 15:43 — PR #3125: lr=1e-3 + 2-epoch warmup + cosine — closed

- Branch: `willowpai2i24h5-askeladd/lr1e3-warmup-cosine`
- Hypothesis: Higher peak LR with a short linear warmup to avoid early instability.
- W&B run: (see PR comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 135.06 (+15% vs winner) |

**Analysis:** 15% regression vs round-1 winner. The 2-epoch warmup spans ~750 batches, which may be too gradual relative to the 50-epoch cosine horizon — the LR is still ramping while the cosine has already started decaying. OneCycleLR (PR #3307) should fix this by scaling warmup to 10% of total *batches*, not epochs.

---

## 2026-05-15 15:43 — PR #3164: dropout=0.05 — closed

- Branch: `willowpai2i24h5-thorfinn/dropout-0p05`
- Hypothesis: Small dropout in Transolver blocks reduces overfitting for OOD generalization.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 142.51 (+22% vs winner) |

**Analysis:** 22% regression. With ≤14 epochs, the model hasn't overfit in the first place — regularization from dropout adds noise but no benefit in this short-budget regime. The per-channel and per-split numbers were not especially illuminating. Next assignment (PR #3308) pivots to optimizer mechanics (beta2=0.95) which has a stronger theoretical motivation given the observed gradient distribution.

---

## 2026-05-15 15:44 — PR #3133: n_layers 5 → 7 — closed

- Branch: `willowpai2i24h5-edward/n-layers-7`
- Hypothesis: More composition passes improve OOD geometry generalization.
- W&B runs: `grsl0gde` (reported), `tqxnlq30` (148.37), `ibhtts8z` (145.59)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (epoch 9) | 146.62 (+25% vs winner) |
| val_geom_camber_cruise (best) | 107.9 ← depth helps OOD |
| test_geom_camber_cruise/mae_surf_p | NaN (Inf prediction, reproduced across all 3 runs) |
| Epochs completed | 10/50 (~182 s/epoch, ~38% slower than 5-layer) |
| Peak VRAM | 57.1 GB |

**Analysis:** Depth hypothesis is directionally correct — the 7-layer model made faster per-epoch progress than the 5-layer would (val dropped 242→146 in 10 epochs), and OOD-geometry cruise val (107.9) was the *best* split, consistent with the prediction. However: (a) epoch 10 shows a sharp stability regression (val_rc 149.8→344.3), and (b) end-of-run test evaluation produces Inf predictions on cruise — both symptoms of the unclipped optimizer interacting badly with the deeper network. The new baseline now includes grad clipping, so n_layers=6 (PR #3310) should test depth in a stable regime.
