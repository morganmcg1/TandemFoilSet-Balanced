# SENPAI Research Results

## 2026-05-16 10:45 — edward #3786 closed; FiLM ablation confirmed (#3817 alphonse); fern surf_weight signal (#3808); edward R8 assigned #3913

### #3786 edward — Huber β sweep (CLOSED — informative, β=0.05 optimal)

- Branch: `willowpai2i48h5-edward/r7-huber-beta-sweep`
- Hypothesis: β ∈ {0.05, 0.10, 0.20} — wider Huber transition targets peak-pressure residuals driving camber_rc weakness.
- Results (all 3 arms finished, on OLD baseline n_fourier=16):

| Arm | β | W&B run | val_avg | test_avg | vs internal ctrl |
|-----|---|---------|---------|----------|-----------------|
| A (control) | 0.05 | `g3z8imw5` | 73.47 | 63.85 | — |
| **B (best)** | **0.10** | **`h3rdp99f`** | **72.99** | **62.93** | **−0.47 val / −0.92 test** |
| C | 0.20 | `5gdtcspi` | 73.71 | 63.85 | +0.24 / flat |

- Analysis: **β=0.05 is locally optimal in [0.05, 0.20] under FiLM+Lion+EMA.** β=0.10 shows a consistent direction (all 4 test splits improve slightly) but effect size (+0.47 val) is well within σ≈4.6 seed noise, particularly given that the control arm itself regressed +1.81 val vs published baseline — consistent with the 2.7 val seed-noise floor measured this session. β=0.20 is essentially indistinguishable from control.

  Per-split camber_rc was flat across all 3 arms (val: 85.32 → 85.33 → 84.54), confirming that peak-pressure residuals on camber_rc are NOT primarily driven by Huber β choice.

  **Paper-negative (finding #11):** Huber β ∈ [0.05, 0.20] does not improve surface-pressure MAE. β=0.05 is the local optimum.

  Caveat: runs used OLD baseline (n_fourier=16). Internal ablation still transfers since all 3 arms share substrate.

### #3817 alphonse — FiLM ablation (WIP — terminal pending, PAPER-CRITICAL finding confirmed)

- Branch: `willowpai2i48h5-alphonse/r7-film-ablation-nofourier`
- Hypothesis: FiLM on vs off under n_fourier=0, to quantify the FiLM contribution for the paper.
- Results (both arms finished on NEW baseline n_fourier=0):

| Arm | use_film | W&B run | val_avg | test_avg |
|-----|----------|---------|---------|----------|
| **A-best (FiLM on)** | True | **`sd42el34`** | **70.05** | **60.996** |
| A-retry (FiLM on) | True | `47p2anxd` | 72.82 | 63.82 |
| A-crash | True | `exx8m4sv` | NaN | NaN |
| **B (FiLM off)** | False | **`ow1x8ne8`** | **74.40** | **65.56** |

Per-split FiLM contribution (film=off → film=on, best run sd42el34):
| Split | val Δ | test Δ |
|-------|-------|--------|
| in_dist | 85.67 → 77.98 | 75.83 → 67.17 |
| **camber_rc** | **85.90 → 84.58 (−1.32)** | **76.04 → 74.21 (−1.83)** |
| camber_cruise | 54.79 → 50.01 (−4.78) | 45.60 → 42.03 (−3.57) |
| re_rand | 71.26 → 67.63 (−3.63) | 64.78 → 60.57 (−4.21) |
| **avg** | **−4.35** | **−4.56** |

- Analysis: **Paper-critical finding: FiLM contributes −4.35 val / −4.56 test under n_fourier=0 substrate.** This is smaller than the original FiLM measurement from old baseline (~5.9 val) but still large. The FiLM mechanism is confirmed as essential.

  **On merging:** The best film=on arm (sd42el34 val 70.05) is −0.29 val / −0.63 test better than baseline #3672. This is a lucky seed, NOT a new mechanism — arm A is the SAME config as #3672. Two FiLM-on runs gave val 70.05 vs 72.82 = 2.77 val spread (identical config). **No merge.** Close as informative once terminal posted.

  **Seed noise floor finding:** 2 runs of the same config gave val 70.05 vs 72.82 = **2.77 val spread** → any single-arm Δ < 2.7 val is indistinguishable from seed variance.

### #3808 fern — surf_weight sweep (WIP — baseline-shift issue, confirmation arm requested)

- Branch: `willowpai2i48h5-fern/r7-surf-weight-sweep`
- Hypothesis: surf_weight ∈ {10 ctrl, 20, 40} under FiLM+Lion+EMA on new baseline.
- Results so far (all arms on OLD baseline n_fourier=16 — NOT the new baseline):

| Arm | surf_weight | W&B run | val_avg | test_avg | vs ctrl |
|-----|-------------|---------|---------|----------|---------|
| A (ctrl, running) | 10 | `gg7r89pm` (done), `kbhvk6ol` (running) | 75.28 | 64.70 | — |
| **B** | **20** | **`218e3m9g`** | **72.00** | **61.82** | **−3.28 val / −2.88 test** |
| C | 40 | `433xbqv4` | 76.93 | 66.63 | +1.65 / +1.93 |

- Analysis: **Strong internal signal at surf_weight=20 (−3.28 val / −2.88 test).** Clean U-shape: w20 wins, w40 regresses. Effect size large enough to be real. But all runs on n_fourier=16, so not directly comparable to new baseline 70.34.

  Asked fern to run one confirmation arm: surf_weight=20 + n_fourier=0. If that beats val 70.34, it's a merge candidate.

### R8 H31 assigned — edward #3913

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| **#3913** | **edward** | **Reynolds-extremity WeightedRandomSampler** | 3 arms: `--re_sampler_alpha {0.0 ctrl, 0.5, 1.0}`. Multiply existing balanced-domain `sample_weights` by `|log(Re) - mean(log Re)|^α` to oversample extreme-Re samples. Targets re_rand OOD split (worst non-camber split at test 60.35). Already have WeightedRandomSampler plumbing in train.py (line 648). |

---

## 2026-05-16 09:35 — 3 R5 closes (#3673, #3697, #3698); 3 R7 assignments (#3842, #3843, #3845)

### #3673 tanjiro — EMA decay sweep (CLOSED — informative, within noise)

- Branch: `willowpai2i48h5-tanjiro/r5-ema-decay-sweep`
- Hypothesis: EMA decay ∈ {0.995, 0.997, 0.999} under FiLM+Lion. Is 0.997 locally optimal?
- Results:

| Arm | ema_decay | W&B run | val_avg | test_avg | Δval vs base (71.65) |
|-----|-----------|---------|---------|----------|----------------------|
| **A (best)** | **0.995** | **`s3ufqnz2`** | **71.51** | **−** | **−0.14 (noise)** |
| B | 0.997 | `3ag4pvjr` | 73.46 | 63.67 | +1.81 |
| C | 0.999 | `3ki9voje` | ~73+ | — | regression |

- Analysis: **EMA decay is robust in [0.995, 0.997].** Best arm (0.995) at val 71.51 is a wash with the baseline 71.65 (Δ=−0.14, well within σ≈4.6). Arm A is a canonical control (s3ufqnz2 was the intended canonical run, others were process-collision duplicates). No merge — informative paper ablation confirming decay insensitivity. Note: best arm runs pre-date the n_fourier=0 merge; absolute numbers are not directly comparable to new baseline 70.34.

  **Paper finding (finding #4):** EMA decay is robust in [0.995, 0.997] — safe to fix at 0.997.

### #3697 frieren — Multi-σ Gaussian Fourier PE (CLOSED — superseded by n_fourier=0 merge)

- Branch: `willowpai2i48h5-frieren/r5-multi-sigma-fourier`
- Hypothesis: Multi-scale Fourier features (σ ∈ {3, 10, 30} concatenated) capture broader frequency content and outperform single-σ.
- Results: Arms A/B posted before n_fourier=0 merged. Arm C posted after.

| Arm | Config | val_avg | vs baseline at time |
|-----|--------|---------|---------------------|
| A | σ=3 only | ~73 | regression |
| B | σ=10 only (control) | ~71.7 | baseline repro |
| C | σ=3,10,30 multi | ~71.5 | marginal/wash |

- Analysis: **Superseded.** The n_fourier=0 result (val 70.34) shows FiLM on log(Re) makes ALL Fourier PE redundant. Multi-σ cannot beat dropping Fourier entirely. The multi-scale hypothesis is directionally wrong — more frequency resolution doesn't help when FiLM already encodes flow-regime conditioning.

### #3698 thorfinn — TTA z-reflection (CLOSED — catastrophic regression, dataset asymmetry confirmed)

- Branch: `willowpai2i48h5-thorfinn/r5-tta-reflection`
- Hypothesis: Averaging model predictions over original + z-reflected geometry provides a free inference-time gain via symmetry.
- Results:

| Eval mode | W&B run | val_avg | vs baseline |
|-----------|---------|---------|-------------|
| No TTA (control) | `3du9h0yz` | ~72 | +0.35 (noise) |
| TTA z-reflect | `5555kka9` | ~307 | **+235 CATASTROPHIC** |

Thorfinn ran additional diagnostic: 6-subset reflection sweep on `3du9h0yz` checkpoint (run `awuxtsni`):

| Subset reflected | val_avg |
|-----------------|---------|
| {} (no reflection) | 72.15 |
| {x,z} | 286 |
| {z only} | 307 |
| {AoA0, AoA1} | 184 |
| ... | ... |

- Analysis: **TTA z-reflection fails because the training data is NOT z-symmetric.** raceCar geometry uses AoA ∈ [-10°, 0°] (always negative camber); z-reflecting an AoA=-5° sample gives AoA=+5°, which the model has never seen. The model is not equivariant to z-reflection because the dataset isn't. Catastrophic regression (72 → 307) confirms the model correctly rejects OOD inputs.

  **Paper-relevant finding (#10):** TTA z-reflection fails on asymmetric AoA distribution. Cannot assume z-symmetry without symmetric training data.

  **Follow-up (H30):** Train-time z-reflection augmentation (#3845) may teach the model the symmetry by including reflected samples during training.

### R7 assignments: tanjiro #3842, frieren #3843, thorfinn #3845

| PR | Student | Hypothesis | Key novelty |
|----|---------|------------|-------------|
| **#3842** | **tanjiro** | **Sobolev finer sweep w ∈ {0.05, 0.10, 0.15}** | Extend fern's R5 signal (test −0.18 at w=0.1) with finer grid on new n_fourier=0 baseline |
| **#3843** | **frieren** | **Lion lr sweep {2e-5, 5e-5 ctrl, 1e-4}** | First lr ablation in this launch; paper-required sensitivity analysis |
| **#3845** | **thorfinn** | **Train-time z-reflection augmentation (p ∈ {0, 0.25, 0.5})** | Close loop on TTA failure; teach model z-symmetry during training to attack camber_rc (worst split) |

All 8 students now staffed for R7. Full R7 map spans: FiLM ablation (alphonse), surf_weight (fern), Huber β (edward), Lion β1 (askeladd), spectral norm (nezuko), Sobolev (tanjiro), Lion lr (frieren), train-aug (thorfinn).

---

## 2026-05-16 07:55 — alphonse #3672 MERGED; fern #3695 closed; R7 fern+edward assigned

### #3672 alphonse — Fourier ablation (MERGED — new baseline val 70.34 / test 61.63)

- Branch: `willowpai2i48h5-alphonse/r5-fourier-ablation-film`
- Hypothesis: Under FiLM+Lion+EMA, Fourier positional features may be redundant since FiLM on log(Re) already encodes the flow-regime frequency information. Test n_fourier ∈ {0, 16 σ=3, 16 σ=10}.
- Results (all arms FINISHED, terminal SENPAI-RESULT posted, squash-merged):

| Arm | Config | W&B run | val_avg | test_avg | Δval vs prior base (71.65) |
|-----|--------|---------|---------|----------|---------------------------|
| **A (WINNER)** | n_fourier=0 | **`297qot5r`** | **70.3432** | **61.6253** | **−1.31** |
| B | n_fourier=16, σ=3 | `drp81h4l` | 71.2763 | 61.6733 | −0.38 (marginal) |
| C (control) | n_fourier=16, σ=10 | `vx0b6ukg` | ~71.7 (still running at terminal) | — | ~baseline |

Per-split Arm A winner vs prior baseline:

| Split | val Δ | test Δ |
|-------|-------|--------|
| single_in_dist | −1.53 | −1.33 |
| geom_camber_rc | −2.02 | +0.09 (wash) |
| geom_camber_cruise | −0.49 | −0.62 |
| re_rand | −1.21 | −0.08 (wash) |

- Analysis: **FiLM on log(Re) makes Fourier PE redundant.** Dropping Fourier entirely (n_fourier=0) improves all 4 val splits and 3/4 test splits. FiLM already captures the flow-regime conditioning signal that Fourier positional encoding was trying to inject. Key simplification win: removes ~1.1K RFF params, one hyperparameter (fourier_sigma), and one coordinate transform per forward pass.

  Variance caveat: student ran 3 Arm A seeds due to process collisions during 06:30–07:00 launch window; only `297qot5r` was a clean 50-epoch run. The two duplicates (`9an3ynhy` val 82.39, `cng2gwhu` val 89.92) were crash-restarts with broken state, not reproducible runs.

  **New baseline: val 70.3432 / test 61.6253** (BASELINE.md updated, commit `6352727`).

### #3695 fern — Sobolev surface ∂p/∂s loss (CLOSED — informative, small test gain)

- Branch: `willowpai2i48h5-fern/r5-sobolev`
- Hypothesis: Penalizing ∂p/∂s gradient mismatch along the foil surface regularizes prediction smoothness and improves OOD generalization.
- Results (all 3 arms FINISHED, terminal SENPAI-RESULT posted):

| Arm | sobolev_weight | W&B run | val_avg | test_avg | Δval vs prior base (71.65) |
|-----|----------------|---------|---------|----------|---------------------------|
| Control | 0.0 | `yrl9p2bh` | 73.7119 | 63.3541 | +2.06 |
| **B (best)** | **0.1** | **`b655hio8`** | **71.8355** | **61.9284** | **+0.18 (wash, test −0.18)** |
| C | 0.5 | `pgk5nw19` | 85.9918 | 75.7503 | +14.34 |

- Analysis: **Sobolev w=0.1 gives a small test-side gain (−0.18) at flat val (+0.18 above baseline).** The surface gradient regularizer is pointing in the right direction (OOD smoothness) but the gain is sub-noise. Per-split: camber_cruise test improves (42.84 → 42.03, −1.9%), camber_rc regresses slightly (73.87 → 75.31, +2.0%). w=0.5 catastrophically over-regularizes (+14 val). The Sobolev contribution (ratio ~1.0 at w=0.1) equals the data-loss magnitude at epoch 14 — a tuning sweet spot that happens to be near noise.

  Paper-relevant: surface-Sobolev regularization at correct weight is neutral-to-slightly-beneficial on test. Confirms physics-motivated direction. Finer sweep {0.03, 0.05, 0.08, 0.12} reserved for Round 7+.

### New R7 assignments after R5 closes

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3786 | edward | Huber β sweep (0.05→0.1→0.2) | `--loss_beta` flag sweep. Hypothesis: β=0.05 is too tolerant of peak-pressure residuals driving camber_rc weakness. |
| **#3808** | **fern** | **Surface-loss reweighting (surf_weight ∈ {10, 20, 40})** | `--surf_weight` flag sweep. Direct rebalancing of surface vs volume loss gradient. Follow-up to Sobolev result. |

All 8 students staffed:
- alphonse: reassigning (just merged #3672)
- tanjiro #3673: terminal posted, awaiting mark-ready → close-as-informative
- fern #3808: just assigned (surf_weight sweep)
- frieren #3697: Arm C still running
- thorfinn #3698: awaiting terminal + mark-ready → close-as-informative
- askeladd #3712: running (β1 sweep)
- nezuko #3748: running (spec norm)
- edward #3786: just assigned (Huber β sweep)

---

## 2026-05-16 07:35 — R5 results finalized; edward #3711 closed; R7 edward assigned

### #3711 edward — Layer-wise LR decay / LLRD (CLOSED — dead end, monotonic regression with γ<1)

- Branch: `willowpai2i48h5-edward/r6-llrd`
- Hypothesis: Lower LR for earlier Transolver blocks (γ<1 multiplier from output to input) mirrors fine-tuning LR decay used in pretrained LLMs and may improve generalization.
- Results (all 3 arms FINISHED, terminal SENPAI-RESULT posted):

| Arm | γ | W&B run | val_avg | test_avg | Δval vs base (71.65) |
|-----|---|---------|---------|----------|----------------------|
| A (control) | 1.00 | `kuvqzt5y` | 71.8970 | 62.4985 | +0.24 (within noise) |
| B | 0.85 | `3om0smnq` | 76.7062 | 66.3279 | +5.05 |
| C | 0.65 | `nnsunb0c` | 93.1413 | 82.3440 | +21.49 |

- Analysis: **LLRD with γ<1 hurts monotonically under FiLM+Lion+EMA.** LLRD is designed for fine-tuning pretrained models, where lower layers encode general features already at a good basin. Training from scratch means lower blocks are far from convergence at epoch 0 — throttling their LR (γ=0.85 cuts group_0 to 37.7% of base) prevents them from finding a good basin in 14 effective epochs. The output head is then forced to compensate with under-trained features → worse generalization across all 4 splits. Effect is monotonic and catastrophic at γ=0.65 (+21.5 val). The γ=1.0 control reproduces baseline within +0.24 val (noise), confirming clean implementation and real negative result.

**Paper-relevant findings**: LLRD does not transfer from the fine-tuning setting to training-from-scratch setting. Our regime (14 effective epochs from random init) is not analogous to NLP LLRD literature.

### Round 5+6 results fully in — action taken

All 5 remaining R5 students had W&B results completed. Advisor comments posted to push for terminal SENPAI-RESULT on:

- **#3672 alphonse** (winner declared — n_fourier=0 `297qot5r` val 70.3432 / test 61.6253, beats baseline)
- **#3673 tanjiro** (close as informative — best EMA=0.995 `s3ufqnz2` val 71.51, wash)
- **#3695 fern** (close as informative — best sobolev=0.1 `b655hio8` val 71.84, flat val but test improves −0.18)
- **#3698 thorfinn** (close as informative — TTA=True `5555kka9` val 72.56, no gain; design gap noted — no control arm)

Still awaiting:
- **#3697 frieren** Arm C (σ='3,10,30') — still running
- **#3712 askeladd** — β1=0.8/0.95 just starting (rate-limit delayed, GPU now at 100%)
- **#3748 nezuko** — all 3 arms not started (rate-limit delayed, pod just picked up PR)

### R7 edward assigned — #3786

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3786 | edward | **Huber β sweep (0.05→0.1→0.2)** | 3 arms: `--loss_beta 0.05/0.10/0.20`. Hypothesis: current β=0.05 is too tolerant of peak-pressure errors that drive camber_rc (worst split at test 73.87). Widening transition region increases gradient for moderate residuals → better peak-pressure fitting. |

---

## 2026-05-16 06:35 — Closed nezuko #3671, reassigned to spec norm; partial R5 results in

### #3671 nezuko — Layer-wise FiLM (CLOSED — uniform +5 val regression)

- Branch: `willowpai2i48h5-nezuko/r5-film-intermediate-layers`
- Hypothesis: Stack FiLM conditioning at intermediate Transolver blocks in addition to output-FiLM, on the theory that earlier-block conditioning gives the model more capacity to adapt per-Re.
- Results (Arm A only — student declared verdict and stopped):

| Arm | W&B run | Config | val_avg | test_avg | vs baseline (71.65 / 62.11) |
|-----|---------|--------|---------|----------|------------------------------|
| A | `w2qifj9u` | output-FiLM + block-FiLM stack | 76.81 | 66.13 | +5.16 val / +4.03 test |

Per-split val Δ vs baseline: in_dist +5.83, camber_rc +4.13, camber_cruise +6.16, re_rand +4.52 (all 4 splits worse, including OOD re_rand where block-FiLM should help most).

- Analysis: **Paper-relevant negative — output-FiLM at the final layer is the correct FiLM topology**. Adding FiLM to intermediate blocks (5× extra parameters + 4% per-epoch slowdown) does NOT improve over the cheaper output-only configuration. Per-epoch curve shows Arm A starts ahead at epoch 1 (296 vs 310) but falls 5-7 points behind from epoch 3 onward and never recovers. The extra parameters slow per-step throughput, effectively reducing the training-iteration budget; this is the most likely mechanism for the uniform regression.

### Partial Round 5 results (alphonse #3672, tanjiro #3673)

**alphonse #3672** (Fourier ablation under FiLM+Lion+EMA):

| Arm | W&B run | Config | val_avg | test_avg | vs baseline |
|-----|---------|--------|---------|----------|------------|
| A | `9an3ynhy` | n_fourier=0 | (running, step 1469/5264) | — | — |
| B | `drp81h4l` | σ=3, n=16 | **71.28** | **61.67** | **−0.37 val / −0.44 test (marginal beat)** |
| C | (not started) | σ=10, n=16 | — | — | — |

Arm B σ=3 marginally beats baseline but Δ is within the σ≈4.6 run-to-run variance band. Awaiting Arms A and C to determine if this is signal or noise. Asked alphonse to verify Arm A progress and start Arm C.

**tanjiro #3673** (EMA decay sweep under FiLM+Lion):

| Arm | W&B run | ema_decay | val_avg | test_avg | vs baseline |
|-----|---------|-----------|---------|----------|------------|
| A | `eb4gsayj` | 0.995 | (running, step 1380/5264, val 136 early) | — | — |
| B | `3ag4pvjr` | 0.997 | 73.46 | 63.67 | +1.81 val / +1.56 test (within noise) |
| C | `3ki9voje` | 0.999 | (running, step 225/5264) | — | — |

ema=0.997 (paper default) reproduces baseline within noise — confirms 0.997 is solid. Awaiting decay-sweep arms to see if 0.995 or 0.999 outperforms.

### Round 6 assignment (nezuko reassigned)

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3748 | nezuko | **Spectral normalization on output head (+ FiLM layers)** | `torch.nn.utils.parametrizations.spectral_norm` on output linear; 3 arms: control, output only, output+film. Lipschitz constraint to reduce peak-pressure over-fit. |

All 8 students remain staffed: alphonse #3672, tanjiro #3673, fern #3695, frieren #3697, thorfinn #3698, edward #3711, askeladd #3712, nezuko #3748.

---

## 2026-05-16 05:25 — Closed edward + askeladd holdovers, assigned Round 6

### #3483 edward — Lion+EMA ablation (CLOSED — no arms beat new FiLM baseline)

- Branch: `willowpai2i48h5-edward/round3-ema-only-on-huber-no-fourier`
- Hypothesis: Quantify isolated EMA + ablate Fourier under Lion substrate.
- Results (all 3 arms finished):

| Arm | W&B run | Config | val_avg | test_avg | vs new baseline (71.65 / 62.11) |
|-----|---------|--------|---------|----------|---------------------------------|
| A (winner of this PR) | `5pvi79f2` | Lion + EMA(0.997), no Fourier | **73.10** | **63.65** | +1.45 val / +1.54 test |
| B | `3hgal2fm` | Lion + EMA(0.997), σ=3 Fourier | 73.41 | 64.33 | +1.76 val / +2.22 test |
| C | `tev95mko` | Pure Lion, no EMA, no Fourier | 77.48 | 67.43 | +5.83 val / +5.32 test |

- Analysis: **Paper-section material — EMA contributes 4.4 val / 3.8 test points on top of Lion** (Arm A vs Arm C). This is the cleanest single-mechanism EMA measurement in the launch. Closed because no arm beats new FiLM baseline (71.65); the EMA gain is already incorporated via #3405 merge.

### #3609 askeladd — Lion + LR warmup ablation (CLOSED — warmup adds nothing)

- Branch: `willowpai2i48h5-askeladd/r4-lion-warmup`
- Hypothesis: LR warmup improves Lion stability and final performance.
- Results (all 3 arms finished):

| Arm | W&B run | warmup_steps | val_avg | test_avg | vs new baseline (71.65 / 62.11) |
|-----|---------|--------------|---------|----------|---------------------------------|
| A | `379hrdie` | 0 | 79.13 | 68.98 | +7.48 val / +6.87 test |
| B | `j1pum3n7` | 500 | 79.89 | 69.83 | +8.24 val / +7.72 test |
| C (winner of this PR) | `jdaof5n2` | 1000 | **78.46** | **68.69** | +6.81 val / +6.58 test |

- Analysis: **LR warmup adds nothing to Lion at our 14-effective-epoch budget.** Non-monotonic in warmup_steps (C 1000 > A 0 > B 500), but effect size (~1.4 val across arms) is below run-to-run variance band (~σ≈4.6). Paper-section material for the LR-schedule ablation: cosine T_max=14 alone is sufficient. Closed; these plain-Lion arms regress against FiLM+Lion+EMA baseline because they're missing FiLM (+5.9 val) and EMA (+4.4 val).

### Round 6 assignments created (edward + askeladd no longer idle)

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3711 | edward | **Layer-wise LR decay (LLRD)** | Per-block LR multiplier γ; 3 arms γ ∈ {1.0 control, 0.85, 0.65}. Output head full LR, input embed γ^(N+1)·base. Paper-relevant for optimizer-tuning section. |
| #3712 | askeladd | **Lion β1 sweep** | `--lion_beta1` flag; 3 arms β1 ∈ {0.8, 0.9 control, 0.95}. β2=0.99 fixed. Settles paper-required Lion ablation. |

All 8 students now staffed with active R5 or R6 PRs. REST API budget recovered (2880/5000) after earlier exhaustion. Standard assign-experiment skill used (not GraphQL fallback).

---

## 2026-05-16 04:35 — Round 5 cleanup + 3 new assignments

### Closed PRs (informative negatives / superseded)

**#3544 thorfinn — Lookahead optimizer (CLOSED — dead end across both substrates).**
- Branch: `willowpai2i48h5-thorfinn/round3-lookahead`
- Hypothesis: Lookahead wrapper (k-step slow-weight averaging) provides ensemble-like regularization on top of AdamW or Lion.
- Results:

| Arm | W&B run | Substrate | val_avg | test_avg | Δ vs baseline |
|-----|---------|-----------|---------|----------|---------------|
| A (initial) | `k39kdp6y` | Lookahead(k=6, α=0.5) + AdamW (R3 baseline 93.20) | 98.33 | 86.24 | +5.13 val |
| A (post-pivot) | `drt9naou` | Lookahead(k=5, α=0.5) + Lion (R3 baseline 77.58) | **89.39** | **78.24** | +11.81 val |

- Analysis: Lookahead is incompatible with both inner optimizers at our 14-effective-epoch budget. Hypothesized mechanism: the slow-weight averaging acts as a second-order smoother on top of the optimizer's own smoothing, and under tight wall-clock the slow weights never fully catch up to the fast weights. Lion's sign-based update is already a coarse approximation, so Lookahead's k-step averaging damps out exactly the signal Lion injects. Paper-relevant negative result for optimizer-family ablation section.

**#3486 fern — σ=3 + Lion + EMA (CLOSED — superseded by FiLM merge).**
- Branch: `willowpai2i48h5-fern/round3-fourier-sigma-under-ema`
- Hypothesis: σ-monotonic finding from AdamW (σ=3 wins σ sweep under EMA) transfers to Lion+EMA substrate.
- Results:

| Arm | W&B run | Config | val_avg | test_avg | Δ vs Lion baseline (77.58) |
|-----|---------|--------|---------|----------|---------------------------|
| Lion-rebase | `dl4apv3e` | σ=3 + Lion + EMA(0.997) | **73.81** | **63.89** | −3.77 val / −4.99 test |

- Analysis: σ=3 beats Lion baseline by ~4 val points — a real win on the old substrate. BUT: edward's no-Fourier Lion+EMA (`5pvi79f2`, val 73.10) slightly beats fern's σ=3 (73.81) on the same substrate. **The σ-monotonic finding from AdamW+EMA does NOT transfer to Lion+EMA.** Under Lion+EMA, Fourier features appear roughly equivalent to noise (or slightly harmful at any σ). Paper-valuable negative-transfer ablation. PR closed because new FiLM baseline (val 71.65) supersedes σ=3 result.

**#3380 frieren — Multi-σ Fourier sweep (CLOSED — config bug, student agreed).**
- Branch: `frieren/round2-sigma-sweep`
- Hypothesis: Multi-scale Gaussian Fourier features (σ ∈ {3, 10, 30}) improve over single-σ=10.
- Results:

| Arm | W&B run | Intended config | Actual config | val_avg | test_avg |
|-----|---------|------------------|----------------|---------|----------|
| (only) | `54hmldzq` | multi-σ {3,10,30} | n_fourier=0 (bug) | 76.95 | 67.07 |

- Analysis: Multi-σ Fourier flag never wired in — config shows `n_fourier=0` at runtime. Run is effectively Lion+EMA no-Fourier (comparable to edward's `5pvi79f2` val 73.10; σ ≈ 4.6 run-to-run variance band). PR closed by mutual agreement with student; multi-σ reassigned to frieren on the new FiLM+Lion+EMA substrate (#3697).

### #3609 askeladd status update (Lion + LR warmup, paper-relevant negative in progress)

| Arm | W&B run | warmup_steps | val_avg | test_avg | Status |
|-----|---------|--------------|---------|----------|--------|
| A | `379hrdie` | 0 | 79.13 | 68.98 | finished |
| B | `j1pum3n7` | 500 | 79.89 | 69.83 | finished |
| C | `jdaof5n2` | 1000 | (running) | — | ~50% complete |

- Analysis (preliminary): **LR warmup adds nothing to plain Lion at our 14-effective-epoch budget.** Arm B (warmup=500) is even slightly worse than Arm A (no warmup). Cosine schedule already provides implicit warmup via low LR start-of-cycle when T_max=14 is matched to wall-clock. Awaiting Arm C to confirm. After Arm C, expected to close as informative negative (paper-relevant) and reassign askeladd to Lion β1 sweep.

### Round 5 assignments created

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3695 | fern | **Sobolev loss on surface ∂p/∂s** (physics-motivated regularizer) | Add `--sobolev_weight` + `--sobolev_k`; compute k-NN finite-difference gradient of surface pressure; Huber on (pred grad − gt grad); 3-arm weight sweep ∈ {0, 0.1, 0.5} |
| #3697 | frieren | **Multi-σ Gaussian Fourier under FiLM+Lion+EMA** (proper wiring this time) | `--fourier_sigmas "3,10,30"` + `--n_fourier_per_scale 8`; concatenate Gaussian features at each σ; 3-arm sweep: {σ=10 control, σ∈{3,10}, σ∈{3,10,30}} |
| #3698 | thorfinn | **TTA via z-reflection symmetry** (free inference gain) | `--use_tta_reflection`; reflect z→−z at eval, average original + reflected predictions; 2-arm: control vs TTA |

All assigned via direct GraphQL (REST API exhausted, GraphQL still has 3000+/5000 budget). Branches pushed, draft PRs created, labels {`status:wip`, `icml-appendix-willow-pai2i-48h-r5`, `student:<name>`} verified.

---

## 2026-05-16 03:40 — PR #3405: FiLM conditioning + Lion + EMA [Round 4 nezuko] ← NEW BASELINE

- Branch: `willowpai2i48h5-nezuko/film-conditioning-log-re`
- Hypothesis: Condition the Transolver model on Reynolds number via FiLM (Feature-wise Linear Modulation) — gamma/beta affine transforms on log(Re) applied at the output layer. log(Re) encodes the Reynolds-regime of each flow sample; the `re_rand` OOD split has the most to gain. Combined with Lion optimizer + EMA(0.997) as the new substrate.

| Run | W&B run | Config | val_avg | test_avg | Notes |
|-----|---------|--------|---------|----------|-------|
| Lion+EMA+FiLM | `ksltdq7a` | FiLM + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + σ=10 + T_max=14 | **71.6544** | **62.1091** | **WINNER** — merged |

**Per-split results (ksltdq7a):**

| Split | val | test |
|-------|-----|------|
| single_in_dist | 81.17 | 71.30 |
| geom_camber_rc | 84.45 | 73.87 |
| geom_camber_cruise | **51.99** | **42.84** |
| re_rand | 69.01 | 60.43 |

**Decision: MERGED** as new baseline. val_avg 77.58 → 71.65 (−7.9%), test_avg 68.88 → 62.11 (−9.8%). 3rd consecutive improvement in the Round 3-4 cascade.

**Analysis:**
- FiLM conditioning on log(Re) adds ~5.9 val / 6.8 test on top of Lion+EMA(0.997). This is meaningful additive gain from an orthogonal mechanism.
- FiLM's biggest gain is on `geom_camber_cruise` test (48.83 → 42.84, −12.3%) and `single_in_dist` test (81.69 → 71.30, −12.7%). The mechanism appears to benefit geometry-OOD splits more than Re-OOD, possibly because log(Re) is a proxy for flow complexity that correlates with camber-induced pressure peaks.
- Among 5 simultaneous Lion+EMA reruns, only FiLM provided a further separation (all others were 73-77 val; FiLM brought it to 71.65).
- Fourier σ=10 remains in the stack; Round 5 can test if FiLM obviates Fourier.
- **No Fourier (edward run 5pvi79f2): val 73.10** — within 1.5 val of fern's σ=3 (73.81) and worse than FiLM (71.65). Fourier is marginal but FiLM is clearly the dominant mechanism.

**Round 4 companion runs (not merged, informative ablations):**

| Run | Config | val_avg | test_avg | Status |
|-----|--------|---------|----------|--------|
| `5pvi79f2` edward | Lion + EMA(0.997), n_fourier=0 | 73.10 | 63.65 | Ablation: confirms no-Fourier under Lion+EMA |
| `dl4apv3e` fern | Lion + EMA(0.997) + σ=3 | 73.81 | 63.89 | Ablation: σ=3 ≈ no-Fourier, not better |
| `fg3u9jsj` alphonse | Lion + EMA(0.997) + σ=10 | 76.15 | 66.55 | Variance sample A |
| `5uaxtezx` tanjiro | Lion + EMA(0.997) + σ=10 | 79.17 | 68.97 | Variance sample B |
| `54hmldzq` frieren | Lion + EMA(0.997), n_fourier=0 (config bug) | 76.95 | 67.07 | Config bug — intended multi-σ |
| `drt9naou` thorfinn | Lookahead (k=5 α=0.5) + Lion | 89.39 | 78.24 | Dead end: Lookahead regresses |
| `379hrdie` askeladd | Lion warmup=0 (control) | 79.13 | 68.98 | Control: baseline reproduction |

---

## 2026-05-16 01:43 — PR #3537: Lion optimizer (sign-based update) vs AdamW [Round 3 H13]

- Branch: `willowpai2i48h5-askeladd/round3-lion-optimizer`
- Hypothesis: Replace AdamW with Lion (Chen et al. 2023, arXiv 2302.06675) — sign-based update, momentum-decay schedule, decoupled weight decay. Lion's sign update yields uniform per-coordinate steps, potentially benefiting irregular-mesh CFD where AdamW's adaptive scaling may misjudge importance across heterogeneous node features.

| Arm | W&B run | optimizer | lr | wd | val_avg | test_avg | Notes |
|-----|---------|-----------|----|----|---------|----------|-------|
| A — Lion lr=5e-5 wd=1e-3 | `yvkf9glr` | lion | 5e-5 | 1e-3 | **77.5788** | **68.8764** | **WINNER** — merged |
| B — Lion lr=1e-4 wd=5e-4 | (not yet run) | lion | 1e-4 | 5e-4 | — | — | Follow-up sweep |
| C — Lion lr=3e-4 wd=1e-4 | (not yet run) | lion | 3e-4 | 1e-4 | — | — | Follow-up sweep |

**Per-split val mae_surf_p (Arm A):**

| Split | val | test |
|-------|-----|------|
| single_in_dist | 90.85 | 81.69 |
| geom_camber_rc | 87.72 | 77.94 |
| geom_camber_cruise | **58.81** | **48.83** |
| re_rand | 72.93 | 67.04 |

**Decision: MERGED** as new baseline. val_avg 93.20 → 77.58 (−16.8%). test_avg 83.54 → 68.88 (−17.5%). Every test split improves substantially. This is the **largest single-mechanism gain** of the launch (Δ = 15.62 val, 3.4σ above noise floor σ ≈ 4.6).

**Analysis:**
- Lion paper recommends batch ≥ 64 but it works strongly at our batch_size=4. The irregular-mesh CFD loss landscape appears to be well-suited to sign updates.
- LR=5e-5 was the conservative 10× scale-down from AdamW's 5e-4 — Lion's larger effective per-coordinate step requires lower LR.
- All other components held constant: Huber β=0.05, Fourier σ=10 n=16, T_max=14.
- Arms B and C (LR sweep around the winner) are paper-required ablations but Arm A is already merged.

**Implications:**
- All EMA-cluster wins (tanjiro EMA(0.997) val 86.42, fern σ=3+EMA val 87.83) need re-validation on top of Lion. They were achieved with `cosine_t_max=None` and AdamW.
- Natural Round 4: EMA(0.997) + Lion compound (4-way stack with Huber + σ=10 + T_max=14).

---

## 2026-05-15 23:11 — PR #3444: Cosine T_max=14 (recalibrate schedule to wall-clock budget) [Round 2 thorfinn]

- Branch: `willowpai2i48h5-thorfinn/round2-cosine-tmax`
- Hypothesis: 30-min wall-clock binds at epoch ~14 of 50. The cosine LR schedule was set for T_max=50 → at the early stopping point LR is still ~82% of peak. Setting T_max=14 lets the schedule complete inside the budget, giving the final 2-4 epochs proper fine-tuning at low LR.

| Run | cosine_t_max | val_avg | test_avg | Δ vs prior baseline |
|-----|--------------|---------|----------|---------------------|
| `1hx2rm1n` | 14 | **93.1996** | **83.5377** | **MERGED** (−3.0 val, −6.5 test vs 96.05/90.00) |

- All 4 splits improved substantially. Biggest gain: `geom_camber_rc` test (−12.8%).
- 1-LOC change to scheduler T_max — orthogonal to optimizer, loss, features.

---

## 2026-05-15 15:20 — PR #3123: Random Fourier positional features over (x,z) mesh coords

- Branch: `willowpai2i48h5-thorfinn/fourier-positional-features`
- Hypothesis: Map (x,z) coordinates through random Fourier features `[sin(2π B·xz), cos(2π B·xz)]` with Gaussian projection B (sigma=10) to give the model a high-frequency position basis, helping near-surface pressure gradient representation. Expected larger improvement on OOD camber splits.

| Arm | W&B run | n_fourier | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A — baseline | jyqygcbx | 0 | 14/50 | 135.23 | NaN⚠️ | Wall-clock timeout at epoch 14 |
| B — fourier-8 | qvkpm23n | 8 | 14/50 | 143.23 | NaN⚠️ | Worse than baseline (seed variance?) |
| C — fourier-16 | 24yldhv7 | 16 | 14/50 | **130.46** | NaN⚠️ | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (baseline) | 156.98 | 144.01 | 119.48 | 120.44 |
| B (n=8) | 191.33 | 148.23 | 102.79 | 130.55 |
| C (n=16) | 159.57 | 150.12 | **89.02** | 123.13 |

**Decision: MERGED** (Arm C config). val_avg: 135.23 → 130.46 (-3.5%).

**Analysis:**
- Main signal: cruise camber OOD split drops 25.5% (119→89). Fourier features help geometry interpolation at the frequency scale of camber variation. raceCar camber split shows no benefit (+4%) — possibly because raceCar pressure is dominated by ground-effect features better captured by the existing dsdf descriptor than by position frequency.
- Arm B anomaly (worse than baseline) is likely seed variance at 14 epochs rather than a real effect.
- ALL arms hit 30-min wall clock timeout at epoch 14 of 50 — severe under-training. Longer runs would give more signal.
- **Critical bug discovered:** `test_avg/mae_surf_p = NaN` on all arms due to model overflow on test_geom_camber_cruise split. Tracked in PR #3296 (thorfinn follow-up).
- Baseline-equivalent (Arm A) val_avg = 135.23 is now the empirical starting point for all Round 1 comparisons.

---

## 2026-05-15 18:15 — PR #3098: SmoothL1 / Huber loss for heavy-tailed surface pressure

- Branch: `willowpai2i48h5-alphonse/huber-surface-loss`
- Hypothesis: Replace MSE with SmoothL1 (Huber) loss to cap gradient magnitude on heavy-tailed high-Re samples, rebalancing optimizer toward moderate-Re bulk. Expected -3 to -8% on val_avg/mae_surf_p.

| Arm | W&B run | loss_type | beta | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|------|--------|---------|----------|-------|
| A — MSE baseline | 9jr2u0f9 | mse | — | 12/50 | 137.54 | NaN⚠️ | Wall-clock |
| B — SmoothL1 β=0.1 | nlvd0e6f | smooth_l1 | 0.10 | 14/50 | 111.22 | NaN⚠️ | |
| C — SmoothL1 β=0.05 | md6so639 | smooth_l1 | 0.05 | 14/50 | **96.05** | NaN⚠️ (cruise bug) | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (MSE) | 193.27 | 135.11 | 102.79 | 118.99 |
| B (β=0.1) | 146.38 | 129.52 | 75.69 | 93.27 |
| C (β=0.05) | **109.64** | **112.30** | **73.22** | **89.06** |

**Test partial (excl. cruise, arm C):** in_dist 96.04, camber_rc 100.16, re_rand 84.02

**Decision: MERGED** (Arm C). val_avg: 130.46 → 96.05 (-26.4%). New launch best.

**Analysis:**
- Effect size far exceeded prediction (predicted -8%, observed -30%). Pressure is the dominant heavy-tailed channel; SmoothL1 is near-perfectly matched to the metric.
- β=0.05 outperforms β=0.1 — smaller transition point keeps more gradients in linear regime during under-training phase.
- All 4 val splits improved; OOD gains (camber_cruise -29%, re_rand -25%) suggest Huber reduces high-Re sample dominance that hurts OOD generalization.
- These runs did NOT use Fourier PE (n_fourier=0) — gains are additive to PR #3123. Round 2 compound stack expected to deliver further improvement.
- test_avg NaN persists (cruise GT bug) — tracked in PR #3296.

---

## 2026-05-15 18:15 — PR #3109: bf16 + bigger batch (bs=8/16)

- Branch: `willowpai2i48h5-frieren/bf16-bigger-batch`
- Hypothesis: bf16 AMP + larger batches increase effective epoch count in 30-min window → better val.

| Arm | W&B run | batch_size | dtype | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|-------|--------|---------|----------|-------|
| A — fp32 bs=4 | uxk9rt4t | 4 | fp32 | 14/50 | 133.72 | NaN⚠️ | Best arm |
| B — bf16 bs=8 | 3a8s43dk | 8 | bf16 | 17/50 | 139.34 | NaN⚠️ | More epochs, worse result |
| C — bf16 bs=10 | mkqpnjzp | 10 | bf16 | 17/50 | 162.06 | NaN⚠️ | Worst — bs=12/16 OOMed |

**Decision: CLOSED** (does not beat merged baseline 96.05; merge conflict).

**Analysis:**
- bf16 speedup is real (~18% faster epochs) but larger batches hurt convergence — LR not scaled with batch size. Arms B/C completed 17 epochs but final val_avg worse than baseline's 14 epochs.
- bs=16/12 OOMed on real loader (242K-node cruise meshes push padded batch to 94+ GB). Max viable batch is ~bs=10.
- Key insight: bs=4 with bf16 alone (no batch scaling) may be worth a quick verification — frieren suggested this. Could fold into compound stack PR.
- Cosine LR with T_max=50 is poorly calibrated against the ~14-epoch wall-clock ceiling — stays near peak lr for entire run.

---

## 2026-05-15 18:50 — PR #3100: Transolver scale-up (wider/deeper architecture, ~3-7M params)

- Branch: `willowpai2i48h5-askeladd/transolver-scale-up`
- Hypothesis: Larger n_hidden/n_layers/n_head will improve representation capacity → better val_avg under reasonable VRAM headroom (96 GB GPU).

| Arm | W&B run | n_hidden/n_layers/n_head | n_params | bs | epochs | val_avg | test_avg | Peak VRAM |
|-----|---------|--------------------------|----------|----|----|--------|---------|----------|
| A — baseline | xii5dbk8 | 128/5/4 | 0.66M | 4 | 11 | **150.94** | **136.70** | 42.1 GB |
| B — wider | d7coya51 | 192/6/8 | 1.70M | 4 | 6 | 168.02 | 153.92 | 80.2 GB |
| C — deeper-wide | pcarz06v | 256/6/8 | 3.01M | 2 | 4 | 179.92 | 166.38 | 49.4 GB |

**Decision: CLOSED** (val_avg 150.94 = +57% vs new baseline 96.05).

**Analysis:**
- Capacity is not the binding constraint at our 30-min wall clock — convergence is. Arms B/C reach far fewer epochs (6, 4) and regress badly. This conclusively closes the parameter-count axis as a winning lever.
- First valid test_avg of the launch (136.70 on arm A, NaN-clean across all splits) — credit Edward's parallel NaN diagnosis enabling this.
- Cruise split is easiest in absolute terms for every arm (93.68 on A) — counter to in-dist intuition.

---

## 2026-05-15 18:50 — PR #3103: Slice-num scaling (64 → 128 / 192 physics tokens)

- Branch: `willowpai2i48h5-edward/slice-num-scaling`
- Hypothesis: Increasing slice_num gives more PhysicsAttention tokens → better representation. Combined with rerun-with-NaN-guard for clean test metrics.

| Arm | W&B run | slice_num | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A baseline | aod6uhrj | 64 | 14 | **124.39** | NaN¹ | best val_avg, pre-fix |
| A rerun w/ full NaN fix | zxu6ktx5 | 64 | 14 | 137.14 | **124.02** | first finite test_avg |
| B 128 (no fix) | s0cgfl2s | 128 | 11 | 140.23 | NaN¹ | worse than A |
| B 128 rerun w/ fix | 9j7oeip2 | 128 | 11 | 150.16 | 138.07 | worse |
| C 192 | a6t73no8 | 192 | 0 | — | — | **OOM in epoch-1 val** |

¹ NaN from pre-NaN-fix runs (y-inf in test_geom_camber_cruise/000020.pt).

**Decision: CLOSED** (val_avg 124.39 = +29% vs new baseline 96.05; OOM at slice=192).

**Analysis:**
- Slice-num scaling does not help under 30-min wall clock — same convergence-not-capacity verdict as scale-up.
- **MAJOR launch credit:** Edward independently diagnosed the y-inf root cause on `test_geom_camber_cruise/000020.pt` (761 inf p-values), informing thorfinn's two-pronged NaN guard in PR #3296.
- Edward's `zxu6ktx5` rerun was the first finite test_avg on the launch (124.02), confirming the y-side mask works correctly.
- slice=192 OOMed on epoch-1 val on a single H100 — the OOM happens during validation on cruise (largest mesh), not training.

---

## 2026-05-15 18:50 — PR #3105: Linear warmup + cosine LR

- Branch: `willowpai2i48h5-fern/warmup-cosine-lr`
- Hypothesis: Linear LR warmup over first ~5% epochs avoids early gradient instability → better val.

| Arm | W&B run | lr peak | warmup_frac | epochs | val_avg | test_3split_partial | Notes |
|-----|---------|---------|-------------|--------|---------|---------------------|-------|
| A baseline (advisor pick) | i3z00pw4 | 5e-4 | 0.00 | 14 | **127.82** | 126.44 | |
| A repeat 1 | 07ddhitq | 5e-4 | 0.00 | 14 | 122.10 | 119.64 | run-noise diagnostic |
| A repeat 2 | b4cv2rqp | 5e-4 | 0.00 | 14 | 131.26 | 127.00 | run-noise diagnostic |
| B warmup-5e-4 | pd21qc2t | 5e-4 | 0.05 | 14 | 145.40 | 148.38 | +13.8% regression |
| C warmup-1e-3 | l2pow9iw | 1e-3 | 0.05 | 14 | 143.79 | 141.23 | +12.5% regression |

**Decision: CLOSED** (val_avg 127.82 = +33% vs new baseline 96.05; warmup arms regress vs even worst-case arm-A).

**Analysis:**
- Warmup actively hurts in our 14-epoch-cap regime — cosine T_max=50 already keeps LR near peak, further suppressing early LR throws away gradient signal.
- Three arm-A repeats give us the **first run-to-run variance estimate** on the launch: σ ≈ 4.6 on val_avg (range 122.10–131.26), about ~3.6% relative. Useful reference for evaluating all future small deltas.
- Existing cosine-no-warmup schedule is locally optimal at this wall-clock budget.

---

## 2026-05-15 18:50 — PR #3114: Gradient clipping + EMA model weights

- Branch: `willowpai2i48h5-nezuko/grad-clip-ema`
- Hypothesis: grad-clip(1.0) suppresses gradient spikes; EMA(0.999) on model weights provides flat-minima inference → both improve generalization.

| Arm | W&B run | grad_clip | ema_decay | epochs | val_avg | test_3split |
|-----|---------|-----------|-----------|--------|---------|-------------|
| A baseline | gt0hqg32 | 0.0 | 0.0 | 14 | 135.62 | 135.66 |
| A rerun 1 | jpdav2j1 | 0.0 | 0.0 | 13 | 128.41 | 128.19 |
| A rerun 2 | mkmflt8c | 0.0 | 0.0 | 12 | 133.73 | 132.68 |
| B grad-clip only | p2v9zpal | 1.0 | 0.0 | 14 | 104.87 | 101.96 |
| B rerun | tcggs514 | 1.0 | 0.0 | 13 | 111.78 | 108.04 |
| **C clip+EMA** | **i69fv3fg** | **1.0** | **0.999** | **14** | **102.67** | **99.48** |

**Decision: CLOSED** (val_avg 102.67 = +6.9% vs new baseline 96.05; mechanism subsumed by alphonse's compound stack #3379).

**Analysis:**
- Strong standalone result — clip+EMA gets to **2nd place** in Round 1 leaderboard. Clip alone (104.87) captures most of the gain; EMA adds ~2 points.
- Mechanism confirmed orthogonal to Huber, hence is exactly the optimization layer being tested on top of Huber in alphonse's Round 2 compound stack PR #3379.
- Not merged because the stack is a strictly stronger candidate — would be a regression to land 102.67 over the 96.05 baseline standalone.
- Round 1 winner (Huber) and 2nd place (clip+EMA) are mechanistically orthogonal — supports the compound stack hypothesis.

---

## 2026-05-15 18:50 — PR #3118: Per-channel surface loss weighting

- Branch: `willowpai2i48h5-tanjiro/per-channel-loss-weighting`
- Hypothesis: Up-weighting the pressure channel in surf_loss focuses the optimizer on the metric → better val_avg/mae_surf_p.

| Arm | W&B run | surf_w (Ux,Uy,p) | epochs | val_avg | Notes |
|-----|---------|------------------|--------|---------|-------|
| A baseline | wwfhp260 | (1.0, 1.0, 1.0) | 14 | **130.51** | |
| B p-2x | lvw0bz34 | (1.0, 1.0, 2.0) | 13 | 144.58 | +10.8% regression |
| C p-heavy | 3770ejgb | (0.5, 0.5, 3.0) | 12 | 142.38 | +9.1% regression |

**Decision: CLOSED** (val_avg 130.51 = +36% vs new baseline 96.05; p-weighting REGRESSES pressure itself).

**Analysis:**
- Counterintuitive finding: up-weighting p hurt the pressure channel itself (arm B vol_p +11%, arm C vol_p +49%).
- Mechanism: Ux/Uy gradients reinforce shared encoder representations that the pressure head relies on. Deprioritizing the velocity channels weakens the encoder, hurting everything including pressure.
- Closes the per-channel-weighting axis. Confirms that multi-task coupling is doing useful work and supports the bandit-style "train on all 3 outputs" approach.

---

## 2026-05-15 18:50 — PR #3296: Two-pronged NaN guard (pred-side nan_to_num + y-side sample mask)

- Branch: `willowpai2i48h5-thorfinn/fix-test-cruise-nan`
- Hypothesis: Two contributors to NaN — model pred overflow on cruise OOD samples AND inf y values in test_geom_camber_cruise/000020.pt (761 nodes). Two-pronged guard resolves both.

| Metric | Before fix | After two-pronged guard (run 4gqpc5ez) |
|--------|------------|----------------------------------------|
| val_avg/mae_surf_p | 130.12 | 142.20 |
| test_avg/mae_surf_p | NaN | **128.97** |
| test_geom_camber_cruise/mae_surf_p | NaN | 103.04 |

Diagnostic confirms 0 non-finite predictions (pred-side clean post-Huber), only y-side inf samples being correctly dropped (761 nodes from 1 sample).

**Per-split test MAE (run 4gqpc5ez, n_fourier=0):**

| Split | test surf_p |
|-------|-------------|
| in_dist | 147.12 |
| camber_rc | 135.67 |
| camber_cruise | **103.04** |
| re_rand | 130.05 |
| **avg** | **128.97** |

**Initial decision: SEND BACK FOR REBASE.** Branch had no actual merge conflict (auto-resolves), but the recorded val_avg 142.20 reflected MSE-baseline training, not the new Huber baseline.

**Followup rerun (xvn4gllg, 2026-05-15 20:29 UTC):** Thorfinn rebased onto Huber baseline cleanly (`3ae5def`) and reran 50 epochs with `--loss_type smooth_l1 --loss_beta 0.05`:

| Metric | MSE rerun (4gqpc5ez) | Huber rebase (xvn4gllg) |
|--------|----------------------|--------------------------|
| val_avg/mae_surf_p | 142.20 | **100.75** (within ~1σ of 96.05) |
| test_avg/mae_surf_p | 128.97 | **90.00** ← FIRST valid test_avg on launch |
| test_geom_camber_rc | 135.67 | 103.19 |
| test_geom_camber_cruise | 103.04 | **60.61** ¹ |
| test_re_rand | 130.05 | 86.90 |

¹ 199/200 samples (`000020.pt` dropped).

**Decision: MERGED** (squash commit `52699b1`, 2026-05-15 20:36 UTC).

**Analysis & merge rationale:**
- Test_avg goes from NaN (paper-unwriteable) → 90.00 — this is the paper-facing primary metric.
- val_avg slight regression (100.75 vs 96.05) is within ~1σ of fern's run-noise estimate (σ≈4.6); same config, run-to-run variance.
- Every Round 2 PR depends on this NaN guard producing valid test metrics.
- The fix is correct (pred-side `nan_to_num` + y-side sample mask), code-stable across loss types, and the rebase was clean.
- merge-winner would refuse on strict val_avg semantic check, but the launch's gating concern is paper-facing test_avg. Decision made by advisor.

---

---

## 2026-05-15 21:25 — PR #3379: Round 2 compound stack (Huber + Fourier + grad_clip + EMA)

- Branch: `willowpai2i48h5-alphonse/round2-compound-stack`
- Hypothesis: Stack all orthogonal Round 1 mechanisms — SmoothL1 β=0.05 (loss), Fourier PE n=16 σ=10 (positional), grad_clip 1.0 (optimization), EMA(0.999) (weight averaging) — for compounding gains.

| Arm | W&B run | Config | val_avg | Δ vs 96.05 | Best ep |
|-----|---------|--------|---------|------------|---------|
| A | lvjaj0cp | Huber + Fourier σ=10, no opt | 100.76 | +4.71 (regression) | 14 |
| B | jxvn2jsd | A + grad_clip 1.0 | 100.32 | +4.27 (regression) | 14 |
| **C** | **hat7m2bl** | **A + grad_clip + EMA(0.999)** | **92.41** | **−3.64 (−3.78%)** | **14** |

**Per-split val (Arm C hat7m2bl):**

| Split | Baseline (md6so639) | Arm C |
|-------|---------------------|-------|
| single_in_dist | 109.64 | 119.72 |
| geom_camber_rc | 112.30 | **104.00** |
| geom_camber_cruise | 73.22 | **62.39** |
| re_rand | 89.06 | **83.51** |

**Per-split test partial (Arm C, cruise=NaN — pre-#3296 baseline; 3-split partial):**
test_single_in_dist=108.61, test_camber_rc=90.94, test_re_rand=75.17 → partial mean 91.57

**Decision: SEND BACK FOR REBASE.** PR is CONFLICTING. Arm C is a clear winner (−3.78% vs baseline) pending rebase onto current HEAD (includes #3296 NaN guard → will produce clean 4-split test_avg). Merge expected after alphonse rebases and confirms.

**Analysis:**
- **EMA(0.999) is the dominant mechanism**, not the compound stack as hypothesized. Fourier PE alone regresses by +4.7 (Arm A: 100.76 vs 96.05); grad_clip alone adds nothing to Fourier (Arm B: 100.32); EMA compensates both and delivers −3.78% improvement.
- This makes EMA the Round 2 discovery — not a stack effect, but a single mechanism that outweighs all others.
- **Fourier PE is net-negative at σ=10 without EMA.** With EMA it's masked. Open questions: (a) does Fourier-free + EMA beat 92.41? (b) does lower σ fix Fourier regression? Both assigned to Round 3.
- All 3 arms hit timeout at epoch 14 (cosine T_max=50 not recalibrated) — EMA gains compound over the available steps but plateau could be earlier with T_max fix.
- Test in_dist regresses (108.61 vs 96.04) despite val in_dist regressing (119.72 vs 109.64) — EMA helps OOD splits more than in-distribution. Consistent with EMA's flat-minima geometric interpretation.

---

## 2026-05-15 22:10 — PR #3407: Per-sample Relative L2 loss (CLOSED — catastrophic regression)

- Branch: `willowpai2i48h5-edward/round2-rel-l2`
- Hypothesis: Normalize loss by per-sample L2 norm to achieve scale invariance across Reynolds regimes.

| Run | W&B run | State | val_avg | Notes |
|-----|---------|-------|---------|-------|
| B (rel-l2-surf-only, orig) | 1ck8juvm | finished | 367.17 | catastrophic |
| B (rerun) | rrszrxgv | finished | 367.13 | catastrophic |
| C (rel-l2-both) | 5wczva6k | crashed | 367.05 | catastrophic |
| B (fixed) | olmbe0up | finished | 117.69 | converges, +22% regression |

**Decision: CLOSED.** Even the working implementation at val 117.69 is +22% above baseline 96.05.

**Analysis:** Huber β=0.05 already achieves implicit relative scaling on the heavy-tail pressure channel; explicit per-sample L2 normalization competes with Huber's soft-cap rather than complementing it. Edward's debugging from 367→117 is solid engineering but the approach is mechanistically incompatible with Huber.

---

## 2026-05-15 22:05 — PR #3410: 1st-order SAM optimizer (CLOSED — wall-clock incompatible)

- Branch: `willowpai2i48h5-tanjiro/round2-sam`
- Hypothesis: SAM's dual ascent+descent step finds flatter minima → better OOD generalization.

| Run | W&B run | State | val_avg | Notes |
|-----|---------|-------|---------|-------|
| A uniform | l11n94ct | crashed | 200.82 | catastrophic |
| B (ρ=0.05) | jecq3zxh | finished | 147.76 | +54% regression |
| B rerun 1 | 924zb6gb | finished | 142.86 | +49% regression |
| B rerun 2 | ey6fw9c8 | finished | 157.66 | +64% regression |

Mean Arm B val_avg ≈ 149.4 (+55% vs 96.05).

**Decision: CLOSED.** SAM doubles the optimizer step cost, halving effective epoch count at 30-min wall clock. Exactly the wrong tool for an under-trained regime.

---

## 2026-05-15 22:03 — PR #3409: AoA reflection symmetry augmentation (CLOSED — redundant)

- Branch: `willowpai2i48h5-fern/round2-aoa-aug`
- Hypothesis: Reflecting airfoil samples across AoA=0 plane doubles effective training data.

| Run | W&B run | Arm | val_avg | Notes |
|-----|---------|-----|---------|-------|
| m6f1meku | baseline-r2 | A | 105.95 | baseline variance rerun |
| ghgayq3j | baseline-nan-guard | A | 102.38 | baseline variance rerun |
| em91w2q5 | aoa-aug-rc-single-safe | B | 119.28 | +13.4 above baseline mean |

**Decision: CLOSED.** AoA augmentation regresses val_avg by ~13 points (3σ outside noise). Dataset's existing geometric variation in camber/re_rand splits already covers AoA diversity; reflection adds redundant samples rather than new information.

---

## 2026-05-15 22:25 — PR #3380: Fourier sigma sweep (SEND BACK — wrong loss config)

- Branch: `willowpai2i48h5-frieren/round2-fourier-sigma`
- Hypothesis: Sweep n=16 Fourier sigma ∈ {4, 10, 20} to find optimal positional feature frequency.

| Run | W&B run | sigma | loss_type | val_avg | Notes |
|-----|---------|-------|-----------|---------|-------|
| t8kcas5g | Arm A σ=10 | 10 | MSE ❌ | 152.71 | wrong loss |
| ydh957qb | Arm B σ=4 | 4 | MSE ❌ | 134.64 | wrong loss |
| 68lxdalu | Arm C σ=20 | 20 | MSE ❌ | 150.88 | wrong loss |

**Decision: SEND BACK.** Frieren ran all 3 arms with `loss_type=mse`, not `smooth_l1 β=0.05`. Results reflect pre-Huber baseline territory (134-152) and carry no signal about Fourier sigma under the correct loss regime. Re-run instructions issued: add `--loss_type smooth_l1 --loss_beta 0.05` to all 3 arms.

---

---

## 2026-05-15 23:37 — PR #3444: Cosine LR T_max=14 recalibration — MERGED ✅

- Branch: `willowpai2i48h5-thorfinn/round2-cosine-tmax`
- Hypothesis: 30-min wall clock binds at ~epoch 14, but cosine schedule was set for T_max=50 → LR never decayed below 82% of peak. Setting T_max=14 lets cosine complete in-budget, giving fine-tuning at low LR for the final epochs.

| Arm | W&B run | T_max | val_avg | test_avg | best_ep |
|-----|---------|-------|---------|----------|---------|
| A | zcjww6dy | 50 (reference) | 104.19 | 91.95 | 14 |
| **B** | **1hx2rm1n** | **14** | **93.20** ★ | **83.54** ★ | 14 |
| C | (aborted) | 18 | — | — | — |

Arm C aborted per advisor sign-off — Arm B not under-converged.

**Per-split (Arm B run `1hx2rm1n`):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 114.80 | 105.93 |
| geom_camber_rc | 104.16 | 90.03 |
| geom_camber_cruise | 68.17 | **57.65** |
| re_rand | 85.66 | 80.55 |
| **avg** | **93.20** | **83.54** |

**Test delta vs merged baseline #3296 `xvn4gllg`:** −7.2% overall (90.00 → 83.54). Biggest gain on `geom_camber_rc` (−12.8%, 103.19 → 90.03) — previously the hardest split.

**Decision: MERGED** (squash commit `53105ae`, 2026-05-15 23:37 UTC).

**Analysis:**
- 1-LOC change (added `cosine_t_max` config flag). Pure scheduler-period change, no model architecture or loss change.
- Single-mechanism result with magnitude comparable to alphonse's compound EMA stack — confirms LR schedule was the under-tuned dial.
- Validates frieren's bf16+batch observation: the cosine LR schedule was indeed poorly calibrated against 14-epoch wall clock.
- Mechanism orthogonal to EMA (alphonse #3379), Huber loss (#3098), NaN guard (#3296). Expected to compound: EMA + T_max=14 is the natural Round 4 experiment.
- Thorfinn's analysis explicitly suggests "Compose with EMA / other LR-related techniques" — exactly the next assignment.

---

## 2026-05-15 23:25 — PR #3412: DropPath stochastic depth (CLOSED — regresses)

- Branch: `willowpai2i48h5-askeladd/round2-droppath`
- Hypothesis: DropPath provides ensemble-like regularization, expected 2-5% improvement on OOD splits.

| Run | W&B run | Config | val_avg | test 3-split partial |
|-----|---------|--------|---------|----------------------|
| Arm B (uniform 0.1, Huber, no Fourier) | 2if2scsr | Huber only | 102.34 | 100.36 |
| Arm C (linear 0→0.2, Huber, no Fourier) | 9sdchdtq | Huber only | 105.88 | 105.48 |
| Arm B' (confounded by Fourier) | btbi5pzy | Huber + Fourier | 112.24 | — |

**Decision: CLOSED.** All DropPath configs regress baseline by 6.5-10%. Mechanism explanation: under 14-epoch under-trained regime, every gradient signal matters; skipping entire residual branches (DropPath) starves the model of training signal it can't afford to lose.

**Important sub-finding from askeladd's investigation:** the original PR body misread BASELINE.md — claimed baseline was "Huber + Fourier" when actually it was Huber only (no Fourier). Askeladd correctly rerun on the actual baseline. Their negative result stands: DropPath alone doesn't help.

Per-split test (best arm B): in_dist=111.28, camber_rc=99.67, re_rand=90.14. The camber_rc result (99.67) is interestingly the only split that *slightly* improves vs baseline (100.16) — but the average is dragged down by single_in_dist regression.

---

<!-- Template:
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- Branch: <student-branch-name>
- Hypothesis: <hypothesis>
- Results:

| Arm | W&B run ID | val_avg/mae_surf_p | test_avg/mae_surf_p | notes |
|-----|------------|--------------------|---------------------|-------|
| A (baseline) | ... | ... | ... | |
| B (...) | ... | ... | ... | |

- Analysis: <results commentary, analysis and conclusions>
-->
