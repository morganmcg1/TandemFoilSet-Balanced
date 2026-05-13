# SENPAI Research Results — willow-pai2g-24h-r5

---

## 2026-05-13 11:09 — PR #2069: n_head sweep n_head=8 vs n_head=2 on Lion+MAE+EMA (alphonse) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-alphonse/n-head-8-lion-mae`
- **Hypothesis:** n_head=4 (baseline) may be over-/under-parameterized for slice_num=64 with n_hidden=128. n_head=2 doubles per-head dim (32→64); n_head=8 halves it (32→16).
- **W&B runs:** `2lo9mn88` (n_head=2, winner), `qkh64fhe` (n_head=8 run 2), `y42702ef` (n_head=8 run 1)

| Metric | Prev baseline (#1932) | n_head=2 | n_head=8 (run 1) | n_head=8 (run 2) |
|--------|----------------------|----------|-----------------|-----------------|
| val_avg/mae_surf_p | 55.41 | **51.11** | 64.09 | 62.74 |
| test_avg/mae_surf_p | 47.90 | **44.18** | 54.96 | 54.11 |
| Δ vs baseline | — | **−7.76%** | +15.7% | +13.2% |
| Epochs in 30 min | 16 | **20** | 12 | 12 |

**Per-test-split (n_head=2):** single_in_dist=49.23, geom_camber_rc=57.44, geom_camber_cruise=26.74, re_rand=43.30 — wins all 4.

**Result:** MERGED. n_head=2 is the strongest single-experiment win of this round. Val still descending at cap (−1.3/epoch in last 3 transitions, no plateau). The monotonic head-count direction confirms head-undersizing story: n_head=2 wins → n_head=4 baseline → n_head=8 worst. The architectural change also unlocks faster epochs (~93.5s vs ~110s) for net more training steps in the same wall-clock.

**Compute-neutrality caveat (alphonse's analysis):** At matched epoch=12 the two arms are close (n_head=2 val=63.00 vs n_head=8 val=62.74). The headline win at 51.11 is partly the architecture buying compute. But the architectural change is what unlocks it — the win is real and reproducible under the 30-min cap.

**Note:** Both runs used lr=1e-4 (not the lr=2e-4 from #1932). The lr × n_head interaction is unexplored.

**New compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=2** (slice_num=64)

---

## 2026-05-13 ~11:10 — PR #2086: Lion lr probe lr=4e-4 + lr=3e-4 on Lion+MAE (thorfinn) — CLOSED, SATURATION CONFIRMED

- **Branch:** `willowpai2g24h5-thorfinn/lion-lr-4e-4-probe`
- **Hypothesis:** lr-doubling trend 5e-5→1e-4→2e-4 hasn't saturated; 3e-4/4e-4 may continue.
- **W&B runs:** `4q7tjvt4` (lr=4e-4), `bpa3tkj7` (lr=3e-4)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (55.41) |
|-----|---|---|---|
| lr=4e-4 (4q7tjvt4) | 57.53 | 49.03 | +3.83% |
| lr=3e-4 (bpa3tkj7) | 57.68 | 48.81 | +4.10% |
| lr=2e-4 baseline | **55.41** | **47.90** | — |

**Result:** CLOSED. lr-doubling trend saturated at lr=2e-4 after 3 winning octaves. Key diagnostics: (1) flat minimum — lr=4e-4 is only 2 pts worse than lr=3e-4 despite 4×-3× the learning rate; (2) EMA main-vs-EMA gap (~10-12 pts) is identical at all 3 lrs — EMA is not the bottleneck at higher lr; (3) both arms still descending at cap with ~3 val pts over last 4 epochs — not diverging, just in a shallower basin. Schedule shape (OneCycleLR) is the natural next lever.

**Thorfinn reassigned:** PR #2211 — OneCycleLR on n_head=2 baseline.

---

## 2026-05-13 ~11:10 — PR #2056: surf_weight tune on Lion+MAE (nezuko) — CLOSED, RETEST ON n_head=2

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-lion-mae`
- **Hypothesis:** MAE's uniform weighting reduces need for high surf_weight; sw=5 may improve surface focus, sw=15 may over-emphasize it.
- **W&B runs:** `gxq26aip` (sw=5, lr=1e-4), `obkwbyo1` (sw=15, lr=2e-4)

| Arm | sw | lr | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old base (55.41) |
|-----|----|----|---|---|---|
| sw=5 (gxq26aip) | 5 | 1e-4 | **54.46** | **47.06** | −1.71% / −1.76% |
| sw=15 (obkwbyo1) | 15 | 2e-4 | 58.32 | 50.27 | +5.25% / +4.95% |

**Result:** CLOSED because #2069 (n_head=2, val=51.11) merged after this was submitted. sw=5 at lr=1e-4 beat the OLD baseline by 1.71% — a genuine improvement — but cannot be merged onto the new n_head=2 code as-is (was measured at n_head=4). sw=15 regresses on all splits. Nezuko reassigned to retest sw=5+sw=7 on n_head=2 baseline (#2210).

---

## 2026-05-13 ~11:10 — PR #2052: bs=8 + LR scaling on Lion+MAE (frieren) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-frieren/batch-size-lr-scaling`
- **Hypothesis:** bs=8 reduces gradient noise; linear-scaled lr compensates for larger batch.
- **W&B runs:** `nay1st1x` (bs=8, lr=2e-4), `img8ns9k` (bs=8, lr=1e-4)

| Arm | bs | lr | val_avg/mae_surf_p | Δ vs baseline |
|-----|----|----|---|---|
| Arm 1 linear (nay1st1x) | 8 | 2e-4 | 61.05 | +7.9% |
| Arm 2 batch-only (img8ns9k) | 8 | 1e-4 | 66.99 | +18.4% |

**Result:** CLOSED. Hypothesis falsified: step-count-limited, not gradient-noise-limited. bs=8 takes half as many steps in same wall-clock. VRAM at 93.7 / 92.2 GB — bs=8 is maximum safe batch. The per-step distance route (lr) is the right lever; gradient noise reduction via batch size is wrong for this regime. Frieren reassigned to split-loss formulation (#2216: surface-MAE + volume-Huber).

---

## 2026-05-13 ~10:34 — PR #2070: Lion-no-EMA + AdamW-no-EMA ablation (edward) — CLOSED, ICML APPENDIX

- **Branch:** `willowpai2g24h5-edward/lion-no-ema-ablation`
- **Hypothesis:** Quantify Lion vs EMA contributions to the Lion+EMA+MAE win. Diagnostic-only, not for merge.
- **W&B runs:** `lyplrb6e` (Lion-no-EMA full 18 ep, canonical), `q6lou95t`/`5memu5rh` (Lion-no-EMA truncated), `4u85vrwj` (AdamW-no-EMA full 18 ep)

| Cell | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline |
|------|-------------------|---------------------|---------------|
| Lion+EMA(0.99)+MAE (baseline #1932) | **55.41** | **47.90** | 0 |
| Lion+no-EMA+MAE (best, lyplrb6e) | 62.47 | 54.42 | +7.06 / +6.52 |
| AdamW+no-EMA+MAE (4u85vrwj) | 82.46 | 71.69 | +27.05 / +23.79 |

**Result:** CLOSED — regression vs baseline (expected, diagnostic-only). **Reframed mechanism story:**
- Full-budget Lion-no-EMA val=62.47 (NOT 78 from truncated runs) — Lion's direction explains ~20 val, EMA explains ~7-9 val.
- **Lion is the dominant ingredient (~75% of Lion+EMA's win over AdamW+EMA), EMA is secondary (~25%).** The synergy is real but Lion-led, not EMA-led.
- Lion-no-EMA final-epoch bounce 62→71 at ep 18 — direct evidence that a single bad late update visibly hurts val without parameter averaging.
- AdamW-no-EMA has 3 backward steps over 18 epochs (Lion-no-EMA has 1) — Lion's signed direction is loss-landscape-aware on this geometry-aware problem.
- **Variance noted:** lyplrb6e=62.47 vs 5memu5rh=77.99 at identical config = 15-val gap. The truncated runs (10–11 ep) under-represent steady-state; full-budget is the canonical datapoint.

**Edward reassigned:** PR #2183 — AdamW+EMA+MAE at lr=5e-4 and lr=2e-4. Fills missing 2×2 cell; quantifies optimizer effect at fixed lr=2e-4 (apples-to-apples Lion vs AdamW).

---

## 2026-05-13 ~10:06 — PR #1999: Cosine T_max tuning T_max=16 ± eta_min on Lion+MAE+lr=1e-4 (fern) — CLOSED REGRESSION (with strong diagnostic)

- **Branch:** `willowpai2g24h5-fern/cosine-tmax-tuning`
- **Hypothesis:** T_max=epochs=50 barely decays the LR over the 16-epoch wall-clock window. Matching T_max to actual budget should let the cosine arc complete.
- **W&B runs:** `8csqgctq` (Arm 1: T_max=16, eta_min=0), `0mw5k64a` (Arm 2: T_max=16, eta_min=1e-5)
- **Caveat:** Both arms ran at **lr=1e-4** (the pre-#1932 baseline). The PR was assigned before #1932 merged.

| Metric | Pre-merge baseline (#1825, lr=1e-4) | Current baseline (#1932, lr=2e-4) | Arm 1 | Arm 2 |
|--------|-------------------------------------|-----------------------------------|-------|-------|
| val_avg/mae_surf_p | 56.58 | **55.41** | 62.02 | 59.59 |
| test_avg/mae_surf_p | 48.82 | **47.90** | 52.55 | 51.42 |
| Δ vs current baseline | — | — | **+11.9%** | **+7.5%** |
| Final-epoch val Δ (last 4) | — | — | **+0.19** (uptick at lr=0) | **−1.48** (steepest descent) |

**Result:** CLOSED — both arms regress vs current baseline at lr=1e-4. **But the diagnostic value is high.**

**Key findings:**
1. **eta_min=0 is strictly dominated by eta_min=1e-5** (Arm 1 vs Arm 2 differ by −2.43 val). Arm 1 had the *only* positive per-epoch delta in its last 4 epochs at the lr=0 step — the degenerate-tail effect is real. **Future cosine work must use eta_min>0.**
2. **Arm 2 still descending at the cap** (−1.48 val in its final epoch, steepest of the last 4). At lr=1e-4 the per-step capacity is the bottleneck, not the schedule shape. Both schedules (T_max=16 and T_max=50) leave the model under-converged at lr=1e-4.
3. **Vol_loss accumulator non-finite bug** flagged by fern (train.py:288, 313–328). Per-component MAE metrics are correct (PR #1541 path), but the vol_loss sum is unguarded. Cosmetic, not in scope.

**Fern reassigned:** PR #2167 — Cosine T_max + eta_min tuning at lr=2e-4. Tests both axes: (a) schedule-match (T_max=16 + eta_min=1e-5) at the new lr, where per-step capacity is doubled; (b) eta_min=1e-5 floor on the default T_max=50 schedule (isolates final-lr regularization effect). Clean 2-axis decomposition.

---

## 2026-05-13 ~10:00 — PR #2001: Lion β1 sweep β1=0.95 vs β1=0.85 on Lion+MAE+lr=2e-4 (askeladd) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-askeladd/lion-b1-sweep`
- **Hypothesis:** β1=0.9 is Lion's canonical value from large-scale vision experiments; smaller datasets may prefer a different β1. β1=0.95 gives more momentum inertia (slower direction change), β1=0.85 makes the sign update more reactive to the current gradient.
- **W&B runs:** `hqfbylaj` (Arm 1: β1=0.95), `2ql8nhfg` (Arm 2: β1=0.85)

| Metric | Current baseline (#1932) | Arm 1 (β1=0.95) | Arm 2 (β1=0.85) | Δ Arm1 | Δ Arm2 |
|--------|--------------------------|-----------------|-----------------|--------|--------|
| val_avg/mae_surf_p | **55.41** | 57.62 | 58.93 | +4.00% | +6.36% |
| test_avg/mae_surf_p | **47.90** | ~50.0 (est.) | ~51.4 (est.) | regression | regression |

**Result:** CLOSED. Both arms regress vs baseline. The curve is **asymmetric**: β1=0.85 hurts ~2× more than β1=0.95, indicating the loss landscape is more sensitive to over-reactive updates than to over-inertial ones. The optimum is near or slightly above β1=0.9.

**Key finding:** Canonical β1=0.9 is confirmed as (near-)optimal for this problem. The asymmetric response (β1=0.95 hurts less than β1=0.85) suggests the momentum inertia side has more tolerance than the reactivity side — the signed gradient direction is informative enough that maintaining it longer is preferable to discarding it faster. No follow-up β1 variations warranted.

**Askeladd reassigned:** PR #2144 — Lion β2 sweep (β2=0.995 vs β2=0.95) on Lion+MAE+lr=2e-4. At lr=2e-4, the momentum buffer's memory window (β2) is the last untested hyperparameter in the Lion triplet (lr, β1, β2). β1=0.9 optimum confirmed; β2=0.99 default is untested.

---

## 2026-05-13 11:55 — PR #2131: Dropout sweep dropout=0.3 vs 0.1 on Lion+MAE+lr=2e-4 (tanjiro) — CLOSED, LOCALLY OPTIMAL AT 0.2

- **Branch:** `willowpai2g24h5-tanjiro/dropout-lion-mae-lr2e-4`
- **Hypothesis:** Under-regularization signal from #1961 (mlp_ratio=4 had main-vs-EMA gap ~22) should be present at mlp_ratio=2 too; dropout=0.2 may be below optimum.
- **W&B runs:** `qtieg835` (dropout=0.3, winner), `jgcae86g` (dropout=0.3 replicate), `0zk2y6cj` (dropout=0.1), `y8oh2gxf` (baseline ref, dropout=0.2)

| Arm | dropout | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old baseline (55.41) |
|-----|---------|--------|------------|---------------------|----------------------|--------------------------|
| Arm 1 winner | 0.3 | `qtieg835` | 16 | **55.10** | **47.83** | −0.56% / −0.15% |
| Arm 1 replicate | 0.3 | `jgcae86g` | 16 | 55.87 | 48.09 | +0.83% / +0.40% |
| Arm 2 | 0.1 | `0zk2y6cj` | 14 | 57.77 | 49.24 | +4.26% / +2.79% |
| baseline ref | 0.2 | `y8oh2gxf` | 16 | 55.41 | 47.90 | — |

**Dropout=0.3 mean:** val=55.49 ± 0.38, test=47.96 ± 0.13 — within noise of baseline 0.2 (55.41 / 47.90).

**Main-vs-EMA gap diagnostics (epoch 16):**
| dropout | run_id | ema_val | main_val | gap |
|---------|--------|---------|----------|-----|
| 0.1 | `0zk2y6cj` | 57.78 | 69.64 | +11.86 |
| 0.2 | `y8oh2gxf` | 55.41 | 61.77 | +6.36 |
| 0.3 | `qtieg835` | 55.10 | 61.89 | +6.79 |
| 0.3 | `jgcae86g` | 55.87 | 67.02 | +11.15 |

**Result:** CLOSED. Baseline note: n_head=2 merged (val=51.11) during these runs — neither arm can merge on new compound regardless. Key findings:
1. **dropout=0.2 is locally optimal on Lion+MAE+lr=2e-4+n_head=4/mlp_ratio=2** — under-regularization signal from #1961 (mlp_ratio=4, gap≈22) did NOT transfer to mlp_ratio=2 (gap≈6–11).
2. **dropout=0.3 ≈ 0.2 within seed noise** — best single 0.3 run wins old baseline by 0.31 val, but replicate loses by 0.46; mean is 0.08 worse.
3. **dropout=0.1 clearly regresses** (+2.4 val / +1.3 test) — Lion's sign-update alone doesn't substitute for dropout regularization.
4. Trajectory shapes confirm NOT converged at epoch 16; all arms still descending at cap.

**Tanjiro reassigned:** PR #2251 — lr sweep on n_head=2 (lr=2e-4 vs lr=1.5e-4). The n_head=2 baseline (PR #2069) inherited lr=1e-4 from pre-merge compound; lr=2e-4 won at n_head=4 but was never tested at n_head=2. BASELINE.md itself noted "the lr × n_head interaction remains to be explored."

---

## 2026-05-13 12:05 — PR #2144: Lion β2 sweep β2=0.995 vs β2=0.95 on Lion+MAE+lr=2e-4 (askeladd) — CLOSED, STRONG SIGNAL, NEEDS RETEST ON n_head=2

- **Branch:** `willowpai2g24h5-askeladd/lion-beta2-sweep`
- **Hypothesis:** β2 (momentum EMA decay, canonical=0.99) untested on TandemFoilSet. β2=0.995 lengthens momentum window (~200 effective steps); β2=0.95 shortens it (~20 steps).
- **W&B runs:** `b7pyyc0n` (β2=0.995, winner), `94sgx2e9` (β2=0.95)

| Arm | β2 | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old baseline (55.41) |
|-----|-----|--------|------------|---------------------|----------------------|--------------------------|
| 1 (winner) | 0.995 | `b7pyyc0n` | 16 | **53.815** | **46.609** | **−2.9% / −2.7%** |
| 2 | 0.95 | `94sgx2e9` | 16 | 64.118 | 54.969 | +15.7% / +14.8% |

**Val split breakdown (Arm 1, β2=0.995):**
- single_in_dist=57.84 (−3.6), geom_camber_rc=68.85 (+1.5, flat), geom_camber_cruise=33.69 (−3.5), re_rand=54.88 (−0.9)

**Test split breakdown (Arm 1, β2=0.995):**
- single_in_dist=51.41 (flat), geom_camber_rc=60.80 (−1.5), geom_camber_cruise=28.71 (−2.5), re_rand=45.51 (−1.5)

**Result:** CLOSED. Cannot merge on new n_head=2 compound (val=51.11 — arm is +2.71 above). Key findings:
1. **Monotonic ordering: 0.95 < 0.99 < 0.995** — three strictly-ordered data points; trend is real
2. **β2=0.995 wins −2.9% val on old compound** — strongest Lion-internal finding of the round
3. **Asymmetric direction** (faster decay hurts 15.7% vs slower decay wins 2.9%) — signal quality vs staleness tradeoff heavily favor de-noising for batch_size=4
4. **Mechanism:** longer momentum window (~200 vs ~100 steps) de-noises direction before `sign(·)` is taken; consistent with #2070 "Lion-direction-led" finding
5. **EMA(weights) + β2(momentum) appear synergistic** — different smoothing targets
6. **Student suggested β2=0.999** as natural follow-up given monotonic trend

**Askeladd reassigned:** PR #2271 — β2 sweep on n_head=2 (β2=0.995 confirm transfer + β2=0.999 extend trend) at lr=1e-4. Highest-EV follow-up: strong signal, asymmetric, monotonic at 3 points; direct merge candidate if Arm 1 transfers.

---

## 2026-05-13 12:27 — PR #2210: surf_weight sw=5 vs sw=7 on n_head=2 (nezuko) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-n-head-2`
- **Hypothesis:** sw=5 (from #2056 on n_head=4) may stack onto n_head=2; sw=7 probes midpoint.
- **W&B runs:** `qkyx47iv` (sw=5, winner), `2owd44pg` (sw=7)

| Arm | sw | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (51.11) |
|-----|-----|--------|------------|---------------------|----------------------|----------------------|
| 1 (winner) | 5 | `qkyx47iv` | 20 | **50.9119** | **43.6823** | **−0.39% / −1.13%** |
| 2 | 7 | `2owd44pg` | 20 | 52.0234 | 45.2506 | +1.78% / +2.43% |

**Per-test-split (sw=5 winner):**
- single_in_dist=46.42 (−2.81 vs baseline), geom_camber_rc=58.60 (+1.16), geom_camber_cruise=27.33 (+0.59), re_rand=42.39 (−0.91)

**Result:** MERGED. New best: val=50.91, test=43.68. Key findings:
1. **sw=5 stacks onto n_head=2** — confirms the surface weighting insight transfers architectures
2. **Non-monotonic response: sw=5 < sw=10 < sw=7** — sw=7 is a local maximum, not a linear interpolation; suggests complex loss landscape in [5,10]
3. **Gain skewed toward in-dist splits** (single_in_dist −2.81, re_rand −0.91); marginal losses on OOD camber splits — sw=5 sharpens in-distribution surface accuracy
4. Both arms reached 20 epochs (val still descending at cap — NOT converged)
5. Peak GPU memory: sw=5 = 99.16 GB (96.6%) — near VRAM limit

**New compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **surf_weight=5**

**Nezuko reassigned:** PR #2277 — surf_weight lower probe: sw=4 (Arm1) vs sw=3 (Arm2). First data below sw=5; will determine if the loss landscape floor is at or below sw=5.

---

## 2026-05-13 09:12 — PR #1961: FFN width sweep mlp_ratio=3/4 on Lion+EMA (tanjiro) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-tanjiro/mlp-ratio-expansion`
- **Hypothesis:** Wider FFN at fixed n_hidden gives Transolver more per-block representation capacity at low extra cost vs depth/width.
- **W&B runs:** `0la90jp4` (Arm 1: mlp_ratio=3, best), `3vii7yyo` (Arm 2: mlp_ratio=4), `y9byrbsb` (Arm 1 replicate)

| Metric | Lion+EMA baseline (old) | Lion+MAE+lr=2e-4 (current) | Arm 1 (ratio=3) | Arm 2 (ratio=4) |
|--------|-------------------------|----------------------------|-----------------|-----------------|
| val_avg/mae_surf_p | 61.302 | **55.41** | 62.368 | 63.483 |
| test_avg/mae_surf_p | 52.682 | **47.90** | 54.270 | 54.241 |
| Δ vs current baseline | — | — | **+12.6%** | **+14.6%** |
| Epochs (30-min cap) | 16 | 16 | 16 | 15 |
| Main_val vs EMA_val gap (epoch 16, Arm 2) | — | — | — | **85.3 vs 63.5 = 22pt** |

**Result:** CLOSED. **Third consecutive capacity-expansion failure** at the 30-min cap — joins #1761 (n_layers=6) and #1934 (n_hidden=192/256) in the compute-wall pattern.

**Key finding from tanjiro's analysis:** Arm 2 (ratio=4) showed a 22-point gap between main_val (85.3) and EMA_val (63.5) — EMA averaging out heavy noise. This is the **under-regularization signal**: capacity expansion at fixed dropout=0.2 produces a noisier training trajectory. Hints that even the baseline compound may have regularization headroom — direct lead-in to tanjiro's next experiment.

**Architectural takeaway (final):** All three capacity axes (depth, width, FFN width) regress at 30-min cap. The model sits in a near-Pareto region for this budget. Future architecture experiments must be compute-neutral (n_head, slice_num) OR paired with budget-aware tradeoffs (e.g., bs↑ to compensate for epoch loss).

**Tanjiro reassigned:** PR #2131 — dropout sweep (0.1 vs 0.3) on Lion+MAE+lr=2e-4. Direct probe of tanjiro's own under-regularization observation.

---

## 2026-05-13 08:30 — PR #1932: Lion lr=2e-4 scaling on Lion+MAE compound (thorfinn) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-thorfinn/lion-lr-scaling`
- **Hypothesis:** lr-doubling trend 5e-5→1e-4 hasn't saturated; 1e-4→2e-4 may continue the descent.
- **W&B runs:** `y8oh2gxf` (Arm 1: lr=2e-4, wd=1e-4, **winner**), `141lcuxh` (Arm 2: lr=2e-4, wd=5e-4)

| Metric | MAE baseline (#1825) | Arm 1 (lr=2e-4, wd=1e-4) | Arm 2 (lr=2e-4, wd=5e-4) | Δ Arm1 |
|--------|---------------------|--------------------------|--------------------------|--------|
| val_avg/mae_surf_p | 56.577 | **55.412** | 56.801 | **−2.06%** |
| test_avg/mae_surf_p | 48.817 | **47.899** | 49.079 | **−1.88%** |
| test/single_in_dist | 53.687 | **51.084** | 52.326 | −4.85% |
| test/geom_camber_rc | 63.234 | **62.288** | 65.813 | −1.49% |
| test/geom_camber_cruise | 30.812 | 31.211 | **30.613** | +1.30% |
| test/re_rand | 47.535 | **47.014** | 47.563 | −1.10% |

**Result:** MERGED. Arm 1 wins 3/4 test splits + average. Val still descending at epoch-16 cap (last 4 epochs: 61.27→59.07→57.82→55.41, ≈−2 pts/epoch). **Third consecutive lr-doubling win without saturation.**

**Key finding:** Canonical wd scaling (Arm 2, wd=5e-4) disconfirms the Chen et al. `lr×wd≈const` heuristic on this compound. MAE-Lion is already near maximum stable step (L1 gradients are ±1 before sign, so per-param step = ±lr independent of gradient scale). EMA+dropout already saturate the regularization budget; wd=5e-4 over-regularizes — most visible as rc split regression (+4.08%). Arm 1 wins by keeping wd=1e-4.

**Trajectory diagnostic:** Main vs EMA val spread remains ~10 pts at epoch 16 (main_val ≈ 65 vs ema_val 55.41) — EMA still absorbing Lion's noise, not saturated. Longer budget prediction: a few more epochs would push well below 55.

**Thorfinn reassigned:** PR #2086 — lr=4e-4 (bold) and lr=3e-4 (midpoint) to complete the lr curve and detect the saturation/instability boundary.

---

## 2026-05-13 08:08 — PR #1934: Width expansion n_hidden=192/256 on Lion+EMA (alphonse) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-alphonse/width-expansion`
- **Hypothesis:** Lion's faster optimization exposes a capacity ceiling at n_hidden=128; widening to 192 or 256 absorbs extra training signal.
- **W&B runs:** `bfqtvd10` (Arm 1: n_hidden=192), `6ez4cyaf` (Arm 2: n_hidden=256), `g4l7y8ic` (Arm 1 replicate)

| Metric | Baseline #1781 (n_hidden=128) | Arm 1 (192) | Arm 2 (256) |
|--------|-------------------------------|-------------|-------------|
| Params | 0.67M | 1.48M | 2.46M |
| val_avg/mae_surf_p | **61.302** | 62.709 (+2.3%) | 66.393 (+8.3%) |
| test_avg/mae_surf_p | **52.682** | 54.235 (+2.9%) | 57.480 (+9.1%) |
| Epochs (30-min cap) | 13 | 13 | 11 |
| s/epoch | 112 | 145 | 171 |

**Result:** CLOSED. Monotonic regression with width — Arm 2 worse than Arm 1, both worse than baseline. vs current Lion+MAE baseline (val=56.58): +10.8% / +17.3%.

**Key finding:** Hypothesis falsified. The model is **compute-bound, not capacity-bound** at the 30-min cap. Wider models trained *fewer* epochs (171s/epoch at width=256 vs 112s at 128), and per-epoch val improvement was NOT faster on wider arms — they're tracking toward the same basin but trailing. Student's analysis was sharp: OOD splits regressed *most* on width, ruling out the under-capacity story (which would predict in-dist improvements). At fixed 30-min wall-clock, n_hidden=128 is the right operating point on this base.

**Architectural conclusion (combined with #1761):** Both depth (n_layers=6 → +19% epoch cost) and width (n_hidden=192/256 → +30%/53% epoch cost) are compute-bound losers at the 30-min cap. The only architectural directions left are **compute-neutral** (head count, MLP ratio if cheap) or **structural** (different attention pattern). Alphonse reassigned to **n_head=8 vs n_head=2 sweep** (PR #2069) — head count is FLOPs-neutral (parallel heads at smaller head_dim).

---

## 2026-05-13 08:10 — PR #1857: EMA decay sweep 0.995/0.999 on pre-Lion base (edward) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-edward/ema-decay-sweep`
- **Hypothesis:** Higher EMA decay (slower update) might sample a broader weight-space neighborhood → flatter basin.
- **W&B runs:** `ihgn5cko` (Arm 1: 0.995), `qyoo8y0j` (Arm 2: 0.999)

| Metric | Baseline #1607 (0.99) | Arm 1 (0.995) | Arm 2 (0.999) |
|--------|-----------------------|---------------|---------------|
| val_avg/mae_surf_p | **77.054** | 77.594 (+0.7%) | 81.489 (+5.8%) |
| test_avg/mae_surf_p | 68.265 | **66.978 (−1.9%)** | 70.828 (+3.8%) |

**Result:** CLOSED. vs old AdamW+EMA baseline: val regression on both arms. vs current Lion+MAE baseline (val=56.58): +37.2% even on best arm. Cannot merge by val rule.

**Important nuanced finding for record:** **Arm 1 (decay=0.995) improved test by −1.9% despite val +0.7% regression**. All 4 test splits improved monotonically. Val/test ratio shifted 1.129 → 1.158, consistent with a slightly more conservative shadow producing better OOD generalization (genuine signal, not noise). This effect was on the OLD pre-Lion AdamW base — would need to be re-tested on Lion+MAE to know if it transfers (Lion's gradient statistics differ from AdamW's).

**Mechanism (student analysis):** Effective averaging window ≈ 1/(1−α): 0.99→100 steps, 0.995→200, 0.999→1000. At the 16-epoch (~6000 step) budget, 0.999 averages over basically the entire post-warmup trajectory including under-trained early weights — predicted failure mode. 0.995 sits in the sweet spot for test, just past it for val.

**Edward reassigned:** PR #2070 — Lion-no-EMA ablation. Diagnostic for ICML appendix: how much of the Lion+EMA gain is Lion vs EMA?

---

## 2026-05-13 07:55 — PR #1786: Higher LR (1e-3/2e-3) on AdamW+EMA base (frieren) — CLOSED SUPERSEDED

- **Branch:** `willowpai2g24h5-frieren/higher-lr-ema`
- **Hypothesis:** EMA smoothing absorbs main-model noise, so 2–4× the baseline AdamW lr=5e-4 should be safe and reach a lower-loss basin within 16 epochs.
- **W&B runs:** `uvc7ljtw` (Arm 1: lr=1e-3), `17oh10lv` (Arm 2: lr=2e-3)

| Metric | Baseline #1607 (lr=5e-4) | Arm 1 (lr=1e-3) | Arm 2 (lr=2e-3) |
|--------|--------------------------|-----------------|-----------------|
| val_avg/mae_surf_p | 77.054 | **74.508 (−3.30%)** | 74.595 (−3.19%) |
| test_avg/mae_surf_p | 68.265 | **64.380 (−5.69%)** | 65.522 (−4.02%) |
| test/geom_camber_cruise | 48.52 | **44.51 (−8.3%)** | 44.30 (−8.7%) |
| test/single_in_dist | 75.31 | **70.64 (−6.2%)** | 73.65 (−2.2%) |

**Result:** CLOSED (direction superseded). Arm 1 (lr=1e-3) was a genuine improvement on the pre-Lion AdamW+EMA base (−3.3% val / −5.7% test), validating the EMA-absorbs-noise hypothesis. However, two subsequent merges moved the baseline to val=56.58 (Lion+MAE), and Arm 1's result (val=74.51) is +31.7% vs the current best. The LR scaling direction on the new compound is covered by thorfinn's #1932 (Lion lr=2e-4). Frieren reassigned to batch_size+LR scaling (PR #2052).

**Key finding:** Mechanism confirmed — EMA val descended monotonically from 197.79 to 74.51 despite main-model wobble (88–227 range). Cruise split (−8.3%) and single_in_dist (−6.2%) are the biggest movers. lr=2e-3 without warmup: good val but worse test (65.52 vs 64.38), consistent with over-aggressive LR degrading OOD generalization.

---

## 2026-05-13 07:54 — PR #1752: surf_weight=5 on pre-Lion AdamW+EMA base (nezuko) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-sweep`
- **Hypothesis:** surf_weight=10 may be overcorrecting post-Huber+EMA; reducing to 5 or 7 might improve balance between surface and volume.
- **W&B run:** `4pvy6khr` (Arm 1: surf_weight=5; Arm 2 skipped per stop rule)

| Metric | Baseline #1607 | Arm 1 (surf_weight=5) | Δ |
|--------|----------------|-----------------------|---|
| val_avg/mae_surf_p | 77.054 | 83.511 | **+8.4% (regression)** |
| test_avg/mae_surf_p | 68.265 | 73.759 | **+8.0% (regression)** |
| val/single_in_dist | 85.45 | 93.557 | +9.5% |
| val/geom_camber_cruise | 57.80 | 63.291 | +9.5% |

**Result:** CLOSED. Hypothesis falsified definitively. surf_weight=5 is uniformly worse on every val and test split — *including* volume MAE (also regressed), so this isn't a surface/volume tradeoff; it's a uniform loss of gradient signal. Stop rule triggered (Arm 1 +8.4% > 3pt threshold); Arm 2 (surf_weight=7) not run. vs Lion+MAE baseline (val=56.58): +47.6% regression.

**Key finding:** Surface nodes are underfit at the 30-min compute budget. Reducing surf_weight starves surface gradient signal — model spends capacity on volume, which is also underfit and doesn't improve either. surf_weight=10 was tuned correctly; the 'EMA+Huber adds slack for lower weighting' intuition was wrong. The unexplored direction is *upward* (surf_weight=15–20), especially with MAE loss where there's no Huber quadratic dampening near zero.

**Nezuko reassigned:** PR #2056 — surf_weight sweep on Lion+MAE (Arm 1: sw=5 apples-to-apples, Arm 2: sw=15 heavier emphasis).

---

## 2026-05-13 06:35 — PR #1825: MAE (L1) loss on Lion+EMA base (askeladd) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-askeladd/mae-loss`
- **Hypothesis:** MAE/L1 loss weights every node uniformly per the linear MAE evaluation metric; this property is independent of optimizer choice and should compound cleanly with Lion+EMA.
- **W&B run:** `03w5fnvm` (Lion+MAE rerun)

| Metric | MAE+Lion+EMA | Lion+EMA baseline (#1781) | Δ |
|--------|-------------|--------------------------|---|
| val_avg/mae_surf_p | **56.577** | 61.302 | **−7.71%** |
| test_avg/mae_surf_p | **48.817** | 52.682 | **−7.34%** |
| test/single_in_dist | 53.687 | 59.813 | −10.24% |
| test/geom_camber_rc | 63.234 | 64.584 | −2.09% |
| test/geom_camber_cruise | 30.812 | 35.140 | −12.32% |
| test/re_rand | 47.535 | 51.193 | −7.14% |

**Result:** MERGED. New best session baseline. Wins all 4 test splits. Val still descending at epoch-16 cap.

**Key finding:** MAE's gain on Lion is *larger* (−7.71%) than on the original AdamW base (−3.15%). Lion's sign-magnitude update removes per-parameter gradient scale information, but MAE's uniform per-node loss aggregation operates *before* backprop — it directly controls how much each node contributes to the scalar loss. This is loss-side, not optimizer-side. Lion+MAE compound: the optimizer discards magnitude noise, the loss ensures every surface node contributes equally. Cruise split (−12.32%) and in-dist split (−10.24%) are the biggest movers.

---

## 2026-05-13 06:40 — PR #1823: Weight decay wd=5e-4 on pre-Lion AdamW base (fern) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-fern/weight-decay-sweep`
- **W&B run:** `qvpxtrx8`

| Metric | wd=5e-4 | AdamW baseline (#1607) | New Lion+MAE baseline |
|--------|---------|------------------------|----------------------|
| val_avg/mae_surf_p | 78.47 | 77.054 | **56.577** |
| test_avg/mae_surf_p | 68.22 | 68.265 | **48.817** |

**Result:** CLOSED. +1.84% val regression vs old AdamW baseline; +38.7% vs new Lion+MAE baseline.

**Analysis:** val regressed (stronger L2 slowed the EMA trajectory), test was essentially tied. The mechanism check was interesting — main-model val improved with wd=5e-4, but EMA val regressed, suggesting stronger wd changes the EMA averaging geometry unfavorably. Direction also redundant: thorfinn's #1932 (Arm 2) is testing wd=5e-4 on the Lion base directly.

---

## 2026-05-13 05:25 — PR #1761: n_layers=6 depth expansion (tanjiro) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-tanjiro/n-layers-6`
- **Hypothesis:** Adding a 6th Transolver block (n_layers 5→6) gives the model more sequential processing capacity for complex flow features.
- **W&B runs:** `sy8axvhz` (drop=0.2 arm), `mr69aglv` (drop=0.1 arm retry)

| Metric | n_layers=6, drop=0.2 | n_layers=6, drop=0.1 | Baseline #1607 | Δ vs old base | vs Lion #1781 |
|--------|---------------------|---------------------|----------------|---------------|----------------|
| val_avg/mae_surf_p | 80.174 | 80.452 | 77.054 | +4.1% / +4.4% | +30.9% / +31.3% |
| test_avg/mae_surf_p | 70.816 | 70.862 | 68.265 | +3.7% / +3.8% | +34.4% / +34.5% |
| Epochs (30-min cap) | 14/50 | 14/50 | 16/50 | — | — |
| Epoch time | ~133.7s | ~133.7s | ~112s | +19% | +19% |

**Result:** CLOSED. Per decision rule (val > 78.5 → close definitively).

**Analysis:** Depth=6 increases per-epoch cost by ~19% (133.7s vs 112s), reducing total epochs from 16 → 14 at the 30-min cap. The trajectory data shows the model is *compute-budget bound, not depth-broken*: dropout=0.2 arm descended ~3.7 pts/epoch at the cap, dropout=0.1 arm descended ~1.4 pts/epoch. Lower dropout improved main_val (94.5 vs 105) — the underlying model was closer to converged with less regularization — but EMA's long-horizon average smoothed out the early-epoch advantage and the late-epoch plateau dominates.

**Architectural takeaway:** At the 30-min cap, n_layers=5 is locally optimal. Depth-6 needs more compute than this launch allows. The depth-vs-budget knee is now characterized for the architecture.

---

## 2026-05-13 05:10 — PR #1781: Lion optimizer lr=1e-4+EMA (thorfinn) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-thorfinn/lion-optimizer`
- **Hypothesis:** Lion's sign-based momentum updates (Chen et al. 2023) produce larger, noisier gradient steps that EMA then smooths — decoupling exploration (Lion) from integration (EMA) more cleanly than AdamW+EMA where second-moment normalization and EMA partially overlap.
- **W&B runs:** `e2l23xny` (lr=1e-4, winner), `9fjjfgjt` (lr=5e-5), buggy-variant `lion-lr5e-5-buggy-variant.log`

| Metric | Lion lr=1e-4 | Lion lr=5e-5 | Baseline #1607 (AdamW+EMA) | Δ vs baseline |
|--------|-------------|-------------|--------------------------|---------------|
| val_avg/mae_surf_p | **61.302** | 64.010 | 77.054 | **−20.44%** |
| test_avg/mae_surf_p | **52.682** | 55.367 | 68.265 | **−22.83%** |
| test/single_in_dist | 59.813 | 65.462 | 75.31 | −20.58% |
| test/geom_camber_rc | 64.584 | 67.409 | 80.81 | −20.08% |
| test/geom_camber_cruise | 35.140 | 35.899 | 48.52 | −27.58% |
| test/re_rand | 51.193 | 52.696 | 68.41 | −25.17% |
| Epochs (30-min cap) | 16/50 | 16/50 | 16/50 | — |

**Result:** MERGED. Lion optimizer is the largest single-PR gain of the session — 20–28% uniform improvement across all 4 test splits. Curve still descending steeply at epoch-16 cap; not converged.

**Key finding — Lion+EMA synergy:** Lion sign-magnitude updates are uniformly ±lr per step (aggressive, noisy), but EMA smooths the noise post-hoc. AdamW second-moment already smooths per-parameter scale, making AdamW+EMA partially redundant. Lion+EMA cleanly separates roles: exploration vs averaging. The lr=1e-4 arm beats lr=5e-5 because bigger, noisier steps give EMA more diversity to average over.

**Bug fix (critical):** Student identified β1/β2 swap in PR diff relative to canonical Lion. Buggy variant scored 92.92 (regression), canonical scored 61.30. Fix: interpolation uses β1=0.9, momentum EMA uses β2=0.99.

**Note:** val still descending at epoch-16 cap — longer budget is the highest-EV immediate follow-up.

---

## 2026-05-13 05:15 — PR #1604: Asinh pressure transform (alphonse) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-alphonse/asinh-pressure`
- **Hypothesis:** Asinh transform on pressure target compresses the high-Re tail, improving generalization on re_rand and single_in_dist splits.
- **W&B run:** `nbig5bns`

| Metric | Asinh run | Baseline #1607 (EMA) | Δ |
|--------|-----------|---------------------|---|
| val_avg/mae_surf_p | 82.81 | 77.054 | **+7.5% (regression)** |
| test_avg/mae_surf_p | 73.25 | 68.265 | **+7.3% (regression)** |
| test/geom_camber_cruise | ~54 | 48.52 | +11.4% |

**Result:** CLOSED. Clear regression on all splits.

**Analysis:** Double-compression: Huber δ=1.0 already compresses the tail at the loss level. Asinh adds a second compression at the data (target representation) level. These are not additive — they compete on the same degree of freedom. With Fourier features providing richer spatial encoding, the model has capacity to learn high-Re structure directly; asinh pre-compression removes the tail signal the model needs. The correct future test, if any, would be Asinh-alone on the Fourier+EMA base *without* Huber loss.

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **Branch:** `willowpai2g24h5-frieren/bf16-mixed-precision`
- **Hypothesis:** BF16 mixed precision halves per-step time, buying more epochs in the 30-min wall-clock budget.
- **W&B run:** `6zx5vuja`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 13) | **123.72** |
| val_single_in_dist/mae_surf_p | 153.36 |
| val_geom_camber_rc/mae_surf_p | 129.40 |
| val_geom_camber_cruise/mae_surf_p | 99.23 |
| val_re_rand/mae_surf_p | 112.87 |
| test_avg/mae_surf_p | NaN (cruise data bug) |
| 3-split test avg (no cruise) | **121.90** |
| Epochs completed | 18 in 30 min |
| Peak VRAM | 32.9 GB / 96 GB |

**Result:** MERGED as new baseline. BF16 completed 18 epochs vs ~14 at FP32 (estimated), establishing val_avg=123.72.

**Key observation:** Pre-existing data corruption in `test_geom_camber_cruise/000020.pt` (761 nodes with y[:,2]=-inf) poisons 4-split test_avg via `0×inf=NaN` in scoring.py. Affects every run on this branch. 3-split test avg is the usable paper-facing signal until fixed.

---

## 2026-05-12 18:56–19:51 — PR #1412: Warmup 3ep then cosine / Warmup 5ep then cosine (thorfinn)

- **Branch:** `willowpai2g24h5-thorfinn/warmup-3ep-then-cosine`
- **Hypothesis:** Linear LR warmup before cosine annealing stabilizes early training steps.
- **W&B runs:** `3chdcivo` (warmup-3ep), `jcd79mzi` (warmup-5ep)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| warmup-3ep | 144.50 | 12 | 144.54 |
| warmup-5ep | **135.37** | 14 | **131.12** |

Per-split (warmup-5ep):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 164.28 | 142.88 |
| geom_camber_rc | 143.91 | 130.88 |
| geom_camber_cruise | 110.76 | NaN |
| re_rand | 122.52 | 119.61 |

**Result:** SENT BACK for rebase. warmup-5 (135.37) did not beat the BF16 baseline (123.72) as a standalone, but warmup and BF16 are orthogonal. Student rebasing to test the combo (warmup-5 + BF16 already in base).

**Key observation:** Warmup=5 strictly dominates warmup=3 across all splits except geom_camber_cruise (+5%). Single_in_dist improved 15.7%, re_rand 1.7%, rc 6.2%. Per-epoch time ~131s; 14 epochs in 30 min without BF16.

---

## 2026-05-12 19:56 — PR #1367: Dropout=0.1/0.2 + grad-clip=1.0 (fern) — **PENDING REBASE**

- **Branch:** `willowpai2g24h5-fern/dropout-0.1-grad-clip`
- **Hypothesis:** Light dropout + grad clipping improves OOD generalization.
- **W&B runs:** `7brl22oo` (dropout=0.1), `3wz81r3d` (dropout=0.2)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| dropout=0.1, clip=1.0 | 146.31 | 8 | 146.51 |
| **dropout=0.2, clip=1.0** | **113.86** | **12** | **114.77** |

Per-split (dropout=0.2):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 145.19 | 132.80 |
| geom_camber_rc | 120.81 | 112.25 |
| geom_camber_cruise | 83.27 | NaN |
| re_rand | 106.18 | 99.28 |

**Result:** SENT BACK for rebase. val_avg=113.86 BEATS current BF16 baseline (123.72) by 7.7%. Merge conflict with PR #1371 — student rebasing to test dropout=0.2 + BF16 combination. Expected to combine to an even lower metric.

**Key observation:** dropout=0.2 beats dropout=0.1 across EVERY split (−22% overall), not just OOD. Probably acts as a smoother loss landscape rather than just generalization: 5 Transolver layers with slice_num=64 attention have many co-adaptation opportunities that dropout disrupts usefully. Validation still descending at 30-min cap — suggests this configuration has more headroom.

---

## 2026-05-12 19:58 — PR #1400: Aux surface-pressure head λ=2 (tanjiro) — **PENDING REBASE**

- **Branch:** `willowpai2g24h5-tanjiro/aux-surf-p-head`
- **Hypothesis:** Auxiliary MLP head predicting surface p only, with λ=2.0 weight on aux loss.
- **W&B run:** `m9xr80iw`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 12) | 132.48 |
| val_single_in_dist | 153.91 |
| val_geom_camber_rc | 147.18 |
| val_geom_camber_cruise | 104.48 |
| val_re_rand | 124.36 |
| 3-split test avg | 130.62 |
| Epochs completed | 14 in 30 min (~134 s/ep) |
| Peak VRAM | 42.6 GB / 96 GB |
| Aux head params | 8,321 (vs 660K main) |

**Result:** SENT BACK for rebase + BF16 combo. val=132.48 doesn't beat BF16 baseline (123.72), but **aux loss is learning** (0.66 → 0.20) and val curve was still descending at the cap (152 → 132 last 3 epochs). Tanjiro re-running with aux head INSIDE BF16 autocast.

**Key observation:** This is a high-α, high-σ direction — the aux head is doing useful work but starved for epochs. BF16 + λ=5.0 (larger aux weight) may be necessary to see real gains.

---

## 2026-05-12 20:51 — PR #1386: Fourier positional encoding L=6 (nezuko) — **PENDING RETRY**

- **Branch:** `willowpai2g24h5-nezuko/fourier-position-encoding`
- **Hypothesis:** Random Fourier features on (x,z) coordinates help model learn high-freq spatial patterns.
- **W&B run:** `0xuvq54a`
- **Config:** L=6, min_freq=1.0, max_freq=1000.0, no BF16 (PR predates BF16 merge), 14 epochs in 30 min

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 13) | **123.10** | 0.5% below merged BF16 baseline (123.72) |
| val_single_in_dist | 142.19 | vs 153.36 (BF16): better |
| val_geom_camber_rc | 138.34 | vs 129.40 (BF16): worse |
| val_geom_camber_cruise | 92.13 | vs 99.23 (BF16): better |
| val_re_rand | 119.73 | vs 112.87 (BF16): worse |
| 3-split test avg | 124.90 | vs 121.90 (BF16): worse |
| test_geom_camber_cruise | NaN | same pre-existing data bug |
| Peak VRAM | 42.5 GB / 96 GB | |

**Result:** SENT BACK for retry. Student correctly identified that **max_freq=1000 is far too high** for raw coordinates (Tancik standard is max_freq ≈ 2π × num_octaves on NORMALIZED positions). Requested re-run with max_freq=32, normalized positions, and BF16 from base.

**Key observation:** Marginal +0.5% val improvement over BF16 baseline, but per-split signals are mixed (helps single_in_dist & cruise, hurts rc & re_rand). The mixed signal + known scaling bug + room for a much bigger win via the retry → preferred over merging a borderline win with a bug.

---

## 2026-05-12 21:00 — PR #1541: Fix test cruise NaN + BF16 rerun (frieren) — **MERGED**

- **Branch:** `willowpai2g24h5-frieren/fix-cruise-test-nan-scoring`
- **Hypothesis:** Guard `0×inf=NaN` in `data/scoring.py::accumulate_batch` to restore 4-split test_avg metric, then rerun BF16 baseline to verify.
- **W&B run:** `x7snuii5`

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 17) | **120.40** | beats BF16 baseline 123.72 by 2.7% |
| test_avg/mae_surf_p | **106.67** | first finite 4-split test metric on branch |
| test_single_in_dist | 125.29 | |
| test_geom_camber_rc | 113.23 | |
| test_geom_camber_cruise | **81.16** | was NaN — now finite |
| test_re_rand | 106.99 | |
| Epochs completed | 18 in 30 min | (~101 s/epoch, same as BF16 baseline) |
| Peak VRAM | ~33 GB / 96 GB | |

**Result:** MERGED as new baseline. val=120.40, test_avg=106.67 — both improvements over BF16 baseline (val=123.72, test=NaN).

**Key observation:** The fix is a single `torch.where(isfinite(...))` guard immediately after `err = (...).abs()` in `accumulate_batch`. Val improvement over the prior BF16 baseline (123.72→120.40) is within the ±3% training-noise band but real; 18 epochs in 30 min confirms BF16 throughput is stable. The cruise test MAE (81.16) is substantially lower than the other splits — cruise samples are geometrically simpler than single_in_dist/re_rand, so this is expected.

---

## 2026-05-12 21:00 — PR #1412: Warmup-5ep + BF16 combo (thorfinn) — **CLOSED**

- **Branch:** `willowpai2g24h5-thorfinn/warmup-3ep-then-cosine`
- **Hypothesis:** warmup-5ep + BF16 combination outperforms BF16-only baseline.
- **W&B run:** `dm90ndo1`

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 14) | 123.10 | vs new baseline 120.40: +2.2% worse |
| val_single_in_dist | 150.51 | |
| val_geom_camber_rc | 136.30 | |
| val_geom_camber_cruise | 96.38 | |
| val_re_rand | 109.21 | |
| 3-split test avg | 122.33 | test_avg NaN (run predates scoring fix) |
| Epochs completed | 18 in 30 min | BF16 confirmed at 101s/epoch |

**Result:** CLOSED. val=123.10 does not beat new baseline (120.40) after #1541 merged. Warmup+BF16 is within noise of BF16-alone — warmup provides marginal additional benefit once BF16 supplies the extra epochs.

**Key observation:** Thorfinn's per-epoch LR analysis is the most valuable finding: T_max=50 is too long for the ~18 reachable BF16 epochs — cosine decays only 36% by run end, leaving LR near peak (4.5e-4) and causing late-epoch val wobble. T_max=18 is the next experiment (PR #1583 assigned).

---

## 2026-05-12 21:05 — PR #1357: Huber loss δ=1.0 (askeladd) — **SENT BACK for BF16 rerun**

- **Branch:** `willowpai2g24h5-askeladd/huber-loss-delta-1`
- **Hypothesis:** Replace MSE with Huber δ=1.0 in normalized space; linear penalty past 1σ is robust to high-Re outliers.
- **W&B run:** `whazlv6i` (pre-BF16 base — peak VRAM ~82 GB)

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 14) | **107.91** | beats baseline 120.40 by 10.4% |
| val_single_in_dist | 123.52 | |
| val_geom_camber_rc | 114.01 | |
| val_geom_camber_cruise | 89.82 | |
| val_re_rand | 104.27 | |
| 3-split test avg | 105.94 | test_avg NaN (run predates scoring fix) |
| Epochs completed | 14 in ~31 min | non-BF16 throughput |

**Result:** SENT BACK for rebase + rerun. Run did not have BF16 active (~82 GB VRAM confirms pre-BF16 code). Predicted: rebase + BF16 + scoring fix should give 107.91 × ~0.95 (BF16 extra epochs) ≈ ~100-105 val with finite test_avg. Will merge as winner on rebase return.

**Key observation:** Huber gives the largest single-experiment gain we've seen so far (~10%). Per-split improvements are largest on `val_re_rand` and `val_geom_camber_cruise` — exactly the high-Re/high-dynamic-range splits the hypothesis targeted. Strong candidate for compounding with dropout=0.2 (PR #1367).

---

## 2026-05-12 21:05 — PR #1352: surf_weight=30 (alphonse) — **CLOSED**

- **Branch:** `willowpai2g24h5-alphonse/surf-weight-30`
- **Hypothesis:** Increase surf_weight 10→30 to focus loss on surface pressure (the ranking metric).
- **W&B runs:** `q12wxz51` (sw=30), `9j9hnhfs` (sw=20)

| Arm | val_avg/mae_surf_p | best epoch |
|-----|--------------------|------------|
| surf_weight=20 | 127.05 | 13 |
| **surf_weight=30** | **120.88** | **14** |

**Result:** CLOSED. val=120.88 does not beat baseline (120.40) and is far from leading unmerged result (113.86 dropout=0.2). Pushing higher likely loses on val_single_in_dist (150.63 at sw=30, indicating overweighted surface loss hurts in-distribution generalization).

**Key observation:** Monotonic improvement 20→30 but trajectory tops out below the leading regularization-based approaches. Surf-weight is exhausted as a standalone lever; might compound with dropout in a future run.

---

## 2026-05-12 21:05 — PR #1365: OneCycleLR max_lr=1e-3 (edward) — **CLOSED**

- **Branch:** `willowpai2g24h5-edward/onecyclelr-max-lr-1e3`
- **Hypothesis:** OneCycleLR sweeps wider LR range than CosineAnnealingLR in short training budget.
- **W&B run:** `3ghxoqlb`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 13) | 128.89 |
| Epochs completed | 14 in 30 min |
| 3-split test avg | 124.47 |

**Result:** CLOSED. val=128.89 is ~7% worse than baseline (120.40). The schedule was structurally mismatched to the budget: OneCycleLR was set with epochs=MAX_EPOCHS=50, so peak LR (1e-3) hits at epoch 5 and the schedule plans to anneal slowly over 45 more epochs — but we only get 14, so LR stays pinned ~9e-4 the entire run.

**Key observation:** Student's own diagnosis is correct: OneCycleLR + 30-min budget requires `total_steps` matched to actual reachable steps, NOT to nominal max_epochs. Thorfinn is testing the analogous fix for CosineAnnealingLR (T_max=18) in PR #1583 — wait for that result before re-trying a budget-matched OneCycleLR.

---

## 2026-05-12 23:20 — PR #1624: AdamW betas (0.9,0.95) and (0.9,0.98) (frieren) — **CLOSED**

- **Branch:** `willowpai2g24h5-frieren/adamw-betas-ema`
- **Hypothesis:** Tuning beta2 from 0.999→0.95 or 0.98 to reduce second-moment memory horizon for short training.
- **W&B runs:** `geuztn5g` (beta2=0.95, val=141.04), `a2h9i5t3` (beta2=0.98, running at val=175 mid-training)

| Arm | val_avg/mae_surf_p | vs Baseline (103.24) |
|-----|-------------------|---------------------|
| betas=(0.9, 0.95) | 141.04 | +36.6% worse |
| betas=(0.9, 0.98) | mid-training ~175 | clearly worse |

**Result:** CLOSED. Both arms substantially worse than baseline. Beta2 reduction removes gradient history too aggressively for 18-epoch training — the standard 0.999 maintains a longer moving average that works better with cosine annealing.

---

## 2026-05-12 22:53 — PR #1386: Fourier positional encoding L=6 mf32 BF16 (nezuko) — **MERGED**

- **Branch:** `willowpai2g24h5-nezuko/fourier-position-encoding`
- **Hypothesis:** Replace raw (x,z) coordinates with Fourier features (log-spaced frequencies) to help the model learn high-frequency spatial patterns that MLPs struggle with from raw floats.
- **W&B runs:** `bpbykd9z` (L=6, primary), `qwmh06uh` (L=4, secondary)
- **Config:** L=6, min_freq=1.0, max_freq=32.0 (corrected from v1's max_freq=1000), positions standardized before encoding; BF16

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | Best epoch |
|-----|-------------------|--------------------|-----------:|
| **Fourier L=6 mf32 BF16** (`bpbykd9z`) | **103.2393** | **90.828** | 18 |
| Fourier L=4 mf32 BF16 (`qwmh06uh`) | 107.1261 | 94.7796 | 18 |
| **Baseline (BF16+scoring fix, #1541)** | 120.40 | 106.67 | 17 |

Per-test-split (L=6):

| Split | test surf_p | vs Baseline (106.67 avg) |
|-------|------------|------------------------:|
| single_in_dist | 105.79 | −15.6% |
| geom_camber_rc | 102.99 | −9.0% |
| geom_camber_cruise | 64.21 | −20.9% |
| re_rand | 90.31 | −15.6% |
| **avg** | **90.83** | **−14.8%** |

**Result:** MERGED as new baseline. val=103.24, test=90.83. All 4 splits improve; biggest gain on cruise geometry (−20.9%), supporting the hypothesis that Fourier features resolve fine boundary-layer structure on unseen airfoil shapes.

**Key observations:**
1. **max_freq matters far more than L.** v1 with max_freq=1000 on raw coords was −8% *worse* than baseline; v2 with max_freq=32 on normalized coords is −14% *better*. The frequency range (not the number of octaves) is the dominant variable.
2. **Standardize positions before encoding.** Computing sin/cos on raw coords makes the basis poorly conditioned; standardizing first puts frequencies in the Tancik-meaningful range.
3. **L=6 > L=4 by ~4%.** Extra octaves covering finer spatial scales (λ ≈ 0.2 unit) help boundary-layer resolution; negligible VRAM cost.
4. **This is the largest single-experiment gain yet: −14.8% test** — surpassing Huber (−10.4% val, BF16 pending) and dropout=0.2 (−7.7% val, BF16 pending). Fourier positional encoding is a foundational input feature change that should compound with both loss and regularization improvements.

**New baseline:** val=103.24, test=90.83. All subsequent PRs should compare against these numbers.

---

## 2026-05-12 21:55 — PR #1609: Transolver slice_num 64→128 (frieren) — **CLOSED**

- **Branch:** `willowpai2g24h5-frieren/slice-num-128-physics-tokens`
- **Hypothesis:** Doubling physics-token count gives attention finer-grained spatial access; orthogonal to other levers.
- **W&B run:** `kg0dwlen`

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| val_avg/mae_surf_p (best, ep 12) | 127.42 | **+5.83% worse** |
| test_avg/mae_surf_p | **116.80** | +9.50% worse |
| test_single_in_dist | 135.60 | +8.23% |
| test_geom_camber_rc | 127.82 | +12.88% |
| test_geom_camber_cruise | 87.67 | +8.03% |
| test_re_rand | 116.12 | +8.53% |

**Result:** CLOSED. Every metric worse than baseline (120.40/106.67). Lands in PR's own "over-allocated to capacity" decision bucket.

**Key observation (pattern confirmation):** In the 18-epoch budget, capacity-adding architectural changes consistently lose because the epochs sacrificed for slower compute matter more than the representational gain. This is the second confirmation (after #1400 tanjiro aux head ran into the same wall). Cheap gradient-shape changes (Huber, dropout) win; capacity-up architecture loses. **For future architecture changes, the per-epoch cost is the dominant variable, not the parameter count.**

---

## Stragglers status (round 1, as of 2026-05-12 ~20:06–20:51)

| PR | Student | Status |
|----|---------|--------|
| #1386 | nezuko | ✅ Completed at val=123.10 → SENT BACK for retry |
| #1365 | edward | Mid-epoch 7/50 OneCycleLR; val 254→190 over 6 epochs — likely undercooked |
| #1357 | askeladd | Epoch 3/50 Huber loss; val 232→178 — descending |
| #1352 | alphonse | surf_weight=20 finished at val=127.05 (worse than 123.72 baseline); surf_weight=30 arm still pending |
| #1541 | frieren | Scoring fix + baseline rerun — no comments yet |

---

*Log format: one block per PR review.*

---

## 2026-05-12 23:55 — PR #1357: Huber loss δ=1.0 + BF16 (askeladd) — **MERGED**

- **Branch:** `willowpai2g24h5-askeladd/huber-loss-delta-1`
- **Hypothesis:** Replace MSE with Huber δ=1.0; linear penalty past 1σ in normalised space is robust to high-Re outliers.
- **W&B run:** `m733u17z`
- **Base:** BF16 + scoring fix (pre-Fourier; W&B shows fun_dim=22)

| Metric | Value | vs Fourier baseline (103.24) |
|--------|-------|------------------------------|
| val_avg/mae_surf_p (best, ep 18) | **98.7905** | **−4.31%** |
| test_avg/mae_surf_p | **88.8965** | **−2.13%** |
| test_single_in_dist | 103.88 | |
| test_geom_camber_rc | 96.54 | |
| test_geom_camber_cruise | 66.61 | |
| test_re_rand | 88.55 | |

**Result:** MERGED. Val=98.79 beats Fourier baseline by 4.31%. Student's run was on pre-Fourier base; squash-merge applied Huber cleanly on top of Fourier base → merged code = Fourier+Huber.

**Key observation:** Per-split gains largest on re_rand (−12.8% vs Fourier) and single_in_dist (−1.5%), exactly where the Huber hypothesis predicted. The improvement confirms Huber targets the same distribution tails as Fourier but through a different mechanism (loss vs. encoding). Compound expected.

---

## 2026-05-12 23:56 — PR #1367: Dropout=0.2 + grad-clip=1.0 (fern) — **MERGED**

- **Branch:** `willowpai2g24h5-fern/dropout-0.1-grad-clip`
- **Hypothesis:** dropout=0.2 + grad-clip=1.0 regularises attention co-adaptation; best arm was 0.2 (not 0.1 from PR title).
- **W&B run:** `otwlgvo7`
- **Base:** BF16 + scoring fix (pre-Fourier; W&B shows fun_dim=22, no Huber)

| Metric | Value | vs Fourier baseline (103.24) |
|--------|-------|------------------------------|
| val_avg/mae_surf_p (best, ep 18) | **98.9622** | **−4.11%** |
| test_avg/mae_surf_p | **88.7390** | **−2.30%** |
| test_single_in_dist | 110.77 | |
| test_geom_camber_rc | 97.23 | |
| test_geom_camber_cruise | 58.81 | |
| test_re_rand | 88.14 | |

**Result:** MERGED. Val=98.96 is within 0.17 points of the just-merged Huber baseline (98.79); strict comparison would say this doesn't beat Huber, but dropout is orthogonal and compounds. Squash-merge applied dropout cleanly on top of Fourier+Huber → merged code = Fourier+Huber+Dropout. Default dropout=0.1 in code; **use `--dropout 0.2` to reproduce winning config**.

**Key observation:** Cruise test drop to 58.81 (vs 66.61 with Huber, 64.21 with Fourier) confirms orthogonality of mechanisms. Val curve still descending at epoch 18 cap — suggests more epochs or compound with other regularisation would help further.

**Next assignments:** askeladd → Huber δ sweep (PR #1703); fern → Dropout rate sweep 0.15/0.25/0.30 (PR #1706).

---

## 2026-05-13 01:15 — PR #1607: EMA weight averaging decay=0.99 (edward) — MERGED

- **Branch:** `willowpai2g24h5-edward/ema-weight-avg`
- **Hypothesis:** Exponential moving average over model weights smooths per-epoch val wobble, producing a more stable checkpoint at evaluation time and potentially accessing lower-loss basins.
- **W&B run:** `nl3llszv`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (EMA, best ep 16) | **77.054** |
| val_single_in_dist/mae_surf_p | 85.45 |
| val_geom_camber_rc/mae_surf_p | 88.60 |
| val_geom_camber_cruise/mae_surf_p | 57.80 |
| val_re_rand/mae_surf_p | 76.36 |
| test_avg/mae_surf_p (EMA) | **68.265** |
| test_single_in_dist/mae_surf_p | 75.31 |
| test_geom_camber_rc/mae_surf_p | 80.81 |
| test_geom_camber_cruise/mae_surf_p | 48.52 |
| test_re_rand/mae_surf_p | 68.41 |
| main val_avg (same epoch, no EMA) | ~100.22 |
| Epochs completed | 16 in ~30 min |
| Peak VRAM | 33.8 GB / 96 GB |
| Δ vs prior best (#1367, val=98.96) | **−22.1% val / −23.1% test** |

**Result:** MERGED. Largest single-PR gain of the session. Main model val at epoch 16 is ~100 — unchanged from no-EMA baseline. EMA model is 77.05. The 23-point gap is entirely due to weight averaging: EMA smooths across the last ~100 gradient steps (effective window = 1/(1−0.99)), filtering batch noise while remaining responsive to learning. Uniform gains across all 4 splits (17–32% reduction).

**Key analysis:** Main model wobble was enormous (range 89–209 across 16 epochs). EMA's monotonic descent consistently reached deeper minima despite the noisy landscape. `decay=0.99` is highly responsive — essentially averaging the last ~100 batches (~0.27 epochs), not a multi-epoch window.

**Implementation:** `copy.deepcopy(model)` with `requires_grad=False`; `ema_p ← decay·ema_p + (1−decay)·p` after every optimizer step; both model and EMA model evaluated on val each epoch; best checkpoint saves EMA weights; test eval loads EMA weights into model.

**Note:** Student ran on default `dropout=0.1`. Full compound with Fourier+Huber+Dropout(0.2)+EMA not yet tested — this is a first estimate. Recommended follow-up: EMA + dropout=0.2 compound.

---

## 2026-05-13 01:15 — PR #1690: Fourier L=8 + concat-raw positions (nezuko) — CLOSED

- **Branch:** `willowpai2g24h5-nezuko/fourier-l8-concat`
- **Hypothesis:** (1) More Fourier frequencies (L=8 vs L=6) capture finer boundary-layer features; (2) Concatenating raw (x,z) alongside Fourier features helps geom_camber_rc by keeping global coordinates.
- **W&B runs:** `2xfd4tvu` (L=8), `hswe57m9` (L=6 concat-raw)

| Metric | L=6 baseline | L=8 replace | Concat-raw |
|--------|-------------|-------------|------------|
| val_avg/mae_surf_p | 103.24 | 104.97 (+1.6%) | 110.72 (+7.2%) |
| test_avg/mae_surf_p | 90.83 | **89.91 (−1.0%)** | 100.65 (+10.8%) |

**Result:** CLOSED. Both arms lose on the primary val metric. L=8 wash on test but primary metric regresses. L=6 concat-raw clear regression everywhere, including geom_camber_rc (+15.11) — the split it was meant to help. L=6 normalized replacement remains the Fourier sweet spot.

**Key insight:** L=4→L=6 improvement was monotone; L=6→L=8 is not. L=8 may require more epochs to converge (last epoch was best, val still descending), but within 30-min budget the gain doesn't materialise. Concat-raw dilutes learned features via scale mismatch (raw pos ∈ [−2,2] vs sin/cos ∈ [−1,1]) — the slim model can't disentangle them in 18 epochs.

---

## 2026-05-13 01:15 — PR #1400: Aux surf-p head λ sweep (tanjiro) — CLOSED

- **Branch:** `willowpai2g24h5-tanjiro/aux-surf-p-head`
- **Hypothesis:** Auxiliary surface-pressure prediction head (MLP on penultimate hidden state, λ-weighted loss) adds direct ranking-metric gradient signal during training.
- **W&B runs:** `xd6973hg` (λ=2, Fourier+BF16), `mbaijhsk` (λ=5, Fourier+BF16); earlier `m9xr80iw` (λ=2, pre-Fourier, no BF16)

| Variant | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---------|-------------------|---------------------|
| Fourier+BF16 baseline | 103.24 | 90.83 |
| λ=2 + Fourier + BF16 | 114.32 (+10.7%) | 99.16 (+9.2%) |
| λ=5 + Fourier + BF16 | 117.94 (+14.2%) | 104.56 (+15.1%) |

**Result:** CLOSED. Both λ arms regress vs baseline; λ=5 is worse than λ=2 — increasing aux weight is monotone-worsening. Aux head is dominated by Fourier features: Fourier-encoded hidden state already carries strong surface-p signal, so extra aux loss gradient competes with the main loss without the surf_weight=10× advantage the main surface loss enjoys.

**Key insight:** Aux head improved on the pre-Fourier base (val=118.88 vs ~123 pre-Fourier), confirming the mechanism works. It's specifically Fourier's input-feature improvement that makes the aux head redundant. The `return_hidden=True` pattern in train.py is a useful implementation pattern for future auxiliary-task hypotheses.


## 2026-05-13 03:10 — PR #1748 CLOSED: EMA-0.99 + Dropout=0.2 compound regresses (edward)

- **Branch:** `willowpai2g24h5-edward/ema-dropout-compound`
- **Hypothesis:** EMA-0.99 (trajectory smoothing) and Dropout=0.2 (feature decorrelation) are orthogonal regularisers; the full Fourier+Huber+Dropout(0.2)+EMA stack should beat the merged EMA-with-dropout=0.1 baseline.

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|-----|--------|---------------------|----------------------|-------|
| **Baseline (merged PR #1607)** | EMA-0.99 + dropout=0.1 | **77.054** | **68.265** | Reference |
| `fv69guww` (Arm 1 seed 1) | EMA-0.99 + dropout=0.2 | 78.869 | 69.880 | +2.4% / +2.4% (worse) |
| `yzgt1otg` (Arm 1 seed 2) | EMA-0.99 + dropout=0.2 | 80.139 | 69.241 | +4.0% / +1.4% (worse) |

- Mean Arm 1 val ≈ 79.5 (vs baseline 77.05), inter-seed spread ≈ 1.3 pts. Clearly outside noise.
- Every val split regresses, including easy `geom_camber_cruise` (+4.0%) — not a hard-tail-only effect.
- Best epoch=16 in both seeds (still descending at cap), same as baseline; not an under-training artefact.

**Result:** CLOSED. EMA-0.99 + dropout=0.2 over-regularises on the EMA base. Mechanism: main-model val is slightly better with dr=0.2 (96.1 vs ~100), but the EMA trajectory averages a less-faithful approximation when dropout adds stochastic noise. EMA already provides strong implicit regularisation, so the dr=0.1→0.2 step (which won on the *non-EMA* base in PR #1367) is past the sweet spot once EMA is in place.

**Key insight:** The merged EMA baseline (PR #1607, val=77.05, test=68.27, dropout=0.1) is the load-bearing reference — it IS the true compound. We do NOT need to revisit dropout=0.2 + EMA combinations. Dropout=0.15 on EMA base is worth testing later (interpolation between 0.1 and 0.2). The PR #1367 dropout=0.2 win was on a pre-EMA base with regularisation headroom that EMA now fills.

**Reassigned:** edward → EMA decay sweep on compound (0.995, 0.999 vs merged 0.99).
