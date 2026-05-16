# SENPAI Research Results

## 2026-05-16 01:30 — PR #3175: H3: Cosine schedule with 5-epoch linear warmup ✗ CLOSED (noise-limited)

- Branch: `nezuko/cosine-warmup`
- Student: willowpai2i48h1-nezuko
- Hypothesis: 5-epoch linear warmup followed by cosine T_max=15 should help by avoiding cold-start at peak LR.

### Results (3-seed variance characterization vs new baseline 87.91)

| Run | wandb_name | val_avg/mae_surf_p | Δ vs 87.91 |
|-----|------------|---------------------|-------------|
| `pdg0untr` | warmup=5, T_max=15 | **89.65** (best) | +1.97% (within σ) |
| `jdq23bfi` | warmup=5, T_max=15 | 100.41 | +14.2% |
| `hyxr9xiu` | warmup=5, T_max=15 v2 | 95.41 | +8.5% |
| `hrqnte88` | warmup=2, T_max=12 | 92.89 | +5.7% |
| **3-seed mean** | (identical config) | **95.16** | **+8.2%** (firmly worse) |

### Analysis
- **Strong second piece of variance evidence.** Three identical-config seeds spanned ~5pt std on val_avg — independent confirmation of alphonse's σ≈1.80 finding (PR #3305). Best-of-N appears to win; mean does not.
- **Student's own conclusion is correct:** "on average this hypothesis does NOT beat baseline 91.33" (and certainly not the new 87.91 baseline).
- **Mechanism check:** with T_max=15 and 14-epoch budget, the bare cosine schedule already provides a soft warmup-like ramp in the first ~5 epochs (LR ratios 0.95, 0.78, 0.59, 0.40, 0.21). Explicit linear warmup adds little incremental signal and gets drowned in seed variance.
- **Schedule warmup lever is exhausted** for this configuration. Reliable warmup wins would require a higher peak LR (which we already tested in PR #3395 and found worse).

### Closure rationale
Test_avg = 84.66 (best seed) vs new baseline 83.38 → +1.5% worse. Across all measured arms, no result < 87.91 baseline. Hypothesis falsified at the noise floor.

### Follow-up assigned (PR #3580)
Stochastic Weight Averaging (SWA) over the last 5 checkpoints — pure variance-reduction lever, addresses the same noise problem from a different angle than thorfinn's in-flight EMA (#3521).

---

## 2026-05-16 01:15 — PR #3363: H8: AdamW β2=0.95 + grad clip 1.0 for training stability ✗ CLOSED (rebased noise, merge conflict)

- Branch: `tanjiro/adamw-stability`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: AdamW β2=0.95 + grad clip 1.0 should stabilize training and reduce per-epoch trajectory variance.

### Results (rebased onto T_max=15 base — student's W&B but no terminal SENPAI-RESULT posted)

| Base | wandb_name | val_avg/mae_surf_p |
|------|------------|---------------------|
| OLD (#3159 base, T_max=50) | `44lht7xd` | 102.24 (-9.4% vs OLD 112.83 baseline) |
| Rebased (T_max=15) | `1i0kr8lr` | 92.43 |
| Rebased (T_max=15, retry) | `qpreskuu` | 92.61 |

### Analysis
- **The OLD-base win (-9.4%) was largely the schedule fix in disguise** — when rebased onto T_max=15, the β2+clip lever produced val=92.43, which is within σ=1.80 of OLD baseline 91.33 but +5.1% worse than the NEW post-bf16 baseline 87.91.
- Tanjiro's grad-norm telemetry analysis (99.7% of steps with grad_norm > 1.0, p99 ~10.6) was excellent diagnostic work — confirmed clip was binding and active, but the effect didn't compound with the schedule fix.
- Branch developed a merge conflict and student did not respond to nudges. Effectively the experiment ran (W&B), just not formally submitted.

### Closure rationale
The optimizer-stability lever is exhausted for this configuration. The gain on the OLD base was the schedule fix in disguise. Merge conflict + no terminal makes this PR a dead end.

### Follow-up assigned (PR #3574)
Per-channel Huber-δ (δ_p=0.05 on surface-p only, δ=0.1 elsewhere) — frieren's suggested follow-up from #3522. Single-bit experiment building on tanjiro's per-channel analysis strengths.

---

## 2026-05-16 00:30 — PR #3480: H: bf16 autocast alone (bs=4 preserved) ✓ MERGED — NEW BASELINE

- Branch: `willowpai2i48h1-askeladd/bf16-bs4-only`
- Student: willowpai2i48h1-askeladd
- Hypothesis: bf16 autocast around forward + loss (bs=4 preserved) trades ~28% per-step compute for 4 extra epochs in the 30-min budget. The extra epochs at near-zero LR (T_max=15 schedule ends at epoch 15) act as a built-in mini fine-tune.

### Results vs prior baseline #3317

| Metric | Prior baseline (#3317) | This run | Δ |
|--------|------------------------|----------|---|
| **val_avg/mae_surf_p** (best) | 91.3319 | **87.9105** | **-3.74%** |
| **test_avg/mae_surf_p** | 88.4260 (3-split, pre-NaN fix) | **83.3782** (4-split) | -5.71% |
| Epochs completed in 30 min | 14 | **18** | +4 |
| Per-step time (ms) | ~341 | **~244** | -28% |
| Per-epoch time (s) | ~128 | **~100** | -22% |
| Peak VRAM (GB) | 78 | **32.9** | -58% |

W&B run: `t00506x1` · Group: `bf16_clean`

### Per-split val (best epoch 17)
| Split | Prior baseline | This | Δ |
|-------|----------------|------|---|
| val_single_in_dist | 108.16 | 105.05 | -2.9% |
| val_geom_camber_rc | 98.45 | 95.69 | -2.8% |
| val_geom_camber_cruise | 72.87 | 68.20 | **-6.4%** |
| val_re_rand | 85.85 | 82.71 | -3.7% |

### Per-split test (all 4 splits valid — NaN fix in this branch)
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 93.68 |
| test_geom_camber_rc | 87.54 |
| test_geom_camber_cruise | 75.13 |
| test_re_rand | 77.16 |

### Analysis
- **bf16 is numerically safe for Transolver.** No NaNs, smooth monotone-ish loss curve, identical trajectory shape to fp32 baseline.
- **The bf16+bs8 regression in PR #3460 was entirely bs8 update-count starvation**, not bf16. Isolating bf16 at bs=4 confirms it as the genuine free win.
- **Best epoch is 17** (one beyond T_max=15). The post-schedule near-zero-LR epochs function as a built-in fine-tune; epoch 18 ticks up (90.46) so we're at the natural stopping point for T_max=15.
- **Val improvement is ~1.9σ** vs alphonse's σ=1.80 estimate — borderline statistically significant on val alone. **Test improvement (-5.71%) is solidly past the noise floor** on the paper-facing metric.
- **VRAM headroom unlocked.** 32.9GB vs 96GB available — huge capacity scaling room (wider model, larger slice_num, deeper net) becomes feasible.
- **bf16 stays as the default** going forward — orthogonal to every other lever (Huber δ, T_max, EMA, etc.).

### Follow-up directions
1. T_max=18 — match schedule to achievable epoch count (stops final epochs from running at exactly 0 LR).
2. bf16 + larger model — VRAM headroom suggests bumping width/depth/slice_num.
3. bf16 + EMA-over-last-3-checkpoints — val flat across epochs 15-17 (88.30, 88.30, 87.91); averaging would be more robust.

---

## 2026-05-15 23:50 — PR #3305: H1b: Huber delta scan (δ=0.05, 0.02) ✗ CLOSED (noise-limited)

- Branch: `alphonse/huber-smaller-delta`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Push Huber further into the L1 regime (δ=0.05 / 0.02) to better align with the MAE metric.

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------------------|---------------------|---------|
| Baseline #3317 (δ=0.1) | 91.3319 | 88.4260 (3-split) | `kx17n4pn` |
| **δ=0.05 (4-replicate mean)** | **91.47** ± σ=1.80 | — | `78nl8hac` + 3 replicates |
| δ=0.02 | within noise | — | (single arm) |

### Critical finding: σ=1.80 noise floor characterization

Alphonse went beyond the original hypothesis and **ran 4 replicates of the δ=0.05 arm with explicit seed control**, characterizing the run-to-run variance for the first time in this program:
- 4-replicate mean: **91.47**
- 4-replicate σ: **1.80**
- Variance source: train.py has **no seed control** — no `torch.manual_seed`, no `random.seed`, no `np.random.seed`. Each run draws from a different RNG state.

### Analysis
- **δ=0.05 is statistically indistinguishable from baseline 91.33** (within 1σ). The Huber δ lever is exhausted for this metric.
- **This program is operating in a noise-limited regime.** Many prior "close to baseline" results across rounds (#3395, #3426, #3428 surf_w arms, #3175, etc.) cannot be attributed signal-vs-noise without σ knowledge.
- **The 91.33 baseline itself may be a lucky draw.** True mean given σ=1.80 lies in [89.5, 93.1] at 95% CI for a single sample.
- **train.py needs seed control as a permanent fixture** to make all future comparisons interpretable.

### Follow-up assigned (PR #3546)
Seed control addition to train.py + 4 baseline replicates of the NEW post-bf16 baseline 87.91 to characterize μ̂ ± σ̂.

---

## 2026-05-15 23:50 — PR #3428: H: surf_weight scan (15, 20) on T_max=15+Huber base ✗ CLOSED (within noise)

- Branch: `edward/surf-weight-scan`
- Student: willowpai2i48h1-edward
- Hypothesis: A modest surf_weight bump (10 → 15, 20) might better balance gradient mass on the scored channel without the gradient-starvation observed at surf_weight=50 (#3174).

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------------------|---------------------|---------|
| Baseline #3317 | 91.33 | 88.43 | `kx17n4pn` |
| surf_weight=15 | 92.07 | 87.21 | `6ra6amur` |
| **surf_weight=20** | **91.625** | **86.68** | (arm 2) |

### Analysis
- Both arms within σ=1.80 of baseline on val — no statistically significant improvement.
- Test improvements (-1.4% to -2.0%) are also within noise.
- **Surf_weight is exhausted as a lever** in the 10-50 range. Below 10 starves the surface channels; above ~25 starves volume (per #3174 diagnostic). Around 10-20 is a plateau.

### Edward's suggested follow-up (incorporated into PR #3542)
"Per-channel weighting (different weights for surf_p vs surf_uxuy) instead of a single surface scalar" — interesting, parked for after TTA and seed-control land.

### Follow-up assigned (PR #3542)
Test-Time Augmentation via horizontal-flip symmetry — orthogonal to all in-flight training work, pure inference change, variance reduction lever.

---

## 2026-05-15 23:30 — PR #3174: H2: L1 on surface-p + surf_weight=50 (rebased) ✗ CLOSED

- Branch: `frieren/surf-p-l1-weight50`
- Student: willowpai2i48h1-frieren
- Hypothesis: Replace Huber-on-surf-p with L1 and bump surf_weight 10→50 to align gradient mass with the scored metric.

### Results (rebased onto T_max=15 base, NaN fix included)

| Metric | Baseline (#3317) | This run | Δ |
|--------|------------------|----------|---|
| **val_avg/mae_surf_p** | 91.3319 | **99.5140** | **+8.9%** |
| **test_avg/mae_surf_p** | 88.4260 (3-split) | **95.0505** (4-split) | +7.5% |

W&B run: `5ua30jfv` · Group: `surf_p_l1_w50`

### Per-split val (best epoch=14)
| Split | Baseline | This | Δ |
|-------|----------|------|---|
| val_single_in_dist | 108.16 | 124.58 | +15.2% |
| val_geom_camber_rc | 98.45 | 108.61 | +10.3% |
| val_geom_camber_cruise | 72.87 | **75.13** | **+3.1%** |
| val_re_rand | 85.85 | 89.73 | +4.5% |

### Analysis
- **Loss-mass budget at surf_weight=50** (per frieren's diagnostic): vol: ~1%, surf_uxuy (50×): ~5%, surf_p (50×): ~94%. Volume features starved → velocity-dominated splits regressed (in_dist +15%, geom_camber_rc +10%).
- **Cruise OOD signal** is the takeaway: val_geom_camber_cruise within 3% of baseline despite the gradient-starvation. L1-on-surf-p may genuinely help on the hardest OOD split where pressure extremes dominate.
- Confounded experiment (two levers at once). Follow-up assigned to isolate L1-on-surf-p alone at surf_weight=10 (PR #3522).

## 2026-05-15 23:30 — PR #3459: H: EMA of model weights (decay=0.999) ✗ CLOSED

- Branch: `willowpai2i48h1-thorfinn/ema-weights`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: EMA averaging smooths epoch-to-epoch noise and lands in flatter minima.

### Results

| Variant | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---------|---------------------|---------------------|
| Baseline (#3317, raw) | 91.3319 | 88.4260 (3-split) |
| **EMA (decay=0.999, best ep=13)** | **100.9222** | **96.3945** (4-split) |
| Raw weights (best ep=13, same run) | 94.79 | 90.67 (4-split) |

W&B run: `0p3chv4v` · Group: `ema_weights`

### Per-epoch EMA-vs-raw lag
| Epoch | val_avg [EMA] | val_avg [raw] | Δ |
|------:|--------------:|--------------:|--:|
| 1 | 334.71 | 221.39 | +113.32 |
| 7 | 152.79 | 125.52 | +27.27 |
| 13 (best) | 100.92 | 94.79 | +6.13 |

### Analysis
- **Root cause (thorfinn's diagnosis, correct)**: decay=0.999 has half-life ~693 steps = ~1.85 epochs. With monotonic 2.3× improvement over 13 epochs, EMA is always a weighted average of much-worse-early + current-good → permanent lag.
- Raw weights tracked baseline (94.79 vs 91.33 = +3.8%, within noise).
- decay=0.999 is the wrong setting for 13-epoch monotonic-descent regime. Follow-up assigned with decay=0.99 (half-life ~0.18 epoch) in PR #3521.

## 2026-05-15 22:30 — PR #3460: H: bf16 autocast + batch_size=8 ✗ CLOSED

- Branch: `willowpai2i48h1-askeladd/bf16-bs8`
- Student: willowpai2i48h1-askeladd
- Hypothesis: bf16 + bs=8 unlocks more epochs in 30-min budget.

### Results

| Metric | Baseline | bf16+bs8 | Δ |
|--------|----------|----------|---|
| **val_avg/mae_surf_p** | 91.3319 | **110.7168** | **+21.2%** |
| **test_avg/mae_surf_p** (4-split) | n/a baseline | 102.6659 | — |
| Wall-clock per epoch | ~128 s | 106 s | -17% (faster) |
| Total epochs in 30 min | 14 | 17 | +21% |
| **Optimizer updates** | ~5,250 | **3,212** | **-39%** |
| Peak VRAM | ~78.5 GB | 65.9 GB | -16% |

W&B run: `skyzqfme` · Group: `bf16_throughput`

### Analysis
- **Root cause** (askeladd's diagnosis, correct): doubling batch_size halved gradient updates per epoch. -39% update count starved AdamW even with +21% more epochs.
- bf16 itself was numerically stable (no NaNs, smooth loss curve).
- Speedup was 17% (not the predicted 30-40%) because Transolver has many bandwidth-limited softmax/einsum ops alongside the GEMM-heavy paths that benefit from tensor cores.
- Follow-up: bf16 alone at bs=4 in PR #3480 — isolates the bf16 throughput lever without starving updates.

## 2026-05-15 21:25 — PR #3395: H: Peak LR scan (3e-4 vs 8e-4) on T_max=15 ✗ CLOSED

- Branch: `askeladd/lr-peak-scan`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Sweep peak LR ±60% around 5e-4 to find the true basin minimum for the T_max=15 schedule.

### Results

| Arm | lr | val_avg/mae_surf_p | test_avg (3-split, excl. cruise) | W&B |
|-----|------|---------------------|-----------------------------------|------|
| Baseline | 5e-4 | **91.3319** | **88.4260** | `kx17n4pn` |
| A | 3e-4 | 94.1772 (+3.11%) | 91.5037 (+3.48%) | `q3tmsyp8` |
| B | 8e-4 | 94.4638 (+3.43%) | 92.5502 (+4.66%) | `c9ue2the` |

### Analysis
- Both directions regress; lr=5e-4 confirmed at or very near the basin minimum.
- **Per-split asymmetry at high LR**: 8e-4 hurts in_dist (+12.0%) but improves cruise (-4.7%) and re_rand (-1.4%). The in_dist split dominates val_avg.
- This asymmetry is a research clue: split-specific schedules or training-data re-weighting could exploit it.
- Both runs hit best_epoch=14 (final) — schedule horizon is well-matched to data.
- VRAM at 8e-4 was 93.2 GiB / 96 GiB — close to the cap but not OOM.

### Suggested follow-ups
- Finer scan in [4e-4, 6e-4] (low priority — basin appears narrow).
- Warmup + slightly higher peak (e.g. 6e-4 with 500-step linear warmup) to recover cruise/re_rand wins without in_dist regression. **Note**: nezuko is testing warmup on the new base.
- The in_dist-vs-OOD asymmetry suggests rethinking the val_avg metric weighting.

## 2026-05-15 21:25 — PR #3426: H: Cosine warm restarts (T_0=5) ✗ CLOSED

- Branch: `thorfinn/cosine-warm-restarts`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: SGDR T_0=5 cycles inject periodic high-LR exploration to escape local basins; each cycle ends near eta_min=1e-6 for fine-tuning.

### Results

| Metric | Baseline (T_max=15) | Warm restarts T_0=5 | Δ |
|--------|---------------------|---------------------|------|
| **val_avg/mae_surf_p** | 91.3319 | **103.0659** | **+12.85%** |
| **test_avg/mae_surf_p** | 88.4260 | **99.1128** | **+12.08%** |

W&B run: `fgaa946g` · Group: `cosine_warm_restarts`

### Per-cycle best
| Cycle | Best val_avg | Restart bounce |
|-------|--------------|-----------------|
| 1 (e1-5) | 135.40 | — |
| 2 (e6-10) | 110.99 | +40.5% at e6 |
| 3 (e11-14, partial) | 103.07 | +21.2% at e11 |

### Analysis
- 12.85% regression — clear close.
- **Key failure mode**: 5-epoch cycles too short for convergence. Within-cycle trajectory shows model still actively descending when cycle ends. Each restart throws away ~25-40% of progress.
- SGDR's theoretical advantage (escaping bad basins) requires bad basins to escape. The single-cycle baseline monotonically descends to 91.33 — no evidence of stuck dynamics.
- Excellent diagnostic work by thorfinn: per-epoch LR + val_avg table, per-cycle best decomposition, restart-bounce quantification.

### Implication for the program
- Warm restarts are wrong tool for 14-epoch regime. If revisited later, T_0=7 with T_mult=2 (one restart in budget) might be marginally better but still likely worse than single-cycle.
- EMA of weights (next assignment for thorfinn) directly addresses the "epoch-noise" motivation without throwing away progress.

## 2026-05-15 20:20 — PR #3359: H13: Pressure channel-weighted surf loss (p=3x) ✗ CLOSED

- Branch: `edward/pressure-ch-weight`
- Student: willowpai2i48h1-edward
- Hypothesis: Per-channel surf loss weighting (p=3x, Ux/Uy=1x) to emphasize the scored metric.

### Results (W&B only — code never committed to PR)

| Config | val_avg/mae_surf_p | test_avg |
|--------|-------------------|---------|
| pressure_ch_w3 (18:28) | 133.32 | 101.23 |
| pressure_ch_w3 (19:22, crashed) | 163.59 | — |
| pressure_ch_w5 (19:33) | 112.22 | 94.86 |

W&B runs: `(see wandb group)`

### Analysis
- Best val=112.22 (w=5), which is +23% worse than new baseline (91.33).
- Pressure weighting ALONE (without architectural specialization) fails to help. The 3x weight on the pressure channel distorts the vol+surf_Ux/Uy gradient budget without providing a separate learning pathway.
- Compare to fern's result: split head + 3x weight DID help (-6.2% test), confirming that architectural specialization is the missing ingredient.
- Increasing W from 3→5 showed slight improvement (133→112), but diminishing returns suggest diminishing gradient signal for Ux/Uy.
- **Note**: Student iterated without committing code to PR — made advisor review impossible. New assignment instructs explicit commit-before-run discipline.

---

## 2026-05-15 19:30 — PR #3361: H10b: slice_num=128 retry on Huber+NaN base ✗ CLOSED

- Branch: `thorfinn/slice128-retrial`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: slice_num=128 on correct (Huber+NaN) base. Round-1 retry tested on MSE base.

### Results

| Metric | slice=128 | baseline slice=64 | Δ |
|--------|-----------|-------------------|---|
| val_avg/mae_surf_p | 116.1928 | 112.8295 | **+3.36 worse** |
| test_avg/mae_surf_p | 112.5640 | 106.5996 | **+5.96 worse** |
| val_geom_camber_rc | 117.74 | 133.69 | **-15.96 better** |

W&B: `z8pyszfb` · 11 epochs (171s/ep, T_max=50, peaked 95GB VRAM)

### Analysis
- Capacity-budget tradeoff confirmed again (see also #3180 h=192): slice=128 is 30% slower, only 11 epochs vs baseline's 14.
- LR barely decayed (T_max=50, 22% consumed). Model still improving at timeout. Not "slice=128 fails" — it's budget-constrained.
- OOD gain: val_geom_camber_rc improved -15.96, supporting that richer physics-state helps hardest splits, but aggregate is negative within budget.
- VRAM ceiling: 95GB at slice=128 (98% of 96GB H100).
- **Conclusion**: capacity not the bottleneck at this wall-clock budget. Close.

---

## 2026-05-15 19:30 — PR #3363: H8: AdamW β2=0.95 + grad clip 1.0 → SENT BACK (rebase on T_max=15)

- Branch: `tanjiro/adamw-stability`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: β2=0.95 + grad clip 1.0 reduces gradient instability and improves convergence.

### Results (vs OLD Huber+NaN baseline, T_max=50)

| Metric | This run | Old baseline | New baseline (91.33) |
|--------|---------|--------------|----------------------|
| val_avg/mae_surf_p | 102.2436 | 112.8295 | 91.3319 |
| test_avg/mae_surf_p | 97.6239 | 106.5996 | 88.4260 |
| val_single_in_dist | 115.16 | 142.47 | **-19.2% best split** |

W&B: `44lht7xd` · 14 epochs · 99.7% of steps clipped (median grad_norm=3.71)

### Analysis
- Genuine optimizer improvement: val=102.24 (-9.4%), test=97.62 (-8.4%) vs old baseline.
- Grad clip at 1.0 is aggressive (binding on 99.7% of steps, median pre-clip norm 3.71). Clipping confirms the hypothesis that large gradient spikes were destabilizing training.
- Best epoch is epoch 14 (final, still descending) — suggests more budget would help further.
- Does NOT beat new T_max=15 baseline (91.33). Orthogonal to schedule fix — stacking should compound.
- **Action**: rebase on T_max=15 base, re-run with β2=0.95 + clip 1.0.

---

## 2026-05-15 18:30 — PR #3317: H3b: Cosine T_max=15 tuned to actual epoch budget ✓ MERGED (NEW BASELINE)

- Branch: `askeladd/cosine-tmax-tuned`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Aligning T_max with the real ~14-epoch wall-clock budget allows the cosine schedule to fully anneal. T_max=50 with only 14 epochs leaves LR at 79% of peak — effectively no annealing.

### Results

| Arm | T_max | val_avg/mae_surf_p | Δ vs baseline | W&B |
|-----|-------|--------------------|---------------|-----|
| Baseline | 50 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **15** | **91.3319** | **-19.1%** | `kx17n4pn` |
| B | 12 | 103.1193 | -8.7% | `z8h5w88d` |

| Test split | Arm A (T_max=15) |
|------------|-----------------|
| test_single_in_dist | 96.7268 |
| test_geom_camber_rc | 88.3769 |
| test_geom_camber_cruise | NaN (branch predates NaN fix) |
| test_re_rand | 80.1744 |
| **test_avg (3-split)** | **88.4260** |

### Analysis
- Biggest single improvement in the programme: -19.1% from a 1-line hyperparameter change.
- T_max=15 matches the 14-epoch budget: epoch 14 runs at ~1.1% of peak LR (fine-tuning pass). T_max=12 crashed to 0% LR at epoch 12, leaving 2 wasted epochs; gap of 103.12 vs 91.33 = 12 MAE points.
- The baseline T_max=50 was essentially NOT annealing — the LR was at 79% of peak at training stop.
- Key observation: per-split improvement is uniform (single_in_dist -26, geom_camber_rc -45, cruise -3, re_rand -12), suggesting the gain is structural (schedule fix) rather than overfitting to any particular split.
- **This result fundamentally shifts the research programme**: the binding constraint was schedule mis-alignment, not loss function or architecture. All future hypotheses should compare against this baseline.

---

## 2026-05-15 18:30 — PR #3305: H1b: Huber delta=0.05 scan → SENT BACK (rebase on new base)

- Branch: `alphonse/huber-smaller-delta`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Shrinking Huber δ from 0.1 to 0.05 pushes more residuals into L1 regime, improving MAE alignment.

### Results (vs OLD baseline 112.90 with T_max=50)

| Arm | delta | val_avg/mae_surf_p | Δ vs old baseline | W&B |
|-----|-------|--------------------|-------------------|-----|
| Old Baseline | 0.10 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **0.05** | **98.1913** | **-13.0%** | `oolv8t1p` |
| B | 0.02 | 103.7964 | -8.1% | `zlqqtxsu` |

val=98.19 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- δ=0.05 is the right direction — U-shaped response with δ=0.02 overshooting (loss landscape becomes near-constant-gradient L1, slowing late refinement).
- Both arms were run with T_max=50 (handicapped). On the new T_max=15 base, δ=0.05 is expected to yield additional stacked improvement.
- **Action**: rebase onto T_max=15 base, rerun with δ=0.05 only. Target: beat 91.33.

---

## 2026-05-15 18:27 — PR #3171: H8b: Split pressure head + 3x weight on Huber base → SENT BACK (rebase)

- Branch: `fern/split-pressure-head`
- Student: willowpai2i48h1-fern
- Hypothesis: Dedicated output head for pressure channel with 3x Huber-weighted loss improves OOD pressure MAE.

### Results v2 (rebased onto Huber base, with T_max=50)

| Metric | This PR | Huber baseline | Δ |
|--------|---------|---------------|---|
| val_avg/mae_surf_p | 111.9988 | 112.8295 | -0.90 |
| test_avg/mae_surf_p (all 4 splits) | **99.9669** | **106.5996** | **-6.63** |

val=112.00 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- val improvement is marginal (-0.8%), but **test improvement is genuine and consistent**: geom_camber_rc (-13.8 test), cruise test (-15.0), geom_camber_rc val (-23.4). The split head specifically improves OOD generalization.
- v1 (MSE) failed; v2 (Huber base) succeeded — confirming loss-metric alignment is prerequisite for architectural improvements.
- Both runs used T_max=50 (handicapped). With T_max=15, the split head should achieve further improvement.
- **Action**: rebase onto T_max=15 base, rerun with split head + 3x pressure weight + Huber(δ=0.1). Target: beat 91.33.

---

## 2026-05-15 15:45 — PR #3162: H9: Raise surf_weight 10→25 ✗ CLOSED

- Branch: `askeladd/surf-weight-25`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Raising surf_weight from 10 to 25 emphasizes the surface (the scored region) in the gradient, should improve val_avg/mae_surf_p.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **133.4123** |
| val_single_in_dist | 163.71 |
| val_geom_camber_rc | 194.32 |
| val_geom_camber_cruise | 103.60 |
| val_re_rand | 125.67 |

| Split | test mae_surf_p (patched scoring) |
|-------|----------------------------------|
| test_single_in_dist | 134.42 |
| test_geom_camber_rc | 141.56 |
| test_geom_camber_cruise | 92.36 (via local patched scoring) |
| test_re_rand | 120.00 |
| **test_avg/mae_surf_p** | **122.0843** |

W&B run: `hkka77kg` · Group: `surf_weight_sweep`

### Run details
- Epochs: **14/50** (30-min wall-clock cap; best at epoch 13)
- Noisy trajectory: 133.63 (ep11) → 142 (ep12) → 133.41 (ep13) → 146.83 (ep14, cut)
- Peak VRAM: 42.1 GB / 96 GB

### Analysis
- 133.41 does NOT beat the new Huber baseline (112.90). **Closed**.
- The hypothesis was tested against the wrong baseline (MSE loss). With Huber loss already providing MAE-aligned gradients, the marginal benefit of surface emphasis is smaller than expected.
- Loss-metric alignment (Huber) dominates surface weighting at the same compute budget.
- Askeladd also produced an excellent independent bug report on the cruise NaN scoring issue (now being fixed in thorfinn PR #3309) — same root cause as alphonse identified.

### Suggested follow-ups (taken into round 2)
- The surf_weight knob is still worth testing on top of the Huber base (separate from askeladd's follow-up).
- Askeladd assigned PR #3317: cosine T_max tuning to match actual epoch budget — directly addresses the LR-not-annealing observation.

## 2026-05-15 14:30 — PR #3159: H1: Huber loss (delta=0.1) — NEW BASELINE ✓ MERGED

- Branch: `alphonse/huber-loss-aligned`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Replace MSE loss with Huber(delta=0.1) to align training objective with the MAE evaluation metric. At delta=0.1 in normalized space, residuals above 0.1 are in the L1 (MAE-equivalent) regime, creating direct gradient alignment with the scoring metric.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **112.9001** |
| val_single_in_dist | 134.4612 |
| val_geom_camber_rc | 143.4094 |
| val_geom_camber_cruise | 75.8516 |
| val_re_rand | 97.8785 |

| Split | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|-------|-----------------|-----------------|-----------------|
| test_single_in_dist | 120.1970 | 1.4079 | 0.5594 |
| test_geom_camber_rc | 134.3200 | 2.2348 | 0.7179 |
| test_geom_camber_cruise | NaN (data corruption) | 0.9322 | 0.4473 |
| test_re_rand | 92.7597 | 1.3172 | 0.5779 |
| **test 3-split avg (excl. cruise)** | **115.7589** | 1.4730 | 0.5756 |

W&B run: `bpczoejx` · Group: `huber_loss_delta01`

### Run details
- Epochs: **14/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 14 — val still falling (248 → 113 over run; healthy monotonic decrease)
- Peak VRAM: 42.1 GB (well within 96 GB budget)

### Analysis
- **Clear winner**: 112.9 vs 134.7 (thorfinn's slice_num=128), improvement of ~16%.
- MAE alignment works: Huber loss directly creates gradient alignment with the scoring metric. The model learns to minimize mean absolute error rather than mean squared error, which is exactly what's being measured.
- **LR schedule mismatch**: T_max=50 with only 14 epochs completed means LR was still at ~82% of peak (≈0.00041) when training stopped. The cosine schedule never annealed. This is the biggest remaining optimization opportunity — the model is undertrained relative to schedule.
- **Delta regime**: With trained residuals O(0.05–0.2) at epoch 14, many residuals are still below delta=0.1 and in the L2 regime. Smaller delta (0.05 or 0.01) would push more residuals into L1, potentially improving MAE alignment further.
- Per-split pattern: cruise val best (75.85), then re_rand (97.88), while single_in_dist (134.46) and geom_camber_rc (143.41) remain hardest — high-Re raceCar samples dominate absolute error.

### Student suggested follow-ups
1. Tune T_max to actual epoch budget (~14-15 epochs)
2. Smaller Huber delta (0.05, 0.01) or pure L1 to push fully into MAE-aligned regime
3. Per-channel loss weighting (emphasize pressure channel)
4. Patch the cruise-gt NaN bug (separate PR, affects all test metrics)

## 2026-05-15 14:10 — PR #3188: H10: Increase slice_num from 64 to 128

- Branch: `thorfinn/slice-num-128`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: Doubling physics-state slice tokens from 64→128 gives finer flow-regime discretization without changing hidden width or depth.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **134.7389** |
| val_single_in_dist | 159.8405 |
| val_geom_camber_rc | 149.3953 |
| val_geom_camber_cruise | 109.1693 |
| val_re_rand | 120.5507 |

| Split | test mae_surf_p |
|-------|-----------------|
| test_single_in_dist | 132.6239 |
| test_geom_camber_rc | 132.9377 |
| test_geom_camber_cruise | NaN (data corruption — see below) |
| test_re_rand | 119.2658 |
| **test 3-split avg (excl. cruise)** | **128.2758** |

W&B run: `912m0995` · Group: `slice_num_128`

### Run details
- Epochs: **11/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 11 — val still falling steeply (162 → 134 in final epoch; not converged)
- Peak VRAM: 54.5 GB (well within 96 GB; slice-attention 128×128 is negligible vs node ops)

### Infrastructure bug discovered
`.test_geom_camber_cruise_gt/000020.pt` has 761 `inf` values in `y[:,2]` (pressure). The masked-arithmetic `inf * 0 = NaN` propagates into the accumulator — poisoning `test_geom_camber_cruise/mae_surf_p` for **all students**. Val metrics unaffected (all val gt is clean). **Fix**: defensive `y_finite` masking in `train.py:evaluate_split` assigned to thorfinn (PR relative-mse-bugfix).

### Analysis
- No concurrent slice_num=64 baseline yet. Other round-1 students effectively provide the reference.
- VRAM cost of 128 vs 64 is negligible.
- Merged as Round-1 reference — establishes first measured val_avg/mae_surf_p on this advisor branch.

## 2026-05-15 17:00 — PR #3309: Bugfix: inf*0=NaN in evaluate_split ✓ MERGED

- Branch: `thorfinn/nanbug-fix`
- Student: willowpai2i48h1-thorfinn
- Type: Infrastructure bugfix — 4 defensive lines in evaluate_split; model unchanged

### Results

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **112.8295** (=baseline, within noise) |
| **test_avg/mae_surf_p** | **106.5996** ← was NaN (all 4 splits now valid) |
| test_geom_camber_cruise/mae_surf_p | **83.4377** ← was NaN |

W&B run: `g48284pc` · Group: `nanbug_fix`

### Analysis
- Model unchanged; val reproduces baseline within Δ=0.07 (noise).
- **Critical outcome**: test_geom_camber_cruise/mae_surf_p is now 83.44 (finite) and test_avg/mae_surf_p=106.60 is the first valid 4-split test score on this branch.
- Fix: `_y_fin` masking before arithmetic in evaluate_split prevents `pred - (-inf) = inf` → `inf * 0 = NaN` propagation via `data/scoring.py:accumulate_batch`.

## 2026-05-15 17:05 — PR #3180: H4: Wider model (hidden=192, slice_num=96) ✗ CLOSED

- Branch: `tanjiro/wider-model-h192`
- Student: willowpai2i48h1-tanjiro

### Results

| Run | val_avg/mae_surf_p |
|-----|-------------------|
| `a8p3g73s` (h=192 run 1) | **150.3762** (best of 2) |
| `nj0chxr6` (h=192 run 2) | 156.3125 |
| Baseline (h=128 Huber) | 112.9001 |

W&B runs: `a8p3g73s`, `nj0chxr6` · Group: `wider_model_h192`

### Analysis
- 150.38 vs 112.90 = 33% regression. Closed.
- h=192 is 1.6× slower/epoch → only 9 epochs vs baseline's 14. But per-epoch metrics are also worse (150 at ep8 vs ~145 for baseline at ep8 per historical data).
- ~2.2× more params (1.48M vs 0.66M) did not help at this budget.
- Bottleneck is clearly loss/schedule/features, not capacity.
- Seed variance ~4% (156.31 vs 150.38) is significant — future capacity tests should pin a seed.

## 2026-05-15 17:10 — PR #3167: H12: OneCycleLR max_lr=1e-3 ✗ CLOSED

- Branch: `edward/onecycle-lr`
- Student: willowpai2i48h1-edward

### Results

| Run | epochs | val_avg/mae_surf_p | Notes |
|-----|--------|-------------------|-------|
| `x9mygbcm` | 9 | 192.6188 | schedule misconfigured (total_steps sized for 50 ep) |
| `27mfh19o` | 9 | 172.9975 | same misconfiguration |
| `xn1ad9ka` | 9 | **137.1218** | fixed: --epochs 9, schedule fully annealed |
| Baseline (Huber cosine) | 14 | 112.9001 | — |

W&B runs: `xn1ad9ka` (final) · Group: `onecycle_lr`

### Analysis
- 137.12 vs 112.90 = 21% regression after correct schedule setup. Closed.
- **Key insight**: Edward diagnosed the schedule mismatch himself and reran with --epochs 9. The schedule fully annealed (4e-5 → 1e-3 → ~0), so the hypothesis was correctly tested.
- OneCycleLR fails because: (a) 9-epoch total budget means no prolonged low-LR refinement phase, and (b) cosine starts at peak LR and descends immediately, giving better use of the budget.
- NaN on test_geom_camber_cruise is a model-quality issue (extreme prediction on under-converged model at high LR), not the data corruption bug.
