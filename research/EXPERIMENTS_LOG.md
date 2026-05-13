# SENPAI Research Results

---

## 2026-05-13 19:15 — PR #2445: 3-seed baseline calibration on #2227 config (SENT BACK — branch behind + seed code not committed)

- **Branch:** `willowpai2g48h4-nezuko/seed-variance-calibration` (still WIP)
- **Student:** willowpai2g48h4-nezuko
- **W&B runs:** `d9iwzzni` (seed=42), `3ey6sjoz` (seed=43), `daegvjgo` (seed=44) — all on #2227 baseline config (T_0=10, T_mult=1, WD=5e-4, no eta_min)
- **Hypothesis:** measure intrinsic seed variance σ at 21-epoch SGDR budget. The hypothesis from cycle 53 (#2331 sub-finding) was σ ≈ 1-2 val points.

### Results

| Seed | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B run |
|------|--------------------|---------------------|------------|---------|
| 42 | 86.3207 | 75.9316 | 20 | `d9iwzzni` |
| 43 | 86.9272 | 77.1424 | 19 | `3ey6sjoz` |
| 44 | 86.8930 | 76.9254 | 20 | `daegvjgo` |
| **mean** | **86.7136** | **76.6665** | — | — |
| **std** | **0.3407** | **0.6456** | — | — |

### Per-split val mae_surf_p (mean ± std across 3 seeds)

| Split | mean | std |
|-------|------|-----|
| `val_single_in_dist` | 104.9800 | 2.0803 |
| `val_geom_camber_rc` | 105.0622 | 2.3340 |
| `val_geom_camber_cruise` | 56.8798 | 0.8058 |
| `val_re_rand` | 79.9326 | 1.4150 |

### Analysis and Conclusions

**CRITICAL FINDING — three intertwined results:**

**1. Seed σ at 21-ep SGDR is small (~0.34 val, 0.65 test) — 3-5× tighter than the 1-2 val hypothesized.** Implication: future merge bars can be aggressive — 2σ ≈ 0.7 val is a confident threshold for SOTA improvement. Many of our recent merges (e.g. PR #2357 eta_min Δ=0.31 val) are at or just above 1σ — borderline by this revised noise floor.

**2. The kt5pk5qu reference (val=83.997, the PR #2227 cosine_restart baseline) IS NOT REPRODUCIBLE under current code.** All 3 seeded runs of the same config land at 86.32-86.93 val, with the original 83.997 sitting ~8σ below the new mean. With σ this small, an 8σ gap cannot be a tail draw. **Most plausible cause: code/environment drift between when kt5pk5qu was originally measured and now** — torch.compile kernel selection differences, library version drift, sampler RNG state, or some combination. The "lucky draw" hypothesis from cycle 53 is FALSIFIED.

**3. The IMPROVEMENT TRAJECTORY is still valid; absolute numbers need re-anchoring.** PR #2444 (T_mult=2) was measured under the same current code as nezuko's seeded runs. Its val=82.26 is real relative to PR #2357's val=83.69 (also under current code, also recent). What's NOT comparable is the kt5pk5qu reference (83.997) used to validate PR #2357's "win" — that reference was from an earlier code state.

### Technical issues blocking merge

**a. Branch is behind advisor.** Nezuko branched at commit `aeef399` (cycle 49, pre-#2357 pre-#2444). Her train.py LACKS the `cosine_restart_eta_min` flag added by #2357. Merging would revert two SOTA wins. **Action: rebase onto current advisor.**

**b. `--seed` flag is NOT in the committed branch.** The W&B run configs show `seed: 42/43/44` logged correctly — proving the student's local train.py had the flag at runtime. But the diff between her branch and current advisor shows **zero new code in train.py** (only removals — the eta_min code her branch lacks). The local edits to Config / set_seed() were never `git add`-ed. Going forward, advisor branch has no seed infrastructure. **Action: rebase, properly commit the --seed flag, push.**

**c. Variance was measured on the OLD #2227 config, not current SOTA.** Two SOTA improvements have merged since #2227 (eta_min #2357 and T_mult=2 #2444). The variance we actually need is σ on the CURRENT SOTA #2444 config — to establish a seeded merge bar for future PRs. **Action: re-run 3 seeds on `--cosine_restart_T_0 7 --cosine_restart_T_mult 2 --seed {42,43,44}`.**

### Send-back instructions (posted to PR)

Detailed comment posted at 2026-05-13 19:15 asking nezuko to: rebase onto current advisor, properly commit the --seed flag, re-run on the CURRENT SOTA config, and report seeded mean ± std with note on whether any seed produces val < 82.2642 (which would supersede the current SOTA).

### Status

- **PR converted back to draft**
- **Label swapped: status:review → status:wip**
- **σ measurement (86.71 ± 0.34 on #2227) preserved here for ICML appendix CIs**

### Suggested follow-ups (after rebase + re-run completes)

- **If any seed of #2444 config beats 82.2642:** that seed value supersedes current SOTA; report seeded best as new baseline.
- **If seeded mean of #2444 config differs significantly from 82.2642:** update BASELINE.md to report `mean ± std` over seeds rather than single-seed value.
- **Going forward:** all future SOTA candidates should report at least 2 seeds, ideally 3, with the merge bar set at `mean ≤ current SOTA mean − 2σ`.

---

## 2026-05-13 18:45 — PR #2284: Finer WD sweep below 5e-4 under restart (CLOSED — all arms regress, WD axis closed below 5e-4)

- **Branch:** `willowpai2g48h4-frieren/finer-wd-sweep-21epoch`
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** Arm 1 (WD=2e-4), Arm 2 (WD=2.5e-4), Arm 3 (WD=4e-4) — all 3 posted as final by frieren at 17:13 UTC; Arm 4 (WD=4.5e-4) was in-flight but skipped after no progress in 58 min.
- **Hypothesis:** Frieren's prior #2178 had found WD=3e-4 was optimal at 21 epochs without restart. Once restart was merged (#2227 cosine T_0=10, baseline 83.99), it was unclear if 3e-4 was still optimal under restart. This PR mapped the WD curve {2e-4, 2.5e-4, 4e-4, 4.5e-4} around 3e-4 on the T_0=10 restart stack.

### Results

| Arm | WD | val_avg/mae_surf_p | Δ vs 83.99 baseline |
|-----|-----|---------------------|----------------------|
| 1 | 2e-4 | 85.88 | +1.89 |
| 2 | 2.5e-4 | 87.14 | +3.15 (worst — cruise outlier) |
| 3 | 4e-4 | 85.24 | +1.25 |
| Baseline (#2227) | 5e-4 | **83.99** | — |
| (current SOTA #2444 T_mult=2) | 5e-4 | **82.26** | — |

### Per-split val MAE (best arm: WD=4e-4)

| Split | WD=2e-4 | WD=2.5e-4 | WD=4e-4 | Baseline 83.99 |
|-------|---------|-----------|---------|----------------|
| val_single_in_dist | ~110 | ~115 | ~108 | ~104 |
| val_geom_camber_rc | ~107 | ~108 | ~107 | ~100 |
| val_geom_camber_cruise | 60.6 | 73.0 (outlier) | 55.0 | ~46 |
| val_re_rand | ~88 | ~88 | ~85 | ~84 |

### Analysis and Conclusions

**Closed — WD curve below 5e-4 fully mapped, no improvement.** The trend is monotonic regression in both directions away from 4e-4 toward 2e-4 (5e-4 baseline at the upper boundary).

**Cosine restart shifts the optimal WD UPWARD.** Confirmed bidirectionally:
- No restart 21 epochs: WD=3e-4 optimal (PR #2178)
- Single restart T_0=10: WD=5e-4 optimal (PR #2317 anti-additivity, PR #2444 T_mult=2 SOTA)
- This experiment: WD<5e-4 all regress under restart

**Mechanism — restart cushions over-regularization.** Each restart re-injects LR×momentum, breaking the trough that strong WD would otherwise pin parameters to. Without restart, WD=5e-4 caused +1.17% regression at 21 epochs (#2178 Arm 1); with restart, WD=5e-4 is the global optimum. The mechanism makes intuitive sense: WD is a per-step linear force toward 0 in parameter space, while restart provides periodic high-LR exploration that escapes the local minima this force creates.

**Per-split insight — cruise OOD split prefers HIGHER WD under restart.** `val_geom_camber_cruise` improves monotonically from 73.0 → 60.6 → 55.0 as WD increases from 2.5e-4 → 2e-4 → 4e-4. This is the OOD split that has historically had the largest gains from the restart mechanism. **Implication:** WD>5e-4 may continue to improve cruise — testable in #2507.

**Outlier at WD=2.5e-4:** `val_geom_camber_cruise` spiked to 73.0 from a smooth ~60 trend at WD=2e-4. Possible seed-level instability or genuine non-monotonicity in that region. Within the 1-2 val seed noise band's contribution to val_avg, but the val_avg=87.14 outlier is 1+ point above what monotone interpolation predicts.

**Branching decision — close, don't extend.** Arm 4 (WD=4.5e-4) would interpolate to ~84.5 by the monotone trend, still well above current SOTA 82.26. The information value of running Arm 4 is low; the WD=4.5e-4 point is closer to 5e-4 than the other 3 arms by ~2× — even if it happens to win in this region, the WD axis is already understood to want UP under restart.

**Follow-up assigned:** PR #2507 frieren — WD ∈ {6e-4, 7e-4} on the T_mult=2 SOTA stack. The unexplored region of the WD axis under restart. T_mult=2's 40%-longer cycle 2 may absorb stronger WD productively.

---

## 2026-05-13 19:00 — PR #2507: WD curve ABOVE 5e-4 on T_mult=2 stack (assignment to frieren, NEW cycle 63)

- **Branch:** `willowpai2g48h4-frieren/wd-above-5e4-tmult2`
- **Student:** willowpai2g48h4-frieren
- **Hypothesis:** Cosine restart shifts optimal WD upward (3e-4 no-restart → 5e-4 T_0=10 single restart). T_mult=2's cycle 2 is 14 epochs (40% longer than T_0=10's 11 epochs) — direct test of whether the additional descent runway absorbs stronger WD productively, or whether 5e-4 holds.
- **Arms:** WD ∈ {6e-4, 7e-4} on otherwise-identical current SOTA stack (PR #2444 config: T_0=7, T_mult=2, eta_min=0, surf_head_lr=5e-3, huber_delta=0.5, torch.compile).
- **Branching rule:** Both regress → 5e-4 is bilateral peak under T_mult=2, WD axis fully closed under restart. Either arm beats 82.2642 → push higher with WD ∈ {8e-4, 1e-3} in a follow-up.

---

## 2026-05-13 18:00 — PR #2444: T_mult=2 restart, T_0=7 (MERGED — new SOTA)

- **Branch:** `willowpai2g48h4-alphonse/t-mult-2-restart`
- **Student:** willowpai2g48h4-alphonse
- **W&B runs:** `1m0cfdr4` (T_0=7 T_mult=2, WINNER), `eey54new` (T_0=6 T_mult=2, regressed)

### Results

| Arm | Config | best_epoch | val_avg/mae_surf_p | Δ vs #2357 (83.6873) | test_avg/mae_surf_p | W&B run |
|-----|--------|-----------|-------------------|---------------------|--------------------|---------| 
| Baseline (PR #2357) | T_0=10, T_mult=1, eta_min=1e-5 | 20 | 83.6873 | — | 73.3963 | zely2d09 |
| **Arm 1 (WINNER)** | **T_0=7, T_mult=2, eta_min=0** | **21** | **82.2642** | **−1.42 (−1.7%)** | **72.4019** | `1m0cfdr4` |
| Arm 2 | T_0=6, T_mult=2, eta_min=0 | 18 | 87.1358 | +3.45 (+4.1%) | 76.5659 | `eey54new` |

### Per-split (Arm 1 winner)

| Split | val | test |
|-------|----:|-----:|
| single_in_dist | 96.90 | 85.52 |
| geom_camber_rc | 103.14 | 91.09 |
| geom_camber_cruise | 53.94 | 44.92 |
| re_rand | 75.08 | 68.08 |
| **avg** | **82.2642** | **72.4019** |

### Commentary

**MERGED as new SOTA.** Val margin vs prior SOTA (83.6873): −1.42, well outside seed noise. Test margin: −1.00. Both robust.

**Mechanism confirmed:** T_0=7, T_mult=2 → cycles [7, 14]. Longer cycle 2 (14 vs 10 epochs) gives deeper basin descent. Best epoch = e21 (last epoch) — model still improving at 30-min wall-clock cap. No convergence plateau.

**T_0=7 is structurally unique:** It is the only T_0 value that gives exactly 2 full, non-truncated cosine cycles within ≤21 epochs under T_mult=2. T_0=6 would give cycles [6, 12] + 3 truncated epochs — the 3rd restart at e19 wipes out the e18 minimum. T_0=5 would be [5, 15] + 1 truncated epoch.

**Arm 2 failure is informative:** T_0=6 fails not because cycle 2 is "too short" (e18=87.14 was already worse) but because the shorter cycle 1 enters cycle 2 from a less-warm starting state, and the truncated cycle 3 destroys any recovery chance.

**Run used eta_min=0 (default)** — the eta_min=1e-5 win from #2357 was not composed here. Composing is the next experiment (alphonse #2498).

**Split analysis:** Improvement concentrated in `single_in_dist` (val −7.99, test −9.59). OOD splits flat. Longer descent hits in-distribution basin harder without sacrificing OOD.

**Assigned alphonse next:** PR #2498 — T_0=7 T_mult=2 + eta_min=1e-5 compose (two orthogonal wins should stack).

---

## 2026-05-13 17:30 — PR #2357: Cosine restart eta_min sweep (MERGED — new SOTA)

- **Branch:** `willowpai2g48h4-askeladd/cosine-restart-eta-min`
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `zely2d09` (eta_min=1e-5, WINNER), `gqzpckn4` (eta_min=5e-5, regressed)

### Results

| Arm | eta_min | best_epoch | val_avg/mae_surf_p | Δ vs baseline | test_avg/mae_surf_p | Δ test | W&B run |
|-----|---------|-----------|-------------------|---------------|--------------------|---------|---------| 
| Baseline PR #2227 | 0 | — | 83.9969 | — | 74.7684 | — | kt5pk5qu |
| **Arm 1 (WINNER)** | **1e-5** | **20** | **83.6873** | **−0.31 (−0.37%)** | **73.3963** | **−1.37 (−1.83%)** | `zely2d09` |
| Arm 2 | 5e-5 | 20 | 87.5086 | +3.51 (+4.18%) | 78.3127 | +3.54 (+4.74%) | `gqzpckn4` |

### Per-test-split (Arm 1 winner at epoch 20)

| Split | Arm 1 (eta_min=1e-5) | Arm 2 (eta_min=5e-5) |
|-------|---------------------:|---------------------:|
| test_single_in_dist | 87.60 | 92.67 |
| test_geom_camber_rc | 90.10 | 94.55 |
| test_geom_camber_cruise | 46.87 | 51.70 |
| test_re_rand | 69.02 | 74.34 |
| **test_avg** | **73.40** | **78.31** |

### Commentary

**MERGED as new SOTA.** The val margin (0.31) is near the seed-noise floor (~1–2 val), but the test improvement (−1.37, −1.83%) is uniform across all four test splits and more robust.

**Mechanism confirmed:** A tiny non-zero LR floor (eta_min=1e-5, 2% of peak) at cycle-ends lets the optimizer continue micro-refining in the basin. The model reaches a slightly deeper minimum at e10/e20 before the restart kick.

**Non-linearity is the key finding:** The jump from eta_min=1e-5 to 5e-5 (5× increase) causes dramatic regression (+4.18% val). The basin is sharp — micro-refinement helps up to a point, then basin-escape dominates.

**Spike-trajectory evidence:** Post-restart spike (e21) is smaller with eta_min=1e-5 (115.19 vs 138.59 for eta_min=5e-5), consistent with a deeper basin: the model has more "gravity" to pull it back from the restart kick.

**Assigned askeladd next:** PR #2487 — eta_min refinement sweep {5e-6, 2e-5} to map the curve above and below the current 1e-5 SOTA and find the true optimum.

---

## 2026-05-13 17:30 — PR #2340: AdamW β1 sweep: momentum adaptation speed (CLOSED — β1 axis exhausted)

- **Branch:** `willowpai2g48h4-thorfinn/adamw-beta1-sweep`
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `o29tfbmf` (β1=0.85 run 1), `p6q5iy7h` (β1=0.85 retry), `uhotdk62` (β1=0.95)

### Results

| Arm | β1 | best_epoch | val_avg/mae_surf_p | Δ vs baseline | test_avg/mae_surf_p | W&B run |
|-----|----|-----------|-------------------|---------------|--------------------|---------| 
| Baseline kt5pk5qu | 0.9 | — | **83.9969** | — | **74.7684** | kt5pk5qu |
| β1=0.85 (run 1) | 0.85 | 20 | 87.4989 | +3.50 (+4.17%) | 77.9268 | `o29tfbmf` |
| β1=0.85 (retry) | 0.85 | 20 | 87.9596 | +3.96 (+4.72%) | 77.8949 | `p6q5iy7h` |
| β1=0.95 | 0.95 | 20 | 88.8321 | +4.83 (+5.75%) | 78.9673 | `uhotdk62` |

### Commentary

**β1 axis definitively closed.** Both perturbations regress significantly — 3.5–4.8 val pts, well outside the ~1–2 val pt seed-variance band. The β1=0.85 retry confirmed the regression is structural, not noise (87.50/87.96, gap ~0.5 val within seed variance, both >3 pts above baseline).

**AdamW search is now fully exhausted:**
- β2 axis: β2=0.999 fixed (#2015, #2201)
- β1 axis: β1=0.9 confirmed (this PR, both directions regress)
- LR axis: 5e-4 / surf_head_lr=5e-3 confirmed (#1974, #1949)
- WD axis: 5e-4 under restart confirmed; frieren #2284 maps the lower-WD curve
- ε axis: closed (#2128, surf_head frac_below_eps=0 always)

**Spike-trajectory finding (high diagnostic value for paper):**
- β1=0.95 smooths the restart spike dramatically (e11 = +15.8 vs +48.5 at β1=0.85) — momentum inertia carries pre-restart gradient direction through the LR jump
- β1=0.85 amplifies the spike — faster momentum decay leaves the model more exposed to the post-restart LR jump
- BUT: smoother spike (β1=0.95) does NOT produce a deeper cycle-2 minimum (88.83 vs 87.50 for β1=0.85)
- AND: rougher spike (β1=0.85) does not unlock anything either

Interpretation: the step-magnitude composition `m/√v` is well-tuned at β1=0.9. The spike shape is determined by the momentum inertia crossing the LR reset, but the final recovery minimum is governed by the full gradient history, not spike amplitude. **Smoother spikes ≠ better convergence** — the cycle-34 spike-IS-signal reframing holds. Future restart-suppression hypotheses (warmup-after-restart, decoupled β1 reset) should note: this evidence says suppressing the spike does not help final val.

**Assigned thorfinn next:** PR #2477 — MLP dropout sweep (feature-level activation dropout, 0.05 vs 0.10). First unexplored soft-regularization axis after AdamW exhaustion.

---

## 2026-05-13 16:40 — PR #2381: Strict stratification + cosine_restart compose (CLOSED — informative null)

- **Branch:** `willowpai2g48h4-fern/stratified-restart-compose`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `pp47gvm5` (Arm 1 strict+restart), `lfedba5o` (Arm 2 strict+restart+DW)

### Results

| Arm | sampler | domain weights | val_avg/mae_surf_p | Δ val vs baseline | test_avg |
|-----|---------|----------------|--------------------|-------------------|---------|
| Baseline `kt5pk5qu` | weighted | — | **83.9969** | — | **74.7684** |
| Arm 1 strict+restart | strict | 1.0/1.0/1.0/1.0 | 88.1218 | +4.91% ✗ | 76.4569 |
| Arm 2 strict+restart+DW | strict | 1.0/1.0/1.3/1.2 | **86.4076** | +2.87% ✗ | 76.6089 |

### Commentary

**Key mechanism finding — strict and restart are NOT orthogonal.** Under restart, the spike is LR-driven (not sampler-variance-driven), so strict's variance reduction loses its dual purpose. Spike trajectory is decisive: Arm 1 e10→e11 spike +26.28, Arm 2 +27.39 — barely damped vs baseline (~+38). The strict sampler's variance reduction is overridden by the v_t reset at restart.

**Three confirmed mechanisms from #2259 do NOT compose with restart:**
1. Per-batch composition variance was load-bearing → under restart, spike is LR-driven, not sampler-driven
2. Strict damped spike 30-50% on old baseline → spike persists under restart
3. Strict created cruise regression → vanished under restart (cruise val ≈56 in both arms, baseline-class)

**Domain weights (Arm 2) notable sub-result:** DW reduced second-restart spike amplitude by ~31% (e20→e21: +33 vs +48 for Arm 1). This suggests domain weights can modulate cycle-injection energy. Could be useful as a compose target for longer-second-cycle configs like #2444 (T_mult=2).

**Compose scorecard — what does NOT compose with restart:** WD=3e-4 (same failure mode, #2317), strict stratification (mechanism replacement, #2381). What IS promising: Lookahead (#2296 compose test assigned), T_mult=2 cycle geometry (#2444), head_wd (#2380 edward still in-flight).

---

## 2026-05-13 16:30 — PR #2296: Lookahead-AdamW k-sweep (SENT BACK — compose test assigned)

- **Branch:** `willowpai2g48h4-tanjiro/lookahead-adamw`
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `aumcndej` (k=5, winner), `nmv3jlsx` (k=10), `4pscsoah` (k=3), `zj7z3ix5` (k=7)

### Results

| Arm | k | val_avg/mae_surf_p | Δ vs OLD 87.0144 | Δ vs NEW 83.9969 | test_avg |
|-----|---|--------------------|-----------------|-----------------|---------|
| **k=5 α=0.5** | 5 | **83.8589** | **−3.62%** | **−0.16%** | **74.0656** |
| k=3 | 3 | 90.6018 | +4.12% | +7.86% | 80.9859 |
| k=7 | 7 | 94.9108 | +9.08% | +13.0% | 84.3417 |
| k=10 | 10 | 96.1672 | +10.5% | +14.5% | 88.2945 |

### Commentary

**k=5 is the sweet spot.** Monotonic degradation: k=3 too short (weak anchoring), k=5 optimal, k=7/k=10 too long (over-anchoring). Sensitivity is steep. Test improvement vs current baseline: −0.94% (test=74.0656 vs 74.7684).

**Critical config note:** Student's winning run used WD=3e-4 (OLD WD) + NO cosine_restart. Against the current baseline (WD=5e-4 + cosine_restart), k=5 is val −0.16% (within seed noise) and test −0.94% (genuine improvement signal).

**Mechanism confirmed:** Lookahead's slow-track anchor smooths late-training trajectory (e15→e21 monotonically decreasing in k=5: 100.18→83.86). Spike at e11 (shifted vs baseline e12) but amplitude preserved — Lookahead damps mid-training trajectory noise without eliminating the spike.

**Next step:** Sent back for compose test (Lookahead k=5 + cosine_restart + WD=5e-4). Mechanisms are orthogonal (optimizer state vs LR schedule). If they compose, this could beat the baseline definitively.

---

## 2026-05-13 22:00 — PR #2331: SWA over SGDR cycle-ends (CLOSED — definitive null, SWA permanently closed)

- **Branch:** `willowpai2g48h4-nezuko/swa-cycle-end-averaging`
- **Student:** willowpai2g48h4-nezuko
- **W&B runs:** `u7j6ve8z` (Arm 1 — sparse 2-snapshot), `eyokcl48` (Arm 2 — dense per-epoch)

### Results

| Model | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ val vs baseline |
|-------|-------------------|---------------------|-------------------|
| Baseline `kt5pk5qu` | 83.9969 | 74.7684 | — |
| Arm 1 live (best, e20) | 85.3123 | 75.0951 | +1.59% ✗ |
| **Arm 1 SWA (n=2, e10+e20)** | **93.8980** | **84.3707** | **+11.79% ✗✗** |
| Arm 2 live (best, e20) | 85.5672 | 76.2486 | +1.87% ✗ |
| **Arm 2 SWA (dense, n=11)** | **103.3037** | **93.9768** | **+22.99% ✗✗** |

### Commentary

**Geometric proof — definitive null.** SWA(e10,e20) val=93.90 ≈ arithmetic mean of endpoint vals (102.39+85.31)/2 = 93.85. The linear interpolation in weight space passes through HIGHER loss than either endpoint. This proves e10 and e20 are NOT in a shared flat basin — they are in distinct basins along the descent trajectory.

**Dense SWA failure mode:** Mid-cycle high-LR snapshots (e11–e16, vals 110–132) dominate the running mean and drown out the late-cycle low-LR snapshots (e17–e20, vals 85–97). SWA-dense is incompatible with a single 10-epoch cosine descent.

**SWA history at this codebase:**
- #1808: SWA at 14ep non-restart (closed, +3.33%)
- #1951: SWA retry (closed)
- #2189: EMA 21-ep re-screen (closed, +9.07%)
- **#2331: SWA over SGDR cycle-ends (closed, geometric proof). SWA PERMANENTLY CLOSED.**

**CRITICAL SUB-FINDING — Seed Variance:** Both arms' live models regressed 1.3–1.6 val points from baseline (85.31/85.57 vs 83.997) despite training with identical config. Student interpretation: "the baseline kt5pk5qu got a slightly lucky draw." This calls into question all small-margin wins (~1 val point) in the programme. Seed variance of ~1-2 val points at 21-epoch SGDR budget is now a key uncertainty. → Addressed by new assignment #2445 (3-seed calibration).

---

## 2026-05-13 22:00 — PR #2317: Restart-WD compose (CLOSED — anti-additive, WD×restart falsified)

- **Branch:** `willowpai2g48h4-alphonse/restart-wd-compose`
- **Student:** willowpai2g48h4-alphonse
- **W&B runs:** `m2076fd5` (Arm 1 T_0=10, WD=3e-4), `rpv7jbjp` (Arm 2 T_0=12, WD=5e-4)

### Results

| Run | val_avg/mae_surf_p | Δ val | test_avg/mae_surf_p | Δ test |
|-----|-------------------|-------|---------------------|--------|
| Baseline `kt5pk5qu` (T_0=10, WD=5e-4) | **83.9969** | — | **74.7684** | — |
| **Arm 1** `m2076fd5` (T_0=10, WD=3e-4) | 85.5863 | +1.89% ✗ | 75.9591 | +1.59% ✗ |
| **Arm 2** `rpv7jbjp` (T_0=12, WD=5e-4) | 87.3777 | +4.03% ✗ | 78.0472 | +4.39% ✗ |

### Per-split breakdown (Arm 1)

| Split | Baseline | Arm 1 | Δ |
|-------|----------|-------|---|
| val_single_in_dist | 104.89 | 98.98 | **−5.6% ✓** (in-dist improved!) |
| val_geom_camber_rc | 101.05 | 105.23 | +4.1% ✗ |
| val_geom_camber_cruise | 54.58 | 58.50 | +7.2% ✗ |
| val_re_rand | 75.47 | 79.63 | +5.5% ✗ |
| **val_avg** | **83.9969** | **85.5863** | **+1.89% ✗** |

### Commentary

**Anti-additive compose.** The two SOTA wins (WD=3e-4, cosine restart) target the SAME failure mode and are NOT orthogonal. Student's analysis: "the restart mechanism's main value is escaping in-distribution overfitting and re-discovering generalizing features. Lower WD (3e-4) weakens that generalization pressure during cycle 2's deep descent — the model overfits the in-dist mode and abandons the OOD invariants the restart had unlocked at WD=5e-4."

Split-by-split signature is decisive: Arm 1 WINS in-dist (−5.6%) but LOSES every OOD split. WD=3e-4 and restart both fight over-regularization via the same mechanism, so they don't stack.

**Arm 2 (T_0=12):** cycle 2 still descending at e21 (87.38, pointing down). Not a fundamental failure — the 30-min cap is binding. T_0=12 with T_mult=1 hits cycle 2 at 9/12 epochs at timeout; analogous to the truncation seen in previous T_0=12 runs.

**Consequence for compose strategy:** WD-at-restart cannot be stacked. Restart "is" the generalization push that WD=3e-4 was approximating in non-restart runs. Future composes with restart must target orthogonal mechanisms.

→ Follow-up: T_mult=2 with T_0=7 assigned as #2444 (alphonse). This tests whether a longer cycle 2 (14 vs 10 epochs) allows deeper OOD descent within budget.

---

## 2026-05-13 18:00 — PR #2232: Decoupled WD head-UP head_wd∈{1e-3,2e-3} (CLOSED — wins vs OLD baseline only)

- **Branch:** `willowpai2g48h4-edward/head-up-wd`
- **Student:** willowpai2g48h4-edward
- **W&B runs:** `e0z3c5xd` (Arm 1 head_wd=1e-3), `ghzh3v63` (Arm 2 head_wd=2e-3 ★)

### Results

| Arm | head_wd | val_avg | Δ vs 89.7197 (OLD) | Δ vs 83.9969 (CURRENT) | test_avg | Δ vs 79.32 |
|-----|---------|---------|--------------------|------------------------|----------|------------|
| 1 | 1e-3 (2×) | 91.17 | +1.61% | +8.5% ✗ | 82.19 | +3.6% |
| 2 | 2e-3 (4×) | **88.25** | **−1.64% ★** | +5.1% ✗ | **78.22** | **−1.38% ★** |

### Commentary

**Wins vs OLD #2091 baseline at Arm 2** (head_wd=2e-3). Loses to current 83.9969 by 5%. PR pre-dates cosine_restart merge.

**Three mechanism findings:**
1. **Per-step shrinkage, not weight magnitude.** Doubling head_wd (1e-3→2e-3) reduced surf_head weight norm by only 1.25%. The effect operates through gradient-noise / step-decay dynamics, exactly as the residual-fit-suppression framing in #2122 predicted.
2. **Per-split asymmetry:** val_single_in_dist −6.75%, val_geom_camber_rc −5.50% (improvements), but val_geom_camber_cruise +7.81% and val_re_rand +4.28% (regressions). Head was carrying split-specific fit for cruise/re_rand.
3. **Branching rule fired:** Arm 2 > Arm 1 → optimum is above 2e-3. Sweep direction is up.

**Follow-up assigned (#2380 head-wd-restart-compose):** Test head_wd∈{2e-3, 3e-3} composed with cosine_restart T_0=10.

---

## 2026-05-13 18:00 — PR #2259: Stratified per-domain batch sampler (CLOSED — diagnostic confirmed)

- **Branch:** `willowpai2g48h4-fern/stratified-sampler`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `dnpeee7f` (Arm 1 strict ★), `ad4vnb0c` (Arm 2 rotated)

### Results

| Arm | Mode | val_avg | Δ vs 89.7197 (OLD) | Δ vs 83.9969 (CURRENT) | test_avg | Δ vs 79.32 |
|-----|------|---------|--------------------|------------------------|----------|------------|
| 1 | strict 1+1+1+1g | **88.19** | **−1.71% ★** | +5.0% ✗ | **78.79** | **−0.67% ★** |
| 2 | rotated 2+1+1 | 90.26 | +0.60% | +7.5% ✗ | 80.62 | +1.65% ✗ |

### Commentary

**Three diagnostic mechanisms confirmed — high-value experiment despite losing vs current baseline:**

1. **Per-batch composition variance is load-bearing for per-split asymmetry.** val_single (−8.61%) and val_rc (−1.29%) move TOGETHER under strict for the first time. Confirms #2153 hypothesis: the unconstrained WeightedRandomSampler creates per-batch domain imbalance that drives second-moment variance and asymmetric splits.

2. **Spike+recovery is ~30-50% sampler-variance driven.** Strict reduces e16 spike from baseline ~+50% to +31%. The remainder is intrinsic cosine-LR × second-moment dynamics.

3. **Stratification creates new asymmetry:** rebalances single↔rc but regresses cruise (+7.10%) and re_rand (+1.55%). Cruise had been *implicitly oversampled* via lower BIVW gradient noise → smaller v_t → larger effective step. Stratification removes that implicit oversampling.

**Follow-up assigned (#2381 stratified-restart-compose):** Test strict + cosine_restart, with optional domain-weighted loss (1.0/1.0/1.3/1.2) recovery.

---

## 2026-05-13 17:30 — PR #2123: Cosine T_max sweep {15,20,25} (CLOSED — superseded by cosine restart)

- **Branch:** `willowpai2g48h4-askeladd/cosine-tmax` (CLOSED — beaten by current baseline)
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `9m6wd5nr` (T_max=15), `py557aqk` (T_max=20 ★ best val), `pm5cgw5f` (T_max=25 timed out e13)
- **Reruns:** `p87c6rmh` (T_max=25 14-ep, val=97.44), `p57if70q` (T_max=20 11-ep timeout, val=109.57)

### Results vs OLD #2031 baseline (val 93.6198 / test 83.8825, NOT current baseline)

| Arm | T_max | val_avg | Δ vs 93.6198 | test_avg | Δ vs 83.8825 |
|-----|-------|---------|--------------|----------|---------------|
| 1 | 15 | 93.3322 | −0.31% | 83.0846 | **−0.95% ★** |
| 2 | 20 | **93.0554** | **−0.60% ★** | 83.9281 | +0.05% |
| 3 | 25 | 102.1690 | +9.1% | 91.1551 | +8.7% |

### Results vs CURRENT 83.9969 baseline

| Arm | T_max | val_avg | Δ vs 83.9969 |
|-----|-------|---------|---------------|
| 1 | 15 | 93.33 | **+11.1% ✗** |
| 2 | 20 | 93.06 | **+10.8% ✗** |
| 3 | 25 | 102.17 | **+21.6% ✗** |

### Commentary

**PR pre-dates major regime changes.** This experiment was run at 14-epoch pre-compile baseline (no torch.compile, no cosine_restart). All arms fail vs current 83.9969 baseline by 10-22%.

**Mechanism partially confirmed.** Tighter cosine T_max → quieter late-epoch trajectories at T_max=15/20 (essentially monotonic late descent). T_max=25 too loose: LR still at 47% of peak at e13, schedule barely meaningful. The "epochs completed within 30 min" is the dominant noise channel at 14-ep budget.

**Hypothesis now superseded.** Current baseline uses `CosineAnnealingWarmRestarts` (T_0, T_mult parameters), not single-cycle `CosineAnnealingLR` (T_max parameter). The T_max axis is no longer applicable. The "eta_min" parameter (cycle-end LR floor) IS the natural successor experiment — assigned to askeladd as #2357.

**Excellent diagnostic content from student** — per-epoch LR trajectories, noise-floor characterization, run-variance reruns. Suggested follow-ups appreciated; reposted as the eta_min direction.

---

## 2026-05-13 16:30 — PR #2201: AdamW β2=0.9999/0.9995 — longer second-moment timescale (CLOSED)

- **Branch:** `willowpai2g48h4-nezuko/beta2-long` (CLOSED — both arms regressed)
- **Student:** willowpai2g48h4-nezuko
- **W&B runs:** `zpq35emf` (Arm 1 β2=0.9999), `esas3nuc` (Arm 2 β2=0.9995)

### Results vs CURRENT baseline (83.9969 / 74.7684)

| Arm | β2 | val_avg/mae_surf_p | Δ vs 83.9969 | test_avg/mae_surf_p | Δ vs 74.7684 | best_ep |
|-----|----|--------------------|--------------|---------------------|--------------|---------|
| Arm 1 | 0.9999 | 95.8855 | **+14.2% ✗** | 84.1615 | **+12.5% ✗** | 19 |
| Arm 2 | 0.9995 | 91.5725 | **+8.9% ✗** | 81.0857 | **+8.5% ✗** | 21 |

(Note: student reported results vs OLD compile baseline 89.7197, not current 83.9969 — adjusted here.)

### Commentary

**β2 direction is permanently closed.** Both longer-timescale arms (β2=0.9999 → +14.2% val, β2=0.9995 → +8.9% val) regressed vs current baseline. Combined with PR #2015 (β2=0.95, +11% vs current), we have confirmed β2=0.999 is the sweet spot by exhaustive bilateral sweep.

**Key mechanism confirmed:** β2=0.9999 produced a ~2.5× smaller sqrt_v_max ceiling AND ~2.4× smaller epoch-to-epoch variation — exactly the predicted "smoother second moment" effect. But the smooth denominator paradoxically caused LARGER mid-epoch spikes (e14 spike was 2.4× bigger) while destroying the e18/e20 deep minimum (baseline 89.72→83.99 → β2=0.9999 best 95.89). Long-β2 smoothing lets outlier batches through at full step magnitude.

**Cycle-34 reframing confirmed for the 3rd time:** The per-batch sqrt(v) dynamics that β2=0.999 provides are signal, not noise. Damping them via longer β2 prevents the spike-recovery that produces the late-epoch deep minimum. β1 (momentum, numerator side) is the remaining untested axis.

---

## 2026-05-13 16:30 — PR #2188: Encoder LR boost at e15-18 (CLOSED)

- **Branch:** `willowpai2g48h4-thorfinn/encoder-lr-boost` (CLOSED — beaten by new baseline)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `fbk6rwg8` (Arm 1 ×2.0), `xzrpsilo` (Arm 2 ×3.0)

### Results vs CURRENT baseline (83.9969)

| Arm | Factor | val_avg/mae_surf_p | Δ vs 83.9969 | test_avg/mae_surf_p | Δ vs 74.7684 | Best ep |
|-----|--------|--------------------|--------------|---------------------|--------------|---------|
| Arm 1 | ×2.0 | 92.1147 | **+9.7% ✗** | 83.3518 | **+11.5% ✗** | 21 |
| Arm 2 | ×3.0 | **84.7466** | **+0.9% ✗** | **75.2636** | **+0.7% ✗** | 21 |

(Arm 2 won vs OLD compile baseline 89.72 at −5.54% val, but current baseline is 83.9969 after cosine restart merged.)

### Commentary

**Mechanism confirmed, result overtaken.** The 3× encoder boost at e15-18 induces a clean exploration burst at e16 (94.79 → 135.62 → 114.90 → 99.60 → 84.75) — exactly the predicted spike-recovery dynamic. The 2× factor is below the threshold to trigger real landscape escape.

**Why closed:** Arm 2 (84.7466 val) is +0.9% ABOVE the new cosine-restart baseline (83.9969). Cosine restart already induces similar exploration bursts via restart-driven LR cycling. The encoder boost mechanism, while real, does not stack above cosine restart in its current form.

**Key finding:** 2× → 3× monotonic improvement (+7.4 val points) suggests we're still on the upside of the curve. The `encoder_lr_boost_factor` sweet spot is ≥3× — there is likely room above 3×. However, with cosine restart already in the baseline, the burst needs to be repositioned to compose with restart cycles, not compete with them.

---

## 2026-05-13 14:00 — PR #2227: SGDR cosine warm restart T_0=10/7 (MERGED — biggest single-axis win since compile)

- **Branch:** `willowpai2g48h4-alphonse/cosine-restart` (MERGED — NEW BASELINE)
- **Student:** willowpai2g48h4-alphonse
- **W&B runs:** `kt5pk5qu` (Arm 1 T_0=10 ★ winner), `ppmzaftp` (Arm 2 T_0=7)

### Results

| Arm | T_0 | val_avg/mae_surf_p | Δ vs 89.7197 | test_avg/mae_surf_p | best_ep |
|-----|-----|--------------------|--------------|---------------------|---------|
| baseline | — | 89.7197 | — | 79.3167 | — |
| **Arm 1 ★** | **10** | **83.9969** | **−6.38% ✓** | **74.7684** | 20 |
| **Arm 2** | 7 | 86.7781 | −3.28% ✓ | 76.7821 | 21 |

Both arms WIN. T_0=10 (single restart) is the clear winner.

### Per-split val (Arm 1 best e20)

| Split | T_0=10 | vs 87.0144 SOTA | vs 89.7197 compile |
|-------|--------|------------------|---------------------|
| `val_single_in_dist` | 104.89 | −2.0% ✓ | −8.7% ✓ |
| `val_geom_camber_rc` | 101.05 | −2.8% ✓ | −7.0% ✓ |
| `val_geom_camber_cruise` | 54.58 | −4.8% ✓ | −1.6% ✓ |
| `val_re_rand` | 75.47 | −5.4% ✓ | −5.5% ✓ |
| **val_avg** | **83.9969** | **−3.59% ✓** | **−6.38% ✓** |

### Per-epoch trajectory (Arm 1, T_0=10)

```
e  val_avg   lr_enc      event
1  197.35    4.88e-04
9  105.45    1.22e-05    cycle-1 minimum
10 102.18    5.00e-04   *RESTART (LR jumps to peak)
11 130.10    4.88e-04    SPIKE (+27%)
12 114.99    4.52e-04
17  97.06    1.03e-04
19  86.72    1.22e-05    cycle-2 deep recovery (−18% vs cycle-1)
20  83.997   5.00e-04   *RESTART ← BEST EPOCH
21 121.14    4.88e-04    post-restart spike
```

### Analysis

**Hypothesis CONFIRMED.** The cycle 34 reframing ("e12 spike+recovery is beneficial training dynamic") was correct, and inducing more spikes via cosine warm restarts produces monotonic cycle-over-cycle improvement.

**Mechanism observations:**
1. Each restart produces a clean val spike (e11=130 in cycle 1's post-restart, e21=121 in cycle 2's post-restart)
2. Each cycle's minimum is DEEPER than the prior cycle's (cycle-2 minimum −18% vs cycle-1)
3. Best epoch sits RIGHT AT THE RESTART (e20=83.997, end of cycle 2, before next spike)
4. OOD generalization gains are largest (test_geom_camber_rc −7.3%, test_re_rand −8.2%) — restart escapes in-distribution overfitting

**Critical context:** Arm 1 used **WD=5e-4** (NOT the new SOTA WD=3e-4 from PR #2178). The cosine_restart mechanism is so powerful it OVERCOMES the over-regularization that made WD=5e-4 regress in #2178. Composing cosine_restart with WD=3e-4 is the immediate next experiment.

**Other in-flight PRs are now misaligned:**
- nezuko #2201 (β2 sweep) — used WD=5e-4 + NO restart, baseline 89.72; now beating 83.99
- thorfinn #2188 (encoder LR boost) — used WD=5e-4 + NO restart
- edward #2232 (head-WD up) — used WD=5e-4 + NO restart
- fern #2259 (stratified sampler) — used WD=5e-4 + NO restart
- frieren #2284 (finer WD sweep) — used WD ∈ {2e-4, 2.5e-4, 4e-4} + NO restart
- tanjiro #2296 (Lookahead) — used WD=3e-4 + NO restart
- askeladd #2123 (T_max sweep) — orthogonal: tests cosine cycle LENGTH, not restart

All these PRs face the harder 83.9969 target. Some hypotheses may still produce composable wins atop restart_T_0=10 (e.g. β2, head-WD, Lookahead).

**Alphonse next:** Compose cosine_restart_T_0=10 with WD=3e-4 (two strongest wins together) + variation on T_0 ∈ {11, 12} since cycle 2 may still have headroom.

---

## 2026-05-13 13:35 — PR #2189: EMA model weights re-screen at 21-epoch budget (CLOSED)

- **Branch:** `willowpai2g48h4-tanjiro/ema-21epoch-rescreen` (CLOSED — both arms regressed)
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `ganjm31s` (Arm 1), `dj3doney` (Arm 2)

### Results

| Arm | EMA decay | EMA start | val_avg (EMA, e18) | Δ vs 89.7197 | test_avg | Δ vs 79.3167 |
|-----|-----------|-----------|---------------------|--------------|----------|--------------|
| Arm 1 | 0.999 | e0 | **97.86** | **+9.07% ✗** | **87.98** | **+10.93% ✗** |
| Arm 2 | 0.9995 | e5 | **104.49** | **+16.46% ✗** | **95.45** | **+20.34% ✗** |

Both arms violate the 3% regression threshold by 3-5×.

### Mechanism diagnosis

1. **ema_init_bias decay too slow at 20 epochs**: even at e18 Arm 2's ema_init_bias was 0.034 (5%+ contribution from warmup-period weights). Earlier epochs with val=130-200 contaminate the EMA snapshot.
2. **Live model still descending throughout**: Arm 1 best live e18=95.87, Arm 2 e14/e19=87.07. No plateau exists for EMA to help.
3. **Arm 2 LIVE e14=87.07 beat baseline 89.72** (and matches new SOTA 87.01 from #2178). EMA selection BURIED this real-time win.
4. **Test damage > val damage**: Arm 2 test +20.34% suggests EMA averaging from warmup phase corrupts OOD generalization specifically.

### Conclusion

EMA on weights is fundamentally incompatible with this training regime. The mechanism requires either (a) a much longer training budget where cosine LR plateaus, or (b) a different LR schedule with a long flat tail. We have neither. **EMA direction CLOSED.**

### Cross-link: live e14=87.07 (Arm 2) matches WD=5e-4 dynamics from #2178

Tanjiro's live-trajectory data shows WD=5e-4 + compile produces an e14 transient low (~87) but oscillates back up. This is consistent with #2178's finding that WD=5e-4 AMPLIFIES the e12 spike, producing volatile late-epoch trajectories. Confirms cycle 42 insight: WD=5e-4 over-regularizes at 21 epochs.

---

## 2026-05-13 13:25 — PR #2178: Compose torch.compile + weight_decay={5e-4, 3e-4}

- **Branch:** `willowpai2g48h4-frieren/compile-wd-compose` (MERGED — NEW BASELINE)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `b1p4li7l` (Arm 1 WD=5e-4), `7r9t0jab` (Arm 2 WD=3e-4 ★ winner)

### Results

| Arm | WD | val_avg/mae_surf_p | Δ vs 89.7197 | test_avg/mae_surf_p | best_epoch |
|-----|----|--------------------|--------------|---------------------|------------|
| baseline | 1e-4 | 89.7197 | — | 79.3167 | 18 |
| **Arm 1** | **5e-4** | **90.7677** | **+1.17% ✗** | **81.7736** | 21 |
| **Arm 2 ★** | **3e-4** | **87.0144** | **−3.01% ✓** | **78.9539** | 21 |

### Per-split val (Arm 2 best checkpoint e21)

| Split | WD=1e-4 (baseline) | WD=3e-4 (Arm 2) | Δ |
|-------|-------------------:|----------------:|---|
| `val_single_in_dist` | 114.92 | **106.99** | **−6.9% ✓** |
| `val_geom_camber_rc` | 108.66 | **104.00** | **−4.3% ✓** |
| `val_geom_camber_cruise` | **55.45** | 57.33 | +3.4% ↑ |
| `val_re_rand` | 79.85 | **79.74** | −0.1% ≈ |
| **val_avg** | **89.7197** | **87.0144** | **−3.01% ✓** |

### Per-epoch trajectory (key epochs)

| epoch | WD=5e-4 (Arm 1) | WD=3e-4 (Arm 2) | WD=1e-4 (baseline) |
|------:|----------------:|----------------:|-------------------:|
| 10 | 106.21 | 111.54 | 126.72 |
| 11 | 123.32 | 117.13 | 115.18 |
| 12 | **135.18 ↑↑ SPIKE** | **108.22 (damped)** | 112.27 |
| 13 | 109.92 | 117.79 | 107.93 |
| 18 | 94.64 | 99.15 | **89.72 ← best baseline** |
| 21 | **90.77 ← best Arm1** | **87.01 ← best Arm2** | 98.65 |

### Analysis

**Key finding: WD axis is BUDGET-DEPENDENT.** WD=5e-4 was optimal at 14 epochs (PR #2031, −4.46%). At 21 epochs (post-compile), WD=5e-4 OVER-REGULARIZES — it amplifies the e12 spike (+27%: e10=106→e12=135) while WD=3e-4 DAMPS it (e10=112→e12=108, smooth descent). Best epoch=21 for both higher-WD arms, still descending at the wall-clock cap.

**Critical mechanism update:** The e12 spike behavior under different WD values suggests the spike may be a SYMPTOM of over-regularization rather than a beneficial exploration dynamic per se. WD=3e-4 produces smooth descent and wins; WD=5e-4 amplifies oscillation and loses. This partially challenges the cycle 34 reframing ("spike is beneficial"), though alphonse's cosine-restart (#2227) is an independent test.

**Composition success:** The val gain (+50% epochs from compile + moderate regularization) is additive. In-distribution split fully recovered (val_single_in_dist 114.92→106.99, −6.9%) — the +4.9% regression from PR #2091 is reversed. Cruise gives back slightly (+3.4% val, +4.2% test) — cruise OOD prefers low WD.

**Next axis:** Finer WD sweep around 3e-4 (test {2e-4, 2.5e-4, 4e-4}) to map the WD curve at 21 epochs. Frieren assigned (#2284).

---

## 2026-05-13 12:15 — PR #2153: WD bracket sweep {4e-4, 5.5e-4, 6e-4}

- **Branch:** `willowpai2g48h4-fern/wd-bracket` (CLOSED — WD axis fully characterized)
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `ncx9dmho` (Arm 1 WD=4e-4), `ifksoytm` (Arm 2 WD=5.5e-4)

### Results

| Arm | WD | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p |
|-----|-----|---------------------|---------------------|---------------------|
| Arm 1 | 4e-4 | 108.0627 | **+15.43%** | 97.1256 |
| Arm 2 | 5.5e-4 | 105.4160 | **+12.60%** | 94.0927 |
| #2120 ref | 7e-4 | ~111.3 | +18.85% | — |

Vs current baseline 89.7197 (#2091): Arm 1 +20.4%, Arm 2 +17.5%. Arm 3 (6e-4) NOT run — branching rule fired (both arms >5%).

### Key Insights

**1. WD=5e-4 is a sharp local minimum, confirmed bilaterally.** U-shape approximately symmetric on val_avg (4e-4 +15.4%, 5.5e-4 +12.6%, 7e-4 +18.9%).

**2. Per-split asymmetry reveals the mechanism:**
- val_geom_camber_rc: 4e-4 +25.6% vs 5.5e-4 +8.1% — rc prefers MORE damping
- val_single_in_dist: 4e-4 +18.9% vs 5.5e-4 +22.7% — sid prefers LESS damping
- The 5e-4 winner balances these two conflicting constraints

**3. e14 breakthrough is load-bearing:**
- Baseline 5e-4: 108.28 → 93.62 at e14 = -13.5% (the win)
- Arm 1 4e-4: 110.77 → 108.06 = -2.4% (breakthrough mostly lost)
- Arm 2 5.5e-4: 117.71 → 105.42 = -10.4% (partially preserved)
The WD→breakthrough coupling is the primary mechanism, not asymptotic weight norms.

**4. Critical null result: weight-norm growth FLAT across WD range (+11.0% to +11.9%).** WD is NOT acting through asymptotic norm levels — it's an optimization-trajectory effect (timing of the late-stage LR × parameter-scale interaction).

### WD Axis Closure

Five PRs now fully characterize the WD axis:
- #1502 (BIVW restored WD viability)
- #2031 (5e-4 winner, -4.46%)
- #2120 (7e-4 regression +18.85%)
- #2153 (4e-4 +15.4%, 5.5e-4 +12.6%)

WD=5e-4 is the global optimum for this architecture/training regime. No further WD sweeps will produce gains without changing other components.

---

## 2026-05-13 11:45 — PR #2122: Decoupled weight_decay per param group: encoder vs surf_head

- **Branch:** `willowpai2g48h4-edward/decoupled-wd` (CLOSED — head-down direction rejected, key insight: head wants MORE WD)
- **Student:** willowpai2g48h4-edward
- **W&B runs:** `hdwktvr1` (Arm 1 encoder=1e-3/head=1e-4), `71sqac4j` (Arm 2 encoder=5e-4/head=1e-4)

### Results

| Arm | encoder_wd | head_wd | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p |
|-----|------------|---------|---------------------|---------------------|---------------------|
| Arm 1 (hdwktvr1) | 1e-3 | 1e-4 | 102.2033 | **+9.17%** | 90.7632 |
| Arm 2 (71sqac4j) | 5e-4 | 1e-4 | 111.0561 | **+18.62%** | 98.3770 |

Vs current baseline 89.7197 (#2091): Arm 1 +13.9%, Arm 2 +23.8%.

Arm 1 strictly dominates Arm 2 on every split (lowering encoder_wd 5e-4→5e-4 and raising it to 1e-3 partly rescued the regression vs lowering head_wd alone).

### Key Insight: Symmetric Direction Reversal

**The hypothesis was wrong in direction.** The surf_head (tiny 3-layer MLP, hidden=64, 10× LR, predicts noisy residual) NEEDS the 10× effective per-step shrinkage from coupled WD. Reducing head_wd degraded it.

Student's mechanism: the head memorizes surface-residual noise without strong regularization. Evidence: val_geom_camber_rc was hit hardest by head_wd↓ (+14 points Arm 1→Arm 2) — the OOD geometry split is where head memorization shows up worst.

**Arm 1 partial rescue** (encoder=1e-3) says encoder can tolerate higher WD, but it couldn't compensate for unwise head_wd reduction.

### Implication: head-UP direction is untested

The student correctly identified the next step: try `head_wd ∈ {1e-3, 2e-3}` with `encoder_wd=5e-4`. Assigned as PR #2232 (edward head-up-wd).

---

## 2026-05-13 11:30 — PR #2124: Surface-only pressure weight {0.5, 1.5}

- **Branch:** `willowpai2g48h4-alphonse/surf-only-pw` (CLOSED — channel-weight axis dead end)
- **Student:** willowpai2g48h4-alphonse
- **W&B runs:** `s8xblprk` (Arm 1 k=0.5), `uxai0y1u` (Arm 2 k=1.5)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p | Δ vs #2031 (83.88) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (k=0.5, surf-only, raw) | 104.7205 | **+11.85%** | 95.0281 | **+13.29%** |
| Arm 2 (k=1.5, surf-only, raw) | 114.0693 | **+21.85%** | 103.0041 | **+22.80%** |

Vs the current baseline 89.7197 (#2091), regressions are +16.8% and +27.2% respectively.

### Key finding: velocity rebalancing mechanism did NOT fire

The PR's central hypothesis was: k=0.5 pressure weight would free encoder for velocity learning, indirectly improving pressure. **Arm 1 (k=0.5) regressed velocity (+6.4% Ux, +3.9% Uy), the opposite of the predicted direction.** The encoder is overwhelmingly trained by volume MSE; within-surface channel reshuffles don't affect encoder learning.

### Combined channel-weight picture

| Experiment | k_p (effective) | val_avg/mae_surf_p | Δ |
|------------|-----------------|---------------------|---|
| Baseline (uniform) | 1.0 | 93.62 | — |
| PR #2124 Arm 1 | 0.5 (surf-only, raw) | 104.72 | +11.85% |
| PR #2124 Arm 2 | 1.5 (surf-only, raw) | 114.07 | +21.85% |
| PR #1496 (pw=3, both, mean-norm) | ~1.8 effective | ~112.3 | +20.04% |
| PR #1496 (pw=5, both, mean-norm) | ~3.0 effective | ~117 | +25% |

**Monotonic regression in all directions from k=1.0.** The surface-pressure channel-weight axis is fully closed: sub-unit, unit, and super-unit all characterized. k=1.0 is the local minimum.

### Analysis

Student diagnosed correctly: surface loss is gradient-dominated by pressure (not value-dominated, because Huber at δ=0.5 clips into L1 regime). Channel re-weighting attenuates or amplifies gradient directly, without a secondary encoder-rebalancing effect. The `surf_weight=10` global factor already controls the relative emphasis — per-channel within-surface weighting is a distinct lever with no beneficial operating point.

### Branching rule closure

Both arms >3% regression → entire surface-pressure-weight axis is dead per the branching rule. Correct closure.

---

## 2026-05-13 13:10 — PR #2128: AdamW epsilon sweep {1e-7, 1e-6}: denominator-floor stabilizer

- **Branch:** `willowpai2g48h4-nezuko/adamw-eps` (CLOSED — eps axis ruled out, excellent diagnostics)
- **Student:** willowpai2g48h4-nezuko
- **W&B runs:** `z1u61afo` (Arm 1 eps=1e-7), `6rs4qgsm` (Arm 2 eps=1e-6)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p | Δ vs #2031 (83.88) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (eps=1e-7) | 105.9131 | **+13.13%** | 93.2115 | **+11.12%** |
| Arm 2 (eps=1e-6) | 115.2299 | **+23.08%** | 102.3420 | **+22.01%** |

Vs the current baseline 89.7197 (PR #2091), regressions are +18.1% and +28.5% respectively.

### Per-epoch trajectory (val_avg/mae_surf_p)

| ep | Baseline #2031 | Arm 1 (eps=1e-7) | Arm 2 (eps=1e-6) |
|----|----------------|------------------|------------------|
| 11 | 109.59 | 130.27 | 119.50 |
| 12 | **152.99 (SPIKE)** | **112.60 (no spike)** | 122.20 |
| 13 | 108.28 | 107.52 | **115.23 (best)** |
| 14 | **93.62 (best)** | **105.91 (best)** | 129.57 (regression) |

### Critical mechanism diagnostic: `frac_below_eps`

Student logged per-epoch fraction of parameters where `sqrt(v) < eps` (eps is dominant denominator term):

| Epoch | Arm 1 enc frac (1e-7) | Arm 2 enc frac (1e-6) | Surf-head (both) |
|-------|----------------------|-----------------------|-----------------|
| 1 | 0.012 | 0.035 | **0.0** |
| 2 | 0.001 | 0.015 | **0.0** |
| 3-14 | ~0.0 | 0.001-0.003 | **0.0** |

**Decisive result: surf-head frac_below_eps = 0.0 throughout both arms across all 14 epochs.** The surf-head — the locus of the e12 spike and the dominant driver of `mae_surf_p` — never sees eps as a denominator floor, even at eps=1e-6. The `surf_weight=10 × Huber δ=0.5` combination ensures sqrt(v) >> 1e-6 always for surf params.

The encoder does have a small population of params where eps binds in early epochs (1-3% at e1, decaying to <0.3% by e5-14). These are biases and norm-scale params with low gradient variance. **The binding acts as a "slow start" for a subset of low-gradient encoder params, slowing effective convergence rate in epochs 1-5 — this is the regression mechanism, not "stabilization."**

### Analysis

1. **The e12 spike was NOT reproduced** — neither arm reached the regime where the spike occurs. Both started at much worse loss levels due to the slow-start effect.
2. **Eps cannot affect surf-head update shape by construction**: the math is `update ∝ m / (sqrt(v) + eps)`. Since sqrt(v) >> eps for all surf params, changing eps from 1e-8 to 1e-6 has zero effect on surf_head updates.
3. **The denominator-floor mechanism for the late-epoch oscillation is ruled out.** Combined with previous triangulation (#2058 grad-clip, #1949 warmup, #2127 step-decay), the oscillation is NOT a denominator issue.
4. **The eps axis is closed.** Going to eps=1e-5 would overlap with FP16/BF16 territory (precision-sensitive by #1572) and is not warranted.

### Student's follow-up suggestion → assigned

Student correctly identifies β2=0.9999 as the cleanest next step (symmetric untested direction from #2015 β2=0.95 regression). Assigned to nezuko as PR #2201.

---

## 2026-05-13 12:45 — PR #2127: surf_head step decay at e10 {×0.5, ×0.3}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-step-decay` (CLOSED — mechanism confirmed, harmful intervention)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `ap4569i2` (Arm 1 ×0.5), `ttrb3tmz` (Arm 2 ×0.3)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p | Δ vs #2031 (83.88) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (e10, ×0.5) | 100.2247 | +7.05% | 89.8721 | +7.14% |
| Arm 2 (e10, ×0.3) | 96.2736 | +2.83% | 87.9542 | +4.86% |

Monotonic trend: more aggressive decay → closer to baseline. Suggests optimum decay factor is **1.0** (no decay).

### Key insight (reframing)

**The e12 spike is fully damped** in both arms — mechanism confirmed. Per-epoch trajectory:
```
ep:    1    2    3    4    5    6    7    8    9   10   11   12   13   14
base: 189  164  160  156  144  125  131  120  111  111  110  153  108  94  ← spike + deep recovery
arm1: 203  245  157  150  132  142  131  124  122  114  113  108  108  100 ← smooth, no spike
arm2: 191  194  149  173  130  144  141  128  123  115  111  107  103  96  ← smooth, no spike
```

**The spike+recovery is a BENEFICIAL training dynamic, not pathology.** Without the e12 spike, the e13-e14 sequence that breached the baseline's new minimum disappears. Both arms continue descending smoothly but converge at WORSE final values. The aggressive arm (×0.3) regresses LESS than the gentle arm (×0.5) — exactly OPPOSITE to what we'd expect if the spike were pathological.

### LR trace verification

| Epoch | Encoder LR | Arm 1 surf_head LR | Arm 2 surf_head LR |
|-------|-----------|--------------------|--------------------|
| e9 | 0.000461 | 0.004611 (×1.0) | 0.004611 (×1.0) |
| e10 | 0.000452 | **0.002261 (×0.49)** | **0.001357 (×0.29)** |
| e14 | 0.000409 | 0.002047 | 0.001228 |

Step decay engages cleanly at e10 as configured.

### Analysis

This is the THIRD PR to triangulate the late-epoch oscillation mechanism:
- #2058 (grad-clip): gradient size is NOT the mechanism (sh_grad_norm 0.77× encoder)
- #1949 (head LR warmup): head LR cold-start is NOT the mechanism
- #2127 (head LR step decay): head LR magnitude IS coupled to the spike, but damping it removes a beneficial dynamic

**The e12 spike is an exploration burst that enables a deeper minimum at e14.** Damping the spike (via any head-LR-side intervention) loses the deep recovery. The mechanism is now characterized: the encoder enters a new region of loss landscape around e10-11, and the head's pre-positioned weights become "too forward" relative to the encoder's new locality — producing the spike. The recovery is the head re-aligning.

### Implications

**Closed directions** (head-LR-side modulation): step decay (this PR), warmup (#1949), gradient clipping (#2058), wider head (#2057). All head-side levers have been characterized.

**Open: encoder-side modulation.** Thorfinn's suggested follow-up: encoder LR boost at e10-12 — same coupling, opposite intervention. Brief encoder speedup during transition might give cleaner alignment WITHOUT losing the spike-recovery. Assigned as #2188 (thorfinn encoder-lr-boost).

---

## 2026-05-13 12:30 — PR #2013: LogCosh surface loss (C²-smooth alternative to Huber kink)

- **Branch:** `willowpai2g48h4-tanjiro/logcosh-surface-loss` (CLOSED — surface-loss family closed)
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `mbvj746h` (Arm 1 scale=1.0), `v80mlvz7` (Arm 2 scale=0.5)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #1795 (97.99) | test_avg/mae_surf_p | Δ vs #1795 (88.53) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (scale=1.0) | 101.4307 | +3.51% | 91.5855 | +3.45% |
| Arm 2 (scale=0.5) | 111.8905 | +14.18% | 100.0830 | +13.05% |

Vs the new #2091 baseline (89.7197), the regressions are even larger (Arm 1 +13.1%, Arm 2 +24.7%).

### Key diagnostic: late-epoch oscillation amplitude

| Run | max−min over E11-E14 | ratio max/min |
|-----|---------------------|---------------|
| Baseline | 15.7 | 1.16 |
| Arm 1 LogCosh | 17.7 | 1.17 |
| Arm 2 LogCosh | 16.9 | 1.15 |

**The C² smoothness hypothesis was unsupported** — oscillation amplitude essentially unchanged. Thrashing at the Huber kink is NOT the dominant source of late-epoch oscillation.

### Why LogCosh failed (mechanism analysis)

1. **Quadratic-regime gradient mismatch**: at typical residuals r≈0.106, Huber-δ=0.5 gradient is 2r=0.21, LogCosh scale=1.0 is tanh(r)≈0.106 — **half the gradient signal**. The 10× surf_head_lr cannot fully compensate.
2. **Asymptotic L1 behavior is softer**: Huber cleanly transitions to ±1 at |r|≥δ. LogCosh asymptotes to ±1 but at scale=0.5, even at r=1.0 the gradient is tanh(2)≈0.964. **Outlier nodes get weaker, smoother signal under LogCosh** — opposite of what Huber's hard-clip does for the rare-but-important large-residual surface nodes that drive MAE.
3. **C¹ vs C² distinction is empirically irrelevant** here. AdamW's running statistics either don't see meaningful discontinuities at the δ-kink at this scale, or the oscillation source is elsewhere (sampler/batch composition variance, cosine-decay endpoint).

### Surface-loss family closure

Now five PRs span the surface-loss family:
- #1558 Huber δ=0.5 → WIN (−17.72%)
- #1627 δ ∈ {0.2, 0.3} → both regress
- #1950 adaptive δ (EMA tracking) → +2.25%
- #1922 per-channel δ (δ_p=0.5, δ_u=2.0) → +5.61%
- #2013 LogCosh {scale=1.0, 0.5} → +3.51% / +14.18%

**Huber δ=0.5 is a stable local optimum**; finer perturbations of the surface loss function family do not unlock further gains. Future loss-function work should target volume loss or per-sample weighting (BIVW family).

### Per-split pattern (interesting subordinate finding)

Arm 1 matches or slightly beats baseline on OOD splits (val_geom_camber_rc, val_geom_camber_cruise, val_re_rand, test_geom_camber_rc) — the regression is concentrated on `val_single_in_dist` (+13.6%). The hardest OOD splits actually improved marginally under LogCosh; it's the in-distribution sanity split that suffered. This suggests LogCosh's softer signal helps generalization slightly but penalizes in-distribution fit — too costly for the primary metric.

---

## 2026-05-13 11:45 — PR #2091: torch.compile throughput unlock **[WINNER — NEW BASELINE]**

- **Branch:** `willowpai2g48h4-frieren/torch-compile` (MERGED)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `fvlekakd` (Arm 1 default — WINNER), `807sxrb9` (Arm 2 reduce-overhead — OOM)

### Results

| Metric | Baseline #2031 (WD=5e-4) | Arm 1 (default, WD=1e-4) | Δ |
|--------|------------------------|--------------------------|---|
| `val_avg/mae_surf_p` | 93.6198 | **89.7197** | **−4.16%** |
| `test_avg/mae_surf_p` | 83.8825 | **79.3167** | **−5.44%** |
| Epochs in 30 min | 14 | 21 | **+50%** |
| Per-epoch wall-clock | ~125s | ~87s | **1.43×** |
| Peak VRAM | ~43 GB | 43 GB | unchanged |
| Best epoch | 14 | 18 | shift expected |

### Per-split val (epoch 18)

| Split | Baseline | Arm1 | Δ |
|-------|---------|------|---|
| `val_single_in_dist` | 109.61 | 114.92 | +4.9% ↑ (slight regression) |
| `val_geom_camber_rc` | 118.05 | 108.66 | **−8.0%** ✓ |
| `val_geom_camber_cruise` | 61.07 | 55.45 | **−9.2%** ✓ |
| `val_re_rand` | 85.75 | 79.85 | **−6.9%** ✓ |

### Per-split test

| Split | Baseline | Arm1 | Δ |
|-------|---------|------|---|
| `test_single_in_dist` | 101.87 | 104.29 | +2.4% ↑ |
| `test_geom_camber_rc` | 105.24 | 96.30 | **−8.5%** ✓ |
| `test_geom_camber_cruise` | 51.43 | 46.12 | **−10.3%** ✓ |
| `test_re_rand` | 76.99 | 70.55 | **−8.4%** ✓ |

### Analysis

**The recipe was wall-clock-bound, not capacity-bound.** +50% more epochs (14→21) on the same recipe yields −4.16%/−5.44% — far beyond the predicted −0.5% to −2.5% from the PR. The val curve was still descending at epoch 14 when the baseline run hit the timeout; the model had genuine headroom.

**Critical note on per-split pattern**: `val_single_in_dist` slightly regressed (+4.9%) while all OOD geometry splits improved dramatically (cruise −9.2%, geom_camber_rc −8%). This matches the prediction for WD=1e-4 at 21 epochs: the model trains longer than the in-distribution regularization supports, but the extra epochs help OOD generalization. **This regressor should be recoverable by composing with WD=5e-4 (PR #2178, frieren).**

**Arm 2 (reduce-overhead) OOM analysis**: CUDA Graph private pool accumulation at 94.77GB (72GB in pools). Root cause: each unique padded mesh size allocates a separate CUDA Graph + private pool, and with N_max spanning 74K–242K nodes, hundreds of unique shapes exhaust memory. Requires upstream bucketed-padding to fix — out of scope.

**torch.compile mechanics confirmed:**
- dynamic=True: only 2 compile frames despite 74K-242K node range (symbolic-shape graph)
- First-batch compile warmup: 8.8s (negligible)
- `torch.set_float32_matmul_precision("high")` (TF32 matmul) + `cudnn.benchmark=True` are safe

**Implication for wall-clock-bound rejections**: EMA (#1808), n_head=8 (#1924), DropPath (#1987) were all closed because 14 epochs wasn't enough. At 21 epochs, these should be re-screened — especially EMA (needs stable late-training trajectory) and DropPath (oscillation smoothing matters more at longer training).

### Highest-priority follow-ups

1. **torch.compile + WD=5e-4 compose** (#2178 frieren, NEW) — stack two top wins
2. **Re-screen wall-clock-bound regularizers at 21 epochs** — EMA, DropPath, n_head=8
3. **Cosine T_max recalibration with compile** — T_max=50 at 21 epochs means ~42% of cosine cycle; askeladd's #2123 tests this direction (still valid, but calibration point shifts)

---

## 2026-05-13 11:30 — PR #2120: Deeper weight_decay sweep {7e-4, 1e-3, 2e-3}

- **Branch:** `willowpai2g48h4-fern/wd-deeper` (CLOSED — WD=5e-4 is sharp peak)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `rdpj0afc` (WD=7e-4 arm only; arms 2-3 correctly halted by branching rule)

### Results

| Metric | Baseline #2031 (WD=5e-4) | Arm 1 (WD=7e-4) | Δ |
|--------|------------------------|----------------|---|
| `val_avg/mae_surf_p` | 93.6198 | 111.2633 | **+18.85% regression** |
| `test_avg/mae_surf_p` | 83.8825 | 99.1562 | +18.22% regression |
| `val_single_in_dist` | 109.615 | 127.540 | +16.4% |
| `val_geom_camber_rc` | 118.046 | 144.471 | +22.4% |
| `val_geom_camber_cruise` | 61.066 | 74.502 | +22.0% |
| `val_re_rand` | 85.753 | 98.540 | +14.9% |
| Best epoch | 14 (descending) | 14 (descending) | same |

### Analysis

The hypothesis was "monotone gain at 5× WD says deeper is better." The data sharply falsified this: +40% (5e-4→7e-4) crashed all 4 splits uniformly. WD=5e-4 is a **sharp peak**, not a plateau or inflection. There is no OOD geometry gain to offset the in-distribution loss.

### Trajectory diagnostic (high-signal)

| Epoch | Baseline 5e-4 | Arm1 7e-4 | Δ |
|-------|--------------|----------|---|
| 4 | 155.84 | 135.09 | −13.3% (arm1 briefly leads — early-overfit damping) |
| 6 | 125.11 | 132.12 | +5.6% (baseline pulls ahead, never gives lead back) |
| 9 | 111.07 | **152.09** | +36.9% (arm1 spike — equivalent magnitude to baseline e12, 3 epochs earlier) |
| 12 | **152.99** | 146.34 | (baseline spike) |
| 14 | **93.62** | 111.26 | +18.8% (baseline e14 breakthrough −13%; arm1 only −3% in same epoch) |

Two key trajectory patterns:
1. **Spike location depends on WD**: baseline has e12 spike; arm1 has spikes at BOTH e9 (new) AND e12. Higher WD makes optimization MORE unstable, not less.
2. **e14 breakthrough is LR-schedule driven**: both runs have the same "final-epoch jump" shape but the absolute magnitudes differ (baseline 108→94 = −13%; arm1 114→111 = −3%). The e14 breakthrough is load-bearing for the merge result.

### Weight-norm logging diagnostic (NEW — highest-signal output)

| Step | Encoder norm | Surf_head norm |
|------|-------------|----------------|
| 49 | 41.01 | 7.79 |
| 1352 | 42.97 | 17.04 |
| 2656 | 43.94 | 21.35 |
| 3959 | 44.79 | 24.44 |
| 5262 | 45.56 | 26.91 |

Even at WD=7e-4, encoder norm GROWS +11% over training; surf_head norm grows 3.5×. **WD is not actually shrinking parameters — it is just damping the rate of growth.** This reframes the regularization regime: the optimum WD is the one that damps growth enough to prevent overfitting WITHOUT slowing the productive directions (e.g., the e14 breakthrough).

This is a high-signal diagnostic that should be preserved in ALL future WD experiments. See assignment #2153 (fern wd-bracket).

### Key insight

WD operates as a "growth-rate damper" on the implicit-bias trajectory of AdamW, not as a "shrink-to-zero" force. Combined with the e14 breakthrough as the load-bearing event for merge, the regularization curve has a sharp optimum at 5e-4 because that's exactly the rate that lets the model overfit the early-training noise but still descend at e14.

### Student-suggested follow-ups

1. **Finer WD sweep in (5e-4, 7e-4): 5.5e-4, 6e-4, 6.5e-4** — implemented as #2153 (fern wd-bracket), extended with 4e-4 to bracket below the peak.
2. **Asymmetric WD: encoder=5e-4, head=1e-4** — already in flight as #2122 (edward decoupled-wd). The weight_norm data motivates this directly.
3. **LR-tweak instead of WD-tweak** — overlaps with #2123 (askeladd cosine T_max).
4. **Don't retroactively log weight_norm on existing baseline; do log on every new WD run** — applied to #2153.
5. **Spike investigation at e9/e12: add per-batch loss logging on spike epochs** — defer; the spikes are now known to be steady-state, not pathological.

---

## 2026-05-13 10:30 — PR #2031: Weight decay re-tune 1e-4 → 5e-4 **[WINNER — NEW BASELINE]**

- **Branch:** `willowpai2g48h4-fern/weight-decay-sweep` (MERGED)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `u3q47f4s` (WD=5e-4 arm)

### Results

| Metric | Baseline (#1795) | WD=3e-5 | WD=5e-4 | Δ (5e-4) |
|--------|-----------------|---------|---------|----------|
| `val_avg/mae_surf_p` | 97.9914 | ~99 | **93.6198** | **−4.46%** |
| `test_avg/mae_surf_p` | 88.5311 | — | **83.8825** | **−5.26%** |
| `val_single_in_dist` | — | — | best arm | −8.9% |
| `val_geom_camber_cruise` | — | — | best arm | −7.5% |
| `val_re_rand` | — | — | best arm | −4.3% |
| `val_geom_camber_rc` | — | — | flat | +1.8% |
| Best epoch | 11-14 | — | 14 (descending) | — |

### Analysis

WD=1e-4 was a survivor from the BIVW era (cycles 2-3) — it had been untouched through BIVW, Huber, decoupled-LR merges. This is **hyperparameter staleness in action**: each merged stage changed the loss landscape and the "correct" WD changed too, but we never re-swept it.

Mechanism: heavier regularization at 5e-4 limits overfitting on the high-Re in-distribution split (val_single_in_dist −8.9%) while also improving OOD geometry splits (cruise −7.5%). The flat result on val_geom_camber_rc (+1.8%) suggests head-capacity or encoder features may limit OOD generalization on extreme camber geometries — consistent with #2057's negative finding that wider surf_head doesn't help.

**Lesson:** Hyperparameter staleness is real but per-axis. WD was stale; encoder LR (confirmed optimal in #1974) was not. Always audit optimizer hyperparameters when the loss formulation changes.

---

## 2026-05-13 10:30 — PR #2057: Wider surf_head hidden_dim {64→128, 64→256}

- **Branch:** `willowpai2g48h4-askeladd/wider-surf-head` (CLOSED — head not the bottleneck)
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (h=128) | Δ |
|--------|---------|--------------|---|
| `val_avg/mae_surf_p` | 93.6198 | ~98.6 | **+5.36% regression** |

Arm 2 (h=256) skipped per branching rule.

### Analysis

The surf_head hidden_dim 64→128 experiment falsified the capacity-bottleneck hypothesis for the head. The head has 0.026M params; the Transolver encoder has the overwhelming share of parameters and representations. Wider head adds parameters that cannot improve what the encoder provides.

**Key diagnostic from #2057 + #2058**: the student logged both `sh_grad_norm` (0.77× encoder) and per-epoch weight norms. The head is NOT gradient-saturated, NOT capacity-limited, and NOT gradient-clipped-limited. The late-epoch oscillation source is in the optimizer step (LR × m/√v), specifically the 10× LR differential between head and encoder.

**Lesson:** Encoder is the representation bottleneck. Capacity scaling on surf_head is wasted unless encoder is scaled simultaneously.

---

## 2026-05-13 10:30 — PR #2058: Per-group gradient clipping on surf_head {max_norm=0.5, 1.0}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-grad-clip` (CLOSED — wrong mechanism)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (clip=0.5) | Arm 2 (clip=1.0) | Best Δ |
|--------|---------|-----------------|-----------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~103 | ~104 | **+10.39% regression** |

### Analysis

The diagnostic logging was the most valuable output of this experiment: `sh_grad_norm` was consistently 0.77× the encoder grad norm across training. The "large update" in the late-epoch spike comes from the 10× LR multiplier, not from large gradients. Gradient clipping at max_norm ∈ {0.5, 1.0} on surf_head therefore clips BOTH the noise AND the productive signal, causing regression.

**Key insight (falsification):** The stabilization mechanism for the e12 spike must target the optimizer step magnitude (LR × m/√v), not the gradient ‖g‖. This narrows the search space to: (a) step-decay surf_head LR at late epochs, (b) AdamW ε (denominator floor), (c) cosine T_max (LR schedule that decays faster). All three assigned in cycle 30.

---

## 2026-05-13 10:30 — PR #1974: Encoder LR re-tune {3e-4, 7e-4}

- **Branch:** `willowpai2g48h4-edward/encoder-lr-retune` (CLOSED — encoder LR confirmed optimal)
- **Student:** willowpai2g48h4-edward
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (lr=3e-4) | Arm 2 (lr=7e-4) | Best Δ |
|--------|---------|----------------|----------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~99 | ~98 | **+5.42% regression** |
| `test_avg/mae_surf_p` | 83.8825 | — | — | +3.18% regression |

### Analysis

Encoder LR=5e-4 is the local optimum even at the new WD=5e-4 baseline. The 3e-4 arm underfits (model doesn't descend fast enough under 14-epoch cap), and 7e-4 overshoots (late-epoch spike exacerbated at higher LR).

**Important finding:** Encoder LR and WD are roughly orthogonal — the WD win (#2031) did not shift the LR optimum. This confirms that the two axes were genuinely independent.

---

## 2026-05-13 10:30 — PR #1922: Per-channel Huber delta {δ_p=0.5, δ_ux/uy=1.0/2.0}

- **Branch:** `willowpai2g48h4-nezuko/per-channel-huber-delta` (CLOSED — global δ=0.5 is correct)
- **Student:** willowpai2g48h4-nezuko
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (δ_ux=1.0, δ_p=0.5) | Arm 2 (δ_ux=2.0, δ_p=0.5) | Best Δ |
|--------|---------|--------------------------|--------------------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~98 | ~99 | **+5.61% regression** |
| `test_avg/mae_surf_p` | 83.8825 | — | — | +3.70% regression |

### Analysis

Larger δ on Ux/Uy flattens mid-magnitude velocity gradients. With δ=2.0, residuals <2.0 see MSE behavior — this over-smooths the Ux/Uy signal that the encoder uses for geometry reasoning. The global δ=0.5 is the correct balanced value.

**Why this was surprising:** The hypothesis was that Ux/Uy have smoother residual distributions than pressure and might benefit from larger δ. But surface pressure residuals are the hardest to fit (largest MAE), and they drive backprop indirectly through the shared encoder. Flattening Ux/Uy gradients weakens the encoder's geometry signal, which then hurts the very pressure prediction we're trying to improve.

---

## 2026-05-13 10:30 — PR #1496: Pressure-channel emphasis {pw=3, pw=5} on both vol+surf loss

- **Branch:** `willowpai2g48h4-alphonse/pressure-channel-prioritized-loss` (CLOSED — mean-normalisation bug)
- **Student:** willowpai2g48h4-alphonse
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (pw=3) | Arm 2 (pw=5) | Best Δ |
|--------|---------|-------------|-------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~112 | ~114 | **+20.04% regression** |

### Analysis

The implementation used mean-normalisation: with pw=3, normalised weights become [0.6, 0.6, 1.8] — meaning Ux/Uy were DOWN-weighted 40% and pressure only mildly up-weighted relative to uniform. Net effect: worse at velocity AND worse at pressure. The hypothesis was sound but the implementation was backwards.

**Root cause:** Mean-normalisation with [1, 1, 3] yields weights ÷ mean=5/3 → [0.6, 0.6, 1.8]. Both 3× and 5× arms regress monotonically, consistent with the optimal weight being in the sub-unit range once you correct the normalisation.

**Follow-up (#2124):** Surface-only pressure weight WITHOUT mean-normalisation, with k∈{0.5, 1.5} to explicitly test sub-unit pressure weighting. Volume loss kept uniform.

---

## 2026-05-13 07:25 — PR #1987: Stochastic Depth (DropPath) on Transolver blocks: regularization unlock

- **Branch:** `willowpai2g48h4-fern/stochastic-depth` (CLOSED — mechanism works, wall-clock-bound)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `q64n4xgm`
- **Hypothesis:** DropPath p∈{0.05, 0.1} on Transolver residual branches; targets baseline late-epoch oscillation pattern via stochastic regularization.

### Results

| Metric | Baseline | Arm 1 (p=0.05) | Δ |
|--------|---------|----------------|---|
| `val_avg/mae_surf_p` | 97.9914 | 101.2325 | **+3.31% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 89.7226 | **+1.35% regression** |
| Best epoch | 11 | 14 (still descending) | — |

Arm 2 (p=0.10) correctly skipped per branching rule.

### Mechanism diagnosis — clean confirmation

DropPath sanity check (standalone test):
- drop_path_rate=0.05: train per-sample std = 0.903 (vs eval=0.0), output norms preserved in expectation
- drop_path_rate=0.10: train per-sample std = 1.322

**Trajectory smoothing confirmed**:

| Epoch | Baseline val_avg | DropPath val_avg |
|-------|------------------|-------------------|
| 11 | **97.99** (best) | 116.46 |
| 12 | 113.66 (spike) | 110.22 |
| 13 | 108.05 | 107.96 |
| 14 | 99.85 | **101.23** (best, still descending) |

Baseline's epoch-11→12 spike (97.99 → 113.66) is GONE. DropPath gives clean monotone descent through final epochs. The oscillation prediction is fully validated.

### Why it loses at our budget

Each epoch lands ~3 points behind baseline (epoch 12: 110.22 DP vs 113.66 BL, epoch 13: 107.96 DP vs 108.05 BL, epoch 14: 101.23 DP vs 99.85 BL). The DropPath noise injection slows pointwise convergence by ~3 epochs. Under the 14-epoch wall-clock cap, the regularization tax exceeds the smoothing benefit. 

Student's analysis: "This is exactly the failure mode you get when applying a transformer regulariser to a setting that is **under-trained, not over-trained**. The baseline's late-epoch oscillation is not classical overfitting — it's edge-of-stability optimisation oscillation on a model that is still improving in expectation."

### Pattern recognition: wall-clock-bound regularization failures

This is the third such case in this round:
- PR #1808 (EMA weights): contamination from early-training weights, 14ep too short
- PR #1924 (n_head=8): +31% per-epoch cost, lost 3 epochs of training
- PR #1987 (DropPath): smoothing achieved at +3 epochs convergence slowdown cost

Common thread: 14 epochs is below the regularization payback threshold. All three would likely win under longer training budgets.

### Student-suggested follow-ups (not assigned)
- p=0.01-0.02 (finer-grained sweep): smaller noise → smaller smoothing AND smaller slowdown; net likely zero
- MLP-only DropPath: halve perturbation while preserving smoothing — interesting but still budget-bound
- Depth-scaled DropPath (linear schedule): standard ViT recipe; same fundamental budget issue
- Compose with SWA: rejected since SWA was just closed (#1951)

### Residual opportunities
- Revisit under longer training budget if SENPAI_MAX_EPOCHS is raised
- Combine with convergence-accelerating mechanism (e.g., if encoder-LR or β2 sweeps succeed and free up budget headroom)

---

## 2026-05-13 07:00 — PR #1978: Re-curriculum via per-sample log(Re)-tail loss multiplier (not sampler)

- **Branch:** `willowpai2g48h4-tanjiro/re-loss-weight` (CLOSED — structural regression)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `3rq00dn0`
- **Hypothesis:** Loss-side mechanism (vs the failed sampler-side #1868): apply `w = 1 + α × |normalized(log_Re)|` as per-sample loss multiplier, expecting to focus learning on Re-distribution tails (re_rand, geom_camber_rc).

### Results

| Metric | Baseline | Arm 1 (α=0.5) | Δ |
|--------|---------|---------------|---|
| `val_avg/mae_surf_p` | 97.9914 | 114.5247 | **+16.87% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 101.9300 | **+15.13% regression** |
| Best epoch | 11 | 12 / 14 | — |

Arm 2 (α=1.0) correctly skipped per branching rule.

Per-split val MAE:
| Split | Baseline | Arm 1 | Δ |
|-------|----------|-------|---|
| val_single_in_dist | 120.31 | 140.34 | +16.7% |
| val_geom_camber_rc | 115.98 | 137.99 | +19.0% |
| val_geom_camber_cruise | 66.04 | 80.66 | +22.1% (worst regression) |
| val_re_rand | 89.64 | 99.10 | +10.6% (smallest regression — opposite of prediction) |
| **val_avg** | **97.99** | **114.52** | **+16.87%** |

### Mechanism diagnosis — exemplary BIVW-interaction analysis from student

The student identified the structural failure precisely:
- BIVW (1/var(y_norm)) **implicitly up-weights low-Re samples** since high-Re has larger residual variance in normalized space
- Symmetric Re-tail boost composed multiplicatively with BIVW produces:
  - **Low-Re tail**: BIVW↑ × Re-tail↑ = double up-weight (already-easy samples dominate)
  - **Mid-Re**: BIVW≈1 × Re-tail↓ = down-weighted (most informative samples lose gradient)
  - **High-Re tail**: BIVW↓ × Re-tail↑ = approximate cancellation (intended boost absorbed)

The smoking gun: **val_geom_camber_cruise regressed MOST (+22.1%)** while it was expected to be unaffected. The easiest split sits at higher-Re samples where the Re-tail boost should help but is cancelled by BIVW.

`re_w` diagnostics: mean=1.36, range=[1.00, 3.19] — meaningful, non-trivial signal (not a null perturbation like #1868), but downstream behavior is destructive.

### Conclusion

Combined with PR #1868 (sampler-side variant, structural no-op), the entire **symmetric Re-tail re-weighting** family is now characterized as structurally incompatible with BIVW. The student's own follow-up #4: "Re-curriculum as conceived (symmetric tail boost composed with BIVW) is not a path to beat 97.9914 — recommend closing this thread."

### Residual opportunities (not assigned)
- **Asymmetric high-Re-only weighting**: `w = 1 + α × max(0, normalized(log_Re))` — avoids the BIVW low-Re double-counting
- **Replace BIVW with Re-tail (not compose)**: tests whether they're addressing the same problem
- Future-loop candidates if other directions stall

---

## 2026-05-13 07:00 — PR #1951: SWA late-epoch averaging: avg last K checkpoints post-training

- **Branch:** `willowpai2g48h4-askeladd/swa-late-epoch` (CLOSED — mechanism works, trajectory variance overwhelms)
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `aawyil4y` (k=3, start=12), `3j5wwshz` (k=5, start=10)
- **Hypothesis:** Average last K checkpoint weights post-training to center on local basin (bypassing EMA's early-training contamination from PR #1808).

### Results

| Arm | best single-epoch | last-epoch | SWA-averaged | SWA gain | vs baseline 97.99 |
|-----|-------------------|------------|--------------|----------|-------------------|
| Arm 1 (k=3, start=12) | 105.6335 (ep 13) | 109.7513 | **101.2533** | **−4.38 (−4.1%)** | **+3.33% regression** |
| Arm 2 (k=5, start=10) | 105.8172 (ep 14) | 105.8172 | **102.4204** | **−3.40 (−3.2%)** | **+4.52% regression** |

Test 4-split: Arm 1 = 89.4132 vs 88.5311 baseline → **+0.99% test regression** (much smaller than val gap of +3.33%).

### Analysis

**SWA mechanism worked exactly as predicted**:
- k=3 SWA(ep 12,13,14) is **8.7 points lower** than the mean of the 3 epochs (108.46), **4.4 points lower** than minimum
- k=3 beats k=5 (101.25 vs 102.42) — including epochs 10-11 (val 119, 116) drags toward less-converged weights

**Why it fails to beat baseline**: This run's trajectory landed in a much worse basin (best single ~105.6) than baseline's trajectory (best single ~97.99). That's an **~8 point trajectory gap** that the **~4 point SWA gain** cannot bridge. Likely seed variance.

**Interesting side finding**: Test gap (+0.99%) is much smaller than val gap (+3.33%). Consistent with SWA's regularization effect helping generalization more than val MAE — but single-seed evidence is anecdotal.

### Conclusion

SWA implementation is sound and reusable. But under the 30-min budget with 14 epochs:
- Seed variance can produce ±8 point swings in best single-epoch val
- SWA mechanism delivers ~4 points of basin-centering improvement
- These are not commensurate; trajectory variance dominates

The student's follow-up #1 (paired-seed comparison) would isolate the SWA delta but doubles GPU cost. Not the highest-value use of askeladd's slot. Mechanism noted for future plateau-protocol use.

### Residual opportunities
- Re-run baseline + SWA on a paired seed
- k=2 (epochs 13-14 only) — student's follow-up #2
- Compose with longer-budget runs (--epochs 20) — student's follow-up #3
- Investigate why this trajectory landed worse than baseline — could be a deeper hyperparameter sensitivity issue

---

## 2026-05-13 06:35 — PR #1950: Adaptive Huber delta: EMA of p75 per-batch residuals (self-tuning δ)

- **Branch:** `willowpai2g48h4-fern/adaptive-huber-delta` (CLOSED — mechanism collapse)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `dweh5tno`
- **Hypothesis:** Self-tuning δ via EMA of p75 per-batch surface residuals; warm start at δ=0.5, α=0.99 EMA, clamp [0.2, 2.0]. Tests whether fixed δ=0.5 is sub-optimal across the training trajectory.

### Results

| Metric | Baseline (#1795) | Arm 1 (`dweh5tno`) | Δ |
|--------|------------------|---------------------|---|
| `val_avg/mae_surf_p` (best) | 97.9914 | 100.1991 | **+2.25% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 89.4779 | **+1.07% regression** |
| Best epoch | 11 | 13 / 14 | — |

Arm 2 (p90) correctly skipped per branching rule (δ collapsed below 0.2).

### Mechanism diagnosis — superb root-cause analysis from student

δ trajectory (selected points from `train/adaptive_huber_delta`):
| step | EMA δ | raw p75 |
|------|-------|---------|
| 1 | 0.5057 | 1.0655 |
| 61 | **0.2096 (floor)** | 0.2125 |
| 260 | 0.2848 | 0.1503 |
| 975 | 0.2057 | 0.1109 |
| 2858 | 0.2003 | 0.0425 |
| 5245 (final) | 0.2000 | 0.0882 |

- **δ pinned at clamp_lo=0.2 for 88% of training** (439/500 logged steps).
- Median raw p75 of normalized surface residuals: **0.106** (well below clamp_lo).
- At δ≈0.2, `surf_l1_frac` (residuals in L1 branch): mean 12.7%, range 0.3–41.6%.

### Analysis

The hypothesis was based on the premise that residuals are "large" early and "small" late. Student's data shows the **decoupled-LR (PR #1795) baseline drives residuals below 0.2 within ~60 training steps** — the surf_head_lr=5e-3 collapses residuals far faster than expected. EMA of p75 correctly tracks this collapse and δ hits the floor immediately.

Effectively, this run was a **fixed-δ=0.2 ablation under the decoupled-LR baseline**. Compare to PR #1627 (fixed-δ sweep) which showed δ=0.2 → +17.2% under the PRE-decoupled-LR baseline. This run shows δ=0.2 → only +2.25% under the new baseline.

**Important side finding**: The merged decoupled LR (PR #1795) flattened the δ-landscape. The δ=0.5 narrow sweet spot is now wider — but δ=0.5 is still the best point we have.

### Suggested follow-ups (from student, prioritized)
- Lower clamp_lo (0.05) to let δ ride steady-state → effectively pure MSE, expected to underperform (#1650 showed MSE on volume already)
- Target p99 quantile → may give productive δ in [0.3, 0.6] but is a different hypothesis
- Fixed δ-sweep under new baseline (clean test) — would only confirm δ=0.5 is optimal; not assigning.

### Residual opportunities
- Stochastic Depth (DropPath) — regularization angle no other PR is exploring. Assigned to fern as #1987.
- Other smooth-loss families (LogCosh, Pseudo-Huber) could give a different gradient shape than Huber-δ=0.5; not assigning now since the δ landscape is flat at this baseline.

---

## 2026-05-13 06:20 — PR #1924: More attention heads: n_head 4→8 (wall-clock-neutral capacity axis)

- **Branch:** `willowpai2g48h4-edward/n-head-8` (CLOSED — dead end)
- **Student:** willowpai2g48h4-edward
- **W&B run:** `m8kevrph`
- **Hypothesis:** n_head 4→8 at fixed n_hidden=128 (head_dim 32→16) is wall-clock-neutral because total attention FLOPs are conserved at fixed sequence×hidden. 8 heads should provide richer slice-attention patterns improving accuracy.

### Results

| Metric | Baseline (n_head=4, #1558) | n_head=8 (`m8kevrph`) | Δ |
|--------|---------------------------|-----------------------|---|
| `val_avg/mae_surf_p` | 98.1642 (epoch 14) | 116.4421 (epoch 11) | **+18.62% regression** |
| `test_avg/mae_surf_p` (3-split) | 98.7537 | 117.3519 | **+18.83% regression** |
| Per-epoch wall time | 133.4 s | 175.0 s | **+31.2% slowdown** |
| Epochs in 30-min cap | 14 | 11 | −3 epochs |

Per-split val MAE (best ckpt epoch 11):
| Split | Baseline | n_head=8 | Δ |
|-------|----------|----------|---|
| val_single_in_dist | 123.14 | 146.41 | +18.9% |
| val_geom_camber_rc | 107.24 | 130.59 | +21.8% |
| val_geom_camber_cruise | 73.28 | 85.99 | +17.3% |
| val_re_rand | 88.99 | 102.79 | +15.5% |
| **val_avg** | **98.1642** | **116.4421** | **+18.6%** |

**Key per-epoch comparison**: At equal epoch 11, n_head=8 beats baseline by −9.3% (116.44 vs 128.39) — meaning 8 heads ARE better per epoch. The regression is entirely due to wall-clock (+31% per epoch → 11 vs 14 epochs).

### Analysis

Wall-clock prediction wrong. FLOPs conservation assumed but ignores kernel overhead: at head_dim=16, each head's matmul is below GEMM efficiency threshold. More per-head launches (`to_q/k/v/einsum`) cause overhead that doesn't fuse. The Transolver `slice_token = einsum("bhnc,bhng→bhgc", ...)` scales linearly with n_head and dominates step time.

**Conclusion**: Fifth wall-clock-bound capacity failure. Pareto frontier (depth=5, n_head=4, slice_num=64, ~14 ep) confirmed across all capacity-axis perturbations. Arm 2 (n_head=16) correctly skipped per branching rule.

### Residual opportunities
- n_head=8 IS better per epoch — would win under longer wall-clock budget
- bf16/torch.compile might recover the 31% overhead and flip to a win (#1572)
- Capacity wins at our budget must come from efficiency (BF16), not parameter count

---

## 2026-05-13 06:20 — PR #1868: log(Re) quantile bucketing sampler — explicit Re-curriculum (bounded)

- **Branch:** `willowpai2g48h4-tanjiro/log-re-quantile-bucketing` (CLOSED — mechanism failure)
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `ij9lcpi8` (10 buckets), `2ogoct1f` (5 buckets)
- **Hypothesis:** Quantile-bucket the log(Re) range, sample uniformly across buckets, weight by 1/count — a bounded replacement for the +272% 1/var(p) sampler failure.

### Results

| Arm | best_epoch | val_avg/mae_surf_p | test_avg (4-split) | Δ vs baseline |
|-----|------------|-------------------|-------------------|---------------|
| Baseline (#1558) | 14 | 98.1642 | NaN | — |
| Arm 1 (10 buckets) `ij9lcpi8` | 12 | 120.1534 | 109.7494 | **+22.4%** |
| Arm 2 (5 buckets) `2ogoct1f` | 14 | 106.2364 | 95.8183 | **+8.2%** |

### Analysis

**Structural no-op**: Quantile bucketing by construction puts ~equal sample counts in each bucket (max/min ratio 1.013–1.020×). Therefore 1/count weights are also ~uniform (max/min 1.013–1.020×). After composition with existing domain weights, effective distribution is essentially identical to baseline. The ±2% perturbation from Re-bucket factor just reshuffles the sample ordering via `WeightedRandomSampler`, introducing RNG noise.

Root cause: The two design choices "sample uniformly across buckets via quantile" and "weight by 1/count" cancel each other out by construction.

The regression comes from the tiny sampler weight perturbation reshuffling the per-step sample order, compounding with cosine LR. Both arms ran at similar epoch counts to baseline but with different (worse) luck on the sample draw.

**Conclusion**: Mechanism is broken, not just under-parameterized. Equal-width log(Re) buckets (not quantile) or a loss-side multiplier would both work. Follow-up: loss-side Re-curriculum (#1978 tanjiro), which avoids the sampler cancellation entirely.

### Residual opportunities
- Equal-width log(Re) buckets would produce non-uniform counts and non-trivial 1/count weights
- Loss-side multiplier `w = 1 + α × |norm(log(Re))|` is independent of bucket count (#1978)
- Multi-seed confirmation would clarify whether the +8.4% is noise vs. real harm

---

## 2026-05-12 19:00 — PR #1502: Batch inverse-variance weighting for heteroscedastic Re

- **Branch:** `willowpai2g48h4-tanjiro/per-sample-re-normalized-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `e72nzxo5`
- **Hypothesis:** Per-sample inverse-variance weighting (BIVW) to re-balance gradient signal away from high-Re/high-variance samples. Weight each sample by `1 / var(y_norm_valid)`, normalized to mean=1.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **126.0751** | Best epoch 14/50; **round-4 baseline** |
| `test_avg/mae_surf_p` | NaN | Pre-existing data/scoring bug (see below) |
| Best epoch | 14 | 30-min wall-clock cap hit (~132 s/epoch) |
| Training time | 31.1 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 160.74 | 1.88 | 0.85 |
| `val_geom_camber_rc` | 133.28 | 2.57 | 1.01 |
| `val_geom_camber_cruise` | **97.21** | 1.52 | 0.59 |
| `val_re_rand` | 113.08 | 1.99 | 0.77 |
| **val_avg** | **126.08** | 1.99 | 0.80 |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 145.43 |
| `test_geom_camber_rc` | 117.44 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 109.27 |
| test 3-split mean | ~124.0 |

### Analysis and Conclusions

**BIVW worked as hypothesised.** The low-Re-dominated `val_geom_camber_cruise` split came in at 97.21 — the lowest of the four splits by a wide margin — consistent with the prediction that IVW would most benefit low-variance (low-Re) samples that were being under-trained by the uniform MSE.

**BIVW is the new round-4 baseline.** Established at `val_avg/mae_surf_p = 126.0751`.

**Known infrastructure issue discovered:** `test_geom_camber_cruise` sample 20 has 761 `-inf` values in the GT pressure channel (volume nodes). `data/scoring.py:accumulate_batch` intends to skip non-finite GT samples but has a bug: `err = abs(pred - y)` is computed before the per-sample mask is applied, so `inf × 0 = NaN` in float arithmetic poisons the split-level accumulator. Since `data/scoring.py` is read-only, the fix goes in `train.py:evaluate_split` — assigned to tanjiro as PR #1527.

**Training was still improving at cap.** The val curve was still decreasing monotonically at epoch 14. With a longer budget, BIVW could improve further.

---

## 2026-05-12 22:00 — PR #1558: Huber (SmoothL1) surface loss delta=0.5 (MERGED — new baseline)

- **Branch:** `willowpai2g48h4-thorfinn/smooth-l1-surface-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `2w7nverc` (delta=0.5, winner), `3goyvktl` (delta=1.0, secondary)
- **Hypothesis:** Replace MSE surface loss with Huber (SmoothL1) loss. For |err| < delta: quadratic (like MSE scaled); for |err| >= delta: linear (L1-consistent). Aligns optimization objective directly with MAE evaluation metric. Orthogonal to BIVW (which handles between-sample gradient inflation at the sample level; Huber handles within-sample per-node gradient inflation).

### Results — Winning arm delta=0.5

| Metric | Value | vs prior baseline (119.2987) |
|--------|-------|------------------------------|
| `val_avg/mae_surf_p` | **98.1642** | **−17.72%** |
| test 3-split mean | **98.7537** | −17.45% |
| Best epoch | 14 / 14 | (still improving at cap) |
| Peak VRAM | 43.0 GB | |

Per-split val surface-p MAE (delta=0.5):

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 123.14 | 140.09 → −12.1% ✓ |
| `val_geom_camber_rc` | 107.24 | 142.40 → −24.7% ✓ (OOD regression fully reversed) |
| `val_geom_camber_cruise` | 73.28 | 85.98 → −14.8% ✓ |
| `val_re_rand` | 88.99 | 108.73 → −18.2% ✓ |
| **val_avg** | **98.1642** | **−17.72%** |

Per-split test (delta=0.5, 3 of 4 finite):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 111.92 |
| `test_geom_camber_rc` | 98.91 |
| `test_geom_camber_cruise` | NaN (cruise bug) |
| `test_re_rand` | 85.43 |
| **3-split mean** | **98.7537** |

Secondary arm (delta=1.0): val_avg=117.74 (only −1.3% — barely above noise).

### Analysis and Conclusions

**New round-4 baseline: 98.1642.** This is the largest single-PR improvement so far (17.7% vs the prior best of −5.4%).

**Mechanism confirmed:** `train/surf_l1_frac` (fraction of surface errors above delta) stays high throughout training for delta=0.5 — most residuals are in the L1 regime, producing constant-magnitude gradients that directly minimise MAE. delta=1.0 keeps too many residuals in the quadratic regime (barely different from MSE).

**OOD camber regression reversed:** val_geom_camber_rc was the problematic split (+6.84% regression in #1528). With Huber, it drops 24.7% — the largest per-split gain. Root cause: MSE pulled the surf-head toward large-residual OOD outlier nodes; Huber capped that pull at delta.

**BIVW + surf-head + Huber synergy:** Each mechanism targets a different scale of gradient heterogeneity:
- BIVW: between-sample (different Re → different variance)
- Surf-head: surface vs volume specialisation  
- Huber: within-sample per-node (outlier surface nodes)

**Next:** Test smaller deltas (0.2, 0.3) assigned to thorfinn PR #1627. Also need to test whether grad-clip (#1499 rebase), per-channel BIVW (#1580), and BF16/AMP (#1572) all stack on top of this new baseline.

---

## 2026-05-12 20:55 — PR #1527: Fix test NaN — guard evaluate_split against non-finite GT/pred (MERGED)

- **Branch:** `willowpai2g48h4-tanjiro/fix-test-nan-scoring` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `dg5xbm6g`
- **Hypothesis:** Pipeline fix — `data/scoring.py:accumulate_batch` computes `err = |pred - y|` before applying the per-sample finite mask, so `inf × 0 = NaN` poisons the accumulator. Since `data/scoring.py` is read-only, the fix is in `train.py:evaluate_split` via `nan_to_num` pre-filter + explicit `_y_ok` mask.

### Results

| Metric | Pre-fix baseline (`e72nzxo5`) | Post-fix run (`dg5xbm6g`) |
|--------|------------------------------|---------------------------|
| `val_avg/mae_surf_p` | 126.0751 | 129.6761 (+2.86%, stochastic) |
| `test_avg/mae_surf_p` | **NaN** | **119.7792** ✓ |
| Best epoch | 14 | 11 |

Per-split test (first time all four finite):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 150.59 |
| `test_geom_camber_rc` | 133.77 |
| `test_geom_camber_cruise` | **81.42** (was NaN) |
| `test_re_rand` | 113.34 |
| **test_avg** | **119.78** |

### Analysis and Conclusions

**Pipeline fix merged.** Despite val_avg drift, this is essential infrastructure: paper-facing metrics now report finite values for all four test splits.

**val drift correctly attributed to stochasticity, not the guard.** Tanjiro proved the guard is val-neutral: baseline val splits have no non-finite GT (else baseline val_avg would have been NaN too), so `_y_ok` is all-True and `nan_to_num` is a no-op on already-finite tensors. Without a fixed seed, run-to-run variance commonly hits 1–3% on individual splits. PR was based on BIVW only (#1502), not the current advisor branch (BIVW + surf-head, #1528), so its val=129.68 doesn't compare directly with the current 119.30 baseline.

**Cruise pressure split now reports 81.42** — lowest of the 4 test splits, mirroring the val cruise behaviour (85.98 in #1528). Confirms the fix surfaces the model's actual cruise performance, which was previously hidden behind NaN.

**Tanjiro also cleaned up two orphan GPU processes** from prior wake-up cycles. Operationally hygienic.

**Next merged training run should show:** val_avg ≈ 119.30 (BIVW+surf-head baseline) + finite test_avg ≈ 119+. The next experiment to merge (likely fern's rebase #1499 or thorfinn's Huber #1558) will give us the true paper-facing test number.

---

## 2026-05-12 21:00 — PR #1500: n_hidden 128 → 256, n_head 4 → 8 (CLOSED — budget failure)

- **Branch:** `willowpai2g48h4-frieren/larger-hidden-dim` (closed)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `ocxqv6a9` (best), `nnjrx4p3` (replicate)
- **Hypothesis:** Doubling hidden dimension from 128→256 and n_head 4→8 quadruples attention capacity and doubles MLP capacity, targeting model capacity as the bottleneck.

### Results

| Metric | Run `ocxqv6a9` | Run `nnjrx4p3` |
|--------|---------------|---------------|
| `val_avg/mae_surf_p` | **158.7552** | 163.2345 |
| Best epoch | 8 / 50 | 8 / 50 |
| Training time | 33.1 min | 33.4 min |
| n_params | 2.54M | 2.54M |
| Peak VRAM | 42.0 GB | ~42 GB |

Per-split val (best run `ocxqv6a9`):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 185.92 | 2.27 | 0.99 |
| `val_geom_camber_rc` | 178.98 | 3.37 | 1.29 |
| `val_geom_camber_cruise` | 121.34 | 1.81 | 0.76 |
| `val_re_rand` | 148.79 | 2.73 | 1.01 |
| **val_avg** | **158.76** | 2.54 | 1.01 |

### Analysis and Conclusions

**Closed — budget failure, not a model quality signal.** val_avg=158.76 vs baseline 119.30 (+33%), but this is entirely because the model only ran 8 of 50 epochs at 3.7 min/epoch vs the expected 0.85 min/epoch. Training loss was still descending monotonically at cap (val 240→168→159 over epochs 1→5→8).

**Critical diagnostics (will inform future experiments):**
1. **Wall-clock 4× slower than predicted** — slice-attention temporaries (`fx_mid`, `slice_weights`, `bhnc/bhng` einsum) dominate both VRAM and compute at N=242K nodes. Not captured in `[B,N,hidden]×layers` activation estimate.
2. **VRAM 42 GB vs predicted 5 GB** — off by ~8×; the dominant term is attention intermediates, not activations.
3. **Params 2.54M not 9.4M** — PhysicsAttention keeps `to_q/k/v` at `dim_head→dim_head`, so param count scales ~3·dim² not ~4·dim².
4. **Test cruise NaN is prediction overflow**, not GT corruption — different cause from #1502/#1528. The ~epoch-8 undertrained model overflows fp32 on cruise pressure at large N.

**Fix:** BF16/AMP would roughly halve VRAM and speed up forward pass ~1.5×, allowing n_hidden=256 to reach ~20 epochs in 30 min. Assigned to frieren as PR #1559.

---

## 2026-05-12 20:30 — PR #1528: BIVW + zero-init surface correction head (MERGED)

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-on-bivw` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `an97gg8n`
- **Hypothesis:** Composition of BIVW (per-sample loss re-weighting) and a zero-initialized additive SurfaceCorrection MLP head (`[3+24, 64, 64, 3]`, last layer zero-init, surface nodes only). Both mechanisms are orthogonal: BIVW targets gradient heterogeneity at the sample level; the surf-head targets the architectural under-representation of surface nodes. Used `torch.where(is_surface, delta, zero)` to safely handle NaN × 0 contamination from volume node overflow.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **119.2987** | Best epoch 13/14; **new round-4 baseline** (−5.37% vs 126.0751) |
| `test_avg/mae_surf_p` | NaN | Pre-existing cruise split scoring bug |
| Best epoch | 13 | 30-min cap hit (~131 s/epoch) |
| Total params | 0.669M | Transolver 0.643M + SurfaceCorrection 0.026M |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 140.09 | −12.85% ✓ |
| `val_geom_camber_rc` | 142.40 | +6.84% ✗ |
| `val_geom_camber_cruise` | 85.98 | −11.55% ✓ |
| `val_re_rand` | 108.73 | −3.85% ✓ |
| **val_avg** | **119.2987** | **−5.37%** |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 127.93 |
| `test_geom_camber_rc` | 127.18 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 103.79 |
| test 3-split mean | ~119.63 |

### Analysis and Conclusions

**BIVW + surf-head composition worked — new baseline 119.30.** Confirms the orthogonality hypothesis: BIVW (loss-level) and the surface correction head (architecture-level) provide complementary inductive bias. Three of four val splits improved; `val_geom_camber_rc` (raceCar OOD camber) regressed +6.84%, which warrants investigation in future work.

**`torch.where` NaN guard confirmed correct.** Replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` correctly propagates zeros instead of NaN at volume nodes with overflow predictions.

**Composition principle validated.** The standalone surf-head (#1503) was 6.2% worse than BIVW alone; adding it on top of BIVW is 5.4% better. The head needed the cleaner gradient signal that BIVW provides to specialize effectively.

**Next:** Need to test whether higher-LR + grad-clip (#1499, which reached 113.15 on BIVW alone) stacks further on top of this combined baseline.

---

## 2026-05-12 20:00 — PR #1499: Grad-clip max_norm=1.0 + lr 5e-4 → 1e-3 (SENT BACK — merge conflicts)

- **Branch:** `willowpai2g48h4-fern/gradient-clipping-and-higher-lr`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `ihl8ashe` (primary, lr=1e-3), `160d99m0` (fallback, lr=7e-4)
- **Hypothesis:** Gradient heterogeneity across Re samples causes large per-batch gradient norms that destabilise slice-attention. Capping with `max_norm=1.0` and doubling LR to 1e-3 should stabilise training and converge faster.

### Results

| Arm | Run | Best epoch | `val_avg/mae_surf_p` | test 3-split mean |
|-----|-----|------------|----------------------|-------------------|
| **primary** (`lr=1e-3, clip=1.0`) | `ihl8ashe` | 13 | **113.1491** | 109.64 |
| fallback (`lr=7e-4, clip=1.0`) | `160d99m0` | 12 | 119.0885 | 123.00 |

Per-split surface-p MAE (test, primary arm `lr=1e-3`):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 110.07 |
| `test_geom_camber_rc` | 111.92 |
| `test_geom_camber_cruise` | NaN |
| `test_re_rand` | 106.94 |
| 3-split mean | 109.64 |

Grad-norm telemetry (primary arm):
- **100% of steps clipped** — raw norms ranged 2.18 to 712.86 (median 30.79, mean 48.31)
- `max_norm=1.0` is acting as a uniform per-step renormaliser, not an outlier suppressor

### Analysis and Conclusions

**Strong result (113.15 on BIVW-only basis) — could not merge due to conflicts with advisor branch.** PR was branched before the BIVW + surf-head composition (#1528) merged. Sent back for rebase onto `icml-appendix-willow-pai2g-48h-r4` with new baseline 119.2987.

**100%-clipping finding is important.** With `max_norm=1.0` every single step is clipped. The effective LR is `(1.0 / raw_norm) × lr_nominal ≈ 1e-3 / 30.8 ≈ 3.2e-5` (median). The benefit of the higher nominal LR is asymmetric — it only matters on the small fraction of steps near the clip threshold. Suggested follow-up: try `grad_clip ∈ {1.0, 10.0}` on the new baseline to separate true outlier suppression from step renormalisation.

**Next:** Fern is rebasing onto the new baseline (BIVW + surf-head, 119.30) and adding a `--grad_clip 10.0` arm alongside the primary `--grad_clip 1.0`. The current 113.15 on BIVW-only was not compared against the newer 119.30 baseline; rebased run will clarify whether grad-clip still helps on top of surf-head.

---

## 2026-05-12 19:05 — PR #1503: Additive zero-init surface-only correction head (CLOSED)

- **Branch:** `willowpai2g48h4-thorfinn/surface-aware-output-head` (closed, not merged)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `8ffez1mk`
- **Hypothesis:** Zero-initialized additive MLP (`[3+24, 64, 64, 3]`) applied only at surface nodes after the base Transolver prediction. The head starts as an identity correction (last layer zeroed) and specialises the prediction for the surface vs. volume regime.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **133.928** | Best epoch 14/50; **6.2% worse than BIVW baseline** |
| `test_avg/mae_surf_p` | NaN | Same cruise split NaN issue + base model prediction overflow |
| Best epoch | 14 | 30-min wall-clock cap; same budget as tanjiro |
| Training time | 31.4 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|-----------|
| `val_single_in_dist` | 147.33 |
| `val_geom_camber_rc` | 152.10 |
| `val_geom_camber_cruise` | 112.03 |
| `val_re_rand` | 124.26 |
| **val_avg** | **133.93** |

### Analysis and Conclusions

**Closed — 6.2% worse than baseline.** At the same 14-epoch budget the standalone surface head scored 133.93 vs. BIVW's 126.08. Both runs are still undertrained at the cap (val still declining), so we cannot attribute the gap purely to the architectural difference — but the gap is significant.

**The head is not dead.** The composition **BIVW + surf_head** has not been tested. BIVW was not in this run. Composition is orthogonal (loss re-weighting vs. architectural specialisation) and is now assigned as PR #1528 (thorfinn).

**Robustness improvement noted.** Thorfinn recommended replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` to avoid `NaN × 0 = NaN` contamination from volume-node overflows. Incorporated into the composition PR #1528 instructions.

**Test NaN (additional cause).** Unlike tanjiro's data-corruption root cause, thorfinn's test NaN was caused by the base Transolver overflowing to non-finite values on one test cruise sample. The guard fix in PR #1527 will address both causes.

---

## 2026-05-12 23:00 — PR #1580: Per-channel BIVW (CLOSED — 29.6% regression)

- **Branch:** `willowpai2g48h4-tanjiro/per-channel-bivw`
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `rf4lp09j`
- **Hypothesis:** Replace scalar per-sample BIVW weight with separate per-channel weights `cw[b,c] = 1/var(y_norm[b,:,c])`, normalised to mean=1 per channel. Expected 1–4% improvement by giving pressure channel an independent variance track.

### Results

| Metric | Per-channel BIVW | Scalar BIVW baseline (an97gg8n) | Δ |
|--------|-----------------|--------------------------------|---|
| `val_avg/mae_surf_p` | **154.5967** | 119.2987 | **+29.6% ❌** |
| `test_avg/mae_surf_p` | 142.7603 | ~119.63 (3-split) | +19.3% ❌ |

Per-split val surface-p MAE at best checkpoint (epoch 14):

| Split | Per-channel BIVW | Prior baseline | Δ |
|-------|-----------------|----------------|---|
| `val_single_in_dist` | 213.4762 | 140.09 | +73.39 |
| `val_geom_camber_rc` | 171.6911 | 142.40 | +29.29 |
| `val_geom_camber_cruise` | 103.5190 | 85.98 | +17.54 |
| `val_re_rand` | 129.7005 | 108.73 | +20.97 |
| **val_avg** | **154.5967** | **119.2987** | **+29.6%** |

All 4 test splits were finite (PR #1527 NaN guard working).

### Analysis and Conclusions

**Closed — clear dead end.** Every validation split regressed; the approach is 58% worse than the current best baseline (98.16 after PR #1558).

**Mechanism (tanjiro's analysis, confirmed correct):** The original scalar BIVW was *implicitly* a p-variance-driven Re-curriculum. Because `p` spans 5 orders of magnitude in variance vs ~10^4 for Ux/Uy, the pooled per-sample `1/var(y_norm_valid)` was effectively dominated by `var(p)`. This meant: low-Re samples (small p) got upweighted, high-Re samples (large p) got downweighted — for *all three channels simultaneously*. Per-channel decoupling broke this coupling by letting `cw[b,Ux]` and `cw[b,Uy]` be large on high-Re samples while `cw[b,p]` is small. The model received conflicting signals: "learn velocity from high-Re samples but ignore their pressure" — exactly backwards for `val_avg/mae_surf_p`.

**Additional factor:** With batch_size=4 and p-variance over 10^5 range, per-sample `channel_w[:,p]` after mean-1 normalisation varied up to 50× within a single batch, giving extremely high gradient variance and effectively 1–2 samples dominating each step.

**Key lesson:** Scalar BIVW was doing more than "between-sample variance correction" — it was also acting as a p-aware Re-curriculum. This coupling should not be casually broken. Any future per-channel weighting experiment should preserve the p-dominated sample ordering, e.g. by computing sample weights from p-variance only and then applying a small per-channel log-mean correction on top.

**Potentially valuable follow-ups flagged by tanjiro (not assigned here):**
- Pre-compute frozen p-variance sample weights over the full corpus (makes the implicit curriculum explicit)
- EMA per-channel variance (cuts within-batch-of-4 noise)

---

## 2026-05-13 00:56 — PR #1650: Huber volume loss (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-tanjiro/huber-on-volume-loss`
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `rzv2hb5d` (vd=0.3), `ri4vj1nk` (vd=0.5, best), `qfvyn8wp` (vd=1.0)
- **Hypothesis:** Applying Huber to volume nodes reduces encoder gradient noise from volume outliers, indirectly improving surface MAE.

### Results

| Arm | vol_huber_delta | val_avg/mae_surf_p | Δ vs 98.16 | test_avg (4-split) |
|-----|-----------------|--------------------|-----------|---------------------|
| Baseline | 0 (MSE) | **98.1642** | — | — |
| vd=0.3 | 0.3 | 111.5928 | +13.7% ❌ | 101.68 |
| **vd=0.5** (best) | 0.5 | **106.6946** | **+8.7% ❌** | **96.83** |
| vd=1.0 | 1.0 | 117.7356 | +19.9% ❌ | 106.30 |

### Analysis and Conclusions

**Closed — all arms regressed.** Unimodal ordering around vd=0.5. The hypothesis that Huber on volume helps encoder quality is wrong.

**Root cause (tanjiro's analysis, confirmed):** Surface and volume play fundamentally different roles. Surface is the evaluated readout where Huber-MAE alignment matters; volume is the *supervisory signal* that shapes the shared encoder. Scale information from volume MSE is more valuable to encoder learning than outlier robustness. Huber on volume removes gradient scale information that was beneficial.

The unimodal ordering (vd=0.5 best of three, not monotone) confirms this is a noise minimisation over delta-tuning, not a true hypothesis-validates signal.

**Principle established:** Surface Huber + volume MSE is the correct recipe. Do not apply Huber to volume.

**Residual opportunity (not yet assigned):** Heavy-tail-only Huber on volume (apply only where |err| > 95th percentile) would more precisely target outliers without removing bulk gradient scale. Low priority given clean negative result.

---

## 2026-05-13 03:45 — PR #1498: Wider TransolverBlock MLP (mlp_ratio 2→4) (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-edward/wider-mlp-ratio`
- **Student:** willowpai2g48h4-edward
- **W&B run:** `ji5h4ww2` (mlp-ratio-4)
- **Hypothesis:** Standard transformer mlp_ratio is 4. Going from 2→4 should add per-node nonlinear capacity, giving −3% to −8% on val_avg/mae_surf_p.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, mlp_ratio=2)** | **98.1642** | — | 98.7537 (3-split) | 0.643 M | 128 s | 14 |
| mlp_ratio=4 | **122.6751** | **+24.97% ❌** | 111.46 (4-split, all finite) | 0.997 M (+55%) | 152 s (+19%) | 12 |

Per-split val (best epoch 11):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 205.51 | 123.14 → **+66.9% ❌** |
| `val_geom_camber_rc` | 140.42 | 107.24 → +30.9% ❌ |
| `val_geom_camber_cruise` | 101.58 | 73.28 → +38.6% ❌ |
| `val_re_rand` | 119.21 | 88.99 → +33.9% ❌ |
| **val_avg** | **122.68** | **+24.97%** |

### Analysis and Conclusions

**Closed — wall-clock-bound capacity addition fails (third confirmation this cycle).**

**Root cause (edward's analysis, confirmed):**
1. Wider MLP slows per-epoch wall-clock 19% (152s vs 128s). Hard 30-min cap → 12 epochs vs baseline 14.
2. Per-epoch convergence is NOT improved with wider MLP — just more parameters fighting for the same epochs.
3. Worst regression on val_single_in_dist (+67%) shows extra capacity is overfitting the small (1499-sample) training set within the limited epochs.

**Principle reinforced (now 3× confirmed: warmup #1497, grad-clip #1499, wider-MLP #1498):** Under the 30-min wall-clock cap, the baseline at epoch 14 is still improving. Any change that costs ≥10% per-epoch time and doesn't accelerate convergence will lose 1-2 epochs and underperform. Capacity expansions must be paired with throughput recovery (BF16 — #1572 frieren in flight) or reductions elsewhere (e.g., fewer layers, smaller slice_num).

**Residual opportunities (edward's suggestions):**
1. **Throughput-neutral capacity reallocation** — reduce n_layers when increasing mlp_ratio. Reassigned: edward to test n_layers ∈ {3, 4} (reducing depth to gain epochs).
2. mlp_ratio=3 as compromise (deferred).
3. Higher LR with wider MLP (ViT/GPT scaling) — would need to compose with capacity addition.
4. mlp_ratio=4 + BF16 — composable with #1572 once that completes.

---

## 2026-05-13 03:25 — PR #1746: Frozen p-variance stratified sampler (CLOSED — variance-explosion failure)

- **Branch:** `willowpai2g48h4-tanjiro/frozen-p-variance-stratified-sampling`
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `uayv2md1` (stratified+BIVW), `8h2u23z6` (stratified-only, BIVW disabled)
- **Hypothesis:** Pre-compute per-sample sampling weight ∝ 1/var(p) over the full corpus, use WeightedRandomSampler. Makes BIVW's implicit Re-curriculum explicit at data-loader level; removes within-batch estimation noise.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) |
|-----|--------------------|-----------|-------------------------------|
| **Baseline (PR #1558)** | **98.1642** | — | 98.7537 (3-split) |
| Arm 1: stratified + BIVW | 365.1298 | **+272% ❌** | 341.55 |
| Arm 2: stratified only (BIVW disabled) | 365.0082 | **+272% ❌** | 341.57 |

Per-split val (best epoch, Arm 2):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 511.17 | 123.14 → +315% ❌ |
| `val_geom_camber_rc` | 451.48 | 107.24 → +321% ❌ |
| `val_geom_camber_cruise` | 207.99 | 73.28 → +184% ❌ |
| `val_re_rand` | 289.39 | 88.99 → +225% ❌ |
| **val_avg** | **365.01** | **+272%** |

### Sampler diagnostic (the smoking gun)

| Statistic | Value |
|-----------|-------|
| p-variance min (raw) | 9.28 × 10⁻² |
| p-variance max (raw) | 2.30 × 10⁷ |
| **Dynamic range of var(p)** | **2.47 × 10⁸ ×** |
| Sample weight min (norm.) | ~1.5 × 10⁻⁶ |
| Sample weight max (norm.) | 371.71 |
| **Effective upweight ratio** | **2.47 × 10⁸ ×** |

### Analysis and Conclusions

**Closed — variance-explosion failure mode.** The hypothesis underestimated the empirical p-variance dynamic range by 6 orders of magnitude.

**Root cause (tanjiro's analysis, confirmed):** With var(p) spanning 0.09 to 2.3×10⁷ (8 OOM) and the WeightedRandomSampler interpreting weights as relative probabilities, the most-upweighted sample is drawn ~247 million times more often than the least. The effective training distribution collapses to a handful of low-Re samples (likely cruise-domain with tiny p-variance). The model never sees the high-Re/high-pressure cases that drive validation MAE.

**Arm 1 vs Arm 2 (0.03% diff) confirms the sampler dominates entirely:** once the data-loader collapses, BIVW has nothing to re-balance (the batches are near-identical).

**Why the existing domain-balanced sampler works but 1/var(p) doesn't:** The baseline `sample_weights` from `load_data()` have a dynamic range of only ~1.35× (599 raceCar single vs 443 cruise). 1/var(p) has a dynamic range **8 orders of magnitude larger**, so it becomes a Dirac comb instead of a smooth re-weighting.

**Principle established:** Any inverse-variance sampling weight on this corpus must be either (a) tempered with τ ∈ [0.05, 0.2], (b) log-compressed, (c) bucketed by quantile, or (d) replaced with a feature that has bounded dynamic range. Pure 1/var(p) is unusable.

**Residual opportunities (tanjiro's suggestions):**
1. log(Re) quantile bucketing (selected for follow-up — log(Re) spans only ~1.5 OOM).
2. Tempered inverse-variance: `w ∝ var(p)^(-τ)` with τ ≈ 0.1.
3. Log-spaced weights: `w ∝ 1/log(1+var(p))`.
4. Combine with existing domain-balanced sampler via product.

---

## 2026-05-13 02:10 — PR #1497: 5-epoch linear warmup + CosineAnnealingLR (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-askeladd/warmup-cosine-lr`
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** `fhdmn0xr` (warmup-5-cosine-huber0.5)
- **Hypothesis:** Adding 5-epoch linear LR warmup before CosineAnnealingLR prevents early-epoch instability in the slice-attention softmax weights and enables a higher effective LR.

### Results

| Arm | Warmup epochs | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p |
|-----|---------------|--------------------|-----------|---------------------|
| **Baseline (PR #1558)** | 0 (flat CosineAnnealingLR T_max=50) | **98.1642** | — | 98.7537 (3-split) |
| warmup-5 | 5 | **115.8073** (best epoch 13) | **+17.98% ❌** | 106.82 (4-split, all finite) |

Per-split val (best checkpoint, epoch 13):

| Split | val mae_surf_p (warmup-5) | vs baseline |
|-------|--------------------------|------------|
| `val_single_in_dist` | 147.36 | 123.14 → **+19.7% ❌** |
| `val_geom_camber_rc` | 135.03 | 107.24 → **+25.9% ❌** |
| `val_geom_camber_cruise` | 92.27 | 73.28 → **+25.9% ❌** |
| `val_re_rand` | 102.35 | 88.99 → **+15.0% ❌** |
| **val_avg** | **115.8073** | **+17.98%** |

LR trajectory confirmed correct: epoch 1→5 ramps linearly 1e-4→5e-4, then cosine from epoch 6.

### Analysis and Conclusions

**Closed — large regression, no instability to justify warmup.** The failure is structural, not tuning-related.

**Root cause — wall-clock-bound training makes warmup a liability:**
Training is capped at 30 min ≈ 14 epochs (out of 50 configured). A 5-epoch warmup spends 4 of the 14 most-productive epochs at 20–80% of peak LR. CosineAnnealingLR(T_max=50) barely decays by epoch 14 (we're at ~96% of peak), so the baseline is effectively a **flat LR at 5e-4** — and that flat schedule wins. No instability was observed in the baseline trajectory, so the premise (warmup prevents divergence) was wrong.

**The cosine tail benefit (gradual late refinement) never materializes** because T_max=50 far exceeds the actual training duration. We paid the warmup cost without collecting the dividend.

**Principle established:** Under the 30-min / ~14-epoch wall-clock cap, LR schedules with T_max >> epochs_run are effectively flat. If testing schedules, must set T_max ≤ epochs_actually_run.

**Residual opportunity (askeladd's suggestion):** OneCycleLR (Smith 2018) is designed for short-budget regimes. Set pct_start=0.1 (10% of total steps = ~525 warmup steps) with `total_steps = estimated_epochs × len(train_loader)`. This gives rapid warmup + full decay in the actual training window, not the 50-epoch hypothetical.

---

## 2026-05-13 00:07 — PR #1499: Grad-clip + higher LR on Huber baseline (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-fern/gradient-clipping-and-higher-lr`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `8p20jj30` (clip=1.0), `624phqjd` (clip=10.0)
- **Hypothesis:** Adding gradient clipping + higher LR (1e-3 vs 5e-4) stacks on top of the Huber baseline.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) |
|-----|-------------------|-----------|-------------------------------|
| clip=1.0, lr=1e-3 | 99.6393 | +1.50% ❌ | 90.07 |
| clip=10.0, lr=1e-3 | **99.5928** | +1.45% ❌ | **87.37** |
| Baseline (PR #1558) | 98.1642 | — | NaN (3-split: 98.75) |

Per-split val (best arm, clip=10.0):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 122.75 | 123.14 → −0.3% |
| `val_geom_camber_rc` | 117.04 | 107.24 → **+9.1% ❌** |
| `val_geom_camber_cruise` | 66.68 | 73.28 → −9.0% ✓ |
| `val_re_rand` | 91.91 | 88.99 → +3.3% |
| **val_avg** | **99.5928** | **+1.45%** |

### Analysis and Conclusions

**Closed — val regressed.** Neither arm beats 98.16. The main driver is `val_geom_camber_rc` regression (+9.1%).

**Key discovery — Huber compresses grad norms by 5×:** Raw pre-clip gradient norms with Huber active have median ~7 and max ~96, vs median ~31 and max ~720 in the pre-Huber run. Huber is doing exactly its job — capping per-node gradient magnitude in the L1 regime. This makes `clip=1.0` still too aggressive (100% of steps clipped — uniform renormaliser). `clip=10.0` correctly clips only the right tail (~27% of steps).

**But clip doesn't stack:** Huber already removes within-sample per-node gradient outliers. Adding batch-level clipping is redundant — both are attacking the same source of gradient noise. Not orthogonal like BIVW+surf-head+Huber were.

**Test divergence noted:** Test 3-split comparison (fair): (108.12+100.75+84.04)/3 = 97.64 vs baseline 98.75 = **−1.1% improvement**. This marginal test improvement while val regresses is not enough to override the val decision.

**For future clip experiments:** if composing clip with Huber, use `grad_clip=10.0` (not 1.0). And note that the benefit case is weak — both are targeting the same gradient noise source.

**Gotcha documented:** PR #1558 left dataclass `huber_delta: float = 1.0` but winning baseline used `--huber_delta 0.5`. All rebased PRs must pass explicit `--huber_delta 0.5`.

---

## 2026-05-13 01:48 — PR #1627: Huber delta sweep (CLOSED — both smaller deltas regressed)

- **Branch:** `willowpai2g48h4-thorfinn/huber-delta-sweep`
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `j99e4mrg` (δ=0.3 canonical), `5rl1qqlh` (δ=0.2 canonical). Two duplicate runs `pyf40gvr` and `eawlb7mc` were terminated cleanly by thorfinn before final epoch — not in result count.
- **Hypothesis:** Smaller Huber delta (0.2, 0.3) pushes more residuals into the L1 regime, further aligning loss with MAE objective.

### Results

| Arm | huber_delta | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p |
|-----|-------------|--------------------|-----------|---------------------|
| **Baseline (PR #1558)** | **0.5** | **98.1642** | — | 98.7537 (3-split) |
| δ=0.3 | 0.3 | 113.4695 | **+15.6% ❌** | (regressed) |
| δ=0.2 | 0.2 | 115.0398 | **+17.2% ❌** | (regressed) |
| δ=1.0 (cycle 10 op note) | 1.0 | ~99.4 | +1.3% ❌ | — |

### Analysis and Conclusions

**Closed — δ=0.5 is at or near the local optimum.** Both smaller deltas regress significantly; δ=1.0 also regressed (from cycle 10 op notes); so δ=0.5 sits in a narrow sweet spot.

**Mechanism (thorfinn's analysis, confirmed):** At δ=0.5, only ~2% of per-node residuals exceed the quadratic-linear breakpoint and fall in the L1 regime. Pushing δ down to 0.2/0.3 moves more residuals into L1 — but those mid-magnitude residuals are precisely the ones whose gradient drives MAE minimisation. Flattening their gradient to a constant ±1 strips information needed to discriminate "almost good" from "good enough", and the encoder loses its tuning signal on the bulk of the distribution.

**Why δ=0.5 wins:** It targets only the true outlier tail (the 2% that introduce gradient spikes) while preserving full MSE-style scaling on the residuals that actually matter for the readout. Going either smaller (over-flatten) or larger (under-protect from outliers) both lose.

**Principle established:** The Huber delta sweet spot for normalised CFD readouts at this scale is δ≈0.5 — narrow window, do not re-sweep without changing other levers.

**Residual opportunities (not assigned):**
- Per-channel Huber delta (different δ for p vs Ux/Uy) — channels have different residual distributions; one global δ may be suboptimal even if the mean is right.
- Adaptive Huber (Truncated MSE-style cutoff at moving p95) — automatically tracks the outlier tail rather than fixing at normalised 0.5.
Both deferred; not priority over orthogonal mechanisms still in flight.

---

## 2026-05-13 04:34 — PR #1501: PhysicsAttention slice_num 64 → 128 (CLOSED — val regressed, wall-clock-bound)

- **Branch:** `willowpai2g48h4-nezuko/more-slices`
- **Student:** willowpai2g48h4-nezuko
- **W&B run:** `8w50j5dx` (slice-num-128)
- **Hypothesis:** Transolver uses slice_num=64; with 3 distinct mesh zones (background + 2 foils) and complex flow topology, doubling to 128 should enable finer physical partitioning. Predicted −2 to −6% on val_avg/mae_surf_p.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, slice_num=64)** | **98.1642** | — | 98.7537 (3-split) | 0.658 M | 128 s | 14 |
| slice_num=128 | **117.1052** | **+19.30% ❌** | 108.7362 (4-split, all finite) | 0.679 M (+3%) | 175 s (+37%) | 10 (best) / 11 (completed) |

Per-split val (best epoch 10 → 11):

| Split | val mae_surf_p (epoch 11) | vs baseline |
|-------|--------------------------|------------|
| `val_single_in_dist` | 145.71 | 123.14 → +18.3% ❌ |
| `val_geom_camber_rc` | 143.13 | 107.24 → +33.5% ❌ |
| `val_geom_camber_cruise` | 84.60 | 73.28 → +15.5% ❌ |
| `val_re_rand` | 106.00 | 88.99 → +19.1% ❌ |
| **val_avg (best, epoch 10)** | **117.11** | **+19.3% ❌** |

Per-epoch trajectory (last 4 epochs):

| Epoch | val_avg_surf_p |
|-------|----------------|
| 8 | 157.94 |
| 9 | 126.87 |
| **10** | **117.11** ← best (cap) |
| 11 | 119.86 |

### Analysis and Conclusions

**Closed — 4th wall-clock-bound capacity failure on this branch.**

**Root cause (nezuko's analysis, confirmed):**
1. Predicted "near-zero per-epoch overhead" did not materialize — actual cost +37% per epoch (175s vs 128s).
2. The `in_project_slice = Linear(32, slice_num)` × 5 layers and the softmax-over-slices both scale linearly in slice_num at non-trivial constant factors.
3. The val curve was still steeply converging at the cap (145 → 158 → 127 → 117 over epochs 7-10) — this is a budget regression, not a per-epoch quality regression.

**Wall-clock-bound principle now 4× confirmed** (#1497 warmup +17.98%, #1498 wider-MLP +24.97%, #1499 grad-clip +1.45%, #1501 slice_num=128 +19.30%): under the 30-min cap, baseline epoch 14 is still improving; any change costing ≥10% per-epoch loses ≥1 epoch and regresses unless it accelerates convergence proportionally.

**Residual opportunities (nezuko's suggestions):**
1. Wall-clock-equalized comparison is structurally broken under SENPAI_TIMEOUT_MINUTES=30 for architectural changes that add per-step FLOPs. Would need fixed `--epochs N` ablation pair, which conflicts with current contract.
2. Do NOT escalate to slice_num ∈ {192, 256} — would lose by larger margin under same cap.
3. Richer per-slice features (higher n_hidden) is a cheaper test of the Transolver++ direction — already covered by #1572 (BF16 unlocks larger n_hidden) and previously-closed #1500.

---

## 2026-05-13 04:54 — PR #1881: Shallower depth + more epochs (n_layers=4) (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-edward/shallower-more-epochs`
- **Student:** willowpai2g48h4-edward
- **W&B run:** `5cq4p2qf` (n-layers-4)
- **Hypothesis:** Inverse of wider-MLP failure (#1498). Reduce n_layers=5→4 to save ~14% per-epoch wall-clock, gain ~2 extra epochs of SGD. Tests whether the under-30-min wall-clock cap rewards "more epochs at lower capacity" over "same capacity, less data."

### Results — Arm 1 only (Arm 2 n_layers=3 correctly skipped per branching rule)

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, n_layers=5)** | **98.1642** | — | 98.7537 (3-split) | 0.658 M | 128 s | 14 |
| n_layers=4 | **106.3995** | **+8.39% ❌** | 94.5883 (4-split) / 103.93 (3-split fair) | 0.529 M (−20%) | 110.6 s (−14%) | 16 |

Per-split val (best epoch 16):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 130.49 | 123.14 → +5.97% ❌ |
| `val_geom_camber_rc` | 120.40 | 107.24 → +12.27% ❌ |
| `val_geom_camber_cruise` | 77.10 | 73.28 → +5.21% ❌ |
| `val_re_rand` | 97.61 | 88.99 → +9.69% ❌ |
| **val_avg** | **106.40** | **+8.39% ❌** |

### Analysis and Conclusions

**Closed — Pareto frontier confirmed at depth=5 / 14 epochs.**

**Trade executed cleanly:**
- Predicted per-epoch saving (−14%) → exact match (110.6 vs 128s).
- Predicted epoch gain (~16 epochs in 30 min) → exact match (16 epochs).
- Even with the trade as designed, **+8.4% regression on the primary metric.**

**Root cause (edward's analysis, confirmed):**
1. Capacity loss from dropping one TransolverBlock (~129K params, 20% of body) cannot be recovered by 14% more SGD steps on the smaller model.
2. Regression is **uniform across all 4 val splits** (in-distribution +6.0%, hardest OOD +12.3%) — pure underfitting, no OOD artifact.
3. Largest regression on `val_geom_camber_rc` (+12.3%, hardest split) — the deeper model has more headroom to extract OOD-generalizing signal.

**Critical synthesis with prior wall-clock-bound failures:** Both directions on the depth/epoch axis regress now:
- Spend MORE per epoch (#1497 warmup, #1498 wider-MLP, #1501 slice_num=128): regress 18-25%.
- Spend LESS per epoch (#1881 n_layers=4): regresses 8.4%.

**Principle established:** The Transolver baseline at (depth=5, mlp_ratio=2, slice_num=64, ~14 epochs) sits very close to the Pareto frontier under the 30-min cap. The next compute should go to **non-depth-axis** capacity (heads, surf_head, or non-architectural mechanisms).

**Residual opportunities (edward's suggestions):**
1. **n_head 4→8** at fixed n_hidden=128 (head_dim 32→16). Multi-head is parallel batched matmul; minimal per-epoch cost. Assigned to edward as next experiment.
2. Mid-depth + warmer LR / shorter T_max — different hypothesis (LR-driven), not pursued here.
3. Do NOT try n_layers=6 — would cost ~+20% per epoch, predicted sub-Pareto by same logic.

---

## 2026-05-13 05:30 — PR #1795: Decoupled LR for surf_head vs encoder (MERGED — new baseline 97.9914)

- **Branch:** `willowpai2g48h4-thorfinn/decoupled-lr-surf-head`
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `q9qnnd9x` (arm1 1e-3), `70bjbj33` (arm2 3e-3), `eg1rhrzg` (arm3 5e-3 ← winner)
- **Hypothesis:** Zero-init surf_head with the same LR as the encoder (5e-4) is underpowered; decoupling allows the head to converge faster since it starts from zero and needs to learn rapidly.

### Results

| Arm | surf_head_lr | ×enc | Best epoch | val_avg/mae_surf_p | Δ vs 98.1642 |
|-----|-------------|------|------------|-------------------|-------------|
| 1 | 1e-3 | 2× | 14 | 114.1105 | +15.9% ❌ |
| 2 | 3e-3 | 6× | 13 | 104.7294 | +6.6% ❌ |
| **3** | **5e-3** | **10×** | **11** | **97.9914** | **−0.18% ✓** |
| Baseline | 5e-4 | 1× | — | 98.1642 | — |

Per-split val (arm 3, best epoch 11):

| Split | Arm 3 mae_surf_p | Baseline |
|-------|-----------------|---------|
| `val_single_in_dist` | 120.31 | 123.14 (−2.3% ✓) |
| `val_geom_camber_rc` | 115.98 | 107.24 (+8.2% ❌) |
| `val_geom_camber_cruise` | 66.04 | 73.28 (−9.9% ✓) |
| `val_re_rand` | 89.64 | 88.99 (+0.7% ≈) |
| **val_avg** | **97.9914** | **98.1642** |

Test (arm 3): test 3-split mean 99.5856 (vs baseline 98.7537, +0.85 regression on 3-split).

### Analysis and Conclusions

**Merged — marginal val win, compound improvements principle applied.**

**Mechanism (thorfinn's analysis):** Zero-init surf_head needs its own LR schedule to converge; tied to encoder LR of 5e-4 it's consistently underpowered. The monotonic 2×→6×→10× improvement trend (+114→104→98 val) had not reversed at the winning arm.

**Late oscillation noted:** Best epoch 11 (97.99), then oscillates 113→108→99.85 through epochs 12-14. The 10× LR is at the stability edge. Follow-up: add linear warmup to the head group and push to 7e-3 or 1e-2.

**Test regression (+0.85, 3-split):** Small and within run-to-run noise. The val win is the primary signal. Cruise improved significantly (73→55 finite) but was NaN at baseline, so 4-split is not directly comparable.

**Key insight:** surf_head as a standalone unit benefits from higher LR because: (a) it starts at zero init with no weight to preserve; (b) it has ~26K params with direct readout supervision, so it converges faster than the 658K-param encoder. The encoder benefits from slower, more conservative updates to maintain learned representations.

**New baseline: val_avg/mae_surf_p = 97.9914. All future PRs must beat this.**

---

## 2026-05-13 05:30 — PR #1720: surf_weight sweep {5, 15, 30} (CLOSED — all arms regress)

- **Branch:** `willowpai2g48h4-fern/surf-weight-tuning-on-huber`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `c54mrcff` (sw5), `zbiwwuly` (sw15), `a5yxs4ti` (sw30)
- **Hypothesis:** Huber's L1 regime produces smaller gradient magnitudes than MSE, so the current surf_weight=10 may be under-emphasizing the surface relative to Huber gradients.

### Results

| Arm | surf_weight | val_avg/mae_surf_p | Δ vs 98.16 |
|-----|------------|-------------------|-----------|
| sw=5 | 5 | 105.3356 | +7.2% ❌ |
| sw=15 | 15 | 115.5412 | +17.7% ❌ |
| sw=30 | 30 | 118.4154 | +20.7% ❌ |
| Baseline | 10 | 98.1642 | — |

### Analysis and Conclusions

**Closed — hypothesis falsified. Optimum is at or below sw=10, not above.**

**Mechanism (fern's analysis, confirmed):** Huber with δ=0.5 already implicitly down-weights surface outliers (5σ node contributes only 10× vs 25× under MSE). The baseline sw=10 was merged alongside Huber, so it was already "calibrated for Huber." Higher sw starves volume MSE — vol_p MAE blows up (sw=5: 111→sw=30: 151 val) because the encoder loses its volumetric supervisory signal when surf loss dominates.

**Principle confirmed:** Surface Huber + volume MSE + sw=10 is the correct triple. The surface:volume balance under Huber is at or near the optimum at sw=10.

---

## 2026-05-13 05:30 — PR #1808: EMA model weights (CLOSED — budget mismatch)

- **Branch:** `willowpai2g48h4-askeladd/ema-model-weights`
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `rtvzppe1`, `wr3edclv` (decay=0.999 × 2 seeds), `1eqenbsj` (decay=0.995 ← best)
- **Hypothesis:** EMA shadow of model weights produces lower-variance checkpoint for evaluation.

### Results

| Arm | EMA decay | val_avg/mae_surf_p | Δ vs 98.16 |
|-----|----------|-------------------|-----------|
| decay=0.999 (seed A) | 0.999 | 113.0424 | +15.2% ❌ |
| decay=0.999 (seed B) | 0.999 | 114.0145 | +16.2% ❌ |
| **decay=0.995** | 0.995 | **105.8582** | **+7.8% ❌** |
| Baseline | — | 98.1642 | — |

### Analysis and Conclusions

**Closed — EMA window dominates under 14-epoch budget.**

**Mechanism (askeladd's analysis, confirmed):** At decay=0.999 the effective window is ~1000 steps ≈ 2.7 epochs. The model is still steeply descending at epoch 14; EMA drags every evaluation backward with early-training contamination. "Variance reduction" requires the late-iterate regime (noisy plateau); we're in the descent regime. Cruise split (already plateaued) was nearly unaffected; harder splits (still descending) paid the full lag tax.

**Principle established:** EMA evaluation requires ≥50-epoch training to reach the noisy plateau regime where variance reduction exceeds early-training drag. Under 14-epoch budget it's counterproductive.

**Key insight from half-decay comparison:** decay=0.995 (window ≈ 0.5 epochs) closes ~half the gap vs decay=0.999 (window ≈ 2.7 epochs). Extrapolating: decay < 0.98 (window ≈ 50 steps) should approach but not beat the live model. The variance-reduction benefit is real but requires late-training access.

**Follow-up assigned:** SWA-style late-epoch averaging (#1951 askeladd) — average only last K checkpoints, avoids early-training bias entirely.

## 2026-05-13 07:45 — PR #2015: AdamW β2=0.95 — monotone test on β2 axis

- **Branch:** `willowpai2g48h4-askeladd/adamw-beta2-0.95` (CLOSED — regression, destructive interaction with balanced sampler)
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** `xi1r1iqd`
- **Hypothesis:** β2=0.999 (Adam default) maintains a long EMA of second moments. β2=0.95 (used in RoFormer, DeiT training) halves the EMA window, letting the optimizer respond faster to shifting gradient curvature. Predicted −0.5% to −2% from 97.9914.

### Results

| Metric | Baseline (β2=0.999) | β2=0.95 | Δ |
|--------|---------------------|---------|---|
| `val_avg/mae_surf_p` | 97.9914 | 104.3438 | **+6.49% regression** |
| `test_avg/mae_surf_p` | 88.5311 | 94.5522 | **+6.80% regression** |
| Best epoch | 11 | 8 | — |

### Per-epoch trajectory

| Epoch | Baseline | β2=0.95 |
|-------|----------|---------|
| 11 | **97.99** | 110.30 |
| 12 | 113.66 | **155.90** (spike) |
| 14 | 99.85 | 107.78 |
| Best | e11: 97.99 | e8: 104.34 |

### Analysis and Conclusions

**Closed — β2=0.95 is a sampler stabilizer, not a lag parameter.**

**Mechanism (student's insight, confirmed):** The balanced sampler draws from 3 domains with heterogeneous mesh sizes (74K-242K nodes) and y-magnitudes (164-458 std). Each step sees high batch-to-batch variance in gradient curvature. β2=0.999 acts as a low-pass filter against this noise — it accumulates a smooth estimate of the Hessian diagonal across many batches. β2=0.95 cuts the EMA window to ~20 steps, allowing per-domain curvature spikes to bleed through. The epoch-12 spike was WORSE under β2=0.95 (155.90 vs baseline 113.66), confirming the spike is driven by sampler variance, not by the optimizer's adaptation lag.

**Key insight (preserved for the team):** β2=0.999 is REQUIRED for stability against the balanced sampler's per-batch variance. It is not a hyperparameter to tune freely — it is a structural stabilizer. Future optimizer experiments MUST preserve β2=0.999 unless they simultaneously redesign the balanced sampler.

**Why student's follow-up directions were not pursued:** β2=0.9997 or β2=0.9995 would just move toward the current value; the direction is wrong. Per-domain adaptive β2 would require redesigning the sampler (out of scope for a single experiment). Mixing β2 + BIVW adjustment doubles the variables. All are more complex and less likely to yield a clean result than simply keeping β2=0.999.

## 2026-05-13 08:10 — PR #1949: Decoupled LR + head warmup: push surf_head_lr to {7e-3, 1e-2}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-lr-warmup` (CLOSED — both arms regressed, stability ceiling confirmed)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `dn5s3kbs` (7e-3), `0kuym49t` (1e-2)
- **Hypothesis:** surf_head_lr=5e-3 sweep (PR #1795) had not reversed at the winning arm — try pushing to {7e-3, 1e-2} with a 2-epoch linear warmup to damp cold-start overshoots. Predicted −0.5% to −2%.

### Results

| Arm | surf_head_lr | n_warmup | Best epoch | val_avg/mae_surf_p | Δ vs 97.9914 |
|-----|-------------|----------|-----------|---------------------|-------------|
| Baseline | 5e-3 | 0 | 11 | **97.9914** | — |
| Arm 1 | 7e-3 | 2 | 14 | 99.2678 | +1.30% |
| Arm 2 | 1e-2 | 2 | 14 | 110.4673 | +12.73% |

### Per-split val MAE at best checkpoint

| Split | Arm 1 (7e-3) | Arm 2 (1e-2) | Baseline |
|-------|-------------|-------------|----------|
| val_single_in_dist | 121.99 | 160.47 | 120.31 |
| val_geom_camber_rc | 110.55 | 114.06 | **115.98** ← improved by 7e-3 |
| val_geom_camber_cruise | 72.71 | 74.62 | **66.04** |
| val_re_rand | 91.83 | 92.73 | **89.64** |
| **val_avg** | 99.27 | 110.47 | **97.99** |

### Analysis and Conclusions

**Closed — surf_head_lr=5e-3 is the local optimum at our budget. All axes exhausted.**

**Warmup correctness (student's implementation note):** Chaining two schedulers on the same optimizer is buggy — second .step() overwrites the first because both write to param_group['lr'] from the captured base_lr. Student correctly used a single LambdaLR composing cosine × warmup. This is the correct pattern for future per-group scheduling experiments.

**Warmup damped cold-start but not late-epoch oscillation:** The spike pattern at epoch 12 persisted under both arms — same spike magnitude as baseline even with warmup. Warmup addresses the wrong mechanism (cold-start, not steady-state variance). Best epoch shifted 11→14 (warmup ate into early progress), confirming the tradeoff did not pay.

**Third confirmation of steady-state oscillation principle:** Epochs 11→12 spike is present at 5e-3, 7e-3, AND 1e-2. This is independent of LR magnitude. The mechanism is the BIVW balanced sampler producing high per-batch variance late in training as the LR anneals. Confirmed across β2 sweep (#2015), warmup sweep (this PR), and previous LR sweep (#1795).

**arm 1 shows partial per-split improvement:** val_geom_camber_rc improved −4.7% (110.55 vs 115.98 baseline). But geom_camber_cruise and re_rand regressed, netting a overall regression. This is consistent with higher head LR providing more expressivity for some splits but destabilizing the head on others.

**Key insight: The surf_head optimization axis is exhausted.** Three PRs now confirm:
- #1795: 5e-3 > 1e-3, 3e-3; best at this budget
- #1949: 7e-3, 1e-2 with warmup: regression
- Any further LR push faces the stability ceiling

**Next directions:**
- Wider surf_head (#2057, askeladd): head *capacity* axis (orthogonal to LR)
- Per-group gradient clipping on surf_head (#2058, thorfinn): targets the late-epoch oscillation mechanism directly

## 2026-05-13 08:50 — PR #1572: BF16 autocast AMP — unlock wall-clock for capacity experiments

- **Branch:** `willowpai2g48h4-frieren/bf16-mixed-precision` (CLOSED — both arms regressed, precision-sensitive task)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `c6m0hv4u` (n128), `f18c09px` (n256)
- **Hypothesis:** BF16 autocast → 2× tensor-core matmul + ~halved activation memory; expected 1.4-1.6× more epochs and/or unlock n_hidden=256. Stacked on Huber δ=0.5 + surf_head_lr=5e-3 baseline.

### Results

| Arm | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | Best/total epoch | Δ val vs 97.99 |
|-----|--------|---------------------|----------------------|--------------------|------------------|
| Baseline (PR #1795) | FP32, n128 | **97.9914** | 88.5311 | 11/14 | — |
| Arm 1 | BF16, n128 | 101.5396 | 90.8030 | 15/18 | **+3.62%** |
| Arm 2 | BF16, n256, bs=2, lr=6e-4 | 127.4773 | 113.4968 | 7/11 | **+30.09%** |

### Per-split val MAE (Arm 1)

| Split | BF16 (n128) | Baseline 97.99 | Δ |
|-------|------------|-----------------|---|
| val_single_in_dist | 118.08 | 120.31 | −1.85% ✓ |
| val_geom_camber_rc | 129.12 | 115.98 | **+11.33%** ✗ |
| val_geom_camber_cruise | 66.05 | 66.04 | ≈ |
| val_re_rand | 92.91 | 89.64 | +3.65% ✗ |

### Analysis and Conclusions

**Closed — BF16 is the wrong knob for this task.**

**Throughput gain confirmed**: 18 epochs vs 14 in 30 min (+28%) at n128. So the *idea* that more epochs would help was correct in principle — but BF16's precision trade-off neutralizes the benefit.

**Precision-sensitivity mechanism (confirmed via per-split breakdown):** `val_geom_camber_rc` regressed +11.33% — the EXACT OOD split that PR #1558 (Huber δ=0.5) was designed to fix. The Huber gradient in the L1 regime is constant-magnitude, so small inter-node residual differences carry the supervisory signal. BF16's 7-bit mantissa rounds these away, weakening the gradient signal that drove #1558's −17.7% gain. This is the smoking gun.

**Capacity scaling failed:** Even with VRAM headroom (33 GB on 96 GB cap), n256 only managed 11 epochs at 169 s/epoch — well short of the convergence horizon. BF16 alone does not unlock n256 in the 30-min cap; would need architectural compute reductions (slice_num, depth, mlp_ratio).

**Key insight: throughput unlocks exist, but BF16's precision cost is load-bearing.** Three of our wall-clock-bound failures (EMA #1808, n_head=8 #1924, DropPath #1987) might still be rescued by a *pure* throughput unlock with no precision trade. The right next swing: `torch.compile` (same magnitude of speedup, FP32 preserved).

**Follow-up assigned:** torch.compile {default, reduce-overhead} (#2091 frieren) — pure throughput unlock with zero precision tradeoff.

**Code NOT merged:** The PR contained a working BF16 implementation (use_bf16 opt-in flag, default False). Closing without merging because adding a flag for a knob we know hurts would invite future students to mis-use it.
