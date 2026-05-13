# SENPAI Research State

- **Last updated:** 2026-05-13 22:35 (Loop 19: #2500 alphonse anchor-mean CLOSED — both arms regress (λ=1: val 45.90, λ=5: val 47.28); mean-drift fix invalid in post-#2311 regime. 4 banked findings inc. **NEW per-split asymmetry: spread expansion HURTS `geom_camber_rc` (+2.72 val) but HELPS other splits** — important context for fern's #2604. Assigned #2616 alphonse film_mid_dim sweep {32, 128} — only untouched single-CLI capacity axis. All 8 students active. 5 PRs still stale_wip pending API rate-limit recovery.)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (~13-15 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Parser gotcha:** Avoid inline `SENPAI-RESULT:` substring in advisor comments — parser treats any line with that substring as a terminal marker and tries `json.loads` on what follows. Use "terminal-result post" or "SENPAI_RESULT" (underscore) in prose.

## ⭐ Current baseline (PR #2311 merged 2026-05-13 19:10 — hybrid Lion+AdamW Kendall σ fix on σ=0.5 stack)

- **val_avg/mae_surf_p:** **45.2181** (seed 0, SWA-model)
- **test_avg/mae_surf_p:** **38.7661** (seed 0, SWA-model, 4-split finite)
- Improvement over prior #2168 (45.7648 / 39.6619): val **−1.20%**, test **−2.26%**
- Cumulative improvement vs #1757: val **−32.14%**, test **−33.53%**
- Config: Transolver + FiLM (mid_dim=64) + Huber β=0.3 + Re-weight + Kendall σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=0.5) + **Lion lr=3e-4 wd=3e-4 (model) + AdamW lr=5e-4 wd=0 (log_σ heads, hybrid_kendall_lr=5e-4)**
- W&B confirmation run: `objur0b9`

### Per-split (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 46.967 | 40.340 |
| geom_camber_rc | 58.126 | 52.781 |
| geom_camber_cruise | 29.496 | 23.712 |
| re_rand | 46.283 | 38.231 |
| **avg** | **45.218** | **38.766** |

### Prior baseline (PR #2168 — RFF σ=0.5 on Lion)
- val 45.7648 / test 39.6619 (for reference in reviewing pending in-flight PRs)

## ✓ Merged improvements (all-time)

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 | smooth-l1-loss | MSE→Huber | val=100.77, test=90.38 |
| #1554 | swa-on-huber | SWA | val=99.07, test=88.90 |
| #1586 | re-weight | Per-sample Re | val=95.75, test=86.17 |
| #1585 | film-on-huber | FiLM | val=80.82, test=71.30 |
| #1731 | grad-clip-1p0 | max_norm=1.0 | val=74.62, test=66.14 |
| #1831 | max-norm-0p5 | max_norm=0.5 | val=73.81, test=65.04 |
| #1906 | kendall-uncertainty | Kendall σ | val=71.43, test=62.99 |
| #2082 | fourier-coord-features | RFF σ=1.0 | val=70.63, test=62.09 |
| #1757 | beta-0p3-on-rff-kendall | Huber β=0.3 | val=66.66, test=58.32 |
| #2063 | lion-optimizer-on-beta0p3 | Lion lr=3e-4 | val=47.64, test=40.57 |
| #2168 | rff-sigma-0p5-on-lion | RFF σ=0.5 (lower-freq prior) | val=45.76, test=39.66 |
| **#2311** | **hybrid-adamw-kendall-sigma** | **Hybrid Lion+AdamW(σ=5e-4), σ-spread restored** | **val=45.22, test=38.77 ← CURRENT** |

## Current active PRs — Wave 12 (post-hybrid-optimizer baseline 45.22/38.77)

**Decision rule (vs hybrid baseline 45.22/38.77):**
- val < 45.22: **merge** (true compound win on new baseline)
- 45.22–45.76: directional win on old σ=0.5 stack only, doesn't beat new baseline — request changes (composition test on new train.py if mechanism is orthogonal)
- val ≥ 45.76: regression vs σ=0.5 baseline — likely close
- ⚠ **All WIP PRs trained on the OLD codebase** (pre-hybrid-optimizer). If a PR's mechanism is orthogonal to σ-collapse (most SWA/RFF/wd PRs should be), it should compose. If it interacts with Kendall σ dynamics (unlikely but check per-PR), may need rerun on new train.py.

| PR | Student | Status | Mechanism | Notes |
|---|---|---|---|---|
| **#2604** | **fern** | wip (new, training) | hybrid_kendall_lr push {1e-3, 2e-3} on hybrid baseline | Test σ-spread ceiling; motivated by β–σ coupling discovery from #2540. **Per-split caveat from #2500:** spread expansion hurts geom_camber_rc — watch the per-split breakdown. |
| **#2606** | **thorfinn** | wip (new, training) | max_norm sweep {1.0, 2.0} on hybrid baseline | Relax saturating clip_fraction=1.0; 4 independent confirmations of clip saturation. Single CLI flag. |
| **#2616** | **alphonse** | wip (new, training) | film_mid_dim sweep {32, 128} on hybrid baseline | Only untouched single-CLI capacity axis. Motivated by #2500 per-split asymmetry. Single CLI flag, zero code change. |
| **#2484** | **frieren** | wip (training done, pending API) | Skip-SWALR entirely on σ=0.5 | Baseline shift notice sent. SWA mechanism orthogonal to σ. |
| **#2481** | **edward** | wip (training done, pending API) | SWA anneal_epochs=1 on σ=0.5 | Baseline shift notice sent. |
| **#2463** | **tanjiro** | wip (training done, pending API) | swa_lr ∈ {0.05x, 0.5x} sweep on σ=0.5 Lion stack | Baseline shift notice sent. SWA mechanism fully orthogonal to σ. |
| **#2442** | **nezuko** | wip (training done, pending API) | n_head ∈ {2, 8} bidirectional sweep at n_hidden=128 on σ=0.5 | Equal-compute capacity reshuffle; targets geom_camber_rc. Baseline shift notice sent. |
| **#2390** | **askeladd** | wip (training done, pending API) | Lion wd 2-arm {3e-3, 1e-2} on σ=0.5 | Mechanism validated on σ=1.0 (wd=3e-3 wins −0.56 val); composition test on σ=0.5 stack done. Baseline shift notice sent. |

**⚠ Mid-wave baseline shift:** σ=0.5 merged while 7 PRs were in-flight on σ=1.0 Lion stack. Notice posted to all 7 with updated thresholds. Triage rule for these landing runs:
- val < 45.76 → MERGE (mechanism compounded with σ=0.5 implicitly via independence)
- val ∈ [45.76, 47.64] → directional win on σ=1.0 stack only, NOT a beat on σ=0.5 stack → rebase + rerun on σ=0.5
- val ≥ 47.64 → regression on its own σ=1.0 baseline → close

**Mechanism-independence assumption check:** σ knob (input encoding) should compose orthogonally with optimizer (Lion fine-tuning: warmup, wd, T_max, grad-clip) and capacity (slice_num) — but NOT with loss-surface mechanisms that interact with RFF spectrum (none currently in-flight). Hybrid Lion+AdamW-for-σ (#2311 fern) is structurally orthogonal to RFF σ.

## Key banked mechanisms

1. **Lion = the biggest single lever** — 28.5% win vs β=0.3 baseline; 37.7% vs RFF baseline; 52.8% vs Kendall baseline
2. **Lion collapses Kendall σ heads** — all 6 log_σ identical. Lion+Kendall ≡ Lion+uniform-weight. **Structural, NOT capacity- OR encoding-driven**: confirmed at 1.61M params (nezuko #2354) AND at RFF σ ∈ {0.25, 0.5, 1.0} (thorfinn #2168). **Hybrid Lion(model) + AdamW(log_σ) confirmed as the structural fix (fern #2311):** 0.81 log-unit spread restored at AdamW lr=1e-3; surface-velocity emphasis 5× volume; mechanism prediction fully validated. Next: lr sweep + σ=0.5 rebase to convert mechanism win into metric win.
3. **β=0.3 and Lion compound** — val improved 50.97→47.64 going from β=0.0 to β=0.3 on Lion stack. Mechanisms are independent.
4. **max_norm=0.5 is the right grad-clip setting under Lion (#2347 CLOSED)** — Arm A (no clip) regressed +9% val; Arm B (max_norm=2.0) flat. Lion's sign-update does NOT smooth out outlier gradient spikes — the EMA carries forward magnitude-biased sign sequences. Refutes the Lion-paper intuition that grad-clip is optional under sign-update. **Fourth independent confirmation of σ-collapse robustness** (across max_norm ∈ {0.0, 0.5, 2.0}).
5. **SWA frac bounded below** — only frac≥0.75 averages in flat-loss region. EMA decay=0.999 (#2285) did NOT fix it (val=70.34, regression) — its 5-epoch window dilutes late-epoch low-lr updates with stale high-lr snapshots. Right fix is schedule shape: faster cosine T_max → eta_min plateau covers more averaging window. → testing now in #2342.
6. **RFF σ↓ wins under Lion+β=0.3 (NEW BASELINE)** — σ=0.5 compounds (−3.94% val / −2.23% test vs σ=1.0); σ=0.25 wins test by additional −0.65 (mechanism: low-freq Fourier = OOD-geometry smoothness prior in z-score-normalized coords). σ floor not yet found — #2407 probes σ=0.1.
7. **LayerScale γ=1e-4 fails at 5 layers** — ReZero γ=1.0 is the fix (#2269 fern)
8. **β=0.3 = β optimum, axis CLOSED** — β=0.1 (#2171) regressed +7.5%, β=0.2 (#2243) flat on val/+0.46% on test. Both directions exhausted. Edward's Kendall σ-relaxation mechanism confirmed (lower β → all 6 log_σ drift toward uniform).
9. **RFF spectral-dim axis CLOSED at n=16** — n=32 (#2170) gave mixed val/test direction; banked SWA-window-gating mechanism (timeout limits useful averaging) directly feeds tanjiro's #2342.
10. **Gradient Centralization axis CLOSED at small-data regime** — frieren #2240 cleanly disproved transfer from ImageNet (clip_fraction unchanged, SWA basin disrupted, OOD prediction direction opposite). Three banked findings inform follow-ups.
11. **clip_fraction=100% under default max_norm=0.5** — corroborated by frieren #2240 and edward's planning for #2347. Strong evidence max_norm=0.5 is over-constraining (whether under AdamW or Lion).
12. **Width-scaling capacity bumps gated by SWA window in 30-min cap** — n_hidden=192 took 43% longer per step, killed SWA. Future capacity bumps need either (a) earlier swa_start_frac OR (b) linear-cost dimensions (depth, slice_num) like #2378.
13. **Optimizer × σ × β=0.3 interaction is non-monotonic (NEW)** — σ↓ wins under Lion+β=0.3 and AdamW+RFF-only, LOSES under AdamW+β=0.3 (#2168 Arm 1: +0.45 val vs AdamW+σ=1.0 reference). AdamW's per-coord adaptive LR cancels σ↓ benefit at β=0.3; Lion's sign-update restores it. **Implication:** future σ-modifying experiments must check optimizer × loss-shape interaction.
14. **σ=0.25 wins paper-facing test but loses val by within-noise margin (#2168)** — test geom_camber_rc −4.88% / cruise −6.11% at σ=0.25 are the strongest OOD-geom test gains observed. Test curve hasn't bottomed out → #2407 probes σ=0.1.
15. **slice_num is NOT a capacity axis at fixed n_head/dim_head (#2378 CLOSED)** — only sizes `nn.Linear(dim_head, slice_num)` → +5K params at slice=96 (vs 64), not +310K. slice_num=96 regressed val +4.19 (+9.16%) and test +4.94 (+12.46%) vs σ=0.5 baseline. Also +16% step time → killed SWA window to 1 epoch. **5th σ-collapse confirmation** (invariant to slice_num at fixed heads). Future capacity bumps must touch n_hidden / n_layers / n_head — slice_num at fixed heads is dead.
16. **clip_fraction ≥0.995 even at 2× max_norm under β=0.3-Huber (#2270 CLOSED)** — both arms of alphonse's relaxation sweep failed to unclamp the optimizer; gradients exceed cap on >99.4% of steps regardless. Combined with #2347's no-clip + 4× cap results, this brackets clip-relaxation top-to-bottom on AdamW *and* Lion stacks. Mechanism: β=0.3-Huber is near-linear → systematically large pre-clip grad norms. **Axis fully closed** for {AdamW, Lion} × {β=0.3}. **Hypothesis cannot fire** without max_norm ≥ 2.0 or pre-clip diagnostics.
17. **AdamW+β=0.3+Kendall log_σ equilibrium values reconfirmed by #2270** — surf_p=−1.34, surf_ux=−1.49, surf_uy=−1.47, vol_p=−1.38, vol_ux=−1.34, vol_uy=−1.35. Within 0.05 of #1906. **High-confidence init target for #2443 (alphonse Kendall init at AdamW-equilibrium on Lion stack)** — single-arm structural alt to fern hybrid optimizer.
18. **SWALR overrides cosine immediately at swa_start_epoch — there is NO cosine eta_min plateau available to SWA (#2342 CLOSED, GOLD finding)** — `SWALR(swa_lr=cfg.lr*0.2=6e-5, anneal_epochs=2)` hijacks the lr schedule the moment `epoch >= swa_start_epoch` and ramps the optimizer toward swa_lr regardless of where cosine left it. **The "SWA averages cosine eta_min plateau" mental model behind #2187, #2285, #2342, and partially #2429 is mechanically wrong.** The current baseline is matched by COINCIDENCE — at swa_start_frac=0.75 of T_max=15, cosine lands at ≈5.9e-5 just below swa_lr=6e-5. **T_max compression is always harmful** (cuts useful cosine annealing + creates SWALR upward ramp during averaging). **Direction-of-ramp** is the right axis to test (now being probed by #2463 tanjiro swa_lr sweep). **Sharpened prediction for #2429 edward swa_start_frac sweep**: going EARLIER should win because cosine_lr at frac=0.5 ≈ 1.5e-4 → SWALR ramps DOWN to 6e-5 → plateaus 4-6 epochs at swa_lr.
19. **Lion wd is directionally correct (Chen 2023 confirmed) BUT acts through a non-shrinkage channel (#2390 askeladd SENT BACK)** — wd=3e-3 (10× baseline) beats wd=3e-4 by 0.56 val / 0.36 test on σ=1.0 stack, largest gain on `geom_camber_rc` (−1.49 val on the bottleneck). **Mechanism is NOT weight shrinkage:** param_norm differs <0.2% across 30× wd range. Lion's sign-step `±lr` overwhelms wd's pull-toward-zero; wd acts through some other channel — probably per-coord corrections to sign direction accumulating into better implicit regularization. **6th independent σ-collapse confirmation** (wd cannot rescue Kendall σ axis — log_σ params are in no-wd group). Composition with σ=0.5 being tested in rebased #2390 (wd ∈ {3e-3, 1e-2}). **NEW DIAGNOSTIC**: `train/param_norm` + `train/param_rms` logging added; will travel with the rebase.
20. **swa_start_frac < 0.75 axis CLOSED on Lion+cosine (#2429 edward) — compounding two regressions** — Arm 1 (frac=0.6) val=47.02 (+1.26), Arm 2 (frac=0.5) val=48.77 (+3.01); monotonic worse with smaller frac. **Mechanism characterization is the gold finding here**: (a) Lion's plateau onset at our 13-epoch budget is ≥ epoch 12, NOT 7-9 — train/loss drops 0.57-1.06 units AFTER SWA start in both arms; (b) **3rd independent SWALR-overrides-cosine confirmation** (after #2342 tanjiro, partial #2390 askeladd); (c) earlier swa_start_frac compounds two costs: pre-plateau averaging AND earlier base-model LR cut (model trains at LR ≤ 1.04e-4 for 5-7 epochs vs baseline 3 epochs); (d) **`geom_camber_rc` is the dominant error contributor (~2× the other splits)** — now the load-bearing OOD split for all future architecture/data work. **8th independent σ-collapse confirmation.** Sharpens follow-ups: #2481 anneal_epochs=1 (cooldown speed), #2484 skip-SWALR entirely.
21. **Lion does NOT have a chaotic init phase (#2363 frieren CLOSED)** — 3-epoch linear warmup made epoch-1 val WORSE (390.91 with warmup lr=1e-4 vs baseline 189.70 at lr=3e-4). Lion's sign(EMA(grad)) update direction is a SIGN — warmup at lower LR lands worse because magnitude shrinks but direction is unchanged. **Adam→Lion mental model transfer for warmup is refuted**, the high-loss epoch-1 in Adam is a chaotic-init phenomenon that Lion does NOT exhibit. Banked finding: don't assume Adam-paper recommendations transfer to Lion just because both are momentum-based.
22. **clip_fraction is invariant to lr schedule (#2363 frieren CLOSED)** — epochs 1-3 at lr ∈ {1e-4, 2e-4, 3e-4} all show clip_fraction ≈ 99-100%. Clipping happens on raw ‖g‖ pre-optimizer-scaling — the gradient norm distribution is set by the loss landscape, not by hyperparameters. **The persistent-clipping signature is a property of (model + data + loss), NOT a hyperparameter problem warmup/lr can fix.** Implication: future grad-clip experiments should manipulate `max_norm` or upstream loss/architecture, NOT lr schedule, to change clip_fraction. **9th independent σ-collapse confirmation** (log_σ trajectory identical to baseline through warmup phase).
23. **clip_fraction definition discrepancy flagged for follow-up (#2363 frieren)** — frieren measured 99-100% clip_fraction under max_norm=0.5 while BASELINE.md cites 74% from #2063. Worth investigating in a future diagnostic PR whether (a) measurement definitions differ (per-step boolean vs ratio) or (b) gradient norms have shifted across the merge series. Connects to alphonse's #2270 finding about pre-clip grad_norm distributions.
24. **Budget-binding interaction with lr-schedule manipulations (#2363 frieren CLOSED, banked principle)** — at SENPAI_TIMEOUT_MINUTES=30 → ~13 effective epochs, any lr modification costing >1 epoch of full-lr training will struggle to recover. By epoch 9 the warmup run was AHEAD on trajectory (67.17 vs baseline 78.74) but the budget cut before SWA could convert the lead. Future lr-schedule PRs (warmup, longer T_max, EMA warm-up) should account for the 13-epoch effective budget upfront when computing expected SWA gain.
25. **σ-collapse fix #2: Init at AdamW-equilibrium alone prevents collapse under Lion (#2443 alphonse CLOSED, banked mechanism)** — **strong-form refutation** of the previously-banked finding that "Lion's sign-update is wholly responsible for collapse". Per-channel Kendall gradient SIGN is sufficient signal to maintain differentiation given a non-degenerate starting point. log_σ trajectory: 0.150 (init) → 0.478 (final), monotonically growing, never collapsing. **Three-tier σ-spread ordering on σ=0.5 Lion stack:** Lion+Kendall baseline (spread 0.000) < AdamW+Kendall (spread ~0.15) < Lion+AdamW-eq-init (spread 0.478) < Hybrid Lion+AdamW (spread 0.81, fern #2311 pending). **Test improved by −0.40 on paper-facing metric** (geom_camber_rc −0.72, single_in_dist −1.04) despite val regression of +0.61. **Open mechanism:** Lion's sign-update still drifts mean(log_σ) ~0.6 nats more negative (−1.40 → −1.99), inflating all effective weights ~3× → val regression. **#2500 alphonse tests the mean-anchor fix** (L2 anchor loss on mean(log_σ) at AdamW-eq target).
26. **Val/test divergence first observed in this direction on Wave 12 (#2443 alphonse)** — differentiated Kendall weighting (Lion+AdamW-eq init) acts as an OOD regularizer: slight val degradation, real test gain especially on harder OOD splits. For paper-facing test number this is interesting; for the merge gate it's a regression. Connects to #2390 askeladd's wd-not-shrinkage finding — both suggest Lion has multiple knobs that act through OOD-regularization channels distinct from the model parameter values directly.
27. **σ-collapse fix #1 confirmed and 0.55-val winner on σ=0.5 stack (#2311 fern Arm 2)** — hybrid Lion(model) + AdamW(log_σ) at lr=5e-4: val 45.22 (−0.55) and test 38.77 (−0.90). Spread 0.475, mean −1.98 (5× more spread than AdamW equilibrium, comparable to alphonse's #2443 init mechanism). **Test wins on 3/4 splits including single_in_dist −2.11 and geom_camber_rc −1.82** — biggest gains on the load-bearing OOD splits. Branch rebased CLEAN; awaiting confirmation rerun before merging. **Compound potential with #2500 anchor-mean** is the natural next step — both mechanisms (optimizer split + mean anchor) attack different parts of the σ-stability problem and should stack.
28. **σ axis exhausted on val; σ=0.25 test win NOT seed-robust (#2407 thorfinn CLOSED)** — five σ values tested (0.1, 0.25×2 seeds, 0.5, 0.75, 1.0). σ=0.1 hits val floor (+1.03%); σ=0.25 cross-seed test mean (39.72) ties σ=0.5 (39.66). σ→smaller direction dead on primary metric; **σ=0.5 stays canonical**. Seed gap +1.42 at σ=0.25 is well above val-gap noise estimate ~0.86 → seed-0 favorable noise draw, not a real improvement.
29. **σ=0.1 mechanism revision: REGULARIZER not OOD-prior (#2407 thorfinn CLOSED)** — gain driver is **single_in_dist test −3.22%**, NOT OOD-geometry. geom_camber_cruise REVERSES (+3.43% worse than σ=0.5); geom_camber_rc only tied with σ=0.25. **The original mechanism story partially survives at rc but fails at cruise.** Low-σ Fourier features become near-degenerate (rff_mean=0.44 cos-dominated; rff_std=0.553 vs theory 0.707) → reduced per-point positional info → smoother predictions → in-distribution test win. **Directly motivates #2512 multi-scale RFF**: if σ=0.1 acts as regularizer, it should compose with σ=0.5's coordinate resolution.
30. **2nd independent clip_fraction≈0.99 confirmation under Lion+max_norm=0.5 (#2407 thorfinn)** — both arms: 0.992 / 0.997 (matches frieren #2363's 0.99 flag). BASELINE.md's "~0.74" note is from a different regime / earlier stack — under our current σ=0.5 baseline, max_norm=0.5 clips every step → Lion effectively becomes sign-of-sign at fixed norm. **Banked for future max_norm relaxation pass on the merged stack** (alphonse #2270 already closed max_norm relaxation under β=0.3+Huber+AdamW; under-Lion remains open).
31. **10th + 11th independent σ-collapse confirmations (#2407)** — both Arm 1 (σ=0.1) and Arm 2 (σ=0.25 seed-1) show `final/log_sigma_*` = −0.9037 (Kendall clamp floor). σ-collapse mechanism is **invariant to RFF σ ∈ {0.1, 0.25, 0.5, 1.0}** at the optimizer level — independent of input encoding bandwidth. Reinforces the structural-not-data nature of the collapse identified in #2311 and #2443.
32. **β–σ coupling mechanism (#2540 fern CLOSED, NEW)** — Higher Huber β systematically collapses the per-channel Kendall log_σ differentiation established by #2311's hybrid AdamW fix. σ-spread: β=0.3=0.475 → β=0.5=0.401 (−16%) → β=1.0=0.304 (−36%). The β knob is NOT orthogonal to σ-differentiation; widening β throws away part of the hybrid-fix gain. Mechanism: as β↑, smooth-L1 residuals quadratically attenuate where |r|<β; gradient magnitudes ∝ 1/β; cross-channel SNR in log_σ gradient signal shrinks → per-channel heads converge toward uniform. **β=0.3 is doubly load-bearing — wins via both direct loss optimum AND σ-preservation. Directly motivates #2604 hybrid_kendall_lr push (test if more spread compounds).**
33. **Refuted grad-norm prediction for smooth-L1 (#2540 methodology finding)** — PR-body predicted p99‖g‖ ordering β=1.0 > β=0.5 > β=0.3. Observed: β=0.5 (36.06) > β=1.0 (28.57). Smooth-L1 grad slope is 1/β in |r|<β region; bulk of residual mass at mid-late training lies there → slope-1/β regime dominates → lower β → higher grad-norm. **grad-norm is NOT a valid proxy for "MSE-likeness" in this regime.** Useful for future Huber-tuning hypotheses.
34. **12th independent clip_fraction confirmation (#2540 β=0.5 arm: 0.9949, β=1.0 arm: 0.9590)** — clip-saturation under max_norm=0.5 is now 12-strong. Strongly motivates #2606 max_norm relaxation experiment.
35. **σ axis decisively closed at σ=0.5 (#2512 thorfinn CLOSED)** — multi-scale RFF 8×σ=0.5 + 8×σ=0.1 does NOT compose on hybrid stack (val +2.66%, test +1.20%). Mechanism revision: σ=0.1's #2407 single_in_dist test gain was **σ-collapse-conditional** — once #2311 fixed σ-collapse via hybrid AdamW, the regularizer effect dissolves. **single_in_dist test trajectory:** 42.45 (collapsed-σ + σ=0.5 RFF) → 40.34 (hybrid-σ + σ=0.5 RFF, #2311 win) → 41.39 (hybrid-σ + multi-scale RFF, this PR). Banked structural finding: **Multi-scale RFF and Kendall σ-differentiation share an information channel — #2311 substitutes for the regularization role low-σ RFF was filling.** 10 distinct σ configs evaluated total since Lion adoption; σ=0.5 wins everywhere. **Next σ-related axes (multi-scale, σ-bracketing) are exhausted.**
36. **Anchor mean(log_σ) is a clean control knob — but the natural hybrid-AdamW equilibrium dominates (#2500 alphonse CLOSED)** — λ=1 → mean=−1.996, λ=5 → mean=−1.748; mechanism decouples mean from per-channel differentiation. **Important finding:** the natural hybrid-AdamW equilibrium (~mean −2.0) outperforms target mean −1.4 (AdamW-only-eq) by val ≈ 1–2 MAE units. The AdamW-eq target measured in #2270/#1906 was specific to AdamW with weight decay on the main model; the new hybrid (lr=5e-4 + wd=0 on log_σ) reaches a structurally different per-channel optimum at deeper σ. **The hybrid optimizer doesn't just unblock differentiation — it shifts the optimum mean too.**
37. **AdamW-eq init expands σ-spread (#2500 alphonse, NEW)** — scalar init at −1.4 produces final spread 0.70 vs hybrid baseline's 0.475 from zero-init. Mechanism: scalar init at "deeper σ" makes initial residuals smaller, log_σ_i differentiates faster, spread expansion accelerates. **Init magnitude affects equilibrium spread, NOT just convergence speed.**
38. **Per-split σ-spread asymmetry: spread expansion HURTS `geom_camber_rc`, HELPS other splits (#2500 alphonse, CRUCIAL)** — Arm 1 vs new baseline (val): single_in_dist −0.13 (WIN), cruise −0.45 (WIN), re_rand +0.57 (LOSS), **geom_camber_rc +2.72 (LOSS)**. The OOD bottleneck has the OPPOSITE preference from other splits on σ-spread axis. **Direct implication for fern's #2604 hybrid_kendall_lr push:** pushing spread UP may regress further on geom_camber_rc, even if it wins on average. Per-split tension may require architectural-level fixes (e.g., FiLM capacity, alphonse's #2616) rather than σ-tuning alone.

## Key open bottlenecks

1. **geom_camber_rc** — **load-bearing OOD split, ~2× the other splits' error**. val=58.13, test=52.78 at new hybrid baseline (improved from 58.29/54.60 at σ=0.5 baseline; PR #2311 gave −1.82 test gain here). Being attacked via #2442 nezuko (n_head sweep — capacity axis), #2390 askeladd (Lion wd), #2512 thorfinn (multi-scale RFF). **Future architecture/data work should prioritize this split.**
2. **SWA window only averages 2-3 epochs at swa_lr=6e-5** (timeout at 13/15 epochs) — **MULTIPLE axes now in flight after the SWALR-overrides-cosine mechanism characterization**:
   - #2463 tanjiro swa_lr value (0.05x vs 0.5x baseline floor)
   - #2481 edward anneal_epochs=1 (1-epoch ramp vs 2-epoch) — gets all 3 averaged epochs at the swa_lr floor
   - #2484 frieren skip-SWALR entirely — averages cosine-tail weights instead of SWALR-floor weights
   - **swa_start_frac<0.75 axis CLOSED** (#2429 edward); start_frac=0.85 (later) untouched
3. **Lion+Kendall σ-collapse** — **3 INDEPENDENT FIXES NOW IDENTIFIED**: (a) #2311 fern hybrid Lion+AdamW (PENDING WINNER: val 45.22 / test 38.77, spread 0.81); (b) #2443 alphonse init at AdamW-equilibrium (BANKED: spread 0.478, test −0.40 but val +0.61); (c) #2500 alphonse mean-anchor loss (in flight; targets mean drift mechanism uncovered by #2443). Compound (hybrid + init + anchor) is the natural future best.
4. **Lion lr = 3e-4 confirmed near optimum** (#2297 V-shape). Lr axis CLOSED.
5. **Lion warmup axis CLOSED on this 13-epoch budget** (#2363 frieren) — Adam→Lion mental model fails; warmup at lower LR makes early epochs WORSE not better.
6. **Capacity-axis dead zone** — width (#2354), depth (legacy), slice_num (#2378) all gated by SWA window or step-time cost. n_head sweep (#2442) tests equal-compute reshuffle at fixed n_hidden. If n_head=2 or n_head=8 wins, banked axis; if both regress, capacity bottleneck is genuinely orthogonal to attention granularity.

## Potential next directions (Wave 12 / post-hybrid-optimizer baseline)

1. ~~**Lion + hybrid AdamW for Kendall σ heads (#2311 fern)**~~ — **MERGED ✓** (val 45.22 / test 38.77).
2. ~~**Multi-scale RFF 8×σ=0.5 + 8×σ=0.1 (#2512 thorfinn)**~~ — **CLOSED ✗** (val +2.66%, test +1.20%; σ axis exhausted).
3. ~~**huber_beta re-sweep on hybrid merged baseline (#2540 fern)**~~ — **CLOSED ✗** (β=0.3 robust; new β–σ coupling mechanism banked).
4. ~~**Anchor mean(log_σ) loss at AdamW-eq + init at eq (#2500 alphonse)**~~ — **CLOSED ✗** (both arms regress; per-split asymmetry banked).
5. **hybrid_kendall_lr push {1e-3, 2e-3} on hybrid (#2604 fern)** — directly tests if σ-spread is the lever. β–σ coupling from #2540 confirms σ-spread is load-bearing; push lr=5e-4→{1e-3, 2e-3} to widen spread further. **Caveat from #2500:** may regress geom_camber_rc.
6. **max_norm sweep {1.0, 2.0} on hybrid (#2606 thorfinn)** — 4 independent confirmations of clip_fraction=1.000 (saturating clip on every step under max_norm=0.5); relax to see if more gradient direction through compounds.
7. **film_mid_dim sweep {32, 128} on hybrid (#2616 alphonse NEW)** — only untouched single-CLI capacity axis; targets the geom_camber_rc per-split asymmetry from #2500.
7. **swa_lr ∈ {0.05x, 0.5x} sweep on σ=0.5 (#2463 tanjiro)** — averaging-lr level axis from #2342 mechanism finding.
8. **SWA anneal_epochs=1 on σ=0.5 (#2481 edward)** — SWALR ramp speed; all 3 averaged epochs at swa_lr.
9. **Skip-SWALR entirely on σ=0.5 (#2484 frieren)** — direct test of SWALR-overrides-cosine mental model; cosine-tail vs SWALR-floor averaging.
10. **n_head ∈ {2, 8} bidirectional at n_hidden=128 on σ=0.5 (#2442 nezuko)** — equal-compute attention-granularity reshuffle; targets geom_camber_rc.
11. **Lion wd sweep on σ=0.5 (#2390 askeladd)** — orthogonal Lion fine-tuning axis.
12. **Delay-SWA-start (frac=0.85)** — narrow window but truly low-lr averaging (tanjiro #2342 suggested follow-up #3). Composes with #2463 Arm A. **Opposite direction from CLOSED #2429.**
11. **Second seed on σ=0.5 baseline** — confirm win magnitude (currently single seed; val effect ~3.9% is well above seed noise).
12. **Coordinate system rethink** — polar/arc-length around airfoil; geom_camber_rc primary target. May compound with RFF σ↓ since both reduce frequency content.
13. **Test-time augmentation** — if test still falling at lower σ, maybe inference-time geometry perturbation could push test_geom_camber_rc further.
14. **Fixed (non-learnable) per-channel loss weights** matching #1906 pattern — even simpler alt than #2443 if init-drift turns out to be the issue.
15. **n_layers depth sweep at fixed n_hidden** — if #2442 closes head axis, depth at fixed compute is the next architectural axis.
16. **clip_fraction definition audit / pre-clip grad_norm p50/p90/p99 logging** — resolve the 99% (frieren #2363, thorfinn #2407 confirmed) vs 74% (#2063 BASELINE.md) discrepancy; informs future grad-clip and gradient-distribution work.
17. **max_norm relaxation under Lion+σ=0.5** — clip_fraction=99% under max_norm=0.5 means Lion effectively sign-of-sign at fixed norm. Under Lion (not yet tested on this stack) try max_norm=1.0 or 2.0; orthogonal to alphonse #2270's AdamW result.
18. **Cyclic LR through SWA window** (Izmailov's original recommendation) — if skip-SWALR (#2484) wins or ties, the natural follow-up is cosine-restarts during averaging window.

## Mechanism-axis coverage

### ✓ Landed (10 axes)
1. Huber β=1.0 (#1452), 2. Per-sample Re-weight (#1586), 3. FiLM (#1585), 4. Grad-clip 1.0 (#1731), 5. Grad-clip 0.5 (#1831), 6. Kendall σ (#1906), 7. RFF σ=1.0 (#2082), 8. Huber β=0.3 (#1757), 9. Lion lr=3e-4 wd=3e-4 (#2063), 10. **RFF σ=0.5 (#2168)** ← CURRENT

### 🔬 In-flight (Wave 12 — post-Loop-19)
- **hybrid_kendall_lr push {1e-3, 2e-3} on merged hybrid (#2604 fern)** — test σ-spread ceiling; β–σ coupling from #2540 confirms σ-spread is the lever; per-split caveat from #2500
- **max_norm sweep {1.0, 2.0} on merged hybrid (#2606 thorfinn)** — relax saturating clip (4 independent confirmations of clip_fraction=1.0)
- **film_mid_dim sweep {32, 128} on merged hybrid (#2616 alphonse NEW)** — only untouched single-CLI capacity axis; targets geom_camber_rc per-split asymmetry from #2500
- Lion wd sweep on σ=0.5 {3e-3, 1e-2} (#2390 askeladd, REBASED) — pending API recovery
- n_head ∈ {2, 8} bidirectional sweep at n_hidden=128 on σ=0.5 (#2442 nezuko) — pending API recovery
- swa_lr ∈ {0.05x, 0.5x} sweep on σ=0.5 (#2463 tanjiro) — pending API recovery
- SWA anneal_epochs=1 on σ=0.5 (#2481 edward) — pending API recovery
- Skip-SWALR entirely on σ=0.5 (#2484 frieren) — pending API recovery

### ⭐ Merged (Wave 12 winners)
- hybrid Lion+AdamW for Kendall σ heads (#2311 fern, 2026-05-13 19:10): val 45.2181 / test 38.7661, −1.20% val / −2.26% test. σ-spread 0→0.475. Structural σ-collapse fix.

### ✗ Closed (30+ axes)
- drop-grad-clip-on-Lion (#2347, 2026-05-13 16:00): max_norm=0.5 is right setting, 4 banked findings inc. 4th σ-collapse confirmation
- slice_num=96 (#2378, 2026-05-13 16:30): NOT a capacity axis (+5K params not +310K), regressed +4.19 val, 5 banked findings inc. 5th σ-collapse confirmation
- max_norm relaxation {0.75, 1.0} (#2270, 2026-05-13 16:35): clip_fraction stays ≥0.995 even at 2× cap under β=0.3-Huber, 5 banked findings inc. AdamW-equilibrium log_σ targets for #2443
- T_max ∈ {10, 12} cosine compression (#2342, 2026-05-13 17:00): clean regression +5.5% to +8.9% val; **6 banked findings inc. SWALR-overrides-cosine pathology — the most valuable mechanistic finding of Wave 12**; opens new swa_lr direction axis via #2463
- swa_start_frac ∈ {0.5, 0.6} (#2429 edward, 2026-05-13 17:30): monotonically worse with smaller frac; **3rd SWALR-overrides-cosine confirmation**; geom_camber_rc identified as load-bearing OOD split (~2× other splits)
- Lion + linear warmup 3 epochs (#2363 frieren, 2026-05-13 17:30): epoch-1 val WORSE with warmup → Adam→Lion mental model refuted; **Lion has no chaotic init phase**; clip_fraction invariant to lr; 9th σ-collapse confirmation
- Kendall log_σ init at AdamW-equilibrium on σ=0.5 Lion (#2443 alphonse, 2026-05-13 18:10): **cleanest σ-collapse mechanism finding on wave — init alone prevents collapse**; spread 0.000→0.478, test −0.40, but mean drift inflates eff_w 3× → val +0.61; motivates #2500 mean anchor
- RFF σ=0.1 + σ=0.25 seed-1 bracket (#2407 thorfinn, 2026-05-13 18:45): σ axis exhausted on val; σ=0.25 test win refuted as seed-0 outlier; **σ=0.1 = regularizer NOT OOD-prior** (driver = single_in_dist, not OOD-geom); 2nd clip_fraction≈0.99 confirmation; 10th + 11th σ-collapse confirmations; motivates #2512 multi-scale RFF
- Multi-scale RFF 8×σ=0.5 + 8×σ=0.1 (#2512 thorfinn, 2026-05-13 20:15): does NOT compose on hybrid stack (val +2.66%, test +1.20%); **σ axis decisively closed at σ=0.5** (10 configs evaluated since Lion adoption); banked: σ=0.1 regularizer effect was σ-collapse-conditional — #2311 substitutes for it; 3rd–4th clip_fraction=1.0 confirmations; motivates #2606 max_norm sweep
- Huber β sweep {0.5, 1.0} on hybrid (#2540 fern, 2026-05-13 20:45): monotonic regression both arms (β=0.5 +3.54% val, β=1.0 +10.12% val); **β=0.3 robust to hybrid optimizer change**; **NEW β–σ coupling mechanism** (β↑ collapses σ-spread); 12th clip_fraction confirmation; refuted grad-norm prediction (smooth-L1 slope 1/β); motivates #2604 hybrid_kendall_lr push
- Anchor mean(log_σ) + AdamW-eq init on hybrid (#2500 alphonse, 2026-05-13 22:15): both arms regress (λ=1 val 45.90, λ=5 val 47.28); mean-drift fix invalid post-#2311; 4 banked findings inc. **NEW per-split asymmetry: spread expansion hurts geom_camber_rc but helps other splits** (important context for #2604), AdamW-eq init expands spread to 0.70, natural hybrid equilibrium (~mean −2.0) outperforms target −1.4; motivates #2616 film_mid_dim sweep
