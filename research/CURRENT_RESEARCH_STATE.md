# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-14 ~05:20 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| **#2489** | **wd=3e-4 on n_head=2+slice32+n_layers=3+Lion+MAE** | **42.00** | **35.96** |
| #2400 | n_layers=3 on n_head=2+slice32+Lion+MAE | 43.14 | 36.95 |
| #2338 | n_head=1 on slice32+Lion+MAE+lr=1e-4 | 46.67 | 40.69 |
| #2335 | slice32+sw=5 on n_head=2 | 48.57 | 41.48 |
| #2218 | slice_num=32 on n_head=2+Lion+MAE+lr=1e-4 | 49.86 | 42.19 |
| #2210 | sw=5 on n_head=2+Lion+MAE+lr=1e-4 | 50.91 | 43.68 |
| #2069 | n_head=2 on Lion+MAE+lr=1e-4 | 51.11 | 44.18 |
| #1932 | Lion lr=2e-4 (wd=1e-4) on Lion+MAE | 55.41 | 47.90 |
| #1825 | MAE (L1) loss on Lion+EMA | 56.58 | 48.82 |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |

**Current compound:** Fourier + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, **wd=3e-4**) + **n_head=2** + slice_num=32 + **n_layers=3** + surf_weight=10

## Active experiments (8/8 students assigned)

| PR | Student | Config | Compound | Status |
|----|---------|--------|----------|--------|
| **#2758** | **edward** | **clip_grad_norm=0.5/2.0 sweep on new compound** | NEW | WIP (just assigned) |
| **#2784** | **thorfinn** | **fourier_max_freq=16/48 sweep at L=6 on new compound** | NEW | WIP (just assigned) |
| **#2789** | **frieren** | **fourier_min_freq=0.5/2.0 sweep at L=6** | NEW | WIP (just assigned) |
| #2491 | fern | sw=5/sw=3 stack on n_layers=3 | NEW | WIP (pod rate-limit blocked) |
| #2482 | askeladd | n_layers=2/n_layers=1 (speed-dividend extension) | NEW | WIP (pod rate-limit blocked) |
| #2483 | tanjiro | n_head=1 + n_layers=3/2 cross-axis | NEW | WIP (pod rate-limit blocked) |
| #2470 | alphonse | sw=15/sw=20 on n_head=1 (sw-reversal test) | OLD (n_head=1) | WIP (pod rate-limit blocked) |
| #2446 | nezuko | mlp_ratio=4/1 on n_head=1 | OLD (n_head=1) | WIP (pod rate-limit blocked) |

**Wave split:** 6 students testing the NEW n_head=2+slice32+n_layers=3 compound (baseline now 42.00 after #2489 merge); 2 students completing isolated-axis data on the OLD n_head=1+n_layers=5 compound (vs 46.67). Note: fern's #2491 (sw=5/3 lower) still pending — will bracket the sw axis on the new compound when returned.

## Infrastructure note
Multiple student pods (alphonse/nezuko/askeladd/tanjiro) hitting `GraphQL: API rate limit already exceeded` and unable to poll their assigned PRs. Root cause: fleet-wide rate-limit pressure on the student polling mechanism. Pods retry every 15s and recover when rate limits reset. Not individual student-agent failure.

## Key findings this cycle (26–34)

26. **sw signal REVERSES at n_head=1 (#2416):** sw=5 synergistic at n_head=2 (−2.54 val); at n_head=1 sw=5 REGRESSES +1.84 val (+3.94%). Mechanism: n_head=1 (dim=128) head must allocate surface capacity; sw=5 strips it with no backup.
27. **lr does NOT transfer across slice_num resolutions (#2376):** lr=1.5e-4 regresses +1.8% on slice32 (opposite of slice64 where lr=1.5e-4 won).
28. **SPEED-DIVIDEND CONFIRMED MASSIVELY (#2400):** n_layers=3 → 53.1 s/ep → 34 epochs in 30 min → val=43.14/test=36.95 (−7.6%/−9.2%). Monotonic: n_layers=3 < n_layers=4 < n_layers=5. All 4 test splits improve 7–13%. Val still descending at cap.
29. **β2=0.99 LOCKED at n_head=1 (#2438):** Plateau-at-canonical across n_head ∈ {1,2,4}: n_head=4 → β2=0.995, n_head=2/1 → β2=0.99. β2 axis closed.
30. **slice_num × n_head SUBSTITUTIVE at high per-head dim (#2430):** slice_num=16 won on n_head=2 (#2337) but anti-stacks on n_head=1 (+8.85% val, +17.7% vs new baseline). Per-head-dim × slice_num jointly determine spatial bandwidth.
31. **lr=1.25e-4 test-only effect on n_head=1 (#2419):** Val flat (+0.42%, within noise), test wins by −2.56% vs old #2338. Val/test divergence at n_head=1. Doesn't beat new baseline.
32. **slice_num INVERTS with depth (#2490):** slice_num=16 won at n_layers=5 (#2337) but anti-stacks at n_layers=3. Mechanism: depth and tokens not independent. **slice_num=32 confirmed as local optimum at n_layers=3.**
33. **wd=3e-4 signal transfers depth-independently (#2489 — NEW BEST):** val=42.0040/test=35.9573 (−2.64%/−2.69% vs #2400, all 4 splits). wd=3e-4 operating point confirmed at n_layers=3; wd=1e-3 over-regularizes at this shallower depth. Regularization benefit is independent of network depth in this regime.
34. **wd INVERTS at n_head=1 (#2448):** wd=3e-4 won at n_head=2 (#2356/#2489) but +0.51 val at n_head=1. wd is n_head-specific — n_head=2 spreads attention across 2 heads (variance/dilution that wd compensates), n_head=1 is single-head and already near wd=1e-4 optimum. At wd=3e-4 the n_head=1 and n_head=2 paths meet at val=47.18 → wd and n_head are substitutive regularizers. Pairs with finding 30 (slice × n_head substitutive at high per-head dim).
35. **REGULARIZATION SATURATION at n_layers=3 (#2551):** dropout=0.25 (+6.50% val) and 0.30 (+3.95% val) both regress on the wd=3e-4 compound. All 4 splits regress uniformly — no OOD-specific dropout effect. wd=3e-4 + dropout=0.20 + BF16 + EMA(0.99) is at the Pareto frontier; adding more of any regularizer is substitutive, not additive. Capacity floor at 3 attention blocks: each block carries more representational load than at n_layers=5, so dropout=0.30 strips too many activations to compensate.
36. **COSINE T_MAX INVERTS direction (#2542):** matched T_max=34 (+10.14% val) and T_max=44 (+9.07% val) both regress. Monotonic ordering {Arm 1 < Arm 2 < baseline T_max=50}: less annealing strictly better. Arm 1 val *went up* at the final epoch as lr→0 (over-annealing collapse); baseline at lr=0.485×lr_init at cap is in a sweet spot. Lion + lr=1e-4 brittle at very low lr — fine-tuning regime is harmful. 'Val still descending at cap' is load-bearing, not a missed opportunity. **Next test: T_max LARGER than 50 (--epochs 80/100) — extends monotonic direction UP.**
37. **NO SPEED DIVIDEND ON BATCH_SIZE AXIS (#2587):** bs ∈ {2, 4, 8} produces 51-54 s/ep (essentially batch_size-INVARIANT). Per-step GPU compute is NOT the bottleneck — data loading + padding to N_max=242K + val pass dominate. Both arms regress (bs=8 +15.1%, bs=2 +9.6%). GPU is near-saturated at bs=4 (97.8% reserved memory). bs=16 would likely OOM. Step count drives performance, with Lion sign-update + small dataset compounding gradient variance into a noise penalty at bs=2. bs=4 is the noise-step Pareto sweet spot. **Speed-dividend mechanism applies to architecture-driven per-step compute reduction (n_layers, slice_num) but NOT to batch_size axis where data loading + padding dominate.** Padding-waste reduction (length-bucketed sampler) is the real lever for budget-extension — requires data-loader code, not a CLI flag.
38. **PER-HEAD DIM=64 SLICE-ATTENTION SWEET SPOT (#2563):** Monotonic {n_head=2 (dim_head=64) < n_head=4 (dim_head=32) < n_head=8 (dim_head=16)} at n_layers=3+wd=3e-4. n_head=4: +4.58% val. n_head=8: +11.73% val. Below dim_head=32 the soft virtual-token mechanism loses rank — single_in_dist worst-hit (+15.72% at n_head=8). Architecture cost is NON-zero: n_head=4 +11.7% s/ep, n_head=8 +35.6% s/ep. Speed-dividend at higher per-head dim wins. wd × n_head ruled out as explanation (regression magnitude dwarfs any wd correction). **n_head=2 + slice_num=32 + n_layers=3 are mutually Pareto-optimal at our budget — three independent axes converge to a single operating point.** n_head=2 optimum is depth-independent.
39. **DROPOUT DIRECTION CLOSED BOTH SIDES; wd × dropout COMPLEMENTARY NOT SUBSTITUTIVE (#2645):** dropout=0.10 (+2.85% val) and 0.05 (+0.89% val) both regress vs baseline d=0.20. Combined with #2551 closing upper direction (0.25/0.30 regress), **dropout=0.20 is locked as local optimum**. Critically REFUTES the substitutive-regularizer prior: lowering dropout did NOT free capacity under wd=3e-4. **wd × dropout are COMPLEMENTARY** — distinct interaction sign from wd × n_head substitutive (finding 34) and slice × n_head substitutive (finding 30). Non-monotonic 0.10 < 0.05 hints mid-strength dropout is worst case. **Different regularizer pairs have different interaction signs — paper-grade finding on regularizer-interaction topology.** In-dist single_in_dist test IMPROVES on both arms while OOD splits regress.
40. **lr=1e-4 LOCKED AS LOCAL OPTIMUM ON NEW COMPOUND; LR SURFACE NARROW (#2641):** lr=8e-5 (+2.58% val) and lr=1.5e-4 (+1.05% val) both regress. wd=3e-4 stabilization did NOT shift lr optimum. **wd and lr are independent axes** under Lion+MAE. lr surface is narrow: ±0.5×–1.5× regresses by 1–2.6 points. Interesting regime-trade: Arm 2 lr=1.5e-4 IMPROVES single_in_dist test (−8.9%) but worsens re_rand (+8.7%) — higher lr trades in-dist fidelity for OOD robustness. **Reinforces finding 37: step count is the bottleneck** (val still descending at cap across all three lrs). Only architecture or padding-waste reduction can free more steps.
41. **SURF_WEIGHT UPPER DIRECTION CLOSED; sw × SPLIT-TYPE INTERACTION (#2707):** sw=15 regresses +2.63 val / +2.12 test on new compound. **Strong asymmetric per-split pattern:** single_in_dist IMPROVES (−2.32%) while ALL OOD splits regress, **re_rand worst at +13.81%**. Mechanism: over-emphasizing surface loss starves volume field; OOD extrapolation depends on volume context anchoring surface prediction. Both geom_camber splits regress, REFUTING original OOD-improvement hypothesis. Surface emphasis is NOT a substitute for balanced surface+volume signal of MAE. **For higher-surf direction would need pairing with volume-loss-floor or per-channel reweighting** (different hypothesis, requires code). With fern's pending #2491 sw=5/3 lower test, sw axis is bracketed on new compound. **The sw × split-type interaction is paper-grade appendix material.**
45. **COSINE T_MAX=50 IS U-SHAPE MINIMUM, BOTH DIRECTIONS CLOSED (#2618):** T_max=80 (val=46.22, +10.0%) and T_max=100 (val=43.64, +3.9%) both regress. Full U-curve: T_max=34(46.26) < T_max=44(45.81) < T_max=50(42.00) > T_max=80(46.22) > T_max=100(43.64). Mechanism: at 30-min cap all variants get ~33 epochs; T_max=50 reaches lr≈0.485×lr_init (polish phase); T_max>50 keeps lr too high → no polish → slower val descent that never catches up. **"Val still descending" is misleading — it's slower, not better.** All splits regress OOD-worst. Scheduler CLI axis fully closed. Linear-warmup+cosine-decay-to-zero (code edit) parked for next tier.
44. **FOURIER_L=6 LOCKED, BOTH DIRECTIONS CLOSED (#2742):** L=4 val=42.57 (+0.57 worse)/test=35.73 (−0.23 ≈ noise); L=8 val=43.59 (+1.59)/test=37.23 (+1.27) — clearly regressive. Aliasing at max_freq=32 on standardized coords explains L=8 failure: top octave near Nyquist for typical mesh spacing. L=4 is within noise (channel count unchanged at 0.45M params). **Key finding: L vs max_freq are distinct axes.** The natural follow-up is max_freq=16/48 at L=6 to isolate aliasing from channel count. Pre-existing bug noted: test_geom_camber_cruise normalized-space loss logs as NaN/Inf in W&B summaries (denormalized MAE correct).
43. **EMA DECAY LOCKED AT 0.99, HIGHER DIRECTION CLOSED (#2729):** ema_decay=0.995 (+1.77 val worse, +1.15 test worse) and 0.999 (val −0.31 worse, test +0.15 ≈ noise) both fail to beat baseline. At the 30-min step budget where val is still descending at cap, higher decay produces *EMA lag* rather than deeper smoothing — the EMA averaging window exceeds remaining training convergence horizon. Main-EMA gap dynamics confirm: 0.995 widens gap (+4.32) = sluggish; 0.999 narrows gap (+1.92) = can't keep up. Arm 2 per-split: trades single_in_dist gain (−3.14) for re_rand regression (+2.39) — echoes sw×split-type pattern (finding 41). **ema_decay=0.99 was already optimally tuned for this budget.** EMA bias correction (handles first-epoch init-anchoring at 0.999) noted as future code axis if budget extends.
42. **MAE DOMINATES HUBER ON MATURE COMPOUND; LOSS-FORMULATION AXIS CLOSED (#2708):** Both Huber arms regress significantly: δ=0.5 (+4.26 val, +3.79 test), δ=1.0 (+3.57 val, +2.96 test). All splits regress on both arms. δ=1.0 > δ=0.5 (closer to MAE = better) — monotonic. **Early #1825 MAE-vs-Huber result REPLICATES on mature compound** — Lion + wd=3e-4 + n_layers=3 + EMA + Fourier do NOT change the loss landscape. Per-node uniform MAE is essential. Huber HURTS OOD specifically (re_rand worst-hit +5-6 pts). Step-budget effect: Huber's slower per-epoch descent (~−0.3 to −0.7/ep) cannot catch up at 30-min cap. **Pairs with finding 41 to define loss-balance landscape:** surface weighting hurts OOD, curvature smoothing hurts everywhere. **Loss-formulation axis closed without code edits** (Huber+MAE blend requires train.py modification). Numerical bug: huber_loss(reduction='none')+BF16 returns nan on test_geom_camber_cruise.

## Priority for current wave

**Highest priority — NEW COMPOUND (n_head=2+slice32+n_layers=3+wd=3e-4, baseline=42.00):**
- **--epochs 80/100 LARGER T_max (frieren #2618):** extends monotonic 'less annealing → better' direction UP
- **n_layers=2/n_layers=1 (askeladd #2482):** speed-dividend extension; val still descending at cap
- **n_head=1 + n_layers=3/2 (tanjiro #2483):** cross-axis test of n_head=1 at shallow depth
- **sw=5/sw=3 (fern #2491):** stack sw synergistic interaction at n_layers=3+wd=3e-4
- **edward #2758:** clip_grad_norm=0.5/2.0 sweep on new compound — last untested CLI-only optimizer axis; Lion+sign-grad makes clipping subtle, both directions informative
- **thorfinn #2784:** fourier_max_freq=16/48 sweep at L=6 — direct aliasing test; max_freq=32 may add noise above Nyquist on standardized coords; max_freq=16 drops potentially-aliased octave; max_freq=48 extends into clearly-aliased band
- **frieren #2789:** fourier_min_freq=0.5/2.0 sweep at L=6 — low-frequency encoding band test; min_freq=1.0 default may miss global-scale features (0.5) or pack unnecessary long-period components (test 2.0). Parallel to thorfinn's max_freq sweep — together these bracket the encoding band

**Plateau Protocol status:** CLI-only axes nearly exhausted — wd (3e-4), dropout (0.2), n_head (2), slice_num (32), n_layers (3), batch_size (4), cosine T_max (50, U-shape confirmed), lr (1e-4), ema_decay (0.99), fourier_L (6), clip_grad_norm (in-flight #2758), fourier_max_freq (in-flight #2784), fourier_min_freq (in-flight #2789). After these 3 return: **escalate to code-edit tier** — linear-warmup+cosine-decay scheduler, width axis (n_hidden), length-bucketed sampler (data-loader), EMA bias correction. **6+ consecutive CLI-axis negatives signal code-edit tier readiness.** The substitutive-vs-complementary regularizer-interaction framework (findings 30, 34, 35, 39) and T_max U-shape mechanism (finding 45) are paper-grade findings.

**Ongoing OLD-compound isolated-axis data (vs 46.67):**
- **#2470 alphonse:** sw=15/20 — sw-reversal mechanism test at n_head=1
- **#2446 nezuko:** mlp_ratio=4/1 — FFN-vs-attention compute rebalance

## Closed experiments this cycle

- **#2618 (frieren):** cosine T_max=80/100 extension on new compound — CLOSED. T_max=50 confirmed U-shape minimum. Scheduler CLI axis fully closed. Finding 45. Frieren reassigned to fourier_min_freq sweep (PR #2789).
- **#2742 (thorfinn):** fourier_L=4/L=8 sweep on new compound — CLOSED. L=6 confirmed. L=4 within-noise; L=8 aliasing regression. Finding 44. Thorfinn reassigned to max_freq sweep (PR #2784).
- **#2729 (edward):** ema_decay=0.995/0.999 higher direction on new compound — CLOSED. ema=0.99 locked. EMA lag mechanism confirmed; higher decay = lagged EMA at this step budget, not deeper smoothing. Arm 2 (0.999) trades single_in_dist for re_rand regression. Finding 43.
- **#2708 (thorfinn):** Huber loss δ=0.5/1.0 vs MAE on n_layers=3+wd=3e-4 — CLOSED. MAE dominates on mature compound, early #1825 replicates. Loss-formulation axis closed. Finding 42.
- **#2707 (edward):** surf_weight=15 UPPER direction on n_layers=3+wd=3e-4 — CLOSED. sw upper closed. Strong sw × split-type asymmetric interaction (single_in_dist improves, all OOD regress, re_rand worst +13.81%). Finding 41. Paper-grade appendix.
- **#2645 (thorfinn):** dropout=0.10/0.05 lower-direction on n_layers=3+wd=3e-4 — CLOSED. Dropout direction closed both sides; dropout=0.20 locked. wd × dropout COMPLEMENTARY not substitutive. Finding 39.
- **#2641 (edward):** lr=8e-5/1.5e-4 on n_layers=3+wd=3e-4 — CLOSED. lr=1e-4 confirmed local optimum; lr surface narrow; wd and lr independent. Finding 40.
- **#2563 (thorfinn):** n_head=4/8 sweep on n_layers=3+wd=3e-4 — CLOSED. Per-head dim=64 is sweet spot; n_head=2 optimum depth-independent. Finding 38.
- **#2587 (edward):** batch_size=8/2 sweep on n_layers=3+wd=3e-4 — CLOSED. No speed dividend on bs axis; GPU near-saturated. Finding 37.
- **#2542 (frieren):** cosine T_max=34/44 — CLOSED. T_max inverts direction; less annealing is better. Finding 36.
- **#2551 (edward):** dropout=0.25/0.30 stack on n_layers=3+wd=3e-4 — CLOSED. Regularization saturation; finding 35.
- **#2448 (thorfinn):** wd=3e-4/1e-3 on n_head=1 — CLOSED. wd inverts at n_head=1; substitutive crossover with n_head=2 at wd=3e-4.
- **#2489 (edward):** wd=3e-4 stack on n_layers=3 — **MERGED** val=42.00, test=35.96. NEW BEST.
- **#2490 (frieren):** slice_num=16/8 stack on n_layers=3 — CLOSED. slice signal inverts with depth; slice32 is the local optimum.
- **#2400 (askeladd):** n_layers=3 — **MERGED** val=43.14, test=36.95. MASSIVE WIN via speed-dividend.
- **#2438 (fern):** β2 sweep n_head=1 — CLOSED. β2=0.99 locked.
- **#2430 (frieren):** slice_num=16 on n_head=1 — CLOSED. Substitutive at high per-head dim.
- **#2419 (edward):** lr sweep n_head=1 — CLOSED. lr=1.25e-4 test-only win.
- **#2376 (tanjiro):** lr=1.5e-4 on slice32 — CLOSED. Regresses.
- **#2416 (alphonse):** n_head=1+sw=5 — CLOSED. sw reversal at n_head=1.
- **#2356 (thorfinn):** wd=3e-4 on n_head=2+slice32 — CLOSED. Monotonic wd confirmed.
- **#2372 (nezuko):** sw=2/3 on n_head=2+slice32 — CLOSED. U-curve at sw=3.
- **#2295 (fern):** EMA decay sweep — CLOSED. ema=0.99 confirmed optimal.
- **#2337 (frieren):** slice_num=16 on n_head=2 — CLOSED. Monotonic trend (doesn't transfer to n_layers=3).

## Potential next directions (post-current-wave)

- **Compound winners stacking:** dropout+wd+sw if all three prove additive; e.g. dropout=0.25+wd=3e-4+sw=5 triple compound
- **n_layers=2 follow-ups:** if askeladd #2482 finds new best, optimize axes on even shallower compound
- **n_head=1 + n_layers=3 follow-ups:** if tanjiro #2483 finds new best, full axis re-test at dim=128+34ep regime
- **Extended budget** — every win still descending at 30-min cap; extended timeout would materially improve all current configs
- **Width axis** — n_hidden=160 now may be affordable at n_layers=3 (no --n_hidden flag, requires Config edit)
- **Loss formulation revisit** — MAE+sw=5 won; explore Huber+MAE blend on new compound
