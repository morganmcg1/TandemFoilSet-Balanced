# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~22:55 UTC
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
| **TBD** | **edward** | **lr=8e-5/1.5e-4 sweep on n_layers=3+wd=3e-4** | NEW | Assigning now |
| #2618 | frieren | --epochs 80/100 (T_max LARGER) on new compound | NEW | WIP |
| #2563 | thorfinn | n_head=4/n_head=8 sweep on n_layers=3+wd=3e-4 | NEW | WIP (stale) |
| #2491 | fern | sw=5/sw=3 stack on n_layers=3 | NEW | WIP (pod rate-limit blocked) |
| #2482 | askeladd | n_layers=2/n_layers=1 (speed-dividend extension) | NEW | WIP (pod rate-limit blocked) |
| #2483 | tanjiro | n_head=1 + n_layers=3/2 cross-axis | NEW | WIP (pod rate-limit blocked) |
| #2470 | alphonse | sw=15/sw=20 on n_head=1 (sw-reversal test) | OLD (n_head=1) | WIP (pod rate-limit blocked) |
| #2446 | nezuko | mlp_ratio=4/1 on n_head=1 | OLD (n_head=1) | WIP (pod rate-limit blocked) |

**Wave split:** 5 students testing the NEW n_head=2+slice32+n_layers=3 compound (baseline now 42.00 after #2489 merge); 3 students completing isolated-axis data on the OLD n_head=1+n_layers=5 compound (vs 46.67).

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

## Priority for current wave

**Highest priority — NEW COMPOUND (n_head=2+slice32+n_layers=3+wd=3e-4, baseline=42.00):**
- **lr=8e-5/1.5e-4 (edward, assigning):** does wd=3e-4 stabilization permit higher lr for more effective steps?
- **--epochs 80/100 LARGER T_max (frieren #2618):** extends monotonic 'less annealing → better' direction UP
- **n_head=4/8 (thorfinn #2563):** does more heads help on the depth-3 compound where wd=3e-4 won?
- **n_layers=2/n_layers=1 (askeladd #2482):** speed-dividend extension; val still descending at cap
- **n_head=1 + n_layers=3/2 (tanjiro #2483):** cross-axis test of n_head=1 at shallow depth
- **sw=5/sw=3 (fern #2491):** stack sw synergistic interaction at n_layers=3+wd=3e-4

**Ongoing OLD-compound isolated-axis data (vs 46.67):**
- **#2470 alphonse:** sw=15/20 — sw-reversal mechanism test at n_head=1
- **#2446 nezuko:** mlp_ratio=4/1 — FFN-vs-attention compute rebalance

## Closed experiments this cycle

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
