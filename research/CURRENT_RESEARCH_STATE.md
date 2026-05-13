# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~19:30 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| **#2400** | **n_layers=3 on n_head=2+slice32+Lion+MAE** | **43.14** | **36.95** |
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

**Current compound:** Fourier + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=2** + slice_num=32 + **n_layers=3** + surf_weight=10

## Active experiments (8/8 students assigned)

| PR | Student | Config | Compound | Status |
|----|---------|--------|----------|--------|
| **TBD** | **frieren** | **cosine T_max via --epochs (34/28)** | NEW | Assigning now |
| #2489 | edward | wd=3e-4/wd=1e-3 stack on n_layers=3 | NEW | WIP |
| #2491 | fern | sw=5/sw=3 stack on n_layers=3 | NEW | WIP |
| #2482 | askeladd | n_layers=2/n_layers=1 (speed-dividend extension) | NEW | WIP (pod rate-limit blocked) |
| #2483 | tanjiro | n_head=1 + n_layers=3/2 cross-axis | NEW | WIP |
| #2470 | alphonse | sw=15/sw=20 on n_head=1 (sw-reversal test) | OLD (n_head=1) | WIP (pod rate-limit blocked) |
| #2448 | thorfinn | Lion wd=3e-4/1e-3 on n_head=1 | OLD (n_head=1) | WIP |
| #2446 | nezuko | mlp_ratio=4/1 on n_head=1 | OLD (n_head=1) | WIP (pod rate-limit blocked) |

**Wave split:** 5 students testing the NEW n_head=2+slice32+n_layers=3 compound (baseline=43.14); 3 students completing isolated-axis data on the OLD n_head=1+n_layers=5 compound (vs 46.67). The OLD-compound runs still produce useful axis-isolation data for the appendix narrative.

## Infrastructure note
Multiple student pods (alphonse/nezuko/askeladd) hitting `GraphQL: API rate limit already exceeded` and unable to poll their assigned PRs (`No assigned PRs or issues` despite PRs being open and labeled). Root cause of repeated stale_wip pattern is rate-limit pressure on the student polling mechanism, NOT individual student-agent failure. Pods retry every 15s and recover when rate limits reset.

## Key findings this cycle (26–32)

26. **sw signal REVERSES at n_head=1 (#2416):** sw=5 synergistic at n_head=2 (−2.54 val); at n_head=1 sw=5 REGRESSES +1.84 val (+3.94%). Mechanism: n_head=1 (dim=128) head must allocate surface capacity; sw=5 strips it with no backup.
27. **lr does NOT transfer across slice_num resolutions (#2376):** lr=1.5e-4 regresses +1.8% on slice32 (opposite of slice64 where lr=1.5e-4 won).
28. **SPEED-DIVIDEND CONFIRMED MASSIVELY (#2400 — new best):** n_layers=3 → 53.1 s/ep → 34 epochs in 30 min → val=43.14/test=36.95 (−7.6%/−9.2%). Monotonic: n_layers=3 < n_layers=4 < n_layers=5. All 4 test splits improve 7–13%. Val still descending at cap.
29. **β2=0.99 LOCKED at n_head=1 (#2438):** Plateau-at-canonical across n_head ∈ {1,2,4}: n_head=4 → β2=0.995, n_head=2/1 → β2=0.99. Main−EMA gap monotonic in β2 (9.14/6.66/7.10 for 0.95/0.99/0.995). β2 axis closed.
30. **slice_num × n_head SUBSTITUTIVE at high per-head dim (#2430):** slice_num=16 won on n_head=2 (#2337) but anti-stacks on n_head=1 (+8.85% val, +17.7% vs new baseline). Mechanism: per-head dim × slice_num jointly determine spatial bandwidth. At dim=64 head capacity bottlenecks; at dim=128 tokens bottleneck.
31. **lr=1.25e-4 test-only effect on n_head=1 (#2419):** Val flat (+0.42%, within noise), test wins by −2.56% vs old #2338. Val/test divergence at n_head=1. Doesn't beat new baseline.
32. **slice_num INVERTS with depth (#2490):** slice_num=16 won at n_layers=5 (#2337) but anti-stacks at n_layers=3 (slice16 +3.08 val, slice8 +3.53 val). Mechanism (frieren): depth and tokens are not independent — at deeper stacks the network compensates for coarser tokens via stacked representation; at shallow depth finer tokens preserve input bandwidth. Per-head-dim story does NOT survive depth change. **slice_num=32 is now the local optimum at n_layers=3.**

## Priority for current wave

**Highest priority — NEW COMPOUND (n_head=2+slice32+n_layers=3, baseline=43.14):**
- **n_layers=2/n_layers=1 (askeladd #2482):** speed-dividend extension; val still descending at n_layers=3 cap
- **n_head=1 + n_layers=3/2 (tanjiro #2483):** cross-axis test of n_head=1 advantage on shallow depth
- **wd=3e-4/1e-3 (edward #2489):** stack monotonic wd signal (#2356 won −5.4% on n_layers=5)
- **sw=5/sw=3 (fern #2491):** stack #2335 synergistic interaction at shallower depth
- **cosine T_max via --epochs (frieren, assigning):** match scheduler T_max to realized epoch count — val still descending at cap suggests lr over-cooling because default --epochs=50 sets T_max=50 but training only reaches ~34 epochs. Direct test of scheduler/realization mismatch.

**Ongoing OLD-compound isolated-axis data (vs 46.67):**
- **#2470 alphonse:** sw=15/20 — sw-reversal mechanism test (predicts higher sw helps at n_head=1)
- **#2448 thorfinn:** wd=3e-4/1e-3 — direct stack of monotonic wd at n_head=1
- **#2446 nezuko:** mlp_ratio=4/1 — FFN-vs-attention compute rebalance

## Closed experiments this cycle

- **#2490 (frieren):** slice_num=16/8 stack on n_layers=3 — CLOSED. slice signal inverts with depth; slice32 is the local optimum at n_layers=3.
- **#2400 (askeladd):** n_layers=3 — **MERGED** val=43.14, test=36.95. MASSIVE WIN via speed-dividend.
- **#2438 (fern):** β2 sweep n_head=1 — CLOSED. β2=0.99 locked across n_head ∈ {1,2,4}. Plateau-at-canonical.
- **#2430 (frieren):** slice_num=16 on n_head=1 — CLOSED. Substitutive at high per-head dim.
- **#2419 (edward):** lr sweep n_head=1 — CLOSED. lr=1.25e-4 test-only win, doesn't beat new baseline.
- **#2376 (tanjiro):** lr=1.5e-4 on slice32 — CLOSED. Regresses +1.8%.
- **#2416 (alphonse):** n_head=1+sw=5 — CLOSED. sw reversal at n_head=1.
- **#2356 (thorfinn):** wd=3e-4 on n_head=2+slice32 — CLOSED. Monotonic wd confirmed; signal transferred.
- **#2372 (nezuko):** sw=2/3 on n_head=2+slice32 — CLOSED. U-curve at sw=3.
- **#2295 (fern):** EMA decay sweep — CLOSED. ema=0.99 confirmed optimal.
- **#2337 (frieren):** slice_num=16 on n_head=2 — CLOSED. Monotonic trend confirmed.

## Potential next directions (post-current-wave)

- **Compound winners on n_layers=3:** if edward/fern stacks succeed, the next wave compounds them (e.g. wd+sw) on the n_layers=3 baseline
- **n_layers=2 follow-ups:** if askeladd #2482 finds new best, optimize axes on that even shallower compound
- **n_head=1 + n_layers=3 follow-ups:** if tanjiro #2483 finds new best, all axes need re-testing at the dim=128 + 34-epoch regime
- **Extended budget** — every win still descending at 30-min cap; if timeout relaxed, all current best configs would extend
- **Width axis** — n_hidden=160 was unviable at n_layers=5; at n_layers=3 the model is much smaller and width may now be affordable (requires Config edit — no --n_hidden flag currently)
- **Cosine T_max tuning** — assigned to frieren this cycle (--epochs 34 vs 28)
- **Loss formulation revisit** — MAE+sw=5 won; explore alternative aggregations (median, Huber+MAE blend) on new compound
- **n_layers=4 + slice24 sweet-spot probe** — frieren's observation: depth-slice trade. Mid-depth, mid-slice may find a balanced point (would require slice values beyond power-of-2 if not already supported).
