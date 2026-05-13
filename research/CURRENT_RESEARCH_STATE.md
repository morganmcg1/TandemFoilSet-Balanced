# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~18:35 UTC
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

**MAJOR SHIFT THIS CYCLE:** #2400 (n_layers=3) merged as new best — val=43.14/test=36.95, a −7.6%/−9.2% improvement via the speed-dividend. Compound reverted to n_head=2 (from n_head=1 in #2338). All active n_head=1 experiments (alphonse, edward, fern, frieren, nezuko, thorfinn) are now running on the old-compound territory (vs 46.67). Their results are still scientifically valid (isolating axes on the n_head=1 compound), but the primary optimization target is now the n_head=2+slice32+n_layers=3 compound.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status | Baseline context |
|----|---------|--------|--------|-----------------|
| **TBD** | **askeladd** | **n_layers=2 on n_head=2+slice32: extend speed-dividend below n_layers=3** | **Assigning now** | vs NEW baseline 43.14 |
| **TBD** | **tanjiro** | **n_head=1 on n_head=2+slice32+n_layers=3 compound** | **Assigning now** | vs NEW baseline 43.14 |
| **#2470** | **alphonse** | sw=15/sw=20 on n_head=1: test sw-reversal (sw<10 hurts, sw>10?) | WIP | vs OLD compound (46.67) |
| **#2448** | **thorfinn** | Lion wd=3e-4/wd=1e-3 on n_head=1: stack wd signal | WIP | vs OLD compound (46.67) |
| **#2430** | **frieren** | slice_num=16 on n_head=1 compound: monotonic slice × n_head | WIP | vs OLD compound (46.67) |
| **#2446** | **nezuko** | mlp_ratio=4 vs mlp_ratio=1 on n_head=1 | WIP | vs OLD compound (46.67) |
| **#2419** | **edward** | lr sweep on n_head=1: lr=1.5e-4 vs lr=1.25e-4 | WIP | vs OLD compound (46.67) |
| **#2438** | **fern** | Lion β2 sweep on n_head=1: β2=0.95 vs β2=0.995 | WIP | vs OLD compound (46.67) |

## Key findings this cycle (26–28)

26. **sw signal REVERSES at n_head=1 (#2416):** sw=5 synergistic at n_head=2 (−2.54 val); at n_head=1 sw=5 REGRESSES +1.84 val (+3.94%). Mechanism: n_head=1 (dim=128) head must allocate surface capacity; sw=5 strips it with no backup. Prediction: sw>10 should help → #2470 alphonse.
27. **lr does NOT transfer across slice_num resolutions (#2376):** lr=1.5e-4 regresses +1.8% on slice32 (opposite of slice64 where lr=1.5e-4 won). Slice32 lower-variance gradients do not benefit from higher lr.
28. **SPEED-DIVIDEND CONFIRMED MASSIVELY (#2400 — new best):** n_layers=3 → 53.1 s/ep → 34 epochs in 30 min → val=43.14/test=36.95 (−7.6%/−9.2% vs #2338). Monotonic: n_layers=3 < n_layers=4 < n_layers=5. All 4 test splits improve 7–13%. Val still descending at cap. n_layers=2 is highest-EV next step.

## Priority for current wave

**Highest priority — NEW COMPOUND (n_head=2+slice32+n_layers=3, baseline=43.14):**
- **n_layers=2 (askeladd, assigning now):** Does speed-dividend continue? ~44 epochs projected in 30 min if ~41 s/ep. Model still undertrained at n_layers=3.
- **n_head=1 on n_layers=3 compound (tanjiro, assigning now):** n_head=1 won over n_head=2 by −3.9% at n_layers=5 (#2338). At n_layers=3 it's a different regime; this tests the cross-axis interaction. Potential source of next major win.

**Ongoing n_head=1 compound experiments (vs 46.67 — still useful data):**
- #2470 alphonse: sw=15/20 on n_head=1 — confirms/denies sw-reversal mechanism
- #2448 thorfinn: wd=3e-4/1e-3 on n_head=1 — stack wd signal
- #2430 frieren: slice_num=16 on n_head=1 — slice monotone × n_head compound
- #2446 nezuko: mlp_ratio=4/1 on n_head=1 — FFN-vs-attention compute rebalance
- #2419 edward: lr=1.5e-4/1.25e-4 on n_head=1 — lr sweetspot at n_head=1
- #2438 fern: β2=0.95/0.995 on n_head=1 — complete n_head×β2 story

**Note on n_head=1 experiments:** These experiments are valuable even though #2400 changed the baseline compound back to n_head=2. Their results will inform whether the best axes from n_head=1 (e.g. sw, wd, β2) should also be tested on the new n_head=2+n_layers=3 compound in the next wave.

## Closed experiments this cycle

- **#2400 (askeladd):** n_layers=3 on n_head=2+slice32 — **MERGED** val=43.14, test=36.95. Speed-dividend mechanism confirmed. Monotonic: n_layers=3 < 4 < 5. MASSIVE WIN.
- **#2376 (tanjiro):** lr=1.5e-4 on slice32 — CLOSED. Regresses +1.8%. lr×slice_num interaction: slice32 prefers lr≤1e-4.
- **#2416 (alphonse):** n_head=1+sw=5 — CLOSED. sw reversal at n_head=1: +3.94% regression. Mechanism confirmed; higher sw may help.
- **#2356 (thorfinn):** wd=3e-4 on n_head=2+slice32 — CLOSED (beats old #2218 but loses to #2338). Monotonic wd confirmed. Signal transferred to n_head=1 → #2448.
- **#2372 (nezuko):** sw=2/3 on n_head=2+slice32 — CLOSED (sw=3 beats old #2218). U-curve at sw=3. Transferred to mlp_ratio test (#2446).
- **#2295 (fern):** EMA decay sweep — CLOSED. ema=0.99 confirmed optimal. EMA axis exhausted.
- **#2337 (frieren):** slice_num=16 on n_head=2 — CLOSED (beats #2218, loses to #2338). Monotonic trend confirmed.

## Potential next directions (post-current-wave)

- **n_layers=2** (askeladd) — highest EV; val still descending at n_layers=3 cap, n_layers=2 projects ~44 epochs
- **n_head=1 × n_layers=3 compound** (tanjiro) — cross-axis test; n_head=1 won +3.9% at n_layers=5
- **wd=3e-4 on new n_head=2+n_layers=3 compound** — wd signal showed +5.4% on old compound; may stack
- **sw=5 on new n_head=2+n_layers=3 compound** — sw was synergistic with n_head=2+slice32 (#2335); still valid on new compound
- **n_head=1 + n_layers=2** — if both axes independently win, compound test is the obvious next step
- **Extended budget** — every run descends at 30-min cap; n_layers=3+34 epochs still not converged; highest projected gain if timeout increases
- **Width axis: n_hidden=160/192** — previously regressed at n_layers=5; at n_layers=3 the model is smaller, width expansion may now be viable
- **mlp_ratio on new compound** — nezuko's result at n_head=1 will tell us if mlp_ratio=1 speed-dividend stacks with n_layers speed-dividend (could compound)
