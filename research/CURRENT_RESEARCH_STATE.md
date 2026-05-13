# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~17:05 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| **#2338** | **n_head=1 on slice32+Lion+MAE+lr=1e-4** | **46.67** | **40.69** |
| #2335 | slice32+sw=5 on n_head=2 | 48.57 | 41.48 |
| #2218 | slice_num=32 on n_head=2+Lion+MAE+lr=1e-4 | 49.86 | 42.19 |
| #2210 | sw=5 on n_head=2+Lion+MAE+lr=1e-4 | 50.91 | 43.68 |
| #2069 | n_head=2 on Lion+MAE+lr=1e-4 | 51.11 | 44.18 |
| #1932 | Lion lr=2e-4 (wd=1e-4) on Lion+MAE | 55.41 | 47.90 |
| #1825 | MAE (L1) loss on Lion+EMA | 56.58 | 48.82 |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current compound:** Fourier + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=1** + slice_num=32 + surf_weight=10

**Note:** Every merged win has had val still descending at the 30-min cap. n_head=1 is the fastest configuration yet: 71.1s/ep → 26 epochs in 30 min. The sw=5 interaction at n_head=1 is untested and could provide another ~−1–2 val (assigned to alphonse #2416).

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#2416** | **alphonse** | **n_head=1 + surf_weight=5 interaction: stack both wins on slice32** | **WIP — new** |
| **#2356** | **thorfinn** | **Lion wd sweep on slice_num=32 compound: wd=3e-4 vs wd=3e-5** | **WIP** |
| **#2430** | **frieren** | **slice_num=16 on n_head=1 compound: test if monotonic slice trend stacks** | **WIP — new** |
| **#2446** | **nezuko** | **mlp_ratio sweep on n_head=1: mlp_ratio=4 vs mlp_ratio=1** | **WIP — new** |
| **#2419** | **edward** | **lr sweep on n_head=1 compound: lr=1.5e-4 vs lr=1.25e-4** | **WIP — new** |
| **#2438** | **fern** | **Lion β2 sweep on n_head=1: β2=0.95 vs β2=0.995 — complete n_head×β2 story** | **WIP — new** |
| **#2376** | **tanjiro** | **lr sweep on slice_num=32 compound: lr=1.5e-4 vs lr=1.25e-4** | **WIP — new** |
| **#2400** | **askeladd** | **n_layers reduce on slice_num=32: n_layers=4 vs n_layers=3** | **WIP — new** |

## Closed experiments this round

- **#2337 (frieren):** slice_num=16 on n_head=2+sw=10 — CLOSED (val=48.08/test=41.02). Beats old #2218 baseline but loses to current #2338 (val=46.67). Monotonic trend confirmed: 16 < 32 < 64 < 128. Speed: 75.9s/ep, 24 epochs in 30 min, val still descending. Reassigned to #2430 (slice_num=16 on n_head=1 compound).
- **#2338 (edward):** n_head=1 on slice32+sw10 — **MERGED** val=46.67, test=40.69. Monotonic trend extends to n_head=1 (1 < 2 < 4 < 8). 26 epochs in 31 min (71.1s/ep). All 4 splits improve. Reassigned to #2419 (lr sweep on n_head=1).
- **#2335 (alphonse):** slice32+sw5 interaction on n_head=2 — **MERGED** val=48.57, test=41.48. Synergistic: observed −2.54 val vs additive −1.45 (1.75×). 3/4 splits improve. Reassigned to #2416 (n_head=1+sw5).
- **#2372 (nezuko):** sw=2/sw=3 on n_head=2+slice32 — sw=3 (48.47/41.20) beats #2218 by −2.78% but loses to #2338 (46.67). sw=2 regresses all 4 splits. U-curve at sw=3 minimum on n_head=2+slice32. cruise gain at sw=3 (−7.59%) strongest single-split this round. Closed; reassigned to #2446 (mlp_ratio=4/1 on n_head=1).
- **#2295 (fern):** EMA decay sweep on n_head=2+sw=5 — ema_decay=0.99 confirmed locally optimal. Both directions regress (0.999: 50% crash + 12-pt spread; 0.95: +2.32 val). Main-vs-EMA gap at best epoch identical (5.06/5.00) — sweet spot at ~100-step window (¼ epoch). Closed; reassigned to #2438 (β2 sweep on n_head=1).
- **#2271 (askeladd):** β2=0.995 vs β2=0.999 on n_head=2+slice_num=64 — β2 effect reverses from n_head=4 (#2144). Canonical β2=0.99 confirmed optimal on n_head=2. Main-vs-EMA gap diagnostic: 7.31 (β2=0.995) vs 2.81 (β2=0.999). Closed; reassigned to #2400 (n_layers reduce).
- **#2251 (tanjiro):** lr=2e-4 vs lr=1.5e-4 on n_head=2+slice_num=64 — Arm 2 (lr=1.5e-4, val=50.36, test=42.53) beats #2069 (−0.75/−1.65) but loses to new #2218 baseline; both ran on slice_num=64 (pre-#2218 default). Key signal: lr=1.5e-4 is the optimal lr at slice_num=64. 60% crash rate at lr=2e-4 (instability). Closed; reassigned to #2376 (lr=1.5e-4 vs lr=1.25e-4 on slice_num=32).
- **#2277 (nezuko):** sw=4/sw=3 on n_head=2+slice_num=64 — sw=3 wins vs old #2210 (val=50.23 vs 50.91, −1.34%) but loses to new #2218 (49.86) by +0.7%. Non-monotonic in [3,5]: sw=3 < sw=5 < sw=4. Strong geom_camber_cruise improvement (−5.6%) at lower sw. Closed; reassigned to #2372 (sw=2/sw=3 on slice_num=32).
- **#2218 (alphonse):** slice_num=32 — **MERGED** val=49.86, test=42.19. Monotonic: 32 < 64 < 128. Also 23 epochs in budget (vs 20) — speed dividend. Interaction with sw=5 untested (#2335).
- **#2216 (frieren):** Split loss (surf-MAE + vol-Huber/MSE) — all 3 arms regress (+4.5–7.1%). Formulation adds tension without signal benefit. Closed; reassigned to #2337 (slice_num=16).
- **#2183 (edward):** AdamW+EMA+MAE diagnostic — val=73.38 vs baseline 49.86 (+47%). 2×2 mechanism table complete. Lion ~2× EMA. Closed; reassigned to #2338 (n_head=1).
- **#2211 (thorfinn):** OneCycleLR pct_start=0.3/0.1 on n_head=2 — both regress (+5.8%/+6.7% and +15.5%/+15.7% vs baseline #2210). Root cause: OneCycleLR integrated LR ~half of cosine baseline (div_factor=25, start=4e-6); Lion sign-steps at lower integrated LR = fewer effective explorations. Arm 1 also crashed at OneCycle peak (zbwa7pwv, step 3066 diverge). Third consecutive schedule experiment to regress. Closed; reassigned to #2339 (Lion wd sweep).
- **#2086 (thorfinn):** Lion lr=4e-4/3e-4 — both regress vs baseline (+3.83%/+4.10%). **lr-doubling trend saturated at lr=2e-4** after 3 octaves. Flat minimum: 4e-4 only 2 pts worse than 3e-4 despite 4× higher lr. Closed; reassigned to #2211 (OneCycleLR).
- **#2069 (alphonse):** n_head=2 — **MERGED** val=51.11, test=44.18. Biggest win of the round.
- **#2056 (nezuko):** sw=5 at lr=1e-4 beat OLD baseline (54.46 < 55.41) but can't merge onto new n_head=2 code as-is. Closed; reassigned to #2210 (sw=5+sw=7 on n_head=2).
- **#2052 (frieren):** bs=8 — both arms regress (+7.9% / +18.4%). Step-count-limited regime; VRAM near limit. Closed; reassigned to #2216 (split-loss formulation).
- **#2070 (edward):** Lion-no-EMA + AdamW-no-EMA ablation — both regress (+7.06 / +27.05). Mechanism reframed: Lion direction ~75%, EMA ~25%. Full-budget Lion-no-EMA = 62.47 (NOT 78 from truncated runs). Closed; reassigned to #2183 (fill AdamW+EMA+MAE 2×2 cell).
- **#1999 (fern):** Cosine T_max=16 ± eta_min at lr=1e-4 — both regress (+11.9%/+7.5%). eta_min=0 strictly dominated by eta_min=1e-5. Closed; reassigned to #2167 at lr=2e-4.
- **#2167 (fern):** Cosine T_max=16 + eta_min=1e-5 at lr=2e-4 on n_head=2 — both arms regress (+10.9%/+13.4% vs new baseline). Schedule-matching has signal (T_max=16 beats T_max=50 by 1.31 val) but insufficient. lr=2e-4 on n_head=2 regresses ~5.5 val. Pattern: cosine changes consistently hurt at 30-min cap. Closed; reassigned to #2295 (EMA decay sweep).
- **#2210 (nezuko):** sw=5 on n_head=2 — **MERGED** val=50.91, test=43.68. Non-monotonic: sw=5 < sw=10 < sw=7. Reassigned to #2277 (sw=4 vs sw=3).
- **#2144 (askeladd):** Lion β2=0.995 wins −2.9% val on OLD compound (monotonic: 0.95<0.99<0.995). Can't merge on n_head=2 compound (was +5.3%). Closed; reassigned to #2271 (β2 sweep on n_head=2: confirm 0.995, push to 0.999).
- **#2131 (tanjiro):** Dropout=0.3/0.1 on n_head=4 — **dropout=0.2 locally optimal** (0.3 mean val=55.49 ± 0.38 ≈ baseline 55.41 within noise; 0.1 regresses +4.3%). Under-reg signal from mlp_ratio=4 did NOT transfer to mlp_ratio=2. Closed; reassigned to #2251 (lr sweep on n_head=2).
- **#2001 (askeladd):** Lion β1=0.95/β1=0.85 — regression both arms. Canonical β1=0.9 confirmed optimal. Closed.
- **#1932 (thorfinn):** Lion lr=2e-4 — **MERGED** val=55.41, test=47.90.
- **#1825 (askeladd):** MAE loss on Lion+EMA — **MERGED** val=56.58, test=48.82.
- Earlier rounds (see EXPERIMENTS_LOG.md for full list)

## Key findings (all rounds)

1. **n_head=2 is a major architectural win (#2069):** −7.78% val/test vs lr=2e-4 baseline; wins all 4 test splits; 20 epochs in 30 min (vs 16 at n_head=4). Mechanism: at slice_num=64, n_head=4 had per-head dim=32 (undersized); n_head=2 doubles to 64. Monotonic: n_head=2 wins → n_head=4 baseline → n_head=8 worst.
2. **Lion lr-doubling trend saturated at lr=2e-4 (#2086):** Both lr=3e-4 and lr=4e-4 regress (+3.8–4.1%). Flat minimum, not sharp peak. EMA absorbs same noise level (~10-12 pt main-vs-EMA gap) at all lrs. Schedule shape (OneCycleLR) is the natural next lever.
3. **MAE (L1) loss on Lion+EMA:** −7.71% val — uniform per-node weighting compounds cleanly with Lion.
4. **Lion optimizer + EMA:** −20.4% val. Mechanism quantified (#2070): Lion direction ~75%, EMA ~25%. Synergy Lion-led; full-budget Lion-no-EMA = 62.47 (+7.06 vs 55.41).
5. **EMA weight averaging (decay=0.99):** −22.1% val — foundational.
6. **Fourier positional encoding:** −14.8% test — foundational.
7. **Canonical wd scaling disconfirmed on Lion+MAE:** wd=5e-4 regresses; EMA+dropout already saturate regularization.
8. **Architectural compute-wall:** all capacity axes (depth, width, FFN-width) regress at 30-min cap. Compute-neutral changes (n_head, slice_num) are the only viable architecture levers — and n_head=2 proves this route.
9. **Lion β1=0.9 confirmed optimal (#2001):** β1=0.95 (+4.0%) and β1=0.85 (+6.4%) both regress; asymmetric — over-reactive hurts 2× more than over-inertial.
10. **Batch size falsified (#2052):** bs=8 halves optimizer steps in same wall-clock; step-count-limited not gradient-noise-limited. VRAM near limit at bs=8.
11. **BF16:** foundational (+4 epochs in 30-min window).
12. **Dropout=0.2 confirmed locally optimal (#2131):** dropout=0.3 mean ≈ 0.2 within noise (±0.38 val), 0.1 regresses +4.3%. Under-regularization signal from mlp_ratio=4 (#1961) does NOT transfer to mlp_ratio=2; main-vs-EMA gap already moderate (~6–11) on this compound.
13. **β2 effect REVERSES at n_head=2 (#2271):** On n_head=4 (#2144): β2=0.995 won −2.9% (monotonic 0.95<0.99<0.995). On n_head=2 (#2271): **β2=0.99 < β2=0.995 < β2=0.999** — canonical β2=0.99 is now optimal. Doubling per-head dim 32→64 shifts optimal momentum window; richer heads already filter noise internally. **β2 sweep closed on n_head=2 compound.**
14. **surf_weight=5 merged (#2210):** −0.39%/−1.13% val/test vs n_head=2 baseline. Non-monotonic response: sw=5 < sw=10 < sw=7. Win concentrated in in-dist splits (single_in_dist −2.81). Lower probe (sw=3/4) in #2277.
15. **Cosine schedule changes consistently regress at 30-min budget (#1999, #2167, #2211):** Three independent schedule experiments (T_max-matching, eta_min, OneCycleLR) all regress. At both lr=1e-4 and lr=2e-4 the model is still in the early exploration regime at the cap — forcing LR down faster cuts off productive learning. Schedule shape is not a viable lever at current budget.
16. **AdamW+EMA+MAE diagnostic (#2183, in progress):** Both lr arms show val~73-74 (+44-45% vs baseline). 2×2 mechanism table: Lion+EMA=50.91, Lion-no-EMA=62.47, AdamW+EMA=73.38, AdamW-no-EMA=82.46. Lion contributes ~2× more than EMA; EMA contribution is larger in the noisier AdamW cell.
17. **slice_num=32 MERGED (#2218):** val=49.86/test=42.19 — −2.06%/−3.40% vs #2210 baseline. Monotonic: 32 < 64 < 128. Coarser slicing is faster (81.4s/ep vs 93.5s) → 23 epochs in 30 min vs 20. All 4 test splits improve.
18. **Split-loss formulation falsified (#2216):** surf-MAE + vol-Huber/MSE — 3 runs, 2 formulations, all regress (+4.5–7.1%). MAE uniform weighting already aligned with the metric.
19. **AdamW+EMA+MAE 2×2 table complete (#2183):** Lion+EMA=49.86, Lion-no-EMA=62.47, AdamW+EMA=73.38, AdamW-no-EMA=82.46. Lion contributes ~2× EMA.
20. **slice32+sw5 synergistic interaction MERGED (#2335):** val=48.57/test=41.48. Stacking slice32 and sw5 yields 1.75× additive gain on val. OOD splits (camber, re_rand) gain most; single_in_dist regresses slightly.
21. **n_head=1 MERGED (#2338):** val=46.67/test=40.69 — new best. Monotonic trend extends: n_head=1 < 2 < 4 < 8. Per-head dim=128 concentrates global attention; 26 epochs in 31 min (71.1s/ep). All 4 test splits improve. val still descending at cap.
22. **slice_num=16 confirmed monotonic on n_head=2 (#2337):** val=48.08/test=41.02 — beats old #2218 (49.86/42.19) but loses to new #2338 (46.67). Trend holds: 16 < 32 < 64 < 128. Speed gain 75.9s/ep (−6.8% vs slice32), +1 epoch in 30 min. Val still descending at cap. Next: test slice_num=16 on n_head=1 compound (#2430 frieren).
23. **ema_decay=0.99 confirmed locally optimal (#2295):** Both decay=0.999 (+1.76 val, 50% crashes) and decay=0.95 (+2.32 val) regress on n_head=2+sw=5. EMA-main gap at best epoch identical (5.06/5.00) — window length matters mid-training not at convergence. 0.99 (100-step window ≈ ¼ epoch) is the sweet spot for this training horizon. EMA decay sweep closed.
24. **sw U-curve on n_head=2+slice32 confirmed at sw=3 minimum (#2372):** sw=3 (48.47) beats old #2218 (49.86) by −2.78% but loses to #2338 (46.67). sw=2 regresses all 4 splits (+2.84%). U-shaped curve: sw=10 (49.86) ≥ sw=5 (48.57) ≈ sw=3 (48.47) ≪ sw=2 (51.28). Cruise gain at sw=3 (−7.59%) is the largest single-split win this round; floor at ≥3× weighting on slice32.

## Priority for current wave

**Architecture — slice_num × n_head compound (highest priority):**
- **slice_num=16 on n_head=1 (#2430 frieren)** — slice_num=16 confirmed +1.78 val win on n_head=2; n_head=1 confirmed −1.77 val win; if stacking holds, combined gain could push val toward ~44-45; speed projection: ~65-68s/ep → 27-28 epochs
- **n_head=1 + sw=5 (#2416 alphonse)** — stack sw=5 on top of n_head=1 compound; sw=5 was synergistic with slice32 (+1.75× additive); test if same synergy transfers to n_head=1
- **lr sweep on n_head=1 (#2419 edward)** — lr=1.5e-4 vs lr=1.25e-4; slice64 winner was 1.5e-4; n_head=1 may prefer different lr
- **n_layers reduce (#2400 askeladd)** — n_layers=4/3 on slice_num=32; speed-dividend thesis
- **mlp_ratio=4 vs mlp_ratio=1 (#2446 nezuko)** — unexplored FFN axis; Transolver paper used mlp_ratio=4; mlp_ratio=1 trades capacity for epochs (~30-32 vs 26)

**Optimizer hparams:**

**Optimizer hparams:**
- Lion wd sweep (#2356 thorfinn) — wd=3e-4 vs wd=3e-5 on new slice_num=32 compound
- **Lion β2 CLOSED (#2271)** — canonical β2=0.99 confirmed optimal on n_head=2; direction reversed from n_head=4
- **lr sweep on slice_num=32 (#2376 tanjiro)** — lr=1.5e-4 (slice64 winner) vs lr=1.25e-4; transfer lr signal to new compound; #2251 confirmed optimal ~1.5e-4 at slice_num=64

**EMA:**
- EMA decay CLOSED (#2295) — 0.99 confirmed locally optimal. No further EMA sweeps needed.

**Lion momentum:**
- **β2 on n_head=1 (#2438 fern)** — complete the monotonic n_head×β2 story: β2=0.995 won at n_head=4; β2=0.99 won at n_head=2; now testing β2=0.95 vs β2=0.995 at n_head=1 (per-head dim=128)

**Loss:**
- Split loss: surface-MAE + volume-Huber (#2216 frieren) — aligns train signal with eval metric; volume Huber reduces outlier noise
- surf_weight lower probe (#2277 nezuko) — sw=4 vs sw=3; first data below current sw=5 minimum

**EMA weight averaging:**
- EMA decay sweep (#2295 fern) — decay=0.999 vs 0.95 on sw=5+n_head=2; hasn't been re-tuned since #1607

**lr × architecture interaction:**
- lr sweep on slice_num=32 (#2376 tanjiro) — #2251 confirmed lr=1.5e-4 is optimal at slice_num=64+n_head=2; now testing transfer to slice_num=32 compound; {1.25e-4, 1.5e-4} bracket

## Potential next directions (post-current-wave)

- **n_head=1 (head_dim=128):** monotonic trend continued; may hit a "single global projection" regime
- **Longer training budget** — every win descending at cap; highest-EV change if wall-clock extended
- **n_head=2 + n_hidden=192** — width expansion previously regressed at n_head=4; with larger per-head dim at n_head=2, may work differently
- **EMA decay sweep on n_head=2** — EMA tuned for n_head=4 compound; new architecture may prefer different decay
- **β2 × lr interaction** — if β2=0.995 also wins on n_head=2, test whether β2=0.995 + lr=2e-4 compounds (tanjiro's #2251 and askeladd's #2271 results will inform this)
