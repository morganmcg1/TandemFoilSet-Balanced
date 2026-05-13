# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~13:05 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| **#2210** | **sw=5 on n_head=2+Lion+MAE+lr=1e-4** | **50.91** | **43.68** |
| #2069 | n_head=2 on Lion+MAE+lr=1e-4 | 51.11 | 44.18 |
| #1932 | Lion lr=2e-4 (wd=1e-4) on Lion+MAE | 55.41 | 47.90 |
| #1825 | MAE (L1) loss on Lion+EMA | 56.58 | 48.82 |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current compound:** Fourier + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **surf_weight=5**

**Note:** Every merged win has had val still descending at the 30-min cap. The model is NOT converged. n_head=2 ran 20 epochs in the same window as n_head=4's 16 — architecture unlocking faster epochs is a compound benefit.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#2218** | **alphonse** | **slice_num sweep on n_head=2: slice_num=32 (Arm1) vs slice_num=128 (Arm2)** | **WIP — new** |
| **#2211** | **thorfinn** | **OneCycleLR on n_head=2: pct_start=0.3 (Arm1) vs pct_start=0.1 (Arm2)** | **WIP — new** |
| **#2216** | **frieren** | **Split loss on n_head=2: surf-MAE + vol-Huber (Arm1), surf-MAE + vol-MSE (Arm2)** | **WIP — new** |
| **#2277** | **nezuko** | **surf_weight lower probe on n_head=2+sw=5 baseline: sw=4 (Arm1) vs sw=3 (Arm2)** | **WIP — new** |
| #2183 | edward | AdamW+EMA+MAE: lr=5e-4 (Arm1) + lr=2e-4 (Arm2) — fill missing 2×2 cell | WIP |
| **#2295** | **fern** | **EMA decay sweep on n_head=2+sw=5: ema_decay=0.999 (Arm1) vs 0.95 (Arm2)** | **WIP — new** |
| #2251 | tanjiro | lr sweep on n_head=2: lr=2e-4 (Arm1) vs lr=1.5e-4 (Arm2) | WIP — new |
| **#2271** | **askeladd** | **Lion β2 on n_head=2: β2=0.995 confirm (Arm1) + β2=0.999 push (Arm2) at lr=1e-4** | **WIP — new** |

## Closed experiments this round

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
13. **Lion β2=0.995 wins −2.9% on OLD compound (#2144, closed):** Monotonic ordering 0.95<0.99<0.995 at three points. β2=0.95 regresses +15.7% (asymmetric). Mechanism: longer momentum window (~200 steps) de-noises direction signal before `sign(·)` taken. Retest on n_head=2 in #2271 — direct merge candidate if win transfers.
14. **surf_weight=5 merged (#2210):** −0.39%/−1.13% val/test vs n_head=2 baseline. Non-monotonic response: sw=5 < sw=10 < sw=7. Win concentrated in in-dist splits (single_in_dist −2.81). Lower probe (sw=3/4) in #2277.
15. **Cosine schedule changes consistently regress at 30-min budget (#1999, #2167):** At both lr=1e-4 and lr=2e-4, T_max=16 and eta_min changes hurt. Model is under-converged at default schedule — forcing LR down faster cuts off productive learning. Schedule shape is not a viable lever at current budget.
16. **AdamW+EMA+MAE diagnostic (#2183, in progress):** Both lr arms show val~73-74 (+44-45% vs baseline). 2×2 mechanism table: Lion+EMA=50.91, Lion-no-EMA=62.47, AdamW+EMA=73.38, AdamW-no-EMA=82.46. Lion contributes ~2× more than EMA; EMA contribution is larger in the noisier AdamW cell.

## Priority for current wave

**Architecture:**
- slice_num sweep (#2218 alphonse) — last unexplored architecture dimension; n_head=2 may pair differently with finer/coarser slices

**Schedule / optimization:**
- OneCycleLR on n_head=2 (#2211 thorfinn) — schedule shape now the frontier after lr-saturation

**Loss:**
- Split loss: surface-MAE + volume-Huber (#2216 frieren) — aligns train signal with eval metric; volume Huber reduces outlier noise
- surf_weight lower probe (#2277 nezuko) — sw=4 vs sw=3; first data below current sw=5 minimum

**EMA weight averaging:**
- EMA decay sweep (#2295 fern) — decay=0.999 vs 0.95 on sw=5+n_head=2; hasn't been re-tuned since #1607

**Optimizer momentum:**
- AdamW+EMA+MAE 2×2 fill (#2183 edward) — diagnostic-only; both arms regressing ~44%, confirms Lion dominance. Awaiting terminal SENPAI-RESULT.
- Lion β2 on n_head=2 (#2271 askeladd) — β2=0.995 wins on old compound; direct merge candidate if transfer confirmed

**lr × architecture interaction:**
- lr sweep on n_head=2 (#2251 tanjiro) — lr=2e-4 won at n_head=4, never retested on n_head=2; per-head dim doubled, optimal lr may have shifted

**Lion momentum parameter (β2):**
- β2 sweep on n_head=2 (#2271 askeladd) — β2=0.995 wins −2.9% on old compound; confirm transfer + push to β2=0.999; merge candidate

## Potential next directions (post-current-wave)

- **n_head=1 (head_dim=128):** monotonic trend continued; may hit a "single global projection" regime
- **Longer training budget** — every win descending at cap; highest-EV change if wall-clock extended
- **n_head=2 + n_hidden=192** — width expansion previously regressed at n_head=4; with larger per-head dim at n_head=2, may work differently
- **EMA decay sweep on n_head=2** — EMA tuned for n_head=4 compound; new architecture may prefer different decay
- **β2 × lr interaction** — if β2=0.995 also wins on n_head=2, test whether β2=0.995 + lr=2e-4 compounds (tanjiro's #2251 and askeladd's #2271 results will inform this)
