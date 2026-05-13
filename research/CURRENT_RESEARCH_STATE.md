# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~01:20 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val (standalone) | test (standalone) |
|----|------|-----------------|-------------------|
| #1607 | EMA decay=0.99 | **77.05** | **68.27** |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current advisor branch:** Fourier + Huber(δ=1.0) + Dropout(default=0.1) + BF16 + scoring fix + **EMA(decay=0.99)**

**IMPORTANT:** EMA is the dominant improvement (−22% val). The merged EMA run used dropout=0.1 (default). The full compound with dropout=0.2 has NOT yet been verified — that's edward's #1748 assignment.

## Active experiments

| PR | Student | Config | Status |
|----|---------|--------|--------|
| #1748 | edward | EMA=0.99 + dropout=0.2 compound verification | WIP |
| #1752 | nezuko | surf_weight sweep: 5 (primary), 7 (secondary) on EMA+Huber+Dropout base | WIP |
| #1761 | tanjiro | n_layers=6: +1 Transolver depth block | WIP |
| #1781 | thorfinn | Lion optimizer (lr=5e-5 primary, 1e-4 secondary) on EMA base | WIP |
| #1786 | frieren | Higher LR (1e-3 primary, 2e-3 secondary) on EMA base | WIP — new |
| #1706 | fern | Dropout rate sweep: 0.15/0.25/0.30 on Fourier+Huber+Dropout base | WIP |
| #1703 | askeladd | Huber δ sweep: δ=0.5, δ=2.0 on compound base | WIP |
| #1604 | alphonse | Asinh transform on pressure target (rebasing onto EMA base) | WIP — re-rebase |

All 8 students active.

## Closed experiments this round

- **#1690 (nezuko):** Fourier L=8 and L=6 concat-raw — both arms regress. L=6 normalized remains sweet spot.
- **#1400 (tanjiro):** Aux surf-p head λ∈{2,5} — dominated by Fourier, consistently worse on compound base.
- **#1583 (thorfinn):** T_max=18 cosine schedule — closed as stale (2.5h silent) and direction dominated by EMA (which solves late-epoch wobble more elegantly than schedule truncation).
- **#1694 (frieren):** n_head=8 attention — closed as stale (2.5h, zero commits/comments after assignment). Reassigned to higher-LR experiment (#1786) to test pod responsiveness with a quick-to-validate hypothesis.

## Key findings (all rounds)

1. **EMA weight averaging (decay=0.99):** −22.1% val / −23.1% test — single largest gain of the session. Main model val unchanged; EMA val reached 77.05. Pure averaging benefit on a noisy optimization landscape.
2. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test — biggest single gain before EMA. Foundational input feature.
3. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
4. **Huber loss (δ=1.0):** −4.31% val vs Fourier baseline; targets high-Re gradient outliers.
5. **Dropout=0.2 + clip=1.0:** −4.11% val vs Fourier baseline. Default in code is 0.1 — use `--dropout 0.2` for the winning value.
6. **Frequency scaling crucial:** Fourier max_freq=1000→32 flipped result from −8% to +14%.
7. **Aux head dominated by Fourier:** Once Fourier hidden state carries surface-p signal, aux head gradient competes rather than helps.
8. **L=6 Fourier is the sweet spot:** L=8 wash, concat-raw hurts.

## Priority for round 3+

**Critical follow-ups (in flight):**
- EMA + dropout=0.2 full compound (#1748 edward) — most important; true stacked baseline not yet established
- surf_weight sweep (#1752 nezuko) — surf_weight=10 predates Huber/EMA, likely tunable
- n_layers=6 depth (#1761 tanjiro) — architectural expansion

**Sweeps (in flight, assigned pre-EMA):**
- Huber δ sweep (askeladd #1703) — results still informative; optimal δ may shift with EMA
- Dropout rate sweep (fern #1706) — optimal p on compound base
- n_head=8 (frieren #1694), asinh (alphonse #1604), T_max=18 (thorfinn #1583)

**Potential after current experiments land:**
- EMA decay sweep (0.995, 0.999) on full compound
- Learning rate tuning on EMA base
- n_hidden=192 if architecture experiments suggest capacity headroom
- SWA (Stochastic Weight Averaging) as complement or alternative to EMA
- Lion/Adan optimizer
