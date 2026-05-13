# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~00:10 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val (standalone) | test (standalone) |
|----|------|-----------------|-------------------|
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current advisor branch:** Fourier + Huber(δ=1.0) + Dropout(default=0.1) + BF16 + scoring fix

Best individual val=98.79 (Huber), best individual test=88.74 (dropout). **First compound run on this branch will establish the true stacked baseline.** Expected compound val: ~90-95 range.

**To reproduce compound baseline:**
```bash
cd "target/" && python train.py --dropout 0.2 --huber_delta 1.0 --agent <student> --wandb_name "<name>/compound-baseline"
```

## Active experiments

| PR | Student | Config | Status |
|----|---------|--------|--------|
| #1703 | askeladd | Huber δ sweep: δ=0.5 (primary), δ=2.0 (secondary) | WIP — new |
| #1706 | fern | Dropout sweep: 0.25 (primary), 0.15, 0.30 | WIP — new |
| #1690 | nezuko | Fourier L=8 + concat-raw positions | WIP |
| #1694 | frieren | n_head=8 + n_hidden=192 secondary | WIP |
| #1604 | alphonse | Asinh transform + rebase on Fourier base | WIP — rebasing |
| #1583 | thorfinn | T_max=18 + rebase on Fourier base | WIP — rebasing |
| #1607 | edward | EMA weight avg + rebase on Fourier base | WIP — rebasing |
| #1400 | tanjiro | Aux surf-p head + Fourier compound | WIP — running compound |

All 8 students active.

## Key findings (all rounds)

1. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test — biggest single gain. Foundational input feature.
2. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
3. **Scoring bug fixed (#1541):** NaN guard in scoring.py.
4. **Huber loss (δ=1.0):** −4.31% val vs Fourier baseline; targets high-Re gradient outliers.
5. **Dropout=0.2 + clip=1.0:** −4.11% val vs Fourier baseline; smooths co-adaptation, OOD cruise hit 58.81.
6. **Three improvements appear orthogonal:** Fourier (input), Huber (loss), Dropout (regularisation) each address different aspects. Compound expected.
7. **Frequency scaling crucial:** Fourier max_freq=1000→32 flipped result from −8% to +14%.
8. **Capacity-up architecture loses:** slice_num=128, AdamW betas tune both regressed.

## Priority for round 3+

**Sweeps (in flight):**
- Huber δ sweep: find optimal δ on compound base
- Dropout rate sweep: find optimal p on compound base

**Compounds to test (after sweeps return):**
- Fourier + optimal Huber + optimal Dropout (full compound)
- Fourier + T_max=18 (if thorfinn's rebase wins)
- Fourier + EMA (if edward's rebase wins)

**Architecture exploration (in flight):**
- n_head=8 (frieren)
- n_hidden=192 secondary (frieren)
- Fourier L=8 + concat-raw (nezuko)

**Potential after architecture exhausted:**
- n_layers=6
- Lion/Adan optimizer
- Per-sample y-normalization
- Sobolev divergence penalty
- Dropout + surf_weight retune (lower surf_weight=5-7 now that Huber is robust)
