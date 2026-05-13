# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~05:20 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val (standalone) | test (standalone) |
|----|------|-----------------|-------------------|
| #1781 | Lion optimizer lr=1e-4+EMA | **61.30** | **52.68** |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current advisor branch:** Fourier + Huber(δ=1.0) + Dropout(default=0.1) + BF16 + scoring fix + EMA(decay=0.99) + **Lion optimizer(lr=1e-4)**

**CONFIRMED:** Lion+EMA is the dominant compound. Lion's sign-magnitude updates decouple exploration from EMA's smoothing, yielding 20–28% uniform improvement across all 4 test splits. Model still descending at epoch-16 cap — not converged. Longer budget is highest-EV direction.

**CONFIRMED:** EMA-0.99 + dropout=0.1 (default) is the correct dropout anchor on EMA base. PR #1748 showed dropout=0.2 over-regularises on EMA alone.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#1932** | **thorfinn** | **Lion lr=2e-4 (Arm 1) + lr=2e-4 wd=5e-4 (Arm 2)** | **WIP — new** |
| **#1934** | **alphonse** | **Width expansion: n_hidden=192 (Arm 1), n_hidden=256 (Arm 2)** | **WIP — new** |
| #1857 | edward | EMA decay sweep: 0.995 (Arm 1), 0.999 (Arm 2) | WIP |
| #1752 | nezuko | surf_weight sweep: 5 (primary), 7 (secondary) on EMA+Huber+Dropout base | WIP |
| #1761 | tanjiro | n_layers=6: +1 Transolver depth block (dropout=0.1 retry) | WIP |
| #1786 | frieren | Higher LR (1e-3 primary, 2e-3 secondary) on EMA base | WIP |
| #1823 | fern | Weight decay sweep: 5e-4 (primary), 1e-3 (secondary) vs default 1e-4 on EMA base | WIP |
| #1825 | askeladd | MAE (L1) loss replacing Huber — match training loss to ranking metric | WIP |

**Note on WIP experiments #1857, #1752, #1761, #1786, #1823, #1825:** These were designed on the pre-Lion AdamW+EMA base (val=77.05). If they complete and beat their old base but not the new Lion baseline (val=61.30), they should be sent back for re-running on the Lion compound. Their direction may still be valuable stacked on top of Lion.

## Closed experiments this round

- **#1781 (thorfinn):** Lion optimizer lr=1e-4 — **MERGED. New best: val=61.30, test=52.68.** −20.4%/−22.8%, uniform 20-28% across all 4 test splits. Session-defining win.
- **#1604 (alphonse):** Asinh pressure transform — regression (+7.5% val). Huber+Asinh double-compress the tail. Closed.
- **#1748 (edward):** EMA=0.99 + dropout=0.2 compound — regresses. val 78.87 vs 77.05 (two seeds). EMA already fills regularisation headroom; dropout=0.1 (default) is correct anchor.
- **#1706 (fern):** Dropout rate sweep (0.15/0.25/0.30) — closed stale, reassigned to weight decay.
- **#1703 (askeladd):** Huber δ sweep (0.5, 2.0) — closed stale, reassigned to MAE loss.
- **#1690 (nezuko):** Fourier L=8 and L=6 concat-raw — both arms regress. L=6 normalized remains sweet spot.
- **#1400 (tanjiro):** Aux surf-p head λ∈{2,5} — dominated by Fourier, consistently worse on compound base.
- **#1583 (thorfinn):** T_max=18 cosine schedule — closed stale; direction dominated by EMA.
- **#1694 (frieren):** n_head=8 attention — closed stale. Reassigned to #1786 (higher LR).

## Key findings (all rounds)

1. **Lion optimizer (lr=1e-4) + EMA:** −20.4% val / −22.8% test — largest single gain of the session. Lion's sign-magnitude updates + EMA smoothing decouple exploration from integration. Still descending at 30-min cap (epoch 16/50). All 4 test splits improve uniformly by 20-28%.
2. **EMA weight averaging (decay=0.99):** −22.1% val / −23.1% test — before Lion was discovered; now absorbed as part of Lion compound.
3. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test — biggest single gain before EMA. Foundational input feature.
4. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
5. **Huber loss (δ=1.0):** −4.31% val vs Fourier baseline; targets high-Re gradient outliers.
6. **Dropout=0.2 + clip=1.0:** On the non-EMA base. With EMA+Lion, dropout=0.1 is the correct setting.
7. **Frequency scaling crucial:** Fourier max_freq=1000→32 flipped result from −8% to +14%.
8. **L=6 Fourier is the sweet spot:** L=8 wash, concat-raw hurts.
9. **Asinh transform fails with Huber:** Double-compression removes tail signal the model needs. If testing asinh again, must remove Huber first.

## Priority for current wave

**Lion follow-up (highest priority):**
- Lion lr=2e-4 (#1932 thorfinn) — push further on the lr doubling trend that proved large
- Width expansion n_hidden=192/256 (#1934 alphonse) — model not converged at cap; capacity may be limiting

**On pre-Lion AdamW base (results need re-evaluation against new Lion baseline):**
- EMA decay sweep (#1857 edward) — 0.995/0.999 may be orthogonal to Lion
- surf_weight sweep (#1752 nezuko)
- n_layers=6 depth (#1761 tanjiro)
- Higher LR on AdamW base (#1786 frieren) — less relevant now that Lion handles LR differently
- Weight decay sweep (#1823 fern)
- MAE/L1 loss (#1825 askeladd)

## Potential next directions (post-current-wave)

- **Longer training budget** — both Lion arms hit cap at epoch 16/50 still descending steeply; a 60-min or 100-epoch run could push substantially lower. Highest EV if budget can be extended.
- **Lion-no-EMA ablation** — confirms the synergy mechanism for ICML appendix
- **Lion lr=3e-4 or 5e-4** — if lr=2e-4 still wins, explore further
- **n_hidden=192 on Lion+EMA** — if width helps, this stacks on top of Lion
- **EMA decay tuning on Lion base** — edward's sweep (#1857) will reveal if 0.995/0.999 are orthogonal to Lion
- **SWA (Stochastic Weight Averaging)** as alternative or complement to EMA
- **n_hidden=192 + n_layers=6** — compound architectural expansion after individual tests
- **OneCycleLR with Lion** — max_lr=2e-4, pct_start=0.3; Lion's aggressive steps might pair well with cyclic schedule
- **Dropout=0.15 on Lion base** — interpolation between 0.1 and 0.2 (0.2 over-regularises on EMA alone; worth re-testing on Lion base where optimizer noise may substitute)
- **Gradient clipping** — Lion has no natural scale, clip_grad_norm_ might stabilize high-lr runs
