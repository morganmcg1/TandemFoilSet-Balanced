# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~03:15 UTC
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

**CONFIRMED:** EMA-0.99 + dropout=0.1 (default) is the correct compound. PR #1748 showed EMA+dropout=0.2 over-regularises — every split regresses, two seeds confirm. The merged baseline IS the true compound.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#1857** | **edward** | **EMA decay sweep: 0.995 (Arm 1), 0.999 (Arm 2)** | **WIP — new** |
| #1752 | nezuko | surf_weight sweep: 5 (primary), 7 (secondary) on EMA+Huber+Dropout base | WIP — GPU active |
| #1761 | tanjiro | n_layers=6: +1 Transolver depth block (dropout=0.1 retry) | WIP |
| #1781 | thorfinn | Lion optimizer (lr=5e-5 primary, 1e-4 secondary) on EMA base | WIP |
| #1786 | frieren | Higher LR (1e-3 primary, 2e-3 secondary) on EMA base | WIP |
| #1604 | alphonse | Asinh transform on pressure target (rebasing onto EMA base) | WIP — actively rebasing |
| #1823 | fern | Weight decay sweep: 5e-4 (primary), 1e-3 (secondary) vs default 1e-4 on EMA base | WIP |
| #1825 | askeladd | MAE (L1) loss replacing Huber — match training loss to ranking metric | WIP |

## Closed experiments this round

- **#1748 (edward):** EMA=0.99 + dropout=0.2 compound — regresses. val 78.87 vs 77.05 (two seeds). EMA already fills regularisation headroom; dropout=0.1 (default) is correct anchor.
- **#1706 (fern):** Dropout rate sweep (0.15/0.25/0.30) — closed stale, reassigned to weight decay.
- **#1703 (askeladd):** Huber δ sweep (0.5, 2.0) — closed stale, reassigned to MAE loss.
- **#1690 (nezuko):** Fourier L=8 and L=6 concat-raw — both arms regress. L=6 normalized remains sweet spot.
- **#1400 (tanjiro):** Aux surf-p head λ∈{2,5} — dominated by Fourier, consistently worse on compound base.
- **#1583 (thorfinn):** T_max=18 cosine schedule — closed stale; direction dominated by EMA.
- **#1694 (frieren):** n_head=8 attention — closed stale. Reassigned to #1786 (higher LR).

## Key findings (all rounds)

1. **EMA weight averaging (decay=0.99):** −22.1% val / −23.1% test — single largest gain of the session. EMA-0.99 + dropout=0.1 is the confirmed optimal pairing. Dropout=0.2 over-regularises on EMA base (PR #1748).
2. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test — biggest single gain before EMA. Foundational input feature.
3. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
4. **Huber loss (δ=1.0):** −4.31% val vs Fourier baseline; targets high-Re gradient outliers.
5. **Dropout=0.2 + clip=1.0:** −4.11% val vs Fourier baseline, but **on the non-EMA base**. With EMA, dropout=0.1 is superior.
6. **Frequency scaling crucial:** Fourier max_freq=1000→32 flipped result from −8% to +14%.
7. **Aux head dominated by Fourier:** Once Fourier hidden state carries surface-p signal, aux head gradient competes rather than helps.
8. **L=6 Fourier is the sweet spot:** L=8 wash, concat-raw hurts.

## Priority for current wave

**High confidence (in flight):**
- EMA decay sweep (#1857 edward) — 0.99 was the first value tried; 0.995/0.999 may squeeze more
- surf_weight sweep (#1752 nezuko) — surf_weight=10 predates Huber/EMA, likely tunable (GPU active now)
- n_layers=6 depth (#1761 tanjiro) — architectural expansion

**Hyperparameter tuning on EMA base:**
- Higher LR (#1786 frieren) — EMA smoothing should permit larger steps
- Lion optimizer (#1781 thorfinn) — sign-based momentum vs AdamW
- Weight decay sweep (#1823 fern)
- MAE/L1 loss (#1825 askeladd) — match training loss to ranking metric

**Long-running:**
- Asinh pressure target (#1604 alphonse) — actively rebasing

## Potential next directions (post-current-wave)

- EMA with warmup (start decay at 0.0, ramp to 0.99 over N epochs to avoid poisoning early averaging)
- SWA (Stochastic Weight Averaging) as complement or alternative to EMA
- n_hidden=192 if depth/width experiments suggest capacity headroom
- Auxiliary physics losses (divergence, pressure-Poisson residual)
- Multi-scale Fourier features with learnable frequency bandwidths
- Dropout=0.15 on EMA base (interpolation between 0.1 and 0.2 per PR #1748 finding)
- Cross-attention pooling instead of slice softmax
- Mixup / CutMix on coordinate grids (geometric augmentation)
