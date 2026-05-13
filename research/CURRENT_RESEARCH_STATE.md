# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~08:10 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| #1825 | MAE (L1) loss on Lion+EMA | **56.58** | **48.82** |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current compound:** Fourier + MAE loss + Dropout(0.1 default) + BF16 + EMA(0.99) + Lion(lr=1e-4)

**Note:** Both wins (#1781 Lion, #1825 MAE) still descend at epoch-16 cap. The model is NOT converged; longer budget is highest-EV improvement if allowed.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#2070** | **edward** | **Lion-no-EMA ablation (Arm1) + AdamW-no-EMA (Arm2) on MAE compound** | **WIP — new (diagnostic)** |
| **#2069** | **alphonse** | **n_head sweep: n_head=8 (Arm1, more heads), n_head=2 (Arm2, fewer) on Lion+MAE** | **WIP — new** |
| **#2056** | **nezuko** | **surf_weight sweep on Lion+MAE: sw=5 (Arm1), sw=15 (Arm2)** | **WIP — new** |
| **#2052** | **frieren** | **batch_size=8: lr=2e-4 linear scaling (Arm1), lr=1e-4 batch-only (Arm2)** | **WIP — new** |
| #2001 | askeladd | Lion β1 sweep: β1=0.95 (Arm1), β1=0.85 (Arm2) on Lion+MAE+EMA | WIP |
| #1999 | fern | Cosine T_max tuning: T_max=16 (Arm1), T_max=16+eta_min=1e-5 (Arm2) | WIP |
| #1932 | thorfinn | Lion lr=2e-4 (Arm1), lr=2e-4+wd=5e-4 (Arm2) | WIP |
| #1961 | tanjiro | FFN width: mlp_ratio=3 (Arm1), mlp_ratio=4 (Arm2) | WIP — nudged (stale) |

## Closed experiments this round

- **#1825 (askeladd):** MAE loss on Lion+EMA — **MERGED** val=56.58, test=48.82.
- **#1823 (fern):** wd=5e-4 on AdamW base — +1.84% val regression, tied test. Closed.
- **#1781 (thorfinn):** Lion optimizer — **MERGED** val=61.30, test=52.68.
- **#1761 (tanjiro):** n_layers=6 — both dropout arms regress (+4%). Compute-budget bound at 30-min cap.
- **#1604 (alphonse):** Asinh pressure transform — +7.5% regression. Huber+Asinh double-compress.
- **#1748 (edward):** EMA=0.99 + dropout=0.2 — regresses. EMA fills regularisation headroom.
- **#1786 (frieren):** Higher LR (1e-3/2e-3) on AdamW+EMA — direction superseded by Lion compound. Arm 1 had genuine −3.3% val / −5.7% test gain on pre-Lion base but +31.7% vs current best.
- **#1752 (nezuko):** surf_weight=5 on pre-Lion AdamW+EMA — +8.4% val regression (uniform across all splits, including volume). Hypothesis decisively falsified.
- **#1934 (alphonse):** n_hidden=192/256 — monotonic regression with width. Compute-bound at 30-min cap (per-epoch +30%/+53%).
- **#1857 (edward):** EMA decay 0.995/0.999 — val regression. **Nuanced finding:** decay=0.995 improved test by −1.9% (uniform across splits) despite val regression on pre-Lion base; potential signal for re-test on Lion+MAE.

## Key findings (all rounds)

1. **MAE (L1) loss on Lion+EMA:** −7.71% val / −7.34% test — largest gain on Lion base. Loss-side property (uniform per-node weighting) is independent of optimizer, compounds cleanly with Lion.
2. **Lion optimizer (lr=1e-4) + EMA:** −20.4% val / −22.8% test — decouples exploration (Lion sign updates) from integration (EMA averaging). Still descending at cap.
3. **EMA weight averaging (decay=0.99):** −22.1% val / −23.1% test — foundational for the session.
4. **Fourier positional encoding (max_freq=32, L=6):** −14.8% test — foundational input feature.
5. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
6. **Architectural compute-budget wall at 30 min:** Depth (n_layers=6, +19% epoch cost) and width (n_hidden=192/256, +30%/+53%) BOTH lose. Capacity expansion is the wrong direction at this budget. **Only compute-neutral architecture changes (head count, mlp_ratio if cheap) remain on the table.**
7. **Huber loss (δ=1.0):** Superseded by MAE on Lion base. Huber's quadratic well competes with MAE's uniform per-node weighting.
8. **Asinh+Huber double-compression:** Both compress the high-Re tail; they compete rather than stack.
9. **EMA decay test/val asymmetry:** decay=0.995 improved test on the OLD base while regressing val (genuine flatter-basin signal). Re-test on Lion+MAE is warranted but is *not* a top-priority lever.

## Priority for current wave

**Highest priority (Lion+MAE base):**
- Lion β1 sweep (#2001 askeladd) — β1=0.9 from large-scale vision; small-data optimal may differ
- Cosine T_max tuning (#1999 fern) — current T_max=50 barely decays LR in 16 epochs
- Lion lr=2e-4 scaling (#1932 thorfinn) — lr doubling trend from 5e-5→1e-4 hasn't saturated
- surf_weight on Lion+MAE (#2056 nezuko) — sw=15 untested direction with MAE's uniform per-node weighting
- batch_size + LR scaling (#2052 frieren) — Lion+EMA may absorb noise reduction from doubled batch
- mlp_ratio=3/4 FFN expansion (#1961 tanjiro) — compute-cheap capacity expansion
- **n_head=8 vs n_head=2 (#2069 alphonse)** — compute-neutral architecture, only attention-side knob untested
- **Lion-no-EMA ablation (#2070 edward)** — diagnostic for ICML appendix; isolates Lion vs EMA mechanism

## Potential next directions (post-current-wave)

- **Longer training budget** — both recent merges hit cap mid-descent; highest EV if wall-clock budget extended
- **Surface-only MAE + volume Huber** — MAE only where metric is measured; Huber on well-behaved interior nodes
- **OneCycleLR** — peak in middle; pairs with Lion's aggressive exploration
- **EMA decay tuning on Lion+MAE** — #1857 closed result hinted at test improvement at 0.995 on AdamW base
- **slice_num sweep on Lion+MAE base** — last unexplored architecture dimension (current=64)
- **n_head + n_hidden joint search** — wider model with FEWER heads might dodge compute wall via different cost profile
- **Lookahead-on-Lion** — outer loop slow-update over Lion's fast inner steps (similar mechanism story to Lion+EMA)
