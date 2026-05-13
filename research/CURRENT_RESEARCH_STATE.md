# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~09:12 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| #1932 | Lion lr=2e-4 (wd=1e-4) on Lion+MAE | **55.41** | **47.90** |
| #1825 | MAE (L1) loss on Lion+EMA | 56.58 | 48.82 |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current compound:** Fourier + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=2e-4, wd=1e-4)

**Note:** Every merged win has had val still descending at the 30-min/16-epoch cap. The model is NOT converged; longer budget remains highest-EV improvement if allowed. The lr-doubling trend (5e-5→1e-4→2e-4) has not saturated after 3 octaves.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#2086** | **thorfinn** | **Lion lr probe: lr=4e-4 (bold) + lr=3e-4 (midpoint) on Lion+MAE** | **WIP — new** |
| #2070 | edward | Lion-no-EMA ablation + AdamW-no-EMA (diagnostic for ICML appendix) | WIP |
| #2069 | alphonse | n_head sweep: n_head=8 (more) and n_head=2 (fewer) on Lion+MAE | WIP |
| #2056 | nezuko | surf_weight sweep on Lion+MAE: sw=5 (Arm1), sw=15 (Arm2) | WIP |
| #2052 | frieren | batch_size=8: lr=2e-4 linear (Arm1), lr=1e-4 batch-only (Arm2) | WIP |
| #2001 | askeladd | Lion β1 sweep: β1=0.95 (Arm1), β1=0.85 (Arm2) | WIP |
| #1999 | fern | Cosine T_max tuning: T_max=16 (Arm1), T_max=16+eta_min=1e-5 (Arm2) | WIP |
| **#2131** | **tanjiro** | **Dropout sweep on Lion+MAE+lr=2e-4: dropout=0.3 (Arm1), dropout=0.1 (Arm2)** | **WIP — new** |

## Closed experiments this round

- **#1932 (thorfinn):** Lion lr=2e-4 — **MERGED** val=55.41, test=47.90.
- **#1825 (askeladd):** MAE loss on Lion+EMA — **MERGED** val=56.58, test=48.82.
- **#1823 (fern):** wd=5e-4 on AdamW base — regression. Closed.
- **#1781 (thorfinn):** Lion optimizer — **MERGED** val=61.30, test=52.68.
- **#1961 (tanjiro):** mlp_ratio=3/4 — monotonic regression, third capacity-expansion failure. Main-vs-EMA gap diagnostic flagged for follow-up.
- **#1761 (tanjiro):** n_layers=6 — compute-budget bound, +4% regression.
- **#1934 (alphonse):** n_hidden=192/256 — monotonic regression, compute-bound.
- **#1857 (edward):** EMA decay 0.995/0.999 — val regression (nuanced test signal at 0.995).
- **#1786 (frieren):** Higher LR on AdamW — superseded by Lion.
- **#1752 (nezuko):** surf_weight=5 — +8.4% regression, hypothesis falsified.
- **#1604 (alphonse):** Asinh transform — +7.5% regression.
- **#1748 (edward):** EMA+dropout=0.2 — regression.

## Key findings (all rounds)

1. **Lion lr-doubling trend (3 octaves):** 5e-5→1e-4→2e-4, each a win, none saturated. Val still steep at cap each time. **4e-4 is the next test.**
2. **MAE (L1) loss on Lion+EMA:** −7.71% val — uniform per-node weighting compounds cleanly with Lion.
3. **Lion optimizer + EMA:** −20.4% val — sign-magnitude updates decoupled from EMA smoothing.
4. **EMA weight averaging (decay=0.99):** −22.1% val — foundational.
5. **Fourier positional encoding:** −14.8% test — foundational.
6. **Canonical wd scaling disconfirmed on Lion+MAE:** wd=5e-4 regresses vs wd=1e-4 at lr=2e-4. EMA+dropout already saturate regularization; additional wd over-constrains.
7. **Architectural compute-budget wall (all 3 capacity axes confirmed):** depth (n_layers=6, +19%/epoch), width (n_hidden=192/256, +30-53%/epoch), AND FFN-width (mlp_ratio=3/4, +1ep cost + main/EMA 22pt gap) ALL regress at 30-min cap. Only compute-neutral architecture changes viable (n_head, slice_num).
8. **Under-regularization signal (new):** mlp_ratio=4 showed main_val=85.3 vs ema_val=63.5 — 22pt gap, EMA absorbing heavy noise. Suggests baseline compound has regularization headroom. Tanjiro's #2131 dropout sweep probes this directly.
8. **BF16:** foundational (+4 epochs in 30-min window).

## Priority for current wave

**Optimization frontier:**
- Lion lr=4e-4 / 3e-4 (#2086 thorfinn) — test saturation boundary of the lr-doubling trend
- Lion β1 sweep (#2001 askeladd) — β1=0.9 from large-scale vision; small-data optimal may differ
- Cosine T_max tuning (#1999 fern) — T_max=50 barely decays LR in 16 epochs
- surf_weight on Lion+MAE (#2056 nezuko) — sw=15 direction untested with MAE's uniform weighting
- batch_size + LR scaling (#2052 frieren)
- n_head sweep (#2069 alphonse) — only compute-neutral attention-side knob untested
- Dropout sweep on Lion+MAE+lr=2e-4 (#2131 tanjiro) — direct probe of under-regularization signal from #1961

**Diagnostic / paper:**
- Lion-no-EMA ablation (#2070 edward) — mechanism story for ICML appendix

## Potential next directions (post-current-wave)

- **Longer training budget** — every win has hit cap mid-descent; highest-EV change if wall-clock extended
- **Surface-only MAE + volume Huber** — MAE only where metric is measured
- **OneCycleLR** — peak in middle; pairs with Lion's large exploration amplitude
- **EMA decay=0.995 retest on Lion+MAE** — #1857 hint suggests test improvement at 0.995
- **slice_num sweep** — last unexplored architecture dimension (current=64)
- **β2 (Lion momentum EMA) sweep at lr=2e-4** — at higher lr, β2=0.99 may be the noise bottleneck
