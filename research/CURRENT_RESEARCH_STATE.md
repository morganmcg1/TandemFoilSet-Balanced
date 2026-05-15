# SENPAI Research State

- **Date:** 2026-05-15 17:15
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm; data NaN bug tracked in issue #1569, fixed in merged baseline)_

## Current best

- **PR #3213 (frieren, merged):** Huber loss delta=0.3
- **val_avg/mae_surf_p: 103.18** | **test_avg/mae_surf_p: 92.02**
- Per-split test surf_p: single=111.93, rc=102.85, cruise=62.84, re_rand=90.45
- Key insight: heavy-tailed pressure distribution (y_std 50→2,077 across Re) responds strongly to robust loss. Huber-0.3 ≈ L1-like regime gives -19% vs the ~128 empty baseline. NaN-safe evaluate_split fix also shipped.

## Wave-2: currently active

| Student | PR | Hypothesis |
|---|---|---|
| frieren | #3333 | LR schedule alignment (T_max=14/20 with huber_delta=0.3) |
| tanjiro | #3334 | Wider Transolver n_hidden=192 with huber_delta=0.3 |

## Rebasing wave-1 PRs (compose with Huber baseline)

| Student | PR | Task |
|---|---|---|
| fern | #3199 | Dualhead architecture on Huber baseline; run `--huber_delta 0.3` |
| askeladd | #3182 | grad_clip=0.5 + huber_delta=0.3; also try max_norm=0.25 |
| thorfinn | #3227 | surf_weight anneal + huber_delta=0.3; also try terminal=30 |

## Fresh wave: active

| Student | PR | Hypothesis |
|---|---|---|
| alphonse | #3355 | Physics-informed input features: Re_x proxy, gap×log(Re), sin/cos AoA |

## Still WIP (original assignments)

| Student | PR | Hypothesis |
|---|---|---|
| edward | #3192 | EMA checkpoint averaging |
| nezuko | #3221 | Fourier positional features |

## Closed

| Student | PR | Reason |
|---|---|---|
| tanjiro | #3225 | Multiscale (32+128) too slow: ~248s/ep vs 130s baseline, only 8ep in budget |
| alphonse | #3178 | Per-sample scale (val 122.42) + Huber composition anti-synergistic (both downweight outliers) |

## Current research themes

1. **Heavy-tail / robust loss** — Huber-0.3 is the big winner (-19% vs baseline). Next: test the delta continuum, and compose with other improvements.

2. **LR schedule alignment** — discovered all experiments hit ~14 epochs in 30-min budget but T_max=50 means LR barely decays. Frieren testing T_max=14/20 to give proper cosine annealing within budget.

3. **Capacity scaling** — now that Huber stabilizes gradients, wider models may benefit. Tanjiro testing n_hidden=192.

4. **Composition wave** — fern (dualhead), askeladd (grad-clip), thorfinn (surf-anneal) all rebasing to compose their changes on top of Huber. Each mechanism is orthogonal; expect stacking gains.

5. **Physics-informed features** — alphonse now testing Re_x proxy + gap×log(Re) + sin/cos AoA (input feature level vs loss level). B-GNN literature shows 3 derived features can match 6× larger model accuracy on boundary nodes.

## Potential next research directions

- **Composite champion:** Huber-0.3 + dualhead + grad-clip-0.5 + LR-alignment in one PR once individual compositions confirm
- **Huber delta sweep:** try 0.1 (nearly L1) to find the floor; try 0.05
- **Log-MAE pressure loss** (#4 from catalogue): sign-preserving log transform on p channel — orthogonal to Huber, targets low-pressure regime specifically
- **Homoscedastic uncertainty weighting** (#10 from catalogue): auto-learn surf_weight balance via Kendall et al. 2018; complements thorfinn's anneal
- **n_hidden=256** if tanjiro's n_hidden=192 wave-2 shows clean scaling
- **Stochastic depth** for larger architectures if overfitting appears (timm.DropPath, ~15 lines)
- **RoPE-2D full version** if nezuko's Fourier features show gain
- **Domain-conditioned MoE slice projections** (NESTOR, #9 from catalogue): if plateau continues after current wave

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries with concrete code recipes)
