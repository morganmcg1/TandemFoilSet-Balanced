# SENPAI Research State

- **Date:** 2026-05-15 16:30
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm; data NaN bug tracked in issue #1569, fixed in merged baseline)_

## Current best

- **PR #3213 (frieren, merged):** Huber loss delta=0.3
- **val_avg/mae_surf_p: 103.18** | **test_avg/mae_surf_p: 92.02**
- Per-split test surf_p: single=111.93, rc=102.85, cruise=62.84, re_rand=90.45
- Key insight: heavy-tailed pressure distribution (y_std 50→2,077 across Re) responds strongly to robust loss. Huber-0.3 ≈ L1-like regime gives -19% vs the ~128 empty baseline. NaN-safe evaluate_split fix also shipped.

## Wave-1 summary (all students returned results)

| Student | PR | val_avg/mae_surf_p | verdict |
|---|---|---|---|
| frieren | #3213 | **103.18** | **MERGED** — new baseline |
| askeladd | #3182 | 113.90 (clip=0.5) | sent back: combine with Huber-0.3 |
| fern | #3199 | 122.40 (dualhead) | sent back: combine dualhead + Huber-0.3 |
| thorfinn | #3227 | 130.40 (anneal) | sent back: combine surf-anneal + Huber-0.3 |
| tanjiro | #3225 | 163.20 (multiscale) | closed: 2× slow/epoch, ~8ep in budget |
| alphonse | #3178 | (still WIP) | per-sample scale normalization |
| edward | #3192 | (still WIP) | EMA checkpoint averaging |
| nezuko | #3221 | (still WIP, just started) | Fourier positional features |

Reference: askeladd's no-clip baseline arm = **128.69 val_avg** (confirms empty baseline level).

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

## Current research themes

1. **Heavy-tail / robust loss** — Huber-0.3 is the big winner (-19% vs baseline). Next: test the delta continuum (0.1? 0.5?), and compose with other improvements.

2. **LR schedule alignment** — discovered that all experiments hit ~14 epochs in 30-min budget but T_max=50 means LR barely decays. Frieren testing T_max=14/20 to give proper cosine annealing within budget.

3. **Capacity scaling** — now that Huber stabilizes gradients, wider models may benefit. Tanjiro testing n_hidden=192.

4. **Composition wave** — fern (dualhead), askeladd (grad-clip), thorfinn (surf-anneal) all rebasing to compose their changes on top of Huber. Each mechanism is orthogonal; expect stacking gains.

## Potential next research directions

- **Composite champion:** Huber-0.3 + dualhead + grad-clip-0.5 in one PR once individual compositions confirm
- **Huber delta sweep:** try 0.1 (nearly L1) to find the floor
- **Per-sample normalization (alphonse):** if works, compose with Huber — scale handling at both loss level AND input level
- **Physics-informed features:** Re_x proxy, gap×log(Re) interaction (idea #8 from catalogue)
- **Smaller batch + larger LR:** clip_frac=1.0 even at max_norm=0.5 suggests room for more gradient steps per minute (bs=2, lr=1e-3)
- **n_hidden=256** if n_hidden=192 shows clean scaling (tanjiro wave-2 informs this)
- **Stochastic depth** for larger architectures if overfitting appears
- **RoPE-2D** if nezuko's Fourier features show gain
- **Domain-conditioned MoE slice projections** (NESTOR analogue)

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries with concrete code recipes)
