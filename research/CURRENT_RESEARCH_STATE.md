# SENPAI Research State

- **Date:** 2026-05-15 19:35
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm; data NaN bug tracked in issue #1569, fixed in merged baseline)_

## Current best

- **PR #3182 (askeladd, merged):** Huber loss delta=0.3 + grad_clip_max_norm=0.25
- **val_avg/mae_surf_p: 98.62** | **test_avg/mae_surf_p: 88.14**
- Per-split test surf_p: single=104.75, rc=104.65, cruise=59.24, re_rand=83.90
- Key insight: Huber-0.3 and clip-0.25 are ADDITIVE — both attack heavy-tail gradients at different scales (per-sample residual suppression vs batch-level update variance). clip_frac=1.0 at both thresholds confirms clipping is still doing real work on top of Huber.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| (round 5 baseline) | ~128 (no-clip baseline from askeladd) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | **98.62** | **88.14** | -4.4% |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| frieren | #3333 | LR schedule T_max=14/20 alignment + Huber-0.3 | WIP (no results yet) |
| thorfinn | #3227 | Surf-anneal 1→20 + Huber-0.3 + terminal=30 | WIP (rebase ongoing) |
| fern | #3199 | Dualhead arch + Huber-0.3 | WIP (rebase ongoing) |
| edward | #3192 | EMA checkpoint averaging | WIP (pod restarted, retrying) |
| nezuko | #3221 | Fourier positional features | WIP (pod restarted, retrying) |
| tanjiro | #3419 | n_hidden=160 + T_max-aligned LR | NEW (wave-3) |
| alphonse | #3420 | Log-space pressure loss (sign-preserving log transform) | NEW (wave-3) |
| askeladd | #3424 | Tighter clip max_norm=0.1 × Huber delta=0.3/0.1 sweep | NEW (wave-3) |

## Closed this round

| PR | Reason |
|---|---|
| #3225 (tanjiro) | Multiscale (32+128) too slow: ~248s/ep, only 8ep in budget |
| #3178 (alphonse) | Per-sample scale + Huber anti-synergistic; both downweight outliers |
| #3334 (tanjiro) | n_hidden=192: 5× per-epoch slowdown, only 9ep in budget |
| #3355 (alphonse) | Physics features: arithmetic combos of existing inputs, MLP synthesizes implicitly |


## Current research themes

1. **Heavy-tail / robust loss + clipping stack** — Huber-0.3 + clip-0.25 = 98.62. Next: Is there headroom from tighter clip (0.1/0.05)? From lower Huber delta (0.1)?

2. **LR schedule alignment** — frieren #3333 testing T_max=14/20. Results pending.

3. **Budget-aware capacity scaling** — tanjiro #3419 testing n_hidden=160 with T_max matched to actual epoch budget (~25-30 epochs).

4. **Loss formulation** — alphonse #3420 testing log-space pressure loss: sign-preserving log transform compresses dynamic range, helps cruise/re_rand splits without per-sample normalization penalty.

5. **Composition wave** — fern (dualhead) and thorfinn (surf-anneal) still rebasing. Once results land, can consider compositing best individual changes.

## Potential next research directions

- **Askeladd assignment needed**: Huber delta sweep (0.1) OR higher LR (1e-3/2e-3) with clip enabled
- **Clip sweep**: try max_norm=0.1, 0.05 — clip_frac=1.0 at 0.25 means floor not yet found
- **Composite champion**: Huber-0.3 + clip-0.25 + dualhead + LR-alignment once composition wave completes
- **Homoscedastic uncertainty weighting** (#10): auto-learn surf_weight tradeoff
- **Stochastic depth** (#13): regularization via DropPath, cheap from timm
- **Re-stratified curriculum** (#12): two-phase sampler oversampling low-Re in early epochs
- **Per-split loss weighting**: askeladd data shows rc split consistently hardest for tight clipping; a domain-adaptive loss could recover that split

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
