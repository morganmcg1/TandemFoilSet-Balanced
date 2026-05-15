# SENPAI Research State

- **Date:** 2026-05-15 20:05
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3221 (nezuko, merged):** Fourier positional features n_freqs=10 + Huber-0.3
- **val_avg/mae_surf_p: 89.27** | **test_avg/mae_surf_p: 79.43**
- Per-split test surf_p: single=93.65, rc=88.94, cruise=56.92, re_rand=78.20
- Key insight: Replacing raw (x,z) with log-spaced Fourier embeddings (space_dim=42) gives 13.5% val improvement with ~4K extra params. Both wall-clock-capped arms (14 epochs) still improving at cutoff. Scaling 6→10 freqs showed continued gain — n=10 may not be the saturation point.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | 98.62 | 88.14 | -4.4% |
| **#3221 (nezuko, merged)** | **Fourier n=10 + Huber-0.3** | **89.27** | **79.43** | **-9.5%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| frieren | #3333 | LR T_max=20 → now: Fourier+Huber+clip+T_max=20 | Sent back (rebase on Fourier) |
| thorfinn | #3227 | Surf-anneal 1→20 + Fourier + Huber + clip | Sent back (rebase on Fourier) |
| edward | #3192 | EMA checkpoint averaging on Fourier baseline | Stale (rebase requested) |
| tanjiro | #3419 | n_hidden=160 + T_max-aligned LR | WIP |
| alphonse | #3420 | Log-space pressure loss (sign-preserving log transform) | WIP |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta=0.3/0.1 | WIP |
| nezuko | #3438 | Fourier freq sweep: n_freqs∈{12,14} + grad_clip=0.25 | NEW (wave-4) |
| fern | #3439 | Gaussian random Fourier features: σ∈{1.0,5.0} | NEW (wave-4) |

## Closed this round

| PR | Reason |
|---|---|
| #3225 (tanjiro) | Multiscale (32+128) too slow: ~248s/ep, only 8ep in budget |
| #3178 (alphonse) | Per-sample scale + Huber anti-synergistic; both downweight outliers |
| #3334 (tanjiro) | n_hidden=192: 5× per-epoch slowdown, only 9ep in budget |
| #3355 (alphonse) | Physics features: arithmetic combos of existing inputs, MLP synthesizes implicitly |
| #3199 (fern) | Dualhead redundant with Huber (both address surface heavy-tail); neither arm beats new Fourier baseline |

## Current research themes

1. **Fourier positional feature scaling** — n=10 gave 13.5% improvement; still improving at wall-clock cutoff. Next: Is n=12/14 better? (nezuko #3438). Does the Gaussian random basis (Tancik 2020) outperform log-spaced at the same n? (fern #3439).

2. **Full-stack composition wave** — Every improvement so far was orthogonal: Huber-0.3, clip-0.25, Fourier n=10, LR T_max=20. None have been fully combined yet. Frieren is testing the full combo (Fourier+Huber+clip+T_max=20). Expected to compound.

3. **Heavy-tail / robust training** — Huber-0.3 + clip-0.25 = additive gain. Askeladd (#3424) testing tighter clip (0.1) and lower Huber delta (0.1). Clip_frac=1.0 at both thresholds confirms tail pressure still present.

4. **LR schedule alignment** — T_max=20 confirmed as free-lunch (8.7% val improvement over T_max=50). Not yet tested on Fourier baseline — frieren's assignment.

5. **Budget-aware capacity scaling** — tanjiro #3419 testing n_hidden=160 with T_max matched to ~25-30 epoch budget. Previous n_hidden=192 was too slow.

6. **Structural search** — thorfinn surf-anneal (loss curriculum) and alphonse log-space pressure loss still pending results. Both need rebase on Fourier baseline.

## Potential next research directions

- **Fourier on other geometric inputs** — arc-length (saf), shape descriptors (dims 4-11) could benefit from frequency lifting; nezuko suggested this
- **Higher LR with clip** — now that clip is in the stack, can we push lr from 5e-4 → 1e-3? A higher LR + Fourier combination untested.
- **Homoscedastic uncertainty weighting** (#10): auto-learn surf_weight tradeoff
- **Stochastic depth** (#13): regularization via DropPath, cheap from timm
- **Re-stratified curriculum** (#12): two-phase sampler oversampling low-Re in early epochs
- **AMP (torch.cuda.amp)**: mixed precision → 30-50% faster epochs → 20-22 epochs in 30-min budget vs 14 now
- **Per-split loss weighting**: rc split consistently hardest for tight clipping; domain-adaptive loss weighting could recover rc
- **Relative position attention**: Fourier lifts absolute position; rotary embeddings in attention heads would lift pairwise mesh-node distances

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
