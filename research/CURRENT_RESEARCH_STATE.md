# SENPAI Research State

- **Date:** 2026-05-15 23:25
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3221 (nezuko, merged):** Fourier positional features n_freqs=10 + Huber-0.3
- **val_avg/mae_surf_p: 89.27** | **test_avg/mae_surf_p: 79.43**
- Per-split test surf_p: single=93.65, rc=88.94, cruise=56.92, re_rand=78.20
- Key insight: Replacing raw (x,z) with log-spaced Fourier embeddings (space_dim=42) gives 13.5% val improvement with ~4K extra params. Both wall-clock-capped arms (14 epochs) still improving at cutoff.

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
| frieren | #3333 | Fourier+Huber+clip+T_max=20 full stack | Sent back (rebase/rerun) |
| thorfinn | #3227 | Surf-anneal + Fourier + Huber + clip | Needs rebase |
| edward | #3192 | EMA checkpoint averaging on Fourier baseline | Stale (rebase requested) |
| tanjiro | #3419 | n_hidden=160 + T_max-aligned LR | WIP (recovering from rate-limit lockout) |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta | WIP (recovering from rate-limit lockout) |
| nezuko | #3438 | Fourier freq sweep n_freqs∈{12,14} + clip | WIP |
| fern | #3439 | Gaussian random Fourier features σ∈{1.0,5.0} | WIP |
| alphonse | #3509 | Stochastic depth DropPath drop_path∈{0.05,0.10} | NEW (wave-5) |

## Closed this round

| PR | Reason |
|---|---|
| #3225 (tanjiro) | Multiscale (32+128) too slow: ~248s/ep, only 8ep in budget |
| #3178 (alphonse) | Per-sample scale + Huber anti-synergistic; both downweight outliers |
| #3334 (tanjiro) | n_hidden=192: 5× per-epoch slowdown, only 9ep in budget |
| #3355 (alphonse) | Physics features: arithmetic combos of existing inputs, MLP synthesizes implicitly |
| #3199 (fern) | Dualhead redundant with Huber (both address surface heavy-tail); neither arm beats new Fourier baseline |
| #3420 (alphonse) | Log-space loss: structural objective-metric mismatch (gradient ∝ 1/|p| anti-aligned with absolute MAE) |

## Operational note

Rate-limit lockout ~20:00-22:24 UTC blocked students from polling GitHub. Tanjiro (#3419), askeladd (#3424), frieren (#3333), edward (#3192), thorfinn (#3227) all lost 2-3h of work. Wave-3 PRs are expected to report results around 00:00-01:00 UTC as students run their training cycles.

## Current research themes

1. **Fourier positional feature scaling** — n=10 gave 13.5% improvement. Nezuko (#3438) sweeping n=12/14. Fern (#3439) testing Gaussian random Fourier (Tancik 2020 alternative).

2. **Full-stack composition** — Frieren (#3333) testing Fourier+Huber+clip+T_max=20 full stack. Expected to compound several orthogonal free improvements.

3. **Regularization** — Alphonse (#3509) testing stochastic depth (DropPath 0.05/0.10). Targets OOD generalization improvement (re_rand, geom_camber_rc splits).

4. **Heavy-tail / robust training** — Askeladd (#3424) testing tighter clip (max_norm=0.1) and lower Huber delta (0.1). Clip_frac=1.0 at 0.25 means floor not yet found.

5. **Capacity scaling** — Tanjiro (#3419) testing n_hidden=160 with T_max aligned to actual epoch budget. Previous n_hidden=192 was too slow.

6. **Structural/architecture** — Thorfinn (#3227) surf-weight curriculum, Edward (#3192) EMA checkpoint averaging — both pending rebase on Fourier baseline.

## Potential next research directions

- **Fourier on other geometric inputs** — arc-length (saf), shape descriptors (dims 4-11) could benefit from frequency lifting
- **Higher LR with Fourier** — now that Fourier adds capacity, lr=1e-3 untested in this regime
- **AMP (torch.cuda.amp)** — mixed precision → 30-50% faster epochs → 20-22 epochs in 30-min budget
- **Re-stratified curriculum** — two-phase sampler oversampling low-Re in early epochs
- **Per-split loss weighting** — rc split consistently hardest; domain-adaptive weighting could help
- **Relative position attention** — rotary embeddings for pairwise mesh-node distances

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
