# SENPAI Research State — Charlie Round 3

- **Last updated:** 2026-05-15 12:00 UTC
- **Branch:** `icml-appendix-charlie-pai2h-48h-r3`
- **Tag:** `charlie-pai2h-48h-r3`
- **Active baseline:** `target/train.py` defaults (Transolver, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4, surf_weight=10.0). No PR-merged improvement yet. See `BASELINE.md`.

## Most recent direction from the human research team

None at round start — no GitHub Issues filed against `charlie-pai2h-48h-r3` or `team` at 2026-05-15 12:00 UTC. Default research direction is in effect: drive down `val_avg/mae_surf_p` (and downstream `test_avg/mae_surf_p`) on a Transolver-style baseline with the four-split validation contract pinned by `target/program.md`.

## Current research focus

This is round 3 of a 48-hour, 2-hour-per-run Charlie-arm sprint. The launch is locally-tracked (no remote experiment-tracking back-end) with eight students at 1 GPU each. The round opens with no prior committed metrics on this branch, so the first wave is a *broad, breadth-first* sweep across the high-leverage axes for CFD surrogate regression. We pick one hypothesis per student, one lever per hypothesis, and prefer ideas whose effect should be visible within 30 minutes of training so that round 3 ends with strong evidence about where to invest deeper runs.

### Wave-1 levers (one per student)

The first wave covers the eight axes most likely to move `val_avg/mae_surf_p`:

1. **Loss reformulation** — handle the order-of-magnitude per-sample y-std heterogeneity (per-sample standardization / relative loss).
2. **Architecture scaling** — push `n_hidden`/`n_layers`/`slice_num` upwards given 96 GB VRAM headroom.
3. **Surface-emphasis mechanism** — separate prediction head or surface-aware attention bias for the surface nodes (where the metric is computed).
4. **Optimization schedule** — warmup + cosine + tuned LR/WD, since the default 5e-4 with no warmup is conservative.
5. **Positional / spectral features** — Fourier features for `(x, z)` (replacing the raw 2D coord input) to give the network higher-frequency capacity.
6. **Physics-informed regularizer** — divergence-free prior on velocity, or pressure-Poisson auxiliary, as a soft physics constraint.
7. **Attention / aggregation reformulation** — slice-count and slice-design tweaks; alternative aggregation (perceiver-style cross-attention, multi-scale).
8. **Input feature engineering** — Fourier-encoded `log(Re)`, foil-relative coordinates, or richer NACA encoding to lift the geometry-OOD splits.

Specific hypothesis assignments are tracked as open PRs on the advisor branch and in `research/EXPERIMENTS_LOG.md` as results arrive.

## Potential next research directions

- **Compound the wave-1 winners.** Each merged improvement should be re-tested in combination with the others — small wins often stack across orthogonal levers (loss + architecture + optimization).
- **Deepen the strongest single lever** with longer / wider variants (e.g. if architecture scaling lifts the metric the most, push to `n_hidden=384` or `n_layers=8` for a focused confirmation run).
- **Investigate per-split disagreement.** Wave 1 may reveal that a hypothesis lifts only one of the four val tracks. The follow-up should target the split-specific failure modes (e.g. unseen-camber generalization vs Re-OOD).
- **Cross-pollinate domains.** If raceCar-single dominates the gain, try a domain-conditioned head or stratified surf-weight to recover cruise / tandem.
- **Hard-sample mining or curriculum.** If high-Re samples are over-dominating the loss, weight by *gradient* not by sampler.
- **Spectral / wavelet representations.** If Fourier features are decisive, push to learned spectral bases (SIREN, FNO-style) or wavelet decompositions over the mesh.
- **Operator-learning alternatives.** If Transolver saturates, propose a perceiver-IO or graph-neural-operator family as a follow-up round-4 swap.

The plateau protocol is in place: 5 consecutive non-improvements triggers a strategy-tier escalation (loss → architecture → representation, or commission a fresh literature pull from `researcher-agent`).
