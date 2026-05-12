# SENPAI Research State — willow-pai2g-24h-r1

- 2026-05-12 (launch start)
- Most recent research direction from human researcher team: This launch is a
  controlled 24h/48h Charlie-vs-Willow logging ablation. Each training run is
  capped at `SENPAI_TIMEOUT_MINUTES=30`. The fleet is scoped to research tag
  `willow-pai2g-24h-r1` and advisor branch
  `icml-appendix-willow-pai2g-24h-r1`. Do not inspect or compare to any other
  PR / branch / experiment outside this scope.
- Current research focus and themes:
  - Round 1 — atomic single-knob variants over the default `train.py`
    Transolver configuration on TandemFoilSet. Goal is to fill all 8 student
    slots with diverse, well-motivated isolated changes so the fleet produces
    a clean set of comparable W&B training-curve traces for the Charlie/Willow
    logging-ablation comparison. The variants span the major orthogonal
    hyperparameter axes (loss weighting, optimizer, depth, width, attention,
    FFN, regularization) so attribution is easy.
- Round 1 assignments (one isolated change each):
  - alphonse — `surf-weight-25` (loss weighting on the primary metric channel)
  - askeladd — `lr-1e-3` (optimizer step size)
  - edward — `n-layers-7` (depth capacity)
  - fern — `slice-num-128` (Transolver physics-slice resolution)
  - frieren — `n-head-8` (attention heads)
  - nezuko — `mlp-ratio-4` (FFN width)
  - tanjiro — `n-hidden-192` (model width)
  - thorfinn — `wd-3e-4` (AdamW weight decay)
- Potential next research directions and themes (once round 1 lands):
  - Compound the wins from round 1 (e.g. best loss-weight × best lr × best
    depth) into a multi-knob frontier run.
  - Loss reformulation — Huber/Charbonnier robust losses for the volume term
    (high-Re samples push extreme y values that pure MSE penalizes
    disproportionately). Per-channel weighting (Ux/Uy/p) instead of a single
    surf_weight.
  - LR schedule variants — OneCycle, warmup+cosine, lower min-lr.
  - Sampler variants — switching off WeightedRandomSampler, per-Re stratified
    sampling, curriculum on Re or mesh size.
  - Architecture — RoPE / Fourier positional encoding on (x, z), gated FFN
    (SwiGLU), pre-norm vs post-norm placement, residual scaling.
  - Data — coordinate normalization (centering on foil 1, z-score on positions),
    per-domain stats vs global stats.
