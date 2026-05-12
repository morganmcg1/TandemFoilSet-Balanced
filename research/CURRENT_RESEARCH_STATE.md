<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent direction from human researcher team:** none (controlled 24h/48h Charlie-vs-Willow logging ablation; no inbound human directives at boot).

## Current research focus

Round 2 of the 24h Charlie-vs-Willow logging ablation on TandemFoilSet, a CFD surrogate task. Primary metric `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits); paper-facing comparator `test_avg/mae_surf_p`. The launch is isolated from other branches — no comparison or cross-pollination with parallel Charlie/Willow rounds is permitted.

**Per-training-run wall-clock cap is hard 30 minutes** (`SENPAI_TIMEOUT_MINUTES=30`). This is the dominant constraint shaping the hypothesis space — only ideas that produce signal in the first 5–15 epochs are testable. Hypotheses that need long convergence are off the table for this round.

## Themes for r2

Across 8 idle students, I'm initially assigning across 5 angles to maximize information per launch:

1. **Loss prioritization (3 students)** — surface pressure is the metric; baseline loss treats Ux/Uy/p uniformly and underweights surface nodes given their tiny fraction (~3%) of total mesh nodes. Three independent attacks: (a) raise `surf_weight` (region balance), (b) raise pressure-channel weight inside the per-node loss (intra-channel balance), (c) replace global-normalized MSE with per-sample Huber rescaling (Re-scale fairness).

2. **Optimizer / fast convergence (1 student)** — the conservative `lr=5e-4` may under-use the 30-min budget. `lr=2e-3` with grad clip 1.0 is a sharp discriminating test.

3. **Capacity (1 student)** — 2M-param baseline may be capacity-limited on the geometry+Re variation. Width 128→256 / heads 4→8 doubles representational budget.

4. **Architectural decoupling (1 student)** — separate per-field output heads (Ux/Uy/p) so pressure projection can specialize away from velocity.

5. **Throughput (1 student)** — bf16 autocast + grad accumulation: more epochs per 30 min, larger effective batch, gradient-noise reduction.

Plus **1 student running a stock-config baseline** because the W&B project had zero runs at launch — we need a measured reference to compare every hypothesis against.

## Potential next research directions (round 2 spares / round 3 candidates)

Held back for the next round, sourced from `research/RESEARCH_IDEAS_2026-05-12_initial.md`:

- **H4** — LR warmup (5%) + `CosineAnnealingWarmRestarts(T_0=5 or 10)`. Schedule angle; small expected delta but free.
- **H6** — `slice_num` 64 → 128 (with batch_size=2 if needed). Tests slice-attention granularity for tandem geometry.
- **H7** — 7 layers + stochastic depth p=0.1. Depth instead of width.
- **H8** — SiLU activation (one-flag change). Zero-risk free experiment.
- **H10** — `log1p(dsdf)` + Re-scaled position transform. Input feature engineering; stats.json normalization risk.
- **H13** — Re-conditioned Fourier position encoding (`sin(2π·pos·√Re)`). Physics-informed position.
- **H14** — `slice_num=256, batch_size=1` cruise diagnostic.

After r2 first wave returns, also worth exploring if any winners emerge:

- **Stacked hypotheses**: if H3 (surf_weight=30) wins, try H3+H2 stack (region + channel). If H1 (Huber) wins, try H1+H3 stack. Be careful of confounds.
- **Loss shape**: SmoothL1 / MAE-direct in normalized space (no Huber rescaling, just absolute loss) — if Huber wins for the right reason, MAE-direct might win more.
- **Sampler**: per-batch domain-balanced batching (instead of weighted-random) — guarantees coverage.
- **EMA of weights**: cheap regularizer; may show signal in 10+ epochs if any.
- **Output residual over an analytic prior**: e.g. predict δ over a coarse linear potential-flow estimate — if pressure has a strong "easy" component, this should help.
- **Surface-aware sampling/positional emphasis**: surface nodes get more attention slices or a dedicated head.

## Operational notes

- All 8 students were idle at boot; 8 PRs created. Re-survey shortly to confirm pods pick up assignments.
- Baseline metrics in `BASELINE.md` will be backfilled once `alphonse`'s baseline run logs to W&B; update the file at that point.
- Plateau protocol not yet relevant — this is round 2 of round 2, no consecutive nulls yet.
