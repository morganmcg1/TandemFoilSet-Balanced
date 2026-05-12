# SENPAI Research State

- **As of:** 2026-05-12
- **Round:** willow-pai2g-48h-r4 (initial bring-up; advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none in this launch yet — operator note in effect: controlled 24/48 h Charlie-vs-Willow logging ablation, each training run hard-capped at `SENPAI_TIMEOUT_MINUTES=30`)

## Current research focus and themes

The advisor branch is freshly forked from `icml-appendix-willow` with only the cap-clarification commit. **No baseline metrics are yet committed to this branch and no prior in-round PR results exist** — all 8 students were idle at boot. The opening round assigns each student one diverse hypothesis covering loss-design, schedule, optimizer-stability, capacity, and head-design families. Goal of this round: surface 1-3 PR-mergeable wins and identify which family is most productive on TandemFoilSet under the 30-min cap.

Live PRs (all `status:wip`, draft, awaiting first-run results):

| # | Student | Slug | Family | Predicted Δ on val_avg/mae_surf_p |
|---|---------|------|--------|-----------------------------------|
| 1496 | alphonse | pressure-channel-prioritized-loss | Loss | −5% to −12% |
| 1497 | askeladd | warmup-cosine-lr | Schedule | −4% to −10% |
| 1498 | edward | wider-mlp-ratio (2→4) | Architecture (MLP) | −3% to −8% |
| 1499 | fern | gradient-clipping-and-higher-lr | Optimizer / stability | −3% to −8% |
| 1500 | frieren | larger-hidden-dim (128→256) | Architecture (scale) | −3% to −7% |
| 1501 | nezuko | more-slices (64→128) | Architecture (capacity) | −2% to −6% |
| 1502 | tanjiro | per-sample-re-normalized-loss | Loss (IVW) | −3% to −9% |
| 1503 | thorfinn | surface-aware-output-head | Head specialization | −2% to −5% |

## Working hypotheses driving the round

1. **Loss-objective alignment is the cheapest lever.** The metric measures only surface-`p` MAE (L1); the training loss is uniform MSE over all 3 channels. PRs #1496 and #1502 attack this misalignment from two angles (channel weighting; heteroscedastic per-sample weighting).
2. **Slice-attention is softmax-gated and benefits from warmup.** PR #1497 tests this — if it wins, all future scheduling work should default to warmup + cosine.
3. **The baseline is plausibly capacity-bottlenecked.** ~2.4M params on 1500 samples × 100K+ nodes is small. PRs #1498, #1500, #1501 each grow capacity in a different sub-module (MLP, hidden, slices) — disentangles where capacity matters most.
4. **Gradient heterogeneity destabilizes early training.** PR #1499 is the standard stability cocktail (clip + higher LR) that often unlocks faster convergence on this kind of data.

## Potential next research directions (round 5+)

Once round-4 results land, the following are queued by family:

- **Compose top-1 + top-2** of round-4 winners (typically loss × schedule × architecture are orthogonal). E.g. `pressure-channel-weight + warmup-cosine + wider-mlp-ratio` should compose if all win.
- **Surface-loss reformulation: L1 directly** — switch `surf_loss` from MSE to L1 in normalized space (closer alignment to L1 MAE metric than channel weighting alone). Held in reserve; orthogonal to channel-weighting.
- **EMA of weights (`AveragedModel` with decay ~0.999) for checkpoint selection** — held in reserve; quick add-on once a strong recipe is established.
- **Domain-stratified or Re-bin-stratified sampling** — the `WeightedRandomSampler` is balanced across the three physical domains, but not stratified within domain by Re. PR #1502 tackles this via loss weighting; a sampler-side variant is the complement.
- **Augmentation by mesh subsampling / dropout** — randomly drop 10-20% of mesh nodes during training; cheap regularization, particularly relevant for variable-mesh inputs.
- **Stochastic depth on TransolverBlocks** — drop entire residual blocks with `p~0.1`; standard ViT regularization.
- **Fourier / random-feature encoding of node positions** — embed (x, z) coords into higher-frequency basis to help MLP capture sharp surface gradients.
- **Pre-norm vs post-norm experimentation** — Transolver uses pre-norm; verify this is optimal or try GroupNorm/RMSNorm variants.
- **Surface-pressure-only auxiliary loss** — add an explicit `surf_p_l1` term in addition to the existing surface MSE; targets the metric directly.
- **Transolver++ direction** — if `more-slices` wins, follow up with the eigen-slice / adaptive-slice variants from arXiv:2505.02107.
- **BF16/AMP for compute headroom** — when a winner is identified, enabling bf16 cheaply buys ~1.5× more epochs in 30 min, often a small free win.

## Operational notes

- **Hard isolation:** This launch is scoped to `willow-pai2g-48h-r4` only — do not consult any prior round's PRs, branches, or W&B history.
- **Cap discipline:** `SENPAI_TIMEOUT_MINUTES=30` and `SENPAI_MAX_EPOCHS` are hard upper bounds on every individual training run; the host-side harvest controls fleet total runtime.
- **Round-4 baseline calibration:** Since no baseline `val_avg/mae_surf_p` is committed yet, the **first completed run across this round** sets the de-facto baseline. Each PR's reproduce command includes a `baseline-control` invocation the student can use to establish a control if they want one — but otherwise, results will be ranked head-to-head once 2+ runs return.

## Living-doc status

Edit this file each cycle to keep it current with what's running and what's queued. Old PR references should be removed once they merge or close.
