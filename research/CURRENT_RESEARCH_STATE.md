# SENPAI Research State

- **Date**: 2026-05-15
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received yet on this branch
- **Per-student GPU budget**: 1 × 96GB, 24h wall-clock per training run

## Current research focus

Round 1 dispatched (8/8 students busy). The Transolver baseline (`n_hidden=128`,
`n_layers=5`, `n_head=4`, `slice_num=64`, lr=5e-4, AdamW, MSE, cosine LR,
surf_weight=10, 50 epochs) has never been measured on this branch. Each
student tests a single orthogonal axis so we can stack winners in round 2.

## Round 1 hypotheses in flight

| PR | Student | Axis | One-line summary |
|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup (5%) + cosine annealing + epochs 50→100 |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) in normalized space instead of MSE |
| #3130 | edward | Width | n_hidden 128→192, n_head 4→6 |
| #3134 | fern | Slice count | slice_num 64→128 |
| #3136 | frieren | Surface weighting | surf_weight 10→25 |
| #3137 | nezuko | EMA | EMA weights (decay 0.999) for eval/test/checkpoint |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position |
| #3144 | thorfinn | Depth | n_layers 5→8 |

## Round 2 candidate hypotheses (from researcher-agent)

See `research/RESEARCH_IDEAS_2026-05-15_round1.md` for full details. Top
follow-ups to queue based on round 1 outcomes:

- **Surface loss with per-channel p-weight** (Idea 2 refinement) — if frieren wins, sweep `surf_weight ∈ {20, 30, 50}` × `p_surf_weight ∈ {2, 3, 5}` to find the optimal channel weighting.
- **Add grad clipping** (Idea 1 refinement) — if alphonse wins, layer `clip_grad_norm_(max_norm=1.0)` on top.
- **Larger model** (Idea 7) — if edward or thorfinn wins, push to `n_hidden=256, n_layers=8, slice_num=128, n_head=8` (~12M params, still fits in 96GB).
- **ReScaler / Re-conditioned output** (Idea 4) — `MLP(log_Re) → exp(scale)` applied to model output per sample. Directly addresses the order-of-magnitude per-sample y-std variation in `program.md`.
- **Domain-conditional FiLM** (Idea 13) — derive domain (single / raceCar tandem / cruise) from gap+AoA features; inject as FiLM shift+scale into LayerNorm.
- **bf16 + larger batch** (Idea 3) — `autocast(bfloat16)` + batch_size 4→8 for ~1.5–2× throughput, enabling more epochs.
- **Temperature annealing** (Idea 10) — anneal `PhysicsAttention.temperature` from 2.0→0.1 across training.
- **Separate prediction heads** (Idea 12) — replace shared `mlp2` with three (Linear→GELU→Linear) heads for Ux, Uy, p.
- **Stochastic depth** (Idea 14) — drop entire blocks with linearly-increasing per-layer probability; pairs well with deeper/wider models.
- **Separate surface/volume encoders** (Idea 15) — two MLPs sharing input, merged at the first attention block via `torch.where(is_surface)`.
- **Curriculum: low-Re first** (Idea 16) — modify `sample_weights` to upweight low-Re for first 30% of epochs.
- **OneCycleLR** (Idea 18) — alternative to warmup+cosine if alphonse fails.

## Stacking plan for round 2

Once round 1 results land, the expected merge order is:
1. EMA (nezuko) and warmup (alphonse) — orthogonal, low-risk, should stack first.
2. The best of {wider, deeper, more-slices} — pick the largest single-axis win.
3. Best loss-side change {Huber, surf_weight=25} — independent axis.
4. Fourier features (tanjiro) — likely independent of all the above.

Round 2 will then add a refined Idea 2 (per-channel p-weight) plus Idea 4
(ReScaler) on top of the stacked round 1 winners.

## Stop / pivot criteria

- If the best round 1 PR shows <2% improvement on `val_avg/mae_surf_p` vs baseline → pivot toward Idea 4 (ReScaler) and Idea 13 (domain FiLM) before further hyperparameter tuning, since the bottleneck is likely the way Re-scale variation is handled rather than capacity or schedule.
- If multiple PRs diverge or crash → that's a stability signal pointing toward Idea 1's grad clip + warmup before any other change is layered on.
