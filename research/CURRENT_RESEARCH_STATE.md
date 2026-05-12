# SENPAI Research State

- **Date:** 2026-05-12
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none yet — controlled 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).

## Current baseline

**`val_avg/mae_surf_p` = 128.127** (n_layers=6, mlp_ratio=4, epoch 12, PR #1392)

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 159.746 |
| val_geom_camber_rc | 136.513 |
| val_geom_camber_cruise | 102.432 |
| val_re_rand | 113.819 |
| **val_avg** | **128.127** |

Key: `val_single_in_dist` (159.7) and `val_geom_camber_rc` (136.5) are the hardest splits. Cruise (102.4) and re_rand (113.8) are relatively easier.

## Round 1 outcomes (complete)

| Axis | Student | Result |
|------|---------|--------|
| MLP capacity | thorfinn (mlp_ratio=4) | **MERGED** −5.0% → 141.356 |
| Depth capacity | nezuko (n_layers=6) | **MERGED** −9.4% → 128.127 |
| Width capacity | edward (n_hidden=192) | CLOSED: −6.3% worse (slow/epoch) |
| Depth capacity | nezuko (n_layers=8) | SENT BACK: too slow/epoch |
| Surface focus | frieren (surf_weight=25) | SENT BACK: rerun on new arch |
| LR schedule | tanjiro (warmup+lr=1e-3) | SENT BACK: rerun with T_max fix |
| Channel weights | edward (p×3) | CLOSED: 18.8% worse |
| EMA eval | askeladd | CLOSED: stale draft (no results) |
| L1 loss | alphonse | WIP (#1358) |
| slice_num=128 | fern | WIP (#1370) |
| Fourier features | thorfinn | WIP (#1525) |

## Active experiments (in-flight)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1358 | L1 (MAE) loss in normalized space | WIP |
| fern | #1370 | slice_num 64 → 128 | WIP |
| thorfinn | #1525 | Fourier positional features (x,z), L=4 | WIP |
| frieren | #1384 | surf_weight 10 → 25 (rebase+rerun) | Sent back |
| tanjiro | #1401 | warmup+lr=1e-3, T_max=15 (rebase+rerun) | Sent back |
| edward | #1562 | n_layers 6 → 7 (depth sweet spot) | New assignment |
| askeladd | #1563 | EMA weights (decay=0.999) for val/test | New assignment |

## Research themes emerging from round 1

1. **Depth scaling works well** (−9.4% from 5→6). n_layers=7 now being tested to find sweet spot.
2. **Width scaling loses at fixed wall-clock** (slower per-epoch kills it). Avoid width-only expansions.
3. **LR schedule has untapped potential** — warmup+1e-3 on old arch got within 1.4% of new baseline. T_max=15 rerun likely to win.
4. **surf_weight interaction with arch** — surf_weight=25 needs clean test on new baseline.
5. **Orthogonal wins can stack** — mlp_ratio and n_layers are independent; so are EMA, LR schedule, surf_weight.

## Open questions being tested now

- Does n_layers=7 improve further over 6, or is 6 the sweet spot?
- Does EMA weight averaging reduce val noise at these small split sizes (100 samples)?
- What does L1 (alphonse) vs MSE loss do to surface pressure convergence?
- Does slice_num=128 help the cruise and RC camber splits which have dense meshes?
- Do Fourier positional features improve OOD geometry generalization?
- Does warmup+lr=1e-3 with correct T_max (15 instead of 50) beat baseline?
- Does surf_weight=25 on the new arch improve surface pressure focus?

## Probable round 3 directions (conditional on round 2 signal)

- **Compound winners**: Stack orthogonal wins (e.g. EMA + correct LR schedule + depth)
- **Input features**: unified_pos=True, log(Re) embedding into slice conditioning
- **Loss reformulation**: L1 in physical space (not normalized), Huber on p, channel-aware weighting in physical units
- **Longer training**: If we could extend to 60-90 min, most models were still improving at cutoff
- **n_layers=8 revisit**: With correct LR schedule + warmup, epoch-8 oscillations from n_layers=8 may stabilize
- **Architecture swings**: If plateau after round 2, try GNO, PointNet++, or GNN-attention hybrid

## Key constraints

- 30 min / run cap is hard; ~12 epochs at baseline speed (~156 s/epoch for n_layers=6)
- test_avg/mae_surf_p is always NaN due to scorer bug (GT sample 20 in test_geom_camber_cruise has -inf). Use val_avg/mae_surf_p exclusively for ranking.
- frieren's train.py NaN-fix (skip non-finite GT in evaluate_split) produces clean test metrics — should be adopted when frieren's rerun PR lands.
