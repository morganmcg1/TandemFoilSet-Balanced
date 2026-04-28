# SENPAI Research State

- 2026-04-28 23:45
- No directives yet from the human researcher team
- **Current research focus**: Round 3/4 — all experiments now include `surf_weight=20` and properly tuned T_max/epochs. Key leads: per-sample normalized loss (PR #747 showed −14.3% at sw=10; combining with sw=20 via PR #845), surf_weight=40 (PR #849), lower LR (PR #812), zero weight decay (PR #813), per-channel p weighting (PR #838), Re-stratified sampling (PR #839), wider hidden dim (PR #735 resubmit). **Current official baseline: 128.83. Target: break sub-100 by combining per-sample norm + surf_weight=20.**

## Current Baseline

- **PR #738** — Surface loss weight 10 → 20
- **val_avg/mae_surf_p = 128.8320** (lower is better)
- Architecture: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Training: `lr=5e-4`, `surf_weight=20.0`, `weight_decay=1e-4`, `batch_size=4`, `--epochs 14`

## Experiment History Summary

| PR | Student | Hypothesis | Result | Decision |
|----|---------|------------|--------|----------|
| #738 | edward | surf_weight 10→20 | **128.83** | MERGED (baseline) |
| #735 | alphonse | n_hidden 128→256 | 140.06 → sent back | Respin: n_hidden=192, n_layers=6, T_max=11 |
| #740 | fern | n_layers 5→7 | 147.21 → sent back | Respin with sw=20, epochs=12 |
| #741 | frieren | mlp_ratio 2→4 | 141.54 | CLOSED — too slow per epoch |
| #736 | askeladd | slice_num 64→128 | 135.96 | CLOSED — sent back, subsequent closed |
| #746 | tanjiro | n_head 4→8 | 128.96 (tied baseline) | CLOSED — new baseline (110.37) made it irrelevant |
| #747 | thorfinn | per-sample norm loss (sw=10) | 110.37 | AWAITING REBASE (−14.3% win, rebasing to advisor branch) |
| #802 | edward (r5) | bf16 + batch_size=8 | 129.14 | CLOSED (r5 track) |
| #812 | edward | LR 5e-4→2e-4 w/ sw=20, epochs=14 | WIP | — |
| #813 | frieren | zero weight decay w/ sw=20, epochs=14 | WIP | — |
| #838 | nezuko | Per-channel p weighting: p_weight=5.0 | WIP | — |
| #839 | tanjiro | Re-stratified mini-batch sampling | WIP | — |
| #845 | fern | per-sample norm loss + sw=20 | WIP | HIGH PRIORITY — combining #747 win with baseline |
| #849 | askeladd | surf_weight 20→40 + T_max=15 | WIP | — |

## Active WIP Experiments (Round 3 / Round 4)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #735 | alphonse | Wider hidden dim 128→256 (+ surf_weight=20, T_max=10) | wip (resubmitted) |
| #747 | thorfinn | Per-sample normalized loss (rebasing onto current advisor branch) | wip (rebasing) |
| #812 | edward | Lower LR 5e-4 → 2e-4 with surf_weight=20, --epochs 14 | wip |
| #813 | frieren | Zero weight decay (weight_decay=0.0) with surf_weight=20, --epochs 14 | wip |
| #838 | nezuko | Per-channel pressure weighting: boost p-channel loss weight | wip |
| #839 | tanjiro | Re-stratified mini-batch sampling to balance regime coverage | wip |
| #845 | fern | Per-sample norm loss + surf_weight=20 (combine #747 winner with baseline) | wip |
| #849 | askeladd | surf_weight 20→40 + T_max=15 to maximize surface pressure convergence | wip |

## Key Technical Findings

1. **Per-split heterogeneity is substantial** — 57-point spread (val_single_in_dist=157 vs val_geom_camber_cruise=100). This appears across all experiments and reflects training data imbalance across Re regimes.
2. **surf_weight=20 is clearly better than 10** — established by PR #738. All new experiments must include `--surf_weight 20.0`.
3. **T_max must match achievable epoch count** — critical misconfiguration lesson. Always set T_max = expected_epochs_in_budget (set via `--epochs N`).
4. **Per-sample normalized loss (PR #747) is the biggest win so far** — 110.37 vs baseline 128.83 (−14.3%), but run at sw=10. Combining with sw=20 is the highest-priority experiment (PR #845, fern).
5. **Depth/width hurt under wall-clock budget** — both n_layers=7 and n_hidden=256 lose because they're 40-100% slower per epoch, limiting achievable epochs. Mixed design (n_hidden=192, n_layers=6) is being retested.
6. **NaN in test_geom_camber_cruise** — sample 20 has -inf ground truth; val splits unaffected.

## Potential Next Research Directions

### Tier 1 — High confidence, directly motivated by findings
1. **Per-sample norm + surf_weight=20** — in-flight PR #845 (fern). If this wins, it becomes the new gold standard.
2. **Combine winning Round 3 results** — if lower-LR (#812) AND zero-wd (#813) both win, combine them; also combine with per-sample norm if #845 wins.
3. **Per-sample norm + lower LR (2e-4)** — per-sample normalization changes effective loss scale; may benefit from lower LR.
4. **surf_weight=40 (PR #849)** — direct extrapolation of the surf_weight=10→20 gain. High chance of further improvement.

### Tier 2 — Architecture and optimization
5. **n_layers=6 Pareto compromise** — 5 is baseline, 7 too slow; 6 layers (~155s/epoch → ~11-12 epochs in 30 min) could be the sweet spot.
6. **CosineAnnealingWarmRestarts** — cyclic restarts may help escape local minima within the 30-min budget.
7. **Gradient clipping** — reduce oscillation on surface-pressure loss spikes.
8. **Mixed capacity: n_hidden=192, n_layers=6** — being tested by PR #735 alphonse respin.

### Tier 3 — Loss and feature innovation
9. **Learned uncertainty weighting (Kendall et al.)** — adaptive channel weights instead of fixed scalars.
10. **Input feature enrichment** — add curvature, arc-length gradients, local normals as extra node features.
11. **Gradient-based surface emphasis** — weight loss by local pressure gradient magnitude.
12. **Physics-consistency auxiliary loss** — soft enforcement of continuity equation.

### Tier 4 — Bold structural changes (if plateau persists)
13. **GNN-augmented architecture** — replace or augment Transolver with GNN layers using mesh connectivity.
14. **Completely different architecture** — FNO, DeepONet, or U-Net style encoder-decoder.
15. **Ensemble / checkpoint averaging** — average final K checkpoints (free post-training gain).
