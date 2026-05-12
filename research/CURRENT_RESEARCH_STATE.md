# SENPAI Research State

- **Last updated:** 2026-05-12 (initial state on this advisor branch)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received yet on this branch

## Current research focus

Establish a first-round baseline on this fresh advisor branch by sweeping **eight distinct, single-axis hypotheses** against the default Transolver config in `train.py`. The 8-PR cohort will be ranked by `val_avg/mae_surf_p` and `test_avg/mae_surf_p`; the winner becomes the new merged baseline for subsequent rounds.

The current Transolver baseline (default in `train.py`) is:
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (≈2-3M params)
- AdamW lr=5e-4, wd=1e-4, CosineAnnealingLR(T_max=epochs), batch_size=4
- Loss = `vol_loss + 10 * surf_loss` with MSE in normalized space
- Cosine schedule + 30-min wall-clock cap → ~10-20 epochs feasible per run
- Primary ranking metric: `val_avg/mae_surf_p` (mean surface pressure MAE across 4 splits)

## In-flight round-1 assignments (8 students, 8 hypotheses, all `status:wip`)

| PR | Student | Slug | Axis |
|----|---------|------|------|
| #1414 | alphonse | `smooth-l1-loss` | Loss form (Huber β=0.1 instead of MSE) |
| #1418 | askeladd | `pressure-channel-weight` | Loss weighting (channel weights Ux/Uy/p = 1/1/3) |
| #1421 | edward | `surf-weight-25` | Loss balance (surf_weight 10 → 25) |
| #1424 | fern | `warmup-cosine-1e-3` | LR schedule (1ep linear warmup + peak 1e-3) |
| #1426 | frieren | `hidden-192-head-6` | Width (n_hidden 128→192, n_head 4→6) |
| #1429 | nezuko | `slice-128-mlp-4` | Capacity (slice_num 64→128, mlp_ratio 2→4) |
| #1432 | tanjiro | `wall-distance-feature` | Input feature (log min-dist to surface) |
| #1435 | thorfinn | `unified-pos-ref8` | Pos encoding (Transolver `unified_pos=True, ref=8`) |

Hypothesis selection rationale: span loss / training-recipe / architecture / feature / pos-encoding axes so the winner's gain is interpretable and orthogonal directions can be stacked in subsequent rounds.

## Potential next-wave research directions

A researcher-agent run is in flight to produce a longer list at `research/RESEARCH_IDEAS_2026-05-12_round1.md`. Seed directions to revisit after round 1 results:

1. **CFD-aware loss reformulations** — relative MAE (per-sample y_std normalized), frequency-aware loss for wake/shock structures, asymmetric loss on extreme-Re samples.
2. **Mesh-aware attention** — KNN local attention, multi-resolution slicing (different slice_num per block), graph-pooling between blocks.
3. **Physics priors** — Galilean-invariant features (velocity in foil-local frame), Re-dependent normalization, dimensional-analysis features (Mach proxy from Re, p/(0.5·ρ·U²)).
4. **Output head innovations** — separate per-channel heads, separate surface vs volume heads, predict log-magnitude residuals.
5. **Training tricks for the tight time budget** — EMA / SWA weights for evaluation, stochastic depth, mixed-precision (bf16 throughout), AdamW + Lookahead.
6. **Data-side levers** — curriculum on Re (low → high), hard-sample mining on the worst predicted samples in the previous epoch, mesh-node sub-sampling for cheaper training batches.

## Operational notes

- Branch isolation strict: do not touch any PR/branch outside `icml-appendix-charlie-pai2g-48h-r2` and the 8 assigned student PRs.
- No W&B/wandb usage on this branch — local JSONL metrics only under `models/<experiment>/metrics.jsonl`.
- Wake-up scheduled to survey PR progress and trigger review/merge as terminal `SENPAI-RESULT` markers appear.
