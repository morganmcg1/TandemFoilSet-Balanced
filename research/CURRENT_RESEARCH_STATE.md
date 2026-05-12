# SENPAI Research State

- **Last updated:** 2026-05-12 ~21:25 (Smooth L1 huge win −26%; alphonse rebasing to stack; frieren/nezuko reassigned; 3 more PRs in flight)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 122.6395`** — PR #1418 (pressure channel weight 3×), 14 epochs, 0.66M param Transolver.  
Config: Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2), AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, batch_size=4, surf_weight=10, channel_weights=[1,1,3].  
See `BASELINE.md` for per-split details.

**Expected new baseline very soon:** alphonse's Smooth L1 (β=0.1) produced val_avg=90.585 (−26%) on old code. Rebasing now with channel_weights=[1,1,3] stacked. Likely to become the new baseline when re-run lands.

**Known issue:** `test_avg/mae_surf_p` is NaN (GT sample 000020 in test_geom_camber_cruise has Inf in pressure). Use `val_avg/mae_surf_p` for ranking. Multiple students now have NaN-skip fixes in `train.py:evaluate_split` — these will propagate to baseline when any of their PRs merge. See BASELINE.md for fix note.

## In-flight PRs

| PR | Student | Slug | Axis | vs. Baseline |
|----|---------|------|------|---|
| #1414 | alphonse | `smooth-l1-rebased` | Smooth L1 β=0.1 + channel weights stacked | SENT BACK (was **90.58** on old code → −26%; rebasing now) |
| #1421 | edward | `surf-only-channel-weight` | Decouple vol/surf channel weights | SENT BACK (was 124.96 vs 122.64) |
| #1424 | fern | `warmup-7e-4-clip` | Refined: 7e-4 + 2ep warmup + grad clip | WIP |
| #1432 | tanjiro | `wall-distance-rebased` | Wall-dist + channel weights stacked | SENT BACK (121.46 alone; rebasing) |
| #1435 | thorfinn | `unified-pos-ref16-nopad` | Unified pos encoding ref=16, no zero-pad | SENT BACK (ref=8 was +1.5% worse; signal on cruise OOD) |
| #1517 | askeladd | `ema-0.99-adaptive` | timm-style adaptive EMA (max=0.99) | SENT BACK (0.999 was +10.5% worse; horizon mismatch) |
| #1597 | frieren | `depth-6-layers` | Depth n_layers 5→6, width unchanged | WIP (new assignment) |
| #1598 | nezuko | `mlp-ratio-4-alone` | mlp_ratio 2→4 only, decoupled from slice_num | WIP (new assignment) |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse, only 9 epochs at 30-min cap
- #1429 nezuko slice-128-mlp-4: +6.97% worse, model output overflow at slice_num=128

## Current research focus

1. **🔥 Smooth L1 rebase (alphonse r2)** — most important pending run. If stacked result confirms ~90, this becomes new baseline by a factor of 1.36. ALL future PRs will be measured against it. Loss-shape hypothesis validated: L1 matches MAE eval criterion.
2. **Stack wall-distance on top of channel weights (tanjiro r2)** — small but real positive signal (−0.96%). Should stack additively with both Smooth L1 and channel weights.
3. **Refined EMA (askeladd r3)** — timm-style adaptive decay. Will confirm or kill EMA direction.
4. **Depth experiment (frieren #1597)** — n_layers=5→6, much cheaper than widening. Tests depth efficiency.
5. **MLP-ratio=4 alone (nezuko #1598)** — decoupled from slice_num doubling. Cheaper epoch time, tests post-attention capacity.
6. **Warmup + LR stability (fern)** — orthogonal to loss changes, should stack.
7. **Decoupled channel weights for vol/surf (edward)** — loss-weighting axis, orthogonal to loss shape.

## Key research insights so far

- **Biggest lever: Loss shape.** Smooth L1 (β=0.1) → −26% improvement by matching training criterion to eval metric (MAE). This dominates everything else tested.
- **Second lever: Channel weighting.** [1,1,3] on pressure → −9.5% improvement. Tells the optimizer to focus on the metric's preferred channel.
- **Architecture axes**: width-scaling is too expensive for 30-min budget (frieren). Depth and MLP-ratio yet to be confirmed.
- **Positional encoding**: signal on cruise OOD split but hurts re_rand — grid resolution too coarse at ref=8. Ref=16 pending.
- **EMA**: wrong decay for training horizon. Adaptive decay pending.
- **Wall-distance**: small but real input-feature improvement, stacking pending.

## Next research directions (from researcher-agent, 2026-05-12)

1. **Fourier positional encoding** (H2) — replace raw (x,z) with 64-dim RFF; untested, high potential for OOD generalization
2. **Per-sample adaptive loss scaling** (H1) — normalize MSE by per-sample std; interacts with Smooth L1 result
3. **Multi-resolution slice pooling** (H9) — vary slice_num per block; complementary to mlp_ratio test

## Operational notes

- Branch isolation: only inspect `icml-appendix-charlie-pai2g-48h-r2` and student branches for this launch.
- No W&B/wandb — local JSONL metrics only.
- Primary ranking metric: `val_avg/mae_surf_p` (test_avg is NaN-poisoned until scoring bug resolved).
- Epoch throughput: ~131s/epoch, ~14 epochs achievable in 30 min. T_max=20 leaves 6 epochs un-annealed.
- Pod rate-limit note: GitHub GraphQL rate-limits caused some heartbeat cycles to report "no work" incorrectly (iterations 22-25 for multiple students). Students auto-recover on next cycle.
