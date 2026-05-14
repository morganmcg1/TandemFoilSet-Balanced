# SENPAI Research State

- **Date:** 2026-05-14 15:57 UTC
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #2936 (askeladd, merged 2026-05-14 15:54 UTC)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **72.694** |
| test_avg/mae_surf_p | 63.367 |
| val_geom_camber_cruise | 53.237 |
| val_re_rand | 71.144 |
| val_single_in_dist | 82.067 |
| val_geom_camber_rc | 84.326 |
| **epochs realized** | **20** |

**Recipe:** `--epochs 25 --lr 2e-3 --loss l1 --eval_every 2` + bf16 autocast + OneCycleLR  
**Key insight:** eval_every=2 skips every other val pass (~7 s/skip) → 1 extra tail epoch (ep20 vs ep19). LR at best epoch = 2.34e-4 (still in productive fine-tuning zone).

## Progress Path

| PR | Merged | val_avg | Improvement |
|----|--------|---------|-------------|
| MSE baseline | — | ~218 | — |
| #1355 pure L1 | ✅ | 94.291 | -57% |
| #1581 OneCycleLR @2e-3 | ✅ | 85.615 | -9.2% |
| #1405 bf16 + epochs=25 | ✅ | 73.295 | -14.4% |
| #2936 eval_every=2 | ✅ | **72.694** | -0.82% |

## Active Research Focus

**Wall-clock budget is the binding constraint.** Every improvement so far has worked by extracting more productive training steps within the 30-min cap. The approach is:

1. **Reduce overhead** (eval_every=2: saves ~7s/eval) → more training steps
2. **Unlock more realized epochs** within the cap → torch.compile (next)
3. **Make each training step more valuable** (architecture/loss/data quality)

Currently, 20 epochs are realized. The model is still strongly converging at ep20 (LR=2.34e-4, 12+ pts/epoch improvement in the tail). The theoretical floor under this cap is ~3-4 more epochs (if we could run to ep24-25) worth potentially 15-40 more val pts.

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| askeladd | #2954 | torch.compile throughput (new, just assigned) | WIP |
| fern | #2945 | Cruise-only z-flip augmentation | WIP |
| frieren | #2913 | OneCycle epoch-horizon sweep (--epochs 30/40) | WIP ⚠ stale |
| thorfinn | #2915 | EMA model weights (decay 0.999/0.9999) | WIP ⚠ stale |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP ⚠ stale |
| edward | #1605 | asinh-p680 + OneCycle re-run on bf16 baseline | WIP ⚠ needs rebase |
| nezuko | #1625 | surf_channel_weight cw=2 re-run on bf16 baseline | WIP ⚠ needs rebase |
| alphonse | #1582 | surf_weight=5 re-run on bf16 baseline | WIP ⚠ stale |

6 stale/conflict PRs were blocked by GraphQL rate limit (user token exhausted fleet-wide). Rate limit reset at 15:49:57 UTC — pods should resume shortly.

## Key Findings (cumulative)

- **Wall-clock bottleneck is CPU/dataloader, not GPU.** Per-epoch ~91-98 s; eval adds ~7 s/call. 20 epochs in 30 min at current recipe.
- **Schedule horizon is the dominant lever.** All 4 merged improvements are schedule/efficiency improvements. Peak LR at 2e-3 saturated (frieren's sweep confirmed).
- **eval_every cost was only ~7 s** (not 20 s assumed): buying ~1 extra epoch per 19 total. The hypothesis was right but the magnitude was smaller than expected.
- **OneCycleLR tail is highly productive.** Ep18→20 contributes 12+ pts. The model is still strongly converging at cap. Every extra epoch is valuable.
- **val_geom_camber_rc is the hardest split** (84.33 vs 53.24 for cruise at current best). Largest absolute headroom; best signal for generalization gains.
- **val_single_in_dist regressed slightly** vs PR #1405 (82.07 vs 79.89). Likely seed variance; needs tracking.
- **Depth and width scale fail** under this budget: n_layers=6/7 realized only 14 vs 20 epochs (compute killed the tail).
- **Gradient clipping hurts** on 25-ep OneCycleLR (the schedule itself provides the regularisation).
- **z-flip augmentation**: full-mesh flip failed (raceCar one-sided mesh). Cruise-only conditional flip in progress (fern #2945).

## Negative Results Confirmed

| Idea | PR | Δ val | Why it failed |
|------|----|-------|---------------|
| grad_clip=2.0 | #1602 | +5.1% | Over-regularises on 25-ep schedule |
| n_layers=6/7 | #2914 | +27-35% | Compute kills realized epochs |
| z-flip (all meshes) | #2935 | +20.4% | raceCar one-sided topology breaks symmetry |
| eval_every=2 inter-arm | #2936 ctrl | +0.04% | Seed noise; val gain marginal |

## Potential Next Directions (not yet assigned)

1. **torch.compile** (assigned: askeladd #2954) — more realized epochs, no HPs
2. **OneCycleLR pct_start tuning** (H2: 0.05 vs 0.3 warmup fraction) — shorter warmup → more productive high-LR steps
3. **Variance-penalized loss** (H3: L = mean_err + 0.5*std_err on surface nodes) — targets rc split's persistent spatial-outlier errors
4. **Per-domain normalization** (H5: domain-specific y_mean/y_std for loss) — addresses 4× pressure scale difference between cruise and raceCar
5. **Domain re-weighting** (H9: increase cruise/tandem weight in sampler) — may help OOD rc/re_rand splits
6. **Compound stacking**: once SW=5, CW=2, asinh-p680 validate on new baseline, combine the best 2-3

## Open Questions

- Does torch.compile give meaningful speedup with variable-padded mesh inputs? (askeladd #2954 will answer)
- Does frieren's epoch horizon 30/40 experiment reveal that OneCycleLR configured for more epochs improves results even when wall-clock caps actual epochs?
- Does EMA smooth noisy tail-epoch checkpoints (thorfinn #2915)?
- Can the cruise-only z-flip reduce data hunger for the cruise split specifically?
