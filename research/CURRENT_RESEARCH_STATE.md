# SENPAI Research State

- **Date:** 2026-05-14 (latest advisor review cycle)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm)
- **Most recent human directive:** None — no GitHub issues from human team

## Current Best Baseline — PR #1405 (tanjiro, merged 2026-05-14)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **73.295** |
| test_avg/mae_surf_p (3-of-4 finite) | 63.911 |
| val_geom_camber_cruise | 54.423 |
| val_re_rand | 71.041 |
| val_single_in_dist | 79.894 |
| val_geom_camber_rc | 87.823 |

**Recipe:** `--epochs 25 --lr 2e-3 --loss l1` + bf16 autocast (now default) + OneCycleLR  
**Key insight:** bf16 reduces VRAM 42→33 GB. Combined with `--epochs 25`, the OneCycleLR schedule is still meaningful (LR ~3.3e-4) at the 30-min wall-clock cutoff (~19 realized epochs). Epochs 15-19 each contributed 3-7+ val MAE improvement — the "schedule tail" is where the gains are.

## Progress Path

| PR | Merged | val_avg | Improvement |
|----|--------|---------|-------------|
| MSE baseline | — | ~218 | — |
| #1355 pure L1 | ✅ | 94.291 | -57% |
| #1581 OneCycleLR @2e-3 | ✅ | 85.615 | -9.2% |
| #1405 bf16 + epochs=25 | ✅ | 73.295 | -14.4% |

## Active Research Focus

**The wall-clock budget is the primary constraint.** All three merged improvements reduced the "wasted training budget" problem:
1. L1 gave better gradient signal per step
2. OneCycleLR extracted more from the available steps
3. bf16+25ep extended the productive schedule window within the cap

The next questions are:
1. **How far can we push the schedule horizon?** (frieren #2913: try --epochs 30/40)
2. **Does depth help now that bf16 gives headroom?** (askeladd #2914: n_layers=6/7)
3. **Does EMA smooth the noisy endpoint?** (thorfinn #2915: ema_decay=0.999/0.9999)
4. **Does doubling batch size give more realized epochs?** (tanjiro #2916: bs=8, epochs=50)
5. **Do orthogonal improvements (gc=2.0, asinh-p680, cw=2, sw=5) survive on new baseline?** (fern #1602, edward #1605, nezuko #1625, alphonse #1582 — all sent back for re-runs)

## Students — Current State

| Student | PR | Hypothesis | State |
|---------|-----|-----------|-------|
| frieren | #2913 | OneCycle epoch-horizon sweep (--epochs 30/40) | WIP |
| askeladd | #2914 | Transolver depth n_layers=6/7 on bf16 baseline | WIP |
| thorfinn | #2915 | EMA model weights (decay 0.999/0.9999) | WIP |
| tanjiro | #2916 | bf16 batch_size=8 + extended schedule | WIP |
| fern | #1602 | gc=2.0 + OneCycle re-run on bf16 baseline | WIP |
| edward | #1605 | asinh-p680 + OneCycle re-run on bf16 baseline | WIP |
| nezuko | #1625 | surf_channel_weight cw=2 re-run on bf16 baseline | WIP |
| alphonse | #1582 | surf_weight=5 re-run on bf16 baseline | WIP |

All 8 students active — zero idle GPUs.

## Key Findings (cumulative)

- **Wall-clock bottleneck is CPU/dataloader, not GPU compute.** At bs=4 bf16 or fp32, per-epoch time is ~97-131 s — unchanged by mixed precision. VRAM headroom gained from bf16 is available for larger models/batches.
- **Schedule horizon is the dominant lever.** All three current winners are schedule improvements. Peak LR at 2e-3 is saturated (frieren's sweep confirmed). The OneCycleLR `total_steps` configuration determines how productive each realized epoch is.
- **val_geom_camber_rc is the hardest split** (87.82 at current best vs 54.42 for cruise). Experiments should track rc specifically — it's the best signal for generalization gains.
- **val_geom_camber_cruise improves fastest** — from 66.44 to 54.42 (-18%) while rc only went from 94.61 to 87.82 (-7%). Any approach that strongly differentiates between these two OOD splits should be flagged.
- **SAM and wider models both fail** due to 2× compute cost halving realized epochs under the cap.

## Potential Next Research Directions (round 4+)

1. **Schedule optimization**: Optimal epochs horizon under wall-clock cap; cosine decay shape (pct_start); warmup length
2. **Larger effective batch**: bs=8 with bf16 (~33 GB as tested) or gradient accumulation. May enable more realized steps per epoch if GPU-bound.
3. **Orthogonal compound**: Stack gc=2.0 + asinh-p680 + cw=2 + sw=5 if all validate on new baseline
4. **Data augmentation**: Geometric symmetry (z→-z, AoA→-AoA, Uy→-Uy) for free 2× effective training set
5. **Depth increase**: n_layers=6/7 now testable with bf16 (askeladd #2914)
6. **torch.compile**: Could reduce per-epoch time 15-25%, unlocking 22-23 realized epochs
7. **Domain re-weighting**: Current WeightedRandomSampler weights single/tandem-rc/tandem-cruise equally; reweighting toward the OOD-hard rc domain may help
8. **Per-domain normalization**: Different stats for single-foil vs tandem vs Re-randomized
9. **Learnable target transform**: Box-Cox or monotone network replacing fixed asinh scale
10. **Reduce eval frequency**: Eval on val every 2 epochs instead of 1 to save ~10% wall time

## Open Questions / Risks

- Does the val trajectory (still improving 3 pts/ep at epoch 19) mean we're just undertrained? Would a 2-hr run converge? Or is there a floor at ~55-60 val?
- The geom_camber_rc split consistently underperforms other splits under aggressive LR schedules (OneCycle) — this may indicate an architecture limitation for extreme-tandem geometry extrapolation, not just a training recipe issue.
- bf16 numerical stability: the eval fp64 accumulation safeguard (from tanjiro's PR) is important — ensure future PRs keep it.
