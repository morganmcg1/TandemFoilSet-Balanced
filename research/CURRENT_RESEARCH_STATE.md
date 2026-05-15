# SENPAI Research State

- **Date:** 2026-05-15 17:30
- **Launch:** willow-pai2i-48h-r1 (round 1 complete, round 2 starting)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~9–14 epochs achievable)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric now also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline
- **val_avg/mae_surf_p = 112.8295** (PR #3309, NaN fix)
- **test_avg/mae_surf_p = 106.5996** (all 4 splits, fully valid)
- W&B: `bpczoejx` (Huber) + `g48284pc` (NaN fix)

Full metrics in `BASELINE.md`.

## Merged PRs
| PR | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|----|-----------|---------------------|---------------------|
| #3159 | Huber loss δ=0.1 | 112.9001 | 115.7589 (3/4 splits) |
| #3309 | NaN fix (cruise test) | 112.8295 | **106.5996** (4/4 valid) |

## Round-2 WIP — 8/8 students assigned
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3305 | alphonse | Huber delta scan (0.05, 0.02) | WIP |
| #3317 | askeladd | Cosine T_max tuned (15 vs 12 ep) | WIP |
| #3359 | edward | Pressure channel-weighted surf loss (p=3×) | WIP — fresh |
| #3171 | fern | Split pressure output head (3× p weight) | WIP — rebase in progress |
| #3174 | frieren | L1 surf pressure + surf_weight=50 | WIP — rebase in progress |
| #3175 | nezuko | Cosine warmup (5-ep linear) | WIP |
| #3361 | thorfinn | slice_num 64→128 retry on Huber base | WIP — fresh |
| #3363 | tanjiro | AdamW β2=0.95 + grad clip 1.0 | WIP — fresh |

## Closed PRs (round 1 failures)
| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| #3162 | surf_weight=25 (MSE base) | 133.41 | Loss alignment >> surface weighting |
| #3188 | slice_num=128 (MSE base) | 134.74 | Predates Huber; being retried in #3361 |
| #3167 | OneCycleLR max_lr=1e-3 | 137.12 | 9-ep budget insufficient for one-cycle |
| #3180 | Wider model h=192 | 150.38 | ~1.6× slower/epoch, bottleneck not capacity |

## Key observations after round 1
- **Loss alignment wins big**: Huber(δ=0.1) gave ~16% improvement over MSE
- **Binding constraint**: 30-min timeout → 9–14 epochs, cosine T_max=50 never anneals
- **Capacity is NOT the bottleneck**: h=192 regressed 33%, consistent with schedule-starvation
- **Noisy training**: epoch-to-epoch val bouncing observed in 3+ runs (gradient stability issue)
- **Test metrics now valid**: cruise test NaN fixed in #3309; test_avg=106.60

## Active hypotheses being tested
1. **Smaller Huber delta** (alphonse): push more residuals into L1 regime
2. **T_max tuning** (askeladd): match cosine schedule to actual ~14-epoch budget
3. **Pressure channel weighting** (edward): 3× gradient on the scored channel
4. **Split pressure head** (fern): dedicated output branch for pressure
5. **L1 surf + surf_weight=50** (frieren): maximize surface pressure gradient
6. **Cosine warmup** (nezuko): warm into peak LR over 5 epochs
7. **slice_num=128 on Huber base** (thorfinn): capacity retrial on correct base
8. **AdamW stability** (tanjiro): β2=0.95 + grad clip 1.0

## Next directions after round 2
- Combine best schedule fix (askeladd/nezuko winner) + loss delta (alphonse)
- Unified positional encoding (`unified_pos=True`) — single config toggle
- Per-domain normalization — pressure ranges differ by domain
- Train-time symmetry augmentation — horizontal flip (needs camber-aware sign flips)
- Spectral/FNO-style blocks — if architecture is still bottleneck after schedule fix
- Multi-scale prediction (residual on coarse base)
