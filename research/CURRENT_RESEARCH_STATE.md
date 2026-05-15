# SENPAI Research State

- **Date:** 2026-05-15 15:30
- **Launch:** willow-pai2i-48h-r1 (round 1, 48h horizon)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Students (8):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn (1 GPU each)
- **Budget per run:** 30 min wall clock, 50 epochs max (~14 epochs achievable at current speed)
- **Latest direction from human team:** None (no open Issues for this launch)

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) — the equal-weight mean surface pressure MAE across the 4 val/test splits on TandemFoilSet. Lower is better.

## Current best baseline
- **val_avg/mae_surf_p = 112.9001** (PR #3159, alphonse, Huber loss delta=0.1)
- W&B run: `bpczoejx`
- Merged to advisor branch

Full metrics in `BASELINE.md`.

## Round-1 status

### Merged / Closed
| PR | Student | Hypothesis | Result |
|----|---------|-----------|--------|
| #3159 ✓ MERGED | alphonse | Huber loss delta=0.1 | **112.9001** — new baseline |
| #3188 ✗ CLOSED | thorfinn | slice_num 64→128 | 134.7389 — did not beat Huber baseline |

### In flight (WIP)
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3167 | edward | OneCycleLR max_lr=1e-3 | WIP — just un-drafted, should start soon |
| #3171 | fern | Split pressure head (3× p weight) | WIP — just un-drafted |
| #3174 | frieren | L1 loss + surf_weight=50 | WIP — just un-drafted |
| #3175 | nezuko | Cosine warmup (5-ep linear) | WIP — just un-drafted |
| #3180 | tanjiro | Wider model (hidden=192, slice_num=96) | WIP — just un-drafted |
| #3305 | alphonse | Huber delta scan (0.05, 0.02) | WIP — fresh assignment |
| #3309 | thorfinn | NaN bug fix in evaluate_split | WIP — fresh assignment |

Note: PRs #3167–#3180 were created earlier but stuck in draft state (student pods skip draft PRs). Un-drafted at 15:30 UTC — students should pick up on next poll cycle.

## Current research focus
Round 1 explores the recipe-level lever set. The Huber(delta=0.1) win (PR #3159) establishes a clear hypothesis: **metric alignment between loss and evaluation drives significant improvement**. Key themes being tested:

1. **Loss alignment** (merged: Huber; in-flight: alphonse's delta scan, frieren's L1+high surf_weight)
2. **LR schedule** (in-flight: edward's OneCycleLR, nezuko's cosine warmup)
3. **Output head architecture** (in-flight: fern's split pressure head)
4. **Capacity** (in-flight: tanjiro's wider model)
5. **Infrastructure** (in-flight: thorfinn's NaN fix — unblocks real test_avg metrics)

## Key insight from round-1 so far
- Huber loss alignment is the largest lever found so far (~16% improvement)
- **Binding constraint**: 30-min timeout → ~14 epochs only. T_max=50 means LR is still at ~82% of peak when training stops. The cosine schedule never anneals. This is a major opportunity: any experiment that uses a schedule fitted to ~14 epochs should benefit.
- val_geom_camber_cruise (75.85) is already well-predicted; val_geom_camber_rc (143.41) and val_single_in_dist (134.46) are hardest and drive the average up.

## Infrastructure issue (partially resolved)
`.test_geom_camber_cruise_gt/000020.pt` has 761 `-inf` values in pressure channel.  
→ `test_geom_camber_cruise/mae_surf_p` = NaN for ALL students.  
→ Fix in thorfinn PR #3309: defensive `y_finite` masking in `evaluate_split`.  
Val metrics are unaffected. Once merged, real `test_avg/mae_surf_p` (4 splits) will be available.

## Themes / next directions to consider
- **T_max / schedule tuning**: Most important remaining lever. OneCycleLR (edward) and warmup (nezuko) both address this. If neither wins, explicitly tune T_max=14.
- **Huber delta refinement**: alphonse testing delta=0.05 and 0.02. Smaller delta → more L1 regime → better MAE alignment.
- **Compound improvements**: Once the LR schedule and loss are both dialled in, test capacity increases (slice_num=128, hidden=192) on top of the combined base.
- **Surface-specific attention**: Cross-attention from surface to volume slice tokens.
- **Train-time symmetry augmentation** (horizontal flip + sign flips on AoA/Ux) — needs care due to camber.
- **Per-channel normalization**: Current stats are global; pressure ranges differ by domain.
- **Spectral / FNO-style operator blocks** alongside Transolver attention.
- **Relative MSE loss**: Normalises large-magnitude (raceCar) errors relative to the target scale.
