# SENPAI Research State

- **Date:** 2026-05-12 ~23:00
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none yet — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).

## Current baseline

**`val_avg/mae_surf_p` = 101.810** (L1 loss + n_layers=6 + mlp_ratio=4, PR #1358)
**`test_avg/mae_surf_p` = 91.708** (first reliable test numbers — NaN-fix merged)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 124.150 | 110.726 |
| geom_camber_rc | 112.699 | 99.692 |
| geom_camber_cruise | 76.570 | 66.879 |
| re_rand | 93.820 | 89.536 |
| **avg** | **101.810** | **91.708** |

## What we've learned

### Big wins (merged)
1. **mlp_ratio=4**: −5% (PR #1408)
2. **n_layers=6**: −9.4% (PR #1392)
3. **L1 loss**: −20.5% (PR #1358) ← dominant factor

### Dead ends
- Width (n_hidden=192): too slow/epoch
- Channel-weighted loss p×3 in normalized space: counterproductive
- n_head=8: +43% per-epoch cost, +15.7% worse
- slice_num=128: +12% per-epoch cost, +17.8% worse
- Warmup+lr=1e-3: too hot (even T_max=15 fails at ~175 s/epoch)
- EMA (decay=0.999): cold-start drag (+41% worse); requires 100+ epochs or non-random init
- lr=3e-4: undertraining at 30-min cap; cosine barely decays with T_max=50 (sent back to try lower LR + short T_max later)
- n_layers=7: +51% worse, 9 epochs/30min, reproducible NaN on test_geom_camber_cruise
- grad clip max_norm=1.0: too aggressive for L1 constant-magnitude gradients; retry at max_norm=10 in progress

### Key insights
1. **L1 loss is the dominant lever** (−20.5% alone)
2. **Budget is the constraint**: 30 min → ~12-13 epochs at n_layers=6, ~9 at n_layers=7. Any deeper/wider arch needs a batching change.
3. **n_layers=6 + mlp_ratio=4 is the sweet spot** for the 30-min budget
4. **LR/schedule coupling**: can't tune LR without also fixing T_max; lower LR needs shorter T_max to decay properly
5. **Gradient clip threshold**: with L1 loss and 1.18M params, max_norm=1.0 clips virtually all updates; threshold must be >>1.0 to be useful

## Active experiments

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1592 | Cosine T_max 50 → 14 (align to budget) | WIP |
| nezuko | #1593 | Gradient clipping (max_norm=10.0 re-run) | WIP (sent back) |
| tanjiro | #1634 | Batch size 4 → 8 (lower gradient noise) | New |
| fern | #1595 | Huber/SmoothL1 loss (beta=1.0) | WIP |
| edward | #1632 | Dropout=0.1 in attention (OOD regularization) | New |
| askeladd | #1622 | AdamW betas (0.9,0.999)→(0.95,0.99) | WIP |
| frieren | #1384 | surf_weight 10 → 25 (rebase needed) | WIP/CONFLICTING |
| thorfinn | #1525 | Fourier positional features (L=4) | WIP |

## Round 3/4 themes and open questions

1. **Schedule completion** (alphonse): T_max=14 = does aligning decay to actual epoch budget help?
2. **Gradient stability at scale** (nezuko retry): max_norm=10 — right clip threshold for L1 grad magnitudes?
3. **Batch size effect** (tanjiro): batch=4→8 — fewer, cleaner gradient steps; does this trade-off favor accuracy at our budget?
4. **OOD regularization** (edward): dropout=0.1 — improves hardest OOD splits?
5. **Huber vs L1** (fern): SmoothL1 beta=1.0 — middle ground between MSE and L1?
6. **AdamW betas for L1** (askeladd): (0.95, 0.99) vs default — better second-moment tracking for constant-magnitude gradients?
7. **surf_weight with L1** (frieren): 10 → 25 on new baseline, after rebase
8. **Fourier features** (thorfinn): input feature engineering — does L=4 Fourier encoding help OOD?

## Probable round 4/5 directions (conditional on round 3/4 signal)

- **Stacked winners**: Compound all orthogonal wins (e.g. optimal T_max + optimal batch + dropout + L1)
- **LR+schedule joint tuning**: lr=3e-4 + T_max=14 (if alphonse's T_max=14 shows benefit)
- **surf_weight exploration**: If surf_weight=25 fails, try surf_weight=5 (L1 may already rebalance)
- **Physical-space L1**: Compute loss in denormalized units for direct metric alignment
- **Data augmentation**: Mesh-coarsening, AoA jitter for OOD robustness
- **SWA with late-start**: Initialize from post-epoch-8 live weights (fixes EMA cold-start issue)
- **Lion optimizer**: Sign-based update complements L1 constant-gradient dynamics
- **mlp_ratio=8**: Bigger feedforward; note we're at width limit (n_hidden=128 × 8 = 1024-dim MLP). Per-epoch time TBD.

## Key constraints

- 30 min / run cap: n_layers=6 → ~12-13 epochs (~175 s/epoch with L1)
- Per-epoch time budget eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205 s/epoch)
- EMA eliminates: decay=0.999 with random init (cold-start drag, half-life 693 steps)
- test_avg/mae_surf_p is RELIABLE since PR #1358 NaN-fix: 91.708 is the first accurate test baseline
- Gradient clip threshold must be >>1.0 with L1 loss and 1.18M params (L2 grad norm >> 1.0 for constant ±1 grads)
