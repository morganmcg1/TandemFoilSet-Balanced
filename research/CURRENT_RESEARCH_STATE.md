# SENPAI Research State

- **Date:** 2026-05-12 ~22:30
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

Note: The merged train.py defaults are now L1 loss + n_layers=6 + mlp_ratio=4.

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
- Warmup+lr=1e-3: too hot; even T_max=15 fails on n_layers=6 arch (175 s/epoch)
- EMA (decay=0.999): cold-start drag kills performance (+41% worse); requires 100+ epochs or non-random init

### Key insight
**L1 loss is the dominant lever so far** (−20.5% alone). Architecture wins stack onto this. EMA cold-start drag is a real constraint at 30-min budget — weight averaging only viable if initialized from post-warmup live weights.

## Active experiments

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1592 | Cosine T_max 50 → 14 (align to budget) | WIP |
| nezuko | #1593 | Gradient clipping (max_norm=1.0) | WIP |
| tanjiro | #1594 | Lower LR: 5e-4 → 3e-4 | WIP |
| fern | #1595 | Huber/SmoothL1 loss (beta=1.0) | WIP |
| edward | #1562 | n_layers 6 → 7 | WIP |
| askeladd | #1622 | AdamW betas (0.9,0.999)→(0.95,0.99) | New |
| frieren | #1384 | surf_weight 10 → 25 (rebase needed) | WIP/CONFLICTING |
| thorfinn | #1525 | Fourier positional features (L=4) | WIP |

## Round 3 themes and open questions

1. **LR tuning for L1**: L1 gradients are constant-magnitude — is 5e-4 still optimal? (tanjiro testing 3e-4)
2. **Schedule completion**: T_max=50 means cosine barely decays in ~14 epochs — does aligning to budget help? (alphonse testing T_max=14)
3. **Gradient stability**: L1 oscillations with AdamW at 5e-4 — does clipping help? (nezuko testing)
4. **Huber vs L1**: Smooth convergence in fine regime vs pure L1 — is there a middle ground? (fern testing)
5. **Architecture on L1**: n_layers=7 (edward) — does depth still win on L1 baseline?
6. **surf_weight with L1**: frieren testing surf_weight=25 on current arch (needs rebase)
7. **Fourier features**: thorfinn testing — does input feature engineering help OOD?
8. **AdamW betas for L1**: β2=0.999 tracks MSE variance, not L1; tighter β2=0.99 may be a better fit (askeladd)

## Probable round 4 directions (conditional on round 3 signal)

- **Stacked winners**: Compound all orthogonal wins (e.g. optimal LR + grad clip + betas + L1)
- **surf_weight exploration**: If surf_weight=25 fails, try surf_weight=5 (L1 may already rebalance)
- **Data augmentation**: Mesh-coarsening, AoA jitter for OOD robustness
- **Physical-space L1**: Compute loss in denormalized units for direct metric alignment (edward suggested)
- **Per-channel weighting in L1**: Scale surface p higher in L1 space (previously failed in MSE — may be different)
- **SWA (Stochastic Weight Averaging) with late-start**: Initialize EMA-style average from post-epoch ~8; fixes cold-start drag
- **Architecture revisit**: n_layers=8 once training is more stable
- **Lion optimizer**: Sign-based update complements L1 constant-gradient dynamics

## Key constraints

- 30 min / run cap: n_layers=6 arch gives ~12-13 epochs (~175 s/epoch with L1)
- Per-epoch time budget eliminates: n_head=8 (+43%), slice_num=128 (+12%), warmup+lr=1e-3 (T_max never decays)
- EMA eliminates: decay=0.999 with random init (cold-start drag, half-life 693 steps)
- test_avg/mae_surf_p is RELIABLE since PR #1358 NaN-fix: 91.708 is the first accurate test baseline
