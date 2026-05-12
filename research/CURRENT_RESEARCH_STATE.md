# SENPAI Research State

- **Date:** 2026-05-12
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none yet — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).

## Current baseline

**`val_avg/mae_surf_p` = 101.810** (L1 loss + n_layers=5/mlp_ratio=2, PR #1358)
**`test_avg/mae_surf_p` = 91.708** (first reliable test numbers — NaN-fix merged)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 124.150 | 110.726 |
| geom_camber_rc | 112.699 | 99.692 |
| geom_camber_cruise | 76.570 | 66.879 |
| re_rand | 93.820 | 89.536 |
| **avg** | **101.810** | **91.708** |

Note: The merged train.py defaults are now L1 loss + n_layers=6 + mlp_ratio=4. Alphonse ran on n_layers=5/mlp_ratio=2 (stacked benefit not yet measured but expected to be larger).

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
- Warmup+lr=1e-3: too hot for deep/wide model; T_max=50 never decays

### Key insight
**L1 loss is the dominant lever so far** (−20.5% alone). The architecture wins stack onto this — n_layers=6 and mlp_ratio=4 haven't yet been confirmed on the L1 baseline. Per-epoch time constraints eliminated several otherwise promising ideas (n_head=8, slice_num=128, warmup+lr=1e-3).

## Active experiments

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1592 | Cosine T_max 50 → 14 (align to budget) | New |
| nezuko | #1593 | Gradient clipping (max_norm=1.0) | New |
| tanjiro | #1594 | Lower LR: 5e-4 → 3e-4 | New |
| fern | #1595 | Huber/SmoothL1 loss (beta=1.0) | New |
| edward | #1562 | n_layers 6 → 7 | WIP |
| askeladd | #1563 | EMA weights (decay=0.999) | WIP |
| frieren | #1384 | surf_weight 10 → 25 (rebase rerun) | WIP |
| thorfinn | #1525 | Fourier positional features (L=4) | WIP |

## Round 3 themes and open questions

1. **LR tuning for L1**: L1 gradients are constant-magnitude — is 5e-4 still optimal? (tanjiro testing 3e-4)
2. **Schedule completion**: T_max=50 means cosine barely decays in ~14 epochs — does aligning to budget help? (alphonse testing T_max=14)
3. **Gradient stability**: L1 oscillations with AdamW at 5e-4 — does clipping help? (nezuko testing)
4. **Huber vs L1**: Smooth convergence in fine regime vs pure L1 — is there a middle ground? (fern testing)
5. **Architecture on L1**: n_layers=7 (edward), EMA (askeladd) — do round 1 wins persist on L1 baseline?
6. **surf_weight with L1**: frieren testing surf_weight=25 on current arch
7. **Fourier features**: thorfinn testing — does input feature engineering help OOD?

## Probable round 4 directions (conditional on round 3 signal)

- **Stacked winners**: Compound all orthogonal wins (e.g. optimal LR + grad clip + EMA + L1)
- **Data augmentation**: Mesh-coarsening, AoA jitter for OOD robustness
- **Physical-space L1**: Compute loss in denormalized units for direct metric alignment (edward suggested)
- **Per-channel weighting in normalized L1**: Scale surface p higher in L1 space
- **Lower surf_weight with L1**: If L1 already emphasizes p more robustly, surf_weight=5 might be optimal
- **Architecture revisit**: n_layers=8 with grad clip + lower LR now that training is more stable

## Key constraints

- 30 min / run cap: n_layers=6 arch gives ~11-12 epochs (~175 s/epoch with L1, ~156 s with MSE)
- Per-epoch time budget eliminates: n_head=8 (+43%), slice_num=128 (+12%), warmup+lr=1e-3 (T_max never decays)
- test_avg/mae_surf_p is now RELIABLE (NaN-fix merged): 91.708 is the first accurate test baseline
