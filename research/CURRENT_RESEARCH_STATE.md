# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-13 ~06:45 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current merged improvements

| PR | What | val | test |
|----|------|-----|------|
| #1825 | MAE (L1) loss on Lion+EMA | **56.58** | **48.82** |
| #1781 | Lion optimizer lr=1e-4+EMA | 61.30 | 52.68 |
| #1607 | EMA decay=0.99 | 77.05 | 68.27 |
| #1541 | BF16 + scoring fix | 120.40 | 106.67 |
| #1386 | Fourier L=6 mf32 | 103.24 | 90.83 |
| #1357 | Huber δ=1.0 | 98.79 | 88.90 |
| #1367 | Dropout=0.2 + clip | 98.96 | 88.74 |

**Current compound:** Fourier + MAE loss + Dropout(0.1 default) + BF16 + EMA(0.99) + Lion(lr=1e-4)

**Note:** Both wins (#1781 Lion, #1825 MAE) still descend at epoch-16 cap. The model is NOT converged; longer budget is highest-EV improvement if allowed.

## Active experiments (8/8 students assigned)

| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#2001** | **askeladd** | **Lion β1 sweep: β1=0.95 (Arm1), β1=0.85 (Arm2) on Lion+MAE+EMA** | **WIP — new** |
| **#1999** | **fern** | **Cosine T_max tuning: T_max=16 (Arm1), T_max=16+eta_min=1e-5 (Arm2)** | **WIP — new** |
| #1932 | thorfinn | Lion lr=2e-4 (Arm1), lr=2e-4+wd=5e-4 (Arm2) | WIP |
| #1934 | alphonse | Width: n_hidden=192 (Arm1), n_hidden=256 (Arm2) | WIP |
| #1961 | tanjiro | FFN width: mlp_ratio=3 (Arm1), mlp_ratio=4 (Arm2) | WIP |
| #1857 | edward | EMA decay sweep: 0.995 (Arm1), 0.999 (Arm2) | WIP — stale |
| #1752 | nezuko | surf_weight sweep: 5 (primary), 7 (secondary) | WIP — stale |
| #1786 | frieren | Higher LR (1e-3/2e-3) on AdamW+EMA base | WIP — stale |

**Note on stale WIPs (#1857, #1752, #1786):** All on pre-Lion AdamW+EMA base. Current baseline is Lion+MAE (val=56.58). If they return results > 56.58 (which they almost certainly will), they should be sent back for re-running on the Lion+MAE compound. The directions themselves (EMA decay, surf_weight, LR) remain worth testing on the new base.

## Closed experiments this round

- **#1825 (askeladd):** MAE loss on Lion+EMA — **MERGED** val=56.58, test=48.82.
- **#1823 (fern):** wd=5e-4 on AdamW base — +1.84% val regression, tied test. Closed.
- **#1781 (thorfinn):** Lion optimizer — **MERGED** val=61.30, test=52.68.
- **#1761 (tanjiro):** n_layers=6 — both dropout arms regress (+4%). Compute-budget bound at 30-min cap.
- **#1604 (alphonse):** Asinh pressure transform — +7.5% regression. Huber+Asinh double-compress.
- **#1748 (edward):** EMA=0.99 + dropout=0.2 — regresses. EMA fills regularisation headroom.

## Key findings (all rounds)

1. **MAE (L1) loss on Lion+EMA:** −7.71% val / −7.34% test — largest gain on Lion base. Loss-side property (uniform per-node weighting) is independent of optimizer, compounds cleanly with Lion.
2. **Lion optimizer (lr=1e-4) + EMA:** −20.4% val / −22.8% test — decouples exploration (Lion sign updates) from integration (EMA averaging). Still descending at cap.
3. **EMA weight averaging (decay=0.99):** −22.1% val / −23.1% test — foundational for the session.
4. **Fourier positional encoding (max_freq=32, L=6):** −14.8% test — foundational input feature.
5. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window. Foundational.
6. **Huber loss (δ=1.0):** Superseded by MAE on Lion base. Huber's quadratic well competes with MAE's uniform per-node weighting.
7. **Depth=6 compute-budget bound:** n_layers=6 loses 2 epochs to overhead; can't beat n_layers=5 at 30-min cap.
8. **Asinh+Huber double-compression:** Both compress the high-Re tail; they compete rather than stack.

## Priority for current wave

**Highest priority (Lion+MAE base):**
- Lion β1 sweep (#2001 askeladd) — β1=0.9 from large-scale vision; small-data optimal may differ
- Cosine T_max tuning (#1999 fern) — current T_max=50 barely decays LR in 16 epochs
- Lion lr=2e-4 scaling (#1932 thorfinn) — lr doubling trend from 5e-5→1e-4 hasn't saturated
- Width expansion n_hidden=192/256 (#1934 alphonse) — model not converged at cap
- mlp_ratio=3/4 FFN expansion (#1961 tanjiro) — cheap capacity, orthogonal to depth/width

**Stale pre-Lion work (will need rerun on Lion+MAE base when results come in):**
- EMA decay sweep (#1857 edward)
- surf_weight sweep (#1752 nezuko)
- Higher LR on AdamW (#1786 frieren) — largely superseded by Lion; depends on result

## Potential next directions (post-current-wave)

- **Longer training budget** — both recent merges hit cap mid-descent; highest EV if wall-clock budget extended
- **Lion-no-EMA ablation** — for ICML appendix narrative on Lion+EMA synergy
- **Surface-only MAE + volume Huber** — MAE only where the metric is measured; Huber on well-behaved interior nodes
- **surf_weight tuning on Lion+MAE** — nezuko's sweep was on old base; optimal weight may differ with MAE loss
- **n_head=8 on Lion+MAE base** — previously failed on AdamW; Lion might enable this
- **Batch size=8** — doubled batch with Lion; reduces gradient noise at cost of fewer updates/epoch
- **OneCycleLR** — peak in middle; pairs with Lion's aggressive exploration
- **EMA decay tuning on Lion+MAE** — edward's sweep should run on current compound
