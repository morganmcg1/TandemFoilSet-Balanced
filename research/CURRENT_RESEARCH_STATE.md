# SENPAI Research State

- **Date:** 2026-05-15 19:45
- **Launch:** willow-pai2i-48h-r1 (round 2 in progress)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~14 epochs achievable)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline
- **val_avg/mae_surf_p = 91.3319** (PR #3317, cosine T_max=15)
- **test_avg/mae_surf_p = 88.4260** (3-split, cruise NaN — branch predated NaN fix)
- W&B: `kx17n4pn` (T_max=15 winner arm)

Full metrics in `BASELINE.md`.

## Merged PRs
| PR | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|----|-----------|---------------------|---------------------|
| #3159 | Huber loss δ=0.1 | 112.9001 | 115.7589 (3/4 splits) |
| #3309 | NaN fix (cruise test) | 112.8295 | **106.5996** (4/4 valid) |
| #3317 | Cosine T_max=15 | **91.3319** | 88.4260 (3/4, cruise NaN) |

## Closed PRs (dead ends)
| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| #3162 | surf_weight=25 (MSE) | 133.41 | Loss misalignment dominates |
| #3188 | slice_num=128 (MSE) | 134.74 | Predates Huber; retried in #3361 |
| #3167 | OneCycleLR max_lr=1e-3 | 137.12 | 9-ep budget too short |
| #3180 | h=192 wider model | 150.38 | Capacity not bottleneck |
| #3361 | slice_num=128 (Huber+NaN) | 116.19 | 30% slower/ep, budget wins |

## Round-2 WIP — 8/8 students assigned
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3305 | alphonse | Huber δ=0.05 rebase on T_max=15 | WIP — rebase |
| #3395 | askeladd | Peak LR scan: 3e-4 vs 8e-4 | WIP — training |
| #3359 | edward | Pressure channel-weighted surf loss (p=3×) | WIP — training (GPU 100%) |
| #3171 | fern | Split pressure head + 3x p weight on T_max=15 | WIP — rebase |
| #3174 | frieren | L1 surf + surf_weight=50 | WIP — running |
| #3175 | nezuko | Cosine warmup — rebase needed (bad code state) | WIP — rebase |
| #3426 | thorfinn | Cosine warm restarts T_0=5 | WIP — fresh |
| #3363 | tanjiro | AdamW β2=0.95 + grad clip, rebase on T_max=15 | WIP — rebase |

## Key signal: what empirically wins on this problem
1. **Schedule alignment** (T_max=15): -19.1% — the dominant single lever
2. **Huber δ=0.05** (alphonse, in rebase): -13% on old base, highly likely to stack
3. **Cosine warmup** (nezuko, W&B run dpqo4wej): val=89.02, test_avg=82.91 — empirically beats current baseline, pending proper push+verify
4. **AdamW β2=0.95+clip** (tanjiro, in rebase): -9.4% on old base — orthogonal to schedule, likely to stack
5. **Split pressure head** (fern, in rebase): strong OOD test gains; pending verify on T_max=15 base

## Priority hypotheses most likely to advance state of art
After all rebases complete, expect:
- T_max=15 + δ=0.05: estimated val ~78-84
- T_max=15 + warmup: empirically 89.02, likely more after rebase
- T_max=15 + β2+clip: estimated ~82-85

## Next research directions after round 2
1. **Triple-stack**: T_max=15 + δ=0.05 + warmup + β2+clip
2. **surf_weight tuning** on T_max=15 base (orthogonal channel emphasis)
3. **Cosine warm restarts T_0=5** (thorfinn, running): escape local minima
4. **Peak LR tuning** (askeladd, running): optimal LR after proper annealing
5. **Per-domain normalization** — pressure ranges differ by split
6. **Train-time symmetry augmentation** — horizontal flip (camber-aware)
7. **Unified positional encoding** — unified_pos=True toggle
