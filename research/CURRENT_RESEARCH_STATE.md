# SENPAI Research State

- **Date:** 2026-05-15 21:50
- **Launch:** willow-pai2i-48h-r1 (round 2 / round 3 starting)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~14 epochs achievable in fp32 at bs=4)
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
| #3359 | pressure_ch_w3 (no commit) | 133.32 | Edward iterated without pushing |
| #3395 | Peak LR scan (3e-4 vs 8e-4) | 94.18/94.46 | lr=5e-4 at basin minimum (both directions worse) |
| #3426 | Cosine warm restarts T_0=5 | 103.07 | +12.85% regression — 5-ep cycles too short to converge |

## Round-2 / Round-3 WIP — 8/8 students assigned
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3305 | alphonse | Huber δ=0.05 rebase on T_max=15 | WIP — rebase |
| #3460 | askeladd | bf16 autocast + batch_size=8 (NEW) | WIP — fresh |
| #3428 | edward | surf_weight scan (15, 20) on T_max=15 | WIP — clarified instructions sent |
| #3171 | fern | Split pressure head + 3x p weight, rebased on T_max=15 | WIP — training (GPU 100%) |
| #3174 | frieren | L1 surf + surf_weight=50 | WIP — running |
| #3175 | nezuko | Cosine warmup on T_max=15 base | WIP — rebase |
| #3459 | thorfinn | EMA of model weights (decay=0.999) (NEW) | WIP — fresh |
| #3363 | tanjiro | AdamW β2=0.95 + grad clip, rebased on T_max=15 | WIP — re-running |

## Key signal: what empirically wins on this problem
1. **Schedule alignment** (T_max=15): -19.1% — the dominant single lever
2. **Huber δ=0.05** (alphonse, in rebase): -13% on old base, highly likely to stack
3. **Cosine warmup** (nezuko, W&B run dpqo4wej): val=89.02, test_avg=82.91 — empirically beat OLD baseline, pending rebased run on T_max=15
4. **AdamW β2=0.95+clip** (tanjiro, re-running on T_max=15): -9.4% on old base; on-base run pending after fresh rebase
5. **Split pressure head** (fern, training now): -6.2% test on OLD base, strong on OOD splits
6. **lr=5e-4 is at basin minimum** (askeladd PR #3395 confirmed): both 3e-4 and 8e-4 regress
7. **Warm restarts wrong tool for 14-ep budget** (thorfinn PR #3426 confirmed)

## Pattern emerging: timeout-bound
Every recent result reports `best_epoch = final_epoch`, meaning the model is still improving when wall-clock cuts off. This makes throughput a first-order lever, not just an engineering detail. **askeladd's new bf16+bs8 assignment directly targets this**.

## Priority hypotheses most likely to advance state of art
After rebases complete, expect (compounded onto baseline 91.33):
- T_max=15 + δ=0.05 (alphonse): estimated val ~78-84
- T_max=15 + warmup (nezuko): empirically 89.02 last time, likely lower with proper rebase
- T_max=15 + β2+clip (tanjiro re-running): estimated ~82-85
- T_max=15 + EMA (thorfinn): expected ~88-90, smoother trajectory
- T_max=15 + split head + 3x p (fern, training): test_avg improvement likely
- T_max=15 + bf16/bs=8 (askeladd): more epochs → val likely 86-88 from convergence alone

## Next research directions after round 3
1. **Triple-stack**: T_max=15 + δ=0.05 + warmup + β2+clip + EMA — once individual results validate
2. **surf_weight tuning** on T_max=15 (edward arm) — orthogonal channel emphasis
3. **Per-domain normalization** — pressure ranges differ by split (8e-4 LR result showed asymmetry)
4. **Train-time symmetry augmentation** — horizontal flip (camber sign + Uy sign flip)
5. **Unified positional encoding** — Transolver `unified_pos=True` toggle
6. **Test-time augmentation (TTA)** with flip+average
7. **Stochastic Weight Averaging (SWA)** during final epochs
8. **Layer-wise LR decay** — different LR for different depths
9. **Larger batch via gradient accumulation** if bf16 path doesn't pan out
