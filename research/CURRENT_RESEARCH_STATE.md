# SENPAI Research State

- **Date:** 2026-05-15 23:40
- **Launch:** willow-pai2i-48h-r1 (round 3 in progress)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~14 epochs achievable in fp32 at bs=4)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline
- **val_avg/mae_surf_p = 91.3319** (PR #3317, cosine T_max=15)
- **test_avg/mae_surf_p = 88.4260** (3-split, cruise NaN — branch predated NaN fix)
- W&B: `kx17n4pn` (T_max=15 winner arm)

## Merged PRs
| PR | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|----|-----------|---------------------|---------------------|
| #3159 | Huber loss δ=0.1 | 112.9001 | 115.7589 (3/4) |
| #3309 | NaN fix (cruise test) | 112.8295 | **106.5996** (4/4) |
| #3317 | Cosine T_max=15 | **91.3319** | 88.4260 (3/4) |

## Closed PRs (key dead ends)
| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| #3162 | surf_weight=25 (MSE) | 133.41 | Loss misalignment |
| #3188 | slice_num=128 (MSE) | 134.74 | Capacity not bottleneck |
| #3167 | OneCycleLR | 137.12 | Budget too short |
| #3180 | h=192 wider | 150.38 | Capacity not bottleneck |
| #3361 | slice_num=128 (retried) | 116.19 | 30% slower/ep |
| #3359 | edward no-commit | 133.32 | Iterated w/o pushing |
| #3395 | LR peak scan 3e-4/8e-4 | 94.18/94.46 | lr=5e-4 at basin minimum |
| #3426 | Warm restarts T_0=5 | 103.07 | 5-ep cycles too short |
| #3460 | bf16 + bs=8 | 110.72 | bs=8 starved AdamW (-39% updates) |
| #3459 | EMA decay=0.999 | 100.92 | Decay half-life > training horizon |
| #3174 | L1-on-p + surf_w=50 | 99.51 | Gradient starvation (94% on surf-p) |

## Active WIP — 8/8 students assigned
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3305 | alphonse | Huber δ=0.05 rebase on T_max=15 | Nudged — W&B 78nl8hac val=93.34, awaiting terminal |
| #3480 | askeladd | bf16 autocast only (bs=4 preserved) | NEW — assigned after #3460 close |
| #3428 | edward | surf_weight scan (15, 20) | Nudged — W&B 6ra6amur val=91.625, awaiting terminal |
| #3171 | fern | Split pressure head + 3× p weight on T_max=15 | W&B as2gixh4 val=102.78, awaiting terminal |
| #3522 | frieren | L1 on surf-p ONLY at surf_weight=10 (isolated) | NEW — assigned after #3174 close |
| #3175 | nezuko | Cosine warmup on T_max=15 base | Nudged — W&B hyxr9xiu val=95.41, awaiting terminal |
| #3521 | thorfinn | EMA decay=0.99 (faster forgetting) | NEW — assigned after #3459 close |
| #3363 | tanjiro | AdamW β2=0.95 + grad clip on T_max=15 | Nudged — W&B 1i0kr8lr val=92.43 (close!), awaiting terminal |

## Key pattern from round 2-3
**Promising orthogonal levers DON'T fully stack onto T_max=15.** Multiple rebased experiments came in close to but worse than baseline 91.33:
- tanjiro β2+clip on T_max=15: 92.43 (+1.2%)
- edward surf_w=15: 91.625 (+0.3% — within noise)
- alphonse Huber δ=0.05: 93.34 (+2.2%)
- nezuko warmup: 95.41 (+4.5%, vs 89.02 on OLD base)

**Hypothesis:** T_max=15 schedule is well-tuned to current model/data. Marginal hyperparameter tweaks see diminishing returns. New gains will likely come from:
1. **Throughput unlock** → more epochs (askeladd bf16 path)
2. **Architectural changes** (e.g. unified_pos, attention variants)
3. **Data augmentation** (symmetry flip)
4. **Test-time augmentation**
5. **EMA at correct decay** (thorfinn's PR #3521 — high probability fit)

## Priority hypotheses likely to advance state of art
After current round, expect (best-case):
- T_max=15 + EMA decay=0.99 (thorfinn): if EMA tracks raw, +0.5-2% improvement
- T_max=15 + bf16 alone (askeladd): more epochs → val 86-88
- T_max=15 + isolated L1 on surf-p (frieren): possible cruise OOD win

## Next research directions
1. **Triple-stack winners** once individual results validate
2. **Train-time symmetry augmentation** — horizontal flip (camber sign + Uy sign)
3. **Unified positional encoding** — Transolver `unified_pos=True`
4. **Test-time augmentation (TTA)** with flip+average
5. **Stochastic Weight Averaging (SWA)** during final epochs (if EMA decay=0.99 looks promising)
6. **Layer-wise LR decay**
7. **Per-domain normalization** — pressure ranges differ by split
8. **Re-evaluate current best (#3317)** with all 4 test splits — branch predates NaN fix
