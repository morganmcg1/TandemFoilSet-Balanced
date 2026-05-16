# SENPAI Research State

- **Updated:** 2026-05-16 05:45 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 84.44**, **test_avg/mae_surf_p = 74.75** (PR #3602,
slice_num=16, single-seed, best epoch 18).

Per-split val: single=100.09, rc=94.49, cruise=63.60, re_rand=79.60.
Per-split test: single=88.51, rc=83.91, cruise=53.62, re_rand=72.94.

**Slice_num progression:** 64(96.17) → 32(90.58, -5.81%) → 16(84.44, -6.78%). Both times model was still improving at cap, confirming compute-bound regime. Probing 8 next.

**Key gap:** `val_single_in_dist=100.09` remains worst split (+5.9% above next-worst rc=94.49). All recent slice_num wins improved single proportionally (-7%+), but gap persists structurally.

## Critical lessons (R1–R7)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Reverse strategy (smaller/cheaper = more epochs) is the dominant discovery.
2. **slice_num is the primary lever.** Three wins in a row via O(K²) cost reduction: 64→32→16. Model is not capacity-limited at these slice counts for ~20k-node meshes. Next probe: 8.
3. **Model is compute-bound, not capacity-bound.** Best epoch = final epoch in ALL recent wins. Adding training compute (more epochs) consistently beats adding model capacity.
4. **EMA-0.998 is promising (pending re-test).** 2-seed evidence on slice_num=32 base showed clean -4.07%. Sent back for confirmation on slice_num=16 base.
5. **Dropout saturated at 0.1, surf_weight at 10.** Both axes closed.
6. **EMA tighter direction closed.** EMA-0.9995 catastrophic (+34.5%). EMA-0.999 current best (0.998 pending).
7. **LR schedule effectively flat.** CosineAnnealingLR(T_max=50) over 18 actual epochs delivers LR 5e-4 → 3.9e-4. Three open probes on this axis (#3572 WarmRestarts, #3573 lr=7e-4, #3603 T_max=16).

## Round wins merged (R1–R7)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% | MERGED |
| #3400 | SmoothL1 beta=0.25 | 97.15 | -1.32% | MERGED |
| #3402 | dropout=0.1 | 96.17 | -1.01% | MERGED |
| #3533 | slice_num=64→32 | 90.58 | -5.81% | MERGED |
| #3602 | slice_num=32→16 | **84.44** | **-6.78%** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Round |
|----|---------|------------|-------|-------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | R6 |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | R6 |
| #3554 | edward   | weight_decay=5e-4 (2-seed) | regularization | R6 |
| #3558 | tanjiro  | racecar_single 2x upweight | data sampling | R6 |
| #3560 | thorfinn | surf per-channel (1,1,3) | loss channel | R6 |
| #3603 | frieren  | CosineAnnealingLR T_max=16 | LR schedule | R7 |
| #3601 | alphonse | EMA decay 0.999→0.998 (re-test on slice_num=16) | EMA tuning | R7 |
| #3677 | askeladd | slice_num=16→8 | compute budget | R8 |

All students notified of new baseline (val=84.44). All asked to rebase onto advisor branch (slice_num=16 now in place).

## Plateau / saturation map (R1–R8)

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** H100 memory-bandwidth-bound.
- **Capacity up (n_hidden=160, mlp_ratio=4):** timeout-bound.
- **Capacity down (n_hidden=96):** only 4% speedup, +1.50% val regression. n_hidden=128 optimal.
- **Stochastic depth:** wrong regime.
- **Dropout:** saturated at 0.1.
- **surf_weight:** saturated at 10.
- **EMA tighter (0.9995):** catastrophic regression. Closed from tighter direction.

**OPEN axes (in-flight):**
- **slice_num (8):** askeladd #3677 — is improvement monotone below 16?
- **EMA looser (0.998) on slice_num=16:** alphonse #3601 — confirm compound win.
- **LR schedule (WarmRestarts):** nezuko #3572.
- **LR schedule (matched T_max=16):** frieren #3603.
- **LR value (7e-4):** fern #3573.
- **Regularization (wd=5e-4):** edward #3554.
- **Data sampling (single upweight):** tanjiro #3558.
- **Loss channel weighting (1,1,3):** thorfinn #3560.

## Potential next research directions (R8+)

1. **Compound winner stack** — once EMA-0.998 confirmed on slice_num=16 base, stack with any of: optim wins (lr, wd), loss channel, schedule.
2. **slice_num=8 result** — if still improving: probe slice_num=4 or find crossover. If regresses: slice_num=16 is expressiveness floor.
3. **Re-conditioned FiLM** — log(Re) → (γ, β) per Transolver block. Cross-regime conditioning; targets val_re_rand and single OOD.
4. **n_layers=4** — with slice_num now cheap, FFN blocks dominate compute. Drop one block, save 20% per-step cost.
5. **Matched cosine T_max** — if T_max=16 wins on old budget, now budget is 18 epochs so T_max=18 would be even better.
6. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely.

## Plateau plan

Progress: consecutive -5.81% and -6.78% wins in R6-R7. Streak reset.
Next trigger fires if R8 AND R9 both land 0 winners vs the 84.44 baseline.
