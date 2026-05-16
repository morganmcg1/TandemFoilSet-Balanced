# SENPAI Research State

- **Updated:** 2026-05-16 09:30 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 80.88**, **test_avg/mae_surf_p = 71.18** (PR #3783,
EMA decay=0.997, slice_num=16, single-seed, best epoch 18).

Per-split val: single=94.59, rc=90.88, cruise=61.04, re_rand=77.02.
Per-split test: (from PR #3783) single≈83, rc≈82, cruise≈51, re_rand≈70.

**EMA axis CLOSED.** Progression: 0.9995→catastrophic, 0.999→84.44, 0.998→81.16 (-3.88%), 0.997→80.88 (-0.34%). Diminishing returns confirmed — 10× smaller gain, per-split reversal, axis converged in [0.997, 0.998].

**Key gap:** `val_single_in_dist=94.59` remains worst split (+3.9% above next-worst rc=90.88). Previous EMA tightening improved single most (-6.04% on 0.998 merge), but the final 0.997 step slightly regressed it. The single split appears to need architectural or data-level attention.

## Critical lessons (R1–R10)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Reverse strategy (smaller/cheaper = more epochs) is the dominant discovery.
2. **slice_num axis CLOSED at 16.** 64→32→16: three consecutive wins (-12.1% total). 8 regressed. Per-batch FFN overhead now dominates.
3. **Model is compute-bound, not capacity-bound.** Best epoch = final epoch in ALL wins. More epochs beats more capacity.
4. **EMA axis CLOSED at [0.997, 0.998].** 0.999→84.44, 0.998→81.16 (-3.88%), 0.997→80.88 (-0.34%). Window tightening converged. Further probe unlikely to yield gains.
5. **Dropout saturated at 0.1, surf_weight at 10.** Both closed.
6. **LR schedule (cosine T_max) closed.** Tested T_max=16 and T_max=18 — ID/OOD reversal fingerprint, no win.
7. **wd=5e-4 closes regularization axis.** Train +42% underfit at wd=5e-4. wd=1e-4 optimal.
8. **val_single_in_dist structurally hardest.** Worst split by +4 pts consistently. May need architectural conditioning (FiLM on Re number) or data strategy to close.

## Round wins merged (R1–R10)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% | MERGED |
| #3400 | SmoothL1 beta=0.25 | 97.15 | -1.32% | MERGED |
| #3402 | dropout=0.1 | 96.17 | -1.01% | MERGED |
| #3533 | slice_num=64→32 | 90.58 | -5.81% | MERGED |
| #3602 | slice_num=32→16 | 84.44 | -6.78% | MERGED |
| #3601 | EMA 0.999→0.998 (sn16 confirm) | 81.16 | -3.88% | MERGED |
| #3783 | EMA 0.998→0.997 (diminishing returns) | **80.88** | **-0.34%** | **MERGED — current baseline** |

**Total improvement from calibration baseline:** 143.52 → 80.88 = **-43.7%**

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Round |
|----|---------|------------|-------|-------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | R6 |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | R6 |
| #3558 | tanjiro  | racecar_single 2x upweight | data sampling | R6 |
| #3560 | thorfinn | surf per-channel (1,1,3) | loss channel | R6 |
| #3743 | askeladd | bf16 autocast — attack per-batch overhead | compute budget | R8 |
| #3769 | edward   | n_layers=5→4 (drop a Transolver block) | compute/capacity | R9 |
| #3772 | frieren  | gradient clipping max_norm=1.0 | training stability | R9 |
| #3841 | alphonse | n_head=4→8: finer attention, zero compute cost | architecture | R10 |

All students notified of new baseline (val=80.88). All should rebase onto advisor branch.

## Plateau / saturation map

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** H100 memory-bandwidth-bound.
- **Capacity up:** timeout-bound.
- **Capacity down (n_hidden=96):** +1.50% regression.
- **Stochastic depth:** wrong regime.
- **Dropout:** saturated at 0.1.
- **surf_weight:** saturated at 10.
- **EMA:** converged in [0.997, 0.998]. 0.9995 catastrophic, 0.997 → diminishing returns.
- **slice_num:** optimum at 16. Fully closed.
- **LR schedule (cosine T_max):** T_max=16 and T_max=18 both regressed.
- **weight_decay:** wd=5e-4 underfits. wd=1e-4 optimal.

**OPEN axes (in-flight):**
- **n_head=4→8:** alphonse #3841 — finer attention head split, zero compute, unexplored architecture axis.
- **bf16 autocast:** askeladd #3743 — H100 matmul speedup → 25-30 epochs vs 18.
- **n_layers=4:** edward #3769 — drop one Transolver block, ~20% per-step speedup.
- **gradient clipping (max_norm=1.0):** frieren #3772 — stability lever.
- **LR schedule (WarmRestarts):** nezuko #3572.
- **LR value (7e-4):** fern #3573.
- **Data sampling (single upweight):** tanjiro #3558.
- **Loss channel weighting (1,1,3):** thorfinn #3560.

## Potential next research directions (R11+)

1. **Compound compute-efficiency wins** — if bf16 (askeladd) and/or n_layers=4 (edward) land, stack them. Both target the same bottleneck (per-batch cost) — they're substitutable, so merge best-first and test second on new stack.
2. **Re-conditioned FiLM** — log(Re) → (γ, β) conditioning per Transolver block. Targets val_single_in_dist structural gap. Moderate complexity but highest-potential architectural idea.
3. **mlp_ratio=1** — if n_layers=4 doesn't win, try halving FFN width instead. Different attack on same bottleneck (FFN compute).
4. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling; robust in compute-bound regime.
5. **n_head compound** — if n_head=8 wins, try n_head=16 (dim_head=8, aggressive).
6. **Data augmentation** — physics-preserving transforms (rotation, Re scaling) to improve OOD splits.

## Plateau plan

Progress: 7 consecutive wins (R1-R10), -43.7% total from calibration. Streak continues.
Next trigger: fires if R10 AND R11 both land 0 winners vs the 80.88 baseline.
