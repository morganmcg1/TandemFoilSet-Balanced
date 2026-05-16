# SENPAI Research State

- **Updated:** 2026-05-16 09:00 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 81.16**, **test_avg/mae_surf_p = 71.77** (PR #3601,
EMA decay=0.998, slice_num=16, single-seed, best epoch 18).

Per-split val: single=94.05, rc=92.73, cruise=60.45, re_rand=77.42.
Per-split test: single=83.37, rc=82.79, cruise=51.08, re_rand=69.85.

**EMA axis:** 0.9995→catastrophic, 0.999→84.44, 0.998→81.16 (-3.88%). Monotone trend — not yet bracketed from looser side. Probe: 0.997 (alphonse #3783).

**Key gap:** `val_single_in_dist=94.05` remains worst split (+1.4% above next-worst rc=92.73). Gap narrowing — EMA-0.998 improved single most (-6.04%), suggesting tighter EMA window benefits the hardest ID split disproportionately.

## Critical lessons (R1–R9)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Reverse strategy (smaller/cheaper = more epochs) is the dominant discovery.
2. **slice_num is the primary lever — now closed.** Three wins: 64→32→16. Axis fully closed at 16. 8 regressed (+1.52%) due to per-batch FFN overhead dominating.
3. **Model is compute-bound, not capacity-bound.** Best epoch = final epoch in ALL recent wins. More epochs beats more capacity.
4. **EMA window must match compute budget.** EMA-0.998 (window ≈ 500 steps, 14-17% of budget) beats EMA-0.999 (window ≈ 1000 steps, 33%). Tighter window focuses on converged tail. Trend continues to 0.997.
5. **Dropout saturated at 0.1, surf_weight at 10.** Both axes closed.
6. **EMA tighter direction closed.** EMA-0.9995 catastrophic (+34.5%).
7. **LR schedule (cosine T_max) axis closed.** T_max=16 and T_max=18 both tested — per-split reversal fingerprint (ID improves, OOD regresses) confirms conflict with slice_num=16 implicit reg.
8. **wd=5e-4 closes regularization axis.** Train loss +42% = underfit on slice_num=16 stack. wd=1e-4 optimal.

## Round wins merged (R1–R9)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% | MERGED |
| #3400 | SmoothL1 beta=0.25 | 97.15 | -1.32% | MERGED |
| #3402 | dropout=0.1 | 96.17 | -1.01% | MERGED |
| #3533 | slice_num=64→32 | 90.58 | -5.81% | MERGED |
| #3602 | slice_num=32→16 | 84.44 | -6.78% | MERGED |
| #3601 | EMA 0.999→0.998 (re-test on sn16) | **81.16** | **-3.88%** | **MERGED — current baseline** |

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
| #3783 | alphonse | EMA decay 0.998→0.997 (continue bracketing) | EMA tuning | R9 |

All students notified of new baseline (val=81.16). All should rebase onto advisor branch.

## Plateau / saturation map

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** H100 memory-bandwidth-bound.
- **Capacity up (n_hidden=160, mlp_ratio=4):** timeout-bound.
- **Capacity down (n_hidden=96):** only 4% speedup, +1.50% val regression. n_hidden=128 optimal.
- **Stochastic depth:** wrong regime.
- **Dropout:** saturated at 0.1.
- **surf_weight:** saturated at 10.
- **EMA tighter (0.9995):** catastrophic regression. Closed from tighter direction.
- **slice_num:** local optimum at 16. 64→32→16→8 all tested. Axis fully closed.
- **LR schedule (cosine T_max):** T_max=16 and T_max=18 both regressed on slice_num=16 base. Axis closed.
- **weight_decay:** wd=5e-4 underfits (+42% train loss). wd=1e-4 optimal. Axis closed.

**OPEN axes (in-flight):**
- **EMA looser (0.997) on slice_num=16+EMA-0.998 stack:** alphonse #3783 — continue bracketing the looser EMA window direction. Monotone trend from 0.999→0.998 not yet at floor.
- **bf16 autocast:** askeladd #3743 — H100 1.5-2× matmul speedup → expected 25-30 epochs vs 18; no capacity loss.
- **n_layers=4:** edward #3769 — drop one Transolver block, ~20% per-step speedup → 22-23 epochs.
- **gradient clipping (max_norm=1.0):** frieren #3772 — stability lever, untested in this launch.
- **LR schedule (WarmRestarts):** nezuko #3572.
- **LR value (7e-4):** fern #3573.
- **Data sampling (single upweight):** tanjiro #3558.
- **Loss channel weighting (1,1,3):** thorfinn #3560.

## Potential next research directions (R10+)

1. **EMA=0.997 bracketing** — if win, continue to 0.996; if loss, optimum is between 0.997-0.998.
2. **Compound winner stack** — once EMA-0.997 resolved, stack with any compute-efficiency win (bf16, n_layers=4). Two orthogonal improvements should compound.
3. **bf16 + EMA compound** — bf16 enables ~8-10 more epochs; EMA quality matters more with longer training. Natural pairing.
4. **Re-conditioned FiLM** — log(Re) → (γ, β) per Transolver block. Cross-regime conditioning; targets val_re_rand and single OOD.
5. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely; potentially more robust in compute-bound regime.
6. **Per-split data analysis** — val_single_in_dist still worst (-6.04% improved but still #1 worst). What structural feature of the single-config domain causes persistent underperformance?
7. **Label smoothing on pressure** — SmoothL1 already near-L1 at beta=0.25; can we do better with asymmetric loss weighting?

## Plateau plan

Progress: 6 sequential wins (R1-R9, merged). val_avg improved from 143.52 → 81.16 (-43.5% total). Still compute-bound; every best epoch = final epoch.
Plateau trigger: fires if R9 AND R10 both land 0 winners vs the 81.16 baseline.
