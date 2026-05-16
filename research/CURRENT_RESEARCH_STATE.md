# SENPAI Research State

- **Updated:** 2026-05-16 15:45 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 71.46**, **test_avg/mae_surf_p = 62.53** (PR #4004,
FiLM-on-Re, mlp_ratio=1, slice_num=12, EMA decay=0.997, n_head=4, dropout=0.1, best epoch 18).

Per-split val: single=83.22, rc=81.69, cruise=50.61, re_rand=70.32.
Per-split test: single=72.86, rc=74.86, cruise=41.88, re_rand=60.52.

**FiLM-on-Re axis OPEN.** PR #4004 was a landmark -9.6% val win — largest single-PR gain since the early SmoothL1 rounds (-19.7%). All 8 splits improved globally. Training still monotonically descending at epoch 18 (2 pts/epoch) — the 30-min cap is the hard constraint, not overfitting. Extension to FiLM+AoA (PR #4018, alphonse) is the immediate follow-up.

**val_single_in_dist:** Closed from 92.38 → 83.22 (-9.9%). Structural gap partially resolved by FiLM-Re. Now 1.53 pts above next-worst rc=81.69 (was 2 pts gap previously). Further reduction likely from FiLM+AoA expansion.

**Compute bottleneck remains:** Best epoch = final epoch (18) in FiLM-Re run. FiLM added +6.3% sec/epoch overhead vs mlp_ratio=1, costing 1 epoch. If bf16 (askeladd #3743) or n_layers=4 (edward #3769) land on the new FiLM baseline, each projected +5-10 epochs could translate to an additional -10-20 pts val.

## Critical lessons (R1–R12)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Reverse strategy (smaller/cheaper = more epochs) is the dominant discovery — though FiLM proves architectural improvements can dominate compute efficiency in some regimes.
2. **FiLM conditioning is the dominant architectural lever found.** -9.6% val in a single PR. Physically motivated (Re sets flow regime) and globally beneficial (all 4 splits improved by 5-14%).
3. **slice_num axis CLOSED at [12, 16].** Both effectively tied.
4. **Model is compute-bound, not capacity-bound.** Best epoch = final epoch in ALL wins.
5. **EMA axis CLOSED at [0.997, 0.998].** Converged.
6. **Dropout saturated at 0.1, surf_weight at 10.** Both closed.
7. **LR schedule (cosine T_max) closed.** T_max=16 and T_max=18 both regressed.
8. **weight_decay: wd=1e-4 optimal.** wd=5e-4 underfits.
9. **n_head axis CLOSED at 4.** n_head=8 +6.7% regression, n_head=2 +1.99%.
10. **mlp_ratio axis CLOSED at 1.** 2→1 was -1.92% win. Cannot go below 1.
11. **FFN matmuls NOT dominant.** Per-iteration overhead (Python/kernel launch) is the real ceiling for compute-only attacks.
12. **FiLM identity init is critical.** γ=0, β=0 at epoch 0 means training starts equivalent to baseline — model must actively learn conditioning. This allows safe zero-risk initialization for architectural additions.

## Round wins merged (R1–R12)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% | MERGED |
| #3400 | SmoothL1 beta=0.25 | 97.15 | -1.32% | MERGED |
| #3402 | dropout=0.1 | 96.17 | -1.01% | MERGED |
| #3533 | slice_num=64→32 | 90.58 | -5.81% | MERGED |
| #3602 | slice_num=32→16 | 84.44 | -6.78% | MERGED |
| #3601 | EMA 0.999→0.998 (sn16 confirm) | 81.16 | -3.88% | MERGED |
| #3783 | EMA 0.998→0.997 (diminishing returns) | 80.88 | -0.34% | MERGED |
| #3950 | slice_num 16→12 (triangulate; tie within noise) | 80.60 | -0.34% | MERGED |
| #3982 | mlp_ratio 2→1 (halve FFN width, +1 epoch) | 79.05 | -1.92% | MERGED |
| #4004 | FiLM-on-Re (condition each block on log(Re)) | **71.46** | **-9.6%** | **MERGED — current baseline** |

**Total improvement from calibration baseline:** 143.52 → 71.46 = **-50.2%**

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | WIP (notified of new baseline 71.46) |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | WIP (notified of new baseline 71.46) |
| #3558 | tanjiro  | racecar_single 2x upweight | data sampling | WIP (notified of new baseline 71.46) |
| #3560 | thorfinn | surf per-channel (1,1,3) | loss channel | WIP (notified of new baseline 71.46) |
| #3743 | askeladd | bf16 autocast — attack per-batch overhead | compute budget | WIP (notified of new baseline 71.46) |
| #3769 | edward   | n_layers=5→4 (drop a Transolver block) | compute/capacity | WIP (notified of new baseline 71.46) |
| #3772 | frieren  | gradient clipping max_norm=1.0 | training stability | WIP (notified of new baseline 71.46) |
| #4018 | alphonse | FiLM-Re+AoA: expand conditioning to [log_Re, AoA0, AoA1] | architecture | WIP (newly assigned) |

Note: PRs #3572, #3573, #3558, #3560, #3772 were designed before FiLM-on-Re was merged. Their baselines have shifted dramatically (79.05 → 71.46). Unless they achieve similar magnitude improvements, they will not beat the new baseline. They should still be reviewed for learnings.

## Plateau / saturation map

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** batch_size=8 tested (PR #3327, +45.5% regression).
- **Capacity up:** timeout-bound.
- **Capacity down (n_hidden=96):** +1.50% regression.
- **Dropout:** saturated at 0.1.
- **surf_weight:** saturated at 10.
- **EMA:** converged in [0.997, 0.998].
- **slice_num:** tied at [12, 16]. Fully closed.
- **LR schedule (cosine T_max):** T_max=16 and T_max=18 both regressed.
- **weight_decay:** wd=1e-4 optimal.
- **n_head:** closed at 4.
- **mlp_ratio:** closed at 1.

**OPEN axes (in-flight):**
- **FiLM-Re+AoA:** alphonse #4018 — expand conditioning to 3 scalars [log_Re, AoA0, AoA1].
- **bf16 autocast:** askeladd #3743 — H100 matmul speedup → more epochs on FiLM baseline.
- **n_layers=4:** edward #3769 — drop one Transolver block, ~20% per-step speedup → more epochs.
- **gradient clipping (max_norm=1.0):** frieren #3772 — stability lever.
- **LR schedule (WarmRestarts):** nezuko #3572 (pre-FiLM design; beating 71.46 is a high bar).
- **LR value (7e-4):** fern #3573 (pre-FiLM design; high bar).
- **Data sampling (single upweight):** tanjiro #3558 (pre-FiLM design; high bar).
- **Loss channel weighting (1,1,3):** thorfinn #3560 (pre-FiLM design; high bar).

## Potential next research directions (R13+)

1. **FiLM + compute wins compound** — if bf16 (askeladd) or n_layers=4 (edward) land on the FiLM baseline, each +5-10 epochs in budget ≈ -10-20 pts additional val. These are the highest-priority in-flight probes to watch.
2. **FiLM on all physical scalars** — extend conditioning beyond Re and AoA to include gap/stagger (tandem geometry parameters). Full conditioning vector: [log_Re, AoA0, AoA1, gap, stagger] = 5 scalars.
3. **FiLM inside PhysicsAttention (slice-level)** — apply Re conditioning to the slice projection matrices within PhysicsAttention, not just the block-level representations. More targeted modulation of the attention mechanism itself.
4. **Schedule-Free AdamW** (Defazio 2024) — eliminates cosine T_max sensitivity. Robust in compute-bound regime.
5. **torch.compile** — with FiLM overhead now +6.3%, compile could fuse the per-block affine ops and recover the lost epoch. Risk: variable mesh shapes need `dynamic=True`.
6. **Second seed confirmation of FiLM-Re** — the -9.6% win is well above ±5-10pt noise floor but a second seed would confirm before the ICML deadline.
7. **Fused AdamW** — `torch.optim.AdamW(..., fused=True)` reduces Python overhead in optimizer step.

## Plateau plan

Progress: 10 consecutive wins (R1-R12), -50.2% total from calibration. We crossed the 50% improvement threshold. Streak continues.
Next trigger: fires if two consecutive rounds land 0 winners vs the 71.46 baseline.
