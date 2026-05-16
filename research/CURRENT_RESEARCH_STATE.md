# SENPAI Research State

- **Updated:** 2026-05-16 14:30 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 79.05**, **test_avg/mae_surf_p = 69.76** (PR #3982,
mlp_ratio=1, slice_num=12, EMA decay=0.997, n_head=4, best epoch 19).

Per-split val: single=92.38, rc=90.41, cruise=58.997, re_rand=74.42.
Per-split test: single=81.55, rc=79.44, cruise=49.32, re_rand=68.73.

**mlp_ratio axis:** 2→1 confirmed -1.92% val. Clean win: all 8 splits improved, +1 epoch from compute saving. 7% sec/epoch savings (NOT the predicted 25%) — student correctly identified FFN matmuls aren't dominant. **Real bottleneck: per-iteration overhead (Python/optimizer/kernel launches).**

**Compute-budget model refined:** O(K²) attention savings (slice_num) and FFN matmul savings (mlp_ratio) both yield modest wall-clock improvements (~5-7%). Dataloader already well-tuned (num_workers=4, pin_memory, persistent_workers). The next big compute win requires different attack: fused kernels, bf16, torch.compile, or batch size changes.

**val_single_in_dist structural gap:** Worst split at 92.38 (+2 pts vs next-worst rc=90.41). Has resisted all closed axes. n_head=8 attempted and regressed it most. Structural — FiLM-on-Re (PR #4004) is the current architectural bet.

## Critical lessons (R1–R11)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Reverse strategy (smaller/cheaper = more epochs) is the dominant discovery.
2. **slice_num axis CLOSED at [12, 16].** 64→32→16→12: four consecutive wins total (-12.1% total). sn=8 regressed. sn=12 ties sn=16 within noise; both effective.
3. **Model is compute-bound, not capacity-bound.** Best epoch = final epoch in ALL wins. More epochs beats more capacity.
4. **EMA axis CLOSED at [0.997, 0.998].** 0.999→84.44, 0.998→81.16 (-3.88%), 0.997→80.88 (-0.34%). Converged. Further probe very unlikely to yield gains.
5. **Dropout saturated at 0.1, surf_weight at 10.** Both closed.
6. **LR schedule (cosine T_max) closed.** Tested T_max=16 and T_max=18 — ID/OOD reversal fingerprint, no win.
7. **wd=5e-4 closes regularization axis.** Train +42% underfit at wd=5e-4. wd=1e-4 optimal.
8. **val_single_in_dist structurally hardest.** Worst split by ~2 pts consistently. n_head=8 regressed it most (+8.9%). n_head=2 tied baseline. FiLM-on-Re (PR #4004) is current bet.
9. **n_head axis CLOSED at 4.** Both n_head=8 (+6.7% regression) and n_head=2 (+1.99%) regressed. Mechanism: wall-clock penalty from extra softmax launches (n_head=8) or no savings (n_head=2, FFN dominates).
10. **FFN matmuls NOT dominant.** mlp_ratio 2→1 saved only 7% sec/epoch (not predicted 25%). Per-iteration overhead (Python/kernel launch) is the real ceiling.
11. **mlp_ratio axis:** 2→1 confirmed -1.92%. mlp_ratio=0 not viable (would eliminate FFN). Axis closed.

## Round wins merged (R1–R11)

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
| #3982 | mlp_ratio 2→1 (halve FFN width, +1 epoch) | **79.05** | **-1.92%** | **MERGED — current baseline** |

**Total improvement from calibration baseline:** 143.52 → 79.05 = **-44.9%**

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
| #4004 | alphonse | FiLM-on-Re: condition each Transolver block on log(Re) | architecture | R11 |

All students notified of latest baseline (val=79.05, PR #3982). All should be rebased onto advisor branch.

## Plateau / saturation map

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** batch_size=8 tested (PR #3327, +45.5% regression). Memory-bandwidth bound + LR interaction.
- **Capacity up:** timeout-bound.
- **Capacity down (n_hidden=96):** +1.50% regression.
- **Stochastic depth:** wrong regime.
- **Dropout:** saturated at 0.1.
- **surf_weight:** saturated at 10.
- **EMA:** converged in [0.997, 0.998].
- **slice_num:** tied at [12, 16]. Fully closed.
- **LR schedule (cosine T_max):** both T_max=16 and T_max=18 regressed.
- **weight_decay:** wd=5e-4 underfits. wd=1e-4 optimal.
- **n_head:** closed at 4. n_head=8 +6.7% regression, n_head=2 +1.99%.
- **mlp_ratio:** closed at 1 (-1.92% win). Below 1 would eliminate FFN.

**OPEN axes (in-flight):**
- **FiLM-on-Re:** alphonse #4004 — architectural conditioning, targets val_single_in_dist structural gap.
- **bf16 autocast:** askeladd #3743 — H100 matmul speedup → potentially 25-30 epochs.
- **n_layers=4:** edward #3769 — drop one Transolver block, ~20% per-step speedup.
- **gradient clipping (max_norm=1.0):** frieren #3772 — stability lever.
- **LR schedule (WarmRestarts):** nezuko #3572.
- **LR value (7e-4):** fern #3573.
- **Data sampling (single upweight):** tanjiro #3558.
- **Loss channel weighting (1,1,3):** thorfinn #3560.

## Potential next research directions (R12+)

1. **Compound compute-efficiency wins** — if bf16 (askeladd) and/or n_layers=4 (edward) land, stack them and test FiLM-on-Re on the new base. Per-step overhead reduction + architectural improvement could compound well.
2. **torch.compile (dynamic=True)** — if per-iteration overhead truly dominates, torch.compile with dynamic shape support could reduce Python overhead. Risk: variable mesh sizes trigger recompile per batch. Test after bf16 lands.
3. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR schedule entirely. Particularly clean in compute-bound regime where best epoch = final epoch. Robust to LR choice.
4. **FiLM-on-AoA + geometry** — if FiLM-on-Re wins, generalize to condition on AoA, NACA, gap/stagger. All geometric parameters are broadcast-constant within a sample.
5. **Data augmentation** — physics-preserving transforms (AoA perturbation within Re regime) to improve OOD splits.
6. **Fused AdamW** — `torch.optim.AdamW(..., fused=True)` fuses optimizer step into single CUDA kernel. Attacks one component of per-iter overhead.
7. **EMA → SWA (full average)** — if late-training is compute-bound with monotone improvement, stochastic weight averaging over last N epochs may be more robust than EMA.

## Plateau plan

Progress: 9 consecutive wins (R1-R11), -44.9% total from calibration. Streak continues.
Next trigger: fires if two consecutive rounds land 0 winners vs the 79.05 baseline.
