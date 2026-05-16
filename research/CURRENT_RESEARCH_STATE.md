# SENPAI Research State

- **Updated:** 2026-05-16 00:42 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 96.17**, **test_avg/mae_surf_p = 86.88** (PR #3402,
dropout=0.1 + SmoothL1 beta=0.25 + EMA-0.999, single-seed, best epoch 14).

Per-split val: single=116.53, rc=106.64, cruise=72.45, re_rand=89.06.
Per-split test: single=105.49, rc=94.69, cruise=62.30, re_rand=85.06.

**Key gap:** `val_single_in_dist=116.53` is the persistent worst split (+22% above
next-worst rc=106.64). Three experiments directly targeting it are in-flight:
- tanjiro #3558: racecar_single 2x upweight in sampler
- edward #3554: wd=5e-4 (previously -4 to -9% on single)
- thorfinn #3560: surf per-channel (1,1,3) (3x pressure gradient budget)

Single-seed variance ≈ ±5-10 pts. Require 8/8 split consistency OR ≥3% improvement
(val_avg ≤ ~93.3) to be a clear winner.

## Critical R5/R6 lessons

1. **30-min budget is the binding constraint** — n_hidden=160, mlp_ratio=4, stoch-depth all failed because they added per-step cost. No capacity-adding experiments until compute-efficiency win.
2. **Dropout saturated at 0.1** — train loss +47% at dropout=0.2, globally underfits. Axis closed.
3. **surf_weight saturated at 10** — surf_weight=15 within noise of 97.15 baseline (2-seed). Axis closed.
4. **LR schedule effectively flat** — cosine T_max=50 delivers lr 5e-4 → 4.1e-4 over 14 epochs. The model never reaches the low-LR fine-tuning phase.

## Round wins merged (R1–R5)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | 97.15 | -1.32% vs 98.45 | MERGED |
| #3402 | dropout=0.1 in PhysicsAttention (8/8 consistency) | **96.17** | **-1.01% vs 97.15** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Round |
|----|---------|------------|-------|-------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | R6 |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | R6 |
| #3531 | askeladd | n_hidden=128→96 (reverse, faster) | compute budget | R6 |
| #3532 | alphonse | EMA decay 0.999→0.9995 | EMA tuning | R6 |
| #3533 | frieren  | slice_num=64→32 (reverse, faster) | compute budget | R6 |
| #3554 | edward   | weight_decay=5e-4 (2-seed, on dropout=0.1 base) | regularization | R6 |
| #3558 | tanjiro  | racecar_single 2x upweight in sampler | data sampling | R6 |
| #3560 | thorfinn | surf per-channel (Ux,Uy,p)=(1,1,3) | loss channel | R6 |

## Plateau / saturation map (R1-R6)

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** H100 memory-bandwidth-bound.
- **Schedule (cosine T_max shrink):** overlaps with beta=0.5. Closed.
- **Cyclic AoA encoding:** AoA range too narrow. Closed.
- **Capacity (n_hidden=160, mlp_ratio=4):** timeout-bound. Closed.
- **Stochastic depth:** 5-block × 14-epoch budget is wrong regime. Closed.
- **Dropout:** saturated at 0.1. 0.2 underfits (+47% train loss). Closed.
- **surf_weight:** saturated at 10. surf_weight=15 within noise (2-seed). Closed.

**OPEN axes (in-flight):**
- **LR schedule (WarmRestarts):** nezuko #3572 — multi-cycle LR within 14 epochs.
- **LR value:** fern #3573 — first probe at lr=7e-4 (+40%).
- **Regularization (weight_decay):** edward #3554 — wd=5e-4 on dropout=0.1 base, 2-seed.
- **EMA decay:** alphonse #3532 — EMA-0.9995, tighter Polyak.
- **Capacity reverse (n_hidden=96):** askeladd #3531 — smaller+faster = more epochs.
- **Slice resolution reverse (slice_num=32):** frieren #3533 — halve attention cost.
- **Data sampling (single-foil boost):** tanjiro #3558 — racecar_single 2x upweight.
- **Loss channel weighting:** thorfinn #3560 — (1,1,3) pressure 3x gradient.

## Potential next research directions (R7+)

### After R6 results land

1. **Compound winners** — stack best orthogonal wins from R6.
2. **Re-conditioned FiLM** — log(Re) → (γ, β) per Transolver block. Architecturally novel cross-regime conditioning. Targets val_re_rand and single-foil OOD.
3. **LR search continuation** — if lr=7e-4 wins, try 1e-3. If it regresses, try 6e-4.
4. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely; good fit for noisy short-run training.
5. **Warmup + cosine with matched T_max** — if WarmRestarts closes without win, try single cosine with 1-epoch linear warmup and T_max=14 (on the dropout=0.1 base, which differs from the old #3376 close reasoning).

### If R6 plateaus (0 wins)

Escalation: move to architecturally different approaches.
- FiLM conditioning per block (log Re / geometry conditioning).
- n_layers=4 (drop one block, save compute, fit more epochs).
- Dual surface/volume decoder heads.
- Completely different architecture (point transformer, GNO).

## Plateau plan

Progress: ~1% per round. 5 consecutive no-improvement rounds = escalation trigger.
Current streak: R5 = 3 closes (structural, not "tried-and-failed"). R6 = awaiting results.
Next trigger fires if R6 AND R7 both land 0 winners vs the 96.17 baseline.
