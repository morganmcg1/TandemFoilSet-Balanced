# SENPAI Research State

- **Updated:** 2026-05-16 00:15 UTC
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

**Key gap:** `val_single_in_dist=116.53` remains the worst split by a large
margin. Every round it survives while others improve. Three distinct attack
vectors now in flight targeting it:
- Sampling: tanjiro racecar_single 2x upweight (#3558)
- Regularization: edward wd=5e-4 which previously gave -4 to -9% on single (#3554)
- Loss channel: thorfinn (1,1,3) weights p channel directly (#3560)

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p. For merges, require
8/8 split directional consistency OR ≥3% improvement (val_avg ≤ ~93.3).

## Critical R5 lesson: the 30-min budget is the binding constraint

- n_hidden=160 (+34% per-epoch → only 11 epochs → +11.5% regression) — CLOSED
- mlp_ratio=4 (+18% per-epoch → only 13 epochs → +38% regression) — CLOSED
- stoch-depth p=0.1 (slows gradient, same budget) → +4.8% regression — CLOSED

**Strategy shift:** R6+ prioritizes (a) smaller/faster models, (b) zero-compute
loss/optim levers, (c) data-level changes. Capacity experiments are off the
table until we have a compute-efficiency win.

## Round wins merged (R1–R5)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | 97.15 | -1.32% vs 98.45 | MERGED |
| #3402 | dropout=0.1 in PhysicsAttention (single-seed) | **96.17** | **-1.01% vs 97.15** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Round |
|----|---------|------------|-------|-------|
| #3510 | nezuko   | dropout=0.1→0.2 push | regularization | R5 |
| #3482 | fern     | surf_weight=10→15 | loss weighting | R5 |
| #3531 | askeladd | n_hidden=128→96 (reverse) | compute budget | R6 |
| #3532 | alphonse | EMA decay 0.999→0.9995 | EMA tuning | R6 |
| #3533 | frieren  | slice_num=64→32 (reverse) | compute budget | R6 |
| #3554 | edward   | weight_decay=5e-4 v2 (2-seed) | regularization | R6 |
| #3558 | tanjiro  | racecar_single 2x upweight | data sampling | R6 |
| #3560 | thorfinn | surf per-channel (1,1,3) | loss channel | R6 |

## Plateau / saturation map (R1-R6)

**CLOSED axes:**
- **Loss formulation:** SmoothL1 beta=0.25 is the optimum. Axis closed.
- **Throughput / batch size:** H100 memory-bandwidth-bound. No lever.
- **Schedule shape:** Cosine T_max shrink overlaps with beta=0.5. Closed for simple shrinking.
- **Cyclic AoA encoding:** CLOSED. AoA range too narrow.
- **Capacity (n_hidden=160, mlp_ratio=4):** CLOSED — timeout-bound at 30-min cap.
- **Stochastic depth:** CLOSED. 5-block depth × 14-epoch budget is wrong regime.

**OPEN axes (in-flight):**
- **Regularization (dropout):** 0.1 merged. 0.2 in flight (nezuko).
- **Regularization (weight_decay):** 5e-4 re-run on dropout=0.1 base (edward #3554, 2-seed).
- **Loss weighting (surf_weight):** fern at 15 in flight.
- **Loss channel (per-channel surf):** thorfinn (1,1,3) in flight.
- **Data sampling (single-foil boost):** tanjiro 2x racecar_single in flight.
- **Capacity (n_hidden reverse):** askeladd n_hidden=96 — smaller+faster.
- **Slice resolution reverse:** frieren slice_num=32 — halve attention cost.
- **EMA tuning:** alphonse EMA-0.9995 — tighter Polyak.

## Potential next research directions (R7+)

### After R6 results land

1. **Compound winners** — stack best orthogonal wins from R6.
   - (wd=5e-4 + dropout=0.1) if edward wins — dual regularization.
   - (n_hidden=96 + slice_num=32) if both reverse-capacity wins — compound budget.
   - (single-foil-upweight + channel-113) if both targeting single win.

2. **Re-conditioned FiLM** — log(Re) → (γ, β) per Transolver block.
   Explicit cross-regime conditioning. Most architecturally novel untried lever.
   Targets val_re_rand and single-foil OOD together.

3. **Cosine WarmRestarts (SGDR)** — T_0=4, multiple restarts. No compute cost.
   Different from the T_max=14 single-cycle experiment that was closed.

4. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely.
   Better calibrated for noisy short-run (14-epoch) training.

5. **n_layers=4** — drop one block if smaller-faster wins (parallel to n_hidden=96).
   Saves compute, fits more epochs.

### If R6 plateaus (0 wins)

Escalation: move to architecturally different approaches.
- Dual decoder heads (surface vs volume).
- FiLM conditioning (log Re per block).
- Mesh-node subsampling in training.
- Try a completely different architecture (e.g. point transformer, GNO).

## Plateau plan

Progress rate: ~1% per round. 5 consecutive no-improvement rounds not yet reached
(R5 produced 3 closes without wins — NOT a no-improvement count because they're
structural-budget closes, not "tried and failed to beat baseline"). Continue
compounding small wins. Next escalation trigger: if R6 and R7 both land 0 winners.
