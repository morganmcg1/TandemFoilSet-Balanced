# SENPAI Research State

- **Updated:** 2026-05-15 23:42 UTC
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

**Key gap:** `val_single_in_dist=116.53` (and `test_single=105.49`) remain
the worst splits. Single-foil geometry is OOD relative to tandem-heavy
training data. Best remaining levers: compute-efficient changes (no extra
params), loss weighting (surf_weight), and EMA tuning.

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p. New baseline is
single-seed — for merges against it, require 8/8 split directional consistency
OR ≥3% improvement (val_avg ≤ ~93.3) to be a clear winner.

## Critical R5 lesson: the 30-min budget is the binding constraint

R5 closed with three regressions that share a common root cause:
- **#3471** stoch-depth p=0.1: train loss +9% higher → convergence slowed → regression
- **#3472** n_hidden=160: per-epoch +34% slower → only 11 epochs in 30 min → undertrained
- **#3124** mlp_ratio=4: per-epoch +18% slower → only 13 epochs → undertrained + NaN

**Conclusion:** At this depth (5 blocks, 14 epochs, 30-min cap), any technique that
slows convergence or adds per-epoch compute produces undertrained regressions.
R6 strategy: reverse capacity directions (smaller, faster) and explore zero-compute levers.

## Round wins merged (R1–R5)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | 97.15 | -1.32% vs 98.45 | MERGED |
| #3402 | dropout=0.1 in PhysicsAttention (single-seed) | **96.17** | **-1.01% vs 97.15** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3510 | nezuko   | dropout=0.1→0.2 push | dropout+beta+EMA | regularization sweep (R5) |
| #3482 | fern     | surf_weight=10→15 | beta+EMA+dropout | loss weighting (R5) |
| #3325 | edward   | weight_decay=5e-4 rebase | beta+EMA+dropout | regularization (R4, rebase pending) |
| #3286 | tanjiro  | surf_weight=25 | beta+EMA+dropout | loss weighting (R4, stale) |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) | beta+EMA+dropout | channel weighting (R4, stale) |
| #3531 | askeladd | n_hidden=128→96 (reverse) | beta+EMA+dropout | compute budget (R6) |
| #3532 | alphonse | EMA decay 0.999→0.9995 | beta+EMA+dropout | EMA tuning (R6) |
| #3533 | frieren  | slice_num=64→32 (reverse) | beta+EMA+dropout | compute budget (R6) |

**Note on tanjiro/thorfinn (old PRs):** These PRs predate the dropout=0.1 merge. Results above 96.17 → send back for rebase.

## Plateau / saturation map (R1-R6)

- **Loss formulation:** SmoothL1 beta=0.25 is the optimum in the smooth Huber family. Beta curve at {1.0, 0.5, 0.25} flattened. Axis closed.
- **Throughput / batch size:** H100 memory-bandwidth-bound. No batch-size lever.
- **NaN-safe scoring + EMA-0.999:** Banked.
- **Schedule shape:** Cosine T_max=14 overlaps with beta=0.5. Axis closed for simple T_max shrinking. WarmupCosine/SGDR remain viable.
- **Regularization (dropout):** 0.1 merged (+1% improvement, 8/8 consistency). Dropout=0.2 in flight (nezuko #3510). Train loss showed no slowdown at 0.1 — headroom exists.
- **Regularization (weight_decay):** Open — edward rebase pending.
- **Regularization (stoch-depth):** CLOSED. p=0.1 on 5-block net within 14-epoch budget slows convergence without OOD benefit. (+4.8% regression, all splits worse).
- **Cyclic AoA encoding:** CLOSED. AoA range too narrow for cyclic benefit.
- **Capacity (n_hidden):** Open in REVERSE — askeladd n_hidden=96 (R6) testing smaller+faster. n_hidden=160 closed (timeout-bound). Optimum unknown in reverse direction.
- **Capacity (mlp_ratio):** CLOSED at mlp_ratio=4 (timeout-bound, 13 epochs). mlp_ratio=2 banked.
- **Slice resolution (slice_num):** Open — frieren slice_num=32 (R6) testing smaller+faster.
- **EMA tuning:** Open — alphonse EMA-0.9995 (R6). Current 0.999 was first value tried, never tuned.
- **Architectural regularization (stoch depth):** CLOSED (see above).
- **Loss weighting (surf_weight):** Open — fern at 15, tanjiro at 25 in flight.
- **Channel weighting (Ux,Uy,p)=(1,1,3):** Open — thorfinn in flight.

## Potential next research directions (R7+)

### After R6 results land

1. **Compound winners** — once R6 completes, stack best-performing orthogonal knobs.
   - Priority: (dropout=0.1 + weight_decay) if edward's rebase succeeds.
   - Priority: (EMA-0.9995 + dropout=0.1) if alphonse wins.
   - If n_hidden=96 wins, test (n_hidden=96 + slice_num=32) as compound.

2. **Re-conditioned FiLM** per Transolver block (log(Re) → (γ, β)) — explicit
   cross-regime conditioning. Targets val_re_rand and single-foil OOD. Most
   architecturally novel lever not yet tried.

3. **Cosine WarmRestarts (SGDR)** — T_0=4, multiple restart cycles in 14 epochs.
   Different from the T_max=14 experiment (single-cycle shrink). No added compute.

4. **WeightedRandomSampler rebalancing** — upsample single-foil training samples
   (currently underrepresented in tandem-heavy data) to attack val_single_in_dist=116.53.
   Single-line change to sampling weights. High-leverage if the single-foil gap is
   a data-distribution problem.

5. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely.
   Potentially better calibrated for noisy short-run training.

### Architectural (if R6 hits plateau)

- **Dual surface vs volume decoder heads** — separate linear for surface/volume.
- **n_layers=4** (drop one block) — reduce depth, save compute, fit more epochs.
- **FiLM conditioning** — explicit Re/geometry conditioning.

## Plateau plan

Progress rate: ~1% per round. 5 consecutive no-improvement rounds not yet reached.
R5 produced three closes — useful saturation signal (capacity doesn't help in budget).
R6 continues compounding. Next escalation trigger: 5+ rounds at ≤0% improvement.
