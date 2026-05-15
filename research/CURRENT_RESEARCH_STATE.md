# SENPAI Research State

- **Updated:** 2026-05-15 22:40 UTC
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
training data. Best remaining levers: capacity (n_hidden=160, mlp_ratio=4),
architectural regularization (stochastic depth), loss weighting (surf_weight).

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p. New baseline is
single-seed — for merges against it, require 8/8 split directional consistency
OR ≥3% improvement (val_avg ≤ ~93.3) to be a clear winner.

## Round 4/5 wins merged

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | 97.15 | -1.32% vs 98.45 | MERGED |
| #3402 | dropout=0.1 in PhysicsAttention (single-seed) | **96.17** | **-1.01% vs 97.15** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3510 | nezuko   | dropout=0.1→0.2 push | dropout+beta+EMA | regularization sweep (R5) |
| #3471 | alphonse | Stochastic depth p=0.1 | beta+EMA+dropout | architectural regularization (R5) |
| #3472 | askeladd | n_hidden=128→160 | beta+EMA+dropout | capacity (R5) |
| #3482 | fern     | surf_weight=10→15 | beta+EMA+dropout | loss weighting (R5) |
| #3325 | edward   | weight_decay=5e-4 rebase | beta+EMA+dropout | regularization (R4, sent back) |
| #3124 | frieren  | mlp_ratio=4 | beta+EMA+dropout | capacity (R4, baseline-update sent) |
| #3286 | tanjiro  | surf_weight=25 | beta+EMA+dropout | loss weighting (R4, stale) |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) | beta+EMA+dropout | channel weighting (R4, stale) |

**Note on tanjiro/thorfinn/frieren (old PRs):** These PRs predate the dropout=0.1 merge. Their experiments may not include dropout=0.1 in the base. New baseline now includes dropout; if results land above 96.17, send back for rebase.

## Plateau / saturation map (R1-R5)

- **Loss formulation:** SmoothL1 beta=0.25 is the optimum in the smooth Huber family. Beta curve at {1.0, 0.5, 0.25} flattened. Axis closed.
- **Throughput / batch size:** H100 memory-bandwidth-bound. No batch-size lever.
- **NaN-safe scoring + EMA-0.999:** Banked.
- **Schedule shape:** Cosine T_max=14 overlaps with beta=0.5. Axis closed for simple T_max shrinking. WarmupCosine/SGDR remain viable.
- **Regularization (dropout):** 0.1 merged (+1% improvement, 8/8 consistency). Dropout=0.2 in flight (nezuko #3510). Train loss showed no slowdown at 0.1 — headroom exists.
- **Regularization (weight_decay):** Open — edward rebase pending.
- **Cyclic AoA encoding:** CLOSED. AoA range too narrow for cyclic benefit.
- **Capacity (n_hidden, mlp_ratio):** Open — askeladd n_hidden=160 and frieren mlp_ratio=4 in flight.
- **Architectural regularization (stoch depth):** Open — alphonse in flight.
- **Loss weighting (surf_weight):** Open — fern at 15, tanjiro at 25 in flight.
- **Channel weighting (Ux,Uy,p)=(1,1,3):** Open — thorfinn in flight.

## Potential next research directions (R6+)

### After R5 results land

1. **Compound winners** — once R5 completes, stack best-performing orthogonal knobs.
   - Priority compound: (dropout + weight_decay) if both win.
   - Priority compound: (stoch_depth + dropout) if both win.

2. **Re-conditioned FiLM** per Transolver block (log(Re) → (γ, β)) — explicit
   cross-regime conditioning. Targets val_re_rand and single-foil OOD. Most
   architecturally novel lever not yet tried.

3. **EMA decay = 0.9995** — tighter Polyak. Cheap one-liner. Not tried yet.

4. **n_layers=6** — add one more Transolver block. Depth vs width trade-off
   vs askeladd's n_hidden=160 experiment.

5. **Cosine WarmRestarts (SGDR)** — T_0=4, multiple restart cycles in 14 epochs.
   Different from the T_max=14 experiment (which was single-cycle shrink).

### Architectural (if R5 hits plateau)

- **Dual surface vs volume decoder heads** — separate linear for surface/volume.
- **Mesh-node subsampling** — 50% of non-surface nodes per step.
- **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling.

## Plateau plan

Progress rate: ~1% per round. 5 consecutive no-improvement rounds not yet reached.
Continue compounding small wins. Next escalation trigger: 5+ rounds at ≤0% improvement.
