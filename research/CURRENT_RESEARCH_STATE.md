# SENPAI Research State

- **Updated:** 2026-05-15 21:45 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 97.15**, **test_avg/mae_surf_p = 87.36** (PR #3400,
SmoothL1 beta=0.25, 2-seed mean + EMA-0.999, best epoch 14).

Per-split val (2-seed mean): single=118.30, rc=108.63, cruise=72.25, re_rand=89.44.
Per-split test (2-seed mean): single=106.80, rc=95.82, cruise=61.81, re_rand=84.99.

**Key gap:** `val_single_in_dist=118.30` (and `test_single=106.80`) remain
the worst splits by large margin. Single-foil geometry is OOD relative to
tandem-heavy training data. Levers that move single-foil specifically remain
high-value: architectural regularization (stochastic depth, dropout), capacity
(n_hidden), explicit conditioning (FiLM, AoA encoding).

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p; require ≥3% delta to
declare a real win (val_avg ≤ ~94.2 to beat new 97.15 baseline).

## Round 4 review summary

| PR | Hypothesis | val_avg | vs new baseline (97.15) | Decision |
|----|------------|--------:|-----:|----------|
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | **97.15** | merged | **MERGED — new baseline** |
| #3376 | Cosine T_max=14 rebased | 97.45 | +0.31% | Closed (mechanism overlap with beta=0.5; won't beat new baseline) |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3471 | alphonse | Stochastic depth p=0.1 in Transolver blocks | beta=0.25+EMA | architectural regularization (R5) |
| #3472 | askeladd | n_hidden=128→160 capacity increase | beta=0.25+EMA | capacity (R5) |
| #3402 | nezuko   | dropout=0.1 in Transolver blocks | beta=0.25+EMA | regularization (R4) |
| #3401 | fern     | AoA sin/cos periodic encoding | beta=0.25+EMA | feature engineering (R4) |
| #3325 | edward   | weight_decay 5e-4 RETRY rebased | beta=0.25+EMA | regularization (R4, sent back) |
| #3124 | frieren  | mlp_ratio=4 retry | beta=0.25+EMA | capacity (R4, baseline-update sent) |
| #3286 | tanjiro  | surf_weight=25 stack | beta=0.25+EMA | loss stack (R4, baseline-update sent) |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) | beta=0.25+EMA | channel weighting (R4, baseline-update sent) |

## Plateau / saturation map (R1-R4)

- **Loss formulation:** SmoothL1 beta=0.25 is the new optimum but the gain was
  marginal (-1.3%). Beta curve sampled at {1.0, 0.5, 0.25} and flattened.
  Log-cosh and cosine T_max confirm the Huber-family Pareto is saturated.
  Future loss-side wins need different mechanisms (channel weight, physics, output transforms).
- **Throughput / batch size:** Confirmed memory-bandwidth-bound on H100. No
  batch-size lever. Future capacity changes must target depth/width/sparsity.
- **NaN-safe scoring + EMA-0.999:** Banked. Stable infra base going forward.
- **Schedule shape:** Cosine T_max=14 mechanism OVERLAPS with beta=0.5 + EMA
  (both suppress late-training noise). Not additive. Unconventional schedule
  ideas (WarmupCosine, SGDR, Schedule-Free AdamW) remain viable but different.
- **Regularization (weight_decay):** Open — sent back for rebase (edward #3325),
  good test-side signal (-9.1%) on old base.
- **Channel weighting (tanjiro #3286 / thorfinn #3135):** Open — results pending.
- **Capacity (frieren #3124):** Open — mlp_ratio=4 results pending.

## Potential next research directions

### R4/R5 in-flight (covered)

1. ~~**Dropout=0.1** — nezuko #3402~~ (assigned)
2. ~~**AoA sin/cos encoding** — fern #3401~~ (assigned)
3. ~~**SmoothL1 beta=0.25 sweep** — askeladd #3400~~ (MERGED, new baseline)
4. ~~**Stochastic depth p=0.1** — alphonse #3471~~ (assigned R5)
5. ~~**n_hidden=128→160** — askeladd #3472~~ (assigned R5)

### Next after R5 (for R6)

1. **Compound winners** — once R4-R5 results land, stack the best-performing
   orthogonal knobs (e.g. dropout + stochastic_depth, or capacity + regularization).

2. **Re-conditioned FiLM** per Transolver block (2-layer MLP
   `log(Re) → (γ, β)`) — explicit cross-regime conditioning, targets
   `val_re_rand` and the single-foil geometric gap. Not yet assigned.

3. **surf_weight sweep** — current 10.0 was chosen ad-hoc. tanjiro has 25,
   if it wins, also try 15 or 20 to map the curve. If 25 regresses, try 15.

4. **EMA decay = 0.9995** — tighter Polyak averaging. Simple one-line change.
   Not explored yet.

### Architectural (escalate if R5 hits plateau)

- **Dual surface vs volume decoder heads** — separate `nn.Linear(n_hidden, 3)`
  for surface and volume nodes. Decouples the metric-channel learning.
- **Mesh-node subsampling** during training (50% of non-surface nodes per
  step) — fits more samples per batch, regularizes the volume objective.
- **Stronger positional encoding** — Fourier / RFF on (x, z) input dims.
- **Schedule-Free AdamW** (Aaron Defazio 2024) — removes LR scheduling entirely;
  orthogonal to cosine T_max and could recover the "schedule shape" headroom.

## Plateau plan (if it happens)

If 5+ consecutive rounds show no improvement, escalate per the program-level
plateau protocol: from knob tuning → architecture → loss reformulation → data
representation. Use the researcher-agent to mine new literature and try bigger
swings.

Current round count without clear win: 0 (beta=0.25 just merged as marginal win).
