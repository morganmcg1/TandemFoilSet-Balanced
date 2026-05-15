# SENPAI Research State

- **Updated:** 2026-05-15 14:15 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch yet. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 115.17** (PR #3111, SmoothL1 Huber beta=1.0, -19.7% vs MSE default).
Splits: single=144.61, rc=124.04, cruise=89.33, re_rand=102.70.

## Currently in flight (8 WIP PRs)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3279 | alphonse | data/scoring.py NaN-safe fix | SmoothL1 | infra bug fix |
| #3280 | askeladd | SmoothL1 beta=1.0 → 0.5 | SmoothL1 | loss tuning |
| #3116 | edward   | surf_weight 10 → 25 (MSE base) | MSE | loss alignment |
| #3285 | fern     | EMA weights decay=0.999 | SmoothL1 | OOD generalization |
| #3124 | frieren  | mlp_ratio 4 (retry on SmoothL1) | SmoothL1 | model capacity |
| #3129 | nezuko   | bf16 autocast | MSE | throughput |
| #3286 | tanjiro  | SmoothL1 + surf_weight=25 stack | SmoothL1 | loss stack |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) (MSE base) | MSE | channel weighting |

Note: edward, nezuko, thorfinn (#3116, #3129, #3135) were assigned before
SmoothL1 merged, so they run on the old MSE base — results will be interpreted
relative to MSE baseline (143.52). All others are on the SmoothL1 base.

## Round 1 summary (reviewed so far)

| PR | Hypothesis | val_avg/mae_surf_p | vs MSE | Decision |
|----|------------|-------------------:|-------:|----------|
| #3107 | baseline (MSE default) | 143.52 | — | Closed (calibration) |
| #3111 | SmoothL1 beta=1.0 | **115.17** | **-19.7%** | **MERGED ← new baseline** |
| #3132 | LR warmup (10%) | 141.73 | -1.3% | Closed (noise) |
| #3124 | mlp_ratio=4 | 134.14 | -6.5% | Sent back (retry on SmoothL1) |
| #3120 | slice_num=128 | 147.74 | +2.9% | Closed (regression) |

## Potential next research directions

Round 2 candidates (from researcher-agent on 2026-05-15 12:30 — full file at
`research/RESEARCH_IDEAS_2026-05-15_12:30.md`), prioritized assuming the round
1 winners point us at loss alignment first:

- **EMA / Polyak averaging** of weights with decay 0.999 — free OOD boost,
  especially for the camber holdouts.
- **Re-conditioned FiLM** per Transolver block (2-layer MLP `log(Re) → (γ, β)`)
  — explicit cross-regime conditioning for `val_re_rand`.
- **Dual surface vs volume decoder heads** — separate `nn.Linear(n_hidden, 3)`
  for surface and volume nodes.
- **Log-cosh loss** — alternative to SmoothL1; if SmoothL1 wins, log-cosh is
  the natural variant.
- **OneCycleLR** instead of cosine — sharper peak, often beats cosine on
  short-horizon trainings.
- **AoA sin/cos encoding** — replace raw radians with periodic encoding.
- **Mesh-node subsampling** during training (e.g. 50% of non-surface nodes per
  step) to fit more samples per batch.
- **Stronger positional encoding** — `unified_pos=True` with various ref grids,
  Fourier / RFF on (x, z) input dims.
- **Regularization for OOD** — dropout sweep, stochastic depth on transformer
  blocks.
- **Compound winners** — once round 1 lands, stack the best loss formulation
  with the best capacity setting in round 2.

## Plateau plan (if it happens)

If 5+ consecutive rounds show no improvement, escalate per the program-level
plateau protocol: from knob tuning → architecture → loss reformulation → data
representation. Use the researcher-agent to mine new literature and try bigger
swings.
