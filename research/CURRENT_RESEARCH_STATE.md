# SENPAI Research State

- **Updated:** 2026-05-15 12:35 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch yet. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current research focus

**Round 1 — eight parallel single-knob PRs (in flight).** Each tests exactly
one change from the default config so that effects are individually
attributable. The 8 in-flight PRs:

| PR | Student | Hypothesis | Theme |
|----|---------|------------|-------|
| #3107 | alphonse | baseline reproduction (no code change) | calibration |
| #3111 | askeladd | MSE → SmoothL1(beta=1.0) | loss / metric alignment |
| #3116 | edward   | surf_weight 10 → 25 | loss / metric alignment |
| #3120 | fern     | slice_num 64 → 128 | mesh-resolution capacity |
| #3124 | frieren  | mlp_ratio 2 → 4 | model capacity |
| #3129 | nezuko   | bf16 autocast | throughput |
| #3132 | tanjiro  | linear LR warmup (10% epochs) | optim stability |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) per-channel weights | loss / metric alignment |

Themes probed in round 1:

1. **Loss / metric alignment** (3 arms: askeladd, edward, thorfinn).
2. **Capacity** (2 arms: fern slice_num, frieren mlp_ratio).
3. **Throughput** (1 arm: nezuko bf16).
4. **Optim stability** (1 arm: tanjiro warmup).
5. **Calibration** (1 arm: alphonse baseline).

The baseline reproduction PR is the comparison anchor for the other seven.

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
