# SENPAI Research State

- **Updated:** 2026-05-15 17:40 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch yet. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 104.52** (PR #3285, SmoothL1 + EMA-0.999).
`test_avg/mae_surf_p = 99.49` from #3279; #3285's run pre-dated the NaN fix, so
its 3-finite-split test mean = 103.36 is the apples-to-apples test number.
Splits (val): single=130.72, rc=112.51, cruise=79.47, re_rand=95.36.

**Key gap:** `val_single_in_dist=130.72` is now the worst of the four splits.
Single-foil geometry is OOD relative to tandem-heavy training data. The R3
hypotheses target this directly (regularization, capacity, schedule).

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p; require ≥5% delta to
declare a real win.

## Currently in flight (8 WIP — 0 idle)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3376 | alphonse | cosine T_max=50→14 (match wall-clock budget) | SmoothL1+EMA | schedule / LR |
| #3280 | askeladd | SmoothL1 beta=1.0 → 0.5 | SmoothL1+EMA | loss tuning |
| #3325 | edward   | weight_decay 1e-4 → 5e-4 | SmoothL1+EMA | regularization |
| #3324 | fern     | log-cosh loss | SmoothL1+EMA | loss formulation |
| #3124 | frieren  | mlp_ratio 4 (retry on SmoothL1) | SmoothL1+EMA | model capacity |
| #3327 | nezuko   | bf16 + batch=8 + lr=1e-3 | SmoothL1+EMA | throughput / capacity |
| #3286 | tanjiro  | SmoothL1 + surf_weight=25 stack | SmoothL1+EMA | loss stack |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) | SmoothL1+EMA | channel weighting |

Note: askeladd/frieren/tanjiro/thorfinn were assigned before EMA merged; their
runs pick up EMA from the advisor base when they pull. alphonse, edward, fern,
nezuko were assigned after EMA so they run on the full SmoothL1+EMA+NaN-fix
base. All effects are orthogonal so the comparison is well-defined.

Note: edward, nezuko, thorfinn (#3116, #3129, #3135) were assigned before
SmoothL1 merged, so they run on the old MSE base — results will be interpreted
relative to MSE baseline (143.52). All others are on the SmoothL1 base.

## Rounds 1 & 2 summary

| PR | Hypothesis | val_avg/mae_surf_p | vs prev baseline | Decision |
|----|------------|-------------------:|-------:|----------|
| #3107 | baseline (MSE default) | 143.52 | — | Closed (calibration) |
| #3111 | SmoothL1 beta=1.0 | 115.17 | -19.7% (vs MSE) | MERGED |
| #3132 | LR warmup (10%) | 141.73 | -1.3% (vs MSE) | Closed (noise) |
| #3124 | mlp_ratio=4 | 134.14 | -6.5% (vs MSE) | Sent back (retry on SmoothL1) |
| #3120 | slice_num=128 | 147.74 | +2.9% (vs MSE) | Closed (regression) |
| #3116 | surf_weight=25 (MSE) | 127.86 | -10.9% (vs MSE) | Closed (subsumed by #3286) |
| #3129 | bf16 autocast | 111.99 | +3.2% (vs 108.47) | Closed (regression, no throughput) |
| #3279 | NaN-safe scoring (infra) | 108.47 | -5.8% (stochastic) | MERGED |
| #3285 | EMA-0.999 weights | **104.52** | **-3.6% (vs 108.47)** | **MERGED ← new baseline** |
| #3299 | OneCycleLR max_lr=1e-3 (T_max=50 bug) | 132.61 | +27% (vs 104.52) | Closed (regression; T_max mismatch) |

## Potential next research directions

After round 3 lands (8 PRs in flight covering schedule/loss/regularization/
capacity/throughput), the remaining bench from the original researcher-agent
list (`research/RESEARCH_IDEAS_2026-05-15_12:30.md`):

- **Re-conditioned FiLM** per Transolver block (2-layer MLP `log(Re) → (γ, β)`)
  — explicit cross-regime conditioning, targets `val_re_rand` and the
  single-foil geometric gap.
- **Dual surface vs volume decoder heads** — separate `nn.Linear(n_hidden, 3)`
  for surface and volume nodes. Decouples the metric-channel learning.
- **AoA sin/cos encoding** — replace raw radians with periodic encoding for
  the foil-pitch conditioning input.
- **Mesh-node subsampling** during training (50% of non-surface nodes per
  step) — fits more samples per batch, regularizes the volume objective.
- **Stronger positional encoding** — `unified_pos=True` with various ref grids,
  Fourier / RFF on (x, z) input dims.
- **Stochastic depth** on transformer blocks (Huang et al. 2016) — modern ViT
  regularization, complements EMA + weight_decay.
- **Compound winners** — stack the best R3 levers (schedule + regularization +
  loss tuning) into a single end-of-program "all-in" run.
- **Sweep EMA decay** — fern noted val was still descending at the cutoff; try
  decay=0.9995 (longer half-life) or shorter cap to confirm the EMA shape.

**Plateau watch:** R3 represents the third tier of single-knob experiments.
If no further wins land after R3, escalate per the plateau protocol —
researcher-agent + architectural changes (FiLM, dual heads).

## Plateau plan (if it happens)

If 5+ consecutive rounds show no improvement, escalate per the program-level
plateau protocol: from knob tuning → architecture → loss reformulation → data
representation. Use the researcher-agent to mine new literature and try bigger
swings.
