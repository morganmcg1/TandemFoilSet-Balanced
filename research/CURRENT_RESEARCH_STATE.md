# SENPAI Research State

- **Updated:** 2026-05-15 18:40 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch yet. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 98.45**, **test_avg/mae_surf_p = 87.63** (PR #3280,
SmoothL1 beta=0.5 + EMA-0.999 + NaN-safe scoring, best epoch 14).

Per-split val: single=119.70, rc=108.17, cruise=74.09, re_rand=91.84.
Per-split test: single=106.01, rc=94.91, cruise=63.44, re_rand=86.17.

**Key gap:** `val_single_in_dist=119.70` (and `test_single=106.01`) remain
the worst splits. Single-foil geometry is OOD relative to tandem-heavy
training data. Hypothesis levers that move single-foil specifically are
high-value: regularization (weight_decay, dropout), geometric augmentation,
explicit conditioning (FiLM, AoA encoding).

Single-seed variance ≈ ±5-10 pts on val_avg/mae_surf_p; require ≥5% delta to
declare a real win.

## Round 3 review summary (just landed)

| PR | Hypothesis | val_avg | vs new baseline (98.45) | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | **98.45** | merged | **MERGED — new baseline** |
| #3325 | weight_decay=5e-4 | 101.17 (mean of 2) | +2.8% | Sent back for rebase + compound |
| #3327 | bf16 + bs=8 + lr=1e-3 | 131.32 | +33.4% | Closed (regression; H100 bandwidth-bound) |
| #3324 | log-cosh loss | 103.47 (mean of 2) | +5.1% | Closed (tie within noise, loss Pareto saturated) |

## Currently in flight (4 WIP — 4 newly idle after R3 reviews)

| PR | Student | Hypothesis | Base | Theme |
|----|---------|------------|------|-------|
| #3376 | alphonse | cosine T_max=14 (match wall-clock) | SmoothL1 beta=1.0 + EMA | schedule (will need re-eval on new base) |
| #3325 | edward   | weight_decay 5e-4 RETRY on new base | SmoothL1 beta=0.5 + EMA | regularization compound |
| #3124 | frieren  | mlp_ratio=4 retry | SmoothL1 beta=1.0 + EMA | capacity (will need re-eval) |
| #3286 | tanjiro  | SmoothL1 + surf_weight=25 stack | SmoothL1 beta=1.0 + EMA | loss stack (will need re-eval) |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) | MSE (pre-SmoothL1) | channel weighting (was rate-limited) |

**Idle after R3 reviews:** askeladd, fern, nezuko (PRs closed/merged).
Edward sent back, so likely returns to WIP shortly.

**Pending assignments (rate-limited at 18:42 UTC, GH core resets 19:19 UTC):**
PR bodies prepared and ready to submit at `/tmp/pr_bodies/`:
- `askeladd_smoothl1_beta025.md` — SmoothL1 beta=0.25 sweep continuation (build on own R3 win)
- `fern_aoa_sincos.md` — AoA periodic encoding (cyclic features for foil pitch)
- `nezuko_dropout01.md` — dropout=0.1 inside Transolver blocks (regularization orthogonal to wd)

Also need to verify #3135 (thorfinn) state on next invocation — was rate-limited
last cycle.

**Note on rebase risk:** PRs #3376, #3124, #3286 were authored against earlier
bases (SmoothL1 beta=1.0 + EMA-0.999). On merge of #3280 (beta=0.5), they
should rebase or their results need to be re-interpreted against the new
baseline 98.45.

## Plateau / saturation map (R1-R3)

- **Loss formulation:** SmoothL1 beta=0.5 is the local optimum within
  smooth Huber-family losses. Log-cosh confirms saturation. Future
  loss-side wins need different mechanisms (channel weight, physics, output
  transforms).
- **Throughput / batch size:** Confirmed memory-bandwidth-bound on H100. No
  batch-size lever. Future capacity changes must target depth/width/sparsity.
- **NaN-safe scoring + EMA-0.999:** Banked. Stable infra base going forward.
- **Schedule shape (alphonse #3376):** Open — first valid cosine T_max
  experiment pending result.
- **Regularization (weight_decay):** Open — single-knob result was marginal
  on val but strong on test; compound test on top of beta=0.5 pending.
- **Channel weighting (tanjiro #3286 / thorfinn #3135):** Open — pending
  results, both pre-date the new baseline.

## Potential next research directions

### Highest priority (target idle students next)

1. **Stochastic depth** on transformer blocks (Huang et al. 2016, ~p=0.1) —
   modern ViT regularization, orthogonal to EMA + weight_decay. Single-knob
   change in Transolver block forward pass. Targets the single-foil OOD gap
   the same way wd does, via a different mechanism.

2. **Dropout=0.1 inside Transolver MLP blocks** — paired naturally with the
   `weight_decay` retry; if both regularizers work, a future compound
   experiment becomes the next assignment.

3. **AoA sin/cos encoding** — replace raw radians with periodic encoding for
   the foil-pitch conditioning input. Cheap, principled, addresses the
   geometric conditioning gap directly.

4. **Re-conditioned FiLM** per Transolver block (2-layer MLP
   `log(Re) → (γ, β)`) — explicit cross-regime conditioning, targets
   `val_re_rand` and the single-foil geometric gap.

### Architectural (escalate if R4 hits plateau)

- **Dual surface vs volume decoder heads** — separate `nn.Linear(n_hidden, 3)`
  for surface and volume nodes. Decouples the metric-channel learning.
- **Mesh-node subsampling** during training (50% of non-surface nodes per
  step) — fits more samples per batch, regularizes the volume objective.
- **Stronger positional encoding** — `unified_pos=True` with various ref grids,
  Fourier / RFF on (x, z) input dims.

### Confirmation / sweep

- **Sweep SmoothL1 beta** in {0.25, 0.5, 0.75, 1.0} — confirm 0.5 is
  optimum; could be lower.
- **Sweep EMA decay** in {0.999, 0.9995} — verify R2 winner generalizes.
- **Compound winners** — stack the best R3-R4 levers (schedule + regularization
  + loss tuning) into a single end-of-program "all-in" run.

## Plateau plan (if it happens)

If 5+ consecutive rounds show no improvement, escalate per the program-level
plateau protocol: from knob tuning → architecture → loss reformulation → data
representation. Use the researcher-agent to mine new literature and try bigger
swings.
