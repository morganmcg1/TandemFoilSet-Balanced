# SENPAI Research State

- 2026-05-15 17:50 — **first winner merged**. PR #3150 (tanjiro warmup+cosine)
  is now the baseline. Assigned new orthogonal work to tanjiro (#3370 GLU MLPs).
- No directives from the human researcher team yet on this launch.

## Current research focus

Round 2+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`. **One winner
merged**: tanjiro's warmup+cosine schedule (PR #3150). New baseline:
**`val_avg/mae_surf_p = 125.83`** (was 130 ± 3 pre-merge). Eight hypotheses
in flight.

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6`

## Key constraints / insights confirmed

- **30-min wall-clock cap is the dominant compute budget.** Width (#3148)
  and depth (#3145) scaling both lost because per-step cost outpaces
  sample-efficiency gain within the cap. AMP (#3330) directly targets this.
- **Warmup is a near-free win for transformer-style architectures.** PR #3150
  showed −11.6% vs internal control with no extra compute.
- **Run-to-run variance ~3-4 mae_surf_p units.** Improvements <3 units need
  multiple seeds.
- **Higher peak LRs don't help in this budget** even with warmup. Both 1e-3
  and 1.5e-3 underperformed in #3150. Probably budget-bound rather than
  optimization-bound at 30 min.
- **Shared-readout bottleneck identified** in #3149 (loss-weighting test).
  Architectural fix in flight as #3331 (per-channel output heads).
- **`test_avg/mae_surf_p` is None for all runs** due to Inf in cruise test GT.
  Students compute `test_avg_3splits/mae_surf_p` as a partial paper-facing
  companion metric. Known multi-launch issue (#1569/#1567/#3292).

## In-flight hypotheses (9 active PRs)

| PR | Student | Lever | Notes |
|----|---------|-------|-------|
| #3138 | alphonse  | PhysicsAttention slice count (`slice_num` ∈ {64,128,256}) | wip |
| #3142 | askeladd  | Surface-loss weight (`surf_weight` ∈ {10,30,80}) | wip (training) |
| #3143 | edward    | Robust loss (Charbonnier ε=1e-3 vs MSE) | wip (training) |
| #3151 | thorfinn  | EMA model weights (`ema_decay` ∈ {0, 0.999, 0.9999}) | wip (training) |
| #3330 | frieren   | bf16 AMP mixed precision (more epochs per 30-min budget) | wip |
| #3331 | nezuko    | Separate per-channel output heads | wip |
| #3348 | fern      | Fourier position encoding (multi-scale spatial frequency basis) | wip |
| #3370 | tanjiro   | Gated MLPs (SwiGLU / GeGLU in TransolverBlocks) | wip (new) |

## Closed / merged this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; closed |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal; closed |
| #3145 | fern    | Deeper Transolver — same wall-clock confound as #3148; closed |
| #3150 | tanjiro | **Warmup + cosine LR — merged ⭐ (val_avg/mae_surf_p 125.83)** |

## Strategic insight: orthogonal levers and compounding

Of the 8 in-flight PRs, 6 are clean orthogonal axes to the merged warmup
schedule (slice_num, surf_weight, Charbonnier, EMA, AMP, separate heads,
Fourier features, GLU MLPs). Each winner that lands should compound onto
the new baseline cleanly.

The most expected high-value paths into the 110-115 range:
1. **AMP (#3330)** lets every other arm finish more epochs → may unlock
   re-running width/depth experiments that lost on wall-clock.
2. **GLU MLPs (#3370)** is the strongest single architectural lever in
   transformer literature; +1-3% reported.
3. **Fourier pos enc (#3348)** for the multi-scale CFD spatial structure.
4. **Separate output heads (#3331)** breaks the shared-readout bottleneck
   we already diagnosed in PR #3149.

## Potential next-round directions (after current 8 PRs review)

- **Compound merged winners** — once 2-3 PRs land, stack and re-baseline.
- **Per-step warmup** (tanjiro's own follow-up suggestion #3) — finer ramp,
  may unlock higher LRs that failed in #3150.
- **Gradient clipping** (max_norm=1.0) — virtually free stabilizer,
  pairs with future per-step warmup.
- **RMSNorm** vs LayerNorm — slightly faster, often comparable.
- **Larger batch with AMP** — if bf16 wins, retest batch_size 8/16.
- **Data augmentation** — chord rotation + NACA-symmetric flip for raceCar
  domains (mathematical symmetries of the dataset).
- **Curriculum** — order training by mesh size or Re for stability.
- **Mesh-aware encodings** — SDF, surface normals, KNN positional encoding
  (extends fern's Fourier work if it wins).
- **Test-time augmentation** — average predictions over k symmetric forward
  passes.
- **Sample weighting refinements** — hardness-based weighting on top of
  current dataset-level domain weights.
- **Data-side fix for cruise NaN** — would unblock the paper-facing test_avg
  metric for all runs (currently 3-split partial only).
