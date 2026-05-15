# SENPAI Research State

- 2026-05-15 18:40 — **second winner merged**. PR #3143 (edward Charbonnier
  robust loss, ε=1e-3) is now the baseline. Assigned new orthogonal work to
  edward (#3398 Charbonnier ε sweep). Sent #3138 back to alphonse: slice_num
  was a dead end but his \`evaluate_split\` NaN bug fix is gold — asked him
  to repurpose the PR into a bug-fix-only PR off the new advisor head.
- No directives from the human researcher team yet on this launch.

## Current research focus

Round 3+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`. **Two
winners merged**: warmup+cosine (#3150) and Charbonnier robust loss
(#3143). New baseline: **`val_avg/mae_surf_p ≈ 98.60`** (down from
~130 pre-merge); composed warmup + Charbonnier number will be confirmed
by edward's #3398 \`charb_eps1e-3\` control arm. 8 hypotheses in flight.

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143)

## Key constraints / insights confirmed

- **Charbonnier robust loss is a massive win.** −18.6% on val_avg/mae_surf_p
  vs MSE control. Largest gain on `val_geom_camber_cruise` (−25.6%), which
  has highest pressure dynamic range — direct evidence that MSE was being
  dominated by a few near-stagnation outliers per mesh. Robust loss bounds
  their gradient and unlocks the bulk of the surface. Improves every
  channel, not just surf_p.
- **30-min wall-clock cap is still the dominant compute budget.** Width
  (#3148), depth (#3145), and slice_num (#3138) sweeps all lost on the
  same wall-clock confound — per-step cost outpaces sample-efficiency
  gain within the cap. AMP (#3330) directly targets this.
- **Warmup is a near-free win for transformer-style architectures.**
  PR #3150 showed −11.6% vs internal control with no extra compute.
- **Run-to-run variance ~3-4 mae_surf_p units.** Improvements <3 units
  need multiple seeds.
- **Shared-readout bottleneck identified** in #3149. Architectural fix
  in flight as #3331 (per-channel output heads).
- **`test_avg/mae_surf_p` Inf bug** — alphonse (#3138) found and fixed
  it at the `evaluate_split` boundary. The fix is: filter samples with
  non-finite `y` _before_ the `err * mask` step, sidestepping
  `NaN * 0 == NaN`. Repurposed his PR to be a bug-fix-only PR off the
  new advisor head — should land fast and unblock paper-facing test
  metric for every other PR.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Notes |
|----|---------|-------|-------|
| #3138 | alphonse  | **`evaluate_split` NaN-fix only** (repurposed) | sent back; fixes test_avg/mae_surf_p for all PRs |
| #3142 | askeladd  | Surface-loss weight (`surf_weight` ∈ {10,30,80}) | wip; stale (last update ~6h ago) |
| #3151 | thorfinn  | EMA model weights (`ema_decay` ∈ {0, 0.999, 0.9999}) | wip; stale (last update ~6h ago) |
| #3330 | frieren   | bf16 AMP mixed precision (more epochs per 30-min budget) | wip |
| #3331 | nezuko    | Separate per-channel output heads | wip |
| #3348 | fern      | Fourier position encoding (multi-scale spatial frequency basis) | wip |
| #3370 | tanjiro   | Gated MLPs (SwiGLU / GeGLU in TransolverBlocks) | wip |
| #3398 | edward    | **Charbonnier ε sweep** ({3e-4, 1e-3, 3e-3}) (new) | wip |

### Stale-WIP investigation

`#3142` (askeladd) and `#3151` (thorfinn) haven't updated since ~12:43
UTC (about 6 hours). The prior iteration noted 6 of 8 student pods were
stuck on a GitHub REST rate-limit exhaustion. Some have since unblocked
(edward delivered, alphonse reviewed). Worth a pod-status check next
iteration if these two don't move by the next wake-up. REST core is
currently 0/5000 (advisor side); reset in ~40 min.

## Closed / merged this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; closed |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal; closed |
| #3145 | fern    | Deeper Transolver — same wall-clock confound as #3148; closed |
| #3150 | tanjiro | **Warmup + cosine LR — merged ⭐ (val_avg/mae_surf_p 125.83)** |
| #3143 | edward  | **Charbonnier robust loss — merged ⭐⭐ (val_avg/mae_surf_p 98.60, −18.6%)** |
| #3138 | alphonse | slice_num sweep — closed; PR repurposed for bug fix |

## Strategic insight: orthogonal levers and compounding

Of the 7 fresh in-flight PRs (excluding alphonse's bug fix), all 7 are
clean orthogonal axes to the merged baseline (slice_num, surf_weight,
EMA, AMP, separate heads, Fourier features, GLU MLPs, Charbonnier ε).
Each winner that lands should compound onto the new baseline cleanly.

Two big merges in a row (warmup, Charbonnier) suggest we're early in
the curve and there's plenty of headroom. The Charbonnier win in
particular was much bigger than the warmup win, which was already a
solid 11.6% — robust-loss is doing real work, not just regularizing
edge cases. This validates the broader thesis that for CFD surrogates
with dynamic-range targets, the loss form matters more than the
architecture knobs we've been tuning.

Most expected high-value paths from the merged baseline of 98.60:
1. **AMP (#3330)** unlocks more epochs in the budget; will let us
   re-examine width/depth/slice that lost on wall-clock.
2. **GLU MLPs (#3370)** strongest single architectural transformer
   lever; +1-3% reported.
3. **Charbonnier ε sweep (#3398)** dials the L1/L2 trade-off; may
   yield another 1-3%.
4. **Fourier pos enc (#3348)** for multi-scale CFD spatial structure.
5. **Separate output heads (#3331)** breaks shared-readout bottleneck
   diagnosed in PR #3149.

## Potential next-round directions (after current 8 PRs review)

**Loss-form follow-ups (high-leverage after Charbonnier win):**
- **Huber / pseudo-Huber / Cauchy / Lorentzian losses** — natural
  extensions of the Charbonnier robust-loss win. Cauchy in particular
  has unbounded influence reduction (asymptotically constant) vs
  Charbonnier's linear, which may help with extreme outliers.
- **Per-channel ε** — surf_p has highest dynamic range, may want a
  larger ε on Ux/Uy than on p.
- **Re-tune surf_weight after Charbonnier** — gradient magnitudes
  rebalanced; the optimal weight may have shifted from 10.

**Architecture:**
- **RMSNorm vs LayerNorm** — slightly faster, often comparable.
- **Per-layer slice_num schedule** (coarse-to-fine).
- **Mesh-aware encodings** — SDF, surface normals, KNN positional
  encoding (extends fern's Fourier work if it wins).

**Training:**
- **Gradient clipping** (max_norm=1.0) — virtually free stabilizer.
- **Per-step warmup** (tanjiro's #3 follow-up) — finer ramp.
- **Larger batch with AMP** — if bf16 wins, retest batch_size 8/16.

**Data:**
- **Augmentation** — chord rotation + NACA-symmetric flip for
  raceCar (mathematical symmetries of the dataset).
- **Curriculum** — order training by mesh size or Re.
- **Sample weighting refinements** — hardness-based weighting on top
  of current dataset-level domain weights.
- **Test-time augmentation** — average predictions over k symmetric
  forward passes.

**Compounding (after 2+ winners on the new baseline):**
- Re-baseline and stack winners. With warmup+Charbonnier already
  composing, the next winner forms a 3-stack.
