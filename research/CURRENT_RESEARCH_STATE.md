# SENPAI Research State

- 2026-05-15 17:25 — updated after reviewing #3145 (closed) and assigning round-2 work to fern.
- No directives from the human researcher team yet on this launch.

## Current research focus

Round 1+2 on advisor branch `icml-appendix-willow-pai2i-24h-r1`. Three PRs
reviewed and closed (no winners). Implicit baseline:
**`val_avg/mae_surf_p` ≈ 130 ± 3** (mean of three baseline-equivalent control
arms: 128.46, 129.07, 132.33). Eight hypotheses currently in flight.

Key constraints confirmed across multiple PRs:
- **30-min wall-clock cap is the dominant budget constraint.** Both width
  scaling (#3148) and depth scaling (#3145) lose because per-step compute
  cost outpaces sample-efficiency gains within the cap. The bf16 AMP work
  in #3330 directly targets this bottleneck.
- **Run-to-run variance ~3-4 mae_surf_p units** — improvements <3 units
  need multiple seeds to confirm.
- **Baseline is still improving at cutoff** — best_epoch 13-14/50 for the
  baseline width arms; the cosine LR schedule doesn't get a chance to fully
  anneal in 30 min.
- **Shared-readout bottleneck identified.** Per-channel loss weighting
  (#3149) showed surf-p upweighting trades off volume metrics. Architectural
  fix is in flight as #3331 (separate per-channel output heads).
- **`test_avg/mae_surf_p` is None for all runs** due to Inf in cruise test
  GT (multi-launch known issue #3292/#1569/#1567). Rank by
  `val_avg/mae_surf_p`. Some students compute a 3-split partial test mean.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Notes |
|----|---------|-------|-------|
| #3138 | alphonse  | PhysicsAttention slice count (`slice_num` ∈ {64,128,256}) | wip |
| #3142 | askeladd  | Surface-loss weight (`surf_weight` ∈ {10,30,80}) | wip |
| #3143 | edward    | Robust loss (Charbonnier ε=1e-3 vs MSE) | wip |
| #3150 | tanjiro   | Warmup + cosine LR (`lr` ∈ {5e-4,1e-3,1.5e-3} with 3-epoch warmup) | wip |
| #3151 | thorfinn  | EMA model weights (`ema_decay` ∈ {0, 0.999, 0.9999}) | wip (poller rate-limited) |
| #3330 | frieren   | bf16 AMP mixed precision (more epochs per 30-min budget) | wip |
| #3331 | nezuko    | Separate per-channel output heads (break shared-readout bottleneck) | wip |
| #3348 | fern      | Fourier position encoding (multi-scale spatial frequency basis) | wip (new) |

## Closed this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; wider models don't converge in 30 min |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal |
| #3145 | fern    | Deeper Transolver — same wall-clock confound as #3148; deeper arms ~2× slower per epoch |

## Strategic insight from rounds 1-2

The **wall-clock cap is the launch-wide primary bottleneck**, confirmed by
two independent compute-scaling experiments. Both reverse-engineer to:
*deeper/wider models would win with more time but don't get it in budget.*
Two orthogonal high-value levers follow:
1. **Throughput multipliers** (AMP, bf16, compiled fwd, larger effective
   batch) — buy more epochs in the same wall-clock. #3330 testing this.
2. **Sample-efficiency multipliers** (better priors at input, better
   readout) — get a better minimum per epoch. #3331 (separate heads) and
   #3348 (Fourier features) testing this from two angles.

If any of the three (#3330, #3331, #3348) wins, all the closed
compute-scaling work (#3148, #3145) becomes worth re-running.

## Potential next-round directions (after current 8 PRs review)

- **Compound round winners** — all current axes are orthogonal by design.
- **Gradient clipping** (`max_norm=1.0`) — virtually free stabilizer,
  often unlocks higher LR; pair with tanjiro's warmup if that wins.
- **SwiGLU / GeGLU** in MLP blocks — known 1-3% gain on transformers.
- **RMSNorm** vs LayerNorm — slightly faster, often comparable.
- **Larger batch with AMP** — if bf16 wins, retest batch_size 8/16.
- **Data augmentation** — chord rotation + NACA-symmetric flip for
  raceCar domains (mathematical symmetries of the dataset).
- **Curriculum** — order training by mesh size or Re for stability.
- **Mesh-aware encodings** — signed-distance field, surface normals,
  KNN positional encoding (extends fern's Fourier work if it wins).
- **Test-time augmentation** — average predictions over k symmetric
  forward passes.
- **Sample weighting refinements** — currently `WeightedRandomSampler`
  uses dataset-level domain weights; consider hardness-based weighting.
