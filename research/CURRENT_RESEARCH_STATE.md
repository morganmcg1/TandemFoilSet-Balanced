# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

bf16 (PR #359) remains the round baseline at **val_avg/mae_surf_p = 121.85**, **test_avg = 111.15** — a 15.5% improvement over the original PR #312 baseline. The throughput frontier is the dominant theme: bf16 unlocked 5 extra epochs of cosine annealing and ~63 GB of headroom. **Throughput-via-sampler is now ruled out** (PR #360 bsz=8 and PR #384 bucketed both regressed); next throughput plays would be `torch.compile` or attention flavor swaps.

The new themes for this round:

1. **Capacity scale-up on bf16** (alphonse PR #393, in flight) — does the half-step wider+deeper model finally win now that throughput is unblocked?
2. **Cosine T_max alignment** (fern PR #407, just-assigned) — at the bf16 baseline, the LR is still 78% of peak at the best epoch (16) because T_max=50 doesn't decay in 19 reachable epochs. `--epochs 20` is the single-flag fix.
3. **Round-1 hypotheses transfer to bf16.** The pre-bf16 round-1 cohort (#314, #321, #324, #327, #333) was assigned against val_avg=144.21. Their gains may not transfer to the new 121.85 baseline. askeladd's PR #313 already came back +13% vs new baseline (after being −4% vs old) — sent back to rebase + re-run on bf16.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-weighted MSE (5x p) — **sent back** to rebase + re-run on bf16 |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | 5-epoch warmup + cosine, peak=7e-4 (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #327 | tanjiro | Spatial inductive bias | Fourier features for (x, z), K=8 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #393 | alphonse | Capacity (on bf16) | Half-step scale-up h=160/L=5/heads=5/slices=80 |
| **#407** | **fern** | **Schedule (on bf16)** | **Cosine T_max alignment via `--epochs 20`** |

## Reviewed (round 1)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded by #359 | Initial baseline: val_avg=144.21. Cherry-picked `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. Now retried as #393 on bf16. |
| #321 | frieren | Sent back | +2.9%. Variation: peak=7e-4 (in flight). |
| #360 | fern | Closed | +3.12%. bsz=8 alone doesn't help — trainer not launch-bound. |
| #359 | alphonse | **Merged (NEW BASELINE)** | bf16 autocast: val_avg=121.85 (−15.5%), test_avg=111.15. |
| #313 | askeladd | Sent back | −4.2% vs old baseline but +13.4% vs new (pre-bf16 run). Rebase + re-run on bf16. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified by allocator fragmentation + pipeline mismatch. |

## Note on round-1 baseline shift

The round-1 cohort (#314, #321 sent back, #324, #327, #333) was assigned against the pre-bf16 baseline (144.21). When their results land, comparison is now against **121.85** — a much harder target. **Expected:** more of the cohort will need rebase + re-run on bf16, similar to askeladd's #313. Not closing them preemptively because a 4% intervention on the pre-bf16 baseline (like askeladd's) could still stack to a real win on bf16.

## Potential next directions

- **`torch.compile` pilot** in dynamic-shapes mode. Risky but the only
  remaining throughput lever once sampler tricks have been ruled out.
- **Stack winners.** If alphonse #393 (capacity) and any of the round-1
  loss-alignment ideas (#313, #314, #333) win on bf16, combine them into
  a single round-2 PR.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` loss accumulator —
  flagged 3+ times now. Can be folded into any next non-throughput PR.
- **Ideas warm for round 2** if simple levers plateau:
  - Test-time augmentation: mirror-flip x for cruise foils.
  - Per-Re weighting in the sampler (high-Re drives the metric tail).
  - Surface-only auxiliary head trained on surface nodes only.
  - Mesh-aware encoders: kNN/GAT/PointNet local message passing before
    slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf| as extra input channels.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer,
    multi-scale slice transformer, hierarchical FNO.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) **still binding** at
  bf16 baseline (19/50 epochs). Cosine T_max alignment may release 1-3%
  more from the fully-decayed schedule tail.
- `data/scoring.py` patched (b78f404) — `test_avg/mae_surf_p` is finite.
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
