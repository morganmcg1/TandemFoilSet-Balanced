# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

bf16 (PR #359) remains the round baseline at **val_avg/mae_surf_p = 121.85**, **test_avg = 111.15**. Two follow-on hypotheses around this baseline are now in flight:

- **Cosine T_max alignment** (fern PR #407, `--epochs 20`) — likely reliable small win; sets the schedule shape for all future comparisons.
- **`torch.compile` pilot** (alphonse PR #416, `dynamic=True`) — high-variance but the highest remaining throughput lever now that bf16 is in and sampler tricks have been ruled out.

Capacity scale-up (PR #393) closed +7.55%, well-diagnosed as a T_max/schedule mismatch rather than capacity-doesn't-help. Will retest after fern's schedule alignment lands.

The original round-1 cohort (#314, #324, #327, #333) was assigned vs the pre-bf16 baseline (144.21). Their gains may not transfer to 121.85 — askeladd's #313 already needed rebase + re-run on bf16 for fair comparison.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-weighted MSE (5x p) — sent back to rebase + re-run on bf16 |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | 5-epoch warmup + cosine, peak=7e-4 (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #327 | tanjiro | Spatial inductive bias | Fourier features for (x, z), K=8 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #407 | fern | Schedule (on bf16) | Cosine T_max alignment via `--epochs 20` |
| **#416** | **alphonse** | **Throughput (on bf16)** | **`torch.compile(dynamic=True)` pilot** |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded by #359 | Initial baseline: val_avg=144.21. Cherry-picked `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. Variation: peak=7e-4 (in flight). |
| #360 | fern | Closed | +3.12%. bsz=8 alone didn't help — trainer not launch-bound. |
| #359 | alphonse | **Merged (NEW BASELINE)** | bf16 autocast: val_avg=121.85 (−15.5%), test_avg=111.15. |
| #313 | askeladd | Sent back | −4.2% vs old baseline but +13.4% vs new (pre-bf16 run). Rebase + re-run on bf16. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified by allocator fragmentation + pipeline mismatch. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max=50 / 14-epoch mismatch. Parked. |

## Throughput levers status

- bf16 autocast: **MERGED** (-26% wall, -22% peak GPU)
- larger batch size: **RULED OUT** (HBM-bound, padding scales with B)
- domain-bucketed sampler: **RULED OUT** (allocator fragmentation, pipeline mismatch)
- cosine T_max alignment: **IN FLIGHT** (PR #407, fern)
- `torch.compile` pilot: **IN FLIGHT** (PR #416, alphonse)
- gradient checkpointing: queued (only useful when scaling capacity)
- attention-flavor swaps: queued (round 2)

## Potential next directions

- **Stack winners.** Once fern #407 or alphonse #416 land, the next baseline
  is whichever wins (or both, if compatible). Then retest capacity (#393)
  and the round-1 loss-alignment cohort against it.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` — flagged 3+
  times. Worth folding into any next non-throughput PR.
- **Beyond round 1** ideas, kept warm for round 2:
  - Test-time augmentation: mirror-flip x for cruise foils.
  - Per-Re weighting in the sampler.
  - Surface-only auxiliary head on surface nodes only.
  - Mesh-aware encoders (kNN/GAT/PointNet) before slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf| as extra input channels.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) **still binding** at
  bf16 baseline (19/50 epochs). If `torch.compile` works it could push
  this to 25-30+ epochs.
- `data/scoring.py` patched (b78f404).
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known
  and not affecting MAE rankings.
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
