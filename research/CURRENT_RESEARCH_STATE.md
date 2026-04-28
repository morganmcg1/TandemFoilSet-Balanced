# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

The round baseline is now **PR #327 (tanjiro, sinusoidal Fourier features K=8 on (x, z), on top of bf16): val_avg/mae_surf_p = 106.92, test_avg = 96.82** — a cumulative −25.9% from the original PR #312 reference (144.21).

Two themes are now in play:

1. **Spatial frequency representation** — FF was the biggest single win of the round. Tanjiro is now testing **Gaussian random Fourier features (RFF)** as the followup; Tancik et al. show it usually outperforms the deterministic ladder.
2. **Throughput / schedule** — fern is on cosine T_max alignment (#407), alphonse on `torch.compile` pilot (#416). Both should compose with FF if they win.

The original round-1 cohort (#314, #324, #333, plus sent-back #313 / #321) is now competing against an **even harder** baseline (106.92 vs the 144.21 they were assigned against). Many will need rebases when results land.

Key per-split signal from #327: **the held-out rc-camber split (M=6-8) gained only −3.3% from FF**, while cruise + single-in-dist gained 17–20%. OOD geometry generalisation is bottlenecked more by camber→pressure mapping than by spatial-frequency representation. Targeted hint for round 2.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-weighted MSE (5x p) — sent back to rebase + re-run on bf16 |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | warmup + cosine peak=7e-4 (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #407 | fern | Schedule (on bf16+FF) | Cosine T_max alignment via `--epochs 20` |
| #416 | alphonse | Throughput (on bf16+FF) | `torch.compile(dynamic=True)` pilot |
| **#443** | **tanjiro** | **Spatial features (on bf16+FF)** | **Gaussian random Fourier features K=16 σ=10 (replacing deterministic K=8)** |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded twice | Initial baseline: val_avg=144.21. + `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. Variation: peak=7e-4 (in flight). |
| #360 | fern | Closed | +3.12%. bsz=8 alone didn't help — trainer not launch-bound. |
| #359 | alphonse | Merged → superseded by #327 | bf16 autocast: val_avg=121.85 (−15.5%). |
| #313 | askeladd | Sent back | −4.2% vs old baseline but +13.4% vs new (pre-bf16 run). Rebase + re-run on bf16. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max=50 / 14-epoch mismatch. Parked. |
| **#327** | **tanjiro** | **Merged (NEW BASELINE)** | **FF K=8: val_avg=106.92 (−12.2%), test_avg=96.82 (−12.9%). Largest single win.** |

## Throughput levers status

- bf16 autocast: **MERGED**
- Sinusoidal Fourier features (x,z) K=8: **MERGED** (largest single win)
- Larger batch size: **RULED OUT** (HBM-bound, padding scales with B)
- Domain-bucketed sampler: **RULED OUT** (allocator fragmentation, pipeline mismatch)
- Cosine T_max alignment: **IN FLIGHT** (#407, fern)
- `torch.compile` pilot: **IN FLIGHT** (#416, alphonse)
- Half-step capacity: parked, retest after #407

## Potential next directions

- **Stack winners.** When #407 / #416 / #443 land, compose with FF baseline.
- **Targeted OOD-camber experiment.** rc-camber held-out split lagged FF
  by an order of magnitude. Investigate camber→pressure mapping
  specifically. Candidate: NACA-camber-aware feature embedding (one-hot or
  learned camber bins), or per-camber stratified loss reweighting.
- **FF on saf/dsdf** (tanjiro followup #4) — extend FF idea to the
  distance-based shape descriptor and signed arc-length features.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` — flagged 4+
  times now. Could fold into any next non-throughput PR.
- **Round-1 cohort transfer status:** the loss / regularization /
  schedule PRs (#314, #324, #333, sent-back #313 / #321) were assigned vs
  144.21. They now have to beat 106.92. Sent-back ones must rebase onto
  the bf16+FF advisor branch. Expect more "send back to rebase" outcomes.
- **Round 2 ideas (kept warm):**
  - Test-time augmentation: mirror-flip x for cruise foils.
  - Per-Re weighting in the sampler.
  - Surface-only auxiliary head.
  - Mesh-aware encoders (kNN/GAT/PointNet) before slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf|.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) **still binding** at
  bf16+FF baseline (19/50 epochs). Cosine T_max alignment may release more
  from the schedule tail.
- `data/scoring.py` patched (b78f404).
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
