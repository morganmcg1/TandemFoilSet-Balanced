# SENPAI Research State

- **Date:** 2026-05-15 18:00 UTC (Round 2 → Round 3 transition on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

Round 2 partial results on TandemFoilSet — Transolver CFD surrogate for tandem-airfoil flow fields. Primary ranking metric: `val_avg/mae_surf_p`.

**Current best baseline (PR #3290 merged):** `val_avg/mae_surf_p = 101.519`, `test_avg/mae_surf_p = 98.735` (3 finite splits, bf16 AMP).

**Dominant operational insight (confirmed Round 2):** Every run is wall-clock-truncated at ~14 epochs fp32 / ~19 epochs bf16 (out of configured 50). bf16 is a free 1.345× throughput multiplier — it is now the mandatory default for all new experiments. New baseline exploits 5 extra epochs (cosine decays deeper into anneal).

**Two key research questions for Round 3:**
1. **LR peak on bf16**: With 19 epochs available and cosine still near peak (80% of lr=5e-4 at ep14, only slightly better at ep19), does raising lr_peak to 1e-3 unlock significant improvement?
2. **Batch size on bf16**: 32.9 GB VRAM (vs 42.1 GB fp32) leaves room for bs=6/8. Does bigger batch reduce gradient noise enough to matter in 19 epochs?

## Round 1 results summary

| Hypothesis | PR | Result |
|-----------|-----|--------|
| Huber loss | #3094 | **MERGED** −15.7% (val 132.3 → 111.5) |
| OneCycleLR | #3131 | Closed: schedule sized for 50 ep, only 14 ran; +11.1% |
| surf_weight sweep | #3108 | Closed: uniform +11-13% regression vs baseline |
| Scale-aware loss | #3128 | Closed: directional misalignment with unweighted-MAE eval |

## Round 2 status snapshot (2026-05-15 18:00)

- **Merged (1):** #3290 bf16 AMP (askeladd) → new baseline 101.519 (−8.98% vs Huber)
- **Closed (1):** #3278 channel weighting (alphonse) — static p-upweighting regresses single_in_dist/rc; root cause is 10× y_std variance across splits, not channel imbalance
- **Sent back (2):**
  - #3117 Fourier features (fern, 2nd send-back) — scale=10 too high; try scale=2,4 + concat raw+Fourier + bf16
  - #3122 FiLM conditioning (frieren, updated) — rebase onto Huber + add bf16
- **Round 2 WIP still pending (4):**
  - #3321 tanjiro — LR=1e-3+warmup (fp32 baseline; intra-PR delta still informative)
  - #3289 thorfinn — cosine T_max=15 (fp32 baseline)
  - #3126 nezuko — EMA weights (fp32 baseline)
  - #3113 edward — slice_num=96 (fp32 baseline)

## Round 3 assignments (2026-05-15 18:00)

| Student | PR | Hypothesis | Axis | Baseline |
|---------|----|-----------|-|---------|
| alphonse | #3364 | LR=1e-3 + 3-ep warmup on bf16 | LR peak × bf16 | 101.519 |
| askeladd | #3365 | batch_size=6/8 on bf16 | Throughput × batch | 101.519 |

**All 8 students productively occupied — zero idle GPUs.**

### LR/schedule convergence (3 experiments covering the same axis space)

- **#3321 tanjiro** (fp32): lr_peak=1e-3 + warmup — intra-PR delta signal
- **#3289 thorfinn** (fp32): cosine T_max=15 — schedule decay signal
- **#3364 alphonse** (bf16): lr_peak=1e-3 + warmup — directly on current baseline

If alphonse wins (#3364), it's immediately mergeable. If tanjiro/thorfinn win, we compose with bf16 in Round 4.

## Potential next research directions (Round 3+ queue)

1. **Compose Round 3 winners** — bf16 + lr_peak=1e-3 + bs=8 + cosine_T_max_fit. Each orthogonal axis contributes an independent multiplier.
2. **Schedule-Free AdamW** — eliminates LR schedule entirely. With 19 bf16 epochs, schedule choice is less critical, but Schedule-Free removes the need for T_max tuning entirely. High risk, potentially high payoff.
3. **Model capacity on bf16** — n_hidden=192 or n_layers=6 with bf16 might now get ~12-15 epochs. Round 1's capacity failure was "only 7 epochs"; 19 bf16 epochs may change the picture.
4. **Fourier features lower scale** — fern's in-flight send-back. scale=2 or 4 on normalised coordinates; concat raw+Fourier.
5. **FiLM conditioning on bf16** — frieren's in-flight rebase. If intra-PR delta was −1.55% on MSE baseline, the Huber+bf16 baseline could amplify this.
6. **EMA weights on bf16** — nezuko's in-flight test. Evaluation with EMA should benefit from more epochs (19 vs 14).
7. **Per-domain y normalization** — correct fix for the 10× y_std heterogeneity that closed channel weighting. Normalize loss by per-domain target std. Requires training domain labels.
8. **SDF input features** — signed distance to foil surface as extra input channel. Direct geometry signal for surface pressure prediction. Implementation cost: KD-tree on surface nodes, precomputed.
9. **Gradient accumulation** — effective bs=16 without VRAM cost. Alternative to actual bs=8 if OOM.
10. **Sobolev loss** — penalize prediction-gradient errors near the surface. Pairs with surface MAE focus.

## Operational notes

- **New baseline: val_avg/mae_surf_p = 101.519** (bf16 + Huber, PR #3290 merged 2026-05-15 17:40)
- **All new experiments must include `--amp_dtype bf16`** — bf16 is default from here
- **All in-flight fp32 experiments (#3321, #3289, #3126, #3113, #3122) still informative via intra-PR delta**
- 30-min wall-clock cap × 50-epoch cap; effective ~19 epochs with bf16
- Local JSONL metrics only (no remote experiment tracking)
- Known scoring bug: `test_geom_camber_cruise/mae_surf_p` NaN (sample 20 inf in ground-truth p); `data/scoring.py` read-only. Affects all arms identically.
- fern's `train.py` NaN-filter fix (merged via #3290's update) rescues test_avg from nan; `test_avg/mae_surf_p` now computable as 3-split average manually.
