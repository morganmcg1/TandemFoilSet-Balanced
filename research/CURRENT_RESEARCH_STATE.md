# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 15:37 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force: only this advisor branch and PRs from the 8 assigned students are in scope.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits (single-foil sanity, unseen front-camber raceCar, unseen front-camber cruise, Re holdout). Baseline is a stock Transolver with physics-aware slice attention. Per-run budget is 30 minutes wall clock × 50 epochs hard cap.

## Current focus

This is round 1 (R1) of the willow-pai2i-24h-r4 track. **First credible baseline measurement landed via edward's paired vanilla run `17fia1vd`: `val_avg/mae_surf_p = 128.34`, 3-split `test_avg/mae_surf_p = 127.29`** (4-split test_avg is NaN due to the systemic eval bug). `BASELINE.md` will be populated when the first rerun-fix PR merges with finite 4-split test_avg. The R1 portfolio covers six orthogonal axes:

| # | Student | Hypothesis | Axis | PR | Status |
|---|---------|-----------|------|-----|--------|
| 1 | tanjiro  | Huber loss (delta=0.5) instead of MSE | Loss-metric alignment | #3256 | WIP |
| 2 | frieren  | Surface MAE + per-channel p-weight 3× | Loss-metric alignment | #3257 | **sent back** (val_avg=99.91 promising, test_avg=NaN; NaN-guard rerun in flight) |
| 3 | fern     | Grad-clip 1.0 + 5-epoch LR warmup     | Training stability     | #3258 | WIP |
| 4 | nezuko   | Surface-biased slice routing (`is_surface` into slice projection) | Architectural — attention   | #3260 | WIP |
| 5 | alphonse | Wider-shallower (256d, 3 layers, 8 heads) | Architectural — capacity | #3261 | WIP |
| 6 | edward   | Random Fourier Features σ=1.0 on (x, z) | Input/feature encoding | #3262 | **sent back** (val_avg=115.78 = −9.8% win; test_avg=NaN; same NaN-guard rerun, σ=1.0 only) |
| 7 | thorfinn | FiLM log(Re) conditioning on hidden state | Physics conditioning | #3263 | WIP |
| 8 | askeladd | Dropout p=0.1 in MLP and attention    | Regularization (OOD)   | #3264 | WIP |

## Standings so far (val_avg/mae_surf_p, lower is better)

| Rank | PR | Hypothesis | val_avg | vs baseline 128.34 | 4-split test_avg | Status |
|------|----|------------|--------:|-------------------:|-----------------:|--------|
| 1 | #3257 (frieren) | Surf-MAE + p-weight 3× | 99.91 | **−22.2%** | NaN (pending rerun) | sent back |
| 2 | #3262 (edward) | RFF σ=1.0 | 115.78 | **−9.8%** | NaN (pending rerun) | sent back |
| — | baseline (`17fia1vd`) | vanilla Transolver | 128.34 | — | NaN (3-split 127.29) | reference |

Two promising results already on the board, both blocked on the same `data/scoring.py` NaN-poisoning bug. Once the eval-only `torch.nan_to_num` patch is applied and both reruns return finite 4-split test_avg, they should merge in this order (frieren first, then edward layered on top — different code paths, likely orthogonal).

## Predicted top performers (updated advisor read)

- **Tier A (confirmed strong val wins, awaiting clean test):** frieren (−22.2% val), edward (−9.8% val).
- **Tier B (high-variance bets, results pending):** thorfinn (FiLM Re; should move `val_re_rand` most), nezuko (surf-biased slice; expected to move `val_geom_camber_*`).
- **Tier C (safer but smaller deltas, pending):** tanjiro (Huber), fern (clip+warmup), askeladd (dropout), alphonse (wider-shallower).

Splits to watch:
- `val_geom_camber_rc` / `val_geom_camber_cruise` for OOD geometry generalization — overall the hardest tracks; dropout, FiLM, and RFF (confirmed) all play here.
- `val_re_rand` for cross-regime Re generalization — FiLM should help most.
- `val_single_in_dist` for in-distribution capacity — wider-shallower and Fourier features (confirmed by edward) most likely.

## Plateau-protocol queue (R2+ candidates, ranked)

If R1 plateaus or only narrow wins land, the next-round backlog (in rough priority order):

1. **Re-stratified loss reweighting** — weight each sample by `1 / per_sample_y_std` so low-Re samples get a fair gradient share. Direct fix for the dynamic-range problem.
2. **Multi-scale slice tokens** — give Transolver two groups of slices (coarse global, fine surface-focused) to exploit the natural length scales (foil chord, gap).
3. **Augment with geometry-aware features** — node distance to nearest surface as an additional input, computed at preprocessing time from already-loaded `is_surface`.
4. **Loss decomposition by domain** — track and balance per-domain (single, raceCar tandem, cruise tandem) loss explicitly rather than relying solely on the balanced sampler.
5. **EMA of weights** — exponential moving average of model parameters for evaluation, common in CFD surrogate papers, often gives a "free" few-percent gain.
6. **Attention on the relative geometry within each foil** — leverage that overset zones are partially known (Zone 0 background vs. Zone 1/2 foil-local).
7. **AdamW betas / weight-decay sweep** — cheap finalization once we have a competitive architecture.
8. **Test-time augmentation via mesh perturbation** — small geometric jitter at inference for OOD robustness.

## Open issues spotted

- **Eval NaN-poisoning** (surfaced via #3257, confirmed in #3262 baseline). `data/scoring.py:accumulate_batch` only filters non-finite ground truth, not non-finite predictions. A single bad predicted value contaminates the entire split sum. Read-only file, so the fix has to live in `train.py:evaluate_split` via `torch.nan_to_num`. Sent #3257 and #3262 back with the same patch. Confirmed bug is present in vanilla baseline too — affects `test_geom_camber_cruise/mae_surf_p` consistently. Reuse the same fix for any other PR that lands with NaN test metric.
- **Cosine LR schedule mismatch.** `T_max=cfg.epochs=50` but every run is timeout-capped at ~14 epochs (30-min wall clock with batch_size=4 on full meshes). The model never sees the low-LR tail of the schedule — implicit under-annealing for everyone in R1. Candidate R2 hypothesis: set `T_max` to an estimated *real* epoch count (e.g., 12–14) rather than the 50 nominal max. Cheap, applies to every run.
- **GitHub API rate limit exhaustion (intermittent).** Combined advisor + 8-student polling can exceed the shared 5000 req/hr quota for the `morganmcg1` token. Observed twice on 2026-05-15 (~14:48–15:18 UTC, ~15:30–16:19 UTC). Recovery is automatic on reset, but it delays advisor send-back label swaps. Not blocking research; flagged for infra awareness.

## Next checkpoint

When students complete their R1 runs and post terminal `SENPAI-RESULT` markers, the priority order for review is:
1. Rank by `val_avg/mae_surf_p` (best validation checkpoint) — only PRs with finite `test_avg/mae_surf_p` are merge-eligible.
2. Verify each result against the W&B run (don't trust the PR comment alone).
3. Merge winners in order, starting with the best. Each merged win updates the baseline that R2 hypotheses must beat.
4. Send promising-but-not-winning PRs back with a specific next variation under the same `wandb_group`.
5. Close clear dead ends (>5% regression or fundamentally broken) and immediately re-assign that student a fresh hypothesis from the R2 queue.
6. For any PR that lands with a NaN test metric, paste the same `torch.nan_to_num` evaluate_split patch as #3257 and ask for a single rerun.
