# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 16:30 UTC
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
| 3 | fern     | Grad-clip 1.0 + 5-epoch LR warmup     | Training stability     | #3258 | **sent back** (val_avg=115.73 = −18.5% win, test_avg=NaN; same NaN-guard rerun, clip1.0-wu5 only) |
| 4 | nezuko   | Surface-biased slice routing (`is_surface` into slice projection) | Architectural — attention   | #3260 | WIP |
| 5 | alphonse | Wider-shallower (256d, 3 layers, 8 heads) | Architectural — capacity | #3261 | WIP |
| 6 | edward   | Random Fourier Features σ=1.0 on (x, z) | Input/feature encoding | #3262 | **sent back** (val_avg=115.78 = −9.8% win; test_avg=NaN; same NaN-guard rerun, σ=1.0 only) |
| 7 | thorfinn | FiLM log(Re) conditioning on hidden state | Physics conditioning | #3263 | WIP |
| 8 | askeladd | Dropout p=0.1 in MLP and attention    | Regularization (OOD)   | #3264 | WIP |

## Standings so far (val_avg/mae_surf_p, lower is better)

| Rank | PR | Hypothesis | val_avg | vs ed baseline 128.34 | vs fern baseline 141.94 | 4-split test_avg | Status |
|------|----|------------|--------:|---------------------:|-----------------------:|-----------------:|--------|
| 1 | #3257 (frieren) | Surf-MAE + p-weight 3× | 99.91 | **−22.2%** | **−29.6%** | NaN (pending rerun) | sent back |
| 2 | #3258 (fern) | Grad-clip 1.0 + warmup 5 | 115.73 | **−9.8%** | **−18.5%** | NaN (pending rerun) | sent back |
| 3 | #3262 (edward) | RFF σ=1.0 | 115.78 | **−9.8%** | **−18.4%** | NaN (pending rerun) | sent back |
| — | baseline (`17fia1vd`, edward) | vanilla Transolver | 128.34 | — | — | NaN (3-split 127.29) | ref |
| — | baseline (`nylo2tvd`, fern) | vanilla Transolver | 141.94 | — | — | NaN (3-split 139.34) | ref |

Three promising results on the board, all blocked on the same `data/scoring.py` NaN-poisoning bug. **Important caveat:** the two measured baselines disagree by ~13pt (run-to-run variance with unclipped, large-gradient training), so improvement percentages above are uncertain. Fern's clip+warmup eliminates this variance source. Merge order plan once reruns return finite test_avg:
1. **fern first** (foundational training fix, applies universally)
2. **frieren second** (loss reformulation, orthogonal to training stability)
3. **edward third** (RFF on top of merged baseline; re-evaluate whether RFF still helps post-fern — #3258 and #3262 produce nearly identical val_avg, possibly redundant)

## Predicted top performers (updated advisor read)

- **Tier A (confirmed strong val wins, awaiting clean test):** frieren (−22.2% val vs ed baseline), fern (−18.5% vs own baseline; clearest mechanistic story via grad-norm trace), edward (−9.8% vs ed baseline).
- **Tier B (high-variance bets, results pending):** thorfinn (FiLM Re; should move `val_re_rand` most), nezuko (surf-biased slice; expected to move `val_geom_camber_*`).
- **Tier C (safer but smaller deltas, pending):** tanjiro (Huber), askeladd (dropout), alphonse (wider-shallower).

**Fern's gradient-norm finding is the most actionable single piece of mechanism insight from R1:** Transolver's pre-clip grad norms are median 56, peak 1100 — 50–1000× the natural clip cap. This explains the run-to-run instability we've been seeing in baselines. Every R1 hypothesis was likely partly noise-dominated by this; clipping should make subsequent rounds much more reliable.

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

- **Eval NaN-poisoning** (surfaced via #3257, confirmed across #3258, #3262, and both measured baselines). `data/scoring.py:accumulate_batch` only filters non-finite ground truth, not non-finite predictions. A single bad predicted value contaminates the entire split sum. Read-only file, so the fix has to live in `train.py:evaluate_split` via `torch.nan_to_num`. Sent #3257, #3258, #3262 back with the same patch. Affects `test_geom_camber_cruise/mae_surf_p` consistently across every run, including vanilla baselines. Reuse the same fix for any other PR that lands with NaN test metric.
- **Huge Transolver gradient norms** (uncovered by fern's #3258 trace). Pre-clip grad norms: median 56, mean 87, p99 445, max 1110 — 50–1000× the natural clip cap. Likely sources: PhysicsAttention slice-softmax temperature 0.5 (sharp distribution amplifies gradients through einsum) and/or `surf_weight=10` weighting against imbalanced surface/volume node counts. This is the root cause of large run-to-run baseline variance (~13pt on val_avg between unclipped runs). R2 candidate: diagnostic per-layer grad histogram + try softer slice softmax temperature.
- **Cosine LR schedule mismatch.** `T_max=cfg.epochs=50` but every run is timeout-capped at ~14 epochs (30-min wall clock with batch_size=4 on full meshes). The model never sees the low-LR tail of the schedule — implicit under-annealing for everyone in R1. Candidate R2 hypothesis: set `T_max` to an estimated *real* epoch count (e.g., 12–14) rather than the 50 nominal max. Cheap, applies to every run. Synergistic with fern's clip+warmup.
- **Run-to-run baseline variance is large (~13pt).** Same vanilla code: edward measured 128.34, fern measured 141.94. Implication: R1 improvement percentages have ±5–10% uncertainty. Mitigations: (1) once fern's clip+warmup merges, variance should drop substantially; (2) multi-seed runs would be ideal but expensive at 30 min/run; (3) bias-correct by averaging both baselines (~135) as a more honest reference.
- **GitHub API rate limit exhaustion (intermittent).** Combined advisor + 8-student polling can exceed the shared 5000 req/hr quota for the `morganmcg1` token. Observed twice on 2026-05-15 (~14:48–15:18 UTC, ~15:30–16:19 UTC). Recovery is automatic on reset. Students' polling scripts threw `JSONDecodeError` on rate-limit responses, which the entrypoint treated as "no work assigned" (5-min sleep). Not blocking research but a real cause of perceived stale_wip flags; flagged for infra awareness.

## Next checkpoint

When students complete their R1 runs and post terminal `SENPAI-RESULT` markers, the priority order for review is:
1. Rank by `val_avg/mae_surf_p` (best validation checkpoint) — only PRs with finite `test_avg/mae_surf_p` are merge-eligible.
2. Verify each result against the W&B run (don't trust the PR comment alone).
3. Merge winners in order, starting with the best. Each merged win updates the baseline that R2 hypotheses must beat.
4. Send promising-but-not-winning PRs back with a specific next variation under the same `wandb_group`.
5. Close clear dead ends (>5% regression or fundamentally broken) and immediately re-assign that student a fresh hypothesis from the R2 queue.
6. For any PR that lands with a NaN test metric, paste the same `torch.nan_to_num` evaluate_split patch as #3257 and ask for a single rerun.
