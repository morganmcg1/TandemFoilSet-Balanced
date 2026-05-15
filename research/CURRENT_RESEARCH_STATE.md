# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-15 18:35 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force: only this advisor branch and PRs from the 8 assigned students are in scope.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits (single-foil sanity, unseen front-camber raceCar, unseen front-camber cruise, Re holdout). Baseline is a stock Transolver with physics-aware slice attention. Per-run budget is 30 minutes wall clock × 50 epochs hard cap.

## Current focus

**R1 has a first winner. PR #3257 (frieren — surface MAE + p-weight 3× + canonical NaN guard) merged at 18:25 UTC.** New BASELINE.md target:

- `val_avg/mae_surf_p = 106.67` (was 117.89, −9.5%)
- `test_avg/mae_surf_p = **94.35**` (was 106.23, −11.2%) — W&B `szru1ogx`

All remaining R1 PRs must now beat **test_avg=94.35**. Two were already in flight with the corrected NaN patch (#3258 fern, #3262 edward) — both have been sent back to rebase onto the new baseline (frieren's loss change introduced a merge conflict and the result needs re-measuring on the new base). Mechanisms (grad-clip+warmup, RFF) are orthogonal to loss reformulation, so should compose.

## R1 portfolio status

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| 1 | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test_avg=94.35 |
| 2 | tanjiro  | Huber loss (delta=0.5) | #3256 | WIP (just unblocked from rate limit, run started 18:28 UTC) |
| 3 | fern     | Grad-clip 1.0 + 5-epoch warmup | #3258 | **sent back for rebase** (rerun on old base returned test_avg=105.70; rebase onto frieren-base in flight) |
| 4 | nezuko   | Surface-biased slice routing | #3260 | WIP (rate-limit-recovered, no commits since assign — pod active) |
| 5 | alphonse | Cosine T_max fix (T_max=14, was 50) | #3358 | WIP |
| 6 | edward   | RFF σ=1.0 on (x, z) | #3262 | **sent back for rebase** (corrected-patch rerun never landed before merge; rebase + rerun in flight) |
| 7 | thorfinn | FiLM log(Re) conditioning | #3263 | WIP |
| 8 | askeladd | EMA weights β=0.999 | #3351 | WIP |
| — | frieren  | **(idle, awaiting R2 assignment)** | — | — |

## Standings — current best per merge target (test_avg/mae_surf_p, lower is better)

| Rank | PR | Hypothesis | val_avg | test_avg | vs old / new baseline | Status |
|------|----|------------|--------:|--------:|---------------------:|--------|
| **1** | **#3257 (frieren)** | **Surf-MAE + p-weight 3×** | **106.67** | **94.35** | **−9.5% / −11.2%** | **MERGED — new baseline** |
| 2 | #3258 (fern) old-base rerun | clip1.0+wu5 | 117.31 | 105.70 | −0.5% / +12.0% | rebase in flight |
| — | baseline (`xfayvdk2`, alphonse) | vanilla + NaN guard | 117.89 | 106.23 | — | pre-R1 anchor |
| ✗ | #3261 (alphonse, closed) | Wider-shallower 256d | 146.26 | 133.34 | +24.1% / +41% | CLOSED |
| ✗ | #3264 (askeladd, closed) | Dropout 0.1 | 140.57 | NaN | +19.3% / — | CLOSED |

## Predicted top performers for R2-on-frieren-base

- **Tier A (orthogonal mechanism, high expected gain after compounding with #3257):**
  - **fern's clip+warmup #3258** — gradient norms are a function of model+data, not loss. Bound clipping at median 56 / peak 1004 still binds 100% of steps on the new loss. Predicted: val ~95, test ~85.
  - **alphonse's cosine T_max fix #3358** — schedule mismatch is universal; near-zero LR tail should help every config. Predicted gain 2–5%, free.
  - **edward's RFF σ=1.0 #3262** — feature-encoding orthogonal to loss. Old gain −9.8% val, expect ~5% on new base.
- **Tier B (high-variance bets, smaller predicted gain):**
  - **thorfinn's FiLM #3263** — Re conditioning should move re_rand most (cross-regime). New loss already does well on cruise/re_rand so headroom may be smaller.
  - **nezuko's surf-biased slice #3260** — slice routing × p-weighted loss may overlap; result will tell us.
- **Tier C (uncertain / orthogonal pure regularization):**
  - **askeladd's EMA #3351** — usually 2–4%, free. Composes with everything.
  - **tanjiro's Huber #3256** — direct competitor to frieren's MAE. Mechanism-wise similar; expected to lose against #3257 since MAE-with-p-weight already captured what Huber would.

## Open issues / live diagnostics

- **Eval NaN-poisoning — FIXED in #3257 merge.** Canonical patch (`y_finite_per_sample`, `nan_to_num(y)`, `n_skipped_y_samples`) is now in `train.py:evaluate_split` on the advisor branch. Every PR built on top automatically inherits it. Future students don't need to re-derive.
- **Huge Transolver gradient norms still unaddressed at root.** Fern's #3258 traced median 56, peak 1110. Frieren's MAE loss may have partially tamed them (MAE has bounded per-sample gradient unlike MSE) — the rebased rerun will show this in the grad-norm trace. If still ~50, clip+warmup will still be a big win. If now ~10, the loss change captured most of it. **R2 hypothesis (saved):** soft slice-softmax temperature (current 0.5 is sharp).
- **Cosine LR schedule mismatch still unaddressed (in flight via #3358).** `T_max=50` but only ~14 epochs trained. Free win likely.
- **Run-to-run baseline variance.** Vanilla unclipped runs span ~13pt on val_avg (128 to 142). Once clip+warmup lands, this should drop substantially. Until then, R1 deltas have ±5–10% uncertainty.
- **GitHub API rate limit exhaustion (intermittent).** Combined advisor + 8-student polling can exceed the 5000 req/hr shared quota. Observed 14:48-15:18 UTC, 15:30-16:19 UTC, **and 17:56-18:21 UTC** (this last window cost tanjiro and nezuko ~25 min of progress). Auto-recovers; not blocking but persistent. Flagged for infra.

## Plateau-protocol queue (R2+ candidates, ranked)

1. **Re-stratified loss reweighting** — weight each sample by `1 / per_sample_y_std` so low-Re samples get a fair gradient share. Direct fix for the dynamic-range problem.
2. **Multi-scale slice tokens** — give Transolver two groups of slices (coarse global, fine surface-focused) to exploit the natural length scales (foil chord, gap).
3. **Geometry-aware features** — node distance to nearest surface as an additional input. Cheap, mechanistically motivated.
4. **Loss decomposition by domain** — track and balance per-domain (single, raceCar tandem, cruise tandem) loss explicitly rather than relying on balanced sampler.
5. **EMA of weights** — already assigned to askeladd as #3351. Common in CFD surrogate papers.
6. **Soft slice-softmax temperature** — current temperature 0.5 makes attention sharp and may be the source of huge gradient norms. Diagnostic-first hypothesis.
7. **AdamW betas / weight-decay sweep** — cheap finalization once we have a competitive architecture.
8. **Test-time augmentation via mesh perturbation** — small geometric jitter at inference for OOD robustness.
9. **Re-evaluate surf_weight=10** — currently surfaces are weighted 10× in the loss. With p-weight=3 on the surface MAE, this is effectively 30× on surface p — may be over-weighted now. Sweep {5, 10, 20}.

## Next checkpoint

When remaining R1 students complete:

1. Rank PRs by `test_avg/mae_surf_p` vs new baseline 94.35.
2. Merge winners sequentially, best first. After each merge, update BASELINE.md and rebase remaining PRs onto new head.
3. Send promising-but-not-winning PRs back with a specific variation on the same `wandb_group`.
4. Close clear dead ends (>5% regression) and immediately reassign that student from the R2 queue or researcher-agent fresh ideas.
5. Frieren is idle now — needs R2 assignment from the queue or new researcher-agent ideas.
