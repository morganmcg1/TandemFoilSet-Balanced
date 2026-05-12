# SENPAI Research State

- **Date:** 2026-05-12
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Round 1 — in flight (first two reviews completed)

Broad sweep over orthogonal levers. First two PRs landed and were sent back (#1427, #1451) due to a `data/scoring.py` NaN-propagation bug that NaN-poisons `test_avg/mae_surf_p`. Both showed promising val signals (134.14 and 136.69 respectively at incomplete epoch budgets). All 8 students currently have WIP PRs. Each PR targets `icml-appendix-willow-pai2g-48h-r5`.

### Active issue: `data/scoring.py` NaN propagation

- `.test_geom_camber_cruise_gt/000020.pt` contains **761 `-inf` values** in y[:, 2] (volume p channel). The exact value `-65504.0 = -fp_max(bf16)` is the smoking gun — preprocessing overflowed in bf16 upstream. Diagnosed jointly by thorfinn (#1451) and alphonse (#1419).
- Scoring's intended sample-skipping is poisoned by IEEE `(+/-inf) * 0 = NaN` and then `NaN * 0 = NaN`.
- Workaround (in `train.py` only — `data/` is read-only): pre-mask non-finite samples + `nan_to_num(y, nan=0, posinf=0, neginf=0)` before `accumulate_batch`. Snippet broadcast to all WIP PRs; alphonse's rerun of #1419 will be the canonical merged version that propagates the fix.
- Val is unaffected; only the end-of-run test eval needs the fix.

### Round-1 leaderboard (preliminary, pending reruns with fix)

| PR | Student | Hypothesis | Val (best) | Test (clean) | Epochs in cap |
|----|---------|-----------|-----------:|-------------:|---:|
| #1419 | alphonse | bf16 autocast | **110.84** | **99.79** (offline) | 18 |
| #1427 | askeladd | surf_weight=30 | 134.14 | NaN; 130.65 (3-split) | 12 |
| #1451 | thorfinn | slice_num=128 (bs=2 OOM-fallback) | 136.69 | NaN; 132.59 (3-split) | 11 |

Alphonse leads decisively. The big driver is the bf16 speedup yielding 18 epochs vs 11-12. Other hypotheses haven't had time to fully reveal their effect yet.

| Student | PR | Hypothesis | Lever |
|---------|----|-----------|-------|
| alphonse | #1419 | bf16 autocast (mixed precision) | Optimisation / throughput |
| askeladd | #1427 | `surf_weight=30` (3×) | Loss weighting |
| edward | #1430 | `lr=1e-3` + 5% linear warmup + cosine | Optimisation |
| fern | #1436 | Smooth L1 (Huber) loss | Loss form |
| frieren | #1442 | Wider Transolver `n_hidden=192` | Architecture (capacity) |
| nezuko | #1445 | Per-channel surface weighting `(0.5, 0.5, 2.0)` | Loss / metric alignment |
| tanjiro | #1447 | `batch_size=8` (2×) | Gradient noise / throughput |
| thorfinn | #1451 | `slice_num=128` (2×) | Architecture (attention partitioning) |

## Round-2 candidate pool (from `research/RESEARCH_IDEAS_2026-05-12_round1.md`)

Round-1 already covers H2 (warmup ≈ edward #1430), H6 (per-channel p weighting ≈ nezuko #1445), and H7 (Huber ≈ fern #1436). Remaining researcher-agent suggestions, to seed round 2 once round-1 results land:

- **H1 — gradient clipping** (`clip_grad_norm_(model.parameters(), 1.0)`): low risk, plausible stabiliser given high-Re y magnitudes up to ±29K
- **H3 — `n_layers=8`**: matches original Transolver paper default; baseline is L=5
- **H4 — `slice_num=96`** (alternative finer-grain to thorfinn's 128)
- **H5 — `mlp_ratio=4`**: capacity increase, all standard transformers use 4×
- **H8 — dropout 0.1**: regularisation for the 3-of-4 OOD val splits, needs Transolver kwarg check first

## Broader follow-up directions (post-round-2)

- Compound winners of round 1 (loss + optimiser + architecture often orthogonal)
- Surface-aware decoder / dual-head architecture (separate volume and surface heads)
- Re-conditioning: explicit Re-aware embeddings, log-Re aware positional encoding
- Spectral / Fourier neural operator hybrids
- Per-domain auxiliary heads or domain adversarial features
- Curriculum on mesh size or Re regime
- Test-time augmentation (TTA) using physical symmetries
- EMA of model weights for eval

This is a living document — refresh as round 1 lands and round 2 planning starts.
