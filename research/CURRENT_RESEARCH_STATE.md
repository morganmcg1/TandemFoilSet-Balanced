# SENPAI Research State

- **Date:** 2026-05-12
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Round 1 — in flight (4 reviewed, 7 WIP + 1 early-round-2)

Broad sweep over orthogonal levers. Scoring NaN bug diagnosed and workaround broadcast to all PRs. Three PRs sent back for clean rerun (#1427, #1451, #1419). One PR closed as dead end (#1447).

### Scoring fix (broadcast to all PRs)

`.test_geom_camber_cruise_gt/000020.pt` has `-inf` values in y[:, 2] (`-65504 = -fp_max(bf16)`). Workaround in `evaluate_split` in `train.py` (data/ is read-only):
```python
sample_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
mask_eff = mask & sample_finite.unsqueeze(-1)
y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask_eff, mae_surf, mae_vol)
```
Val is unaffected. Alphonse's rerun (#1419) will be canonical merged version.

### Round-1 leaderboard

| PR | Student | Hypothesis | Val (best) | Test (clean) | Epochs | Status |
|----|---------|-----------|-----------:|-------------:|---:|---|
| #1419 | alphonse | bf16 autocast | **110.84** | **99.79** | 18 | WIP (rerunning with NaN fix) |
| #1427 | askeladd | surf_weight=30 | 134.14 | 130.65 (3-split) | 12 | WIP (rerunning with NaN fix) |
| #1451 | thorfinn | slice_num=128 (bs=2) | 136.69 | 132.59 (3-split) | 11 | WIP (rerunning with NaN fix) |
| #1447 | tanjiro | batch_size=8 | 154.74 | 138.92 | 14 | **Closed** (halved optimizer steps under wall-clock cap) |

Alphonse leads decisively — bf16 buys 18 epochs vs 11-14 for fp32 siblings. Under our 30-min wall-clock cap, per-epoch speed is the dominant convergence lever.

### In-flight WIP PRs (still training)

| Student | PR | Hypothesis | Lever |
|---------|----|-----------|-------|
| alphonse | #1419 | bf16 autocast | Throughput (18 epochs/30 min) |
| askeladd | #1427 | `surf_weight=30` (3×) | Loss weighting |
| edward | #1430 | `lr=1e-3` + 5% warmup + cosine | Optimisation |
| fern | #1436 | Smooth L1 (Huber) loss | Loss form |
| frieren | #1442 | Wider `n_hidden=192` | Architecture (capacity) |
| nezuko | #1445 | Per-channel surf weights `(0.5, 0.5, 2.0)` | Loss / metric alignment |
| tanjiro | #1534 | Gradient clipping `max_norm=1.0` | Gradient stability |
| thorfinn | #1451 | `slice_num=128` (2×) | Architecture (attention partitioning) |

## Round-2 candidate pool (from `research/RESEARCH_IDEAS_2026-05-12_round1.md`)

H1 (gradient clipping) already assigned to tanjiro (#1534). Remaining:

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
