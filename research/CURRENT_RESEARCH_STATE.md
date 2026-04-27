# SENPAI Research State

- **Date:** 2026-04-27 23:40
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **101.87** | PR #293 (edward, L1 loss) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 102.61 | PR #293 |

L1 loss in normalized space replaced MSE. Everything else at the original `train.py` defaults (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `lr=5e-4`, `surf_weight=10.0`, `batch_size=4`, plain CosineAnnealingLR, `epochs=50`).

## Current research focus

L1 loss won round 1 by a wide margin (val_avg ~30% lower than the next-best round-1 PR), validating the "align loss with eval metric" thesis. **All MSE-era round-1 measurements are now stale** — the round-1 PRs that don't beat L1 directly are being rebased onto L1 if the hypothesis is plausibly orthogonal, or closed if the hypothesis is structurally penalized by the timeout regime.

The 30-min `SENPAI_TIMEOUT_MINUTES` cap is binding: only ~14 epochs of the configured 50 are reached. Per-epoch wall time is the dominant cost. Cheap-per-epoch changes (loss formulation, feature augmentation, light regularization, schedule fixes) outperform capacity-heavy changes (deeper, wider, more attention). All round-2+ assignments respect this constraint.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR this round. Independent diagnoses from edward (#293), alphonse (#278), nezuko (#301), askeladd (#290): test sample 20 has 761 non-finite values in the volume-cell `p` channel of GT; `data/scoring.accumulate_batch` computes `(pred_orig - y).abs()` before masking, so `NaN * 0 = NaN` propagates into the per-channel sum. `data/scoring.py` is read-only per program constraints. Until upstream-fixed, rank PRs by the **3-clean-split test mean** alongside `val_avg/mae_surf_p`.

## Open PRs

### Round 1 sent back to rebase onto L1 (status:wip)

| PR | Axis | Student | Hypothesis | Why send-back |
|----|------|---------|------------|---------------|
| #278 | Loss | alphonse | `surf_p_weight=5` (pressure-channel up-weight in surface loss) | Plausibly stacks with L1 |
| #296 | Optim | fern | Warmup → cosine, peak `lr=1e-3`, `--epochs 14` budget-matched | Schedule had to be matched to wall-clock budget |
| #301 | Loss | nezuko | `surf_weight=30` (surface emphasis) | Optimal weight may differ on L1 vs MSE |

### Round 1 still in flight (status:wip, not yet measured)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #299 | Architecture | frieren | Deeper model: `n_layers` 5 → 8 |
| #303 | Optim | tanjiro | EMA weights (decay 0.999) for eval and final checkpoint |

These are still on the pre-L1 baseline. When they land, they'll be evaluated against the L1 baseline and likely sent back to rebase if the hypothesis is plausibly additive.

### Round 2 (status:wip, on top of L1)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber (smooth_l1, beta=1.0) — quadratic-near-zero on top of L1 |
| #365 | Feature | thorfinn | Fourier positional features (8 freqs on normalized `x, z`) |
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |

### Round 1 closed

| PR | Student | Reason |
|----|---------|--------|
| #290 | askeladd | `n_hidden=192, slice_num=96` — per-epoch ~205s gave 9/50 epochs at 30-min cap; 33% worse than L1 at iso-wall-clock. Reassigned to drop-path (#369). |
| #305 | thorfinn | `n_head=8, slice_num=128, dim_head=16` — per-epoch ~252s gave 8/50 epochs; dim_head=16 produced non-finite test predictions. Reassigned to Fourier features (#365). |

## Round-1 ranking (val_avg/mae_surf_p, all measured on pre-L1 MSE baseline)

| Rank | PR | Student | Axis | val_avg | Verdict |
|---:|----|---------|------|---------:|---------|
| 1 | #293 | edward | L1 loss | **101.87** | Merged |
| 2 | #296 | fern | LR warmup 1e-3 | 137.32 | Sent back (schedule mismatched) |
| 3 | #301 | nezuko | surf_weight=30 | 141.56 | Sent back (rebase to L1) |
| 4 | #290 | askeladd | wider 192 | 152.24 | Closed (budget penalty) |
| 5 | #278 | alphonse | surf_p_weight=5 | 156.16 | Sent back (rebase to L1) |
| 6 | #305 | thorfinn | slices+heads 2x | 160.68 | Closed (budget + instability) |
| ? | #299 | frieren | n_layers=8 | TBD | Running |
| ? | #303 | tanjiro | EMA | TBD | Running |

## Potential next research directions

When the round-2 / sent-back PRs land:
- **Stack winners.** L1 × {Huber? drop-path? Fourier? pressure-weighting? surf_weight tweak? warmup schedule?} — the cheap regularizers / loss tweaks compose orthogonally.
- **Best-val checkpoint averaging.** Average top-K best-val checkpoints rather than picking one (poor man's SWA).
- **Mesh-aware augmentation.** Random node-loss subsampling during training (model is permutation-invariant by design).
- **Domain conditioning.** Explicit token / FiLM on (raceCar single | raceCar tandem | cruise tandem) — currently inferred only from input features.
- **Output residual from a free-stream estimate** for `Ux, Uy` — reduces the dynamic range to learn from scratch.
- **Per-channel volume weighting** — currently the volume term treats Ux/Uy/p equally; pressure dominates the eval metric.
- **Gradient clipping.** Cheap robustness against the kind of test-time NaN that hit thorfinn's narrow-head config.
- **Extended-budget schedule.** When fern's `--epochs 14` rerun lands, generalize: budget-matched schedule for all PRs.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
