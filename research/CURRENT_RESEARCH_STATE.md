# SENPAI Research State

- **Date:** 2026-04-27 23:30
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **101.87** | PR #293 (edward, L1 loss) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 102.61 | PR #293 |

L1 loss in normalized space replaced MSE; everything else at the original `train.py` defaults (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `lr=5e-4`, `surf_weight=10.0`, `batch_size=4`, plain CosineAnnealingLR, `epochs=50`).

## Current research focus

L1 loss won round 1 by a wide margin, validating the "align loss with eval metric" thesis. The remaining round-1 hypotheses (architecture and optimizer changes) are still in flight on the unmodified-baseline branch — they'll be evaluated against the new L1 baseline as they land. Round 2 starts stacking ideas on top of L1.

The 30-min `SENPAI_TIMEOUT_MINUTES` cap is the binding constraint: only ~14 epochs of the configured 50 are reached, and the cosine schedule was designed for 50. Cheap-per-epoch changes (loss formulation, feature augmentation, regularization, schedule fixes) outperform capacity-heavy changes (deeper, wider, more attention) in this regime.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR this round. Edward's diagnosis (PR #293): GT in test sample 20 has 761 non-finite values in the `p` channel; `data/scoring.accumulate_batch` computes `(pred - y).abs()` before masking so the NaN propagates into the per-channel sum even though the surrounding code reads as a sample-skip. `data/scoring.py` is read-only per program constraints. Until upstream-fixed, rank PRs by the **3-clean-split test mean** alongside `val_avg/mae_surf_p`.

## Open PRs

### Round 1 still in flight (status:wip, on the pre-L1 baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #278 | Loss | alphonse | Per-channel pressure-up-weighting (`surf_p_weight=5`) inside the surface loss |
| #290 | Architecture | askeladd | Wider model: `n_hidden` 128 → 192, `slice_num` 64 → 96 |
| #299 | Architecture | frieren | Deeper model: `n_layers` 5 → 8 |
| #301 | Loss | nezuko | `surf_weight` 10 → 30 |
| #303 | Optimization | tanjiro | EMA weights (decay 0.999) |

When these land, students will need to rebase onto the now-L1 baseline so we measure the change net of L1.

### Round 1 send-back (post-L1)

| PR | Axis | Student | Hypothesis | Reason |
|----|------|---------|------------|--------|
| #296 | Optimization | fern | Linear warmup → cosine, peak `lr` 1e-3, **`--epochs 14`** budget-matched | Original schedule decayed over 45 epochs while only 14 reachable |

### Round 2 (status:wip, on top of L1)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber (smooth_l1, beta=1.0) — quadratic-near-zero on top of L1 |
| #365 | Feature | thorfinn | Fourier positional features (8 freqs on normalized `x, z`) |

### Round 1 closed

| PR | Student | Reason |
|----|---------|--------|
| #305 | thorfinn | `n_head=8, slice_num=128, dim_head=16` — 2× per-epoch cost (~252 s) made only 8/50 epochs reachable, plus dim_head=16 produced non-finite test predictions on cruise. Reassigned to Fourier features (#365). |

## Potential next research directions

Once round-1 results land and we can stack:
- **Stack winners.** Best round-1 winners × L1 × {Huber? Fourier?} compose orthogonally.
- **Per-channel volume weighting** — currently the volume loss term treats Ux/Uy/p equally; pressure dominates the eval metric and the volume term may be diluting the gradient signal.
- **Mesh-aware augmentation.** Random node subsampling during training (the model is permutation-invariant by design, so this is "free" regularization).
- **Domain conditioning.** Explicit token / FiLM on (raceCar single | raceCar tandem | cruise tandem).
- **Gradient clipping.** Cheap robustness against the same kind of test-time NaN that hit thorfinn's narrow-head config — also a hedge against future capacity bumps.
- **Output residual from a free-stream estimate** for `Ux, Uy` — reduces the dynamic range the model has to learn from scratch, particularly at high Re.
- **Best-val checkpoint averaging.** Average the top-3 best-val checkpoints rather than picking one. Cheap ensemble-without-extra-training.
- **Extended-budget schedule.** When round-1 schedule fix (fern) converges, generalize: set `cosine T_max = epochs` with `epochs = 14` (budget-matched) as the default and have all PRs rebase to this.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
