# SENPAI Research State

- **Date:** 2026-04-28 02:40
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **74.44** | PR #387 (alphonse, L1 + warmup + Fourier + sw=30 + grad-clip-1.0) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 72.14 | PR #387 |

Per-split val on the new baseline: `val_single_in_dist=86.68`, `val_geom_camber_rc=85.92`, `val_geom_camber_cruise=53.29`, `val_re_rand=71.88`.

**Five orthogonal axes now stacked:** L1 loss (PR #293) × linear warmup → cosine, peak `lr=1e-3`, budget-matched `--epochs 14` (PR #296) × 8-band Fourier features on normalized (x, z) (PR #365) × `surf_weight=30` (PR #301) × **gradient clipping `max_norm=1.0`** (PR #387). All other knobs at originals.

Total improvement vs the original `train.py` baseline: estimated **>50% reduction** in `val_avg/mae_surf_p` (round-1 PRs on plain MSE were getting 130–160, now we're at 74.44).

## Current research focus

The full orthogonal stack is now established. Diminishing returns are starting to appear on each new axis (alphonse's grad-clip dropped from −13.5% on partial stack to −2.92% on full stack — much of the trajectory-smoothing work is already being absorbed by Fourier and warmup). Going forward, the priority is:

1. **Refine each axis at its current setting** — sweeps around the merged values (`grad_clip_norm` 0.5 / 0.25, `weight_decay` 1e-3, Huber `beta=0.5`).
2. **Stack the remaining cheap-per-epoch hypotheses** (EMA, ckpt-avg, drop-path, dsdf-Fourier).
3. **Identify when overlap dominates and a hypothesis becomes net-zero or net-negative** — the diminishing-returns regime is starting to bite.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR. Diagnosed independently 8+ times; `data/scoring.py` is read-only. Rank by 3-clean-split test mean.

## Open PRs

### Sent back to rebase onto current advisor (status:wip)

| PR | Axis | Student | Hypothesis | Status |
|----|------|---------|------------|--------|
| #303 | Weights | tanjiro | EMA weights (decay 0.999) | rebasing |
| #364 | Loss | edward | Huber smooth_l1 **beta=0.5** (refined from beta=1.0) | rebasing |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging top-3 + val-on-averaged | rebasing |
| #385 | Regularization | fern | `weight_decay` 1e-4 → 5e-4 | rebasing |
| #414 | Feature | thorfinn | Fourier on dsdf channels (4 freqs) | rebasing with iso-epoch concern |

### Round 2 carry-over (status:wip, on stale L1-only baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |

### Round 5 (status:wip, on top of full current stack)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #444 | Loss balance | nezuko | `surf_p_extra=3.0` — boost surface-p channel only |
| #464 | Stability | alphonse | Tighten grad clip 1.0 → 0.5 (motivated by gradient-norm telemetry) |

## Round-1+2+3+4 ranking (val_avg/mae_surf_p)

| Rank | PR | Student | Stack | val_avg | Verdict |
|---:|----|---------|-------|---------:|---------|
| 1 | #387 | alphonse | full + grad-clip-1.0 | **74.44** | **Merged (current baseline)** |
| 2 | #301 | nezuko | L1+warmup+Fourier+sw=30 | 76.68 | Merged (previous baseline) |
| 3 | #385 | fern | L1+warmup+Fourier+wd=5e-4 (sw=10) | 77.29 | Sent back |
| 4 | #387 (1st) | alphonse | L1+warmup+grad-clip (no Fourier) | 81.81 | Superseded by rerun #387 |
| 5 | #364 | edward | L1+warmup+Fourier+Huber-beta-1.0 | 85.58 | Sent back, refined to beta=0.5 |
| 6 | #365 | thorfinn | L1+warmup+Fourier | 87.86 | Merged |
| 7 | #385 (1st) | fern | L1+warmup+wd=5e-4 (no Fourier) | 87.27 | Superseded |
| 8 | #296 | fern | L1+warmup+budget | 94.54 | Merged |
| 9 | #293 | edward | L1 only | 101.87 | Merged |

## Notable directional findings

1. **alphonse's gradient-norm telemetry** is the most valuable instrumentation of the project. Pre-clip ‖∇‖ went from peak 105/end 25 (pre-Fourier) to peak 270/end 63 (post-Fourier) — Fourier features ~2.5× the gradient signal. Clipping is doing more work, not less, post-Fourier. This generalizes to all PRs — every future PR should include the per-epoch grad-norm telemetry in its JSONL.

2. **Diminishing returns starting to appear.** Each axis tested individually gave 8–14% improvements. Now stacking on a 5-axis full stack, the per-axis gain is dropping to 2–3%. Some of this is overlap (e.g. Fourier and clipping both smooth the optimizer trajectory); some is closer-to-optimum dynamics.

3. **`val_geom_camber_rc` Fourier anomaly persists.** Across multiple PRs, this split has the smallest gain (and sometimes regresses). Most likely explanation: the residual error is geometry-extrapolation-dominated, not anything our current axes target. Future direction: domain conditioning, test-time augmentation, per-Re conditioning.

4. **edward's residual-magnitude analysis (PR #364):** beta=1.0 in smooth_l1 is mis-calibrated for unit-variance normalized targets — residuals at convergence are <<1σ, so Huber operates in MSE-mode for late training. Next: beta=0.5 on full stack.

5. **Volume regression with surf_weight=30 (PR #301):** `val_avg/mae_vol_p` regressed by +13.2%. Volume isn't ranked, but worth tracking; nezuko's #444 (surf_p_extra=3.0) attempts to localize the surface emphasis to just the p channel without further hurting volume.

## Potential next research directions

When the pending PRs land:
- **Stack the remaining round-3 reruns** (EMA, ckpt-avg, wd=5e-4, dsdf-Fourier, Huber-beta-0.5) — see how far the orthogonal-additivity story holds before saturation.
- **`val_geom_camber_rc` deep-dive** — domain-conditioning, test-time augmentation, or per-Re conditioning to specifically attack this geometry-extrapolation-dominated split.
- **Mesh-aware augmentation** — random node-loss subsampling.
- **Output residual from a free-stream estimate** for `Ux, Uy`.
- **Trainable Fourier projection** (Tancik 2020).
- **Mixed precision (bf16)** — speedup → more epochs in same budget.
- **Pressure-channel-only volume weighting** — currently `vol_loss` treats Ux/Uy/p equally; pressure dominates the eval metric.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
- For PRs that are CLI-flag-only changes (no train.py diff), the Config default is updated on the advisor branch in a follow-up commit at merge time so future PRs reproduce the new baseline without explicit flags.
- Per-epoch grad-norm telemetry is now in the merged train.py (PR #387) — every future PR's metrics.jsonl includes `train/grad_norm_avg`. Use this for diagnostic when reviewing.
