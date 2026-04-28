# SENPAI Research State

- **Date:** 2026-04-28 02:05
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **76.68** | PR #301 (nezuko, L1 + warmup + Fourier + sw=30) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 73.40 | PR #301 |

Per-split val on the new baseline: `val_single_in_dist=87.59`, `val_geom_camber_rc=88.15`, `val_geom_camber_cruise=55.71`, `val_re_rand=75.26`.

**Volume-pressure regression flag:** `val_avg/mae_vol_p = 104.43` at the sw=30 baseline (vs 92.29 at the prior Fourier-only baseline) — a +13.2% regression. Volume isn't ranked, but if `surf_weight` keeps creeping up, the joint flow representation could degrade enough to eventually undermine surface accuracy. We're now testing pressure-only-channel boosts (PR #444) to see if we can decouple surface-p emphasis from this volume tradeoff.

Four orthogonal axes now stacked: **L1 loss** (PR #293) × **linear warmup → cosine** with peak `lr=1e-3` and budget-matched `--epochs 14` (PR #296) × **8-band Fourier features on normalized (x, z)** (PR #365) × **`surf_weight=30`** (PR #301). All other knobs at originals: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `weight_decay=1e-4`, `batch_size=4`, `fun_dim=54`.

## Current research focus

Round-3 reruns on the post-Fourier baseline are landing thick and fast. The pattern: every cheap-per-epoch axis tested so far (loss formulation, schedule, feature augmentation, regularization, surface emphasis) gives ~7–14% improvements when stacked on the right base, and they appear to be *mostly orthogonal* — pairs of orthogonal axes give roughly the sum of their individual deltas.

The pipeline of pending stacked measurements is large, so the next several merges will be sequential: each PR rebases onto the latest baseline, reruns to confirm the stack still helps, and merges.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR. Diagnosed independently 7+ times: test sample 20 has 761 non-finite values in volume `p` channel of GT; `data/scoring.accumulate_batch` computes `(pred_orig - y).abs()` before masking, so `NaN * 0 = NaN` propagates. `data/scoring.py` is read-only per program constraints. Rank by **3-clean-split test mean** alongside `val_avg/mae_surf_p`.

## Open PRs

### Sent back to rebase onto current advisor (status:wip)

These are all sitting at status:wip with the student rerunning on the current baseline. As reruns land, expect a sequence of merges in val-rank order.

| PR | Axis | Student | Hypothesis | Last-measured val (on prior baseline) |
|----|------|---------|------------|--------:|
| #303 | Weights | tanjiro | EMA weights (decay 0.999) | 127.6 (on MSE) |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging top-3 + val-on-averaged | (val on averaged not measured yet; test 91.13 on L1) |
| #385 | Regularization | fern | `weight_decay` 1e-4 → 5e-4 | 77.29 (on L1+warmup+Fourier; need rebase to sw=30) |
| #387 | Stability | alphonse | Gradient clipping `max_norm=1.0` | **81.81** (on L1+warmup, NO Fourier; standout — stacking on Fourier expected to give substantial new best) |

### Round 2 carry-over (status:wip, on top of L1 only — pre-warmup, pre-Fourier, pre-sw=30)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber (smooth_l1, beta=1.0) |
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |

When these results land they'll be on a 3-axis-stale baseline; expect they'll need to rebase + rerun to get a measurement against the current advisor.

### Round 4 / 5 (status:wip, on top of current baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #414 | Feature | thorfinn | Fourier features on dsdf channels (4 freqs, dims 2–11) |
| #444 | Loss balance | nezuko | `surf_p_extra=3.0` — boost surface-p channel only, no Ux/Uy starvation |

## Round-1+2+3 ranking (val_avg/mae_surf_p)

| Rank | PR | Student | Stack | val_avg | Verdict |
|---:|----|---------|-------|---------:|---------|
| 1 | #301 | nezuko | L1+warmup+Fourier+sw=30 | **76.68** | **Merged (current baseline)** |
| 2 | #385 | fern | L1+warmup+Fourier+wd=5e-4 (sw=10) | 77.29 | Sent back (need rebase to sw=30) |
| 3 | #387 | alphonse | L1+warmup+grad-clip (no Fourier) | 81.81 | Sent back (rebase + rerun on full stack) |
| 4 | #365 | thorfinn | L1+warmup+Fourier | 87.86 | Merged (previous baseline) |
| 5 | #385 (run #1) | fern | L1+warmup+wd=5e-4 (no Fourier) | 87.27 | Superseded by rerun |
| 6 | #296 | fern | L1+warmup+budget | 94.54 | Merged (older baseline) |
| 7 | #293 | edward | L1 only | 101.87 | Merged (oldest baseline) |
| — | #380 | frieren | L1+ckpt-avg (single-best val) | 104.43 | Sent back |
| — | #278 | alphonse | L1+surf_p_weight=5 | 108.63 | Closed (falsified at 5×) |
| — | #303 | tanjiro | EMA on MSE | 127.65 | Sent back |
| — | #299 | frieren | n_layers=8 | 139.29 | Closed (budget penalty) |
| — | #290 | askeladd | wider 192 | 152.24 | Closed (budget penalty) |
| — | #305 | thorfinn | slices+heads 2x | 160.68 | Closed (budget + dim_head=16) |

## Notable directional findings

1. **alphonse's gradient-norm telemetry (PR #387):** pre-clip ‖∇‖ was 25–100× the threshold throughout training. Under L1 loss specifically, gradient magnitudes don't naturally decay with residuals, so cosine-decayed LR alone can't control step sizes. This is doing fundamental optimization work, not just stability — and **it generalizes to all PRs on this branch** since they all use L1 now. Candidate for default-level inclusion in a future merged PR.

2. **fern's train/val gap analysis (PR #385):** Fourier features *widened* the gap from −0.185 to −0.330 — they gave the model more capacity that the same WD has to discipline. Suggests room for stronger regularization in the Fourier+sw=30 regime.

3. **Per-split gain pattern shifted with stack depth:** in early rounds, gains concentrated on `val_single_in_dist`. With Fourier added, gains shifted toward `val_geom_camber_rc` and `val_re_rand` (OOD splits). The "WD targets OOD" framing only emerged on the richer-input baseline.

4. **`val_geom_camber_rc` Fourier anomaly (PR #365):** improved least under Fourier (−1.0% vs −8.5% to −10.8% on other splits), suggesting that split's residual is **geometry-extrapolation-dominated** rather than spectral-bias-dominated. Worth a future deep-dive (domain conditioning, test-time augmentation, per-Re conditioning).

5. **Volume regression with surf_weight=30 (PR #301):** `val_avg/mae_vol_p` regressed by +13.2%. Not ranked, but worth tracking — too aggressive a surf_weight may eventually undermine the joint flow representation.

## Potential next research directions

When the stack converges and pending PRs land:
- **Pressure-only surface boost** (active: PR #444) — addresses the volume regression directly.
- **Stack alphonse's grad-clip on top of full stack** — predicted to give the next big jump (his PR's 14% delta on a partial stack suggests substantial room remains).
- **WD sweep on top of full stack** — if fern's wd=5e-4 stacks, push to 1e-3 / 2e-3.
- **`val_geom_camber_rc` deep-dive** — domain-conditioning, test-time augmentation, or per-Re conditioning.
- **Mesh-aware augmentation** — random node-loss subsampling.
- **Output residual from a free-stream estimate** for `Ux, Uy`.
- **Trainable Fourier projection** (Tancik 2020).

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
- For PRs that are CLI-flag-only changes (no train.py diff), the Config default is updated on the advisor branch in a follow-up commit at merge time so future PRs reproduce the new baseline without explicit flags.
