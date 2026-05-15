# SENPAI Research State

- **Last updated**: 2026-05-15 ~15:45 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline
- `val_avg/mae_surf_p` = **117.66** (PR #3237, edward, `huber-loss`, epoch 13)
- Per-split: single=147.77, camber_rc=125.08, camber_cruise=88.98, re_rand=108.81
- Change from default: Huber(δ=1.0) instead of MSE. Everything else at default.

## Key observations from round 1
1. **30-min cap is the main bottleneck**: Every student hit the 30-min wall-clock cap at epoch 14/50. The val loss was still descending at timeout. Getting more epochs per budget is high-priority.
2. **Huber loss works**: 2-line change, cleanly improves over MSE. Now the default on this branch.
3. **Local-Re feature is promising** but needs Huber on top. With MSE it lands at 124.27 (askeladd, PR #3235 sent back for revision with Huber + saf arc-length variant).
4. **Curriculum failed**: Re-sorted ascending order without domain balance hurt generalization badly. The design was also never fully tested (curriculum phase never finished in 30 min). Closed.
5. **NaN bug in test metric**: `test_geom_camber_cruise/000020.pt` has `inf` in GT. `data/scoring.py` correctly identifies it but `inf * 0 = NaN` poisons the accumulator. All `test_avg/mae_surf_p` values are NaN. Ranking on `val_avg/mae_surf_p` is unaffected. See EXPERIMENTS_LOG.md.
6. **5 other experiments still in progress**: alphonse (per-sample-scale-norm), fern (dual-branch-heads), frieren (fourier-pos-enc), nezuko (hflip-augment), tanjiro (ema-weights) — all running.

## Active PRs

| # | Student | Slug | Status |
|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP |
| #3235 | askeladd | `local-re-feature` | WIP — rerun with Huber + saf coord |
| #3238 | fern | `dual-branch-heads` | WIP |
| #3239 | frieren | `fourier-pos-enc` | WIP |
| #3240 | nezuko | `hflip-augment` | WIP |
| #3241 | tanjiro | `ema-weights` | WIP |
| #3300 | edward | `bf16-mixed-precision` | WIP |
| #3303 | thorfinn | `surf-weight-50` | WIP |

## Human research direction
None received yet.

## Current research themes

**Budget efficiency** (edward #3300):
- BF16 mixed precision to get ~20–25 epochs in 30 min vs. current 14
- If this works, it becomes the new baseline and unlocks the true potential of all other techniques

**Loss formulation** (thorfinn #3303, alphonse #3177):
- surf_weight=50 with Huber: maximize surface focus at expense of vol accuracy (which doesn't rank)
- per-sample-scale-norm: equalize Re-regime gradient magnitudes (complementary to Huber)

**Features / Architecture** (askeladd #3235, fern #3238, frieren #3239):
- local-Re feature + Huber: boundary-layer physics signal on surface nodes
- dual-branch heads: specialized surface/volume output MLPs
- Fourier positional encoding: multi-scale spatial features over (x, z)

**Data / Augmentation** (nezuko #3240):
- z-reflection symmetry: physical symmetry of NS equations, free 2x effective data

**Optimization** (tanjiro #3241):
- EMA weight averaging: smoother checkpoint, reduces late-training noise

## Potential round 3 directions (post-round 2 results)
1. **Compose winners**: stack BF16 + Huber + best-performing feature/arch change
2. **Per-channel pressure loss**: extra loss term on channel 2 (p) only, since mae_surf_p is the metric
3. **Warmup-cosine schedule**: 3-epoch linear warmup then cosine — helps if model diverges early
4. **Larger model** (n_hidden=192 or 256): test if capacity is the remaining bottleneck
5. **Per-domain normalization**: separate (y_mean, y_std) per domain to remove cross-domain scaling artifacts
6. **Huber delta sweep** (δ=0.5, 2.0): test sensitivity around the current δ=1.0
7. **Larger batch** (batch_size=8): reduces gradient variance, pairs well with BF16
8. **Fix scoring.py NaN bug**: unblock paper-facing test_avg/mae_surf_p (requires read-only waiver or human coordinator action)

## No idle students
All 8 students are assigned.
