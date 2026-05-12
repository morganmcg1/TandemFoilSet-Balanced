# SENPAI Research State

- **Date**: 2026-05-12 19:30 (round 1 partial results; Huber-loss winner pending rebase)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 117.17** — PR #1479 (grad-clip-1), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=50)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, MSE loss.

**Pending winner**: PR #1473 (huber-loss, tanjiro) achieved **val_avg/mae_surf_p = 111.296** (~5% improvement) but was forked pre-grad-clip. Sent back for rebase + re-verify; will merge as new baseline on confirmation.

---

## Current Research Focus

**Round 1 finding**: The Transolver baseline was gradient-unstable without clipping (pre-clip norms 50–800, 100% clip rate). `grad_clip=1.0` is now mandatory infrastructure. All future experiments must build on the grad-clip baseline.

**Round 1 loss-reformulation signal**: Huber(delta=0.5) on the pre-grad-clip base gave 111.30 (vs 117.17 baseline on grad-clip base). Even without head-to-head, the L2-fraction trajectory (75%→93%) shows healthy outlier-capping behavior. **Composing Huber + grad-clip is the highest-confidence path to a new baseline.**

**Key open questions**:
1. Does Huber compound on top of grad-clip, or does grad-clip already absorb the outlier-gradient signal? (Tanjiro rebase will answer this.)
2. Do architecture scaling (wider-deeper) and LR changes (higher-lr-cosine-14) compound on top of the stable grad-clip baseline?
3. Once Huber merges, can per-channel delta (delta_p < delta_Ux ≈ delta_Uy) extract additional gains given pressure has the widest residual distribution?

**Per-split profile** at baseline:
- Easiest: `val_geom_camber_cruise` (87.04) — cruise tandem, lower Re, smaller y magnitudes
- Hardest: `val_geom_camber_rc` (134.17) and `val_single_in_dist` (134.83) — raceCar regime, high Re, extreme p values
- Huber improves the hardest splits most (rc: 134.17 → 122.56, re_rand: 112.66 → 99.57), consistent with outlier-capping theory

**Known bug**: `test_geom_camber_cruise` has a corrupted sample (±Inf in GT pressure). Fix: add y-sanitization wrapper in `train.py:evaluate_split` before calling `accumulate_batch`. Tanjiro independently diagnosed the bug — fix now standard in all sent-back PRs.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (round 1) | Pre-grad-clip base; may need re-run if not winner |
| #1457 | askeladd | `surf-weight-50` | WIP (v2, sent back) | surf_weight=30 + grad-clip baseline |
| #1458 | edward | `wider-deeper` | WIP (v2, sent back, dirty) | batch_size=4 + grad-clip baseline |
| #1460 | fern | `relative-l2-loss` | WIP (round 1) | Pre-grad-clip base |
| #1462 | frieren | `warmup-cosine` | WIP (v2, sent back, dirty) | 1-epoch warmup + grad-clip baseline |
| #1467 | nezuko | `more-slices-128` | WIP (round 1) | Pre-grad-clip base |
| #1473 | tanjiro | `huber-loss` | WIP (v2, sent back) | **Winner candidate** — rebase + grad-clip + NaN fix; re-merge target |
| #1518 | thorfinn | `higher-lr-cosine-14` | WIP (new) | lr=1e-3, T_max=14, exploits grad-clip stability |

---

## Potential Next Research Directions

Round 2 candidates (pending round 1 final results + Huber rebase confirmation):

- **Compound: Huber + bf16-amp**: If bf16 gives more epochs and Huber-on-grad-clip merges, the compound should dominate.
- **Per-channel Huber delta**: `delta_p < delta_Ux ≈ delta_Uy` given pressure has the widest residuals (tanjiro suggested follow-up #3). Try `delta = {p: 0.3, Ux: 0.7, Uy: 0.7}`.
- **Huber delta sweep**: Once Huber-on-grad-clip is the baseline, sweep `delta ∈ {0.3, 0.7}`.
- **surf_weight sweep**: Test 25, 50, 75 once stable Huber baseline confirmed.
- **`lion-optimizer`**: Lower memory + potential convergence benefit; needs LR tuning.
- **Fourier/position encoding**: Improve geometric encoding for OOD camber generalization (the harder splits are raceCar geometry and single-foil, not cruise).
- **SiLU activation**: Cheap, orthogonal to all other changes.
- **EMA**: Smooth val metrics from noisy 14-epoch trajectories.
- **CosineAnnealingWarmRestarts**: May suit the 14-epoch budget better than standard cosine.

**Plateau protocol**: If 5 consecutive experiments fail to beat the current confirmed baseline, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
