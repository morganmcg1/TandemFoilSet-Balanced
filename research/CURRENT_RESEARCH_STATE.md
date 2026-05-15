# SENPAI Research State

- **Date:** 2026-05-15 (updated 16:00 after round-1 batch review)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline:** `val_avg/mae_surf_p = 110.83`, `test_avg/mae_surf_p (excl cruise) = 109.75`
  - Previously 135.30/135.54; improved −18.1%/−19.0% by Huber merge (PR #3155) and −8.9%/−10.7% by LR warmup merge (PR #3147).

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model because at least one test-cruise sample produces a non-finite pressure prediction whose squared error propagates Inf through `data/scoring.py:accumulate_batch`. Validation cruise is finite — only test cruise is broken. Until fixed, every PR uses the 3-split `test_avg/mae_surf_p (excl cruise)` for paper-facing comparison.

Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions before accumulate_batch. Deferred to a dedicated small PR after round-2 clears.

## Round-1 outcomes summary

| PR | Student | Hypothesis | Δ val vs canonical | Δ test vs canonical | Decision |
|---|---|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192, 4→6 heads) | +18.7% | +13.9% | Closed |
| #3147 | askeladd | LR warmup + peak 5e-4→1e-3 | **−8.9%** | **−10.7%** | **Merged** |
| #3152 | edward | Per-channel p×3 MSE upweight | +0.6% (noise) | +3.1% | Request changes |
| #3155 | fern | Huber loss (SmoothL1 delta=1.0) | **−18.1%** | **−19.0%** | **Merged** |
| #3161 | frieren | Per-sample loss normalization | +13.0% | +9.4% | Closed |
| #3165 | nezuko | Depth scaling 5→8 layers | +25.4% | +29.1% | Closed |
| #3169 | tanjiro | MLP ratio 2→4 | stale WIP — pending triage | — | TBD |
| #3172 | thorfinn | Fourier pos features + slice 96 | stale WIP — pending triage | — | TBD |

**Key learnings from round 1:**
1. **Robust loss is the dominant lever.** Huber −18.1% dominates all other interventions. MSE was vulnerable to outlier pressure samples; linear Huber tails reduce their influence dramatically.
2. **LR warmup compounds with loss changes.** −8.9% from higher peak + warmup; likely orthogonal to Huber.
3. **Capacity scaling is blocked under the 30-min cap.** Width, depth, and MLP-ratio expansions all incur ~1.55× epoch-time penalty, cutting epoch count by ~36%. No capacity expansion survives this regime.
4. **Per-sample loss normalization hurts.** Equal-weight-per-sample destabilizes gradient balance across variable-size meshes.

## Current focus: round-2 assignments (4 students idle, pending dispatch)

| Student | Hypothesis | Family | Rationale |
|---|---|---|---|
| fern | Huber delta sensitivity: delta=0.5 and 2.0 arms | Loss tuning | Determine whether 1.0 is optimal for this pressure scale |
| askeladd | Warmup duration sweep: 2-epoch and 5-epoch warmup | Optimization | Tune the key warmup hyperparameter |
| frieren | AoA reflection augmentation (negate AoA, flip z, sign-flip Uy for raceCar) | Data augmentation | Highest-EV S-risk idea; doubles raceCar data; attacks rc split |
| nezuko | Attention entropy regularization (PhysicsAttention slice uniformity) | Architecture | Light-touch; no per-step cost increase |

Also in flight:
- **alphonse**: SOAP optimizer PR #3283 (WIP)
- **tanjiro**: mlp_ratio 2→4 PR #3169 (stale, under investigation)
- **thorfinn**: Fourier pos + slice_num 96 PR #3172 (stale, under investigation)
- **edward**: Surface-only p×3 upweight follow-up PR #3152 (WIP, bounced for changes)

## Potential next research directions

After round-2 results, **stack confirmed winners** and explore:

1. **SOAP optimizer** (in flight — alphonse #3283) — Hessian preconditioning may reduce vol/surf gradient conflict.
2. **Ada-Temp slice reparameterization** — per-point temperature in PhysicsAttention softmax; targets slice-collapse.
3. **Log-Re sinusoidal embedding** — 8-dim sinusoidal features on log(Re); targets `val_re_rand` OOD.
4. **Alternative robust losses** — Cauchy/Welsch/Tukey biweight (Huber win signals outlier-robustness is a key lever; deeper exploration warranted).
5. **Physical-units scale-aware loss** — normalize each field loss by physical scale (edward's analysis: normalized-space underweights p).
6. **Divergence-free auxiliary loss** — soft penalty on div(U) via finite differences; incompressibility constraint.
7. **Per-domain normalization** — domain-conditioned y stats vs global pooling.
8. **Separate surface decoder head** — dedicated wider MLP for surface nodes; orthogonal to loss-side changes.
