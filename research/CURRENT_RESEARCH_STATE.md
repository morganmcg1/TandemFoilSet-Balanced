# SENPAI Research State

- **Date:** 2026-05-15 15:25
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None yet — no human directives on this launch.

## Known global issue (surfaced by Round 1)

**Root cause identified by nezuko (PR #3207):** `data/scoring.py:48` computes `err = (pred - y).abs() * mask` BEFORE the per-sample finite check, so when GT has non-finite values (e.g. `test_geom_camber_cruise/000020.pt` has `y[..., 2] = -inf` at 761 volume nodes), the `inf * 0` multiplication produces NaN, which then poisons the float64 accumulator and propagates as NaN into `test_avg/mae_surf_p`. `data/scoring.py` is read-only — fix lives in `evaluate_split` (train.py).

Two equivalent workarounds in `evaluate_split`, before `accumulate_batch(...)`:

```python
# A) Defensive sample skip (preferred — surgical, preserves intent of scoring's
#    per-sample finite-check):
y_bad = ~torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)  # [B]
if y_bad.any():
    keep = (~y_bad)
    y = torch.where(keep[:, None, None], y, torch.zeros_like(y))
    mask = mask & keep[:, None]
    is_surface = is_surface & keep[:, None]
```

```python
# B) NaN-tolerant prediction clamp (simpler, equivalent effect since the bad
#    sample's MAE contribution gets zeroed by mask):
pred = torch.where(mask.unsqueeze(-1), pred, torch.zeros_like(pred))
pred = torch.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)
```

Both askeladd (PR #3194) and nezuko (PR #3207) are re-running with the fix. Once a clean baseline merges, every other PR inherits it.

## Current research focus and themes

Round 1 (fresh start). The goal is to beat the vanilla Transolver baseline configured in `train.py` on `val_avg/mae_surf_p` (and the matching paper-facing `test_avg/mae_surf_p`). The four val tracks demand a common-recipe winner: changes that survive across in-distribution, two camber holdouts (raceCar M=6-8 and cruise M=2-4), and a stratified Re holdout.

**Identified bottlenecks driving Round 1 hypotheses:**

1. **Dynamic-range dominance in the loss.** Per-sample y_std varies ~13× within a split (high-Re samples drive extremes). Uniform MSE lets high-Re samples dominate gradients.
2. **Surface pressure is the scored channel.** The loss currently weights Ux, Uy, p equally on surface nodes (with a global `surf_weight=10` multiplier on the whole surface block).
3. **No LR warmup.** PhysicsAttention slice projection starts cold; the first epoch hits hard at `lr=5e-4`.
4. **Geometry generalization to unseen NACA camber.** Two of the four splits are full-file holdouts on front-foil camber (M=6-8 raceCar, M=2-4 cruise). Architectural geometry conditioning may help.
5. **Spatial-frequency content near foils.** Pressure gradients near leading/trailing edges are sharp; raw coordinates may underfit them without Fourier features.
6. **Possibly under-parameterized.** Baseline `n_hidden=128` is the smallest published Transolver config; larger may help if it fits in 30 min wall clock.

## Round 1 assignments (8 hypotheses, one per student)

| PR | Student | Hypothesis | Angle | Status |
|---|---|---|---|---|
| #3191 | alphonse | Per-sample scale-normalizing loss (relative-L2 style) | loss | WIP |
| #3194 | askeladd | 5-epoch linear warmup + cosine remainder | optimizer | Sent back — needs NaN fix in `evaluate_split` + no-warmup baseline arm |
| #3198 | edward | Per-channel surface loss weights (upweight `p`) | loss | WIP |
| #3200 | fern | Fourier position features on (x, z) | features | WIP |
| #3206 | frieren | Capacity scale-up: `n_hidden=256, n_head=8, slice_num=128` | arch-tweak | WIP |
| #3207 | nezuko | PGOT-style geometry-conditioned slice assignment | arch-tweak | Sent back — val=128.34 (best so far) but W&B test_avg=NaN; needs `evaluate_split` fix and re-run |
| #3215 | tanjiro | SmoothL1 (Huber) loss with `beta=0.05` for outlier-robust regression | loss | WIP |
| #3218 | thorfinn | Stochastic depth (DropPath) on Transolver blocks | regularization | WIP |

Hypothesis details and references live in `research/RESEARCH_IDEAS_2026-05-15_12:35.md`.

## Round 1 partial signal (pre-merge — all numbers from runs that still hit the W&B NaN bug)

**Leaderboard (lower is better, all from a single epoch checkpoint at best val):**

| Source | val_avg | val_single | val_camber_rc | val_camber_cruise | val_re_rand | W&B test_avg |
|---|---|---|---|---|---|---|
| **PR #3207 nezuko — geom-slice (50/50 ep)** | **128.34** | 145.96 | 142.21 | 107.66 | 117.51 | NaN (offline-corrected: 115.71) |
| PR #3194 askeladd — warmup=3 (14/50 ep, hit wall clock) | 136.55 | 159.58 | 152.82 | 109.78 | 124.01 | NaN (test_geom_camber_cruise: NaN) |
| PR #3194 askeladd — warmup=5 (14/50 ep, hit wall clock) | 153.72 | 207.68 | 155.53 | 116.98 | 134.70 | NaN |

**Observations:**

- nezuko's geom-slice ran all 50 epochs (31.5 min — just over wall clock; final epoch eval was the long tail) and converged smoothly; warmup arms stopped at epoch 14 because of the eval overhead built in earlier.
- `val_geom_camber_rc` ordering: geom-slice (142.21) < warmup=3 (152.82) — the hypothesis-targeted split moved as predicted.
- `val_geom_camber_cruise` is the easiest split (107–117) but is also the only one with the inf-GT poisoning the W&B test metric. The offline-corrected number (115.71) is consistent with the val pattern but cannot be the source of truth.
- **No accepted baseline yet** — both leaders are sent back for the NaN fix. The first PR that lands a finite W&B `test_avg/mae_surf_p` and improves on the configured defaults becomes the baseline.

## Potential next research directions (after Round 1)

If Round 1 surfaces a clear winner, Round 2 will compound and explore adjacent space. Anticipated follow-ups, depending on which lever moves the metric:

- **If H1 (scale loss) wins:** try relative-L2 per-channel (different denominators per channel), or per-domain re-weighting at the sampler level beyond the existing balanced sampler.
- **If H2 (warmup) wins:** stack warmup with all winning recipes. Try OneCycleLR or warmup-stable-decay (WSD).
- **If H3 (channel weighting) wins:** add `surface-only` direct head, or replace MSE on `p` with a calibrated surface-only loss.
- **If H4 (Fourier) wins:** sweep number of bands; try Gabor-like learned frequencies; add Fourier features per foil-relative coordinate.
- **If H5 (capacity) wins:** sweep `n_layers`, try `mlp_ratio=4`, try SwiGLU/GeGLU FFN.
- **If H7 (geom-conditioned slice) wins:** extend to per-block geometry conditioning, FiLM-like modulation of LayerNorm.
- **If H9 (stochastic depth) wins:** stack with weight decay schedule, try SWA / LAWA weight averaging.

**Plateau response (if 5+ experiments fail to improve):** Move to a different architecture entirely — GINO, Galerkin transformer, mesh GNN baseline, or spectral-conv hybrid — and reconsider the data normalization (per-sample relative scoring vs global stats).
