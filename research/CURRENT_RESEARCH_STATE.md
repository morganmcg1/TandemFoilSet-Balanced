# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-12
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** none received — this is a controlled 24/48h Charlie-vs-Willow logging ablation. Treated as research-isolated.

## Research focus

Round 1 baseline now established by PR #1504 (mask-aware PhysicsAttention, merged 2026-05-12 21:52):

- **`val_avg/mae_surf_p` = 119.450**, **`test_avg/mae_surf_p` = 109.669**
- Mask-aware code is now the reference `train.py`. It fixes the `test_geom_camber_cruise=None` failure mode that affected every other unmasked round-1 run.

Round 1 in-flight (7 PRs) is now running on the **pre-merge** code (no mask fix). Their test_geom_camber_cruise will likely return None. Plan: let them finish their current runs, then send back any that report `test_avg=None` for a clean rebase + re-run on the merged baseline. The underlying hypotheses (Huber, wider, more slices, surf_weight 25, warmup+lr, deeper, AdamW betas) are still valuable; only the comparison surface needs to be fixed.

## Round 1 portfolio (status)

| PR    | Student   | Hypothesis axis                  | Status |
|-------|-----------|----------------------------------|--------|
| #1504 | alphonse  | Mask-aware PhysicsAttention      | **MERGED** (val=119.45, test=109.67) |
| #1505 | askeladd  | Huber surface loss (β=0.5)       | WIP, pre-merge code |
| #1506 | edward    | Wider hidden (128→192)           | WIP, pre-merge code |
| #1507 | fern      | More slices (64→128)             | WIP, pre-merge code |
| #1508 | frieren   | surf_weight 10→25                | WIP, pre-merge code |
| #1509 | nezuko    | Warmup + lr=1e-3                 | WIP, pre-merge code |
| #1510 | tanjiro   | Fourier pos enc (L=6)            | CLOSED (cruise NaN, pre-merge) |
| #1511 | thorfinn  | Deeper (5→7 layers)              | WIP, pre-merge code |
| #1589 | tanjiro   | AdamW betas (0.9, 0.95)          | WIP, pre-merge code |
| #1623 | alphonse  | mlp_ratio 2→4                    | WIP, post-merge baseline |

Axes covered: **architecture (width, depth, slice count, pos enc, FFN ratio), loss (Huber, surf weight), optimization (LR/warmup, AdamW betas), correctness (mask) → done**.

## Potential next research directions

After Round 1 lands we'll know which axis is most productive. Likely follow-ups, by axis:

- **If mask-aware attention wins**: combine with Huber, then try mask-aware AND fine slicing together.
- **If loss changes win**: explore relative-error / per-sample-std weighting, and revisit `surf_weight` sweep with the new loss.
- **If width/depth wins**: scale to n_hidden=256 or n_layers=9; check VRAM headroom; consider mlp_ratio=4.
- **If Fourier pos enc wins**: tune `n_freq` and `pos_scale`; consider learned multi-scale Fourier (NeRF Gaussian features).
- **If LR warmup wins**: revisit AdamW betas, weight decay, and try OneCycle/CyclicLR variants.

Larger swings reserved for later rounds (after baseline is established):
- Surface-anchored cross-attention (boundary nodes as queries against volume tokens)
- Per-sample Re-normalization to balance high-Re vs low-Re gradient magnitudes
- Non-NACA foil disambiguation feature
- Optimizer swaps (Muon, LION, SOAP) — only if the current optimizer is plausibly suboptimal after we've tuned schedule
- Multi-resolution slice attention (multi-scale hierarchies of slice tokens)

## Open questions and ruled-out paths

- **Strong signal (2026-05-12 21:15):** PR #1504 (mask-aware PhysicsAttention) is the only finished round-1 run with a populated `test_avg/mae_surf_p` — all other finished baselines hit `test_geom_camber_cruise=None`. This includes the Fourier PR #1510 (closed) and the finished runs of #1506/#1508/#1511. Strong evidence that the baseline's unmasked slice softmax produces inf/NaN on at least one cruise test sample, and mask-aware attention fixes it. **PR #1504 is now the highest-priority merge candidate** of round 1 — both a metric improvement and a correctness fix on the paper-facing metric.
- **Open:** Is the loss/metric mismatch (MSE training vs MAE evaluation) actually a big lever, or has the surface weight already absorbed it?
- **Open:** Is the dataset bottleneck on the geometry-camber holdouts (M=6-8 raceCar, M=2-4 cruise) inductive-bias-bound, or capacity-bound?
- **Resolved:** Padding in `pad_collate` does produce measurable noise — see PR #1504 evidence above.
- **Ruled-out:** Fourier positional encoding (PR #1510, both `pos_scale=1.0` and `pos_scale=0.1`) cannot be fairly evaluated until the cruise NaN is resolved. Not a Fourier-spectrum problem.

## Operating notes

- `SENPAI_TIMEOUT_MINUTES=30` per run; `SENPAI_MAX_EPOCHS` unset (defaults to `cfg.epochs=50`). Most runs will exit on wall clock first.
- All runs grouped in W&B project `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r3` with `--wandb_group` set per hypothesis family.
- This launch is isolated to `icml-appendix-willow-pai2g-48h-r3` and the 8 assigned student PR branches. No cross-launch comparison.
