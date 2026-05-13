# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-12
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** none received — this is a controlled 24/48h Charlie-vs-Willow logging ablation. Treated as research-isolated.

## Research focus

Round-1 baseline has shifted **twice** in the past 2 hours through stacking compatible winners:

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 baseline |
| PR #1505 (Huber β=0.5) | 2026-05-13 00:00 | **113.794** | **101.782** | −4.74% val, −7.19% test |

The current `train.py` has both fixes stacked. The mask is applied after slice softmax (line 195 area) and the surface loss is Huber β=0.5 at both train (line 508) and eval (line 260).

**Pattern emerging from round 1:**
- Correctness and loss-formulation changes are both winning by 5-10% each, stacking additively.
- All 4 of the "compute-bound" hypotheses (slice=128, deeper=7, surf_weight=25, fern's slice retry) closed because the per-epoch cost cut them below the wall-clock convergence horizon.
- The `test_geom_camber_cruise` NaN affected every unmasked baseline. Resolved by mask. Worth a follow-up sanity check whether the underlying data issue (`scoring.py` cruise sample 20 with inf pressure) is fully neutralized or just masked.

Round 1 in-flight (8 PRs) on various baselines:
- **#1623 alphonse (mlp_ratio=4)**: pre-Huber baseline (post-mask-aware)
- **#1506 edward (n_hidden=192)**, **#1509 nezuko (warmup+lr=1e-3)**, **#1511 thorfinn (n_layers=7)**: pre-mask baseline — need full rebase
- **#1589 tanjiro (AdamW betas)**: rebasing from pre-mask
- **#1692 fern (grad_clip=1.0)**: from mask-aware baseline, pre-Huber
- **#1712 askeladd (Huber β=0.25)**: from current merged baseline
- **#1715 frieren (bf16 AMP)**: from current merged baseline

All 6 pre-Huber PRs got a heads-up about the new baseline; they need to clear val < 113.79 / test < 101.78 to merge.

## Round 1 portfolio (status)

| PR    | Student   | Hypothesis axis                  | Status |
|-------|-----------|----------------------------------|--------|
| #1504 | alphonse  | Mask-aware PhysicsAttention      | **MERGED** 21:52 (val=119.45, test=109.67) |
| #1505 | askeladd  | Huber surface loss (β=0.5)       | **MERGED** 00:00 (val=113.79, test=101.78) |
| #1506 | edward    | Wider hidden (128→192)           | WIP, pre-mask code, heads-up posted |
| #1507 | fern      | More slices (64→128)             | CLOSED (compute-bound, +27%) |
| #1508 | frieren   | surf_weight 10→25                | CLOSED (compute-bound, +16%) |
| #1509 | nezuko    | Warmup + lr=1e-3                 | WIP, pre-mask code, heads-up posted |
| #1510 | tanjiro   | Fourier pos enc (L=6)            | CLOSED (cruise NaN, pre-mask) |
| #1511 | thorfinn  | Deeper (5→7 layers)              | WIP, pre-mask code, heads-up posted |
| #1589 | tanjiro   | AdamW betas (0.9, 0.95)          | WIP, rebasing onto mask-aware |
| #1623 | alphonse  | mlp_ratio 2→4                    | WIP, post-mask, pre-Huber |
| #1692 | fern      | Gradient clipping (max_norm=1.0) | WIP, post-mask, pre-Huber |
| #1712 | askeladd  | Huber β=0.25 (β-tune)            | WIP, current baseline (just assigned) |
| #1715 | frieren   | bf16 mixed-precision (AMP)       | WIP, current baseline (just assigned) |

**Merged:** 2 (mask-aware, Huber). **Closed:** 3 (Fourier, slice=128, surf_weight=25). **Open:** 8 (5 needing rebase + 3 fresh).

## Potential next research directions

Confirmed winners so far (both stack): correctness (mask) + loss formulation (Huber). Likely follow-ups:

- **If Huber β-tuning wins (#1712 askeladd):** sweep β around the optimum, consider per-channel β for surface p vs Ux vs Uy (different normalized scales).
- **If bf16 AMP wins (#1715 frieren):** the compute-bound hypotheses (slice=128, deeper=7, surf_weight=25) become reviewable again at the larger epoch budget. Re-open those as "AMP+X" combinations.
- **If grad_clip wins (#1692 fern):** explore weight decay tuning and learning-rate revisits, since clipping decouples optimizer stability from those.
- **If mlp_ratio=4 wins (#1623 alphonse):** try mlp_ratio=8 (transformer-recipe standard); also revisit n_hidden scaling now that FFN is wider.

Larger swings to queue for round 2 if the above plateau:
- **Surface-anchored cross-attention** (boundary nodes as queries against volume tokens) — directly addresses the "surface inherits from volume" structural relationship that frieren's surf_weight result highlighted.
- **Mirror data augmentation** (y-axis flip + Uy negation) — strong CFD physical prior, effectively 2× training data; orthogonal to all architectural/optimization changes.
- **Per-sample Re normalization** (Reynolds-number-aware feature embedding) — re_rand split is where Huber gave the largest gain, suggesting Re-dependent error structure.
- **Quantile / Pinball loss** — more aggressive median-targeting than Huber if β=0.25 still leaves gains on the table.
- **OneCycleLR / longer cosine warmup** — alternative schedule, may help if AMP enables more epochs.
- **Cruise data sanity check** (`scoring.py` cruise sample 20 inf-pressure) — mask currently hides the upstream data issue; worth a dedicated PR to either clean or document.

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
