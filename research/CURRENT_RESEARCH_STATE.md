# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-13
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** none received — this is a controlled 24/48h Charlie-vs-Willow logging ablation. Treated as research-isolated.

## Research focus

Round-1 baseline has shifted **three times in <8 hours** through stacking compatible winners:

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 baseline |
| PR #1505 (Huber β=0.5) | 2026-05-13 00:00 | 113.794 | 101.782 | −4.74% val, −7.19% test |
| **PR #1715 (bf16 AMP)** | **2026-05-13 02:00** | **89.597** | **79.907** | **−21.3% val, −21.5% test** |

The current `train.py` has all three stacked: mask after slice softmax (~line 125), Huber β=0.5 at train+eval (lines 260, 508), and bf16 autocast wrapping the forward in both eval (line 255) and train (line 506). Best epoch shifted 13 → 17 of 18 (still descending at termination).

**Pattern emerging from round 1:**
- **All three winners are orthogonal mechanisms** (correctness, loss formulation, compute). They stack additively — cumulative val 119.45 → 89.60 (−25%).
- **The "compute-bound undertraining" cluster is now alive again.** Four hypotheses (#1506 wider, #1507 slice=128, #1511 deeper=7, #1623 mlp_ratio=4) closed because they extended per-epoch cost below the wall-clock convergence horizon. On bf16's 18-epoch budget, those axes may be back in-play — flagged for round-2 priority.
- **The trajectory at epoch 17/18 of bf16 is still descending.** Compute remains the binding constraint at the 30-min cap — `torch.compile` (PR #1810, frieren) is the next compute-side lever.
- The `test_geom_camber_cruise` NaN issue is fully resolved by the mask fix; bf16 truncation × `1/(slice_norm + 1e-5)` did not re-introduce it.

Round 1 in-flight (8 PRs), most on stale baselines vs the current bf16 baseline:
- **#1506 edward (n_hidden=192)**, **#1511 thorfinn (n_layers=7)**: pre-mask baseline — need full rebase
- **#1589 tanjiro (AdamW betas)**: rebasing from pre-mask
- **#1692 fern (grad_clip=1.0)**: from mask-aware baseline, pre-Huber
- **#1735 alphonse (SwiGLU FFN)**: from Huber baseline, pre-bf16
- **#1810 frieren (torch.compile + bf16)**: current bf16 baseline
- **#1843 nezuko (Cosine T_max=18)**: current bf16 baseline
- **#1882 askeladd (Huber β=0.75)**: current bf16 baseline (just assigned, bounds β-axis from above)

All got bf16 heads-up where applicable. New merge bar: **val < 89.60, test < 79.91, all four test splits finite.**

**Latest diagnostic finding (2026-05-13 03:00 from PR #1509 close):** The cosine schedule `T_max=MAX_EPOCHS=50` mis-tunes the LR decay to a never-reached horizon. At the bf16 baseline's 18 epochs, end-of-run LR is at ~81% of peak (4.07e-4 vs 5e-4) — the schedule never actually decays. PR #1843 isolates this as a single-axis test.

**Portfolio constraint update 2026-05-13 02:00 (after #1715 merge):** The compute-bound axes (#1506 width, #1507 slices, #1511 depth, #1623 mlp_ratio) that closed earlier may be **back in-play** on the bf16 baseline (18-epoch budget vs 14). They are not auto-re-opened — flagged for round-2 priority queue after current round-1 PRs land. The portfolio rule from #1623 (capacity moves should change *what* is computed, not scale existing components) still applies as the default; bf16 simply opens a controlled exception for retest.

## Round 1 portfolio (status)

| PR    | Student   | Hypothesis axis                  | Status |
|-------|-----------|----------------------------------|--------|
| #1504 | alphonse  | Mask-aware PhysicsAttention      | **MERGED** 21:52 (val=119.45, test=109.67) |
| #1505 | askeladd  | Huber surface loss (β=0.5)       | **MERGED** 00:00 (val=113.79, test=101.78) |
| #1506 | edward    | Wider hidden (128→192)           | WIP, pre-mask code, bf16 heads-up posted |
| #1507 | fern      | More slices (64→128)             | CLOSED (compute-bound, +27%) — bf16-revisit candidate |
| #1508 | frieren   | surf_weight 10→25                | CLOSED (compute-bound, +16%) |
| #1509 | nezuko    | Warmup + lr=1e-3                 | CLOSED (+13.4% val on bf16; diagnostic surfaced T_max issue) |
| #1510 | tanjiro   | Fourier pos enc (L=6)            | CLOSED (cruise NaN, pre-mask) |
| #1511 | thorfinn  | Deeper (5→7 layers)              | WIP, pre-mask code, bf16 heads-up posted — bf16-revisit candidate |
| #1589 | tanjiro   | AdamW betas (0.9, 0.95)          | WIP, rebasing onto mask-aware, bf16 heads-up posted |
| #1623 | alphonse  | mlp_ratio 2→4                    | CLOSED (compute-bound, +18% val) — bf16-revisit candidate |
| #1692 | fern      | Gradient clipping (max_norm=1.0) | WIP, post-mask pre-Huber, bf16 heads-up posted |
| #1712 | askeladd  | Huber β=0.25 (β-tune)            | CLOSED (+6.6% val on bf16; bounds β from below) |
| #1715 | frieren   | bf16 mixed-precision (AMP)       | **MERGED** 02:00 (val=89.60, test=79.91) |
| #1735 | alphonse  | SwiGLU FFN (matched params)      | WIP, pre-bf16, bf16 heads-up posted |
| #1810 | frieren   | torch.compile (dynamic=True)     | WIP, current bf16 baseline |
| #1843 | nezuko    | Cosine T_max=18 (not 50)         | WIP, current bf16 baseline |
| #1882 | askeladd  | Huber β=0.75 (β-tune from above) | WIP, current bf16 baseline (just assigned) |

**Merged:** 3 (mask-aware, Huber, bf16). **Closed:** 6 (Fourier, slice=128, surf_weight=25, mlp_ratio=4, warmup+lr=1e-3, β=0.25 — 3 of the closed are bf16-revisit candidates). **Open:** 8 (4 needing rebase + 4 on current baseline).

## Potential next research directions

Confirmed winners so far (all three stack): correctness (mask) + loss formulation (Huber) + compute (bf16). Likely follow-ups:

- **If torch.compile wins (#1810 frieren):** further compute-side optimizations become low-priority (we'd be near the H100 ceiling). The bottleneck shifts to actual model quality — round-2 priorities pivot to architecture and OOD generalization.
- **If Huber β-tuning wins (#1712 askeladd):** sweep β around the optimum, consider per-channel β for surface p vs Ux vs Uy (different normalized scales).
- **If grad_clip wins (#1692 fern):** explore weight decay tuning and LR revisits, since clipping decouples optimizer stability from those.
- **If SwiGLU FFN wins (#1735 alphonse):** the gating mechanism's success would suggest other modern transformer-FFN moves are worth trying (e.g. GeGLU variant, larger gating dimension, per-block residual gating). Pairs well with bf16.
- **If width/warmup/depth/AdamW-betas land:** harvest the wins, stack them with bf16, and revisit compute-bound axes on the resulting baseline.

Round-2 priority queue (post-round-1-cleanup):

**Compute-bound revisits on bf16** (3 of the 4 closed PRs are candidates):
- **n_layers=7** (#1511 retry on bf16) — was the cleanest compute-bound regression, simplest revisit.
- **mlp_ratio=4** (#1623 retry on bf16) — alphonse's first attempt closed compute-bound; SwiGLU (#1735) is the alternative track; on bf16 both could be winners stacked together.
- **slice_num=128** (#1507 retry on bf16) — fern's compute-bound axis; could substantially increase model capacity if epoch budget supports it.

**Larger swings if round-1+round-2-revisits plateau:**
- **Surface-anchored cross-attention** (boundary nodes as queries against volume tokens) — directly addresses the "surface inherits from volume" structural relationship.
- **Mirror data augmentation** (y-axis flip + Uy negation) — strong CFD physical prior, effectively 2× training data; orthogonal to architectural/optimization changes. Note: requires careful per-feature flipping of ~9 signed columns in X.
- **Per-sample Re normalization** (Reynolds-aware feature embedding) — re_rand and cruise are where bf16 gave the largest test gains (−22.7% / −20.0%), suggesting Re-dependent error structure remains.
- **Quantile / Pinball loss** — more aggressive median-targeting than Huber if β-tune still leaves gains on the table.
- **OneCycleLR / longer cosine warmup** — alternative schedule for the now-18-epoch budget.
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
