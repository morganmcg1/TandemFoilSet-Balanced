<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 3 of the appendix-willow PAI2I sub-track)
- **Most recent human research direction:** None received this launch.

## Current focus

Round 3 fresh-slate exploration on TandemFoilSet (Transolver surrogate for tandem-airfoil
CFD). Goal: minimize `val_avg/mae_surf_p` — equal-weight surface-pressure MAE across the
four validation tracks (in_dist, geom_camber_rc, geom_camber_cruise, re_rand). The
test-time decision metric is `test_avg/mae_surf_p`.

Baseline configuration for this branch (no prior merged PRs):

- Transolver: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- AdamW: `lr=5e-4`, `weight_decay=1e-4`
- Loss: MSE in normalized space, `surf_weight=10`, equal channel weights
- Optimizer schedule: `CosineAnnealingLR(T_max=epochs)` with no warmup
- `batch_size=4`, `epochs=50` (subject to `SENPAI_TIMEOUT_MINUTES`)

**Hard budget constraint:** `SENPAI_TIMEOUT_MINUTES=30` per run. With baseline L=5 ~160–210
s/epoch, only ~9–14 epochs of the 50-epoch cosine schedule complete before timeout —
the LR barely anneals from its peak. This shapes every round-3 result.

## Round 3 dispatch + status

| PR | Student | Hypothesis | Status | val_avg/mae_surf_p | test_avg/mae_surf_p (nansafe) |
|----|---------|------------|--------|--------------------|------------------------------|
| #3243 | alphonse | deeper-transolver (n_layers 5→8) | **review (hold)** | **147.85** (E9/9, timeout) | 138.60 |
| #3244 | askeladd | warmup-cosine-grad-clip | WIP | — | — |
| #3245 | edward | surf-p-weighted-loss ([1,1,3]) | WIP | — | — |
| #3247 | fern | larger-slice-num (64→128) | WIP | — | — |
| #3248 | frieren | huber-robust-loss (δ=2.0) | WIP | — | — |
| #3249 | nezuko | ema-model-averaging (decay=0.999) | WIP | — | — |
| #3250 | tanjiro | re-conditioned-loss-weighting | WIP | — | — |
| #3251 | thorfinn | naca-camber-fourier-features | WIP | — | — |
| #3282 | alphonse | bf16-mixed-precision (L=5 default + bf16) | WIP | — | — |

Full hypothesis specifications in `RESEARCH_IDEAS_2026-05-15_initial.md`.

### Decision principle for round 3

No PR is merged until the cohort is in. We then rank all completed PRs by
`val_avg/mae_surf_p` and merge the strongest. Holding PR #3243 in review for this
reason — it is the only result so far and was severely time-limited (9 of 50 epochs,
cosine T_max=50 means LR effectively constant). The 7 L=5 in-flight PRs will likely
fit ~14 epochs each in the same 30-min budget; head-to-head will be informative.

## Known infra bug — data/scoring.py NaN propagation

PR #3243 (alphonse) discovered and characterized a NaN-propagation bug in `data/scoring.py`
(read-only file):

- `test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]` (interior
  pressure nodes; **no** surface node is affected).
- `accumulate_batch` masks the sample but computes `err = (pred - y).abs()` first, so
  `err` is `NaN` at those nodes; `NaN * 0 = NaN` in IEEE-754, and the NaN propagates
  into the per-channel accumulator, poisoning the entire split's surface metric.
- Net effect: in-tree `test_avg/mae_surf_p` is reported as `None`/`NaN` for every
  submission on this branch.
- **Workaround adopted:** every student will additionally log a NaN-safe variant:
  `test/<split>/mae_surf_p_nansafe` (filtered finite-only) and `test_avg_nansafe/mae_surf_p`.
  The W&B run summary should also carry `data_bug/cruise_idx20_p_neginf_*` diagnostics.
- The nansafe variant is the paper-facing comparison number for this round until/unless
  `data/scoring.py` itself is patched by the human team.

## Operational observations from round 3 so far

1. **Time budget is binding.** PR #3243 ran out of clock at 30.89 min/E9. Every student
   in this round is likely to be cut by the same cap. Future hypotheses should consider
   throughput (bf16, torch.compile, mixed precision) as first-class levers.
2. **Cosine T_max=epochs is too long for the actual run.** With epochs=50 in the config
   but only ~9–14 actually completing, the LR schedule's annealing tail is never
   exercised. A future bug-fix or assignment should set `T_max` to an estimate of
   completed-epoch count, so the cosine actually anneals.
3. **No GPU/system metrics in the W&B summary.** Peak VRAM was not logged automatically;
   future assignments should request it explicitly in the results table.

## Next research directions (after round 3 cohort completes)

Conditional on results coming in, plausible follow-ups (ordered by expected impact):

1. **Throughput levers.** bf16 (alphonse's PR #3282 just dispatched), `torch.compile`,
   eventual fp16 + GradScaler if bf16 wins. Doubling throughput directly addresses the
   binding constraint that distorted PR #3243.
2. **T_max correction.** Set cosine `T_max` to estimated completed epochs so the LR
   actually anneals. Could be folded into the eventual winner's confirmation arm.
3. **Stack winners.** Combine top-2 or top-3 round-3 winners (likely from different
   tiers — e.g., loss alignment + EMA + bf16) into a confirmation run.
4. **Larger attention bottleneck.** If `larger-slice-num` wins, try `slice_num=192` and
   investigate per-domain slice token specialization.
5. **Targeted geometry conditioning.** If `naca-camber-fourier-features` shows OOD gains,
   extend to gap/stagger/AoA features and consider per-foil geometry conditioning tokens.
6. **Physics-residual losses.** If geometry generalization remains weak, add a soft
   divergence-free penalty on predicted velocity and an inviscid-Bernoulli surface
   pressure consistency term.
7. **Surface-aware architecture.** Surface-only token attention or a dedicated
   surface-decoder head that takes `is_surface` mask as a conditioning input.
8. **Per-channel normalization beyond global stats.** Per-domain or per-Re-bucket
   normalization to align value magnitudes across the dataset.
9. **Optimizers beyond AdamW.** Lion, SOAP, Muon — better effective lr-scaling under
   heterogeneous gradient regimes.
10. **Re-evaluate L=8.** Once bf16 unlocks ~2x epochs, re-test depth hypothesis with
    the proper number of epochs and a sensible T_max.

## Operational notes

- All assignments use `--wandb_group` matching the hypothesis slug so iterations cluster
  in W&B.
- Going forward, every PR body asks students to log NaN-safe test metrics in addition to
  the in-tree scorer's output. The nansafe number is the paper-facing comparison.
- Hard limits: `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` govern each training run.
  Do not override.
