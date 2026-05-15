<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15 15:50
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 3 of the appendix-willow PAI2I sub-track)
- **Most recent human research direction:** None received this launch.

## Current focus

Round 3 fresh-slate exploration on TandemFoilSet (Transolver surrogate for tandem-airfoil
CFD). Goal: minimize `val_avg/mae_surf_p` — equal-weight surface-pressure MAE across the
four validation tracks (in_dist, geom_camber_rc, geom_camber_cruise, re_rand). The
test-time decision metric is `test_avg/mae_surf_p` (nansafe variant, per the
`data/scoring.py` bug below).

Baseline configuration for this branch (no prior merged PRs):

- Transolver: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- AdamW: `lr=5e-4`, `weight_decay=1e-4`
- Loss: MSE in normalized space, `surf_weight=10`, equal channel weights
- Optimizer schedule: `CosineAnnealingLR(T_max=epochs)` with no warmup
- `batch_size=4`, `epochs=50` (subject to `SENPAI_TIMEOUT_MINUTES`)

**Hard budget constraint:** `SENPAI_TIMEOUT_MINUTES=30` per run. With baseline L=5 ~160–210
s/epoch, only ~11–14 epochs of the 50-epoch cosine schedule complete before timeout —
the LR barely anneals from its peak. This shapes every round-3 result.

**Fresh-slate baseline anchor:** edward's `7fa1s7vm` run (clean default config) hit
`val_avg/mae_surf_p = 129.99` at epoch 14/50. This is the reference for round-3 deltas.

## Round 3 cohort — interim standings (15:50)

W&B-sourced ranking of finished runs (in-tree test_avg = NaN for all; per-split test
available for 3 of 4 splits). All values are `val_avg/mae_surf_p` (lower better):

| Rank | Agent | PR | Hypothesis | Best val | Status |
|---|---|---|---|---:|---|
| 1 | askeladd | #3244 | warmup-cosine-grad-clip | **109.99** | WIP (4th arm running) |
| 2 | frieren | #3248 | huber-robust-loss (δ=2.0) | 124.66 | WIP (3rd arm running) |
| 3 | tanjiro | #3250 | re-conditioned-loss-weighting | 125.07 | WIP (3rd arm running) |
| — | edward | (closed #3245) | baseline anchor | **129.99** | reference only |
| 4 | nezuko | #3249 | ema-model-averaging (decay=0.999) | 130.17 | WIP (3rd arm running) |
| 5 | thorfinn | #3251 | naca-camber-fourier-features | 138.36 | WIP (3rd arm running) |
| closed | fern | (closed #3247) | larger-slice-num (S=128) | 133.73 | test NaN, model inf |
| closed | edward | (closed #3245) | surf-p-weighted [1,1,3] | 135.66 | +4.4% vs baseline |
| closed | alphonse | (closed #3243) | deeper-transolver (L=8) | 147.85 | undertrained |

**New WIP assignments (15:50):**
- fern → #3312 `lion-optimizer` (Lion at lr=1.5e-4 wd=3e-4)
- edward → #3313 `grad-accum-eff-batch-16` (accum_steps=4, eff batch 16)

**Holding:** alphonse on #3282 `bf16-mixed-precision` (debug nudge sent — `1t41l8sx`
crashed at 0.1 min; awaiting next run).

### Cohort signal (preliminary)

Training-stability changes dominate so far. askeladd's warmup-cosine-grad-clip leads by
~14 over the next tier (huber loss, re-conditioned loss weighting), and all three of those
beat the fresh-slate baseline (129.99). The lower tiers tried EMA, NACA-Fourier features,
larger slice_num, and L=8 — all of which were neutral-to-worse.

**Hypothesis-of-the-hypothesis:** the round-3 winners are eating the same lever (training
stability at the start of the cosine ramp). Adding warmup gives the optimizer 1–2 free
epochs at sane gradient scale; grad-clip prevents the rare spike from poisoning the rest
of training. Huber and re-cond loss weighting are doing a softer version of the same
thing by capping per-sample gradient magnitude.

### Decision principle for round 3

No PR is merged until the cohort is fully terminal (5 WIP students need to post terminal
`SENPAI-RESULT` comments with final arm complete). Once cohort is in, rank by
`val_avg/mae_surf_p` and **merge the strongest first** (likely askeladd's PR #3244),
then assess what stacks orthogonally for round 4.

## Known infra bugs

### 1. `data/scoring.py` NaN propagation (read-only file, project-wide)

- `test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]` (interior
  pressure nodes; **no** surface node affected).
- `accumulate_batch` masks the sample but computes `err = (pred - y).abs()` first;
  `NaN * 0 = NaN` in IEEE-754, NaN propagates into the per-channel accumulator,
  poisoning the entire split's surface metric.
- Net effect: in-tree `test_avg/mae_surf_p` is `None`/`NaN` for every submission on
  this branch.
- **Workaround:** every PR must log `test/<split>/mae_surf_p_nansafe` (filtered
  finite-only) and `test_avg_nansafe/mae_surf_p`. Nansafe is the paper-facing number.
- Identified by alphonse on PR #3243.

### 2. PhysicsAttention numerical instability at `slice_num=128` (new — fern)

- At `slice_num=128`, the model produces `±inf` pressure predictions on at least one
  `test_geom_camber_cruise` sample (reproducible across runs `pf6dwz1f`, `kcpsgrot`).
- Cruise val is fine (104.24), but cruise test → `vol_loss=inf` and `mae_surf_p=NaN`.
- Likely causes (from fern's analysis): (a) softmax temperature decaying near zero in
  PhysicsAttention, (b) `slice_token / (slice_norm + 1e-5)` underflow with small slice
  norms at S=128, (c) attention softmax accumulator overflow.
- **Not currently being fixed** — future `slice_num` arms must pair with a stability
  guard (fp32-stable softmax in slice projection, output logit clamp, or norm floor).
- Identified by fern on PR #3247.

## Operational observations from round 3

1. **Time budget is binding.** Every L=5 run hits 11–14 epochs in 30 min; cosine `T_max=50`
   means the LR schedule never anneals. Throughput levers (bf16, torch.compile) are
   first-class research targets.
2. **Cosine T_max should match completed-epoch count, not the nominal 50.** Future
   variants should set `T_max=epochs_estimate` so the schedule actually exercises its
   annealing tail.
3. **Multi-arm per PR is the norm.** Students naturally launched 2–4 W&B runs per
   hypothesis to sweep within the assigned lever. Working as intended; the cohort decision
   waits on terminal `SENPAI-RESULT` per PR.
4. **In-tree test_avg is unusable** until/unless `data/scoring.py` is patched. Nansafe is
   the comparison number; per-split test is more informative than the broken aggregate.

## Next research directions (after round 3 cohort completes)

Conditional on results coming in, plausible follow-ups (ordered by expected impact):

1. **Merge askeladd's warmup-cosine-grad-clip** as new baseline (assuming it terminates
   above frontier 109.99). Then revisit which losers regress less against the new base.
2. **Throughput levers.** bf16 (alphonse PR #3282, debugging), torch.compile, eventual
   gradient checkpointing for larger effective batch. Doubling throughput unlocks the
   undertrained hypotheses (L=8, larger slice_num with stability guard, longer-cosine).
3. **T_max correction.** Set cosine `T_max` to estimated completed epochs so LR actually
   anneals. Could be folded into the winner's confirmation arm.
4. **Stack winners.** Once askeladd lands, try Lion + warmup (fern's lion-optimizer
   stacks naturally with warmup), or grad-accum + warmup (edward + askeladd).
5. **PhysicsAttention stability fix.** fp32-stable softmax in slice projection — unlocks
   `slice_num=128/192/256` work blocked by the model-side inf bug.
6. **Larger attention bottleneck (post-fix).** If slice_num scales smoothly after the
   stability fix, try `slice_num=192/256` at `batch_size=2`.
7. **Targeted geometry conditioning.** If `naca-camber-fourier-features` terminal results
   show OOD gains on geom_camber splits, extend to gap/stagger/AoA features.
8. **Physics-residual losses.** Soft divergence-free penalty on predicted velocity and
   inviscid-Bernoulli surface-pressure consistency — particularly for OOD camber splits.
9. **Surface-aware decoder.** Dedicated surface-decoder head that takes `is_surface` mask
   as conditioning. Targets the primary metric directly.
10. **Per-domain or per-Re-bucket normalization.** Align value magnitudes across the
    heterogeneous dataset (raceCar ground-effect vs cruise freestream; Re spanning 110K–5M).

## Operational notes

- All assignments use `--wandb_group` matching the hypothesis slug so iterations cluster
  in W&B.
- Every PR body asks students to log NaN-safe test metrics in addition to the in-tree
  scorer's output. Nansafe is the paper-facing comparison.
- Hard limits: `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` govern each training run.
  Do not override.
- Active PRs (status:wip):
  - #3244 askeladd warmup-cosine-grad-clip
  - #3248 frieren huber-robust-loss
  - #3249 nezuko ema-model-averaging
  - #3250 tanjiro re-conditioned-loss-weighting
  - #3251 thorfinn naca-camber-fourier-features
  - #3282 alphonse bf16-mixed-precision (debug)
  - #3312 fern lion-optimizer (new)
  - #3313 edward grad-accum-eff-batch-16 (new)
- All 8 student GPUs allocated; zero idle.
