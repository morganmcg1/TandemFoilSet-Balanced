<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15 16:30
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

## Round 3 cohort — interim standings (16:30)

W&B-sourced ranking of finished runs. All values are `val_avg/mae_surf_p` (lower better).
Each cohort student has 3-4 finished arms; each just launched a *new* arm (2-5 min in)
and none have posted terminal SENPAI-RESULT yet. Nudge comments sent at 16:30 asking
each to post terminal SENPAI-RESULT once their current arm completes.

| Rank | Agent | PR | Hypothesis | Best val | Status |
|---|---|---|---|---:|---|
| 1 | **frieren** | #3248 | **huber-robust-loss (δ=2.0)** | **107.5** | WIP (4th arm running) |
| 2 | askeladd | #3244 | warmup-cosine-grad-clip | 110.0 | WIP (4th arm running) |
| 3 | **alphonse** | #3282 | **bf16-mixed-precision** | **111.6** | WIP (2nd arm running) |
| — | edward | (closed #3245) | baseline anchor | 129.99 | reference only |
| 4 | thorfinn | #3251 | naca-camber-fourier-features | 123.3 | WIP (4th arm running) |
| 5 | tanjiro | #3250 | re-conditioned-loss-weighting | 124.8 | WIP (4th arm running) |
| 6 | nezuko | #3249 | ema-model-averaging (decay=0.999) | 130.2 | WIP (4th arm running) |
| closed | fern | (closed #3247) | larger-slice-num (S=128) | 133.73 | test NaN, model inf |
| closed | edward | (closed #3245) | surf-p-weighted [1,1,3] | 135.66 | +4.4% vs baseline |
| closed | alphonse | (closed #3243) | deeper-transolver (L=8) | 147.85 | undertrained |

**Key shift since 15:50:**
- **frieren `mp8s8okf` displaced askeladd as cohort leader at 107.5** (was 124.66 before).
- **alphonse's bf16 is now working cleanly** — `tup20e60` at best_val=111.6 (full bf16 run).
  Note: `1t41l8sx` earlier was a clean debug run, not a crash — my misdiagnosis (corrected
  on PR #3282). bf16 at L=5 trains cleanly; the L=8 revisit is now viable.
- alphonse's `tup20e60` has best_val=111.6 vs final_val=171.4 — late-cosine divergence,
  best checkpoint covers it for paper-facing but signals bf16+grad-clip stacks could help.
- thorfinn jumped from 138 → 123.3 with `n2i46t6r` — what config? need terminal result.

**New WIP assignments (15:50):**
- fern → #3312 `lion-optimizer` (Lion at lr=1.5e-4 wd=3e-4) — picked up iter 38 at 16:20
- edward → #3313 `grad-accum-eff-batch-16` (accum_steps=4, eff batch 16) — picked up iter 38 at 16:22

### Cohort signal (updated 16:30)

Top-3 are tightly bunched (107.5, 110.0, 111.6) and reflect three independent levers:
- **Loss-side stability** (frieren's Huber δ=2.0 — caps gradient magnitude on outlier samples)
- **Schedule-side stability** (askeladd's warmup + cosine + grad-clip — sane LR ramp + clip)
- **Throughput** (alphonse's bf16 — same baseline config but ~17 epochs in 30 min vs 14)

All three attack the **same root cause** from different angles: the 30-min cap forces
training to live in the high-LR, high-gradient-noise early regime where outliers dominate.
Huber dampens outlier loss; warmup gives a cleaner ramp; bf16 buys more steps to escape
the noisy regime. This is consistent with the lower-tier results (EMA, NACA-Fourier, larger
slice_num, L=8) all being moderate — they tried to add capacity or features *to* a noisy
training regime that hadn't been fixed.

**Implication for round 4:** the top-3 levers are likely **complementary**, not redundant.
A round-4 confirmation run that stacks Huber + warmup-cosine-grad-clip + bf16 should
plausibly clear 100 if the levers are orthogonal. Lion (fern, in flight) and grad-accum
(edward, in flight) attack the same family (optimizer + effective-batch stability) — if
they show competitive results, they extend the menu.

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
