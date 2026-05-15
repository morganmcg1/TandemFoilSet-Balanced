<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15 17:45
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 3 of the appendix-willow PAI2I sub-track)
- **Most recent human research direction:** None received this launch.

## Current focus

Round 3 cohort has its winner. **Frieren #3248 posted terminal SENPAI-RESULT at 17:39 UTC**
— `mp8s8okf` at `val_avg/mae_surf_p=107.4641`, `test_avg_nansafe/mae_surf_p=101.9848`. She
is the round-3 cohort leader, beating askeladd's warmup-cosine-grad-clip (109.99) by 2.3%.

6 of 8 cohort PRs are now terminal; 1 pending (alphonse #3282 bf16, with `tup20e60` at 111.6
visible in W&B but no terminal comment yet); 1 in flight (edward #3313 grad-accum).

**Merge plan (deferred to next invocation due to GH rate limit, reset 18:19 UTC):**
1. Merge frieren #3248 (huber-robust-loss, val 107.46) as round-3 winner.
2. Merge askeladd #3244 (warmup-cosine-grad-clip, val 109.99) as compound on top of
   frieren. Both touch orthogonal levers (loss function vs optimizer schedule) so they
   should stack cleanly.
3. Close losers (#3249 nezuko neutral, #3250 tanjiro, #3251 thorfinn, #3312 fern).
4. Nudge alphonse #3282 for terminal; assign round-4 stacking hypotheses.

Goal remains: minimize `val_avg/mae_surf_p` — equal-weight surface-pressure MAE across
the 4 validation tracks (in_dist, geom_camber_rc, geom_camber_cruise, re_rand). The
paper-facing metric is `test_avg_nansafe/mae_surf_p` (nansafe variant, per the
`data/scoring.py` bug below).

Baseline configuration for this branch (no prior merged PRs):

- Transolver: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- AdamW: `lr=5e-4`, `weight_decay=1e-4`
- Loss: MSE in normalized space, `surf_weight=10`, equal channel weights
- Optimizer schedule: `CosineAnnealingLR(T_max=epochs)` with no warmup
- `batch_size=4`, `epochs=50` (subject to `SENPAI_TIMEOUT_MINUTES`)

**Hard budget constraint:** `SENPAI_TIMEOUT_MINUTES=30` per run. With baseline L=5
~130–210 s/epoch, only ~11–14 epochs of the 50-epoch cosine schedule complete before
timeout — the LR barely anneals from its peak. This shapes every round-3 result.

**Fresh-slate baseline anchor:** edward's `7fa1s7vm` run (clean default config) hit
`val_avg/mae_surf_p = 129.99` at epoch 14/50. This is the reference for round-3 deltas.

## Round 3 cohort — terminal results (17:30 UTC)

5 of 8 cohort PRs have posted terminal `SENPAI-RESULT` comments. All values:
`val_avg/mae_surf_p` (lower better). Test column is `test_avg_nansafe/mae_surf_p`
(3-split mean, excluding cruise — bug below).

| Rank | Agent | PR | Hypothesis | val_avg | test_nansafe | Status |
|---|---|---|---|---:|---:|---|
| **1** | **frieren** | **#3248** | **huber-robust-loss (δ=2.0)** | **107.46** | **101.98** | ✓ **TERMINAL** — round-3 winner |
| 2 | askeladd | #3244 | warmup-cosine-grad-clip | **109.99** | 107.44 | ✓ terminal |
| 3 | alphonse | #3282 | bf16-mixed-precision | 111.6 | — | WIP (terminal pending; late-divergence noted) |
| 4 | fern | #3312 | lion-optimizer | **115.49** | **109.10** | ✓ terminal |
| 5 | thorfinn | #3251 | naca-camber-fourier-features | **123.35** | 123.25 | ✓ terminal |
| 6 | tanjiro | #3250 | re-conditioned-loss-weighting | **124.76** | 124.29 | ✓ terminal |
| 7 | nezuko | #3249 | ema-model-averaging (decay=0.999) | **130.18** | 127.12 | ✓ terminal — **neutral**, ≈baseline |
| 8 | edward | #3313 | grad-accum-eff-batch-16 | (in flight) | — | WIP (just started ~16:22 UTC) |

**Closed (round 3 dead-ends):**
- fern's earlier #3247 `larger-slice-num` (133.73 val, test NaN from model inf)
- edward's earlier #3245 `surf-p-weighted [1,1,3]` (135.66, +4.4% vs baseline)
- alphonse's earlier #3243 `deeper-transolver L=8` (147.85, undertrained 9 epochs)

## Round 3 merge plan (deferred to next invocation due to GH rate limit)

GitHub REST rate limit hit at 17:33 UTC; resets ~18:19 UTC. This invocation is read-only
on GitHub via GraphQL. Merges and label swaps deferred to next harness re-entry.

**Execution order on next invocation:**

1. **Merge PR #3248 frieren** (round-3 winner, val 107.46, test_nansafe 101.98).
   - Use `senpai:merge-winner` skill with args `3248 target/`.
   - First swap label `status:wip` → `status:review` if needed; PR is currently draft.
     Will need `gh pr ready 3248` before `senpai:merge-winner`.
   - After merge, advisor branch's `train.py` has Huber loss with `huber_delta=2.0`.
     BASELINE.md updates to `val_avg/mae_surf_p=107.46`,
     `test_avg_nansafe/mae_surf_p=101.98`.

2. **Merge PR #3244 askeladd** (val 109.99 on fresh-slate; compounds onto frieren).
   - Frieren's loss change is orthogonal to askeladd's schedule/grad-clip changes;
     conflict scope is narrow (train.py loss line vs optimizer setup block).
   - Resolve any minor conflict if it appears; the levers stack cleanly.
   - After merge, advisor branch has Huber + warmup-cosine + grad-clip. Update
     BASELINE.md again (but only if the post-merge val number is verified — we
     don't have a stacked-trained metric yet; mark as "stacked, awaiting confirmation").

3. **Close PR #3249 nezuko** (ema-model-averaging, val 130.18 ≈ baseline 129.99 = neutral).
   - EMA hypothesis failed at the 13-epoch budget; the EMA buffer barely averages
     over a steady-state model. Reassign nezuko to a different lever in round 4.

4. **Handle PRs #3312 fern, #3251 thorfinn, #3250 tanjiro**:
   - All three beat fresh-slate baseline (129.99) but lose to the round-3 winner
     on their old measurements.
   - Their levers (Lion, NACA-Fourier, re-cond-loss-weighting) are orthogonal to
     huber-loss + warmup-cosine-grad-clip and may compound. The clean test is
     round-4 stacking, not round-3 merging.
   - **Decision: close all three with explicit round-4 reassignment** that tests
     the lever on top of merged (frieren+askeladd) baseline. Each gets a fresh PR.

5. **Nudge alphonse #3282** to post terminal SENPAI-RESULT.
   - Alphonse's `tup20e60` has best-vs-final divergence (111.6 vs 171.4) — needs a
     follow-up arm with grad-clip or smaller late-LR to fix the late-cosine instability.

6. **Idle students after merge cycle:** frieren, askeladd, fern, nezuko (close),
   thorfinn (close), tanjiro (close). Edward (#3313) keeps running grad-accum.
   Assign round-4 stacking hypotheses (see plan below).

## Round 4 hypothesis plan (stacking experiments)

After the round-3 merges, the new baseline has Huber loss (frieren) + warmup-cosine-grad-clip
(askeladd) stacked. Round-4 experiments test orthogonal levers on top of this baseline.

| Student | New hypothesis | Slug | Rationale |
|---|---|---|---|
| askeladd | **T_max=epochs_realized + warmup_epochs=2 + grad_clip=5.0** | `warmup-tmax-fix` | His own follow-up: cosine actually anneals, less warmup overhead, less aggressive clip. Predicted ~−3 on val_avg. |
| fern | **Lion + new stacked baseline** | `lion-stacked` | Lion was +5 worse than warmup-cosine-grad-clip at fresh-slate; on stacked baseline it tests whether Lion's update geometry adds over schedule fixes. |
| nezuko | **Per-domain output normalization** | `per-domain-output-norm` | New lever; aligns Ux/Uy/p scales across raceCar (ground effect) vs cruise (freestream). Orthogonal to loss+schedule. |
| thorfinn | **NACA Fourier features stacked** | `naca-fourier-stacked` | Re-test thorfinn's lever on merged baseline; was +12% on fresh-slate. |
| tanjiro | **Re-conditioned loss weighting stacked** | `re-cond-stacked` | Re-test tanjiro's lever on merged baseline (combine with Huber, not replace it). |
| edward | (wait for grad-accum result) | TBD | Round-3 still in flight; round-4 depends on outcome. |
| frieren | **Surface-only Huber + tighter δ=1.0** | `huber-surface-only` | Her own suggested follow-up: keep MSE on volume, Huber on surface, tighter δ. Predicted to extend her lead since surface is what the primary metric uses. |
| alphonse | **bf16 + grad_clip late-divergence fix** | `bf16-stable` | His own follow-up: bf16 + grad_clip + LR floor to fix the best-vs-final divergence. |

These are 8 round-4 assignments — one per student — to be issued sequentially after the
merge cycle completes.

### Cohort signal (refined 17:40)

Round 3 confirms three orthogonal levers attack the binding constraint (30-min cap +
high-noise early training):
- **Loss-side stability** (frieren's Huber δ=2.0) — caps gradient magnitude on outliers
- **Schedule-side stability** (askeladd's warmup + cosine + grad-clip)
- **Throughput** (alphonse's bf16 — same baseline config, more epochs in 30 min)

All three are likely **complementary**, not redundant. Round 4's stacking experiments
will test this hypothesis directly. Lion (fern) extends the schedule-side stability
menu by giving uniform per-parameter steps regardless of gradient magnitude — should
compose nicely with warmup.

Middle-tier round-3 results (NACA-Fourier 123, re-cond-loss 125, EMA 130) are weaker
because they didn't address the binding constraint. Their value is conditional: each
might compound nicely once the training-stability foundation is fixed.

## Known infra bugs

### 1. `data/scoring.py` NaN propagation (read-only file, project-wide)

- `test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]` (interior
  pressure nodes; **no** surface node affected).
- `accumulate_batch` masks the sample but computes `err = (pred - y).abs()` first;
  `NaN * 0 = NaN` in IEEE-754, NaN propagates into the per-channel accumulator,
  poisoning the entire split's surface metric.
- Net effect: in-tree `test_avg/mae_surf_p` is `None`/`NaN` for every submission.
- **Workaround:** every PR logs `test/<split>/mae_surf_p_nansafe` (finite-only) and
  `test_avg_nansafe/mae_surf_p` (3-split mean). Nansafe is the paper-facing number.
- Identified by alphonse on PR #3243.

### 2. PhysicsAttention numerical instability at `slice_num=128` (fern's finding)

- At `slice_num=128`, the model produces `±inf` pressure predictions on at least one
  `test_geom_camber_cruise` sample (reproducible across runs).
- Cruise val is fine (104.24), but cruise test → `vol_loss=inf`, `mae_surf_p=NaN`.
- Likely causes: softmax temperature decaying near zero, slice_norm underflow at S=128,
  attention softmax accumulator overflow.
- **Not currently fixed** — future `slice_num` arms must pair with a stability guard
  (fp32-stable softmax in slice projection, output logit clamp, or norm floor).
- Identified by fern on PR #3247.

### 3. Late-training divergence with bf16 (alphonse's finding)

- alphonse's `tup20e60` (bf16, L=5, default schedule) shows best_val=111.6 at epoch ~9
  but final_val=171.4 at epoch ~17 — model state diverges late in training.
- Best-checkpoint saving covers paper-facing metric, but signals bf16 needs additional
  stability (grad-clip or LR floor or fp32-fallback near end).
- Will be addressed in alphonse's round-4 `bf16-stable` hypothesis.

## Operational observations from round 3

1. **Time budget is binding.** Every L=5 run hits 11–14 epochs in 30 min; cosine
   `T_max=50` means the LR schedule never anneals. Throughput levers (bf16,
   torch.compile, grad-accum) are first-class research targets.
2. **Cosine T_max should match completed-epoch count.** Future variants should set
   `T_max=epochs_estimate` so the schedule actually exercises its annealing tail.
3. **Multi-arm per PR is the norm.** Students naturally launched 2–4 W&B runs per
   hypothesis. Works as intended; the cohort decision waits on terminal `SENPAI-RESULT`.
4. **In-tree test_avg is unusable** until/unless `data/scoring.py` is patched. Nansafe
   is the comparison number; per-split test is more informative than the broken aggregate.
5. **GitHub REST rate limit hit at 17:33 UTC** (5000/hr exhausted). Use GraphQL for
   reads when this happens; defer writes (close/comment/merge/label) to next invocation.

## Operational notes

- All assignments use `--wandb_group` matching the hypothesis slug so iterations cluster
  in W&B.
- Every PR body asks students to log NaN-safe test metrics in addition to the in-tree
  scorer's output. Nansafe is the paper-facing comparison.
- Hard limits: `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` govern each training run.
  Do not override.
- All 8 student GPUs allocated; zero idle (pending review for askeladd, fern who'll
  be reassigned next invocation).

## Active PRs (17:40 UTC)

| PR | Student | Status | Action next invocation |
|---|---|---|---|
| #3248 | frieren | wip (terminal posted) | **MERGE 1st** (round-3 winner, val 107.46) — `gh pr ready 3248` then `senpai:merge-winner 3248 target/` |
| #3244 | askeladd | review | **MERGE 2nd** (compound onto frieren, val 109.99) |
| #3249 | nezuko | review | CLOSE (neutral, +0.14% vs baseline) → reassign `per-domain-output-norm` |
| #3250 | tanjiro | review | CLOSE → reassign `re-cond-stacked` |
| #3251 | thorfinn | review | CLOSE → reassign `naca-fourier-stacked` |
| #3282 | alphonse | wip | nudge for terminal; round-4 reassign to `bf16-stable` |
| #3312 | fern | review | CLOSE → reassign `lion-stacked` |
| #3313 | edward | wip | leave running |
