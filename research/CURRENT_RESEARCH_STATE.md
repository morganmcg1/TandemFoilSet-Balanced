# SENPAI Research State

- 2026-04-28 09:15 — round 2 nearing completion, round 3 starting;
  **5 merges (compile is the round-2 winner)** + 13 closes; 8 axes in flight
  including the first round-3 capacity-axis retry (#660); all hands on deck
- **Baseline updated (5 merges total):** PR #328 + #330 + #367 + #399 + **#553
  (torch.compile, the round-2 winner)**.
  Anchor val_avg/mae_surf_p = **115.61** (fp32 anchor, run `uip4q05z`).
  Operative baseline: **compile+bf16+Huber 3-seed mean = 80.70 (σ=2.20)**,
  test_avg = **71.45 (σ=2.88)**. **−28 % vs bf16+Huber baseline (~14σ
  outside noise)**, 2.23× speedup, 18 GB memory freed. Default config now:
  `n_hidden=128 n_layers=5 n_head=4 slice_num=128 mlp_ratio=2 lr=5e-4
  weight_decay=1e-4 batch_size=4 surf_weight=10.0 epochs=50` +
  `loss=Huber(beta=1.0)` + `amp_dtype=bf16` + **`compile_mode=default`** +
  `nan_to_num` scoring guard. **All round-2 PRs now compare against 80.70**.
  Wall-clock cap allows ~29 epochs (vs 13 at bf16-only, vs 11 at fp32-eager).
- **Round 2 candidates in flight:**
  - **PR #415** (frieren, asinh on pressure target): NEW
    assignment. Pairs with merged Huber β=1 — orthogonal
    mechanism on the same high-Re-tail failure mode.
  - **PR #399** (askeladd, bf16 mixed precision): throughput unlock
    for the 30-min cap.
- **Pending round-1 merge candidates** (need on-baseline
  confirmation; both pre-#330):
  - **PR #311** (alphonse, width-160): 126.18 on slice_num=64 (now
    11.7 % WORSE than the new 115.61 baseline — needs to land below
    115.61 to be a winner, well outside the ±10 % noise band).
  - PR #332 (nezuko, surf_weight=25): **CLOSED** this cycle —
    multi-seed mean 151.94 on slice-128 was 18 points above
    baseline, ~11 SE outside noise band; surf_weight axis doesn't
    stack with slice_num=128.
- **Bug fix landed:** PR #367 (fern, scoring NaN `nan_to_num`)
  merged. `test_avg/mae_surf_p` is finite on the next run of every
  PR — paper-facing metric finally works.
- **Noise floor: ±10 % at single seed.** Thorfinn (PR #337) ran the same
  config twice (kon60q79=153.19, nphltrz9=139.39) — a ~10 % spread
  purely from random seed / data-order randomness. **This is a
  methodological constraint affecting every PR's number on this
  branch.** Single-seed deltas <10 % vs baseline (133.55) are not
  statistically distinguishable; merge decisions on borderline PRs
  should require multi-seed replication. Frieren's 109.47 sits outside
  this noise band, so it remains a real signal.
- Primary metric: `val_avg/mae_surf_p`. Paper-facing:
  `test_avg/mae_surf_p` (currently NaN — bug-fix PR #367 in flight).
- W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2`.
- Wall-clock cap per run: `SENPAI_TIMEOUT_MINUTES=30`. Empirically this
  binds at **9–14 epochs** for round-1 configurations, not the planned
  50 — every finished run is undertrained.

## Round 1 status (after this cycle's actions)

| PR  | Student   | Axis                       | Status (now)   | Best val_avg/mae_surf_p |
|-----|-----------|----------------------------|----------------|-------------------------|
| 311 | alphonse  | width 128 → 192 → 160       | **closed** (3-seed mean 126.76 = +9.6 %, all 3 seeds above 115.61 baseline; per-split: cruise improves, in-dist + camber-rc regress — width-axis exhausted at 30-min cap) | superseded |
| 485 | alphonse  | round 2: RMSNorm           | **closed** (3-seed mean 127.74 = +10.5 % vs 115.61, slightly outside band; throughput delta -2.3 % below 3 % bar; PyTorch fused LayerNorm already near-optimal at this scale; 6th per-distribution-shift instance) | superseded |
| 568 | alphonse  | round 2: fp16 with scoped autocast + GradScaler | **closed** (catastrophic divergence on all 3 seeds; +0.2 % slower than bf16 — bf16 captured all AMP throughput at our scale) | superseded |
| 648 | alphonse  | round 2: LayerScale (learned residual gain) | NEW assignment (status:wip) | n/a |
| 325 | askeladd  | depth 5 → 8                | **closed** (21 % regression at 30-min cap) | 150.06 / 162.05 (two seeds) |
| 399 | askeladd  | round 2: bf16 mixed precision | NEW assignment (status:wip) | n/a |
| 326 | edward    | mlp_ratio 2 → 4            | **closed** (FFN axis exhausted; 21 % worse than new baseline) | mlp-3=139.79, mlp-2-control=136.54 (slice-128, MSE; monotone trend smaller-is-better) |
| 429 | edward    | round 2: per-channel loss weighting on `p` | **closed** (3-seed mean 118.65 = +2.6 % vs 115.61 baseline; seed=0=112.07 was a lucky pull from σ=6.24 noise distribution; mechanism: Huber's gradient clipping already absorbs the channel-weighting effect) | superseded |
| 547 | edward    | round 2: layer-wise learning rate decay (LLRD) | NEW assignment (status:wip) | n/a |
| 328 | fern      | slice_num 64 → 128         | **MERGED ★**   | **133.55 (new baseline)**|
| 330 | frieren   | MSE → Huber β=1            | **MERGED ★ (new baseline 115.61)** | rebased run = 115.61 (slice-128, epoch 11/50, run uip4q05z) |
| 399 | askeladd  | round 2: bf16 mixed precision | **MERGED ★ infrastructure** | 3-seed bf16+Huber mean 112.13 (σ=4.53) on slice-128 — at-baseline within new noise floor; throughput unlock (1.23× / +30 % epochs); first finite test_avg=101.82 |
| 553 | askeladd  | round 2: torch.compile on top of bf16 | **MERGED ★ metric+infrastructure (round-2 winner)** | 3-seed mean 80.70 (σ=2.20); 2.23× speedup; +123 % epochs in budget; 18 GB memory freed; σ tightens 4.53 → 2.20 |
| 660 | askeladd  | round 3: depth-8 on top of compile+bf16 (capacity retry) | NEW assignment (status:wip) | n/a — first round-3 capacity-axis re-validation |
| 415 | frieren   | round 2: asinh on pressure target | NEW assignment | n/a (assigned this cycle) |
| 332 | nezuko    | surf_weight 10 → 25 (sweep)| **closed** (3-seed mean 151.94 on slice-128 = +13.8 %, ~11 SE outside noise; vol_p also up; val curve oscillation) | superseded |
| 472 | nezuko    | round 2: Lion optimizer    | NEW assignment (status:wip) | n/a |
| 335 | tanjiro   | warmup + cos, peak 1e-3    | **closed** (3-seed mean 122.38 on slice-128+Huber = +5.85 % vs 115.61 baseline; schedule axis doesn't stack with Huber's implicit gradient shaping) | superseded |
| 517 | tanjiro   | round 2: stochastic depth (DropPath) | **closed** (3-seed mean 122.61 ± 2.95 — statistically null vs in-PR dp=0.0 control 122.93; throughput regressed 3.5 %; depth=5 too shallow per literature) | superseded |
| 649 | tanjiro   | round 2: auxiliary smoothness loss (physics-informed) | NEW assignment (status:wip) | n/a |
| 337 | thorfinn  | BS 4→8, lr 7e-4            | **closed** (hardware-blocked at BS≥8 on slice-128; BS=6 3-seed mean 162.63 is 41 % worse than 115.61) | superseded |
| 457 | thorfinn  | round 2: EMA weight averaging | NEW assignment (status:wip) | n/a |
| 367 | fern      | bug fix: cruise-NaN scoring| **MERGED ★** | rebased + verified on Huber baseline: test_avg: NaN → 117.59, cruise_p: NaN → 96.92 |
| 452 | fern      | round 2: push slice_num to 192/256 | **closed** (slice-192 single-seed = 133.30, +15.3 % vs 115.61 baseline; per-split mechanism-inversion confirms slice-128 is the ceiling) | superseded |
| 478 | fern      | round 2: curriculum by per-sample y-std | **closed** (W=3 single-seed = 116.90 = +1.1 % vs baseline; mechanism: global y_std quantile filtering inadvertently flips domain balance and amplifies Huber's implicit cruise-favoring) | superseded |
| 559 | fern      | round 2: per-sample loss weight ∝ y_std (counter-balance Huber) | **closed** (mechanism falsified: α=0.5 3-seed mean 136.48 ± 6.35 = +21.7 % vs baseline; α=1.0 catastrophic; predicted-to-improve splits regressed most) | superseded |
| 659 | fern      | round 2: per-domain stratified data weighting | NEW assignment (status:wip) | n/a — fern's #559-closing follow-up #1 |

PRs surfaced for advisor review this cycle: **#553, #559**. Actions:
**#553 (askeladd torch.compile) MERGED ★ as round-2 winner.**
3-seed mean val_avg = 80.70 (σ=2.20), test_avg = 71.45 — **−28 %
vs bf16+Huber baseline (~14σ outside noise)**. 2.23× speedup
(63s/epoch vs 141s), 29 epochs in budget vs 13, 18 GB memory freed,
σ tightens 4.53 → 2.20. Mechanism: small model + small batch → Python
dispatch dominates → graph capture eliminates it; plus inductor's
aggressive Blackwell kernel fusion; plus PhysicsAttention's small-
tensor pointwise structure is exactly what `default` mode optimizes
hardest. **Biggest single-axis win since Huber #330** (-13.4 % at
the time of merge). **All round-2 PRs now compare against 80.70**.
**Reassigned askeladd to #660 (depth-8 retry on top of compile)** —
first round-3 capacity-axis re-validation. Closes the throughput-
engineering loop: round-1 closed depth-8 because of budget
confounding; round-2 unlocked the budget; now we test whether the
budget-confounded round-1 axes converge cleanly with 29 epochs.

**#559 (fern y_std weighting) closed — mechanism falsified.**
α=0.5 3-seed mean = 136.48 ± 6.35 (+21.7 %, ~3.1σ outside noise on
wrong side); α=1.0 catastrophic. **Wrong splits regressed**:
predicted-to-improve val_single_in_dist + val_geom_camber_rc
regressed *most*; predicted-to-regress val_geom_camber_cruise
regressed *least*. Up-weighting raceCar globally degraded raceCar's
own held-out splits — Huber-clipping-induced cruise favoring is
**not** the underlying cause of the per-distribution-shift pattern.
**Reassigned fern to #659 (per-domain stratified data weighting)** —
fern's own #559-closing follow-up #1. Tests whether the data-axis
mechanism works when within-domain difficulty is decoupled from
between-domain class structure. **Last data-axis lever** before
round-3 axes pivot to per-region (spatial) or per-channel mechanisms.

Cycle 22 actions (recap): #568 closed (fp16 catastrophic; bf16 stays
right precision), #517 closed (DropPath statistically null at
depth=5), #648 assigned (alphonse LayerScale), #649 assigned
(tanjiro smoothness).

Earlier cycle actions:
**#335 closed** — schedule axis doesn't stack on Huber+slice-128.
3-seed mean 122.38 (seeds 124.45, 119.61, 123.08) vs 115.61
baseline = +5.85 % on mean, all 3 above. Same per-distribution-shift
pattern as alphonse #311 (cruise improves, in-dist regresses) —
**third independent confirmation of "interior optima are
architecture-dependent"** (after nezuko #332 + alphonse #311).
Tanjiro's schedule infrastructure (`cosine_t_max` flag + warmup +
SequentialLR + `SENPAI_SEED` seeding) stays in train.py for round-2
reuse.
**Reassigned tanjiro to round-2 axis #517 (stochastic depth /
DropPath)** — drop entire TransolverBlock residual contributions
during training with prob `drop_path` (linear-by-depth scaling per
DeiT/Swin recipe). Sweep `drop_path ∈ {0.0, 0.05, 0.1, 0.2}` with
multi-seed at the winner. Truly orthogonal regularization-axis
lever (no other in-flight axis touches block-level dropout).

Cycle 16 actions (recap): #311 closed (width axis exhausted),
#485 assigned (alphonse RMSNorm).

**Five merges**: #328 (slice_num=128, anchored prior baseline at
133.55) + #330 (Huber β=1, was anchor 115.61) + #367 (scoring NaN
fix, infrastructure) + #399 (bf16, infrastructure mean 112.13) +
**#553 (torch.compile, ROUND-2 WINNER, current operative baseline
80.70 ± 2.20)**. **Thirteen closed axes** (#311 width, #325 depth,
#326 FFN, #332 surf_weight, #335 schedule, #337 BS+LR, #429 p_weight,
#452 slice-192, #478 curriculum, #485 RMSNorm, #517 DropPath, #559
y_std-weighting, #568 fp16-retry — all axes that fight the 30-min
cap, overlap with Huber's implicit gradient shaping, are sub-
literature-threshold at our depth=5 scale, or operate at the wrong
mechanism level for the per-distribution-shift pattern).
**Eight axes currently in flight** (7 round-2 + 1 round-3):
#415 frieren asinh-on-pressure, #457 thorfinn EMA, #472 nezuko Lion,
#547 edward LLRD, #648 alphonse LayerScale, #649 tanjiro smoothness-
regularization, #659 fern per-domain stratified weighting,
**#660 askeladd depth-8 ON COMPILE (first round-3 capacity-axis
retry)**. **All eight students busy on actionable WIP work.**

## What we learned this cycle (and last)

0. **Single-seed noise floor on this branch is ≈ ±10 %.** Thorfinn's
   accidental same-config replicate (kon60q79 vs nphltrz9, both
   `bs=8 lr=7e-4`) showed a 13.8-point spread (153.19 vs 139.39)
   purely from random seed / data-order. This means:
   - Differences smaller than ~10 % between single-seed runs are not
     statistically meaningful.
   - The 134-138 cohort (alphonse, fern, edward, nezuko surf-15) is
     effectively tied within noise.
   - Merge decisions for results inside ±10 % of baseline must use
     multi-seed replication to disambiguate.
   - Frieren's 109.47 (18 % over baseline) is the only round-1
     result outside the noise band on a single seed; it's the only
     result we can be confident about without replication.

1. **Loss formulation is the highest-leverage axis we've found.**
   Frieren's Huber-β=1 result (109.47 on slice_num=64) is roughly 4×
   the magnitude of any architecture-axis gain seen in round 1.
   Likely combination of (a) Huber's L1-tail behavior matches the
   L1 metric we're ranked on better than MSE; (b) at high Re,
   normalized residuals exceed 1.0 enough that gradient clipping
   prevents tail samples from dominating. Direct evidence of (b):
   `val_re_rand` is best-of-cohort (100.85, 17 % over next-best).
2. **`slice_num` is the most pay-per-FLOP architecture axis.**
   Doubling slice tokens (64 → 128) gave the round-1 architecture
   winner at minimal extra cost (~10–15 % slowdown). Per-split
   signal matched the slice-bottleneck hypothesis cleanly: 7 %
   improvement on `val_geom_camber_rc` vs the next-best run.
3. **Capacity-scaling axes (`n_hidden`, `n_layers`, `mlp_ratio`)
   are confounded by the 30-min cap.** They each chop the epoch
   budget by 30–80 %, so the wall-clock-equal comparison favors
   cheaper changes even if the asymptotic answer might prefer
   wider models. Compute-equal middle grounds (width-160,
   mlp_ratio=3) are the right next step — both already in flight.
4. **The cruise NaN bug is a real blocker for paper-facing metrics.**
   Both edward and fern independently identified the same root cause
   (sample 20 of `test_geom_camber_cruise/.gt` has 761 NaN in pressure;
   `0.0 * NaN = NaN` in `accumulate_batch` defeats the `y_finite`
   mask). PR #367 puts up the 2-line `nan_to_num` fix.
5. **Branches need rebases as soon as the advisor branch moves.**
   Confirmed across the round on every PR created during the
   round-1 assignment phase. The send-back-for-rebase pattern
   catches it. Reviewers should diff against current advisor head
   before merging anything.

7. **Huber absorbs small-effect optimization-axis levers** (the
   round-2 dominant lesson, surfaced from edward #429 + tanjiro #335
   independently, then refined by fern #478 to a deeper mechanism).
   Huber β=1's gradient clipping past |residual|=1 already implicitly
   does the work that several round-2 axes tried to do explicitly,
   AND it bakes in a per-distribution gradient imbalance that any
   subsequent perturbation amplifies:
   - **schedule** (cosine T_max=15) — closed; Huber's tail-clipping
     covers what late-epoch LR shrinkage was doing on MSE.
   - **per-channel weighting** (p_weight=2) — closed; Huber already
     up-weights the channel with the largest residuals (pressure on
     high-Re).
   - **surf_weight=25** — closed; similar mechanism.
   - **batch+LR scaling** — closed (also hardware-blocked).
   - **curriculum** — closed; global y_std filtering flips domain
     balance and amplifies Huber's cruise-favoring.
   The corollary for round-2 axis selection: only mechanisms that
   are **structurally orthogonal** to gradient-magnitude shaping
   are likely to compound (target distribution via asinh, throughput
   via bf16/torch.compile, parameter-noise via EMA, normalization
   via RMSNorm, regularization via DropPath, update rule via Lion,
   per-layer LR via LLRD). **And one direct counter-axis**:
   per-sample loss weighting ∝ y_std (#559) which targets Huber's
   per-distribution gradient imbalance directly — first round-2 PR
   to attack the per-distribution-shift pattern at the mechanism
   level rather than working around it.

8. **Per-distribution-shift pattern: 7 of 7 round-2 perturbations,
   with one sign reversal.** alphonse #311 width-160, tanjiro #335
   schedule, edward #429 p_weight, fern #478 curriculum, nezuko #332
   surf_weight (partial), alphonse #485 RMSNorm, tanjiro #517
   DropPath. The first 6 share a sign (cruise improves / raceCar
   regresses); DropPath uniquely **reverses the sign on
   `val_single_in_dist`** (helps -3 to -5 %) while still hurting
   `val_geom_camber_rc` (+3 to +4 %). **Anchored as a property of
   the y_std-heterogeneous data distribution**, not perturbation-
   specific. **Cause is NOT Huber-clipping** (falsified by fern #559;
   per-sample y_std weighting against the predicted direction
   degraded the predicted-to-improve splits most). Round-3 axes
   should look at **per-region (spatial)** or **per-channel**
   mechanisms instead — possibly surface-vs-volume per-region
   weighting, or joint-channel y_std weighting, or per-domain
   architecture specialization.

9. **Throughput axis was the highest-leverage round-2 lever** (the
   round-2 winner). bf16 (#399) + torch.compile (#553) compounded
   multiplicatively (1.23× × 1.81× = 2.23×), and the extra epoch
   budget converted directly to metric improvement (29 epochs →
   80.70 vs 13 epochs → 112.13). The 30-min wall-clock cap was
   the binding constraint across round 2; lifting it (via efficiency,
   not by extending wall-clock) was the right move. **Round-3 should
   re-validate the round-1 capacity-axis closes** (depth-8, width-160,
   slice-256, mlp_ratio=4) because they were budget-confounded, not
   architecturally unfit. First retry assigned (#660, askeladd
   depth-8-on-compile).

10. **Compile + Huber tightens the noise floor** σ ≈ 2.20 (down from
    σ ≈ 4.53 at bf16+Huber, ±10 % at fp32-eager). Multi-seed runs
    discriminate effects in fewer seeds. **Updated round-3 multi-seed
    budgeting**: 3-seed σ ≈ 1.3, 2-seed σ ≈ 1.6 — small-effect axes
    that were lost in fp32 noise should now be measurable.
6. **Interior hyperparameter optima are architecture-dependent.**
   nezuko #332 found a clean interior optimum at surf_weight=25 on
   slice_num=64 (sweep curve + val_vol_p secondary signal). It did
   not transfer to slice_num=128 — multi-seed mean was 18 points
   above baseline because the surface-pressure bottleneck shifted
   when slice_num doubled. Future round-3 stacking decisions should
   anticipate this: a hyperparameter winner found on one
   architecture cannot be assumed to generalize across an
   architecture change without re-tuning.

## Round 2 candidate stacks (post round-1 settle, will compound on the new baseline)

- **Push `slice_num` further.** ASSIGNED as PR #452 to fern.
  Continuation of the merged #328 win. Sweep slice_num ∈ {192, 256}.
- **Stack slice-128 + width.** alphonse's width axis on top of the
  merged slice-128 baseline — directly tests the
  arch-stack hypothesis. Pending alphonse's width-160 result first.
- **bf16 mixed precision.** **MERGED ★ as infrastructure (#399).**
  3-seed bf16+Huber mean 112.13 (σ=4.53), 1.23× speedup, +30 %
  epoch headroom, first finite test_avg=101.82.
- **`torch.compile`.** **MERGED ★ as round-2 winner (#553).**
  3-seed compile+bf16+Huber mean **80.70 (σ=2.20)**, 2.23× total
  speedup over fp32-eager, +123 % epochs in budget vs bf16-only,
  18 GB memory freed by kernel fusion. Single biggest round-2 win.
- **Schedule that fits the budget.** OneCycleLR over `total_steps`
  (not epochs) is robust to 30-min wall-clock cuts. May supersede
  `T_max=epochs` cosine entirely. Pending tanjiro's iteration.
- **Per-channel loss weighting on `p`.** CLOSED in #429 — Huber's
  gradient clipping already absorbs the channel-weighting effect;
  multi-seed mean 118.65 vs 115.61 baseline = +2.6 %.
- **Layer-wise learning rate decay (LLRD).** ASSIGNED as PR #547
  to edward: split parameters into per-layer groups, geometric LR
  decay by depth (foundation slower, output faster). Sweep decay
  ∈ {0.7, 0.85, 1.0 anchor}. Truly orthogonal to all other
  optimizer axes (Lion = different update rule; EMA = post-step
  averaging; LLRD = per-layer step-size splitting).
- **Target-space reformulation.** ASSIGNED as PR #415 to frieren:
  `asinh` on pressure channel of `y_norm` only (Ux/Uy unchanged).
  Pairs naturally with the merged Huber-β=1 — orthogonal mechanisms
  on the same high-Re-tail failure mode (distribution flattening vs
  gradient clipping). per-sample-std normalization is a separate
  follow-up if asinh doesn't capture the gain.
- **Geometry-preserving augmentation.** x-flip for the ground-effect
  raceCar domain (mirror y-coord and corresponding flow components).
- **Curriculum learning.** CLOSED in #478 — global y_std quantile
  filtering inadvertently flips domain balance and amplifies Huber's
  implicit cruise-favoring. Per-domain stratified curriculum
  remains a round-3 candidate.
- **Per-sample loss weighting ∝ y_std (counter-balance Huber).**
  CLOSED in #559 — **mechanism falsified**. α=0.5 3-seed mean
  136.48 ± 6.35 (+21.7 %, ~3.1σ outside noise on wrong side); α=1.0
  catastrophic. Wrong splits regressed (predicted-to-improve splits
  regressed *most*). Up-weighting raceCar globally degrades
  raceCar's own held-out splits — Huber-clipping is NOT the cause
  of the per-distribution-shift pattern.
- **Per-domain stratified data weighting.** ASSIGNED as PR #659 to
  fern: weight per-sample loss by within-domain rank rather than
  global rank. Fern's own #559-closing follow-up #1. Tests whether
  data-axis-difficulty works when decoupled from between-domain
  class structure. Last data-axis lever before pivot to per-region /
  per-channel mechanisms.
- **Weight averaging (EMA).** ASSIGNED as PR #457 to thorfinn:
  `torch.optim.swa_utils.AveragedModel` with custom EMA `avg_fn`,
  per-epoch dual val with best-of (live, EMA) checkpoint selection.
  Decay sweep {0.9, 0.99, 0.999}. Direct fit for the undertrained-
  at-30-min-cap regime where every PR's val curve is still
  descending at the wall-clock cut.
- **Lion optimizer.** ASSIGNED as PR #472 to nezuko: replaces
  AdamW with a sign-based update + single momentum buffer (~33 %
  optimizer-state memory savings). 3-value LR sweep
  {3e-5, 1e-4, 3e-4}. Reported in literature to outperform AdamW
  on transformers at smaller LR.
- **RMSNorm.** CLOSED in #485 — multi-seed mean 127.74 outside
  ±10 % band; fused PyTorch LayerNorm is already near-optimal at
  our scale (RMSNorm only saves the mean-subtraction reduction;
  -2.3 % epoch time, below the ≥3 % bar).
- **fp16 with scoped autocast + GradScaler.** CLOSED in #568 —
  catastrophic divergence on all 3 seeds (PhysicsAttention slice
  pathway overflows fp16's range); +0.2 % slower than bf16, no
  throughput unlock at our scale on Blackwell. **bf16 stays the
  right precision for this branch.**
- **Stochastic depth (DropPath).** CLOSED in #517 — multi-seed mean
  statistically null vs in-PR control; throughput regressed 3.5 %
  instead of unlocking; depth=5 below literature threshold (≥12).
- **LayerScale (learned residual gain).** ASSIGNED as PR #648 to
  alphonse: per-channel learned scalars on each block's residual
  contributions, init small (1e-4 to 1e-2). Modern transformer
  recipe (CaiT, ViT-S+, DeiT). Predicted 1-3 %.
- **Auxiliary smoothness loss (physics-informed).** ASSIGNED as PR
  #649 to tanjiro: penalize prediction differences between random
  pairs of nearby mesh nodes (Gaussian-decay weighted by spatial
  distance). Truly novel orthogonal axis filling the un-claimed
  "auxiliary loss / physics prior" bucket. Random-pair sampling
  sidesteps O(N²) k-NN.

## Round-3 capacity-axis retries (NEW WAVE — enabled by compile merge)

The compile merge (#553) doubled the in-budget epoch count from 13 → 29
and freed 18 GB of memory. Round-1 axes that closed because of budget
confounding now have ~3× more training and ~2× more memory. **First
retry assigned this cycle**:

- **depth-8 retry on top of compile** (PR #660 to askeladd). Round-1
  #325 closed at val=162.05 with 9/50 epochs. With 29 epochs in
  budget at depth-5 (and ~18-22 expected at depth-8), depth-8 should
  now have meaningfully more training. Tests "depth-8 was just
  budget-confounded" hypothesis directly.

Future round-3 retry candidates (assignment after current axes settle):
- **width-160 retry on top of compile** (alphonse closed-and-
  carried-forward axis from #311; multi-seed mean was 126.18).
- **slice_num=256 retry on top of compile** (fern closed #452;
  single-seed was 133.30).
- **mlp_ratio=4 retry on top of compile** (edward closed #326;
  multi-seed was 137.83).
- **Stochastic depth (DropPath).** ASSIGNED as PR #517 to tanjiro:
  drop entire `TransolverBlock` residual contributions during
  training with prob `drop_path` (linear-by-depth scaling per
  DeiT/Swin recipe). Sweep {0.0, 0.05, 0.1, 0.2}. Truly orthogonal
  regularization-axis lever; modern transformer default.
- **Per-distribution architecture specialization (round-3
  candidate).** Surfaced from alphonse's closed #311 iter-3
  per-split data: width-160 helps `val_geom_camber_cruise` (−6.1 %)
  but hurts `val_single_in_dist` and `val_geom_camber_rc`
  (+17–20 %). Capacity scaling has split-specific returns at the
  30-min cap. Future axes to explore: per-mesh-size-conditional MLP
  scaling, or specialised heads per domain. Round-3 work after
  AMP/bf16 throughput unlock makes wider models budget-feasible.

## Active blockers

1. **`test_avg/mae_surf_p = NaN` for every run** until PR #367 lands
   (in flight, fern). Round-1 winners are being selected on
   `val_avg/mae_surf_p` only — this is fine for the round-1 ranking
   but blocks paper-facing comparisons.

## Constraints respected

- `SENPAI_MAX_EPOCHS=999` and `SENPAI_TIMEOUT_MINUTES=30` not overridden.
- `data/scoring.py` and `train.py::evaluate_split` are normally
  read-only for experiment PRs — bug-fix PR #367 is the documented
  exception (metric-preserving 2-line patch).
- One hypothesis per PR. Send-backs ask for variants of the same
  axis or compute-equal restatements; not bundled changes.
