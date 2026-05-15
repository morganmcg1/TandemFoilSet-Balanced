# SENPAI Research Results — `willow-pai2i-24h-r1`

Rolling log of completed experiment PRs reviewed by the advisor. Metrics
sourced from W&B (project `wandb-applied-ai-team/senpai-v1`); rankings use
`val_avg/mae_surf_p` (lower is better). Test-side ranking is currently
contaminated by an Inf in the cruise test ground truth (see notes).

## 2026-05-15 18:25 — PR #3143: Charbonnier robust loss vs MSE — **MERGED ⭐⭐ MAJOR WIN**

- Student branch: `willowpai2i24h1-edward/charbonnier-robust-loss`
- Student: `willowpai2i24h1-edward`
- Hypothesis: replace MSE with Charbonnier `sqrt(diff² + ε²)` so the gradient
  becomes ~linear for large residuals. Surface pressure has order-of-magnitude
  dynamic range — a few near-stagnation outliers per mesh dominate MSE gradient
  and bias the model toward them at the expense of the bulk surface MAE.

| Arm | wandb run | `val_avg/mae_surf_p` | test 3-split avg | best_epoch | Notes |
|-----|-----------|----------------------|------------------|-----------|-------|
| **charbonnier_eps1e-3** ⭐ | lukq8jry | **98.60** | **98.03** | — | merged |
| mse_baseline (within-PR control) | 9npuojl6 | 121.14 | — | — | MSE-only, pre-warmup base |

Per-split val (charbonnier_eps1e-3): single_in_dist=126.05, geom_camber_rc=106.97,
geom_camber_cruise=73.34, re_rand=88.04. Per-channel val_avg: surf_Ux=1.377,
surf_Uy=0.639, surf_p=98.60, vol_p=103.37 — Charbonnier improves _every_ channel,
not just surf_p, with the biggest gain on surf_p where the dynamic range is largest.

Per-split partial test (cruise excl. due to NaN bug):
single_in_dist=115.46, geom_camber_rc=93.44, re_rand=85.18.

**Conclusion:** Charbonnier robust loss delivers **−18.6%** on the primary
metric vs the within-PR MSE control (121.14 → 98.60). The largest per-split
gain is on `val_geom_camber_cruise` (−25.6%), which has the highest pressure
dynamic range — exactly the failure mode the robust loss targets. Run-to-run
noise (~3 units) cannot explain a 22-unit gap; this is a real effect.

**Decision:** **merged**. New advisor-branch baseline.
- Adds `--loss_fn` (default mse → now charbonnier on the advisor branch) and
  `--charbonnier_eps` (default 1e-3) flags. The per-node loss is
  `sqrt(diff² + ε²)` for charbonnier, applied identically in train and
  evaluation (so the eval metric for `mae_surf_p` itself is unchanged — it's
  an L1-style MAE on denormalized predictions, independent of loss choice).
- **Compositional caveat.** Edward's branch forked before PR #3150 (warmup +
  cosine) was merged, so the 98.60 result is _Charbonnier alone_ vs MSE
  without warmup. The merge composes with the warmup schedule already on the
  advisor branch. Next student control runs against this composed baseline
  will confirm the post-warmup + Charbonnier number.
- Pre-warmup MSE control was 121.14, post-warmup MSE control should be
  around 125.83 (PR #3150 winner). If Charbonnier composes additively we'd
  expect the new baseline to land in the ~95-100 range; if it composes
  multiplicatively (−18.6% × 125.83) we'd expect ~102. We'll see.

**Edward's suggested follow-ups (worth running):**
1. ε sweep (1e-4, 3e-3, 1e-2) — ε=1e-3 is the only point tested
2. Other robust losses: Huber, Cauchy/Lorentzian, pseudo-Huber
3. Re-check `surf_weight` after residual linearization (the optimal weight
   may have shifted; surf gradient magnitudes are now compressed relative to
   vol gradient magnitudes)

## 2026-05-15 17:43 — PR #3150: Warmup + cosine schedule, peak LR sweep — **MERGED ⭐ WINNER**

- Student branch: `willowpai2i24h1-tanjiro/warmup-cosine-schedule`
- Student: `willowpai2i24h1-tanjiro`
- Hypothesis: a 3-epoch linear warmup before CosineAnnealingLR(eta_min=1e-6)
  yields lower `val_avg/mae_surf_p` at the baseline peak LR, and may unlock
  higher peak LRs.

| Arm | wandb run | `val_avg/mae_surf_p` | test 3-split avg | best_epoch | Notes |
|-----|-----------|----------------------|------------------|-----------|-------|
| **lr5e-4_wu3** ⭐ | sb39atyp | **125.83** | **122.01** | 13 | merged |
| lr5e-4_wu0 (no-warmup ref) | bww3uk1z | 142.28 | 136.67 | 12 | internal control |
| lr1.5e-3_wu3 | n5v1kwy3 | 139.27 | 132.57 | 13 | |
| lr1e-3_wu3   | xxt85yy2 | 149.07 | 145.70 | 12 | |

**Conclusion:** warmup at the baseline peak LR is a clear win — −16.45 (−11.6%)
versus the internal no-warmup control, and −3.2% versus the pre-merge
implicit baseline of 130 ± 3. Better on **every** val split, not just the
average. Higher peak LRs (1e-3, 1.5e-3) underperform even with warmup —
the model is already in a stable training regime at lr=5e-4 with AdamW+LN,
so warmup mitigates initial dynamics but doesn't expand the useful peak-LR
range in this budget.

**Decision:** **merged**. Now the new advisor-branch baseline.
- Adds `--warmup_epochs` flag (default 3); replaces scheduler with
  `SequentialLR(LinearLR start_factor=1e-3, CosineAnnealingLR eta_min=1e-6)`.
- Note from student analysis: part of the within-PR gap (16.45) may be partly
  attributable to the wu0 reference decaying to eta_min=0 vs wu3 decaying to
  1e-6 — disentangling pure warmup vs eta_min is a worthwhile future probe.

## 2026-05-15 16:35 — PR #3145: Deeper Transolver: n_layers 5/8/10

- Student branch: `willowpai2i24h1-fern/deeper-transolver`
- Student: `willowpai2i24h1-fern`
- Hypothesis: deepening Transolver (`n_layers` 5 → 8 → 10) improves
  `val_avg/mae_surf_p` because more mixing stages help model multi-scale
  CFD structure; ~1.27M params at depth 10 is still tiny for 96 GB VRAM.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | epochs reached | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|----------------|------------------------------------------|
| depth5  | 0g36hqgg | **129.07** | 14 | 14 | rc=120.44 / sid=140.92 / re=118.32 |
| depth8  | 3x6sfou4 | 172.83 | 9  | 9  | rc=183.88 / sid=193.75 / re=147.69 |
| depth10 | b539x67l | 164.10 | 8  | 8  | rc=164.56 / sid=186.37 / re=142.99 |

**Conclusion:** depth5 (baseline control) wins by a wide margin. depth8 is
+34%, depth10 is +27% — both clear regressions. Per-epoch wall-clock scales
~linearly with depth (131s → 207s → 255s), so deeper arms only complete ~half
as many epochs before hitting the 30-min cap.

**Important finding from student analysis.** Per-epoch, the deeper arms are
*more sample-efficient*: at epoch 8, depth10 = 164.10 < depth5 = 185.67.
But depth5's 2× faster per-epoch lets it race past at epoch ~10 and keep
improving. **This is the same wall-clock confound as PR #3148 (width sweep)** —
the 30-min cap is the binding constraint; any lever that scales per-step
compute without proportional sample-efficiency gain loses in this budget.

**Decision:** close. Worth revisiting after bf16 AMP (PR #3330) lands as a
winner — that would roughly halve per-epoch time and unlock depth/width
experiments to converge within budget.

## 2026-05-15 15:36 — PR #3148: Wider Transolver: n_hidden 128/192/256

- Student branch: `willowpai2i24h1-frieren/wider-transolver`
- Student: `willowpai2i24h1-frieren`
- Hypothesis: widening `n_hidden` (and proportionally `n_head`) from the
  baseline 128/4 reduces `val_avg/mae_surf_p` because the baseline is small
  for 96 GB VRAM.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | total_min | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|-----------|------------------------------------------|
| w128 | qmyih0vv | **128.46** | 14 | 30.64 | rc=141.6 / sid=129.3 / re=114.4 |
| w192 | u9udr95v | 149.32 | 7 | 30.65 | rc=152.3 / sid=165.3 / re=132.3 |
| w256 | o1ax3h3f | 173.63 | 7 | 30.36 | rc=171.4 / sid=202.0 / re=153.6 |

**Conclusion:** widening *hurts* in this budget. All three arms hit the
30-min wall-clock cap; the wider arms reached their best val at epoch 7 vs
epoch 14 for the baseline width — the wider models simply did not have
enough wall-clock to converge. Width 192 is +16% on the primary metric,
width 256 is +35% — both above the close threshold (>5% regression).

**Decision:** close as dead end *in this training budget*. Wider models are
worth revisiting with (a) more epochs, (b) warmup + higher peak LR, or
(c) substantially smaller width steps. Implementation itself is clean —
the `--n_hidden` / `--n_head` plumbing in `train.py` is preserved for
future experiments.

## 2026-05-15 15:31 — PR #3149: Per-channel surface-loss weights focusing on p

- Student branch: `willowpai2i24h1-nezuko/surface-pressure-loss`
- Student: `willowpai2i24h1-nezuko`
- Hypothesis: explicitly upweighting the p channel inside `surf_loss` directly
  pushes the optimizer toward what the ranking metric measures, reducing
  `val_avg/mae_surf_p` without much volume cost.

| Arm | wandb run | `val_avg/mae_surf_p` | best_epoch | total_min | partial `test_avg/mae_surf_p` (3 splits) |
|-----|-----------|----------------------|-----------|-----------|------------------------------------------|
| surfp1  | 7d1rlw4w | **132.33** | 13 | 30.81 | rc=136.9 / sid=139.1 / re=122.5 |
| surfp4  | 7tuf0qsy | 132.71 | 13 | 30.75 | rc=132.8 / sid=135.3 / re=117.8 |
| surfp10 | 84u5mine | 140.66 | 13 | 30.81 | rc=136.4 / sid=154.1 / re=133.9 |

**Conclusion:** per-channel surface-p upweighting did not improve
`val_avg/mae_surf_p`. surfp4 ties baseline (+0.3%, within run-to-run noise);
surfp10 is +6.3% worse. The infra is correct (mean-normalization keeps
surfp1 identical to current baseline; diagnostic per-channel surf MSEs are
logged in W&B).

**Decision:** close — no improvement on the primary metric. The lever
exists but at the tested weights doesn't help. Future variants worth
trying: downweight Ux/Uy (mathematically equivalent at the surface but
keeps the volume loss balanced) and/or combine with a higher overall
`surf_weight`.

## Cross-PR observations (round 1)

- **Run-to-run variance is ~3-4 units in `mae_surf_p`.** Three
  nominally-identical baseline configs across PRs gave 128.46 (frieren w128),
  129.07 (fern depth5), 132.33 (nezuko surfp1) — a ~4% spread. Improvements
  smaller than ~3-4 mae_surf_p units should not be treated as winners on a
  single seed.
- **30-min wall-clock cap binds at 50 epochs** for the baseline width.
  All 6 reviewed runs hit ~30.4-30.8 min total, meaning training stopped
  exactly at the cap. Best-val epoch was 13-14 for the baseline-width
  arms — the model is still improving at the end of training. This
  suggests longer schedules or faster convergence (warmup, larger LR)
  could move the baseline number.
- **`test_avg/mae_surf_p` is None for all 6 runs.** Root cause:
  `test_geom_camber_cruise` has Inf in its hidden ground truth `y` somewhere;
  `data/scoring.py` does an `Inf * 0 = NaN` operation when masking
  non-finite samples (line ~49 in `accumulate_batch`, the mask-and-sum
  expression). This contaminates the cruise test MAE accumulator with NaN,
  which serializes as None in W&B. Validation cruise is unaffected.
  Filing as a separate issue to the human researcher team.
- **6 of 8 student pods are stuck waiting** due to GitHub API rate-limit
  exhaustion on the shared token — their entrypoint pollers cannot see
  their assigned PRs. This is an operational/throughput issue, not a
  research signal; expected to self-resolve when the limit resets.
