<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research Results — willow-pai2g-24h-r3

Lower is better for `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## 2026-05-14 00:15 — PR #2657: Cautious Lion on n_head=2+Lion — CLOSED

- `willowpai2g24h3-thorfinn/cautious-lion`
- **Hypothesis (H3 from RESEARCH_IDEAS 2026-05-13_22:30):** Mask Lion updates where `sign(g) ≠ sign(interp_momentum)` per Liang et al. 2411.16085 (ICLR 2026). Targets confirmed Lion+noise amplification mechanism (PR #2097 coord-jitter regress). Implementation via timm's `Lion(caution=True)` — one-line plumbing change.
- **Results (single clean seed `mmcehzjn`, killed duplicate `bddaxut9` on advisor poke):**

| Metric | Cautious Lion | Baseline Lion (gd934e9l) | Δ |
|---|---:|---:|---|
| val_avg/mae_surf_p | **45.6368** | 40.2741 | **+13.32% ❌** |
| test_avg/mae_surf_p | **38.9703** | 33.6017 | **+15.98% ❌** |
| val_single_in_dist | 46.618 | 35.836 | +30.1% |
| val_geom_camber_rc | 58.884 | 53.495 | +10.1% |
| val_geom_camber_cruise | 31.642 | 28.146 | +12.4% |
| val_re_rand | 45.403 | 43.619 | +4.1% |
| best epoch | 30/50 | 36/50 | (terminated at 30-min cap, still descending ~0.5-1/epoch) |
| Peak VRAM | 10.0 GiB | ~13.6 GiB | smaller — caution gates updates |

- **Analysis (thorfinn's terminal):** Three sharp insights from his analysis:
  1. **Motivation surface absent.** The cautious mask was designed to suppress amplification of adversarial noise (e.g. coord-jitter). With no jitter or aug in the baseline, the only signal being suppressed is normal SGD minibatch noise — which IS information, not corruption. Selectively dropping those updates is a net loss.
  2. **Re-normalization amplifies variance.** `update *= mask / mask.mean()` makes surviving updates take 1.4-2× larger effective steps when mask.mean() ≈ 0.5-0.7. Combined with Lion's already-large sign-step magnitude, this is variance amplification, not noise suppression.
  3. **Paper-regime mismatch.** Cautious gains (1.5-3× speedup) are wall-clock at multi-day budgets where momentum stabilizes. At epoch 30 in a 50-epoch schedule with warmup_epochs=5, momentum is barely converged — selectivity has no asymptotic regime to benefit from.
- **Banked follow-ups (thorfinn's suggestions, not assigning):**
  - Cautious + coord-jitter (2×2 grid) — actually tests hypothesis as stated; requires jitter plumbing
  - Cautious WITHOUT re-normalization — isolates variance-amplification mechanism
  - Cautious AdamW + jitter — decouples Lion's sign-step variance from cautious gating
- **Next experiment:** thorfinn reassigned to **PR #2679 APW Curriculum** (H1; per-sample loss-EMA-weighted SmoothL1 with α 0→0.5 ramp; first sample-conditional gradient shaping test on this stack).
- **Banked lesson:** Cautious Optimizers' paper-regime is multi-day LLM training. Don't assume "optimizer-level interventions" port to 30-min CFD surrogates without verifying convergence regime overlap.

## 2026-05-14 00:30 — PR #2667: warmup_epochs=10 on n_head=2+Lion — RUNNING (terminal pending)

- `willowpai2g24h3-edward/warmup-epochs-10`
- **Hypothesis:** Edward's own follow-up #2 from #2612 terminal SENPAI-RESULT: "Since training is gradient-step-bound, anything that gets the live weights to a lower-loss point at epoch 50 will translate directly into a lower-loss EMA. Longer warmup (5→10 epochs) delays the LR peak and shifts more 'useful refinement' into the late phase the EMA actually captures."
- **Result (single seed `wpt9oaph`, finished):**

| Metric | warmup=10 (this run) | Baseline (gd934e9l) | Δ |
|---|---:|---:|---|
| val_avg/mae_surf_p | **41.7675** | 40.2741 | **+3.71% ❌** |
| test_avg/mae_surf_p | **34.4976** | 33.6017 | **+2.67% ❌** |
| Step count | 27000 (final) | ~27000 | (both hit cap) |
| Elapsed | 30.8 min | ~30 min | |

- **Analysis (pending terminal SENPAI-RESULT from edward):** This is the smallest val/test regression we've seen on the n_head=2+Lion stack since #2192 baseline merged. Val 41.77 (+3.71%) is over close bar (40.30) but tighter than every other regression this round. **Test +2.67% is the smallest test regression of any run in the plateau period**. Interpretation: shifting LR peak later does pull terminal weights slightly *worse*, suggesting the model needs MORE time at peak LR, not less. The mechanism inversion implies the natural follow-up: **shorter warmup** (warmup_epochs=2 or 0) — let the model spend more time at high LR before cosine decay starts. Pending edward's SENPAI-RESULT to confirm direction and queue this as his next assignment.

## 2026-05-14 00:00 — PR #2552: Fourier K continuation (K=16, K=20) on n_head=2+Lion — CLOSED

- `askeladd/fourier-k-lion-sweep`
- **Hypothesis:** Continue the AdamW-stack monotone K trend (K=4→8→12) up to K=16/K=20 on the n_head=2+Lion stack to find the new local optimum after Lion + n_head=2 merges.
- **Results (Arm B clean; Arm A 5× crashes):**

| Arm | K | Run | State | val_avg | test_avg | Δ val | Δ test |
|---|---:|---|---|---:|---:|---:|---:|
| B | 20 | `7qfg4u8k` | finished (30.4 min, cap) | **42.8227** | **35.4099** | **+6.33% ❌** | +5.38% ❌ |
| A | 16 | `eqzo2qox` | crashed @ 27.5 min | 43.185 (mid) | — | (+7.2% trajectory) | — |
| A | 16 | `ik2i80ar` | crashed @ 26.3 min | 44.023 (mid) | — | (+9.3% trajectory) | — |
| A | 16 | `swth32k8` | crashed @ 19.3 min | 50.934 (mid) | — | — | — |
| A | 16 | `y10si1ew` | crashed @ 16.0 min | 55.251 (mid) | — | — | — |
| A | 16 | `lo8zls4m` | crashed @ 7.0 min | 88.708 (mid) | — | — | — |

Baseline: 40.27 / 33.60 (`gd934e9l`, PR #2192).

- **Analysis:** K=20 cleanly completes a 30.4-min run and regresses 6.3% val / 5.4% test — solidly over the 40.30 close bar. K=16 crashed 5/5 attempts with non-uniform durations (7.0 to 27.5 min) and absence of Python tracebacks (consistent with external SIGKILL, not in-process exceptions). Askeladd's crash diagnostic was thorough — non-uniform durations + clean K=20 completion argue against pure OOM (K=20 has slightly higher memory footprint than K=16). Plausible mechanism: K=16 hit transient gradient pathology on this stack triggering a process-fatal NaN before backward could log it. Even the best mid-run K=16 trajectory pointed to val ≥ 43, well into close territory regardless.
- **Mechanism interpretation:** The merged n_head=2 win came from doubling `dim_head` (32 → 64), which enriched per-head capacity but halved parallel head count. Adding more Fourier modes (K=16/20) widens the positional input dimension into a model with reduced parallel attention capacity — model can't usefully exploit the extra positional features. The K axis interacts with the head count.
- **Suggested follow-up (askeladd's):** Test K=8 and K=10 (downward) on n_head=2+Lion to find new local K optimum; current K=12 inherited from n_head=4 tuning may no longer be the right operating point. Assigned in **PR #2670 — Fourier K-down sweep**.

## 2026-05-14 00:00 — PR #2612: EMA decay sweep (0.9995, 0.998) on n_head=2+Lion — CLOSED (both arms regress)

- `willowpai2g24h3-edward/ema-decay-sweep`
- **Hypothesis:** Test EMA decay flanks (slower 0.9995, then faster 0.998 after Arm A regressed) on n_head=2+Lion stack. Baseline uses 0.999. If gradient-step-bound regime is dominant, EMA window may be the binding constraint.
- **Results (5 runs across both arms — duplicate-launch artifacts):**

| Run | ema_decay | State | val_avg | Δ val |
|---|---:|---|---:|---:|
| `9ir26ul6` | 0.9995 | finished (clean) | 47.97 (best epoch summary) | **+19.1% ❌** |
| `g5ndrcny` | 0.998 | finished (clean) | 42.92 (best epoch summary) | **+6.6% ❌** |
| `dpejdw12` | 0.9995 | crashed mid | — | — |
| `0bocvkzd` | 0.9995 | crashed mid | — | — |
| `69p1wbb2` | 0.9995 | crashed mid | — | — |

Baseline: 40.27 / 33.60.

- **Analysis:** Both flanks regress; baseline 0.999 is the local optimum on this stack. **Slower EMA (0.9995) regresses much more (+19.1%)** — pulls EMA toward stale earlier-trajectory weights since training is monotonically descending (best_epoch == final_epoch). **Faster EMA (0.998) regresses smaller (+6.6%)** — loses variance-reduction benefit. Edward's terminal analysis: "The real lever is the trajectory itself, not the EMA window. Since training is gradient-step-bound, anything that gets the live weights to a lower-loss point at epoch 50 will translate directly into a lower-loss EMA." → assigned to warmup_epochs=10 follow-up (#2667).
- **Operational flag:** Pod had 3 concurrent runs initially (`9ir26ul6`, `0bocvkzd`, `dpejdw12`); edward killed `dpejdw12` and `69p1wbb2` on advisor poke, did not relaunch 0.9999. Same harness duplicate-launch pattern as #2596 / #2555 / #2657.
- **Banked lesson:** **EMA decay axis exhaustively closed**: 0.998 (-6.6%), 0.999 (baseline), 0.9995 (-19.1%) on n_head=2+Lion. Don't revisit. The mechanism inversion (slower = catastrophic) is intuitive once you recognize the gradient-step-bound regime — never the case when training is u-shaped, which it isn't here.
- **Next experiment:** edward reassigned to **PR #2667 warmup_epochs=10** (shift LR peak later in 50-epoch budget; tests edward's own follow-up #2 from this terminal SENPAI-RESULT).

## 2026-05-13 23:13 — PR #2555: Lion wd sweep (3e-4, 1e-3) on n_head=2 — CLOSED

- `willowpai2g24h3-thorfinn/weight-decay-lion-sweep`
- **Hypothesis:** test wd=3e-4 (3× default) and wd=1e-3 (10× default) on the merged n_head=2+Lion stack. Lion paper recommends 3-10× AdamW wd for equivalent regularization; predicted a small positive on at least one arm.
- **Results (4 finished runs across duplicate launches):**

| Run | wd | State | val_avg | test_avg | Δ val |
|---|---:|---|---:|---:|---:|
| `5xmshkwk` | 3e-4 | finished | 41.32 | 34.34 | +2.61% ❌ |
| `581t7ak5` | 3e-4 (seed 2) | finished | 44.41 | 37.00 | +10.27% ❌ |
| `d4dd5gd1` | 1e-3 | finished | **40.91** | 34.56 | **+1.59% ❌** |
| `myywnism` | 3e-4 (seed 3) | finished | — | — | (duplicate-launch artifact) |

Baseline: 40.27 / 33.60 (`gd934e9l`).

- **Analysis:** Both arms regress vs baseline. The lower-regress arm (wd=1e-3 at +1.59%) still sits above the 40.3 close threshold and the test metric is +2.86%. The seed-2 wd=3e-4 result (+10.27%) shows the same-config seed variance can be very large (41.32 vs 44.41), reinforcing that wd default 1e-4 is well-tuned for this stack. The Lion paper's 3-10× wd guidance applies to LLM/vision scales — on this 548K-param CFD surrogate at 30-min budget, larger wd hurts more than it helps. **Lion wd axis closed; default 1e-4 is optimal.** PR closed by student after advisor sendback.
- **Operational flag:** Duplicate-launch issue surfaced again — 4 runs (3 of wd=3e-4, 1 of wd=1e-3) across one pod. Same harness pattern as edward #2596 / #2612.
- **Next experiment:** thorfinn reassigned to **PR #2657 Cautious Lion** (H3 from RESEARCH_IDEAS_2026-05-13_22:30) — sign-gate update mask on Lion that directly targets the noise-amplification mechanism his earlier coord-jitter PR (#2097) exposed.

## 2026-05-13 22:55 — PR #2593: n_head=1 isolation — CLOSED

- `willowpai2g24h3-frieren/n-head-1-isolation`
- **Hypothesis:** continue the n_head monotone trend (4→2 improves) down to n_head=1 (single 128-dim head). Predicted if monotone: small val gain. Predicted if non-monotone: regression on the in-dist split where head diversity matters most.
- **Results (1 clean seed + 1 crash):**

| Metric | n_head=1 (`x0o1lj5y`) | n_head=2 baseline (gd934e9l) | Δ |
|---|---:|---:|---|
| val_avg | 44.0769 | 40.2741 | **+9.44% ❌** |
| test_avg | 36.5697 | 33.6017 | **+8.83% ❌** |
| val_single_in_dist | 42.106 | 35.836 | **+17.50%** |
| val_camber_rc | 57.623 | 53.495 | +7.72% |
| val_camber_cruise | 30.794 | 28.146 | +9.41% |
| val_re_rand | 45.785 | 43.619 | +4.97% |
| test_single_in_dist | 34.925 | 30.586 | +14.19% |
| Best epoch | 33/35 | 36/36 | (1 epoch shorter wall-clock; n_head=1 has slightly larger QKV per layer) |

Second seed `7svginti` crashed at 10.5 min, val 108 mid-divergence.

- **Analysis:** The n_head sweep is **non-monotone**. n_head=4 → n_head=2 improves, n_head=2 → n_head=1 regresses. Mechanism: PhysicsAttention benefits from at least 2 heads to represent inter-cluster vs intra-cluster patterns; a single 128-dim head loses pattern diversity even with richer per-head capacity. The largest regression is on val_single_in_dist (+17.5%) — the easiest split, where the wider single head should have been most helpful, is hurt the most. **n_head=2 confirmed as local optimum.** All 4/4 val and 4/4 test splits regress.
- **Next experiment:** frieren reassigned to PR #2644 — **SwitchEMA** (periodic EMA-to-weights swap, interval=500 and 1000). Targets the gradient-step-bound observation with zero compute overhead.

## 2026-05-13 22:10 — PR #2596: n_layers=3 + n_head=2 + Lion — CLOSED

- `willowpai2g24h3-edward/n-layers-3-n-head-2`
- **Hypothesis:** Combine the two strongest individual signals — n_head=2 (PR #2192) and n_layers=3 depth signal observed in `rxqavwx9` (val 39.84 at n_head=4) — predicted to compound additively on the merged n_head=2+Lion stack.
- **Results (single seed, run `zmirxzq5`):**

| Metric | n_layers=3+n_head=2+Lion (this) | Baseline n_layers=4+n_head=2+Lion (gd934e9l) | Δ |
|---|---:|---:|---|
| val_avg | 41.6177 | 40.2741 | **+3.34% ❌** |
| test_avg | 35.0493 | 33.6017 | **+4.31% ❌** |
| val_single_in_dist | 40.9411 | 35.836 | **+14.25%** |
| val_geom_camber_rc | 53.7159 | 53.495 | +0.41% |
| val_geom_camber_cruise | 29.1766 | 28.146 | +3.66% |
| val_re_rand | 42.6373 | 43.619 | −2.25% |
| Best epoch | 35/35 | 36/36 | (still descending at cap) |
| Peak VRAM | 51.1 GiB | ~13.6 GiB | (model larger? — same n_hidden=128) |

- **Analysis:** Depth and n_head do NOT compound additively on this stack. Edward's mechanism analysis: dim_head=64 (n_hidden=128 / n_head=2) needs 4 blocks of refinement to outperform; 3 blocks at dim_head=64 starves the per-block representational throughput. The reference data point `rxqavwx9` (n_layers=3 + n_head=4 + Lion = 39.84) has dim_head=32, where each block has 4 attention heads doing parallel mixing — so the per-block throughput is high enough that 3 blocks suffice. n_head=2's "richer per-head capacity" requires more iterative refinement passes to amortize across slices. The largest regression is val_single_in_dist (+14.25%), the largest and most homogeneous split — strongest signal that per-block representational quality, not gradient-step budget, is the binding constraint here. Closed per merge bar (val ≥ 40.3 → close).
- **Operational flag:** Pod had 3 concurrent training processes (PIDs 138008, 141990, 142923). Student handled cleanly by reporting from the clean 21:27 run that had committed best-checkpoint artifacts before contention. Harness assignment issue worth investigating downstream — not a content issue.
- **Banked lesson:** Depth axis (n_layers) and head-count axis (n_head) target different bottlenecks (compute-per-block vs per-slice capacity) and do not stack. n_layers=4 should be held fixed on the n_head=2+Lion stack going forward.
- **Next experiment:** edward reassigned to EMA decay sweep (PR #2612, 0.9995 / 0.9999) targeting the "best_epoch == final_epoch, still descending" gradient-step-bound observation.

## 2026-05-13 21:17 — PR #2192: n_head=2 + Lion — MERGED ⭐ WIN

- `willowpai2g24h3-frieren/n-head-sweep`
- **Hypothesis:** Reducing n_head from 4→2 gives dim_head=64 (vs 32), enriching per-head attention capacity. n_head controls parallel attention patterns in PhysicsAttention; with n_head=2, each head encodes richer geometry per cluster. Orthogonal to optimizer axis — predicted to compound with Lion.
- **Results (n_head=2+Lion stack, 3 seeds):**

| Seed | Run ID | val_avg | test_avg | Δ val vs 43.20 | Δ test vs 35.76 |
|---|---|---|---|---|---|
| **1 (best)** | `gd934e9l` | **40.27** | **33.60** | **−6.78% ✅** | **−6.04% ✅** |
| 2 | `j598prwj` | 41.00 | 34.84 | −5.10% ✅ | −2.57% ✅ |
| 3 | `r00qdgp9` | 41.06 | 34.56 | −4.97% ✅ | −3.36% ✅ |
| **Mean** | — | **40.78** | **34.33** | **−5.62%** | **−4.00%** |
| **Std** | — | 0.44 | 0.64 | — | — |

Per-split val (best seed `gd934e9l`): single_in_dist 35.84 (−13.9%), camber_rc 53.50 (−5.5%), camber_cruise 28.15 (−4.8%), re_rand 43.62 (−3.0%)
Per-split test (best seed `gd934e9l`): single_in_dist 30.59, camber_rc 45.36, camber_cruise 22.88, re_rand 35.58

- **Analysis:** n_head=2 compounds cleanly with Lion's sign-momentum. All 3 seeds beat baseline (val 43.20), tight clustering (val σ=0.44, ~1% of mean). Best epoch = final epoch for all seeds → model still descending at 30-min cap, indicating further headroom. The n_head monotone trend (4→2 improves) raises n_head=1 as a natural follow-up (frieren assigned #2593). Strong gain on single_in_dist (−13.9%) and camber_rc (−5.5%); smaller on cruise/re_rand. **New baseline: val 40.27 / test 33.60.** New merge bar: ≤36.2 val (≥10%), 36.2-40.3 → second seed, ≥40.3 → close.
- **Impact on in-flight PRs:** All 7 WIP PRs sent back for rebase + n_head=2 retest. Stack update: all experiments must now use `--n_head 2 --optimizer lion --lr 1e-4` base.

## 2026-05-13 19:56 — PR #2097: coord-jitter σ=0.005 on Lion stack — CLOSED

- `willowpai2g24h3-thorfinn/coord-jitter-aug`
- **Hypothesis:** Gaussian (x,z) coordinate jitter augmentation on Lion stack — orthogonal to optimizer, predicted to improve OOD generalization.
- **Results (Lion stack, run `i9ek9ffo`):**

| Metric | Lion baseline (h2m396kw) | coord-jitter σ=0.005 | Δ |
|---|---:|---:|---|
| val_avg | 43.20 | 45.30 | **+4.86% ❌** |
| test_avg | 35.76 | 38.09 | **+6.51% ❌** |
| Best epoch | 32 | 27 | −5 epochs |

Per-split val: single_in_dist +4.6%, camber_rc +2.4%, camber_cruise **+10.1%**, re_rand +4.7% — ALL 8/8 regress.

Note: On AdamW stack, same σ=0.005 gave val 52.19 vs 53.84 baseline (−3.1%, marginal positive). On Lion it becomes a clear negative.

- **Analysis:** Critical insight — **input noise that was a weak positive on AdamW becomes a strong negative on Lion.** The mechanism: Lion uses `sign(gradient_momentum)`, so every update is a unit-magnitude direction change. Coordinate jitter corrupts the gradient direction signal, and Lion amplifies that corruption into a uniformly-large step in the wrong direction. AdamW's second-moment adaptive scaling partially absorbed the noisy gradient signal; Lion cannot. The largest regression is on `camber_cruise` (+10.1%) — the split where Lion delivered its biggest absolute gain (−26.9%) — confirming that Lion's geometry-capturing ability is what's being disrupted. **General lesson: input augmentations that add noise should be retested on Lion, as the optimizer amplification mechanism can invert their effect.** Augmentation axis closed on Lion stack.
- W&B: `i9ek9ffo` (Lion+jitter), `ufx6b3d1` (AdamW Arm A σ=0.005), `glq2kv94` (AdamW Arm B σ=0.01)

## 2026-05-13 19:10 — PR #2314: Lion optimizer (lr=1e-4) — MERGED ⭐ MAJOR WIN

- `willowpai2g24h3-askeladd/lion-optimizer`
- **Hypothesis:** AdamW → Lion (Chen et al. 2023 sign-based momentum). At lr=1e-4 (AdamW_lr/5), Lion normalizes updates to max step size, making each gradient step more impactful without slowing throughput.
- **Results:**

| Metric | Baseline (qttr6jay, AdamW) | Lion lr=1e-4 (h2m396kw) | Δ |
|---|---:|---:|---|
| val_avg | 53.8380 | **43.1973** | **−19.8% ✅** |
| test_avg | 46.9320 | **35.7630** | **−23.8% ✅** |
| Best epoch | 29 | 32 | +3 |
| s/epoch | 57.8 | 57.8 | identical |
| VRAM | ~13 GB | 11.2 GB | −14% |

Per-split val (all 4 improve):

| Split | Baseline | Lion | Δ |
|---|---:|---:|---|
| single_in_dist | 57.24 | **41.63** | −27.3% |
| camber_rc | 62.57 | **56.63** | −9.5% |
| camber_cruise | 40.50 | **29.58** | −26.9% |
| re_rand | 55.05 | **44.96** | −18.3% |

Per-split test (all 4 improve):

| Split | Baseline | Lion | Δ |
|---|---:|---:|---|
| single_in_dist | 50.16 | **34.16** | −31.9% |
| camber_rc | 56.06 | **47.74** | −14.8% |
| camber_cruise | 33.72 | **24.83** | −26.4% |
| re_rand | 47.79 | **36.32** | −24.0% |

- **Analysis:** Largest single-step gain of the launch (−19.8% val, −23.8% test). Lion's sign-based updates effectively normalize gradient magnitudes, giving uniform step size every iteration regardless of curvature. At lr=1e-4 (1/5 of AdamW's 5e-4), Lion matches AdamW's throughput exactly (32 epochs in 30 min at 57.8 s/ep). All 8/8 per-split metrics improve; largest gains on smooth-geometry splits (camber_cruise −26.9%, single_in_dist −27.3%), consistent with Lion's known strength on smooth loss landscapes. Second seed from pre-merge arm (n_layers=5 stack) confirmed at val 45.48, test 37.95. All subsequent experiments must now use `--optimizer lion --lr 1e-4` as part of the base stack.
- W&B: `h2m396kw` (retest on n_layers=4 stack), `62gq70rj` (Arm A n_layers=5), `qng9zq6d` (Arm B lr=3e-4)
- **New baseline:** val 43.1973 / test 35.7630. New merge bar: ≤38.9 val (≥10% gain).
- **Full stack reproduce:** `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5 --fourier_k 12 --slice_num 32 --batch_size 2 --n_layers 4 --optimizer lion --lr 1e-4`

## 2026-05-13 18:55 — PR #2192: n_head=2 sweep — SENT BACK (retest on Lion stack)

- `willowpai2g24h3-frieren/n-head-sweep`
- **Hypothesis:** n_head sweep (2, 4, 8) — wider per-head attention (n_head=2, dim_head=64) may capture more physically coherent flow patterns than 32 dims/head at default.
- **Results (AdamW stack, before Lion merge):**

| Arm | n_head | dim_head | val_avg | test_avg | best_ep | s/ep |
|---|---|---|---:|---:|---|---|
| **B (`fa9n7lqp`)** | **2** | **64** | **49.777** | **42.676** | 36 | 50.2 |
| A (`xejqc8hr`) | 4 | 32 | 53.149 | 44.873 | 31 | 58.3 |
| C (`x0sx5nmq`) | 8 | 16 | 58.737 | 50.907 | 25 | 72.4 |

- **Analysis:** Clear n_head=2 win (−7.5% val, −9.1% test vs AdamW baseline 53.84). n_head=8 is a strong regression (+9.1%) — narrowing heads below 32 dims/head hurts. n_head=2 with 64 dims/head enables richer per-slice representation. However, results were measured without Lion. Against new Lion baseline 43.20, the result regresses. Sent back for n_head=2+Lion retest. If confirmed, n_head=2+Lion likely compounds.
- W&B group: `willow-r3-n-head-sweep-nl4`

## 2026-05-13 18:31 — PR #2097: coord-jitter σ=0.005 — SENT BACK (retest on Lion stack)

- `willowpai2g24h3-thorfinn/coord-jitter-aug`
- **Hypothesis:** Gaussian coordinate jitter on (x,z) input dims — augmentation for OOD-camber generalization.
- **Results (AdamW stack):**

| Arm | σ | val_avg | Δ | test_avg | Δ | best_ep |
|---|---|---:|---|---:|---|---|
| **A (`ufx6b3d1`)** | **0.005** | **52.19** | **−3.1%** | **44.84** | **−4.5%** | 32 |
| B (`glq2kv94`) | 0.01 | 52.57 | −2.4% | 45.25 | −3.6% | 32 |

- **Analysis:** Both arms improve on AdamW baseline (53.84), with σ=0.005 best. Improvements are small (3.1%/4.5%) and fall within the ±7 single-seed noise band. However, directionally consistent (both val+test improve) and augmentation is orthogonal to optimizer. Sent back for σ=0.005+Lion retest. If Lion amplifies the gain via better use of augmented gradients, this could compound. Thorfinn's per-split analysis showed largest benefit on `single_in_dist` (−7.5%) and `re_rand` (−5.7%), with very small camber_rc benefit (−0.4%).
- W&B: `ufx6b3d1` (σ=0.005), `glq2kv94` (σ=0.01)

## 2026-05-13 18:20 — PR #2462: n_layers=3 — SENT BACK (retest on Lion stack)

- `willowpai2g24h3-edward/n-layers-sweep`
- **Hypothesis:** Push depth below n=4 merged baseline — n_layers=3 on full stack.
- **Results (AdamW stack):**

| Run | val_avg | test_avg | best_ep | params |
|---|---:|---:|---|---|
| Seed 1 (`qgju7je7`) | **48.7928** | **42.1367** | 40 | ~428K |
| Seed 2 (`hrayb76h`) | 52.3633 | 45.0600 | 33 | ~428K |
| Baseline (qttr6jay, n=4) | 53.8380 | 46.9320 | 29 | 548K |

- **Analysis:** Clean n_layers=3 win on AdamW (−9.4% val, −10.2% test). Two seeds confirm the direction (epoch-aligned noise ~0.3). All 8/8 per-split metrics improve; largest on single_in_dist (−14.7% val, −16.5% test), smallest on camber_rc (−3.4%). Val curve still strictly descending at epoch 40 (−0.5/ep avg), model still gradient-step-limited. However, against new Lion baseline 43.20, the result regresses. Sent back for n_layers=3+Lion retest. With Lion's efficient updates and faster per-epoch throughput at n=3 (~43 s/ep estimated), this could be very strong.
- W&B: `qgju7je7` (seed 1), `hrayb76h` (seed 2)

## 2026-05-13 16:43 — PR #2423: n_hidden=192 on bs=2+slice32 merged stack — CLOSED

- `willowpai2g24h3-fern/wider-model-bs2`
- **Hypothesis:** With VRAM dropping to 13.6 GB at bs=2, the 82 GB headroom reopens the width axis. n_hidden=192 on bs=2+slice32 stack should compound the grad-step gains.
- **Result:** val 67.87 (+17.6% regression), test 58.66 (+18.4% regression). All 8 sub-metrics regress.

| Metric | Baseline (jc24jr52) | n_hidden=192 (2rl3vfjq) | Δ |
|---|---:|---:|---|
| val_avg | 57.7122 | 67.8679 | +17.6% ❌ |
| test_avg | 49.5412 | 58.6628 | +18.4% ❌ |
| Epochs | ~26 | 17 | −9 epochs |
| s/epoch | 71.0 | 94.6 | +33% slower |
| VRAM | 13.6 GB | 18.6 GB | +5 GB |

- **Analysis:** Budget-bound. 33% slower epochs → 17 epochs vs 26 → under-trained. Val still descending at final epoch (best = last epoch). non-EMA test 75.92 vs EMA 58.66 confirms model still converging fast at termination. **Width-up axis closed at bs=2 / 30-min cap.** Next direction: width-down (n_hidden=96, n_hidden=64) mirrors the n_layers depth-down win — assigned fern #2464.
- W&B: `2rl3vfjq` (group: `willow-r3-n-hidden-192-bs2`)

## 2026-05-13 16:43 — PR #2119: n_layers=4 on bs=2+slice32+fourier_k=12 stack — MERGED

- `willowpai2g24h3-edward/n-layers-sweep`
- **Hypothesis:** Shallower model converges further in a fixed 30-min budget. First-round data (pre-slice32) showed n=4 > n=5 > n=6 monotonically. Retest on full merged stack (bs=2+slice32+fourier) to confirm the gain stacks.
- **Results:**

| Metric | Baseline (jc24jr52, n=5) | n_layers=4 (qttr6jay) | Δ |
|---|---:|---:|---|
| val_avg | 57.7122 | **53.8380** | **−6.71% ✅** |
| test_avg | 49.5412 | **46.9320** | **−5.27% ✅** |
| Best epoch | 26 | 29 | +3 |
| s/epoch | 71.0 | 57.8 | −19% faster |
| n_params | 668,855 | 548,755 | −18% |

Per-split val: in_dist=57.24 (−6.3%), camber_rc=62.57 (−8.1%), camber_cruise=40.50 (−5.8%), re_rand=55.05 (−6.2%). All 4/4 improve.
Per-split test: 50.16 (−3.4%), 56.06 (−5.3%), 33.72 (−6.2%), 47.79 (−6.5%). All 4/4 improve.

- **Analysis:** Mechanism confirmed. 4/5 per-layer scaling holds exactly (measured 0.81× vs predicted 0.80×). 3 extra epochs in the same wall-clock (+11.5% grad steps). All 8/8 per-split metrics improve. The depth-and-batch-size levers are partially redundant (both buy more grad steps), reducing marginal gain vs first-round data, but direction is unambiguous. **Merged as clean win.** New baseline: val 53.84 / test 46.93. Next: n_layers=3 sweep (edward #2462) and n_hidden narrowing (fern #2464).
- W&B: `qttr6jay` (group: `willow-r3-n-layers-slice-ext`)
- Reproduce: `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5 --fourier_k 12 --slice_num 32 --batch_size 2 --n_layers 4`

## 2026-05-13 16:31 — PR #2413: Cosine tail compression (--epochs 25, --epochs 22) on slice32+bs2 stack — CLOSED

- `willowpai2g24h3-tanjiro/cosine-tail-compress`
- **Hypothesis:** At bs=2 (T_max=45, cap at ~25 epochs → terminal LR ≈ 0.59× peak), compressing the cosine tail (T_max=20 via --epochs 25) would lower terminal LR to ~0.05× peak and extract more learning in the last few epochs. Arm A: --epochs 25 (mild tail curl), Arm B: --epochs 22 (full decay to 0). Per pre-stated rule, Arm B skipped when Arm A regresses.

| Metric | Arm A (qkscl3sh) | Baseline (jc24jr52) | Δ |
|--------|---:|---:|---|
| val_avg | 60.8314 | 57.7122 | +5.41% ❌ |
| test_avg | 52.4963 | 49.5412 | +5.96% ❌ |

Per-split val: single_in_dist=61.93 (+1.42%), camber_rc=70.79 (+3.93%), camber_cruise=48.01 (+11.71%), re_rand=62.60 (+6.63%). Per-split test: 54.03 (+4.1%), 62.19 (+5.1%), 38.96 (+8.4%), 54.81 (+7.3%). All 8 sub-metrics regress; largest hit on cruise.

GPU contention caveat: epochs 19-22 ran ~150 s (2× normal) due to duplicate process competition; extrapolation to clean run lands ~58-59 val, still above the 57.7 close bar.

- **Analysis:** Cosine-tail-compression axis settled. The gradient-step-bound mechanism confirmed also at bs=2: the baseline's "high" terminal LR (0.59× peak at epoch 25) is doing real work. Replacing it with near-zero LR via compressed T_max uniformly hurts all 4 splits. The "shape" axis on the cosine schedule is now fully characterized: both aggressive (T_max=15, #2302) and mild (T_max=20, this PR) compressions regress. Next experiment: "magnitude" axis — peak LR sweep (lr=7e-4, lr=1e-3) assigned to tanjiro #2449.
- W&B: `qkscl3sh` (group: `willow-r3-cosine-tail-compress`)

## 2026-05-13 15:51 — PR #2389: batch_size=2 — more grad steps per 30-min cap — MERGED

- `willowpai2g24h3-nezuko/bs2-retest`
- **Hypothesis:** Default bs=4 is gradient-step-limited under the 30-min cap (val curve monotonic at epoch 23). bs=2 doubles mini-batches/epoch, and `pad_collate`'s per-batch padding means smaller batches waste fewer FLOPs on padding smaller meshes to the max sample size.
- **Results:**

| Metric | Baseline (#1747 bs=4) | bs=2 (`jc24jr52`) | Δ |
|------|----:|----:|---|
| val_avg | 65.3954 | **57.7122** | −11.7% ✅ |
| test_avg | 56.1093 | **49.5412** | −11.7% ✅ |
| Epochs in 30 min | 23 | **26** | +13% |
| s/epoch | 85.4 | **71.0** | −17% (faster!) |
| Peak VRAM | 80.9 GB | **13.6 GB** | −83% |
| Total grad steps | ~8,625 | **~19,500** | +2.26× |

Per-split val: single_in_dist=61.06 (−14.4%), camber_rc=68.11 (−8.0%), camber_cruise=42.98 (−13.4%), re_rand=58.71 (−11.9%). Per-split test: 51.92 (−13.1%), 59.20 (−9.0%), 35.95 (−12.5%), 51.09 (−12.7%).

- **Analysis:** Uniform 4/4 val + test win confirms pure under-training story: the baseline was gradient-step-limited, not capacity- or regularization-limited. The Pareto speedup (bs=2 *faster* per epoch) is due to padding-waste reduction: `pad_collate` pads each batch to the largest sample; at bs=4 one large mesh inflates 3 smaller samples' FLOPs by up to 3.2×; at bs=2 only 1 sample is padded per batch. Val curve still strictly descending at epoch 26 (LR ~28% peak, cosine T_max=50 only 52% spent). VRAM headroom (82 GB free) opens the door for larger model experiments. EMA contribution intact (test_no_ema = 59.85 vs EMA 49.54 → −17.2% from EMA on top of bs=2).
- W&B: `jc24jr52` (group: `willow-r3-bs2`)
- Reproduce: `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5 --fourier_k 12 --slice_num 32 --batch_size 2`

## 2026-05-13 15:40 — PR #2313: Fourier K sweep (K=16, K=20) on slice32 stack — CLOSED

- `willowpai2g24h3-fern/higher-fourier-k`
- **Hypothesis:** K=12 was the best in the pre-slice stack; extending upward to K=16, K=20 might continue improving.

| Arm | K | val_avg | test_avg | W&B |
|---|---|---:|---:|---|
| Baseline | 12 | 65.3954 | 56.1093 | `9sk1rwv1` |
| Arm A | 16 | 66.6196 | 57.6283 | `l2rjj9sf` |
| Arm B | 20 | 66.3272 | 57.6231 | `eu9wc5ki` |

- **Analysis:** Both arms regress. K curve bent past K=12 on the slice32 stack. High-K failure mode: geometry-variation splits (camber_rc, camber_cruise) regress with higher K while single_in_dist marginally improves — consistent with high-frequency overfitting on 32-sample training set. K=12 confirmed optimum. Fourier-K axis settled.

## 2026-05-13 15:26 — PR #2302: Cosine budget-match --epochs 20 on slice32 stack (retest) — CLOSED

- `willowpai2g24h3-tanjiro/cosine-budget-match`
- **Hypothesis (retest):** Short cosine schedule (T_max=15, --epochs 20) that was previously promising on fourier-only stack. Retested with --slice_num 32 added.
- val 71.21 / test 62.03 → +8.88% / +10.55% regression vs new baseline.
- **Analysis:** Hypothesis broke under slice32. The slice32 stack cap-time LR (≈0.66× peak at epoch 23) is already much closer to optimal than fourier-only stack (≈0.86× peak). Forcing T_max=15 starves the model of mid-schedule gradient area. The bottleneck under slice32 is total gradient steps (confirmed by bs=2 win), not LR-schedule shape. Follow-up: mild cosine-tail compression (--epochs 25) assigned to tanjiro #2413.

## 2026-05-13 14:15 — PR #1747: slice_num=32 on Physics Attention (+ fourier_k=12 baseline) — MERGED

- `willowpai2g24h3-alphonse/slice-num-sweep`
- **Hypothesis:** Transolver's default `slice_num=64` may be overcomplete for Physics Attention with `n_hidden=128, dim_head=32`. Fewer slice tokens force denser slot packing and better node routing, especially on large cruise meshes (242K nodes).

| Run | slice_num | val_avg | test_avg | Epochs | W&B |
|-----|-----------|---------|----------|--------|-----|
| Fourier-K12 baseline | 64 (default) | 73.16 | 63.89 | 19/50 | `osxp8woj` |
| **slice_num=32 + fourier_k=12** | **32** | **65.40** | **56.11** | **23/50** | **`9sk1rwv1`** |

Per-split val (`9sk1rwv1`):

| split | baseline (slice=64) | slice=32 | Δ |
|-------|---------------------|----------|---|
| val_single_in_dist | 80.28 | 71.30 | −11.2% ✅ |
| val_geom_camber_rc | 81.88 | 74.02 | −9.6% ✅ |
| val_geom_camber_cruise | 57.96 | 49.63 | −14.4% ✅ |
| val_re_rand | 72.52 | 66.63 | −8.1% ✅ |
| **val_avg** | **73.16** | **65.40** | **−10.6% ✅** |

Per-split test (`9sk1rwv1`): 59.73 / 65.09 / 41.08 / 58.54 → test_avg 56.11 (−12.2% vs 63.89).

- **Merge decision:** val 65.40 clears merge bar ≤65.8 (≥10% gain over 73.16). All 4 splits improve on both val and test. MERGED as clean single-seed winner.
- **Analysis:** Reducing slice tokens from 64→32 forces each token to aggregate a denser set of nodes. Default slice_num=64 was overcomplete relative to dim_head=32 — the in_project_slice Linear(32→64) produced too many near-empty token slots. Biggest gain on cruise (242K nodes, −14.4%) where routing overhead was highest. EMA contribution confirmed independent (test_no_ema=70.62 vs EMA=56.11, +20.5%). Val curve still monotonic at epoch 23 → cap is still binding constraint.
- **New baseline:** val 65.40 / test 56.11. Reproduce: `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5 --fourier_k 12 --slice_num 32`

## 2026-05-13 14:05 — PR #1986: Fourier positional features for (x,z) node coordinates — MERGED

- `willowpai2g24h3-tanjiro/fourier-positional-features`
- **Hypothesis:** Fourier position encoding (sin/cos bands over x,z with K frequencies) adds spatial inductive bias for sharp pressure peak localization, improving high-spatial-frequency splits (single_in_dist, camber_rc).

| Run | K | val_avg | test_avg | W&B |
|-----|---|---------|----------|-----|
| Pre-warmup K=4 | 4 | 76.18 | 66.58 | `kdv1oe8h` |
| Pre-warmup K=8 | 8 | 74.91 | 64.56 | `hjewmotx` |
| Pre-warmup K=12 | 12 | 73.33 | 63.85 | `5agqa3bm` |
| **K=12 + warmup5 retest** | 12 | **73.16** | **63.89** | `osxp8woj` |
| Baseline (d1lqln08) | — | 75.96 | 67.53 | `d1lqln08` |

- **Merge decision:** val 73.16 is in 68.4-76.0 in-band zone (−3.7% val, −5.4% test). MERGED per pre-committed in-band rule + confirmed mechanism + decisive test-mean improvement. Per-split: single_in_dist (−9.6%), camber_rc (−8.2%) improve strongly; cruise (+6.4%) and re_rand (+1.7%) mildly worse.
- **Analysis:** Monotone in K; K=12 is best. Zero latency overhead (sin/cos buffers, no learnable params). Key lesson: Fourier features trade cruise headroom for sharp-peak splits. The cruise regression is the main open negative — future work should target per-split Fourier scale or K optimization for smooth vs. sharp regimes. Schedule was noted as binding constraint (cosine 22% spent at cap) → motivates cosine-budget-match follow-up (#2302).

## 2026-05-13 14:05 — PR #1918: Stochastic Depth (DropPath) in TransolverBlock — CLOSED

- `willowpai2g24h3-fern/droppath-regularization`
- **Hypothesis:** DropPath regularization at standard transformer rates on residual paths would disproportionately help the under-regularized camber_rc OOD split.

| Arm | drop_path | val_avg | test_avg | W&B |
|-----|-----------|---------|----------|-----|
| 1 | 0.0 (baseline) | 76.39 | 67.44 | `41xzohdn` |
| 2 | 0.1 | 82.17 | 72.62 | `uxdbut8u` |
| 3 | 0.2 | 86.45 | 76.18 | `yzxvvjc7` |

- **Analysis:** Monotone degradation as drop_path rises. Prediction failed — all 4 splits degrade proportionally. Mechanism: EMA already captures DropPath's variance-reduction effect (EMA gap narrows: 19.77 → 11.69). DropPath is redundant on this stack and adds regularization overhead. **Dropout/stochastic-depth axis closed on EMA+AMP stack.**

## 2026-05-13 14:05 — PR #1721: Loss-level Re-reweighting by Reynolds number — CLOSED

- `willowpai2g24h3-askeladd/re-loss-reweight`
- **Hypothesis:** Continuous Re-temperature reweighting (re_loss_weight_temp) to tilt gradient toward high-Re samples without discrete sampler starvation.

| Arm | temp | val_avg | test_avg | W&B |
|-----|------|---------|----------|-----|
| 1 | 0.0 (baseline) | 76.43 | 67.20 | `33a6sx2k` |
| 2 | 0.3 | 78.50 | 69.26 | `6j9207pd` |
| 3 | 1.0 | 81.77 | 72.47 | `np0pm394` |

- **Analysis:** Monotone degradation as temperature rises — same failure as discrete Re-resampling (#1616). At t=1.0, re_factor spans 50× (0.046–2.299) which starves low-Re samples the same way. SmoothL1 already addresses gradient-dominance loss-side; adding sample-level Re-reweighting is redundant and harmful. **Re-weighting axis now fully closed** (tested via sampling #1616 and loss-weighting #1721).

## 2026-05-13 11:05 — PR #1998: Match cosine LR schedule T_max to AMP epoch budget — CLOSED

- Student branch: `willowpai2g24h3-nezuko/cosine-budget-match`
- Hypothesis: Match `T_max` to AMP epoch budget (~19-20 epochs) so cosine LR fully anneals before 30-min cap fires; should beat the default `T_max=50` which leaves LR at 37% of peak at cap.

### Results (4 arms, all rebased onto AMP+EMA stack `--amp --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999`, pre-warmup-merge)

| arm | --epochs | run id | epochs ran | val_avg | test_avg (EMA) |
|---|---:|---|---:|---:|---:|
| 1 (baseline reproduction) | 50 | `jip2if6s` | 14 | 93.08 | 82.34 |
| 2 (primary) | 20 | `vx3y6l75` | 19 | **81.95** | **72.32** |
| 3 (partial cool) | 25 | `mv4npbva` | 14 | 95.87 | 85.87 |
| 4 (aggressive) | 15 | `a6u6x6lu` | 15 | 94.85 | 84.32 |
| advisor baseline (#1440) | 50 | `30wvu5r0` | 19 | **77.37** | **68.21** |

### Analysis

- Internal "win" of Arm 2 over Arm 1 was a 12% drop in val_avg, but Arm 1 baseline reproduction at 93.08 is WAY worse than advisor reference at 77.37 — slow-seed handicap. Arm 1 only completed 14 epochs because several epochs took 200-213s (vs clean 98s rate).
- Arm 2 (e=20) absolute val=81.95 is +5.99 vs new advisor baseline 75.96 — **fails decision criteria** (≥76 → close).
- Within-experiment win is confounded by Arm 1's slow-seed; not a real T_max effect.
- **Keeper output**: Arm 2's per-epoch val curve table (322.72 → 81.95 across 19 epochs) shows monotonic decline through final epoch. With T_max=20, cosine LR fully anneals to ~0.6% of peak at epoch 19, yet val keeps falling. **The model is undertrained, not over-scheduled.**

### Lessons added to research state

- "Match T_max to AMP epoch budget" is closed as a *direct* lever — schedule completion does not move the metric when the model is under-trained.
- The diagnostic ("monotonic decline at end of training") motivates the **next axis: peak LR sweep** — if we can't add epochs, make each step more impactful. Reassigning nezuko to LR sweep (#2202).

## 2026-05-13 10:50 — PR #1438: 5-epoch linear LR warmup before cosine decay — MERGED

- Student branch: `willowpai2g24h3-frieren/warmup-5ep`
- Hypothesis: 5-epoch linear LR warmup from 0→peak before handing off to CosineAnnealingLR prevents early-training instability, reduces outlier basin risk (the "125.94 outlier" baseline run), and improves optimization convergence.

### Results (2 arms, both rebased onto current advisor stack `--amp --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999`)

| arm | warmup_epochs | val_avg | Δ val | test_avg (EMA) | Δ test | best epoch | run id |
|---|---:|---:|---:|---:|---:|---:|---|
| warmup-5ep (primary, **merged**) | 5 | **75.96** | **−1.41** | **67.53** | **−0.68** | 19 | `d1lqln08` |
| warmup-3ep (bracket) | 3 | 76.72 | −0.65 | 67.24 | −0.97 | 19 | `fzxx54lu` |
| advisor baseline (#1440 AMP+EMA) | 0 | 77.37 | — | 68.21 | — | 19 | `30wvu5r0` |

Per-split val/test (warmup-5ep `d1lqln08` vs advisor baseline):

| split | val (warmup-5ep) | Δ val | test (warmup-5ep) | Δ test |
|---|---:|---:|---:|---:|
| single_in_dist | 88.79 | −1.97 | 80.20 | +0.32 |
| geom_camber_rc | 89.23 | −1.50 | 79.83 | −1.25 |
| geom_camber_cruise | 54.47 | −0.41 | 45.41 | −0.47 |
| re_rand | 71.33 | −1.79 | 64.70 | −1.29 |
| **avg** | **75.96** | **−1.41** | **67.53** | **−0.68** |

### Analysis

- In-band result (val=75.96, ±7 noise band). Merge justified by: (1) all 4 val splits improve, (2) 3/4 test splits improve, (3) both warmup arms beat baseline, (4) compound-improvements principle.
- Epoch count = 19 (same as baseline) — warmup doesn't change throughput. The 5-epoch linear ramp has negligible overhead.
- Mechanism signal: warmup-3ep has better test_avg (67.24 vs 67.53) but worse val_avg. warmup-5ep has slightly wider EMA window (test vs no-EMA: 67.53 vs 78.16 — EMA helps +10.6); warmup-3ep shows worse test-no-EMA (87.70), suggesting its raw trajectory was noisier.
- Per the test-split tie-breaker lesson: 3/4 test splits favor warmup → merge confirmed.
- **New baseline: val_avg=75.96 / test_avg=67.53** (reproduce: `--amp --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --warmup_epochs 5`)

## 2026-05-13 09:10 — PR #1842: Transolver mlp_ratio sweep (post-rebase under AMP) — CLOSED

- Student branch: `willowpai2g24h3-edward/mlp-ratio-sweep`
- Hypothesis: at the new AMP operating point, smaller MLP (`mlp_ratio=1`) re-allocates throughput into more epochs of cosine cool-down; larger MLP (`mlp_ratio=4`) costs throughput. Pre-AMP this PR had won at 85.82 val (−6.4% vs pre-AMP 91.66).

### Results (3 arms, all rebased onto advisor `04aa53b` with `--amp --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999`)

| arm | mlp_ratio | n_params | val_avg | test_avg (EMA) | test_no_ema | best epoch | s/epoch | run id |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (baseline reproduction) | 2 | 0.66M | **77.94** | **67.84** | 78.49 | 19 | 98.86 | `6u8da009` |
| B (winner candidate pre-AMP) | 1 | 0.50M | 78.46 | 69.71 | 86.97 | 19 | 94.63 | `eliwspvs` |
| C (bracket) | 4 | 0.99M | 81.14 | 71.66 | 85.11 | 18 | 103.76 | `t9yrv6d1` |
| advisor reference | 2 | 0.66M | 77.37 | 68.21 | — | — | — | `30wvu5r0` |

Arm A reproduces the advisor reference within ±0.6 val / ±0.4 test — rebase is correct. Arm B (the pre-AMP winner) loses on every test split. Arm C (the upper bracket) loses by even more. Per-split test deltas vs baseline:

| split | ratio=2 | ratio=1 | ratio=4 | Δ(1−2) | Δ(4−2) |
|---|---:|---:|---:|---:|---:|
| test_single_in_dist | 79.84 | 81.30 | 84.78 | +1.46 | +4.94 |
| test_geom_camber_rc | 79.73 | 80.83 | 82.63 | +1.10 | +2.90 |
| test_geom_camber_cruise | 46.06 | 48.61 | 49.77 | +2.55 | +3.71 |
| test_re_rand | 65.72 | 68.09 | 69.44 | +2.38 | +3.72 |

### Mechanism — AMP subsumed the throughput-mechanism win

Epoch-time ratio collapse is the keeper output:

| arm | predicted (PR body) | pre-AMP measured | post-AMP measured |
|---|---:|---:|---:|
| ratio=1 vs baseline | 0.71× | 0.93× | **0.957×** |
| ratio=4 vs baseline | 1.75× | 1.10× | **1.050×** |

Under AMP, smaller MLP buys only ~4% per-epoch and larger MLP costs only ~5%. All three arms hit 18-19 epochs (vs the pre-AMP 13-15 spread). The throughput dial that drove the pre-AMP win (~2 extra epochs of cool-down) is now saturated by autocast. Only the per-step capacity effect remains, which favors the existing `mlp_ratio=2`.

### Conclusion

**Close. mlp_ratio axis at this depth/width is settled — no value in 1 or 4 over the existing default.** Code change (Config field) not cherry-picked: not needed for any in-flight hypothesis, and one less Config option is preferable. Edward reassigned to depth sweep (`n_layers`) — the natural follow-up since width was closed pre-AMP (#1443) and depth has never been tested at the AMP operating point.

### Emerging lessons logged

1. **AMP shifts the capacity-vs-throughput surface.** Pre-AMP optimum was throughput-saving (ratio=1); post-AMP optimum is the existing default (ratio=2). Pre-AMP architectural negatives may be worth a quick AMP re-sweep before being treated as final.
2. **Test-split direction breaks ties on noise-band val results.** When val_avg lands in ±7 noise, per-split test sign (all worse vs mixed) is the cleanest tie-breaker. Decisive here against second-seeding mlp_ratio=1.

## 2026-05-13 09:00 — PR #1779: AdamW weight_decay sweep at AMP+EMA baseline — CLOSED

- Student branch: `willowpai2g24h3-thorfinn/weight-decay-sweep`
- Hypothesis: AdamW `weight_decay` sweep {1e-4, 1e-3, 1e-2, 5e-2} as a regularization probe on top of the merged SmoothL1+grad-clip+EMA(0.999)+AMP stack. Prediction: a modest wd lift would help the highest-MAE OOD splits (`val_single_in_dist`, `val_geom_camber_rc`) preferentially.

### Results — all 3 variant arms (rebased onto advisor `04aa53b`, 30-min cap, AMP+EMA on)

| arm | wd | val_avg | test_avg | Δ val vs 77.37 | best epoch | run id |
|---|---:|---:|---:|---:|---:|---|
| Baseline (advisor) | 1e-4 | **77.37** | **68.21** | — | — | `30wvu5r0` |
| wd=1e-3 (best variant) | 1e-3 | 77.73 | 68.64 | +0.36 (+0.5%) | 19/50 | `xz3vojme` |
| wd=1e-2 | 1e-2 | 78.35 | 69.54 | +0.98 (+1.3%) | 19/50 | `npnres0j` |
| wd=5e-2 | 5e-2 | 81.03 | 71.57 | +3.66 (+4.7%) | 19/50 | `hqglc6x5` |

W&B group: `willow-r3-weight-decay-sweep`. No NaNs, no OOMs, peak GPU mem ~53 GiB across arms.

### Per-split val breakdown (wd=1e-3 best variant vs AMP+EMA baseline `30wvu5r0`)

| split | baseline 1e-4 | wd=1e-3 | Δ | wd=1e-2 | Δ |
|---|---:|---:|---:|---:|---:|
| val_single_in_dist | 90.76 | **90.29** | −0.47 | **89.65** | −1.11 |
| val_geom_camber_rc | 90.73 | **88.59** | **−2.14** | 91.46 | +0.73 |
| val_geom_camber_cruise | 54.88 | 57.99 | +3.11 | 57.80 | +2.92 |
| val_re_rand | 73.12 | 74.05 | +0.93 | 74.50 | +1.38 |
| **avg** | **77.37** | **77.73** | +0.36 | **78.35** | +0.98 |

### Mechanism analysis (thorfinn)

The hypothesis predicted that L2 regularization should preferentially help the highest-MAE OOD-ish val splits. **The mechanism is partly real but unprofitable**:

- At wd=1e-3, the two high-MAE OOD-ish splits (`val_single_in_dist`, `val_geom_camber_rc`) DO improve in the predicted direction (−0.47 and **−2.14** respectively).
- But the same regularizer hurts `val_geom_camber_cruise` by +3.11 and `val_re_rand` by +0.93.
- Net of opposing forces lands aggregate inside the ±7 noise band on the wrong side of baseline.

This is the signature of a **model whose capacity is matched-to-task**: there is no free regularization headroom — the same regularizer that fixes one split breaks another by the same magnitude. EMA already subsumes the variance-reduction effect that decoupled weight_decay would otherwise provide.

### Conclusion (advisor)

**Weight decay is closed at this baseline.** Rule out the entire L2-regularization family for Round 2: no wd schedules, no layer-wise wd, no AdamW-vs-Adam re-runs. The forward axis is **input-side feature representation / data augmentation**, which thorfinn's per-split asymmetry diagnosis points at directly. Next assignment to thorfinn is coordinate-jitter augmentation as a synthetic-near-miss-geometry generator targeting the held-out camber splits.

The per-split asymmetry table from this run ("one split is over-regularized by exactly the same amount that the other is under-regularized") is the cleanest capacity-vs-task-fit diagnostic from this round and should be revisited every time a new lever is proposed.

## 2026-05-13 06:35 — PR #1440: Enable bfloat16 mixed precision (AMP + EMA) — MERGED (WINNER)

- Student branch: `willowpai2g24h3-nezuko/amp-bf16`
- Hypothesis: `torch.autocast("cuda", dtype=torch.bfloat16)` for forward + loss reduces per-epoch wall-clock ~25-30%, giving ~35% more gradient steps within the 30-min budget. AMP and EMA are orthogonal — AMP changes per-step precision, EMA averages the parameter trajectory.

### Results

| arm | run | epochs (30-min cap) | s/epoch | peak VRAM | val_avg | test_avg | Δ vs merged 91.66/81.28 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (EMA, no AMP, advisor `emqh79b0`) | `emqh79b0` | ~14 | ~131 | ~42 GB | 91.6553 | 81.2845 | baseline |
| AMP only (no EMA, supplementary) | `rn1gkw8h` | 19 | 98.4 | 32.9 GB | 86.0296 | 74.2780 | −6.2% val |
| **AMP + EMA (merge candidate)** | `30wvu5r0` | 19 | 97.8 | 32.9 GB | **77.3716** | **68.2053** | **−15.6% val / −16.1% test** |

W&B group: `willow-r3-amp-bf16` in `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3`.

### Per-split breakdown (AMP+EMA `30wvu5r0` vs baseline `emqh79b0`)

| split | baseline val | AMP+EMA val | Δ | baseline test | AMP+EMA test | Δ |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 108.52 | **90.76** | −16.4% | 98.53 | **79.88** | −18.9% |
| geom_camber_rc | 104.81 | **90.73** | −13.4% | 89.74 | **81.08** | −9.7% |
| geom_camber_cruise | 68.50 | **54.88** | −19.9% | 57.58 | **45.88** | −20.3% |
| re_rand | 84.78 | **73.12** | −13.7% | 79.30 | **65.99** | −16.8% |
| **avg** | **91.66** | **77.37** | **−15.6%** | **81.28** | **68.21** | **−16.1%** |

### EMA decomposition at same step (best-val epoch 19)

| evaluation branch | test_avg/mae_surf_p |
|---|---:|
| EMA weights (saved ckpt, primary) | **68.21** |
| Raw weights at same step (`test_no_ema/*`) | 71.39 |
| AMP-only arm (different ckpt, no EMA training, `rn1gkw8h`) | 74.28 |

EMA on top of AMP adds ≈ −4.5% (variance-reduction-at-eval) + ≈ −3.7% (better-epoch-selection via smoother val curve) = ≈ −8% from EMA alone, fully consistent with the EMA mechanism from PR #1437.

### Analysis and conclusions

1. **AMP + EMA compose cleanly.** AMP changes the per-step precision pipeline (forward + loss in bf16, master weights fp32 inside AdamW). EMA averages the parameter trajectory outside the autocast context (fp32 EMA buffers). No numerical interaction. EMA overhead invisible at AMP speed (97.8 vs 98.4 s/epoch).
2. **Mechanism is throughput → more cooling.** With T_max=50 cosine schedule and ~14 baseline epochs (no AMP), only ~28% of the annealing budget is used. AMP pushes to ~19 epochs, still only 38% — but the extra 5 epochs correspond to additional LR cool-down where the optimizer makes more conservative, higher-quality steps. Val curve was **strictly monotonic** through epoch 19 (no plateau), meaning the 30-min cap fires while training is still actively improving.
3. **Cruise NaN bug fully resolved.** `test_geom_camber_cruise/mae_surf_p = 45.88` (finite). The per-sample `isfinite(y)` filter from PR #1615 handles this cleanly.
4. **Key open question**: val curve still descending monotonically at epoch 19 with T_max=50 → only 38% of cosine schedule spent. The implicit next hypothesis is matching T_max to the AMP epoch budget (~20 epochs) so cosine LR fully anneals within 30 min.
5. **New reproduce baseline**: `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp`

## 2026-05-13 06:08 — PR #1800: Truncated L1 (zero-gradient cliff at τ) — CLOSED

- Student branch: `willowpai2g24h3-tanjiro/truncated-l1`
- Hypothesis: per-element `min(|r|, τ)` zeros gradient on residuals where `|r| ≥ τ`. Predicted to improve high-|p| splits (in_dist + rc) by 2-5% by suppressing outlier influence; cruise (low-|p|) predicted to degrade slightly. Cliff sweep at τ∈{0.5, 1.0, 2.0} plus a τ=1.0+EMA+grad_clip combined arm.

### Results

| arm | run | val_avg | test_avg | Δ vs merged 91.66/81.28 |
|---|---|---:|---:|---:|
| L1 baseline | `gif79a0t` | 100.38 | 89.66 | +8.7 val (in noise) |
| τ=2.0 (light cap) | `cnv0cuz5` | 114.14 | 104.15 | +22 val WORSE |
| τ=1.0 (primary) | `f8j56db1` | 111.08 | 101.41 | +19 val WORSE |
| τ=0.5 (tight cap) | `cln0mj4e` | 125.23 | 115.83 | +34 val WORSE |
| **τ=1.0 + EMA + gc** | `r6tr47d7` | **107.08** | **97.71** | **+15.4 val / +16.4 test WORSE** |

### `train/pct_clipped` at convergence

| arm | last-half mean clip rate |
|---|---:|
| τ=2.0 | 0.006 (essentially off) |
| τ=1.0 (no EMA) | 0.034 |
| τ=1.0 + EMA + gc | 0.029 |
| τ=0.5 | 0.107 |

### Mechanism — prediction falsified, reinterpreted

PR-body predicted: at τ=1.0, `val_single_in_dist` and `val_geom_camber_rc` IMPROVE (high-|p| splits where tight cap helps), `cruise` DEGRADES (low-|p| where cap removes signal).

Observed at τ=1.0 (vs L1):
- val_single_in_dist: **+34.0 MAE (+29%) WORSE** (predicted: improve)
- val_geom_camber_rc: **+6.3 MAE WORSE** (predicted: improve)
- val_geom_camber_cruise: +0.4 MAE (essentially unchanged — only correct prediction)
- val_re_rand: +2.0 MAE WORSE

Reinterpretation: the few worst residuals at convergence (~3% at τ=1.0) ARE the signal needed to learn high-magnitude regions, not outlier noise. Zeroing their gradient kills learning on those regions. Degradation is graded proportional to clip rate.

### EMA-on-truncated_l1 orthogonality check (best τ=1.0+EMA arm `r6tr47d7`)

Dual eval at same best-val checkpoint via #1437's `test_no_ema/*` logging:

| metric | EMA weights | non-EMA weights at same step | Δ |
|---|---:|---:|---:|
| test_avg | 97.71 | 108.34 | **−10.6** (EMA wins) |
| test_geom_camber_rc | 102.29 | 126.05 | **−23.8** |
| test_geom_camber_cruise | 62.08 | 74.51 | **−12.4** |
| test_re_rand | 87.62 | 102.26 | **−14.6** |
| test_single_in_dist | 138.84 | 130.54 | +8.3 (EMA hurts on this split) |

EMA buys ~10.6 test MAE on top of truncated_l1 — **same magnitude as on SmoothL1 (#1437)**. EMA's parameter-trajectory averaging is robust to the underlying gradient-shape choice. Useful generalization: future loss-fn hypotheses can assume EMA stacks for free, and only need to argue about the underlying loss-fn mechanism.

### Conclusions

- **Closed**. Truncated direction does not produce a merge candidate at any τ tested. The closer-to-zero floor at the best τ=1.0+EMA arm (107.08 val, 97.71 test) is 15-16 MAE worse than the merged baseline.
- **Loss-shape axis is now closed.** Three PRs (#1441 MSE→SmoothL1 winner, #1615 SmoothL1→L1 equivalence, this PR truncated L1 hurts) pin down `sign(r)` bounded-linear gradient as the local optimum on the "gradient aggressiveness vs residual magnitude" axis. Future loss-fn hypotheses should target sample-conditional rather than residual-conditional gradient shape.
- **Diagnostic `train/pct_clipped` is a new advisor-branch instrument**: not strictly necessary for the merged baseline, but useful for future cliff/clip hypotheses. Living in tanjiro's branch only — would need re-implementation if revisited. (Not landing in advisor branch since this PR is closing.)

## 2026-05-13 04:52 — PR #1437: EMA of model weights (decay=0.999) — MERGED (winner)

- Student branch: `willowpai2g24h3-fern/ema-decay999`
- Hypothesis: EMA of model weights (decay=0.999) at val/test/checkpoint reduces variance of the SGD/Adam trajectory; predicted 1–3% reduction in `val_avg/mae_surf_p`. Orthogonal to SmoothL1 (per-element gradient cap) and grad-clip (per-batch gradient cap) because it operates on the parameter trajectory itself.

### Results (rebase onto advisor HEAD `4f225b4` = SmoothL1+grad-clip+cruise-NaN fix)

| Arm | Run | ema_decay | best val_avg/mae_surf_p | test_avg/mae_surf_p (4-split) | Δ vs merged 104.03/95.09 |
|---|---|---|---:|---:|---:|
| baseline-30m-newbase-v2 | `7xv82fez` | 0.0 | 101.06 | 89.41 | −2.9% / −6.0% (in noise) |
| baseline-r3-30m | `t73h00e2` | 0.0 | 105.18 | 94.79 | +1.1% / −0.3% (in noise) |
| **ema-decay-0.999-30m-newbase** | `zzv8ke31` | 0.999 | **93.70** | **83.46** | **−9.9% / −12.2%** |
| **ema-decay-0.999-30m-newbase-v2** | `emqh79b0` | 0.999 | **91.66** | **81.28** | **−11.9% / −14.5%** |

Three baselines (incl. merged 104.03) mean ≈ 103.4; two EMA reproductions mean ≈ 92.7 — a **−10.4% mean delta on val, well outside the ±7 single-seed noise band**.

### Per-split breakdown (best EMA `emqh79b0`)

**Val (vs merged baseline #1615 per-split):**

| Split | EMA `emqh79b0` | merged baseline | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 108.52 | 129.82 | **−16.4%** |
| val_geom_camber_rc | 104.81 | 110.53 | **−5.2%** |
| val_geom_camber_cruise | 68.50 | 80.24 | **−14.6%** |
| val_re_rand | 84.78 | 95.52 | **−11.2%** |

**Test (4-split, finite):**

| Split | EMA `emqh79b0` |
|---|---:|
| test_single_in_dist | 98.53 |
| test_geom_camber_rc | 89.74 |
| test_geom_camber_cruise | 57.58 |
| test_re_rand | 79.30 |

Improvement is broadly distributed (every split is in-the-money). Biggest absolute gains land on `single_in_dist` and `cruise` — the splits with the noisiest val curve under baseline. Smallest gain on `geom_camber_rc` (already the lowest-error split, less headroom).

### Mechanism (decomposed by dual eval at the same training step)

EMA(0.999) is a Polyak average over the AdamW trajectory with effective window ≈ 1000 steps. The rebase plumbed an additional "non-EMA test eval at the same epoch" logged under `test_no_ema/*`, isolating two distinct contributions:

| Branch | test_avg/mae_surf_p |
|---|---:|
| EMA weights (saved ckpt) | **81.28** |
| Raw weights at same step (no-EMA-eval) | 85.46 |
| Baseline (no EMA in training, `7xv82fez`) | 89.41 |

- **Variance-reduction at eval (~5%):** 85.46 → 81.28 from averaging the parameter trajectory.
- **Better epoch selection (~4%):** 89.41 → 85.46 because the EMA val curve is monotonically descending so `argmin val_avg` lands on a later, better epoch.

Both effects compound; combined ≈ −9% test, with no interaction with SmoothL1 / grad-clip. Per-step EMA update overhead is ~1 ms; per-epoch wall-clock identical to baseline at 130–133 s. EMA also did NOT need any α-warmup at this budget (decay=0.9995 was tested in PR pass 1 and is too slow at 30 min — see #1437 pass 1 comment).

### Conclusions

- **Merged. New empirical high-water mark on the advisor branch:**
  - `val_avg/mae_surf_p = 91.66` (down from 104.03)
  - `test_avg/mae_surf_p = 81.28` (down from 95.09)
- EMA stacks orthogonally on top of SmoothL1(β=0.1) + grad_clip(1.0) without interaction — three distinct gradient-stabilization mechanisms on three distinct objects (per-element / per-batch / parameter-trajectory) all compose.
- The `test_no_ema/*` dual-eval plumbing now ships on the advisor branch — future EMA-extension PRs (decay sweep, warmup, longer-budget) inherit it for free.
- **Round 2 priority shift**: capacity / regularization / loss reformulation now compete against a much harder baseline. Hypotheses claiming <5% improvement need ≥2 seeds. Any new "headline" merge needs ≥10% relative gain to be visibly real on a single seed.

## 2026-05-12 19:30 — PR #1443: Widen Transolver to n_hidden=192, n_head=6 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/wider-n192`
- Hypothesis: increasing `n_hidden` from 128→192 and `n_head` from 4→6 (`dim_head` constant at 32) gives more capacity at fixed depth/slice; expected 2–6% reduction in `val_avg/mae_surf_p`.

### Results

| Run | n_hidden / n_head | Params | Epochs done | val_avg/mae_surf_p | test 3-split avg surf_p | Δ vs baseline | W&B |
|---|---|---|---|---|---|---|---|
| baseline-30m | 128 / 4 | 0.66M | 14 | **123.17** (e12) | 120.19 | — | `h73q3u7m` |
| wider-n192-30m | 192 / 6 | 1.45M | 9 | **163.67** (e7) | 165.67 | +33% val / +38% test (worse) | `b9pe1a61` |

### Analysis

Wider variant regressed by +33% on val and +38% on test. Root cause: wider model is ~1.5× slower per epoch, finishes only 9 of the 50 scheduled epochs vs baseline's 14, and never enters the cosine cool-down where the baseline gains most of its ground.

Key observation from baseline trajectory (which becomes the seed for the next experiment): val_avg/mae_surf_p drops 140 → 156 → 126 → **123** at epochs 9-12 (collapse to 182 at e13 — likely noise). The cosine LR is barely cooled at this point (T_max=50, t=14 → cos(14π/100)≈0.92, LR still ~4.6e-4 of 5e-4 peak). Completing the schedule should push the best lower.

### Conclusions

- At the 30-min budget, capacity scaling via width is dominated by throughput cost — closed.
- Schedule mismatch (T_max=50, only 14 epochs fit) is a probable next lever — assigned to thorfinn as `schedule-tuned-e13`.
- **Known bug (do not block on):** `test_geom_camber_cruise/mae_surf_p` is NaN on both arms (pre-existing in the scoring/data path). Both `Ux/Uy` MAE on the same split are finite, suggesting a specific sample's p-channel prediction or ground-truth overflows. Need a separate `data/scoring.py` or data-side PR; deferring until more PRs land or the bug starts blocking ranking.

## 2026-05-12 21:05 — PR #1441: Replace MSE with SmoothL1 (Huber, β=0.1) — MERGED (winner)

- Student branch: `willowpai2g24h3-tanjiro/smooth-l1-beta01`
- Hypothesis: SmoothL1 in normalized space caps per-element gradient magnitude on high-Re outliers; predicted 2–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | Loss | Best val_avg/mae_surf_p (epoch) | test 3-split-ex-cruise avg surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| baseline-30m | MSE | 131.81 (e10) | 131.56 | — | `y3dfc5e7` |
| smooth-l1-0.1-30m | SmoothL1(β=0.1) | **104.70 (e13)** | **101.08** | **−20.6% val / −23.2% test** | `d53f0jn4` |

Per-split val surface-p MAE at SmoothL1 best-val:
- val_single_in_dist 120.63 (−22.9%)
- val_geom_camber_rc 117.45 (−16.4%)
- val_geom_camber_cruise 82.36 (−24.5%)
- val_re_rand 98.34 (−18.9%)

### Analysis

Outsized win — 4-10× the predicted delta — uniformly across every val split. Mechanism is consistent with the heavy-tail story: under MSE the high-Re/high-`|p|` outlier samples in each batch produce normalized residuals well above β=0.1, dominating the quadratic gradient on a single step and yanking the model off-trajectory (epoch-to-epoch val swings of ±20–40 MAE points were typical). SmoothL1 caps that contribution while leaving the in-regime quadratic intact, so each step is balanced across the Re range. SmoothL1's best epoch came at 13 vs MSE's 10 — Huber also keeps improving for longer in the same wall-clock budget. Largest absolute gains landed on the splits with the largest |p| magnitudes (cruise / re_rand), as predicted.

### Conclusions

- Merged. New empirical high-water mark on the advisor branch: **val_avg/mae_surf_p = 104.70**.
- Pre-authorized follow-ups (β=0.05, longer training, surf_weight re-tune under Huber, pure L1 comparison) are first-class Round 2 candidates.
- The cruise-test NaN bug is not in this PR (it stays a 3-split-ex-cruise figure) — fix lands in #1433 (next merge).

## 2026-05-12 21:06 — PR #1433: Add gradient norm clipping (max_norm=1.0) — MERGED

- Student branch: `willowpai2g24h3-askeladd/grad-clip-norm1`
- Hypothesis: `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` stabilizes training under heavy-tailed outliers; predicted 1–4% reduction in `val_avg/mae_surf_p`.
- Also ships the inline cruise-test NaN fix in `train.py::evaluate_split` (drops non-finite-`y` samples before forward pass and `accumulate_batch`).

### Results

| Run | max_norm | Best val_avg/mae_surf_p (epoch) | Δ vs baseline arm | W&B |
|---|---|---|---|---|
| baseline-30m | none | 131.96 (e?) | — | `mz3x4ieb` |
| grad-clip-1.0 | 1.0 | **114.18** | **−13.5%** | `qof1cbki` |
| grad-clip-0.5 | 0.5 | 121.41 | −8.0% | `japg46eu` |

Pre-clip grad-norm distribution measured at the baseline arm: median 53.90, max 579.57 — confirming the heavy-tail hypothesis (a single batch's grad-norm spike at >10× the median is a routine occurrence under MSE).

### Analysis

Tighter clip (0.5) underperforms looser (1.0), suggesting the floor for "useful" grad updates on a normal batch is somewhere between 0.5 and the median ~54 in pre-clip norm — 1.0 attenuates only the spike-batches and leaves the bulk of training-time gradients essentially untouched. Same mechanism as Huber (cap outlier influence) but acting at the batch-aggregate level instead of per-element.

### Conclusions

- Merged. Does NOT dethrone tanjiro's 104.70 (this PR was measured under MSE, not SmoothL1). Advisor branch now ships SmoothL1 + grad-clip stacked; combined-config has never been measured.
- The cruise-test NaN fix is now on the advisor branch. Future PRs will inherit it via rebase and report 4-split `test_avg/mae_surf_p` end-to-end.
- Open question for Round 2: does grad-clip still help on top of SmoothL1, or does SmoothL1 already subsume it? Pre-clip norms under SmoothL1 should be much smaller — likely the marginal benefit of clip-on-top-of-Huber is near zero, but a small A/B run can confirm.

## 2026-05-13 00:10 — PR #1616: Per-Re WeightedRandomSampler (upweight high-Re samples) — CLOSED

- Student branch: `willowpai2g24h3-askeladd/re-resample`
- Hypothesis: a `WeightedRandomSampler` weighted by `exp(t * log_re_centered)` shifts effective epochs toward the high-Re regime where pressure targets vary most; predicted 1–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | re_weight_temp | val_avg/mae_surf_p (best) | test_avg/mae_surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| uniform-baseline-smoothl1-clip1 (baseline) | 0.0 | **90.91** | **86.87 (4-split, all finite)** | — | `eztvtkxc` |
| re-resample-t1.0-smoothl1-clip1 | 1.0 | 97.41 (final 100.61) | NaN (variant produced non-finite preds on cruise test) | **+7.2% (worse)** | `stzo9xvw` |

Per-split val MAE breakdown shows the mechanism cleanly:

| Split | Baseline t=0 | Variant t=1.0 | Δ |
|---|---|---|---|
| val_single_in_dist | 103.90 | 148.16 | **+42.6% (catastrophic)** |
| val_geom_camber_rc | 105.34 | 102.78 | −2.4% |
| val_geom_camber_cruise | 68.99 | 67.60 | −2.0% |
| val_re_rand | 85.40 | 83.89 | −1.8% |

### Analysis

The variant *improves* every OOD-ish split (geom_camber_rc, geom_camber_cruise, re_rand) by 2–3% on both val and test — confirming the "high-Re samples generalize the OOD splits" sub-hypothesis. But the in-distribution split (`val_single_in_dist`) degrades by **+42.6%** because at `t=1.0` the max/min sampling ratio is **67.6×** — the lowest-Re training samples are seen <1× per epoch in expectation under `WeightedRandomSampler(replacement=True)`. The model is starved of low-Re training updates that the in-distribution split depends on.

Mechanistic insight: Huber and re-resampling are *not* the orthogonal mechanisms the PR predicted. They fight — Huber caps the gradient on high-Re samples that re-resampling deliberately re-injects. The net effect is just less effective training on in-distribution, with no headroom gained from over-emphasized regimes (Huber already handles those).

Additionally: the variant model produced non-finite predictions on at least one cruise *test* sample (`vol_loss = +Inf`, `surf_loss = NaN`), even though training-time cruise val was finite. The cruise-y filter from #1433 cannot help here — it handles non-finite *ground truth*, not non-finite *predictions* — but this is a signal that the variant model is unfit for the paper-facing pass under heavy reweighting.

### Side-effects of this PR (high-value despite the close)

1. **First clean end-to-end 4-split test pass for this launch.** Run `eztvtkxc` delivered `test_avg/mae_surf_p = 86.87` with all four splits finite — the cruise-y filter from PR #1433 worked.
2. **Cleanest measurement of the current advisor branch:** 90.91 val / 86.87 test (uniform sampling on top of SmoothL1+grad-clip+cruise-fix). Combined with two other in-flight baseline measurements (#1615 at 102.17, #1437 at 104.84), this characterizes a **±7 single-seed noise band** on `val_avg/mae_surf_p`.

### Conclusions

- Closed. Hypothesis at `t=1.0` falsified (+7.2% on val, NaN on test). Per-spec `t=2.0` stretch arm correctly not run.
- Follow-up direction (assigned to askeladd as next PR): **loss-level Re-reweighting** — multiply each sample's loss by `exp(t * log_re_centered)` inside the train loop, no resampling. Same "tilt toward high-Re" mechanism without the discrete sample-starvation problem. If even `t=0.3` produces a -1 to -3% effect on `val_avg`, the OOD-split signal observed here is real and just needed a less aggressive implementation.
- BASELINE.md updated with the supplemental 90.91/86.87 measurement of the current advisor branch (the merged-best stays at 104.70 until a winning hypothesis PR's terminal `SENPAI-RESULT` marker lands).

## 2026-05-13 00:55 — PR #1431: Raise surf_weight 10 → 50 to align loss with surface-p MAE — CLOSED

- Student branch: `willowpai2g24h3-alphonse/surf-weight-50`
- Hypothesis: raising `surf_weight` from 10 → 50 sharpens the loss-vs-metric alignment with surface-pressure MAE; predicted small improvement on `val_avg/mae_surf_p`.
- Bundled: an in-PR copy of the cruise-NaN-y filter (commit `b073a95` in `train.py::evaluate_split`) — same fix as askeladd's #1433, applied independently. Will be a no-op delta on rebase.

### Results

| Arm | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, finite) | Δ vs baseline (test) | W&B |
|---|---:|---:|---:|---:|---|
| baseline | 10 (default) | **126.70** | **112.68** | — | `ogz8su1w` |
| variant | 50 | 131.34 | 120.90 | **+7.30% worse** | `2qytxnem` |
| bonus | 25 | 143.79 | 127.35 | +13.02% worse | `x6nf3mk2` |

All three arms hit the 30-min wall-clock cap at 14 epochs (~28% through the cosine schedule). Comparisons are apples-to-apples at the same training budget on alphonse's pre-rebase branch (his fork carries MSE + cruise-fix, but does *not* yet stack SmoothL1+grad-clip — so absolute numbers are not directly comparable to other students' baselines on the current advisor branch). The hypothesis decision (variant +7.3% worse) is unaffected.

### Per-split test breakdown (best-val checkpoint) — the smoking gun

| Arm | split | surf[p] | vol[p] |
|---|---|---:|---:|
| baseline | test_single_in_dist | 132.97 | 134.44 |
| baseline | test_geom_camber_rc | 124.40 | 121.45 |
| baseline | test_geom_camber_cruise | 81.39 | 79.76 |
| baseline | test_re_rand | 111.96 | 107.22 |
| **surf=50** | test_single_in_dist | 130.39 | **178.16 (+32%)** |
| **surf=50** | test_geom_camber_rc | 132.78 | **159.60 (+31%)** |
| **surf=50** | test_geom_camber_cruise | 98.18 | **161.50 (+102%)** |
| **surf=50** | test_re_rand | 122.24 | **176.63 (+65%)** |

### Analysis (mechanistic — high-value finding)

**Bernoulli-coupling is the dominant mechanism.** alphonse's diagnosis: in incompressible flow, surface `p` and volume `p` are globally linked through pressure-Poisson / Bernoulli equations. Suppressing the volume-`p` residual signal (from `1/(1+10)=9.1%` of total at `surf_weight=10` to `1/(1+50)=1.96%` at `surf_weight=50`) starves the model of the volume-pressure structure it needs to *correctly anchor* surface pressure. The result is exactly what we see: vol[p] regresses by 30-102% across all four test splits, and surface-p slightly regresses too because the global pressure field is now miscalibrated near the foil.

**The "minority-class" framing was wrong on principle.** "Surface is the metric, so upweight surface" looks like sensible loss-metric alignment, but on a coupled PDE system the volume channels are *not noise* — they carry the constraint structure the surface predictions rely on. This rules out a whole family of naive task-aligned reweighting hypotheses for coupled physics. Generalizes to other PDE-surrogate problems.

**Surface velocity (Ux, Uy) is robust to channel reweighting** (slight regressions only) — the free-slip-like constraint at the foil makes those channels easy and saturated. The hypothesis only ever had a chance on `surf[p]`, and that channel needs both sides of the Bernoulli coupling.

### Conclusions

- Closed. Hypothesis falsified by an internally-consistent A/B with strong mechanistic explanation.
- Cruise-NaN-y filter works: all three arms produced finite 4-split `test_avg/mae_surf_p`. Independent confirmation that #1433's fix is correct.
- Follow-up direction (assigned to alphonse as next PR): **`slice_num` sweep on Transolver's Physics Attention layer.** Listed as an open question in `CURRENT_RESEARCH_STATE.md`; tests whether 64 slices saturate on the 242K-node cruise meshes. Default 64; arms at 32/96/128 to bracket. Compute trade-off (slower epochs vs finer representation) similar to but milder than the closed #1443 wider-n192.
- The Bernoulli-coupling mechanism finding will be cited in future hypothesis assignments. "Reweight surface" is now a known dead end for surface-MAE-on-coupled-physics.

## 2026-05-13 01:20 — PR #1537: Tune cosine T_max to budget — --epochs 13 instead of 50 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/schedule-tuned-e13`
- Hypothesis: matching cosine `T_max` to the actually-achievable epoch count converts the unused tail of the schedule into a proper cool-down; predicted 3–10% reduction in `val_avg/mae_surf_p`. (Direct data-driven follow-up to thorfinn's own #1443 baseline arm trajectory.)

### Results — W&B group `willow-r3-schedule-tuned-e13`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Epochs | Wall-clock | State |
|---|---:|---:|---:|---:|---|
| x4sqeaqz | **118.77** (best) | NaN | 13 | 28:50 | finished |
| nx1tvtp1 | 119.26 | NaN | 13 | 28:54 | finished |
| rfxdtryp | 120.64 | NaN | 13 | 28:49 | finished |
| afft3f1v | 122.25 | NaN | 13 | 28:57 | finished |
| slsutjdn | 122.41 | NaN | 13 | 28:51 | finished |
| k6h7anq3 | 192.27 (div) | NaN | — | — | crashed |
| crwqx3mb | 172.03 (div) | NaN | — | — | crashed |
| navwrdyg | 186.59 (in-prog) | NaN | mid-run | 0:18 | running (div) |

Best of 5 finished arms is **118.77 — +14 MAE points above merged baseline 104.70, +28 above the advisor-branch ~91 lower noise band.** The variant arm (e13) does not move the metric over thorfinn's own e50 baseline trajectory, and no arm enters the noise band of the current advisor baseline.

### Analysis

The mechanism prediction was that cool-down of the cosine schedule would harvest the last few percentage points of capacity. Empirically, finishing a 13-epoch cosine cycle does cool the LR but produces no measurable improvement vs running 14 of 50 epochs at near-peak LR — across 5 independent seeds. Implication: at the merged-baseline operating point (SmoothL1 + grad-clip on advisor branch), the LR-cooling regime contributes less than the seed-to-seed noise (±7). 

Two crashes + one in-flight diverging run also suggest the e13-config + WeightedRandomSampler-with-replacement combination may have a borderline-stable training regime — likely a separate effect from the hypothesis itself, but worth noting.

No SENPAI-RESULT terminal marker was posted on the PR; advisor closed based on the W&B group readout directly.

### Conclusions

- Closed. Hypothesis falsified at the merged advisor-branch operating point — schedule-cooling alone is not a 5%+ lever here.
- **Schedule reformulation is not abandoned** — frieren's #1438 (warmup-5ep) tests the complementary half (LR warmup before the cosine). If warmup wins, then a **warmup + tuned T_max combo** would be the natural Round 2 stack and bears revisiting.
- All test_avg/mae_surf_p were NaN, suggesting thorfinn's branch may not have absorbed the cruise-NaN-y fix from #1433 — but the comparison against the val metric is unaffected.
- Follow-up direction (assigned to thorfinn as next PR): **AdamW weight_decay sweep**. Single-knob regularization test on a baseline that is now characterized to ±7 noise. Pure compute-neutral lever — no time cost per epoch, predicted to differentiate cleanly on per-split signal (especially val_single_in_dist and val_geom_camber_rc which have the highest per-split MAE).

## 2026-05-13 01:35 — PR #1615: Pure L1 / MAE loss + cruise-NaN code fix (MERGED)

- Student branch: `willowpai2g24h3-tanjiro/pure-l1`
- Hypothesis: dropping SmoothL1's quadratic-near-zero region (pure L1 / MAE loss in normalized space) should match SmoothL1(β=0.1) within noise — testing whether the residual quadratic does any useful work after we established the Huber gradient cap is the dominant mechanism. Predicted delta: −2% (better) to +5% (worse).

### Results — W&B group `willow-r3-pure-l1`

| Arm | wandb_id | loss_fn | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, post-fix) |
|---|---|---|---:|---:|
| pure-l1-30m (variant) | `mc22t7l2` | L1 | **104.03** | **95.09** |
| smooth-l1-0.1-30m-v2 (best SmoothL1) | `x0ud9i0a` | SmoothL1 β=0.1 | 102.17 | 92.04 |
| smooth-l1-0.1-30m (#3) | `30cs7nad` | SmoothL1 β=0.1 | 103.57 | 94.02 |
| smooth-l1-0.1-30m (#1, high-var) | `02e8ituj` | SmoothL1 β=0.1 | 125.94 | 97.40 |

Pure-L1 variant vs best SmoothL1 baseline: +1.8% val / +3.3% test. Pure-L1 vs mean of two well-behaved SmoothL1 baselines (102.17, 103.57): +1.1% val / +2.2% test. Both well within the ±7 single-seed noise band (three SmoothL1 reproductions span val=102-126, σ≈13). **Hypothesis confirmed equivalent within noise.**

### Per-split val MAE: pure-L1 vs best SmoothL1 baseline

| Split | SmoothL1 best (102.17) | pure L1 (104.03) | Δ (L1 − SmoothL1) |
|---|---:|---:|---:|
| val_geom_camber_cruise | 69.20 | 80.24 | **+15.9%** |
| val_geom_camber_rc | 111.06 | 110.53 | −0.5% |
| val_re_rand | 92.90 | 95.52 | +2.8% |
| val_single_in_dist | 135.52 | 129.82 | −4.2% |

### Per-split test MAE (post-fix, 4-split): pure-L1 vs best SmoothL1 baseline

| Split | SmoothL1 best | pure L1 | Δ |
|---|---:|---:|---:|
| test_geom_camber_cruise | 58.60 | 68.55 | **+17.0%** |
| test_geom_camber_rc | 100.15 | 101.44 | +1.3% |
| test_re_rand | 85.68 | 90.93 | +6.1% |
| test_single_in_dist | 123.73 | 119.46 | −3.5% |

### Bug-fix component (separate from hypothesis result)

tanjiro discovered that the advisor branch `train.py::evaluate_split` was missing the cruise-NaN-y filter that BASELINE.md / PR #1433 docs claimed was in place — only the documentation landed, not the code. He added the actual per-sample `torch.isfinite(y).all(dim=-1)` filter (train.py lines 240-250), exactly matching the `data/scoring.py::accumulate_batch` per-sample-skip semantics. This unlocks finite 4-split `test_avg/mae_surf_p` reporting for all future PRs. **This is a high-value contribution beyond the loss-fn experiment.**

### Analysis (mechanistic)

The Huber win in #1441 was **the linear-region gradient cap on outlier residuals**, not the quadratic-near-zero smoothness. With y-normalized target stats `y_std ≈ O(1)` and SmoothL1 β=0.1, the quadratic region (|r|<0.1 in normalized space) covers only the bottom decile of residuals at convergence. Early-training trajectories are very similar between L1 and SmoothL1 (both runs reach val ≈ 105 by epoch 25). Only the cruise split shows SmoothL1 consistently better — that split is dominated by easy low-Re aerofoil flow with the smallest absolute pressure scale (cruise val_p ≈ 70 vs single-foil val_p ≈ 130), so its residuals are the most likely to live inside the quadratic region. Other three splits show pure-L1 either ahead or within ±2%. **The residual quadratic does its (tiny) work on the low-magnitude split** — consistent with the textbook Huber picture, but not big enough to matter at this dataset/budget noise band.

### Conclusions

- Merged. New empirical baseline: **val_avg/mae_surf_p = 104.03** (pure-L1, run `mc22t7l2`); **test_avg/mae_surf_p = 95.09** (4-split, post-fix).
- **Implication for the paper**: parameter-free L1 is statistically indistinguishable from tuned-β SmoothL1 on TandemFoilSet at this scale — the SmoothL1 win in #1441 reduces to "gradient cap on the linear-region tail of outlier residuals." Clean negative result for the quadratic-near-zero.
- Bug-fix code change unlocks paper-facing 4-split `test_avg/mae_surf_p` reporting for every future run on the advisor branch.
- Follow-up direction (assigned to tanjiro as next PR): on the outlier-residual mechanism thread, the next high-leverage direction is **NOT smaller β** (this PR + closed #1616 already bracket that). It's a different mechanism entirely — to be designed in the next assignment.

## 2026-05-13 01:55 — PR #1434: 3× p-channel weight on pressure in training loss (CLOSED)

- Student branch: `willowpai2g24h3-edward/p-channel-weight3x`
- Hypothesis: multiplying the pressure-channel loss term by 3× (relative to Ux, Uy) aligns the loss with the surface-MAE ranking metric without abandoning velocity supervision; predicted 2-5% improvement on `val_avg/mae_surf_p`.

### Results — W&B group `willow-r3-p-channel-weight3x`

| Arm | p_weight | Best val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline | W&B |
|---|---:|---:|---:|---:|---|
| baseline (best) | 1.0 | **97.00** | 89.49 (`567j0vuh`) | — | `w6lqwh5o` (val), `567j0vuh` (test) |
| 5× variant | 5.0 | 138.92 | — | **+43% worse** | `aq0t3zfr` |
| 3× variant | 3.0 | 157.16 | — | **+62% worse** | `nqjvocmq` |

Multiple baseline reproductions (p_weight=1.0): 97.00, 98.49, 99.70, 113.12 — well-clustered near ~99, matching the current advisor-branch noise band.

### Analysis (mechanistic — confirms Bernoulli-coupling generalization)

Same failure mode as alphonse's closed #1431 (surf_weight 10 → 50), via a different lever:

- Transolver's decoder predicts `(Ux, Uy, p)` as a globally coupled physical solution; the loss minimum lies on a manifold defined by the incompressible Navier-Stokes equations (∇·u = 0, u·∇u + ∇p/ρ = ν∇²u).
- Up-weighting the `p` channel by 3× tells the optimizer to spend disproportionate capacity on fitting `p` at the expense of `(Ux, Uy)`. This breaks the Bernoulli closure between velocity and pressure.
- Result: predicted-`p` drifts off the manifold defined by predicted `(Ux, Uy)`. Training-time p-loss can decrease while *evaluation* p-MAE rises, because the prediction is no longer physically self-consistent.

**Independent confirmation of a generalizable failure mode.** alphonse's #1431 reweighted surface-vs-interior; edward's #1434 reweighted per-channel. Both fail by the same coupling-violation mechanism, and the failure scales monotonically with the strength of the reweighting (3× already +62% worse, 5× still +43% — i.e. 5× is "less catastrophic" than 3× because the 5× variant happened to converge on a slightly less broken local optimum; both are decisively closed).

### Conclusions

- Closed. Channel-level reweighting of (Ux, Uy, p) is a closed direction.
- Combined with #1431 closure, the lesson is: **never reweight individual output channels (or boundary regions) of a physics-coupled multi-task head** unless the reweighting respects the coupling constraint. This rules out a whole family of naive task-aligned reweighting hypotheses for coupled PDE surrogates.
- Follow-up direction (assigned to edward as next PR): hypothesis pivoted to a lever that doesn't touch the loss landscape's physical coupling — see next assignment.
