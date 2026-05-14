# SENPAI Research Results — charlie-pai2g-48h-r1

## 2026-05-14 14:23 — PR #1602: grad_clip=2.0 on bf16+OneCycle@25ep baseline ❌ CLOSED (negative)

- **Student branch:** `charliepai2g48h1-fern/grad-clip-l1`
- **Hypothesis:** gc=2.0 + OneCycle (winner from round-2 re-run on old baseline) carries over to new bf16+25ep baseline.

### Result (2 seeds, vs tanjiro #1405 baseline 73.295)

| Arm | val_avg (best/mean) | test_avg 4-split (best/mean) | Δ vs baseline |
|-----|--------------------:|-----------------------------:|--------------:|
| gc=2.0+bf16+OneCycle@25ep | **76.359 / 77.018** | **66.474 / 66.642** | **+3.72 val (+5.1%)** |
| **Baseline (#1405)** | — / **73.295** | — / **63.911** | — |

Per-split (mean 2 seeds): cruise +1.0, re_rand +2.6, single_in_dist **+10.7** (+13.4%), rc +0.6

Artifacts: `models/model-gc2-onecycle-bf16-25ep-20260514-130421/` and `models/model-gc2-onecycle-bf16-25ep-20260514-134136/` on student branch.

### Action: CLOSED — >5% val regression; direction dropped

**Mechanism characterised:** Mean pre-clip norm ~38; gc=2.0 clips every batch at ~5% of natural magnitude — pure per-step renormaliser, not spike cap. On the OLD baseline (fp32+14ep), this renormalisation regularised effectively. On the NEW baseline (bf16+25ep), the longer LR plateau (~10-12 high-LR epochs) already supplies the same regularisation — clipping on top over-regularises, specifically on `val_single_in_dist` (+13.4%). The effect has fully inverted: from the best relative gain to the worst split.

**Boundary finding (useful for paper):** Grad-clip helps under shorter, cooler schedules (≤14ep cosine/OneCycle); hurts under longer, hotter schedules (25ep OneCycle + bf16). The binding constraint that clipping was solving no longer exists in the new recipe.

**New assignment:** #2935 fern → geometric z-flip augmentation (next)

---

## 2026-05-14 14:33 — PR #2914: Transolver depth n_layers=6/7 on bf16 baseline ❌ CLOSED (compute-budget failure)

- **Student branch:** `charliepai2g48h1-askeladd/transolver-depth-increase-bf16`
- **Hypothesis:** Deeper Transolver (n_layers 5→6/7) improves capacity; bf16 VRAM headroom makes this tractable.

### Result (vs tanjiro #1405 baseline 73.295)

| Arm | n_layers | VRAM | s/epoch | realized epochs | best epoch | val_avg | Δ vs baseline |
|-----|----------|------|---------|-----------------|------------|---------|---------------|
| A | 6 | 38.9 GB | ~117–251 | 14/25 | 13 | 98.735 | **+34.7%** |
| B | 7 | 44.9 GB | ~135 | 14/25 | 14 | 93.611 | **+27.7%** |
| Baseline | 5 | ~33 GB | ~97 | 19/25 | 19 | **73.295** | — |

Artifacts: `models/model-transolver-layers6-bf16-20260514-125952/` and `models/model-transolver-layers7-bf16-20260514-135156/`

### Action: CLOSED — compute-budget failure, not capacity failure

Both arms realized only 14 epochs vs baseline 19 (+20-40% per-step overhead reduced realized epochs). LR at best epoch = 9.6e-4, still mid-plateau (not the fine-tuning tail). Baseline gets to epoch 19 AND reaches LR~1e-9 wind-down. The architecture can't compensate for missing the LR decay regime.

Confirms the pattern established by #1381 (wider model): extra architecture cost under 30-min cap prevents OneCycleLR fine-tuning tail. Closing the depth lever for this budget.

**New assignment:** #2936 askeladd → eval-every-2 (save eval wall-time to unlock extra training epochs)

---

## 2026-05-14 14:35 — New assignment: PR #2936 askeladd → eval-every-2

- **Student:** charliepai2g48h1-askeladd, branch `charliepai2g48h1-askeladd/eval-every-2`
- **Hypothesis:** Eval every 2 epochs instead of every 1 epoch saves ~200s of wall-clock time (9-10 skipped eval calls × 20s) ≈ 2 extra training epochs in the OneCycleLR fine-tuning tail. Zero architecture/hyperparameter change.
- **Beat:** val_avg/mae_surf_p < 73.295

---

## 2026-05-14 14:25 — New assignment: PR #2935 fern → z-flip augmentation

- **Student:** charliepai2g48h1-fern, branch `charliepai2g48h1-fern/z-flip-augmentation`
- **Hypothesis:** Per-batch geometric z-flip (z→-z, AoA→-AoA, Uy→-Uy) with p=0.5 effectively doubles training data for free using TandemFoilSet's physical mirror symmetry. No compute cost, no architecture change. Expected -3 to -7 val pts, largest gains on `val_geom_camber_rc`.
- **Beat:** val_avg/mae_surf_p < 73.295

---

## 2026-05-13 00:10 — PR #1602: Gradient clipping sweep (0/0.5/1.0) ↩ SENT BACK for OneCycle re-run

- **Student branch:** `charliepai2g48h1-fern/grad-clip-l1`
- **Hypothesis:** `clip_grad_norm_(max_norm)` stabilizes training, reduces overfitting on `val_single_in_dist`; sweep 0/0.5/1.0.

### Result (on OLD L1-only baseline 94.291)

| Arm | grad_clip | val_avg/mae_surf_p | test_avg (3/4) | Δ vs control mean |
|-----|----------:|-------------------:|---------------:|------------------:|
| A1 control (run 1) | 0.0 | 91.373 | 89.293 | — |
| A2 control (run 2) | 0.0 | 90.965 | 89.022 | — |
| **B (winner)** | **1.0** | **89.196** | **88.320** | **-2.0%** |
| C | 0.5 | 94.184 | 92.442 | +3.3% (worse) |

Within-PR 2-seed control mean = 91.169 ± 0.20. gc=1.0 vs control delta = -2.0% val / -0.7% test.

### Action: SENT BACK — beaten by OneCycle baseline (85.615), but signal is real

**Why send-back not close:** gc=1.0 has a real within-PR -2.0% effect on L1+cosine. But ran with default `--lr 5e-4` (no OneCycleLR), so 89.196 is +4.2% worse than the merged 85.615 baseline.

**Re-run instructions issued:** drop gc=0.5 (strictly worse), sweep gc=1.0 vs gc=2.0 on OneCycleLR+L1 baseline. Tests whether clipping helps under OneCycle's 4× higher peak LR (where grad spikes should be more destabilizing).

**Key mechanism insight** (load-bearing for the re-run): `max_norm=1.0` is acting as a **per-step renormaliser** (mean pre-clip grad norm ~45-50), not a rare-spike cap. It's equivalent to a constant-magnitude / adaptive-direction update — closer to LARS/LAMB than to conventional clipping. This is why gc=0.5 is worse than no clipping (starves the optimizer of LR magnitude) and why higher LR may compensate.

---

## 2026-05-13 00:05 — PR #1605: asinh transform on pressure target ↩ SENT BACK for OneCycle re-run

- **Student branch:** `charliepai2g48h1-edward/asinh-pressure-target`
- **Hypothesis:** `asinh(p/scale)` compresses the heavy pressure tail without distorting low-magnitude regions, balancing L1 gradients across pressure ranges. Two scales: 100 (aggressive) vs 680 (≈σ_p, gentle).

### Result (on OLD L1-only baseline 94.291)

| Arm | scale | best ep | val_avg/mae_surf_p | test_avg (3/4) | Δ vs L1 baseline |
|-----|------:|--------:|-------------------:|---------------:|-----------------:|
| A | 100 | 14 | 89.893 | 88.904 | -4.66% |
| **B (winner)** | **680** | **14** | **88.643** | **86.505** | **-5.99%** |

Per-split val (Arm B): cruise=64.73 (-9.67%), re_rand=83.54 (-4.52%), single=105.45 (-4.49%), rc=100.84 (-6.27%). All four splits improved.

### Setup deviation: recomputed pressure stats on `asinh(p/scale)`

Edward caught a showstopper before launching: applying raw-`p` stats (mean=-129.22, std=679.45) on top of `asinh(p/scale)` collapsed `y_norm_p` to std ~0.003 (near-constant), shutting off pressure gradient. **1-epoch test of this broken version showed val_avg=264k** (vs 94.29). Fix (approved): recompute `y_mean[2]` and `y_std[2]` on `asinh(p/scale)` from the training set. With fix applied, `y_norm_p` mean~0/std~O(1) and inverse round-trip stable.

Recomputed stats logged to metrics.yaml:
- scale=100: post-asinh (mean, std) = (-0.2141, 1.5242)
- scale=680: post-asinh (mean, std) = (-0.1093, 0.5707)

### Action: SENT BACK — beaten by OneCycle baseline (85.615), but signal is real and clean

**Why send-back not close:** asinh@680 has a real -6.0% effect on L1+cosine. But ran with default `--lr 5e-4` (no OneCycleLR), so 88.643 is +3.5% worse than the merged 85.615 baseline.

**Re-run instructions issued:** single arm with scale=680 on OneCycleLR+L1 baseline.

**Compound expectation:** if asinh stacks with OneCycle, expect ~80-81 val_avg (L1 baseline 94.29 → OneCycle -9.2% → 85.6 → asinh -6% → ~80). If interaction is destructive (e.g. asinh's gentler gradients can't tolerate higher peak LR), arm may regress. Either outcome is informative.

---

## 2026-05-12 23:15 — PR #1601: EMA of model weights (decay 0.999 vs 0.9999) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-thorfinn/ema-model-weights`
- **Hypothesis:** EMA of model weights (exponential moving average) as a near-zero-cost regularizer provides a smoother ensemble of checkpoints, reducing overfitting on the 1499-sample training set and improving OOD generalization.

### Result

| Arm | EMA decay | best epoch | val_avg/mae_surf_p | test_avg (3/4) |
|-----|-----------|------------|---------------------|----------------|
| A | 0.999 | 14 | **94.014** | ~91.6 (est.) |
| B | 0.9999 | 14 | ~94.2 | — |
| Baseline (L1 only) | — | 14 | 94.291 | 91.859 |
| **New baseline (L1+OneCycle)** | — | 14 | **85.615** | 83.328 |

### Action: CLOSED — beaten by OneCycle baseline

**Root cause of failure:** EMA provides benefit when gradient updates are noisy and the optimizer oscillates near a minimum — the EMA smooths over the oscillations. L1 + OneCycleLR produces a **smooth monotone descent**: bounded L1 gradients + warm-up/wind-down schedule drive a clean trajectory with no oscillation to average over. EMA sees no gradient noise to exploit.

**Timing issue:** Thorfinn ran results against the old L1-only baseline (94.29). Arm A's val=94.014 is marginally better (+0.3%) than that old baseline, but:
1. The new OneCycleLR baseline (85.615) was merged while #1601 was in-flight
2. 94.014 is +9.8% worse than 85.615 — not a winner against the current best
3. EMA decay=0.9999 effective window ≈ 70 epochs >> 14 realized epochs; EMA barely converged

**Round-3 follow-up dispatched: #`TBD` (thorfinn, SAM optimizer)**
SAM (Sharpness-Aware Minimization) is the orthogonal lever: it reshapes the loss landscape rather than averaging weights, and is known to help small-dataset OOD generalization (Foret et al. 2021). Arms: rho=0.05 vs rho=0.1.

---

## 2026-05-12 23:05 — PR #1405: bfloat16 autocast + batch_size 8 + sqrt-scaled lr — sent back for rebase

- **Student branch:** `charliepai2g48h1-tanjiro/amp-bf16-batch8`
- **Hypothesis:** bfloat16 autocast + larger batch (4→8) + sqrt-scaled lr (5e-4→7e-4) doubles throughput → more realized epochs within the 30-min cap.

### Result (pre-rebase, on L1-only baseline)

| Arm | bf16 | batch | lr | epochs cfg | epochs done | val_avg/mae_surf_p | test_avg (3/4) |
|-----|------|-------|-----|------------|-------------|---------------------|----------------|
| Arm 1 (main hypothesis) | ✅ | 8 | 7e-4 | 15 | 15 | 100.77 | 89.61 |
| **Arm 2 (control)** | ✅ | 4 | 5e-4 | 25 | **19** | **82.70** | **73.06** |

**Arm 2 is a huge win**: val=82.70 is -12.3% vs L1 baseline (94.29). The surprise: throughput hypothesis failed (bs=8 gave no speedup — CPU/dataloader is the bottleneck), but running --epochs 25 let the cosine schedule consume its full tail (19 realized epochs vs 14 before), and that's where the gain came from.

### Why sent back (not merged yet)

PR has merge conflicts (CONFLICTING state) because PR #1355 already added the `--loss l1` branches that tanjiro also implemented locally. Additionally, **PR #1581 (frieren, OneCycleLR@2e-3) merged at 22:55 UTC with val=85.61**, which changes the baseline recipe. Tanjiro's Arm 2 result (82.70) was on cosine+lr=5e-4 with 19 realized epochs — it's not directly comparable to the new OneCycleLR baseline.

Sent back with instructions to:
1. Rebase and drop duplicate L1 implementation (keep only bf16 + eval safety changes)
2. Re-run on top of new baseline: Arm A (bf16 + OneCycle@2e-3 + 14ep), Arm B (bf16 + OneCycle@2e-3 + 25ep configured)

**Key scientific insight:** CPU/dataloader throughput is the per-epoch bottleneck, not GPU. On these irregular meshes (74K–242K nodes), `num_workers=4, prefetch=2` starves the GPU at both bs=4 and bs=8. The win from "more epochs" (19 vs 14 realized) is pure cosine-tail scheduling. This strongly motivates profiling the dataloader or using a pre-batched dataset before any further batch-size experiments.

---

## 2026-05-12 22:55 — PR #1581: L1 + OneCycleLR@peak=2e-3 (frieren) ✅ MERGED

- **Student branch:** `charliepai2g48h1-frieren/l1-onecycle-compound`
- **Hypothesis:** OneCycleLR (warmup→peak→cosine wind-down, per-batch stepping) compounded with L1 loss, at peak_lr=1e-3 and peak_lr=2e-3.

### Result

| Arm | peak_lr | val_avg/mae_surf_p | Δ vs L1 baseline | test_avg (3/4) | Δ vs test baseline |
|-----|--------:|-------------------:|------------------:|-------------------:|------------------:|
| **B (winner)** | 2e-3 | **85.615** | **-9.20%** | **83.328** | **-9.29%** |
| A | 1e-3 | 87.574 | -7.12% | 84.617 | -7.88% |
| Baseline | 5e-4 cosine | 94.291 | — | 91.859 | — |

Per-split val: cruise=66.44, rc=94.61, re_rand=81.89, single=99.52

### Action: MERGED as new baseline (val_avg=85.615)

Both arms beat baseline; Arm B wins. Key findings:
- **Compound stacks well**: loss change (→L1) gave ~57%; adding OneCycle@2e-3 gives another 9.2% on top
- **L1 tolerates higher peak LR**: L1 gradient is bounded at 1 in normalized units, so large raw steps don't explode. Arm B (2e-3) outperforms Arm A (1e-3) by 2.1pp additional
- **Schedule fully consumed**: best epoch=14/14 with LR wound to 10⁻⁹ — no truncated tail
- **`val_geom_camber_rc` biggest absolute improvement**: 107.6 → 94.6 (-12%), still the hardest split

Round-2 follow-up dispatched: #1667 (frieren, peak LR push to 3/4/5e-3 — monotone signal warrants pushing further).

---

## 2026-05-12 22:05 — PR #1399: Per-channel surface loss pressure weighting (3-arm corrected sweep) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-nezuko/surf-channel-pressure-weight`
- **Hypothesis:** Per-channel surface-loss weighting `CHANNEL_W=[1,1,k]` (boosting p relative to Ux/Uy) improves `val_avg/mae_surf_p` by directing gradients toward the ranking metric.

### Result — corrected 3-arm sweep (`.mean()` denominator, `surf_weight=10`)

All arms use MSE loss (round-1 dispatch, pre-L1-baseline).

| Arm | `CHANNEL_W` | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4) | val_geom_camber_rc |
|-----|-------------|---------|---------------------|---------------------------|---------------------|
| **C (winner)** | `[1,1,3]` | 14 | **120.448** | 118.886 | 128.573 |
| B   | `[1,1,2]`   | 14 | 120.992             | 120.960                   | 132.772             |
| A (control) | `[1,1,1]` | 14 | 122.961     | **118.579**               | 133.982             |

Artifacts: `models/model-baseline-control-sw10-20260512-192251/`, `models/model-surf-pw2-fixed-sw10-20260512-200712/`, `models/model-surf-pw3-fixed-sw10-20260512-205059/` on the student branch.

### Action: CLOSED — MSE-era results 28%+ worse than L1 baseline (94.29)

The absolute numbers are dominated by MSE-vs-L1: the merged L1 baseline (94.29) is 23% better than nezuko's Arm A control (122.96) using the same architecture and surf_weight. No arm can be merged.

**But the val signal is real and actionable:**
- Monotone improvement on `val_avg/mae_surf_p` as k rises: 122.96 → 120.99 → 120.45 (-2.0%)
- Strongest effect on `val_geom_camber_rc` (hardest OOD split): 133.98 → 132.77 → 128.57 (-4.0%)
- Signal holds across all four val splits — not a single-split fluke
- Test mean is inconclusive (non-monotone, within noise), consistent with nezuko's honest caveat about single-seed variance

### Round-2 follow-up dispatched: #1625

Same lever compounded with L1. Hypothesis: the ~2% val gain from channel weighting stacks on top of the ~57% gain from L1. Arms at k∈{2,3,5}.

---

## 2026-05-12 21:20 — Round-2 dispatch (5 PRs)

After establishing L1 as the new measured baseline (PR #1355, val_avg/mae_surf_p=94.29),
five round-2 PRs were dispatched to compound the L1 winner with orthogonal levers that
are cheap (no extra params, no compute hit) and target known weak spots of
short-budget training in normalized space.

| PR | Student | Lever | Arms |
|----|---------|-------|------|
| #1581 | frieren | L1 + OneCycleLR compound | peak_lr=1e-3 vs 2e-3 |
| #1582 | alphonse | surf_weight sweep on L1 | surf_weight=5 / 10 (control) / 20 |
| #1601 | thorfinn | EMA of model weights on L1 | decay=0.999 vs 0.9999 |
| #1602 | fern | Gradient clipping on L1 | max_norm 0 (control) / 0.5 / 1.0 |
| #1605 | edward | asinh transform on pressure target with L1 | scale=100 (aggressive) vs 680 (~σ_p, gentle) |

All five dispatched against the L1 baseline (`--loss l1` is the recipe-level
default in every PR body). The three still-running round-1 PRs (#1381 wider
askeladd, #1399 nezuko channel-weight corrected replan, #1405 tanjiro bf16)
remain in flight; if any beats 94.29 it will be merged ahead of the round-2
results.

---

## 2026-05-12 21:15 — PR #1385: Finer physics attention (slice_num 64→128, n_head 4→8) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-edward/slices128-heads8`
- **Hypothesis:** Doubling `slice_num` (64→128) and `n_head` (4→8) gives the
  Transolver physics attention finer slicing and better feature interaction.
  Predicted -2% to -5% on `val_avg/mae_surf_p`.

### Result

| Arm | slice_num | n_head | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4) |
|-----|-----------|--------|---------|---------------------|---------------------------|
| A   | 128       | 8      | 9       | 151.92              | 144.78                    |
| B   | 128       | 4      | 9       | 156.83              | 149.51                    |

### Action: CLOSED — 61% regression vs the new L1 baseline (94.29)

Both arms regressed massively. The finer attention slicing combined with
more heads roughly **doubled** attention compute per layer; under the
hard 30-min wall-clock cap, only 9 epochs of 15 ran — same schedule-mismatch
pathology as the deeper-net experiment (#1389). Even discounting the
schedule artifact, the gap to L1 baseline is too large for further sweeps
under the current epoch budget. Closing the slice/head lever for this arm.

---

## 2026-05-12 21:12 — PR #1389: Deeper Transolver (n_layers 5→8) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-fern/deeper-8-layers`
- **Hypothesis:** Going from 5 → 8 layers gives more iterations of slice
  attention → MLP refinement.

### Final result (after Arm C `--epochs 9` rerun)

| Arm | lr | epochs | best ep | val_avg/mae_surf_p |
|-----|------|--------|---------|---------------------|
| A   | 5e-4 | 9 / 15 | 8       | 153.48              |
| B   | 3e-4 | 9 / 15 | 8       | 147.40              |
| C   | 3e-4 | 9 / 9  | 8       | ~142–145 (schedule-matched, still 50%+ worse than L1) |

### Action: CLOSED — 42%+ regression vs L1 baseline

Arm C (cosine T_max=9 matching the realized epoch count) closed only ~3
points vs Arm B's schedule-mismatched 147.40. Even with the schedule
artifact removed, depth=8 is ~50% worse than the 94.29 L1 baseline at
this epoch count. Closing the depth lever for the current epoch budget;
re-test when total compute budget allows full 15-epoch realization of
deeper nets.

Reusable finding: more depth raises per-step compute by ~55% (so realized
epochs drop from 14 → 9 under the 30-min cap), and the cosine schedule
needs to be retuned to T_max=realized_epochs whenever the per-step cost
materially changes.

---

## 2026-05-12 21:08 — PR #1410: Multi-scale Fourier features for (x,z) coords ❌ CLOSED

- **Student branch:** `charliepai2g48h1-thorfinn/fourier-position-features`
- **Hypothesis:** Adding learned/fixed sinusoidal frequency encodings of (x,z)
  coordinates should give the network a richer geometric representation
  than the raw 2D coords alone, particularly helping the geometry-OOD splits.
  Predicted -3% to -8% on `val_avg/mae_surf_p`.

### Result

| Arm | n_freq_bands | scale_max | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4) |
|-----|--------------|-----------|---------|---------------------|---------------------------|
| A   | 8            | 10        | 13      | 105.05              | 102.78                    |
| B   | 16           | 32        | 12      | 109.20              | 106.94                    |

### Action: CLOSED — 11% regression vs L1 baseline (94.29)

Both Fourier-feature arms regressed against L1 baseline. The geometry-OOD
splits (`val_geom_camber_rc/cruise`) did not improve relative to the L1
baseline either — the synthetic frequency bands seem to add noise without
adding geometric information that the raw coords don't already provide
to the Transolver's slice attention.

This is informative: it suggests the Transolver's physics-aware attention
on raw coords is already extracting geometric structure effectively, and
the next geometric-representation experiment should attack the slicing
mechanism itself rather than the input coordinate encoding.

---

## 2026-05-12 20:50 — PR #1355: Smooth L1 / pure L1 loss vs MSE on normalized residuals ✅ MERGED

- **Student branch:** `charliepai2g48h1-alphonse/smooth-l1-loss`
- **Hypothesis:** L1-family losses (Smooth L1 / Huber β=1.0, pure L1) align
  training objective with the eval metric (MAE in original space), unlike MSE
  which over-penalizes outliers. Predicted -2% to -8% on `val_avg/mae_surf_p`.

### Result

| Arm | Loss | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 finite) | best ep |
|-----|------|---------------------|----------------------------------|---------|
| **B (winner)** | Pure L1 | **94.291** ⭐ | **91.859** | 14 |
| A | Smooth L1 / Huber β=1 | 97.791 | 94.393 | 14 |
| baseline MSE | MSE | 218.388 @ ep3 (partial) | — | — |

Metrics: `models/model-pure-l1-20260512-191540/{metrics.jsonl,metrics.yaml}` and
`models/model-charliepai2g48h1-alphonse-smooth-l1-huber-20260512-175942/{...}` on advisor branch.

### Action: MERGED as new baseline

Pure L1 is the winner by 3.6% over Smooth L1 (and ~15% over implied full-epoch MSE). The
metric-objective alignment argument is confirmed: pure L1 in normalized space ≈ MAE
in original space, so the training gradient points exactly at what we measure.

Key finding: Pure L1's advantage grows through training (Smooth L1 briefly leads at ep3 by
convergence smoothness, but Pure L1 overtakes by ep9 and dominates). The `val_geom_camber_cruise`
split improved most dramatically (71.66 vs 79.99 vs implied ~90+ MSE) — the hardest OOD split
benefits most from better loss alignment.

**New baseline as of 2026-05-12 20:52:** `val_avg/mae_surf_p = 94.291`. All future experiments
use `--loss l1`.

---

## 2026-05-12 20:04 — PR #1393: OneCycleLR with warmup replacing CosineAnnealingLR

- **Student branch:** `charliepai2g48h1-frieren/onecycle-lr`
- **Hypothesis:** OneCycleLR (warmup → peak → cosine wind-down, per-batch
  stepping) at peak_lr=1e-3 should beat vanilla CosineAnnealingLR for short
  (14-epoch) training runs. Predicted -2% to -6% on `val_avg/mae_surf_p`.

### Result

| Arm | peak_lr | epochs | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 splits) |
|-----|---------|--------|---------|---------------------|----------------------------------|
| **A (winner)** | 1e-3 | 14 / 15 | 14 | **111.2984** ⭐ | 107.54 |
| B   | 5e-4 | 14 / 15 | 14 | 113.8337            | 108.63                           |

Per-epoch ~131 s, peak GPU memory 42.12 GB. Per-batch LR stepping confirmed
firing as intended (lr 4e-5 → 9.97e-4 by ep2 warmup → 1.34e-5 by ep14 for
Arm A). Arm A wins by 2.5 points (-2.2% rel) — well inside the predicted
band. The full 15-epoch schedule didn't complete because the 30-min cap
cut at epoch 14 (~7% of schedule tail wasted, but LR was already in deep
decay so unlikely to matter much).

### Action: SEND BACK to push (not yet mergeable)

**The student branch is empty.** Only the `assign` commit is on
`charliepai2g48h1-frieren/onecycle-lr` — no `train.py` diff, no
`models/model-onecycle-*` artifacts. Cause: the GitHub API rate-limit
storm (18:30–19:50 UTC) let `gh pr comment` retries succeed (the
SENPAI-RESULT comment posted at 20:04Z), but the separate `git push` for
code + metrics artifacts never landed and was not retried. The result is
real and the methodology checks out — but the PR has no diff to merge.

Sent back at 20:0Xz with concrete push commands. Will merge as new
baseline once the diff is on the branch (this is the first cleanly
terminal round-1 result with no loss-formulation caveat — Arm A becomes
the new `val_avg/mae_surf_p` floor).

### Pre-existing issue (not this PR)

`test_geom_camber_cruise/mae_surf_p` came back NaN on **both** arms; pre-existing
(reproduces identically across arms, confirmed by alphonse PR #1355). Same
`+Inf in y` sample 000020.pt. `data/scoring.py` stays read-only.

---

## 2026-05-12 20:02 — PR #1389: Deeper Transolver (n_layers 5 → 8)

- **Student branch:** `charliepai2g48h1-fern/deeper-8-layers`
- **Hypothesis:** Going from 5 → 8 layers gives more iterations of slice
  attention → MLP refinement. Predicted -2% to -6% on `val_avg/mae_surf_p`,
  bigger gains on tandem OOD splits.

### Result

| Arm | lr | epochs | best ep | val_avg/mae_surf_p | test_avg (3/4 splits, fern's local recompute) |
|-----|------|--------|---------|---------------------|------------------------------------------------|
| A   | 5e-4 | 9 / 15 | 8       | 153.4759            | 139.91                                         |
| B   | 3e-4 | 9 / 15 | 8       | **147.3969**        | 134.38                                         |

Per-epoch ~205 s (≈55% slower than baseline due to extra blocks), peak GPU
memory 64.5 GB. `n_params=1,025,827` (~1.0M, lower than the predicted
1.6-1.8M because Transolver blocks add ~125k each, not 200k). **Both arms
realized only 9 of the 15 configured epochs** before the 30-min cap → the
cosine schedule was set for T_max=15 so the LR decayed for 60% of where it
"thought" it was → effective LR was still elevated at the cut.

### Action: SEND BACK with push + Arm C (not a winner at current state)

**Same push problem as #1393** — the branch has no `train.py` diff, no
metrics. Need the push first regardless of merge decision.

On the result itself: Arm B's 147.40 is ~32% worse than the round-1 leader
(frieren OneCycle Arm A: 111.30). But the comparison is contaminated by the
schedule mismatch (cosine T_max=15 vs realized 9 epochs). Asked fern to run
one more arm with `--epochs 9` so the cosine fully completes within the
wall-clock — that isolates "does depth help once the schedule matches
realized epochs?" from "did the depth lever just get a half-decayed LR?".

If Arm C lands at <120 then depth is salvageable for round 2 compounding;
if it stays >135 we close the depth lever as a confirmed regression.

The hypothesis-aligned finding (Arm B improves `val_geom_camber_rc` by 42.5
over Arm A) is genuine and consistent with the "more layers help tandem
OOD" theory — it's just dominated by the schedule artifact in the average.

---

## 2026-05-12 19:09 — PR #1399: Surface loss pressure-channel weight 2× + surf_weight sweep

- **Student branch:** `charliepai2g48h1-nezuko/surf-channel-pressure-weight`
- **Hypothesis:** Per-channel surface-loss weighting (`[Ux, Uy, p] = [1, 1, 2]`)
  should improve `val_avg/mae_surf_p` because the ranking metric is surface
  pressure. Predicted -3% to -8%.

### Result

| Arm | surf_weight | CHANNEL_W | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 splits) |
|-----|-------------|-----------|---------|---------------------|----------------------------------|
| A   | 10          | [1,1,2]   | 13      | **111.7978**        | 110.9876                         |
| B   | 20          | [1,1,2]   | 13      | 126.2973            | 125.9050                         |

Metrics: `models/model-surf-pw2-sw10-20260512-175612/{metrics.jsonl,metrics.yaml}`
and `models/model-surf-pw2-sw20-20260512-183156/{...}` on the student branch.

### Action: send back (not merged)

The student's results comment uncovered a normalization error in the PR's loss
formulation. With denominator `surf_mask.sum() * surf_channel_weight.sum()`,
the new `surf_loss` is ~3× smaller in magnitude than the baseline `surf_loss`
when channel weights are `[1,1,1]`. So Arm A's *effective* surf:vol ratio is
`10/3 ≈ 3.3` (in baseline-equivalent units) and Arm B's is `~6.7` — both
**below** the baseline's `10`. That makes A-vs-B mostly a sweep of effective
surf_weight, not the per-channel-weighting hypothesis we wanted to test.

Sent back with a 3-arm replan (all `surf_weight=10`, fixed denominator using
`surf_channel_weight.mean()`):

- Arm A control: `CHANNEL_W=[1,1,1]` — exactly recovers baseline; first
  true baseline measurement on this branch.
- Arm B: `CHANNEL_W=[1,1,2]` — corrected version of the original hypothesis.
- Arm C (if time): `CHANNEL_W=[1,1,3]` — dose-response.

### Pre-existing issue (not this PR)

`test_geom_camber_cruise/mae_surf_p` came back NaN on both arms while the
matching `val_geom_camber_cruise/mae_surf_p` was finite (87.41 for Arm A).
This affects the pressure channel only on that one test split. Pre-existing —
likely a numerical instability in model predictions on at least one extreme
cruise test sample, not a scoring bug (since val_finite + same split's
Ux/Uy_test were finite). Logged; will revisit if other PRs hit the same
NaN. Test_avg reported here is the partial mean over the 3 finite splits.

### Trajectory (Arm A)

| epoch | val_avg/mae_surf_p | seconds | peak_mem_GB |
|-------|---------------------|---------|-------------|
|  1 | 223.35 | 133 | 41.7 |
|  5 | 163.80 | 130 | 42.1 |
| 10 | 139.59 | 132 | 42.1 |
| 13 | **111.80** ⭐ best | 132 | 42.1 |
| 14 | 112.80 | 130 | 42.1 |

Per-epoch ~2.2 min; 14 of 15 configured epochs ran before the 30-min cap.
Peak memory ~42 GB of 96 GB — large headroom for wider/deeper models.

---

# Round 3/4 Results — 2026-05-14 Advisor Review

---

## 2026-05-14 — PR #1405: bf16 autocast + OneCycle@25ep (tanjiro) ✅ MERGED — NEW BEST

- **Student branch:** `charliepai2g48h1-tanjiro/amp-bf16-batch8`
- **Hypothesis:** bf16 autocast reduces VRAM (42→33 GB), and configuring `--epochs 25` stretches the OneCycleLR schedule so that at the 30-min wall-clock cutoff (~19 realized epochs) the LR is still meaningful (~3.3e-4) rather than near-zero (as in the cfg=14 recipe).

### Result

| Arm | bf16 | epochs cfg | epochs realized | wall (min) | val_avg/mae_surf_p | test_avg/mae_surf_p | peak VRAM |
|-----|------|------------|-----------------|------------|---------------------|----------------------|-----------|
| A: bf16+OneCycle, cfg=14 | ✅ | 14 | 14 | 22.9 | 83.951 | 73.316 | 32.94 GB |
| **B (winner): bf16+OneCycle, cfg=25** | **✅** | **25** | **19** | **31.0** | **73.295** | **63.911** | **32.95 GB** |
| Baseline (PR #1581, fp32) | — | 14 | 14 | — | 85.615 | 83.328 | 42 GB |

Per-split val (Arm B, epoch 19):

| Split | mae_surf_p | Δ vs PR #1581 |
|-------|-----------|----------------|
| val_geom_camber_cruise | 54.423 | -18.1% |
| val_re_rand | 71.041 | -13.2% |
| val_single_in_dist | 79.894 | -19.7% |
| val_geom_camber_rc | 87.823 | -7.2% |
| **val_avg** | **73.295** | **-14.4%** |

Val trajectory (Arm B): ep15=93.09, ep16=88.83, ep17=81.33, ep18=76.40, ep19=73.30. Still improving ~3 pts/epoch at cutoff.

**Metric artifacts:** `models/model-amp-bf16-onecycle-25ep-20260512-233756/metrics.jsonl` on advisor branch.

### Analysis

The key mechanism is the **OneCycleLR schedule horizon**:
- `cfg=14`: schedule completes fully at epoch 14 (LR → ~8e-9). Model optimizes well but has no LR headroom after ep14.
- `cfg=25`: schedule configured for 25 epochs. At epoch 19 (cap), LR = ~3.3e-4 (76% through schedule). The model is still in active fine-grained descent.

bf16 itself doesn't speed up training (97 s/epoch identical to fp32 — CPU/dataloader bottleneck, not GPU compute). The VRAM savings (33 vs 42 GB) are headroom for future experiments.

The 5-epoch improvement window (ep14→ep19 with cfg=25) produced -12.3 val pts total — that's the "schedule tail" that was cut off in all previous experiments.

### Action: MERGED — new baseline val_avg=73.295

---

## 2026-05-14 — PR #1602: Grad-clip gc=2.0 + OneCycle (fern) ↩ SENT BACK

- **Student branch:** `charliepai2g48h1-fern/grad-clip-l1`
- **Round 2 (on OneCycleLR baseline):** gc=2.0 achieved val=81.861 / test=77.818 (best seed) — **−4.4% vs PR #1581 (85.615)**

### Round-2 Result

| Arm | gc | seeds | val_avg best / mean | test_avg (3/4) best / mean |
|-----|----|------:|--------------------:|---------------------------:|
| A (gc=1.0) | 1.0 | 3 | 84.038 / 84.751 | 80.516 / 80.999 |
| **B winner (gc=2.0)** | **2.0** | **2** | **81.861 / 82.731** | **77.818 / 79.030** |
| Baseline (PR #1581) | — | — | — / 85.615 | — / 83.328 |

Mechanism: pre-clip grad-norm mean ~36 (clips on every step); gc=2.0 preserves 2× more per-batch magnitude than gc=1.0, which helps under OneCycle's higher peak LR.

### Action: SENT BACK — beaten by PR #1405 new baseline (73.295)

gc=2.0's val=81.861 is worse than the new bf16+25ep baseline. Sent back to re-run with `--epochs 25 --batch_size 4` (bf16 default) + `--grad_clip 2.0`.

---

## 2026-05-14 — PR #1605: asinh-p680 + OneCycle (edward) ↩ SENT BACK

- **Student branch:** `charliepai2g48h1-edward/asinh-pressure-target`
- **Round 2 (on OneCycleLR baseline):** asinh-p680 + OneCycle achieved val=83.259 / test=80.301 — **−2.75% vs PR #1581 (85.615)**

### Round-2 Result

| Run | val_avg/mae_surf_p | test_avg_3of4 | Δ vs OneCycle |
|-----|-------------------|--------------|---------------|
| OneCycle baseline (PR #1581) | 85.615 | 83.328 | — |
| **asinh-p680 + OneCycle** | **83.259** | **80.301** | **-2.75%** |

Per-split: cruise 60.63 (-8.74%), re_rand 80.15 (-2.13%), single 95.75 (-3.80%), rc 96.52 (+2.01% small regression).

### Action: SENT BACK — beaten by PR #1405 new baseline (73.295)

83.259 is above the new 73.295 baseline. Sent back to re-run with `--epochs 25` + `--asinh_p_scale 680.0` on the bf16+OneCycle baseline. Additional arm at scale=400 suggested (rc regression under OneCycle may be recoverable).

---

## 2026-05-14 — PR #1625: Per-channel weight cw=2 on cosine L1 (nezuko) ↩ SENT BACK

- **Student branch:** `charliepai2g48h1-nezuko/surf-channel-pressure-weight`
- **Result on OLD cosine L1 baseline:** `surf_channel_weight=[1,1,2]` → val=90.362 / test=87.643 — **−4.2% vs PR #1355 (94.291)**

### Result

| Arm | cw | val_avg | test_avg_3of4 | Δ vs #1355 |
|-----|----|--------:|--------------|-----------|
| A (cw=2) ★ | [1,1,2] | 90.362 | 87.643 | -4.2% |
| B (cw=3) | [1,1,3] | 94.367 | 92.922 | +0.08% |
| C (cw=5) | [1,1,5] | 93.928 | 91.107 | -0.4% |

### Action: SENT BACK — ran on cosine baseline, must re-test on bf16+OneCycle

cw=2 validates the signal on L1+cosine but both the OneCycle and bf16 baselines have since merged. Sent back to run single arm `--surf_channel_weight "1.0,1.0,2.0" --epochs 25 --lr 2e-3 --loss l1`.

---

## 2026-05-14 — PR #1582: surf_weight=5 on cosine L1 (alphonse) ↩ SENT BACK

- **Student branch:** `charliepai2g48h1-alphonse/surf-weight-sweep`
- **Result on OLD cosine L1 baseline:** `surf_weight=5` → val=90.776 / test=88.984 — **−3.7% vs PR #1355 (94.291)**

### Result

| Arm | surf_weight | val_avg | test_avg_3of4 | Δ vs #1355 |
|-----|------------|--------:|--------------|-----------|
| A (sw=5) ★ | 5 | 90.776 | 88.984 | -3.7% |
| B (sw=10) | 10 | 93.805 | 91.426 | -0.5% (control) |
| C (sw=20) | 20 | 94.203 | 93.191 | -0.09% |

Mechanism: L1's bounded gradients make the existing 10× surf amplification over-aggressive; 5× is the sweet spot on this loss.

### Action: SENT BACK — ran on cosine baseline, must re-test on bf16+OneCycle

---

## 2026-05-14 — PR #1697: SAM optimizer (thorfinn) ❌ CLOSED

- **Hypothesis:** SAM finds flatter minima → better OOD generalization, particularly on camber-OOD splits.
- **Result:** Best SAM arm (rho=0.05, 7 epochs) = val=134.744 — **57% worse than baseline 85.615**
- **Root cause:** SAM's 2× per-step cost halves realized epochs (7 vs 14). Flat-minima benefit requires a converged model; at 7 epochs, the model is far from convergence. The 2× cost is intrinsic and irrecoverable at 30-min cap.
- **Action: CLOSED** — fundamental compute-budget mismatch.

---

## 2026-05-14 — PR #1381: Wider Transolver n_hidden=256 (askeladd) ❌ CLOSED

- **Hypothesis:** Doubling hidden dim gives more capacity for multi-scale pressure gradients.
- **Result:** Best arm = val=138.022 — **61% worse than baseline 85.615**
- **Root cause:** 3.9M params → 2× slower per epoch → OOM at bs=4, forced bs=2, only 5-7 epochs realized. Undertrained model, not capacity-limited. Trajectory still steeply descending at cutoff.
- **Action: CLOSED** — width without throughput is a losing trade. bf16 now merged; depth experiment (askeladd, PR #2914) is the right follow-up.

---

## 2026-05-14 — PR #1667: OneCycleLR peak LR push 3/4/5e-3 (frieren) ❌ CLOSED

- **Hypothesis:** LR-saturated signal (1e-3→2e-3 gave +2.1pp); push to 3/4/5e-3.
- **Result:** 3e-3 → val=86.872, 4e-3 → val=89.321, 5e-3 → DIVERGED (val=316.4 at ep1)
- **Root cause:** AdamW + L1 + OneCycle(pct_start=0.1) is LR-saturated at 2e-3. The OneCycle wind-up at peak >2e-3 enters high-curvature regions before warmup completes. 
- **Action: CLOSED** — peak LR direction exhausted at 2e-3.
