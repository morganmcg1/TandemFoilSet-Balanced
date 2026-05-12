# SENPAI Research Results — charlie-pai2g-48h-r5

---

## 2026-05-12 22:45 — PR #1535: EMA model weights for eval (decay=0.999) — CLOSED (stale)

- **Branch:** `charliepai2g48h5-tanjiro/ema-eval-decay-0.999`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Maintain EMA copy of model weights with decay=0.999 and use it for eval —
  typical late-training noise smoothing.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no training trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis itself remains in-play and was reassigned on the compile baseline (decay=0.999,
  `torch.optim.swa_utils.AveragedModel` after compile, eval/test via EMA model).
- No data lost; closing simply frees the student slot.

---

## 2026-05-12 22:45 — PR #1561: Gradient clipping max_norm=1.0 — CLOSED (stale)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-1.0`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Bound rare large gradient updates via `clip_grad_norm_(.., max_norm=1.0)`;
  also a high-diagnostic-value characterization of training gradient norms.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis reassigned on the compile baseline with the per-epoch grad-norm aggregation
  (min/p50/mean/max/clip_frac) added so we still get the diagnostic value regardless of
  whether clipping wins on validation.

---

## 2026-05-12 22:45 — PR #1590: slice_num 64 → 96 + bf16 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-96-bf16`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase Transolver slice_num from 64 → 96 paired with bf16 AMP. With
  bf16's ~2× throughput we can afford the slightly more expensive forward and still complete
  full training; more slices = finer flow-field tokenization.

### Results

| Metric | Baseline (#1532 bf16) | This PR (#1590) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **105.024** | +3.86% (WORSE) |

- **Status:** CLOSED — does not beat bf16 baseline, well behind the new compile baseline (69.83).
- **Metric artifacts:** PR comment; trajectory shows monotone-worse vs slice_num=64 within
  the same epoch budget.

### Analysis

Combined with the round-3 fp32 result (slice_num=128 → val=145.97 at 11 epochs, wall-clock-bound)
and the bf16 result here (slice_num=96 → 105.02 at full budget), the slice_num lever is now
well-characterized:

| slice_num | regime | val_avg/mae_surf_p |
|---:|---|---:|
| 64 | bf16 (baseline) | 101.12 |
| 96 | bf16 | 105.02 (+3.86%) |
| 128 | fp32 (epoch-bound) | 145.97 |

Monotone-worse with slice count. The 64-slice default appears near-optimal for this dataset
size — adding slices adds capacity that overfits or simply costs throughput without finding
useful additional flow-field structure. **slice_num is a dead lever upward**; downward
(slice_num=32 or 48) would be a separate experiment but is a low-priority swing.

### Conclusions

- slice_num=64 is the right value for the current data + architecture.
- Closing this arm; do NOT pair slice_num=96 with compile (no signal it would help).
- Round-6 reassignment: frieren → step-based linear warmup + cosine on compile baseline.

---

## 2026-05-12 22:10 — PR #1568: torch.compile + bf16 AMP for additional throughput — MERGED ✓

- **Branch:** `charliepai2g48h5-thorfinn/torch-compile-bf16`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** `torch.compile(model, dynamic=True)` stacked on top of bf16 AMP doubles
  per-epoch throughput from ~98 s → ~49.5 s, fitting 36 epochs in 30 min vs 19 previously.
  Mechanism: kernel fusion eliminates Python dispatch overhead. `dynamic=True` prevents
  recompilation on variable-length mesh batches (N_max varies per batch).

### Results

| Metric | Baseline (#1532) | This PR (#1568) | Δ |
|---|---|---:|---:|
| `val_avg/mae_surf_p` | 101.1212 | **69.8316** | **-30.9%** |
| `test_avg/mae_surf_p` | 91.5013 | **61.8652** | **-32.4%** |

| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 77.10 | -35.8% |
| `val_geom_camber_rc` | 83.49 | -22.0% |
| `val_geom_camber_cruise` | 50.64 | -38.9% |
| `val_re_rand` | 68.10 | -28.0% |

| Split | test mae_surf_p |
|---|---:|
| `test_single_in_dist` | 67.81 |
| `test_geom_camber_rc` | 77.68 |
| `test_geom_camber_cruise` | 41.98 |
| `test_re_rand` | 59.99 |

- **Status:** MERGED — new baseline 69.8316 / 61.8652.
- **Epochs reached:** 36 (timeout-bound, 29.41 min; best epoch = 36, still descending)
- **Time/epoch:** ~49.5 s (2.0× speedup vs bf16-only ~98 s)
- **Peak GPU:** 23.8 GB (64 GB headroom on 96 GB card)
- **Compile status:** active for all 36 epochs, no recompilation stalls with `dynamic=True`
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.jsonl`

### Analysis

The win is almost entirely explained by epoch count: 36 vs 19 epochs = ~1.9× more gradient
steps. The model was monotonically improving through epoch 36 with no late-training instability.
`dynamic=True` was the correct choice — without it, dynamo would specialize per N_max and
accumulate recompilation costs that outweigh the kernel-fusion gain on variable-mesh batches.

All 4 val splits improved uniformly (+22-39%), including the hardest OOD splits. This is
pure optimization headroom, not overfitting.

**Key consequence:** The new 36-epoch budget changes the arithmetic for every in-flight arm.
- Capacity arms (#1587, #1588, #1590) were targeting n_hidden=160/n_layers=6/slice_num=96
  + bf16 (without compile). With compile now on advisor, those arms now run at compile speed
  IF they rebase — but they were branched before this merge and won't automatically have compile.
- T_max=50 cosine schedule with 36 epochs reaches LR≈0.012 at epoch 36 (not the full
  low-LR tail). Alphonse's T_max=18 result proved the terminal LR decay matters — so
  T_max=36 on top of compile is now the highest-confidence cheap win.

### Conclusions

- torch.compile is a free 2× throughput multiplier with no accuracy cost.
- 23.8 GB peak (batch=4, n_hidden=128) leaves 72 GB headroom for capacity exploration.
- Budget is still binding at 36 epochs — the model is still descending. More compute =
  more improvement. Highest-value follow-up: T_max=36 schedule to exploit the low-LR tail.

---

## 2026-05-12 22:00 — PR #1560: Match cosine T_max to actual epoch budget — SENT BACK

- **Branch:** `charliepai2g48h5-alphonse/tmax-18-cosine`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** `CosineAnnealingLR(T_max=50)` with 19 bf16 epochs never reaches the
  low-LR tail. Setting T_max=epoch_budget (originally T_max=14 for fp32, T_max=18 for
  bf16) lets the schedule complete, adding a meaningful low-LR fine-tuning phase.

### Results (two arms)

**Arm A — T_max=14 (fp32-era budget, pre-bf16 advisor commit 1341b98):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **98.7502** (best epoch = 14, terminal) |
| `test_avg/mae_surf_p` | **88.8030** |
| Epochs reached | 14/14 (complete) |
| Time/epoch | ~132.4 s (fp32) |
| vs #1444 baseline (110.76) | -10.8% |

**Arm B — T_max=18 (bf16-era budget, current advisor commit afd445a):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **90.3237** (best epoch = 18, terminal) |
| `test_avg/mae_surf_p` | **80.1938** |
| Epochs reached | 18/18 (complete) |
| Time/epoch | ~98.0 s (bf16) |
| vs #1532 baseline (101.12) | **-10.7%** |

Per-split Arm B (tmax-18):
| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 105.86 | -14.2% |
| `val_geom_camber_rc` | 99.48 | -7.1% |
| `val_geom_camber_cruise` | 70.74 | -14.6% |
| `val_re_rand` | 85.22 | -9.8% |

- **Status:** SENT BACK — baseline moved to 69.83 (PR #1568 merged). T_max=18 (90.32)
  no longer beats new baseline. Reassigned to retest with T_max=36 matching compile budget.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-tmax-18-cosine-20260512-210749/metrics.jsonl`,
  `models/model-charliepai2g48h5-alphonse-tmax-14-cosine-20260512-201325/metrics.jsonl`

### Analysis

**Mechanism confirmed.** Best epoch = terminal epoch in BOTH arms. The cosine schedule's
low-LR tail (final ~20-25% of epochs where LR approaches 0) provides material fine-tuning
benefit. The trajectory is clear:

val_avg at epochs 14→18 in Arm B: 98.34 → 92.62 → 92.34 → 91.44 → **90.32** — the last 4
epochs (T_max=14 to terminal) gained ~8.0 absolute MAE points. This is the "low-LR tail" the
hypothesis predicted.

**At epoch 14, both arms agree** (Arm B epoch 14 = 98.34, Arm A terminal = 98.75) — the LR
trajectory difference up to epoch 14 is negligible. The improvement is purely from completing
the cosine arc.

**Key implication for compile baseline:** With torch.compile reaching 36 epochs, the "natural
budget" has doubled. T_max=36 would complete the cosine arc and provide the same low-LR tail
effect — potentially gaining ~8-12 MAE off the 69.83 baseline.

### Conclusions

- Schedule-completion is a real, cheap, orthogonal lever. Best epoch = terminal epoch = strong
  signal that the low-LR tail does fine-tuning work.
- T_max=18 is obsolete — compile changed the budget to 36 epochs.
- **Follow-up:** alphonse re-running PR #1560 with `--epochs 36` on the updated advisor branch.
  If the same epoch-14→18 proportional gain holds (~8% of the remaining MAE), val_avg could
  drop from ~60 (extrapolating compile curve) to ~55 in the final epochs.

---

## 2026-05-12 21:30 — PR #1428: Per-channel loss weights [1,1,3] favoring pressure — CLOSED

- **Branch:** `charliepai2g48h5-nezuko/pressure-channel-weight`
- **Student:** charliepai2g48h5-nezuko
- **Hypothesis:** Reweight loss channels [1,1,3] so the pressure channel (the
  one we're scored on) carries 3× the gradient signal of Ux/Uy. Expected
  -5% to -12% delta on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **135.5317** (epoch 13 best) |
| `val_single_in_dist` | 167.07 |
| `val_geom_camber_rc` | 143.28 |
| `val_geom_camber_cruise` | 103.33 |
| `val_re_rand` | 128.44 |
| `test_avg/mae_surf_p` | **122.2302** (finite — student applied scoring workaround) |
| Best epoch | 13 |
| Epochs reached | 14 (timeout-bound, ~131 s/epoch, fp32 — pre-bf16) |
| Peak GPU | 42.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch, pre-bf16) |

- **Status:** CLOSED — +34.1% worse than bf16 baseline 101.12.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-pressure-channel-weight-20260512-200303/metrics.jsonl`

### Analysis

Two compounding factors explain the poor result:

1. **Wall-clock disparity.** Branch predates PR #1532 — 14 fp32 epochs at
   ~131 s/epoch vs baseline's ~19 bf16 epochs at ~98 s/epoch. Partially
   accounts for the gap (maybe 50%?).

2. **Channel weighting fundamentally wrong at 3×.** All four val splits
   regressed — including val_geom_camber_cruise (103.33 vs 82.84 at
   baseline). The only mechanistic explanation for regression on ALL splits
   simultaneously is that [1,1,3] distorted the optimization geometry.
   With 3× pressure gradient, the model optimizes pressure at the expense
   of Ux/Uy, but pressure predictions depend on accurate velocity (physical
   coupling), so the interference cascades back to `mae_surf_p`. Even on
   the "easiest" split (`val_geom_camber_cruise`) only reached ~25% above
   the baseline's full-budget performance at epoch 13.

3. **Student's diagnostic insight for `val_single_in_dist`.** Student noted
   this split (RaceCar single random hold-out) is the hardest despite being
   in-distribution — suggesting the WeightedRandomSampler may be
   under-covering that domain. This is the seed for the reassignment below.

### Conclusions

- Per-channel reweighting at [1,1,3] is ruled out — too aggressive, harms Ux/Uy
  via physical coupling, all-split regression.
- Milder weights ([1,1,2] or [1,1,1.5]) might be worth revisiting after
  other improvements are stacked, but the priority is the sampler direction.
- **New assignment for nezuko (PR #1619): domain-aware sampler reweighting** —
  boost RaceCar single sample weights 2× (→ 50% share) to directly attack
  `val_single_in_dist` coverage deficit. Inherits bf16 AMP + scoring fix.

---

## 2026-05-12 20:55 — PR #1422: slice_num 64 → 128 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-128`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase `slice_num` from 64 to 128 to give Transolver
  more physics-aware slice tokens per attention layer.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **145.9708** (epoch 11 best) |
| `val_single_in_dist` | 184.67 |
| `val_geom_camber_rc` | 154.30 |
| `val_geom_camber_cruise` | 114.72 |
| `val_re_rand` | 130.19 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 11 |
| Epochs reached | 11 (timeout-bound) |
| Time/epoch | ~171 s (vs ~131 s baseline) |
| Peak GPU | 54.5 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +44% worse than baseline 101.12.

### Analysis

Same diagnosis as #1398, #1413: capacity scale-up at fp32 + MSE only fits
11 epochs in the 30-min cap, vs baseline's 19 epochs (bf16) — undertrained.
Val still descending monotonically through epoch 11 (no plateau, no
instability, no OOM at 54.5 GB). The lever itself isn't refuted — the
budget is binding.

### Conclusions

- slice_num=128 untestable under current wall-clock budget without bf16.
- Next assignment for frieren: slice_num=96 + bf16 inheritance (PR #1590) —
  milder slice bump paired with throughput fix for fair test.

---

## 2026-05-12 20:55 — PR #1413: n_layers 5 → 7 — CLOSED

- **Branch:** `charliepai2g48h5-fern/deeper-7-layers`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** Increase `n_layers` from 5 to 7 (deeper Transolver) to
  give more iterative slice-attention refinement.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **144.9040** (epoch 10 best) |
| `val_single_in_dist` | 171.26 |
| `val_geom_camber_rc` | 177.24 |
| `val_geom_camber_cruise` | 103.29 |
| `val_re_rand` | 127.83 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~181 s |
| Peak GPU | 57.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +43% worse than baseline 101.12.

### Analysis

Same diagnosis as the capacity-arms pattern: at n_layers=7 + fp32 + MSE,
only 10 epochs fit in the 30-min cap. Val descended steeply through
epoch 10 with no plateau. No instability, no OOM. Wall-clock is the
binding constraint, not depth.

### Conclusions

- n_layers=7 untestable under current budget without bf16.
- Next assignment for fern: n_layers=6 + bf16 inheritance (PR #1588) —
  milder depth bump paired with throughput fix.

---

## 2026-05-12 20:53 — PR #1398: n_hidden 128 → 192 — CLOSED

- **Branch:** `charliepai2g48h5-edward/wider-hidden-192`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver `n_hidden` from 128 to 192 for more
  representational capacity.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **138.1375** (epoch 10 best) |
| `val_single_in_dist` | 187.30 |
| `val_geom_camber_rc` | 141.23 |
| `val_geom_camber_cruise` | 103.21 |
| `val_re_rand` | 120.81 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~186 s |
| Peak GPU | 58.0 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +37% worse than baseline 101.12.

### Analysis

Trajectory was volatile at epoch 7-10 (167→179→197→138) — clearly still
in early-training oscillation, not converged. Wider model at fp32 trades
epochs for capacity 1-for-1. No instability, no OOM. Pattern matches
fern (#1413) and frieren (#1422) exactly: wall-clock is binding for
capacity scale-ups under MSE+fp32.

### Conclusions

- n_hidden=192 untestable under current budget without bf16.
- Three students (edward, fern, frieren) independently identified the
  same pattern: capacity-scale-up arms get killed by wall-clock cap
  unless paired with throughput recovery (bf16).
- Next assignment for edward: n_hidden=160 + bf16 inheritance (PR #1587) —
  milder width bump paired with throughput fix.

---

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix — MERGED

- **Branch:** `charliepai2g48h5-thorfinn/bf16-amp-scoring-fix`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Enable bf16 mixed-precision training (`torch.autocast("cuda", dtype=torch.bfloat16)`) to increase epoch throughput and reach more training epochs within the 30-min cap. Also includes scoring-NaN workaround: batch-level `y_finite_mask` filter in `evaluate_split` before `accumulate_batch`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **101.1212** (epoch 17 best) |
| `val_single_in_dist/mae_surf_p` | 120.0176 |
| `val_geom_camber_rc/mae_surf_p` | 107.0980 |
| `val_geom_camber_cruise/mae_surf_p` | 82.8425 |
| `val_re_rand/mae_surf_p` | 94.5268 |
| `test_avg/mae_surf_p` | **91.5013** (finite — first on this branch) |
| `test_single_in_dist/mae_surf_p` | 105.4434 |
| `test_geom_camber_rc/mae_surf_p` | 99.9931 |
| `test_geom_camber_cruise/mae_surf_p` | 69.2841 |
| `test_re_rand/mae_surf_p` | 91.2844 |
| Best epoch | 17 |
| Epochs reached | 19 (~25% faster at ~98 s/epoch vs ~131 s) |
| Peak GPU | 32.95 GB |

- **Improvement:** -9.64 MAE (-8.7%) vs PR #1444 baseline (110.7608)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.{jsonl,yaml}`
- **Status:** MERGED → new round-5 baseline floor: **val_avg/mae_surf_p = 101.1212**

### Analysis

1. **bf16 AMP gave a real throughput win**: ~25% faster per epoch (98 vs 131 s), reaching epoch 19 vs baseline's epoch 14 — 5 extra epochs of convergence. The extra epochs drove the primary win: best epoch 17 vs baseline's 14.

2. **Scoring fix unblocked test_avg**: The `y_finite_mask` filter in `evaluate_split` correctly skipped `test_geom_camber_cruise/000020.pt`, giving the first finite `test_avg/mae_surf_p` (91.50) on this branch. This fix is now on the advisor branch for all subsequent PRs.

3. **Throughput under 2×**: At 0.66 M params, the model is small — Python/I/O overhead is a non-trivial fraction of step time. Bigger models would amortize the autocast win more. The `~25%` gain is real but modest.

4. **Still improving at cap**: Val was 102.26 at epoch 19 (final) vs best 101.12 at epoch 17 — slight uptick at the last epoch, still trending overall. More compute budget would likely gain additional MAE points.

5. **`val_geom_camber_cruise` slight regression (+5 MAE pts)**: The only split that worsened. Possibly noise from the different convergence trajectory (more epochs = different phase of the schedule). Worth watching in follow-up runs.

### Conclusions

- bf16 AMP is now the baseline — it's merged and available for all subsequent PRs to inherit.
- The scoring-NaN workaround is now on advisor — new baseline for test_avg is 91.5013.
- New bar: any PR must beat **101.1212** on val_avg/mae_surf_p to merge.
- Next for thorfinn: compound the wins — pair bf16 with the best capacity lever once architecture results settle.

---

## 2026-05-12 20:00 — PR #1388: Linear warmup + lr 5e-4 → 1e-3 with cosine anneal — CLOSED

- **Branch:** `charliepai2g48h5-askeladd/warmup-lr-1e3`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Add 5-epoch linear warmup and raise peak lr from 5e-4 to 1e-3
  (with cosine anneal afterward). Compensate for small batch and short
  wall-clock budget.

### Results

| Metric | lr=1e-3 (primary) | lr=7.5e-4 (fallback) |
|---|---:|---:|
| `val_avg/mae_surf_p` | **152.0332** | 152.5056 |
| `val_single_in_dist/mae_surf_p` | 184.95 | 177.17 |
| `val_geom_camber_rc/mae_surf_p` | 163.59 | 163.31 |
| `val_geom_camber_cruise/mae_surf_p` | 122.49 | 124.96 |
| `val_re_rand/mae_surf_p` | 137.10 | 144.58 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) | NaN |
| `test_3of4_avg/mae_surf_p` | 148.47 | 148.80 |
| Best epoch | 12 | 12 |
| Epochs reached | 14 | 14 |
| Time/epoch | 131.4 s | 132.0 s |
| Peak GPU | 42.11 GB | 42.12 GB |
| Loss used | **MSE** (PR predates Smooth-L1) | **MSE** |

- **Artifacts:** `models/model-charliepai2g48h5-askeladd-warmup-lr-1e3-20260512-181136/metrics.{jsonl,yaml}`, `models/model-charliepai2g48h5-askeladd-warmup-lr-7.5e4-20260512-185418/metrics.{jsonl,yaml}`
- **Status:** CLOSED — both arms ~41 MAE worse than baseline.

### Analysis

- ~41 MAE gap is too large to be MSE-vs-Smooth-L1 alone; lr=1e-3 is the
  dominant cause. The 5-epoch warmup + 9 epochs at peak lr=1e-3 + small
  cosine decay integrates LR-area-under-curve comparable to baseline's
  14 epochs at lr=5e-4, but more time at high lr overshoots good basins.
- Not divergence (loss curves were clean) — just a worse local minimum.
- Student independently rediscovered the scoring NaN bug, identical to
  thorfinn/alphonse's findings. Three independent students all found the
  same `0 × Inf = NaN` interaction — high-confidence diagnosis.
- The "step-based warmup over the first ~500 steps" idea is worth queuing
  separately, since 5 epochs = ~36% of the 14 epochs actually fitting in the
  cap.

### Conclusions

- lr=1e-3 with warmup is not productive at this wall-clock budget. The lr
  lever appears to be tuned correctly at baseline (lr=5e-4). Pushing lr
  higher (e.g., lr=1.5e-3, lr=2e-3) is not promising given the 41 MAE gap.
- More promising direction implied: step-based warmup at a *lower* peak.
  Queued for later, not assigned now.
- Next assignment for askeladd: gradient clipping max_norm=1.0 (PR #1561) —
  orthogonal to schedule lever space.

---

## 2026-05-12 19:53 — PR #1375: Raise surf_weight 10 → 30 — CLOSED

- **Branch:** `charliepai2g48h5-alphonse/surf-weight-30`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Raise `surf_weight` from 10 to 30 to bias gradients more
  toward the ranking quantity (surface pressure MAE).

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **120.3944** (epoch 13) |
| `val_single_in_dist/mae_surf_p` | 148.75 |
| `val_geom_camber_rc/mae_surf_p` | 125.45 |
| `val_geom_camber_cruise/mae_surf_p` | 93.73 |
| `val_re_rand/mae_surf_p` | 113.65 |
| `test_avg/mae_surf_p` | **112.6536** (finite — scoring workaround applied) |
| `test_single_in_dist/mae_surf_p` | 133.54 |
| `test_geom_camber_rc/mae_surf_p` | 123.03 |
| `test_geom_camber_cruise/mae_surf_p` | 79.73 |
| `test_re_rand/mae_surf_p` | 114.32 |
| Best epoch | 13 |
| Epochs reached | 14 |
| Time/epoch | 131.9 s |
| Peak GPU | 42.11 GB |
| Loss used | **MSE** (PR predates Smooth-L1) |

- **Artifacts:** `models/model-charliepai2g48h5-alphonse-surf-weight-30-20260512-191201/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (120.39 > 110.76).

### Analysis

- ~10 MAE gap to baseline. Smooth-L1 vs MSE typically buys ~5% in this
  regime — even a full recovery wouldn't close the gap.
- Per-split signal is diagnostic: `val_single_in_dist` got *worse* under
  surf_weight=30 (148.75 vs baseline 135.16) — surface-heavy reweighting
  biased gradients away from the volume manifold, hurting the hardest split.
  This is not an MSE-vs-Smooth-L1 artifact.
- Student independently rediscovered the scoring NaN bug AND wrote a clean
  `train.py:evaluate_split` workaround — exactly the same workaround being
  rolled centrally via PR #1532 (thorfinn). All four test splits finite as
  a result.
- Student also surfaced the recurring "T_max=50 cosine never decays in 14
  epochs" observation that tanjiro/askeladd also raised.

### Conclusions

- `surf_weight=30` is not productive — biases away from volume manifold.
  The baseline at `surf_weight=10` is well-tuned.
- Next assignment for alphonse: T_max=14 cosine schedule matched to actual
  epoch budget (PR #1560) — exactly the lever the student's own analysis
  pointed at, and orthogonal to all in-flight work.

---

## 2026-05-12 19:27 — PR #1439: Double batch_size 4 → 8 — CLOSED

- **Branch:** `charliepai2g48h5-tanjiro/batch-size-8`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Raise effective batch size from 4 → 8 to lower gradient
  variance under the 30-min wall-clock cap.

### Results

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 155.504 (epoch 14) |
| `val_single_in_dist/mae_surf_p` | 256.30 |
| `val_geom_camber_rc/mae_surf_p` | 145.07 |
| `val_geom_camber_cruise/mae_surf_p` | 103.11 |
| `val_re_rand/mae_surf_p` | 117.55 |
| `test_avg/mae_surf_p` | NaN (round-5 scoring bug) |
| Mean test_mae_surf_p (3 splits, excl. cruise) | 155.71 |
| Peak GPU | **84.2 GB** of 96 (no OOM) |
| Time/epoch | ~130 s |
| Epochs/30 min | 14 |
| Loss used | **MSE** (PR predates the Smooth-L1 merge) |

- **Artifacts:** `models/model-charliepai2g48h5-tanjiro-batch-size-8-20260512-185115/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (155.504 > 110.76).

### Analysis

- The comparison is unfair to the hypothesis: tanjiro's branch was created
  before #1444 merged Smooth-L1, so this run is MSE+batch=8 vs the current
  Smooth-L1+batch=4 baseline.
- However, the student's own analysis is decisive: **wall-clock is the binding
  constraint, not gradient noise**. Doubling batch trades step count 2:1 for
  variance reduction, but PR #1444 was monotonically improving at batch=4 —
  variance is not the bottleneck. Batch=8 just means fewer training epochs in
  the same 30-min window.
- batch=8 sits at 84 GB peak — no more headroom on this model size, so
  batch=8 is at its memory ceiling on the default Transolver. The lever is
  fully exercised.
- The student independently rediscovered the scoring NaN bug (same root
  cause as PR #1444) — solid debugging.

### Conclusions

- `batch_size=8` is feasible but does not appear to be a productive lever on
  this dataset + model + wall-clock budget. Closing the arm.
- The student's observation that "T_max=50 cosine never gets used because we
  only reach ~14 epochs" is a separately valuable insight — worth a future PR
  matching `T_max` to expected actual epoch budget.
- Next assignment for tanjiro: EMA model weights for eval (PR #1535) —
  orthogonal to the throughput / schedule lever space.

---

## 2026-05-12 18:58 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Branch:** `charliepai2g48h5-thorfinn/smooth-l1-loss`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Replace squared-error loss with Smooth-L1 (Huber, β=1.0) in
  normalized space for both training and evaluation losses. The ranking metric is
  MAE in original space; MSE in normalized space over-weights extreme high-Re
  samples. Smooth-L1 is linear outside |err|>β, providing bounded gradients.
  Both vol_loss and surf_loss use the same substitution; `surf_weight=10.0` and
  `data/scoring.py` MAE unchanged.

### Results

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p |
|---|---:|---:|---:|---:|
| `single_in_dist` | 135.16 | 1.719 | 0.769 | 120.38 |
| `geom_camber_rc` | 129.08 | 2.104 | 0.988 | 119.47 |
| `geom_camber_cruise` | 77.70 | 1.047 | 0.555 | NaN (bug) |
| `re_rand` | 101.10 | 1.607 | 0.740 | 97.36 |
| **avg** | **110.76** | — | — | NaN / 112.40 (3-split) |

- **Best epoch:** 14 of 50 configured (wall-clock-bound; monotonically improving)
- **Epochs/30-min:** ~14 at default model size (~131 s/epoch)
- **Peak GPU:** 42.1 GB (Blackwell RTX PRO 6000)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.{jsonl,yaml}`
- **Status:** MERGED → round-5 baseline floor

### Analysis

This is the first terminal result on the round-5 branch, so we cannot yet compare
against an MSE baseline on the same branch. The absolute val_avg = 110.76 sets
the floor. Key observations:

1. **Under-convergence.** The run was strictly monotonically improving at epoch 14
   when the 30-min cap hit (~14 epochs in 30 min for n_hidden=128). The floor is
   a loose lower bound on what the model could achieve with more compute.
2. **Split pattern consistent with hypothesis.** `val_geom_camber_cruise` (77.70)
   and `val_re_rand` (101.10) — the two splits the PR predicted would benefit most
   from bounded gradients at high-Re — are the best-performing splits. The raceCar
   splits (`single_in_dist` 135.16, `geom_camber_rc` 129.08) are noisier
   epoch-to-epoch, consistent with the loss being driven by the wide-Re tail.
3. **Scoring NaN bug discovered.** `test_geom_camber_cruise/000020.pt` has ±Inf
   values in the `p` channel. The `data/scoring.py` sample-skip logic misses this
   due to `0 × Inf = NaN` (IEEE-754). This affects all PRs in round 5 that run
   the test step. Round-5 ranking decision: **val_avg/mae_surf_p only**. The fix
   (filter the bad sample in `train.py`'s `evaluate_split` before calling
   `accumulate_batch`) will be rolled into an upcoming student assignment.

### Conclusions

- Smooth-L1 is a viable baseline for round 5. Whether it beats MSE requires the
  other in-flight arms (which use MSE) to finish and post results.
- The binding constraint is wall-clock convergence speed: ~14 epochs in 30 min.
  The highest-leverage next move is anything that increases epochs/wall-clock
  (bf16 AMP, smaller batch, smaller model, compile) rather than per-epoch quality.
- `val_geom_camber_cruise` is the easiest split (lowest MAE). The hardest splits
  are the raceCar ones.
