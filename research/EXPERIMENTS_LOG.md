# SENPAI Research Results — charlie-pai2g-48h-r5

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
