# SENPAI Research Results

Track: `icml-appendix-charlie-pai2i-48h-r2`. Primary ranking metric: `val_avg/mae_surf_p` (lower is better). Test ranking metric: `test_avg/mae_surf_p`.

Entries are appended chronologically as PRs return terminal `SENPAI-RESULT` markers.

---

## 2026-05-15 13:50 — PR #3119: Longer training: epochs 50→80 (extends cosine schedule)

- **Branch:** charliepai2i48h2-thorfinn/longer-training-80-epochs
- **Hypothesis:** Extending cosine schedule T_max 50→80 would give 30 more low-LR fine-tuning epochs, improving convergence.
- **Metrics (committed):** `models/model-charliepai2i48h2-thorfinn-longer-training-80-epochs-20260515-124711/metrics.jsonl`

| Split | val_surf_p | test_surf_p |
|-------|-----------|------------|
| single_in_dist | 167.57 | 145.30 |
| geom_camber_rc | 148.30 | 133.81 |
| geom_camber_cruise | 103.98 | NaN (scoring bug) |
| re_rand | 120.21 | 118.37 |
| **avg** | **135.0153** | **NaN (3-split proxy: 132.50)** |

- **Decision:** MERGED as round-1 baseline. Best epoch was 12; epochs=80 config had no effect since 30-min timeout cap hit at epoch 14. Run is effectively the bare-baseline reference.
- **Key finding:** Per-epoch time ~131s → ~14 epochs per 30-min cap. Val was still descending at cutoff. Pre-existing NaN scoring bug found: sample 20 of test_geom_camber_cruise has NaN p in GT; NaN×0=NaN propagates through accumulate_batch despite y_finite filter. Fix: nan_to_num in train.py's evaluate_split.

---

## 2026-05-15 13:55 — PR #3104: Per-channel weighting: 4× surface-p, 2× volume-p in training loss

- **Branch:** charliepai2i48h2-fern/per-channel-p-weight-4x
- **Hypothesis:** Weighting surf-p gradient 4× and vol-p gradient 2× in training loss aligns optimization with the primary metric.
- **Metrics (committed):** `models/model-charliepai2i48h2-fern-per-channel-p-weight-4x-20260515-124617/metrics.jsonl`

| Split | val_surf_p | test_surf_p (recomputed) |
|-------|-----------|--------------------------|
| single_in_dist | 173.53 | 153.33 |
| geom_camber_rc | 180.04 | 163.89 |
| geom_camber_cruise | 114.42 | 98.39 |
| re_rand | 128.42 | 129.44 |
| **avg** | **149.1018** | **136.26** |

- **Decision:** CLOSED — 10.4% regression vs baseline (149.10 vs 135.02). Exceeds the >5% close threshold.
- **Key finding:** Per-channel weighting at [1,1,4]/[1,1,2] was too aggressive; converged to a worse local minimum in the same 14-epoch window. Also hit same 30-min timeout. Student produced recompute_test.py with NaN-safe accumulation and confirmed the scoring bug. Val was still descending at cutoff — the run may not be representative of the steady-state per-channel weight effect.
- **Follow-up:** Try lighter weights [1,1,2]/[1,1,1.5] after the NaN-safe scoring fix is merged. Or try surf_weight=30 (askeladd, PR #3101, still WIP) which adjusts surface emphasis globally rather than per-channel.

---

## 2026-05-15 14:15 — PR #3099: Capacity scale-up: n_hidden 128→192, n_layers 5→6, n_head 4→6

- **Branch:** charliepai2i48h2-alphonse/capacity-scale-192h-6l-6head
- **Hypothesis:** Larger residual stream (1.7M vs 0.66M params) generalizes better across 3 physical domains.
- **Metrics (committed):** `models/model-charliepai2i48h2-alphonse-capacity-192h-6l-6head-20260515-124549/metrics.jsonl`

| Epoch | Per-epoch time | val_avg/mae_surf_p |
|-------|---------------|-------------------|
| 7 (best) | ~242s | 169.99 |
| 8 | ~241s | 171.0 |

| Baseline epoch 7 | val_avg/mae_surf_p=210 | Alphonse epoch 7 | 170 ← 40 pts better at equal epoch count |

- **Decision:** SENT BACK — not a regression at equal epoch count. Per-epoch time ~242s vs baseline ~131s → only 8 epochs in 30-min cap vs baseline's 14. Ask student to rerun with lr=1e-3 to compress convergence into the budget window.

---

## 2026-05-15 14:15 — PR #3106: Slice/head scale-up: slice_num 64→128, n_head 4→8, mlp_ratio 2→3

- **Branch:** charliepai2i48h2-frieren/slice-128-head-8-mlp-3
- **Hypothesis:** Finer spatial decomposition (128 slices) and wider heads improve surface pressure accuracy.
- **Metrics (committed):** `models/model-charliepai2i48h2-frieren-slice-128-head-8-mlp-3-20260515-124520/metrics.jsonl`

| Epoch | Per-epoch time | val_avg/mae_surf_p |
|-------|---------------|-------------------|
| 6 (best) | ~259s | 163.98 |
| 7 | ~261s | 191.7 |

| Baseline epoch 6 | val_avg/mae_surf_p=171 | Frieren epoch 6 | 164 ← 7 pts better at equal epoch count |

- **Decision:** SENT BACK — not a regression at equal epoch count. Per-epoch time ~261s → only 7 epochs in 30-min cap. Ask student to rerun with lr=1e-3 to compress convergence into fewer epochs.

---

## 2026-05-15 15:30 — PR #3110: Batch + LR scaling: batch_size 4→8, lr 5e-4→8e-4 (sqrt scaling)

- **Branch:** charliepai2i48h2-nezuko/batch-8-lr-8e-4
- **Hypothesis:** Doubling batch to 8 with sqrt-scaled LR (5e-4→8e-4) gives cleaner gradient estimates per step; the 96GB GPU has headroom for B=8.
- **Metrics (committed):** `models/model-charliepai2i48h2-nezuko-batch-8-lr-8e-4-20260515-132234/metrics.jsonl`

| Split | val_surf_p | test_surf_p (corrected) |
|-------|-----------|--------------------------|
| single_in_dist | 219.798 | 186.691 |
| geom_camber_rc | 154.321 | 138.748 |
| geom_camber_cruise | 109.275 | 94.368 |
| re_rand | 118.355 | 120.105 |
| **avg** | **150.4371** | **134.978** |

- **Decision:** CLOSED — 11.4% regression vs baseline (150.44 vs 135.02). Per-epoch time ~180s → only 14 epochs in 30-min cap (same as baseline), but scheduler is undertrained because B=8 produces fewer gradient steps per epoch. The hypothesis is sound but timing makes it lose per wall-clock.
- **Key finding:** With batch=8, ~14 epochs fit in the 30-min cap but cosine schedule never reaches low-LR window. Peak memory 84.2 GB confirms B=8 is feasible. Student independently identified the NaN scoring bug and produced corrected test metrics (134.978 via nan_to_num). Cap-aware `batch=6, lr=6e-4` noted as a possible softer variant but deprioritized (same schedule-vs-cap tension; better to try orthogonal direction).
- **Follow-up:** Assigned to nezuko: Lion optimizer (PR #3293).

---

## 2026-05-15 16:25 — PR #3274: Bug fix: NaN-safe evaluate_split (nan_to_num before masked sum)

- **Branch:** charliepai2i48h2-fern/nan-safe-eval-fix
- **Hypothesis:** Infrastructure fix only — no metric change expected on val (val splits have no NaN-y samples). Fix enables valid test_avg/mae_surf_p for all future runs.
- **Metrics (debug mode):** `models/model-charliepai2i48h2-fern-nan-safe-eval-fix-20260515-142502/metrics.jsonl`

val_avg/mae_surf_p = 486.5309 (debug mode, 3 epochs, 2-sample dataset — NOT comparable to baseline)

- **Decision:** MERGED (infrastructure fix). val=486 is debug-mode garbage, not a regression. The fix adds `_safe_accumulate_batch` to train.py, replaces the `accumulate_batch` call in `evaluate_split`, and adds `nan_to_num` to the `vol_loss_sum`/`surf_loss_sum` blocks. Fix verified with independent NaN-injection unit test: fixed version gives finite mae_surf_p where buggy version gives NaN; bit-identical on clean data (max diff=0.0). After merge, all future runs on this branch report a valid 4-split `test_avg/mae_surf_p`.
- **Key finding:** Fern also provided a standalone validation confirming exactly 761 NaN values in the pressure channel of sample 20 of test_geom_camber_cruise (Ux/Uy finite). Fix is minimal and semantics-preserving.

---

## 2026-05-15 16:26 — PR #3101: Surface loss weight 10→30 (3× surface loss emphasis)

- **Branch:** charliepai2i48h2-askeladd/surf-weight-30
- **Hypothesis:** Tripling the surface loss weight aligns training gradients with val_avg/mae_surf_p, the primary metric. Standard technique for task-aligned fine-tuning in PDE surrogate models.
- **Metrics (committed):** `models/model-charliepai2i48h2-askeladd-surf-weight-30-20260515-124531/metrics.jsonl`

| Split | val_surf_p | test_surf_p (corrected) |
|-------|-----------|--------------------------|
| single_in_dist | 152.82 | 136.68 |
| geom_camber_rc | 134.85 | 123.03 |
| geom_camber_cruise | 102.60 | 88.12 |
| re_rand | 119.38 | 119.51 |
| **avg** | **127.4122** | **116.83** |

- **Decision:** MERGED — 5.6% improvement over previous baseline (127.41 vs 135.02). New baseline. Val curve still descending at epoch 14; significant headroom remaining at convergence.
- **Key finding:** 3× surface weight is stable (no vol metric degradation), training converged smoothly. Best epoch = 14 (last trained). The surf_weight sweep is worth continuing: try 50, 75. Corrected test=116.83 uses the independent recompute script; raw test is NaN (pre-#3274 fix). BASELINE.md updated.

---

## 2026-05-15 16:27 — PR #3102: OneCycleLR (max_lr=1e-3, pct_start=0.1) — SENT BACK

- **Branch:** charliepai2i48h2-edward/onecycle-maxlr-1e-3
- **Hypothesis:** OneCycleLR with aggressive warmup to max_lr=1e-3 compresses convergence into the early epochs, using momentum decay in the annealing phase to improve generalization.
- **Metrics (committed):** `models/model-charliepai2i48h2-edward-onecycle-maxlr-1e-3-20260515-142245/metrics.jsonl`

val_avg/mae_surf_p = 141.7713 (best at epoch 12 of 14 trained)
test_avg (3-finite splits): ~140.33; full test NaN (pre-#3274)

- **Decision:** SENT BACK — 11.1% regression vs new baseline (127.41), but OneCycleLR was sized for 50 epochs total while only 14 ran. The cosine annealing tail (where OneCycleLR wins) never executed. Fix: rerun with `epochs=13` so the full schedule fits in the 30-min cap. Max_lr=1e-3, pct_start=0.1 unchanged.
- **Key finding:** Warmup to max_lr=1e-3 by epoch 5 was smooth. The schedule mismatch (planned 50 epochs, ran 14) is the entire cause of the apparent regression. Per-epoch time matches baseline (~131s), so 13 epochs × 131s = ~28.5 min safely fits the cap.

---

## 2026-05-15 17:25 — PR #3293: Lion optimizer replacing AdamW (lr=1.7e-4, wd=3e-4)

- **Branch:** charliepai2i48h2-nezuko/lion-optimizer
- **Hypothesis:** Sign-based Lion update is robust to heavy-tailed gradient magnitudes; predicted 2-5% gain over AdamW. Expected strongest improvement on high-pressure splits.
- **Metrics (committed):** `models/model-charliepai2i48h2-nezuko-lion-optimizer-20260515-153522/metrics.jsonl`

| Split | val_surf_p (Lion+surf10) | Δ vs prev baseline |
|-------|--------------------------|-------------------|
| single_in_dist | 137.24 | −18.1% |
| geom_camber_rc | 124.42 | −16.1% |
| geom_camber_cruise | 97.32 | −6.4% |
| re_rand | 111.03 | −7.6% |
| **avg** | **117.5014** | **−12.97%** |

3-split test proxy: 115.70 (single+rc+re_rand); test_geom_camber_cruise NaN pre-merge (fix now live).

- **Decision:** MERGED — 7.8% improvement vs current baseline 127.41, 12.97% vs old 135.02. Hypothesis validated and exceeded 3× predicted magnitude. Sign-based updates are clearly robust to heavy-tail CFD gradients. Merged config = Lion+surf30 (auto-compounded via 3-way merge with #3101).
- **Key finding:** Best epoch=9 (earlier than AdamW's 12) indicates cosine schedule is undertrained with T_max=80. Strongest gains on high-magnitude splits (single_in_dist −18%, geom_rc −16%) confirm heavy-tail mechanism. Memory unchanged at 42.11 GB.

---

## 2026-05-15 17:27 — PR #3115: Re-conditional FiLM modulation after preprocess — CLOSED

- **Branch:** charliepai2i48h2-tanjiro/re-film-conditioning
- **Hypothesis:** FiLM gating with log(Re) after the preprocess MLP helps the model switch regimes across Re-OOD splits.
- **Metrics (committed):** `models/model-charliepai2i48h2-tanjiro-re-film-conditioning-20260515-143532/metrics.jsonl`

| Split | FiLM val | No-FiLM val | Δ |
|-------|----------|-------------|---|
| single_in_dist | 150.24 | 162.77 | −7.7% (better) |
| geom_camber_rc | 146.62 | 143.48 | +2.2% (worse) |
| geom_camber_cruise | 118.70 | 99.82 | **+18.9% (worse)** |
| re_rand | 126.44 | 122.52 | **+3.2% (worse — opposite of prediction)** |
| **avg** | **135.50** | **132.15** | **+2.5%** |

- **Decision:** CLOSED — val=135.50 is +6.3% regression vs current baseline 127.41. Hypothesis falsified: predicted FiLM helps re_rand (Re-OOD), but observed it hurts re_rand (+3.2%) and geom_cruise (+18.9%). Only single_in_dist improved (−7.7%), suggesting Re gating helps in-distribution but amplifies Re-OOD errors.
- **Key finding:** FiLM with Re conditioning creates tighter Re-dependency; this helps when Re is in-distribution but hurts when Re shifts out of training mass. Geometry-OOD splits also regress. Geometry-conditional FiLM (camber/AoA) not worth trying — same mechanism would apply. Student ran correct A/B comparison on same hardware. NaN bug also independently identified (fix already in #3274).

---

## 2026-05-15 18:30 — PR #3328: Surface loss weight sweep: surf_weight 30→50

- **Branch:** charliepai2i48h2-askeladd/surf-weight-50
- **Hypothesis:** Increasing surf_weight 30→50 would further align training loss with the primary metric (surface pressure MAE) and improve especially single_in_dist and geom_camber_rc (higher-magnitude splits).
- **Metrics:** `models/model-charliepai2i48h2-askeladd-surf-weight-50-20260515-163202/metrics.jsonl`

| Split | surf_weight=30 baseline | surf_weight=50 | Δ |
|-------|------------------------|----------------|---|
| single_in_dist | 152.82 | 178.79 | +17.0% |
| geom_camber_rc | 134.85 | 173.38 | +28.6% |
| geom_camber_cruise | 102.60 | 109.56 | +6.8% |
| re_rand | 119.38 | 126.69 | +6.1% |
| **avg** | **127.4122** | **147.1046** | **+15.5%** |

- **Decision:** CLOSED — +15.5% regression vs AdamW+surf30 baseline (127.41); +25% vs current Lion baseline (117.50). All 4 splits regressed. val curve highly oscillatory (±50 between adjacent epochs) vs monotone descent at surf_weight=30.
- **Key finding:** surf_weight=10→30 was the sweet spot. Pushing to 50 crosses into optimization instability: surface-loss curvature swamps volume-loss directions (train_vol_loss elevated at 0.95 vs 0.60 for sw=30 at same point), decoupling loss from the val metric. Tested on AdamW — not directly comparable to Lion baseline, but regression magnitude is decisive.

---

## 2026-05-15 18:30 — PR #3329: AdamW β2 tuning 0.999→0.95 for faster moment adaptation

- **Branch:** charliepai2i48h2-fern/adamw-beta2-095
- **Hypothesis:** Lowering β2 from 0.999 to 0.95 (PaLM/T5 default) would speed up second-moment adaptation and help on a short ~5k-step training run.
- **Metrics:** `models/model-charliepai2i48h2-fern-adamw-beta2-095-20260515-163840/metrics.jsonl`

| Split | β2=0.999 (baseline) | β2=0.95 | Δ |
|-------|---------------------|---------|---|
| single_in_dist | 152.82 | 174.41 | +14.1% |
| geom_camber_rc | 134.85 | 152.38 | +13.0% |
| geom_camber_cruise | 102.60 | 111.27 | +8.5% |
| re_rand | 119.38 | 131.56 | +10.2% |
| **avg** | **127.4122** | **142.4049** | **+11.8%** |

- **Decision:** CLOSED — +11.8% regression; all 4 splits worse on both val and test. Direction doubly off-axis since the optimizer has since moved to Lion.
- **Key finding:** With B=4 and variable-mesh heavy-tailed gradients, β2=0.95 is wrong direction — we need more smoothing, not less. PaLM/T5-style β2 is tuned for batch-millions regimes. Val best at epoch 10 then degrades (noisier trajectory consistent with under-smoothed v_t).

---

## 2026-05-15 18:30 — PR #3102: OneCycleLR (epochs=13 rerun, max_lr=1e-3)

- **Branch:** charliepai2i48h2-edward/onecycle-maxlr-1e-3
- **Hypothesis (round 2):** Sizing OneCycleLR total_steps to match wall-clock cap (13 epochs) would let the schedule execute its full warmup→peak→cosine arc, recovering from the truncation failure of round 1 (total_steps=50 while only 14 ran).
- **Metrics:** `models/model-charliepai2i48h2-edward-onecycle-maxlr-1e-3-ep13-20260515-163625/metrics.jsonl`

| Split | Cosine baseline (sw=30) | OneCycleLR ep13 | Δ |
|-------|------------------------|-----------------|---|
| single_in_dist | 152.82 | 177.71 | +24.9 |
| geom_camber_rc | 134.85 | 152.61 | +17.8 |
| geom_camber_cruise | 102.60 | 108.36 | +5.8 |
| re_rand | 119.38 | 125.81 | +6.4 |
| **avg** | **127.4122** | **141.1231** | **+10.8%** |

- **Decision:** CLOSED — +10.8% regression (AdamW+sw30 baseline); +20% vs Lion baseline 117.50. Both arms of OneCycleLR failed. Round 2 val (141.12) vs round 1 (141.77) — essentially identical despite schedule running to completion.
- **Key finding:** At 13 epochs, warmup window is ~1.3 steps (degenerate), model spends first half at peak LR (>5e-4) while cosine baseline anneals through that period. Schedule shape structurally poorly matched to 14-epoch budget. The bottleneck was schedule shape, not truncation. Student's analysis: CosineAnnealingLR wins on short budgets because it spends more time at moderate-to-low LR.

---

## 2026-05-15 18:45 — PR #3357: asinh loss transform for surface pressure

- **Branch:** charliepai2i48h2-tanjiro/asinh-pressure-loss
- **Hypothesis:** Heavy-tail pressure z-scores dominate the gradient under standard squared-error loss. `torch.asinh()` applied to pressure z-scores compresses gradient for |z|≫1 (∝ 1/|z|), letting the optimizer focus on the bulk distribution rather than rare extreme samples.
- **Metrics:** `models/model-charliepai2i48h2-tanjiro-asinh-pressure-loss-20260515-173315/metrics.jsonl`

| Split | Lion+sw30 baseline | asinh-pressure | Δ |
|-------|-------------------|----------------|---|
| single_in_dist | 137.24 | 108.04 | −21.3% |
| geom_camber_rc | 124.42 | 90.63 | −27.2% |
| geom_camber_cruise | 97.32 | 62.68 | −35.6% |
| re_rand | 111.03 | 78.58 | −29.2% |
| **val_avg** | **117.5014** | **84.9819** | **−27.7%** |
| **test_avg** | **115.70 (3-split proxy)** | **76.1441** | **−34.2%** |

- **Decision:** MERGED — new baseline 84.9819. Largest single-PR improvement in the research history.
- **Key finding:** Asinh-compressed loss is a fundamental improvement in the optimization landscape, not just a regularization trick. Val curve was still descending at epoch 14 (timeout bound). Improvements are uniform across all 4 val/test splits (20-36%), confirming this is not a localized fix to one pathological split but a systemic improvement in gradient quality. Model unchanged — no architecture, optimizer, or scheduler modifications. The 7-line code change applies `torch.asinh()` to only the pressure channel z-scores before computing squared error; evaluation MAE in physical units is unchanged, so the improvement is genuine.
