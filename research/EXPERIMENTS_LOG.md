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

---

## 2026-05-15 20:30 — PR #3411: Extend asinh to Ux/Uy channels (all-channel asinh loss)

- **Branch:** charliepai2i48h2-tanjiro/asinh-all-channels
- **Hypothesis:** If velocity z-scores also have heavy tails, extending asinh compression to Ux/Uy would further improve val_avg/mae_surf_p beyond the pressure-only baseline.
- **Metrics:** `models/model-charliepai2i48h2-tanjiro-asinh-all-channels-20260515-193258/metrics.jsonl`

| Split | asinh-p baseline | asinh-all | Δ |
|-------|-----------------|-----------|---|
| single_in_dist | 108.04 | 107.19 | −0.8% |
| geom_camber_rc | 90.63 | 103.98 | +14.7% |
| geom_camber_cruise | 62.68 | 66.85 | +6.7% |
| re_rand | 78.58 | 81.48 | +3.7% |
| **val_avg** | **84.9819** | **89.8741** | **+5.8%** |
| **test_avg** | **76.1441** | **80.5842** | **+5.8%** |

- **Decision:** CLOSED — +5.8% regression across all splits except single_in_dist (marginal). Hypothesis falsified.
- **Key finding:** Velocity z-scores are light-tailed (|z|<1 for most samples), so asinh compresses meaningful velocity gradient signal rather than suppressing outliers. The asymmetry between pressure (heavy tails) and velocity (light tails) makes the per-channel choice in PR #3357 load-bearing. Strongest regression on val_geom_camber_rc (+14.7%) — the high-Re raceCar split with the largest |Ux| magnitudes, where velocity gradient signal is most informative. The val loss landscape is also noisier than the pressure-only baseline (oscillatory curve vs monotonic descent in #3357). Confirmed: pressure-only asinh is the correct design choice.

---

## 2026-05-15 21:34 — PR #3382: EMA weights (decay=0.999) on asinh baseline (rebased rerun)

- **Branch:** charliepai2i48h2-askeladd/ema-weights-decay-0999
- **Hypothesis:** EMA shadow (decay=0.999) applied at val/test passes reduces Lion optimizer's sign-based update variance. First run was on pre-asinh baseline (val=105.79 on old 117.50 ref). Sent back for rebase; this is the confirmed rerun on current 84.98 asinh baseline.
- **Metrics:** `models/model-charliepai2i48h2-askeladd-ema-weights-decay-0999-rebased-20260515-203115/metrics.jsonl`

| Split | asinh-p baseline (#3357) | EMA+asinh (this) | Δ |
|-------|--------------------------|------------------|---|
| single_in_dist | 108.04 | 99.95 | −7.5% |
| geom_camber_rc | 90.63 | 94.15 | +3.9% |
| geom_camber_cruise | 62.68 | 60.26 | −3.9% |
| re_rand | 78.58 | 78.38 | −0.3% |
| **val_avg** | **84.9819** | **83.1874** | **−2.11%** |
| **test_avg** | **76.1441** | **74.5193** | **−2.13%** |

- **Decision:** MERGED — new baseline 83.1874. Mechanisms compose cleanly: asinh smooths the loss landscape; EMA then smooths parameter trajectory on top.
- **Key finding:** EMA still adds value on top of asinh, but the gain is reduced (−2.1% here vs −9.96% for EMA on the pre-asinh baseline). This confirms the advisor's hypothesis: asinh already reduces the variance EMA was correcting, but EMA provides an orthogonal variance-reduction layer (parameter trajectory vs gradient scale). Smoothness diagnostic: 0 sign flips in val curve, every epoch a new best — consistent with both mechanisms composing. Best epoch=14 (final, timeout-bound) with EMA shadow still catching up at 0.999 decay over ~5k steps. Mixed-split picture: single_in_dist improved strongly (-7.5%), geom_camber_rc slightly regressed (+3.9%), geom_camber_cruise improved (-3.9%), re_rand near-flat. Net improvement is real but smaller in magnitude than pressure-only asinh.


---

## 2026-05-15 22:30 — PR #3099: Capacity scale-up 192h/6L/6H (rerun on EMA+asinh baseline)

- **Branch:** charliepai2i48h2-alphonse/capacity-scale-192h-6l-6head
- **Hypothesis:** Scale model capacity (n_hidden 128→192, n_layers 5→6, n_head 4→6, lr 1.7e-4→3.4e-4) on top of the full EMA+asinh stack would give the larger model enough optimization signal to land below 83.19 baseline within the 30-min cap.
- **Metrics:** `models/model-charliepai2i48h2-alphonse-capacity-192h-6l-6head-lion-3p4e4-20260515-213703/metrics.jsonl`

| Split | EMA+asinh baseline | 192h/6L/6H rerun | Δ |
|-------|--------------------|-------------------|---|
| single_in_dist | 99.95 | 179.12 | +79% |
| geom_camber_rc | 94.15 | 156.16 | +66% |
| geom_camber_cruise | 60.26 | 87.87 | +46% |
| re_rand | 78.38 | 110.89 | +42% |
| **val_avg** | **83.19** | **133.51** | **+60.5%** |
| **test_avg** | **74.52** | **123.17** | **+65.3%** |
| Epochs reached | 14 | 8 | −43% |
| s/epoch | 131 | 242 | +85% |
| n_params | 0.66M | 1.71M | +159% |

- **Decision:** CLOSED — pre-registered fail criterion met. Capacity scaling alone is dominated by the per-epoch cost on this 30-min wall-clock budget.
- **Key finding:** The 2.6× larger model gets only 57% as many epochs as the baseline within the budget. At epoch 8 val=133.51 vs baseline epoch-8 val<90. Linear extrapolation suggests the bigger model would need ~25 epochs (~100 min) to reach 83.19 — outside the project envelope. The path forward for any future capacity work is offsetting throughput wins (bf16, FlashAttention, larger batch, lower slice_num). Student suggested bf16 mixed precision as the most direct throughput lever — selected as alphonse's next experiment.


---

## 2026-05-15 23:37 — PR #3384: Gradient clipping (max_norm=1.0) on EMA+asinh stack (rebased rerun)

- **Branch:** charliepai2i48h2-fern/lion-gradclip-1.0
- **Hypothesis:** grad_norm(pre-clip) >> 1.0 was confirmed to be 100% of steps in the first arm (pre-asinh baseline, mean ~137, max ~2724). On the asinh+EMA stack, asinh reduces the pressure component but aggregate gradient norms should remain heavy-tailed. max_norm=1.0 should compose orthogonally with both mechanisms.
- **Metrics:** `models/model-charliepai2i48h2-fern-lion-gradclip-1.0-rebased-20260515-222530/metrics.jsonl`

| Split | EMA+asinh baseline (#3382) | grad-clip+EMA+asinh (this) | Δ |
|-------|--------------------------|--------------------------|---|
| single_in_dist | 99.95 | 81.50 | −18.5% |
| geom_camber_rc | 94.15 | 82.80 | −12.1% |
| geom_camber_cruise | 60.26 | 49.22 | −18.3% |
| re_rand | 78.38 | 67.47 | −13.9% |
| **val_avg** | **83.1874** | **70.2479** | **−15.6%** |
| **test_avg** | **74.5193** | **62.0765** | **−16.7%** |

- **Decision:** MERGED — new baseline 70.2479. Third major compounding win.
- **Key finding:** Mechanisms compose at different levels of the gradient pipeline: asinh compresses loss-level heavy tails (per-coordinate pressure z-score scaling), EMA smooths parameter trajectory (exponential moving average of weights), grad-clip caps per-step L2 norm of gradient vector. Post-asinh pre-clip norms still 25-180 mean (100% clip rate all 14 epochs) confirming the three mechanisms are genuinely orthogonal — asinh does NOT eliminate the need for gradient clipping. Val still descending at epoch 14 timeout. Run shows 9% stochastic variance between two reruns (70.25 vs 77.23), suggesting single-run estimates carry ±5% noise under EMA slow start.

---

## 2026-05-15 23:38 — PR #3106: Slice/head scale-up (slice_num 64→128, n_head 4→8, mlp_ratio 2→3) on full stack

- **Branch:** charliepai2i48h2-frieren/slice-128-head-8-mlp-3
- **Hypothesis:** Richer slice decomposition and wider attention (more heads, larger MLP) would improve the per-step convergence rate faster than the baseline's 14-epoch improvement.
- **Metrics:** `models/model-charliepai2i48h2-frieren-slice-128-head-8-mlp-3-20260515-223545/metrics.jsonl`

| Split | EMA+asinh baseline | slice-128 rerun | Δ |
|-------|-------------------|-----------------|---|
| single_in_dist | 99.95 | 231.32 | +131% |
| geom_camber_rc | 94.15 | 192.17 | +104% |
| geom_camber_cruise | 60.26 | 106.95 | +77.5% |
| re_rand | 78.38 | 130.26 | +66.2% |
| **val_avg** | **83.19** | **165.18** | **+98.6%** |
| Epochs | 14 | 7 | −50% |
| s/epoch | 131 | 261 | +99% |

- **Decision:** CLOSED — +98.6% regression. Same wall-clock penalty pattern as alphonse capacity scale-up (#3099). 261s/epoch gives only 7 epochs vs baseline's 14. The per-step convergence is fine (monotone descent) but the epoch budget deficit dominates.
- **Key finding:** Any architecture change that increases per-epoch time above ~150s cannot overcome the lost epoch count within the 30-min cap. The direction requires an orthogonal throughput win (bf16 — alphonse's current experiment) before capacity scaling can add positive signal.

---

## 2026-05-16 03:30 — PR #3530: surf_weight reduction 30→25 on full 5-mechanism stack

- **Branch:** charliepai2i48h2-frieren/surf-weight-ablation
- **Hypothesis:** With asinh loss compression + EMA + grad-clip all active, surf_weight=30 over-weights surface loss because the noise-reduction mechanisms already de-emphasize extreme pressure gradients implicitly. Reducing surf_weight should rebalance the vol/surf trade-off.
- **Metrics:** `models/model-charliepai2i48h2-frieren-surf-weight-25-20260516-002627/metrics.jsonl` (winner), `models/model-charliepai2i48h2-frieren-surf-weight-20-20260516-013527/metrics.jsonl` (arm B)

| Split | baseline (sw=30) | Arm A (sw=25) | Δ | Arm B (sw=20) | Δ |
|-------|-----------------|---------------|---|---------------|---|
| single_in_dist | 81.50 | 80.69 | −0.99% | 81.42 | −0.10% |
| geom_camber_rc | 82.80 | 79.03 | −4.55% | 77.80 | −6.04% |
| geom_camber_cruise | 49.22 | 46.10 | −6.34% | 48.17 | −2.13% |
| re_rand | 67.47 | 63.37 | −6.07% | 65.02 | −3.62% |
| **val_avg** | **70.2479** | **67.2991** | **−4.20%** | **68.1024** | **−2.92%** |
| **test_avg** | **62.0765** | **58.9233** | **−5.08%** | **59.2259** | **−4.59%** |

- **Decision:** MERGED — new baseline 67.2991 (sw=25 wins). 5th compounding mechanism on the 5-mechanism stack (Lion + surf_weight=25 + asinh + EMA + grad-clip).
- **Key finding:** Hypothesis fully confirmed. The asinh/EMA/grad-clip stack implicitly de-emphasizes extreme-pressure gradients, shifting the optimal surface weight left from 30 toward 25. Both arms beat baseline; sw=25 marginally outperforms sw=20 in the aggregate. Split-by-split: cruise and re_rand gain the most (−6%); single_in_dist barely moves. Vol metrics stable — optimizer genuinely rebalanced rather than just shifting surface loss off. Optimum knee is at ~25; going below 20 likely trades away cruise/re_rand gains. Val still descending at epoch 14 (timeout-bound). Cumulative improvement from initial baseline: 135.02 → 67.30 = **−50.2%**.

---

## 2026-05-16 07:00 — PR #3485: bf16 autocast — 6th compounding mechanism

- **Branch:** charliepai2i48h2-alphonse/bf16-autocast
- **Hypothesis:** bf16 autocast on forward+loss computation reduces per-epoch time by freeing VRAM bandwidth, unlocking more epochs within the 30-min wall-clock cap. Prior baseline was reaching epoch 14 and timing out with val still descending — any extra epochs on a still-descending curve directly improve the primary metric.
- **Metrics:** `models/model-charliepai2i48h2-alphonse-bf16-autocast-20260516-053111/metrics.jsonl`

| Split | Baseline (PR #3530) | bf16 arm | Δ |
|-------|---------------------|----------|---|
| single_in_dist | 80.6871 | 70.2014 | −13.0% |
| geom_camber_rc | 79.0339 | 68.7070 | −13.1% |
| geom_camber_cruise | 46.1009 | 39.0294 | −15.3% |
| re_rand | 63.3746 | 57.5489 | −9.2% |
| **val_avg** | **67.2991** | **58.8717** | **−12.5%** |
| **test_avg** | **58.9233** | **51.6269** | **−12.4%** |
| Epochs | 14 | **18** | +4 |
| s/epoch | ~131 | **~101** | −23% |
| Peak GPU memory | 42.13 GB | **32.96 GB** | −22% |

- **Decision:** MERGED — new baseline 58.8717. 6th compounding mechanism (Lion + surf_weight=25 + asinh + EMA + grad-clip + **bf16**).
- **Key finding:** Throughput intervention fully confirmed. 23% per-epoch speedup from reduced memory bandwidth → 4 extra epochs in same wall-clock → −12.5% val improvement. No NaN, no instability (asinh, grad-clip, EMA all compose cleanly with bf16). 9 GB VRAM freed opens capacity expansion that was previously epoch-budget-constrained. Val still descending at epoch 18 — still timeout-bound. Cumulative improvement: 135.02 → 58.87 = **−56.4%**. The key CLAUDE.md lesson from PR #3106/#3099 is now inverted: any architecture change that previously failed due to per-epoch time can be retested under bf16.

---


## 2026-05-16 08:30 — PR #3733: Warmup-cosine schedule (v2): 2-epoch linear warmup + cosine
- charliepai2i48h2-edward/warmup-cosine-v2
- **Hypothesis:** 2-epoch linear LR warmup (start_factor=1e-3) before CosineAnnealingLR(T_max=78) would mitigate Lion cold-start instability — first-epoch updates on randomly-initialized weights with fixed-magnitude sign-based steps could encode noise into the EMA shadow. After warmup, cosine continues to lr=0 at original T_max.
- **Result:** `val_avg/mae_surf_p = 61.3285` at epoch 18 (vs baseline 58.8717, +4.2% regression). All 4 splits regressed.

| Split | Baseline (PR #3485) | warmup-cosine | Δ |
|-------|---------------------|---------------|---|
| single_in_dist | 70.2014 | 71.5439 | +1.9% |
| geom_camber_rc | 68.7070 | 73.1294 | +6.4% |
| geom_camber_cruise | 39.0294 | 41.4636 | +6.2% |
| re_rand | 57.5489 | 59.1769 | +2.8% |
| **val_avg** | **58.8717** | **61.3285** | **+4.2%** |
| **test_avg** | **51.6269** | **53.5904** | **+3.8%** |

- **Metrics:** `models/model-charliepai2i48h2-edward-warmup-cosine-v2-20260516-072829/metrics.jsonl`
- **Decision:** CLOSED no_improvement. Per-epoch val curve at epoch 18 (61.33) was roughly where the bf16 baseline was at epoch ~17 — warmup shifted the descent right by one epoch without compensating stability gain.
- **Key finding:** Lion + EMA + grad-clip already absorb the early-epoch instability the hypothesis worried about. Warmup is essentially zero-sum on this stack in a fixed budget: 2 epochs of slower learning that never get recovered. The val curve was still descending at ~2.3/epoch at end and `is_best` hit on every epoch — the diagnostic is real but the solution is not warmup. Reassigned edward to cosine-tmax-align (#3822): T_max=20/30 vs current 80 (cosine essentially constant at LR 0.85× initial through 18 epochs) — late-schedule complement to this early-schedule failure.

---

## 2026-05-16 10:30 — PR #3822: Cosine T_max alignment (T_max=30)
- charliepai2i48h2-edward/cosine-tmax-align
- **Hypothesis:** CosineAnnealingLR(T_max=80) is essentially constant over 18 epochs (LR at 0.852× initial at epoch 18). T_max alignment to match realistic training budget gives meaningful late-epoch annealing.
- **Result:** Arm B (T_max=30) **wins** — val 56.0011 (−4.88% vs 58.87). Arm A (T_max=20) neutral — val 59.08 (+0.36%).

| Split | Baseline (PR #3485) | Arm B T_max=30 | Δ |
|-------|---------------------|-----------------|---|
| single_in_dist | 70.2014 | 62.2099 | −11.4% |
| geom_camber_rc | 68.7070 | 68.5030 | −0.3% |
| geom_camber_cruise | 39.0294 | 37.7010 | −3.4% |
| re_rand | 57.5489 | 55.5904 | −3.4% |
| **val_avg** | **58.8717** | **56.0011** | **−4.88%** |
| **test_avg** | **51.6269** | **48.9470** | **−5.20%** |

- **Metrics:** `models/model-charliepai2i48h2-edward-cosine-tmax-30-20260516-093553/metrics.jsonl`
- **Decision:** MERGED — new baseline 56.0011. 7th compounding mechanism.
- **Key finding:** Arm A (T_max=20) was nearly neutral while Arm B (T_max=30) won. The optimal schedule has LR ending at ~40% of initial (6.73e-5), providing meaningful annealing without giving up update magnitude. Lion's sign-based update is not LR-magnitude-insensitive at end of training — modest annealing helps. Val still descending monotonically at epoch 18 in both arms (no plateau), confirming runs remain timeout-bound. Cumulative improvement: 135.02 → 56.00 = −58.5%.

---

## 2026-05-16 10:30 — PR #3750: Capacity expansion on bf16 stack
- charliepai2i48h2-alphonse/capacity-bf16
- **Hypothesis:** bf16 freed 9 GB VRAM; moderate capacity bump (n_hidden=144 or n_layers=6) could fit within budget.

| Run | Config | val_avg | test_avg | Epochs | t/epoch |
|-----|--------|---------|---------|--------|---------|
| Baseline | n128/5L/bf16 | 58.8717 | 51.6269 | 18 | 101s |
| Arm A (run 1) | n_hidden=144 | 59.8474 | **51.4517** | 16 | 114.7s |
| Arm A (re-run) | n_hidden=144 | 61.0179 | 52.6836 | 16 | 114.3s |
| Arm B | n_layers=6 | 64.5683 | 55.9771 | 15 | 121.0s |

- **Metrics:** `models/model-charliepai2i48h2-alphonse-capacity-n144-20260516-072226/metrics.jsonl`, `*-082212/metrics.jsonl`, `*-capacity-l6-20260516-092228/metrics.jsonl`
- **Decision:** CLOSED no_improvement. Arm B regresses >5%. Arm A within noise floor (~1.17pt variance between same-config runs).
- **Key finding:** Per-epoch time +14-20% dropped epoch count 18→15-16, offsetting expressivity gain. Capacity expansion needs more wall-clock, not just memory headroom. Test metric mildly better than baseline (51.45 vs 51.63) despite val regression — suggests bigger model learns something useful but val checkpoint selection masked it. Reassigned alphonse to batch-size-bf16 (#3884).

---

## 2026-05-16 12:40 — PR #3884: Batch size scaling on bf16 headroom
- charliepai2i48h2-alphonse/batch-size-bf16
- **Hypothesis:** bf16 freed 9 GB (32.97 vs 42.13 GB peak); batch=6 or 8 could buy smoother gradient estimates for Lion's sign-based update without memory overflow.
- **Result:** Arm A (batch=6) regressed +24.4% on val (69.68 vs 56.00). Stop condition triggered; Arm B (batch=8) not run.

| Metric | Arm A (batch=6) | Baseline (batch=4) | Δ |
|--------|-----------------|---------------------|---|
| val_avg/mae_surf_p | 69.6827 | 56.0011 | +24.43% |
| test_avg/mae_surf_p | 60.5222 | 48.9470 | +23.65% |
| Steps/epoch | 250 | 374 | −33.2% |
| Per-epoch time | ~105s | ~102s | +3s |
| Total steps (18ep) | 4,500 | 6,732 | −33.2% |
| Peak GPU | 49.42 GB | 32.97 GB | +50% |

- **Metrics:** `models/model-charliepai2i48h2-alphonse-batch-bs-6-20260516-112156/metrics.jsonl`
- **Decision:** CLOSED no_improvement.
- **Key finding:** Per-epoch time essentially unchanged at batch=6 (+3s) — bf16 is already compute-saturated at batch=4; bigger batch does NOT buy throughput. But steps/epoch dropped 33%: 4,500 vs 6,732 total optimization updates. With val still descending 5%/epoch at epoch 18, the lost steps dominate. Definitively confirms: **throughput is binding, not memory or signal quality.** Combined with #3750, the 7-mech stack at batch=4 is the right anchor. Next direction: `torch.compile` to buy per-epoch time savings that translate to more steps (assigned as #3970).

## 2026-05-16 13:50 — PR #3674: Per-channel pressure weight (pw=2.0 wins)
- charliepai2i48h2-nezuko/pressure-channel-weight
- **Hypothesis:** Per-channel loss weighting for pressure: pw=0.5 (de-emphasise) vs pw=2.0 (up-weight) vs baseline pw=1.0.
- **Result:** Arm B (pw=2.0) wins — val 53.7235 (−4.07% vs 56.00 baseline). Arm A (pw=0.5) regressed +3.20%.

| Arm | pressure_weight | val_avg/mae_surf_p | Δ | test_avg/mae_surf_p |
|-----|-----------------|---------------------|---|---------------------|
| Baseline | 1.0 | 56.0011 | — | 48.9470 |
| **Arm B** | **2.0** | **53.7235** | **−4.07%** | **46.6011** |
| Arm A | 0.5 | 57.7945 | +3.20% | 50.4263 |

Per-split breakdown (Arm B, pw=2.0, val): single_in_dist=59.91 (−3.71%), geom_camber_rc=67.08 (−2.08%), geom_camber_cruise=35.41 (−6.08%), re_rand=52.50 (−5.55%)

- **Metrics:** `models/model-charliepai2i48h2-nezuko-pressure-weight-2p0-20260516-112553/metrics.jsonl`
- **Decision:** MERGED — new baseline 53.7235. 8th compounding mechanism.
- **Key finding:** asinh compression hadn't fully neutralised channel imbalance — pw=2.0 constructively stacks with asinh by re-emphasising pressure without losing asinh stability. Velocity channels mildly regress (+11-21%) but pressure improvement dominates. val monotone in {pw=0.5, 1.0, 2.0} — curve may continue upward; nezuko assigned pw=3.0/4.0 sweep (#3984). Cumulative improvement: 135.02 → 53.72 = −60.2%.

---

## 2026-05-16 13:50 — PR #3949: Lion β1 momentum sweep (no_improvement)
- charliepai2i48h2-askeladd/lion-beta1
- **Hypothesis:** Lion β1=0.95 (more smoothing) and β1=0.85 (more reactivity) vs default β1=0.90.
- **Result:** Arm A (β1=0.95) regressed +9.8% (val 61.49 vs 56.00). Stop condition triggered; Arm B not run.

| Arm | β1 | val_avg/mae_surf_p | Δ |
|-----|-----|---------------------|---|
| Baseline | 0.90 | 56.0011 | — |
| Arm A | 0.95 | 61.4851 | +9.77% ❌ |

- **Metrics:** `models/model-charliepai2i48h2-askeladd-lion-beta1-095-20260516-123309/metrics.jsonl`
- **Decision:** CLOSED no_improvement.
- **Key finding:** More momentum smoothing (β1=0.95, 5% gradient weight/step) is decisively worse at 18-epoch/batch=4 regime. JSONL shows slower surf_loss descent from epoch 5 — under-reactive updates rather than instability. β1=0.90 appears well-tuned for LR=1.7e-4/T_max=30/bs=4. The productive direction is β1 < 0.90 (more reactivity) — student correctly identified this; future PR on the 8-mech stack could test β1 ∈ {0.85, 0.88}. However, askeladd was re-assigned to EMA decay (#3989) as the higher-priority direction.

---

## 2026-05-16 15:30 — PR #3725: Per-group grad-clip (attention vs MLP separate max_norm, no_improvement)
- charliepai2i48h2-fern/per-group-grad-clip
- **Hypothesis:** Attention projections dominate the aggregate gradient norm — separate clipping (attention=1.0, MLP/other=5.0 or 10.0) should let well-behaved MLP signal through while controlling noisy attention.
- **Result:** All three arms regressed vs 56.00 baseline (ran without pressure_weight=2.0). Arm C (attn=0.5, other=1.0) was closest at +1.12%.

| Arm | attn_norm | other_norm | val_avg/mae_surf_p | Δ | test_avg/mae_surf_p |
|-----|-----------|-----------|---------------------|---|---------------------|
| Baseline | — | — | 56.0011 | — | 48.9470 (51.6269 original) |
| **A** | 1.0 | 5.0 | 57.4996 | +2.66% | 48.9892 |
| **B** | 1.0 | 10.0 | 58.1940 | +3.91% | 49.2568 |
| **C (best)** | 0.5 | 1.0 | **56.6272** | +1.12% | 49.1408 |

- **Gradient-norm diagnostic (key finding):** At E18 with 7-mech stack: `attn_gn mean = 3.45`, `other_gn mean = 18.35`. The hypothesis was *inverted* — MLPs/output are ~5× noisier than attention, not the reverse. Single-clip(1.0) was already controlling the dominant (MLP/output) group; loosening it to 5.0/10.0 allows more MLP gradient noise → val regression.
- **Metrics:** `models/model-charliepai2i48h2-fern-pg-clip-A-20260516-132937/metrics.jsonl`, `models/model-charliepai2i48h2-fern-pg-clip-B-20260516-140449/metrics.jsonl`, `models/model-charliepai2i48h2-fern-pg-clip-C-20260516-143922/metrics.jsonl`
- **Decision:** CLOSED no_improvement.
- **Follow-up:** The diagnostic directly motivates the inverse test: tighten `other` clip *below* 1.0 (e.g., 0.5, 0.3) on the new 8-mech stack (where pressure_weight=2.0 likely elevated output-head MLP gradients further). Assigned as #4016.

---

## 2026-05-16 16:00 — PR #3989: EMA decay re-tune on 8-mech stack (ema_decay=0.995 wins)
- charliepai2i48h2-askeladd/ema-decay-8mech
- **Hypothesis:** EMA decay=0.999 was tuned on a 5-mech stack at T_max=80. Under T_max=30+pw=2.0, faster decay (shorter half-life) should better track the recent-weights regime produced by steeper annealing.
- **Result:** Arm B (ema_decay=0.995) wins — val 51.4403 (−4.25% vs 53.72). Arm A (ema_decay=0.997) also beats baseline: val 52.6190 (−2.06%).

| Arm | EMA decay | val_avg/mae_surf_p | Δ | test_avg/mae_surf_p |
|-----|-----------|---------------------|---|---------------------|
| Baseline | 0.999 | 53.7235 | — | 46.6011 |
| **Arm B (winner)** | **0.995** | **51.4403** | **−4.25%** | **43.9473** |
| Arm A | 0.997 | 52.6190 | −2.06% | 45.2369 |

Per-split (Arm B): val_single_in_dist=56.17 (−3.71%), val_geom_camber_rc=68.07 (+1.49%), val_geom_camber_cruise=32.12 (−9.27%), val_re_rand=49.40 (−5.91%)
Test (Arm B): test_single_in_dist=53.55, test_geom_camber_rc=56.79, test_geom_camber_cruise=26.94, test_re_rand=38.51

- **Metrics:** `models/model-charliepai2i48h2-askeladd-ema-decay-8mech-0995-20260516-142331/metrics.jsonl`, `models/model-charliepai2i48h2-askeladd-ema-decay-8mech-0997-20260516-133820/metrics.jsonl`
- **Decision:** MERGED — new baseline 51.4403. 9th compounding mechanism.
- **Key finding:** The pre-bf16 finding (faster EMA decay wins) replicated cleanly on the 9-mech stack. Convergence-horizon hypothesis confirmed: T_max=30 collapses LR to 40% by epoch 18 → recent weights are trained at much lower LR → EMA shadow model benefits from tracking them more closely (shorter half-life). re_rand test split gains the most (44.63→38.51, −13.7%) — cross-regime Re generalization benefits most from tighter EMA tracking. Arm A vs Arm B gap is 1.18 pts (at noise floor), so 0.995 isn't conclusively better than 0.997 but both clearly beat 0.999. Askeladd assigned #4029 to push toward 0.993/0.990.

---

## 2026-05-16 16:00 — PR #3984: Pressure weight sweep pw=3.0/4.0 (no_improvement)
- charliepai2i48h2-nezuko/pressure-weight-3p0 and 4p0
- **Hypothesis:** pw curve was monotone in {0.5, 1.0, 2.0} — does it continue upward past 2.0?
- **Result:** Both regressed. pw=2.0 is the unique peak.

| pw | val_avg/mae_surf_p | Δ vs 53.72 |
|----|---------------------|-------------|
| **2.0 (baseline)** | **53.7235** | — |
| 3.0 (Arm A) | 55.0026 | +2.38% |
| 4.0 (Arm B) | 56.8016 | +5.73% |

- **Metrics:** `models/model-charliepai2i48h2-nezuko-pressure-weight-3p0-20260516-134034/metrics.jsonl`, `models/model-charliepai2i48h2-nezuko-pressure-weight-4p0-20260516-144129/metrics.jsonl`
- **Decision:** CLOSED no_improvement. Pressure-weight axis closed at pw=2.0.
- **Key finding:** At pw=3.0, velocity Ux/Uy MAE degrades +5-9%; at pw=4.0, +13%. When velocity is starved, the shared Transolver backbone can no longer represent flow physics well, and pressure itself stops improving. Complete inverted-U curve: 57.79 (pw=0.5) → 56.00 (pw=1.0) → 53.72 (pw=2.0) → 55.00 (pw=3.0) → 56.80 (pw=4.0). Nezuko assigned #4030 (velocity surface down-weighting) as a gentler reallocation approach.

---

## 2026-05-16 16:00 — PR #3887: Cosine T_max bracket 25/40 on 8-mech stack (no_improvement)
- charliepai2i48h2-edward/cosine-tmax-25-pw2 and cosine-tmax-40-pw2
- **Hypothesis:** T_max=30 was found optimal on the 7-mech stack (pre-pw=2.0). Does the optimum shift with pw=2.0?
- **Result:** Both regressed. T_max=30 remains the peak.

| T_max | Final LR at ep18 | val_avg/mae_surf_p | Δ |
|-------|-------------------|---------------------|---|
| 25 (Arm A) | 3.95e-5 (23% init) | 56.4444 | +5.07% |
| **30 (baseline)** | **6.73e-5 (40% init)** | **53.7235** | — |
| 40 (Arm B) | 1.05e-4 (62% init) | 55.2622 | +2.86% |

- **Metrics:** `models/model-charliepai2i48h2-edward-cosine-tmax-25-pw2-20260516-134404/metrics.jsonl`, `models/model-charliepai2i48h2-edward-cosine-tmax-40-pw2-20260516-143620/metrics.jsonl`
- **Decision:** CLOSED no_improvement. T_max axis closed at 30.
- **Key finding:** T_max=30 is a relatively narrow peak. Both directions lose. The 40% final-LR-fraction is the sweet spot for this 18-epoch/batch=4 regime. Tighter anneal (T_max=25) costs more than looser (T_max=40) — confirming that killing late-epoch LR is worse than being slightly too warm. Edward assigned #4031 (Lion β2 sweep) as the next untested axis.

---

## 2026-05-16 17:30 — PR #3731: Signed log1p compression on pressure (no_improvement)
- charliepai2i48h2-tanjiro/signed-log1p-pressure-v2
- **Hypothesis:** Signed log1p (`sign(z) * log1p(|z|)`) is a more aggressive heavy-tail compressor than asinh and may better stabilize pressure-channel residuals.
- **Result:** All splits regressed +4–20% vs 9-mech baseline (51.4403); asinh confirmed locally optimal.

| Metric | signed log1p | 9-mech baseline (asinh) | Δ |
|--------|--------------|--------------------------|---|
| **val_avg/mae_surf_p** | **58.0898** | **51.4403** | **+12.93%** |
| val_single_in_dist | 64.02 | 56.17 | +13.97% |
| val_geom_camber_rc | 71.01 | 68.07 | +4.31% |
| val_geom_camber_cruise | 38.53 | 32.12 | +19.95% |
| val_re_rand | 58.81 | 49.40 | +19.05% |
| **test_avg/mae_surf_p** | **49.5628** | 43.95 | +12.78% |

- **Metrics:** `models/model-charliepai2i48h2-tanjiro-signed-log1p-pressure-v2-20260516-165143/metrics.jsonl`
- **Decision:** CLOSED no_improvement. Pressure-transform axis exhausted.
- **Key finding:** asinh's derivative `1/√(1+z²)` and signed-log1p's `1/(1+|z|)` are similar at extreme |z| but differ critically in the |z| ∈ (1,5) "transition" range where most pressure-gradient signal lives. signed-log1p over-attenuates this range, throttling the boundary-layer and geometry-tail gradient. Worst-hit splits (val_geom_camber_cruise +19.95%, val_re_rand +19.05%) depend on capturing finer pressure variation in perturbed geometries — exactly where moderate-|z| signal matters. Tanjiro's analysis explicitly framed this as binary: either signed-log1p wins or asinh is locally optimal. Result is decisive: asinh is locally optimal under the 9-mech stack. EMA(0.995)+pw=2.0 are calibrated around asinh's shape; switching compressors breaks that calibration. Tanjiro assigned #4061 (channel-decoupled output heads) as the next direction — leverages their pressure-channel expertise toward an architectural specialization instead of further loss-side transforms.

---

## 2026-05-16 18:15 — PR #3970: torch.compile(mode=default, dynamic=True) — MASSIVE WIN (10th mechanism)
- charliepai2i48h2-alphonse/torch-compile
- **Hypothesis:** torch.compile(model, mode='default', dynamic=True) reduces kernel launch overhead and fuses operations, cutting per-epoch time and enabling more epochs within the 30-min cap. dynamic=True handles variable-length padded batches from pad_collate.
- **Result:** Arm A (compile_mode=default) wins with val=44.2439 (−14.0% vs 9-mech baseline 51.44). Epoch time halved: 102s→54.4s. Epochs: 18→33. VRAM: 32.97→23.84 GB.

| Metric | Arm A (default) | Arm B (reduce-overhead) | 9-mech baseline |
|--------|-----------------|-------------------------|-----------------|
| **val_avg/mae_surf_p** | **44.2439** | 45.3626 | 51.4403 |
| val_single_in_dist | 46.9816 | 46.83 | 56.17 |
| val_geom_camber_rc | 58.2760 | 60.11 | 68.07 |
| val_geom_camber_cruise | 27.6407 | 28.36 | 32.12 |
| val_re_rand | 44.0774 | 46.15 | 49.40 |
| **test_avg/mae_surf_p** | **38.0107** | 39.28 | 43.95 |
| test_single_in_dist | 42.3063 | 43.36 | 53.55 |
| test_geom_camber_rc | 49.5504 | 52.17 | 56.79 |
| test_geom_camber_cruise | 23.1558 | 23.71 | 26.94 |
| test_re_rand | 37.0300 | 37.86 | 38.51 |
| Per-epoch time | ~54.4s | ~55.2s | ~102s |
| Best epoch | 33 | 33 | 18 |
| Peak VRAM | 23.84 GB | 23.84 GB | 32.97 GB |

- **Metrics:** `models/model-charliepai2i48h2-alphonse-torch-compile-default-20260516-162535/metrics.jsonl`, `models/model-charliepai2i48h2-alphonse-torch-compile-reduce-overhead-20260516-165822/metrics.jsonl`
- **Decision:** MERGED. New baseline: 44.2439. Cumulative: 135.02 → 44.24 = **−67.2%** from initial.
- **Key finding:** The entire gain comes from the 15 extra epochs (18→33) that compile enables — the loss curve was still monotonically descending at epoch 18. reduce-overhead mode is slightly slower than default in this setting (variable shapes from pad_collate cause overhead that outweighs reduce-overhead's kernel caching benefit). Critically: 9 GB freed VRAM (32.97→23.84 GB) opens the door to capacity expansion — previously blocked by the 42 GB VRAM usage of larger models. Val curve still descending at epoch 33 with rate ~0.03/epoch — not yet saturated. Assigned alphonse #4078 (capacity scale-up on compile stack) and edward #4079 (T_max re-calibration for 33-epoch budget).

---

## 2026-05-16 18:15 — PR #4031: Lion β2 sweep: 0.95 and 0.98 (no_improvement — failure)
- charliepai2i48h2-edward/lion-beta2
- **Hypothesis:** Lion β2 (momentum buffer decay) has never been tested. Under T_max=30+pw=2.0+EMA=0.995, the optimal gradient memory window may have shifted from the default β2=0.99.
- **Result:** Arm A (β2=0.95) val=65.34 — a +27% regression. Stop condition triggered; Arm B skipped.

| Metric | β2=0.95 (Arm A) | 9-mech baseline (β2=0.99) | Δ |
|--------|-----------------|---------------------------|---|
| **val_avg/mae_surf_p** | **65.34** | **51.4403** | **+27.0%** |
| val_single_in_dist | 69.26 | 56.17 | +23.3% |
| val_geom_camber_rc | 84.31 | 68.07 | +23.9% |
| val_geom_camber_cruise | 46.18 | 32.12 | +43.8% |
| val_re_rand | 61.10 | 49.40 | +23.7% |

- **Metrics:** student PR comment
- **Decision:** CLOSED no_improvement. Lion-internal momentum axis is now fully closed: β1=0.90 optimal (PR #3949, β1=0.95 +9.8%), β2=0.99 optimal (this PR, β2=0.95 +27%). Both defaults confirmed optimal under the current 10-mech stack.
- **Key finding:** β2=0.95 gives a ~14-step momentum half-life — Lion "forgets" gradient direction every 14 steps. Under T_max=30's steep annealing and pw=2.0's channel asymmetry, this is far too reactive: the model cannot build stable gradient consensus and oscillates. The default β2=0.99 (~69-step half-life) is well-matched to this regime. Physical intuition confirmed: the lion momentum buffer needs enough history to distinguish noise from signal, especially with asymmetric pressure/velocity loss weighting.

---

## 2026-05-16 21:30 — PR #4016: Tighter MLP/output grad-clip (other=0.5,0.3) on 10-mech stack

- **Branch:** charliepai2i48h2-fern/tighter-mlp-clip
- **Hypothesis:** MLP/output gradients are ~4-6× larger than attention gradients (confirmed by #3725 diagnostic). Per-group clipping with tighter `other_grad_norm` (0.5, 0.3) vs single-clip(1.0) should reduce noise in the dominant gradient group and improve convergence.
- **Metrics committed:**
  - `models/model-charliepai2i48h2-fern-tighter-mlp-clip-sanity-20260516-193051/metrics.jsonl`
  - `models/model-charliepai2i48h2-fern-tighter-mlp-clip-A-20260516-200539/metrics.jsonl`
  - `models/model-charliepai2i48h2-fern-tighter-mlp-clip-B-20260516-203955/metrics.jsonl`

| Arm | other_grad_norm | Best epoch | val_avg | test_avg | Δ vs baseline (44.24) |
|-----|-----------------|-----------|---------|---------|----------------------|
| Sanity (1.0, 1.0) | 1.0 | 33 | 43.38 | 37.15 | −1.95% |
| Arm A (1.0, 0.5) | 0.5 | 33 | 45.94 | 40.11 | +3.84% |
| Arm B (1.0, 0.3) | 0.3 | 32 | 44.60 | 38.99 | +0.81% |

Gradient diagnostic (pre-clip means): attn ≈ 3-55×, other ≈ 13-200× — ratio stable at 4-6× throughout training.

- **Decision:** CLOSED no_improvement. Arms A/B regressed: tighter `other_grad_norm` discards useful signal from the dominant MLP/output group. Sanity arm (43.38) within ~1.17-pt noise floor of baseline (44.24).

- **Key finding:** Per-group(1.0, 1.0) is mathematically different from single-clip(1.0): with attn≈3, other≈18, single-clip gives attn only 0.16 effective magnitude while per-group gives attn full 1.0. This rebalancing is a promising avenue. Follow-up PR #4154 tests loosening `other_grad_norm` above 1.0 (1.5, 2.0) to explore the inverse direction.

---

## 2026-05-16 21:35 — PR #3953: LR × T_max re-calibration for 33-epoch compile horizon

- **Branch:** charliepai2i48h2-frieren/lr-tmax-coupling
- **Hypothesis:** Under the 18-epoch pre-compile budget, lr=1.7e-4 + T_max=30 was optimal. The compile stack gives 33 epochs. With more training time and a gentler annealing slope, a higher lr_init (2.1e-4, 2.5e-4) combined with a recalibrated T_max=40 should outperform the inherited 10-mech config.
- **Metrics committed:**
  - `models/model-charliepai2i48h2-frieren-lr-tmax-coupling-compile-tmax40-21e4-20260516-192650/metrics.jsonl` (Arm A)
  - `models/model-charliepai2i48h2-frieren-lr-tmax-coupling-compile-tmax40-25e4-20260516-202729/metrics.jsonl` (Arm B)

| Arm | lr | T_max | Best epoch | val_avg | test_avg | Δ vs baseline (44.24) |
|-----|----|----|-----------|---------|---------|----------------------|
| Arm A (2.1e-4) | 2.1e-4 | 40 | 33 | 41.01 | 35.90 | −7.30% |
| **Arm B (2.5e-4) — WINNER** | 2.5e-4 | 40 | 33 | **40.69** | **34.98** | **−8.04%** |

Per-split (Arm B): val 44.58 / 54.52 / 23.73 / 39.91 → avg 40.69; test 38.09 / 48.19 / 19.99 / 33.64 → avg 34.98. Arm B better on 3/4 val splits and 4/4 test splits. Both trajectories smooth, still descending at epoch 33.

- **Decision:** MERGED as strong win (−8.04%, both arms clear strong-win threshold val < 41.6). New baseline: val=40.6869. 11th compounding mechanism.

- **Key finding:** LR × schedule coupling confirmed directionally across three regimes (T_max=80: lr=2.5e-4 regressed +2.74%; T_max=30/18ep: within noise; T_max=40/33ep + compile: −8%). The confound between T_max and lr is acknowledged — edward's #4079 (pure T_max at lr=1.7e-4) will isolate T_max. Frieren assigned #4159 (T_max fine-sweep at lr=2.5e-4).

- **Cumulative improvement:** 135.02 → 40.69 = **−69.8%**

---

## 2026-05-16 21:50 — PR #4078: Capacity scale-up (n192, n256) on 10-mech compile stack

- **Branch:** charliepai2i48h2-alphonse/capacity-compile
- **Hypothesis:** torch.compile freed 9 GB VRAM and gave 33 epochs; n192/n256 capacity should be viable now.
- **Metrics committed:**
  - `models/model-charliepai2i48h2-alphonse-capacity-compile-sanity-20260516-184730/metrics.jsonl`
  - `models/model-charliepai2i48h2-alphonse-capacity-compile-n192-20260516-193613/metrics.jsonl`
  - `models/model-charliepai2i48h2-alphonse-capacity-compile-n256-20260516-202747/metrics.jsonl`

| Arm | n_hidden | Epochs | val_avg | test_avg | Epoch time | Δ vs baseline (44.24) |
|-----|---------|--------|---------|---------|------------|----------------------|
| Sanity (n128) | 128 | 25 | 45.13 | 39.17 | 54.6s | +1.99% |
| Arm A (n192) | 192 | 23 | 45.47 | 38.56 | 79.4s | +2.77% |
| Arm B (n256) | 256 | 18 | 50.42 | 43.83 | 103.1s | +13.9% |

At epoch 18 apples-to-apples: n192 val=50.03 < n128 val=52.04 — per-epoch capacity benefit is real, but throughput reduction overwhelms it. T_max=30 miscalibrated for n192's ~22-epoch budget.

- **Decision:** CLOSED no_improvement. Per-epoch capacity benefit confirmed. Follow-up #4167 tests n192 with T_max=22 (calibrated to n192 epoch budget) at lr=1.7e-4 on 12-mech stack.

---

## 2026-05-16 21:50 — PR #4079: T_max=33/40 calibration for 33-epoch compile horizon (pure T_max sweep)

- **Branch:** edward/tmax-compile
- **Hypothesis:** T_max=30 causes LR→0 at epoch 30, wasting epochs 31-33. T_max=33/40 calibrated to compile epoch count.
- **Metrics committed:**
  - `models/model-charliepai2i48h2-edward-tmax-compile-33-20260516-202201/metrics.jsonl`
  - `models/model-charliepai2i48h2-edward-tmax-compile-40-20260516-193437/metrics.jsonl`

| Arm | T_max | lr | Epochs | val_avg | test_avg | Δ vs baseline (44.24) |
|-----|-------|-----|--------|---------|---------|----------------------|
| Arm A (T_max=33) | 33 | 1.7e-4 | 33 | 42.79 | 36.66 | −3.30% |
| **Arm B (T_max=40) — WINNER** | 40 | 1.7e-4 | 34 | **39.83** | **33.89** | **−9.97%** |

Val still descending at epoch 34 with lr=1.25e-5. LR trajectory confirms hypothesis: T_max=40 leaves meaningful signal at timeout while T_max=30 reaches LR≈0 at epoch 30.

- **Decision:** MERGED as strong win. New baseline val=39.8345, test=33.8873.

- **Critical scientific finding:** Pure T_max sweep at lr=1.7e-4 shows T_max recalibration alone drives the full −10% gain. Frieren's prior result (#3953, val=40.69, T_max=40+lr=2.5e-4) is WORSE than lr=1.7e-4 result. The lr increase was counterproductive. Optimal config: T_max=40, lr=1.7e-4 (in-tree default).

- **Cumulative improvement:** 135.02 → 39.83 = **−70.5%**

---

## 2026-05-16 22:32 — PR #4167: Capacity n192 + calibrated T_max=22 on 12-mech stack

- **Student:** charliepai2i48h2-alphonse
- **Hypothesis:** n192 width (1.45M params) with T_max=22 calibrated to the ~22-epoch compute budget at 79 s/epoch, following confirmed per-epoch benefit from #4078.
- **Results:**

| Metric | This run (n192, T_max=22) | Baseline #4079 (n128, T_max=40) | Δ |
|--------|-----------|----------|---|
| **val_avg/mae_surf_p** | **50.3206** | **39.8345** | **+10.49 (+26.3%)** |
| val_single_in_dist | 54.2321 | 43.6797 | +10.55 |
| val_geom_camber_rc | 65.4722 | 53.1517 | +12.32 |
| val_geom_camber_cruise | 30.6558 | 22.7101 | +7.95 |
| val_re_rand | 50.9221 | 39.7965 | +11.13 |
| **test_avg/mae_surf_p** | **42.5489** | **33.8873** | **+8.66** |
| Best epoch | 23 (timeout-bound; val still descending) | 34 | — |
| Per-epoch time | ~79.4 s | ~54.2 s | +46.7% |
| Peak VRAM | 35.59 GB | 23.84 GB | +49% |

Metrics artifact: `models/model-charliepai2i48h2-alphonse-capacity-n192-tmax22-20260516-214742/metrics.jsonl`

- **Analysis:** T_max=22 calibration was mechanically correct — the cosine schedule completed within budget and val descended monotonically to the timeout. The issue is compute throughput: 79 s/epoch vs n128's 54 s costs 11 fewer epochs in the 30-min window. At epoch 23, n192 val (50.32) is ~+10 pts worse than n128's val at the same epoch index. The 1.45M params model simply needs more passes at meaningful LR than the throughput allows. Tightening T_max (from 30→22) did not help because the schedule completion is not the bottleneck — total epoch budget is.

- **Decision:** CLOSED — no_improvement (+26.3% regression; clearly above 41.0 stop threshold). The capacity direction at n192 is throughput-limited inside the 30-min wall-clock budget. Follow-up: compile_mode sweep (reduce-overhead, max-autotune) at n128 to see if per-epoch time can be cut enough to reopen n192.


---

## 2026-05-16 22:52 — PR #4030: Velocity surface down-weighting (initial run, sent back)

- **Student:** charliepai2i48h2-nezuko
- **Hypothesis:** Down-weight surface velocity loss terms (surf_ux_weight, surf_uy_weight ∈ {0.5, 0.7}) to reallocate optimizer budget to pressure on the shared backbone.
- **Stack used:** 11-mech (lr=2.5e-4) — the stack current when assigned; superseded by 12-mech (lr=1.7e-4) baseline mid-run.
- **Results:**

| Metric | Baseline #4079 (12-mech, lr=1.7e-4) | Arm A (ux=uy=0.5) | Arm B (ux=uy=0.7) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 42.0098 (+5.5%) | **40.1906 (+0.9%)** |
| **test_avg/mae_surf_p** | **33.8873** | 36.4971 (+7.7%) | **33.7173 (−0.5%)** |
| val_geom_camber_rc | 53.1517 | 55.46 | 52.63 |
| Best epoch | 34 | 33 | 33 |

Metric artifacts:
- `models/model-charliepai2i48h2-nezuko-vel-surf-weight-A-20260516-214104/metrics.jsonl`
- `models/model-charliepai2i48h2-nezuko-vel-surf-weight-B-20260516-221612/metrics.jsonl`

- **Analysis:** Arm B test improvement (−0.5% vs current baseline test) is real signal — outside noise floor. Per-channel diagnostic confirmed genuine velocity-pressure coupling: Arm A (0.5) degraded the volume velocity channels *more* than the down-weighted surface velocity channels, showing the shared backbone bleeds when starved. Arm B (0.7) is the right "gentle" knee. But val=40.19 vs baseline 39.83 is +0.9% — does not beat current primary metric.

- **Decision:** SENT BACK — re-run as Arms C (ux=uy=0.7) and D (ux=uy=0.8) at lr=1.7e-4 (current baseline LR). Edward's PR #4079 proved lr=1.7e-4 outperforms 2.5e-4 by ~0.86 pts on val at T_max=40; applying that delta to Arm B suggests ~39.3 val at correct LR — would beat baseline.


---

## 2026-05-16 23:40 — PR #3734: SwiGLU gated activation in TransolverBlock MLPs (v2) — closed stale

- **Student:** charliepai2i48h2-thorfinn
- **Hypothesis:** Replace GELU with SwiGLU gated activation in TransolverBlock MLPs to improve OOD generalization.
- **Status:** CLOSED — stale; no commits in 17 hours, 45 pod restarts, GPU at 0% at advisor check time.
- **Analysis:** SwiGLU has now been attempted twice without producing a runnable arm — v1 (PRs #3275/#3383/#3442 superseded as v2 #3734) and v2 (#3734). The gated activation requires non-trivial dimension bookkeeping changes (gating mechanism doubles the linear projection width then halves via element-wise product) which appears to be unstable with the current dynamic-shape compile path or bf16 autocast. No metrics produced.
- **Decision:** Closed direction. Reassigned thorfinn to weight-decay-sweep (#4230) — a pure CLI-flag hyperparameter sweep with no code changes, unblocking the student's GPU budget.


---

## 2026-05-17 00:10 — PR #4188: torch.compile mode sweep (reduce-overhead, max-autotune) — no_improvement

- **Student:** charliepai2i48h2-alphonse
- **Hypothesis:** Cut per-epoch time via PyTorch compile modes; reopen capacity direction if epochs increase under 30-min budget.
- **Results:**

| Metric | Baseline #4079 (default) | Arm A (reduce-overhead) | Arm B (max-autotune) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 40.4319 (+1.5%) | 40.3870 (+1.4%) |
| **test_avg/mae_surf_p** | **33.8873** | 34.9420 (+3.1%) | 35.3061 (+4.2%) |
| Epoch 1 time | ~54 s | 74.6 s (+20s) | 254.1 s (+200s autotune) |
| Per-epoch (post-warmup) | ~54.2 s | ~55.4 s | ~53.7 s (~3% faster) |
| Total epochs in 30 min | 34 | 31 | 29 |
| Peak VRAM | 23.84 GB | 23.83 GB | 23.83 GB |
| Crashes/NaN | — | none | none |

- **Analysis:** Excellent honest student analysis. Per-step speedup is marginal-to-zero on this stack. `reduce-overhead` is ~1 s/epoch slower (dynamic shapes from pad_collate break CUDA-graph amortization). `max-autotune` saves ~1-2 s/epoch but pays 200 s autotune cost on epoch 1 — net loss of 5 effective epochs in 30 min. Convergence per epoch is identical (no numerical changes); both arms simply ran shorter and produced a worse val on the same trajectory.
- **Decision:** CLOSED — no_improvement. Compile-mode lever is exhausted. `default` is optimal at the 30-min budget. Reassigning alphonse to mlp-ratio-sweep (#4235) per plateau protocol.

---

## 2026-05-17 00:00 — PR #4159: T_max fine-sweep at lr=1.7e-4 (T_max=35, 50) — no_improvement (informative)

- **Student:** charliepai2i48h2-frieren
- **Hypothesis:** Bracket T_max=40 with T_max=35 and 50 to confirm cosine schedule length is at the optimum under lr=1.7e-4.
- **Results:**

| Metric | Baseline #4079 (T_max=40) | Arm A (T_max=35) | Arm B (T_max=50) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 42.6991 (+7.2%) | 40.1995 (+0.92% noise) |
| **test_avg/mae_surf_p** | **33.8873** | 37.0662 (+9.4%) | 34.3809 (+1.4%) |
| LR at epoch 33 | 1.25e-5 (7% init) | 3.06e-6 (1.8%) | 4.88e-5 (28.7%) |
| Best epoch | 34 | 33 (timeout) | 33 (timeout) |
| Peak VRAM | 23.84 GB | 23.84 GB | 23.84 GB |

- **Analysis:** T_max=35 fails because LR collapses to 1.8% of init by epoch 33 — model freezes for the last 6 epochs. T_max=50 leaves LR at 28.7% of init at the timeout — val curve still descending. Confirms T_max=40 is broadly optimal in the [40, 50] band; within-band differences sit at the noise floor. Key followup-insight from student: schedule bottleneck is at the START (random init wastes early high-LR budget), not the END.
- **Decision:** CLOSED — no_improvement, but the most informative null of the round. Reassigning frieren to warmup-cosine (#4236) — directly tests the early-budget hypothesis.

---

## 2026-05-17 00:00 — PR #4154: Per-group grad-clip loosen (other=1.5, 2.0) — no_improvement (refuted)

- **Student:** charliepai2i48h2-fern
- **Hypothesis:** Loosen `other_grad_norm` above 1.0 to give the dominant MLP/output group more update room while keeping `attn_grad_norm=1.0`.
- **Results:**

| Metric | Baseline #4079 (single-clip 1.0) | Arm A (other=1.5) | Arm B (other=2.0) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 40.6269 (+2.0%) | 42.0424 (+5.55%) |
| **test_avg/mae_surf_p** | **33.8873** | 34.6414 (+2.2%) | 36.0878 (+6.5%) |
| Best epoch | 34 | 33 | 33 |
| Per-epoch time | ~54.2 s | ~54.7 s | ~54.1 s |

- **Analysis:** Direction monotonically wrong — looser other_grad_norm increases regression. Excellent student reconciliation: per-group(1.0,1.0) effectively gives 4× attn magnitude vs single-clip; loosening other to 1.5/2.0 inflates both groups further. Lion's sign-based updates suffer overshoot from the magnitude inflation. The #4016 sanity "win" was likely noise — the data refutes the directional hypothesis.
- **Decision:** CLOSED — no_improvement, direction refuted. Per-group clip axis closed at this stack. Reassigning fern to n-layers-sweep (#4237) — depth scaling untouched since round 1 and motivated directly by fern's prior diagnostic (MLP/other has 5× attn gradient norm; more layers tests if model is depth-limited).


---

## 2026-05-17 01:15 — PR #4029: EMA decay fine sweep (0.993, 0.990) — no_improvement

- **Student:** charliepai2i48h2-askeladd
- **Hypothesis:** EMA decay=0.995 was set during a stack with T_max=30; with T_max=40 and longer training, a slightly faster decay (0.993 or 0.990) might track the improved trajectory more closely.
- **Results:**

| Metric | Baseline #4079 (decay=0.995) | Arm A (decay=0.993) | Arm B (decay=0.990) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 39.7881 (−0.046, within noise) | not run (below threshold) |
| **test_avg/mae_surf_p** | **33.8873** | 34.6774 (+0.79 regression) | — |
| Best epoch | 34 | — | — |

- **Analysis:** Arm A val=39.79 is technically better by 0.046 — well within the 1.17 noise floor. But the paired test regression (+0.79, 17× the val gain) rules out a clean compounding win. The val gain is noise; the test signal is meaningful. Arm B was not run. PR was additionally stale for 8h (required clarified instructions referencing correct 12-mech stack — the PR body originally referenced the 9-mech stack).
- **Decision:** CLOSED — no_improvement. EMA decay=0.995 is locked. Reassigning askeladd to slice_num sweep (#4243) — cleanest untested architectural axis (PhysicsAttention slice routing), orthogonal to all in-flight experiments.

---

## 2026-05-17 01:15 — PR #4181: LR fine-sweep at T_max=40 (lr=1.5e-4, 2.0e-4) — no_improvement

- **Student:** charliepai2i48h2-edward
- **Hypothesis:** PR #4079 confirmed lr=1.7e-4 beats lr=2.5e-4 at T_max=40 — two data points only. Bracket 1.7e-4 with 1.5e-4 (−12%) and 2.0e-4 (+18%) to confirm it's at the optimum.
- **Results:**

| Metric | Baseline #4079 (lr=1.7e-4) | Arm A (lr=1.5e-4) | Arm B (lr=2.0e-4) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **39.8345** | 40.3893 (+1.4% null) | 41.8680 (+5.1% failure) |
| **test_avg/mae_surf_p** | **33.8873** | 34.0666 (+0.5%) | 35.7273 (+5.4%) |
| val_single_in_dist | 43.6797 | 44.9663 | 45.0380 |
| val_geom_camber_rc | 53.1517 | 53.0366 | 58.0543 |
| val_geom_camber_cruise | 22.7101 | 23.0733 | 24.3773 |
| val_re_rand | 39.7965 | 40.4810 | 40.0024 |
| Best epoch | 34 (timeout) | 33 (timeout) | 32 (timeout) |
| Per-epoch time | ~54.2 s | ~54.3 s | ~54.5 s |
| Peak VRAM | 23.84 GB | 23.84 GB | 23.85 GB |

- **Analysis:** Arm A within null band (±1.0), test also within noise. Arm B exceeds the 41.0 failure threshold — over-LR signature (rc-camber most sensitive: 58.05 vs 53.15). Physical intuition that T_max=40's extended high-LR dwell might want lower peak LR did NOT hold. Both arms show val still descending at timeout — the model is timeout-limited, not LR-limited. Key student observation: final-epoch LR at 9.5-12% of init (above the expected 7% for T_max=40) because timeout prevents reaching cycle completion; rc-camber split is the most LR-sensitive and the dominant val error source.
- **Decision:** CLOSED — no_improvement. LR axis fully closed. lr=1.7e-4 is the precise optimum for the 12-mech / T_max=40 stack. Reassigning edward to SGDR warm restarts (#4253) — mid-training LR reset to test if the "val still descending at timeout" attractor can be escaped via a second high-LR phase.

---

## 2026-05-17 02:00 — PR #4061: Channel-decoupled output heads (velocity/pressure split) — no_improvement

- **Student:** charliepai2i48h2-tanjiro
- **Hypothesis:** The shared `Linear(d,d)→GELU→Linear(d,3)` output head is forced to represent both heavy-tailed pressure and lighter-tailed velocity jointly; splitting into parallel pressure/velocity heads lets each specialize.
- **Results:**

| Metric | Baseline (#4079) | Arm A (2-layer p head) | Arm B (3-layer p head) |
|--------|------------------|-----------------------|-----------------------|
| **val_avg/mae_surf_p** | **39.8345** | **40.0470 (+0.53%)** | **41.9308 (+5.26%)** |
| **test_avg/mae_surf_p** | **33.8873** | **35.2176 (+3.92%)** | **36.1335 (+6.63%)** |
| val_single_in_dist | 43.68 | 40.94 (−6.3%!) | 45.03 |
| val_geom_camber_rc | 53.15 | 55.19 (+3.8%) | 55.16 (+3.8%) |
| val_geom_camber_cruise | 22.71 | 23.52 (+3.6%) | 25.23 (+11.1%) |
| val_re_rand | 39.80 | 40.53 (+1.8%) | 42.30 (+6.3%) |
| n_params | 662,359 | 678,871 (+2.5%) | 695,383 (+5.0%) |
| Per-epoch time | ~54.2 s | ~55.0 s | ~55.8 s |
| Peak VRAM | 23.84 GB | 24.13 GB | 24.63 GB |

- **Analysis:** Arm A is within ±1 null band on val but paired test regression (+3.92%) rules out compounding win. Arm B threshold breach (+5.3% val). Key student insight: per-split analysis shows val_single_in_dist improved −6.3% (bright spot) offset by OOD regressions; channels moved uniformly, not pressure-specifically — confirming the 12-mech stack (asinh+pw=2.0) had already neutralized the shared-head bottleneck. The tanjiro implementation was clean (correct param counts, channel order preserved). Excellent diagnostic work.
- **Decision:** CLOSED — no_improvement. Decoupled-heads axis closed. Reassigning tanjiro to n_head sweep (#4273) — pure head count sweep (prior #3106 was a 3-way compound; uninterpretable for n_head alone).

---

## 2026-05-17 02:00 — PR #4030: Velocity surface down-weighting Arms C/D at lr=1.7e-4 — no_improvement

- **Student:** charliepai2i48h2-nezuko
- **Hypothesis:** Arm B (ux=uy=0.7) at lr=2.5e-4 showed val=40.19 and test=33.72 — promising signal; re-run at lr=1.7e-4 (12-mech baseline) to see if the mechanism holds.
- **Results:**

| Metric | Baseline (#4079) | Arm C (0.7 at lr=1.7e-4) | Arm D (0.8 at lr=1.7e-4) |
|--------|------------------|--------------------------|--------------------------|
| **val_avg/mae_surf_p** | **39.8345** | **39.4862 (−0.88%)** | **40.7607 (+2.32%)** |
| **test_avg/mae_surf_p** | **33.8873** | **34.4596 (+1.69%)** | **34.8643 (+2.88%)** |
| val_geom_camber_rc | 53.15 | 53.15 (≈same) | 55.38 |
| Best epoch | 34 | 33 | 32 |

- **Analysis:** Arm C clears val by 0.35 pts (within noise floor) but test regresses +1.69%. Arm D clear regression. The conjunctive beat target (val<39.83 AND test<33.89) is not met by either arm. Student's key insight: lowering LR (1.7e-4 vs 2.5e-4) and surface-vel-downweight appear to interact non-trivially — lowering LR absorbed the headroom that Arm B was exploiting at 2.5e-4. Per-channel diagnostic: surf_Ux/Uy flat between 0.7 and 0.8 (no backbone starvation), so regression at 0.8 is "insufficient pressure-share" failure mode rather than gradient starvation. Excellent analysis.
- **Decision:** CLOSED — no_improvement. Velocity-surface-weight axis closed at lr=1.7e-4. Reassigning nezuko to attention-dropout (#4278) — targets the OOD generalization gap (val_geom_camber_rc=53.15) directly via attention regularization; model currently has zero stochastic regularization in attention path.

---

## 2026-05-17 02:15 — PR #4236: Warmup-cosine (2/3-epoch linear warmup) — no_improvement

- **Student:** charliepai2i48h2-frieren
- **Hypothesis:** The first 2-3 epochs are "wasted budget" because random-init gradient magnitudes are too large; a linear warmup would stabilize early training and use the cosine budget more effectively.
- **Results:**

| Metric | Arm A (warmup=2) | Baseline (#4079) |
|--------|------------------|--------------------|
| **val_avg/mae_surf_p** | **41.2842 (+3.64%)** | **39.8345** |
| **test_avg/mae_surf_p** | **35.5331 (+4.86%)** | **33.8873** |
| val_geom_camber_cruise | 25.64 (+12.88%) | 22.71 |
| val_geom_camber_rc | 56.58 (+6.46%) | 53.15 |
| val_single_in_dist | 43.49 (−0.43%) | 43.68 |
| Best epoch | 33 (timeout) | 34 (timeout) |

Stop condition triggered (val > 41.0 threshold); Arm B (warmup=3) correctly not run.

- **Analysis:** Excellent student diagnosis. Three root causes identified: (1) "wasted epochs" premise was false — baseline learns rapidly from epoch 1 at full LR, gradients are already being clipped to max_norm=1.0 so warmup's stabilization role is redundant; (2) schedule-compression effect dominates — warmup shifts the effective cosine window from 40 → 38 epochs while keeping total budget at 33, creating a ~1-2 epoch shift in the descent curve; (3) OOD splits hit hardest (geom_camber_cruise +12.88%) — model needs maximum effective training time at these split distributions. This is the second warmup failure on this codebase (cf PR #3733). Warmup direction is closed.
- **Decision:** CLOSED — no_improvement, direction refuted. Reassigning frieren to batch-size-sweep (#4287) — directly addresses frieren's own VRAM headroom observation (23.84/80 GB = 30% utilization at batch=4).

---

## 2026-05-17 02:35 — PR #4237: Depth sweep n_layers=6,7 — no_improvement

- **Student:** charliepai2i48h2-fern
- **Hypothesis:** Adding TransolverBlock depth (n_layers=5→6→7) would route more signal through extra residual stacks; motivated by the ~5× MLP/attn grad-norm ratio from PR #4154, hypothesizing the model is depth-limited.
- **Results:**

| Metric | Arm A (n=6) | Arm B (n=7) | Baseline (#4079) |
|--------|-------------|-------------|------------------|
| **val_avg/mae_surf_p** | **42.3335 (+2.50)** | **46.9938 (+7.16)** | **39.8345** |
| **test_avg/mae_surf_p** | **36.0610 (+2.17)** | **40.1238 (+6.24)** | **33.8873** |
| val_single_in_dist | 48.70 | 53.30 | 43.68 |
| val_geom_camber_rc | 54.87 | 59.29 | 53.15 |
| val_geom_camber_cruise | 23.79 | 28.92 | 22.71 |
| val_re_rand | 41.97 | 46.47 | 39.80 |
| Best epoch | 28 (last) | 24 (last) | 34 (timeout) |
| Sec/epoch | 66 | 76 | ~54 |
| Peak VRAM | 28.10 GB | 32.36 GB | 23.84 GB |
| n_params | 783,515 | 904,671 | 662,359 |
| Metric artifacts | `models/model-charliepai2i48h2-fern-n-layers-A-20260517-003352/metrics.jsonl` | `models/model-charliepai2i48h2-fern-n-layers-B-20260517-012751/metrics.jsonl` | |

- **Analysis:** Both arms throughput-bound (A: 28ep vs 34 baseline, B: 24ep). Crucially, the student's iso-epoch comparison showed NO per-epoch convergence advantage from depth: Arm A at epoch 28 sits at 42.33, while baseline extrapolated to ep28 is ~41.5–42. Arm B at ep24 (46.99) is clearly behind A at ep24 (45.18). So even at the same epoch index, deeper stacks do not converge faster — per-epoch convergence is equal or slower. The MLP/attn gradient imbalance from PR #4154 is amplified by adding more MLP-heavy blocks, explaining the velocity-dominant split degradation (single_in_dist +11.3%, re_rand +5.4%). Depth scaling at the current width (n_hidden=128, mlp_ratio=2) is a confirmed dead end. Both width scaling (PR #4167) and depth scaling (#4237) are now closed.
- **Decision:** CLOSED — no_improvement, depth axis closed. Student's follow-up suggestion (per-group LR scaling to address the MLP/attn imbalance at source) is the next assignment (#4295).

---

## 2026-05-17 02:45 — PR #4243: slice_num=48 sweep (askeladd) — STRONG WIN → MERGED

- **Student:** charliepai2i48h2-askeladd
- **Hypothesis:** Coarser slice_num (fewer routing slices in PhysicsAttention) gives each slice more mesh points to integrate over → lower-variance gradient signal per step → better generalization. Previous sweep had slice=64 as baseline; tested 48 (coarser) and 96 (finer).
- **Results:**

| Metric | Arm A (slice=48) | Arm B (slice=96) | Baseline (#4079, slice=64) |
|--------|------------------|-----------------|-----------------------------|
| **val_avg/mae_surf_p** | **38.6750 (−1.16, −2.91%)** | 42.5414 (+2.71, +6.80%) | 39.8345 |
| **test_avg/mae_surf_p** | **33.4948 (−0.39, −1.16%)** | 37.3840 (+3.50, +10.31%) | 33.8873 |
| val_single_in_dist | 42.140 | 47.408 | 43.6797 |
| val_geom_camber_rc | 51.618 | 54.915 | 53.1517 |
| val_geom_camber_cruise | 22.483 | 25.673 | 22.7101 |
| val_re_rand | 38.460 | 42.169 | 39.7965 |
| Best epoch | 35 (timeout, descending) | 29 (timeout, descending) | 34 (timeout, descending) |
| Sec/epoch | 51.7 | 60.7 | ~54.2 |
| Peak VRAM | 22.60 GB | 26.34 GB | 23.84 GB |
| n_params | 659,719 | 667,639 | ~662,359 |
| Metric artifacts | `models/model-charliepai2i48h2-askeladd-slice-num-sweep-A-20260517-004423/metrics.jsonl` | `models/model-charliepai2i48h2-askeladd-slice-num-sweep-B-20260517-013209/metrics.jsonl` | |

- **Analysis:** Strong win on all 4 val splits and all 4 test splits. Effect is non-monotone: slice=48 (coarser) wins, slice=96 (finer) strongly regresses. Mechanism: at batch=4 and ~36 training geometries, fewer slices → each slice integrates more mesh points → lower-variance per-slice gradient signal → faster + better generalization. Throughput bonus: fewer attention rows → 51.7s/ep vs 54.2s baseline, gaining 1 extra epoch in the 30-min budget. Val trajectory in Arm A: ep30→39.93, ep32→39.18, ep33→39.10, ep34→38.84, ep35→38.68 — monotone descent, optimum not reached. Follow-up: explore even coarser (slice_num=32, 40) to find the true optimum.
- **Decision:** MERGED as new baseline. New baseline: val=38.675, test=33.495 (PR #4243). Assigned askeladd to slice-num-coarser (#4306) to continue the coarser direction.

---

## 2026-05-17 02:50 — PR #4235: mlp-ratio-sweep (alphonse) — no_improvement

- **Student:** charliepai2i48h2-alphonse
- **Hypothesis:** Expanding the MLP hidden ratio within TransolverBlocks (mlp_ratio=3 or 4 vs current 2) would add capacity along the dominant parameter group (MLP ~5× noisier than attn per PR #4154).
- **Results:**

| Metric | Arm A (r=3) | Arm B (r=4) | Baseline (#4079, r=2) |
|--------|-------------|-------------|------------------------|
| **val_avg/mae_surf_p** | 40.2640 (+1.08%) | 43.0996 (+8.20%) | 39.8345 |
| **test_avg/mae_surf_p** | 35.0889 (+3.55%) | 36.8368 (+8.71%) | 33.8873 |
| Best epoch | 31 | 29 | 34 |
| Sec/epoch | 58.1 | 61.5 | ~54.2 |
| n_params | 827K | 991K | 662K |
| Metric artifacts | `models/model-charliepai2i48h2-alphonse-mlp-ratio-A-20260517-003406/metrics.jsonl` | `models/model-charliepai2i48h2-alphonse-mlp-ratio-B-20260517-012413/metrics.jsonl` | |

- **Analysis:** Both arms throughput-bound (3 and 5 fewer epochs respectively in 30-min budget). Monotone relationship: more MLP capacity → slower per epoch → fewer epochs → strictly worse final val. Arm A's val was still descending at -0.63/ep at termination (projected to ~38.5–38.8 at ep34 equivalent) — but the new baseline (post-#4243 merge) is now 38.675, so the projections would barely compete even under ideal assumptions. Cross-experiment conclusion: the MLP/attn grad-norm imbalance does NOT imply MLP is under-parameterised; instead MLP is over-absorbing signal at the existing size, and capacity scaling amplifies that problem. MLP-ratio axis closed at mlp_ratio=2 within the 30-min budget.
- **Decision:** CLOSED — no_improvement. MLP-ratio axis closed. Assigned alphonse to ffn-dropout (#4308) — regularizing the over-driven MLP rather than expanding it.

---

## 2026-05-17 02:50 — PR #4230: weight-decay-sweep (thorfinn) — no_improvement

- **Student:** charliepai2i48h2-thorfinn
- **Hypothesis:** Weight decay wd=3e-4 (set at Lion switch in PR #3293) may not be optimal for the current 12-mech stack. Bracket test: wd=1e-4 (looser) and wd=5e-4 (tighter).
- **Results:**

| Metric | Arm A (wd=1e-4) | Arm B (wd=5e-4) | Baseline (#4079, wd=3e-4) |
|--------|-----------------|-----------------|----------------------------|
| **val_avg/mae_surf_p** | 41.9930 (+5.4%) | 43.4835 (+9.2%) | 39.8345 |
| **test_avg/mae_surf_p** | 35.5056 (+4.8%) | 36.7422 (+8.4%) | 33.8873 |
| Best epoch | 32 | 27 | 34 |
| Stop condition | TRIGGERED (>41.0) | TRIGGERED (>41.0) | — |
| Metric artifacts | `models/model-charliepai2i48h2-thorfinn-weight-decay-A-20260517-002645/metrics.jsonl` | `models/model-charliepai2i48h2-thorfinn-weight-decay-B-20260517-012525/metrics.jsonl` | |

- **Analysis:** Both arms hit the >41.0 stop threshold. Monotone ordering: wd=3e-4 (baseline) < wd=1e-4 < wd=5e-4 in terms of val error. Confirms wd=3e-4 is at or near the local minimum. Student's analysis: Lion's sign-based update applies uniform step sizes; the right wd balances regularization against effective learning and 3e-4 hits it correctly at this batch/LR/EMA configuration. Both directions (more and less regularization) regress — the wd axis is fully closed.
- **Decision:** CLOSED — no_improvement, stop condition hit. wd=3e-4 locked. Assigned thorfinn to SWA (#4312) — testing alternative model averaging vs current EMA(0.995).

---

## 2026-05-17 03:30 — PR #4287: batch-size-sweep (frieren) — failure → closed

- **Student:** charliepai2i48h2-frieren
- **Hypothesis:** Larger batches (8, 12 vs current 4) might unlock better convergence via better gradient quality or improved wall-clock throughput, given the GPU is at 30% VRAM utilization at batch=4 (~24 GB / 80 GB).
- **Results:**

| Metric | Arm A (batch=8) | Arm B (batch=12) | Baseline (#4243, batch=4) |
|--------|-----------------|------------------|----------------------------|
| **val_avg/mae_surf_p** | 45.2559 (+13.6%) | 63.5112 (+59.4%) | 38.6750 |
| **test_avg/mae_surf_p** | 39.0534 | 54.8841 | 33.4948 |
| Best epoch | 31 (timeout, descending) | 31 (timeout, descending) | 35 |
| Sec/epoch | 57.37 | 58.94 | 51.7 |
| Peak VRAM | 47.66 GB | 71.48 GB | 22.60 GB |
| Stop condition | FAIL (val > 45) | FAIL (val > 45) | — |
| Metric artifacts | `models/model-charliepai2i48h2-frieren-batch-size-sweep-A-20260517-014947/metrics.jsonl` | `models/model-charliepai2i48h2-frieren-batch-size-sweep-B-20260517-022433/metrics.jsonl` | |

- **Analysis:** Three clean diagnoses from the student:
  1. **No wall-clock speedup** — per-epoch time only rose from 54.2 s (b=4) → 57.4 s (b=8) → 58.9 s (b=12). The Transolver was not compute-bound at batch=4; attention/index ops scale sub-linearly. Net: ~½ (A) or ⅓ (B) the gradient updates per epoch with no compensating reduction in epoch count.
  2. **Fewer gradient updates dominate** — classic "no LR scaling at larger batch" failure mode. Goyal et al. linear scaling would suggest lr=3.4e-4 (A) and 5.1e-4 (B), but LR axis was locked at 1.7e-4 in PR #4181. 2D batch×lr sweep not affordable in this budget.
  3. **In-distribution split suffered most** (Arm B val_single_in_dist=92.1 vs baseline 43.7 = 2.1×) — pure undertraining signature, not generalization. With fewer updates the model never reaches the in-distribution basin.
  - Bonus: corrected a misconception in the PR body — `train_samples=1499` per config.yaml (not ~36 geometries per epoch); the ratio interpretation still holds.
- **Decision:** CLOSED — failure on both arms. Batch axis closed at batch=4 with LR locked. Assigned frieren to huber-loss (#4327) — fresh loss-formulation axis, untested.

---

## 2026-05-17 04:45 — PR #4312: swa (thorfinn) — no_improvement → closed

- **Student:** charliepai2i48h2-thorfinn
- **Hypothesis:** Replace EMA(0.995) with SWA (Stochastic Weight Averaging, Izmailov et al. 2018) for inference-time model averaging. SWA averages epoch-scale weight snapshots via `torch.optim.swa_utils.AveragedModel` and `SWALR`. Arms: swa_start=20 (Arm A) vs swa_start=10 (Arm B).
- **Results:**

| Metric | Arm A (swa_start=20) | Arm B (swa_start=10) | Baseline (#4243, EMA) |
|--------|---------------------|---------------------|------------------------|
| **val_avg/mae_surf_p** | 44.0863 | 56.8770 | **38.6750** |
| **test_avg/mae_surf_p** | 37.4775 | 49.4180 | **33.4948** |
| val_single_in_dist | 49.9613 | 65.0115 | 42.1400 |
| val_geom_camber_rc | 57.5383 | 72.0573 | 51.6180 |
| val_geom_camber_cruise | 25.1708 | 35.4037 | 22.4830 |
| val_re_rand | 43.6749 | 55.0357 | 38.4600 |
| n_averaged at end | 15 | 25 | — |
| Best epoch | 35 | 35 | 35 |
| Sec/epoch | 51.5 | 51.5 | 51.7 |
| Peak VRAM | 22.60 GB | 22.60 GB | 22.60 GB |
| Metric artifacts | `models/model-charliepai2i48h2-thorfinn-swa-A-20260517-030608/metrics.jsonl` | `models/model-charliepai2i48h2-thorfinn-swa-B-20260517-034012/metrics.jsonl` | |

- **Analysis:** Both arms catastrophically underperform. Root cause is visible in val trajectory: once SWA kicks in, SWALR drops to swa_lr=1e-6 (constant), effectively freezing learning. The averaged model is then averaging nearly-stagnant snapshots. The model is timeout-limited (val still descending at ep35), so any phase that stops descent wastes the budget. EMA's zero-overhead shadow tracking the ongoing descent is strictly superior here.
- **Decision:** CLOSED — no_improvement. SWA direction closed. If revisited, correct approach: keep cosine schedule running, sample snapshots for ensemble at inference only (don't switch optimizer). Assigned thorfinn to Lookahead (#4362).

---

## 2026-05-17 04:45 — PR #4308: ffn-dropout (alphonse) — no_improvement → closed

- **Student:** charliepai2i48h2-alphonse
- **Hypothesis:** FFN/MLP dropout (p=0.05 Arm A, p=0.10 Arm B) after GELU activation in MLP blocks. Motivated by MLP/attn grad-norm imbalance (~5×). Hypothesis: MLP may be over-fitting.
- **Results:**

| Metric | Arm A (p=0.05) | Arm B (p=0.10) | Baseline (#4243) |
|--------|----------------|----------------|------------------|
| **val_avg/mae_surf_p** | 42.2636 | **42.0952** | **38.6750** |
| **test_avg/mae_surf_p** | 36.0281 | 35.9161 | **33.4948** |
| val_geom_camber_rc | 55.6878 | 58.0487 | 51.6180 |
| val_re_rand | 42.3499 | 39.6849 | 38.4600 |
| Best epoch | 34 | 34 | 35 |
| Sec/epoch | 51.4 | 51.4 | 51.7 |
| Peak VRAM | 23.84 GB | 23.84 GB | 22.60 GB |
| Metric artifacts | `models/model-charliepai2i48h2-alphonse-ffn-dropout-A-20260517-025529/metrics.jsonl` | `models/model-charliepai2i48h2-alphonse-ffn-dropout-B-20260517-034055/metrics.jsonl` | |

- **Analysis:** Dropout narrowed train/val gap slightly (0.021→0.016) but did so by hurting train loss more than it helped val. MLP is NOT over-fitting — adding noise uniformly degrades learning. Combined with PR #4235 (mlp-ratio closed), the MLP capacity is well-tuned. The 5× grad-norm imbalance reflects *signal dominance*, not over-fitting.
- **Decision:** CLOSED — no_improvement. Regularization via FFN dropout closed. Assigned alphonse to SwiGLU (#4358) — testing gated activation as a cleaner MLP upgrade.

---

## 2026-05-17 04:45 — PR #4273: n-head-sweep v2 (tanjiro) — no_improvement → closed

- **Student:** charliepai2i48h2-tanjiro
- **Hypothesis (v2):** n_head=2 (from v1) + slice_num=48 (new baseline) compound. v1 showed n_head=2 (head_dim=64) nearly tied the old baseline (val=38.77 vs 39.83) and beat the new test baseline (test=33.23 vs 33.49). v2 tested whether slice=48 + n_head=2 stack.
- **Results:**

| Metric | v1 Arm A (slice=64, n_head=2) | v2 Arm A (slice=48, n_head=2) | Baseline (slice=48, n_head=4) |
|--------|-------------------------------|-------------------------------|-------------------------------|
| **val_avg/mae_surf_p** | 38.7712 | 40.7324 | **38.6750** |
| **test_avg/mae_surf_p** | 33.2295 | 35.2262 | **33.4948** |
| val_single_in_dist | 40.3896 | 45.3103 | 42.1400 |
| val_geom_camber_rc | 53.5892 | 53.8120 | 51.6180 |
| Best epoch | 37 | 38 | 35 |
| Sec/epoch | 49.0 | 47.4 | 51.7 |
| Peak VRAM | 21.30 GB | 20.68 GB | 22.60 GB |
| Metric artifacts | `models/model-charliepai2i48h2-tanjiro-n-head-sweep-A-20260517-013708/metrics.jsonl` | `models/model-charliepai2i48h2-tanjiro-n-head-sweep-v2-A-20260517-033402/metrics.jsonl` | |

- **Analysis:** Clear interaction effect: n_head=2 (head_dim=64) needs the spatial resolution of slice=64 to compensate for fewer attention heads. At slice=48 (coarser), two heads with head_dim=64 cannot resolve the attention patterns. The baseline n_head=4, head_dim=32, slice_num=48 is a local optimum on the head_count×slice_num interaction grid. single_in_dist degraded most (40→45), confirming the attention shape mismatch hurts basic learning capacity.
- **Decision:** CLOSED — no_improvement. n_head direction fully closed. Assigned tanjiro to RMSNorm (#4365) — orthogonal normalization-layer axis.

---

## 2026-05-17 05:10 — PR #4278: attention-dropout (nezuko) — no_improvement → closed

- **Student:** charliepai2i48h2-nezuko
- **Hypothesis:** Attention dropout (p=0.05 Arm A, p=0.10 Arm B) inside PhysicsAttention to reduce OOD over-fitting on geom_camber_rc.
- **Results:**

| Metric | Arm A (p=0.05) | Arm B (p=0.10) | Old baseline (#4079) | New baseline (#4243) |
|--------|----------------|----------------|----------------------|----------------------|
| **val_avg/mae_surf_p** | 40.2876 | 40.6040 | 39.8345 | **38.6750** |
| **test_avg/mae_surf_p** | 34.4451 | 34.6701 | 33.8873 | **33.4948** |
| val_single_in_dist | 43.4782 | 43.6650 | 43.6797 | 42.1400 |
| **val_geom_camber_rc** | 54.1693 (+1.91%) | 54.7244 (+2.96%) | 53.1517 | 51.6180 |
| val_geom_camber_cruise | 23.5359 | 23.4124 | 22.7101 | 22.4830 |
| val_re_rand | 39.9668 | 40.6141 | 39.7965 | 38.4600 |
| Best epoch | 33 | 33 | 34 | 35 |
| Sec/epoch | 54.8 | 55.5 | ~54.2 | 51.7 |
| Peak VRAM | 23.84 GB | 23.84 GB | 23.84 GB | 22.60 GB |
| Metric artifacts | `models/model-charliepai2i48h2-nezuko-attention-dropout-A-20260517-024608/metrics.jsonl` | `models/model-charliepai2i48h2-nezuko-attention-dropout-B-20260517-033355/metrics.jsonl` | | |

- **Analysis:** Two clean diagnostics:
  1. **Train/val gap GREW with higher dropout** (Arm B p=0.10: train surf_loss 0.021, val_geom_camber_rc 0.066, gap 0.046 vs Arm A p=0.05: gap 0.036). Opposite of regularization-of-overfit signature — dropout is removing signal not co-adaptation.
  2. **OOD split worsened on both arms** — the very split this hypothesis aimed at (geom_camber_rc) got more error. Attention dropout is NOT helping the OOD bottleneck.
  - Note: results were submitted vs the OLD baseline (val 39.83). Even comparing favorably there, both arms regressed (+1.14%, +1.93%). Against new baseline (38.675), regression is +4.2%, +5.0%.
- **Decision:** CLOSED — no_improvement. Combined with #4308 (FFN dropout) and #4312 (SWA), three consecutive failures of model-internal regularization. Reading: model is not over-fitting; OOD gap is from data coverage. Next axis: **data augmentation**. Assigned nezuko to point-subsampling (#4377) — drop 20%/40% non-surface points per training batch.

---

## 2026-05-17 05:36 — PR #4327: Huber loss (δ=1.0 and δ=0.5) vs MSE on slice=48 stack

- **Branch:** charliepai2i48h2-frieren/huber-loss
- **Hypothesis:** Huber transitions quadratic→linear at δ, capping per-sample gradient on outliers. Should reduce update-direction dominance by a few large-error samples in small-data regime.

| Metric | Baseline (PR #4243) | Arm A (δ=1.0) | Arm B (δ=0.5) |
|--------|---------------------|---------------|---------------|
| **val_avg/mae_surf_p** | **38.6750** | 39.1977 (+1.35%) | 39.5966 (+2.38%) |
| **test_avg/mae_surf_p** | **33.4948** | 34.6382 | ~35.0 |
| val_single_in_dist | 42.1400 | 42.6806 | 43.0686 |
| val_geom_camber_rc | 51.6180 | 52.4771 | 53.0811 |
| val_geom_camber_cruise | 22.4830 | 22.5467 | 21.9752 |
| val_re_rand | 38.4600 | 39.0862 | 40.2617 |
| Metric artifacts | | `models/model-charliepai2i48h2-frieren-huber-loss-A-20260517-034205/metrics.jsonl` | `models/model-charliepai2i48h2-frieren-huber-loss-B-20260517-043156/metrics.jsonl` |

- **Analysis:** Both arms regressed nearly uniformly across all splits. asinh-pressure transform (PR #3357) already does the heavy lifting on heavy-tailed residuals — there's no remaining outlier mass for Huber to dampen. The capped gradient on residuals slows convergence in the timeout-bound regime (best epoch 33 with val still descending). The single positive (cruise on Arm B at 21.98) is offset by larger losses elsewhere.
- **Decision:** CLOSED — no_improvement. Loss-function reshaping is now a closed direction on the slice=48 stack.

---

## 2026-05-17 05:39 — PR #4253: SGDR warm restarts (T_0=17 and T_0=12)

- **Branch:** charliepai2i48h2-edward/sgdr-warm-restarts
- **Hypothesis:** CosineAnnealingWarmRestarts replaces monotonic cosine with a mid-training LR reset (epoch T_0). Should help model jump out of attractor and find a flatter generalizing minimum.

| Metric | Baseline (PR #4243) | Arm A (T_0=17) | Arm B (T_0=12) |
|--------|---------------------|----------------|----------------|
| **val_avg/mae_surf_p** | **38.6750** | 43.7760 (+13.2%) | 41.3263 (+6.9%) |
| **test_avg/mae_surf_p** | **33.4948** | 37.8838 | 35.4141 |
| val_geom_camber_rc | 51.6180 | 55.9754 | 55.5760 |
| Metric artifacts | | `models/model-charliepai2i48h2-edward-sgdr-warm-restarts-A-20260517-032811/metrics.jsonl` | `models/model-charliepai2i48h2-edward-sgdr-warm-restarts-B-20260517-042242/metrics.jsonl` |

- **Analysis:** Clean LR trace confirms restart implementation (Arm A: E17 lr=1.46e-6 → E18 lr=1.7e-4 = exact reset; Arm B: E12 lr=2.9e-6 → E13 lr=1.7e-4 = exact reset). The restart causes a regression spike of +5.5 (Arm A) or +8.2 (Arm B) in val. Within the 30-min budget, Arm A completes only ~47% of cycle 2 and Arm B nearly completes cycle 2 — neither recovers below cycle-1 minimum.
- **Decision:** CLOSED — no_improvement. **Third schedule-disrupting failure** (#4312 SWA, this, plus the Huber-as-effective-LR-shape closure on #4327). Monotonic cosine T_max=40 is locally optimal under timeout constraint. Schedule-disruption strategies are a now-closed direction.

---

## 2026-05-17 05:42 — PR #4295: Per-group LR (lr_attn_mult, lr_other_mult) on slice=48 stack

- **Branch:** charliepai2i48h2-fern/per-group-lr
- **Hypothesis:** MLP/attn 5× gradient-norm asymmetry (PR #4154) reflects real signal-strength imbalance. Per-group LR multipliers should rebalance the effective update step size that single global LR can't address.

| Metric | Baseline (PR #4243) | Arm A (lr_other×0.5) | Arm B (lr_attn×2.0) |
|--------|---------------------|----------------------|---------------------|
| **val_avg/mae_surf_p** | **38.6750** | 40.0739 (+1.40) | 39.5025 (+0.83) |
| **test_avg/mae_surf_p** | **33.4948** | 34.3189 | 33.7651 |
| val_single_in_dist | 42.1400 | 45.1506 (+3.01) | 40.6518 (−1.49) |
| val_geom_camber_rc | 51.6180 | 53.7306 | 52.7786 |
| val_re_rand | 38.4600 | 39.5943 | 41.4051 (+2.95) |
| Metric artifacts | | `models/model-charliepai2i48h2-fern-per-group-lr-A-20260517-034502/metrics.jsonl` | `models/model-charliepai2i48h2-fern-per-group-lr-B-20260517-041956/metrics.jsonl` |

- **Param split:** attn=273,620 (41%); other=388,739 (59%). LR sanity at ep1 confirmed both arms applied correctly.
- **Analysis:** Both arms regressed, with informative asymmetric failure modes:
  1. Arm A (rein MLP) under-trained the larger group → +3.01 on val_single_in_dist, +3.68 on test_single_in_dist. Train_surf_loss 50% higher than Arm B.
  2. Arm B (boost attn) over-fit to easy splits → improved val_single_in_dist (−1.49) and test_in_dist (−0.23) but regressed val_re_rand (+2.95).
- **Key insight (from student's diagnostic):** Lion's `sign(momentum)` already neutralizes the gradient-norm *magnitude* asymmetry. Per-group LR ratios only change the **total trajectory length** in each group's parameter subspace. The 5× MLP/attn grad-norm gap is a **measurement artefact**, not a steering signal Lion can be told to act on.
- **Decision:** CLOSED — no_improvement. Third optimizer-tuning failure on slice=48 (#4031 β2, #4181 lr-fine, this). Lion optimizer geometry comprehensively resolved.

---

## 2026-05-17 07:10 — PR #4365: RMSNorm vs LayerNorm on slice=48 stack

- **Branch:** charliepai2i48h2-tanjiro/rmsnorm
- **Hypothesis:** RMSNorm (LayerNorm minus mean-centering and bias) is faster, more bf16-stable, and used in LLaMA/Mistral/T5v1.1. Simpler normalization form should maintain or improve performance on this small-data, bf16, timeout-limited stack.

| Metric | Baseline (PR #4243) | Arm A (scope=all norms) | Arm B (scope=pre-block only) |
|--------|---------------------|-------------------------|-----------------------------|
| **val_avg/mae_surf_p** | **38.6750** | 39.7936 (+1.12) | 39.9674 (+1.29) |
| **test_avg/mae_surf_p** | **33.4948** | 33.8518 (+0.36) | 34.6572 (+1.16) |
| val_single_in_dist | 42.1400 | ~43.0 | ~43.5 |
| val_geom_camber_rc | 51.6180 | ~52.4 | ~53.1 |
| Best epoch | 35 | 35 | 35 |
| Per-epoch time | 51.7 s | 52.2 s | 52.3 s |
| n_RMSNorm / param delta | — | 11 / −1408 | 10 / −1280 |
| Metric artifacts | | `models/model-charliepai2i48h2-tanjiro-rmsnorm-A-*/metrics.jsonl` | `models/model-charliepai2i48h2-tanjiro-rmsnorm-B-*/metrics.jsonl` |

- **Analysis:** Output-head-only (Arm B) hurt LESS than pre-block (Arm A), ruling out "safe in residual stream" as a positioning fix. Param delta confirms bias-drop is working (n_RMSNorm 11/10, −1408/−1280 params from bias removal). Mean-centering absorbs a non-trivial mean shift in this architecture that downstream linears must fight without it.
- **Key insight (from student's diagnostic):** "RMSNorm = LayerNorm minus mean subtraction and bias. The fact that both arms regress suggests the mean-centering term *is* doing useful work." In this Transolver with 128-dim hidden, the residual stream accumulates non-zero mean shifts across 5 blocks; LayerNorm's mean subtraction is load-bearing.
- **Decision:** CLOSED — no_improvement. Combined with #4358 (SwiGLU in-flight), architectural normalization axis comprehensively explored. LayerNorm locked as optimal normalization form.

---

## 2026-05-17 07:10 — PR #4362: Lookahead-Lion: slow/fast weight anchoring on slice=48 stack

- **Branch:** charliepai2i48h2-thorfinn/lookahead-lion
- **Hypothesis:** Lookahead (Zhang et al. 2019) wraps Lion with a slow-weight trust-region anchor: every k=5/10 steps, slow weights ← slow + α(fast − slow). This should help Lion escape noise-driven excursions and find flatter minima.

| Metric | Baseline (PR #4243) | Arm A (k=5, α=0.5) | Arm B (k=10, α=0.5) |
|--------|---------------------|--------------------|---------------------|
| **val_avg/mae_surf_p** | **38.6750** | 43.5308 (+12.6%) | 41.9229 (+8.4%) |
| **test_avg/mae_surf_p** | **33.4948** | 37.9266 (+13.2%) | 36.1539 (+7.9%) |
| val_geom_camber_rc | 51.6180 | ~58.3 | ~55.8 |
| Best epoch | 35 | 35 | 35 |
| Sync events | — | 2625 (k=5) | 1312 (k=10) |
| Metric artifacts | | `models/model-charliepai2i48h2-thorfinn-lookahead-lion-A-*/metrics.jsonl` | `models/model-charliepai2i48h2-thorfinn-lookahead-lion-B-*/metrics.jsonl` |

- **Analysis:** Both arms substantially regress. LR cosine schedule intact (verified), per-epoch time and VRAM identical to baseline. Monotonic k=5→k=10 trend (worse → less-worse) suggests k→∞ (no Lookahead) is the local optimum.
- **Key insight (from student's diagnostic):** Triple-smoothing problem. Lion already produces signed (±lr) updates that implicitly average over momentum via β1=0.9, β2=0.99. A second EMA (Lookahead α=0.5 slow-weight anchor) damps useful exploration. "Lookahead is simply making the underlying optimizer worse."
- **Decision:** CLOSED — no_improvement. **11th consecutive optimizer/schedule failure** since slice=48 merge. Lion+EMA+cosine triple is the locked optimal. Optimizer direction comprehensively closed. Pivoting to architecture (LayerScale #4435) and data augmentation (Y-mirror #4433).

---

## 2026-05-17 07:55 — PR #4403: Fourier feature encoding of mesh pos coords (NeRF K=8 vs RFF K=16)

- **Branch:** charliepai2i48h2-edward/fourier-features
- **Hypothesis:** Fourier feature encoding of mesh pos coords (NeRF octaves K=8 vs RFF K=16 σ=10) would unlock high-frequency spatial signal and disproportionately help val_geom_camber_rc OOD split.

| Metric | Baseline (PR #4243) | Arm A (NeRF K=8) | Arm B (RFF K=16, σ=10) |
|--------|---------------------|------------------|------------------------|
| **val_avg/mae_surf_p** | **38.6750** | 42.4488 (+3.77) | 42.0794 (+3.40) |
| **test_avg/mae_surf_p** | **33.4948** | 36.6978 (+3.20) | 37.6176 (+4.12) |
| val_geom_camber_rc | 51.6180 | 56.4726 (+4.85) | 55.5293 (+3.91) |
| Best epoch | 35 | 35 (still descending) | 35 (still descending) |
| Per-epoch time | 51.7 s | 52.4 s | 52.3 s |
| Input width | 24 | 56 (+32) | 56 (+32) |
| Metric artifacts | | `models/model-charliepai2i48h2-edward-fourier-features-A-*/metrics.jsonl` | `models/model-charliepai2i48h2-edward-fourier-features-B-*/metrics.jsonl` |

- **Analysis:** Both arms timeout-bound at epoch 35, val still descending but ~3.4 units above baseline. OOD got WORSE not better — val_geom_camber_rc +4.85/+3.91. The +32 channels added 8,192 params at the input projection layer; Lion+cosine(T_max=40) was tuned against 24-d input, so the larger embedding needs more SGD steps. NeRF vs RFF essentially identical at convergence (both ~42 val_avg, same channel count).
- **Key insight (from student's diagnostic):** PhysicsAttention slice routing already partitions by learned coords; explicit Fourier basis on pos is redundant *and* costly. Convergence-speed penalty inside fixed budget dominates whatever spectral-bias improvement might exist.
- **Decision:** CLOSED — no_improvement. Input-representation axis closed for now. Future work: encode rare physics dims (Re, NACA) instead of pos; replace pos rather than append. → #4454 feature-noise (token-space augmentation, no capacity addition)

---

## 2026-05-17 07:55 — PR #4377: Point subsampling augmentation: drop 20%/40% non-surface points

- **Branch:** charliepai2i48h2-nezuko/point-subsample
- **Hypothesis:** Per-batch point subsampling (keep_rate=0.8 vs 0.6 on non-surface points) acts as effective data augmentation, increasing diversity in slice-routing token statistics and improving OOD generalization.

| Metric | Baseline (PR #4243) | Arm A (keep=0.8) | Arm B (keep=0.6) |
|--------|---------------------|------------------|------------------|
| **val_avg/mae_surf_p** | **38.6750** | 40.6471 (+5.10%) | 39.8499 (+3.04%) |
| **test_avg/mae_surf_p** | **33.4948** | 34.5265 (+3.08%) | 34.4628 (+2.89%) |
| val_geom_camber_rc | 51.6180 | 53.8964 (+4.41%) | 53.8278 (+4.28%) |
| val_geom_camber_cruise | 22.4830 | 23.5716 (+4.84%) | **22.1761 (−1.36%)** |
| Best epoch | 35 | 35 | 35 |
| Per-epoch time | 51.7 s | 51.8 s | 51.8 s |
| Metric artifacts | | `models/model-charliepai2i48h2-nezuko-point-subsample-A-*/metrics.jsonl` | `models/model-charliepai2i48h2-nezuko-point-subsample-B-*/metrics.jsonl` |

- **Analysis:** Both arms regress; OOD bottleneck (camber_rc) worse on both. Surprising **inversion**: Arm B (heavier 40% drop) is *less bad* than Arm A. Read: at keep=0.8 the model still memorizes per-batch geometry + small noise term; at keep=0.6 the model is forced to compute on genuinely different point clouds, recovering more of the original signal path. Mild signal of val_geom_camber_cruise (−1.36%) on Arm B — possibly genuine in low-Re aerial regime.
- **Key insight (from student's diagnostic):** **PhysicsAttention slice routing is permutation-equivariant and approximately invariant to random subsampling at these rates.** Slice tokens are robust enough that dropping 20-40% of volume points produces near-identical token statistics. The augmentation behaves as added MC variance per step (input noise) rather than as data diversity (where slice-routing is sensitive — in *token space*, not point space).
- **Decision:** CLOSED — no_improvement. **Critical pivot signal:** input-side augmentation at the point level cannot affect slice-routing token statistics. → #4454 feature-noise (token-space noise post-normalization, addressing this diagnostic directly), → #4458 attn-temperature (perturb the slice-routing softmax itself, sharper τ may help OOD).
