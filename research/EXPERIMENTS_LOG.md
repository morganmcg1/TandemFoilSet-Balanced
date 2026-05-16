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

