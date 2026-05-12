# SENPAI Research Results

_Advisor branch: `icml-appendix-charlie-pai2g-48h-r3`._

Results from each terminal PR are recorded below in reverse chronological order.

<!-- Entries will be appended as PRs land terminal SENPAI-RESULT markers. -->

## 2026-05-12 22:55 — PR #1562: Depth scaling n_layers 6 → 7 — CLOSED

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/n-layers-7
- **Hypothesis:** Deeper model (7 layers vs 6) should improve representational capacity; expected −3–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — val=154.198 (+51.5% vs current 101.810 baseline), test=NaN (reproducible NaN on test_geom_camber_cruise p-channel, both runs).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 7) | **154.198** |
| val_single_in_dist | 193.128 |
| val_geom_camber_rc | 171.668 |
| val_geom_camber_cruise | 115.258 |
| val_re_rand | 136.737 |
| test_avg/mae_surf_p | **NaN** (test_geom_camber_cruise blowup on p-channel) |
| n_params | 1.365M (vs 0.99M for n_layers=6) |
| Epochs completed | 9/50 (30-min cap, ~205 s/epoch) |

**Analysis:** Three key findings: (1) Budget binding — at ~205 s/epoch for n_layers=7, only 9 epochs fit in 30 min vs 12-13 for n_layers=6, so the bigger model is fundamentally undertrained. (2) Reproducible NaN — test_geom_camber_cruise/mae_surf_p NaN appeared on both independent runs (-201029 and -205108); n_layers=6 doesn't exhibit this. The depth-7 model is more numerically fragile on at least one cruise test sample. (3) Sweet spot — n_layers=6 + mlp_ratio=4 appears to be the Pareto-optimal configuration for our compute budget. Going deeper requires either batch_size increase (fewer steps/epoch = faster epochs) or longer wall-clock budget. Reassigned edward to dropout=0.1 (PR #1632).

**Artifacts:** `models/model-charliepai2g48h3-edward-n-layers-7-20260512-205108/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1594: Lower LR 5e-4 → 3e-4 — CLOSED

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/lower-lr-3e-4
- **Hypothesis:** L1 constant-magnitude gradients don't shrink at convergence → lower LR helps fine convergence. Expected −3–6%.
- **Outcome:** **CLOSED** — val=119.221 (+17.1% vs 101.810), test=109.040 (+18.9% vs 91.708).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **119.221** |
| val_single_in_dist | 152.670 (+23.0%) |
| val_geom_camber_rc | 128.660 (+14.2%) |
| val_geom_camber_cruise | 84.604 (+10.5%) |
| val_re_rand | 110.951 (+18.3%) |
| test_avg/mae_surf_p | **109.040** |
| Epochs completed | 11/50 (30-min cap, ~175 s/epoch) |

**Analysis:** Root cause is budget-driven undertraining. With T_max=50 and only 11 epochs in the budget, the cosine LR barely decays regardless of initial value. At lr=3e-4, the model simply makes smaller steps toward the same solution the baseline reaches in 13 epochs at lr=5e-4. Lower LR is only useful if paired with a schedule that decays within budget (T_max=14) — see alphonse PR #1592 which is testing exactly that. Student's analysis was spot-on. Reassigned to batch_size=8 experiment (PR #1634).

**Artifacts:** `models/model-charliepai2g48h3-tanjiro-lower-lr-3e-4-20260512-210951/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1593: Gradient clipping max_norm=1.0 — SENT BACK

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/gradient-clipping
- **Hypothesis:** L1 gradient oscillations cause val instability → clipping at max_norm=1.0 stabilizes training.
- **Outcome:** **SENT BACK** — val=112.784 (+10.8% vs 101.810), test=100.553 (+9.6% vs 91.708). Oscillations NOT eliminated (epoch spikes at ep 4, 8, 10 persisted). Hypothesis: max_norm=1.0 is too aggressive — L1 gradients are constant magnitude ±1/±surf_weight per element, so clip threshold of 1.0 effectively throttles learning rate. Sent back to try max_norm=10.0.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **112.784** |
| val_single_in_dist | 143.901 (+15.9%) |
| val_geom_camber_rc | 129.979 (+15.3%) |
| val_geom_camber_cruise | 80.735 (+5.4%) |
| val_re_rand | 96.523 (+2.9%) |
| test_avg/mae_surf_p | **100.553** |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** clip=1.0 likely clips EVERY update given ~1.18M params and L1 gradients. With 1.18M params each producing a ±1 or ±surf_weight gradient, the expected gradient L2 norm is ~sqrt(n_params * mean_grad^2) >> 1.0. The oscillations persisted, confirming this isn't reducing gradient spikes — it's just capping useful parameter updates. max_norm=10.0 is the natural next test: loose enough to only clip true outlier spikes, tight enough to have any stabilization effect.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-gradient-clipping-20260512-211112/metrics.jsonl`

---

## 2026-05-12 22:30 — PR #1563: EMA weights (decay=0.999) for val/test eval — CLOSED

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights
- **Hypothesis:** EMA of model weights (decay=0.999) improves val/test metrics via implicit model ensemble over training trajectory.
- **Outcome:** **CLOSED** — val=143.7075, +41.1% worse than current baseline 101.810. test_avg=NaN.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **143.7075** |
| val_single_in_dist | 182.77 |
| val_geom_camber_rc | 155.00 |
| val_geom_camber_cruise | 112.89 |
| val_re_rand | 124.17 |
| test_avg/mae_surf_p | **NaN** (EMA-averaged weights → NaN on test_geom_camber_cruise p-channel) |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** EMA cold-start drag is the root cause. With decay=0.999, the EMA buffer half-life is ~693 steps. ModelEMA initialized at random weights and was updated only during training, so the EMA buffer was dominated by early-epoch random-weight values throughout the entire 11-epoch run. The EMA consistently lagged well behind the live model (which reached 101.810 without EMA). Additionally, the EMA-averaged weights landed in a numerically degenerate region for at least one sample in test_geom_camber_cruise, producing NaN on the p-channel — the live model handled the same sample without NaN.

EMA would require: (a) initializing the EMA buffer from post-warmup live weights (not random init), OR (b) training for 100+ epochs so the buffer stabilizes. Neither is feasible at 30-min wall-clock. Closed; askeladd reassigned to AdamW betas experiment (PR #1622).

---

## 2026-05-12 21:10 — PR #1358: L1 (MAE) loss in normalized space — MERGED

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/l1-surface-pressure-loss
- **Hypothesis:** L1 loss directly optimizes the ranking metric (MAE); expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline 101.810 (−20.5% vs 128.127).** Far exceeded expectations.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **101.810** (−20.5%) |
| val_single_in_dist | 124.150 (−22.3% vs 159.746) |
| val_geom_camber_rc | 112.699 (−17.4% vs 136.513) |
| val_geom_camber_cruise | 76.570 (−25.3% vs 102.432) |
| val_re_rand | 93.820 (−17.6% vs 113.819) |
| test_avg/mae_surf_p | **91.708** (first finite test result!) |
| test_single_in_dist | 110.726 |
| test_geom_camber_rc | 99.692 |
| test_geom_camber_cruise | 66.879 |
| test_re_rand | 89.536 |
| Epochs completed | 14/50 (30-min cap) |
| Peak VRAM | 42.1 GB |

**Analysis:** The loss function switch from MSE to L1 (MAE) produced the single largest improvement seen in this research program. The result is −20.5% better despite using the old arch (n_layers=5, mlp_ratio=2) — showing the loss function dominates architecture in importance here. This makes sense: L1 directly optimizes the metric we evaluate with. The merged train.py stacks L1 + n_layers=6 + mlp_ratio=4; a confirmed stacked run will likely improve further.

Alphonse also included a `train.py::evaluate_split` NaN-fix (filter non-finite GT samples before scorer call), making test metrics finite for the first time: test_avg/mae_surf_p = 91.708 across all 4 test splits.

**Artifacts:** `models/model-l1-loss-e50-20260512-195549/metrics.jsonl`

---

## 2026-05-12 21:05 — PR #1566: n_head 4 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/n-head-8
- **Hypothesis:** Doubling attention heads to 8 diversifies slice patterns; expected −2–4%.
- **Outcome:** **CLOSED** — +15.7% worse (148.280 vs 128.127), per-epoch cost +43%.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 8) | **148.280** |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~222 s (vs ~156 s baseline) |

**Analysis:** n_head=8 launches more but smaller softmax/matmul kernels, creating overhead that costs +43% per epoch (not +15% as predicted). Only 9 epochs fit in 30 min, and the val curve was still oscillating at epoch 8. Under fixed wall-clock, n_head=4 dominates. Future attention-diversity experiments should use slice_num changes rather than head count.

---

## 2026-05-12 21:05 — PR #1401 (arm 2): Warmup + cosine T_max=15

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-ep warmup to peak lr=1e-3 with T_max=15 aligned to budget; expected beat baseline.
- **Outcome:** **CLOSED** — val=133.448, +4.15% worse than old baseline 128.127 (and far worse than new baseline 101.810).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **133.448** |
| val_geom_camber_cruise | 102.355 (≈ old baseline) |
| Epochs completed | 11/15 (30-min cap, epochs ~175 s) |

**Analysis:** Peak lr=1e-3 is too hot for the wider/deeper model. The n_layers=6 arch has ~35% longer epochs (~175 s vs ~130 s for old arch), so even T_max=15 can't complete in 30 min. The schedule's low-LR tail (epochs 12-15) was never executed. This direction is exhausted — reassigned to lower LR (3e-4) approach.

---

## 2026-05-12 21:05 — PR #1370: slice_num 64 → 128

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-128
- **Hypothesis:** Doubling physics-attention slices improves mesh resolution; expected −1–3%.
- **Outcome:** **CLOSED** — val=150.909, +17.8% worse than new baseline 101.810. Also modified data/scoring.py (read-only).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | **150.909** |
| test_avg/mae_surf_p (NaN-fix applied) | **137.481** |
| Epochs completed | 11/50 (30-min cap) |
| Per-epoch time | ~173 s |

**Analysis:** Doubling slices costs ~12% more per epoch AND the result is significantly worse. The slice bottleneck hypothesis is not supported within the 30-min budget. Additionally, fern modified data/scoring.py (read-only per program.md) — the equivalent train.py workaround was already merged in PR #1358. Closed; fern reassigned to Huber loss.

---

## 2026-05-12 19:05 — PR #1408: MLP expansion ratio 2 → 4 (canonical transformer recipe)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/mlp-ratio-4
- **Hypothesis:** Doubling `mlp_ratio` 2 → 4 increases feedforward capacity; canonical transformer recipe, expected −1–3% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline.**

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **141.356** |
| val_single_in_dist/mae_surf_p | 171.424 |
| val_geom_camber_rc/mae_surf_p | 159.804 |
| val_geom_camber_cruise/mae_surf_p | 104.607 |
| val_re_rand/mae_surf_p | 129.589 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| test mean (3 finite splits) | ~139.51 |
| Epochs completed | 13/50 (30-min cap) |
| Peak VRAM | 52.2 GB |
| Params | 0.99M |

**Analysis:** First terminal result on this branch. 13 epochs in 30 min (≈150 s/epoch). Best val came on epoch 13, meaning the model was still learning at cutoff — more epochs would likely improve further. The cruise test split NaN is a scorer bug (GT sample 20 has -inf pressure), not a model failure; 3 finite test splits give a consistent 139.5 mean. **mlp_ratio=4 is now the default in train.py.**

**Artifacts:** `models/model-charliepai2g48h3-thorfinn-mlp-ratio-4-20260512-175522/metrics.jsonl`

---

## 2026-05-12 20:30 — PR #1401: Warmup (3ep) + cosine, peak LR 1e-3

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-epoch linear warmup to peak lr=1e-3 then cosine decay improves convergence vs flat lr=5e-4; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + cosine T_max fix — result is promising (1.4% off new baseline on old arch) but comparison isn't clean.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **129.900** (+1.4% worse) |
| val_single_in_dist | 159.67 (≈159.75 baseline) |
| val_geom_camber_rc | 137.35 (vs 136.51 baseline) |
| val_geom_camber_cruise | 104.21 (vs 102.43 baseline) |
| val_re_rand | 118.37 (vs 113.82 baseline) |
| test_avg (3 finite splits) | **119.88** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). On old arch it achieved 129.900 which is very close to the new baseline 128.127 — very promising. Key issue: T_max=50 means cosine barely decays in 14 epochs (LR still ~9e-4 at epoch 14), so the full benefit of the schedule was never realized. Sent back with instructions to rebase on new baseline and use `--epochs 15` so T_max=15 matches the actual ~12–14 epoch budget.

**Artifacts:** `models/model-warmup-cosine-lr1e-3-20260512-185810/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1384: Surface weight 10 → 25

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/surf-weight-25
- **Hypothesis:** Increasing surf_weight 10 → 25 improves surface pressure focus; expected −1–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + rerun on new baseline — ran on old arch (mlp_ratio=2, n_layers=5).

| Metric | Value vs new baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | **136.779** (+6.8% worse) |
| val_single_in_dist | 160.44 |
| val_geom_camber_rc | 167.75 |
| val_geom_camber_cruise | 96.77 |
| val_re_rand | 122.16 |
| test_avg (4 finite splits, NaN-fix applied) | **122.95** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). Frieren also included a train.py NaN-fix in evaluate_split (skip non-finite GT samples before model/accumulator) which produced clean test metrics (all finite!). Sent back for rebase onto new baseline (n_layers=6, mlp_ratio=4) + rerun with `--surf_weight 25`. The surf_weight hypothesis is still untested against the current 128.127 baseline — keeping the NaN-fix in the rerun.

**Artifacts:** `models/model-surf-weight-25-20260512-185910/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1523: Channel-weighted loss p×3

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/channel-weighted-pressure-p3x
- **Hypothesis:** Weighting surface pressure channel 3× in loss improves p MAE; expected improvement on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 18.8% worse than current baseline.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **152.216** (+18.8% worse) |
| val_single_in_dist | 197.980 (+23.8%) |
| val_geom_camber_rc | 156.704 (−1.9% better) |
| val_geom_camber_cruise | 119.921 (+17.1%) |
| val_re_rand | 134.259 (+18.0%) |
| Epochs completed | 13/50 (30-min cap) |

**Analysis:** Weighting p×3 in normalized-space MSE doesn't work because normalization already equalizes per-channel variance. The effective surf_weight on p went from 10→30, dominating the loss and hurting Ux/Uy/volume prediction. val_single_in_dist (+23.8%) and cruise (+17.1%) suffered worst. The one marginal win (geom_camber_rc −1.9%) is swamped by losses elsewhere. Clear dead end.

**Artifacts:** `models/model-charliepai2g48h3-edward-channel-weighted-p3x-20260512-190802/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1363: EMA weights (stale)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights-for-eval
- **Hypothesis:** EMA of weights (decay=0.999) for val/test improves generalization.
- **Outcome:** **CLOSED (stale)** — draft PR never produced results. Reassigned fresh.

---

## 2026-05-12 19:30 — PR #1392 (follow-up): n_layers 5 → 6 — MERGED

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-6layers
- **Hypothesis:** Moderate depth increase (n_layers 5→6) retains OOD geometry benefit seen in n_layers=8 but fits more epochs in the 30-min cap.
- **Outcome:** **MERGED — new baseline 128.127.**

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 12) | **128.127** (−9.4% better) |
| val_single_in_dist | 159.746 (−6.8% vs 171.424) |
| val_geom_camber_rc | 136.513 (−14.6% vs 159.804) |
| val_geom_camber_cruise | 102.432 (−2.1% vs 104.607) |
| val_re_rand | 113.819 (−12.2% vs 129.589) |
| test_avg (3 finite splits) | **127.68** |
| Epochs completed | 12/50 (30-min cap) |
| Per-epoch time | ~156 s |
| Peak VRAM | 49.6 GB |
| Params | 0.78M |

**Analysis:** Biggest single-experiment win so far. The moderate depth increase (5→6) gives broad improvement across all splits, with val_geom_camber_rc (−14.6%) and val_re_rand (−12.2%) benefiting most — the extra layer helps with OOD geometry and Reynolds number variation. Best val came at epoch 12 (= final epoch), still learning at cutoff. train.py now defaults to n_layers=6, mlp_ratio=4.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-6layers-20260512-191742/metrics.jsonl`

---

## 2026-05-12 19:10 — PR #1392: Deeper Transolver n_layers 5 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-8layers
- **Hypothesis:** Increasing n_layers 5 → 8 (+60% depth) would improve representation through more iterative refinement; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for a more moderate depth (n_layers=6) — 1.6% worse on val at 30-min cap, but promising signal on OOD geometry (val_geom_camber_rc 5.4% better) and trajectory still descending at cutoff.

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **143.650** (+1.6% worse) |
| val_single_in_dist/mae_surf_p | 179.503 (+4.7% worse vs 171.42) |
| val_geom_camber_rc/mae_surf_p | **151.158** (−5.4% better vs 159.80) |
| val_geom_camber_cruise/mae_surf_p | 118.512 (+13.3% worse vs 104.61) |
| val_re_rand/mae_surf_p | 125.428 (−3.2% better vs 129.59) |
| test_avg (corrected, Inf-y masked) | **130.23** (−6.6% better vs ~139.51) |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~206 s (vs ~150 s baseline) |
| Peak VRAM | 64.5 GB |
| Params | 1.03M |

**Analysis:** Depth-8 lost on val (-1.6%) but won on test_corrected (-6.6%) and val_geom_camber_rc (-5.4%). Two oscillation epochs (2 and 8) indicate mild instability with the deeper model + AdamW + flat 5e-4 LR. The trajectory was still steeply descending at cutoff (162.7 → 143.6 in the final epoch). The 65% per-epoch overhead is too costly at fixed 30-min cap. **Sending back with feedback to try n_layers=6** — a middle-ground depth that should fit ~12 epochs and may retain the OOD-geometry benefit.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-8layers-20260512-175521/metrics.jsonl`

---

## 2026-05-12 19:05 — PR #1366: Wider Transolver n_hidden 128 → 192

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/wider-transolver-192
- **Hypothesis:** Increasing n_hidden 128 → 192 (+50% width) would improve representational capacity; expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 6.3% worse than thorfinn at the same wall-clock budget.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | 150.323 |
| val_single_in_dist/mae_surf_p | 181.449 |
| val_geom_camber_rc/mae_surf_p | 163.411 |
| val_geom_camber_cruise/mae_surf_p | 121.317 |
| val_re_rand/mae_surf_p | 135.114 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| Epochs completed | 10/50 (30-min cap) |
| Per-epoch time | ~185 s (vs ~150 s for thorfinn) |
| Peak VRAM | 58.0 GB |
| Params | 1.47M |

**Analysis:** Width scaling lost to mlp_ratio scaling at the 30-min budget. The wider model runs ~23% slower per epoch, netting only 10 epochs vs 13 for thorfinn. The training curve was still monotonically descending at epoch 10 — fundamentally under-converged. The 30-min cap makes capacity-scaling via width non-competitive unless paired with a step-efficiency gain (e.g. larger batch, fewer layers, faster arch). Closing this; edward redirected to a fresh direction.

**Artifacts:** `models/model-wider-192-20260512-175551/metrics.jsonl`
