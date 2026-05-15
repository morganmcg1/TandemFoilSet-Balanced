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
