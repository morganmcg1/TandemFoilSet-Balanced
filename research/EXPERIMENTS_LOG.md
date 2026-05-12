# SENPAI Research Results — `icml-appendix-charlie-pai2g-48h-r2`

---

## 2026-05-12 18:55 — PR #1418: Per-channel loss weight: upweight pressure 3×

- **Branch:** `charliepai2g48h2-askeladd/pressure-channel-weight`
- **Hypothesis:** Add per-channel loss weights [Ux=1, Uy=1, p=3] (normalized by sum=5) to squared error in training and eval. Upweighting pressure targets the primary `val_avg/mae_surf_p` ranking metric.
- **Status:** MERGED ✅ — new round-1 baseline

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | **122.6395** |
| val_single_in_dist/mae_surf_p | 145.914 |
| val_geom_camber_rc/mae_surf_p | 137.895 |
| val_geom_camber_cruise/mae_surf_p | 94.868 |
| val_re_rand/mae_surf_p | 111.882 |
| test_single_in_dist/mae_surf_p | 126.460 |
| test_geom_camber_rc/mae_surf_p | 127.348 |
| test_geom_camber_cruise/mae_surf_p | NaN ⚠️ (scoring bug) |
| test_re_rand/mae_surf_p | 111.169 |
| test_avg/mae_surf_p (3-split partial) | 121.659 |
| Peak GPU | 42.1 GB |
| Epoch time | ~131s/epoch |
| Epochs completed | 14/20 (30-min timeout) |

**Metrics artifacts:**
- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.jsonl`
- `models/model-charliepai2g48h2-askeladd-pressure-channel-weight-20260512-175622/metrics.yaml`

### Commentary

Channel weighting is a clean win for this metric: the pressure channel receives 60% of the gradient (3 out of 5 total weight) instead of 33%, and the model's best val_avg drops from the initial few-epoch range directly to 122.64 by epoch 14 with a monotonically decreasing curve. The mechanism is straightforward — the optimizer is explicitly told the evaluation metric's channel preference.

The val curve was still descending at timeout, suggesting more epochs would have improved further. Follow-ups should consider pressure-only weighting ([0,0,1]) and channel weight sweeps.

NaN on test_geom_camber_cruise is a GT data quality issue in the test set (sample 000020 contains 761 +inf entries in y), not a model or implementation issue. See BASELINE.md for details.

---

## 2026-05-12 18:53 — PR #1424: Warmup + cosine, peak LR 1e-3

- **Branch:** `charliepai2g48h2-fern/warmup-cosine-1e-3`
- **Hypothesis:** Peak LR 1e-3 with 1-epoch linear warmup and cosine decay. Hypothesis: faster early convergence under the 30-min cap.
- **Status:** SENT BACK for refinement → `warmup-7e-4-clip`

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | 127.170 |
| val_geom_camber_cruise/mae_surf_p | 95.15 |
| val_geom_camber_rc/mae_surf_p | 134.36 |
| val_re_rand/mae_surf_p | 109.46 |
| val_single_in_dist/mae_surf_p | 169.72 |
| test_avg/mae_surf_p (3-split partial) | 126.36 |
| Epochs completed | 14/20 |

### Commentary

3.7% worse than #1418 at the same epoch count. The 1e-3 peak LR caused instability (epoch-5 spike to 267 from 202, slow re-stabilization from epoch 11). The model was still descending at timeout — with a stable schedule it could have caught up, but the instability indicates the LR overshoots the loss landscape at this scale (0.66M params, 96GB-VRAM single GPU). Notably, cruise test overflowed to NaN for a different reason than #1418 (model output overflow, not GT NaN), confirming the instability was real.

Direction is still worth pursuing: faster convergence under the 30-min cap is genuinely valuable. Refinement: peak LR 7e-4 (20% increase over baseline), 2-epoch warmup, gradient clipping at 1.0.

---
