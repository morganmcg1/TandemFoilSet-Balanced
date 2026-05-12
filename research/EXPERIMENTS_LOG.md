# SENPAI Research Results — icml-appendix-charlie-pai2g-24h-r5

## 2026-05-12 18:55 — PR #1459: Raise surf_weight 10→20 (CLOSED — regression)

- Student branch: `charliepai2g24h5-alphonse/surf-weight-20`
- Hypothesis: Doubling `surf_weight` (10 → 20) up-weights the surface-only metric in the loss; expected 3–8% relative improvement on `val_avg/mae_surf_p`.
- Trained 14 epochs (hit 30-min wall-clock cap); best checkpoint at epoch 12.

### Results (vs. effective baseline from #1463 with the same 14-epoch budget)

| Run | val_avg/mae_surf_p | val_geom_camber_cruise | test_avg/mae_surf_p |
|---|---:|---:|---:|
| #1459 surf_weight=20 (this PR) | **135.7367** | 101.3540 | NaN (cruise-test pressure overflow) |
| #1463 baseline (SWA never engaged) | **125.20** | — | NaN (cruise-test pressure overflow, same) |

- Metrics: `models/model-surf_weight_20-20260512-180422/metrics.jsonl`
- Summary: `models/model-surf_weight_20-20260512-180422/metrics.yaml`

### Analysis

surf_weight=20 underperforms baseline (surf_weight=10) by ~8.4% on the primary metric within our 30-min training budget — a clear regression past the 5% close threshold. The hypothesis may still be correct given more epochs (the surface-up-weighted loss landscape needs more updates to reach its new minimum), but our cap doesn't give us those epochs.

### Side-effect: test-time pressure overflow

Both runs (this PR and the baseline-equivalent #1463 measurement) produce NaN on `test_geom_camber_cruise/mae_surf_p` because the model occasionally outputs Inf/NaN pressure predictions on individual cruise test samples, which propagate through the MAE accumulator since `data/scoring.py` only skips samples with non-finite GT (not non-finite predictions). The fix is train.py-side (`nan_to_num` clamp + seed pin) since `data/scoring.py` is read-only. PR #1463 (askeladd) is the next experiment that will adopt this fix.

### Conclusion

Closed. Alphonse reassigned to H10 (warmup + cosine matched to budget). The 8.4% surf_weight regression and the implicit ~125.20 baseline measurement are both useful information for round 5 planning.

---

## 2026-05-12 18:58 — PR #1463: SWA from epoch 25 (SENT BACK — SWA never engaged)

- Student branch: `charliepai2g24h5-askeladd/swa-start25`
- Hypothesis: SWA averaging from epoch 25 onward improves OOD generalisation by 2–6%.

### What we learned

SWA_START_EPOCH=25 is **unreachable in our 30-min budget** — training stops at epoch 14. The student's diagnosis is correct: the SWA-paper recipe assumes the model is in the cosine LR valley before averaging starts. With T_max=50 cosine and only 14 epochs available, LR at epoch 14 is still ~82% of peak — not a valley.

**Effective baseline measurement (SWA never engaged → equivalent to baseline surf_weight=10):**

| Metric | Value | Epoch |
|---|---:|---:|
| val_avg/mae_surf_p (best) | **125.20** | 14 |
| test_avg/mae_surf_p | NaN | — |
| test_geom_camber_cruise/mae_surf_p | NaN (Inf overflow) | — |

This is now our informal round-5 baseline floor. It is not a merged baseline because (a) the test number is NaN and (b) the PR itself was about SWA, not baseline measurement.

### Advisor action

Sent back to student with:
1. Approved option (b): `SWA_START_EPOCH=8`, `--epochs 14` (cosine T_max matched to budget gives SWA a real LR valley to average over).
2. Pin a seed (torch.manual_seed(42)) for reproducibility.
3. Add `torch.nan_to_num` guard on `pred_orig` in `evaluate_split` (train.py only — data/ is read-only) so the cruise-test pressure overflow no longer NaNs the entire split.
4. Report best val_avg/mae_surf_p in BOTH the pre-SWA and post-SWA regimes so we can attribute the SWA contribution cleanly.

Status: WIP, awaiting rerun.
