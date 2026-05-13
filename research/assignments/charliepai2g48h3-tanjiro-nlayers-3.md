# Assignment: n_layers=3 + T_max=22 Depth Sweep (TandemFoilSet / Transolver)

**Student:** tanjiro  
**Branch:** charliepai2g48h3-tanjiro-nlayers-3  
**Date:** 2026-05-13  

---

## Hypothesis

We are continuing a systematic depth sweep driven by the **epoch-count mechanism**: shallower models are faster per epoch, so within a fixed 30-minute wall-clock budget they complete more epochs and achieve better cosine schedule alignment.

The last three wins confirm the pattern:

| Change | Per-epoch time | Epochs gained | val_avg/mae_surf_p delta |
|--------|---------------|---------------|--------------------------|
| n_layers=6 → 5 | faster | +2 epochs | −6.98% |
| n_layers=5 → 4 | faster | +3 epochs | −1.07% |
| **n_layers=4 → 3 (this PR)** | ~75s/epoch (est.) | +5-7 epochs | ? |

At n_layers=4 the best checkpoint was epoch 17/17 — the model was **still descending** at the budget wall. Reducing to n_layers=3 should yield ~22-24 reachable epochs, allowing the cosine schedule to run closer to completion and extract further improvement.

**Predictions:**
- **If this PR wins:** test n_layers=2, continuing the sweep to find the throughput-optimal depth.
- **If this PR loses:** the depth floor lies between 3 and 4 layers — capacity, not epoch count, becomes the bottleneck at n_layers=3, and we should stop the depth sweep and focus on other levers.

---

## Current Baseline (after PR #2080 merge)

**Config:** n_layers=4, slice_num=48, n_head=4, mlp_ratio=4, GeGLU, RMSNorm, Lion lr=1e-4 WD=1e-4, surf_weight=10, bf16, batch=4

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | **46.344** |
| test_avg/mae_surf_p | **39.950** |

**Per-split baseline:**

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 49.979 | 44.746 |
| geom_camber_rc | 61.558 | 54.155 |
| geom_camber_cruise | 27.318 | 22.876 |
| re_rand | 46.518 | 38.025 |

**Timing at n_layers=4:** ~94s/epoch, 17 epochs completed in 30-min budget (26.7 min total), best_epoch=17 (still improving at final epoch).

---

## Your Task

Test **n_layers=3** with **--epochs 22** (T_max=22 via CosineAnnealingLR which matches MAX_EPOCHS).

Expected timing: ~75s/epoch → 22-24 epochs in 30-min budget.

**Budget guardrail:** If epoch 1 wall-clock exceeds 80s, cap at **--epochs 21** instead of 22. Report the actual epoch-1 timing so we can calibrate future depth steps.

All other hyperparameters remain identical to the current baseline. Note that `n_layers` defaults to 5 in `train.py` — you **must** pass `--n_layers 3` explicitly.

---

## Run Command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name nlayers-3-tmax22 \
  --epochs 22 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3
```

If epoch 1 > 80s, rerun with `--epochs 21`.

---

## Reporting Requirements

Please commit all metrics JSONL outputs to this branch and post a comment with the following information:

1. **Per-split val and test mae_surf_p** for all 4 splits vs the baseline table above (average across splits must beat 46.344 val / 39.950 test to be a winner).
2. **Per-split mae_vol_p** (val and test) for all 4 splits.
3. **Per-epoch wall-clock** for epoch 1 and the last completed epoch.
4. **Total wall-clock** and **epochs actually completed**.
5. **Best epoch** number and **parameter count** (expect ~560K vs ~670K at n_layers=4).
6. **Peak GPU memory** (MB).

---

## Result Marker

When complete, post a terminal result comment containing the single-line marker below (fill in actual values):

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```

Ensure `terminal=true` and `pending_arms=false` only when you have final results across all splits. Do not post the marker for intermediate or partial results.
