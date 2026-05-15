# SENPAI Research Results — `icml-appendix-charlie-pai2i-48h-r4`

## 2026-05-15 14:23 — PR #3094 [MERGED]: Huber (smooth L1) loss to align training with MAE eval metric

- **Student branch:** `charliepai2i48h4-alphonse/huber-loss`
- **Hypothesis:** Switching the per-element training loss from MSE to Huber (smooth L1, β=1.0) better aligns the gradient with the MAE eval metric — bounded gradients at large errors reduce outlier amplification while preserving smooth gradients near zero.

### Results

| Arm | Loss | `val_avg/mae_surf_p` | best epoch | epochs run | `test_avg/mae_surf_p` (3 finite splits) |
|-----|------|----------------------|-----------|-----------|-----------------------------------------|
| A (baseline, MSE) | `(pred - y_norm)**2` | 132.282 | 14 | 14 | 133.117 |
| B (Huber, β=1.0) | `F.smooth_l1_loss(...)` | **111.531** | 11 | 14 | **112.479** |
| **Δ (B − A)** | — | **−15.7%** | — | — | **−15.5%** |

Per-split val MAE pressure (lower is better):

| Split | Baseline (MSE) | Huber | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 172.116 | **141.566** | −17.7% |
| `val_geom_camber_rc`     | 141.056 | **116.797** | −17.2% |
| `val_geom_camber_cruise` |  97.342 |  **86.222** | −11.4% |
| `val_re_rand`            | 118.615 | **101.539** | −14.4% |
| **val_avg**              | **132.282** | **111.531** | **−15.7%** |

### Metric artifacts

- `models/model-charliepai2i48h4-alphonse-huber-loss-hyp-20260515-131213/metrics.jsonl`
- `models/model-charliepai2i48h4-alphonse-huber-loss-hyp-20260515-131213/metrics.yaml`

### Analysis & conclusions

- Huber loss is a **uniform win across all four val splits** — not concentrated on one — and the test set (3 finite splits) confirms the val improvement generalizes. The mechanism is consistent with the prediction: MSE was amplifying large per-sample errors (high-Re outliers in particular).
- The win compounds: every Round 2 experiment now starts from the Huber baseline. Loss alignment is a one-time gain — the next axes (architecture, schedule, conditioning) need to deliver independent improvements.
- **Known issue (pre-existing, not from this PR):** `test_geom_camber_cruise/mae_surf_p` is NaN — single-sample NaN propagation in `data/scoring.py:48` taints the entire split accumulator. Same issue appeared on PR #3113. Not a merge blocker (other splits + val are fine). Flagged for separate fix; `data/scoring.py` is read-only per `program.md`.

### Carry-forward config (current best)

```python
# Loss (in train.py)
sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
# Model: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
# Optimizer: AdamW lr=5e-4 wd=1e-4, batch_size=4, CosineAnnealingLR(T_max=50)
# surf_weight=10
```

---

## 2026-05-15 14:06 — PR #3113 [SENT BACK]: Scale Transolver capacity: n_hidden=192 n_layers=7 slice_num=96

- **Student branch:** `charliepai2i48h4-edward/model-bigger`
- **Hypothesis:** Bumping capacity along three axes (n_hidden 128→192, n_layers 5→7, slice_num 64→96) gives the model more representational space for the multi-domain task. Expected −5% to −15% on val_avg/mae_surf_p.

### Results

| Arm | n_hidden | n_layers | slice_num | params | peak VRAM | epochs run | sec/epoch | `val_avg/mae_surf_p` | best epoch |
|-----|---|---|---|---|---|---|---|---|---|
| A (baseline, MSE) | 128 | 5 | 64 | 0.66M | 42.1 GB | 14 | 131s | 135.95 | 14 |
| B (larger, MSE) | 192 | 7 | 96 | 2.02M | 85.4 GB | **7** | **283s** (2.16× slower) | **177.52** (+30.6% vs A) | 6 |

### Analysis & conclusions

- **Architecture isn't broken; the 30-min budget can't reveal it.** Arm B reached val=177.52 at epoch 6 (cosine still at ~96% peak lr); Arm A reached val=177.49 at the same epoch — identical per-epoch curves until A continued and B was timeout-killed.
- Per-epoch wall-clock cost (2.16× slower) ate into the budget catastrophically. Edward's pre-stated risk fired exactly as anticipated.
- **Same NaN test bug** as #3094 on `test_geom_camber_cruise/mae_surf_p`.

### Decision

**Sent back** for a milder single-axis test: keep `n_hidden=128, n_layers=5`, change **only `slice_num=64 → 96`**, re-baseline on Huber. Transolver's slice attention is the canonical capacity lever and has the lowest wall-clock cost relative to its capacity gain.

### Followups queued post-rebase

- If slice_num=96 works on Huber baseline: next try slice_num=128 or compose with depth +1
- If it doesn't: capacity is not the lever at this budget — try shorter cosine T_max or larger batch with mixed precision instead

---

## 2026-05-15 14:39 — PR #3108 [CLOSED, falsified]: Sweep surf_weight (10/25/50) — direct surface MAE optimization

- **Student branch:** `charliepai2i48h4-askeladd/surf-weight-sweep`
- **Hypothesis:** Raising `surf_weight` from baseline 10 to 25 or 50 redirects optimizer capacity to surface nodes (which are 100–1000× rarer than volume nodes) and reduces `val_avg/mae_surf_p`. Predicted −3% to −10%.
- **Loss form:** MSE (this PR predated the Huber merge in #3094).

### Results — hypothesis falsified, decisively

| Arm | `surf_weight` | `val_avg/mae_surf_p` | `val_avg/mae_vol_p` | `test_avg/mae_surf_p` (3 finite splits) | best epoch |
|-----|---------------|----------------------|---------------------|------------------------------------------|------------|
| **A (baseline)** | **10** | **122.456** | **125.184** | **119.235** | 13 |
| B | 25 | 138.185 (+12.8%) | 158.881 | 143.846 | 13 |
| C | 50 | 136.282 (+11.3%) | 160.104 | 135.970 | 13 |

Per-split val (lower is better):

| Arm | single_in_dist | geom_camber_rc | geom_camber_cruise | re_rand |
|-----|---:|---:|---:|---:|
| A (10) | **143.565** | **131.030** | **97.675** | **117.556** |
| B (25) | 164.834 | 148.216 | 108.052 | 131.636 |
| C (50) | 168.287 | 145.104 | 109.377 | 122.359 |

Regression is **uniform across all 4 val splits and the 3 finite test splits** — no slice hides a sweet-spot at higher weighting.

### Metric artifacts

- `models/model-surf-weight-baseline-20260515-124507/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-surf-weight-25-20260515-132543/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-surf-weight-50-20260515-140025/metrics.jsonl`

### Analysis & conclusions

- **`surf_weight=10` is conclusively the right value.** Higher weights destabilize the joint surface-volume optimization without improving surface MAE; volume MAE also degrades, ruling out a "trade-off" interpretation.
- **Useful noise calibration:** askeladd's Arm A (MSE + surf_weight=10, 14 epochs) = 122.456. PR #3094 Arm A (same config) = 132.282. That's ~7% scatter between two seeds at this epoch count — a useful prior for evaluating future small-margin wins.
- **NaN scoring bug** — askeladd's analysis is the most thorough version we have: `test_geom_camber_cruise` sample index 20 has `inf` in ground-truth `p`, and `inf * 0 = NaN` in IEEE 754 poisons the masked sum even when `surf_mask` is False. The `torch.where(mask, err, zeros)` patch is correct; `data/scoring.py` remains read-only per `program.md` so the fix is outside our scope. Same NaN was flagged on #3094 and #3113.

### Decision

**Closed** — clean falsification, no sub-region of the (10, 25, 50] axis is worth re-exploring. Askeladd reassigned to bf16 AMP mixed precision (PR #3290 below).

---

## 2026-05-15 14:50 — Round 2 assignments

After Round 1 we have 1 merge (#3094 Huber), 2 closed misses (#3108 surf_weight, #3131 OneCycle), 1 sent-back (#3113 slice_num revision), and 6 still WIP. Three students have received Round 2 hypotheses targeting orthogonal axes:

| Student | PR | Round 2 hypothesis | Axis | Mechanism |
|---------|----|--------------------|------|-----------|
| alphonse | #3278 | Per-channel loss weighting (Ux, Uy, p) | Loss × channel | Up-weight pressure channel in Huber loss; arms p=2× and p=4× |
| thorfinn | #3289 | Cosine `T_max=15` to match achievable budget | LR schedule | Schedule was sized for 50 epochs but only ~14 achievable; full cosine decay needed |
| askeladd | #3290 | bf16 AMP mixed precision | Throughput | Wrap forward+loss in autocast(bf16) — predict 1.5–2× speedup → ~21–28 epochs in 30 min |

All three target the Huber baseline (`val_avg/mae_surf_p = 111.531`). All three are paired-arm comparisons. All three are composable with each other and with the 5 still-WIP Round-1 PRs (#3117 Fourier, #3122 FiLM, #3126 EMA, #3128 scale-aware loss, #3113 revised slice_num). If multiple Round-2 arms win we'll compose them in Round 3.

### Operational thesis

The dominant Round-1 lesson: **every run is wall-clock-truncated at ~14 epochs, not epoch-truncated**. This makes throughput and schedule-fit hypotheses (thorfinn cosine_t_max, askeladd bf16) particularly high-leverage — each one expands the effective compute budget for every subsequent experiment.

---
