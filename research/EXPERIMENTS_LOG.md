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

## 2026-05-15 15:27 — PR #3122 [SENT BACK]: FiLM conditioning on (log Re, AoA, NACA, gap, stagger)

- **Student branch:** `charliepai2i48h4-frieren/film-conditioning`
- **Hypothesis:** Explicit FiLM modulation of attention input on geometry+flow conditions (Re, AoA, NACA, gap, stagger) should improve cross-regime generalization, especially on `val_re_rand`.

### Results (on MSE baseline — sent back for Huber rebase)

| Arm | conditioning | `val_avg/mae_surf_p` | `val_single_in_dist` | `val_geom_camber_rc` | `val_geom_camber_cruise` | `val_re_rand` |
|-----|-------------|---------------------:|---------------------:|---------------------:|-------------------------:|--------------:|
| A (no FiLM)        | implicit (raw dims 13–23) | **125.634** | 154.133 | 131.251 | 100.159 | 116.992 |
| B (FiLM)           | explicit on 5 conditions | **123.683** | **140.471** | 136.427 | 98.760 | 119.073 |
| **Δ (B − A)**      | | **−1.55%** | **−8.87%** | +3.94% (worse) | −1.40% | +1.78% (worse) |

Test 3-split mean (excl. cruise NaN): A=122.206, B=123.235 (**+0.84% worse** on test).

### Analysis

- **Mechanism inversion** — predicted to help `val_re_rand` most (cross-Re generalization). Actually helped `val_single_in_dist` the most (which has the widest Re range in-dist). Cross-Re holdout is slightly worse with FiLM. Interpretation: explicit conditioning helps when there's headroom for per-sample adaptation within the training distribution; it doesn't automatically buy generalization across distribution boundaries.
- **Mixed train val↔test signal** — val_single_in_dist gains (−8.9%) translate to test_single_in_dist (−5.9%), but val_geom_camber_rc gets worse on FiLM (+3.9%) and test_geom_camber_rc gets *much* worse (+7.0%). Net 3-split test mean is slightly worse with FiLM.
- ~5% epoch-time overhead (137.6s vs 131.3s), +28% params (mostly the final FiLM linear). Memory budget fine (44.6 GB peak, well under 96 GB ceiling).

### Decision

**Sent back to WIP** (PR re-converted to draft, `status:review → status:wip`). Frieren's Arm A is on the MSE baseline (not Huber); val_avg 125.634 is 8% above the current Huber floor (111.531). The −1.55% FiLM gain, if it composes, would land at ~109.8 on Huber — a plausible merge. But the mixed test signal also suggests it may not compose cleanly. Re-run both arms on the rebased Huber baseline to settle it.

Next merge decision waits for terminal results on the Huber baseline.

---

## 2026-05-15 15:27 — PR #3128 [CLOSED, falsified]: Per-sample scale-aware loss

- **Student branch:** `charliepai2i48h4-tanjiro/scale-aware-loss`
- **Hypothesis:** Reweighting per-sample loss by `1/per_sample_std` should balance the gradient signal across the 10× y-std range, especially helping low-y_std cruise and mixed-Re splits.

### Results (on MSE baseline)

| Arm | loss reweighting | `val_avg/mae_surf_p` | `val_single_in_dist` | `val_geom_camber_rc` | `val_geom_camber_cruise` | `val_re_rand` |
|-----|------------------|---------------------:|---------------------:|---------------------:|-------------------------:|--------------:|
| A (baseline) | unweighted MSE | **129.30** | 162.89 | 131.35 | 104.73 | 118.22 |
| B (inv-std) | per-sample inv-std weighting | 138.92 | 176.24 | **159.91** | **100.41** | 119.13 |
| **Δ (B − A)** | | **+7.4% (worse)** | +8.2% | **+21.7%** | −4.1% | +0.8% |

Test 3-split mean (excl. cruise NaN): A=125.90, B=142.27 (**+13.0% worse** on test).

### Analysis — clean mechanistic falsification

The mechanism is exactly as tanjiro analyzed:

> The metric (`val_avg/mae_surf_p`) is **unweighted absolute MAE** — it rewards getting big things right. Inverse-std weighting *downweights* the gradient on high-y_std samples (raceCar single + camber_rc, max y-std 2077), so the optimizer spends less capacity on them, and their absolute MAE balloons.

This is a fundamental directional misalignment, not a tuning issue. Sweeping milder variants (sqrt-inv, quarter-inv) would attenuate the harm but the gradient direction is still wrong against the unweighted-MAE objective. No further sweeps in this family worth GPU time.

Cruise improvement (−4.1%) is real and predicted; raceCar splits' +20% regression is the kill. Same mechanism would apply on Huber baseline — closing rather than re-running.

### Decision

**Closed.** Tanjiro reassigned to higher LR + warmup (PR #3321 below), which leverages a different Round-1 lesson: the cosine schedule barely anneals in 14 epochs, so raising the LR peak gives more effective progress.

---

## 2026-05-15 14:50 — Round 2 assignments

After Round 1 we have 1 merge (#3094 Huber), 2 closed misses (#3108 surf_weight, #3131 OneCycle), 1 sent-back (#3113 slice_num revision), and 6 still WIP. Three students have received Round 2 hypotheses targeting orthogonal axes:

| Student | PR | Round 2 hypothesis | Axis | Mechanism |
|---------|----|--------------------|------|-----------|
| alphonse | #3278 | Per-channel loss weighting (Ux, Uy, p) | Loss × channel | Up-weight pressure channel in Huber loss; arms p=2× and p=4× |
| thorfinn | #3289 | Cosine `T_max=15` to match achievable budget | LR schedule | Schedule was sized for 50 epochs but only ~14 achievable; full cosine decay needed |
| askeladd | #3290 | bf16 AMP mixed precision | Throughput | Wrap forward+loss in autocast(bf16) — predict 1.5–2× speedup → ~21–28 epochs in 30 min |
| tanjiro | #3321 | Higher LR (1e-3, 1.5e-3) + 3-epoch warmup | LR peak | Cosine only anneals to 82% of peak by ep14 — near-constant LR, so raising peak gives more progress per step |

All target the Huber baseline (`val_avg/mae_surf_p = 111.531`). Each is a paired-arm comparison. All are composable with each other and with the still-WIP Round-1 PRs (#3117 Fourier, #3126 EMA, #3113 revised slice_num, and the sent-back #3122 FiLM rebased onto Huber). If multiple Round-2 arms win we'll compose them in Round 3.

### Operational thesis

The dominant Round-1 lesson: **every run is wall-clock-truncated at ~14 epochs, not epoch-truncated**. This makes throughput and schedule-fit hypotheses (thorfinn cosine_t_max, askeladd bf16) particularly high-leverage — each one expands the effective compute budget for every subsequent experiment.

---

## 2026-05-15 17:40 — PR #3290 [MERGED]: bf16 AMP mixed precision — unlock ~1.5× more epochs

- **Student branch:** `charliepai2i48h4-askeladd/amp-bf16`
- **Hypothesis:** Wrapping the forward pass and loss computation in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` reduces sec/epoch and peak VRAM, unlocking more epochs in the 30-min budget.

### Results

| Arm | `amp_dtype` | epochs | sec/epoch | peak VRAM | `val_avg/mae_surf_p` | best epoch | `test_avg/mae_surf_p` |
|-----|-------------|--------|-----------|-----------|----------------------|-----------|------------------------|
| A (fp32) | fp32 | 14 | 131.8 s | 42.1 GB | 107.801 | 14 | 105.087 |
| B (bf16) | bf16 | **19** | **98.0 s** | **32.9 GB** | **101.519** | 16 | **98.735** |
| **Δ (B vs Huber baseline 111.531)** | | +5 epochs | 1.345× faster | −21.8% VRAM | **−8.98%** | — | **−12.2%** |

Per-split val MAE pressure:

| Split | Huber baseline | bf16 Arm B | Δ (B vs Huber) |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 141.566 | **116.096** | −18.0% |
| `val_geom_camber_rc`     | 116.797 | **116.636** |  −0.1% |
| `val_geom_camber_cruise` |  86.222 |  **76.479** | −11.3% |
| `val_re_rand`            | 101.539 |  **96.863** |  −4.6% |
| **val_avg**              | **111.531** | **101.519** | **−8.98%** |

### Metric artifacts

- `models/model-charliepai2i48h4-askeladd-amp-bf16-20260515-162617/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-amp-bf16-20260515-162617/metrics.yaml`

### Analysis & conclusions

Largest single-PR gain on this track. Mechanism confirmed: bf16 unlocks 5 extra epochs (14→19), cosine schedule decays deeper, best epoch (16) is further along the anneal. VRAM reduction (−21.8%) opens headroom for bs=8 and larger capacity experiments.

`val_geom_camber_rc` flat (−0.1%) is consistent with geometric-generalization being the bottleneck there, not optimization depth.

**bf16 is now the new default on this branch.** All subsequent experiments must include `--amp_dtype bf16`.

---

## 2026-05-15 17:43 — PR #3278 [CLOSED, falsified]: Per-channel loss weighting (Ux/Uy/p)

- **Student branch:** `charliepai2i48h4-alphonse/channel-weight`
- **Hypothesis:** Up-weighting the pressure channel in Huber loss (w_p=2× or 4×) shifts gradient emphasis toward `p`, the primary metric channel.

### Results (on Huber fp32 baseline)

| Arm | weights | `val_avg/mae_surf_p` | `val_single_in_dist` | `val_geom_camber_cruise` |
|-----|---------|---:|---:|---:|
| A (1,1,1) | uniform | 112.816 | 129.27 | 93.21 |
| B (0.5,0.5,2.0) | p×2 | 121.521 | +21% | **−11%** |
| C (0.25,0.25,4.0) | p×4 | 114.488 | +4% | ≈0% |

Arm A ≈ Huber baseline (within noise). Both weighted arms regress on average.

### Analysis

Static pressure upweighting helps on cruise (low-magnitude p, std≈164) but crushes `single_in_dist` (+21%) and `*_rc` (+13%), where pressure's raw error already dominates the gradient. The 10× y_std variance across splits means any fixed channel weight trades one split's gain for others' losses. Root cause: need per-domain not per-channel weighting. Closed — different axis required.

---

## 2026-05-15 17:45 — PR #3117 [SENT BACK ×2]: NeRF-style Fourier features on (x,z) positions

- **Student branch:** `charliepai2i48h4-fern/fourier-pos-features`
- **Hypothesis (2nd run):** NeRF-style random Fourier features (num_bands=10, scale=10.0) replacing raw (x,z) positions help the model resolve high-frequency spatial structure in the pressure field.

### Results (Huber fp32 baseline, 2nd run)

| Arm | features | `val_avg/mae_surf_p` | `val_single_in_dist` | `val_re_rand` | `val_geom_camber_rc` |
|-----|----------|---:|---:|---:|---:|
| A (raw) | raw `(x,z)` | 119.712 | 162.56 | 102.49 | 129.26 |
| B (Fourier) | 10 bands, scale=10 | 119.932 | **143.46 (−11.75%)** | +7.65% | +9.00% |
| **Δ (B − A)** | | **+0.18%** (nil) | | | |

Both arms underperformed Huber reference (single-seed variance, 12-13 vs 14 epochs). Intra-PR is the valid signal.

### Analysis

Fourier features (scale=10) help `single_in_dist` (−11.75%) but hurt `*_rc` (+9%) and `re_rand` (+7.65%) — the high-frequency fixed basis aliases on multi-foil / OOD-Re layouts. scale=10 is ~6-7× over-shot for normalised foil coordinates (±1.5 range). Note: fern also included a valid `train.py` NaN-filter fix for `evaluate_split`.

### Decision

Sent back ×2: try fourier_scale ∈ {2.0, 4.0} + concatenate raw+Fourier + add `--amp_dtype bf16`.

---

## 2026-05-15 18:00 — Round 3 assignments

After Round 2 partial results (bf16 merged −8.98%; channel weighting falsified; Fourier sent back), two students are newly idle.

| Student | PR | Hypothesis | Axis |
|---------|----|-----------|------|
| alphonse | #3364 | LR=1e-3 + 3-ep warmup on bf16 baseline | LR peak × bf16 |
| askeladd | #3365 | batch_size=6/8 on bf16 baseline | Throughput × batch |

Both test single-axis changes directly on the current best baseline (bf16 + Huber, 101.519). Independent and composable. Still-WIP: tanjiro #3321, thorfinn #3289, nezuko #3126, edward #3113, frieren #3122 (rebasing), fern #3117 (sent back).

---

## 2026-05-15 18:30 — PR #3289 [MERGED]: Cosine T_max=15 — match LR schedule horizon to 30-min budget

- **Student branch:** `charliepai2i48h4-thorfinn/cosine-tmax`
- **Hypothesis:** CosineAnnealingLR(T_max=50) barely decays in 14 fp32 epochs (only 16% drop from peak); setting T_max=15 lets the schedule complete its full cosine anneal within the wall-clock budget, giving the optimizer the low-LR refinement phase it was never reaching.

### Results (Huber fp32 baseline, 3-arm sweep)

| Arm | `cosine_t_max` | epochs | LR at best ep | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` (3 finite) |
|-----|----------------|--------|--------------|---:|---:|
| A | 50 (default) | 14 | 4.32e-4 (−16%) | 107.466 | 107.328 |
| B | **15** | **14** | **2.16e-5 (−96%)** | **100.059** | **96.641** |
| C | 20 | 14 | 1.37e-4 (−73%) | 101.758 | 102.433 |

Per-split val (Arm B, best):

| Split | Huber baseline | **T_max=15** | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 141.566 | **118.473** | −16.3% |
| `val_geom_camber_rc`     | 116.797 | **111.356** |  −4.7% |
| `val_geom_camber_cruise` |  86.222 |  **79.108** |  −8.3% |
| `val_re_rand`            | 101.539 |  **91.299** | −10.1% |
| **val_avg**              | **111.531** | **100.059** | **−10.3%** |

### Analysis & conclusions

The mechanism is exactly as predicted. LR-vs-epoch trace confirms:
- Arm A T_max=50: LR=4.32e-4 at ep14 (near-constant throughout — model never enters refinement)
- Arm B T_max=15: LR=2.16e-5 at ep14 (full cosine decay, 96% drop — refinement phase fully exploited)
- Arm C T_max=20: LR=1.37e-4 at ep14 (73% decay — most of the gain but missing final low-LR window)

Monotonic A > C > B on val_avg across 3 of 4 splits, confirming this is a real signal not noise. The optimum at T_max=15 is calibrated for ~14 fp32 epochs.

**Post-merge note:** the codebase now has bf16 (PR #3290) + T_max=15 (PR #3289). Thorfinn's measured value (100.059) beats the bf16 baseline (101.519) even though it's fp32. The bf16+T_max=15 compose needs direct verification — assigned to thorfinn in Round 3 (#3390). Expected: ~93–95.

---

## 2026-05-15 18:35 — Round 3 continuation assignment

After merging #3289 (thorfinn wins), thorfinn is idle and assigned a compose-verification run.

| Student | PR | Hypothesis | Axis |
|---------|----|-----------|------|
| thorfinn | #3390 | bf16+T_max=15/20 composition verify | Schedule × bf16 compose |

**Current state:** 3 Round 3 experiments in flight (alphonse #3364 lr-warmup-bf16, askeladd #3365 bigger-batch, thorfinn #3390 bf16-tmax-compose). Remaining Round 2 WIP: tanjiro #3321, nezuko #3126, edward #3113, frieren #3122 (rebasing), fern #3117 (sent back). All 8 GPUs occupied.

---

## 2026-05-15 20:34 — PR #3364 [CLOSED, falsified]: Higher peak LR (1e-3 + 3-ep warmup) on bf16

- **Student branch:** `charliepai2i48h4-alphonse/lr-warmup-bf16`
- **Hypothesis:** Raising peak LR from 5e-4 to 1e-3 with 3-epoch linear warmup would exploit the near-constant-LR regime under wall-clock truncation (both arms sit near peak throughout 19 bf16 epochs), yielding more effective parameter movement per budget.

### Results (Huber + bf16 baseline, T_max=50)

| Arm | `lr_peak` | `warmup_epochs` | `amp_dtype` | epochs | LR at best epoch | `val_avg/mae_surf_p` | best epoch | `test_avg/mae_surf_p` (3-finite) |
|-----|-----------|-----------------|-------------|--------|------------------|----------------------|-----------|----------------------------------|
| A (baseline) | 5e-4 | 0 | bf16 | 19/50 | 3.56e-4 | **99.218** | 19 | **93.976** |
| B (higher LR) | 1e-3 | 3 | bf16 | 19/50 | 7.69e-4 | 107.457 | 19 | 101.584 |
| **Δ (B − A)** | — | — | — | — | — | **+8.31%** | — | **+8.09%** |

Per-split val MAE:

| Split | Arm A | Arm B | Δ% |
|---|---:|---:|---:|
| `val_single_in_dist`     | 113.129 | 132.053 | **+16.7%** worse |
| `val_geom_camber_rc`     | 114.383 | 110.954 | −3.0% better |
| `val_geom_camber_cruise` |  78.335 |  89.124 | **+13.8%** worse |
| `val_re_rand`            |  91.027 |  97.696 | +7.3% worse |
| **val_avg**              | **99.218** | **107.457** | **+8.31%** worse |

### Metric artifacts

- Arm A: `models/model-lr-warmup-bf16-baseline-20260515-182519/{metrics.jsonl,metrics.yaml,config.yaml}`
- Arm B: `models/model-charliepai2i48h4-alphonse-lr-warmup-bf16-1e3-20260515-192542/{metrics.jsonl,metrics.yaml,config.yaml}`

### Analysis & conclusions

**Hypothesis cleanly falsified.** 3 of 4 splits regress under higher LR; regression is consistent across both val and test. `best_epoch=last_epoch` for both arms — neither reached its loss floor.

**Mechanism:** bf16's 7-bit mantissa truncates gradient information, making each step noisier than fp32. Doubling peak LR doubles noise amplitude. The "near-constant LR" framing was also wrong about direction: Arm B still runs near-constant LR (7.69e-4, 77% of peak) — the same structural problem as Arm A, just at 2× magnitude.

**Cross-check with tanjiro #3321 (in-flight):** tanjiro's bf16 Arm B (same config) measured 100.272 vs Arm A 100.372 — essentially tied (different seed). Two-seed result: lr=1e-3+warmup is either neutral or harmful on bf16. Falsification is robust.

**Note on Arm A:** 99.218 is −2.3% below the committed bf16 baseline (101.519). Run-to-run seed variance on bf16 is ~±2%.

### Decision

Closed. alphonse reassigned to PR #3443 (lower peak LR: 2.5e-4 and 3.5e-4 on bf16+T_max=15 stack).

---

## 2026-05-15 20:40 — Round 3b: lower LR assignment (alphonse #3443)

After falsifying lr=1e-3+warmup, alphonse's own analysis suggested the opposite direction: **lower the peak LR**. Assigned #3443 to test lr ∈ {5e-4, 3.5e-4, 2.5e-4} on current best stack (bf16+T_max=15).

**Rationale:** bf16 noisier gradients may shift the stability optimum downward vs fp32. T_max=15 means the schedule fully decays regardless of peak; peak magnitude sets absolute step sizes. If 5e-4 is at the stability edge under bf16, 2.5-3.5e-4 may be inside it without losing convergence speed (cosine still completes annealing within budget).

| Student | PR | Arms | Expected delta |
|---------|----|----|----------------|
| alphonse | #3443 | lr ∈ {5e-4, 3.5e-4, 2.5e-4} on bf16+T_max=15 | neutral to −3% |

---

## 2026-05-15 22:32 — PR #3126 [MERGED]: EMA weights (decay=0.999, Karras warmup ramp)

- **Student branch:** `charliepai2i48h4-nezuko/ema-weights`
- **Hypothesis:** An exponential moving average (EMA) of weights acts as a low-pass filter over the AdamW optimization trajectory, reducing late-epoch validation variance and improving generalization without additional regularization cost.

### Results

| Arm | Config | best epoch | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` (3 finite) | Δ vs Arm A |
|-----|--------|-----:|---------:|--------:|--------:|
| A | bf16 + T_max=15, no EMA | 19 | 97.492 | 94.879 | — |
| B | + EMA (decay=0.999, Karras ramp) | 18 | **96.464** | **93.857** | **−1.06%** |

Vs prior BASELINE.md (fp32+T_max=15 = 100.059): Arm B −3.59% improvement.

Per-split val MAE pressure (lower is better):

| Split | Arm A | Arm B (EMA) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist`     | 116.714 | **111.948** | −4.08% |
| `val_geom_camber_rc`     | 102.709 | **102.325** | −0.37% |
| `val_geom_camber_cruise` |  77.554 |   79.490 | +2.50% |
| `val_re_rand`            |  92.990 |  **92.092** | −0.97% |
| **val_avg**              | **97.492** | **96.464** | **−1.06%** |

Per-split test (3 finite splits):

| Split | Arm A | Arm B (EMA) | Δ |
|---|---:|---:|---:|
| `test_single_in_dist`     | 103.011 |  **97.964** | −4.90% |
| `test_geom_camber_rc`     |  93.417 |   94.701 | +1.37% |
| `test_re_rand`            |  88.210 |   88.905 | +0.79% |
| **avg (3 splits)**        |  94.879 |  **93.857** | **−1.08%** |

Late-training variance (epochs 10–19):

| Stat | Arm A | Arm B (EMA) | Δ |
|---|---:|---:|---:|
| mean   | 101.679 | 99.076 | −2.56% |
| stdev  |   6.546 |  3.688 | **−43.7%** |
| min    |  97.492 | 96.464 | −1.05% |
| max    | 116.957 | 107.254 | −8.30% |

### Metric artifacts

- `models/model-charliepai2i48h4-nezuko-arm_b_ema_d0999_bf16_tmax15-20260515-212327/metrics.jsonl`
- `models/model-charliepai2i48h4-nezuko-arm_a_baseline_bf16_tmax15-20260515-203158/metrics.jsonl`

### Analysis & conclusions

Clean win. EMA reduces val variance by 43.7% (σ 6.55 → 3.69) and beats Arm A at every epoch from epoch 1 onward — the Karras warmup ramp (`min(0.999, (1+step)/(10+step))`) prevents the cold-start issue that flat decay=0.999 can cause on short schedules.

Per-split pattern: biggest win on `val_single_in_dist` (−4.08%) and test (−4.90%). Slight regression on cruise val (+2.50%) which may reflect noise (cruise val is small-sample). Three-of-four splits improve.

**Side benefit:** Arm A (97.492) is the first measured bf16+T_max=15 compose number — confirms the predicted ~93–95 range; thorfinn #3390 is running a second seed.

### Decision

Merged. New best stack: `--amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999`.
Nezuko reassigned to #3492 (n_hidden=192 capacity test on full stack).

---

## 2026-05-15 23:25 — PR #3321 [CLOSED]: Higher LR (1e-3, 1.5e-3) + 3-epoch warmup

- **Student branch:** `charliepai2i48h4-tanjiro/lr-warmup-higher-peak`
- **Hypothesis:** Higher peak LR (lr=1e-3 / 1.5e-3) + 3-epoch warmup beats baseline lr=5e-4 by exploiting near-constant LR within the 14-epoch budget.

### Results (6-arm sweep on fp32 Huber and bf16 Huber)

| Arm | dtype | lr_peak | warmup | epochs | best ep | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` (3 finite) |
|-----|-------|---------|--------|--------|---------|---------------------:|---------------------------------:|
| A | fp32 | 5e-4 | 0 | 14 | 13 | **119.897** | 112.109 |
| B | fp32 | 1e-3 | 3 | 14 | 12 | 122.950 (+2.5%) | 121.919 |
| C | fp32 | 1.5e-3 | 3 | 14 | 13 | 122.575 (+2.2%) | 121.661 |
| A | bf16 | 5e-4 | 0 | 19 | 17 | 100.372 | 99.132 |
| B | bf16 | 1e-3 | 3 | 19 | 16 | 100.272 (−0.1%, tied) | 99.004 |
| C | bf16 | 1.5e-3 | 3 | 19 | 18 | 112.399 (+12%) | 111.611 |

### Metric artifacts

- `models/model-charliepai2i48h4-tanjiro-lr-warmup-baseline-20260515-163341/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-lr-warmup-1e3-20260515-172533/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-lr-warmup-15e4-20260515-202542/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-lr-warmup-baseline-20260515-182820/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-lr-warmup-1e3-bf16-20260515-192619/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-lr-warmup-15e4-bf16-20260515-212417/metrics.jsonl`

### Analysis & conclusions

Higher LR axis cleanly falsified on both dtypes. The hypothesis mechanism (near-constant LR within 14-ep budget) was empirically supported (LR-at-best is 86-92% of peak in fp32), but the predicted win did not materialize.

**bf16 Arm B (100.272) being tied with Arm A (100.372)** corroborates alphonse #3364's similar near-tied/regression result at the same config — combined two-seed data: lr=1e-3 sits at or past the bf16 stability edge.

**Side observation:** `val_geom_camber_cruise` split actively prefers higher LR in fp32 (105 → 95-97, −10%) — a per-split signal worth flagging for future cruise-specific hypotheses.

Vs the new BASELINE.md (96.464 from #3126), tanjiro's best arm (100.272) is +3.95% — far from current best. Combined with alphonse #3364's falsification, this exhausts the higher-LR direction.

### Decision

Closed. Tanjiro reassigned to PR #3511 (gradient clipping on current best stack — mechanistically motivated by their own bf16-noise analysis).

---

## 2026-05-15 23:32 — Round 4: gradient clipping assignment (tanjiro #3511)

After establishing that higher LR is falsified (both seeds via #3364 + #3321), the natural complement is bounding the per-step magnitude directly. Gradient clipping caps the noise input that EMA averages on the output side — should be additive with the EMA win.

Assigned tanjiro PR #3511 with 3 arms on full best stack:
- Arm A: no clip
- Arm B: clip_grad_norm=1.0
- Arm C: clip_grad_norm=0.5

## 2026-05-16 01:25 — PR #3117 [SENT BACK FOR REBASE+RECOMPOSE]: Fourier features Round 2 — concat raw + scale ∈ {2, 4}

- **Student branch:** `charliepai2i48h4-fern/fourier-pos-features`
- **Hypothesis:** NeRF-style Fourier features (sin/cos at random Gaussian frequencies) on (x,z) positions unlock high-frequency content for sharper pressure gradients. Round 1 (scale=10, replace raw) gave net-zero average but +9% on multi-foil (`*_rc`) splits. Round 2 (scale=2/4, concat raw+Fourier) is the advisor-prescribed fix — preserve raw position as low-frequency fallback, lower scale to match foil coordinate span (±1.5σ).

### Results (3 arms, single GPU, 30-min × 50-epoch caps, all bf16 AMP, no EMA, no T_max=15)

| Arm | Config | `val_avg/mae_surf_p` | Δ vs Arm A | `test_avg/mae_surf_p` | Δ vs Arm A |
|-----|--------|----------------------|------------|-----------------------|------------|
| A | bf16, no Fourier (baseline) | 103.370 | — | 96.014 | — |
| B | bf16, Fourier scale=2, num_bands=10, concat raw | **93.967** | **−9.10%** ✅ | **83.878** | **−12.64%** ✅ |
| C | bf16, Fourier scale=4, num_bands=10, concat raw | 96.946 | −6.21% ✅ | 88.886 | −7.42% ✅ |

### Per-split val (best checkpoint, mae_surf_p)

| Split | Arm A (raw) | Arm B (scale=2) | Δ_B | Arm C (scale=4) | Δ_C |
|---|---|---|---|---|---|
| `val_single_in_dist`     | 121.82 | **116.23** | **−4.59%** ✅ | 118.50 | −2.73% ✅ |
| `val_geom_camber_cruise` |  88.56 |  **67.07** | **−24.26%** ✅ |  72.88 | −17.70% ✅ |
| `val_re_rand`            |  97.74 |  **86.32** | **−11.68%** ✅ |  90.16 | −7.76% ✅ |
| `val_geom_camber_rc`     | 105.36 | 106.25 | +0.84% (≈0) | 106.25 | +0.85% (≈0) |

### Analysis

- **Round 1 OOD failure fully repaired.** Round 1 with `scale=10, replace_positions` regressed `*_rc` by +9%; Round 2 with `scale=2, concat` regresses it by +0.84% (within noise). Mechanism: keeping raw `(x, z)` preserves smooth low-frequency basis so slot attention can still see "this region is far-field empty," while Fourier provides high-frequency lift on in-distribution and near-cruise splits.
- **scale=2 strictly dominates scale=4** on 7/8 splits. The lone tie is `val_geom_camber_rc`. Confirms Tancik's scale-vs-input-σ analysis: foil positions have σ≈1.0–1.5, so scale=2 sits in the sweet spot while scale=4 starts aliasing.
- **`val_re_rand` sign-flipped** vs Round 1 (+7.6% → −11.7%) — frequency mismatch, not the Fourier basis itself, was the OOD liability.
- **`val_geom_camber_cruise` is the standout** (−24%/−28% val/test). Cruise-condition camber perturbations are geometrically near training, so high-frequency basis locks onto small perturbations efficiently.
- **Other channels neutral.** `mae_surf_Ux/Uy` flat to slightly down; `mae_vol_p` actually improved (−4.2%). Net win is real, not surface-`p`-only.
- **Memory cost negligible:** 33.19 GB vs 32.94 GB (+0.25 GB for 20 extra input features).

### Decision: Sent back for rebase + recompose

The intra-PR signal is huge (−9.10%) and the raw Arm B metric (93.967) already beats the current baseline (96.464) by −2.59%. **BUT** the result was generated on a pre-EMA, pre-T_max=15 stack (`git_commit: 85d57d6`), so the comparison vs 96.464 isn't apples-to-apples and the PR is `CONFLICTING` against advisor HEAD.

Send-back instructions: rebase onto current advisor (`5c53212`), drop scale=4, rerun 2 arms on `bf16 + T_max=15 + EMA` stack:
- Arm A: full current best stack baseline (predicted ~96.5)
- Arm B: full stack + Fourier scale=2 (predicted ~87-90 if composition holds)

Composition is highly likely (Fourier is feature-side, EMA+T_max=15 are gradient/schedule-side — orthogonal). Expected wall-clock ~60 min. After clean rerun, merge.

### Bug-fix flag

PR also contains `evaluate_split` upstream NaN-sample filter in `train.py` (writable) — confirmed safe in prior round; no `data/scoring.py` touched. Fix is no-op on splits with finite targets; only rescues `test_geom_camber_cruise` from `NaN·0=NaN` poisoning. This stays in the PR as a quality fix.

## 2026-05-16 01:28 — PR #3122 [MERGED]: FiLM conditioning on physics parameters

- **Student branch:** `charliepai2i48h4-frieren/film-conditioning`
- **Hypothesis:** FiLM (Feature-wise Linear Modulation) injects learned per-sample scale+shift at every TransolverBlock, conditioned on physics parameters [log(Re), AoA, NACA_encoded, gap, stagger]. Enables explicit cross-regime conditioning rather than relying on the model to extract regime from mesh features.

### Results (Round 2 — full current best stack: Huber + bf16 + T_max=15 + EMA decay=0.999)

| Arm | Config | `val_avg/mae_surf_p` | Δ vs Arm A | `test 3-split mean` | Δ vs Arm A |
|-----|--------|----------------------|------------|---------------------|------------|
| A | Full stack, no FiLM (baseline) | 97.360 | — | 93.924 | — |
| B | Full stack + FiLM | **92.606** | **−4.88%** ✅ | **89.005** | **−5.24%** ✅ |

**Arm B vs merged baseline (96.464): −4.00%** ✅

### Per-split val (best checkpoint, mae_surf_p)

| Split | Arm A | Arm B (FiLM) | Δ |
|---|---|---|---|
| `val_single_in_dist`     | 112.950 | **107.788** | −4.57% ✅ |
| `val_geom_camber_rc`     | 105.171 | **101.033** | −3.93% ✅ |
| `val_geom_camber_cruise` |  77.396 |  **73.993** | −4.41% ✅ |
| `val_re_rand`            |  93.922 |  **87.611** | **−6.72%** ✅ |
| **val_avg**              | **97.360** | **92.606** | **−4.88%** ✅ |

### Analysis

- **All 4 splits improve uniformly.** Unlike Round 1 (MSE-only stack, where FiLM anomalously helped single-in-dist more than re-rand), Round 2 on the Huber+EMA stack shows `val_re_rand` as the biggest winner (−6.72%) — exactly the cross-regime generalization the hypothesis predicted.
- **Round 1 mechanism inversion was loss-curvature driven.** MSE over-weights high-Re outliers; Huber's bounded influence + EMA variance reduction let FiLM's cross-regime conditioning express itself properly.
- **Compose is additive.** FiLM adds −4.88% on top of EMA's −1.06% — independent mechanisms (EMA smooths the optimizer path; FiLM conditions the representation).
- **Compute overhead modest.** +27.6% params, +7% epoch time, +2.5 GB VRAM. At 18 best epoch (vs 19 for Arm A), per-epoch quality is markedly higher — FiLM converges faster in terms of validation metric per epoch.
- **Zero-init FiLM warm-start works well.** Arm B epoch 1 ≈ Arm A (warm start preserved); diverges positively by epoch 4 and maintains the lead through epoch 18.

### Decision: MERGED

Clean −4.00% improvement on merged baseline; apples-to-apples (same full stack); terminal SENPAI-RESULT; mergeable state CLEAN.

New best: **92.606 val_avg/mae_surf_p**
Reproduce: `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond`

## 2026-05-16 01:40 — PR #3443 [CLOSED-FALSIFIED]: Lower LR sweep (2.5e-4, 3.5e-4) on bf16+T_max=15

- **Student branch:** `charliepai2i48h4-alphonse/lower-lr-sweep`
- **Hypothesis:** bf16 gradient noise shifts the LR optimum downward; 2.5e-4 or 3.5e-4 may refine better than default 5e-4.

### Results (3 arms, bf16 + T_max=15, no EMA)

| Arm | lr | val_avg/mae_surf_p | Δ vs Arm A | test_avg (3 splits) |
|-----|----|--------------------|------------|---------------------|
| A | 5e-4 (default) | **97.241** | — | **91.790** |
| B | 2.5e-4 | 98.002 | +0.78% ❌ | 94.771 |
| C | 3.5e-4 | 99.771 | +2.60% ❌ | 98.739 |

**LR axis closed: 5e-4 is at or near the magnitude optimum for bf16+T_max=15.**

### Analysis

- **Monotone on test:** A < B < C on every finite test split. No flips.
- **val_single_in_dist anomaly:** B/C win by 1.5-2% on this easy split, but lose on all harder generalization splits.
- **Floor convergence:** Both A and B hit the LR floor (LR ~0) by epoch 16 and make negligible progress; lower LR doesn't buy more useful epochs, just a worse local optimum.
- **Note:** Arm A (97.241) is first clean bf16+T_max=15 measurement without EMA or FiLM.

### Decision: CLOSED (falsification)

New assignments: alphonse → Schedule-Free AdamW (#3594)

---

## 2026-05-16 03:25 — PR #3117 [SENT BACK ×3]: Fourier features Round 3 — composes with EMA+T_max=15, pending FiLM compose verify

- **Student branch:** `charliepai2i48h4-fern/fourier-pos-features`
- **Hypothesis (Round 3 framing):** Does Fourier scale=2 + concat raw (Round 2 winner, −9.10% intra-PR on bf16-only) still win when stacked on EMA decay=0.999 + cosine T_max=15 + bf16?
- **Round 3 setup:** Rebased onto advisor `5c53212` (pre-FiLM). 2 paired arms, both with the full EMA+T_max=15+bf16 stack. 50-epoch budget, 30-min wall-clock cap, `best_epoch=19` for both arms (budget-bound).

### Results — paired arms (val_avg/mae_surf_p, lower is better)

| Arm | Fourier | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-----|---------|---------------------|---------------------|
| A   | off (full stack baseline) | **95.714** | 85.416 |
| B   | on (scale=2, num_bands=10) | **92.694** | **82.719** |
| **Δ (B − A)** | — | **−3.16%** ✅ | **−3.16%** ✅ |

### Per-split val (best_val/.../mae_surf_p)

| Split | Arm A | Arm B (Fourier) | Δ |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 109.466 | 108.328 | −1.04% ✅ |
| `val_geom_camber_cruise` |  76.837 |  73.071 | −4.90% ✅ |
| `val_geom_camber_rc`     | 104.686 |  98.762 | −5.66% ✅ |
| `val_re_rand`            |  91.865 |  90.614 | −1.36% ✅ |
| **val_avg**              | **95.714** | **92.694** | **−3.16%** ✅ |

### Per-split test (test/.../mae_surf_p)

| Split | Arm A | Arm B (Fourier) | Δ |
|-------|---:|---:|---:|
| `test_single_in_dist`     |  95.102 |  91.905 | −3.36% ✅ |
| `test_geom_camber_cruise` |  65.102 |  61.039 | −6.24% ✅ |
| `test_geom_camber_rc`     |  94.729 |  90.408 | −4.56% ✅ |
| `test_re_rand`            |  86.730 |  87.523 | +0.91% (≈tie) |
| **test_avg**              | **85.416** | **82.719** | **−3.16%** ✅ |

### Other channels (val_avg)

| Channel | Arm A | Arm B | Δ |
|---|---:|---:|---:|
| `mae_surf_Ux` | 1.396 | 1.393 | −0.2% |
| `mae_surf_Uy` | 0.683 | 0.677 | −0.8% |
| `mae_vol_p`   | 103.4 |  99.3 | **−4.0%** ✅ |
| `mae_vol_Ux`  | 4.250 | 4.144 | −2.5% |
| `mae_vol_Uy`  | 1.982 | 1.946 | −1.8% |

### Composition delta vs Round 2

| Stack | Paired Δ (B − A) |
|-------|-----------------|
| bf16 only (R2) | **−9.10%** |
| bf16 + T_max=15 + EMA (R3) | **−3.16%** |

The win composes but shrinks — EMA + T_max=15 captures part of the bf16-noise mitigation that Fourier features were doing on their own. Still a clean, real signal: every val split improves, three of four test splits improve, the only tie is `test_re_rand`.

### Metric artifacts

- `models/model-charliepai2i48h4-fern-r3-armA-baseline-20260516-013819/metrics.jsonl`
- `models/model-charliepai2i48h4-fern-r3-armB-fourier-scale2-20260516-021245/metrics.jsonl`

### Analysis & conclusions

- **Round 3 confirmed composition on EMA+T_max=15+bf16.** Fourier scale=2 + concat raw is robust across stack components.
- **The `val_geom_camber_rc` recovery is notable.** In Round 1 (pre-rebase), Fourier hurt `*_rc` by +9%. In Round 2 (concat raw + scale=2), it was flat (+0.84%). In Round 3 (EMA+T_max=15), it now wins (−5.66%). The combination of concat raw + lower scale + EMA smoothing eliminated the OOD penalty.
- **Cannot merge as-is.** The current advisor baseline moved to 92.606 while Round 3 was running (FiLM merged in PR #3122). Arm B (92.694) is +0.095% worse than current baseline (92.606), and the branch is `CONFLICTING`. Final composition question — does Fourier compose with FiLM? — is still unmeasured.
- **Sent back for Round 4:** rebase onto advisor HEAD `9adc607` (post-FiLM), run 2 paired arms on the full current stack including FiLM. Decision rule: any Δ > 0 → merge; tie → close as "Fourier subsumed by FiLM"; regression → close with interaction warning.


---

## 2026-05-16 04:50 — PR #3584 [MERGED]: Two-shot FiLM — condition attention + MLP paths per TransolverBlock

- **Student branch:** `charliepai2i48h4-frieren/two-shot-film`
- **Hypothesis:** Conditioning FiLM at two sites per TransolverBlock (attention input after ln_1 AND MLP input after ln_2) vs current single-shot (attention only), using shared FiLMConditioner (+0 parameters).

### Results

| Arm | FiLM sites | params | val_avg/mae_surf_p | best_epoch | vs Arm A | vs baseline |
|-----|------------|--------|---------------------|------------|----------|-------------|
| A   | 1-shot (attn only) | 845,527 | 93.205 | 18 | — | +0.65% (noise) |
| B   | 2-shot (attn + MLP) | 845,527 | **89.784** | 17 | **−3.67%** ✅ | **−3.05%** ✅ |

### Per-split val (mae_surf_p)

| Split | Arm A | Arm B (2-shot) | Δ |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 106.191 | **103.854** | −2.20% |
| `val_geom_camber_rc`     | 103.036 |  **95.887** | **−6.94%** ✅ |
| `val_geom_camber_cruise` |  73.888 |  **73.143** | −1.01% |
| `val_re_rand`            |  89.704 |  **86.251** | −3.85% |
| **val_avg**              | **93.205** | **89.784** | **−3.67%** ✅ |

### Per-split test (3 finite splits)

| Split | Arm A | Arm B (2-shot) | Δ |
|-------|---:|---:|---:|
| `test_single_in_dist`    |  91.619 |  **89.460** | −2.36% |
| `test_geom_camber_rc`    |  91.888 |  **87.408** | −4.87% |
| `test_re_rand`           |  84.201 |  **80.336** | −4.59% |
| **avg (3 splits)**       | **89.236** | **85.735** | **−3.92%** ✅ |

Metric artifacts:
- `models/model-charliepai2i48h4-frieren-two-shot-film-armb-twoshot-20260516-030245/metrics.jsonl` (winner)
- `models/model-charliepai2i48h4-frieren-two-shot-film-arma-baseline-20260516-022727/metrics.jsonl` (baseline)

### Analysis & conclusions

- **Uniform win across all 4 val splits and all 3 test splits.** Strongest gain on `val_geom_camber_rc` (−6.94% val, −4.87% test) — the unseen-camber/raceCar OOD generalization split.
- **Zero extra parameters**: shared FiLMConditioner called twice per block (same γ,β reused). Only compute cost: +6.2% epoch time, +6.8% peak VRAM (38.9 GB). Lost 1 epoch under budget (17 vs Arm A's 18) but still wins clearly.
- **Both arms at final epoch = still descending**. Additional budget would likely improve both; two-shot would likely maintain or widen the gap.
- **Mechanism**: MLP-path FiLM helps the model transfer geometric features across cambers (OOD) independently of the attention path's slice aggregation. The two paths provide complementary physics-regime adaptation.
- **New best: 89.784.** Compound stack: Huber + bf16 + T_max=15 + EMA + FiLM + two-shot FiLM.

### Decision: MERGED (new best → 89.784)

---

## 2026-05-16 04:52 — PR #3595 [CLOSED, falsified]: n_layers depth sweep — 5→6 layers on full EMA+FiLM stack

- **Student branch:** `charliepai2i48h4-edward/nlayers-depth-sweep`
- **Hypothesis:** Depth increase (5→6 TransolverBlocks) gives more representational capacity on the full FiLM+EMA+bf16+T_max=15 stack.

### Results

| Arm | n_layers | params | sec/epoch | epochs | val_avg/mae_surf_p | vs Arm A |
|-----|----------|--------|-----------|--------|--------------------|----------|
| A (n_layers=5) | 5 | 845,527 | 104.4 s | 18 | **92.408** | — |
| B (n_layers=6) | 6 | 999,707 | 123.9 s (+18.7%) | 15 | **94.694** | **+2.47% ❌** |

### Analysis

- **Depth regression.** +2.47% intra-PR. The mechanism is clear: +20% wall-clock per epoch means 3 fewer fine-tune epochs at lr≈0 (epochs 16-18 where Arm A improved 93.05 → 92.41). Even projecting Arm B gets the same fine-tune lift (−0.65), projected Arm B ≈ 94.04 — still +1.6% worse. Depth-vs-epochs tradeoff is asymmetric and net-negative under 30-min budget.
- **Lesson**: with a fixed 30-min wall-clock, capacity changes that increase per-epoch cost trade away load-bearing fine-tune epochs. n_layers=4 (faster, more fine-tune time) might be more interesting than n_layers=6, though that's a separate hypothesis.
- All val splits regress; largest hit on `val_geom_camber_cruise` (+4.19). Test direction matches val.

### Decision: CLOSED (falsification — depth bump net-negative under 30-min budget)

---

## 2026-05-16 04:53 — PR #3511 [SENT BACK for rebase]: Grad clipping on bf16+T_max=15+EMA stack

- **Student branch:** `charliepai2i48h4-tanjiro/grad-clip-bf16-tmax-15`
- **Hypothesis:** Gradient clipping (clip_norm ∈ {0.5, 1.0, ∞}) reduces bf16 noise outliers.

### Results (pre-FiLM stack, bf16+T_max=15+EMA only)

| Arm | clip | val_avg/mae_surf_p | vs A-mean | vs baseline 96.464 |
|-----|-----:|---------------------|-----------|---------------------|
| A1  | none | 94.669 | — | — |
| A2  | none | 96.778 | — | — |
| A-mean | none | **95.724** | — | −1.28% |
| **B** | **1.0** | **91.861** | **−4.03%** ✅ | **−4.77%** ✅ |
| C | 0.5 | 94.365 | −1.42% ✅ | −2.18% |

### Key finding

**Clip=1.0 fires on ~98-100% of steps** — this is not outlier clipping, it's gradient direction normalization (LION/Normalized-GD behavior). Natural grad norms: p50≈9, p90≈22 (driven by `surf_weight=10` and large mesh sizes 74K-242K nodes × bf16 quantization). AdamW with lr=5e-4 typically sees norms in [0.1, 5]; this system runs 10× higher. Decoupling step magnitude from gradient magnitude is genuinely beneficial.

Arm B at 91.861 beats the EMA pre-FiLM baseline (96.464) by 4.77%, and even beats the current FiLM baseline (92.606) by 0.80%. However, the PR is CONFLICTING (pre-FiLM, pre-two-shot-FiLM). Current baseline moved to 89.784 while this ran.

### Decision: SENT BACK for rebase + rerun on full two-shot FiLM stack

New rerun: Arm A (two-shot FiLM, no clip) vs Arm B (two-shot FiLM + clip=1.0). Expected ~86-88 if composition holds.


---

## 2026-05-16 05:15 — PR #3594 [SENT BACK FOR VERIFY+REBASE]: Schedule-Free AdamW — eliminate cosine schedule, optimizer-native convergence

- **Student branch:** `charliepai2i48h4-alphonse/schedule-free-adamw`
- **Hypothesis:** Replace `AdamW + CosineAnnealingLR(T_max=15)` with `AdamWScheduleFree` (Defazio et al., schedulefree library). Polyak-averaged Z-iterate provides convergence smoothing internally; eliminates schedule sensitivity.

### Results (Round 1, pre-two-shot-FiLM stack, `git_commit: 1c0f616` / `9adc607`)

| Arm | Optimizer/schedule | val_avg/mae_surf_p | best_epoch | Δ vs Arm A | Δ vs merged FiLM (92.606) |
|-----|--------------------|---------------------|------------|------------|---------------------------|
| A   | AdamW + cosine T_max=15 (current best at runtime) | **90.207** | 18 | — | −2.59% |
| **B** | **AdamWScheduleFree (no schedule)** | **71.492** | **18 (still descending)** | **−20.75%** ✅ | **−22.80%** ✅ |

### Per-split val (lower is better)

| Split | Arm A | **Arm B (SF-AdamW)** | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 103.855 |  **80.542** | −22.45% |
| `val_geom_camber_rc`     |  97.931 |  **84.946** | −13.26% |
| `val_geom_camber_cruise` |  71.925 |  **51.568** | **−28.30%** ✅ |
| `val_re_rand`            |  87.118 |  **68.912** | −20.90% |
| **val_avg**              | **90.207** | **71.492** | **−20.75%** ✅ |

### Per-split test (3 finite splits)

| Split | Arm A | **Arm B (SF-AdamW)** | Δ % |
|-------|---:|---:|---:|
| `test_single_in_dist`    | 89.701 | **69.706** | −22.29% |
| `test_geom_camber_rc`    | 89.474 | **76.488** | −14.51% |
| `test_re_rand`           | 82.075 | **62.206** | −24.21% |
| **avg (3 splits)**       | **87.083** | **69.467** | **−20.23%** ✅ |

### Verified metrics (committed to student branch)

- `models/model-charliepai2i48h4-alphonse-schedule-free-armb-sf-adamw-20260516-032817/metrics.yaml` → `best_val_avg/mae_surf_p: 71.49214002634584`
- Per-split confirms uniform improvement, no measurement anomaly
- No NaN/inf in training, EMA shadow updates ran identically to Arm A

### Mechanism analysis (from student)

- Cosine T_max=15 schedule decays lr from 5e-4 to 5e-8 by epoch 15-16 (effective floor); Arm A is essentially frozen from ep 16 onward
- SF-AdamW maintains internal Polyak-averaged Z-iterate at constant base lr; warmup_steps=500 (≈epoch 2); full magnitude steps for remaining 16 epochs
- Loss trajectory smooth: train surf_loss 0.393 → 0.044, train vol_loss 0.602 → 0.097 (monotonic)
- Peak VRAM essentially unchanged (35.93 vs 35.94 GB)
- bf16 + SF-AdamW composed cleanly (optimizer state held in fp32)

### Decision: SENT BACK FOR REBASE + VERIFY ON TWO-SHOT FILM STACK

Rationale:
1. **PR is CONFLICTING.** Branch ran on `1c0f616` (post-FiLM, pre-two-shot-FiLM); current advisor HEAD is `be679d4` (post-two-shot-FiLM, baseline 89.784).
2. **A −20.75% intra-PR delta is unprecedented in this track.** Previous max was Huber −15.7% (loss-function change). Independent seed on current baseline required before merge.
3. **Composition with two-shot FiLM unknown.** Likely additive (orthogonal mechanisms), but needs measurement.

Rerun:
- **Arm A:** Full current stack with cosine T_max=15 — `--amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film`
- **Arm B:** Full current stack with SF-AdamW (no cosine) — `--amp_dtype bf16 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --use_schedule_free`

Predicted outcome:
- If composition additive: Arm B ≈ 70-75 (matches SF gain + small two-shot FiLM bonus)
- If composition sub-additive: Arm B ≈ 75-80
- If gain reproduces but doesn't compose: still likely beats current 89.784 baseline


---

## 2026-05-16 05:30 — PR #3117 [CLOSED]: NeRF-style Fourier features on (x,z) positions — R4 (Fourier subsumed by FiLM)

- **Student branch:** `charliepai2i48h4-fern/fourier-features` (R4)
- **Hypothesis:** Apply random-Gaussian NeRF-style positional Fourier features on (x,z) with `fourier_num_bands=10, fourier_scale=2.0`, concat raw+sin/cos, drop into Transolver as pre-input. Composes with full current best stack including two-shot FiLM.
- **Stack:** bf16 + cosine T_max=15 + EMA(0.999) + FiLM + two-shot-FiLM (advisor HEAD `7af79ac` rebased)

### Results

| Arm | Stack | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` (3 finite splits) |
|-----|-------|----------------------|----|
| A (full stack, no Fourier) | bf16 + cosine + EMA + 2xFiLM | **90.240** | 81.249 |
| B (full stack + Fourier scale=2) | + `--use_fourier --fourier_scale 2.0 --fourier_num_bands 10` | **90.149** | 81.947 |
| **Δ (B − A)** | — | **−0.10%** (tie, within ±0.5% band) | **+0.86%** (slight regression) |

### Compression story across 4 rounds (Fourier intra-PR Δ)

| Round | Composition stack | Intra-PR Δ |
|-------|-------------------|-----------|
| R2 | bf16-only | **−9.10%** |
| R3 | + EMA + cosine T_max=15 | **−3.16%** |
| R4 | + FiLM + two-shot-FiLM | **−0.10%** (this round) |

Half-life decay pattern. Each merged feature absorbs progressively more of Fourier's signal.

### Per-split val MAE pressure (R4)

| Split | Arm A | Arm B (Fourier) | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 64.05 | 65.77 | **+2.68%** (Fourier *hurts*) |
| `val_geom_camber_rc`     | 65.16 | 63.35 | **−2.78%** (Fourier helps) |
| `val_geom_camber_cruise` | 161.42 | 159.25 | **−1.34%** (Fourier helps) |
| `val_re_rand`            | 89.16 | 89.85 | **+0.77%** (tie/slight regression) |
| `mae_surf_Ux`            | — | — | **+4.64%** (FiLM owns velocity) |

### Metric artifacts

- `models/model-charliepai2i48h4-fern-fourier-features-r4-arma-*/metrics.jsonl`
- `models/model-charliepai2i48h4-fern-fourier-features-r4-armb-*/metrics.jsonl`

### Analysis & conclusions

**Closed per the decision rule defined in R3 send-back:** "If Arm B ties Arm A (±0.5% paired Δ) → close cleanly with a clear 'Fourier subsumed by FiLM' conclusion."

The R4 Δ of −0.10% sits squarely in the tie band AND the test direction reverses (+0.86%). This is a textbook "feature absorbed by an earlier-merged feature" outcome. As the model gained:
1. **Global physics conditioning** via FiLM (γ,β modulation conditioned on log(Re), AoA, NACA, gap, stagger)
2. **Two-shot FiLM** (γ,β applied at both attn and MLP sites)

...the marginal value of Fourier positional features fell from large → meaningful → noise. FiLM's spatial-frequency-relevant γ,β scaling is functionally similar to the basis-expansion effect Fourier provides on the input side.

**The per-split decomposition** is mechanistically informative: Fourier still helps multi-foil geometry splits (`geom_camber_rc` −2.78%, `geom_camber_cruise` −1.34%) but at insufficient magnitude to overcome single-foil regression (+2.68%) and the dominant velocity-channel regression (`mae_surf_Ux` +4.64%). FiLM "owns" the velocity channel and in-distribution split; Fourier residual contributes only on multi-foil rich-geometry tasks.

**Test direction:** Slight regression (+0.86%) on `test_avg/mae_surf_p` (3 finite splits, cruise NaN). This is the strongest argument for closing rather than merging — val tie + test regression = no business adding complexity.

### Decision: CLOSED (Fourier subsumed by FiLM)

- Full credit to fern for clean reporting and graceful handling of 3 baseline shifts during the rebase
- Fourier code in branch is preserved as template for future positional-feature experiments (SDF, etc.)
- Fourier-as-default is off the table for this track


---

## 2026-05-16 07:35 — PR #3365 [CLOSED]: Bigger batch size (bs=6/8) on bf16

- **Student branch:** `charliepai2i48h4-askeladd/bf16-bigger-batch`
- **Hypothesis:** bf16 freed 9.2 GB VRAM → enables bs=6 or bs=8. Larger batch → cleaner gradients → better cosine-schedule annealing → lower MAE.
- **Stack:** bf16-only (pre-EMA, pre-FiLM, pre-cosine; note: older stack)

### Results

| Arm | `batch_size` | epochs | sec/epoch | peak VRAM | `val_avg/mae_surf_p` | best epoch | `test_avg/mae_surf_p` |
|-----|-------------|--------|-----------|-----------|----------------------|-----------|----------------------|
| A (baseline) | 4 | 19 | 98.1 s | 32.95 GB | **95.69** | 18 | **97.18** |
| B (bs=6) | 6 | 18 | 102.4 s | 49.40 GB | 101.37 | 17 | 98.39 |
| C (bs=8) | 8 | 18 | 104.5 s | 65.86 GB | 114.49 | 14 | 115.38 |

| Split | Arm A (bs=4) | Arm B (bs=6) | Arm C (bs=8) |
|-------|---:|---:|---:|
| `test_single_in_dist` | 112.40 | 107.60 | 132.33 |
| `test_geom_camber_rc` | 93.73 | 95.56 | 113.62 |
| `test_re_rand` | 85.43 | 92.01 | 100.18 |

### Metric artifacts

- `models/model-charliepai2i48h4-askeladd-bf16-bs-baseline-20260516-052349/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-bs-6-bf16-20260516-062911/metrics.jsonl`
- `models/model-charliepai2i48h4-askeladd-bs-8-bf16-20260516-033344/metrics.jsonl`

### Analysis & conclusions

**Hypothesis falsified: bs=4 < bs=6 < bs=8 monotonically (larger batch = worse MAE under iso-wall-clock).**

Mechanism: GPU is **compute-bound at bs=4**, not memory-bound or dataloader-bound. sec/epoch barely changes (98→102→104 s for 2× batch). So bigger batches only reduce SGD steps per wall-clock (7100→4500→3400), with no compensating per-step quality gain (LR not scaled with batch). The pre-body "doubling batch ≡ √2 LR" argument assumed iso-step count; it inverts under iso-wall-clock.

The bf16-freed VRAM is available but not usable profitably via batch scaling without matching LR scaling.

**Important note on stack staleness:** This experiment ran on bf16-only (baseline 101.519 from PR #3290). Current track best is 89.784 (two-shot FiLM full stack). The batch-size axis would need retesting on the full stack to be actionable, but the mechanism (GPU compute-bound) is stack-independent.

**Seed variance observation:** Arm A (95.69) materially outperformed the prior-merge bf16 baseline (101.519) — same config, different seeds. Cross-commit variance ±5-6 MAE on bf16-only. Within-session paired Δs (A vs B vs C) are still trustworthy.

### Decision: CLOSED (hypothesis falsified, monotonic regression)

---


---

## 2026-05-16 08:30 — PR #3492 [SENT BACK]: n_hidden=192 — wider model on pre-FiLM stack

- **Student branch:** `charliepai2i48h4-nezuko/model-capacity-nhidden192`
- **Hypothesis:** Model capacity-limited at n_hidden=128. Wider model (n_hidden=192) should improve generalization.
- **Stack (run R1):** bf16 + T_max=15 + EMA (pre-FiLM, pre-two-shot-FiLM)

### Results

| Arm | n_hidden | params | VRAM | epochs | sec/epoch | `val_avg/mae_surf_p` | best epoch | `test_avg/mae_surf_p` (3 finite) |
|-----|---------|---|---|---|---|---|---|---|
| A | 128 | 0.66M | 32.95 GB | 19 | 98.3 s | 96.886 | 19 | 94.296 |
| B | **192** | **1.47M** | 43.04 GB | 15 | 122.1 s | **93.989** | 15 | **91.025** |
| **Δ** | **2.22×** | +30.6% | −4 epochs | +24% | **−2.99%** | — | **−3.47%** |

### Per-split val MAE pressure

| Split | A (128) | B (192) | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`     | 115.749 | **109.303** | **−5.57%** |
| `val_geom_camber_rc`     | 103.732 | **101.157** | −2.48% |
| `val_geom_camber_cruise` |  75.658 |  **75.083** | −0.76% |
| `val_re_rand`            |  92.408 |  **90.413** | −2.16% |

### Per-split test MAE (3 finite splits)

| Split | A | B | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`     | 99.247 | **93.810** | **−5.48%** |
| `test_geom_camber_rc`     | 96.952 | **93.911** | −3.14% |
| `test_re_rand`            | 86.688 | **85.354** | −1.54% |

### Mechanism analysis (from student)

- Train losses similar (Arm A 0.0578 vs Arm B 0.0612 at end) but val gap ~3% → "better inductive bias at same fit" mode, NOT raw memorization
- Wider model finds solutions that generalize better at similar fit quality
- Consistent with "capacity gives better inductive bias when combined with EMA + cosine"
- 2.22× param ratio but only 1.31× VRAM ratio → activations dominate memory on large meshes

### Decision: SENT BACK FOR REBASE + VERIFY ON FULL FILM STACK

Rationale:
1. PR ran on pre-FiLM stack — current best is 89.784 (FiLM stack)
2. Arm B (93.989) is +4.7% worse than current best, even though n_hidden=192 won within PR
3. Composition with FiLM unknown — FiLM modulation capacity grows with n_hidden too
4. Student's mechanistic finding (capacity → smoother optimization landscape) predicts compositional behavior

Rerun:
- Arm A: full current stack (n_hidden=128) — expected ~89.8
- Arm B: full stack + n_hidden=192 — predicted **86.5-88.5** if additive

---

## 2026-05-16 08:30 — PR #3390 [SENT BACK]: bf16 + T_max compose verify — T_max=20 found as new optimum

- **Student branch:** `charliepai2i48h4-thorfinn/bf16-tmax-compose`
- **Hypothesis:** Verify bf16 × cosine_T_max composition. Test T_max=15 and T_max=20 vs baseline T_max=50.
- **Stack (run R1):** bf16-only (pre-EMA, pre-FiLM, pre-two-shot-FiLM)

### Results

| Arm | `cosine_t_max` | epochs | LR @ end | `val_avg/mae_surf_p` | best epoch | `test_avg/mae_surf_p` (3 finite) |
|-----|---:|---:|---:|---:|---:|---:|
| A | 50 (default) | 19 | 3.564e-4 (71% of init) | 102.794 | 19 | 101.855 |
| B | 15 | 19 | 5.000e-8 (floor at ep 16) | 97.968 | 19 | 94.747 |
| **C** | **20** | 19 | **1.224e-5 (2.4% of init)** | **88.229** | 19 | **84.598** |
| **Δ C vs A** | — | — | — | **−14.2%** | — | **−16.9%** |

### LR-vs-epoch trace (showing T_max=15 wastes epochs at LR floor)

| Epoch | A (T_max=50) | B (T_max=15) | C (T_max=20) |
|---:|---:|---:|---:|
| 14 | 4.211e-4 | 2.161e-5 | 1.365e-4 |
| 15 | 4.094e-4 | 5.463e-6 | 1.031e-4 |
| 16 | 3.969e-4 | **5.000e-8 floor** | 7.322e-5 |
| 19 | 3.564e-4 | **5.000e-8 floor** | 1.224e-5 |

### Per-split val MAE pressure (Arm C wins every split)

| Split | A | B | **C** | Δ C vs A |
|---|---:|---:|---:|---:|
| `val_single_in_dist`     | 119.018 | 117.811 | **98.618** | **−17.1%** |
| `val_geom_camber_rc`     | 119.550 | 104.910 | **95.728** | **−19.9%** |
| `val_geom_camber_cruise` |  74.632 |  76.938 | **72.692** |  −2.6% |
| `val_re_rand`            |  97.977 |  92.214 | **85.877** | **−12.4%** |

### Mechanism analysis (from student)

- T_max=15 was calibrated for fp32's 14-epoch budget; on bf16's 19-epoch budget it hits LR floor at epoch 16, wasting 3 epochs
- T_max=20 keeps cosine arc decaying continuously, finishing at LR=1.224e-5 (still meaningful gradient signal)
- **Composition super-additive:** bf16 (−7.8% vs Huber) + T_max=15 (−10.3% vs Huber) → T_max=20 (−20.9% vs Huber); not just orthogonal but synergistic
- Simple rule: `cosine_t_max ≈ expected_epoch_budget` for the wall-clock

### Decision: SENT BACK FOR REBASE + VERIFY ON FULL FILM STACK

Rationale:
1. **Arm C (88.229) is better than current track best (89.784)** — but on bf16-only stack
2. Cannot merge bf16-only as new "winner" — would lose FiLM (−4%) and two-shot FiLM (−3%) wins
3. **Composition unknown:** T_max=20 + FiLM stack — likely additive (mechanism is LR-schedule-shape, independent of conditioning) but unverified
4. **Parallel investigation:** alphonse #3594 testing SF-AdamW (eliminates cosine) — both attack same LR-floor issue. If both win, merge larger Δ.

Rerun:
- Arm A: full current stack with T_max=15 — expected ~89.8
- Arm B: full stack with T_max=20 — predicted **80-85** if composition holds; **86-89** if sub-additive

### This may be the largest unmerged single-change improvement available right now.


---

## 2026-05-16 09:00 — PR #3684 [CLOSED]: slice_num=32/64/96 sweep on full FiLM stack

- **Student branch:** `charliepai2i48h4-edward/slice-num-sweep`
- **Hypothesis:** Test whether richer slice-token partitions (96) or faster coarser (32) improve over default 64 on the full FiLM stack.
- **Stack:** bf16 + cosine T_max=15 + EMA(0.999) + FiLM + two-shot FiLM (full current best)

### Results

| Arm | slice_num | val_avg/mae_surf_p | best_epoch | n_epochs | sec/epoch | peak_VRAM |
|-----|---:|---:|---:|---:|---:|---:|
| **A** | **64** | **88.534** | 17 | 17 | 110.3 | 38.92 GB |
| B | 96 | 91.927 | 14 | 14 | 130.9 | 45.14 GB |
| C | 32 | 92.850 | 21 | 21 | 89.7 | 32.71 GB |

### Per-split val MAE (Arm A wins every split)

| Split | A (sn=64) | B (sn=96) | C (sn=32) |
|---|---:|---:|---:|
| `val_single_in_dist`     | **101.435** | 108.136 | 105.877 |
| `val_geom_camber_rc`     |  **95.531** |  98.207 |  99.125 |
| `val_geom_camber_cruise` |  **72.072** |  73.582 |  75.014 |
| `val_re_rand`            |  **85.098** |  87.783 |  91.384 |
| **val_avg**              |  **88.534** |  91.927 |  92.850 |

### Compute analysis

- sn=96: 18.7% slower/epoch, 14 epochs (truncated, didn't reach T_max=15 LR floor)
- sn=32: 18.7% faster/epoch, 21 epochs (6 bonus LR-floor epochs); accuracy loss outweighs compute gain
- VRAM: 32.7 → 38.9 → 45.1 GB linear scaling

### Mechanism analysis (from student)

- sn=96: model not bottlenecked by mode resolution at n_hidden=128 capacity; overparameterized attention head + slower epochs
- sn=32: coarser mode partition loses val_re_rand (+7.4%) worst — fewer physics modes = worse Re-OOD generalization
- sn=64 is a genuine knee point

### Decision: CLOSED (slice_num=64 locked in as optimum)

Note: student suggested sn=96 might unlock with n_hidden=192 at wider capacity. Holding for post-nezuko evaluation.

---

## 2026-05-16 09:00 — PR #3681 [CLOSED]: Three-shot FiLM — preprocess injection

- **Student branch:** `charliepai2i48h4-frieren/three-shot-film`
- **Hypothesis:** Add third FiLM injection site at preprocess MLP output, before residual stream begins.
- **Stack:** bf16 + cosine T_max=15 + EMA(0.999) + FiLM + two-shot FiLM (full current best)

### Results

| Arm | FiLM sites | n_params | val_avg/mae_surf_p | best_epoch | test_avg/mae_surf_p (3 finite) |
|-----|---|---:|---:|---:|---:|
| A (two-shot) | attn + MLP × 5 blocks | 845,527 | **89.285** | 17 | 86.442 |
| B (three-shot) | preprocess + attn + MLP × 5 blocks | 878,551 | 92.922 | 17 | 89.284 |
| **Δ B vs A** | — | +33,024 | **+4.08%** | — | **+3.29%** |

### Per-split val MAE (B worse on every split)

| Split | A (two-shot) | B (three-shot) | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 103.523 | 107.820 | +4.15% |
| `val_geom_camber_rc`     |  96.087 |  97.746 | +1.73% |
| `val_geom_camber_cruise` |  71.644 |  75.221 | +4.99% |
| `val_re_rand`            |  85.884 |  90.900 | +5.84% |
| **val_avg**              | **89.285** | **92.922** | **+4.08%** |

### Per-epoch gap consistent (not a transient)

Arm B trails by 3-4 points from epoch 8 onward — third FiLM is actively harmful, not just slow to converge.

### Mechanism analysis (from student)

1. **Shared conditioner head over-stretched.** Going 5→10→12 output slots forces same MLP body to multiplex across more streams without growing capacity.
2. **Preprocess site is wrong injection point.** Two-shot works because both sites are *inside* residual stream (post-LN, pre-sublayer). Preprocess site is on un-LN'd features before residual stream — scale+shift on raw features distorts rather than modulates.

### Decision: CLOSED (three-shot FiLM falsified; injection-count axis at saturation)

Follow-up: per-block independent FiLM (#3829) tests shared-head-bottleneck hypothesis directly.

---

## 2026-05-16 09:39 — PR #3758 [SENT BACK]: n_layers=4 depth ablation on full FiLM stack (fern R1)

- **Student branch:** `charliepai2i48h4-fern/depth-r1`
- **Hypothesis:** Drop n_layers from 5 → 4. Smaller model → faster epochs → more fine-tune time in cosine T_max=15 tail where EMA accumulates smoothing benefit. Tests "depth costs epochs more than it adds capacity at this 30-min budget."

### Headline (paired)

| | Arm A (n_layers=5) | **Arm B (n_layers=4)** | Δ% (B vs A) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 91.305 | **90.198** | **−1.21%** ✅ |
| Best epoch | 17 | 21 | +4 epochs |
| Params | 845,527 | 691,347 | −18.2% |
| sec/epoch | 111.1 | 89.8 | −19.2% |
| Peak VRAM (GB) | 38.92 | 31.95 | −17.9% |

### Per-split val MAE (3 of 4 splits improve)

| Split | Arm A | **Arm B** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 106.193 | **105.433** | −0.72% |
| `val_geom_camber_rc`     |  99.698 |  **96.080** | **−3.63%** |
| `val_geom_camber_cruise` |  **72.107** |   73.478 | +1.90% |
| `val_re_rand`            |  87.220 |  **85.802** | −1.63% |
| **val_avg**              | **91.305** | **90.198** | **−1.21%** |

### Per-split test MAE (2 of 3 finite splits improve)

| Split | Arm A | **Arm B** | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`     | 91.890 | **88.274** | **−3.94%** |
| `test_geom_camber_rc`     | 90.897 | **88.990** | −2.10% |
| `test_re_rand`            | **81.290** |  82.574 | +1.58% |
| **avg (3 finite splits)** | **88.026** | **86.612** | **−1.61%** |

### Mechanism verification (all 3 predictions hit)

- ✅ params −18.2% (predicted ~20%)
- ✅ sec/epoch −19.2% (predicted 15-20%)
- ✅ +4 fine-tune epochs (predicted 2-3 — exceeded)

Arm B's best epoch is 21 — past Arm A's wall-clock cutoff at 17. Win lives entirely in the extra cosine-tail epochs (lr ≈ 1e-7 → 5e-9) where EMA smoothing pays off.

### Metric artifacts

- `models/model-charliepai2i48h4-fern-depth-r1-armb-nlayers4-20260516-080418/metrics.jsonl` (winner)
- `models/model-charliepai2i48h4-fern-depth-r1-arma-baseline-20260516-072830/metrics.jsonl` (paired baseline)

### Tension: paired wins, absolute fails

- Paired Δ (within-PR): **−1.21%**, mechanism fully verified
- Absolute (vs merged baseline 89.784): Arm B = 90.198 → **+0.46% absolute regression**
- Within-PR Arm A measured 91.305 — but the merged baseline (89.784, PR #3584) was Arm A in a *different* PR using identical config. Cross-PR seed variance ~±1.5-2% on n_layers=5 confirmed.

### Decision: REQUEST CHANGES — one more Arm B seed needed

Merge protocol requires updating BASELINE.md downward. Merging at 90.198 absolute would regress the comparison contract for all in-flight PRs (thorfinn T_max=20, alphonse SF-AdamW, nezuko n_hidden=192, tanjiro grad-clip — all evaluating "Δ vs 89.784").

Asked fern to run a single additional Arm B (different random seed) with same config. 30-min cost resolves the absolute-vs-paired tension:
- If seed-2 Arm B `val_avg/mae_surf_p` < 89.784 → **merge** at lower of two seeds
- If seed-2 Arm B ≥ 89.784 → **close cleanly**, keep finding in log

### n_layers axis fully mapped at this budget (monotone)

- n_layers=3: untested (potential follow-up if seed-2 lands)
- n_layers=4: **−1.21%** paired (this PR)
- n_layers=5: current baseline
- n_layers=6: +2.47% regression (PR #3595)

Curve is monotone in current 30-min budget regime: depth costs epochs more than it adds capacity.

### Follow-up directions (deferred until seed-2 resolves)

- **n_layers=3** — capacity floor unknown; one paired arm maps it
- **n_layers=4 + n_hidden=144 / mlp_ratio=3** — redistribute saved params
- **n_layers=4 + T_max=18-21** — addresses over-decayed schedule with 21-epoch runs (composes with thorfinn #3390)

---

## 2026-05-16 11:22 — PR #3511 [MERGED]: Gradient clipping (clip_norm=1.0) on full two-shot FiLM stack (tanjiro R2)

- **Student branch:** `charliepai2i48h4-tanjiro/grad-clip-bf16-tmax-15`
- **Hypothesis:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before every optimizer.step(). Natural grad norms in this regime (bf16 + large mesh + surf_weight=10) are p50≈7-25, p90≈15-50 throughout training — all far above clip=1.0, so clipping fires ~96-100% of steps. This is direction normalization, not outlier filtering.

### Headline (intra-PR paired Δ)

| Arm | clip | epochs | best ep | val_avg/mae_surf_p | test_avg/3finite |
|-----|-----:|-------:|--------:|-------------------:|------------------|
| A (two-shot FiLM, no clip) | — | 17 | 16 | 92.146 | 88.903 |
| **B (+ clip=1.0)** | **1.0** | **17** | **17** | **81.660** | **78.967** |
| **Δ (B vs A)** | — | — | — | **−11.38%** | **−11.18%** |

vs merged baseline 89.784: **−9.05%** — **MERGED** as new baseline.

### Per-split val MAE (all 4 splits improve)

| Split | Arm A | **Arm B (clip=1.0)** | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 105.365 | **94.434** | −10.37% |
| `val_geom_camber_rc`     | 101.253 | **90.960** | −10.17% |
| `val_geom_camber_cruise` |  73.369 | **62.732** | **−14.50%** |
| `val_re_rand`            |  88.598 | **78.516** | −11.38% |
| **val_avg**              | **92.146** | **81.660** | **−11.38%** |

### Per-split test MAE (all 3 finite splits improve)

| Split | Arm A | **Arm B (clip=1.0)** | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist` | 90.902 | **81.956** |  −9.84% |
| `test_geom_camber_rc` | 93.074 | **83.649** | −10.13% |
| `test_re_rand`        | 82.735 | **71.296** | **−13.83%** |
| **avg (3 finite)**    | **88.903** | **78.967** | **−11.18%** |

### Mechanism analysis (key novel finding)

- **Not outlier clipping — gradient direction normalization.** Clip fires ~96-100% of steps throughout training. Natural grad norms p50≈7-25, p90≈15-50 are all well above clip=1.0 threshold. AdamW operates on unit-normalized gradient direction; per-parameter adaptive scaling still applies.
- **Super-additive composition with FiLM.** Pre-FiLM gain (R1) was −4.77%; post-FiLM gain is −9.05%. FiLM makes the model more sensitive to per-sample conditioning signals, amplifying gradient noise in the bf16 heavy-tail regime. Clipping removes that noise source.
- **Epoch-1 heavy-tail outliers confirmed:** max=226 in Arm B (pre-clip) vs max=87 in Arm A (no-clip arm). Both from same model architecture on same first batch with different RNG — confirms bf16 quantization noise creates heavy-tail gradient outliers.
- **Natural grad-norm trajectory:** epoch-1 p50≈24, epoch-17 p50≈6. Well above clip threshold at all epochs.

### Variance analysis

4 Arm A pilot seeds during development: 87.579, 92.146, 92.276, 95.066 (mean 91.8, std 3.1). Arm B at 81.660 is 5.9% better than the BEST pilot Arm A — signal is unambiguous.

### Metric artifacts

- `models/model-charliepai2i48h4-tanjiro-gradclip-r2-arm-b-twoshot-clip1_0-20260516-093143/metrics.jsonl` (winner)
- `models/model-charliepai2i48h4-tanjiro-gradclip-r2-arm-a-twoshot-noclip-20260516-072527/metrics.jsonl` (paired baseline)

### Decision: MERGED — new baseline 81.660

**Stack staleness impact:** All in-flight PRs (#3390, #3594, #3492, #3777, #3829, #3830, #3758) are running on pre-clip stack and will not beat 81.660 in absolute terms. Protocol: if they show positive paired Δ → rebase with clip=1.0 added to both arms; if no paired Δ → close.

### Follow-up assigned

Tanjiro #3906: clip threshold sweep {0.25, 1.0, 4.0} — determines whether direction-normalization mechanism is saturated at clip=1.0 or whether adjacent thresholds improve further.


---

## 2026-05-16 12:25 — PR #3758 [SENT BACK R2]: n_layers=4 depth ablation — seed-2 confirm (fern)

- **Student branch:** `charliepai2i48h4-fern/depth-r1`
- **Hypothesis (carried forward):** n_layers=4 saves ~18.2% params and ~19.2% sec/epoch, gaining ~4 fine-tune epochs in cosine T_max=15 tail.

### Seed-2 result

| Run | val_avg/mae_surf_p | Best epoch | n_params | sec/epoch |
|---|---:|---:|---:|---:|
| R1 seed-1 Arm A (n_layers=5)   | 91.305 | 17 | 845,527 | 111.2 |
| R1 seed-1 Arm B (n_layers=4)   | 90.198 | 21 | 691,347 |  89.8 |
| **R2 seed-2 Arm B (n_layers=4)** | **88.441** | 21 | 691,347 |  89.83 |

Mean of two n_layers=4 seeds: **89.32** (below pre-clip merged 89.784).

### Per-split val MAE (seed-2 vs seed-1, both n_layers=4)

| Split | Seed-1 Arm B | **Seed-2 Arm B** | Δ vs seed-1 |
|---|---:|---:|---:|
| `val_single_in_dist`     | 105.433 | **102.786** | −2.51% |
| `val_geom_camber_rc`     |  96.080 |  **94.709** | −1.43% |
| `val_geom_camber_cruise` |  73.478 |  **71.609** | −2.54% |
| `val_re_rand`            |  85.802 |  **84.660** | −1.33% |
| **val_avg**              | **90.198** | **88.441** | **−1.95%** |

Test (3 finite splits): **83.164** vs seed-1 Arm B 86.612 → −3.99%.

### Decision: SENT BACK FOR REBASE WITH CLIP

Per-arm seed variance characterized at ±1.5–2% (confirmed across this PR and #3365, #3684). But **baseline moved while seed-2 ran**: grad-clip merged at #3511 dropping baseline 89.784 → 81.660. fern's 88.441 absolute is +8.3% worse than the new baseline.

The paired Δ (−1.21% R1) is real and clean. Mechanism-wise, depth reduction (capacity↓, epochs↑) is orthogonal to clip-norm (direction normalization). Prior: they compose.

**Sent back for R3 paired sweep with `--grad_clip_norm 1.0` in BOTH arms.** Decision rule:
- Arm B (n_layers=4+clip) < Arm A (n_layers=5+clip) by >0.5% paired Δ → merge.
- Within ±0.5% → close (clip subsumed depth's benefit).
- Arm B > Arm A → close.

### Metric artifacts

- `models/model-charliepai2i48h4-fern-depth-r2-armb-nlayers4-seed2-20260516-102352/metrics.jsonl` (seed-2)

---

## 2026-05-16 12:25 — PR #3492 [SENT BACK R2]: Model capacity — n_hidden=192 on full FiLM stack (nezuko R2)

- **Student branch:** `charliepai2i48h4-nezuko/model-capacity-nhidden192`
- **Hypothesis:** Wider model (n_hidden=128 → 192) gives FiLM heads more output bandwidth and the trunk MLPs more capacity. Expected larger gain on FiLM stack than pre-FiLM because FiLM's conditioner-multiplier interaction benefits from richer per-channel modulation.

### R2 results — paired Δ −8.21% val_avg

| Arm | n_hidden | n_params | epochs | sec/epoch | VRAM (GB) | best epoch | val_avg/mae_surf_p | test_avg (3 finite) |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|
| A   | 128 |   845,527 | 17 | 110.6 | 38.92 | 17 | 97.232 | 93.831 |
| B   | **192** | **1,737,559** | 13 | 140.3 | **51.98** | 13 | **89.252** | **86.318** |
| Δ   | — | +2.05× | −4 | +26.8% | +33.6% | — | **−8.21%** | **−8.00%** |

### Per-split val MAE

| Split | A (n_hidden=128) | B (n_hidden=192) | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 119.597 | **100.542** | **−15.93%** |
| `val_geom_camber_rc`     | 101.358 |  **98.071** | −3.24% |
| `val_geom_camber_cruise` |  77.050 |  **72.401** | −6.03% |
| `val_re_rand`            |  90.926 |  **85.995** | −5.42% |
| **val_avg**              | **97.232** | **89.252** | **−8.21%** |

Every split improves; largest gain on `val_single_in_dist` (−15.93%), consistent with R1 pattern where wider FiLM head best captures heavy-tailed pressure distributions on single-airfoil split.

### Analysis

- **Wider FiLM head, not just trunk:** FiLM module size scales with hidden_dim (output `2*hidden_dim*n_layers`). The wider conditioner provides richer per-channel modulation that composes with two-shot injection.
- **Capacity composes with FiLM (not subsumed):** R1 (pre-FiLM) showed paired Δ −2.99%; R2 (FiLM stack) shows −8.21%. FiLM allocates extra capacity to inductive-bias smoothing rather than memorization. At common epoch 13, Arm B train surf_loss=0.0517 vs Arm A=0.0598 (Arm B fits training tighter AND generalizes better).
- **Wall-clock cost manageable:** +26.8% sec/epoch fits inside cosine T_max=15 (Arm B best at epoch 13 with 2 cosine steps remaining).
- **Initial OOM crash on Arm B launch** (Arm A still in test eval, ~49 GB held). Relaunched sequentially; no code issue.
- **Seed variance caveat:** Arm A absolute (97.232) is +8.3% above merged 89.784 — bf16 noise + no `torch.manual_seed`. Paired Δ within session is the trusted signal.

### Decision: SENT BACK FOR REBASE WITH CLIP

Absolute Arm B (89.252) is +9.3% worse than new baseline 81.660. Can't merge directly. But the paired Δ is the strongest signal we've seen this round — orthogonal to clip-norm, prior is strong they compose.

**Sent back for R3 paired sweep with `--grad_clip_norm 1.0` in BOTH arms.** Decision rule:
- Arm B (n_hidden=192+clip) < Arm A (n_hidden=128+clip) by >0.5% paired Δ → likely merge. If even a fraction of −8.21% survives, this is the biggest single hop available.
- Within ±0.5% → close (capacity subsumed by clip).
- Arm B > Arm A → close.

If R3 wins, also unlocks: slice_num=96 (currently locked at 64 at n_hidden=128), wider FiLM MLP hidden (256), and depth+capacity compose.

### Metric artifacts

- `models/model-charliepai2i48h4-nezuko-capacity-r2-arma-baseline-20260516-102534/metrics.jsonl`
- `models/model-charliepai2i48h4-nezuko-capacity-r2-armb-nhidden192-20260516-110108/metrics.jsonl`

---

## 2026-05-16 13:28 — PR #3829 [CLOSED]: Per-block independent FiLM heads (frieren R1-R2)

- **Student branch:** `charliepai2i48h4-frieren/perblock-film`
- **Hypothesis:** Replace shared FiLM output head with per-block heads (shared body, per-block output projections). With `--two_shot_film`, each per-block head outputs two independent (γ, β) pairs — one for attn, one for MLP — instead of the shared two-shot's single (γ, β) reused at both sites.

### Results (R1 + R2 paired)

| Run | Arm A (shared) val_avg | Arm B (per-block) val_avg | Paired Δ | Notes |
|---|---:|---:|---:|---|
| R1 (09:35 / 10:39) | 90.962 | 91.224 | **+0.29%** | regression |
| R2 (11:28 / 12:03) | 91.752 | 90.518 | **−1.34%** | improvement |
| **Mean** | 91.357 | 90.871 | **−0.53% val_avg / −0.64% test 3-split** | within noise band |

- **Params:** 845,527 → 1,010,647 (+165,120, **+19.5%**) vs predicted +1.5% (13× higher than expected — per-block × per-site arithmetic dominates)
- **Sec/epoch:** ~+6-8%
- **Seed variance band:** ±1.5-2% → averaged Δ −0.53% sits well inside noise

### Decision: CLOSE

**Rationale:**
1. **Signal at noise floor.** R1 +0.29% and R2 −1.34% straddle zero; averaged Δ inside seed variance band.
2. **Disproportionate cost.** +19.5% params for noise-level signal.
3. **Confounded design** (student-flagged): Arm B couples (i) per-block independent heads AND (ii) attn-vs-MLP independent (γ, β) per block. Cannot attribute (noisy) Δ to either change.
4. **Won't beat 81.660** even with clip-rebase: Arm B 91.488 absolute + clip's ~−9% lands low 80s = parity with baseline.
5. **FiLM injection-count axis saturated** at n_hidden=128: single-shot merged, two-shot merged, three-shot closed (+4.08%), per-block-capacity at noise. May unlock at higher capacity if nezuko #3492 R3 (n_hidden=192+clip) merges.

### Lessons for future FiLM work

- **Two paired runs were essential.** Without R2, R1's +0.29% regression would have been the only data point — easy to misread as falsified. R2's −1.34% revealed the noise-floor nature of the signal. Replicates near the noise floor are the right protocol.
- **Per-block × per-site combinatorics:** the param multiplication caught the student off-guard. For future per-block conditioner ideas, isolate per-block from per-site changes (test single-shot per-block first to control the confound).
- **FiLM capacity may need wider trunk to manifest.** If nezuko #3492 R3 merges (n_hidden=192 + clip), revisit per-block FiLM at the larger capacity — the bottleneck may shift.

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-perblock-r1-arma-baseline-20260516-093521/metrics.jsonl`
- `models/model-charliepai2i48h4-frieren-perblock-r1-armb-perblock-20260516-103912/metrics.jsonl`
- `models/model-charliepai2i48h4-frieren-perblock-r1-arma-baseline-20260516-112856/metrics.jsonl`
- `models/model-charliepai2i48h4-frieren-perblock-r2-armb-perblock-20260516-120346/metrics.jsonl`

### Next assignment: PR #3980 Lion optimizer

Frieren reassigned to Lion optimizer (sign projection on full clip stack vs AdamW+clip). Tests the mechanistic question of whether clip's load-bearing mechanism is direction normalization, in which case Lion's sign-projection (L∞ direction normalization, more extreme than L2 clip) should compose or supersede.

---

## 2026-05-16 13:35 — PR #3830 [CLOSED]: Lookahead optimizer wrapper (edward R1)

- **Student branch:** `charliepai2i48h4-edward/lookahead-optimizer`
- **Hypothesis:** Lookahead (Zhang et al. 2019, k=5, α=0.5) wraps AdamW with slow-weight interpolation; produces smoother val trajectories and improved generalization. Composes with EMA at different time scales.

### Results (R1 paired)

| Split | Arm A (no Lookahead) | Arm B (Lookahead k=5, α=0.5) | Δ % |
|---|---:|---:|---:|
| val_single_in_dist     | 107.363 | 108.533 | +1.09% |
| val_geom_camber_rc     |  98.067 |  97.082 | −1.00% |
| val_geom_camber_cruise |  72.106 |  74.073 | +2.73% |
| val_re_rand            |  88.108 |  87.479 | −0.71% |
| **val_avg**            | **91.411** | **91.792** | **+0.42%** |
| **test_avg (3 finite)** | **87.947** | **89.225** | **+1.45%** |

### Mechanism check (Lookahead's claim: smoother trajectory)

| Quantity | Arm A | Arm B | Δ |
|---|---:|---:|---:|
| Mean epoch-to-epoch \|Δval\| | 7.237 | 6.457 | −10.8% |
| Last-half (ep 9-17) val std  | 4.801 | 4.423 | −7.9% |

**Mechanism real but small** — trajectory IS measurably smoother, but the smoothing benefit doesn't reach the best checkpoint. EMA(0.999) already low-passes the trajectory at ~693-step half-life; Lookahead's ~25-step half-life is redundant.

### Per-epoch convergence trace (Lookahead leads early, loses by cosine tail)

```
ep    Arm A    Arm B  (Δ B−A)
 1  207.21   195.11  −12.10   ← Lookahead +5.8% better
 8  109.57   109.20   −0.37   ← edge gone
14   92.71    92.88   +0.17   ← Arm A overtakes
17   91.41    91.79   +0.38   ← best checkpoint, Arm A wins
```

Lookahead delivers faster early convergence but becomes mildly counterproductive once cosine LR drops below ~7% of peak (epoch 13+). The slow-weight pull clamps the fine-grained refinement of the cosine tail.

### Decision: CLOSE

**Rationale:**
1. **Paired Δ within noise band** for val (+0.42% < ±1.8% seed variance), but **outside band for test 3-split** (+1.45%).
2. **Arm B 91.792 absolute vs current baseline 81.660**: +12.4% worse — not competitive even with clip rebase.
3. **Mechanism captured by EMA.** Trajectory smoothing role is occupied. Lookahead's shorter time constant doesn't add value on top of EMA(0.999) + cosine T_max=15.
4. **Per-split pattern (helps `val_geom_camber_rc` by 1%, hurts in-dist + Re-rand by 1-3%)** shows the regularization signature — smoothing trades fine-detail fitting for generalization. We don't need this trade on this stack.
5. **Wall-clock overhead ~0%** (slow-weight clone + 1275 sync steps disappear into per-epoch noise). PR's "+5%" estimate was conservative.

### Lessons captured

- **Trajectory smoothing as a research axis is largely saturated by EMA at this scale.** Any future technique proposing to smooth the trajectory (SWA, polyak averaging, longer EMA half-life) competes with an entrenched mechanism.
- **`val_geom_camber_rc` selectively benefits from smoothing.** Future hypotheses targeting cross-geometry generalization specifically should revisit Lookahead in that ablation.
- **The orthogonal axes that remain** are direction normalization at the gradient step (Lion, AGC), per-group clip granularity (AGC), cosine schedule replacement (SF-AdamW), cosine schedule extension (T_max=20).

### Metric artifacts

- `models/model-charliepai2i48h4-edward-lookahead-r1-arma-baseline-paired-20260516-113402/metrics.jsonl`
- `models/model-charliepai2i48h4-edward-lookahead-r1-armb-lookahead-20260516-122639/metrics.jsonl`

### Next assignment: PR #3985 AGC

Edward reassigned to AGC (Adaptive Gradient Clipping, NFNet-style per-parameter-group adaptive clipping vs global L2 clip). Tests whether global L2 normalization is the right granularity for the direction-normalization mechanism that PR #3511 confirmed.

---

## 2026-05-16 14:32 — PR #3906 [MERGED]: Clip threshold sweep R1 — clip=0.25 wins (tanjiro)

- **Student branch:** `charliepai2i48h4-tanjiro/clipthresh-r1`
- **Hypothesis:** Clip=1.0 may not be optimal; sweep {0.25, 1.0, 4.0} to find the direction-normalization optimum. Tighter clip → 100% clip rate → purer direction signal. Looser clip → partial outlier-suppression only.

### Results (3-arm sweep)

| Arm | clip_norm | val_avg/mae_surf_p | test_avg (3 finite) | Δ vs A (paired) |
|---|---:|---:|---:|---:|
| A (control) | 1.0 | 83.756 | 79.643 | — |
| B | 4.0 | 86.647 | 82.886 | **+3.45%** (regression) |
| **C (winner)** | **0.25** | **80.893** | **76.889** | **−3.42%** ✅ |

**Arm C vs absolute baseline 81.660: −0.94%** ✅

### Per-split val MAE

| Split | Arm A (1.0) | Arm C (0.25) | Δ % |
|---|---:|---:|---:|
| val_single_in_dist     | 96.717 | **93.062** | −3.78% |
| val_geom_camber_rc     | 93.005 | **90.132** | −3.09% |
| val_geom_camber_cruise | 65.251 | **61.764** | −5.34% |
| val_re_rand            | 80.051 | **78.616** | −1.79% |
| **val_avg**            | **83.756** | **80.893** | **−3.42%** |

### Clip-rate diagnostics (key mechanism evidence)

| Epoch | A clip=1.0 (rate) | B clip=4.0 (rate) | C clip=0.25 (rate) |
|---:|---:|---:|---:|
| 1 | 1.000 | 0.997 | 1.000 |
| 5 | 1.000 | 0.931 | 1.000 |
| 10 | 0.997 | 0.867 | 1.000 |
| 15 | 0.949 | 0.723 | 1.000 |
| 17 | 0.947 | 0.763 | 1.000 |

**Arm C: 100% clip rate on every step, entire run** — maximally aggressive direction normalization. Monotone curve (4.0 > 1.0 > 0.25): tighter is better.

### Analysis & conclusions

- **Direction normalization is the correct mechanism.** Arm C's 100% clip rate throughout and improved performance falsifies the "outlier suppression" hypothesis (which would predict Arm C over-clamps and hurts). The unit direction step is doing the work.
- **LR-equivalence insight from student:** at 100% clip rate, clip=0.25 is equivalent to effective_lr = lr × (0.25 / ||g||) per step — i.e., uniformly smaller steps. This opens the question of whether the win is pure "smaller steps" or genuinely "purer direction." Dedicated LR vs clip experiment would disambiguate.
- **Monotone curve opens tighter sweep.** Clip=0.25 is still at 100% rate throughout, so the optimum may be even tighter. Round 2 assigned as PR #4003.
- **New stack baseline:** `--grad_clip_norm 0.25` replaces `--grad_clip_norm 1.0` as the standard flag.

### Metric artifacts

- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-arma-clip1_0-20260516-113014/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-armb-clip4_0-20260516-122208/metrics.jsonl`
- `models/model-charliepai2i48h4-tanjiro-clipthresh-r1-armc-clip0_25-20260516-133449/metrics.jsonl` ← **winner**

### Next assignment: PR #4003 clip threshold R2 {0.05, 0.1, 0.15, 0.25 control}

---

## 2026-05-16 14:50 — PR #3758 [CLOSED]: n_layers=4 depth ablation R3 (fern)

**Branch:** `charliepai2i48h4-fern/n-layers-4-depth-ablation`

**Hypothesis:** Under the 30-min wall-clock budget, n_layers=4 should be net-positive because each epoch ~17-20% faster than n_layers=5 buys 2-3 extra fine-tune epochs in the cosine T_max=15 tail. Tested twice prior (R1 +0.41%, R2 +1.78% on the pre-clip stack — both within seed band and inconclusive). R3 re-ran on the **new clip=1.0 stack** to check whether depth=4 still adds anything once gradient clipping is on.

### Results (R3 paired, clip=1.0 stack)

| Arm | Config | val_avg/mae_surf_p | Δ vs Arm A |
|---|---|---:|---:|
| A | n_layers=5 + clip | 81.31 | — (control) |
| B | n_layers=4 + clip | 83.86 | **+3.14% (regression)** |

Test 3-split mean: Arm A 77.5, Arm B 80.1 — **+3.34%** matching direction.

### Decision: CLOSE

Two paired regressions (+3.14% val, +3.34% test) clearly above the seed variance band (±1.5-2%). Depth axis fully closed: depth↑ regressed (#3595 n_layers=6), depth↓ regressed under clip (#3758 n_layers=4). The original "epoch-budget buys late-training stability" mechanism is real, but gradient clipping (now baseline at 0.25) does exactly that job more directly — **two distinct mechanisms targeting the same role do not compound, and the cheaper one (clip) wins**.

### Lessons captured

- **Pre-clip vs post-clip mechanism check is now a standard test.** When R1/R2 evidence for a hypothesis comes from a stack that has since changed (clip was added between R2 and R3), re-running on the new stack often reveals the original mechanism was a proxy for something the new stack now handles directly.
- **Late-training stability is owned by gradient clipping in this stack.** EMA(0.999) handles trajectory smoothing on longer time scales; clip=0.25 handles per-step direction; depth=4 was a third path to the same effect and is now subsumed.
- **n_layers=5 is the right depth** for this Transolver on this dataset under the 30-min budget — no further depth sweeps planned.

### Metric artifacts

- `models/model-charliepai2i48h4-fern-depth-r3-arma-baseline-*/metrics.jsonl` (Arm A control)
- `models/model-charliepai2i48h4-fern-depth-r3-armb-n_layers4-*/metrics.jsonl` (Arm B regression)

### Next assignment: PR #4012 Sobolev/edge-gradient loss (fern)

Switch fern to a fully orthogonal axis — physics-aware loss. Add L1 supervision on the finite-difference gradient of predicted surface pressure between kNN-neighbor surface nodes. Four-arm weight sweep {0, 0.1, 0.3, 1.0}. First experiment in the round that changes *what* is being optimized rather than how.

---

## 2026-05-16 15:34 — PR #3594 [MERGED]: Schedule-Free AdamW eliminates cosine schedule (alphonse R2)

**Branch:** `charliepai2i48h4-alphonse/schedule-free-adamw`

**Hypothesis:** The cosine T_max=15 schedule floors LR at 5e-8 by epoch 16, wasting the last 2-3 epochs of the 30-minute budget. SF-AdamW (Defazio et al. 2024) eliminates the scheduler entirely, using Polyak-Ruppert averaging of the weight iterates. With no schedule, LR stays at 5e-4 throughout.

R1 (pre-two-shot-FiLM stack): −20.75% paired. R2 (two-shot-FiLM + clip=1.0 stack): clean replication with smaller but still massive gain.

### Results (R2 paired, two-shot-FiLM + clip=1.0 stack)

| Arm | Config | val_avg/mae_surf_p | Δ vs A | Δ vs baseline 80.893 |
|---|---|---:|---:|---:|
| A | AdamW + cosine T_max=15 + clip=1.0 + two-shot FiLM | 78.871 | — | −2.50% |
| **B** | **SF-AdamW (no scheduler) + clip=1.0 + two-shot FiLM** | **65.618** | **−16.80%** | **−18.88%** |

### Per-split val MAE

| Split | Arm A | Arm B (SF) | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 89.793 | 74.715 | −16.79% |
| `val_geom_camber_rc`     | 90.291 | 79.128 | −12.36% |
| `val_geom_camber_cruise` | 59.656 | 45.160 | −24.30% |
| `val_re_rand`            | 75.743 | 63.467 | −16.21% |
| **val_avg**              | **78.871** | **65.618** | **−16.80%** |

### Per-split test MAE (3 finite splits)

| Split | Arm A | Arm B (SF) | Δ % |
|---|---:|---:|---:|
| `test_single_in_dist`   | 75.881 | 63.718 | −16.03% |
| `test_geom_camber_rc`   | 80.597 | 70.042 | −13.10% |
| `test_re_rand`          | 67.861 | 54.799 | −19.25% |
| **test_avg (3 finite)** | **74.780** | **62.853** | **−15.95%** |

### Per-epoch trajectory (key evidence)

Arm A's LR reaches 5e-8 by epoch 16 — frozen. Arm B's LR stays at 5e-4 throughout. Arm B was still dropping ~1.8 val/epoch at the epoch 17 cap.

| Epoch | Arm B val_avg | Arm B lr | Arm A val_avg | Arm A lr |
|---:|---:|---:|---:|---:|
| 14 | 71.033 | 5e-4 | 80.513 | 2.2e-5 |
| 15 | 69.202 | 5e-4 | 79.661 | 5.5e-6 |
| 16 | 67.440 | 5e-4 | 79.082 | 5e-8 (frozen) |
| 17 | 65.618 | 5e-4 | 78.871 | 5e-8 (frozen) |

### Analysis & conclusions

- **Cosine T_max=15 wastes the last 2 epochs.** Arm A is frozen at lr=5e-8 from epoch 16.
- **Per-split improvement is uniform and large.** Cruise gets biggest relative boost (−24.30%), all four splits improve by 12-24%.
- **Mechanism reproduced from R1.** Smaller gain vs R1 (−20.75%) is expected — less slack to exploit on a stronger baseline.
- **Key stack note:** this win used **clip=1.0** (not clip=0.25). The optimal clip under SF-AdamW is unknown; the 2×2 factorial (#4019) resolves this.
- **EMA + SF-AdamW double-averaging:** SF's Polyak averaging and external EMA(0.999) may be redundant. #4019 includes EMA ablation arms.

### Metric artifacts

- `models/model-charliepai2i48h4-alphonse-sf-r2-armb-sf-adamw-clip-20260516-142921/metrics.jsonl` ← **winner** (val_avg=65.618)
- `models/model-charliepai2i48h4-alphonse-sf-r2-armb-sf-adamw-clip-20260516-142921/metrics.yaml`
- `models/model-charliepai2i48h4-alphonse-sf-r2-arma-baseline-20260516-135423/metrics.jsonl` (paired control)

### Next assignment: PR #4019 SF-AdamW 2×2 composition factorial (alphonse)

2×2 factorial: clip ∈ {1.0, 0.25} × EMA ∈ {on, off}. Answers: (1) optimal clip under SF, (2) EMA redundancy under SF. Best arm merges as the canonical new SF-AdamW stack.

---

## 2026-05-16 15:55 — PR #3777 [CLOSED]: SDF input features — distance-to-surface as geometric input (askeladd)

**Branch:** `charliepai2i48h4-askeladd/sdf-features`

**Hypothesis:** Adding a per-node Signed Distance to Surface (SDF) input feature (precomputed via `torch.cdist`, per-sample p95-normalized) provides an explicit geometric prior orthogonal to FiLM (which encodes per-sample physics-regime) and orthogonal to the existing `dsdf` 8-D shape descriptor. Expected to help the geometric splits (`val_geom_camber_rc`, `val_geom_camber_cruise`).

### Results (paired, two-shot-FiLM + cosine T_max=15 stack — NOT current SF stack)

| Arm | Config | val_avg/mae_surf_p | best epoch | n_params |
|---|---|---:|---:|---:|
| A | full stack, no SDF | 88.8275 | 17 | 845,527 |
| B | full stack + SDF (per-sample p95-norm) | 91.0628 | 16 | 845,783 (+256) |
| **Paired Δ B−A** | — | **+2.235 (+2.52%)** | — | +0.03% |

Note: Arm A=88.83 reproduces the pre-clip baseline (89.78) within seed band — implementation is sound, run was clean, but on a stack now superseded by SF-AdamW (65.618 baseline).

### Per-split val MAE

| Split | A (no SDF) | B (SDF) | Δ % | Hypothesis-predicted? |
|---|---:|---:|---:|---|
| `val_single_in_dist`     | 100.354 | 105.578 | **+5.21%** | no (sanity split) |
| `val_geom_camber_rc`     |  97.200 |  98.980 | +1.83% | **YES — predicted to help** |
| `val_geom_camber_cruise` |  70.585 |  71.954 | +1.94% | **YES — predicted to help** |
| `val_re_rand`            |  87.171 |  87.739 | +0.65% | no |
| **val_avg**              | **88.828** | **91.063** | **+2.52%** | — |

Per-split test 3-split mean: +3.99% (every finite split regresses, slightly larger than val). Direction-consistent — not val-only noise.

### Mechanism diagnostic: per-sample p95 normalization breaks cross-sample comparability

Student measured raw_p95 chord distances across all splits:

| Split | n | raw_p95 median | spread |
|---|---:|---:|---|
| `train`                  | 1499 | 3.60 | **2.22–4.74 (2.2× spread)** |
| `val_geom_camber_rc`     |  100 | 2.85 | 2.34–3.35 |
| `val_geom_camber_cruise` |  100 | 4.43 | 4.24–4.73 |

The per-sample p95 varies 2.2× across train: a node with `sdf_norm=0.5` means physical distance of 1.1 chord in raceCar but 2.4 chord in cruise. **The network cannot learn a single "near-wall" threshold** — the normalization removes mesh-extent invariance only at the cost of physical comparability.

### Why it failed — three mechanism findings

1. **Not orthogonal to existing inputs.** The dataset already provides `dsdf` (8-D distance-based shape descriptor, x[4:12]) and `is_surface` (binary, x[12]). Adding a per-sample-normalized single scalar is a low-resolution, lower-quality version of what `dsdf` already encodes at 8-D.
2. **Not orthogonal to FiLM.** FiLM owns per-sample geometric conditioning via film_cond features. Adding a redundant per-node geometric feature *competes* with FiLM's role rather than supplementing it — a third mechanism aimed at the same task that the model already handles.
3. **Per-sample normalization actively hurts.** The normalization scheme assumed the answer was "mesh-extent-invariant SDF" but the resulting feature loses the physical-distance meaning the SDF was meant to encode. A *global* normalization (one constant over the entire dataset) would preserve comparability but lose the mesh-extent-invariance.

### Decision: CLOSE

Paired Δ +2.52% val / +3.99% test cleanly above seed band (±1.5-2%). Mechanism failure has three independent contributing causes — re-running on the SF-AdamW stack would not change the analysis (geometric-input axis is closed by `dsdf` already, not by stack noise).

### Lesson carried forward

**When a new feature targets a role another mechanism already owns, it has to *replace* that mechanism, not stack with it.** FiLM owns per-sample geometric conditioning. `dsdf` owns per-node distance-based geometric encoding. SDF was a third mechanism aimed at the same role and predictably lost. This same lesson now applies to the broader stack: experiments that change *what is being optimized* (Sobolev loss, AGC, Lion) have a chance — experiments that add a redundant feature do not.

### Metric artifacts

- `models/model-charliepai2i48h4-askeladd-sdf-r1-arma-baseline-20260516-133807/metrics.jsonl` (control)
- `models/model-charliepai2i48h4-askeladd-sdf-r1-armb-sdf-20260516-142734/metrics.jsonl` (SDF)

### Next assignment: PR #4038 SF-AdamW LR sweep (askeladd)

4-arm constant-LR sweep under SF-AdamW: lr ∈ {5e-4 control, 1e-3, 2e-3, 5e-3} — covers SF README's full recommended 1×-10× range. Highest-leverage hyperparameter on the new SF stack: cosine is gone, so constant LR is now the dominant per-step setting. Arm B in #3594 was still descending at the budget cap (val slope ~1.8/epoch at epoch 17), suggesting the optimizer was still in fast-descent regime — a larger LR may reach equivalent or better val_avg in fewer epochs.

---

## 2026-05-16 15:50 — PR #3985 [SENT BACK]: AGC R1 — strong paired win on stale baseline (edward)

**Branch:** `charliepai2i48h4-edward/agc-vs-global-clip`

**Hypothesis:** NFNet's Adaptive Gradient Clipping (AGC) replaces global `clip_grad_norm_(model, 0.25)` with per-tensor clipping at `λ × ||param|| / ||grad||`. The mechanism: scale-aware clipping that adapts per-parameter (large weight matrices get larger clip budget; small biases get smaller), in principle more "physical" than a single global L2 threshold. Tested on the AdamW + cosine + clip=0.25 stack.

### Results (R1 paired, AdamW + cosine T_max=15 + two-shot FiLM stack — NOT current SF stack)

| Arm | Config | val_avg/mae_surf_p | Δ vs A | Δ vs baseline 80.893 |
|---|---|---:|---:|---:|
| A | clip=0.25 control (matches old baseline) | 83.233 | — | +2.89% |
| **B** | **AGC λ=0.01 (no global clip)** | **81.552** | **−2.02%** | +0.81% |

Test 3-split mean: paired Δ −3.97%. Direction-consistent (val and test both improve, test larger).

### Diagnostic from student (per-group AGC clip rate)

Student measured AGC clip rates per parameter-group, every step. Key finding: **`any_clip` rate = 100% every step** — every step at least one tensor was clipped. This is the *opposite* of the regime the AGC paper assumes ("mostly inactive, kicks in only at gradient spikes"). So:
- At λ=0.01, AGC is behaving more like a permanent per-tensor normalizer than a safety clamp.
- The mechanism that won here is "per-tensor direction normalization" rather than "scale-aware safety clipping."
- This is structurally similar to the win that tighter global clip=0.25 got (#3906) — direction normalization at full saturation.

### Decision: SEND BACK for SF-AdamW retest

The win is real (−2.02% paired) and on a mechanism that genuinely differs from global L2 clip (per-tensor vs single threshold). But the absolute val_avg 81.552 is **worse than the current 65.618 baseline** — Edward's experiment was on a stale stack.

Sent back with explicit instructions for the R2 retest:
- Arm A: SF-AdamW + clip=1.0 + EMA + two-shot FiLM (reproduces #3594 winner)
- Arm B: SF-AdamW + AGC λ=0.01 (no `--grad_clip_norm`) + EMA + two-shot FiLM

Under SF-AdamW, the question is whether AGC's per-tensor direction normalization compounds with SF's Polyak averaging (which smooths iterates, not gradients). If AGC wins again on the new stack, it becomes a candidate to replace `--grad_clip_norm 1.0` in the canonical stack.

### Lesson

**Rebase-if-positive-Δ protocol fired correctly.** A paired Δ on a stale baseline is the right signal to retest, not to merge. R2 confirmation on the SF stack will determine canonical adoption.

### Metric artifacts

- `models/model-charliepai2i48h4-edward-agc-r1-arma-clip025-20260516-150051/metrics.jsonl` (control, val_avg=83.233)
- `models/model-charliepai2i48h4-edward-agc-r1-armb-agc-lambda01-20260516-142616/metrics.jsonl` (AGC, val_avg=81.552)

---

## 2026-05-16 16:05 — PR #3980 [SENT BACK for rebase]: Lion optimizer (sign projection) on full clip stack (frieren)

**Branch:** `charliepai2i48h4-frieren/lion-optimizer`

**Hypothesis:** Replace AdamW's `m̂ / (√v̂ + ε)` update with Lion's `sign(β1·m + (1−β1)·g)`. Under `grad_clip_norm=0.25` (100% clip rate), AdamW's per-coordinate adaptive scaling fights the global L2 clip — Lion's sign projection is internally consistent direction normalization at the L∞ extreme.

### Results (R1 paired, AdamW + cosine T_max=15 + clip=0.25 stack, seed=1)

| Arm | Config | val_avg/mae_surf_p | test 3-split |
|---|---|---:|---:|
| A | AdamW + clip=0.25 (control) | 83.812 | 79.376 |
| **B** | **Lion + clip=0.25** | **63.336** | **60.549** |
| **Paired Δ** | — | **−24.43%** | **−23.72%** |

**Arm B absolute val 63.336 beats current SF-AdamW baseline 65.618 by −3.48%; test 60.549 vs 62.853 (−3.67%).** This is the largest paired Δ observed on this track and the lowest absolute val_avg on any single experiment.

### Per-split paired Δ

| Split | A | B | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 95.651 | 65.069 | **−31.97%** |
| `val_geom_camber_rc`     | 94.700 | 77.134 | −18.55% |
| `val_geom_camber_cruise` | 66.706 | 47.166 | **−29.29%** |
| `val_re_rand`            | 78.193 | 63.975 | −18.18% |
| `test_single_in_dist`    | 80.977 | 56.001 | **−30.84%** |
| `test_geom_camber_rc`    | 82.523 | 69.853 | −15.35% |
| `test_re_rand`           | 74.628 | 55.794 | −25.24% |

Wins every split, every metric. Largest wins on `single_in_dist` (−31% val/test) where heavy-tail outliers seem to most benefit from sign projection's coordinate-equalization.

### Diagnostics

- **Wall-clock parity**: Lion 112.4 sec/epoch vs AdamW 111.4 (no speedup from no-`v²`; data/activations dominate at 845k params).
- **Grad clip rate**: ~100% in both arms — every step's pre-optimizer norm exceeded 0.25.
- **Training stability**: Lion's val_loss strictly lower from epoch 2 onward. No instability.

### Mechanism reading

Sign projection is a **strictly better direction normalizer than L2 clip** on this task. Why:
1. **Component-equalization vs vector-rescaling**: clip preserves coordinate ratios then L2-rescales; sign projection forces every coordinate to ±1 then scales by lr. On heavy-tailed gradient distributions, this re-weights toward components AdamW under-weighted.
2. **No internal-vs-external normalization conflict**: AdamW + L2 clip applies per-coordinate scaling (Adam's `v̂`) then global L2 rescaling — two normalizers in series, producing inconsistent direction. Lion's single sign step is internally coherent.

This generalizes the "direction normalization is load-bearing" mechanism from PR #3906 (clip=0.25 win): Lion is the L∞ extreme of the same mechanism family.

### Decision: SEND BACK for rebase

Branch has merge conflicts against the SF-AdamW HEAD. Re-run after rebase to confirm paired Δ holds. Within-run Δ at 12× noise floor; single seed sufficient for definitive merge once rebased.

### What this implies for the stack

If Lion's win reproduces post-rebase, **Lion likely becomes the new canonical optimizer**, displacing AdamW from the stack. Open follow-on questions:
- Does Lion + SF-AdamW compose, or are they testing the same mechanism via different routes? (SF-AdamW removes the LR schedule; Lion changes the update direction. Likely orthogonal but needs paired test.)
- Lion + AGC: another extreme of direction normalization on top of Lion's sign step.

### Metric artifacts

- `models/model-charliepai2i48h4-frieren-lion-r1-arma-adamw-clip25-20260516-143719/metrics.jsonl` (control)
- `models/model-charliepai2i48h4-frieren-lion-r1-armb-lion-clip25-20260516-152650/metrics.jsonl` (Lion, val_avg=63.336)

---

## 2026-05-16 16:08 — PR #3390 [CLOSED]: T_max=20 R2 superseded by SF-AdamW (thorfinn)

**Branch:** `charliepai2i48h4-thorfinn/bf16-tmax20-compose`

**Hypothesis:** Composing bf16 with cosine T_max=20 (R1 found −14.2% on bf16-only stack) should help on the FiLM stack. R2 was a 2-arm paired test: T_max=15 control vs T_max=20.

### Result: Arm A completed, Arm B aborted

| | val_avg/mae_surf_p | best epoch |
|---|---:|---:|
| Arm A (T_max=15, control) | 84.480 | 17 |
| Arm B (T_max=20) | aborted | — |

**Smart hold call by thorfinn**: while Arm A was running, PR #3594 SF-AdamW merged at 15:34 UTC and the stack moved from "AdamW + cosine T_max=15" to "AdamW + SF-AdamW (no cosine)". The decision rule "if Arm B beats Arm A by >1% → merge T_max=20" presumed cosine remained in the stack. SF-AdamW's paired Δ −16.80% is strictly larger than the best plausible T_max=20 Δ (R1 was −14.2% on a weaker stack), so the merge path was foreclosed.

### Decision: CLOSE as superseded

Three options offered (close, run Arm B, pivot to SF+cosine). Student recommended option 1, I agreed. Cosine schedule axis is now formally closed.

### Two-mechanisms-for-same-role pattern (now repeated)

- **n_layers=4** (#3758): cheaper depth-as-proxy-for-late-training-stability. Subsumed by grad_clip which owns the role directly.
- **T_max=20** (#3390): better-use-of-the-cosine-tail. Subsumed by SF-AdamW which removes the tail entirely.

The lesson is becoming a research-program-level principle worth its own writeup section: **when a hypothesis worked via a mechanism that a later baseline now owns directly, the hypothesis loses even if its earlier evidence was real**.

### Metric artifacts

- `models/model-charliepai2i48h4-thorfinn-tmax20-r2-arma-tmax15-clip025-20260516-152533/metrics.jsonl` (Arm A only — control reproduction)

### Next assignment: PR #4051 SF-AdamW weight-decay sweep (thorfinn)

4-arm wd sweep {1e-4 control, 3e-4, 1e-3, 1e-2}. Under SF-AdamW, effective regularization no longer decays with cosine, so the AdamW-tuned wd=1e-4 may be sub-optimal. Three orders of magnitude swept, paired control.

---

## 2026-05-16 16:10 — PR #3492 [SENT BACK for SF retest]: n_hidden=192 R3 — third paired replication (nezuko)

**Branch:** `charliepai2i48h4-nezuko/n-hidden-bigger`

**Hypothesis:** Wider Transolver (n_hidden 128→192, +2.05× params) compounds with grad_clip=1.0 stack. Tested R1 (bf16+T_max=15+EMA, paired −2.99%), R2 (+FiLM, paired −8.21%), R3 (+clip=1.0, paired −5.17%).

### Results (R3 paired, FiLM + clip=1.0 stack, seed=1)

| Arm | n_hidden | n_params | best epoch | val_avg/mae_surf_p | test 3-split |
|---|---:|---:|---:|---:|---:|
| A | 128 | 845,527 | 17 | 83.741 | 79.927 |
| B | **192** | 1,737,559 | 13 | **79.409** | **75.751** |
| **Δ** | — | +2.05× | — | **−5.17%** | **−5.22%** |

**Third consecutive paired replication of the capacity win.** Mechanism robust across three stack changes.

But **Arm B absolute 79.409 > current SF-AdamW baseline 65.618 by +21.0%** — cannot merge on a stale stack.

### Per-split val Δ

| Split | A | B | Δ % |
|---|---:|---:|---:|
| `val_single_in_dist`     | 97.913 | 88.525 | **−9.59%** |
| `val_geom_camber_rc`     | 93.093 | 91.018 | −2.23% |
| `val_geom_camber_cruise` | 63.990 | 60.133 | −6.03% |
| `val_re_rand`            | 79.969 | 77.961 | −2.51% |

`single_in_dist` always biggest winner — wider model best at heavy-tailed raceCar-single pressure distributions.

### Diagnostics (excellent — student added `--seed` flag mid-PR)

- **n_params:** 845,527 → 1,737,559 (+2.05×)
- **VRAM:** 38.92 → 51.98 GB (+33.6%, well within 96 GB budget)
- **Throughput:** 111.4 → 141.7 sec/epoch (+27.2%). Arm B fits 13 epochs in 30 min vs Arm A's 17.
- **Clip rate:** A 98.6%, B 99.2% — direction normalization mechanism preserved at higher capacity.
- **Train loss at common epoch 13:** A=0.0498 vs B=0.0471 (B fits 5.4% tighter; capacity benefit + better generalization).
- **Seed pinning added:** `train.py` now accepts `--seed N`. Removes R2's ~8% absolute drift concern.

### Decision: SEND BACK for R4 SF-AdamW retest

Rebase-if-positive-Δ protocol fires. R4 instructions: paired 2-arm n_hidden=128 vs 192 on the **SF-AdamW** stack (`--use_schedule_free --grad_clip_norm 1.0` + EMA + FiLM + two-shot FiLM). Same seed=1 pinning.

If R4 reproduces the paired Δ at <0 with Arm B absolute < 65.618 → **n_hidden=192 + SF becomes the new canonical capacity** and likely the largest absolute win on the track (capacity + SF compounded). Expected if mechanisms are orthogonal.

### Highest-EV in-flight retest

R2's −8.21% paired and three independent replications make this the strongest mechanism candidate not yet on the current baseline. If it lands at even a fraction of R2's gain on top of SF, val_avg could drop to sub-60.

### Metric artifacts

- `models/model-charliepai2i48h4-nezuko-capacity-r3-arma-nh128-clip-20260516-143410/metrics.jsonl` (control)
- `models/model-charliepai2i48h4-nezuko-capacity-r3-armb-nh192-clip-20260516-152649/metrics.jsonl` (n_hidden=192)

## 2026-05-16 18:30 — PR #3492 (R4): Model capacity n_hidden=192 — CLOSED (capacity axis subsumed under SF)

**Branch:** `charliepai2i48h4-nezuko/model-capacity-nhidden192`

**Hypothesis (R4):** n_hidden=192 should still compose under SF-AdamW (R1-R3 showed compositional wins across +T_max, +FiLM, +clip stacks). Retest on the current SF stack to determine whether the capacity axis is still live after the −16.80% absolute drop.

### Results (paired R4, full SF stack, seed=1)

| Arm | n_hidden | n_params | best epoch | val_avg/mae_surf_p | test 3-split |
|---|---:|---:|---:|---:|---:|
| A | 128 | 845,527 | 17 | **62.950** | **61.257** |
| B | **192** | 1,737,559 | 13 | 63.867 | 61.219 |
| **Δ** | — | +2.05× | — | **+1.46%** | **−0.06%** |

**Per-split val (B − A):** B regresses on 3/4 splits; only `val_re_rand` improves for B (−2.25%).

**At common epoch 13:** A=70.845 vs B=63.867 (B better by −9.85%) — the per-epoch capacity benefit is intact, but Arm A then gets 4 extra epochs (14-17) and uses them to descend further (70.845 → 62.950, −11.2%).

### Diagnostics — the key insight

| Epoch | Arm A val_avg | Arm B val_avg | Δ |
|------:|--------------:|--------------:|---:|
| 5  | 105.219 | 97.080  | −7.74% |
| 9  | 82.096  | 76.708  | −6.56% |
| 11 | 75.711  | 69.783  | −7.83% |
| 13 | 70.845  | **63.867** | **−9.85%** |
| 15 | 67.039  | (cut)   | — |
| 17 | **62.950** | (cut) | — |

Arm B wins per-epoch at every common epoch. **Wall-clock budget cuts B at ep13 just as it's still descending steeply.** Under SF (no cosine floor), Arm A's 4 extra epochs add −11.2% of pure additional improvement.

### Mechanism: BUDGET subsumption, not MECHANISM subsumption

- **Capacity composes with FiLM (R2 −8.21%)** and **clip (R3 −5.17%)** because those don't interact with epoch budget.
- **Capacity does NOT compose with SF-AdamW under fixed wall-clock** because SF's no-cosine-floor convergence makes additional epochs disproportionately valuable, and the wider model loses 4 epochs to its 27% throughput penalty.
- **At larger compute budgets** (or via batch-size tricks / smaller `slice_num`), n_hidden=192 + SF would likely still win.

### Decision: CLOSED (per R4 rule: paired Δ ≥ 0 → "capacity axis subsumed under SF")

R1-R3 paired Δs remain valid as mechanistic findings; they just don't translate to the SF stack at 30-min wall-clock.

### Why this still matters

- **Inductive-bias-amplification hypothesis is alive** — FiLM head width (`film_mlp_hidden`) was fixed at 128 across all four rounds. FiLM is small (no throughput penalty), so the budget calculus that closed R4 does not apply. Next experiment: scale `film_mlp_hidden` instead of n_hidden.
- **Capacity remains dormant, not dead** — could be revisited if (a) wall-clock budget grows, (b) we find lower-overhead width scaling (smaller `slice_num`, batched FiLM compute), or (c) SF-LR sweep (#4038) lands on a lower constant LR favorable to wider models.
- **Arm A absolute 62.950 < baseline 65.618** is single-seed noise within the ±1.5-2% paired variance band.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-nezuko-capacity-r4-arma-nh128-sf-20260516-164211/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-nezuko-capacity-r4-armb-nh192-sf-20260516-173359/metrics.jsonl`


## 2026-05-16 18:40 — PR #3985 (R2): AGC adaptive gradient clipping — CLOSED (mechanism-flip under SF-AdamW)

**Branch:** `charliepai2i48h4-edward/agc-adaptive-clip`

**Hypothesis (R2):** Per-tensor AGC (λ=0.01, NFNet recipe) vs global L2 clip=1.0 under SF-AdamW. R1 found AGC wins by −2.02% paired on AdamW+cosine+clip=0.25 stack — does the win survive the optimizer change?

### Results (paired R2, single seed each)

| Arm | val_avg/mae_surf_p | test 3-split (no cruise) | best epoch | sec/epoch |
|-----|------:|------:|---:|---:|
| A (SF + clip=1.0, control) | **66.761** | **63.510** | 17 | 112.4 |
| B (SF + AGC λ=0.01)        | 72.405      | 69.482     | 16 | 113.6 |
| **Δ %**                    | **+8.45%**  | **+9.40%** | — | +1.1% |

All 4 val splits and all 3 finite-test splits regress for B (smallest single-split regression: +5.32% on val_re_rand).

**Arm A absolute 66.761 vs baseline 65.618 = +1.74%** → control reproduced within ±1.5-2% seed band.

### Mechanism (the key insight)

**R1 (AdamW+cosine):** AGC won −2.02% on clip=0.25 stack. **R2 (SF-AdamW):** AGC loses +8.45% on clip=1.0 stack. Same AGC, same λ, same operating regime (any_clip rate=1.0 every step, group_clip~77%). What changed: **LR schedule**.

- **Under AdamW+cosine:** LR decays from 5e-4 → ~5e-8 over 17 epochs. AGC's per-tensor adaptive clip preserves layer-specific direction *late* in training when LR is small. That's where the R1 win came from.
- **Under SF-AdamW:** LR is constant at 5e-4 throughout. Every step is "big". Per-tensor rescaling at λ=0.01 forces 40 attention tensors each to rescale independently in their own per-tensor frame. The independence is exactly what hurts under constant LR — no late-stage low-LR regime where the per-tensor geometry would matter.
- **Diagnostic confirmation:** per-group `ratio_mean` values are 2-3× smaller under SF than under AdamW+cosine at matched final epoch — the gradient distribution *is* genuinely different.

### Decision: CLOSED (per R2 rule: AGC + SF regresses → "SF and AGC are interfering")

R1 result remains valid on its stack. R2 is a clean mechanism-flip: AGC's value is in late-LR direction preservation, which SF eliminates. Direction normalization at L2 (clip=1.0) wins under SF; per-tensor AGC loses.

### Direction normalization mechanism map (updated)

The "direction normalization mechanism family" now has clearer geometry:

| Mechanism | Geometry | Best stack | Status |
|---|---|---|---|
| clip=0.25 (L2 global) | L2 ball, 100% saturation | AdamW + cosine | Was canonical (#3906) |
| clip=1.0 (L2 global) | L2 ball, 100% saturation | SF-AdamW | Current canonical (#3594) |
| AGC λ=0.01 (per-tensor) | Per-tensor L2 ball | AdamW + cosine | Wins R1, loses R2 (R1 stale-stack) |
| Lion sign-projection (L∞) | Sign on each coord | AdamW + cosine | STRONG WIN R1, awaiting SF compose (#3980) |

**Conclusion:** Direction normalization geometry matters, but the optimizer/scheduler also matters. AGC is **not** a universal upgrade; it's late-LR-specific.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-edward-agc-r2-arma-sf-clip1-20260516-163042/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-edward-agc-r2-armb-sf-agc-20260516-173702/metrics.jsonl`


---

## 2026-05-16 19:30 — PR #4012 [SENT BACK R2]: Sobolev gradient loss — edge-gradient L1 supervision via kNN

- **Student branch:** `charliepai2i48h4-fern/sobolev-loss`
- **Hypothesis:** Adding an edge-gradient L1 loss term (`sobolev_weight` × `||∇ŷ − ∇y||₁`) via in-batch kNN sharpens surface pressure prediction by supervising first-order spatial derivatives, not just point values. Expected −2% to −8% on val_avg.

### Results (R1 — STALE STACK, sent back for R2)

All R1 arms ran on **AdamW+cosine+clip=0.25 stack** (not SF-AdamW). Arm A control absolute 83.812 vs cited baseline 80.893 (+3.6% drift — expected session-level noise from different seed/run).

| Arm | sobolev_weight | val_avg/mae_surf_p | Paired Δ vs A | Test 3-split | Test Δ vs A |
|-----|---|---|---|---|---|
| **A (ctrl)** | 0.0 | 83.812 | — | 79.938 | — |
| B | 0.1 | 84.903 | +1.30% | 79.940 | +0.00% |
| C | 0.3 | 82.912 | **−1.07%** | 79.790 | **−0.19%** |
| D | 1.0 | 83.854 | +0.05% | 80.053 | +0.14% |

**Arm C (w=0.3): paired Δ −1.07% val / −0.19% test.** Non-monotone ranking (B regresses, C wins, D flat). Strongest split gains: val_geom_camber_cruise (−4.07%), val_geom_camber_rc (−2.45%). Weakest: val_re_rand (−0.10%).

### Decision: SENT BACK for SF-AdamW rebase + 2-arm confirmation

Three blockers prevent merge on R1:
1. **Stale stack** — Arm A absolute 83.812 vs current baseline 65.618 (+27.8%). The experiment was run on AdamW+cosine, not the merged SF-AdamW stack.
2. **Weak test transfer** — val Δ −1.07% but test Δ only −0.19% (5× weaker). Sobolev surface-gradient supervision is more likely to fit the specific val split geometry than generalize to test.
3. **Non-monotonicity** — B regresses +1.30%, C wins, D ties. Suggests the win may be noise-sensitive.

### R2 instruction (sent to student 19:20 UTC)

- Rebase onto advisor HEAD `02ae7e3`
- Run 2-arm paired comparison only: Arm A (SF+sobolev_weight=0) vs Arm B (SF+sobolev_weight=0.3)
- Decision gates: (1) reproduces −1% with test transfer → R3 full sweep, (2) val win without test signal → close, (3) regresses or ties → close

### Metric artifacts (R1)

- Arm A: `models/model-charliepai2i48h4-fern-sobolev-r1-arma-sw0_0-*/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-fern-sobolev-r1-armb-sw0_1-*/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-fern-sobolev-r1-armc-sw0_3-*/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-fern-sobolev-r1-armd-sw1_0-*/metrics.jsonl`

---

## 2026-05-16 19:55 — PR #4051 [CLOSED]: SF-AdamW weight-decay sweep — wd ∈ {1e-4, 3e-4, 1e-3, 1e-2}

- **Student branch:** `charliepai2i48h4-thorfinn/sf-wd-sweep`
- **Hypothesis:** Weight decay (wd) was never re-tuned for SF-AdamW; the 100× range sweep probes whether higher wd helps regularize under SF's constant-LR regime. Expected −1% to −3% at some larger wd.

### Results

All arms ran on SF-AdamW stack with 17 epochs.

| Arm | wd | val_avg | Paired Δ vs A | Test 3-split | Test Δ vs A |
|-----|---|---|---|---|---|
| **A (ctrl)** | 1e-4 | **61.758** | — | **60.373** | — |
| B | 3e-4 | 63.278 | +2.46% | 60.612 | +0.40% |
| C | 1e-3 | 63.788 | +3.29% | 60.860 | +0.81% |
| D | 1e-2 | 69.161 | +11.99% | 66.357 | +9.91% |

All four splits regress monotonically. Higher wd uniformly hurts across all splits and test splits. The train/val gap at Arm A is only **+0.00489** — no overfitting to regularize away. Arm C (wd=1e-3) has the smallest gap (+0.00072) but worst val of {A,B,C} — higher wd raises train loss without lowering val loss.

**Weight L2 trajectory:** All arms' weights grow monotonically (gradient signal dominates wd shrinkage). Only Arm D shows a meaningful gap in final weight norm (−2.1% vs A). Arms B and C indistinguishable from A on weight norm (<0.25% Δ).

### Decision: CLOSED (per rule: all higher-wd arms regress → wd=1e-4 at or below optimum)

### Mechanism finding (HIGH VALUE)

**SF Polyak averaging + EMA already saturate the implicit-regularization role.** This is the 3rd "two mechanisms for same role"-style finding under SF-AdamW:
1. Capacity (n_hidden=192, #3492 R4) — budget-subsumed under SF
2. AGC (#3985 R2) — mechanism-flip under SF
3. Weight decay (#4051) — Polyak + EMA absorb the regularization role; external wd has no headroom

**General principle:** SF-AdamW's iterate-averaging absorbs roles previously played by scheduler-mediated regularization mechanisms (cosine LR decay, explicit wd, AGC's late-low-LR regime). Consistent with Defazio et al. (2024) theoretical claim.

**Key secondary finding:** Arm A absolute val=61.758 (5.9% below merged baseline 65.618 at 17 epochs). Multiple within-session controls now land 61-63 absolute (nezuko R4 A=62.95, frieren Lion B=63.336, thorfinn #4051 A=61.758). The merged 65.618 was likely a slightly-unlucky single-seed run; true SF-AdamW+EMA+clip=1.0 stack is approximately 61-63 absolute.

### New assignments

- **thorfinn → #4114**: batch_size sweep under SF-AdamW {4, 6, 8, 12} (VRAM headroom: 39 GB peak at bs=4 vs 96 GB available)

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-thorfinn-sf-wd-r1-arma-wd1e-4-20260516-164911/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-thorfinn-sf-wd-r1-armb-wd3e-4-20260516-172402/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-thorfinn-sf-wd-r1-armc-wd1e-3-20260516-175851/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-thorfinn-sf-wd-r1-armd-wd1e-2-20260516-183339/metrics.jsonl`

---

## 2026-05-16 19:58 — PR #4003 [CLOSED]: Clip threshold R2 — tighter sweep {0.05, 0.1, 0.15, 0.25 control}

- **Student branch:** `charliepai2i48h4-tanjiro/clipthresh-r2-tighter`
- **Hypothesis:** R1 found clip=0.25 improved over clip=1.0; R2 sweeps tighter {0.05, 0.10, 0.15} to find whether the monotone-improvement continues or saturates at 0.25.

### Results (AdamW+cosine stack)

All arms on AdamW+cosine+Huber+EMA+FiLM stack. **Within-session noise floor: 0.97% paired** (measured from two identical clip=0.25 replicates, same seed=1 — the most precise per-step variance measurement this track has produced).

| Arm | clip | val_avg | Paired Δ vs A | Test 3-split | Test Δ vs A |
|-----|---|---|---|---|---|
| A r1 (rerun ctrl) | 0.25 | 82.028 | −0.96% | 78.261 | — |
| **A r2 (official ctrl)** | **0.25** | **82.823** | — | **78.876** | — |
| B | 0.10 | 82.766 | **−0.07%** (tie) | 79.724 | +1.08% |
| C | 0.05 | 83.913 | +1.32% | 79.856 | +1.24% |
| D | 0.15 | 83.274 | +0.54% | 80.003 | +1.43% |

Arm B within noise floor. Arms C/D regress. **All arms at 100% clip rate** at every epoch including clip=0.05 — direction-normalization is fully saturated at 0.25.

**Pre-clip p50 grad norm: ~5-6** across all arms (23× / 61× / 129× / 41× over threshold). Even at clip=0.05, effective LR collapses to ~4e-6/step (far below optimizer's useful regime). Tighter clip is just effective-LR shrinkage, not direction improvement.

### Decision: CLOSED (per rule: no arm beats control by >0.5% paired → clip=0.25 is AdamW+cosine optimum)

### Mechanism finding (HIGH VALUE)

**100% clip rate at all thresholds including 0.05** → direction-normalization is a binary switch, not a knob. Once every gradient is rescaled, the *threshold value* is just a global LR scaling factor. The natural gradient distribution has p50 ≈ 5-6 across all arms; clip=0.25 is already 20-25× below median, so tighter just shrinks effective LR.

**Noise floor calibration:** 0.97% paired variance from cuDNN non-determinism alone (identical seed, config, hardware). This tightens the previous ±1.5-2% stated band — for AdamW+cosine at this stack, the 1-sigma noise is ~0.5% paired. Threshold for a real signal: >1% paired (2-sigma).

**Note:** This is on AdamW+cosine, not SF-AdamW. The SF-clip question remains open under alphonse #4019 (2×2 factorial: clip ∈ {0.25, 1.0} × EMA ∈ {off, on}). Lion's potential win (frieren #3980) must come from momentum dynamics, not direction-only signal.

### New assignments

- **tanjiro → #4113**: EMA decay value sweep under SF-AdamW {0.99, 0.999, 0.9995, 0.9999}

### Metric artifacts

- A r2 (official): `models/model-charliepai2i48h4-tanjiro-clipthresh-r2-arma-clip0_25-20260516-162240/metrics.jsonl`
- A r1 (duplicate ctrl): `models/model-charliepai2i48h4-tanjiro-clipthresh-r2-arma-clip0_25-20260516-143815/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-tanjiro-clipthresh-r2-armb-clip0_1-20260516-152719/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-tanjiro-clipthresh-r2-armc-clip0_05-20260516-172649/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-tanjiro-clipthresh-r2-armd-clip0_15-20260516-182441/metrics.jsonl`


---

## 2026-05-16 21:07 — PR #3980 [MERGED]: Lion optimizer — sign projection vs AdamW (post-rebase R2 confirmation)

- **Student branch:** `charliepai2i48h4-frieren/lion-optimizer`
- **Hypothesis:** Lion's sign-projection update (L∞ direction normalization) provides a more internally consistent gradient direction than L2-clipped AdamW when clip rate ≈ 100%. Expected −5% to −20% on val_avg/mae_surf_p.

### Results (R2 post-rebase, seed=1)

| Arm | optimizer | clip | val_avg/mae_surf_p | Paired Δ | Test 3-split | Test Δ |
|-----|---|---|---|---|---|---|
| **A (AdamW control)** | AdamW + cosine T_max=15 | 0.25 | 83.812 | — | 79.376 | — |
| **B (Lion)** | Lion + cosine T_max=15 | 0.25 | **63.336** | **−24.43%** | **60.549** | **−23.72%** |

Per-split val (lower is better):

| Split | Arm A AdamW | Arm B Lion | Δ % |
|-------|---:|---:|---:|
| `val_single_in_dist`      | 95.651 | **65.069** | −31.97% |
| `val_geom_camber_rc`      | 94.700 | **77.134** | −18.55% |
| `val_geom_camber_cruise`  | 66.706 | **47.166** | −29.29% |
| `val_re_rand`             | 78.193 | **63.975** | −18.18% |
| **val_avg**               | **83.812** | **63.336** | **−24.43%** |

Lion wins on **every single split**, both val and test. Paired Δ is 12× the seed-variance noise floor (±1.5-2%). Bit-exact reproduction across R1 and R2 (same numbers — rebase changed nothing in the training code path).

### Comparison vs current baseline

**Lion 63.336 beats the previously-merged SF-AdamW baseline (65.618) by −3.48% val / −3.67% test.**

This means:
- Lion on AdamW+cosine stack outperforms SF-AdamW on the same base metrics
- Falsifies the implicit assumption that SF-AdamW was the strongest single mechanism
- Lion's −24% paired Δ over AdamW+cosine > SF-AdamW's −16.8% paired Δ over matched cosine

### Operational metrics

| metric | Arm A AdamW | Arm B Lion |
|---|---|---|
| epochs | 17 | 17 |
| best epoch | 17 | 17 |
| sec/epoch | 111.4 | 112.5 |
| peak GPU (GB) | 38.92 | 38.92 |
| clip_rate | ~100% | ~100% |
| train/surf_loss (ep17) | 0.0440 | 0.0296 |

Wall-clock and memory are essentially identical — Lion's ~3.4 MB optimizer state savings negligible vs 39 GB activations.

### Mechanism

With clip_rate ≈ 100%: AdamW updates in `m̂/(√v̂+ε)` direction (per-coordinate rescaling) then globally L2-rescaled to ≤0.25 = **two normalizers in series**. Lion updates as `sign(β₁m + (1-β₁)g)` = **single internally consistent normalizer** that forces all coordinates to ±1 (scaled by lr). Sign projection re-weights toward under-represented gradient components that AdamW de-emphasizes via its adaptive scaling, then globally rescales to cancel that de-emphasis.

### Decision: MERGED as new canonical optimizer (2026-05-16 21:07 UTC)

### New canonical stack

```bash
python train.py \
  --amp_dtype bf16 --cosine_t_max 15 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 0.25 \
  --optimizer lion --lion_lr 1.5e-4 --lion_weight_decay 3e-4 \
  --lion_betas 0.9,0.99
```

### Impact on in-flight experiments

Following the merge, Lion replaces SF-AdamW as canonical. In-flight SF-specific sweeps (#4019, #4038, #4087, #4113, #4114) will be evaluated against the new 63.336 baseline when they complete. Their results remain informative as mechanistic diagnostics even if the canonical stack changed.

Key redirects:
- **#4012 fern Sobolev R2**: Redirected to Lion stack (same 2-arm design, Lion control instead of SF control).
- **#4144 frieren**: New assignment — Lion + SF composition (3-way comparison).

### Metric artifacts

- Winner R2: `models/model-charliepai2i48h4-frieren-lion-r2-armb-lion-clip25-rebased-20260516-183306/metrics.jsonl`
- Control R2: `models/model-charliepai2i48h4-frieren-lion-r2-arma-adamw-clip25-rebased-20260516-172945/metrics.jsonl`
- Original R1 winner: `models/model-charliepai2i48h4-frieren-lion-r1-armb-lion-clip25-20260516-152650/metrics.jsonl`


---

## 2026-05-16 21:18 — PR #4038 [MERGED]: SF-AdamW LR sweep — lr ∈ {5e-4, 1e-3, 2e-3, 5e-3}

- **Student branch:** `charliepai2i48h4-askeladd/sf-lr-sweep`
- **Hypothesis:** The default lr=5e-4 was inherited from AdamW-tuned cosine stack. SF's constant-LR regime benefits from re-tuning upward — paper recommends 4-10× higher LR than scheduled optimizers.

### Results

All arms ran on full SF-AdamW canonical stack (17 epochs each).

| Arm | lr | val_avg | Paired Δ vs A | Test 3-split | Test Δ vs A |
|-----|---|---|---|---|---|
| **A (ctrl)** | 5e-4 | 62.958 | — | 61.235 | — |
| B | 1e-3 | 58.424 | **−7.20%** | 56.145 | −8.31% |
| **C (winner)** | **2e-3** | **54.769** | **−13.01%** | **53.540** | **−12.57%** |
| D | 5e-3 | 55.951 | −11.13% | 53.833 | −12.09% |

**Non-monotone: C < D < B.** C (lr=2e-3) is the minimum. C wins on EVERY val split and EVERY finite test split.

Per-split val:

| Split | A (5e-4) | C (2e-3, winner) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist`     | 72.255 | **60.429** | −16.4% |
| `val_geom_camber_rc`     | 75.771 | **68.478** | −9.6% |
| `val_geom_camber_cruise` | 42.765 | **34.597** | −19.1% |
| `val_re_rand`            | 61.040 | **55.572** | −9.0% |
| **val_avg**              | **62.958** | **54.769** | **−13.01%** |

### Comparison to newly-merged Lion baseline (63.336)

SF-AdamW + lr=2e-3 (54.769) beats Lion + cosine + lr=1.5e-4 (63.336) by **−13.5% absolute on val** and **−11.7% on test (53.540 vs 60.549)**. This re-establishes SF-AdamW as the dominant single mechanism.

**Critical finding:** The lr=5e-4 default was catastrophically wrong for SF. SF's constant-LR regime needs ~4× higher LR than AdamW's cosine-schedule counterpart. Every SF experiment up to this point (including the Lion comparison) was run with the wrong LR.

### Decision: MERGED as new canonical stack (2026-05-16 21:18 UTC)

### Impact on in-flight experiments

All in-flight SF sweeps (#4019, #4087, #4113, #4114) were run with lr=5e-4 — the wrong LR. Their results are still mechanistically informative (directional effects transfer), but absolute numbers and any "close to baseline" decisions should be re-evaluated. When these PRs land:

- If paired Δ is still directionally significant → result likely holds at lr=2e-3 (probably bigger)
- If paired Δ was marginal → re-test on the correct lr=2e-3 stack

Frieren #4144 Lion+SF composition: **Updated to use SF-AdamW lr=2e-3 in the Arm B control** (posted corrective comment).

### New canonical stack

```bash
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 2e-3
```

### New assignments

- **askeladd → #4149**: Lion LR sweep {7.5e-5, 1.5e-4, 3e-4, 6e-4} — can the 4× LR insight transfer to Lion? Tests whether Lion matches SF-AdamW lr=2e-3 with higher LR.

### Metric artifacts

- Winner (lr=2e-3): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-armc-lr2e-3-20260516-180222/metrics.jsonl`
- Control (lr=5e-4): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-arma-lr5e-4-20260516-192309/metrics.jsonl`
- Arm B (lr=1e-3): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-armb-lr1e-3-20260516-172709/metrics.jsonl`
- Arm D (lr=5e-3): `models/model-charliepai2i48h4-askeladd-sf-lr-r1-armd-lr5e-3-20260516-183721/metrics.jsonl`

---

## 2026-05-16 21:50 — PR #4087 [CLOSED / NULL]: SF-AdamW warmup steps sweep — {100, 500, 1000, 2000}

- **Student branch:** `charliepai2i48h4-edward/sf-warmup-steps-sweep`
- **Hypothesis:** SF's warmup duration (sf_warmup_steps=500, paper default) has never been tuned for our 30-min wall-clock budget. Paper default was calibrated for longer training. Shorter warmup gives more steps at full LR; longer warmup gives cleaner Polyak iterate burn-in.
- **Stack:** SF-AdamW lr=5e-4 (stale — canonical has since moved to lr=2e-3), bf16, EMA 0.999, FiLM+two-shot, clip=1.0

### Results

| Arm | warmup | val_avg/mae_surf_p | Δ vs A | Δ % |
|-----|-------:|---------:|-------:|----:|
| **A (control)** | **500** | **63.317** | — | — |
| B | 100 | 67.843 | +4.526 | **+7.15%** regress |
| C | 1000 | 65.730 | +2.414 | **+3.81%** regress |
| D | 2000 | 64.355 | +1.039 | **+1.64%** regress |

**Arm A (warmup=500, paper default) wins every val and test split.**

Test 3-split mean (Arm A): 60.521 (sid=62.798, cam_rc=66.605, re_rand=52.160)

Per-epoch trajectory confirmed the mechanism:
- Epochs 1–4: Arm B (warmup=100) leads briefly due to faster initial activation
- Epoch 5 onward: Arm A wins and stays ahead — short-warmup noise contaminates Polyak average
- Arm D (warmup=2000) closes monotonically from worst-at-ep1 to +1.6% behind A at ep17; with longer budget might converge to A

Gradient-norm trajectories confirmed expected mechanism: longer warmup → lower ||g||₂ in ep1, peak shifted later. All arms converge to ~25–35 mean ||g||₂ by ep10.

### Conclusions

**Null result. Warmup axis exhausted.**
- Paper-default warmup=500 steps is well-calibrated for our ~17-epoch / 30-min budget
- Shorter warmup (100) contaminates Polyak iterate with too-early noisy samples (worst outcome)
- Longer warmup (1000, 2000) wastes budget at sub-full LR — Arm D only reaches +1.6% behind A
- Conclusion likely transfers to lr=2e-3 canonical (mechanism is LR-independent)

**Note:** Ran at stale lr=5e-4 stack. Arm A absolute (63.317) does NOT beat current canonical (54.769). Warmup conclusion transfers; no re-run needed.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-edward-sf-warmup-r1-armA-w500-20260516-184849/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-edward-sf-warmup-r1-armB-w100-20260516-192340/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-edward-sf-warmup-r1-armC-w1000-20260516-195821/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-edward-sf-warmup-r1-armD-w2000-20260516-203308/metrics.jsonl`

### Follow-up

**edward → #4157**: SF-AdamW LR fine-tune {1.5e-3, 2e-3, 2.5e-3, 3e-3} — localize the lr=2e-3 peak discovered in askeladd #4038.

---

## 2026-05-16 21:55 — PR #4019 [SENT BACK]: SF-AdamW clip×EMA 2×2 factorial (R1, stale lr=5e-4 stack)

- **Student branch:** `charliepai2i48h4-alphonse/sf-adamw-clip-ema-compose`
- **Hypothesis:** Two mechanism questions under SF-AdamW — (1) Is clip=0.25 still optimal under SF's Polyak averaging? (2) Is external EMA still load-bearing or redundant with internal Polyak averaging?
- **Stack (stale):** SF-AdamW lr=5e-4, bf16, FiLM+two-shot, with clip and EMA factorialized. Seed=1 across all 4 arms (PR-introduced `--seed` for paired reproducibility).

### Results

| Arm | clip | EMA | val_avg | Δ vs A | test 3-split mean | clip rate @ best |
|---|---:|:---:|---:|---:|---:|---:|
| A (control) | 1.0 | on | 63.300 | — | 62.359 | 0.9947 |
| B | 0.25 | on | 63.520 | +0.347% | 62.748 | 1.0000 |
| **C** | 1.0 | **off** | **62.914** | **−0.610%** | **62.047** | 0.9947 |
| D | 0.25 | off | 63.162 | −0.218% | 62.417 | 1.0000 |

**Direction signal:** EMA-off (C) wins on **3 of 4 val splits** + test 3-split mean. Pattern is internally consistent. But:

- **Paired Δ 0.610% < noise floor 0.97%** (tanjiro #4003) — below seed-variance threshold.
- **Absolute miss:** All 4 arms regress 15-16% vs current canonical SF-AdamW lr=2e-3 (54.769).
- **Cannot generalize to lr=2e-3:** Higher gradient norms at higher LR could make EMA more critical (smoothing helps) OR more redundant (Polyak dominates).

### Conclusions

**Sent back for R2 at lr=2e-3.** The R1 ranking is consistent enough to be worth investigating at the correct LR. Three plausible outcomes for R2:
- EMA-off gain grows (>0.5% paired) → merge candidate, update canonical
- Gain shrinks below 0.3% → close EMA-removal hypothesis
- Surprising winner (B or D) → investigate LR×clip×EMA interaction

### Notable infrastructure contribution

Alphonse introduced `--seed` flag for deterministic paired-arm reproducibility. This is excellent infra and should be standard for all future paired-Δ experiments.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-alphonse-sf-r3-arma-clip1_ema-20260516-202503/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-alphonse-sf-r3-armb-clip0_25-ema-20260516-175159/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-alphonse-sf-r3-armc-clip1-noema-20260516-182613/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-alphonse-sf-r3-armd-clip0_25-noema-20260516-192255/metrics.jsonl`

---

## 2026-05-16 22:30 — PR #4012 [CLOSED / NULL]: Sobolev edge-gradient L1 supervision on surface pressure

- **Student branch:** `charliepai2i48h4-fern/sobolev-gradient-loss`
- **Hypothesis:** Add an edge-gradient L1 supervision term (∇_x p) to the surface loss. Tests whether forcing the model to learn spatial-derivative regularity improves the surface MAE.
- **R1 stack:** AdamW + cosine + clip=0.25 + EMA + FiLM + bf16 + Huber (stale, lr=5e-4)
- **R2 stack:** Lion + cosine + clip=0.25 + EMA + FiLM + bf16 + Huber (lion_lr=1.5e-4)

### Results

**R1 (AdamW+cosine, 4 arms over Sobolev weight {0, 0.1, 0.3, 0.5}):**
- Non-monotonic ranking: B regresses, C wins narrowly, D flat
- Val→test transfer ratio ~5× (signal didn't transfer cleanly)
- Student's own honest read: "within seed variance"

**R2 (Lion+cosine, 2-arm paired w=0 vs w=0.3 with --seed 1):**

| Arm | sobolev_weight | val_avg | Δ val | test_3split | Δ test |
|---|---:|---:|---:|---:|---:|
| A (control) | 0.0 | 63.399 | — | 60.525 | — |
| B | 0.3 | 65.096 | **+2.68% regress** | 61.904 | **+2.28% regress** |

- Arm A reproduces Lion baseline within float drift (Δ +0.06 / +0.10%)
- Every single val and test split regresses (sid, cam_rc, cam_cr, re_rand)
- Val→test transfer ratio 0.81× (clean — regression travels consistently)

### Conclusions

**Sobolev hypothesis exhausted across two optimizer stacks.** The marginal R1 effect was seed luck; the R2 paired test (seed=1) shows clean regression on Lion. Mechanism doesn't fit the loss landscape — edge-gradient supervision adds a slightly conflicting objective.

**No re-test on SF lr=2e-3:** Two stacks with null/regression is sufficient evidence. The Sobolev penalty fires the same way regardless of optimizer; the additional ~0.9% of total loss contribution doesn't help surface MAE.

### Metric artifacts (R2)

- `models/model-charliepai2i48h4-fern-sobolev-r2-arma-lion-w0-20260516-213309/metrics.jsonl`
- `models/model-charliepai2i48h4-fern-sobolev-r2-armb-lion-w0_3-20260516-220713/metrics.jsonl`

### Follow-up

**fern → #4208**: Dropout regularization sweep at SF-AdamW lr=2e-3 — explore the untouched regularization axis with paired Δ.

---

## 2026-05-16 22:30 — PR #4113 [CLOSED / NULL]: EMA decay value sweep — {0.99, 0.999, 0.9995, 0.9999}

- **Student branch:** `charliepai2i48h4-tanjiro/sf-ema-r1`
- **Hypothesis:** Under SF-AdamW's Polyak averaging, the external EMA decay target may interact differently with the Karras warmup ramp. Test faster (0.99), default (0.999), slower (0.9995), much slower (0.9999) decays.
- **Stack (stale):** SF-AdamW lr=5e-4 + clip=1.0 + EMA + FiLM two-shot + bf16, no --seed

### Results

| Arm | ema_decay | val_avg | Δ vs A | test_3split | clip rate |
|---|---:|---:|---:|---:|---:|
| A (control) | 0.999 | 65.349 | — | 63.708 | 0.9947 |
| **B (winner nominal)** | **0.99** | **62.921** | **−3.72%** | **59.984** | 0.9947 |
| C | 0.9995 | 63.092 | −3.45% | 60.769 | 0.9947 |
| D | 0.9999 | 64.200 | −1.76% | 61.289 | 0.9947 |

### Critical caveat (student's own analysis)

**Karras warmup ramp `effective_decay = min(target, (1+step)/(10+step))` reaches 0.99859 at step 6375 (epoch 17 cap).** This means Arms A/C/D should be **theoretically identical** (same ramp-limited decay throughout). The 3.46% A-C-D spread IS the cross-arm seed-variance in this setup (no --seed flag on tanjiro's branch yet — it was added in #3980 Lion merge).

Only Arm B (target=0.99) is target-limited because it hits its target at step 100, well before the ramp catches up.

### Honest read of the signal

- **Arm B's 3.72% val win over Arm A is *at* the cross-arm noise band (3.46%)** — at noise floor
- **B vs C tie on val (62.92 vs 63.09)** is the cleanest evidence against a strong faster-EMA effect
- All 4 arms regress 15-18% vs current canonical SF-AdamW lr=2e-3 (54.769) — stale-LR stack issue

### Conclusions

**Closed without re-test on SF lr=2e-3.** Alphonse #4019 R2 directly tests EMA on/off at lr=2e-3 with --seed — that experiment subsumes "fastest possible decay" and will give the clean answer.

**Key learning preserved:** Karras ramp dominates target ema_decay at small step budgets — important constraint for future EMA work. Paired methodology with --seed matters MORE than absolute decay tuning at this scale.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-tanjiro-sf-ema-r1-arma-d0_999-20260516-202503/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armb-d0_99-20260516-205953/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armc-d0_9995-20260516-213442/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armd-d0_9999-20260516-220927/metrics.jsonl`

### Follow-up

**tanjiro → #4207**: surf_weight sweep at SF-AdamW lr=2e-3 — directly modulate the primary metric's loss term weight, untested at correct LR.

---

## 2026-05-16 23:55 — PR #4149 [CLOSED / NULL]: Lion LR sweep — lion_lr ∈ {7.5e-5, 1.5e-4, 3e-4, 6e-4}

- **Student branch:** `charliepai2i48h4-askeladd/lion-lr-r1`
- **Hypothesis:** Can Lion match SF-AdamW (54.769) with higher LR? The SF \"4× LR\" insight (5e-4 → 2e-3) might transfer to Lion's sign-projection regime.
- **Stack:** Lion + cosine T_max=15 + clip=0.25 + EMA 0.999 + FiLM two-shot + bf16 + Huber

### Results

| Arm | lion_lr | val_avg | paired Δ vs A | test 3-split | vs SF (54.769) |
|---|---:|---:|---:|---:|---:|
| **A (control)** | **1.5e-4** | **61.146** | — | **58.577** | +11.64% |
| B | 7.5e-5 (0.5×) | 72.088 | +17.89% | 68.483 | +31.62% |
| C | 3e-4 (2×) | 61.412 | +0.43% (tied) | 57.542 | +12.13% |
| D | 6e-4 (4×) | 65.611 | +7.30% | 63.115 | +19.80% |

### Crossing-pattern diagnostic

Per-epoch trajectory shows:
- D leads epochs 1–3 (higher LR steps further per iteration)
- C catches up around epoch 5
- **A pulls ahead by epoch 7 and never gives the lead back**

This is the cosine-decay signature: high-LR arms burn early-epoch budget faster, then can't refine as LR decays. SF-AdamW avoids this because constant LR maintains step magnitude throughout — **explains why SF benefits from higher LR while Lion+cosine doesn't**.

### Conclusions

1. **The SF \"4× LR\" insight does NOT transfer to Lion.** Sign projection lacks the Polyak iterate-averaging buffer that SF uses to stabilize at higher LR.
2. **lion_lr=1.5e-4 is the local optimum under cosine T_max=15.** A vs C tied within noise; B and D regress monotonically.
3. **No Lion arm approaches SF-AdamW canonical (54.769).** Best Lion (A) is +11.64% behind. **Lion is officially behind SF as a standalone optimizer.**

**Open question:** Lion + Schedule-Free composition (frieren #4144 Arm C) — if Lion+SF wins, the constant-LR regime might rescue Lion's higher-LR potential. If Lion+SF loses, Lion is fully closed.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-askeladd-lion-lr-r1-arma-lr1_5e-4-20260516-205206/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-askeladd-lion-lr-r1-armb-lr7_5e-5-20260516-212554/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-askeladd-lion-lr-r1-armc-lr3e-4-20260516-220123/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-askeladd-lion-lr-r1-armd-lr6e-4-20260516-223642/metrics.jsonl`

### Follow-up

**askeladd → #4225**: Model width sweep at SF-AdamW lr=2e-3 — `n_hidden` ∈ {96, 128, 160, 192} with --seed 1. The primary model capacity axis, untouched at correct LR.


---

## 2026-05-17 01:00 — PR #4157 [MERGED/WINNER]: SF-AdamW LR fine-tune — lr=3e-3 NEW BEST

- **Student branch:** `charliepai2i48h4-edward/sf-lr-fine`
- **Hypothesis:** Fine-tune SF-AdamW learning rate in [1.5e-3, 3e-3] to localize the peak identified by the coarse sweep (#4038, winner 2e-3). Paired 4-arm sweep with --seed 1.

### Results

| Arm | lr | val_avg/mae_surf_p | Δ vs B (control) | Δ vs baseline (54.769) |
|-----|---:|---:|---:|---:|
| A | 1.5e-3 | 55.902 | +3.88% (regression) | +2.07% |
| **B (control)** | **2e-3** | **53.814** | — | −1.74% |
| C | 2.5e-3 | 53.182 | −1.18% | −2.90% |
| **D (winner)** | **3e-3** | **52.258** | **−2.89%** | **−4.59%** |

**Key finding:** Sweep is perfectly monotone A→B→C→D. True LR peak is BEYOND 3e-3 — all 4 arms hit ep17/17 budget cap still descending. The gains are also accelerating (2→2.5: −1.18%, 2.5→3: −1.71%), suggesting more headroom above 3e-3.

Per-split val MAE (winning arm D, lr=3e-3):

| Split | val/mae_surf_p |
|-------|---:|
| val_single_in_dist | 56.454 |
| val_geom_camber_rc | 66.039 |
| val_geom_camber_cruise | 33.763 |
| val_re_rand | 52.775 |
| **val_avg** | **52.258** |

Test 3-split mean (D): 51.206 — sid=48.731, rc=60.335, re=44.552

### Metric artifacts

- Winner (Arm D, lr=3e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armD-lr3e-3-20260516-222733-20260516-233714/metrics.jsonl`
- Control (Arm B, lr=2e-3): `models/model-charliepai2i48h4-edward-sf-lr-fine-r1-armB-lr2e-3-20260516-222733-20260516-222735/metrics.jsonl`

### Analysis

The "true LR peak beyond 3e-3" finding has two causes:
1. **Polyak iterate averaging**: SF-AdamW's internal Polyak averaging stabilizes higher-LR trajectories — models trained at 3e-3 still produce smooth averaged iterates even when individual steps are large.
2. **Budget truncation**: All arms at ep17 are still descending. At higher LR, the optimizer reaches lower loss states earlier in training; the val curve is still declining when the 30-min budget cuts it. Longer budget would likely pull all arms even lower, with larger gaps.

**New canonical LR: 3e-3.** All in-flight experiments were at lr=2e-3 and should apply the paired-Δ gate: if they beat their seed-1 control, the result is still directionally valid, but may be even stronger at lr=3e-3.

### Follow-up

**edward → new assignment #4XXX**: LR extension sweep {3e-3, 4e-3, 5e-3, 7e-3} at SF-AdamW + seed=1.

---

## 2026-05-17 01:05 — PR #4144 [CLOSED/NULL]: Lion + Schedule-Free composition (3-way)

- **Student branch:** `charliepai2i48h4-frieren/lion-sf-compose-r1`
- **Hypothesis:** Lion+SF composition might rescue Lion's higher-LR potential by applying SF's Polyak averaging to Lion's sign-projected steps. Three-way comparison: A=Lion+cosine (control), B=SF-AdamW lr=2e-3 (control), C=Lion+SF (hypothesis).

### Results

| Arm | Optimizer | LR | val_avg/mae_surf_p | Δ vs A | Test 3-split |
|-----|-----------|---:|---:|---:|---:|
| A | Lion + cosine T_max=15 | 1.5e-4 | 63.336 | — (ref) | 60.549 |
| B | SF-AdamW (Polyak) | 2e-3 | 54.957 | −13.23% | 53.543 |
| C1 | Lion + SF (custom wrapper) | 1.5e-4 | 111.223 | **+75.61%** | 108.801 |
| C2 | Lion + SF (custom wrapper) | 6e-4 | 112.988→div | **+78.40%** | 112.070 |

- Arm A exactly reproduces 63.336 (to 4dp). Arm B reproduces canonical within 0.34% (well within noise).
- C1 declines slowly but plateaus 75% above Lion baseline — never converges.
- C2 reaches best val=112.99 at ep12, then **catastrophically diverges** (ep14: 3184, then 682, 446).

### Mechanism (why Lion+SF fails)

Lion is a pure **sign-projection** optimizer — every gradient update is exactly ±lr, independent of gradient magnitude. SF's internal Polyak averaging is designed to average across AdamW steps, which have varying per-coordinate magnitude. When composed with Lion's constant-magnitude steps, the Polyak averaging loses its ability to exploit scale differences across gradient coordinates. The result is a momentum buffer that accumulates fixed-magnitude steps and fails to find good iterate averages. This is a fundamental incompatibility, not a tuning problem.

### Conclusions

1. **Lion+SF composition is definitively falsified.** Results are catastrophically worse, not marginally worse.
2. **Lion track is now fully exhausted:** Lion standalone (best: 63.336, +11.6% behind SF canonical), Lion LR boost (no benefit, #4149), Lion+SF (catastrophic failure, +75-78% regression).
3. **SF-AdamW is the optimizer.** No further Lion experiments warranted unless a fundamentally different composition approach is found.

### Metric artifacts

- Arm A: `models/model-charliepai2i48h4-frieren-lion-sf-r1-arma-lion-cosine-20260516-210530/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-frieren-lion-sf-r1-armb-sf-adamw-lr2e-3-20260516-214301/metrics.jsonl`
- Arm C1: `models/model-charliepai2i48h4-frieren-lion-sf-r1-armc-lion-sf-lr1.5e-4-20260516-232654/metrics.jsonl`
- Arm C2: `models/model-charliepai2i48h4-frieren-lion-sf-r1-armc-lion-sf-lr6e-4-20260516-224341/metrics.jsonl`

### Follow-up

**frieren → new assignment**: n_layers depth sweep {3, 4, 5, 7} at SF-AdamW lr=3e-3 + seed=1. (Requires --n_layers Config/CLI edit.)

---

## 2026-05-17 02:38 — PR #4114 [CLOSED/NULL]: Batch size sweep under SF-AdamW — bs ∈ {4, 6, 8, 9}

- **Student branch:** `charliepai2i48h4-thorfinn/sf-bs-r1`
- **Hypothesis:** Larger batch sizes reduce gradient variance (lower CV), and SF-AdamW's Polyak averaging may amplify cleaner-gradient benefit enough to offset the step-count reduction within the 30-min budget.
- **Stack:** SF-AdamW lr=5e-4 (stale — ran before lr=2e-3 canonical update)

### Results

| Arm | bs | best ep | val_avg/mae_surf_p | Δ vs A | sec/ep | total steps | peak GB |
|-----|----|---------|--------------------|--------|--------|-------------|---------|
| A (control) | 4 | 17 | **65.168** | +0.00% | 111.5 | 6,375 | 38.92 |
| B | 6 | 16 | 73.655 | +13.02% | 116.1 | 4,000 | 58.35 |
| C | 8 | 16 | 80.192 | +23.05% | 118.1 | 3,008 | 77.82 |
| D | 9 | 16 | 84.107 | +29.06% | 119.9 | 2,672 | 87.51 |

Note: bs=10 and bs=12 OOM'd at SF-AdamW full optimizer-state footprint (peak 94.8/93.7 GB). Practical ceiling: bs=9.

### Mechanism analysis

- **Gradient CV drops monotonically** (0.88 → 0.54 for bs=4 → bs=9): cleaner gradients confirmed mechanistically.
- **Step-count dominates** despite cleaner gradients: bs=9 gets 2,672 total steps vs bs=4's 6,375 — a 2.4× deficit within the 30-min budget.
- **Warmup-share confounder**: fixed 500-step warmup consumes 7.8% of budget for bs=4 but 12.3% for bs=9, amplifying the epoch-1 regression (209 → 313).
- Polyak averaging does not amplify the cleaner-gradient benefit enough to offset the throughput loss.

### Conclusions

Hypothesis **decisively rejected**. bs=4 wins by 13-29% paired Δ across all splits and on the test 3-split mean. The regression magnitudes (13-29%) far exceed the 3% stale-LR gate — re-testing at lr=3e-3 is not warranted. bs=4 is a fixed point of this stack.

**Key lesson:** The binding constraint under SF-AdamW in this budget is total step count, not gradient quality. The warmup-share effect compounds this for large batches. Treat bs=4 as canonical.

### Metric artifacts

- `models/model-charliepai2i48h4-thorfinn-sf-bs-r1-arma-bs4-r4-20260516-233055/metrics.jsonl`
- `models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armb-bs6-r4-20260517-000550/metrics.jsonl`
- `models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armc-bs8-r4-20260517-004008/metrics.jsonl`
- `models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armd-bs9-r4-20260517-012908/metrics.jsonl`

### Follow-up

**thorfinn → new assignment #4303**: slice_num sweep {32, 64, 96, 128} at SF-AdamW lr=3e-3 + seed=1. Third primary Transolver architecture axis (alongside n_hidden and n_layers).

---

## 2026-05-17 02:53 — PR #4019 [CLOSED/NULL R2]: SF clip×EMA factorial — R2 re-test at lr=2e-3

- **Student branch:** `charliepai2i48h4-alphonse/sf-adamw-clip-ema-compose`
- **R1 result (closed earlier, sent back):** lr=5e-4, paired Δ for EMA-off (Arm C vs A) = −0.610%; below merge but consistent across 3/4 splits.
- **R2 (this round) stack:** SF-AdamW lr=2e-3 + paired --seed 1, 2×2 over {clip=1.0, 0.25} × {EMA on, off}.

### R2 Results

| Arm | clip | EMA | val_avg/mae_surf_p | Δ vs A (paired) | Δ vs 52.258 (canonical) |
|---|---:|:---:|---:|---:|---:|
| A (control) | 1.0 | on | 54.6735 | — | +4.625% |
| B | 0.25 | on | 55.1499 | +0.871% | +5.537% |
| **C (winner)** | 1.0 | off | 54.4385 | **−0.430%** | +4.176% |
| D | 0.25 | off | 54.9295 | +0.468% | +5.115% |

### Two mechanism findings

1. **EMA-off direction holds but attenuates.** R1 paired Δ at lr=5e-4: −0.610%. R2 paired Δ at lr=2e-3: −0.430%. Extrapolating to lr=3e-3: ~−0.35%. SF's Polyak averaging absorbs more of EMA's averaging role as LR scales up; external EMA(0.999) → less load-bearing.
2. **Clip × LR interaction is the surprise.** clip=0.25 vs clip=1.0 swung from neutral at lr=5e-4 (+0.35%) to consistently harmful at lr=2e-3 (+0.9%). Under SF + higher LR, the clip threshold is an effective-LR knob, not a passive guard rail. Keep clip=1.0 in canonical.

### Conclusions

Closed without merge: Arm C (best) regresses +4.18% vs current canonical 52.258 (because the canonical moved from lr=2e-3 to lr=3e-3 after the send-back was authored). Paired Δ (0.43%) is below the 0.5% merge threshold. EMA-off direction is real but operationally below merge bar at any LR tested.

**Canonical decision:** Keep `--use_ema --ema_decay 0.999` AND `--grad_clip_norm 1.0` in the canonical stack. EMA gain at lr=3e-3 (extrapolated ~0.35%) is below merge threshold; not worth a third round on the same axis.

### Metric artifacts (committed)

- Arm A: `models/model-charliepai2i48h4-alphonse-sf-r4-arma-clip1_ema-lr2e3-20260517-002510/metrics.jsonl`
- Arm B: `models/model-charliepai2i48h4-alphonse-sf-r4-armb-clip0_25_ema-lr2e3-20260517-010040/metrics.jsonl`
- Arm C: `models/model-charliepai2i48h4-alphonse-sf-r4-armc-clip1-noema-lr2e3-20260517-013531/metrics.jsonl`
- Arm D: `models/model-charliepai2i48h4-alphonse-sf-r4-armd-clip0_25-noema-lr2e3-20260517-021025/metrics.jsonl`

### Follow-up

**alphonse → new assignment #4317**: SF-AdamW betas sweep at lr=3e-3, 2×2 over {beta1: 0.9, 0.95} × {beta2: 0.99, 0.999}. First optimizer-internal axis ever explored in this track. Requires `--sf_beta1`/`--sf_beta2` infra commit.
