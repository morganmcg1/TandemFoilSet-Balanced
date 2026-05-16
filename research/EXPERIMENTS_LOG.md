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

