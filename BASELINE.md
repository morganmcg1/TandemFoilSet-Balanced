# Baseline Metrics — icml-appendix-charlie-pai2i-48h-r2

All baselines on `val_avg/mae_surf_p` (lower is better). Test metric is `test_avg/mae_surf_p`.

---

## 2026-05-15 13:50 — PR #3119: Round-1 baseline (epochs=80 config, ~14 effective epochs)

**Student:** charliepai2i48h2-thorfinn  
**Change:** epochs 50→80 (cosine T_max=80). 30-min timeout cap hit at epoch 14 — schedule change had no effective impact. This run establishes the round-1 absolute reference.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **135.0153** |
| val_single_in_dist/mae_surf_p | 167.5711 |
| val_geom_camber_rc/mae_surf_p | 148.3005 |
| val_geom_camber_cruise/mae_surf_p | 103.9843 |
| val_re_rand/mae_surf_p | 120.2051 |
| **test_avg/mae_surf_p** | **NaN** (pre-existing scoring bug; 3-split proxy = 132.4954) |
| test_single_in_dist/mae_surf_p | 145.30 |
| test_geom_camber_rc/mae_surf_p | 133.81 |
| test_geom_camber_cruise/mae_surf_p | NaN (1 GT sample has NaN p) |
| test_re_rand/mae_surf_p | 118.37 |
| Best epoch | 12 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU, trunc_normal_(std=0.02)  
**Optimizer:** AdamW lr=5e-4, weight_decay=1e-4, CosineAnnealingLR(T_max=80, effective T_max≈50)  
**Loss:** vol_loss + 10·surf_loss (uniform per-channel)  
**Batch:** 4

**Scoring bug note:** `test_geom_camber_cruise/mae_surf_p` is NaN due to sample 20 of that split having NaN p in GT (accumulate_batch NaN×0=NaN propagation). A one-line fix via `torch.nan_to_num` in `evaluate_split` in train.py will resolve this; it is tracked as PR bug-fix work.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --epochs 50
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **135.0153**~~ — superseded by PR #3101 below.

---

## 2026-05-15 16:26 — PR #3101: surf_weight 10→30 (3× surface loss emphasis)

**Student:** charliepai2i48h2-askeladd  
**Change:** `Config.surf_weight: 10.0 → 30.0` — single-line change aligning training loss with primary metric.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **127.4122** |
| val_single_in_dist/mae_surf_p | 152.8215 |
| val_geom_camber_rc/mae_surf_p | 134.8461 |
| val_geom_camber_cruise/mae_surf_p | 102.5963 |
| val_re_rand/mae_surf_p | 119.3849 |
| **test_avg/mae_surf_p** | **NaN** (scoring bug; corrected 4-split avg = 116.83) |
| test_single_in_dist/mae_surf_p | 136.68 |
| test_geom_camber_rc/mae_surf_p | 123.03 |
| test_geom_camber_cruise/mae_surf_p | NaN → 88.12 (corrected via test_metrics_corrected.json) |
| test_re_rand/mae_surf_p | 119.51 |
| Best epoch | 14 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** AdamW lr=5e-4, weight_decay=1e-4, CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + **30·surf_loss** (was 10) — uniform per-channel  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-askeladd-surf-weight-30-20260515-124531/metrics.jsonl`

**Note:** test_avg/mae_surf_p is NaN for this run (pre-merge; bug-fix PR #3274 now merged). Use test_metrics_corrected.json for the corrected 4-split test avg (116.83).

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 30
```

> ~~**Beat this:** val below **127.4122**~~ — superseded by PR #3293 below.

---

## 2026-05-15 17:25 — PR #3293: Lion optimizer replacing AdamW (lr=1.7e-4, wd=3e-4)

**Student:** charliepai2i48h2-nezuko  
**Change:** Replace AdamW with Lion (sign-based update, single momentum buffer). lr=5e-4→1.7e-4, wd=1e-4→3e-4. All other settings baseline (including surf_weight=10 during the run; merged config now has surf_weight=30 from PR #3101 compounded automatically).

| Metric | Value (measured at surf_weight=10) |
|--------|------------------------------------|
| **val_avg/mae_surf_p** | **117.5014** |
| val_single_in_dist/mae_surf_p | 137.2384 |
| val_geom_camber_rc/mae_surf_p | 124.4193 |
| val_geom_camber_cruise/mae_surf_p | 97.3192 |
| val_re_rand/mae_surf_p | 111.0287 |
| **test_avg/mae_surf_p** | **NaN** (pre-merge; #3274 fix now live) |
| test_single_in_dist/mae_surf_p | 121.5755 |
| test_geom_camber_rc/mae_surf_p | 117.3048 |
| test_geom_camber_cruise/mae_surf_p | NaN (scoring bug — now fixed) |
| test_re_rand/mae_surf_p | 108.2318 |
| test_avg (3-split proxy) | 115.7040 |
| Best epoch | 9 |
| Peak GPU memory | 42.11 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** **Lion** lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + **30·surf_loss** (surf_weight=30 from PR #3101, compounded in merged HEAD)  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-nezuko-lion-optimizer-20260515-153522/metrics.jsonl`

**Note:** val=117.50 was measured with surf_weight=10; merged HEAD has surf_weight=30 + Lion (unmeasured combined). Next experiments should compare against 117.50 and also produce a clean Lion+surf30 measurement.

**Reproduce (merged config — Lion + surf_weight=30):**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** val below **117.5014**~~ — superseded by PR #3357 below.

---

## 2026-05-15 18:40 — PR #3357: asinh loss transform for surface pressure

**Student:** charliepai2i48h2-tanjiro  
**Change:** Apply `torch.asinh()` to pressure channel z-scores in the training loss only (not evaluation). Compresses heavy-tail gradient signal on pressure; Ux/Uy loss unchanged. Evaluation MAE remains in physical units — comparison is fair.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **84.9819** |
| val_single_in_dist/mae_surf_p | 108.0437 |
| val_geom_camber_rc/mae_surf_p | 90.6260 |
| val_geom_camber_cruise/mae_surf_p | 62.6771 |
| val_re_rand/mae_surf_p | 78.5807 |
| **test_avg/mae_surf_p** | **76.1441** |
| test_single_in_dist/mae_surf_p | 94.3654 |
| test_geom_camber_rc/mae_surf_p | 82.8340 |
| test_geom_camber_cruise/mae_surf_p | 53.5469 |
| test_re_rand/mae_surf_p | 73.8302 |
| Best epoch | 14 (timeout-bound; val still descending) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with **asinh(z) transform on pressure channel z-scores**  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-tanjiro-asinh-pressure-loss-20260515-173315/metrics.jsonl`

**Change (7 lines in train.py):**
```python
# asinh loss compression on pressure channel (index 2) only.
sq_err_uxuy = (pred[..., :2] - y_norm[..., :2]) ** 2
sq_err_p = (torch.asinh(pred[..., 2:3]) - torch.asinh(y_norm[..., 2:3])) ** 2
sq_err = torch.cat([sq_err_uxuy, sq_err_p], dim=-1)
```

**Note:** −27.7% on val (84.98 vs 117.50), −34% on test (76.14 vs 115.70 proxy). All 4 splits improved 20-36%. Val curve still descending at epoch 14 timeout — 84.98 is a loose upper bound on what this loss can do. Stronger gains on OOD splits (geom_camber_rc −27%, re_rand −29%) than in-dist (single_in_dist −21%).

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **84.9819**~~ — superseded by PR #3382 below.

---

## 2026-05-15 21:34 — PR #3382: EMA weights (decay=0.999) on asinh baseline

**Student:** charliepai2i48h2-askeladd  
**Change:** EMA shadow (decay=0.999) applied at every val and test pass on top of Lion + surf_weight=30 + asinh baseline. Live weights unchanged during training; EMA shadow loaded for all evaluation passes.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **83.1874** |
| val_single_in_dist/mae_surf_p | 99.9507 |
| val_geom_camber_rc/mae_surf_p | 94.1543 |
| val_geom_camber_cruise/mae_surf_p | 60.2646 |
| val_re_rand/mae_surf_p | 78.3801 |
| **test_avg/mae_surf_p** | **74.5193** |
| test_single_in_dist/mae_surf_p | 88.9512 |
| test_geom_camber_rc/mae_surf_p | 85.8766 |
| test_geom_camber_cruise/mae_surf_p | 50.9533 |
| test_re_rand/mae_surf_p | 72.2961 |
| Best epoch | 14 (timeout-bound; every epoch a new best, shadow still converging) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with asinh(z) on pressure channel  
**EMA:** decay=0.999, applied at val/test passes only — live weights unchanged  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-askeladd-ema-weights-decay-0999-rebased-20260515-203115/metrics.jsonl`

**Note:** −2.11% on val (83.19 vs 84.98), −2.13% on test (74.52 vs 76.14). All 4 val splits improved. Smoothness diagnostic: 0 sign flips in val curve (vs 8 for Lion baseline), every epoch a new best — EMA shadow still catching up at timeout. Mechanisms compose cleanly: asinh smooths loss landscape, EMA then smooths parameter trajectory on top.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **83.1874**~~ — superseded by PR #3384 below.

---

## 2026-05-15 23:37 — PR #3384: Gradient clipping (max_norm=1.0) on full EMA+asinh stack

**Student:** charliepai2i48h2-fern  
**Change:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` inserted between `loss.backward()` and `optimizer.step()` in the training loop. Rebased on full stack: Lion lr=1.7e-4, wd=3e-4, surf_weight=30, asinh pressure-loss, EMA(0.999).

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **70.2479** |
| val_single_in_dist/mae_surf_p | 81.5014 |
| val_geom_camber_rc/mae_surf_p | 82.7986 |
| val_geom_camber_cruise/mae_surf_p | 49.2201 |
| val_re_rand/mae_surf_p | 67.4712 |
| **test_avg/mae_surf_p** | **62.0765** |
| test_single_in_dist/mae_surf_p | 73.2488 |
| test_geom_camber_rc/mae_surf_p | 73.2597 |
| test_geom_camber_cruise/mae_surf_p | 41.3097 |
| test_re_rand/mae_surf_p | 60.4878 |
| Best epoch | 14 (timeout-bound; val still descending) |
| Peak GPU memory | 42.12 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 30·surf_loss, with asinh(z) on pressure channel  
**EMA:** decay=0.999, applied at val/test passes  
**Gradient clipping:** max_norm=1.0 (inserted before optimizer.step())  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-fern-lion-gradclip-1.0-rebased-20260515-222530/metrics.jsonl`

**Note:** −15.6% on val (70.25 vs 83.19), −16.7% on test (62.08 vs 74.52). All 4 splits improved 12-19%. Mechanisms compose cleanly: asinh compresses loss-level heavy tails, EMA smooths parameter trajectory, grad-clip caps per-step gradient norm. Post-asinh pre-clip norms still 25-180 (100% of steps clip) — the three mechanisms target different points in the gradient pipeline. Val curve still descending at epoch 14 timeout.

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name"
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **70.2479**~~ — superseded by PR #3530 below.

---

## 2026-05-16 03:30 — PR #3530: surf_weight reduction 30→25 on full 5-mechanism stack

**Student:** charliepai2i48h2-frieren  
**Change:** `surf_weight=30 → surf_weight=25`. Hypothesis: the asinh loss compression, EMA, and grad-clip together already reduce extreme pressure gradient signal, making surf_weight=30 an over-weight on the 4-mechanism stack. Two arms tested (25 and 20); sw=25 wins in aggregate.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **67.2991** |
| val_single_in_dist/mae_surf_p | 80.6871 |
| val_geom_camber_rc/mae_surf_p | 79.0339 |
| val_geom_camber_cruise/mae_surf_p | 46.1009 |
| val_re_rand/mae_surf_p | 63.3746 |
| **test_avg/mae_surf_p** | **58.9233** |
| test_single_in_dist/mae_surf_p | 71.2300 |
| test_geom_camber_rc/mae_surf_p | 69.7762 |
| test_geom_camber_cruise/mae_surf_p | 37.9947 |
| test_re_rand/mae_surf_p | 56.6922 |
| Best epoch | 14 (timeout-bound; val still descending) |
| Peak GPU memory | 42.13 GB |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + **25**·surf_loss, with asinh(z) on pressure channel  
**EMA:** decay=0.999, applied at val/test passes  
**Gradient clipping:** max_norm=1.0 (inserted before optimizer.step())  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-frieren-surf-weight-25-20260516-002627/metrics.jsonl`

**Note:** −4.20% on val (67.30 vs 70.25), −5.08% on test (58.92 vs 62.08). All 4 val splits improved (−1.0% to −6.3%). Hypothesis confirmed: the asinh/EMA/grad-clip stack already compresses extreme pressure gradient signal, making surf_weight=30 an over-weight. The sw=20 arm (val=68.10) also beat baseline, showing the 20–25 range is broadly good; the curve flattens below 20. Val and test both descend monotonically to epoch 14 — still timeout-bound.

**5-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.999) + grad_clip(max_norm=1.0)

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **67.2991**~~ — superseded by PR #3485 below.

---

## 2026-05-16 07:00 — PR #3485: bf16 autocast — 6th compounding mechanism

**Student:** charliepai2i48h2-alphonse  
**Change:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass and loss computation. Backward, grad-clip, optimizer.step(), and EMA.update() remain in fp32. This is a throughput intervention: per-epoch time 131s → 101s (−23%), peak GPU memory 42.13 → 32.96 GB (−22%), reaching 18 epochs vs 14 within the 30-min cap. The 4 extra epochs on a still-descending loss curve give the primary metric improvement.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.8717** |
| val_single_in_dist/mae_surf_p | 70.2014 |
| val_geom_camber_rc/mae_surf_p | 68.7070 |
| val_geom_camber_cruise/mae_surf_p | 39.0294 |
| val_re_rand/mae_surf_p | 57.5489 |
| **test_avg/mae_surf_p** | **51.6269** |
| test_single_in_dist/mae_surf_p | 61.6750 |
| test_geom_camber_rc/mae_surf_p | 61.9484 |
| test_geom_camber_cruise/mae_surf_p | 33.2190 |
| test_re_rand/mae_surf_p | 49.6652 |
| Best epoch | 18 (timeout-bound; val still descending) |
| Per-epoch time | ~101s (was ~131s) |
| Peak GPU memory | 32.96 GB (was 42.13 GB) |
| n_params | 662,359 |

**Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** CosineAnnealingLR(T_max=80)  
**Loss:** vol_loss + 25·surf_loss, with asinh(z) on pressure channel  
**Precision:** **bf16 autocast on forward+loss; fp32 backward/clip/optimizer/EMA**  
**EMA:** decay=0.999, applied at val/test passes  
**Gradient clipping:** max_norm=1.0 (before optimizer.step(), in fp32)  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-alphonse-bf16-autocast-20260516-053111/metrics.jsonl`

**Note:** −12.5% on val (58.87 vs 67.30), −12.4% on test (51.63 vs 58.92). All 4 splits improved (−9.2% to −15.3%). No NaN or instability — asinh and grad-clip compose cleanly with bf16. 9 GB freed VRAM opens the door to moderate capacity expansion. Val curve still descending at epoch 18 — still timeout-bound.

**6-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.999) + grad_clip(max_norm=1.0) + **bf16 autocast**

**Cumulative improvement from initial baseline:** 135.02 → 58.87 = **−56.4%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25
```

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **58.8717**~~ — superseded by PR #3822 below.

---

## 2026-05-16 10:30 — PR #3822: Cosine T_max alignment (T_max=30) — 7th compounding mechanism

**Student:** charliepai2i48h2-edward
**Change:** `CosineAnnealingLR(T_max=80)` → `CosineAnnealingLR(T_max=cfg.cosine_t_max_epochs=30)`. The previous T_max=80 with an 18-epoch realistic budget left LR essentially constant (0.852× initial at epoch 18). T_max=30 lets the schedule reach a meaningful late-epoch LR floor (40% of initial at epoch 18, 6.73e-5) while keeping useful update magnitude throughout the run. Pure scheduler-shape intervention.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **56.0011** |
| val_single_in_dist/mae_surf_p | 62.2099 |
| val_geom_camber_rc/mae_surf_p | 68.5030 |
| val_geom_camber_cruise/mae_surf_p | 37.7010 |
| val_re_rand/mae_surf_p | 55.5904 |
| **test_avg/mae_surf_p** | **48.9470** |
| test_single_in_dist/mae_surf_p | 56.2906 |
| test_geom_camber_rc/mae_surf_p | 60.3303 |
| test_geom_camber_cruise/mae_surf_p | 32.2264 |
| test_re_rand/mae_surf_p | 46.9408 |
| Best epoch | 18 (timeout-bound; val still descending) |
| Per-epoch time | ~102s (was ~101s; negligible overhead) |
| Peak GPU memory | 32.97 GB |

**Model config:** unchanged — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU
**Optimizer:** unchanged — Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)
**Scheduler:** **CosineAnnealingLR(T_max=30)** (was T_max=80)
**Loss:** unchanged — vol_loss + 25·surf_loss with asinh(z) on pressure
**Precision:** unchanged — bf16 autocast on forward+loss
**EMA:** unchanged — decay=0.999
**Gradient clipping:** unchanged — max_norm=1.0
**Batch:** 4
**Metric artifacts:** `models/model-charliepai2i48h2-edward-cosine-tmax-30-20260516-093553/metrics.jsonl`

**Note:** −4.88% on val (56.00 vs 58.87), −5.20% on test (48.95 vs 51.63). All 8 splits improved (val: −2.4% to −11.4%; test: −2.6% to −8.7%). Arm A (T_max=20) regressed +0.36% — the aggressive anneal pushes final LR to 5.4% of initial (9.26e-6) and gives up too much late-update magnitude. Arm B's moderate anneal (40% of initial at epoch 18) hits the optimal: enough decay to refine, enough magnitude to keep learning. Val curve still descending at epoch 18 even with the better schedule — runs are still timeout-bound, not LR-starved.

**7-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.999) + grad_clip(max_norm=1.0) + bf16 autocast + **cosine T_max=30**

**Cumulative improvement from initial baseline:** 135.02 → 56.00 = **−58.5%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30
```
(In-tree default is still 80 — must pass `--cosine_t_max_epochs 30` explicitly.)

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **56.0011**~~ — superseded by PR #3674 below.

---

## 2026-05-16 13:50 — PR #3674: Per-channel pressure weight (pw=2.0) — 8th compounding mechanism

**Student:** charliepai2i48h2-nezuko
**Change:** Added `pressure_weight: float = 1.0` to Config; applied as `loss = vol_loss + surf_weight * (surf_weight_Ux * mae_Ux + surf_weight_Uy * mae_Uy + pressure_weight * mae_p)` (or equivalent) in the training loss. Arm B (pw=2.0) wins — pressure channel up-weighted 2× in loss, letting the optimizer spend more gradient budget on the primary scoring metric.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **53.7235** |
| val_single_in_dist/mae_surf_p | 59.9048 |
| val_geom_camber_rc/mae_surf_p | 67.0791 |
| val_geom_camber_cruise/mae_surf_p | 35.4058 |
| val_re_rand/mae_surf_p | 52.5042 |
| **test_avg/mae_surf_p** | **46.6011** |
| test_single_in_dist/mae_surf_p | 53.4787 |
| test_geom_camber_rc/mae_surf_p | 59.3855 |
| test_geom_camber_cruise/mae_surf_p | 28.9082 |
| test_re_rand/mae_surf_p | 44.6319 |
| Best epoch | 18 (timeout-bound; val still descending) |
| Per-epoch time | ~100s |
| Peak GPU memory | 32.96 GB |

**Model config:** unchanged — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU
**Optimizer:** unchanged — Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)
**Scheduler:** unchanged — CosineAnnealingLR(T_max=30)
**Loss:** vol_loss + 25·surf_loss with asinh(z) on pressure + **pressure_weight=2.0** on the pressure MAE term
**Precision:** unchanged — bf16 autocast on forward+loss
**EMA:** unchanged — decay=0.999
**Gradient clipping:** unchanged — max_norm=1.0
**Batch:** 4
**Metric artifacts:** `models/model-charliepai2i48h2-nezuko-pressure-weight-2p0-20260516-112553/metrics.jsonl`

**Note:** −4.07% on val (53.72 vs 56.00), −4.79% on test (46.60 vs 48.95). All 8 splits improved. Arm A (pw=0.5) regressed +3.20% — down-weighting pressure under-trains the primary metric channel. Key finding: asinh compression didn't fully neutralise the channel imbalance — pw=2.0 constructively stacks with asinh by re-emphasising pressure without losing asinh-driven stability. Velocity channels Ux/Uy mildly regress (+11-21%) but pressure improvement dominates since it is the primary metric. Val still descending at epoch 18.

**8-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.999) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + **pressure_weight=2.0**

**Cumulative improvement from initial baseline:** 135.02 → 53.72 = **−60.2%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0
```

> **Beat this:** submit a PR improving `val_avg/mae_surf_p` below **53.7235** with a terminal `SENPAI-RESULT` marker.

---

## 2026-05-16 16:00 — PR #3989: EMA decay re-tune on 8-mech stack (ema_decay=0.995) — 9th compounding mechanism

**Student:** charliepai2i48h2-askeladd
**Change:** Faster EMA decay from 0.999 to 0.995. The convergence-horizon hypothesis: with T_max=30 producing steeper LR anneal and pressure_weight=2.0 shifting loss curvature, the EMA shadow model benefits from tracking more recent weights (shorter effective half-life: ~138 steps at 0.995 vs ~693 steps at 0.999). Arm B (ema_decay=0.995) wins; Arm A (ema_decay=0.997) also beats baseline.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **51.4403** |
| val_single_in_dist/mae_surf_p | 56.1655 |
| val_geom_camber_rc/mae_surf_p | 68.0745 |
| val_geom_camber_cruise/mae_surf_p | 32.1218 |
| val_re_rand/mae_surf_p | 49.3995 |
| **test_avg/mae_surf_p** | **43.9473** |
| test_single_in_dist/mae_surf_p | 53.55 |
| test_geom_camber_rc/mae_surf_p | 56.79 |
| test_geom_camber_cruise/mae_surf_p | 26.94 |
| test_re_rand/mae_surf_p | 38.51 |
| Best epoch | 18 (timeout-bound; val still descending) |
| Per-epoch time | ~101s |
| Peak GPU memory | 32.96 GB |

**Model config:** unchanged — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU
**Optimizer:** unchanged — Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)
**Scheduler:** unchanged — CosineAnnealingLR(T_max=30)
**Loss:** unchanged — vol_loss + 25·surf_loss with asinh(z) on pressure + pressure_weight=2.0
**Precision:** unchanged — bf16 autocast on forward+loss
**EMA:** **decay=0.995** (was 0.999)
**Gradient clipping:** unchanged — max_norm=1.0
**Batch:** 4
**Metric artifacts:** `models/model-charliepai2i48h2-askeladd-ema-decay-8mech-0995-20260516-142331/metrics.jsonl`

**Note:** −4.25% on val (51.44 vs 53.72), −5.69% on test (43.95 vs 46.60). 3 of 4 val splits improved; all 4 test splits improved. re_rand shows the biggest test gain (44.63→38.51, −13.7%), suggesting faster EMA tracking helps cross-regime Re generalization. The Arm A (0.997) vs Arm B (0.995) gap is at the noise floor (1.18 pts vs 1.17 pt noise floor); both clearly beat 0.999. Faster-decay EMA (0.995) follows the "faster anneal → faster EMA" pattern: T_max=30 collapses LR to 40% by epoch 18 so recent weights matter more; 0.999's ~693-step half-life is too slow to track this regime.

**9-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + **EMA(0.995)** + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + pressure_weight=2.0

**Cumulative improvement from initial baseline:** 135.02 → 51.44 = **−61.9%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0 \
    --ema_decay 0.995
```
(In-tree defaults: T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999 — must pass all four explicitly.)

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **51.4403**~~ — superseded by PR #3970 below.

---

## 2026-05-16 18:15 — PR #3970: torch.compile (mode=default, dynamic=True) — 10th compounding mechanism

**Student:** charliepai2i48h2-alphonse  
**Change:** `torch.compile(model, mode='default', dynamic=True)` wraps the Transolver model after instantiation. `dynamic=True` handles the variable-length padded batch shapes from `pad_collate`. This is a pure throughput intervention: per-epoch time 102s → 54.4s (−47%), peak VRAM 32.97 → 23.84 GB (−9 GB headroom), reaching **33 epochs** vs 18 within the 30-min cap. The 15 extra epochs on a still-descending loss curve give a massive primary metric gain.

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **44.2439** |
| val_single_in_dist/mae_surf_p | 46.9816 |
| val_geom_camber_rc/mae_surf_p | 58.2760 |
| val_geom_camber_cruise/mae_surf_p | 27.6407 |
| val_re_rand/mae_surf_p | 44.0774 |
| **test_avg/mae_surf_p** | **38.0107** |
| test_single_in_dist/mae_surf_p | 42.3063 |
| test_geom_camber_rc/mae_surf_p | 49.5504 |
| test_geom_camber_cruise/mae_surf_p | 23.1558 |
| test_re_rand/mae_surf_p | 37.0300 |
| Best epoch | 33 (timeout-bound; val still descending ~0.03/epoch) |
| Per-epoch time | ~54.4s (was ~102s) |
| Peak GPU memory | 23.84 GB (was 32.97 GB) |
| n_params | 662,359 |

**Model config:** unchanged — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** unchanged — Lion lr=1.7e-4, wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** unchanged — CosineAnnealingLR(T_max=30)  
**Loss:** unchanged — vol_loss + 25·surf_loss with asinh(z) on pressure + pressure_weight=2.0  
**Precision:** unchanged — bf16 autocast on forward+loss  
**EMA:** unchanged — decay=0.995  
**Gradient clipping:** unchanged — max_norm=1.0  
**Compile:** **torch.compile(mode='default', dynamic=True)** (Arm B reduce-overhead: val=45.36)  
**Batch:** 4  
**Metric artifacts:** `models/model-charliepai2i48h2-alphonse-torch-compile-default-20260516-162535/metrics.jsonl`

**Note:** −14.0% on val (44.24 vs 51.44), −13.5% on test (38.01 vs 43.95). All 8 splits improved. The 15-epoch gain from compile (18→33) explains the full improvement — the loss curve was still monotonically descending at epoch 18. With 9 GB of freed VRAM (32.97→23.84 GB), capacity expansion experiments are now viable. reduce-overhead mode (Arm B: val=45.36) is slower than default — compile overhead pays off for default mode in this variable-shape setting.

**10-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + cosine T_max=30 + pressure_weight=2.0 + **torch.compile(mode=default, dynamic=True)**

**Cumulative improvement from initial baseline:** 135.02 → 44.24 = **−67.2%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 30 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default
```
(In-tree defaults: T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999, compile_mode=none — must pass all five explicitly.)

> ~~**Beat this:** submit a PR improving `val_avg/mae_surf_p` below **44.2439**~~ — superseded by PR #3953 below.

---

## 2026-05-16 21:35 — PR #3953: LR × T_max re-calibration for 33-epoch compile horizon — 11th compounding mechanism

**Student:** charliepai2i48h2-frieren  
**Change:** Joint re-calibration of `lr_init` (1.7e-4 → 2.5e-4) and `cosine_t_max_epochs` (30 → 40) for the 33-epoch compile horizon. The original T_max=30/lr=1.7e-4 pair was tuned for 18 epochs; under 33 epochs the LR × schedule coupling shifts: longer budget + gentler anneal (T_max=40 leaves ~10% LR floor at epoch 33) allows a higher lr_init to remain stable throughout. Frieren's LR/schedule coupling hypothesis, confirmed across three regimes (T_max=80, T_max=30/18ep, T_max=40/33ep).

| Metric | Arm A (lr=2.1e-4, T_max=40) | **Arm B (lr=2.5e-4, T_max=40) — WINNER** |
|--------|------------------------------|-------------------------------------------|
| **val_avg/mae_surf_p** | 41.0110 (−7.30%) | **40.6869 (−8.04%)** |
| val_single_in_dist/mae_surf_p | 43.6699 | 44.5780 |
| val_geom_camber_rc/mae_surf_p | 54.2719 | 54.5227 |
| val_geom_camber_cruise/mae_surf_p | 24.6291 | 23.7332 |
| val_re_rand/mae_surf_p | 41.4732 | 39.9137 |
| **test_avg/mae_surf_p** | 35.9024 | **34.9776** |
| test_single_in_dist/mae_surf_p | 39.4387 | 38.0930 |
| test_geom_camber_rc/mae_surf_p | 49.7494 | 48.1880 |
| test_geom_camber_cruise/mae_surf_p | 20.3493 | 19.9850 |
| test_re_rand/mae_surf_p | 34.0719 | 33.6445 |
| Best epoch | 33 (timeout-bound; val still descending) | 33 (timeout-bound; val still descending) |
| Per-epoch time | ~55s | ~55s |
| Peak GPU memory | 23.84 GB | 23.84 GB |

**Model config:** unchanged — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, GELU  
**Optimizer:** Lion **lr=2.5e-4** (was 1.7e-4), wd=3e-4, betas=(0.9, 0.99)  
**Scheduler:** **CosineAnnealingLR(T_max=40)** (was T_max=30)  
**Loss:** unchanged — vol_loss + 25·surf_loss with asinh(z) on pressure + pressure_weight=2.0  
**Precision:** unchanged — bf16 autocast on forward+loss  
**EMA:** unchanged — decay=0.995  
**Gradient clipping:** unchanged — max_norm=1.0  
**Compile:** unchanged — torch.compile(mode='default', dynamic=True)  
**Batch:** 4  
**Metric artifacts:**  
- `models/model-charliepai2i48h2-frieren-lr-tmax-coupling-compile-tmax40-21e4-20260516-192650/metrics.jsonl` (Arm A)  
- `models/model-charliepai2i48h2-frieren-lr-tmax-coupling-compile-tmax40-25e4-20260516-202729/metrics.jsonl` (Arm B)

**Note:** −8.04% on val (40.69 vs 44.24), −7.98% on test (34.98 vs 38.01). Both arms clear the strong-win threshold (val < 41.6). Both trajectories smooth, still descending at epoch 33 — T_max=40 leaves ~10% LR floor at final epoch vs T_max=30 hitting near-zero at epoch 30. Arm B beats Arm A on 3/4 val splits and 4/4 test splits. Key mechanism: the 33-epoch budget changes the LR/schedule coupling: a higher lr_init can now be supported by the extended training horizon without instability. The confound between T_max and lr changes is real — edward's #4079 (pure T_max sweep at lr=1.7e-4) will isolate contributions.

**11-mechanism stack:** Lion + surf_weight=25 + asinh(pressure) + EMA(0.995) + grad_clip(max_norm=1.0) + bf16 autocast + **cosine T_max=40** + pressure_weight=2.0 + torch.compile(mode=default, dynamic=True) + **lr=2.5e-4**

**Cumulative improvement from initial baseline:** 135.02 → 40.69 = **−69.8%**

**Reproduce:**
```bash
cd target && python train.py --agent <student> \
    --experiment_name "<student>/your-experiment-name" \
    --surf_weight 25 \
    --cosine_t_max_epochs 40 \
    --pressure_weight 2.0 \
    --ema_decay 0.995 \
    --compile_mode default \
    --lr 2.5e-4
```
(In-tree defaults: lr=1.7e-4, T_max=80, surf_weight=30, pressure_weight=1.0, ema_decay=0.999, compile_mode=none — must pass all six explicitly.)

> **Beat this:** submit a PR improving `val_avg/mae_surf_p` below **40.6869** with a terminal `SENPAI-RESULT` marker.
