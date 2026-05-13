# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best

### 2026-05-13 12:10 — PR #1968: [max_lr-1e-3] OneCycleLR max_lr 8e-4→1e-3 (+25% peak) on bs=1 (thorfinn)

- **`val_avg/mae_surf_p`:** **59.39** (best epoch 21/21 — still descending at schedule end)
- **`test_avg/mae_surf_p`:** **51.40** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=58.84, geom_camber_rc=70.02, geom_camber_cruise=46.67, re_rand=62.03
- **Per-split surface-p MAE (test):** single_in_dist=51.49, geom_camber_rc=63.38, geom_camber_cruise=38.23, re_rand=52.51
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, wd=1e-4, surf_weight=5.0, batch_size=1, seed=42, OneCycleLR(max_lr=1e-3, pct_start=0.1, anneal_strategy='cos', div_factor=25, final_div_factor=10, T_MAX_EPOCHS=21), AdamW(0.9,0.999, lr=5e-4 initial), smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0), bf16, unified_pos=True, ref=8`
- **Key change:** `max_lr=8e-4 → 1e-3` (+25%). All other schedule params identical. Initial LR = 1e-3/25 = 4e-5; final LR = 1e-3/250 = 4e-6.
- **Improvement vs #2014 (OneCycleLR max_lr=8e-4, val=60.98):** val −1.59 (−2.61%), test −1.08 (−2.06%)
- **Split detail:** single_in_dist improved most (val −4.50 / test −3.26); rc improved val −2.77 / test −0.70; cruise slightly worse val +0.77 / test +0.80; re_rand essentially flat val +0.12 / test −1.15. Single_in_dist and rc drove the win; cruise/re_rand are at similar floor.
- **Training dynamics:** Epoch-1 max grad-norm ~100 (vs ~80 at 8e-4), fully absorbed by clip. Val monotonically decreased ep1→21; best epoch=21 (final), still descending — the model hasn't saturated the 30-min budget.
- **Metric artifacts:** `models/model-charliepai2g48h4-thorfinn-max-lr-1e-3-20260513-112139/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-thorfinn --experiment_name "charliepai2g48h4-thorfinn/max-lr-1e-3"`
- **17th effective merge. SUB-60 VAL MILESTONE HIT.**

**Open questions / next steps after this merge:**
- **max_lr upper bracket still open:** 1e-3 beats 8e-4; is 1.2e-3 even better? Or does instability kick in above 1e-3?
- **Best epoch = final epoch** (still descending) — suggests more epochs or higher max_lr could push val further.
- **cruise split slightly worsened** (+0.77 val / +0.80 test) — higher peak may overshoot the narrow camber-cruise geometry distribution. Architecture (n_hidden=192, edward #2217) may help here.
- **EMA null result** (askeladd #1540, +0.05 val on this base) — confirmed EMA is moot on OneCycleLR, closing.
- **rc split (hardest at val=70.02)** — still the dominant contributor to val avg. Architecture width or pct_start tuning may help.

---

### 2026-05-13 11:05 — PR #2014: [onecycle-lr] OneCycleLR(max_lr=8e-4) replacing SequentialLR on bs=1 (nezuko)

- **`val_avg/mae_surf_p`:** **60.98** (best epoch 20/21)
- **`test_avg/mae_surf_p`:** **52.48** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=63.34, geom_camber_rc=72.79, geom_camber_cruise=45.90, re_rand=61.91
- **Per-split surface-p MAE (test):** single_in_dist=54.75, geom_camber_rc=64.08, geom_camber_cruise=37.43, re_rand=53.66
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, wd=1e-4, surf_weight=5.0, batch_size=1, seed=42, OneCycleLR(max_lr=8e-4, pct_start=0.1, anneal_strategy='cos', div_factor=25, final_div_factor=10, T_MAX_EPOCHS=21), AdamW(0.9,0.999, lr=5e-4 initial), smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0), bf16, unified_pos=True, ref=8`
- **Key change:** SequentialLR(LinearLR(warmup) → CosineAnnealingLR(T_max=17)) **replaced** by OneCycleLR. Single super-convergence schedule: lr rises from 3.2e-5 (=max_lr/25) to 8e-4 over first ~2.1 epochs (pct_start=0.1), then cosine-anneals to 3.2e-6 (=max_lr/250) over remaining 18.9 epochs. Stops exactly at T_MAX_EPOCHS=21. **Note: max_lr=8e-4 is hardcoded; cfg.lr=5e-4 is used as AdamW init but immediately overridden by OneCycleLR on step 1.**
- **Improvement vs #2012 (bs=1+beta=0.5, val=66.32):** val −5.34 (−8.06%), test −7.20 (−12.07%)
- **All 8 splits improved:** rc val −8.28 / test −10.12 (biggest), cruise val −3.21 / test −4.18 (smallest), single_in_dist −6.64/−6.95, re_rand −3.19/−7.54
- **Schedule note:** OneCycleLR with T_MAX_EPOCHS=21 solves the schedule-misalignment problem that the old SequentialLR had (cosine ending at ep18, model still improving at ep21). The super-convergence schedule is fully calibrated to the bs=1 30-min budget.
- **Metric artifacts:** `models/model-charliepai2g48h4-nezuko-onecycle-lr-bs1-20260513-100811/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-nezuko --experiment_name "charliepai2g48h4-nezuko/onecycle-lr-bs1"`
- **16th effective merge**

**Critical impact on in-flight experiments:**
- `cfg.lr` (default 5e-4) is now irrelevant — OneCycleLR overrides it from step 1. Experiments targeting `cfg.lr` changes (thorfinn #1968, alphonse #2106) are MOOT after rebase. Reassign with `max_lr` as the lever.
- `T_max` (CosineAnnealingLR) no longer exists. Experiments targeting T_max (edward #2162) are MOOT. Reassign.
- `beta=0.5` (loss shape) and EMA (askeladd) and β2 (tanjiro) are independent of schedule — those can rebase cleanly.

**Open questions after this merge:**
- **max_lr bracket:** 8e-4 untested against alternatives. What's the peak? Try 6e-4 (−25%) and 1e-3 (+25%).
- **EMA stacking:** Does EMA further smooth the OneCycleLR trajectory? askeladd #1540 testing.
- **beta bracket:** Does beta=0.25 still help under OneCycleLR? fern #2164 testing.
- **β2 with OneCycleLR:** The high-peak-then-anneal profile is different from cosine. Does β2=0.95 or 0.99 help? tanjiro #2125 testing.
- **pct_start bracket:** 10% peak (2.1 epochs). Too fast? 15% (3.15 epochs) may slow the initial rise and help the rc split.
- **T_MAX_EPOCHS=22?** If the budget allows, an extra epoch at the final very-low-lr plateau may help. But first confirm the current budget is fully used.

---

### 2026-05-13 10:00 — PR #2012: [loss-beta-0-5] Halve smooth_l1 beta 1.0→0.5 on bs=1 baseline (edward)

- **`val_avg/mae_surf_p`:** **66.32** (best epoch 21/21)
- **`test_avg/mae_surf_p`:** **59.68** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=69.98, geom_camber_rc=81.07, geom_camber_cruise=49.11, re_rand=65.10
- **Per-split surface-p MAE (test):** single_in_dist=61.70, geom_camber_rc=74.20, geom_camber_cruise=41.61, re_rand=61.20
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=1, seed=42, SequentialLR(LinearLR(1ep warmup) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0), bf16, unified_pos=True, ref=8`
- **Key change:** `beta=1.0 → beta=0.5` in training smooth_l1_loss only. Narrower quadratic zone means more residuals hit the L1 slope sooner — under our grad-clip normalized-step regime this reshapes which examples drive direction updates.
- **Improvement vs #2036 (bs=1+beta=1.0, val=70.30):** val −3.98 (−5.66%), test −1.71 (−2.79%)
- **3/4 val splits improved:** single_in_dist −4.17, cruise −4.56, re_rand −7.18 (largest); rc flat −0.04
- **3/4 test splits improved:** single_in_dist −3.12, cruise −3.12, re_rand −0.89; rc slight +0.28 (noise)
- **Metric artifacts:** `models/model-charliepai2g48h4-edward-loss-beta-0-5-bs1-20260513-091012/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-edward --experiment_name "charliepai2g48h4-edward/loss-beta-0-5-bs1"`
- **15th effective merge**

**Open questions after this merge:**
- **Beta bracket on bs=1:** 0.5 < 1.0 confirmed, but where is the optimum? Try beta ∈ {0.25, 0.75} to pin the curve shape.
- **Pure L1:** Natural limit beta→0 (F.l1_loss). Does the quadratic zone provide any benefit at all under bs=1+grad-clip?
- **EMA (askeladd #1540):** Still training. Expected to stack on top — loss-shape and EMA are independent mechanisms.
- **Beta-volume asymmetry:** Volume residuals (MAE ~3) vs surface residuals (MAE ~80) operate at very different scales. Independent betas per-component could compound.
- **LR retuning at bs=1:** lr=4e-4 bracket (alphonse #2106) still in flight.

---

### 2026-05-13 09:00 — PR #2036: [batch-size-1] Extreme batch bracket: batch_size 2→1 (alphonse)

- **`val_avg/mae_surf_p`:** **70.30** (best epoch 18/19)
- **`test_avg/mae_surf_p`:** **61.39** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=74.15, geom_camber_rc=81.11, geom_camber_cruise=53.67, re_rand=72.28
- **Per-split surface-p MAE (test):** single_in_dist=64.82, geom_camber_rc=73.92, geom_camber_cruise=44.73, re_rand=62.09
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=1, seed=42, SequentialLR(LinearLR(1ep warmup) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0), bf16, unified_pos=True, ref=8`
- **Metric artifacts:** `models/model-charliepai2g48h4-alphonse-batch-size-1-20260513-075224/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-alphonse --experiment_name "charliepai2g48h4-alphonse/batch-size-1"`

---

### 2026-05-13 07:30 — PR #1972: [batch-size-2] Halve batch size 4→2 (alphonse)

- **`val_avg/mae_surf_p`:** **76.24** (best epoch 18/19)
- **`test_avg/mae_surf_p`:** **66.85** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=81.78, geom_camber_rc=87.06, geom_camber_cruise=59.39, re_rand=76.74
- **Per-split surface-p MAE (test):** single_in_dist=70.94, geom_camber_rc=77.56, geom_camber_cruise=50.63, re_rand=68.28
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=2, seed=42, SequentialLR(LinearLR(1ep warmup) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW(0.9,0.999), smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0), bf16, unified_pos=True, ref=8`
- **Key change:** batch_size 4→2. n_batches_per_epoch doubled (375→750) — same wall-clock per epoch (~95s), 2x optimizer steps. VRAM halved (~17 GB). 19 epochs completed (vs 18 for bs=4).
- **Improvement vs directly-comparable reference (#1812 lr-warmup-1ep, bs=4):** val −6.32 (−7.65%), test −7.28 (−9.82%)
- **All 8 splits improved:** single_in_dist val −8.62 / test −10.02 (biggest), rc val −4.34 / test −4.48 (smallest)
- **Metric artifacts:** `models/model-charliepai2g48h4-alphonse-batch-size-2-20260513-060609/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-alphonse --experiment_name "charliepai2g48h4-alphonse/batch-size-2"`

**Open questions after this merge:**
- Does lr=7e-4 (thorfinn #1968) still win on top of bs=2? Thorfinn's result (79.77) beat bs=4 baseline but not this new bs=2 baseline. Sent back to rerun with bs=2 + lr=7e-4.
- Does lr=8e-4 (nezuko #2014 onecycle-lr) win? Both schedule shape and peak LR differ from current.
- Does EMA (askeladd #1540) stack with bs=2? EMA currently training on bs=4 baseline. Results pending.
- Optimal batch size bracket: bs=2 wins; bs=1 assigned to alphonse to probe lower.

---

### 2026-05-13 06:00 — PR #1812: [lr-warmup-1ep] 1-epoch linear warmup + cosine annealing (thorfinn)

- **`val_avg/mae_surf_p`:** **82.56** (best epoch 18/18)
- **`test_avg/mae_surf_p`:** **74.13** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=90.40, geom_camber_rc=91.39, geom_camber_cruise=66.68, re_rand=81.77
- **Per-split surface-p MAE (test):** single_in_dist=80.96, geom_camber_rc=82.04, geom_camber_cruise=56.78, re_rand=76.77
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=4, seed=42, SequentialLR(LinearLR(start_factor=0.01, 1-epoch warmup) → CosineAnnealingLR(T_max=17ep, eta_min=5e-5)), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`
- **Key change:** 1-epoch linear LR warmup (5e-6→5e-4) prepended to cosine schedule. Warmup damps epoch-1 AdamW momentum corruption: epoch-1 mean grad-norm drops from 30-1000+ to 8.66 (max 35.5), preventing chaotic momentum buffer seeding. Cosine portion retains eta_min=5e-5 floor. By epoch 17 the model is at val=85.00 vs ~87.29 in the pure-cosine best — warmup positions the model better for the final descent.
- **Improvement vs directly-comparable reference (#1855 eta_min=5e-5, pure cosine):** val −1.39 (−1.65%), test −0.57 (−0.76%)
- **3 of 4 val splits improved:** single_in_dist −3.05, geom_camber_cruise −0.38, re_rand −2.20; geom_camber_rc +0.06 (flat)
- **Metric artifacts:** `models/model-charliepai2g48h4-thorfinn-lr-warmup-1ep-20260513-051740/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-thorfinn --experiment_name "charliepai2g48h4-thorfinn/lr-warmup-1ep"`

**Open questions after this merge:**
- Does lr=7e-4 + warmup beat lr=5e-4 + warmup? Warmup provides momentum-corruption protection — higher peak LR may now be safe. Thorfinn assigned #XXXX to test.
- Does warmup compound with EMA (askeladd #1540)? Expected to produce another sub-80 jump.
- Second seed validation: −1.39 is within σ≈8.5; confirming with seed 0 or 1 would distinguish signal from noise.

---

### 2026-05-13 04:05 — PR #1855: [eta-min-5e-5] Non-zero cosine LR floor eta_min=0.0→5e-5 (fern)

- **`val_avg/mae_surf_p`:** **83.95** (best epoch 18/18)
- **`test_avg/mae_surf_p`:** **74.70** (from best-val checkpoint, all 4 splits)
- **Per-split surface-p MAE (val):** single_in_dist=93.45, geom_camber_rc=91.33, geom_camber_cruise=67.06, re_rand=83.97
- **Per-split surface-p MAE (test):** single_in_dist=83.68, geom_camber_rc=81.29, geom_camber_cruise=56.91, re_rand=76.93
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=18, eta_min=5e-5), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`
- **Key change:** `eta_min=5e-5` (was 0.0). Non-zero LR floor prevents the cosine schedule from reaching zero at epoch 18. The model continued descending through the final epoch (val epoch 17→18: 87.29→83.95, Δ−3.34). Under grad-clip's normalized steps, even a small non-zero LR (5.34e-5 at epoch 18) produces meaningful updates — each clipped step contributes signal proportional to the LR.
- **Improvement vs directly-comparable reference (#1695 T_max=18, eta_min=0.0):** val −0.72 (−0.85%), test −0.24 (−0.32%)
- **3 of 4 val splits improved:** single_in_dist −2.80, geom_camber_rc −1.92, re_rand +0.19 (~flat); geom_camber_cruise +1.67 (slight regression)
- **Metric artifacts:** `models/model-charliepai2g48h4-fern-eta-min-5e-5-20260513-030913/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-fern --experiment_name "charliepai2g48h4-fern/eta-min-5e-5"`

**Open questions after this merge:**
- Is eta_min=5e-5 the optimum? Try eta_min=1e-4 (20% of lr) or eta_min=1e-5 (2%) to bracket.
- The model was still improving at epoch 18 — would more epochs help? T_max=19 or T_max=20 may squeeze more, but the 30-min wall-clock cap may not allow it.
- Layers-6 regressed under grad-clip (edward #1730, val=93.97 vs 84.67 baseline). Depth bottleneck may not be the lever at this architecture scale.

---

### 2026-05-13 03:00 — PR #1695: [tmax-18] Tune cosine T_max=15→18 to match achievable epoch count exactly (nezuko)

- **`val_avg/mae_surf_p`:** **84.67** (best epoch 18/18)
- **`test_avg/mae_surf_p`:** **74.94** (from best-val checkpoint, all 4 splits clean)
- **Per-split surface-p MAE (val):** single_in_dist=96.25, geom_camber_rc=93.25, geom_camber_cruise=65.39, re_rand=83.78
- **Per-split surface-p MAE (test):** single_in_dist=85.31, geom_camber_rc=83.17, geom_camber_cruise=55.11, re_rand=76.18
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=18, eta_min=0.0), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`
- **Key change:** `CosineAnnealingLR(T_max=18, eta_min=0.0)` (was T_max=15). Under the 30-min wall-clock cap, the model reliably reaches 18 epochs. Setting T_max=18 aligns the cosine schedule minimum with the last achievable epoch, ensuring lr→0 at the true end of training rather than 3 epochs early. Clean scheduling win — no capacity or loss changes.
- **Improvement vs directly-comparable reference (#1762 surf-weight-5, T_max=15):** val −5.91 (−6.5%), test −5.06 (−6.3%)
- **All 4 val splits improved:** single_in_dist −10.06, geom_camber_rc −5.59, geom_camber_cruise −3.96, re_rand −4.04
- **All 4 test splits improved:** single_in_dist −8.30, geom_camber_rc −3.69, geom_camber_cruise −3.26, re_rand −5.02
- **Metric artifacts:** `models/model-charliepai2g48h4-nezuko-tmax-18-20260513-021955/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-nezuko --experiment_name "charliepai2g48h4-nezuko/tmax-18"`

**Open questions after this merge:**
- Does T_max=19 or T_max=20 push further, or is 18 at saturation given the 30-min cap? (18 epochs complete cleanly; 19 may be squeezable depending on mini-batch time variance.)
- Layers-6 edward rerun on new baseline — previously extrapolated ~80.5 from Huber-base delta; with T_max=18 now standard, target shifts to sub-79.
- EMA on current recipe (askeladd #1540) is highest-priority stacking test; estimated sub-79 with EMA+current recipe.

---

### 2026-05-13 02:10 — PR #1762: [surf-weight-5] Halve surface loss weight 10→5 (tanjiro)

- **`val_avg/mae_surf_p`:** **90.58** (best epoch 17/18)
- **`test_avg/mae_surf_p`:** **80.00** (from best-val checkpoint, all 4 splits clean)
- **Per-split surface-p MAE (val):** single_in_dist=106.31, geom_camber_rc=98.84, geom_camber_cruise=69.35, re_rand=87.82
- **Per-split surface-p MAE (test):** single_in_dist=93.61, geom_camber_rc=86.86, geom_camber_cruise=58.37, re_rand=81.18
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=5.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=F.smooth_l1_loss(beta=1.0), clip_grad_norm_(max_norm=1.0)`
- **Key change:** `surf_weight: float = 5.0` (was 10.0 at train.py:383). With grad-clip normalizing every step, surface gradients no longer need to be cranked up to dominate the loss — clipping equalizes step magnitudes. Halving surf_weight better balances surface-vs-volume residuals in the loss, improving all 4 splits simultaneously.
- **Gradient dynamics under new recipe:** Pre-clip gradient norms drop to mean 2–5 (vs 33–106 with surf_weight=10). Every step still clipped (norms >> max_norm=1.0). Effective step magnitude ~half what it was.
- **Improvement vs directly-comparable reference (#1696 grad-clip-1.0, surf_weight=10):** val −6.20 (−6.4%), test −6.56 (−7.6%)
- **All 4 val + all 4 test splits improved** (single_in_dist val −4.07, geom_camber_rc −6.50, geom_camber_cruise −5.79, re_rand −8.45)
- **Metric artifacts:** `models/model-charliepai2g48h4-tanjiro-surf-weight-5-20260513-011227/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-tanjiro --experiment_name "charliepai2g48h4-tanjiro/surf-weight-5"`

**Open questions after this merge:**
- Is surf_weight=5 the optimum or can we push lower? surf_weight ∈ {2, 3} pending on next tanjiro assignment.
- max_norm=0.5 tested on old baseline (surf_weight=10, val=96.78) gave val=95.53 (Δ−1.3%). Needs rerun on new baseline (surf_weight=5) to confirm it still helps.
- All in-flight reruns (layers-6, log-cosh, EMA, T_max=18) will naturally test on new surf_weight=5 HEAD.

---

### 2026-05-13 01:19 — PR #1696: [grad-clip-1.0] Gradient clipping max_norm=1.0 (frieren)

- **`val_avg/mae_surf_p`:** **96.78** (best epoch 17/18)
- **`test_avg/mae_surf_p`:** **86.56** (from best-val checkpoint, all 4 splits clean)
- **Per-split surface-p MAE (val):** single_in_dist=110.38, geom_camber_rc=105.34, geom_camber_cruise=75.14, re_rand=96.27
- **Per-split surface-p MAE (test):** single_in_dist=98.34, geom_camber_rc=94.63, geom_camber_cruise=63.51, re_rand=89.75
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=MSE` + **`clip_grad_norm_(model.parameters(), max_norm=1.0)`**
- **Key change:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` inserted between `loss.backward()` and `optimizer.step()`. Pre-clip gradient norms ranged 33–1000+ (mean 33–106 across epochs), meaning **every step was effectively clipped** — this acts as gradient-direction-following (unit step magnitude scaled by lr), fundamentally changing optimization dynamics rather than just outlier suppression.
- **Run context:** Frieren's run was on the pre-Huber HEAD (started 2026-05-12 23:54 before Huber merge at ~00:08). The squash merge adds grad-clip on top of the current Huber HEAD, giving an untested Huber+grad-clip combination in `train.py`. Huber+grad-clip confirmation is expected from subsequent student runs.
- **Improvement vs directly-comparable reference (#1542 T_max=15 MSE, pre-Huber base):** val −18.03 (−15.7%), test −18.12 (−17.3%)
- **Improvement vs Huber-only best (#1374):** val −13.81 (−12.5%), test −15.72 (−15.4%)
- **All 4 splits improved** (single_in_dist −29.4 val, geom_camber_rc −15.3, geom_camber_cruise −12.6, re_rand −14.8)
- **Metric artifacts:** `models/model-charliepai2g48h4-frieren-grad-clip-1.0-20260512-235444/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-frieren --experiment_name "charliepai2g48h4-frieren/grad-clip-1.0"`

**Note on optimization dynamics:** With max_norm=1.0 and typical grad norms of 30–1000+, clipping scales every step down by 30–1000×. The optimizer is doing normalized gradient descent (AdamW-with-clipping ≈ adaptive scale-free SGD). This may explain the dramatic improvement — the learning rate is effectively per-step adaptive, not just per-parameter adaptive.

**Open questions after this merge:**
- Huber+grad-clip combination is now in `train.py` but untested end-to-end. Next runs from rebased students will measure this.
- Askeladd EMA rebase pending — first test of EMA+Huber+grad-clip stack.
- Cross-seed σ on grad-clip baseline needed after first confirmation run.
- Does max_norm=0.1 or 0.5 push further? Or is 1.0 already at saturation?

---

## Previous bests (chronological)

### 2026-05-13 00:08 — PR #1374: [huber-loss] Smooth L1 (Huber, beta=1.0) instead of MSE (edward)
- **val_avg/mae_surf_p:** 110.59 / **test:** 102.28
- Config: merged recipe + Huber(beta=1.0) + seed=42. All 4 splits clean.
- Artifact: `models/model-charliepai2g48h4-edward-huber-loss-20260512-231342/metrics.jsonl`

### 2026-05-12 23:25 — PR #1542: [cosine-trunc-t15] Truncate cosine T_max 50→15 (nezuko)
- **val_avg/mae_surf_p:** 114.81 / **test:** 104.68
- Config: merged recipe + T_max=15 + seed=42. Per-split val: single_in_dist=139.82, geom_camber_rc=120.59, geom_camber_cruise=87.75, re_rand=111.06
- Artifact: `models/model-charliepai2g48h4-nezuko-cosine-trunc-t15-merged-20260512-215533/metrics.jsonl`

### 2026-05-12 23:05 — PR #1577: [seed42-baseline] Seeding + surf_weight=10 rollback (alphonse)
- **val_avg/mae_surf_p:** 116.43 / **test:** 108.87
- Config: merged recipe (unified_pos + bf16 + scoring-fix), surf_weight=10, seed=42, T_max=50
- Adds determinism infrastructure; byte-identical across 2 runs.

### 2026-05-12 20:10 — PR #1512: [scoring-nan-fix] Default config + NaN-fix patch (fern)

- **`val_avg/mae_surf_p`:** **123.99** (best epoch 14, 30-min cap)
- **`test_avg/mae_surf_p`:** **110.97** (from best-val checkpoint)
- **Per-split surface-p MAE (test):** single_in_dist=134.23, geom_camber_rc=121.93, geom_camber_cruise=76.78, re_rand=110.93
- **Config:** default — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, CosineAnnealingLR(T_max=50), AdamW`
- **Metric artifacts:** `models/model-charliepai2g48h4-fern-scoring-nan-fix-20260512-185620/metrics.jsonl`

## Merged improvements (within noise floor)

| PR | Description | val_avg | test_avg | Status |
|---|---|---|---|---|
| #1513 (tanjiro) | bf16 autocast | 125.40 | 126.57 (3-split) | **MERGED** → 24% per-epoch speedup |
| #1416 (thorfinn) | unified_pos=True, ref=8 | 125.78 | 117.12 | **MERGED** → best cruise OOD |
| #1369 (askeladd) | surf_weight=10→20 | 127.94 | 117.35 | **MERGED but effectively reverted** → regression confirmed (#1570: val=127.86), rolled back via #1577 |
| #1577 (alphonse) | seed=42 + surf_weight=10 rollback | 116.43 | 108.87 | MERGED |
| #1542 (nezuko) | T_max=15 cosine truncation | 114.81 | 104.68 | MERGED → superseded |
| #1374 (edward) | Huber loss (beta=1.0) | 110.59 | 102.28 | MERGED → superseded by #1696 |
| **#1696 (frieren)** | **grad-clip max_norm=1.0** | **96.78** | **86.56** | **MERGED — NEW BEST** |

**Current advisor-branch recipe** (after 8 effective merges):
`unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, loss=Huber(beta=1.0), clip_grad_norm_(max_norm=1.0)`

**Comparison threshold:** cross-seed σ ≈ 3.5 val / 0.5 test (pre-Huber estimate; recalibration on new baseline needed). Use 5+ pt val difference as practical significance threshold. Note: frieren's run was pre-Huber; the combined Huber+grad-clip effect is yet to be measured.

---

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.
- **All future experiments must use `seed=42` to be comparable against this seeded baseline.** Include `--experiment_name` and don't override seed.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
