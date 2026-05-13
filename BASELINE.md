# BASELINE — icml-appendix-charlie-pai2g-24h-r2

## Current best — PR #1477 (2026-05-13)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **84.5393** |
| val_single_in_dist/mae_surf_p | 97.8600 |
| val_geom_camber_rc/mae_surf_p | 96.9600 |
| val_geom_camber_cruise/mae_surf_p | 63.2900 |
| val_re_rand/mae_surf_p | 80.0500 |
| test_avg/mae_surf_p (bs=4, clean) | **74.6655** |
| test_single_in_dist/mae_surf_p (bs=1) | 87.5600 |
| test_geom_camber_rc/mae_surf_p (bs=1) | 87.1800 |
| test_geom_camber_cruise/mae_surf_p (bs=1) | 52.6900 |
| test_re_rand/mae_surf_p (bs=1) | 72.2300 |
| test_avg (bs=1 clean eval) | **74.9122** |
| best_epoch | 15 / 15 (cosine COMPLETED — no timeout-cut) |
| wall_clock | 24.6 min (vs 30 min floor — 18% faster) |
| peak_vram | 32.95 GB (vs 42.11 GB floor — 22% reduction) |

**NOTE:** Non-finite-y prefilter in `evaluate_split` now resolves the bs=4 NaN on `test_geom_camber_cruise` — test_avg at bs=4 is now fully clean (74.67).

**Artifacts:** `models/model-charliepai2g24h2-fern-amp-bf16-on-t12-floor-20260513-075400/` (best seed)

**Change from PR #1751 floor:** Added AMP bf16 autocast around forward pass in train loop and evaluate_split (with `pred.float()` before loss computation to preserve fp32 Huber + chan_w arithmetic). Also added non-finite-y prefilter in `evaluate_split` to skip samples with NaN/Inf ground truth — fixes the bs=4 test_geom_camber_cruise NaN that required `eval_bs1.py` workaround. bf16 reduces peak VRAM by 9 GB (42→33 GB) and epoch time by 37% (156s→98s), enabling the full 15-epoch cosine schedule to complete within 24.6 min vs the prior floor's timeout-cut at epoch 14. val_avg improved **−1.6%** (85.93→84.54, best seed). bs=1 test improved **−3.5%** (77.65→74.91). 2-seed mean: val_avg=85.08, test_avg_bs1=75.28 (both seeds beat floor).

**Config run:**
```bash
cd target && python train.py \
  --lr 7.5e-4 \
  --epochs 15 \
  --agent charliepai2g24h2-fern \
  --experiment_name "charliepai2g24h2-fern/amp-bf16-on-t12-floor"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)
Optimizer: AdamW lr=7.5e-4, wd=1e-4, batch_size=4, chan_w=[1,1,5], surf_weight=10, 3-ep warmup + cosine(T_max=12, eta_min=1e-6), gradient-clip max_norm=1.0, Huber β=0.3, **AMP bf16** + non-finite-y prefilter.
Peak VRAM: 32.95 GB. Wall clock: 24.6 min → **15 epochs (full cosine)**.

## Previous best — PR #1751 (2026-05-13)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **85.9338** |
| val_single_in_dist/mae_surf_p | 108.0187 |
| val_geom_camber_rc/mae_surf_p | 96.3656 |
| val_geom_camber_cruise/mae_surf_p | 61.1470 |
| val_re_rand/mae_surf_p | 78.2041 |
| test_avg/mae_surf_p (bs=4) | NaN† |
| test_single_in_dist/mae_surf_p (bs=1) | 97.6661 |
| test_geom_camber_rc/mae_surf_p (bs=1) | 88.6578 |
| test_geom_camber_cruise/mae_surf_p (bs=1) | 51.3197 |
| test_re_rand/mae_surf_p (bs=1) | 72.9518 |
| test_avg (bs=1 clean eval) | **77.6488** |
| best_epoch | 14 (of 15; timeout-cut at 30 min) |

†bs=4 NaN on test_geom_camber_cruise is the known inference-time attention numerics edge case. bs=1 eval is fully clean via eval_bs1.py.

**Artifacts:** `models/model-charliepai2g24h2-frieren-tighter-cosine-t-max-12-20260513-055432/`

**Change from PR #1849 floor:** Changed `CosineAnnealingLR(T_max=47)` → `T_max=12` to align the cosine half-cycle with the actual ~12-epoch training window. With T_max=47, by epoch 14 the LR had decayed only 23% (still near peak ~7e-4). With T_max=12, by epoch 14 the LR had decayed 92% (5.1e-5). The missing late-LR decay phase was the bulk of remaining headroom. val_avg improved **−18.7%** (105.68 → 85.93). bs=1 test improved **−18.3%** (94.98 → 77.65). The gain is concentrated in epochs 12–14.

**Config run:**
```bash
cd target && python train.py \
  --lr 7.5e-4 \
  --epochs 15 \
  --agent charliepai2g24h2-frieren \
  --experiment_name "charliepai2g24h2-frieren/tighter-cosine-t-max-12"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)
Optimizer: AdamW lr=7.5e-4, wd=1e-4, batch_size=4, chan_w=[1,1,5], surf_weight=10, 3-ep warmup + cosine(**T_max=12**, eta_min=1e-6), gradient-clip max_norm=1.0, **Huber β=0.3**, fp32
Peak VRAM: 42.11 GB. Wall clock: 30 min → 14 epochs.

## Previous best — PR #1849 (2026-05-13)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **105.6808** |
| val_single_in_dist/mae_surf_p | 126.2130 |
| val_geom_camber_rc/mae_surf_p | 116.3601 |
| val_geom_camber_cruise/mae_surf_p | 82.2281 |
| val_re_rand/mae_surf_p | 97.9218 |
| test_avg/mae_surf_p (bs=4) | NaN† |
| test_single_in_dist/mae_surf_p (bs=1) | 113.4328 |
| test_geom_camber_rc/mae_surf_p (bs=1) | 104.1052 |
| test_geom_camber_cruise/mae_surf_p (bs=1) | 68.1068 |
| test_re_rand/mae_surf_p (bs=1) | 94.2933 |
| test_avg (bs=1 clean eval) | **94.9845** |
| best_epoch | 12 (of 50; timeout-cut at 30 min) |

†bs=4 NaN on test_geom_camber_cruise is the known inference-time attention numerics edge case. bs=1 eval is fully clean via eval_bs1.py.

**Artifacts:** `models/model-charliepai2g24h2-edward-huber-beta0p3-20260513-035314/`

**Change from PR #1801 floor:** Reduced Huber β from 1.0 → 0.3 in BOTH the training loop and `evaluate_split`. β=0.3 puts the quadratic-to-linear transition at much smaller residuals, giving a closer-to-L1 loss that better matches the MAE eval metric while keeping quadratic smoothness at tiny residuals. val_avg/mae_surf_p improved **−4.92%** (111.15 → 105.68). bs=1 test improved **−4.11%** (99.06 → 94.98). Largest per-split gain: val_geom_camber_rc −13.1% (133.88 → 116.36). Note: val_geom_camber_cruise was better at β=0.5 (75.41 vs 82.23) — low-residual splits prefer less aggressive L1 penalty.

**Config run:**
```bash
cd target && python train.py \
  --lr 7.5e-4 \
  --agent charliepai2g24h2-edward \
  --experiment_name "charliepai2g24h2-edward/huber-beta0p3"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)
Optimizer: AdamW lr=7.5e-4, wd=1e-4, batch_size=4, chan_w=[1,1,5], surf_weight=10, 3-ep warmup + cosine(T_max=47, eta_min=1e-6), gradient-clip max_norm=1.0, **Huber β=0.3**, fp32
Peak VRAM: ~42 GB. Wall clock: 30 min → 12 epochs.

## Previous best — PR #1801 (2026-05-13)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **111.1516** |
| val_single_in_dist/mae_surf_p | 134.2063 |
| val_geom_camber_rc/mae_surf_p | 133.8828 |
| val_geom_camber_cruise/mae_surf_p | 77.5901 |
| val_re_rand/mae_surf_p | 98.9271 |
| test_avg/mae_surf_p (bs=4) | NaN† |
| test_single_in_dist/mae_surf_p | 117.37 |
| test_geom_camber_rc/mae_surf_p | 118.47 |
| test_geom_camber_cruise/mae_surf_p | NaN† |
| test_re_rand/mae_surf_p | 92.83 |
| test_avg (bs=1 clean eval) | **99.0565** |
| best_epoch | 13 (of 50; timeout-cut at 30 min) |

†bs=4 NaN on test_geom_camber_cruise is the inference-time attention numerics edge case. bs=1 eval is clean. Note: test_avg bs=1 = 99.06 is the first sub-100 result on this branch.

**Artifacts:** `models/model-charliepai2g24h2-edward-huber-loss-pressure-20260513-020521/`

**Change from PR #1573 floor:** Replaced L2 squared-error loss with Huber/SmoothL1 (β=1.0) in both the training loop and `evaluate_split`. Applied to all 3 channels (Ux, Uy, p) with chan_w=[1,1,5] weighting preserved. val_avg/mae_surf_p improved **−9.4%** (122.70 → 111.15). bs=1 test improved **−10.2%** (110.25 → 99.06). Largest per-split gain: val_single_in_dist −15.9% (159.59 → 134.21).

**Config run:**
```bash
cd target && python train.py \
  --lr 7.5e-4 \
  --agent charliepai2g24h2-edward \
  --experiment_name "charliepai2g24h2-edward/huber-loss-pressure"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)
Optimizer: AdamW lr=7.5e-4, wd=1e-4, batch_size=4, chan_w=[1,1,5], surf_weight=10, 3-ep warmup + cosine(T_max=47, eta_min=1e-6), gradient-clip max_norm=1.0, Huber β=1.0, fp32
Peak VRAM: 42.12 GB. Wall clock: 30 min → 13-14 epochs.

## Previous best — PR #1573 (2026-05-13)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **122.7043** |
| val_single_in_dist/mae_surf_p | 159.59 |
| val_geom_camber_rc/mae_surf_p | 134.74 |
| val_geom_camber_cruise/mae_surf_p | 89.18 |
| val_re_rand/mae_surf_p | 107.31 |
| test_avg/mae_surf_p (bs=4) | NaN† |
| test_single_in_dist/mae_surf_p | 137.28 |
| test_geom_camber_rc/mae_surf_p | 121.04 |
| test_geom_camber_cruise/mae_surf_p | NaN† |
| test_re_rand/mae_surf_p | 103.32 |
| test_avg (bs=1 clean eval) | **110.2527** |
| best_epoch | 12 (of 50; timeout-cut at 31.2 min) |

†bs=4 NaN on test_geom_camber_cruise is a deterministic inference-time attention numerics edge case (specific bs=4 batch compositions in PhysicsAttention at specific mesh-size mixes). Not the data bug — gradient clipping did not fix it (train-side clip doesn't affect inference activations). bs=1 eval is fully clean. Askeladd's #1536 (NaN guard) addresses the data bug separately.

**Artifacts:** `models/model-charliepai2g24h2-frieren-warmup-lr75e-4-gradclip-20260512-235148/`

**Change from PR #1482 floor:** Reduced peak lr from 1e-3 → 7.5e-4, added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` immediately before `optimizer.step()`. Warmup+cosine schedule unchanged. val_avg/mae_surf_p improved **−4.2%** (128.09 → 122.70). bs=1 test improved **−6.1%** (117.40 → 110.25).

**Config run:**
```bash
cd target && python train.py \
  --lr 7.5e-4 \
  --agent charliepai2g24h2-frieren \
  --experiment_name "charliepai2g24h2-frieren/warmup-lr75e-4-gradclip"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)
Optimizer: AdamW lr=7.5e-4 (peak), wd=1e-4, batch_size=4, chan_w=[1,1,5], surf_weight=10, 3-ep warmup + cosine(T_max=47, eta_min=1e-6), gradient-clip max_norm=1.0, fp32
Peak VRAM: 42.12 GB. Wall clock: 31.2 min → 12 epochs.

## Previous best — PR #1482 (2026-05-12)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **128.0916** |
| val_single_in_dist/mae_surf_p | 162.05 |
| val_geom_camber_rc/mae_surf_p | 137.15 |
| val_geom_camber_cruise/mae_surf_p | 101.34 |
| val_re_rand/mae_surf_p | 111.83 |
| test_avg/mae_surf_p | NaN† |
| test_single_in_dist/mae_surf_p | 146.76 |
| test_geom_camber_rc/mae_surf_p | 122.86 |
| test_geom_camber_cruise/mae_surf_p | NaN† |
| test_re_rand/mae_surf_p | 111.94 |
| test_avg (bs=1 clean eval) | 117.40 |
| best_epoch | 14 (of 50; timeout-cut, still improving) |

†Two distinct NaN sources on test_geom_camber_cruise: (1) known data bug (`0*NaN` propagation from 000020.pt), and (2) model-level batch-composition sensitivity at lr=1e-3 for specific bs=4 batches in this split. Diagnosed by student: bs=1 eval produces clean test_avg=117.40. Addressed in follow-up PR.

**Note:** This floor was measured WITHOUT chan_w=[1,1,5] in the branch (PR assigned before #1464 merged). The merged advisor branch NOW has BOTH chan_w + warmup; the true floor with both levers stacked is expected to be ≤128.09 but has not been measured yet. Use 128.09 as the conservative floor until askeladd's re-run (#1536) confirms.

**Artifacts:** `models/model-charliepai2g24h2-frieren-warmup-cosine-lr1e-3-20260512-180356/`

**Change:** 3-epoch linear warmup (0.02×peak → 1.0) followed by CosineAnnealingLR(T_max=MAX_EPOCHS−3, eta_min=1e-6). Peak lr=1e-3 (CLI flag). Added per-epoch LR logging to metrics.jsonl.

**Config run:**
```bash
cd target && python train.py \
  --lr 1e-3 \
  --agent charliepai2g24h2-frieren \
  --experiment_name "charliepai2g24h2-frieren/warmup-cosine-lr1e-3"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)  
Optimizer: AdamW lr=1e-3 (peak), wd=1e-4, batch_size=4, surf_weight=10, 3-ep warmup + cosine, fp32  
Peak VRAM: 42.1 GB. Wall clock cap: 30 min → 14 epochs.

## Previous floor — PR #1464 (2026-05-12)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **133.9353** |
| val_single_in_dist/mae_surf_p | 155.84 |
| val_geom_camber_rc/mae_surf_p | 146.50 |
| val_geom_camber_cruise/mae_surf_p | 103.54 |
| val_re_rand/mae_surf_p | 129.86 |
| test_avg/mae_surf_p | NaN* (125.48 with local scoring fix) |
| test_single_in_dist/mae_surf_p | 141.26 |
| test_geom_camber_rc/mae_surf_p | 145.90 |
| test_geom_camber_cruise/mae_surf_p | NaN* (87.74 with fix) |
| test_re_rand/mae_surf_p | 127.03 |
| best_epoch | 14 (of 50; timeout-cut, still improving) |

*NaN due to `data/scoring.py` `0*NaN` propagation from one bad GT sample. 3-split test avg = 125.48 (excl. cruise).

**Artifacts:** `models/model-charliepai2g24h2-alphonse-channel-weight-p5-20260512-181154/`

**Change:** Per-channel loss weighting `chan_w=[1.0, 1.0, 5.0]` applied to `sq_err` in train and evaluate_split.

**Config run:**
```bash
cd target && python train.py \
  --agent charliepai2g24h2-alphonse \
  --experiment_name "charliepai2g24h2-alphonse/channel-weight-p5"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.66M params)  
Optimizer: AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, CosineAnnealingLR, fp32  
Peak VRAM: 42.1 GB. Wall clock cap: 30 min → 14 epochs.

## Previous floor — PR #1486 (2026-05-12)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **143.1450** |
| val_single_in_dist/mae_surf_p | 179.61 |
| val_geom_camber_rc/mae_surf_p | 151.83 |
| val_geom_camber_cruise/mae_surf_p | 114.07 |
| val_re_rand/mae_surf_p | 127.06 |
| test_avg/mae_surf_p | NaN* |
| test_single_in_dist/mae_surf_p | 156.25 |
| test_geom_camber_rc/mae_surf_p | 148.55 |
| test_geom_camber_cruise/mae_surf_p | NaN* |
| test_re_rand/mae_surf_p | 137.14 |
| best_epoch | 14 (of 50; timeout-cut — still improving) |

*NaN on test_geom_camber_cruise due to non-finite GT in sample 000020.pt (p channel). Remaining 3 test splits are clean. 3-split test avg = 147.31.

**Artifacts:** `models/model-charliepai2g24h2-tanjiro-batch-size-8-fallback-20260512-180842/`

**Config run (CLI flags, train.py defaults unchanged):**
```bash
cd target && python train.py --batch_size 8 --lr 7e-4 \
  --agent charliepai2g24h2-tanjiro \
  --experiment_name "charliepai2g24h2-tanjiro/batch-size-8-fallback"
```

Model: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1.4M params)  
Optimizer: AdamW lr=7e-4, wd=1e-4, batch_size=8, surf_weight=10, CosineAnnealingLR, fp32  
Wall-clock cap: SENPAI_TIMEOUT_MINUTES=30 — timeout-cut at 14 epochs

> Note: train.py defaults remain **lr=5e-4, batch_size=4**. This floor was measured at bs=8/lr=7e-4 (fallback after bs=16 OOM). Future PRs testing default config (bs=4/lr=5e-4) may improve on this.

## Reference baseline (target/train.py defaults — unmeasured on this branch)

The unmodified `train.py` defaults represent the canonical configuration to beat:

| Field | Value |
|---|---|
| Model | Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1.4M params) |
| Optimizer | AdamW lr=5e-4, wd=1e-4 |
| Training | bs=4, surf_weight=10, CosineAnnealingLR, 50 epochs, fp32 |
| Wall-clock cap | SENPAI_TIMEOUT_MINUTES=30 |

Primary metric: `val_avg/mae_surf_p` (lower is better). Test-time metric for paper-facing comparisons: `test_avg/mae_surf_p`.

## Update protocol

When a PR beats the current floor (143.1450), update the table above and append a historical entry below.
</content>
</invoke>