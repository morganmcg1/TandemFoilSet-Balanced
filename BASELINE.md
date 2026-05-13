# BASELINE — icml-appendix-charlie-pai2g-24h-r2

## Current best — PR #1801 (2026-05-13)

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