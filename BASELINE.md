# TandemFoilSet — Advisor Baseline

**Branch:** `icml-appendix-charlie-pai2i-24h-r4`
**Round:** charlie-pai2i-24h-r4 (24h budget, 8 students × 1 GPU)
**Primary metric:** `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface pressure MAE across 4 val splits
**Test metric:** `test_avg/mae_surf_p` (now finite — NaN workaround merged with PR #3217)

## Current best (this branch)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p`              | **122.81** | PR #3217 (frieren H5 RFF), epoch 12 |
| `val_single_in_dist/mae_surf_p`   | 144.70 | PR #3217 |
| `val_geom_camber_rc/mae_surf_p`   | 125.95 | PR #3217 |
| `val_geom_camber_cruise/mae_surf_p` | 101.61 | PR #3217 |
| `val_re_rand/mae_surf_p`          | 119.00 | PR #3217 |
| `test_avg/mae_surf_p`             | **111.16** | PR #3217 |
| `test_single_in_dist/mae_surf_p`  | 123.91 | PR #3217 |
| `test_geom_camber_rc/mae_surf_p`  | 114.82 | PR #3217 |
| `test_geom_camber_cruise/mae_surf_p` | 88.14 | PR #3217 |
| `test_re_rand/mae_surf_p`         | 117.78 | PR #3217 |

## Current baseline configuration

`train.py` after merging PR #3226 (H10 Re-strat) + PR #3217 (H5 RFF + NaN fix):

- **Model:** `Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)` (~678K trainable params + 64 non-trainable RFF buffer)
- **Input:** RFF coordinate encoding (n_freq=32, sigma=1.0) replacing raw (x,z) — input to preprocess MLP is now 86-dim (64 RFF + 22 other features)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** CosineAnnealingLR(T_max=epochs)
- **Batch:** 4
- **surf_weight:** 10.0
- **Epochs:** 50 (cap) / `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap
- **Sampler:** WeightedRandomSampler with domain-balanced weights × Re-strat multiplier (Re>1e6 samples × 2.0; ~1303/1499 train samples)
- **NaN workaround:** `evaluate_split` masks out and zero-fills non-finite GT samples before accumulation (fixes test_geom_camber_cruise NaN)
- **Splits dir:** `/mnt/new-pvc/datasets/tandemfoil/splits_v2`

### Reproduce command

```bash
cd target && python train.py --agent <student> --experiment_name "<student>/baseline"
```

---

## Baseline history

### 2026-05-15 16:25 — PR #3217: H5 RFF coord encoding + NaN fix (frieren) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 122.81 (best epoch 12, 30-min cap)
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 144.70
  - `val_geom_camber_rc/mae_surf_p` = 125.95
  - `val_geom_camber_cruise/mae_surf_p` = 101.61
  - `val_re_rand/mae_surf_p` = 119.00
- **test_avg/mae_surf_p:** 111.16 (first finite test metric on this branch)
- **Per-split test:**
  - `test_single_in_dist/mae_surf_p` = 123.91
  - `test_geom_camber_rc/mae_surf_p` = 114.82
  - `test_geom_camber_cruise/mae_surf_p` = 88.14
  - `test_re_rand/mae_surf_p` = 117.78
- **What changed:**
  1. Added `RFFEncoding(n_freq=32, sigma=1.0)` module registered as a non-trainable buffer, replacing raw (x,z) dims 0-1 with 64-dim Fourier expansion. Preprocess MLP input: 24→86.
  2. In `evaluate_split`: added `y_finite_sample = isfinite(y).all(...)` mask + `nan_to_num(y)` to prevent NaN-propagation through the `(pred-y).abs() * surf_mask` multiplication for the one non-finite sample in `test_geom_camber_cruise`.
- **Metric artifact:** `models/model-frieren-rff-nfreq32-sigma1-20260515-140556/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent frieren --experiment_name "frieren/rff-nfreq32-sigma1"`

### 2026-05-15 15:00 — PR #3226: H10 Re-stratified sampler (thorfinn)

- **val_avg/mae_surf_p:** 127.84 (best epoch 14, 30-min cap)
- **Per-split val:** single_in_dist=160.10, geom_camber_rc=148.67, geom_camber_cruise=91.50, re_rand=111.08
- **test_avg/mae_surf_p:** NaN at time of merge (fixed by frieren PR #3217)
- **What changed:** Re>1e6 samples weighted × 2.0 in `WeightedRandomSampler`. 1303/1499 train samples affected.
- **Metric artifact:** `models/model-charliepai2i24h4-thorfinn-re-strat-high2x-*/metrics.jsonl`

---

## Notes for upcoming PRs

- **Beat this:** `val_avg/mae_surf_p < 122.81` to be a merge candidate.
- **Hardest split:** `val_single_in_dist = 144.70`. Two-branch head (thorfinn H7, PR #3291 in progress) targets this.
- **test_avg NaN is resolved:** The `evaluate_split` workaround is now in the baseline — test metrics should be finite for all future PRs unless a different non-finite GT sample is encountered elsewhere.
- **Both RFF and Re-strat are baked in:** Future student branches start from this combined baseline automatically.
