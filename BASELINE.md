# TandemFoilSet — Advisor Baseline

**Branch:** `icml-appendix-charlie-pai2i-24h-r4`
**Round:** charlie-pai2i-24h-r4 (24h budget, 8 students × 1 GPU)
**Primary metric:** `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface pressure MAE across 4 val splits
**Test metric:** `test_avg/mae_surf_p` (finite — NaN workaround baked in since PR #3217)

## Current best (this branch)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p`              | **112.49** | PR #3326 (fern H12 MLP dropout), epoch 13 |
| `val_single_in_dist/mae_surf_p`   | 136.83 | PR #3326 |
| `val_geom_camber_rc/mae_surf_p`   | 118.25 | PR #3326 |
| `val_geom_camber_cruise/mae_surf_p` | 87.31 | PR #3326 |
| `val_re_rand/mae_surf_p`          | 107.55 | PR #3326 |
| `test_avg/mae_surf_p`             | **104.83** | PR #3326 |
| `test_single_in_dist/mae_surf_p`  | 126.77 | PR #3326 |
| `test_geom_camber_rc/mae_surf_p`  | 112.01 | PR #3326 |
| `test_geom_camber_cruise/mae_surf_p` | 75.35 | PR #3326 |
| `test_re_rand/mae_surf_p`         | 105.20 | PR #3326 |

## Current baseline configuration

`train.py` after merging PR #3226 (H10 Re-strat) + PR #3217 (H5 RFF + NaN fix) + PR #3326 (H12 MLP dropout):

- **Model:** `Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)` (~678K trainable params + 64 non-trainable RFF buffer)
- **Input:** RFF coordinate encoding (n_freq=32, sigma=1.0) replacing raw (x,z) — input to preprocess MLP is now 86-dim (64 RFF + 22 other features)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** CosineAnnealingLR(T_max=epochs)
- **Batch:** 4
- **surf_weight:** 10.0
- **Epochs:** 50 (cap) / `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap
- **Sampler:** WeightedRandomSampler with domain-balanced weights × Re-strat multiplier (Re>1e6 samples × 2.0; ~1303/1499 train samples)
- **MLP dropout:** `dropout=0.1` in each `TransolverBlock.mlp` (FFN sub-layers); `PhysicsAttention`, preprocess MLP, and final head remain at `dropout=0.0`
- **NaN workaround:** `evaluate_split` masks out and zero-fills non-finite GT samples before accumulation (fixes test_geom_camber_cruise NaN)
- **Splits dir:** `/mnt/new-pvc/datasets/tandemfoil/splits_v2`

### Reproduce command

```bash
cd target && python train.py --agent <student> --experiment_name "<student>/baseline"
```

---

## Baseline history

### 2026-05-15 18:20 — PR #3326: H12 MLP dropout=0.1 (fern) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 112.49 (best epoch 13, 30-min cap)
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 136.83
  - `val_geom_camber_rc/mae_surf_p` = 118.25
  - `val_geom_camber_cruise/mae_surf_p` = 87.31
  - `val_re_rand/mae_surf_p` = 107.55
- **test_avg/mae_surf_p:** 104.83
- **Per-split test:**
  - `test_single_in_dist/mae_surf_p` = 126.77
  - `test_geom_camber_rc/mae_surf_p` = 112.01
  - `test_geom_camber_cruise/mae_surf_p` = 75.35
  - `test_re_rand/mae_surf_p` = 105.20
- **What changed:** Added `nn.Dropout(0.1)` after each activation in `MLP.linear_pre` and hidden layers. `PhysicsAttention` dropout stays at 0.0. Only `TransolverBlock.mlp` gets dropout.
- **Delta:** -8.4% val_avg (122.81 → 112.49). OOD splits benefited most: geom_camber_cruise -14.1%, re_rand -9.6%, geom_camber_rc -6.1%. In-dist slightly worse on test (+2.3%), consistent with regularizer tradeoff.
- **Metric artifact:** `models/model-fern-mlp-dropout-0p1-20260515-163433/metrics.jsonl`
- **Reproduce:** `cd target && python train.py --agent fern --experiment_name "fern/mlp-dropout-0p1"`

### 2026-05-15 16:25 — PR #3217: H5 RFF coord encoding + NaN fix (frieren)

- **val_avg/mae_surf_p:** 122.81 (best epoch 12, 30-min cap)
- **Per-split val:** single_in_dist=144.70, geom_camber_rc=125.95, geom_camber_cruise=101.61, re_rand=119.00
- **test_avg/mae_surf_p:** 111.16
- **What changed:** RFFEncoding(n_freq=32, sigma=1.0) replacing raw (x,z) coords. evaluate_split NaN workaround added.
- **Metric artifact:** `models/model-frieren-rff-nfreq32-sigma1-20260515-140556/metrics.jsonl`

### 2026-05-15 15:00 — PR #3226: H10 Re-stratified sampler (thorfinn)

- **val_avg/mae_surf_p:** 127.84 (best epoch 14, 30-min cap)
- **Per-split val:** single_in_dist=160.10, geom_camber_rc=148.67, geom_camber_cruise=91.50, re_rand=111.08
- **test_avg/mae_surf_p:** NaN at time of merge (fixed by frieren PR #3217)
- **What changed:** Re>1e6 samples weighted × 2.0 in WeightedRandomSampler.
- **Metric artifact:** `models/model-charliepai2i24h4-thorfinn-re-strat-high2x-*/metrics.jsonl`

---

## Notes for upcoming PRs

- **Beat this:** `val_avg/mae_surf_p < 112.49` to be a merge candidate.
- **Hardest split:** `val_single_in_dist = 136.83`. (-5.4% improvement from 144.70 due to dropout, but still the bottleneck split).
- **Baseline stack:** Re-strat sampler + RFF coord encoding + MLP dropout=0.1 + evaluate_split NaN workaround all baked in.
- **Active WIP PRs rebasing onto this baseline:** #3222 (nezuko cautious-adamw v2), #3201 (edward channel-loss p=1.5), #3224 (tanjiro geom-cond v2), #3318 (frieren H6 SGDR).
