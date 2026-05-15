# TandemFoilSet — Advisor Baseline

**Branch:** `icml-appendix-charlie-pai2i-24h-r4`
**Round:** charlie-pai2i-24h-r4 (24h budget, 8 students × 1 GPU)
**Primary metric:** `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface pressure MAE across 4 val splits
**Test metric:** `test_avg/mae_surf_p` (computed at end of every run from the best val checkpoint; currently NaN on this branch due to known data quirk — see Notes)

## Current best (this branch)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p`              | **127.84** | PR #3226 (thorfinn H10), epoch 14 |
| `val_single_in_dist/mae_surf_p`   | 160.10 | PR #3226 |
| `val_geom_camber_rc/mae_surf_p`   | 148.67 | PR #3226 |
| `val_geom_camber_cruise/mae_surf_p` | 91.50 | PR #3226 |
| `val_re_rand/mae_surf_p`          | 111.08 | PR #3226 |
| `test_avg/mae_surf_p`             | NaN (data quirk; see Notes) | PR #3226 |

## Current baseline configuration

`train.py` after merging PR #3226 (H10 Re-stratified sampler):

- **Model:** `Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)` (~1M params)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** CosineAnnealingLR(T_max=epochs)
- **Batch:** 4
- **surf_weight:** 10.0
- **Epochs:** 50 (cap) / `SENPAI_TIMEOUT_MINUTES=30` wall-clock cap
- **Sampler:** WeightedRandomSampler with domain-balanced weights **× Re-stratification multiplier** (Re>1e6 samples × 2.0; ~1303/1499 training samples affected)
- **Splits dir:** `/mnt/new-pvc/datasets/tandemfoil/splits_v2`

### Reproduce command

```bash
cd target && python train.py --agent <student> --experiment_name "<student>/baseline"
```

---

## Baseline history

### 2026-05-15 15:00 — PR #3226: H10 Re-stratified sampler (thorfinn) — **CURRENT BEST**

- **val_avg/mae_surf_p:** 127.84 (best epoch 14, 30-min cap)
- **Per-split val:**
  - `val_single_in_dist/mae_surf_p` = 160.10
  - `val_geom_camber_rc/mae_surf_p` = 148.67
  - `val_geom_camber_cruise/mae_surf_p` = 91.50
  - `val_re_rand/mae_surf_p` = 111.08
- **test_avg/mae_surf_p:** NaN (data quirk; see Notes)
- **What changed:** Added a one-time startup loop over train_ds to upweight Re>1e6 samples by 2× in the `WeightedRandomSampler`, stacked on top of the existing domain-balanced weights. 1303/1499 samples affected. `x[..., 13]` is already `log(Re)` (not normalized), so threshold is `math.log(1e6)` directly.
- **Metric artifact:** `models/model-charliepai2i24h4-thorfinn-re-strat-high2x-*/metrics.jsonl` (squash-merged into `icml-appendix-charlie-pai2i-24h-r4`)
- **Reproduce:** `cd target && python train.py --agent thorfinn --experiment_name "thorfinn/re-strat-high2x"`

---

## Notes for upcoming PRs

- **Beat this:** `val_avg/mae_surf_p < 127.84` to be a merge candidate.
- **Hardest split:** `val_single_in_dist = 160.10`. Techniques targeting this split (e.g. two-branch head, additional capacity) may yield the next improvement.
- **Branch-wide test_avg NaN:** `data/scoring.py` (read-only) accumulates `(pred-y).abs() * surf_mask`. Sample 20 in `test_geom_camber_cruise` has 761 `inf` values in `y[..., 2]` (p channel). `NaN * 0 = NaN` propagates and contaminates `test_avg`. **Workaround:** rank on `val_avg/mae_surf_p`, report the 3 finite test splits separately. Do not attempt to patch `data/scoring.py`.
- **Re-strat sampler is now baked in:** train.py already includes the Re>1e6 × 2 upweight. Students building new experiments on top of the baseline get it for free.
