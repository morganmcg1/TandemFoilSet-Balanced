# Baseline — `icml-appendix-willow-pai2i-24h-r1`

Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface
pressure MAE across the four validation splits).
Paper-facing metric: **`test_avg/mae_surf_p`** (same metric on the test splits,
computed at end-of-run from the best-val checkpoint). Now **unblocked** — see note below.

Lower is better. Per-split diagnostic metrics (`{split}/mae_surf_{Ux,Uy,p}`,
`{split}/mae_vol_*`) are also reported in W&B for every run.

## Current baseline configuration (head of advisor branch `train.py`)

- Model: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  (~570K params)
- Loss: Charbonnier `sqrt(diff² + ε²)`, ε=1e-3 (**merged from PR #3143**)
  `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10.0`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- Schedule: `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6` (**merged from PR #3150**)
- Eval: non-finite ground truth samples filtered at `evaluate_split` boundary,
  so `test_avg/mae_surf_p` is now finite (**merged from PR #3138**)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

⚠️ **Config default caveat (pending fix).** `Config.loss_fn` default is still
`"mse"` on the advisor branch even though Charbonnier is the documented baseline.
Until a follow-up PR flips the default, pass `--loss_fn charbonnier
--charbonnier_eps 1e-3` **explicitly** in all reproduce commands and student
instructions. Failing to do so silently runs the old MSE baseline (~125 val).

## Current best baseline result (PR #3143 + compose confirmation)

Best single val result: PR #3143 `charbonnier_eps1e-3` (pre-warmup base, Charbonnier alone):

| Source | wandb run | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|--------|-----------|--------------------|--------------------|-------|
| PR #3143 arm `charbonnier_eps1e-3` | lukq8jry | **98.60** | N/A (pre-NaN-fix) | Charbonnier alone, pre-warmup; best single val |

**Compose confirmation** — PR #3138 sanity run (warmup + Charbonnier + NaN-fix):

| Source | wandb run | val_avg/mae_surf_p | **test_avg/mae_surf_p** | Notes |
|--------|-----------|--------------------|-----------------------|-------|
| PR #3138 `nan_fix_sanity` | u2k87wan | 102.25 | **92.71** | All 4 test splits finite; first valid test_avg on this launch |

Per-split val at best epoch (nan_fix_sanity, epoch 14, composed baseline):
- `val_single_in_dist`     mae_surf_p = 116.56
- `val_geom_camber_rc`     mae_surf_p = 127.63
- `val_geom_camber_cruise` mae_surf_p =  72.04
- `val_re_rand`            mae_surf_p =  92.77

Per-split test (ALL SPLITS NOW FINITE — NaN fix merged):
- `test_single_in_dist`    mae_surf_p = 103.60
- `test_geom_camber_rc`    mae_surf_p = 117.05
- `test_geom_camber_cruise` mae_surf_p = **63.99** ← formerly NaN; now unblocked
- `test_re_rand`           mae_surf_p =  86.20
- **`test_avg/mae_surf_p` = 92.71** (all 4 splits, equal-weight mean)

The val difference between PR #3143 (98.60) and the compose sanity (102.25) is
3.65 units — within the expected 3-4 unit run-to-run noise. The compose is
confirmed clean; no interaction between warmup and Charbonnier.

**To beat the baseline**, a PR must achieve `val_avg/mae_surf_p < 98` (i.e.,
better than the best single-seed result across all PRs). With run-to-run
variance ~3-4 units, improvements need to be ≥5 units to be clearly attributable.

## Pre-merge history

| Source | wandb run | val_avg/mae_surf_p | Notes |
|--------|-----------|--------------------|-------|
| PR #3150 arm `lr5e-4_wu3` | sb39atyp | 125.83 | warmup+cosine merge winner |
| frieren w128 #3148 | qmyih0vv | 128.46 | pre-warmup baseline-equivalent control |
| fern depth5 #3145  | 0g36hqgg | 129.07 | pre-warmup baseline-equivalent control |
| nezuko surfp1 #3149| 7d1rlw4w | 132.33 | pre-warmup baseline-equivalent control |
| edward mse_baseline #3143 | 9npuojl6 | 121.14 | within-PR MSE control, pre-warmup |

Implicit pre-warmup baseline ≈ **130 ± 3**, run-to-run variance ~3-4 units.

**Note on test_avg (RESOLVED).** `test_avg/mae_surf_p` was `None` for every
run before PR #3138 (alphonse NaN fix). Root cause: `test_geom_camber_cruise`
ground truth contains Inf values; `data/scoring.accumulate_batch` uses
`err * surf_mask` to skip them, but `NaN * 0.0 == NaN` under IEEE float,
poisoning the accumulator. Fix: filter non-finite samples in `train.py:evaluate_split`
before any arithmetic. The fix is now merged; `test_avg/mae_surf_p` will be
finite for all future runs.

W&B project: `wandb-applied-ai-team/senpai-v1` — research tag
`willow-pai2i-24h-r1`.

Reproduce baseline (explicit flags required until default flip lands):
```
cd target/
python train.py --loss_fn charbonnier --charbonnier_eps 1e-3 \
  --wandb_name baseline_default --wandb_group baseline
```
