# Baseline — `icml-appendix-willow-pai2i-24h-r1`

Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface
pressure MAE across the four validation splits).
Paper-facing metric: **`test_avg/mae_surf_p`** (same metric on the test splits,
computed at end-of-run from the best-val checkpoint).

Lower is better. Per-split diagnostic metrics (`{split}/mae_surf_{Ux,Uy,p}`,
`{split}/mae_vol_*`) are also reported in W&B for every run.

## Current baseline configuration (head of advisor branch `train.py`)

- Model: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  (~570K params)
- Loss: Charbonnier `sqrt(diff² + ε²)`, ε=1e-3 (**merged from PR #3143**, default
  flipped in **PR #3440**)
  `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10.0`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- Schedule: `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6` (**merged from PR #3150**)
- **Gradient clipping**: opt-in via `--grad_clip_max_norm 0.5` (Config default is
  still 0.0). PR #3418 added the lever and demonstrated clip_0p5 wins by 9.4
  units within-PR. Default flip is queued as a follow-up.
- Eval: non-finite ground truth samples filtered at `evaluate_split` boundary,
  so `test_avg/mae_surf_p` is now finite (**merged from PR #3138**)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

✅ **Loss-fn default correct as of PR #3440.** Charbonnier ε=1e-3 auto-applies.

⚠️ **Grad-clip default caveat (pending follow-up flip).** PR #3418 added the
`--grad_clip_max_norm` flag but kept the default at 0.0. The current best
baseline (97.47) used `--grad_clip_max_norm 0.5` explicitly. Until a follow-up
flips the default, pass it explicitly. Bare `python train.py` will land in
~106 (the no_clip arm range), NOT in 97.47.

## Current best baseline result (PR #3418 — grad-clip merged)

Best single val result: **PR #3418 arm `clip_0p5`**:

| Source | wandb run | val_avg/mae_surf_p | test_avg_3splits/mae_surf_p | Notes |
|--------|-----------|--------------------|---------------------|-------|
| PR #3418 arm `clip_0p5` | 221dquoy | **97.47** | 95.96 (3-split, cruise NaN on this branch) | Warmup+Charbonnier+grad_clip_0.5; 4-split test pending re-eval |

Branch was pre-#3138 NaN-fix merge so `test_avg/mae_surf_p` reported as 3-split partial mean only.

Per-split val at best epoch (clip_0p5, epoch 14):
- `val_single_in_dist`     mae_surf_p = ~105
- `val_geom_camber_rc`     mae_surf_p =  ~97
- `val_geom_camber_cruise` mae_surf_p =  N/A (need re-eval with NaN-fix code)
- `val_re_rand`            mae_surf_p =  ~84

(The clip_0p5 PR was finished before its branch picked up the #3138 NaN fix; a
fresh re-run from current advisor head would give full 4-split numbers.)

**Last clean 4-split sanity** — PR #3138 `nan_fix_sanity` (warmup + Charbonnier
+ NaN-fix, no grad clip):

| Source | wandb run | val_avg/mae_surf_p | **test_avg/mae_surf_p** | Notes |
|--------|-----------|--------------------|-----------------------|-------|
| PR #3138 `nan_fix_sanity` | u2k87wan | 102.25 | **92.71** | First valid 4-split test |

Per-split val at best epoch (nan_fix_sanity, epoch 14, composed baseline):
- `val_single_in_dist`     mae_surf_p = 116.56
- `val_geom_camber_rc`     mae_surf_p = 127.63
- `val_geom_camber_cruise` mae_surf_p =  72.04
- `val_re_rand`            mae_surf_p =  92.77

Per-split test (ALL SPLITS NOW FINITE — NaN fix merged):
- `test_single_in_dist`    mae_surf_p = 103.60
- `test_geom_camber_rc`    mae_surf_p = 117.05
- `test_geom_camber_cruise` mae_surf_p =  63.99
- `test_re_rand`           mae_surf_p =  86.20
- **`test_avg/mae_surf_p` = 92.71** (all 4 splits, equal-weight mean)

**To beat the baseline**, a PR must achieve `val_avg/mae_surf_p < 97.47` (i.e.,
better than the best single-seed result across all PRs). With run-to-run
variance ~3-4 units, improvements need to be ≥5 units to be clearly attributable.

## Pre-merge history

| Source | wandb run | val_avg/mae_surf_p | Notes |
|--------|-----------|--------------------|-------|
| PR #3440 `default_charb_sanity` | kqjdf50q | 107.14 | Defaults-only sanity, charbonnier auto-applied, all 4 test splits finite (test_avg/mae_surf_p=97.24) |
| PR #3143 arm `charbonnier_eps1e-3` | lukq8jry | 98.60 | Charbonnier alone, pre-warmup |
| PR #3138 `nan_fix_sanity` | u2k87wan | 102.25 | Warmup+Charbonnier+NaN-fix compose |
| PR #3150 arm `lr5e-4_wu3` | sb39atyp | 125.83 | warmup+cosine merge winner |
| frieren w128 #3148 | qmyih0vv | 128.46 | pre-warmup baseline-equivalent control |
| fern depth5 #3145  | 0g36hqgg | 129.07 | pre-warmup baseline-equivalent control |
| nezuko surfp1 #3149| 7d1rlw4w | 132.33 | pre-warmup baseline-equivalent control |
| edward mse_baseline #3143 | 9npuojl6 | 121.14 | within-PR MSE control, pre-warmup |

Implicit pre-warmup baseline ≈ **130 ± 3**, run-to-run variance ~3-4 units.

W&B project: `wandb-applied-ai-team/senpai-v1` — research tag
`willow-pai2i-24h-r1`.

Reproduce current best baseline (requires explicit `--grad_clip_max_norm 0.5`
flag until default flip lands):
```
cd target/
python train.py --grad_clip_max_norm 0.5 \
  --wandb_name baseline_default --wandb_group baseline
```

Loss-fn default is correct (Charbonnier ε=1e-3, PR #3440). Grad-clip default
will be flipped in an upcoming follow-up PR.
