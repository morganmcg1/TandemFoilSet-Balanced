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
- **Gradient clipping**: `clip_grad_norm_(max_norm=0.5)` — **default is now 0.5**
  (PR #3418 added lever; PR #3494 flipped default from 0.0 → 0.5). Bare
  `python train.py` now clips at 0.5 automatically — no explicit flag needed.
- **Precision**: `bfloat16` AMP mixed precision — **default is now bf16** (**merged
  from PR #3330**). 1.33× per-epoch speedup → deeper cosine decay in 30-min budget.
- Eval: non-finite ground truth samples filtered at `evaluate_split` boundary,
  so `test_avg/mae_surf_p` is now finite (**merged from PR #3138**)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

✅ **Loss-fn default correct as of PR #3440.** Charbonnier ε=1e-3 auto-applies.
✅ **Grad-clip default correct as of PR #3494.** Bare `python train.py` now clips at 0.5. No explicit flag needed.
✅ **Fourier positional encoding default as of PR #3348.** `pos_enc_mode=fourier_basic`, L=8 auto-applies.
✅ **bf16 AMP default correct as of PR #3330.** 1.33× per-epoch speedup auto-applies.

## 2026-05-16 — PR #3330: bf16 AMP mixed precision — **MERGED ⭐⭐ (new val AND test best)**

- **val_avg/mae_surf_p: 83.54** (NEW BEST, −13.93 vs 97.47 prior best, −14.3%)
- **test_avg/mae_surf_p: 73.02** (NEW BEST, −13.20 vs 86.22 prior best, −15.3%)
- **W&B run:** 5a0rym2t (bf16_fourier_v1)
- **Control:** ku86zau9 (fp32_ref_v2, val=103.62, test=92.43) — 1.33× speedup confirmed
- **best_epoch:** 19 (vs 14 for fp32 — deeper cosine decay in same 30-min budget)
- **Speedup:** 1.329× per-epoch (99.4 s/epoch bf16 vs 132.1 s/epoch fp32)
- **Surface MAE (test, bf16_fourier_v1, run 5a0rym2t):**
  - test_single_in_dist mae_surf_p = 87.67
  - test_geom_camber_rc mae_surf_p = 78.83
  - test_geom_camber_cruise mae_surf_p = 52.60
  - test_re_rand mae_surf_p = 72.97
- **Reproduce:**
  ```
  cd target/
  python train.py --pos_enc_mode fourier_basic \
    --wandb_group bf16-amp-fourier --wandb_name bf16_fourier_v1 --epochs 50
  ```
  (No `--grad_clip_max_norm` or `--amp_dtype` flags needed — both now default.)

Notes: Composes multiplicatively with Fourier L=8 (PR #3348). 1.33× per-epoch speedup lets
cosine schedule decay 36% deeper (ep 19 vs 14), with bf16 numerical stability confirmed across
~60 aggregate training epochs and all 4 test splits finite. Post-merge, all subsequent experiments
get the speedup automatically.

---

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

## 2026-05-16 — PR #3494: Default grad_clip flip 0.0 → 0.5 (nezuko)

- **Operational hygiene.** Flips `Config.grad_clip_max_norm` default from 0.0 to 0.5 so bare `python train.py` matches the documented best config.
- **Sanity run:** val_avg/mae_surf_p = 101.19, test_avg/mae_surf_p = 90.50 (W&B w8th8428). No metric improvement expected — just verifies the flip works.
- **W&B run:** w8th8428 (willowpai2i24h1-nezuko/default_clip_sanity)
- **Reproduce:** `cd "target/" && python train.py --epochs 50` (no flags needed)

## 2026-05-16 — PR #3348: Fourier positional encoding L=8 (fern)

- **val_avg/mae_surf_p:** 98.16 (within noise of 97.47 val best — effectively tied)
- **test_avg/mae_surf_p: 86.22 (NEW TEST BEST, −6.49 vs 92.71 prior best)**
- **Surface MAE (test, fourier_L8_charb, run jum9x071):**
  - test_single_in_dist mae_surf_p = 96.53
  - test_geom_camber_rc mae_surf_p = 102.56
  - test_geom_camber_cruise mae_surf_p = 55.77
  - test_re_rand mae_surf_p = 90.04
- **W&B run:** jum9x071
- **Reproduce:**
  ```
  cd target/
  python train.py --pos_enc_mode fourier_basic --grad_clip_max_norm 0.5 \
    --wandb_group fourier-pos-enc-charb --wandb_name fourier_L8_charb --epochs 50
  ```

Notes: val primary metric is within noise (98.16 vs 97.47), but test improvement is −6.49 absolute (−7.0%) — decisive on the paper-facing metric. Per-split test gains largest on geom_camber_cruise (−8.22). Fourier L=8 default baked into merged train.py via `pos_enc_mode` flag.

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

Reproduce current best baseline (all defaults correct — no explicit flags needed):
```
cd target/
python train.py --wandb_name baseline_default --wandb_group baseline
```

All defaults now correct: Charbonnier ε=1e-3 (#3440), grad_clip=0.5 (#3494), Fourier L=8 positional encoding (#3348).
