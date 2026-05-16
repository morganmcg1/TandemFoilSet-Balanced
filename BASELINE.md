# Baseline — `icml-appendix-willow-pai2i-48h-r4`

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better). For paper-facing comparison use `test_avg/mae_surf_p` (NaN until scoring bug fixed — see note below).

---

## 2026-05-16 15:50 — PR #3969: SwiGLU mlp_ratio=2 + epochs=14 (askeladd) — ← CURRENT BEST

- **val_avg/mae_surf_p: 56.4402** (best epoch 14/14, W&B run `dwyzcs0e`) — **−1.60% vs previous best 57.3537**
- **test_avg/mae_surf_p: 48.8947** — **−1.84% vs previous best 49.8024**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not per-split logged) | 55.3088 |
| geom_camber_rc | (not per-split logged) | 61.1577 |
| geom_camber_cruise | (not per-split logged) | 32.0157 |
| re_rand | (not per-split logged) | 47.0968 |
| **avg** | **56.4402** | **48.8947** |

- **Model config:** SwiGLU FFN, n_hidden=160, n_layers=5, n_head=4, slice_num=64, **mlp_ratio=2**, inner_dim=216, ~1.035M params
- **Change from previous baseline:** `--epochs 14` on mlp_ratio=2 stack (vs mlp_ratio=3 at epochs=14 in PR #4002)
- **Val curve (last 4 epochs):** ep11=65.49, ep12=61.10, ep13=58.40, ep14=**56.44** — still descending at budget end; ep13→ep14 delta = −1.96
- **Wall time:** ~41.05 min (2463 s); ~176 s/epoch
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=14); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Key finding:** mlp_ratio=2 + epochs=14 (this PR) BEATS mlp_ratio=3 + epochs=14 (PR #4002) by 0.91 val / 0.91 test. The wider model does not cleanly compound with extended training at this budget. The under-converged model at ep12 (mlp_ratio=3) overtook mlp_ratio=2 at shorter budgets, but the advantage reverses with extended training.

**Reproduce command:**
```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=50 python train.py --epochs 14
```
*(mlp_ratio=2 is default; no extra flag needed)*

---

## 2026-05-16 15:30 — PR #4002: SwiGLU mlp_ratio=3 + epochs=14 (alphonse)

- **val_avg/mae_surf_p: 57.3537** (best epoch 14/14, W&B run `vuod53pk`) — **−2.80% vs previous best 59.0038**
- **test_avg/mae_surf_p: 49.8024** — **−1.84% vs previous best 50.7368**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not per-split logged) | 55.8769 |
| geom_camber_rc | (not per-split logged) | 60.9186 |
| geom_camber_cruise | (not per-split logged) | 33.9840 |
| re_rand | (not per-split logged) | 48.4299 |
| **avg** | **57.3537** | **49.8024** |

- **Model config:** SwiGLU FFN, n_hidden=160, n_layers=5, n_head=4, slice_num=64, **mlp_ratio=3**, inner_dim=320, ~1.285M params
- **Change from previous baseline:** `--epochs 14` (extended training; ep13→ep14 delta = −1.28, still descending)
- **Val curve (selected epochs):** ep12=61.60, ep13=58.63, ep14=**57.35** — still descending at budget end
- **Wall time:** 44.4 min; Peak GPU ~98.5% of 96GB VRAM
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=14); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Note:** 3 of 4 test splits improved; geom_camber_cruise marginally regressed (+0.32, within run-to-run noise ~2.5)

**Reproduce command:**
```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=50 python train.py --epochs 14 --mlp_ratio 3
```

---

## 2026-05-16 14:25 — PR #3908: SwiGLU mlp_ratio=3 (alphonse)

- **val_avg/mae_surf_p: 59.0038** (best epoch 12/12, W&B run `4n7z1mwm`) — **−2.83% vs previous best 60.7195**
- **test_avg/mae_surf_p: 50.7368** — **−2.35% vs previous best 51.9559**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 62.4661 | 57.1946 |
| geom_camber_rc | 71.9954 | 62.6291 |
| geom_camber_cruise | 41.9057 | 33.6630 |
| re_rand | 59.6480 | 49.4603 |
| **avg** | **59.0038** | **50.7368** |

- **Model config:** SwiGLU FFN, n_hidden=160, n_layers=5, n_head=4, slice_num=64, **mlp_ratio=3**, inner_dim=320, ~1.285M params (+0.25M vs mlp_ratio=2 baseline)
- **Change from previous baseline:** `--mlp_ratio 3` (inner_dim 216 → 320); wider gated FFN
- **Val curve:** Best epoch 12/12 (last epoch). Model still converging at budget end.
- **Also tested:** mlp_ratio=4 (inner_dim=424, 1.535M params) → val=59.9421/test=51.1934; ratio=3 wins both val and test
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=12); **Batch:** 4, surf_weight=10.0, grad_clip=1.0

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=40 python train.py --epochs 12 --mlp_ratio 3
```

---

## 2026-05-16 12:45 — PR #3905: SwiGLU + epochs=12 (askeladd)

- **val_avg/mae_surf_p: 60.7195** (best epoch 12/12, W&B run `j4ej0kge`) — **−5.5% vs previous best 64.2430**
- **test_avg/mae_surf_p: 51.9559** — **−6.5% vs previous best 55.5454**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 66.2504 | 58.9275 |
| geom_camber_rc | 71.2693 | 61.2299 |
| geom_camber_cruise | 44.9014 | 36.8245 |
| re_rand | 60.4569 | 50.8417 |
| **avg** | **60.7195** | **51.9559** |

- **Model config:** Same as #3814 (SwiGLU FFN, n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, inner_dim=216, 1.04M params)
- **Change:** `--epochs 12` instead of 10; cosine T_max=12
- **Val curve:** Still descending at epoch 12 (−3.70 from epoch 11→12, comparable to −3.80 from 10→11). Model NOT yet converged.
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=12); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Budget:** 40-min wall clock → 34.8 min / 12 epochs (~173s/epoch); Peak VRAM ~98.2% of 100.78 GB

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=40 python train.py --epochs 12 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(All other flags use defaults: SwiGLU FFN, coord_noise_std=0.01, num_freq=4, lr=5e-4)*

> **Note on convergence:** Val curve still descending at epoch 12 at the same rate as epoch 11. The model is under-converged. Next experiments should try `--epochs 14` or `--epochs 16` with `SENPAI_TIMEOUT_MINUTES=50`.

---

## 2026-05-16 11:30 — PR #3814: SwiGLU FFN in TransolverBlock (askeladd)

- **val_avg/mae_surf_p: 64.2430** (best epoch 10/10, W&B run `dvcj6w25`) — **−22% vs previous best 82.4997**
- **test_avg/mae_surf_p: 55.5454** — **−25% vs previous best 74.1023**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 71.8437 | 64.1005 |
| geom_camber_rc | 74.3348 | 66.0329 |
| geom_camber_cruise | 46.4804 | 37.6118 |
| re_rand | 64.3132 | 54.4363 |
| **avg** | **64.2430** | **55.5454** |

- **Model config:** Transolver `n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1.04M params, +0.7% vs baseline)
- **SwiGLU change:** `self.mlp` in `TransolverBlock` replaced with `SwiGLUFFN(hidden_dim, hidden_dim*mlp_ratio, hidden_dim)`. Inner dim=216 (`round_to_mult(160*2*2/3, 8)`). `mlp2` (output head) left as standard MLP.
- **Augmentation:** `coord_noise_std=0.01`
- **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=10)
- **Batch:** 4, surf_weight=10.0, grad clip max_norm=1.0
- **Budget:** 30-min wall clock → 10 epochs (~173s/epoch); peak VRAM ~54.6 GB
- **Reproducibility:** second seed `msnk1t8p` gave val=64.5454, test=55.8364 (within 0.3 of canonical run — not a seed fluke)
- **Note:** best val still at last epoch (10/10) — model still improving at budget end; gains are likely underestimated at fixed compute. **Future SwiGLU experiments should use --epochs 12** to stack with the epochs win.

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py --epochs 12 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(SwiGLU is now merged into train.py defaults; coord_noise_std=0.01, num_freq=4, lr=5e-4 all default)*

---

## 2026-05-16 08:30 — PR #3691: Longer training 12 epochs (thorfinn)

- **val_avg/mae_surf_p: 82.4997** (best epoch 11/12, W&B run `zqxkh9np`)
- **test_avg/mae_surf_p: 74.1023** — best-val run; note: among 3 seeds, `kkuvnrai` achieved test=72.3393 (−1.96% vs baseline) while `zqxkh9np` shows a small test regression (+0.42%). Mean across 3 seeds: val≈82.96, test≈73.41 — both beat baseline.

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 90.995 | 83.128 |
| geom_camber_rc | 91.548 | 82.735 |
| geom_camber_cruise | 65.790 | 56.332 |
| re_rand | 81.665 | 74.215 |
| **avg (`zqxkh9np`)** | **82.4997** | **74.1023** |

- **Model config:** Transolver `n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1.03M params) — unchanged
- **Augmentation:** `coord_noise_std=0.01` (from #3632 default)
- **Positional encoding:** Fourier PE `num_freq=4` (from #3372 default)
- **Loss:** L1 (`Config.loss_type = "l1"`)
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=12) — key change vs baseline
- **Batch:** 4, surf_weight=10.0, grad clip max_norm=1.0
- **Budget:** 30-min wall clock → 12 epochs (~170s/epoch, fitting within budget)
- **Multi-seed note:** 3 runs with identical config showed val spread 82.50–83.74 and test spread 72.34–74.10. Best-val (`zqxkh9np`) and best-test (`kkuvnrai`) are different seeds — single-seed conclusion has uncertainty ≈ the gain itself. Gain is real in expectation (mean val 82.96 < 83.50 baseline).

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py --epochs 12 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(All other flags use current defaults: coord_noise_std=0.01, num_freq=4, lr=5e-4)*

> **Note on epochs:** `--epochs 12` is now the recommended training budget. The 10-epoch schedule was leaving the model unconverged; best_epoch landed at 11/12 in all 3 replicates. Future experiments should default to `--epochs 12`.

---

## 2026-05-16 04:30 — PR #3632: Coordinate noise augmentation std=0.01 on (x,z) during training (tanjiro)

- **val_avg/mae_surf_p: 83.4954** (best epoch 10/10, W&B run `0q6t1hpc`)
- **test_avg/mae_surf_p: 73.7918** — clean finite metric, −4.28% vs previous best

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 95.1365 | 83.7744 |
| geom_camber_rc | 91.6051 | 80.5539 |
| geom_camber_cruise | 64.7562 | 55.2016 |
| re_rand | 82.4838 | 75.6371 |
| **avg** | **83.4954** | **73.7918** |

- **Model config:** Transolver `n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1.03M params)
- **Augmentation:** `coord_noise_std=0.01` — Gaussian jitter on normalized (x,z) coords during training only
- **Positional encoding:** Fourier PE `num_freq=4` (from PR #3372)
- **Loss:** L1 (`Config.loss_type = "l1"`)
- **Optimizer:** AdamW, lr=5e-4 (Config default), weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=10)
- **Grad clip:** max_norm=1.0
- **Batch:** 4, surf_weight=10.0
- **Budget:** 30-min wall clock → 10 epochs (~170s/epoch)

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py --epochs 10 \
  --coord_noise_std 0.01 --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(coord_noise_std=0.01 will be the new default after this merge; num_freq=4 is already default)*

> **Note on lr:** This winning run used lr=5e-4 (Config default). The previous baseline #3372 used lr=1e-3. Testing lr=1e-3 with coord noise is an open experiment — expected to compound.

---

## 2026-05-16 02:25 — PR #3372: Fourier positional encoding 4-freq on (x,z) coords (askeladd)

- **val_avg/mae_surf_p: 88.2442** (best epoch 10/10, W&B run `qyc68z5k`)
- **test_avg/mae_surf_p: 77.0880** — clean finite metric

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 101.5180 | 87.8840 |
| geom_camber_rc | 97.1550 | 82.7020 |
| geom_camber_cruise | 67.7870 | 59.4070 |
| re_rand | 86.5170 | 78.3590 |
| **avg** | **88.2442** | **77.0880** |

- **Model config:** Transolver `n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1.03M params)
- **Positional encoding:** NeRF-style log-spaced Fourier features, `num_freq=4` on `(x, z)` coords; `fun_dim` grows from 24 → 40 (2 raw coords replaced by `4*num_freq=16` sinusoidal features per coord pair). Config knob: `Config.num_freq = 4`.
- **Loss:** L1 (`Config.loss_type = "l1"`, default — from #3089)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=10)
- **Grad clip:** max_norm=1.0
- **Batch:** 4, surf_weight=10.0
- **Budget:** 30-min wall clock → 10 epochs; per-epoch time ~168s (same as width-160)
- **Peak VRAM:** ~50 GB (no significant overhead from PE — only input layer grows)

**Reproduce command:**
```bash
cd target/ && SENPAI_TIMEOUT_MINUTES=30 python train.py --epochs 10 --num_freq 4 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(Config.num_freq is now 4 by default; no `--num_freq` flag needed after merge)*

---

## 2026-05-16 00:30 — PR #3507: Width scaling n_hidden 128 → 160 (alphonse)

- **val_avg/mae_surf_p: 96.0997** (best epoch 10/10, W&B run `7vxhbv8o`)
- **test_avg/mae_surf_p: 85.5256** — clean finite metric

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not logged per-epoch) | 103.7483 |
| geom_camber_rc | (not logged per-epoch) | 92.4243 |
| geom_camber_cruise | (not logged per-epoch) | 61.3787 |
| re_rand | (not logged per-epoch) | 84.5510 |
| **avg** | **96.0997** | **85.5256** |

- **Model config:** Transolver `n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1.03M params, ↑ from 662k)
- **Loss:** L1 (`Config.loss_type = "l1"`, default)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=10)
- **Grad clip:** max_norm=1.0
- **Batch:** 4, surf_weight=10.0
- **Budget:** 30-min wall clock → 10 epochs; per-epoch time ~168s (↑ from ~134s at n_hidden=128)
- **Peak VRAM:** 50.1 GB (of 96 GB available — significant headroom remains)

**Reproduce command:**
```bash
cd target/ && python train.py --epochs 10 --lr 1e-3 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(Config.n_hidden is now 160 by default; no extra flag needed)*

---

## 2026-05-15 22:31 — PR #3089: L1 loss + NaN scoring fix (alphonse)

- **val_avg/mae_surf_p: 100.5275** (best epoch 10/10, W&B run `14w7wdyb`)
- **test_avg/mae_surf_p: 90.1489** — first clean finite test metric (scoring NaN bug fixed)

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not logged) | 112.07 |
| geom_camber_rc | (not logged) | 98.04 |
| geom_camber_cruise | (not logged) | 64.21 |
| re_rand | (not logged) | 86.28 |
| **avg** | **100.5275** | **90.1489** |

- **Model config:** Transolver `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **Loss:** L1 (`Config.loss_type = "l1"`) — replaces MSE; dispatched via `_pointwise_loss` helper
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs (0.1→1.0), then cosine to 0 over remaining epochs (T_max=10)
- **Grad clip:** max_norm=1.0
- **Batch:** 4, surf_weight=10.0
- **Scoring fix:** `torch.isfinite` per-sample mask in `evaluate_split` (train.py) — makes `test_avg/mae_surf_p` finite for all splits
- **Budget:** 30-min wall clock → 10 epochs (T_max=10, cosine fully anneals)

**Reproduce command:**
```bash
cd target/ && python train.py --epochs 10 --lr 1e-3 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```
*(No `--loss_type` flag needed — `Config.loss_type` default is now `"l1"`)*

---

## 2026-05-15 14:38 — PR #3091: LR warmup + gradient clipping + lr=1e-3

- **val_avg/mae_surf_p: 109.4166** (best epoch 14/15, W&B run `0ez1sqmi`)
- **test_avg/mae_surf_p: NaN** — scoring bug in `data/scoring.py`: `0.0 * NaN = NaN` on accumulation of 1 non-finite cruise test sample; 3-split workaround = **107.4694** (excl. `test_geom_camber_cruise`)

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 119.58 | 111.04 |
| geom_camber_rc | 119.40 | 110.20 |
| geom_camber_cruise | 88.57 | **NaN (bug)** |
| re_rand | 110.12 | 101.17 |
| **avg** | **109.42** | — |

- **Model config:** Transolver `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~3M params)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs (0.1→1.0), then cosine to 0 over remaining epochs
- **Grad clip:** max_norm=1.0 (logs `train/grad_norm`)
- **Batch:** 4, surf_weight=10.0, MSE loss
- **Budget:** 30-min wall clock → 15 epochs of configured 50 (T_max=50, so cosine barely anneals — see note)

**Reproduce command:**
```bash
cd target/ && python train.py --epochs 50 --lr 1e-3 \
  --agent <student-name> --wandb_name <run-name> --wandb_group <group>
```

> **Note on cosine schedule:** `T_max=50` with 30-min budget means only ~14-15 epochs run and LR anneals only ~15% of its range. Experiments targeting proper LR annealing should pass `--epochs 10` (or whatever matches the realistic epoch count) to get T_max=10 so cosine fully decays within the budget.

> **Note on test_geom_camber_cruise NaN:** `data/scoring.py`'s `accumulate_batch` computes `err * surf_mask` where `err` includes NaN from non-finite ground truth at 1 sample; `0 * NaN = NaN` propagates to the sum. Fix is `err.nan_to_num(0.0)` in `accumulate_batch` or clamping `y` before passing to it. This is tracked in edward's follow-up PR. Until fixed, compare on `val_avg/mae_surf_p` (clean) and 3-split test workaround.
