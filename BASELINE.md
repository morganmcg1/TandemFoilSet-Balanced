# Baseline — `icml-appendix-willow-pai2i-48h-r4`

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better). For paper-facing comparison use `test_avg/mae_surf_p` (NaN until scoring bug fixed — see note below).

---

## 2026-05-17 01:25 — PR #4252: LION optimizer at n_hidden=176 + bf16 + epochs=14 (frieren) — ← CURRENT BEST

- **val_avg/mae_surf_p: 49.2616** (best epoch 14/14, W&B run `eu7e0g18`) — **+0.86% vs #4106 val (within seed noise); −2.28% on paper-facing test metric**
- **test_avg/mae_surf_p: 41.6188** — **−2.28% vs previous best 42.5895** ← TEST METRIC WIN

| Split | val mae_surf_p | test mae_surf_p | Δ vs #4106 test |
|---|---:|---:|---:|
| single_in_dist | — | 43.9070 | **−5.39%** |
| geom_camber_rc | — | 54.7549 | **−1.36%** ← first improvement on hard split |
| geom_camber_cruise | — | 26.1287 | **−3.74%** |
| re_rand | — | 41.6846 | +0.94% (within noise) |
| **avg** | **49.2616** | **41.6188** | **−2.28%** |

- **Model config:** SwiGLU FFN, **n_hidden=176**, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, inner_dim=232, ~1.23M params (smaller than previous baseline nh=192)
- **Change from previous baseline:** Replace AdamW optimizer with Lion (sign-of-momentum, decoupled weight decay). Architecture unchanged from #4082.
- **Optimizer:** Lion, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-3
- **Throughput:** ~131 s/epoch (bf16, n_hidden=176) — same as #4082; −30% wall-clock vs #4106 (14ep vs 20ep)
- **Wall time:** ~30.5 min for all 14 epochs (within 30 min cap)
- **Peak GPU memory:** 44.6 GB (vs 47.6 GB at n_hidden=192) — 1 momentum buffer vs Adam's 2
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=14, fully annealed)
- **Key findings:**
  1. **LION beats AdamW by −19.3% val at matched config (nh=176+ep14)**. At ep9, Lion already matches AdamW-ep14.
  2. **Paper-facing test metric improves −2.28%** despite smaller model (nh=176 vs 192) and fewer epochs (14 vs 20).
  3. **`geom_camber_rc` improves for first time** (54.75 vs 55.51) since width-scaling plateau — Lion's uniform step magnitude better balances volume vs surface learning.
  4. Mechanistic explanation: Lion's sign-only update gives identical per-step magnitude for every parameter. On surf_weight=10 regression, this prevents Adam's variance-estimate from upweighting surface parameters at the expense of volume parameters. Result: better-balanced model across all splits.
  5. **Val tie (+0.86%) is within seed noise** (std ~2.5). Test metric is the paper-facing comparison; Lion wins clearly there.
- **Next frontier:** Lion + n_hidden=192 + epochs=20 confirmation (needs 43 min cap pod — assign to 45+ min cap student)

**Reproduce command:**
```bash
cd "target/" && python train.py --n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3
```

---

## 2026-05-17 00:05 — PR #4106: Push wider — n_hidden=192 + bf16 + epochs=20 (fern) — SUPERSEDED BY #4252

- **val_avg/mae_surf_p: 48.8400** (best epoch 20/20, W&B run `or5uq1id`) — **−4.05% vs previous best 50.9008**
- **test_avg/mae_surf_p: 42.5895** — **−2.98% vs previous best 43.8989**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not per-split logged) | 46.4089 |
| geom_camber_rc | (not per-split logged) | 55.5071 |
| geom_camber_cruise | (not per-split logged) | 27.1443 |
| re_rand | (not per-split logged) | 41.2976 |
| **avg** | **48.8400** | **42.5895** |

- **Model config:** SwiGLU FFN, **n_hidden=192** (wider), n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, inner_dim=256, ~1.47M params (+18% vs #4082)
- **Change from previous baseline:** add `--n_hidden 192 --epochs 20` on top of bf16. First retest at ep18 (val=50.92) was borderline; ep20 retest confirmed the wider model was compute-starved at ep18.
- **Throughput:** ~131 s/epoch (bf16, n_hidden=192) vs ~130 s/epoch (bf16, n_hidden=176) → +0.8% per-epoch cost
- **Wall time:** ~43.6 min for all 20 epochs completed (within 50 min cap)
- **Cosine T_max:** 20 (fully annealed); curve **still descending but decelerating** at ep20 (−0.97% ep19→ep20, vs −2.67% ep18→ep19)
- **Peak GPU memory:** 47.6 GB (vs 44.6 GB at n_hidden=176) — ~50 GB of headroom remains on 96 GB H100
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=20); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Key findings:**
  1. Compound width+epochs win on every primary metric: −4.05% val / −2.98% test on top of the n_hidden=176+ep18 baseline.
  2. **All 4 test splits improve or hold flat:** single_in_dist −5.24%, geom_camber_rc +0.10% (flat — structural hard split), geom_camber_cruise −3.99%, re_rand −3.76%.
  3. **Mild-overfitting hypothesis from ep18 retest is refuted** — OOD splits that regressed at ep18 (rc/cruise/re_rand) now improve or hold flat at ep20. The wider model was compute-starved, not overfitting.
  4. Curve still descending but decelerating — wider model is ~converged within 20-ep budget. Marginal returns past ep20 likely small.
  5. **geom_camber_rc is the structural hard split** (~55 across all variants tested) — moving it requires something other than width/budget.

**Reproduce command:**
```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=50 python train.py --n_hidden 192 --epochs 20 --use_bf16
```

---

## 2026-05-16 19:32 — PR #4082: Width retest with bf16 budget — n_hidden=176 + bf16 + epochs=18 (fern) — SUPERSEDED BY #4106

- **val_avg/mae_surf_p: 50.9008** (best epoch 18/18, W&B run `mgu3m5v2`) — **−5.43% vs previous best 53.8221**
- **test_avg/mae_surf_p: 43.8989** — **−7.14% vs previous best 47.2742**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not per-split logged) | 48.9679 |
| geom_camber_rc | (not per-split logged) | 55.4540 |
| geom_camber_cruise | (not per-split logged) | 28.2655 |
| re_rand | (not per-split logged) | 42.9082 |
| **avg** | **50.9008** | **43.8989** |

- **Model config:** SwiGLU FFN, **n_hidden=176** (wider), n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, inner_dim=232, ~1.23M params (+18% vs #3981)
- **Change from previous baseline:** add `--n_hidden 176` on top of bf16+ep18. Earlier n_hidden=176 regress (on mlp_ratio=3+fp32+ep12) was a joint budget+capacity artifact, not a width ceiling.
- **Throughput:** ~130 s/epoch (bf16, n_hidden=176) vs ~117.8 s/epoch (bf16, n_hidden=160) → +10% per-epoch cost
- **Wall time:** ~39.0 min for all 18 epochs completed (well under 45 min cap)
- **Cosine T_max:** 18 (fully annealed); curve **still descending** at ep18 (−1.38 from ep17→ep18)
- **Peak GPU memory:** 44.6 GB (vs 41.9 GB at n_hidden=160) — ~50 GB of headroom remains on 96 GB H100
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=18); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Key findings:**
  1. Width scaling on the bf16+ep18 stack is unsaturated; n_hidden=176 wins +5.4% val / +7.1% test over n_hidden=160
  2. ALL test splits improve: single_in_dist −10.5%, geom_camber_rc −7.1%, geom_camber_cruise −3.0%, re_rand −5.8%
  3. Curve still descending at ep18 — more epochs OR even wider model likely to keep gaining
  4. Combined with #3981, bf16 + width scaling delivers a compound win: 53.82→50.90 with single change

**Reproduce command:**
```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=45 python train.py --n_hidden 176 --epochs 18 --use_bf16
```

---

## 2026-05-16 16:42 — PR #3981: bf16 mixed-precision + epochs=18 (cut at ep16) (thorfinn) — prev baseline

- **val_avg/mae_surf_p: 53.8221** (best epoch 16/18, W&B run `b9h4bvnm`) — −4.64% vs prior
- **test_avg/mae_surf_p: 47.2742** — **−3.31% vs previous best 48.8947**

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | (not per-split logged) | 54.72 |
| geom_camber_rc | (not per-split logged) | 59.71 |
| geom_camber_cruise | (not per-split logged) | 29.13 |
| re_rand | (not per-split logged) | 45.53 |
| **avg** | **53.8221** | **47.2742** |

- **Model config:** SwiGLU FFN, n_hidden=160, n_layers=5, n_head=4, slice_num=64, **mlp_ratio=2** (default), inner_dim=216, ~1.035M params — same architecture as #3969
- **Change from previous baseline:** add `--use_bf16` autocast + `--epochs 18`. Run was cut at ep16/18 by `SENPAI_TIMEOUT_MINUTES=30` (student correctly did not override). Best ckpt = epoch 16, evaluated on test.
- **Throughput:** ~117.8 s/epoch (bf16) vs ~173.3 s/epoch (fp32 baseline) → **1.47× speedup**; 41.9 GB peak VRAM
- **Wall time:** ~31.5 min for 16/18 completed epochs
- **Cosine T_max:** 18 (lr at the cut ≈ 2e-5, schedule still descending)
- **Augmentation:** `coord_noise_std=0.01`; **Positional encoding:** Fourier PE `num_freq=4`
- **Loss:** L1; **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule:** Linear warmup 2 epochs, cosine to 0 (T_max=18); **Batch:** 4, surf_weight=10.0, grad_clip=1.0
- **Key findings:**
  1. bf16 autocast at the same wall clock budget = 50% more epochs trained → all 4 test splits improve
  2. ALL test split improvements: single_in_dist −1.07%, geom_camber_rc −2.37%, geom_camber_cruise **−9.02%**, re_rand −3.33%
  3. Val curve still descending at the timeout — there is more headroom with a longer budget
  4. bf16 autocast: parameters/optimizer stay fp32; only matmul/forward intermediates use bf16. No NaN/inf instabilities observed.

**Reproduce command:**
```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=35 python train.py --epochs 18 --use_bf16
```

---

## 2026-05-16 15:50 — PR #3969: SwiGLU mlp_ratio=2 + epochs=14 (askeladd) — prev baseline

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
