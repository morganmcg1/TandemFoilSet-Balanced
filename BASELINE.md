# Baseline — TandemFoilSet CFD Surrogate (icml-appendix-charlie-pai2e-r4)

Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## 2026-04-28 20:00 — PR #738: Surface loss weight: 10 → 20 to prioritize surface MAE

- **Branch:** charliepai2e4-edward/higher-surf-weight-20
- **Best epoch:** 13 of 14 (30-min timeout — model still converging)
- **Surface MAE (val, best ckpt):** Ux=2.4441, Uy=0.8943, **p=128.8320**
- **Volume MAE (val, best ckpt):** Ux=5.8291, Uy=2.5823, p=145.0063
- **val_avg/mae_surf_p: 128.8320** ← current best
- **Metric summary:** `target/metrics/charliepai2e4-edward-higher-surf-weight-20-wnnqnvav.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 2.2478 | 0.9177 | 157.1632 | 6.5801 | 2.7364 | 183.4634 |
| val_geom_camber_rc     | 3.0704 | 1.0829 | 136.6349 | 6.1857 | 3.2678 | 140.8892 |
| val_geom_camber_cruise | 1.9694 | 0.7030 | 100.2357 | 4.9168 | 1.8728 | 125.3909 |
| val_re_rand            | 2.4887 | 0.8736 | 121.2942 | 5.6338 | 2.4523 | 130.2818 |
| **avg**                | **2.4441** | **0.8943** | **128.8320** | **5.8291** | **2.5823** | **145.0063** |

### Notes
- First merged experiment on this track — establishes the baseline.
- `surf_weight=20` (doubled from default 10) pushes surface accuracy below volume accuracy on every split (avg surf/vol ratio 0.889).
- `test_geom_camber_cruise` pressure is NaN (single-sample overflow in scoring pipeline — not specific to this config).
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, cosine over 50 epochs.

---

## 2026-04-28 — PR #812: Lower LR 5e-4 → 2e-4 with surf_weight=20 for smoother convergence

- **Branch:** charliepai2e4-edward/lower-lr-2e-4-surf-weight-20
- **Best epoch:** 14 of 14 (30-min timeout — best=last suggests further gains with more time)
- **Surface MAE (val, best ckpt):** Ux=1.8782, Uy=0.7963, **p=112.9366**
- **Volume MAE (val, best ckpt):** Ux=5.2301, Uy=2.6142, p=132.0728
- **val_avg/mae_surf_p: 112.9366** ← current best (was 128.8320, **−12.3%**)
- **Metric summary:** `metrics/charliepai2e4-edward-lower-lr-2e-4-surf-weight-20-5vwlbqdz.jsonl`
- **Reproduce:** `cd target/ && python train.py --lr 2e-4 --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.6891 | 0.8207 | 132.7534 | 5.3934 | 2.6561 | 165.0441 |
| val_geom_camber_rc     | 2.4081 | 0.9279 | 125.5523 | 5.7861 | 3.1474 | 129.0143 |
| val_geom_camber_cruise | 1.5603 | 0.6291 |  91.5413 | 4.6879 | 2.0143 | 116.7482 |
| val_re_rand            | 1.8552 | 0.8074 | 101.9034 | 5.0529 | 2.6389 | 117.5847 |
| **avg**                | **1.8782** | **0.7963** | **112.9366** | **5.2301** | **2.6142** | **132.0728** |

### Notes
- Reducing LR from 5e-4 to 2e-4 with `surf_weight=20` gave a 12.3% improvement on the primary metric.
- Best epoch = last epoch: model was still converging at the 30-min wall-clock cap.
- Cosine annealing `T_max` set to actual achievable epochs (~14), not MAX_EPOCHS=50.
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.

---

## 2026-04-28 — PR #735: Wider hidden dim 192, deeper 6 layers, surf_weight=20 (alphonse)

- **Branch:** charliepai2e4-alphonse/wider-hidden-256 (fork: morganmcg1/TandemFoilSet-Balanced)
- **Best epoch:** 9 of ~11 achievable
- **Surface MAE (val, best ckpt):** Ux=2.0889, Uy=0.9002, **p=128.3833**
- **Volume MAE (val, best ckpt):** Ux=5.6798, Uy=2.5484, p=138.1776
- **val_avg/mae_surf_p: 128.3833** — beat old PR #738 baseline (128.8320, −0.35%) but below current best PR #812 (112.9366)
- **Metric summary:** `metrics/charliepai2e4-alphonse-wider192-deep6-sw20-tmax11-qme35ium.jsonl`
- **Reproduce:** `cd target/ && python train.py --n_hidden 192 --n_layers 6 --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 2.0038 | 0.9508 | 156.3134 | 6.4149 | 2.7374 | 181.2448 |
| val_geom_camber_rc     | 2.7272 | 1.0610 | 143.9932 | 5.9975 | 3.1286 | 137.1337 |
| val_geom_camber_cruise | 1.6257 | 0.6789 |  97.4374 | 4.8069 | 1.8280 | 122.9547 |
| val_re_rand            | 1.9991 | 0.9101 | 115.7893 | 5.5000 | 2.4998 | 111.3773 |
| **avg**                | **2.0889** | **0.9002** | **128.3833** | **5.6798** | **2.5484** | **138.1776** |

### Notes
- Widening hidden dim from 128 to 192 and adding a 6th layer beat the very first baseline (PR #738: 128.83) by 0.35%.
- However, this run predates the LR reduction in PR #812 — combining wider architecture with lr=2e-4 is a strong candidate for further improvement.
- PR closed manually (metrics cherry-picked): fork branch had structural conflicts due to divergent repo layout (train.py at root vs target/).
- Best epoch was not the last — model had headroom; best=epoch 9 of 11.
- Architecture: `n_hidden=192`, `n_layers=6`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `surf_weight=20`, cosine T_max=11.
