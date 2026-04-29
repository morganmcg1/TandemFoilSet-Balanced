# SENPAI Research Results

## 2026-04-28 21:30 — PR #735: Wider hidden dim: 128 → 256 for more model capacity

- **Branch:** charliepai2e4-alphonse/wider-hidden-256
- **Hypothesis:** Doubling n_hidden from 128 to 256 gives the model more representational capacity to capture sharp pressure gradients near airfoil surfaces.
- **Results (best checkpoint, epoch 9/9, surf_weight=20, T_max=9):**

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | mae_vol_Ux | mae_vol_Uy | mae_vol_p |
|-------|-------------|-------------|------------|------------|------------|-----------| 
| val_single_in_dist | 1.7986 | 0.8996 | 159.3673 | 6.7937 | 2.6951 | 175.3428 |
| val_geom_camber_rc | 3.2967 | 1.1988 | 145.1024 | 7.8562 | 3.2722 | 142.9192 |
| val_geom_camber_cruise | 1.3705 | 0.6349 | 101.0673 | 4.5688 | 1.7380 | 120.4071 |
| val_re_rand | 2.1119 | 0.8848 | 117.8924 | 5.7632 | 2.3745 | 123.4097 |
| **avg** | **2.1444** | **0.9045** | **130.857** | **6.2455** | **2.5200** | **140.5197** |

- **Primary metric:** `val_avg/mae_surf_p = 130.857` vs baseline 128.832 — **did NOT beat baseline (+1.6%)**
- **Metric summary:** `target/metrics/charliepai2e4-alphonse-wider-hidden-256-sw20-tmax9-0dgn8uv3.jsonl`
- **Decision:** Request changes — promising direction but width alone costs too many iterations in the 30-min budget.

### Analysis and Conclusions

The first run was misconfigured (surf_weight=10, T_max=50 for 9 actual epochs). After corrections (surf_weight=20, T_max=9), the metric improved from 140.06 → 130.86, but still 2.03 points above baseline. The key issue: `n_hidden=256` makes each epoch ~2× more expensive, so the model only sees 9 epochs vs ~17 for the baseline. Despite full cosine schedule completion (LR reached 0 by epoch 9), the model is still converging steeply — the final epoch dropped 5 points (135.89 → 130.86). 

**The width hypothesis is untested under equal-compute conditions.** Under equal wall-clock, wider lost to narrower. Under equal epochs (not tested), the wider model might win. The student's per-split analysis shows data heterogeneity (val_geom_camber_cruise at 101 vs val_single_in_dist at 159) is a major factor — addressing this via per-channel weighting or Re-stratified sampling may have higher ROI than raw capacity scaling.

**Suggested next direction:** Try a mixed-capacity design — n_hidden=192, n_layers=6 — which adds ~20% params but keeps iteration speed tolerable (~14 epochs in 30 min). Combined with surf_weight=20 and T_max calibrated to achievable epochs.

## 2026-04-28 22:15 — PR #740: Deeper Transolver: 5 → 7 layers for richer feature composition

- **Branch:** charliepai2e4-fern/deeper-layers-7
- **Hypothesis:** Adding 2 more Transolver blocks (n_layers=5→7) allows the model to compose richer multi-scale features and better capture pressure gradients near airfoil surfaces.
- **Results (best checkpoint, epoch 10/12, surf_weight=20, T_max=12):**

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | mae_vol_Ux | mae_vol_Uy | mae_vol_p |
|-------|-------------|-------------|------------|------------|------------|-----------|
| val_single_in_dist | — | — | 168.52 | — | — | — |
| val_geom_camber_rc | — | — | 143.27 | — | — | — |
| val_geom_camber_cruise | — | — | 106.95 | — | — | — |
| val_re_rand | — | — | 124.87 | — | — | — |
| **avg** | — | — | **135.90** | — | — | — |

- **Primary metric:** `val_avg/mae_surf_p = 135.90` vs baseline 128.832 — **did NOT beat baseline (+5.5%)**
- **Decision:** Closed as dead end.

### Analysis and Conclusions

Run 1 was misconfigured (surf_weight=10, T_max=50). After corrections (surf_weight=20, T_max=12), the 7-layer model still failed to beat baseline. The core issue: n_layers=7 costs ~180s/epoch vs ~131s/epoch for baseline (43% overhead), so only 10 epochs fit in the 30-min budget vs ~14 for the 5-layer model. Even with proper LR schedule calibration (LR decayed to 3.35e-5 = 7% of start by epoch 10), the model was still descending at termination (142.28 → 135.90 in last 2 epochs), suggesting it needed more epochs. Under fixed wall-clock, depth loses to baseline depth. The depth hypothesis is rejected at this compute budget. Per-split heterogeneity remains substantial (val_single_in_dist=168.52 vs val_geom_camber_cruise=106.95, a 62-point spread), echoing the finding from PR #735 and suggesting data regime heterogeneity is the primary lever to address.

## 2026-04-29 03:00 — PR #935: Extended training epochs=18, T_max=18 on current best: per_sample_norm_mse + lr=2e-4

- **Branch:** charliepai2e4-frieren/extended-epochs-18-per-sample-norm-lr-2e-4
- **Hypothesis:** Extending epochs from 14→18 with T_max=18 keeps a positive learning rate at epoch 14 (the 30-min timeout cutoff), allowing continued gradient flow vs T_max=14 which parks LR≈0 at that epoch. Architecture change: spatial_bias input 2D→4D (adding saf_0 and saf_1 shape coordinates).
- **Results (best checkpoint, epoch 14/18, wall-clock timeout):**

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | mae_vol_Ux | mae_vol_Uy | mae_vol_p |
|-------|-------------|-------------|------------|------------|------------|-----------|
| val_single_in_dist     | 1.2984 | 0.6816 | 109.0267 | 4.9698 | 2.1102 | 144.2117 |
| val_geom_camber_rc     | 1.9591 | 0.8680 | 108.3369 | 5.1580 | 2.7046 | 115.3989 |
| val_geom_camber_cruise | 0.7416 | 0.4688 |  73.1648 | 3.1743 | 1.2715 |  79.3267 |
| val_re_rand            | 1.1660 | 0.7089 |  88.2387 | 4.1590 | 1.8454 |  92.1751 |
| **avg**                | **1.2913** | **0.6843** | **94.6918** | **4.3653** | **1.9829** | **110.2781** |

- **Primary metric:** `val_avg/mae_surf_p = 94.6918` vs baseline 95.6617 — **WINNER (−1.01%)**
- **Metric summary:** `target/metrics/charliepai2e4-frieren-extended-epochs-18-per-sample-norm-lr-2e-4-alsxfigk.jsonl`
- **Decision:** MERGED as winner.

### Analysis and Conclusions

Two changes contributed: (1) T_max=18 cosine schedule shape — at epoch 14 (timeout), LR is still at ~16% of peak (~3.2e-5) rather than ≈0 under T_max=14, giving continued gradient signal. (2) 4D spatial bias using saf_0/saf_1 coordinates alongside x/y provides richer shape-aware context to the attention bias. Best=last pattern confirmed again — the model continues improving through the full 30-min budget. val_single_in_dist (109.03) remains the hardest split, but the gap to val_geom_camber_rc (108.34) narrowed vs PR #871, suggesting the 4D spatial bias helps with geometry-OOD generalization.

The improvement is modest at −1.01%, but compounding. The T_max schedule shape insight is reusable: setting T_max > achievable_epochs is a low-cost way to maintain gradient flow at the timeout boundary.

## 2026-04-29 03:05 — PR #932: Zero weight_decay (0.0) on current best: per_sample_norm_mse + lr=2e-4

- **Branch:** charliepai2e4-edward/zero-weight-decay-per-sample-norm-lr-2e-4
- **Hypothesis:** Per_sample_norm_mse normalizes per-sample gradient scale, potentially making explicit L2 regularization redundant.
- **Results (best checkpoint):**

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist     | ~127.27 |
| val_geom_camber_rc     | ~115.50 |
| val_geom_camber_cruise | ~75.89 |
| val_re_rand            | ~78.38 |
| **avg** | **99.2564** |

- **Primary metric:** `val_avg/mae_surf_p = 99.2564` vs baseline 95.6617 — **FAILED (+3.77%)**
- **Decision:** CLOSED as dead end.

### Analysis and Conclusions

Removing weight_decay causes mild overfitting at lr=2e-4: training loss is lower but validation surface pressure MAE regresses by 3.77% vs the best stack. The per_sample_norm loss normalizes gradient scale across Re-regimes but does NOT replace L2 regularization. At lr=2e-4 with gradient clipping active, weight_decay=1e-4 contributes meaningful regularization. The experiment confirms weight_decay=1e-4 should remain fixed in the best-stack config.

<!-- Format: ## <YYYY-MM-DD HH:MM> — PR #<number>: <title> -->
