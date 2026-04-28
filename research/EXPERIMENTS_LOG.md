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

<!-- Format: ## <YYYY-MM-DD HH:MM> — PR #<number>: <title> -->
