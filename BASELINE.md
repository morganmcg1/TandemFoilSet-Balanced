# BASELINE — TandemFoilSet (willow-pai2i-24h-r4)

## Current best — PR #3257 (frieren, merged 2026-05-15 18:25 UTC)

**Surface MAE loss + pressure-channel weight 3× + canonical NaN guard.**

| Metric | Value | W&B run | Δ vs prior baseline |
|--------|------:|---------|---------------------|
| `val_avg/mae_surf_p` | **106.67** | `szru1ogx` (frieren) | **−9.5%** (from 117.89) |
| `test_avg/mae_surf_p` | **94.35** | `szru1ogx` (frieren) | **−11.2%** (from 106.23) |
| `test_single_in_dist/mae_surf_p` | 122.34 | `szru1ogx` | |
| `test_geom_camber_rc/mae_surf_p` | 106.31 | `szru1ogx` | |
| `test_geom_camber_cruise/mae_surf_p` | 62.47 | `szru1ogx` | |
| `test_re_rand/mae_surf_p` | 86.28 | `szru1ogx` | |

### What changed
- **Loss reformulation:** `train.py` loss replaced MSE with surface-volume MAE + per-channel weight `[1, 1, 3]` on (Ux, Uy, p). Surface nodes still weighted 10× via `surf_weight`.
- **Canonical NaN fix:** `evaluate_split` skips non-finite-y samples in the mask and `nan_to_num`s y before `accumulate_batch`. Root cause was `+inf` in GT at `test_geom_camber_cruise_gt/000020.pt` (y[..., p] at ~761 nodes) → `inf * 0 = NaN` would otherwise poison the masked sum.

### Model config (unchanged from vanilla)
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10, epochs=50`
- `dropout=0.0, grad_clip=none, warmup=none`, cosine LR `T_max=50` (still mismatched at ~14 epochs trained)
- Wall-clock: 30-min cap, 13/14 epochs of 50 completed

### Reproduce command

```bash
cd target && python train.py --wandb_group surf-mae-pweight --wandb_name surf-mae-pweight3
```

---

## History

| Date | PR | Hypothesis | val_avg | test_avg | Merge |
|------|----|------------|--------:|--------:|:-----:|
| 2026-05-15 | #3257 (frieren) | Surface MAE + p-weight 3× + NaN guard | **106.67** | **94.35** | ✓ R1#1 |
| — | vanilla (`xfayvdk2`, alphonse) | NaN-guarded baseline | 117.89 | 106.23 | pre-R1 anchor |
| — | vanilla (`17fia1vd`, edward) | unguarded baseline | 128.34 | NaN | ref only |
| — | vanilla (`nylo2tvd`, fern) | unguarded baseline | 141.94 | NaN | ref only |

Run-to-run variance on unclipped vanilla baselines is ~13pt on val_avg (fern's #3258 grad-norm trace shows median 56, peak 1110). Frieren's win should be reproducible on a fresh seed but margin may shrink ±3pt.
