# BASELINE — TandemFoilSet (willow-pai2i-24h-r4)

## Current best (unmodified Transolver, NaN-guard applied)

**As of 2026-05-15 17:30 UTC — no experiment has merged yet. This is the vanilla reference.**

| Metric | Value | W&B run | Notes |
|--------|------:|---------|-------|
| `val_avg/mae_surf_p` | 117.89 | `xfayvdk2` (alphonse) | Best of measured unclipped vanilla runs |
| `test_avg/mae_surf_p` | **106.23** | `xfayvdk2` (alphonse) | First finite 4-split test_avg (NaN guard applied) |
| `test_single_in_dist/mae_surf_p` | 126.60 | `xfayvdk2` | |
| `test_geom_camber_rc/mae_surf_p` | 113.67 | `xfayvdk2` | |
| `test_geom_camber_cruise/mae_surf_p` | 78.72 | `xfayvdk2` | NaN on all other runs without NaN guard |
| `test_re_rand/mae_surf_p` | 105.92 | `xfayvdk2` | |

### Model config (vanilla Transolver)
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10, epochs=50`
- `dropout=0.0, grad_clip=none, warmup=none, MSE loss`
- Wall-clock: 30-min cap, ~13–14 epochs of 50 completed

### Run-to-run variance (unclipped baseline, important caveat)

Without grad clipping, the vanilla Transolver shows large run-to-run variance (~13pt on val_avg):

| Run | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | NaN-guard applied |
|-----|---------------------:|----------------------:|:-----------------:|
| `xfayvdk2` (alphonse) | 117.89 | **106.23** | ✓ |
| `17fia1vd` (edward) | 128.34 | 127.29 (3-split) | ✗ |
| `nylo2tvd` (fern) | 141.94 | 139.34 (3-split) | ✗ |

Fern's #3258 (grad-clip+warmup) demonstrates that the gradient norms are median 56, peak 1110 — this is why variance is so high. Once fern's clip+warmup lands, the baseline will be more reproducible.

### Reproduce command

```bash
cd target && python train.py
```

Apply the NaN guard in `evaluate_split` to get finite test metrics (see frieren's #3257 commit `34600cf` for the canonical patch).

---

## History

| Date | PR | Hypothesis | val_avg | test_avg | Merge |
|------|----|------------|--------:|--------:|:-----:|
| — | vanilla | Starting point (no improvement merged yet) | 117.89 | **106.23** | pre-R1 |
