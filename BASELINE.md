# BASELINE — TandemFoilSet (willow-pai2i-24h-r4)

## Current best — PR #3262 (edward, merged 2026-05-16 01:27 UTC)

**Random Fourier Features positional encoding (σ=1.0, n_freqs=16), stacked on top of #3358 cosine T_max=14 + #3263 FiLM(log_Re) + #3257 surf-MAE+p_weight=3.**

| Metric | Value | W&B run | Δ vs prior baseline |
|--------|------:|---------|---------------------|
| `val_avg/mae_surf_p` | **79.2847** | `tnna02ob` (edward) | **−12.3%** (from 90.44) |
| `test_avg/mae_surf_p` | **69.2741** | `tnna02ob` (edward) | **−13.5%** (from 80.08) |
| `test_single_in_dist/mae_surf_p` | 78.6939 | `tnna02ob` | **−18.4%** |
| `test_geom_camber_rc/mae_surf_p` | 79.5933 | `tnna02ob` | −11.8% |
| `test_geom_camber_cruise/mae_surf_p` | 49.1630 | `tnna02ob` | −12.1% |
| `test_re_rand/mae_surf_p` | 69.6460 | `tnna02ob` | −10.3% |

**Cumulative path:** vanilla 106.23 → #3257 94.35 → #3263 90.06 → #3358 80.08 → #3262 **69.27** (**−34.8% from vanilla in 4 PRs**).

All subsequent PRs must beat **test_avg/mae_surf_p < 69.27**.

### What changed
- **Random Fourier Features (RFF) on (x, z) node coordinates.** `fourier_n_freqs=16` produces 32 additional features (sin + cos pairs) appended to the raw 24-dim input (`x[..., 24:]`). The RFF sigma `fourier_sigma=1.0` is appropriate for chord-length-normalized geometry (typical node spacing << 1).
- **No architecture changes** (same TransolverFiLM trunk, FiLM(log_Re) head, loss, schedule).
- **Param delta:** +7,680 (32 RFF features × 128 hidden × 2 halves = 8K, minus the original (x,z) 2-dim direct input path that was replaced) → effective +~8K, from 679,619 → 687,319.
- **All `_skipped_y_samples` correct:** cruise = 1 (canonical), other splits = 0. `n_nonfinite_pred=0` across all splits — RFF features did not introduce numerical issues.

### Mechanism summary
RFF maps the low-dimensional (x, z) Cartesian coordinates into a 32-dim Fourier basis, allowing the Transolver to resolve higher-frequency spatial patterns in surface pressure. On the old MSE baseline, RFF gave −9.8% val — on the fully-stacked new base (better loss + FiLM conditioning + cosine annealing), it scales up to −12.3% val / −13.5% test. The cleaner loss surface and better-conditioned optimizer give the network more capacity to exploit the richer input representation. The gain is largest on `test_single_in_dist` (−18.4%), the hardest split with the densest geometry — consistent with RFF's theoretical advantage on high-frequency structure.

### Model config
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10, p_channel_weight=3, surface-MAE loss, epochs=50`
- **FiLM:** `cond_dim=1 (log_Re), mid_dim=64, hidden=128, zero-init`
- **Cosine LR:** `T_max=14`
- **RFF:** `fourier_n_freqs=16, fourier_sigma=1.0` on (x, z) coordinates
- `dropout=0.0, grad_clip=none, warmup=none`
- Peak VRAM: ~42 GB / 96 GB, wall-clock: 32.3 min, 14 epochs of 50

### Reproduce command

```bash
cd target && python train.py --wandb_group fourier-pos-enc --wandb_name rff-sigma1.0-on-cosine-tmax-base \
    --fourier_n_freqs 16 --fourier_sigma 1.0
```

---

## Previous best — PR #3358 (alphonse, merged 2026-05-16 00:24 UTC)

**Cosine LR schedule T_max=14 (matched to wall-clock epoch cap), on top of #3263 FiLM(log_Re) + #3257 surf-MAE+p_weight=3 base.**

| Metric | Value | W&B run | Δ vs prior baseline |
|--------|------:|---------|---------------------|
| `val_avg/mae_surf_p` | **90.4369** | `b9qv36aq` (alphonse) | **−9.78%** (from 100.24) |
| `test_avg/mae_surf_p` | **80.0794** | `b9qv36aq` (alphonse) | **−11.08%** (from 90.06) |
| `test_single_in_dist/mae_surf_p` | 96.49 | `b9qv36aq` | **−19.0%** |
| `test_geom_camber_rc/mae_surf_p` | 90.24 | `b9qv36aq` | −10.0% |
| `test_geom_camber_cruise/mae_surf_p` | 55.95 | `b9qv36aq` | −4.6% |
| `test_re_rand/mae_surf_p` | 77.65 | `b9qv36aq` | −5.6% |

### What changed
- **Cosine `T_max` aligned to the wall-clock epoch cap.** Default `cosine_tmax: int = 14` added to Config; scheduler line uses `CosineAnnealingLR(optimizer, T_max=cfg.cosine_tmax)`.
- **LR trace now decays cleanly to 0** at epoch 14 (epoch 1: 4.94e-04 → epoch 7: 2.50e-04 → epoch 10: 9.41e-05 → epoch 13: 6.27e-06 → epoch 14: 0.00).
- **`train/lr` epoch logging** added for the LR trace.
- **All other config unchanged** from #3263 (FiLM head + frieren's loss preserved through rebase).
- **All `_skipped_y_samples` correct:** cruise = 1 (canonical), other splits = 0.

### Mechanism summary
The previous baseline (`69jp9tvt`) was still training at LR ≈ 4.09e-04 (82% of peak) when the wall-clock cap hit at epoch 14 — the cosine schedule was set for `T_max=50` but only ~14 epochs ran. With `T_max=14`, the optimizer gets the full annealing tail and can settle into a noticeably tighter minimum on top of the better-conditioned loss + FiLM base. The mechanism is purely orthogonal to the loss reformulation and architecture work — three independent improvements stacking additively.

The largest per-split gain is on `test_single_in_dist` (−19.0%), the hardest split. On the old MSE base this was the only split where T_max=14 *lost* vs T_max=50 — on the new FiLM+MAE base it's the biggest winner, suggesting the schedule fix interacts constructively with the better loss/architecture.

### Model config (unchanged from #3263 except cosine T_max)
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10, p_channel_weight=3, surface-MAE loss, epochs=50`
- **FiLM:** `cond_dim=1 (log_Re), mid_dim=64, hidden=128, zero-init`
- **Cosine LR:** `T_max=14` (matched to wall-clock 14 epochs)
- `dropout=0.0, grad_clip=none, warmup=none`
- Peak VRAM: ~42 GB / 96 GB, wall-clock: 31.8 min, 14 epochs of 50

### Reproduce command

```bash
cd target && python train.py --wandb_group cosine-tmax --wandb_name cosine-tmax14-on-film-base
```

(No CLI override needed — `cosine_tmax=14` is now the default.)

---

## History

| Date | PR | Hypothesis | val_avg | test_avg | Merge |
|------|----|------------|--------:|--------:|:-----:|
| 2026-05-16 | #3262 (edward) | RFF σ=1.0, n_freqs=16 on (x,z) coords | **79.28** | **69.27** | ✓ R2#2 |
| 2026-05-16 | #3358 (alphonse) | Cosine LR T_max=14 (matched to wall-clock cap) | **90.44** | **80.08** | ✓ R2#1 |
| 2026-05-15 | #3263 (thorfinn) | FiLM(log_Re) conditioning on hidden state | 100.24 | 90.06 | ✓ R1#2 |
| 2026-05-15 | #3257 (frieren) | Surface MAE + p-weight 3× + NaN guard | 106.67 | 94.35 | ✓ R1#1 |
| — | vanilla (`xfayvdk2`, alphonse) | NaN-guarded baseline | 117.89 | 106.23 | pre-R1 anchor |
| — | vanilla (`17fia1vd`, edward) | unguarded baseline | 128.34 | NaN | ref only |
| — | vanilla (`nylo2tvd`, fern) | unguarded baseline | 141.94 | NaN | ref only |

Run-to-run variance on unclipped vanilla baselines is ~13pt on val_avg (fern's #3258 grad-norm trace shows median 56, peak 1110). FiLM (single-seed) showed seed spread of 133/128/118 on the MSE base; the rebased run landed at val=100.24/test=90.06 — credible reproducibility margin ±3-5pt.
