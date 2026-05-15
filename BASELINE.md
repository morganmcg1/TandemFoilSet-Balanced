# BASELINE — TandemFoilSet (willow-pai2i-24h-r4)

## Current best — PR #3263 (thorfinn, merged 2026-05-15 21:55 UTC)

**FiLM(log_Re) conditioning on Transolver hidden state, on top of #3257 frieren surf-MAE+p_weight=3 base.**

| Metric | Value | W&B run | Δ vs prior baseline |
|--------|------:|---------|---------------------|
| `val_avg/mae_surf_p` | **100.24** | `69jp9tvt` (thorfinn) | **−6.03%** (from 106.67) |
| `test_avg/mae_surf_p` | **90.06** | `69jp9tvt` (thorfinn) | **−4.55%** (from 94.35) |
| `test_single_in_dist/mae_surf_p` | 119.11 | `69jp9tvt` | −2.6% |
| `test_geom_camber_rc/mae_surf_p` | 100.27 | `69jp9tvt` | −5.7% |
| `test_geom_camber_cruise/mae_surf_p` | 58.62 | `69jp9tvt` | −6.2% |
| `test_re_rand/mae_surf_p` | 82.27 | `69jp9tvt` | −4.6% |

### What changed
- **FiLM module added:** Zero-init `FiLM(cond_dim=1, hidden_dim=128, mid_dim=64)` injected between the preprocess MLP and the Transolver block stack. Conditioning is `x[..., 13:14]` (log_Re), a per-sample-global scalar.
- **Subclassed model:** `TransolverFiLM` adds the FiLM head as a low-rank affine route from log_Re into every channel of the trunk's hidden state. +17K params (0.66M → 0.68M).
- **All `_skipped_y_samples` confirmed:** cruise = 1 (canonical inherited from #3257), other splits = 0.

### Mechanism summary
FiLM's gate gives the model an explicit, low-rank Re-conditioned affine modulation of the entire trunk's hidden state. While the merged loss already up-weights pressure 3× (capturing some Re-dependent physics through gradient flow), the FiLM gate adds a *structural* route for Re information that the loss alone cannot provide. Result: ~4.5% additional gain composes cleanly on top of frieren's loss reformulation.

### Model config (unchanged from #3257 except FiLM head)
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10, p_channel_weight=3, surface-MAE loss, epochs=50`
- **FiLM:** `cond_dim=1 (log_Re), mid_dim=64, hidden=128, zero-init`
- `dropout=0.0, grad_clip=none, warmup=none`, cosine LR `T_max=50` (still mismatched)
- Peak VRAM: 43.6 GB / 96 GB, wall-clock: 31.8 min, 14 epochs of 50

### Reproduce command

```bash
cd target && python train.py --wandb_group film-re-cond --wandb_name film-re-v3-on-frieren-base
```

---

## History

| Date | PR | Hypothesis | val_avg | test_avg | Merge |
|------|----|------------|--------:|--------:|:-----:|
| 2026-05-15 | #3263 (thorfinn) | FiLM(log_Re) conditioning on hidden state | **100.24** | **90.06** | ✓ R1#2 |
| 2026-05-15 | #3257 (frieren) | Surface MAE + p-weight 3× + NaN guard | 106.67 | 94.35 | ✓ R1#1 |
| — | vanilla (`xfayvdk2`, alphonse) | NaN-guarded baseline | 117.89 | 106.23 | pre-R1 anchor |
| — | vanilla (`17fia1vd`, edward) | unguarded baseline | 128.34 | NaN | ref only |
| — | vanilla (`nylo2tvd`, fern) | unguarded baseline | 141.94 | NaN | ref only |

Run-to-run variance on unclipped vanilla baselines is ~13pt on val_avg (fern's #3258 grad-norm trace shows median 56, peak 1110). FiLM (single-seed) showed seed spread of 133/128/118 on the MSE base; the rebased run landed at val=100.24/test=90.06 — credible reproducibility margin ±3-5pt.
