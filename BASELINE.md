# Baseline Metrics — icml-appendix-charlie-pai2-r3

## TandemFoilSet

- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits)
- **Current best:** Not yet established — Round 1 sweeps in progress

### Vanilla Transolver defaults (target/train.py at HEAD)
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- Optimizer: AdamW, Scheduler: CosineAnnealingLR (T_max=epochs)
- Model: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, act=gelu, dropout=0.0)
- Loss: MSE normalized space, `vol_loss + surf_weight * surf_loss`

### Historical reference (kagent_v_students track — different cluster, not directly comparable)
The prior track established the following improvement trajectory:
- Vanilla MSE baseline: ~131.99 val_avg/mae_surf_p
- L1 loss (−22%): ~103.04
- surf_weight=1 (−9.6%): ~93.13
- AMP + grad_accum=4 (−5.2%): ~88.27
- Fourier PE σ=0.7 + SwiGLU: ~70.67
- n_head=2, sn=32: ~60.82
- nl=3, sn=32: ~54.48
- nl=3, sn=16 compound: ~49.44

These numbers are from a different hardware/cluster and may not directly reproduce on pai-2. Round 1 will re-establish the baseline on this cluster.

## Reproduce command

```bash
cd target && python train.py --agent <student_name> --experiment_name <experiment_name>
```

Use `python train.py --help` to see all available CLI flags.
