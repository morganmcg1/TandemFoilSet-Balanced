# Baseline — icml-appendix-charlie-pai2e-r3

## Current Best
- **Source**: unmodified `train.py` (default config)
- **PR**: _none yet — first round in flight_
- **Primary**: `val_avg/mae_surf_p` (lower is better)

## Default config (the bar to beat)
```
Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)
AdamW(lr=5e-4, weight_decay=1e-4)
CosineAnnealingLR(T_max=epochs)
batch_size=4, surf_weight=10, epochs=50
WeightedRandomSampler over 3 domains (rc-single, rc-tandem, cruise-tandem)
loss = vol_mse + 10 · surf_mse  (in normalized space)
```

## Metrics to report
Each PR must report from the best-val-checkpoint test evaluation:
- `test_avg/mae_surf_p` (primary)
- per-split `test/<split>/mae_surf_p`
- per-split `test/<split>/mae_vol_{Ux,Uy,p}`
- training wall-clock (min) and peak VRAM (GB)

## Notes
- Branch is `icml-appendix-charlie-pai2e-r3`; PRs target it as base; merges squash into it.
- 4 val splits = `val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`.
- Camber-holdout splits (`val_geom_camber_*`) are typically hardest — front-foil shape unseen at train time.
- Tag every PR with `charlie-pai2e-r3`.
