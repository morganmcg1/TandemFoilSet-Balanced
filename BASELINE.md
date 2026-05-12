# Baseline — `icml-appendix-charlie-pai2g-24h-r4`

Fresh start for round 4 of the Charlie / pai2g 24h logging ablation. No
prior experiments on this branch — the implicit baseline is the unmodified
`train.py` config inherited from `icml-appendix-charlie`. The first merged
winner sets the first numeric reference value.

## Reference config (default `train.py` at HEAD)

- **Model**: Transolver
  - `n_hidden = 128`
  - `n_layers = 5`
  - `n_head = 4`
  - `slice_num = 64`
  - `mlp_ratio = 2`
  - `space_dim = 2`, `fun_dim = X_DIM - 2 = 22`
  - `out_dim = 3` (`Ux`, `Uy`, `p`)
  - `unified_pos = False`
- **Optimizer**: AdamW (`lr = 5e-4`, `weight_decay = 1e-4`)
- **LR schedule**: CosineAnnealingLR with `T_max = MAX_EPOCHS`
- **Loss**: MSE in normalized target space, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight = 10.0`
- **Batch size**: `4`
- **Epochs**: configured `50`, capped by `SENPAI_TIMEOUT_MINUTES = 30`
- **Sampler**: `WeightedRandomSampler` with equal-domain weights from `meta.json`

## Metrics contract

- Primary ranking metric: `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits.
- Paper-facing comparison metric: `test_avg/mae_surf_p` — same aggregation on the four test splits at the best val checkpoint.
- All metrics computed in physical (denormalized) units in `data/scoring.py`.

## Current best result

- **PR**: _none — no winners merged yet on this branch_
- **`val_avg/mae_surf_p`**: TBD (to be set by first merged winner)
- **`test_avg/mae_surf_p`**: TBD

## Reproduce baseline

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <student>/baseline
```
