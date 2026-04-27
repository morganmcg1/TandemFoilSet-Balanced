# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round-1 ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #312, alphonse, 2026-04-27)

- **`val_avg/mae_surf_p` = 144.2118** at epoch 10 (of 14 completed)
- **`test_avg/mae_surf_p` = 131.1823** (best val checkpoint)
- W&B run: [`x33nmv34` / `baseline-default-r1`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/x33nmv34)
- Peak GPU memory: 42.1 GB (of 96 GB) — large headroom for bigger batches.
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` was binding (~131 s/epoch).

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 169.7020 | 1.9334 | 0.9404 |
| val_geom_camber_rc | 170.3406 | 2.9992 | 1.2416 |
| val_geom_camber_cruise | 110.6985 | 1.5809 | 0.6804 |
| val_re_rand | 126.1063 | 2.2099 | 0.9143 |
| **val_avg** | **144.2118** | 2.1809 | 0.9442 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 150.3920 | 1.8518 | 0.8808 |
| test_geom_camber_rc | 155.0540 | 2.8913 | 1.1783 |
| test_geom_camber_cruise | 93.2915 | 1.6136 | 0.6313 |
| test_re_rand | 125.9919 | 2.1377 | 0.9069 |
| **test_avg** | **131.1823** | 2.1236 | 0.8993 |

## Default config (matches PR #312)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- Loss: per-channel-equal MSE in normalized space, with surface vs. volume
  split via `surf_weight`
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22`, `out_dim=3`)

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-default-r1 \
  --wandb_name baseline-default-r1
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- Test-time number reported alongside: `test_avg/mae_surf_p`.
- Both are equal-weight means across the four val/test tracks.
- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) is **binding** at the
  current model size — only ~14 of 50 epochs finished. The cosine LR
  schedule was specified for 50 epochs, so the LR barely decayed before the
  cap. Throughput improvements (AMP/bf16, larger batch using the spare 50+
  GB of VRAM, `torch.compile`) are high-value because every epoch of extra
  training translates almost linearly into improvement at this regime.
- `data/scoring.py` was patched in commit `b78f404` to filter non-finite-y
  samples instead of masking by zero — fixes the
  `0 * Inf = NaN` poisoning that previously made `test_avg/mae_surf_p`
  silently NaN whenever `test_geom_camber_cruise/000020.pt` was scored.
