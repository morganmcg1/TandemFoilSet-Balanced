# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #327, tanjiro, 2026-04-28)

K=8 sinusoidal Fourier features for normalized `(x, z)` concatenated
to the per-node feature vector. Stacks on top of the bf16 baseline.

- **`val_avg/mae_surf_p` = 106.9223** at epoch 19 (of 19 completed)
- **`test_avg/mae_surf_p` = 96.8186** (best val checkpoint)
- W&B run: [`nbyicdne` / `ff-K8-bf16`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/nbyicdne)
- Per-epoch wall: 97–100 s (essentially unchanged from bf16; FF is cheap)
- Peak GPU memory: 33.3 GB / 96 GB
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 19/50 epochs.

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 117.2176 | 1.4740 | 0.7534 |
| val_geom_camber_rc | 125.9391 | 2.3627 | 0.9713 |
| val_geom_camber_cruise | 80.2640 | 1.2281 | 0.6079 |
| val_re_rand | 104.2685 | 1.7971 | 0.8143 |
| **val_avg** | **106.9223** | 1.7155 | 0.7867 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 103.7333 | 1.4222 | 0.7224 |
| test_geom_camber_rc | 116.3529 | 2.2950 | 0.9094 |
| test_geom_camber_cruise | 70.3920 | 1.2256 | 0.5590 |
| test_re_rand | 96.7963 | 1.6182 | 0.7705 |
| **test_avg** | **96.8186** | 1.6403 | 0.7403 |

## Default config (matches PR #327)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- Loss: per-channel-equal MSE in normalized space, with surface vs. volume
  split via `surf_weight`. **Forward + loss inside `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.**
- **Fourier features (K=8) for normalized (x, z), computed in fp32 outside
  the autocast scope** and concatenated to the per-node feature vector
  before the encoder MLP. Per-node feature dim grows 24 → 56.
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, **`fun_dim=22 + 4*8 = 54`**, `out_dim=3`).

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-bf16-ff8 \
  --wandb_name baseline-bf16-ff8
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- Test-time number reported alongside: `test_avg/mae_surf_p`.
- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) **still binding** at
  19/50 epochs. Cosine T_max alignment may release additional headroom from
  the schedule tail (currently being tested in PR #407).
- `data/scoring.py` patched in commit `b78f404` to filter non-finite-y samples.
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.
- Per-split asymmetry: cruise + single-in-dist see ~17–20% wins from FF;
  rc-camber held-out sees only −3.3%. OOD geometry is bottlenecked more by
  camber→pressure mapping than spatial-frequency representation — a hint for
  future targeted experiments.

## Prior baselines (superseded)

- **PR #312** (alphonse, original baseline): val_avg=144.21, test_avg=131.18.
  Default Transolver, no bf16, no FF.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15. Same model
  + bf16 autocast on forward+loss. Superseded by PR #327 on 2026-04-28.
