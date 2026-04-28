# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #504, edward, 2026-04-28)

Pure L1 loss `(pred - y_norm).abs()` replacing SmoothL1(β=1.0) on top of
bf16 + FF K=8 + `torch.compile(dynamic=True)`. Two-line change inside the
autocast block. Cosine schedule `--epochs 50` (T_max=50, run hits 30-min
cap at epoch 36).

- **`val_avg/mae_surf_p` = 57.2858** at epoch 36 (of 36 completed, wall-cap)
- **`test_avg/mae_surf_p` = 51.3504** (best val checkpoint)
- W&B run: [`yi5upb1e` / `pure-l1-on-compile-ff`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/yi5upb1e)
- Per-epoch wall: ~49 s steady state (cold compile epoch 1 ≈ 60 s)
- Peak GPU memory: 24.1 GB / 102.6 GB (~78 GB headroom — compile fuses L1
  and SmoothL1 into the same kernel pattern)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 36/50 epochs.

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 62.1098 | 0.6649 | 0.3570 |
| val_geom_camber_rc | 72.2255 | 1.2436 | 0.5880 |
| val_geom_camber_cruise | 38.0929 | 0.4085 | 0.2557 |
| val_re_rand | 56.7151 | 0.8416 | 0.4115 |
| **val_avg** | **57.2858** | 0.7792 | 0.3985 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 53.9001 | 0.6541 | 0.3390 |
| test_geom_camber_rc | 68.2461 | 1.1800 | 0.5437 |
| test_geom_camber_cruise | 33.0251 | 0.3820 | 0.2311 |
| test_re_rand | 50.2303 | 0.6705 | 0.3669 |
| **test_avg** | **51.3504** | 0.7217 | 0.3702 |

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| PR #314 (edward): + SmoothL1 β=1.0 | 69.83 | −13.6% |
| PR #407 (fern): + cosine T_max=37 alignment | 69.74 | −0.13% |
| **PR #504 (edward): SmoothL1 → pure L1** | **57.29** | **−17.96%** |

Cumulative: **−60.3% on val_avg / −60.9% on test_avg** since PR #312.

## Schedule note: pure L1 vs SmoothL1 may need different T_max

The previous baseline (Huber + T_max=37) reached its win partially from
the late-training low-LR tail (cosine reaching zero). **Pure L1 has a
constant-magnitude gradient (`sign(r)`) regardless of residual size**, so
it keeps making progress even at tiny residuals — but it stops making
progress when lr=0. The mechanism suggests pure L1 may want a *longer*
T_max (`--epochs 50` keeps lr at ~27% of peak through epoch 36) rather
than aligning T_max to the achievable budget.

This is currently being tested directly (edward PR #533 in flight) — sweep
of `--epochs 37` vs `--epochs 50` with pure L1 to settle the schedule
question. Until that lands, the canonical reproduce uses `--epochs 50`,
matching the run that produced this baseline.

## Default config (matches PR #504)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- **`--epochs 50`** (T_max=50; lr does not fully decay in the 30-min
  achievable budget — pure L1 keeps making progress at non-zero LR)
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- **Loss**: pure L1 `(pred - y_norm).abs()` per-element loss in normalized
  space, with surface vs. volume split via `surf_weight`. Inside
  `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Fourier features (K=8)** for normalized (x, z), computed in fp32
  outside the autocast scope, concatenated to the per-node feature vector.
  Per-node feature dim: 24 → 56.
- **`torch.compile(model, dynamic=True)`** wrapper applied right after
  `model.to(device)` (gated on `not cfg.debug`). Save/load via
  `getattr(model, "_orig_mod", model).state_dict()` so the W&B model
  artifact is portable into a non-compiled module.
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22 + 4*8 = 54`, `out_dim=3`).

> **Note on `--epochs`:** The Config dataclass default is still
> `epochs: int = 50`, which matches this baseline. Previously fern's PR
> #407 had recommended `--epochs 37` for Huber (T_max alignment to the
> achievable ~37-epoch budget). With pure L1 now merged, the schedule
> alignment story has changed — see the schedule note above.

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-pure-l1-compile-ff \
  --wandb_name baseline-pure-l1-compile-ff
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap binding at 36/50 epochs.
- VRAM headroom is now 78 GB (24.1 / 102.6).
- `data/scoring.py` patched (`b78f404`).
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.

## Prior baselines (superseded)

- **PR #312** (alphonse, original): val_avg=144.21, test_avg=131.18.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15.
- **PR #327** (tanjiro, FF K=8): val_avg=106.92, test_avg=96.82.
- **PR #416** (alphonse, compile+FF): val_avg=80.85, test_avg=73.41.
- **PR #314** (edward, Huber+compile+FF): val_avg=69.83, test_avg=61.72.
- **PR #407** (fern, T_max=37 on Huber): val_avg=69.74, test_avg=60.48.
