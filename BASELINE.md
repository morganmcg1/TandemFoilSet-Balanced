# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #531 fern, 2026-04-28)

Per-Re weighted sampling stacked on top of pure L1 + bf16 + FF K=8 +
`torch.compile(dynamic=True)` + cosine T_max=50. Within each domain,
weight samples by `sqrt(Re / Re_median[domain])` — high-Re samples
(which drive the pressure-magnitude tail) get ~2.5× the training weight
of low-Re. Sampler-side change in `train.py`; `data/` untouched.

- **`val_avg/mae_surf_p` = 54.0914** at epoch 37 (of 37 completed, wall-cap)
- **`test_avg/mae_surf_p` = 46.3959** (best val checkpoint)
- W&B run: [`ncn6snxe` / `re-weighted-sampling-sqrt-on-l1`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/ncn6snxe)
- Per-epoch wall: ~48 s steady state (compile cold start absorbed identically)
- Peak GPU memory: 24.1 GB / 102.6 GB (~78 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 37/50 epochs.

### Per-split surface MAE (val, best checkpoint = epoch 37)

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 56.7504 |
| val_geom_camber_rc | 68.6043 |
| val_geom_camber_cruise | 35.3607 |
| val_re_rand | 55.6503 |
| **val_avg** | **54.0914** |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 49.8142 |
| test_geom_camber_rc | 61.5704 |
| test_geom_camber_cruise | 29.1724 |
| test_re_rand | 45.0265 |
| **test_avg** | **46.3959** |

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| PR #314 (edward): + SmoothL1 β=1.0 | 69.83 | −13.6% |
| PR #407 (fern): + cosine T_max=37 (Huber-era) | 69.74 | −0.13% |
| PR #504 (edward): SmoothL1 → pure L1 | 57.29 | −17.96% |
| PR #541 (edward): T_max=50 confirmed, fresh seed | 56.22 | −1.07% (rerun) |
| **PR #531 (fern): + per-Re sqrt sampling** | **54.09** | **−3.79%** |

Cumulative: **−62.5% on val_avg / −64.6% on test_avg** since PR #312.

### Per-Re sampling mechanism

Within-domain `sqrt(Re / Re_median[domain])`:

| domain | Re range | Re_median | weight p10 | p50 | p90 |
|---|---|---|---|---|---|
| racecar_single | 100K-5M | 2.53M | 8.45e-4 | 1.75e-3 | 2.32e-3 |
| racecar_tandem | 160K-5M | 2.73M | 1.39e-3 | 2.28e-3 | 2.93e-3 |
| cruise | 121K-5M | 2.90M | 1.40e-3 | 2.38e-3 | 3.00e-3 |

Within-domain p90/p10 spread is 2.1-2.8× (Re distribution is upper-skewed
near Re_max=5M, so the median is biased high and the dynamic range is
gentler than naive expectations). Domain mass remains balanced
post-reweighting. **The mechanism is fully orthogonal to loss / compile / FF**
and stacks at ~94% efficiency.

Test gains exceed val gains (−9.65% test vs −5.58% val vs PR #504; −4.18%
test vs −3.79% val vs PR #541) — sampler reweighting helps **generalization**,
not just training-set fit. Notably, **rc-camber test improved cleanly under
L1** (−9.79% vs PR #504), reversing the near-flat behavior under Huber. Sample
emphasis on high-Re cases gives optimizer useful updates across the
distribution, not just the broad-Re splits.

## Default config (matches PR #531)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- **`--epochs 50`** (T_max=50; lr ends at ~16% of peak which pure L1 uses
  productively)
- **Per-Re sampling**: WeightedRandomSampler weights = `(1/group_size) ×
  sqrt(Re / Re_median[domain])` — built once at startup (~5 s scan over
  train files for log-Re extraction).
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

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-pure-l1-tmax50-rew \
  --wandb_name baseline-pure-l1-tmax50-rew
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap binding at 37/50 epochs.
- Single-seed variance ≈ ±1% on val_avg.
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
- **PR #504** (edward, pure L1): val_avg=57.29, test_avg=51.35.
- **PR #541** (edward, T_max=50 for L1, fresh seed): val_avg=56.22, test_avg=48.42.
