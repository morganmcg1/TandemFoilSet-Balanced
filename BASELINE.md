# Charlie pai2g 48h r1 — Baseline

Branch: `icml-appendix-charlie-pai2g-48h-r1`
Research tag: `charlie-pai2g-48h-r1`

## Status (2026-05-14)

**Three winners merged.** PR #1405 (tanjiro, bf16 + OneCycle@25ep) is now the
current baseline at `val_avg/mae_surf_p = 73.295` (-14.4% vs PR #1581).
All subsequent experiments should use `--loss l1 --lr 2e-3 --epochs 25`
with OneCycleLR scheduler + bf16 autocast enabled. VRAM footprint is ~33 GB
(down from ~42 GB in fp32) due to bf16.

## 2026-05-14 — PR #1405: bf16 autocast + OneCycleLR@25ep (tanjiro) ← CURRENT BEST

- **Primary metric:** `val_avg/mae_surf_p` = **73.295**
- **Paper-facing metric:** `test_avg/mae_surf_p` = **63.911**
- **Improvement vs PR #1581:** -14.4% val / -23.3% test
- **Best epoch:** 19 / 25 configured (30-min wall-clock cap; ~97 s/epoch with bf16, ~33 GB peak VRAM)
- **Key change:** bf16 autocast reduces per-epoch time, enabling more epochs within the 30-min cap;
  `--epochs 25` sets OneCycleLR `total_steps = 25 × len(loader)` so LR stays meaningful through epoch 19.
- **Per-split val breakdown (epoch 19):**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 54.423 |
| val_re_rand | 71.041 |
| val_single_in_dist | 79.894 |
| val_geom_camber_rc | 87.823 |
| **val_avg** | **73.295** |

- **Metric artifacts:** `models/model-amp-bf16-onecycle-25ep-20260512-233756/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 25 --lr 2e-3 --loss l1 \
  --agent charliepai2g48h1-tanjiro --experiment_name amp-bf16-onecycle-25ep
```

Note: bf16 autocast is always-on after this merge (merged `train.py`). VRAM ~33 GB.
Verify OneCycleLR completes: check `train/lr_end_of_epoch` — should be >1e-4 at epoch 14
(meaning more training budget remains under the cap).

## 2026-05-12 22:55 — PR #1581: L1 + OneCycleLR@peak=2e-3 (frieren) [previous best]

- **Primary metric:** `val_avg/mae_surf_p` = **85.615**
- **Paper-facing metric:** `test_avg/mae_surf_p_3of4_finite_splits` = **83.328**
- **Improvement vs PR #1355:** -9.20% val / -9.29% test
- **Best epoch:** 14 / 14 configured (~131 s/epoch, ~42 GB peak VRAM)
- **Per-split val breakdown:**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 66.435 |
| val_re_rand | 81.890 |
| val_single_in_dist | 99.524 |
| val_geom_camber_rc | 94.610 |
| **val_avg** | **85.615** |

- **Metric artifacts:** `models/model-l1-onecycle-peak2e3-20260512-215537/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 14 --lr 2e-3 --loss l1 \
  --agent charliepai2g48h1-frieren --experiment_name l1-onecycle-peak2e3
```

Note: `--lr 2e-3` triggers the OneCycleLR scheduler (see merged `train.py`).
Verify the OneCycleLR schedule fires by checking `train/lr_end_of_epoch`
in the metrics JSONL — should decay from ~2e-3 to ~10⁻⁹ over 14 epochs.

## 2026-05-12 20:52 — PR #1355: Smooth L1 / pure L1 vs MSE (alphonse)

- **Primary metric:** `val_avg/mae_surf_p` = **94.291**
- **Paper-facing metric:** `test_avg/mae_surf_p_3of4_finite_splits` = **91.859**
  (3 finite test splits; `test_geom_camber_cruise/mae_surf_p` is NaN due to
  pre-existing `+Inf` sample 000020.pt — `data/scoring.py` is read-only)
- **Best epoch:** 14 / 15 configured (~131 s/epoch, ~42 GB peak VRAM)
- **Per-split val breakdown:**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 71.660 |
| val_re_rand | 87.503 |
| val_single_in_dist | 110.407 |
| val_geom_camber_rc | 107.595 |
| **val_avg** | **94.291** |

- **Metric artifacts:** `models/model-pure-l1-20260512-191540/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 15 --loss l1 \
  --agent charliepai2g48h1-alphonse --experiment_name pure-l1
```

## Reference configuration (updated after PR #1581)

- **Optimizer:** `AdamW(lr=2e-3, weight_decay=1e-4)` — updated from 5e-4 (PR #1581)
- **Scheduler:** ~~CosineAnnealingLR(T_max=epochs)~~ → **OneCycleLR** (merged PR #1581), per-batch stepping, peak_lr=`lr` arg, total_steps=`epochs * len(loader)`
- **Loss:** ~~MSE~~ → **Pure L1** in normalized space, `vol_loss + 10.0 * surf_loss` (merged PR #1355)
- **Model:** Transolver
  - `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
  - `space_dim=2, fun_dim=22 (= X_DIM - 2), out_dim=3`
  - ~1.1M params
- **Training:** `batch_size=4`, **bf16 autocast** (merged PR #1405, VRAM ~33 GB)
  - `WeightedRandomSampler` for equal-weight domain sampling across the three
    training-domain groups (raceCar single, raceCar tandem, cruise tandem).
- **Wall-clock cap:** `SENPAI_TIMEOUT_MINUTES=30` per training execution.
- **Eval splits:** `val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand` (100 samples each).
- **Test splits:** matching test versions, 200 samples each. Evaluated once at
  the end of training using the best-val checkpoint.

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four
val splits, in the original target space (physical units). Lower is better.

## Paper-facing metric

`test_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four
test splits, evaluated from the best-val checkpoint at the end of training.

## Notes for reviewers

- The default `--epochs 50` is wasted under the 30-min cap: at ~2 min/epoch
  only ~10-15 epochs actually run, so cosine annealing only enters its first
  ~30% of the decay curve. All round-1 PRs explicitly tune `--epochs` to fit
  the cap; this is itself an implicit common-recipe improvement.
- Compute headroom: 96 GB VRAM is heavily underused at batch=4, fp32 — room
  for wider models, larger batches, AMP, etc.
- Local JSONL metrics only on this branch (`models/<exp>/metrics.jsonl`);
  no W&B.
