# Charlie pai2g 48h r1 — Baseline

Branch: `icml-appendix-charlie-pai2g-48h-r1`
Research tag: `charlie-pai2g-48h-r1`

## Status (2026-05-14)

**Seven winners merged.** PR #1582 (alphonse, surf_weight=5) is the
current baseline at `val_avg/mae_surf_p = 53.482` (-1.82% vs PR #2967).
Recipe: `--loss l1 --lr 2e-3 --epochs 35 --eval_every 2 --compile_model --surf_weight 5`
VRAM footprint ~24 GB. Throughput: ~50.9 s/epoch. All 35 epochs fit in 29.7 min.

## 2026-05-14 19:23 — PR #1582: surf_weight=5 on compile+35ep baseline (alphonse) ← CURRENT BEST

- **Primary metric:** `val_avg/mae_surf_p` = **53.482**
- **Paper-facing metric:** `test_avg/mae_surf_p` = **46.104**
- **Improvement vs PR #2967:** -1.82% val / -2.00% test
- **Best epoch:** 35/35 configured (all fit in 29.7 min)
- **Key change:** `--surf_weight 5` reduces the surface:volume loss scalar from 10 to 5.
  The default sw=10 was over-weighting the surface loss; sw=5 gives better surf:vol balance.
  Effect survives migration from cosine/L1/15ep → OneCycleLR/L1/bf16/compile/35ep recipe.
- **Per-split val breakdown (epoch 35):**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 37.156 |
| val_re_rand | 53.973 |
| val_single_in_dist | 56.283 |
| val_geom_camber_rc | 66.515 |
| **val_avg** | **53.482** |

- **Per-split test breakdown:**

| Split | mae_surf_p |
|-------|------------|
| test_geom_camber_cruise | 30.178 |
| test_re_rand | 46.258 |
| test_single_in_dist | 47.954 |
| test_geom_camber_rc | 60.027 |
| **test_avg** | **46.104** |

- **Metric artifacts:** `models/model-sw5-onecycle-ep35-compiled-20260514-184607/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model \
  --surf_weight 5 \
  --agent charliepai2g48h1-alphonse --experiment_name sw5-onecycle-ep35-compiled
```

Note: `--surf_weight 5 --compile_model --epochs 35` is now the **required baseline recipe**.

## 2026-05-14 18:35 — PR #2967: OneCycleLR horizon extension --epochs 35 (askeladd) [previous best]

- **Primary metric:** `val_avg/mae_surf_p` = **54.475**
- **Paper-facing metric:** `test_avg/mae_surf_p` = **47.043**
- **Improvement vs PR #2954:** -17.4% val / -17.2% test
- **Best epoch:** 35/35 configured (all fit in 29.8 min at ~51 s/epoch avg)
- **Key change:** `--epochs 35` extends the OneCycleLR horizon; LR at ep 25 is now 4.57e-4
  (productive mid-tail) vs the old baseline where LR was already at the floor (8e-9) by ep 25.
  Val still improving at ep 35 (monotone decline ep 2–35) — schedule is the binding constraint.
  VRAM and throughput identical to PR #2954 (same compile kernel fusion).
- **Per-split val breakdown (epoch 35):**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 37.613 |
| val_re_rand | 53.733 |
| val_single_in_dist | 57.573 |
| val_geom_camber_rc | 68.980 |
| **val_avg** | **54.475** |

- **Per-split test breakdown (epoch 35 checkpoint):**

| Split | mae_surf_p |
|-------|------------|
| test_geom_camber_cruise | 30.375 |
| test_re_rand | 46.455 |
| test_single_in_dist | 49.797 |
| test_geom_camber_rc | 61.544 |
| **test_avg** | **47.043** |

- **Metric artifacts:** `models/model-onecycle-ep35-compiled-20260514-171905/metrics.jsonl`
  and `models/model-onecycle-ep35-compiled-20260514-171905/metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 35 --lr 2e-3 --loss l1 --eval_every 2 --compile_model \
  --agent charliepai2g48h1-askeladd --experiment_name onecycle-ep35-compiled
```

Note: `--epochs 35` is now the **required** baseline recipe. Wall-clock 29.8 min (under 30 min cap).
All future experiments MUST use `--compile_model --epochs 35`. Without `--epochs 35`, the LR
schedule is exhausted by ep 25 and 10 productive tail epochs are wasted.

## 2026-05-14 — PR #2954: torch.compile (askeladd) [previous best]

- **Primary metric:** `val_avg/mae_surf_p` = **65.953**
- **Paper-facing metric:** `test_avg/mae_surf_p` = **56.825**
- **Improvement vs PR #2936:** -9.3% val / -10.3% test
- **Best epoch:** 25/25 configured (all epochs fit in 21.7 min at ~50 s/epoch)
- **Key change:** `torch.compile(model, dynamic=True, mode="reduce-overhead")` gives 1.86× throughput.
  Compile overhead ~14 s, paid back inside epoch 1 (63.1 s vs 91.9 s without compile).
  VRAM drops: 23.8 GB → vs 32.95 GB uncompiled (Triton fused kernels keep fewer activations live).
- **Per-split val breakdown (epoch 25):**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 49.899 |
| val_re_rand | 64.475 |
| val_single_in_dist | 70.437 |
| val_geom_camber_rc | 79.001 |
| **val_avg** | **65.953** |

- **Metric artifacts:** `models/model-torch-compile-on-20260514-161231/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 25 --lr 2e-3 --loss l1 --eval_every 2 --compile_model \
  --agent charliepai2g48h1-askeladd --experiment_name torch-compile-on
```

Note: `--compile_model` is now **required** for the baseline recipe. Without it, throughput halves
and only 19/25 epochs fit in the cap. All future experiments must include `--compile_model`.

## 2026-05-14 — PR #2936: eval_every=2 (askeladd) [previous best]

- **Primary metric:** `val_avg/mae_surf_p` = **72.694**
- **Paper-facing metric:** `test_avg/mae_surf_p` = **63.367**
- **Improvement vs PR #1405:** -0.82% val / -0.85% test
- **Best epoch:** 20 / 25 configured (30-min wall-clock cap; ~98 s/eval epoch, 20 realized)
- **Key change:** `--eval_every 2` skips validation on odd epochs, saving ~7 s/epoch × 10
  skips ≈ 70 s → 1 extra training epoch in the OneCycleLR tail.
- **Per-split val breakdown (epoch 20):**

| Split | mae_surf_p |
|-------|------------|
| val_geom_camber_cruise | 53.237 |
| val_re_rand | 71.144 |
| val_single_in_dist | 82.067 |
| val_geom_camber_rc | 84.326 |
| **val_avg** | **72.694** |

- **Metric artifacts:** `models/model-eval-every-2-20260514-143831/metrics.jsonl`
  and `metrics.yaml` on this branch.
- **Reproduce:**

```bash
cd target && python train.py --epochs 25 --lr 2e-3 --loss l1 --eval_every 2 \
  --agent charliepai2g48h1-askeladd --experiment_name eval-every-2
```

Note: `--eval_every 2` is the new default for this recipe. `should_eval` gate in
`train.py` forces a validation pass on the last configured epoch regardless.

## 2026-05-14 — PR #1405: bf16 autocast + OneCycleLR@25ep (tanjiro) [previous best]

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
