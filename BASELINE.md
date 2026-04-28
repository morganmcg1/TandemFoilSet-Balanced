# Baseline — icml-appendix-willow-pai2e-r2

Active branch: `icml-appendix-willow-pai2e-r2`.

## Current best (this branch)

- **PR**: #840 — "Per-sample relative MAE loss" (merged 2026-04-28)
- **Config**: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2` + `--loss_type relative_mae` + `--surf_weight 10.0`
- **val_avg/mae_surf_p** (best checkpoint, epoch 32): **64.16**
- **test_avg/mae_surf_p** (best checkpoint): **55.73** (finite across all 4 splits including cruise)
- **W&B run**: `t5p9xzxx` (group `compound-relative-mae`, project `senpai-charlie-wilson-willow-e-r2`) — rebased re-run
- **Params**: 558,134 (unchanged) | **Epochs in 30 min**: ~32 (timed out at 32/50)

### Per-split val metrics (best checkpoint, epoch 32)

| Split | val mae_surf_p |
|-------|---------------|
| `val_single_in_dist`     | 77.07  |
| `val_geom_camber_rc`     | 84.10  |
| `val_geom_camber_cruise` | 36.86  |
| `val_re_rand`            | 58.58  |
| **val_avg/mae_surf_p**   | **64.16** |

### Per-split test metrics (from best checkpoint)

| Split | test mae_surf_p |
|-------|----------------|
| `test_single_in_dist`       | 71.33  |
| `test_geom_camber_rc`       | 70.62  |
| `test_geom_camber_cruise`   | 30.92  |
| `test_re_rand`              | 50.04  |
| **test_avg/mae_surf_p**     | **55.73** |

**Note**: Metrics from rebased re-run (W&B `t5p9xzxx`). Relative MAE loss fixed the cruise NaN — cruise test split is finite (30.92). The gradient-equalization across Re regimes works as predicted: cruise (low-Re, small |y|) benefits most.

### Reproduce

```bash
cd target && python train.py \
    --loss_type relative_mae \
    --surf_weight 10.0 \
    --epochs 50 \
    --wandb_group compound-relative-mae \
    --wandb_name compound-relative-mae \
    --agent willowpai2e2-edward
```

with `model_config` in `train.py` set to:
```python
n_layers=3,
n_head=1,
slice_num=16,
n_hidden=128,
mlp_ratio=2,
```

---

## 2026-04-28 18:00 — PR #840: per-sample relative MAE (merged)

- **val_avg/mae_surf_p:** 64.16 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 55.73 (finite across all 4 splits)
- **Per-split val:** single=77.07, rc=84.10, cruise=36.86, re_rand=58.58
- **Per-split test:** single=71.33, rc=70.62, cruise=30.92, re_rand=50.04
- **Delta vs previous best (PR #783):** −11.77 (−15.5%) on val_avg/mae_surf_p
- **W&B run:** t5p9xzxx (rebased re-run)
- **Status:** Merged into advisor branch

---

## 2026-04-28 16:00 — PR #840: per-sample relative MAE (new best — pending merge)

- **val_avg/mae_surf_p:** 64.73 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 56.92 (finite across all 4 splits)
- **Per-split val:** single=80.41, rc=78.51, cruise=40.13, re_rand=60.73
- **Per-split test:** single=77.25, rc=67.74, cruise=32.35, re_rand=50.35
- **Delta vs previous best (PR #783):** −11.20 (−14.7%) on val_avg/mae_surf_p
- **W&B run:** nz8eev8e
- **Status:** Winner declared; sent back for rebase (merge conflict on advisor branch)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (prev best)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (new best)

- **Surface MAE (val_avg):** 75.93 (epoch 32, timed out at 32/50 — still improving)
- **Per-split val:** single=85.84, rc=91.20, cruise=54.68, re_rand=71.99
- **Per-split test (finite):** single=79.35, rc=82.61, re_rand=64.29; cruise=NaN (scoring bug)
- **Delta vs previous best:** −20.87 (−21.6%) on val_avg/mae_surf_p
- **W&B run:** 2y1lj209
- **Reproduce:** see above — add `--huber_delta 1.0` to the compound anchor command

---

## 2026-04-28 12:00 — PR #779: Round 1 anchor

- **Surface MAE (val_avg):** 96.80
- **W&B run:** ez3f10h3
- **Reproduce:** see above

---

## Reference context (from `target/README.md` leaderboard)

A previous senpai-vs-kagent investigation against this same dataset/Transolver
baseline found that a compounded reduction of model size dominated the
leaderboard. Use these as targets, not as merged baselines on this branch:

- Reference baseline (default config, similar to our `train.py`): `test_avg/mae_surf_p ≈ 80–82`
- Reference compound winner (PR #32 in that older repo): `test_avg/mae_surf_p = 40.927`
  - Configuration: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`
  - Compound was the combination of three independent reductions (depth, slice
    count, single-head attention) on top of the default optimizer/loss.

## Default training command

```bash
cd target && python train.py --epochs 50 --wandb_name <descriptive-name>
```

Architecture parameters (`n_hidden`, `n_layers`, `n_head`, `slice_num`,
`mlp_ratio`) are not CLI flags — students must edit `model_config` in
`target/train.py` to change them. Optimizer and loss parameters (`lr`,
`weight_decay`, `batch_size`, `surf_weight`, `epochs`) are CLI flags via
`Config`.

## Per-split structure (4 val + 4 test tracks)

The primary metric `val_avg/mae_surf_p` is the **equal-weight mean of surface
pressure MAE across the four validation splits**. The same average across the
four held-out test splits is `test_avg/mae_surf_p`. Lower is better. Best
checkpoint is selected on `val_avg/mae_surf_p` and that checkpoint is used for
the end-of-run test eval. See `target/program.md` for the full split design.

| Track | Tests |
|-------|-------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity (single-foil random holdout) |
| `val_geom_camber_rc` / `test_geom_camber_rc` | RaceCar tandem, unseen front-foil camber M=6–8 |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Cruise tandem, unseen front-foil camber M=2–4 |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem domains |
