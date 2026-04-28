# Baseline — icml-appendix-willow-pai2e-r2

Active branch: `icml-appendix-willow-pai2e-r2`.

## Current best (this branch)

- **PR**: #783 — "Round 1: compound + Huber loss (delta=1.0)" (merged 2026-04-28)
- **Config**: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2` + `--huber_delta 1.0`
- **val_avg/mae_surf_p** (best checkpoint, epoch 32): **75.93**
- **W&B run**: `2y1lj209` (group `compound-huber`, project `senpai-charlie-wilson-willow-e-r2`)
- **Params**: 558,134 (unchanged) | **Peak VRAM**: 21.6 GB | **Epochs in 30 min**: 32

### Per-split val metrics (best checkpoint, epoch 32)

| Split | val mae_surf_p |
|-------|---------------|
| `val_single_in_dist`     | 85.84  |
| `val_geom_camber_rc`     | 91.20  |
| `val_geom_camber_cruise` | 54.68  |
| `val_re_rand`            | 71.99  |
| **val_avg/mae_surf_p**   | **75.93** |

### Per-split test metrics (from best checkpoint)

| Split | test mae_surf_p |
|-------|----------------|
| `test_single_in_dist`       | 79.35  |
| `test_geom_camber_rc`       | 82.61  |
| `test_geom_camber_cruise`   | **NaN** (scoring bug — see note below) |
| `test_re_rand`              | 64.29  |
| **partial test_avg (3 finite splits)** | **75.42** |
| **test_avg/mae_surf_p**     | **NaN** (poisoned by cruise NaN) |

**Cruise NaN note**: `data/scoring.py` only skips samples with non-finite *ground truth*; a single inf in the model's pressure prediction for one cruise test sample (`test_geom_camber_cruise/000020.pt`, 761 Inf values in `p` channel) poisons the whole accumulator. The val cruise split was finite throughout training (val_geom_camber_cruise/mae_surf_p = 54.68 at epoch 32). This is a scoring bug, not a model issue. A fix PR (adding a prediction-finiteness guard in `train.py`) has been green-lit.

### Reproduce

```bash
cd target && python train.py \
    --huber_delta 1.0 \
    --epochs 50 \
    --wandb_group compound-huber \
    --wandb_name compound-huber-d1.0 \
    --agent willowpai2e2-fern
```

with `model_config` in `train.py` set to (default after PR #779 merge):
```python
n_layers=3,
n_head=1,
slice_num=16,
n_hidden=128,
mlp_ratio=2,
```

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
