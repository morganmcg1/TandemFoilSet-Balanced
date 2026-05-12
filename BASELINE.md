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
- **Loss**: **L1 (MAE) in normalized target space**, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight = 10.0` _(updated 2026-05-12 by PR #1397)_
- **Stochastic depth**: per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]` (linear schedule, last layer is the output head and never dropped) _(added 2026-05-12 by PR #1552)_
- **`evaluate_split` NaN-safe pre-filter**: skip samples with non-finite `y` before `accumulate_batch` to keep the 4-split test mean finite despite the `test_geom_camber_cruise/000020.pt` data bug _(added 2026-05-12 by PR #1552)_
- **Batch size**: `4`
- **Epochs**: configured `50`, capped by `SENPAI_TIMEOUT_MINUTES = 30`
- **Sampler**: `WeightedRandomSampler` with equal-domain weights from `meta.json`

## Metrics contract

- Primary ranking metric: `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits.
- Paper-facing comparison metric: `test_avg/mae_surf_p` — same aggregation on the four test splits at the best val checkpoint.
- All metrics computed in physical (denormalized) units in `data/scoring.py`.

## Current best result

### 2026-05-12 20:52 — PR #1552 (`charliepai2g24h4-frieren/stoch-depth-0.1`)

Stochastic depth (Huang et al., ECCV 2016) added to the 5-layer Transolver
with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
At eval/test time it is a no-op (all blocks always used). Also bundles the
NaN-safe pre-filter in `evaluate_split` that produces the first finite
4-split `test_avg/mae_surf_p` on this branch.

- **`val_avg/mae_surf_p`** = **98.353** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **87.995** — first finite 4-split test
  reference on this branch; new paper-facing baseline.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) | mae_surf_p (test) |
|-------|-----------:|-----------:|
| single_in_dist     | 119.159 | 104.953 |
| geom_camber_rc     | 111.093 | 101.883 |
| geom_camber_cruise |  73.323 |  62.243 |
| re_rand            |  89.837 |  82.901 |
| **avg**            | **98.353** |  **87.995** |

vs. L1 baseline (PR #1397):
- val_avg: 100.957 → 98.353 (**-2.58% improvement**)
- val_single_in_dist: -6.45% / val_geom_camber_cruise: -5.21% (largest gains)
- val_geom_camber_rc: +0.24% (flat) / val_re_rand: +1.77% (small regression)

The OOD-specific framing was only half-supported — the biggest gain landed
on `val_single_in_dist` (in-distribution), not the camber OOD splits as
predicted. Stoch-depth's implicit ensemble flattened split-specific overfit
modes regardless of OOD axis. Best epoch landed at the wall-clock cap
(epoch 15), so more training time would likely extend the gain.

Caveat: `loss`/`surf_loss` aggregates for `test_geom_camber_cruise` still
show NaN/Inf in `metrics.yaml` because the normalized-space loss path
runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary
ranking metric is clean.

- **Metric artifacts**:
  `models/model-charliepai2g24h4-frieren-stoch-depth-0.1-20260512-201730/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.1 GB, **wall time**: 30 min (cap).

### 2026-05-12 19:05 — PR #1397 (`charliepai2g24h4-alphonse/l1-loss`)

L1 (MAE) loss replaces MSE in normalized-space training. First numeric
baseline on this branch.

- **`val_avg/mae_surf_p`** = **100.9574** (best @ epoch 13/14 before 30 min timeout)
- **`test_avg/mae_surf_p` (3-split mean, excludes `test_geom_camber_cruise`)** = **100.8314**
- **`test_avg/mae_surf_p` (all 4 splits, raw)** = NaN — pre-existing data
  bug: `test_geom_camber_cruise/000020.pt` has 761 nodes with `inf` in
  pressure y. Affects every arm in this round; `data/scoring.py` is
  marked read-only. See PR #1397 comment for full trace and proposed
  fixes. Until resolved we record the 3-split test mean.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 127.371 |
| val_geom_camber_rc     | 110.832 |
| val_geom_camber_cruise |  77.353 |
| val_re_rand            |  88.273 |
| **val_avg/mae_surf_p** | **100.957** |
| test_single_in_dist    | 116.622 |
| test_geom_camber_rc    |  97.209 |
| test_geom_camber_cruise| NaN (data bug, surf_Ux/Uy still ok) |
| test_re_rand           |  88.663 |
| **test_avg/mae_surf_p (3-split)** | **100.831** |

- **Metric artifacts**:
  `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M, **peak GPU memory**: 42.1 GB, **wall time**: 30.7 min.

## Reproduce baseline

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <student>/baseline
```
