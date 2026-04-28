# Round Baseline — `icml-appendix-charlie-pai2d-r2`

Lower is better. Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across the four val splits). Paper-facing metric is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-28 00:25 — PR #363: EMA of model weights (decay=0.999) for evaluation

- **Best `val_avg/mae_surf_p`** (target to beat): **101.350** (epoch 14)
- **`test_avg/mae_surf_p`** (paper-facing): pending finite re-measurement on the EMA-merged baseline (cruise NaN here because PR #361 had not landed when this run started); **3-split test mean = 100.030** — `single_in_dist=113.32, geom_camber_rc=97.44, re_rand=89.33`.
- **Per-split val MAE for `p` (EMA, epoch 14)**:
  - `val_single_in_dist`: 126.323 (−5.76% vs huber)
  - `val_geom_camber_rc`: 109.406 (−0.07%, flat)
  - `val_geom_camber_cruise`: 76.988 (−6.93% vs huber)
  - `val_re_rand`: 92.682 (−5.19% vs huber)
- **Recipe**: huber(δ=1.0) loss in normalized space + EMA copy of weights (decay 0.999), checkpoint = EMA weights. All other defaults unchanged from the merged baseline.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-eval --agent <name>
  ```

## 2026-04-27 23:30 — Previous baseline (PR #282 + #361)

- **Best `val_avg/mae_surf_p`**: 105.999 (PR #282 huber-loss)
- **`test_avg/mae_surf_p`**: 97.957 (first finite measurement, PR #361 NaN-safe eval rerun)
- **Per-split val surface MAE for `p`**:
  - `val_single_in_dist`: 134.048
  - `val_geom_camber_rc`: 109.479
  - `val_geom_camber_cruise`: 82.718
  - `val_re_rand`: 97.751
- **Per-split val Ux / Uy / p (surface)**: see `research/EXPERIMENTS_LOG.md`
- **Model**: Transolver, 0.66M params, default config (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`).
- **Optimizer**: AdamW, lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, epochs=50 (timeout-truncated at 14/50 epochs).
- **Loss**: Huber(δ=1.0) on normalized targets, applied identically in train and val/test eval.
- **Metrics path**: `models/model-charliepai2d2-edward-huber-loss-20260427-223516/{metrics.jsonl,metrics.yaml}`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-loss --agent <name>
  ```

## 2026-04-28 00:10 — PR #361 follow-up: per-split test surface MAE for `p` (first finite test_avg)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| `test_single_in_dist`     | 123.760 | 1.737 | 0.746 |
| `test_geom_camber_rc`     | 104.946 | 2.090 | 0.877 |
| `test_geom_camber_cruise` |  66.144 | 0.959 | 0.480 |
| `test_re_rand`            |  96.978 | 1.532 | 0.706 |
| **avg**                   | **97.957** | **1.579** | **0.702** |

PR #361 added a 3-line filter in `train.py:evaluate_split` that drops samples with any non-finite `y` from the batch before calling `accumulate_batch`. The `data/scoring.py:accumulate_batch` Inf-times-0 propagation bug remains (file is read-only); the workaround triggers exactly once per test pass — on `test_geom_camber_cruise` sample 20 (761 non-finite `y[p]` volume nodes; surface `p` and Ux/Uy unaffected) — and is a no-op everywhere else.

## Ranking note

Future PRs are scored against `val_avg/mae_surf_p < 105.999` (recipe high-water mark from PR #282), **not** against the 108.103 RNG draw from PR #361. The val computation path on PR #361 is byte-identical to the merged recipe (the workaround does not trigger on any val sample); the +1.99% delta is purely run-to-run variance under a 14-epoch timeout-truncated training.
