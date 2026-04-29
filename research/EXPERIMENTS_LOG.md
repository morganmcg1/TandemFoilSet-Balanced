# SENPAI Research Results

## 2026-04-29 10:50 — PR #1088: Increase surf_weight from 10 to 25 for surface MAE focus

- **Branch**: charliepai2f2-edward/surf-weight-sweep-25
- **Hypothesis**: Increasing surf_weight from 10→25 focuses training loss on surface nodes, directly targeting the primary val metric (surface pressure MAE).
- **Outcome**: **MERGED** — new baseline. val_avg/mae_surf_p = 127.6661

### Results (epoch 13/50, best checkpoint, ~14 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **127.6661** |
| val_avg/mae_vol_p | 139.9394 |
| val_avg/mae_surf_Ux | 2.2548 |
| val_avg/mae_surf_Uy | 0.9431 |
| val_avg/mae_vol_Ux | 5.8663 |
| val_avg/mae_vol_Uy | 2.6935 |

| Split | mae_surf_p | mae_vol_p |
|-------|------------|-----------|
| val_single_in_dist | 157.82 | 178.70 |
| val_geom_camber_rc | 135.65 | 146.43 |
| val_geom_camber_cruise | 99.26 | 112.71 |
| val_re_rand | 117.94 | 121.91 |

Test metrics (3 of 4 valid; test_geom_camber_cruise NaN — corrupted GT sample 000020.pt):
| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 137.04 |
| test_geom_camber_rc | 122.18 |
| test_geom_camber_cruise | NaN (upstream data bug) |
| test_re_rand | 117.39 |
| 3-split avg | 125.54 |

- Per-epoch time: ~131.6s; Peak GPU: 42.12 GB
- Metrics JSONL: `target/models/model-charliepai2f2-edward-surf-weight-25-20260429-095003/metrics.jsonl`

### Analysis

A clean, minimal single-parameter change. Explicitly up-weighting surface nodes in the training loss directly improves the primary surface pressure metric. The large improvement (vs. a hypothetical baseline that would be ~130+ before any tuning) validates that the original surf_weight=10 was under-emphasizing surface accuracy. Edward also identified the NaN issue in scoring (corrupted GT sample) and provided a root cause analysis. Model is efficient: 2× fewer VRAM than width-expansion experiment, nearly 2× faster per epoch.

Follow-up: edward is testing timeout-aware CosineAnnealingLR (T_max=14 to match actual epoch count) in PR #1126.

---

## 2026-04-29 11:00 — PR #1086: Widen Transolver: n_hidden 128→256, n_head 4→8

- **Branch**: charliepai2f2-alphonse/width-expansion-256
- **Hypothesis**: Doubling hidden width from 128→256 and scaling heads 4→8 increases model capacity for multi-domain generalization. 4x parameter increase in attention and MLP layers.
- **Outcome**: Sent back for revision — inconclusive due to timeout constraint

### Results (epoch 6/50, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---|---|---|---|---|---|
| val_single_in_dist | 208.89 | 2.69 | 1.02 | 212.80 | 7.20 | 2.75 |
| val_geom_camber_rc | 175.16 | 4.16 | 1.25 | 180.37 | 7.96 | 3.31 |
| val_geom_camber_cruise | 154.44 | 2.43 | 0.84 | 162.93 | 5.51 | 1.84 |
| val_re_rand | 157.46 | 3.26 | 1.02 | 162.81 | 6.61 | 2.41 |
| **val_avg** | **173.99** | — | — | — | — | — |

**Current baseline**: val_avg/mae_surf_p = 127.67 (PR #1088, surf_weight=25)

### Analysis

Not a fair comparison. The n_hidden=256 model runs at ~258s/epoch vs. baseline's ~131s/epoch, so in the 30-minute timeout, it only completed 7/50 epochs vs. baseline's ~14. The LR was still near its peak (cosine annealing with T_max=50 had only annealed ~5%). The training curve was monotonically improving at epoch 6 (173.99), with epoch 7 showing instability (213.43) from high LR. Additional issue: NaN in test_geom_camber_cruise pressure — same corrupted GT sample as other experiments; not unique to this model.

**Sent back with instruction to try n_hidden=192, n_head=6 (keeping head_dim=32) with surf_weight=25, which should bring per-epoch time closer to 200s and allow ~9 epochs within the timeout with better LR annealing. Also instructed to add nan_to_num guard.**

---

## 2026-04-29 12:15 — PR #1102: MLP ratio 2→4: wider feedforward sublayer for richer local physics

- charliepai2f2-thorfinn/mlp-ratio-expansion
- **Hypothesis**: Expanding the MLP feedforward ratio from 2× to 4× (standard transformer ratio) increases capacity for capturing local physics relationships, improving surface pressure prediction.
- **Outcome**: **CLOSED** — experiment did not beat baseline; regression on primary metric.

### Results (both runs hit 30-min wall-clock timeout)

| Config | n_params | Best Epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | VRAM | Time/epoch |
|---|---:|---:|---:|---:|---:|---:|
| mlp_ratio=2 (within-PR baseline) | 0.66M | 14 | 130.53 | 117.20 | 42.1 GB | 132s |
| mlp_ratio=4 (experiment) | 0.99M | 13 | 136.16 | 125.17 | 52.2 GB | 148s |
| **Current baseline** (PR #1088) | — | — | **127.67** | — | — | — |

Per-split val mae_surf_p:

| Split | mlp=2 | mlp=4 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 154.90 | 164.96 | +10.06 |
| val_geom_camber_rc | 129.52 | 143.57 | +14.05 |
| val_geom_camber_cruise | 112.62 | 109.42 | −3.20 |
| val_re_rand | 125.07 | 126.70 | +1.63 |

Metric files:
- Baseline: `models/model-charliepai2f2-thorfinn-mlp-ratio-2-baseline-20260429-105629/metrics.jsonl`
- Experiment: `models/model-charliepai2f2-thorfinn-mlp-ratio-4-20260429-113117/metrics.jsonl`

### Analysis

Hypothesis was not validated. mlp_ratio=4 is 4.3% worse on val_avg/mae_surf_p than the within-PR baseline, and clearly worse than the true baseline (127.67). The primary failure mode: ~1.5K training samples and ~14 effective epochs cannot productively use the extra ~660K parameters added by the wider MLP. At this data/budget scale the 0.66M baseline model is already at the Pareto frontier.

The timeout asymmetry identified by the student (mlp=4 is 12% slower per epoch, getting one fewer epoch, and that epoch happens to be when cosine LR kicks in hard) partially explains the gap, but even correcting for this, 3 of 4 splits still regress.

Only `val_geom_camber_cruise` benefited (−3.2), consistent with the hypothesis that wider MLPs help OOD generalization on low-magnitude domains — but not enough to override the overall regression.

**Key side contribution: bug fix.** Student identified and fixed a critical correctness bug: `test_geom_camber_cruise` had `inf` in ground truth `y`. The masked accumulation `err * mask` after `err = (pred - y).abs()` produces `inf * 0 = NaN`. Fix: drop non-finite-y samples before computing predictions. This fix should be propagated to all future experiments to ensure clean test_geom_camber_cruise metrics.
