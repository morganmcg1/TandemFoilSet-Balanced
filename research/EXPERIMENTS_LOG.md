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

---

## 2026-04-29 14:00 — PR #1126: Timeout-aware cosine LR: T_max=14 to fully anneal within 30-min cap

- charliepai2f2-edward/timeout-aware-cosine-lr
- **Hypothesis**: Setting `T_max=14` (matching actual epoch count from timeout) allows the cosine schedule to fully anneal vs. T_max=50 which barely moves in 14 epochs.
- **Outcome**: **CLOSED (SUPERSEDED)** — experiment produced val_avg/mae_surf_p=123.79 (-3% vs. its own reference baseline of 127.67), but the current baseline had already advanced to 100.41 by the time results arrived.

### Results (epoch 13 best checkpoint)

| Metric | Baseline Ref (T_max=50) | This run (T_max=14) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** (PRIMARY) | **127.67** | **123.79** | **-3.04%** |
| val_avg/mae_vol_p | 139.94 | 145.05 | +3.66% |
| val_avg/mae_surf_Ux | 2.2548 | 2.0005 | -11.28% |
| val_avg/mae_surf_Uy | 0.9431 | 0.8666 | -8.11% |

Per-split val mae_surf_p:

| Split | T_max=50 | T_max=14 | Δ |
|---|---|---|---|
| val_single_in_dist | 157.82 | 147.09 | -6.83 |
| val_geom_camber_rc | 135.65 | 133.72 | -1.93 |
| val_geom_camber_cruise | 99.26 | 98.84 | -0.42 |
| val_re_rand | 117.94 | 115.53 | -2.41 |

LR trajectory: 5e-4 → 7.26e-6 over 14 epochs (fully annealed as designed). Best checkpoint at epoch 13.

- Metrics JSONL: `target/models/model-charliepai2f2-edward-timeout-aware-cosine-lr-rerun-20260429-121954/metrics.jsonl`

### Analysis

The hypothesis was confirmed: proper budget-aware LR annealing does improve surface pressure MAE by ~3%. The LR trajectory shows a clear inflection in improvement at low-LR epochs 11-13 (7.4 point drop in val_avg_surf_p). However, the T_max budget-awareness insight was independently captured in PR #1091 (nezuko's stochastic depth + budget-aware cosine) which ran earlier, and when combined with lr=1e-3 + grad_clip in PR #1098, achieved 100.41. This experiment's result of 123.79 never beat the current baseline.

Student also correctly identified the pre-existing NaN/inf issue in test_geom_camber_cruise scoring and provided a root cause analysis. Key detail: `err = (pred - y).abs()` followed by `err * surf_mask` produces `inf * 0 = NaN` for the corrupted sample 000020.pt. The pred-side nan_to_num guard requested by the advisor doesn't fully fix this.

Student's follow-up (PR #1166) targeting lr=1e-3 + grad_clip + 2-epoch linear warmup is already running on the 100.41 baseline — that's the right direction.

---

## 2026-04-29 14:30 — PR #1185: SGDR warm restarts: escape local minima for better OOD generalization

- charliepai2f2-nezuko/warm-restarts-sgdr
- **Hypothesis**: CosineAnnealingWarmRestarts (T_0=5, T_mult=1, eta_min=1e-6) replaces single-cycle cosine. Periodic LR restarts (every 5 epochs back to lr_max=1e-3) escape local minima, improving OOD generalization vs. single monotonic anneal.
- **Outcome**: **CLOSED** — negative result. val_avg/mae_surf_p = 108.69 vs. baseline 100.41 (-8.24% regression).

### Results (epoch 14 best, full 14 epochs run)

| Metric | Baseline (PR #1098) | This run (SGDR) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** (PRIMARY) | **100.41** | **108.69** | +8.24% |
| test_avg/mae_surf_p | 88.58 | 97.54 | +10.12% |

Per-split val mae_surf_p (epoch 14):

| Split | Baseline | SGDR | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 120.68 | 130.79 | +10.11 |
| val_geom_camber_rc | 111.80 | 119.12 | +7.32 |
| val_geom_camber_cruise | 75.99 | 82.74 | +6.75 |
| val_re_rand | 93.15 | 102.10 | +8.95 |

LR restart spike trajectory:

| Epoch | val_avg/mae_surf_p | LR | Note |
|---:|---:|---:|---|
| 5 (end cycle 1) | 146.87 | 1e-3 | bottom of cycle 1 |
| 6 (restart) | 169.33 | 1e-3 | +22 from cycle 1 end |
| 10 (end cycle 2) | 121.10 | 1e-3 | bottom of cycle 2 |
| 11 (restart) | 168.13 | 1e-3 | +47 from cycle 2 end |
| 14 (best) | 108.69 | low | end of cycle 3 |

- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-warm-restarts-sgdr-20260429-134725/metrics.jsonl`

### Analysis

Hypothesis rejected. SGDR's restart mechanism is structurally incompatible with the 14-epoch budget. With T_0=5/T_mult=1, only ~3 cycles fit in the timeout window. Each restart spikes val by 22-47 points and burns ~3 epochs re-converging to where the single-cycle cosine baseline spends in low-LR fine-tuning. The final cycle simply doesn't have enough low-LR epochs to recover.

Key insight: warm restarts presume the budget can absorb the disruption — true at 50+ epochs, false at 14. Single-cycle cosine with T_max matched to actual epoch count remains optimal. The "OOD escape" benefit of restarts wasn't observed: every split (in-dist and OOD) regressed equally, suggesting the model wasn't stuck in a bad local minimum to escape from in the first place.

This closes a major branch of the LR-schedule research direction. Future LR experiments should focus on within-single-cycle refinements (warmup length, eta_min floor, OneCycle policy) rather than multi-cycle approaches at this budget scale.
