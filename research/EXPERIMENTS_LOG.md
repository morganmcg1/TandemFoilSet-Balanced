# SENPAI Research Results

## 2026-04-28 19:30 — PR #771: Learnable per-channel uncertainty weighting (Kendall & Gal 2018)

- Branch: `willowpai2e1-edward/uncertainty-weighting`
- Hypothesis: Replace fixed MSE loss with Kendall-Gal learnable uncertainty weighting. A scalar log-variance per output channel (Ux, Uy, p) is jointly learned with the model. The natural gradient signal redistributes loss capacity toward channels with the highest residual variance, which we expected to be the pressure channel given its larger dynamic range.

| W&B run | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p | Epoch | Notes |
|---------|------------|---------------------|---------------------|-------|-------|
| [1tvvwlux](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/1tvvwlux) | 10 | 123.243 | 111.227 | 14/50 | Timeout hit |
| [6gjtvi4h](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/6gjtvi4h) | 1 | 123.887 | 113.698 | 14/50 | Timeout hit |

**Note:** No unmodified baseline exists yet. These numbers cannot be interpreted as improvement or regression.

**Analysis and conclusions:**

The UW mechanism is fundamentally misaligned with the task objective. Kendall-Gal UW assigns lower effective loss weight to the channel with the *highest* per-element MSE — because high residual variance is interpreted as high aleatoric uncertainty, not as something the model should work harder on. Since the pressure channel has the largest dynamic range (and hence the highest absolute MSE), `log_var[p]` converged to -1.0 while `log_var[Ux/Uy]` converged to ~-2.25, giving pressure approximately 3.5x *less* effective loss weight than velocity. This is precisely backwards: our ranking metric is `mae_surf_p`, so we need more focus on p, not less.

The approach is a mathematical dead-end for this metric and objective. Inverse/fixed channel weighting (upweighting p explicitly) remains a valid idea worth testing separately.

**Bug found and fixed by student (credited to willowpai2e1-edward):**
`test_geom_camber_cruise/000020.pt` contains 761 non-finite pressure node values. The existing per-sample `y_finite` guard in `data/scoring.py` correctly excluded sample 20 from the accumulation mask, but `NaN * 0.0 = NaN` in IEEE-754 meant the error tensor still poisoned the running sum. The same NaN propagated through `y_norm` into the monitoring loss in `train.py`'s `evaluate_split()`. Both bugs were fixed in commit `49c55ed` on the advisor branch with `torch.nan_to_num(err, nan=0.0)` guards. All future runs on this track will have correct `test_geom_camber_cruise` metrics.

**PR closed** as a dead end.

---

## 2026-04-28 21:55 — PR #773: EMA weight averaging (Polyak) for flatter generalization ✓ MERGED

- Branch: `willowpai2e1-fern/ema-weights`
- Hypothesis: Polyak / EMA averaging of model weights (via `torch.optim.swa_utils.AveragedModel`) produces a flatter validation minimum and better OOD generalization, especially on the held-out geometry splits. A sweep of three decay rates was tested.

| Decay  | Best epoch | val_avg/mae_surf_p | EMA Δ vs live model | test_avg/mae_surf_p | W&B run |
|--------|------------|--------------------|---------------------|---------------------|---------|
| **0.99**   | **13/14**  | **119.35**         | **+6.0% over live** | **108.79**          | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) |
| 0.999  | 9/14       | 145.08             | +10.8%              | 132.17              | [t7x9cjha](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/t7x9cjha) |
| 0.9999 | 14/14      | 158.68             | −12.5% (worse!)     | 146.05              | [3otfhs7r](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/3otfhs7r) |

Per-split val improvement at decay=0.99 vs live model: `val_single_in_dist` +5.5%, `val_geom_camber_rc` +4.7%, `val_geom_camber_cruise` +9.4%, `val_re_rand` +5.2%. Gains are consistent across all 4 splits; geometry OOD splits benefit most (consistent with the "flatter basin = better OOD" story).

**Analysis and conclusions:**

EMA works as predicted. decay=0.99 is optimal for the ~14-epoch budget: fast enough to integrate useful signal (4-step half-life ≈ 100 optimizer steps per epoch ≈ ~400-step effective lookback). decay=0.999 was theoretically better (slower, flatter) but needed more epochs than the 30-min budget allowed — it bested the live model at epoch 9 but had only 4 EMA-active epochs post-warmup. decay=0.9999 was far too slow: still anchored to early-training weights at epoch 14.

Key implementation note from student: checkpoint saves `ema_model.module.state_dict()` (inner module, no `module.` prefix) so it loads cleanly into a plain Transolver for eval or further fine-tuning.

**New baseline: val_avg/mae_surf_p = 119.35, test_avg/mae_surf_p = 108.79. Merged into advisor branch.**

---

## 2026-04-28 21:56 — PR #777: Log-Re input jitter for cross-regime generalization ✗ CLOSED

- Branch: `willowpai2e1-thorfinn/re-jitter-aug`
- Hypothesis: Gaussian jitter on the normalized log(Re) input feature (dim 13) during training improves cross-regime generalization (val_re_rand target).

| Variant     | std  | val_avg/mae_surf_p | Δ vs control | W&B run |
|-------------|------|--------------------:|-------------|---------|
| no-jitter (control) | 0.00 | **124.149** | — | [ze94qebq](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ze94qebq) |
| jitter-0.10 | 0.10 | 132.247 | +6.5% worse | [atr6fwx4](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/atr6fwx4) |
| jitter-0.05 | 0.05 | 140.081 | +12.8% worse | [etpurp6h](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/etpurp6h) |
| jitter-0.20 | 0.20 | 146.337 | +17.9% worse | [ov86n58c](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ov86n58c) |

The control wins on every val split including val_re_rand (the targeted one: 114.57 vs 117.58 for best jitter). Only val_geom_camber_cruise showed marginal jitter-0.10 benefit (−1.6%).

**Analysis and conclusions:**

All runs hit the 30-min timeout at epoch 13-14. Input augmentation slows convergence — the regularization benefit only emerges once the unaugmented model starts to overfit, which never happens before the budget is exhausted. The effect is monotone in std: larger jitter = more damage. Augmentation as a regularization strategy is incompatible with our short-budget regime in its current form.

**Note:** The no-jitter control run (ze94qebq) provides an approximate unmodified baseline at epoch 14: val_avg=124.149, test_single=128.67, test_rc=125.35, test_re_rand=114.09 (test_geom_camber_cruise=NaN, unfixed run). Awaiting PR #846 for the authoritative full-budget baseline.

Follow-up idea (flagged for later): curriculum jitter starting at epoch 30+ once overfitting begins, or per-AoA jitter only. Not worth pursuing until budget is extended.

**PR closed** as a dead end.

---
