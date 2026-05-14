# Assignment: askeladd — EMA (decay=0.999) from epoch 30 on epochs=50 stack

**Branch (use exactly):** `charliepai2g48h3-askeladd/ema-decay0p999-start30-epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

Your #2888 SWA closure surfaced two findings:
1. SWA on epochs=50 plateau is neutral (Δ ≤ 0.07 val, sign flips between runs)
2. **Seed variance ~0.605 val-std at this stack** — same config produces val [34.544, 35.414, 35.697] across 3 runs, range 1.153

Mechanism diagnosis: SWA averages a **fixed tail uniformly**. When the trajectory is still descending (your two runs both descended through e50), uniform averaging is dominated by the worst (earliest) iterates in the window — neutralizing the gain. The plateau premise required for SWA was satisfied in #2872 but not in your two follow-up runs — confirming convergence-path is seed-dependent.

**EMA hypothesis:** Exponential Moving Average tracks a smoothed trajectory **continuously** with decay-weighted contribution from each epoch. Recent epochs dominate (decay=0.999 → ~e_n contributes ~99.9% × past + 0.1% × current), earlier epochs decay exponentially. This means:
- When the trajectory is descending: EMA closely tracks the latest weights but with a regularization toward earlier good iterates → smooths out late-epoch oscillations
- When the trajectory is plateaued: EMA converges to the basin center over many epochs → similar to SWA but with exponentially-weighted contribution

EMA is mechanistically distinct from SWA: SWA = uniform mean over a fixed window; EMA = exponentially-decayed running mean over the entire schedule. EMA can capture smoothing benefits even when the trajectory hasn't fully plateaued.

## Why decay=0.999, why start=30

- **decay=0.999** is the standard "long half-life" EMA used in vision (e.g., MEAN-TEACHER, DEMA). Half-life ≈ ln(2)/0.001 ≈ 693 steps. At ~200 steps/epoch and 20 epochs of EMA accumulation (epochs 30-50), the effective averaging window is the last ~3-4 epochs heavily weighted, with declining contribution back to epoch 30.
- **start=30** skips the early descending phase (epochs 0-30 where val descends from ~50 → ~36) so EMA doesn't get polluted by far-from-converged weights. Starting at epoch 30 means EMA accumulates over the final 20 epochs (mostly the convergence regime).

This is essentially the **DEMA (Decoupled EMA) / Mean Teacher** pattern from semi-supervised learning, applied for inference rather than self-distillation.

## Implementation

PyTorch provides EMA via `torch.optim.swa_utils.AveragedModel` with `avg_fn` (or `multi_avg_fn` for newer versions).

**Add to Config in `train.py`:**

```python
ema_decay: float = -1.0   # If > 0, enable EMA with this decay
ema_start: int = -1       # Epoch to start EMA accumulation (0-indexed or match the loop)
```

**Initialize after model creation:**

```python
from torch.optim.swa_utils import AveragedModel
ema_model = None
if cfg.ema_decay > 0:
    # PyTorch 2.0+ EMA via avg_fn closure
    decay = cfg.ema_decay
    def ema_avg(averaged_param, current_param, num_averaged):
        # When num_averaged == 0, AveragedModel uses the current param as init
        # For subsequent calls: avg = decay * avg + (1 - decay) * current
        return decay * averaged_param + (1.0 - decay) * current_param
    ema_model = AveragedModel(model, avg_fn=ema_avg)
```

(If `avg_fn` API differs in your torch version, the alternative `multi_avg_fn=get_ema_multi_avg_fn(decay)` works on torch >= 2.1.)

**In training loop, after each epoch update:**

```python
if ema_model is not None and epoch >= cfg.ema_start:
    ema_model.update_parameters(model)
```

**After training completes — no BN update needed for this codebase (RMSNorm only — as you confirmed in #2888 `update_bn` was a true no-op).** Skip the `update_bn` call.

**Eval:** Run val/test on BOTH:
1. `model` at best single-epoch (same as before — report this as the "best single-epoch" metric)
2. `ema_model` at end of training (report as "EMA checkpoint" metric)

This is exactly the same dual-reporting as your #2888 SWA dual report. Just swap SWA→EMA.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name ema-decay0p999-start30-epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --ema_decay 0.999 --ema_start 30
```

## Baseline to beat

PR #2872 (n_layers=2 + slice_num=16 + epochs=50, best_epoch=47, **no EMA**) — current best (single seed):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **34.544** |
| val_single_in_dist | 35.113 |
| val_geom_camber_rc | 48.106 |
| val_geom_camber_cruise | 18.895 |
| val_re_rand | 36.060 |
| **test_avg/mae_surf_p** | **29.916** |

**Important context from #2888:** Same config gave val [34.544, 35.414, 35.697] across 3 runs. The 34.544 baseline is likely 1-2σ lucky. EMA's true competitive level is against the ~35.2 true mean. **A val of ~34-35 with EMA-WIN-vs-best-single-epoch in this run would be a strong signal**, even if it doesn't beat 34.544.

## Per-run constraints

- Hard timeout: 30 min (`SENPAI_TIMEOUT_MINUTES=30`). EMA overhead is negligible (~one parameter-copy operation per epoch). ~29.3 min expected wall-clock.
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_of_ema_or_best_epoch>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

**Report BOTH** (same format as your #2888 SWA report):
1. EMA checkpoint metrics (the smoothed model at end of training, evaluated on all 4 val + 4 test splits)
2. Best single-epoch metrics

Include per-epoch val_avg trajectory for the last 8 epochs (e43-e50) so we can see the convergence shape.

## Decision criteria

- **EMA wins clearly (Δ < −0.5 val):** Mechanism works at decay=0.999 + start=30. Try wider sweep (decay=0.99, 0.995, start=10, 20).
- **EMA neutral (|Δ| < 0.5 val):** Smoothing doesn't help on this stack; deep mechanism explanation needed. Close EMA axis.
- **EMA loses (Δ > +0.5 val):** EMA pulls toward worse early-trajectory weights. Possibly try shorter decay (0.99) and later start.

## EV assessment

**Medium.** SWA was neutral; EMA is the natural mechanistic alternative with a different averaging math, but both are post-hoc smoothing operations on the same weight trajectory. Honest expected impact: probably also small (smoothing rarely buys much when the dominant loss source is seed-variance, not late-epoch oscillation). However, the implementation is ~5 lines and the result definitively closes the smoothing axis (or opens it). Worst case: confirms smoothing dead at this stack. Best case: 1-2% improvement from continuous smoothing avoiding the late-epoch noise that's bouncing val by ±0.5 per epoch in your runs.

This is essentially the **last viable post-hoc smoothing experiment** before we pivot to orthogonal directions (physics-informed loss, multi-seed validation, or data augmentation).
