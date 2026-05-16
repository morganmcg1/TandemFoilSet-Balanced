# Hypothesis: ema-weights (edward)

## Hypothesis

The current SOTA (Lion+bf16+clip+floor, PR #3427, val=69.86) has a key property:
**val is still descending at the final epoch 19**. Best epoch = final epoch — no
plateau. This is a classic signal that the model could benefit from **weight averaging**:
the trajectory near the end of training is still moving in a useful direction, but each
individual step has noise. Averaging recent weights extracts the signal from the noise
and typically gives a free 0.5–3% improvement at zero compute cost (just one extra
parameter copy in memory).

**Exponential Moving Average (EMA)** maintains a running average of model weights:
```
ema.data = decay * ema.data + (1 - decay) * model.data
```
At evaluation time, `ema_model` is used instead of `model`. The EMA tracks a smoothed
version of the optimization trajectory — bounded ahead of the noise floor by
~(1-decay) in expectation.

This is **orthogonal to every lever currently in the merged stack** (Lion, bf16, clip,
eta_min, T_max). It costs no training-time compute, only a single extra `param.data.copy_`
per step. It's especially powerful when val is still descending: EMA "looks ahead" of the
current step by averaging recent trajectory.

Your prior T_max=14 work confirmed that LR scheduling matters at our compute budget
(val 93.44 → 83.45 on old Lion baseline). EMA is a complementary lever — instead of
tuning *where* the optimizer stops, EMA tunes *what weights you return* from the
trajectory.

**Predicted improvement:** −0.5 to −3.0 on val_avg/mae_surf_p vs 69.86 baseline.
Conservative: −0.5 to −1.5 from noise reduction alone. Aggressive: −2 to −3 if the
descending-val signal means EMA captures genuinely better-positioned weights from the
end of the cosine arc.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0
+eta_min=1e-5) is the default. ONLY add EMA.

### 2. Add EMA to train.py

Add to Config:
```python
ema_decay: float = 0.0  # 0.0 = disabled; typical values: 0.999, 0.9999
```

In the training setup, after the model is created:
```python
import copy
if cfg.ema_decay > 0:
    ema_model = copy.deepcopy(model)
    ema_model.eval()  # always in eval mode
    for p in ema_model.parameters():
        p.requires_grad_(False)
else:
    ema_model = None
```

In the train loop, **after `optimizer.step()`**:
```python
if ema_model is not None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(cfg.ema_decay).add_(p.data, alpha=1 - cfg.ema_decay)
```

At evaluation time, swap `model` for `ema_model`:
```python
eval_model = ema_model if ema_model is not None else model
# use eval_model in your validate() / test() calls
```

### 3. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — decay=0.999 (recent-heavy, ~1000-step horizon):**
```bash
cd target/ && python train.py \
    --ema_decay 0.999 \
    --wandb_group ema-weights \
    --wandb_name ema-d999 \
    --agent willowpai2i24h3-edward
```

**Arm 2 — decay=0.9999 (slower, ~10000-step horizon, longer history):**
```bash
cd target/ && python train.py \
    --ema_decay 0.9999 \
    --wandb_group ema-weights \
    --wandb_name ema-d9999 \
    --agent willowpai2i24h3-edward
```

With ~150 steps/epoch × 19 epochs ≈ 2850 total steps:
- decay=0.999 → effective horizon ~1000 steps (~7 epochs of history)
- decay=0.9999 → effective horizon ~10000 steps (longer than full training — early-epoch
  bias remains, may need warmup of EMA itself for fairer comparison)

### 5. Report key signals

- **Both `val_avg/mae_surf_p` (model) AND `val_avg_ema/mae_surf_p` (ema_model)** per epoch
  — log both to W&B so we can see the divergence
- Best epoch on each — does EMA's best lag or lead the model's best?
- Val trajectory crossover: at which epoch does ema_model overtake model?
- Per-split values for the best EMA checkpoint
- VRAM usage (should be ~+8 GB for the extra model copy — well within headroom)

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on the EMA checkpoint at the best EMA-val epoch.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_ema_val>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<best_ema_test>}}
```

Use the EMA model's val as the primary metric (since that's the deployed model). Include
the model val in your prose for comparison.

## Baseline

Current best — alphonse's lion-bf16-clip-floor (PR #3427, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **69.8562** |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |
| Stack | Lion + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 |
| Best epoch | 19 (final — val still descending) |
| VRAM | 33 GB / 96 GB |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
