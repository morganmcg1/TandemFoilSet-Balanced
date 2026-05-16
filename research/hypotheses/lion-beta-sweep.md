# Hypothesis: lion-beta-sweep (askeladd)

## Hypothesis

The current SOTA stack uses Lion with betas=(0.9, 0.99) — the defaults from the original
paper. These were tuned on ImageNet-scale tasks; their optimality for our CFD surrogate
(19 epochs, 150 steps/epoch, clip=1.0 engaging on 99.7% of steps) has never been tested.

**Lion's update rule:** `update = sign(β₁ · exp_avg + (1-β₁) · grad)`, then
`exp_avg = β₂ · exp_avg + (1-β₂) · grad`. Two independent levers:

- **β₁ (inner loop)** controls how much momentum contributes to the update sign. At
  β₁=0.9, the momentum EMA (half-life ≈ 6.6 steps) dominates. A higher β₁ (slower
  adaptation) means the update direction is more momentum-driven; lower β₁ (faster
  adaptation) means it follows the current gradient more closely.
- **β₂ (outer loop)** controls the momentum EMA itself. At β₂=0.99, half-life ≈ 69
  steps — about half an epoch. At β₂=0.999, half-life ≈ 693 steps — roughly our entire
  run. β₂ rarely matters for Lion in practice (same sign as higher β₂), but testing it
  is cheap.

**Key interaction with clip=1.0:** Since clip normalizes every gradient to unit sphere,
the EMA `exp_avg` is averaging unit-direction vectors. β₁=0.9 gives a weighted average
of the last ~7 unit direction vectors. β₁=0.95 averages the last ~14, smoothing out more
gradient noise at the cost of slower direction changes. With our 150-step epochs, this
is a significant difference in time scale.

**Predicted improvement:** −0.5 to −2 on val_avg/mae_surf_p if β₁ is suboptimal.
Faster β₁=0.8 may help if gradient direction changes quickly (noisy landscape with clip).
Slower β₁=0.95 may help if momentum should smooth over more steps.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0+
eta_min=1e-5+**T_max=21**) is the default. Change ONLY Lion betas.

### 2. Verify CLI flags for Lion betas

In `target/train.py`, check if `--lion_beta1` and `--lion_beta2` CLI flags are exposed.
If not, they need to be added. The Lion instantiation should become:
```python
optimizer = timm.optim.Lion(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.wd,
    betas=(cfg.lion_beta1, cfg.lion_beta2)
)
```
Default values: `lion_beta1: float = 0.9`, `lion_beta2: float = 0.99`.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — β₁=0.95, β₂=0.99 (slower momentum, smoother direction):**
```bash
cd target/ && python train.py \
    --lion_beta1 0.95 \
    --lion_beta2 0.99 \
    --lr_T_max 21 \
    --wandb_group lion-beta-sweep \
    --wandb_name lion-b1-0p95 \
    --agent willowpai2i24h3-askeladd
```

**Arm 2 — β₁=0.8, β₂=0.99 (faster momentum, more grad-following):**
```bash
cd target/ && python train.py \
    --lion_beta1 0.8 \
    --lion_beta2 0.99 \
    --lr_T_max 21 \
    --wandb_group lion-beta-sweep \
    --wandb_name lion-b1-0p8 \
    --agent willowpai2i24h3-askeladd
```

If either arm diverges (val > 300 by epoch 3, or monotonically increasing after epoch 5),
stop early and note the failure. Report the early result with whatever epochs completed.

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — does either β₁ value descend faster or reach a lower floor?
- Clip engagement rate — does β₁ change the % of steps clipped? (It shouldn't, since
  clip acts before the momentum update, but confirm)
- Early-epoch trajectory (ep 1-5): does β₁=0.8 converge faster initially?
- Late-epoch behavior (ep 15-18): does β₁=0.95 show smoother descent in the low-LR zone?
- Best epoch: earlier or final?

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — tanjiro's lion-tmax21 (PR #3596, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.7375** |
| **test_avg_nansafe/mae_surf_p** | **61.7003** |
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21** |
| Lion betas | **(0.9, 0.99)** — what you are sweeping |
| Best epoch | 18 (val still descending at timeout) |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```
