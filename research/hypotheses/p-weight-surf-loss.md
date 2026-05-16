# Hypothesis: p-weight-surf-loss (fern)

## Hypothesis

The current `surf_loss` computation mixes three surface channels with **equal weight**:
`surf_loss = huber(pred_p, target_p) + huber(pred_Ux, target_Ux) + huber(pred_Uy, target_Uy)`
(all in normalized space, where each channel has been z-scored to ~unit variance).

The **primary ranking metric is `mae_surf_p`** — only the pressure channel counts for
scoring. Ux and Uy surface accuracy is irrelevant to the leaderboard. Yet the model
allocates equal optimization budget to all three channels in `surf_loss`.

Per the current baseline (PR #3427), the test surface MAE breaks down as:
- `mae_surf_Ux` = 0.98 (already very accurate in normalized space)
- `mae_surf_Uy` = 0.46 (even more accurate)
- `mae_surf_p` = 65.88 (the metric we care about — ~66× worse than Uy)

In normalized space, the errors are more comparable (all z-scored), but if p has more
challenging structure (more outliers, multi-modal distribution, strong geometry-dependence),
its gradient contribution per sample may be smaller relative to its residual. Reweighting
`huber(p)` by a factor of 2–4 directly increases the optimization signal on the ranked channel.

This is motivated by nezuko's surf_weight sweep finding: overall surf/vol balance (sw=10)
is near-optimal, but **per-channel reweighting inside surf_loss** is an untested degree of
freedom. The lever is strictly more targeted — it tells the model "optimize p more" without
changing the vol/surf split that was found optimal.

**Predicted improvement:** −2 to −8 on val_avg/mae_surf_p vs 69.86 baseline.
The upper bound assumes p gradients are currently underweighted relative to their
residual magnitude in normalized space. The lower bound assumes equal weighting is
already near-optimal because z-scoring already equalizes the channel scales.

## Instructions

### 1. Start from the current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip+floor) is
the default. Only modify the surface loss channel weighting.

### 2. Find the surf_loss computation in train.py

Look for where `surf_loss` or `surface_loss` is computed — it should combine p, Ux, Uy
predictions. The exact structure may be:

```python
surf_loss = loss_fn(pred[:, :, 0], target[:, :, 0]) + \  # Ux
            loss_fn(pred[:, :, 1], target[:, :, 1]) + \  # Uy
            loss_fn(pred[:, :, 2], target[:, :, 2])     # p
```

Or it may use a vectorized form. Trace through `scoring.py` and `train.py` to find the
exact formulation.

### 3. Add a `p_weight` config parameter

```python
p_weight: float = 1.0  # in Config dataclass — weight for p channel in surf_loss
```

Modify the surf_loss to:
```python
# Assumes channels are [Ux, Uy, p] in order — verify against data format
surf_loss = (loss_fn(pred[:, :, 0], target[:, :, 0]) +    # Ux (weight 1.0)
             loss_fn(pred[:, :, 1], target[:, :, 1]) +    # Uy (weight 1.0)
             cfg.p_weight * loss_fn(pred[:, :, 2], target[:, :, 2]))  # p (weighted)
```

IMPORTANT: verify the channel order by checking `data/dataset.py` or `data/scoring.py`.
The order may not be [Ux, Uy, p] — confirm before applying the weight.

### 4. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 5. Run two arms

**Arm 1 (primary) — 2× weight on p:**
```bash
cd target/ && python train.py \
    --p_weight 2.0 \
    --wandb_group p-weight-surf-loss \
    --wandb_name p-weight-2x \
    --agent willowpai2i24h3-fern
```

**Arm 2 — 4× weight on p:**
```bash
cd target/ && python train.py \
    --p_weight 4.0 \
    --wandb_group p-weight-surf-loss \
    --wandb_name p-weight-4x \
    --agent willowpai2i24h3-fern
```

### 6. Report key signals

- val_avg/mae_surf_p and per-split breakdown
- **Also report mae_surf_Ux and mae_surf_Uy** — do these regress when p is upweighted?
  (expected: small increase in Ux/Uy error, large decrease in p error if hypothesis holds)
- val trajectory: does p-upweighting converge faster or slower?
- Confirm the channel index you used (which index is p in your data) in your result comment.

### 7. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 8. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — alphonse's lion-bf16-clip-floor (PR #3427, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **69.8562** |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| Surface MAE channels | Ux=0.9810, Uy=0.4619, p=65.8812 |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
