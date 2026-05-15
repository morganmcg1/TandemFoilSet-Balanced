# Hypothesis: lion-warmup (frieren)

## Hypothesis

Fern's Lion-stacked baseline (val=94.08) showed early-epoch instability: e1 val=195.75
(very high) before settling. Lion's sign-rule update can produce abrupt early
trajectories before the model finds the loss basin. A short linear LR warmup (2 epochs
from 0 → 1e-4) should smooth the early-epoch loss landscape and let Lion reach a
lower minimum within the 14-epoch budget.

This is fern's own suggested follow-up #2 from his terminal SENPAI-RESULT comment.

**Predicted improvement:** −2 to −5 on val_avg/mae_surf_p vs 94.08 baseline (target
range 89–92).

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — this already has Lion optimizer
(`timm.optim.Lion(lr=1e-4, wd=1e-2)`) and Huber loss (δ=2.0) merged.

### 2. Add linear LR warmup before cosine

In `target/train.py`, find the scheduler instantiation:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

Replace with:

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
warmup_epochs = cfg.warmup_epochs
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1e-3,  # near-zero start
    end_factor=1.0,
    total_iters=warmup_epochs * len(train_loader),
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(MAX_EPOCHS - warmup_epochs) * len(train_loader),
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs * len(train_loader)],
)
```

Also add to `Config`:
```python
warmup_epochs: int = 0  # 0 = no warmup (current behavior); >0 = linear warmup
```

And step the scheduler per-batch (not per-epoch) so the warmup steps actually fire:
```python
# Find scheduler.step() call (likely at end of epoch loop)
# Move it to inside the for-batch loop, after optimizer.step()
```

If per-batch stepping is too invasive, use epoch-level milestones in SequentialLR
(simpler but coarser).

### 3. Run primary arm

```bash
cd target/ && python train.py \
    --warmup_epochs 2 \
    --wandb_group lion-warmup \
    --wandb_name lion-warmup2 \
    --agent willowpai2i24h3-frieren
```

### 4. (Optional) Run second arm with longer warmup

```bash
cd target/ && python train.py \
    --warmup_epochs 1 \
    --wandb_group lion-warmup \
    --wandb_name lion-warmup1 \
    --agent willowpai2i24h3-frieren
```

### 5. Compute nansafe test metrics

Use the merged `eval_nansafe.py` on the best checkpoint.

### 6. Report key signals

- val_avg/mae_surf_p per arm, plus the trajectory (does e1 still spike to ~195?)
- test_avg_nansafe/mae_surf_p per arm
- Total epochs (expect ~14, same as Lion baseline)
- Best epoch (should still be the last epoch unless warmup caused early convergence)
- IMPORTANT: report seed-noise — your prior surface-only PR (#3394) showed 18-point
  spread between two identical runs. Run each arm at least twice if time permits, or
  document that this is a single-seed result.

## Baseline

Current best (fern's Lion-stacked, PR #3387, merged):

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **94.0803** |
| test_avg_nansafe/mae_surf_p | **88.9362** |
| W&B run | `f9w6yzoq` (group: `lion-stacked`) |
| Key note | Val curve descending at timeout (e1=195.75 → e14=94.08) |

Reproduce baseline: `cd "target/" && python train.py --wandb_group lion-stacked --wandb_name lion-lr1e-4-wd1e-2 --agent willowpai2i24h3-fern`

## Re: variance issue from your prior PR

Your #3394 showed 18-point spread between two identical-config runs (val=103.20 vs
121.38). Likely causes: no fixed seed in `train.py`, non-deterministic data shuffling,
or numerical noise in the Huber elbow at δ=1.0. For this PR, **at minimum add a fixed
seed** before model/optimizer/dataloader instantiation:

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

That alone should reduce inter-run spread substantially. If you can run each arm
twice with different seeds, even better — gives a noise floor estimate alongside the
mean.

Post terminal SENPAI-RESULT when done.
