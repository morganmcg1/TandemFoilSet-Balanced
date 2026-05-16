# Hypothesis: lion-lr-wd-sweep (thorfinn)

## Hypothesis

Fern's Lion-stacked baseline (PR #3387, val=94.08) replaced AdamW with `timm.optim.Lion`
but copied the **exact same hyperparameters** AdamW was using: `lr=1e-4, wd=1e-2`. The
Lion paper (Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023) makes a
specific, published recommendation:

> "When training on small datasets, the optimal learning rate for Lion is typically
> 3-10x smaller than that for AdamW, while the optimal weight decay is 3-10x larger."

This is because Lion's sign-based update rule produces a step magnitude that is
*independent of gradient magnitude* (only the sign matters). The effective step per
parameter is `lr` directly — there is no per-parameter adaptive scaling like in Adam.
For a given lr, Lion takes effectively larger and more uniform steps than AdamW, so
the optimal lr is smaller. Meanwhile, the wd term acts as a stronger regularizer
because it competes with a uniform step instead of an adapted-down step.

If the Lion paper's guidance applies here, our current Lion config is in a suboptimal
regime — lr is too high and wd is too low. The fact that we already get val=94.08
suggests Lion is robust enough to be in the ballpark, but the residual ~3-6 points of
sub-optimality may be exactly the headroom that's keeping the val curve descending at
timeout (epoch 14, slope −2.9/epoch).

**Predicted improvement:** −2 to −6 on val_avg/mae_surf_p vs Lion baseline 94.08
(target range 88–92). This is a single-flag sweep, so the failure mode is well-defined
(if all arms are worse than the current Lion config, we learn Lion is unusually robust
to its hyperparameters on this problem).

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — this already has Lion optimizer
(`timm.optim.Lion(lr=1e-4, wd=1e-2)`) and Huber loss (δ=2.0) merged from PR #3387.

### 2. Add CLI flags for lr and wd

If `--lr` and `--wd` (or equivalent flags like `--learning_rate` and `--weight_decay`)
already exist in `target/train.py`, just use them. If not, add them to the `Config`
dataclass and the argparser:

```python
lr: float = 1e-4
wd: float = 1e-2
```

And update the Lion instantiation:

```python
optimizer = timm.optim.Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
```

### 3. Add a fixed seed (prevents the variance issue we keep hitting)

Frieren's recent PRs showed up to 18-point spread between identical-config runs. Before
the model/optimizer/dataloader are instantiated, add:

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

This is a stability-of-result fix, not a hyperparameter — keep it on for all arms.

### 4. Run three arms (in priority order)

The Lion paper's recommended range is lr_Lion = lr_AdamW / 3–10 and wd_Lion = wd_AdamW
× 3–10. The current config is on the upper edge of lr and the lower edge of wd, so we
sweep downward in lr and upward in wd in matched pairs:

**Arm 1 (primary): paper's middle-of-range recommendation**
```bash
cd target/ && python train.py \
    --lr 3e-5 \
    --wd 3e-2 \
    --wandb_group lion-lr-wd-sweep \
    --wandb_name lion-lr3e5-wd3e2 \
    --agent willowpai2i24h3-thorfinn
```

**Arm 2: aggressive end of paper's range**
```bash
cd target/ && python train.py \
    --lr 1e-5 \
    --wd 1e-1 \
    --wandb_group lion-lr-wd-sweep \
    --wandb_name lion-lr1e5-wd1e1 \
    --agent willowpai2i24h3-thorfinn
```

**Arm 3 (if time allows): conservative end (closer to current baseline)**
```bash
cd target/ && python train.py \
    --lr 5e-5 \
    --wd 2e-2 \
    --wandb_group lion-lr-wd-sweep \
    --wandb_name lion-lr5e5-wd2e2 \
    --agent willowpai2i24h3-thorfinn
```

If you only have time for two arms, run Arm 1 and Arm 2 — they bracket the paper's
recommended range and will give the clearest signal about whether the guidance applies
to our problem.

### 5. Compute nansafe test metrics

Use the merged `eval_nansafe.py` on each arm's best checkpoint.

### 6. Report key signals

- val_avg/mae_surf_p per arm
- test_avg_nansafe/mae_surf_p per arm
- Per-epoch val trajectory for each arm — does lr=3e-5 still have a descending curve
  at e14, or does it plateau earlier (a sign that the optimal lr is lower)?
- Compare best epoch across arms — at lower lr, do we still reach the loss basin in
  14 epochs, or does the curve need more time?
- For the winning arm, note whether the curve was still descending at timeout (would
  indicate further headroom for a lower-lr + bf16 stack in round 5).

## Baseline

Current best (fern's Lion-stacked, PR #3387, merged 21:45 UTC):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **94.0803** |
| val_single_in_dist | 108.0536 |
| val_geom_camber_rc | 109.6926 |
| val_geom_camber_cruise | 69.3504 |
| val_re_rand | 89.2247 |
| **test_avg_nansafe/mae_surf_p** | **88.9362** |
| W&B run | `f9w6yzoq` (group: `lion-stacked`) |
| Optimizer | Lion lr=1e-4, wd=1e-2, betas=(0.9,0.99) |
| Loss | Huber δ=2.0 |
| Key note | Val still descending at epoch 14 (slope −2.9/epoch) |

Reproduce baseline: `cd target/ && python train.py --wandb_group lion-stacked --wandb_name lion-lr1e-4-wd1e-2 --agent willowpai2i24h3-fern`

## Why this is high-confidence

This is a published-recipe experiment: the Lion paper itself recommends our current
hyperparameters are suboptimal. Three reasons it's a good fit for your slot:

1. **Orthogonal to all other in-flight Lion experiments.** Fern is running bf16,
   alphonse is rebasing bf16+grad-clip+eta_min, frieren is running warmup, edward is
   running T_max=14. None of them are touching lr or wd.

2. **Single-flag change.** No architectural change, no scheduler change, no autocast
   block. Just CLI flags. If you have time after the primary arms, easy to iterate
   on follow-up variants.

3. **Failure mode is well-defined.** If all arms are worse than 94.08, we learn Lion
   is unusually robust to its hyperparameters on this problem (also informative). If
   one arm beats 94.08, we have a new baseline that stacks cleanly with everyone
   else's in-flight work — round 5 becomes Lion(optimal) + bf16 + T_max=14 + warmup.

Post terminal SENPAI-RESULT when all arms finish:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>",...],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```
