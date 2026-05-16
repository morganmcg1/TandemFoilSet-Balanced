# Hypothesis: lion-tmax-newbase (tanjiro)

## Hypothesis

The merged baseline (Lion+bf16+clip+floor, PR #3427) runs ~19 epochs in 30 min with
bf16 enabled. The CosineAnnealingLR in the code uses `T_max=50` (the MAX_EPOCHS global),
but only 19 of those 50 epochs are completed before the wall-clock timeout hits.

Alphonse's analysis of the winning run confirmed: at epoch 19, the LR was still
`7.16e-5` — the cosine schedule has barely decayed from the initial `1e-4` (only
~28% into the cosine curve). The `eta_min=1e-5` floor is a dead letter at this budget.

Setting `T_max=19` (matching the actual bf16 epoch budget) would:
1. **Fully anneal the LR to 0 within the training budget** → LR reaches eta_min=1e-5 at
   epoch 19, not at epoch 50
2. **Engage the eta_min floor** → the LR settles at 1e-5 for the final epochs, giving
   fine-grained convergence at the end of the run
3. **Correct the cosine schedule** → the whole cosine arc fits in the actual training time,
   rather than spending 30 min on only the first 38% of the curve

This is the same T_max diagnosis that edward confirmed on the old bare Lion baseline
(−3.9% from T_max=14 on AdamW+Huber, and ~val=93.44 on Lion). On the new stacked
baseline, the interaction with `eta_min=1e-5` makes this much more interesting.

**Predicted improvement:** −2 to −6 on val_avg/mae_surf_p vs 69.86 baseline.
If the eta_min floor genuinely helps final-epoch refinement: expect val in 63–68 range.

## Instructions

### 1. Start from the current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0
+eta_min=1e-5) is the default. Change ONLY T_max.

### 2. Verify or add the `lr_T_max` CLI flag

In `target/train.py`, check if a `lr_T_max` or `t_max` CLI override exists. If not, add:

```python
lr_T_max: int = 0  # in Config — 0 = use MAX_EPOCHS; >0 = override
```

And update the scheduler:

```python
t_max = cfg.lr_T_max if cfg.lr_T_max > 0 else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=t_max, eta_min=cfg.eta_min
)
```

If the flag already exists from edward's PR (which may have been squash-merged into the
advisor branch or may not), just use it.

### 3. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — T_max matches bf16 budget exactly:**
```bash
cd target/ && python train.py \
    --lr_T_max 19 \
    --wandb_group lion-tmax-newbase \
    --wandb_name lion-tmax19 \
    --agent willowpai2i24h3-tanjiro
```

**Arm 2 — T_max with 2-epoch cushion (for run-to-run variance):**
```bash
cd target/ && python train.py \
    --lr_T_max 21 \
    --wandb_group lion-tmax-newbase \
    --wandb_name lion-tmax21 \
    --agent willowpai2i24h3-tanjiro
```

T_max=21 is included because the bf16 epoch count can vary by ±1 between runs. A
2-epoch cushion ensures the schedule reaches eta_min even if a run runs slightly longer.

### 5. Report key signals

- val_avg/mae_surf_p per epoch — does the annealed schedule converge faster?
- **LR at final epoch** (should be ~eta_min=1e-5 for T_max=19, not 7e-5 as in baseline)
- Total epochs in 30 min (should still be ~19 with bf16)
- Best epoch: is it early (converged before timeout) or final (still descending)?
- val trajectory: does T_max=19 give a steeper early descent or a later, lower floor?

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — alphonse's lion-bf16-clip-floor (PR #3427, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **69.8562** |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 |
| T_max | **50 (misconfigured — only 38% of cosine arc used in 19 epochs)** |
| LR at ep19 | 7.16e-5 (barely decayed from 1e-4; eta_min=1e-5 floor not engaged) |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
