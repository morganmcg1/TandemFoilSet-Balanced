# Hypothesis: lr-tmax-fix (edward)

## Hypothesis
Fix the cosine annealing T_max from 50 (configured epochs) to ~14 (actual epoch budget
under the 30-min wall-clock cap). This is the most important untested lever in the
codebase. The current `CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)` with MAX_EPOCHS=50
but only 11-14 epochs running means the LR decays from 5e-4 to only ~4.7e-4 over the
budget (3% of the annealing range). The model effectively trains with a *constant* LR.

Askeladd's round-3 warmup-cosine experiment (val=109.99 on MSE baseline) implicitly
tested this via SequentialLR, but bundled it with warmup + grad-clip. The isolated
effect of just the T_max fix has never been measured.

**Predicted improvement:** −3 to −8 on val_avg/mae_surf_p vs 107.46 (Huber baseline).

## Instructions

### 1. Add `lr_T_max` CLI flag to Config in train.py

Find the `@dataclass class Config:` block and add:

```python
lr_T_max: int = 0  # 0 = use MAX_EPOCHS (current behavior); >0 = override
```

### 2. Update the scheduler instantiation

Find the scheduler line:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

Replace with:
```python
t_max = cfg.lr_T_max if cfg.lr_T_max > 0 else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
```

### 3. Run two arms

**Arm 1 (primary):** `--lr_T_max 14` (matches actual epoch budget at L=5)
```bash
cd target/ && python train.py \
    --lr_T_max 14 \
    --wandb_group lr-tmax-fix \
    --wandb_name tmax-14 \
    --agent willowpai2i24h3-edward
```

**Arm 2:** `--lr_T_max 12` (tighter, LR hits near-zero by epoch 12)
```bash
cd target/ && python train.py \
    --lr_T_max 12 \
    --wandb_group lr-tmax-fix \
    --wandb_name tmax-12 \
    --agent willowpai2i24h3-edward
```

If time permits, a third arm with `--lr_T_max 18` (LR not quite at zero, retains some
late-training lr for fine-tuning if epoch budget overshoots estimate).

### 4. Report key signals

- val_avg/mae_surf_p per arm
- LR curve from W&B (should show actual annealing for T_max=14 arm)
- Compare epoch-by-epoch convergence: does the T_max=14 arm reach a lower best_val
  earlier than baseline?

## Baseline

Current best (frieren's Huber loss, PR #3248, merged):

- **val_avg/mae_surf_p:** 107.4641
- **val split breakdown:** single_in_dist=127.91, geom_camber_rc=118.49, geom_camber_cruise=83.35, re_rand=100.11
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`

## Why this is high-priority

This is a **first-principles diagnostic**. Most round-3/round-4 experiments are working
around a fundamentally misconfigured LR schedule. If this fix gives even +3% improvement
in isolation, it becomes the new baseline that ALL future experiments should be stacked on.
If it doesn't help, that's also valuable — it rules out scheduler-anneal as a free lever
and refocuses attention on capacity/loss/optimizer levers.

Post terminal SENPAI-RESULT when both arms finish:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<tmax-14-id>","<tmax-12-id>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best value>},"test_metric":{"name":"test_avg/mae_surf_p_nansafe","value":<number>}}
```
