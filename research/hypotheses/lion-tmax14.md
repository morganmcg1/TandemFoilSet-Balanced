# Hypothesis: lion-tmax14 (edward)

## Hypothesis

Your `lr-tmax-fix` PR confirmed the T_max diagnostic: at T_max=14 (matching actual
14-epoch budget), val_avg=103.30 vs old Huber baseline 107.46 — a −3.9% improvement
in isolation. This validated the original hypothesis that the LR schedule was
misconfigured.

Now stack the T_max=14 fix on top of the **merged Lion+Huber baseline (val=94.08)**.
Two-mechanism compound:
- **Lion**: sign-based update rule, bounded per-parameter step magnitude
- **T_max=14 cosine**: schedule actually anneals to zero within the wall-clock budget
  (vs default T_max=50 which barely decays under 14 epochs)

Predicted: Lion is sensitive to LR (sign-rule has no momentum-like adaptive scaling),
so getting the LR schedule right may give larger absolute gains than the same fix
gave on AdamW.

**Predicted improvement:** −2 to −6 on val_avg/mae_surf_p vs Lion baseline 94.08
(target range 88–92).

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — has Lion + Huber merged. Your
`lr_T_max` CLI flag from PR #3403 already exists in train.py (your previous PR
established the pattern; it's now part of the merged base via Lion's merge of
train.py).

Wait — actually your `lr_T_max` flag is NOT yet in the merged baseline. Lion's merge
preserved the original `CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)` line.

You need to re-add your `lr_T_max` change on top of the new Lion baseline:

### 2. Add `lr_T_max` CLI flag

In `Config`:
```python
lr_T_max: int = 0  # 0 = use MAX_EPOCHS; >0 = override
```

### 3. Update scheduler instantiation

```python
t_max = cfg.lr_T_max if cfg.lr_T_max > 0 else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
```

### 4. Run two arms

**Arm 1 (primary):** `--lr_T_max 14` (matches actual epoch budget)
```bash
cd target/ && python train.py \
    --lr_T_max 14 \
    --wandb_group lion-tmax14 \
    --wandb_name lion-tmax14 \
    --agent willowpai2i24h3-edward
```

**Arm 2:** `--lr_T_max 16` (slight cushion for run-to-run epoch variance)
```bash
cd target/ && python train.py \
    --lr_T_max 16 \
    --wandb_group lion-tmax14 \
    --wandb_name lion-tmax16 \
    --agent willowpai2i24h3-edward
```

T_max=16 is included because Lion is slightly slower per-epoch than AdamW; if T_max=14
turns out to under-shoot, T_max=16 gives the schedule a 2-epoch buffer.

### 5. Compute nansafe test metrics

Use the merged `eval_nansafe.py` on each arm's best checkpoint.

### 6. Report key signals

- val_avg/mae_surf_p per arm
- test_avg_nansafe/mae_surf_p per arm
- LR trajectory from W&B (T_max=14 arm should show full annealing to 0)
- Best epoch per arm — does T_max=14 reach a lower val earlier than Lion baseline?
- Compare to your prior T_max=14 result on AdamW (val=103.30 from `2j268eqn`):
  - If Lion+T_max=14 beats both Lion alone (94.08) AND AdamW+T_max=14 (103.30) by a
    multiplicative amount, it suggests compounding
  - If it only marginally beats Lion (94.08), the LR-schedule lever is weaker once Lion
    is in place (Lion may already be implicitly adapting via sign-rule)

## Baseline

Current best (fern's Lion-stacked, PR #3387, merged):

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **94.0803** |
| test_avg_nansafe/mae_surf_p | **88.9362** |
| W&B run | `f9w6yzoq` (group: `lion-stacked`) |

Reproduce baseline: `cd "target/" && python train.py --wandb_group lion-stacked --wandb_name lion-lr1e-4-wd1e-2 --agent willowpai2i24h3-fern`

Your prior AdamW+T_max=14 result (`2j268eqn`, PR #3403, now closed):
- val_avg/mae_surf_p: 103.3022
- test_avg_nansafe/mae_surf_p: 98.6362
- −3.9% vs old Huber baseline, confirming T_max=14 helps

## Why this matters

Lion's val curve was descending at timeout (slope −2.9/epoch at e14). The natural
question: would a properly-annealed schedule (one that actually decays the LR within
the budget) give Lion's curve a final-epoch refinement that pushes deeper?

If yes (T_max=14 + Lion beats Lion alone): the LR schedule + optimizer choice
together are the binding constraint, not the optimizer alone.

If no (no improvement): Lion's sign-rule is robust to the LR schedule misconfig,
and the gain in your prior result was unique to AdamW's sensitivity to T_max.

Either result is informative for the next round of stacking experiments.

Post terminal SENPAI-RESULT when both arms finish.
