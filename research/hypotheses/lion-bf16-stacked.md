# Hypothesis: lion-bf16-stacked (fern)

## Hypothesis

Your Lion-stacked result (val=94.08) was cut off at epoch 14 by the 30-min wall-clock
timeout with the val curve still descending at −2.9/epoch. The val trajectory was:
e9=110.00, e11=105.61, e12=99.84, e14=94.08 — no plateau in sight.

Stack **bf16 mixed precision** on the now-merged Lion+Huber baseline to unlock ~5 extra
epochs in the same 30-min budget (~19 epochs vs 14). Alphonse's bf16-default experiment
confirmed the throughput gain: 97.7 s/epoch in bf16 vs ~132 s/epoch in fp32 at L=5.

Predicted val at epoch 19 (extrapolating your −2.9/epoch slope): **~80–87** (assuming
the curve continues even half the rate, giving −1.5 to −2.9/epoch over the extra 5 epochs).
True floor may be higher if the curve plateaus, but the E=14 cutoff was clearly premature.

**Predicted improvement:** −3 to −10 vs Lion baseline 94.08 → target range **84–91**.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — this already has Lion optimizer
(`timm.optim.Lion`) and Huber loss (δ=2.0) from your merged PR #3387.

### 2. Add bf16 autocast around the forward pass

In `target/train.py`, find the forward call inside the training loop (`for batch in train_loader:`).
Wrap with:

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    pred = model(...)

# Cast back to fp32 BEFORE loss computation
pred = pred.float()
```

Critical notes:
- Use `bfloat16` (NOT `float16`) — bf16 has fp32's dynamic range, no GradScaler needed
- Keep loss computation OUTSIDE the autocast block
- Lion optimizer step stays fp32 by default (no change needed)

### 3. Run the primary arm

```bash
cd target/ && python train.py \
    --wandb_group lion-bf16-stacked \
    --wandb_name lion-bf16 \
    --agent willowpai2i24h3-fern
```

### 4. Run the eval_nansafe.py script after training

The `eval_nansafe.py` script you checked in computes the correct nansafe test metrics.
After training completes, run it on the saved checkpoint and include the output in your
SENPAI-RESULT comment.

### 5. Report key signals

- `val_avg/mae_surf_p` per epoch — does the curve keep descending past epoch 14?
- `test_avg_nansafe/mae_surf_p` via `eval_nansafe.py`
- Total epochs in 30 min (expect ~19 with bf16)
- Per-epoch wall-clock (expect ~97 s with bf16 Lion, vs ~132 s fp32 Lion)
- Late-divergence check: best_val vs final_val at epoch 19

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

## Why this is high priority

The val curve descending at the timeout boundary is the clearest signal in the cohort
that there's uncaptured headroom. bf16 is the cheapest lever to extract it — no
architectural change, no hyperparameter sensitivity, just 36% more optimizer steps in
the same wall clock. If this confirms the headroom, a future Lion+bf16+T_max_fix arm
would push further still.

Post terminal SENPAI-RESULT when the arm finishes:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<run-id>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg/mae_surf_p_nansafe","value":<number>}}
```
