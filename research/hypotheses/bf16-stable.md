# Hypothesis: bf16-stable (alphonse)

## Hypothesis

Combine three orthogonal stabilization levers on top of the merged Huber baseline:

1. **bf16 mixed precision** (your `tup20e60` throughput win — 19 epochs in 30 min vs ~14 epochs fp32 L=5)
2. **Gradient clipping** (`max_norm=1.0`) to halt the late-cosine divergence you observed (best=111.6 ep16 → final=171.4 ep19)
3. **LR floor** (`eta_min=1e-5` in `CosineAnnealingLR`) so the LR doesn't decay to zero and reduce stability further

With Huber loss (δ=2.0, already merged) capping outlier gradient magnitude at the loss level, plus grad-clip capping it at the optimizer level, plus bf16 giving the optimizer ~36% more steps to anneal, the predicted target is **−4 to −8** on `val_avg/mae_surf_p` vs 107.46.

**Predicted improvement:** −4 to −8 on val_avg/mae_surf_p vs 107.46 (so target range 99–103).

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` (which now includes frieren's merged Huber loss). Do NOT carry over your old branch's changes — start fresh on the new baseline and add only the three levers below.

### 2. Add bf16 autocast wrapper around the forward pass

In `target/train.py`, find the model forward call in the training loop (~line 470 area in current train.py — the line that calls `model(...)` inside the `for batch in train_loader:` loop). Wrap it as:

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    pred = model(...)

# Cast back to fp32 BEFORE loss computation (Huber computation needs fp32)
pred = pred.float()
```

Critical notes:
- Use `bfloat16` (NOT `float16`) — bf16 has fp32's dynamic range
- Do NOT use `GradScaler` (only needed for fp16, not bf16)
- Keep loss computation OUTSIDE the autocast block — cast `pred.float()` before the Huber/MSE block
- Optimizer step stays fp32 by default (no change needed)

### 3. Add gradient clipping

Find the optimizer-step block (after `loss.backward()`, before `optimizer.step()`). Add:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer.step()
```

### 4. Add LR floor (eta_min)

Find the scheduler instantiation:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

Replace with:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
)
```

### 5. Run the experiment

```bash
cd target/ && python train.py \
    --wandb_group bf16-stable \
    --wandb_name bf16-huber-clip-floor \
    --agent willowpai2i24h3-alphonse
```

### 6. (Optional second arm) Test with `--huber_delta 1.0`

If time permits, run a second arm with tighter delta — Huber δ=1.0 may pair well with grad-clip since both cap outlier influence:

```bash
cd target/ && python train.py \
    --huber_delta 1.0 \
    --wandb_group bf16-stable \
    --wandb_name bf16-huber-delta1-clip-floor \
    --agent willowpai2i24h3-alphonse
```

### 7. Report key signals

- val_avg/mae_surf_p (best epoch) per arm
- test_avg_nansafe/mae_surf_p (3-split mean, your manual computation since the cruise data bug poisons the in-tree metric)
- Total epochs in 30 min (should be ~19 vs fp32 L=5's ~14)
- Late-divergence check: best_val_avg vs final_val_avg — grad-clip + eta_min should prevent the gap seen in `tup20e60`
- Peak VRAM (should be ~77 GB / 96 GB, ~80% — bf16 frees memory vs your fp32 baseline)
- Per-epoch wall-clock (should be ~95–100 s)

## Baseline

Current best (frieren's Huber loss, PR #3248, merged):

- **val_avg/mae_surf_p:** 107.4641
- **val split breakdown:** single_in_dist=127.91, geom_camber_rc=118.49, geom_camber_cruise=83.35, re_rand=100.11
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`

Your prior bf16-default result (`tup20e60`):
- val_avg/mae_surf_p: 111.57 (epoch 16/19, best)
- test_avg_nansafe/mae_surf_p (3-split manual): 109.134
- 19 epochs in 30.95 min

## Why this stacks

Each lever attacks a distinct failure mode:
- **Huber** — caps outlier gradient at the loss level (high-std surface pressure samples)
- **Grad-clip** — caps outlier gradient at the optimizer level (any path the loss doesn't catch)
- **eta_min=1e-5** — prevents LR-too-low instability at end of cosine (different from edward's T_max fix)
- **bf16** — gives 5 extra epochs in budget (1.36x), so the cosine schedule actually anneals further

If edward's T_max fix lands separately and shows isolated improvement, a future `bf16-tmax` stack becomes the obvious next step.

Post terminal SENPAI-RESULT when both arms finish:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<arm1-id>","<arm2-id>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best value>},"test_metric":{"name":"test_avg/mae_surf_p_nansafe","value":<number>}}
```
