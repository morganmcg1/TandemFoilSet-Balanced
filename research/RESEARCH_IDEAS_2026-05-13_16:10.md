# Research Ideas — 2026-05-13 16:10

## Hypothesis: Linear LR Warmup (3 epochs, epoch-level LambdaLR)

### One-line summary

Replace the current flat-start CosineAnnealingLR with a 3-epoch linear warmup that ramps lr from 1% to 100% of peak (5e-4), then holds constant — letting Adam moment estimates stabilize before full-sized steps are taken.

### Mechanistic grounding

The current stack has two properties that make early-training instability a plausible bottleneck:

1. **100% gradient clip rate**: grad norm is consistently 18-19, so `max_norm=1.0` fires on every step. This means grad clipping is not protecting against spikes — it is acting as a global step-size normalizer. But the *direction* of step 1 is set before the Adam second-moment estimate (beta2=0.95) has seen any data. At step 0, the second-moment EMA is essentially zero (or the PyTorch default initial value of 0), so the Adam effective step size is large and noisy.

2. **beta2=0.95 means a ~20-step effective window**: the second moment stabilizes fast once training has seen ~20 steps, but the very first steps (especially step 0, where the domain mixture is whatever the sampler draws first) are taken with the noisiest gradient directions. Cross-domain heterogeneity (raceCar meshes at ~85K nodes vs. cruise meshes at ~210K nodes) amplifies this noise.

A 3-epoch linear warmup addresses this: during epochs 0-2, lr is scaled to 33%, 67%, 100% of peak. The model starts from random init with small steps, Adam moment estimates accumulate over ~3×(batch steps per epoch) real gradient observations, and only then does the optimizer commit to full-sized steps. This is the standard recipe for Transformer-class models (ViT, BERT, GPT-2, Transolver's own paper uses warmup).

### Why this is not ruled out by prior experiments

PR #1509 (CLOSED, +13.4% val regression) tested "warmup + lr=1e-3" — but that was a **confounded experiment**: it combined (a) elevated lr from 5e-4 to 1e-3, (b) warmup, (c) pre-grad-clip baseline (max_norm was not yet in the stack). The current 8-stack baseline is materially different. Standalone warmup at lr=5e-4 on the current stack is **untested**.

PR #2379 (nezuko, WIP): tests CosineAnnealingLR + eta_min=1e-5 — this tests the *tail* of the schedule, not the *head*. These are orthogonal mechanisms.

No in-flight PR tests standalone warmup. This is a clean axis.

### Proposed code change

In `train.py`, replace the scheduler definition (currently line ~459):

```python
# BEFORE (current baseline):
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# AFTER (this experiment):
_WARMUP_EPOCHS = 3

def _warmup_lr(epoch: int) -> float:
    if epoch < _WARMUP_EPOCHS:
        return max(0.01, (epoch + 1) / _WARMUP_EPOCHS)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lr)
```

No other changes. `scheduler.step()` is already called once per epoch at line 538 — this is the correct granularity for epoch-indexed `LambdaLR`.

LR trajectory (35 epochs, lr=5e-4):
- Epoch 0: 5e-4 × 0.33 = 1.67e-4
- Epoch 1: 5e-4 × 0.67 = 3.33e-4
- Epoch 2: 5e-4 × 1.00 = 5e-4
- Epochs 3-34: 5e-4 (constant at peak)

### What we expect to see

If the mechanism is alive: early-epoch train loss should converge faster or more smoothly vs. baseline (less variance in first 5 epochs). Val metric at epoch 10-15 should be noticeably lower than baseline at the same epoch. Best val checkpoint should improve by 0.5–2.5% relative on `val_avg/mae_surf_p`.

If the mechanism is dead: early-epoch train loss is indistinguishable from baseline. Final val metric is within noise (±0.3%). This would suggest early-step instability is not a limiting factor in this regime, or that the Adam second-moment initialization is not meaningfully different from a warmed-up state at this scale.

### Falsifying result

If warmup-only (no cosine tail) degrades final val vs. baseline by more than 1%, the failure is interpretable: removing the cosine decay removed more than warmup added. This would be strong evidence that the schedule *tail* (cooling) matters more than the *head* (warmup), pointing toward nezuko's cosine+eta_min direction as the right lever.

### Predicted delta

−0.5% to −2.5% on `val_avg/mae_surf_p` (improvement, lower is better). Conservative lower bound because the current constant-LR schedule has been stable; upper bound informed by warmup improvements seen in similar Transformer-on-mesh settings (e.g., GINO, FactFormer).

### Taste rubric

| Criterion | Score | Reason |
|---|---|---|
| Mechanistic grounding | 3 | Targets specific observed failure (100% clip rate + beta2=0.95 early-step noise); external analogue strong (standard Transformer recipe); confound in #1509 fully explained. |
| Research-state value | 3 | Result separates two explanations: early-step instability vs. schedule-tail cooling. Failure is as interpretable as success. |
| Execution value | 4 | 5-line change, no new packages, no architectural risk, runs within 30-min × 2-seed budget. Directly targets paper-facing metric with clear falsification path. |

Research mode: **frontier refinement** (next untested axis after the current 8-stack baseline).

### Reproduce command for edward

```bash
python train.py \
  --wandb_group lr_warmup \
  --run_name lr_warmup_3ep_seed42 \
  --seed 42

# Second seed:
python train.py \
  --wandb_group lr_warmup \
  --run_name lr_warmup_3ep_seed123 \
  --seed 123
```

### Stop condition

If warmup degrades val_avg/mae_surf_p by more than 2% vs. baseline (58.883), close this direction. If it improves, consider extending to warmup+cosine-tail (SequentialLR) as a follow-up — but only after confirming the warmup head alone is beneficial.

### Current baseline for comparison

- val_avg/mae_surf_p: **58.883**
- test_avg/mae_surf_p: **51.078**
- Stack: Transolver n_hidden=128/n_layers=5/n_head=4/slice_num=64, Huber β=0.5 surf+vol loss, surf_weight=10, AdamW lr=5e-4 betas=(0.9,0.95) wd=2e-4, max_norm=1.0, bf16 AMP, torch.compile dynamic=True, CosineAnnealingLR T_max=MAX_EPOCHS

---

## Secondary Ideas (lower priority, for future assignment)

These are noted but should NOT be assigned to edward this round — they overlap with or depend on results from in-flight WIP PRs.

1. **Warmup + cosine tail (SequentialLR)**: test after warmup-only confirms the head matters. Overlaps with nezuko's cosine+eta_min (#2379).

2. **OneCycleLR**: combines warmup + cosine anneal in one schedule. Lower priority; depends on whether standalone warmup wins.

3. **Per-layer LR groups**: surface nodes vs. volume nodes get different lr. More complex; best deferred until simpler LR levers are exhausted.

4. **SAM / Lookahead optimizer**: sharpness-aware or smoothing optimizer. Plausible but ~2× cost per step; schedule first, then optimizer variants.
