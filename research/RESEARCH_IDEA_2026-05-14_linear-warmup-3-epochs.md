# Round 129 — Linear LR warmup (3 epochs) + 57-epoch cosine

## Hypothesis

Add a **3-epoch linear LR warmup** from 0 → peak (1.5e-4), followed by **57-epoch cosine decay** to ~0. Total epochs=60 (same as baseline). Tests whether smoother early optimization trajectory at this small dataset scale (~3000 samples) can improve final convergence — early training under high LR is famously noisy and warmup is the canonical fix.

## Why this might WIN

1. **Lion + cosine without warmup is empirically aggressive at small data scale.** Lion (Chen et al. 2023) uses sign-based momentum updates that are inherently noisier than AdamW for small datasets, and the cosine schedule starts at peak LR from step 0. Adding 3 epochs of linear warmup smooths the first 1125 steps (~5% of training) — the regime most prone to early-training pathology in batch_size=4 + small-dataset settings.

2. **Warmup is the canonical recipe for transformer training.** Vaswani et al. 2017 introduced inverse-square-root warmup; Devlin et al. 2018 BERT used linear warmup; all modern transformer recipes (GPT, T5, ViT, DeiT, Swin) use some warmup period. Our architecture is a Transolver = transformer variant; we've been training it WITHOUT warmup since boot.

3. **A small targeted change is high-leverage.** This is exactly the kind of mature-recipe gap that's been overlooked in 102+ taxa of architecture exploration. The 8 already-merged optimizer-axis winners (Lion, etc.) haven't tested warmup.

4. **Could specifically improve in_dist precision.** Warmup typically produces flatter loss-landscape exploration in early training, which often translates to better fine-detail fitting in late training. The in_dist split is precision-critical; warmup could help here without sacrificing OOD generalization.

5. **Trivial implementation, zero new params.** Just a wrapper around the existing CosineAnnealingLR scheduler.

## Why this might LOSS

1. **3 epochs is ~5% of training — may be too short.** Standard transformer recipes use 5-10% warmup. 3/60 = 5% is at the low end. Mitigation: 3 is conservative; the falsifiable predictions cover trying longer warmup next.

2. **Lion + warmup may interact poorly.** Lion's sign-based update is already direction-stable; warmup primarily helps Adam-family optimizers that have noisy second moments early in training. Lion may not benefit. Counter: even Adam-family warmup is empirical, not from theory — and sign-based optimizers still benefit from LR-magnitude warmup, just less.

3. **Cosine tail truncation:** 3 epochs of warmup leaves 57 for cosine decay (vs baseline 60). Baseline #2879 timed-out at ep58/60 with full cosine. Our warmup version reaches ep58 of (3 warmup + 57 cosine = 60 total) — so we're at the BEGINNING of the cosine tail when baseline is at the END. If the deep cosine tail is where most improvement comes from, this is a HUGE handicap.

   Mitigation: this is a real risk. Worst case, we lose 1-2 points to a truncated cosine tail. We could also keep total cosine at 60 epochs (so total is 63) but that exceeds SENPAI_TIMEOUT.

4. **Tiny dataset may not need warmup.** With only 3000 samples, the optimization landscape is essentially deterministic by epoch 10; warmup matters mostly when the dataset is large enough to create gradient noise.

## Falsifiable predictions

- **WIN** (val < 30.5605): Warmup helps even at small scale. Try 5-epoch warmup next.
- **WASH** (val ≈ 30.5605 ± 0.5%): No effect; close warmup-length axis at zero.
- **LOSS** (val > 31.0): Warmup hurts (likely via cosine-tail truncation). Try epochs=63 with 3-epoch warmup + 60-epoch cosine if SENPAI_TIMEOUT allows (it won't — 63 epochs at 30 sec ≈ 31.5 min). Close axis.

## Implementation

### Step 1: Replace `CosineAnnealingLR` with a `LambdaLR` schedule

In `train.py`, find the scheduler construction (around line ~620-630, where `CosineAnnealingLR(optimizer, T_max=epochs)` is built). Replace with:

```python
import math

warmup_epochs = 3
steps_per_epoch = len(train_loader)
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = cfg.epochs * steps_per_epoch  # 60 * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        # Linear warmup: 0 -> 1.0
        return float(step) / float(max(1, warmup_steps))
    # Cosine decay over remaining steps
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

**CRITICAL:** This is a **PER-STEP** scheduler — call `scheduler.step()` AFTER EACH OPTIMIZER STEP (not per epoch). Find the existing `scheduler.step()` call in the training loop and verify it's per-step. If it's per-epoch, you must move it to be per-step OR multiply the lambda's effective period by `steps_per_epoch` and keep per-epoch calling.

Recommendation: convert to per-step. Most modern recipes call schedulers per-step.

### Step 2: Add startup diagnostic

```python
print(f"LR schedule: linear warmup {warmup_epochs} epochs ({warmup_steps} steps), then cosine to 0")
print(f"Steps per epoch: {steps_per_epoch}, total steps: {total_steps}")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940
```

### Step 3: Optional — log LR per epoch

Add `lr` to metric logging at each epoch end:
```python
metrics.update({"train/lr": optimizer.param_groups[0]['lr']})
```

This verifies the warmup ramp looks correct (LR rises linearly for the first ~1125 steps, then cosine-decays).

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 |

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-frieren \
    --experiment_name "charliepai2g48h5-frieren/linear-warmup-3-epochs" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 total (3 warmup + 57 cosine). No new CLI flag — warmup is hardcoded as `warmup_epochs = 3`. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **LR curve sanity check:** report LR at start of each epoch — should ramp linearly through epochs 0-2, peak at ep3, cosine-decay through ep59. Include a single table column "epoch | lr" for first ~5 and last ~5 epochs.
4. Param count confirmation (~407,940)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train-loss vs val-loss gap (warmup may shrink early-epoch noise but final gap depends on convergence)
7. **Verdict on warmup hypothesis:** does 3-epoch linear warmup help, hurt, or wash? Compare to baseline's bare CosineAnnealingLR. Note any qualitative shape change in early training-loss curve (warmup should smooth ep1-3 train loss).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
