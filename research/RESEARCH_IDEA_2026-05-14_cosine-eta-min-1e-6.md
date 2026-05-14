# Round 133 — Cosine eta_min=1e-6 (non-zero LR floor)

## Hypothesis

Change the existing `CosineAnnealingLR(T_max=57)` (which decays to 0) to **`CosineAnnealingLR(T_max=57, eta_min=1e-6)`** — keep the same warmup + same schedule shape, but anchor the cosine tail at a small non-zero floor instead of decaying all the way to 0. Tests whether the deep-tail epochs (where LR < 1e-6) contribute nothing useful — and if so, redistributing that final budget at a meaningful LR magnitude can improve convergence.

## Why this might WIN

1. **Student of #2920 explicitly recommended this test.** Their #3 followup: *"Cosine to a non-zero floor (e.g., eta_min=1e-6): prevents the dead-tail epochs where lr < 1e-7 contributes ~nothing."* From the LR table at ep55-60 in #2920, LR drops to 2.8e-6, 1.8e-6, 1.0e-6, 4.5e-7, 1.1e-7, 0 — the last 3 epochs are essentially LR=0 and contribute pure gradient-step-zero "frozen" updates.

2. **Best epoch in baseline #2879 was ep58 of 70** (cosine fully tailing to 0). The model was still slowly improving when LR reached negligible magnitudes. With `eta_min=1e-6`, the final ~5 epochs run at a small but meaningful LR (~1e-6) instead of decaying to 0 — these "extra" useful steps may unlock a few additional units of fine-tuning.

3. **Modern transformer recipes use eta_min > 0.** ViT, BEiT, and most large-scale recipes use `eta_min` in [1e-6, 1e-5] rather than zero. The zero floor is a torch default, not an optimization-derived constant.

4. **The change is mechanistically distinct from removing warmup.** Frieren's parallel experiment removes warmup entirely. This experiment keeps warmup but addresses the tail. Both axes are orthogonal and complementary.

5. **Trivial implementation, zero new params.** One kwarg added to existing scheduler.

## Why this might LOSS

1. **Lion at lr=1e-6 may still produce non-zero updates.** Lion's sign-step magnitude scales with LR, but sign(g) is non-zero whenever g is non-zero. So at lr=1e-6, the model takes ~steps of magnitude 1e-6 per param. Over 5 final epochs at this LR, the model accumulates ~5e-6 of drift per param — potentially perturbing the converged minimum to a slightly worse location.

2. **Lion + small LR could cause sign-noise oscillation.** If gradients are small (model has converged), sign(g) becomes dominated by gradient noise. The model could drift slightly in random directions per the noisy sign.

3. **WASH outcome is likely.** Given how small 1e-6 is, the effect could be within run-to-run variance. The baseline trajectory shows ep58-60 plateau at 30.56/30.79/30.96, suggesting late epochs are noise-bound regardless.

## Falsifiable predictions

- **WIN** (val < 30.5605): Non-zero LR tail provides useful late-training signal. Try eta_min=1e-5 as confirmation.
- **WASH** (val ≈ 30.5605 ± 0.3%): Late-tail LR magnitude doesn't matter; the convergence is gradient-bound. Close axis.
- **LOSS** (val > 31.0): Non-zero floor perturbs converged solution. Try smaller eta_min (1e-7) or close axis.

## Implementation

### Step 1: Locate the CosineAnnealingLR construction

In `train.py`, the current scheduler builds CosineAnnealingLR with T_max=57 (within a SequentialLR with LinearLR warmup):

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=57)
```

### Step 2: Add eta_min=1e-6

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=57, eta_min=1e-6)
```

**That is the only change** — keep warmup, keep T_max=57, keep per-epoch stepping. Just anchor the tail at 1e-6 instead of 0.

### Step 3: Startup diagnostics

```python
print(f"LR schedule: LinearLR warmup 3ep → CosineAnnealingLR(T_max=57, eta_min=1e-6) → floor 1e-6")
print(f"Peak LR: {optimizer.param_groups[0]['lr']:.4e}")
print(f"Expected min LR after 60ep: 1e-6 (not 0)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-epoch LR logging

Log LR at end of every epoch. Especially important: LR at ep55, 56, 57, 58, 59, 60 should all be near 1e-6 (not decaying to 0). Compare to baseline's tail epochs which drop to 0.

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

Current schedule: SequentialLR(LinearLR(0.1→1.0, 3 ep) + CosineAnnealingLR(T_max=57, eta_min=0)), per-epoch.

After change: SequentialLR(LinearLR(0.1→1.0, 3 ep) + CosineAnnealingLR(T_max=57, **eta_min=1e-6**)), per-epoch.

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-fern \
    --experiment_name "charliepai2g48h5-fern/cosine-eta-min-1e-6" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **LR tail diagnostic:** LR at ep55, 56, 57, 58, 59, 60. Confirm tail floors at 1e-6 (not 0).
4. **Late-training trajectory:** val_avg at ep55, 56, 57, 58, 59, 60 — does the model continue to descend at the 1e-6 floor or plateau?
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Best epoch — was it earlier than 60 (cosine tail useless) or at 60 (still useful)?
8. **Plain-language verdict:** WIN (non-zero tail unlocks late descent) / WASH (no late effect) / LOSS (floor perturbs converged solution).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
