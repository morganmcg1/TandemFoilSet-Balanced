# Round 136 — Pure CosineAnnealingLR with peak lr=1.0e-4 (no-warmup disambiguation)

## Hypothesis

**Replicate #2929's pure CosineAnnealingLR(T_max=60) schedule (no warmup) but with peak lr=1.0e-4 instead of 1.5e-4.** Disambiguates whether #2929's LOSS (+6.91% val) came from "no warmup at all" or "too much LR too soon at the cold start."

The baseline's effective LR averaged over ep1-3 (under the 3-epoch 0.1→1.0× warmup) is ~0.55 × 1.5e-4 ≈ 0.825e-4. So #2929 ran at ~1.8× the baseline's effective early-training LR. A no-warmup recipe with reduced peak LR is a more apples-to-apples comparison.

## Why this might WIN

1. **Student of #2929 explicitly recommended this as followup #2.** Verbatim: *"Try peak LR matched to no-warmup schedule. Pure cosine starts at peak immediately; the effective 'warmup-integrated' peak under the baseline schedule is ~0.55× peak averaged over ep1-3. Test lr=1.0e-4 with no warmup to see if the loss comes from 'too much LR too soon' rather than 'no warmup at all.'"*

2. **Disambiguates the schedule-axis closure.** If WIN → the "no warmup" framing of #2929 was wrong; the real issue was LR magnitude. If LOSS → warmup mechanism itself is load-bearing.

3. **Completes the schedule axis analysis.** Combined with #2920 (warmup-from-0) and #2929 (no warmup at 1.5e-4), this experiment pins down the cause of the LR-schedule LOSS by isolating LR magnitude from schedule shape.

4. **Lion's stability at lr=1.0e-4 is well-established.** #2929 proved Lion+sign-step is stable from cold start at 1.5e-4; at 1.0e-4 (33% lower) it's strictly safer.

5. **Trivial implementation, zero new params.** Two changes: scheduler swap (to pure cosine T_max=60) AND --lr 1.0e-4 on the CLI.

## Why this might LOSS

1. **Warmup may not be just LR magnitude — it could be SHAPE.** A gradual ramp from 0.1→1.0× over 3 epochs is mechanistically distinct from a constant-low-LR start. The ramp may give the optimizer time to "discover" the loss landscape structure before fully committing.

2. **lr=1.0e-4 may be too low for 60 epochs.** The total LR budget under pure cosine is ∫₀⁶⁰ lr(t) dt = 0.5 × 60 × peak_lr. At peak=1e-4 this is 3.0e-3 vs baseline ~4.4e-3 (with warmup at 1.5e-4 peak). 33% less total optimization "energy."

3. **Could still hurt in_dist** like #2929. The structural cruise/in_dist trade-off may persist regardless of LR magnitude.

## Falsifiable predictions

- **WIN** (val < 30.5605): "No warmup" framing was wrong; the LOSS came from LR magnitude. Suggests lower peak LR + no warmup is a productive recipe direction. Try lr=0.8e-4 or 1.2e-4 to find optimum.
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): Disambiguation confirmed — LR magnitude was the cause. Probably WASH.
- **WASH** (val 31.5-32.0): LR magnitude reduction recovered ~half the #2929 gap; both factors (LR + ramp shape) contribute partially.
- **LOSS** (val > 32.0): Warmup ramp SHAPE itself is the load-bearing factor — not just average LR. Closes the no-warmup axis completely. Don't pursue further.

## Implementation

### Step 1: Replace SequentialLR with pure CosineAnnealingLR

Same change as #2929:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
```

### Step 2: Use --lr 1.0e-4 (instead of baseline 1.5e-4)

```bash
--lr 1.0e-4
```

That's the only change vs #2929.

### Step 3: Startup diagnostics

```python
print(f"LR schedule: pure CosineAnnealingLR(T_max={cfg.epochs}) — NO WARMUP, peak={optimizer.param_groups[0]['lr']:.4e}")
print(f"vs #2929 pure cosine at 1.5e-4 peak: 33% lower peak, ~33% lower total optimization energy")
print(f"vs baseline warmup schedule: ~21% higher cold-start LR (1e-4 vs ~0.55*1.5e-4=0.825e-4 avg ep1-3)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-epoch logging

LR at ep1, 2, 3, 5, 10, 30, 50, 60. ep1 should be at exactly 1.0e-4. ep60 should be ~0. Compare ep1-3 train loss to #2929's (0.91 → 0.48 → 0.37 at lr=1.5e-4 from cold start) — at lr=1e-4 expect slightly higher ep1 train loss but smoother descent.

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

For comparison:
- #2929 pure cosine, peak 1.5e-4: val 32.6744 (+6.91% LOSS), test 26.9837
- #2920 per-step warmup-from-0, peak 1.5e-4: val 32.6362 (+6.79% LOSS)

After this PR: pure cosine, peak 1.0e-4 (this PR). Schedule shape from #2929, magnitude reduced 33%.

**Beat:** `val_avg/mae_surf_p < 30.5605` (would be a WIN — disambiguates schedule axis with surprising direction)

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-frieren \
    --experiment_name "charliepai2g48h5-frieren/pure-cosine-lr-1e-4" \
    --lr 1.0e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160 AND vs #2929 32.6744 (the prior no-warmup arm at 1.5e-4)
2. Per-split val + test breakdown
3. **LR table:** ep1, 2, 3, 5, 10, 30, 50, 60 (should start 1.0e-4, end ~0)
4. **Train loss ep1-3 comparison vs #2929:** is the descent smoother at lower LR?
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence
8. **Schedule-axis disambiguation:** the prior #2929 LOSS — was it from (a) too much LR too soon, or (b) ramp shape vs constant start, or (c) both? Use this PR's result to attribute.
9. **Plain-language verdict:** WIN (LR magnitude was the issue, not warmup) / WASH (partial recovery, both factors contribute) / LOSS (ramp shape is load-bearing regardless of magnitude).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
