# Round 133 — Pure CosineAnnealingLR (no warmup) — student-suggested follow-up

## Hypothesis

**Remove the existing 3-epoch per-epoch linear warmup** and use **pure `CosineAnnealingLR(T_max=60)` from peak LR**, stepped per-epoch. Tests whether the baseline's existing warmup-from-0.1× is doing anything useful for Lion at lr=1.5e-4 — or whether it's just wasted budget at low LR that could be spent at peak.

## Why this might WIN

1. **Student of #2920 explicitly recommended this test.** Their #1 followup: *"Test cosine without warmup (pure CosineAnnealingLR from peak, per-epoch): isolates whether any warmup helps Lion at lr=1.5e-4 vs no warmup at all. If pure cosine matches or beats the per-epoch warmup baseline, then the existing 3-epoch warmup is dead weight."* This is the most informative possible follow-up after the per-step-from-zero variant failed.

2. **Lion's sign-step is inherently smooth.** Chen et al. 2023 (Lion paper): the sign-based update is direction-stable across any LR magnitude, unlike Adam which has noisy second moments early in training. Warmup is the canonical fix for Adam's early-training noisy variance — Lion doesn't have this pathology. The 3-epoch warmup-from-0.1× may be inherited from Adam-era recipes and unnecessary for Lion.

3. **Lion's update doesn't benefit from low-LR warmup.** Sign(exp_avg + β₁·grad) is bounded in magnitude regardless of LR. The actual update size scales linearly with LR, so warmup acts purely as a smaller step size in early training. For an architecture that's stable (no early-training divergence in baseline), small early steps just slow convergence.

4. **Recover 3 epochs at peak LR.** The current warmup costs ~3 epochs at fractional LR (0.15e-4 → 1.5e-4). Pure cosine from peak gives all 60 epochs at full schedule, gaining ~5% effective high-LR training time.

5. **Trivial implementation, zero new params.** Single-line scheduler change.

## Why this might LOSS

1. **Warmup may be load-bearing for stability with FiLM + SE + LayerScale interactions.** Even if Lion itself doesn't need warmup, the architecture's gated conditioning paths may produce early-training amplification. Mitigation: log ep1-3 train loss carefully.

2. **The per-epoch warmup may already be near-optimal cosine-shape.** start_factor=0.1 + 3 epochs out of 60 = a very gentle warmup that's nearly equivalent to running cosine from peak anyway. The difference may be in the noise.

3. **Could destabilize the trade-off at convergence.** If warmup was implicitly biasing the model toward a particular minimum (the cosine path matters in non-convex), removing it could land in a different basin with different cruise/in_dist trade-off.

## Falsifiable predictions

- **WIN** (val < 30.5605): Existing warmup was dead weight. Try pure cosine with `eta_min=1e-6` next.
- **WASH** (val ≈ 30.5605 ± 0.3%): Warmup-from-0.1× is essentially identity at this scale. Close warmup axis.
- **LOSS without instability** (val > 31.0, ep1-3 normal): Warmup was helping in a non-obvious way. Don't pursue further.
- **LOSS with ep1-3 spike** (train loss diverges at ep1-3): Warmup was load-bearing for stability; confirmed needed. Close axis.

## Implementation

### Step 1: Locate the SequentialLR construction in `train.py`

The current scheduler is:
```python
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=3
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=57),  # 57 = 60 - 3
    ],
    milestones=[3],
)
```

### Step 2: REPLACE with pure CosineAnnealingLR over the full 60 epochs

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
```

That is the **only change**. `scheduler.step()` calls remain per-epoch as before. No other code changes.

### Step 3: Startup diagnostics

```python
print(f"LR schedule: pure CosineAnnealingLR(T_max={cfg.epochs}) — NO WARMUP")
print(f"Peak LR: {optimizer.param_groups[0]['lr']:.4e} (starting LR)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-epoch LR + train loss logging

Critical: report train loss at ep1, ep2, ep3 vs baseline ep1-3 — looking for any spike that would indicate warmup was load-bearing for stability.

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

Current schedule: SequentialLR(LinearLR(0.1→1.0, 3 ep) + CosineAnnealingLR(T_max=57)), per-epoch.

After change: CosineAnnealingLR(T_max=60), per-epoch, from peak.

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-frieren \
    --experiment_name "charliepai2g48h5-frieren/no-warmup-pure-cosine" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **LR sanity table:** LR at ep1, 2, 3, 5, 10, 30, 50, 60 — should start at peak (1.5e-4) and end near 0. No warmup.
4. **Train loss at ep1-3:** compared to baseline ep1-3. Watch for instability spikes.
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence
8. **Plain-language verdict:** WIN (existing warmup was dead weight) / WASH (no effect at this scale) / LOSS-stable (warmup was helping somehow) / LOSS-spike (warmup was load-bearing).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
