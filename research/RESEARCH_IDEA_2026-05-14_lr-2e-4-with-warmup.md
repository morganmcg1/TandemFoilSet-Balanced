# Round 137 — Peak LR=2.0e-4 with the existing 3-epoch warmup (LR-magnitude axis, ramp preserved)

## Hypothesis

**Raise peak LR from 1.5e-4 to 2.0e-4 while keeping the load-bearing 3-epoch LinearLR(0.1→1.0×) warmup intact.** This is the **opposite direction** of the no-warmup axis just closed by #2920, #2929, #2938 — tests whether the baseline lr=1.5e-4 sits at the LR-magnitude optimum or *below it*, with the load-bearing ramp shape preserved (the only schedule shape we have evidence is non-negotiable).

This is **student of #2938's followup #2**, directly motivated by their decisive insight: *"If the schedule axis is worth revisiting, the productive direction is the OPPOSITE: try a longer warmup (4-5 epochs instead of 3), or a HIGHER PEAK LR with the existing warmup (e.g. lr=1.75e-4 or 2.0e-4 with the 3-epoch ramp). The ramp gives headroom that cold-start cannot reproduce."*

## Why this might WIN

1. **#2938 student explicitly recommended this** as the productive opposite-direction probe to the closed no-warmup axis. Verbatim above.

2. **Total optimization energy increase.** Baseline lr=1.5e-4 over 60ep gives ∫LR(t) dt ≈ 4.4e-3. Lr=2.0e-4 gives ≈ 5.9e-3 (+33% more total optimization energy). Combined with #2938's finding that the model is **under-trained at 60 epochs even with warmup**, more total LR may unlock improvement.

3. **Ramp preserved.** The 3-epoch LinearLR(0.1→1.0×) warmup is the only schedule component we have *positive* evidence for. By keeping it intact, this is a clean LR-magnitude axis test — uncontaminated by schedule-shape factors.

4. **Lion is robust to high LR.** Lion's sign-step normalization makes it more tolerant of high LR than Adam. The 1.5e-4 baseline is the conservative end of Lion's typical operating range (1e-4 to 5e-4 in the literature for similar architectures).

5. **Warmup ramp HEADROOM.** With warmup, peak LR reached at ep3 starts from 2.0e-4 × 0.1 = 2.0e-5 (still well below baseline ep1 effective LR), gives the model 3 epochs to adapt before higher peak LR kicks in. The ramp specifically buys headroom that cold-start cannot reproduce per #2938.

6. **LR magnitude was never explicitly tested upward.** Baseline lr=1.5e-4 was set at launch; previous LR-magnitude tests went down (1.0e-4 in #2938) or removed warmup. No experiment in this launch has tested lr UP with the load-bearing ramp preserved.

## Why this might LOSS

1. **Lion at lr=2.0e-4 may destabilize.** Lion's sign-step has implicit normalization but unbounded effective gradient magnitude can still cause instability. If Lion at 2.0e-4 destabilizes mid-training (ep10-30), we'd see val regress.

2. **The baseline tuning may be near the optimum.** The 1.5e-4 setting was likely chosen with some tuning. The function may be relatively flat between 1.0e-4 and 2.0e-4 with 1.5e-4 close to optimum, making +33% LR a small perturbation in either direction.

3. **Over-fitting risk.** Higher LR + warmup → faster early descent → more time at low LR in cosine tail → more capacity to over-fit. Could regress on OOD splits while in_dist improves.

4. **Lion at high LR has been shown to over-shoot.** In some studies Lion plateaus or regresses at lr=2e-4+ on similar small Transformer models.

## Falsifiable predictions

- **WIN** (val < 30.5605): LR magnitude axis is unsaturated upward. Baseline tuning was conservative. Try lr=2.5e-4 to characterize peak. Particularly check in_dist (most LR-sensitive split typically).
- **PARTIAL** (val ≈ 30.5605 ± 0.3% WASH or marginally LOSS): LR axis is flat in this neighborhood. 1.5e-4 is at the optimum. Close LR-magnitude axis upward direction.
- **LOSS** (val > 31.0): Higher LR destabilizes or over-fits with this fixed budget. Likely LR axis caps at ~1.5e-4 for Lion at this scale. Closes LR-magnitude axis upward; consider longer-warmup variant next round.
- **DIVERGE** (NaN or val > 50): Lion destabilizes at 2.0e-4. Closes LR-magnitude axis at the Lion stability boundary — try lr=1.75e-4 next.

## Implementation

### Step 1: Change peak LR to 2.0e-4 via CLI flag

```bash
--lr 2.0e-4
```

This is the **only change**. Keep `--weight_decay 3e-4` baseline, keep `--epochs 60`, no other modifications.

### Step 2: Verify warmup ramp is intact

Confirm in the run output that:
- LR at epoch 1 ≈ 2.0e-5 (10% of peak = 2.0e-4)
- LR at epoch 2 ≈ 8.0e-5 (warmup midpoint, ~40% peak via LinearLR(0.1→1.0))
- LR at epoch 3 ≈ 2.0e-4 (peak reached)
- LR at epoch 30 ≈ 1.0e-4 (cosine midpoint, 50% of peak)
- LR at epoch 60 ≈ 0 (cosine endpoint)

### Step 3: Startup diagnostics

```python
print(f"Peak LR: 2.0e-4 (vs baseline 1.5e-4, +33%)")
print(f"Warmup: 3-epoch LinearLR(0.1 → 1.0×) [baseline schedule preserved]")
print(f"Cosine: CosineAnnealingLR(T_max=57) [baseline schedule preserved]")
print(f"Total ∫LR(t) dt ≈ 5.9e-3 (vs baseline ~4.4e-3, +33% optimization energy)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
print(f"Schedule shape: IDENTICAL to baseline, only peak magnitude scaled by 1.33×")
```

### Step 4: Stability monitoring

Track every 10 epochs:
- Train loss (surf + vol). If NaN/Inf, terminate.
- Grad-norm (if available). If diverges or spikes, log.
- Val loss every epoch (already standard).

If divergence occurs in ep5-15, that's the Lion stability boundary signal.

### Step 5: Per-split test diagnostic

At ep60, report not just val_avg but per-split val + test breakdown. Critical to check whether higher LR helps cruise (the split that typically improves with more capacity / more optimization) vs in_dist (typically over-fits faster).

## Baseline (PR #2879) and prior closure

| Metric | Baseline (lr=1.5e-4 + warmup) | #2938 (lr=1.0e-4 no-warmup) | This PR target |
|---|---|---|---|
| val_avg/mae_surf_p | **30.5605** | 34.6871 (+13.50% LOSS) | beat baseline |
| Schedule | 3ep warmup → cosine | pure cosine (no warmup) | 3ep warmup → cosine |
| Peak LR | 1.5e-4 | 1.0e-4 | 2.0e-4 |
| ∫LR(t) | ~4.4e-3 | ~3.0e-3 | ~5.9e-3 |

For comparison (closing the no-warmup axis, NOT this PR):
- #2920: per-step warmup-from-0, lr=1.5e-4 → val ~32.6 (+6.79% LOSS)
- #2929: pure cosine no-warmup, lr=1.5e-4 → val 32.6744 (+6.91% LOSS)
- #2938: pure cosine no-warmup, lr=1.0e-4 → val 34.6871 (+13.50% LOSS)

For comparison (LR magnitude axis upward, never tested):
- This PR: lr=2.0e-4 + warmup at +33% peak with ramp preserved

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-frieren \
    --experiment_name "charliepai2g48h5-frieren/lr-2e-4-with-warmup" \
    --lr 2.0e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **LR schedule trajectory check:** confirm peak reached at ep3 = 2.0e-4, no warmup shortcuts
4. **Stability:** any NaN/Inf during training? Grad-norm trajectory if available.
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence vs #2938 (under-trained signature)
8. **Meta-signal check:** does higher LR specifically help cruise or in_dist? Or move both uniformly?
9. **LR-magnitude axis verdict:** is baseline 1.5e-4 sub-optimal (WIN at 2.0e-4) or at the optimum (WASH) or at Lion's stability boundary (LOSS / DIVERGE)?
10. **Plain-language verdict:** WIN (LR was sub-optimal) / WASH (LR magnitude axis is flat at this neighborhood) / LOSS (1.5e-4 is at or near optimum, higher destabilizes).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
