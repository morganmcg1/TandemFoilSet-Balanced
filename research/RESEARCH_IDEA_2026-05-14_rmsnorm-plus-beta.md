# Round 137 — RMSNorm + β bias (LayerNorm without mean-subtraction): isolate β-removal vs mean-subtraction

## Hypothesis

**Use RMSNorm but ADD BACK a learnable β bias** — equivalent to a LayerNorm WITHOUT mean-subtraction (`(x / RMS(x)) * γ + β`). This is a clean ablation that isolates the β-bias-removal effect from the mean-subtraction effect that #2939 conflated.

#2939 swapped LayerNorm → RMSNorm and changed TWO things at once: (a) removed the mean-subtraction `x - mean(x)`, and (b) removed the learnable β bias (-864 params total). The LOSS could be from either or both. This experiment isolates the variables.

This is **student of #2939's followup #1**, directly motivated by their decisive: *"Try RMSNorm + add a per-channel learnable bias (β) after the γ scale ('RMSNorm + β' or equivalently a LayerNorm without mean-subtraction). Would test whether the LOSS comes from losing the β bias or from losing the mean-subtraction itself. Cheap, +864 params back, isolates the two factors."*

## Why this might WIN

1. **#2939 student explicitly recommended this as followup #1.** Cleanest ablation experiment.

2. **If RMSNorm+β recovers baseline → LOSS came from β removal.** 864 params of per-channel bias may have been load-bearing — possibly absorbing per-channel DC offsets that mean-subtraction was masking.

3. **If RMSNorm+β still LOSS → mean-subtraction is the load-bearing piece.** This DECISIVELY closes the normalization-axis from the mean-subtraction angle.

4. **Tiny implementation change.** Add `self.bias = nn.Parameter(torch.zeros(dim))` to the RMSNorm module and add `+ self.bias` in forward.

5. **Param count restored to 407,940** (matches baseline). No capacity confound.

## Why this might LOSS

1. **The mean-subtraction may be the load-bearing piece, not β.** Then RMSNorm+β still LOSS — but informative closure.

2. **Per-Reynolds DC offsets in CFD residuals may genuinely need explicit subtraction**, not just β bias absorption.

3. **The current LN+γ+β baseline may already be near-optimal**, so any deviation hurts.

## Falsifiable predictions

- **WIN** (val < 30.5605): β was the load-bearing piece. The mean-subtraction itself is removable; just keep the β bias. Try RMSNorm+β at half the bias capacity (`nn.Parameter(torch.zeros(dim // 2)).repeat(2)`) to characterize.
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): β recovers SOME loss but mean-subtraction also matters. Both factors contribute.
- **WASH** (val ≈ 30.5605 ± 0.3%): β alone fully recovers. Mean-subtraction is incidental at this scale.
- **LOSS** (val > 31.0 OR similar to #2939's 32.79): Adding β alone doesn't help. The mean-subtraction itself was load-bearing. Decisively closes the norm-shape axis. Move to GroupNorm or stop probing.

## Implementation

### Step 1: Modify the RMSNorm module (or create a new class) to add β

If using `nn.RMSNorm` (PyTorch ≥ 2.4):
```python
# nn.RMSNorm does NOT support β by default. Replace with custom module.
class RMSNormWithBias(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # γ
        self.bias   = nn.Parameter(torch.zeros(dim))  # β  ← THIS IS THE NEW PIECE

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight + self.bias
```

### Step 2: Replace every RMSNorm site with RMSNormWithBias

Apply at every site (same 9 sites as #2939: 4× ln_1, 4× ln_2, 1× ln_3 in last block).

### Step 3: Startup diagnostics

```python
norm_sites = sum(1 for m in model.modules() if isinstance(m, RMSNormWithBias))
print(f"RMSNorm+β sites: {norm_sites} (expect 9)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # expect ~407,940 (matches baseline)
print(f"vs baseline LayerNorm (407,940): same count")
print(f"vs #2939 RMSNorm-only (407,076): +864 (the 9 β biases restored)")
print(f"Mechanism: norm WITHOUT mean-subtraction (x / RMS(x) * γ + β)")
```

### Step 4: Update `_init_weights` to cover the new module

If `_init_weights` was widened in #2939 to cover `nn.RMSNorm`, extend to cover `RMSNormWithBias` too — γ=1, β=0 (same as LayerNorm convention).

### Step 5: Per-epoch logging

Track:
- Train surf/vol losses at ep1, 5, 10, 30, 60 — compare to #2939 trajectory
- ep1 train_surf should match #2939's (1.6355) within ~5% if init is consistent
- Convergence speed — RMSNorm+β should converge FASTER than RMSNorm-only if β is load-bearing

## Baseline (PR #2879) and prior closure

| Metric | Baseline (LN) | #2939 RMSNorm (no β) | This PR target |
|---|---|---|---|
| val_avg/mae_surf_p | **30.5605** | 32.7855 (+7.28% LOSS) | beat baseline |
| Param count | 407,940 | 407,076 (-864) | ~407,940 (restored) |

For comparison:
- Baseline (PR #2879): LayerNorm at 9 sites, val 30.5605
- #2939: RMSNorm at 9 sites (no β), val 32.7855 (+7.28% LOSS) — 864 params less
- This PR: RMSNorm + β at 9 sites, val ?, params matching baseline

**Beat:** `val_avg/mae_surf_p < 30.5605`. The disambiguation outcome matters even if WASH:
- WIN → β was load-bearing, mean-subtraction is removable
- LOSS → mean-subtraction was load-bearing, β is incidental
- WASH → both contribute or norm at this scale is forgiving

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/rmsnorm-plus-beta" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160 AND vs #2939 RMSNorm 32.7855 / 28.4832
2. Per-split val + test breakdown with Δ vs baseline and vs #2939
3. **Norm-site count diagnostic:** confirm 9 RMSNormWithBias sites, β bias zero-init verified
4. **Param count confirmation:** ~407,940 (matching baseline)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train→val loss gap at convergence
7. **Disambiguation result:** is the LOSS from #2939 attributable to (a) β bias removal, (b) mean-subtraction removal, or (c) both? Use this PR's outcome to attribute.
8. **β bias diagnostic:** at ep60, log the L∞ norm of each block's β bias vector. If β values diverge significantly from 0, β was indeed absorbing structure. If ~0, β was unused.
9. **Plain-language verdict:** WIN (β was load-bearing — fix is cheap) / WASH (both factors share load) / LOSS (mean-subtraction was the load-bearing piece — keep LayerNorm).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
