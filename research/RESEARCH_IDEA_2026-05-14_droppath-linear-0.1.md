# Round 128 — DropPath (stochastic depth) with linear schedule 0→0.1

## Hypothesis

Add **stochastic depth (DropPath)** to TransolverBlock with a **linear drop-rate schedule** from 0 at block 0 to 0.1 at block 3. Tests whether regularization that reduces inter-block co-adaptation can break the **cruise/in_dist trade-off** observed across the last ~10 experiments (where in_dist over-specialization appears to come at camber_cruise's cost — see "Meta-signal" below).

## Why this might WIN

1. **Directly targets the over-specialization meta-signal.** Four losing experiments (#2889 mlp_ratio=4 -3.60%, #2890 additive geo-FiLM -9.87%, #2899 asymmetric mlp -5.00%, #2903 RMSNorm -0.69%) ALL improved camber_cruise while regressing in_dist. The pattern strongly suggests that the model over-specializes to in_dist features and loses generalization on OOD camber-cruise geometries. DropPath randomly drops residual paths during training, forcing each block to be useful in isolation — this is a textbook anti-co-adaptation regularizer.

2. **Stochastic depth is empirically robust.** Used by virtually all modern vision transformers (DeiT, Swin, ConvNeXt, MAE, DINOv2) with drop rates 0.1-0.4 depending on model size. At our small scale (4 blocks, ~407k params, ~3000 samples), a conservative 0.1 max rate is the standard small-model setting.

3. **Linear schedule with deeper-block bias is principled.** Per Huang et al. 2016 (Deep Networks with Stochastic Depth) and Touvron et al. 2021 (DeiT), drop rates should increase with depth — shallow blocks are needed every forward pass, deep blocks specialize and benefit from path-shuffling. Schedule [0.0, 0.033, 0.067, 0.1] across blocks 0-3 follows this convention.

4. **Block-wise independent stochastic dropping is parameter-free.** No new layers, no new buffers, just torch.rand at training time per block per batch. Adds ~0 params. Pure regularization test.

5. **Compounds well with existing FiLM/SE conditioning.** The conditioning paths (flow-FiLM, SE block-3) are inside blocks and survive when blocks are kept; when blocks drop, the residual passes through unchanged. Should not interact pathologically with the conditioning layers.

## Why this might LOSS

1. **Tiny dataset may not need additional regularization.** ~3000 train samples + L1 + weight_decay=3e-4 + cosine LR already provides regularization. Additional DropPath could under-fit, especially in early epochs.

2. **Test/inference removes DropPath entirely.** During eval, all blocks are kept and outputs are scaled. If the rescaling assumption (output_in_eval ≈ E[output_in_train]) is violated for our specific architecture (especially with SE block-3 + FiLM), eval-time metrics could regress.

3. **Drop rate 0.1 may be too low to see effects** at this small depth — only 4 blocks means at most ~0.4 expected dropped paths per forward pass. May need 0.2-0.3 to see strong regularization.

4. **The cruise/in_dist trade-off may not be co-adaptation-driven.** If it's actually structural (e.g., the network needs different feature representations for different splits), block-level dropping won't help.

## Falsifiable predictions

- **WIN** (val < 30.5605): DropPath reduces over-fit; both cruise and in_dist improve OR cruise improves while in_dist holds. Try linear 0→0.2 next, and per-block drop scheduling variants.
- **PARTIAL WIN** (camber_cruise drops below 17.0 AND in_dist within ±2% of baseline): trade-off partially decoupled. Try DropPath + mixup or DropPath + lower weight_decay.
- **WASH** (val ≈ 30.5605 ± 0.5%): No effect at small drop rate. Try larger drop rate (0.2-0.3) before closing.
- **LOSS** (val > 31.0): DropPath at this scale hurts. Close stochastic-depth axis.

## Implementation

### Step 1: Add a `DropPath` class to `train.py`

Place near the top of the file (after the `class SwiGLUMLP` definition, before `class PhysicsAttention`):

```python
class DropPath(nn.Module):
    \"\"\"Per-sample stochastic depth (Huang et al. 2016).
    During training, drops entire batch elements' residual contribution with probability drop_prob.
    During eval, identity.
    \"\"\"
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, ...) — broadcast across token dim and feature dim
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()  # 0 or 1 mask
        return x.div(keep_prob) * rand
```

### Step 2: Add per-block DropPath modules in `TransolverBlock.__init__`

Modify `TransolverBlock.__init__` to accept a `drop_path` arg and create two DropPath layers (one per residual: attention and MLP):

```python
def __init__(
    self,
    num_heads,
    hidden_dim,
    dropout,
    act,
    mlp_ratio,
    out_dim,
    slice_num,
    last_layer,
    use_se,
    drop_path=0.0,  # NEW
):
    super().__init__()
    # ... existing setup ...
    self.drop_path_attn = DropPath(drop_path)
    self.drop_path_mlp = DropPath(drop_path)
    # ... rest unchanged
```

### Step 3: Apply DropPath in `TransolverBlock.forward`

Find the residual connections around the attention and MLP. The current pattern looks like:

```python
# CURRENT (residuals around line 218-228)
x = x + self.attn(self.ln_1(x))    # attention residual
x = x + self.mlp(self.ln_2(x))     # MLP residual
```

Change to:

```python
# AFTER
x = x + self.drop_path_attn(self.attn(self.ln_1(x)))
x = x + self.drop_path_mlp(self.mlp(self.ln_2(x)))
```

**IMPORTANT:** DropPath wraps the **branch output** (the attention/MLP output), NOT the residual itself. The residual `x` always passes through; when DropPath fires, the branch contributes zero to that sample's residual sum.

### Step 4: Linear drop-rate schedule across blocks in `Transolver.__init__`

In `Transolver.__init__`, before creating `self.blocks`:

```python
# Linear schedule: 0, 0.033, 0.067, 0.1 for blocks 0-3
drop_path_rates = [x.item() for x in torch.linspace(0.0, 0.1, n_layers)]
print(f"DropPath rates per block: {drop_path_rates}")

self.blocks = nn.ModuleList([
    TransolverBlock(
        num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
        act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
        slice_num=slice_num, last_layer=(i == n_layers - 1),
        use_se=(i == n_layers - 1),
        drop_path=drop_path_rates[i],  # NEW
    )
    for i in range(n_layers)
])
```

### Step 5: Diagnostics

At training start, print:
```python
print(f"DropPath schedule: {drop_path_rates}")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")
```

Per epoch, optionally log:
- `model.training` state (should be True during train, False during eval — confirm DropPath is OFF in eval)
- Block-level activation norms (do dropped blocks during training cause feature norm to fluctuate?)

## Baseline (PR #2879, current best)

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

### The meta-signal we're trying to exploit

| Experiment | val_avg Δ | camber_cruise Δ | in_dist Δ |
|---|---|---|---|
| #2889 mlp_ratio=4 | +2.91% LOSS | **-3.60% WIN** | +10.88% LOSS |
| #2890 additive geo-FiLM | +3.09% LOSS | **-9.87% WIN BIGGEST** | +16.85% LOSS |
| #2899 asym mlp [3,3,4,4] | +2.38% LOSS | **-5.00% WIN** | +10.45% LOSS |
| #2903 RMSNorm | +2.10% LOSS | **-0.69% WIN** | +4.83% LOSS |
| **Target this PR** | **<0% WIN** | **<0% retained** | **<0% retained** |

If DropPath reduces over-fit, expect: in_dist holds or slightly improves AND camber_cruise improves (capturing the regularization signal that was being expressed via destabilizing mods).

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/droppath-linear-0.1" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 (not 70) to fit within SENPAI_TIMEOUT=30min. No new CLI flag; DropPath is hardcoded in TransolverBlock + Transolver. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test with delta vs baseline
3. **Trade-off table:** the 4-row table above with this PR's row added — does DropPath retain camber_cruise improvement while NOT regressing in_dist?
4. Param count confirmation (~407,940 — DropPath adds zero params)
5. Epochs completed (target: all 60), sec/epoch, peak GPU memory
6. Training-loss-vs-val-loss gap (DropPath should reduce this gap if regularization is the right diagnosis)
7. **Plain-language verdict on the meta-signal:** did stochastic depth break the cruise/in_dist trade-off? If YES → try 0.2 schedule. If WASH → try 0.2 before closing. If LOSS → close stochastic-depth axis.

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
