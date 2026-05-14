# Round 117 — Stochastic Depth (DropPath) on TransolverBlock residuals

## Hypothesis (Round 117 — Bold-swing #2 in plateau protocol; 16 LOSSes since #2810 merge)

Add **Stochastic Depth (DropPath)** to the TransolverBlock attn and MLP residual paths. During training, randomly drop the entire attn or MLP contribution per-sample at a depth-progressive rate (linear schedule 0 → 0.1 across 4 blocks). At evaluation, no dropping (deterministic forward).

### Why this might WIN

1. **Well-validated regularizer for transformers.** Huang 2016 (Deep Networks with Stochastic Depth, residual nets), Touvron 2021 (DeiT, ImageNet ViT), Liu 2022 (ConvNeXt), Zhang 2023 (Vision Transformers). Standard practice for any transformer at this depth — and it's never been tried in this programme.
2. **Targets the gradient path, not the activation/architecture.** 16 consecutive LOSSes have all been activation-site, attention-site, normalization-site, or capacity-axis tweaks. None have targeted the residual gradient flow. This is genuinely new surface.
3. **Implicit ensemble of sub-networks.** Each forward pass during training samples a random sub-net of 1-4 active blocks. The model learns to be robust to block ablation, which empirically improves generalization across the depth.
4. **Zero added parameters.** Same 333,700 params; only training-time stochastic Bernoulli noise on residual paths.
5. **Direct attack on the recurring OOD pattern.** Recent merge winners (SE-attn-pool #2741, FiLM-embedding #2810) shared an "OOD-positive bias" via regularization rather than capacity. DropPath is in the same class — induces feature redundancy that may help OOD distribution shift (camber_rc, camber_cruise, re_rand all baselined > in-dist).
6. **Compatible with LayerScale.** Our γ=1e-4 LayerScale + DropPath is the exact recipe used in DeiT-III / ConvNeXt — they synergize because LayerScale starts each block near-identity (easy to drop early in training) and DropPath teaches the model to function without each block.

### Why this might LOSS

1. **Small-scale models may not need regularizer.** 333k params on 4 blocks is tiny; DropPath was designed for 100M+ models with 12-24 layers. At our depth=4, dropping 10% of blocks means dropping 0.4 blocks/sample on average — may starve the model of capacity rather than regularize it.
2. **CFD regression may not benefit like image classification does.** ViT DropPath helps with cross-entropy at large scale. L1 regression on continuous physical fields may show different sensitivity.
3. **Lion + DropPath interaction is untested.** Standard recipes use AdamW. Lion's sign-step + stochastic block dropping may compound noise unpredictably.

### Falsifiable predictions

- **WIN** (val < 30.8909): DropPath breaks the plateau; try max_rate=0.15 or 0.2 next, or DropPath on slice-routing instead of full block.
- **WASH** (val ≈ 30.8909 ± 0.5%): regularizer is at the edge of helpful; try cosine-scheduled DropPath rate or test on individual residual paths only.
- **LOSS** (val > 30.8909 + 1%): small-scale model lacks capacity to absorb 10% block drop; close max_rate=0.1, try smaller max_rate=0.05 or disable for early blocks.

## Implementation

### Step 1: Add DropPath module (or use torch.nn.functional)

In `train.py`, add a small helper class (above `TransolverBlock` at line 200):

```python
class DropPath(nn.Module):
    """Per-sample stochastic depth (timm-compatible implementation).

    Drops the entire residual branch with probability drop_prob during training.
    No-op at eval. Scales surviving samples by 1/keep_prob to maintain mean.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: [B, 1, 1, ...] broadcasting over all dims except batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor
```

### Step 2: Modify TransolverBlock to use DropPath

In `TransolverBlock.__init__` (line 200), add `drop_path_rate=0.0` parameter and instantiate two `DropPath` modules:

```python
def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
             mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
             layerscale_init=1e-4, use_se=True, drop_path_rate=0.0):  # NEW
    super().__init__()
    # ... existing init unchanged ...
    self.drop_path_attn = DropPath(drop_path_rate)  # NEW
    self.drop_path_mlp = DropPath(drop_path_rate)   # NEW
```

In `TransolverBlock.forward` (line 223), wrap the residual sub-block outputs with DropPath:

```python
def forward(self, fx, mask=None):
    # BEFORE:
    # fx = self.gamma_attn * self.attn(self.ln_1(fx)) + fx
    # fx = self.gamma_mlp * self.mlp(self.ln_2(fx)) + fx

    # AFTER:
    fx = self.drop_path_attn(self.gamma_attn * self.attn(self.ln_1(fx))) + fx
    fx = self.drop_path_mlp(self.gamma_mlp * self.mlp(self.ln_2(fx))) + fx

    if self.se is not None:
        fx = self.se(fx, mask=mask)
    if self.last_layer:
        return self.mlp2(self.ln_3(fx))
    return fx
```

### Step 3: Linear depth-progressive schedule in Transolver

In `Transolver.__init__` (line 252-262), compute per-block drop_path rates with a linear schedule from 0 to `max_drop_path_rate`:

```python
# In Transolver.__init__, replace the self.blocks = nn.ModuleList([...]) with:
max_drop_path_rate = 0.1  # NEW: hardcode for first run; can be CLI flag later
drop_path_rates = [max_drop_path_rate * i / max(1, n_layers - 1) for i in range(n_layers)]
# For n_layers=4: [0.0, 0.033, 0.067, 0.1]

self.blocks = nn.ModuleList([
    TransolverBlock(
        num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
        act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
        slice_num=slice_num, last_layer=(i == n_layers - 1),
        use_se=(i == n_layers - 1),
        drop_path_rate=drop_path_rates[i],  # NEW
    )
    for i in range(n_layers)
])
```

### Step 4: Startup diagnostic prints

After model construction:

```python
print(f"DropPath: linear schedule 0.0 → 0.1 across {n_layers} blocks")
print(f"Per-block drop_path_rate: {[f'{r:.3f}' for r in drop_path_rates]}")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # expect 333,700 unchanged
```

### Step 5: Diagnostics to log per val pass

Per-block drop-path active-rate sanity check (since DropPath is no-op at eval, this can only be measured during training):

- `drop_path/block_<i>_rate` — the static rate per block (sanity check; should match schedule)
- `train/drop_path_active_frac_block_<i>` — empirical fraction of train batches where block_<i> attn-path or mlp-path was dropped (over the epoch; should converge to drop_path_rate as a sanity check on the Bernoulli implementation)

Standard validation metrics unchanged.

## Baseline (current best — PR #2810 Round 101)

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p | **30.8909** | ep64/65; 20th winner |
| test_avg/mae_surf_p | **26.1964** | from best-val ep64 |
| Param count | **333,700** | (this experiment expected unchanged) |
| val_single_in_dist | **25.2751** | |
| val_geom_camber_rc | **45.8179** | |
| val_geom_camber_cruise | **16.8427** | |
| val_re_rand | **35.6177** | |

**Target to beat:** val_avg/mae_surf_p < **30.8909**

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-tanjiro \
    --experiment_name "charliepai2g48h5-tanjiro/drop-path-0.1-linear" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 70
```

**IMPORTANT**: Use canonical hyperparameters (lr=1.5e-4, weight_decay=3e-4, epochs=70). The DropPath rates are HARDCODED inside `Transolver.__init__` (max=0.1, linear schedule); no new CLI flag needed for this initial test.

**No W&B / wandb** — local JSONL only. `SENPAI_TIMEOUT_MINUTES=30` hard cap.

## Reporting

Post results as a PR comment including:

1. val_avg vs baseline **30.8909**; test_avg vs baseline **26.1964** — anchor to BASELINE.md reference numbers
2. Per-split breakdown (4 val + 4 test splits) — pay attention to whether OOD splits improve relative to in_dist (regularizer signature) or all splits move together (capacity signature)
3. Per-block DropPath rate schedule confirmation (should print at startup as `[0.000, 0.033, 0.067, 0.100]` for n_layers=4)
4. Empirical drop-path active fraction per block over the final 5 epochs — should match the schedule closely (proves Bernoulli implementation works)
5. Param count confirmation: **333,700** expected (unchanged)
6. Total epochs reached, sec/epoch, peak GPU memory
7. Training-loss-vs-val-loss gap — DropPath should INCREASE train loss but DECREASE (or maintain) val loss; report both

Use the terminal result marker:
```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
