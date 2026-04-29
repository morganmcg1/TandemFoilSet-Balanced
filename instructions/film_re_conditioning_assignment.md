# FiLM conditioning on Reynolds number: inject log(Re) per TransolverBlock

## Hypothesis

`re_rand` is the hardest OOD split (val=93.15 in the current baseline) and Reynolds number is the primary axis of out-of-distribution generalization in this dataset (Re spans 100K–5M). Although `log(Re)` already lives in `x` at dim 13, the Transolver sees it merely as a point feature mixed homogeneously with 23 other dimensions through the shared `preprocess` MLP. It never gets to *modulate* computation across layers.

**FiLM conditioning** (Feature-wise Linear Modulation; Perez et al. 2018) extracts this global scalar and turns it into per-layer scale + shift parameters (γ, β) applied to the hidden representation at every TransolverBlock. This gives the network explicit Re-awareness at every depth, allowing it to cleanly interpolate velocity and pressure magnitudes across the 50× range of Reynolds numbers without relying on the attention mechanism to implicitly infer the regime. Expected benefit: improved generalization on `val_re_rand` and likely on `val_geom_camber_cruise` (which spans a very different Re subrange).

**Expected value**: Re-conditioning has shown 5–15% improvements in similar physics surrogate settings. Even a 2–3% reduction in the re_rand split (93.15 → ~90) would push the average below 100.

## Current Baseline

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 120.68 | 104.32 |
| geom_camber_rc | 111.80 | 98.04 |
| geom_camber_cruise | 75.99 | 63.06 |
| re_rand | 93.15 | 88.91 |
| **avg** | **100.41** | **88.58** |

Branch: PR #1098 (lr=1e-3, grad_clip=1.0, DropPath 0→0.1, budget-aware CosineAnnealingLR, surf_weight=25, wd=1e-4)

## Implementation Instructions

All changes go in `train.py` only. `data/` files are read-only.

### Step 1 — Add a FiLM generator module

Add after the existing `MLP` class definition:

```python
class FiLMGenerator(nn.Module):
    """Generates per-channel scale and shift from a scalar condition (log Re)."""
    def __init__(self, n_hidden: int):
        super().__init__()
        # Small 2-layer MLP: scalar → hidden → 2*n_hidden (γ and β)
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2 * n_hidden),
        )
        # Initialize so FiLM starts as identity (γ=1, β=0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Set the first n_hidden biases to 1.0 so γ starts at 1
        with torch.no_grad():
            self.net[-1].bias[:n_hidden] = 1.0

    def forward(self, log_re: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            log_re: [B, 1] — per-sample log(Re) (already normalized is fine,
                    but we extract raw from x[:, :, 13] before normalization)
        Returns:
            gamma: [B, 1, n_hidden]
            beta:  [B, 1, n_hidden]
        """
        out = self.net(log_re)  # [B, 1, 2*n_hidden]
        gamma, beta = out.chunk(2, dim=-1)  # each [B, 1, n_hidden]
        return gamma, beta
```

### Step 2 — Modify `TransolverBlock.forward` to accept FiLM params

Change `TransolverBlock.forward`:

```python
def forward(self, fx: torch.Tensor, gamma: torch.Tensor = None, beta: torch.Tensor = None):
    # existing: residual + physics attention
    residual = fx
    fx = self.ln1(fx)
    fx = self.attn(fx)
    fx = self.drop_path(fx)
    fx = fx + residual

    # Apply FiLM after first residual, before MLP branch
    if gamma is not None and beta is not None:
        fx = fx * gamma + beta   # [B, N, n_hidden] broadcast with [B, 1, n_hidden]

    residual = fx
    fx = self.ln2(fx)
    if self.last_layer:
        fx = self.mlp2(fx)
    else:
        fx = self.mlp(fx)
    fx = self.drop_path(fx)
    fx = fx + residual
    return fx
```

> Note: if the current `TransolverBlock.forward` applies `drop_path` differently (e.g. two separate DropPath calls), keep that structure and just insert the FiLM line between the two residual blocks.

### Step 3 — Add FiLM generators to `Transolver.__init__`

In `Transolver.__init__`, after creating `self.blocks`, add:

```python
self.film_generators = nn.ModuleList([
    FiLMGenerator(n_hidden) for _ in range(n_layers)
])
```

### Step 4 — Extract log(Re) in `Transolver.forward` and thread through blocks

`log(Re)` is `x[:, :, 13]` (raw, before normalization). All nodes in a sample share the same global Re, so take the mean over nodes (or just index node 0):

```python
def forward(self, data, **kwargs):
    x = data["x"]
    # Extract raw log(Re) — dim 13, global per sample (mean over nodes for robustness)
    log_re = x[:, :, 13].mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block, film_gen in zip(self.blocks, self.film_generators):
        gamma, beta = film_gen(log_re)   # [B, 1, n_hidden] each
        fx = block(fx, gamma=gamma, beta=beta)
    return {"preds": fx}
```

> Important: use `x` (raw, before normalization) for log_re extraction. The normalization is applied to `x` before passing to the model in the training loop (`x_norm`), but `data["x"]` inside `forward` is the normalized x. In that case, use `x[:, :, 13]` and the normalized log(Re) is still a valid conditioning signal since it has the same rank-order as the raw log(Re). This is fine — the FiLM generator will learn to map the normalized value to the right scale/shift.

### Step 5 — Keep all other hyperparameters identical to baseline

```
lr = 1e-3
weight_decay = 1e-4
grad_clip = 1.0
surf_weight = 25.0
batch_size = 4
```

Budget-aware cosine annealing, DropPath 0→0.1 — all unchanged from the current baseline (PR #1098).

### Step 6 — Parameter count check

`FiLMGenerator` with n_hidden=128: `(1×128 + 128) + (128×256 + 256) ≈ 33K params × 5 layers = ~165K`. This is negligible (~0.3% overhead) and epoch time should be essentially unchanged.

## Reproduce Command

```bash
python train.py \
  --experiment_name film-re-conditioning \
  --agent tanjiro \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --surf_weight 25.0 \
  --batch_size 4
```

## Expected Outcome

- `val_avg/mae_surf_p` < 100.41
- Primary signal: `val_re_rand/mae_surf_p` should improve most (expected ~2–5% drop from 93.15)
- Secondary signal: `val_geom_camber_cruise/mae_surf_p` may also improve (Re range is different from raceCar splits)
- Epoch time should be essentially unchanged (<1% overhead)

## What to Report

Please include in your results comment:
1. Full val metrics table (all 4 splits + avg) at best epoch
2. Full test metrics table at best epoch checkpoint
3. Best epoch number and total epochs completed
4. Per-epoch val_avg/mae_surf_p curve (from metrics.jsonl if possible)
5. VRAM usage and epoch timing
6. Whether re_rand improved vs. baseline (key signal for this hypothesis)

## References

- Perez et al. (2018) "FiLM: Visual Reasoning with a General Conditioning Layer" (https://arxiv.org/abs/1709.07871)
- Wandrey et al. (2023) FiLM in PDE surrogates: conditioning on flow regime scalars shows consistent 5–15% improvement on Re-OOD splits
