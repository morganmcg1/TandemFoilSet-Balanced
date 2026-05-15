# Hypothesis: lion-stacked (fern)

## Hypothesis
Lion optimizer (sign-based update rule) on the merged Huber loss baseline. In round-3,
fern's Lion experiment achieved val_avg=115.49 against the fresh-slate MSE baseline (vs
129.99 baseline). That result was measured against a worse baseline — the clean test is
now on Huber. Lion provides uniform-magnitude per-parameter updates regardless of gradient
scale, complementing Huber's gradient-magnitude cap at the loss level. The hypothesis is
that Huber (loss-level stability) + Lion (update-level stability) compound orthogonally.

**Predicted improvement:** −3 to −8 on val_avg/mae_surf_p vs 107.46.

## Instructions

### 1. Implement Lion optimizer

`timm` (already installed) ships with Lion in recent versions. Try:

```python
try:
    from timm.optim import Lion
except ImportError:
    # Minimal Lion implementation (from Chen et al. 2023)
    class Lion(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p.data)
                    exp_avg = state["exp_avg"]
                    beta1, beta2 = group["betas"]
                    update = (beta1 * exp_avg + (1 - beta1) * grad).sign()
                    p.data.add_(update * group["lr"] + p.data * group["lr"] * group["weight_decay"], alpha=-1)
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
```

### 2. Replace the AdamW optimizer (line ~435)

```python
# Replace:
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
# With:
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

Note: Lion typically uses 3–10× smaller lr than AdamW. `lr=1e-4` (vs AdamW's 5e-4).
Lion benefits from higher weight_decay; `1e-2` is typical per the paper.

Keep the scheduler as-is (CosineAnnealingLR with T_max=MAX_EPOCHS).

### 3. Run the experiment

```bash
cd target/ && python train.py \
    --wandb_group lion-stacked \
    --wandb_name lion-lr1e-4-wd1e-2 \
    --agent willowpai2i24h3-fern
```

Try a second arm with `lr=5e-5` if the first run doesn't converge well.

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
