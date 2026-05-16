## Hypothesis

**H49: Lion optimizer — sign-based gradient updates as implicit per-parameter normalization.**

Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms") replaces AdamW's gradient/second-moment scaling with element-wise `sign()`. The update is:

```
update = sign( β₁ · m_{t-1} + (1 - β₁) · g_t )
θ_{t+1} = θ_t - lr · ( update + wd · θ_t )      # decoupled wd
m_t = β₂ · m_{t-1} + (1 - β₂) · g_t              # momentum buffer
```

Lion tracks only **one** momentum buffer (vs AdamW's two), saving ~50% optimizer memory. The `sign()` in the update removes magnitude information entirely — every parameter receives an update of magnitude exactly `lr`.

**Why this might help in CFD surrogate context:**
1. **Mixed-Re gradient imbalance.** High-Re samples in our dataset produce gradients ~10× larger than low-Re samples. AdamW's per-parameter EMA scaling helps but doesn't eliminate this. Lion's `sign()` forces every gradient component to contribute equally regardless of magnitude — high-Re samples can't dominate weight updates.
2. **Channel imbalance.** Even with per-channel Huber (δ_p=0.25/δ_vel=0.5), pressure gradients are still 2-3× larger than velocity. Sign normalization further evens this.
3. **Flatter minima.** Lion has been reported to find flatter minima than AdamW (Chen et al. 2023). For our heavily-regularized 891K model on small data, flatter minima generalize better — directly relevant to our OOD splits.
4. **Smaller LR required.** Lion typically uses lr ~3-10× smaller than AdamW because every update has magnitude=lr (not lr × |g|). Standard convention: AdamW lr=1e-3 → Lion lr=1e-4 to 3e-4.

**Two arms test LR + wd combinations:**

- **Arm A — Lion lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99**: Standard Lion config from Chen et al. Conservative LR. wd 10× higher than AdamW (standard for Lion, since the sign-clipped updates make wd act differently).
- **Arm B — Lion lr=3e-4, wd=3e-4, β₁=0.95, β₂=0.99**: Higher LR with slightly higher first-moment momentum. Tests whether Lion at higher LR can match AdamW lr=1e-3 throughput in 14 epochs.

If Arm A > Arm B, conservative LR is required. If Arm B > Arm A, Lion needs higher effective step size to compete with AdamW at our budget.

## Instructions

You will need to add the Lion optimizer to `train.py`. Two implementation options — **pick whichever is faster**:

**Option 1 (preferred): pip-installable**
```bash
pip install lion-pytorch
```
Then in train.py:
```python
from lion_pytorch import Lion
# In optimizer construction block:
optimizer = Lion(model.parameters(), lr=cfg.lr,
                 betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
```

**Option 2 (no extra dependency): self-implement** (~25 lines)
```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']
                p.mul_(1 - lr * wd)                            # decoupled wd
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1 - beta2)         # update buffer
        return loss
```

**Add Config flags** (after `clip_grad_norm` line ~404):
```python
optimizer: str = "adamw"  # 'adamw' (default) or 'lion'
beta1: float = 0.9        # first moment coefficient (Adam: 0.9, Lion: 0.9)
beta2: float = 0.999      # second moment coefficient (Adam: 0.999, Lion: 0.99)
```
And CLI flags: `--optimizer`, `--beta1`, `--beta2`.

**Wire into the optimizer construction block** (around line ~440):
```python
if cfg.optimizer == "lion":
    optimizer = Lion(model.parameters(), lr=cfg.lr,
                     betas=(cfg.beta1, cfg.beta2),
                     weight_decay=cfg.weight_decay)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   betas=(cfg.beta1, cfg.beta2),
                                   weight_decay=cfg.weight_decay)
```

Default `optimizer="adamw"` reproduces current behavior exactly.

**Arm A — Lion lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h49-lion-lr1e4-wd1e3 \
  --agent charliepai2i48h3-edward \
  --optimizer lion --lr 1e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --n_head 2 --clip_grad_norm 1.0
```

**Arm B — Lion lr=3e-4, wd=3e-4, β₁=0.95, β₂=0.99:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h49-lion-lr3e4-wd3e4 \
  --agent charliepai2i48h3-edward \
  --optimizer lion --lr 3e-4 --weight_decay 3e-4 \
  --beta1 0.95 --beta2 0.99 \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- Per-epoch val_avg trajectory (especially: does Lion converge slower/faster than AdamW?)
- Total parameter count (should be unchanged; only optimizer state differs)
- **Peak GPU memory** — expected to DROP vs AdamW because Lion stores one moment buffer vs Adam's two

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if Arm A diverges:** if val_avg at epoch 3 exceeds 200, kill and try lr=5e-5 instead. Lion is highly LR-sensitive; this is the first thing to check.

## Baseline

Current best — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **66.1060** |
| val_single_in_dist/mae_surf_p | 74.3956 |
| val_geom_camber_rc/mae_surf_p | 78.9959 |
| val_geom_camber_cruise/mae_surf_p | 46.4384 |
| val_re_rand/mae_surf_p | 64.5940 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **64.4522** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), **AdamW (β₁=0.9, β₂=0.999)**.

**Beat this: val_avg/mae_surf_p < 66.1060**

This is an **optimizer-tier swap** — separate from any hyperparameter work currently running. If Lion wins, it becomes a candidate for further stacking with our other levers. If both arms regress beyond 68, close immediately — regression tasks are a known weak point for sign-based optimizers, and we shouldn't burn another arm.

Predicted val_avg if Lion mechanism applies cleanly to CFD regression ≈ 64-67. High variance.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h37b-nhead2-lr1e3-clip1 \
  --agent charliepai2i48h3-edward \
  --n_head 2 --lr 1e-3 --clip_grad_norm 1.0
```
