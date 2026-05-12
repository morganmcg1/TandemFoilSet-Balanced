# Research Hypotheses — Round 2
**Date**: 2026-05-12
**Baseline**: val_avg/mae_surf_p = 96.5587 (PR #1518, higher-lr-cosine-14)
**Config at baseline**: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (662K params), AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR(T_max=14), grad_clip=1.0, batch_size=4, surf_weight=10.0, MSE loss in normalized space.

**WIP PRs that must not be duplicated**:
- #1456 alphonse/bf16-amp
- #1457 askeladd/surf-weight-50
- #1458 edward/wider-deeper
- #1460 fern/relative-l2-loss
- #1462 frieren/warmup-cosine
- #1467 nezuko/more-slices-128
- #1473 tanjiro/huber-loss

---

## Causal analysis before proposing

The convergence trace reveals several independent bottlenecks that have not been attacked:

1. **Gradient conflict between vol_loss and surf_loss**: grad_clip fires on 100% of batches with pre-clip max norms 288–740. This is not purely magnitude instability — competing gradient directions from the two loss terms cause the optimizer to thrash. Huber loss (WIP) addresses magnitude; it does not address direction conflict.

2. **Output scale heterogeneity across Re**: Re ranges from 100K to 5M within every split. The target magnitudes vary by an order of magnitude with Re. The model must predict small-Re (low amplitude) and high-Re (large amplitude) flows with the same output head, using the same `y_std` normalization. This is the principal source of hard OOD error on val_re_rand and val_single_in_dist.

3. **Optimizer second-moment mismatch**: AdamW adapts per-parameter, but the gradient distribution in PhysicsAttention has structured correlations across the slice projection weights. Quasi-Newton or preconditioned methods that estimate the gradient covariance can navigate this more efficiently.

4. **Attention temperature saturation**: `temperature = nn.Parameter(init=0.5)` is fixed and likely plateaus early, causing all physics-slice assignments to concentrate into a few slices and neglect the rest. Dynamic annealing from coarse to fine assignments may unlock capacity the current architecture is underusing.

5. **Loss channel imbalance**: Current MSE mixes Ux, Uy, p into a single scalar. The primary metric is surface pressure MAE. The p channel may be underweighted within the loss relative to its contribution to the metric.

6. **Schedule may not have reached bottom**: val was still falling at epoch 14 (100.34 → 98.66 → 96.56). T_max=14 with eta_min=0 means the LR hit 0 at epoch 14. A short cosine tail extended to T_max=20 with a small eta_min floor (1e-5) may extract the remaining gradient signal without disrupting stable convergence.

---

## Ranked Hypotheses

### Priority 1 (HIGH) — Expected > 5% relative improvement, well-supported mechanism

---

### H1 — `soap-optimizer`
**Priority**: HIGH
**Slug**: `soap-optimizer`

**Hypothesis and mechanism**

Replace AdamW with the SOAP (Shampoo As Adam Preconditioner) optimizer (Wang et al., 2025; arxiv 2502.00604). SOAP maintains layer-wise gradient covariance matrices, periodically eigendecomposes them, and applies the resulting curvature-aware preconditioner to each update step. Unlike diagonal Adam, SOAP resolves both:
- **Type I conflicts**: same-direction gradients with very different magnitudes (Adam over-adapts the small one, under-adapts the large one)
- **Type II conflicts**: opposing-direction gradients that cancel in the full-batch but fire alternately in mini-batches

These are exactly the dynamics observed here: gradient norms span 23–740 (Type I), and vol_loss vs. surf_loss produce competing gradient directions (Type II). SOAP on PDE benchmarks showed 2–10× improvement over Adam on turbulent Navier-Stokes and related tasks; the mechanism maps directly to our instability pattern.

**Specific implementation**

1. Add `soap-optimizer` package to `pyproject.toml`: `soap-optimizer>=0.1` (PyPI package `soap-optimizer`).

2. In `train.py`, replace:
   ```python
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=cfg.lr,
       weight_decay=cfg.weight_decay
   )
   ```
   with:
   ```python
   from soap import SOAP
   optimizer = SOAP(
       model.parameters(),
       lr=cfg.lr,           # keep 1e-3 — SOAP uses same LR scale as AdamW
       weight_decay=cfg.weight_decay,
       betas=(0.99, 0.99),  # β₁=β₂=0.99 recommended in Wang et al.
       precondition_frequency=10,  # update preconditioner every 10 steps
       max_precond_dim=512,        # cap eigendecomposition for large layers
   )
   ```

3. Keep `CosineAnnealingLR(T_max=14)` and `grad_clip=1.0` unchanged.

**Expected benefit**: 5–15% relative reduction in val_avg/mae_surf_p. If AdamW is leaving curvature unexploited due to diagonal second-moment mismatch, SOAP should close 30–60% of the remaining gap.

**Risk**: SOAP is more memory-intensive (stores covariance matrices per layer). At 662K params the model is small — peak VRAM should remain well under 96 GB. Main risk is that `precondition_frequency=10` may be too aggressive or too conservative; try 5 and 20 as fallback.

**Falsifying result**: If val_avg/mae_surf_p is within 1% of 96.56 at epoch 14, the bottleneck is not optimizer curvature — it is architecture or data representation.

---

### H2 — `re-conditioned-scaling`
**Priority**: HIGH
**Slug**: `re-conditioned-scaling`

**Hypothesis and mechanism**

The single largest source of hard error in OOD splits is the order-of-magnitude variation in output scale with Reynolds number (Re). At low Re (100K), pressure magnitudes are O(10–100); at high Re (5M), they are O(1000–10000). The global normalization `(y - y_mean)/y_std` compresses this into a narrow band, but the residual scale variation is learned implicitly via the input feature `log(Re)` (dim 13). This learning is imperfect: the model must simultaneously solve the scale estimation and the shape estimation problem through the same attention stack.

Decompose these: let the model predict the **shape** of the solution in a scale-free space, and apply a learned Re-conditioned rescaling **after** the model output. This is analogous to DimINO's Redimensionalization (Huang et al., 2024; arxiv 2410.05894): extract a characteristic scale from the problem parameters and multiply the model output by it.

Specifically: train an additional 2-layer MLP (`ReScaleHead`) that takes `log_Re` (a scalar per sample) as input and predicts a 3-vector `scale_factor` (one per output channel). Multiply `pred * scale_factor` before loss and metric computation. Initialize `scale_factor` to all-ones (identity at training start). The main model learns scale-free patterns; the head learns to rescale them into the correct magnitude.

**Specific implementation**

In `train.py`, after the `Transolver` class definition, add:

```python
class ReScaleHead(nn.Module):
    def __init__(self, hidden=32, out_channels=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )
        # Initialize to identity: softplus(0) ≈ 0.693, but we want 1.0 at init
        # Use bias init: Linear(1→32) default zeros bias is fine
        # final layer bias init to log(1) = 0.0 with softplus ensures output → 1.0
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            self.mlp[-1].weight.zero_()  # output starts at 0 → softplus(0)≈0.693
            # Adjust: use sigmoid+1 or just init bias to log(e-1)≈0.541 → softplus=1.0
            self.mlp[-1].bias.fill_(0.541)  # softplus(0.541) ≈ 1.0

    def forward(self, log_re):
        # log_re: [B, 1] — per-sample log Reynolds number
        # returns: [B, 1, 3] scale factors (broadcast over N nodes)
        raw = self.mlp(log_re)  # [B, 3]
        scale = torch.nn.functional.softplus(raw)  # positive, init≈1.0
        return scale.unsqueeze(1)  # [B, 1, 3]
```

In the training loop, extract `log_re` from the input batch (dim 13 is `log(Re)` from `program.md`):

```python
# After normalization, before model call:
log_re_norm = x_norm[:, 0, 13:14]  # [B, 1] — same across all nodes in a sample
scale = rescale_head(log_re_norm)   # [B, 1, 3]
pred = model({"x": x_norm})["preds"] * scale  # [B, N, 3]
```

Add `rescale_head = ReScaleHead().to(device)` and include its parameters in the optimizer. Keep all other config unchanged.

**Expected benefit**: 5–12% relative reduction. Val_re_rand and val_single_in_dist are the hardest splits (val 92.71 and 108.58); both span the full Re range. This directly targets the scale heterogeneity bottleneck.

**Risk**: The `log_re_norm` extracted from dim 13 is the already-normalized `log(Re)` — it still encodes the Re ordering. If the normalization scrambles the ordering, use `x[:, 0, 13:14]` (raw) instead. The MLP is tiny (32 hidden) so VRAM and compute overhead are negligible.

**Falsifying result**: If `scale` converges to near 1.0 for all Re values at end of training, the model already solves scale estimation implicitly — this head adds nothing.

---

### H3 — `pcgrad-surgery`
**Priority**: HIGH
**Slug**: `pcgrad-surgery`

**Hypothesis and mechanism**

The root cause of 100% gradient clipping is not just magnitude — it is gradient conflict between vol_loss and surf_loss. PCGrad (Yu et al., NeurIPS 2020) surgically removes the conflicting component of one task's gradient before accumulation: for two loss terms A and B, if `g_A · g_B < 0` (opposing directions), project `g_A` onto the plane orthogonal to `g_B` before adding them. This eliminates destructive interference without suppressing either loss term's signal.

The mechanism is directly motivated by the observed instability: clipping fires on 100% of batches with max norms 740×. Gradient surgery would reduce effective step magnitude only for conflicting directions, preserving the aligned component intact. This is strictly better than isotropic norm clipping for the two-task setting.

**Specific implementation**

Implement inline in `train.py` (no new package). After computing `vol_loss` and `surf_loss` but before the combined backward:

```python
# ---- PCGrad surgery (replace existing loss.backward()) ----
optimizer.zero_grad()

# Compute per-task gradients separately
vol_loss.backward(retain_graph=True)
grads_vol = [p.grad.clone() if p.grad is not None else None
             for p in model.parameters()]
optimizer.zero_grad()

surf_loss.backward()
grads_surf = [p.grad.clone() if p.grad is not None else None
              for p in model.parameters()]
optimizer.zero_grad()

# Project: remove component of g_vol that conflicts with g_surf, and vice versa
def _project(g_a, g_b):
    if g_a is None or g_b is None:
        return g_a
    dot = (g_a * g_b).sum()
    if dot < 0:
        # g_a -= (dot / ||g_b||^2) * g_b
        g_a = g_a - (dot / (g_b.norm() ** 2 + 1e-12)) * g_b
    return g_a

for p, gv, gs in zip(model.parameters(), grads_vol, grads_surf):
    if gv is None and gs is None:
        continue
    gv_proj = _project(gv, gs)
    gs_proj = _project(gs, gv)
    p.grad = (gv_proj if gv_proj is not None else torch.zeros_like(p))
    if gs_proj is not None:
        p.grad = p.grad + cfg.surf_weight * gs_proj

# Apply existing grad clip on the combined projected gradient
if cfg.grad_clip:
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

optimizer.step()
# ---- end PCGrad surgery ----
```

Note: retain_graph=True on vol_loss is needed because surf_loss shares the same computation graph. Keep `cfg.surf_weight` as the relative weighting after projection.

**Expected benefit**: 5–10% relative reduction. If gradient conflicts are the dominant cause of the 100% clip rate, PCGrad should reduce effective noise and allow the model to descend more efficiently. The secondary benefit is that grad_clip may fire less often, allowing larger effective steps on aligned directions.

**Risk**: The per-parameter projection loop adds compute overhead (roughly 2× backward pass time). At 662K params this is tolerable but will slow down epochs. Monitor whether epochs stay within the 30-min budget. If too slow, replace with layer-grouped projection (project per-layer norm vectors rather than per-parameter).

**Falsifying result**: If val_avg/mae_surf_p is within 1% of 96.56, gradient conflict direction (not magnitude) was not the bottleneck — clipping fires for magnitude reasons only, and PCGrad does not help.

---

### Priority 2 (MEDIUM) — Expected 2–5% relative improvement, good mechanism

---

### H4 — `sgdr-restarts`
**Priority**: MEDIUM
**Slug**: `sgdr-restarts`

**Hypothesis and mechanism**

The current CosineAnnealingLR(T_max=14) has a single monotone decay from lr=1e-3 to lr=0 over 14 epochs. Val was still improving at epoch 14. The LR hitting 0 exactly at the final epoch risks under-training in the late regime.

SGDR (Loshchilov & Hutter, ICLR 2017) uses cosine annealing with warm restarts: LR resets periodically from a high value to force the model to escape local minima and explore the loss surface more broadly. With T_0=7 and T_mult=1, training cycles twice over 14 epochs: a full cosine cycle from epoch 0-7, restart, and a second cycle from epoch 7-14. This is distinct from WIP #1462 (warmup-cosine), which is a monotone schedule with a linear warmup — no restarts.

The restarts act as implicit annealed averaging: each half-cycle explores a basin, and the second cycle refines the basin found after the restart. This has been empirically reliable on vision and NLP tasks and is particularly effective when training runs for a fixed, short budget.

**Specific implementation**

In `train.py`, replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14)
```
with:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=7,       # half of 14-epoch budget per cycle
    T_mult=1,    # equal-length cycles
    eta_min=1e-5 # small floor to avoid LR hitting exact 0
)
```

No other changes needed. Keep `grad_clip=1.0` and `lr=1e-3`.

**Expected benefit**: 2–5% relative reduction. If the loss surface has multiple nearby basins (likely given multi-domain data), the restart mechanism should help escape the basin found by the first half-cycle. The `eta_min=1e-5` floor also prevents hard shutdown at epoch 14 — val was still falling.

**Risk**: If the loss surface is unimodal, restarts waste cycles re-climbing from high LR. The main downside is the epoch 7 LR reset may temporarily spike val loss before recovery. Monitor epoch 7–8 trajectory.

**Falsifying result**: If val_avg at epoch 7 is markedly worse than at epoch 14 of the monotone baseline, restarts are disrupting productive fine-tuning.

---

### H5 — `lion-optimizer`
**Priority**: MEDIUM
**Slug**: `lion-optimizer`

**Hypothesis and mechanism**

Lion (Chen et al., ICLR 2024) is a sign-based momentum optimizer: updates are `sign(momentum * beta1 + grad * (1 - beta1))`, then momentum is updated with `momentum * beta2 + grad * (1 - beta2)`. It only tracks first-moment (no second moment), making it memory-efficient compared to AdamW, and it applies uniform step sizes (sign) rather than adaptive magnitudes.

Held back from Round 1 because of LR sensitivity (Round 1 notes: "Held back for round 2 — LR sensitivity"). With the current baseline now well-tuned at lr=1e-3 with grad_clip=1.0, we have better calibration for setting Lion's LR correctly. Lion requires 3–5× lower LR than AdamW in practice due to the unit step size magnitude.

**Specific implementation**

Implement Lion inline in `train.py` (no new package needed — 20-line class):

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                beta1, beta2 = group['betas']
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                # Weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                # Update step
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(update.sign_(), alpha=-group['lr'])
                # Update momentum (no in-place to avoid graph issues)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
```

Replace optimizer instantiation in `train.py`:
```python
optimizer = Lion(
    model.parameters(),
    lr=2e-4,          # 5× lower than AdamW's 1e-3 — standard Lion rule of thumb
    betas=(0.9, 0.99),
    weight_decay=cfg.weight_decay,
)
```

Keep `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`.

**Expected benefit**: 2–5% relative reduction. Sign-based updates are immune to gradient magnitude variation — directly addresses the 23-740 norm spread without needing per-parameter adaptive scaling.

**Risk**: LR of 2e-4 may be too conservative or too aggressive. If val_avg at epoch 5 is worse than 120, restart at 3e-4. If training diverges, fall back to 1e-4.

**Falsifying result**: If val_avg at epoch 14 is within 2% of AdamW baseline, the optimizer choice is not the bottleneck at this scale — architecture or data representation matters more.

---

### H6 — `per-channel-loss-weights`
**Priority**: MEDIUM
**Slug**: `per-channel-loss-weights`

**Hypothesis and mechanism**

The primary ranking metric is surface pressure MAE (`mae_surf_p`), but the current MSE loss treats all three output channels (Ux, Uy, p) equally. Ux and Uy velocity fields are predicted alongside pressure, but errors in velocity do not count toward the paper-facing metric. If Ux/Uy errors dominate the loss (they may, since velocity fluctuations are large in inverted-foil flow), the optimizer wastes capacity on channels that don't affect the metric.

Add a per-channel loss weight vector: `loss_weights = [w_ux, w_uy, w_p]` that up-weights the pressure channel. Start with a simple hand-tuned ratio (`w_p=2.0, w_ux=w_uy=0.5`), which doubles the pressure gradient signal relative to the symmetric baseline.

Note: this is distinct from `surf_weight` (which reweights by node type: surface vs. volume). This is a **channel reweighting** within the surface loss term.

**Specific implementation**

In `train.py`, add a `loss_weights` config value:
```python
@dataclass
class Config:
    ...
    # Per-channel loss weights [Ux, Uy, p]
    loss_weights: list = field(default_factory=lambda: [0.5, 0.5, 2.0])
```

In the loss computation block, replace:
```python
sq_err = (pred - y_norm) ** 2  # [B, N, 3]
```
with:
```python
lw = torch.tensor(cfg.loss_weights, device=pred.device)  # [3]
sq_err = ((pred - y_norm) ** 2) * lw  # [B, N, 3] — broadcast over nodes
```

All subsequent vol_loss / surf_loss averaging is unchanged.

**Expected benefit**: 3–7% relative reduction in val_avg/mae_surf_p. By allocating more gradient signal to the p channel, the model should fit pressure more accurately at the expense of slightly worse velocity reconstruction.

**Risk**: If Ux/Uy errors propagate to p through the model's physics (they are coupled in the NS equations), down-weighting velocity may hurt pressure accuracy indirectly. Start with `[0.5, 0.5, 2.0]` and report per-channel val_surf metrics to diagnose coupling.

**Falsifying result**: If mae_surf_ux and mae_surf_uy improve without improvement to mae_surf_p, velocity and pressure errors are decoupled in this architecture and channel weighting does not propagate.

---

### H7 — `attention-temperature-anneal`
**Priority**: MEDIUM
**Slug**: `attention-temperature-anneal`

**Hypothesis and mechanism**

`PhysicsAttention` uses a learned temperature parameter (`nn.Parameter(init=0.5)` per head) that controls the sharpness of physics-slice assignments. Low temperature → sharp (one slice per node); high temperature → soft (many slices). The current fixed initialization at 0.5 may not be optimal — early in training, soft assignments let the model explore which slices matter; later, sharper assignments reduce noise.

Explicitly anneal temperature from high (1.0) to low (0.1) over training using a cosine schedule, decoupling temperature from learned optimization. This is analogous to simulated annealing in combinatorial optimization and curriculum learning in ML: start with broad exploration, sharpen to exploitation.

**Specific implementation**

In `train.py`, add a temperature annealing callback in the training loop. After the PhysicsAttention definition (or as a monkey-patch), expose temperature parameters:

```python
def set_physics_attention_temperature(model, temp):
    """Set temperature in all PhysicsAttention modules."""
    for module in model.modules():
        if hasattr(module, 'temperature'):
            with torch.no_grad():
                module.temperature.fill_(temp)

# In training loop, before each epoch:
temp_high, temp_low = 1.0, 0.1
temp = temp_high - (temp_high - temp_low) * (epoch / (cfg.epochs - 1))
set_physics_attention_temperature(model, temp)
```

This overrides the learned temperature with a scheduled value. Keep all other config unchanged.

**Expected benefit**: 2–4% relative reduction. If the model is stuck in suboptimal slice assignments (a few slices concentrating most of the attention mass), annealing from soft to sharp should improve coverage and reduce redundancy.

**Risk**: Overriding the learned temperature prevents the model from adapting temperature via gradient descent. A cleaner approach is to add a temperature loss term that encourages the learned temperature to follow the schedule (soft constraint). However, the override is simpler and faster to implement for a screening run.

**Falsifying result**: If slice assignment entropy (log of softmax diversity across slices) does not change materially between epoch 1 and epoch 14, temperature was not a binding constraint.

---

### Priority 3 (LOW) — Exploratory, worth trying after high/medium confirmed

---

### H8 — `perchannel-heads`
**Priority**: LOW
**Slug**: `perchannel-heads`

**Hypothesis and mechanism**

The current Transolver decoder maps from the model's internal representation to all 3 output channels (Ux, Uy, p) through a shared MLP. Each channel has different physics: velocity components are governed by the momentum equations; pressure is governed by the divergence-free constraint. Separate per-channel decoder heads may allow the model to learn channel-specific representations without interference.

Held back from Round 1 (explicitly noted: "Held back for round 2"). Add three independent MLP heads (one per output channel), replacing the single final linear decoder. Each head has 2 layers with 64 hidden dims.

**Specific implementation**

In `Transolver.forward()`, replace the final projection:
```python
# Before: single head
self.out_proj = nn.Linear(n_hidden, 3)
out = self.out_proj(x)  # [B, N, 3]
```
with:
```python
# After: three separate heads
self.out_heads = nn.ModuleList([
    nn.Sequential(nn.Linear(n_hidden, 64), nn.GELU(), nn.Linear(64, 1))
    for _ in range(3)  # Ux, Uy, p
])
out = torch.cat([h(x) for h in self.out_heads], dim=-1)  # [B, N, 3]
```

**Expected benefit**: 1–3% relative reduction. The mechanism is plausible but modest — the shared decoder is a small fraction of total model parameters. May compound well with `per-channel-loss-weights`.

**Risk**: Adds ~25K params (tiny). No VRAM concern. Main risk is that the shared representation already encodes channel-specific information implicitly and separate heads offer no additional capacity.

---

### H9 — `silu-activation`
**Priority**: LOW
**Slug**: `silu-activation`

**Hypothesis and mechanism**

The current MLP blocks use GELU activation. SiLU (Swish) is `x * sigmoid(x)`, which is smoother and often performs better in physics surrogate settings due to its non-monotone curvature properties. This is the second smallest change in the hypothesis list — one activation swap throughout.

Held back from Round 1 (explicitly noted: "too small to justify a full slot"). With Round 2 having strong hypotheses, this is a background validation that answers: "does activation choice matter here?"

**Specific implementation**

In `MLP.__init__()` in `train.py`, change the default:
```python
# Before
def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):

# After: pass act="silu" at instantiation site, or change the default
```

In the `Transolver` init, when creating MLP blocks, pass `act="silu"`. Alternatively, in `MLP.forward()`, change the activation lookup:
```python
# After (in MLP forward, wherever act is applied):
acts = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}
act_fn = acts[self.act]
```

Then set all Transolver MLP instantiations to `act="silu"`.

**Expected benefit**: 0–2% relative reduction. Low but essentially free — zero additional parameters or VRAM.

**Risk**: None material. If SiLU diverges or performs worse, revert immediately.

---

### H10 — `extended-cosine-tail`
**Priority**: LOW
**Slug**: `extended-cosine-tail`

**Hypothesis and mechanism**

Val was still falling at epoch 14 (100.34 → 98.66 → 96.56) and the LR hit exactly 0 at T_max=14. Extending T_max slightly to 20 with a small eta_min floor (1e-5) lets the schedule decay more slowly through the final training epochs. Within the 30-min budget, the model completes ~14 epochs — but epochs 10–14 are currently at near-zero LR and still improving. A gentler tail keeps gradient signal alive longer.

**Specific implementation**

In `train.py`, replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14)
```
with:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20, eta_min=1e-5
)
```

No other changes needed. The model will run ~14 epochs in the 30-min cap; the schedule will reach roughly the midpoint of its cosine decay rather than hitting 0.

**Expected benefit**: 1–3% relative reduction. Low-cost diagnostic — answers whether the schedule hitting 0 at epoch 14 was causing under-training in the final epochs.

**Risk**: LR may be too high in the final epochs (still in middle of cosine curve), potentially preventing final convergence. Monitor epoch 12–14 val trajectory.

---

## Summary Table

| Rank | Slug | Priority | Mechanism | Expected Gain | New package? |
|------|------|----------|-----------|---------------|--------------|
| 1 | `soap-optimizer` | HIGH | Curvature-aware quasi-Newton preconditioner | 5–15% | yes: `soap-optimizer` |
| 2 | `re-conditioned-scaling` | HIGH | Re-dependent output scale head | 5–12% | no |
| 3 | `pcgrad-surgery` | HIGH | Per-task gradient conflict removal | 5–10% | no |
| 4 | `sgdr-restarts` | MEDIUM | Cosine restarts, LR floor at eta_min=1e-5 | 2–5% | no |
| 5 | `lion-optimizer` | MEDIUM | Sign-based uniform updates, held back Round 1 | 2–5% | no |
| 6 | `per-channel-loss-weights` | MEDIUM | Up-weight pressure channel in loss | 3–7% | no |
| 7 | `attention-temperature-anneal` | MEDIUM | Soft→sharp physics-slice annealing | 2–4% | no |
| 8 | `perchannel-heads` | LOW | Independent decoders per output channel | 1–3% | no |
| 9 | `silu-activation` | LOW | Swap GELU → SiLU throughout MLPs | 0–2% | no |
| 10 | `extended-cosine-tail` | LOW | T_max=20, eta_min=1e-5 | 1–3% | no |

**Recommended assignment order**: Assign H1 (`soap-optimizer`), H2 (`re-conditioned-scaling`), H3 (`pcgrad-surgery`) immediately — these target the three identified bottlenecks with the strongest external evidence. Follow with H4–H7 as Round 2 confirmations after Round 2a results are in.

**Interaction notes**: H1 + H3 likely to compound (SOAP resolves curvature; PCGrad resolves conflict direction — independent mechanisms). H2 + H6 may compound (Re scaling addresses output magnitude; channel weighting addresses gradient allocation). H5 (Lion) is somewhat redundant with H1 (SOAP) — run only one optimizer replacement at a time.
