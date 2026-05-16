# Research Ideas — 2026-05-16 11:35

Canonical baseline: val_avg/mae_surf_p = 54.494, test = 52.837
Stack: Transolver (slice_num=64, n_hidden=128, n_layers=5), SOAP (betas=0.95, precond_freq=5), EMA (decay=0.99), Huber (beta=0.5), surf_weight=10.0, lr=1e-3

In-flight (do not duplicate): surf_weight finer sweep, EMA decay sweep, slice_num variants, something on precond_freq, log-space loss on pressure (check before assigning H4 below).

---

## H1: Cosine LR Floor (eta_min)

### Hypothesis
CosineAnnealingLR decays to lr=0 by default. In the final 5–10 epochs, the effective learning rate approaches zero and SOAP's preconditioner matrix is stale relative to an essentially frozen parameter trajectory. Adding a non-zero floor (`eta_min=1e-5`) prevents this stall: the optimizer keeps making gradient steps with a well-conditioned preconditioner through the end of training, and the EMA checkpoint captures a model that has continued improving rather than coasting.

This is a near-zero-risk, 1-line change. It is orthogonal to all in-flight PRs. There is strong empirical precedent (cosine floor is standard in ViT/BERT recipes) and the mechanism is clear: the model should not stop learning before the wall-clock cap.

### Implementation

**File:** `train.py`, line 504

Current:
```python
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1),
)
```

Change to:
```python
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1), eta_min=cfg.lr_floor,
)
```

**File:** `train.py`, `Config` dataclass (lines 404–423), add one field:
```python
lr_floor: float = 0.0  # default preserves existing behavior
```

**CLI invocation:**
```bash
python train.py \
  --optimizer soap \
  --ema_decay 0.99 \
  --precondition_frequency 5 \
  --huber_beta 0.5 \
  --surf_weight 10.0 \
  --lr_floor 1e-5 \
  --wandb_group lr-floor-sweep
```

Try three values in sequence if time allows: `lr_floor=1e-5` (primary), `lr_floor=5e-5`, `lr_floor=1e-4`. The primary arm is the hypothesis test; the others are cheap follow-ups if the first moves the metric.

### Expected gain
1–2%. The decay-to-zero schedule is a known sub-optimality for short runs. For a 14-epoch run, the cosine curve hits near-zero LR around epoch 12–13, meaning ~15% of training epochs operate at essentially zero learning rate.

### Risk
Low. If `lr_floor` is too large, loss plateaus rather than converging; but 1e-5 is 100x below peak (1e-3) and will not cause instability. The default=0.0 preserves exact backward compatibility.

### Why now
Every in-flight PR tests a different dimension (surf_weight, EMA, slice_num). This is the only untested schedule parameter. It is 2 lines of code, orthogonal, and has a clear mechanism.

---

## H2: Lookahead Wrapper on SOAP

### Hypothesis
SOAP approximates the full Shampoo preconditioner via the Kronecker eigenbasis and refreshes it every `precondition_frequency` steps. Between refreshes, curvature information is stale. Lookahead (Zhang et al., 2019) maintains a set of "slow weights" that are periodically synced from the fast optimizer trajectory. It was originally proposed for Adam but has been shown to stabilize and improve convergence for second-order optimizers (K-FAC, Shampoo) because the slow-weight sync averages out the noise introduced by stale preconditioner updates. Effect: lower variance in the effective update direction, better final checkpoint quality, orthogonal to EMA (EMA smooths the checkpoint used for eval; Lookahead smooths the training trajectory itself).

### Implementation

**File:** `train.py` — add a `Lookahead` wrapper class immediately after the `EMAModel` class (after line 74). Total addition: ~25 lines.

```python
class Lookahead:
    """Lookahead optimizer wrapper (Zhang et al., 2019).
    
    Maintains slow weights synced from fast optimizer every `k` steps.
    Compatible with any inner optimizer including SOAP.
    """
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        # Store slow weights as a flat list parallel to param groups
        self.slow_weights = [
            [p.data.clone() for p in group["params"]]
            for group in optimizer.param_groups
        ]

    def step(self):
        self.optimizer.step()
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group_slow, group in zip(self.slow_weights, self.optimizer.param_groups):
                for slow_p, p in zip(group_slow, group["params"]):
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
```

**File:** `train.py`, lines 483–495 — wrap the SOAP instantiation:

```python
# existing SOAP instantiation unchanged:
_soap = SOAP(
    model.parameters(),
    lr=cfg.lr,
    betas=(0.95, 0.95),
    weight_decay=cfg.weight_decay,
    precondition_frequency=cfg.precondition_frequency,
    shampoo_beta=0.95,
    eps=1e-8,
    normalize_grads=False,
)
optimizer = Lookahead(_soap, k=cfg.lookahead_k, alpha=cfg.lookahead_alpha) if cfg.use_lookahead else _soap
```

**File:** `train.py`, `Config` dataclass — add:
```python
use_lookahead: bool = False
lookahead_k: int = 5
lookahead_alpha: float = 0.5
```

**CLI invocation:**
```bash
python train.py \
  --optimizer soap \
  --ema_decay 0.99 \
  --precondition_frequency 5 \
  --huber_beta 0.5 \
  --surf_weight 10.0 \
  --use_lookahead \
  --lookahead_k 5 \
  --lookahead_alpha 0.5 \
  --wandb_group lookahead-soap-sweep
```

If k=5 / alpha=0.5 wins, try k=10 / alpha=0.5 as a follow-up. Do not change both simultaneously.

**Note:** The scheduler wraps the inner SOAP optimizer directly (`optimizer.optimizer` for LR state access). Confirm that `SequentialLR` is called on `_soap`, not on the Lookahead wrapper — or patch the scheduler call to use `optimizer.optimizer` when `cfg.use_lookahead` is True.

### Expected gain
1–3%. Lookahead on SOAP has not been tested in this stack. The preconditioner refresh period creates exactly the stale-curvature noise that Lookahead is designed to absorb. On small models (0.66M params), the slow-weight sync adds negligible compute overhead.

### Risk
Medium. Lookahead interacts with LR scheduling in a subtle way: when slow weights are synced back to fast weights, the optimizer's effective LR behavior changes slightly. The implementation above is correct for standard usage, but ensure the scheduler step is called on the inner optimizer's LR, not via the Lookahead wrapper. Also confirm the EMA update still reads from `model.parameters()` (not slow weights) — EMA and Lookahead are both weight-averaging mechanisms but they operate on different timescales and are both needed.

### Why now
Lookahead has never been tried in this stack. Its mechanism targets the specific known limitation of periodic Kronecker-preconditioner refresh in SOAP. The slow-weight sync is a fundamentally different mechanism from EMA (which operates only on the eval branch).

---

## H3: SAM Perturbation Wrapping SOAP

### Hypothesis
Sharpness-Aware Minimization (Foret et al., 2021) seeks parameters where the loss is flat in all directions, not just low. The canonical formulation adds an ascent step (perturb weights by `rho * grad / ||grad||`) before the descent step. For physics-informed surrogates, flat minima generalize better across OOD splits (val_geom_camber_rc, val_re_rand) because OOD queries fall in regions where the sharpest directions of the loss landscape are most likely to fail. The current best val_avg/mae_surf_p captures an average across 4 splits; SAM's flatness bias should preferentially improve the OOD splits.

### Implementation

**File:** `train.py` — add a `SAMWrapper` class after `EMAModel` (after line 74). Total: ~30 lines.

```python
class SAMWrapper:
    """Sharpness-Aware Minimization wrapper (Foret et al., 2021).
    
    Two-step per batch: (1) ascent step to find worst-case neighbor,
    (2) inner optimizer step from that neighbor back toward flat region.
    Compatible with SOAP as inner optimizer.
    
    rho: perturbation radius. 0.05 is standard for small models.
    """
    def __init__(self, optimizer, rho: float = 0.05):
        self.optimizer = optimizer
        self.rho = rho
        self._saved_grads: list | None = None

    def _ascent_step(self):
        """Perturb weights by rho * g/||g||. Save e_w for later restoration."""
        grads = [
            p.grad.detach()
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        grad_norm = torch.stack([g.norm(2) for g in grads]).norm(2)
        scale = self.rho / (grad_norm + 1e-12)
        self._e_ws = []
        param_iter = (
            p for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        )
        for p, g in zip(param_iter, grads):
            e_w = g * scale
            p.data.add_(e_w)
            self._e_ws.append(e_w)

    def _descent_restore(self):
        """Restore weights to pre-ascent position."""
        param_iter = (
            p for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        )
        for p, e_w in zip(param_iter, self._e_ws):
            p.data.sub_(e_w)

    def step(self, closure):
        """closure() must compute loss, call backward(), return loss."""
        # First forward+backward at current weights
        loss = closure()
        self._ascent_step()
        # Second forward+backward at perturbed weights
        self.optimizer.zero_grad()
        loss2 = closure()
        self._descent_restore()
        # Descent step using second gradients
        self.optimizer.step()
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
```

**File:** `train.py`, lines 483–495 — wrap SOAP:
```python
_soap = SOAP(
    model.parameters(),
    lr=cfg.lr,
    betas=(0.95, 0.95),
    weight_decay=cfg.weight_decay,
    precondition_frequency=cfg.precondition_frequency,
    shampoo_beta=0.95,
    eps=1e-8,
    normalize_grads=False,
)
optimizer = SAMWrapper(_soap, rho=cfg.sam_rho) if cfg.use_sam else _soap
```

**File:** `train.py`, training loop (around line 578) — change the standard step to SAM closure pattern:
```python
if cfg.use_sam:
    def closure():
        # recompute loss from current batch (already in scope)
        pred_c = model(x_norm, is_surface)
        sq_c = F.smooth_l1_loss(pred_c, y_norm, reduction="none", beta=cfg.huber_beta)
        vl_c = (sq_c * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        sl_c = (sq_c * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        lc = vl_c + cfg.surf_weight * sl_c
        lc.backward()
        return lc
    optimizer.zero_grad()
    loss = optimizer.step(closure)
else:
    loss.backward()
    optimizer.step()
```

**File:** `train.py`, `Config` dataclass:
```python
use_sam: bool = False
sam_rho: float = 0.05
```

**CLI invocation:**
```bash
python train.py \
  --optimizer soap \
  --ema_decay 0.99 \
  --precondition_frequency 5 \
  --huber_beta 0.5 \
  --surf_weight 10.0 \
  --use_sam \
  --sam_rho 0.05 \
  --wandb_group sam-soap-sweep
```

Try rho=0.02 as a cheaper variant if 0.05 shows training instability (loss spike in first 2 epochs).

**Critical note:** SAM doubles the number of forward passes per step. At the current model size (0.66M params), this should stay within the 30-minute wall-clock cap for ~7 effective epochs. Watch training throughput in the first 2 epochs; if steps/sec drop below 50% of baseline, reduce batch size by 2x to compensate.

### Expected gain
2–4% on OOD splits (val_geom_camber_rc, val_re_rand) specifically. The average val_avg improvement may be 1–2% even if OOD improves more, because the in-distribution split is already well-fit.

### Risk
Medium-high. SAM doubles forward cost, risks not converging within 30-min cap if the model is large relative to wall time. The closure-based implementation requires that the batch tensors are still in scope for the second forward pass — verify that `x_norm`, `y_norm`, `vol_mask`, `surf_mask` are accessible in the closure (they are, since Python closures capture by reference). Do not accumulate gradients from the first closure call into the SOAP state — the `zero_grad()` call between first and second forward is mandatory.

### Why now
SAM has not been tried in this stack. The research shows consistent OOD improvements on physics surrogate problems. The mechanism directly targets the OOD splits that are most physically meaningful (different geometry, different Re). The double-forward cost is acceptable given model size.

---

## H4: Log-Pressure Auxiliary Loss

### Hypothesis
The pressure field `p` spans multiple orders of magnitude across Re 100K–5M: low-Re cases have small absolute pressure values, high-Re cases have large ones. In normalized space (`y_norm`), Huber loss with beta=0.5 applies equal weight to all points regardless of their original dynamic range. A log-space auxiliary loss on the pressure channel captures relative error instead of absolute error: `log(|p|+eps)` penalizes underestimating a large pressure value as severely as overestimating a small one. This is directly motivated by the `val_re_rand` split, which tests generalization across Re regimes.

### Implementation

**File:** `train.py`, lines 566–574 — extend the loss block:

```python
sq_err = F.smooth_l1_loss(
    pred, y_norm, reduction="none", beta=cfg.huber_beta
)
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss

# Log-pressure auxiliary loss (channel index 2 = pressure p)
if cfg.log_p_weight > 0.0:
    eps_log = 1e-3
    log_pred_p = torch.log(pred[..., 2:3].abs() + eps_log)
    log_true_p = torch.log(y_norm[..., 2:3].abs() + eps_log)
    log_err = F.smooth_l1_loss(log_pred_p, log_true_p, reduction="none", beta=cfg.huber_beta)
    surf_log_loss = (log_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    loss = loss + cfg.log_p_weight * surf_log_loss
```

**File:** `train.py`, `Config` dataclass:
```python
log_p_weight: float = 0.0  # default off; try 0.1
```

**CLI invocation:**
```bash
python train.py \
  --optimizer soap \
  --ema_decay 0.99 \
  --precondition_frequency 5 \
  --huber_beta 0.5 \
  --surf_weight 10.0 \
  --log_p_weight 0.1 \
  --wandb_group log-p-aux-sweep
```

Try weights: 0.1 (primary), 0.05, 0.2 in that order. Stop after the first arm if it moves the metric in either direction.

**Caveat before assigning:** Check whether any in-flight PR is already testing a log-space or relative-error loss on the pressure channel. If so, skip H4 and assign H1 or H2 instead.

### Expected gain
1–3%, concentrated in `val_re_rand`. May degrade `val_single_in_dist` slightly if the log loss introduces conflicting gradient signal for low-Re cases where absolute pressure is small. Monitor per-split metrics, not just val_avg.

### Risk
Medium. The log-space loss operates on normalized `y_norm` (already mean/std normalized), so `|p|+1e-3` may be small or negative for some samples after normalization. The `abs()` prevents NaN gradients, but `log(abs(p_norm)+1e-3)` could be dominated by the eps term for many samples. Verify the distribution of `y_norm[...,2]` values empirically before running — if >20% of surface pressure values after normalization are < 0.01, the eps=1e-3 floor will dominate and the loss will carry no signal. Fix: increase eps to 0.1 or apply the log loss in denormalized space (multiply by `y_std[2]` + add `y_mean[2]` before logging).

### Why now
The `val_re_rand` split is the hardest OOD split and the most physically meaningful. Log-space pressure loss directly targets the mechanism that causes high-Re / low-Re prediction asymmetry in the normalized loss.

---

## H5: Mixed-Precision bfloat16 Autocast

### Hypothesis
At 30-minute wall-clock cap, the model runs approximately 14 epochs. With bfloat16 autocast (not float16 — no GradScaler needed, no NaN risk), forward and backward passes run faster on A100/H100 hardware (typical 1.5–2x throughput), allowing ~20–22 effective epochs within the same wall-clock budget. More epochs mean the EMA checkpoint accumulates more smoothing and the SOAP preconditioner is refreshed more times, both of which improve final checkpoint quality. The mechanism is compute-budget amplification, not algorithmic novelty — but within a hard wall-clock cap it is equivalent to free training time.

### Implementation

**File:** `train.py`, training loop — wrap the forward+loss computation in autocast. Approximately lines 563–578.

Current (schematic):
```python
pred = model(x_norm, is_surface)
sq_err = F.smooth_l1_loss(pred, y_norm, ...)
...
loss.backward()
optimizer.step()
```

Change to:
```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
    pred = model(x_norm, is_surface)
    sq_err = F.smooth_l1_loss(pred, y_norm, reduction="none", beta=cfg.huber_beta)
    vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
    surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    loss = vol_loss + cfg.surf_weight * surf_loss

loss.backward()
optimizer.step()
```

**No GradScaler needed** for bfloat16. The autocast does NOT wrap `loss.backward()` or `optimizer.step()` — those remain in float32.

**File:** `train.py`, `Config` dataclass:
```python
use_bf16: bool = False  # default off for reproducibility; set True to test
```

**CLI invocation:**
```bash
python train.py \
  --optimizer soap \
  --ema_decay 0.99 \
  --precondition_frequency 5 \
  --huber_beta 0.5 \
  --surf_weight 10.0 \
  --use_bf16 \
  --wandb_group bf16-autocast-sweep
```

### Expected gain
1–3%, conditional on the throughput gain actually materializing (verify steps/sec in W&B first epoch). If throughput does not improve by at least 30%, the hypothesis is falsified — the model is memory-bandwidth-bound, not compute-bound, and bf16 adds numerical noise without more epochs.

### Risk
Low-medium. bfloat16 has the same dynamic range as float32 (8-bit exponent vs float16's 5-bit exponent) so NaN/Inf from dynamic range is essentially absent. SOAP's preconditioner computation runs in float32 (optimizer.step() is outside autocast), so Kronecker factorization stability is unaffected. The main risk is subtle numerical drift in PhysicsAttention's softmax temperature scaling — check that `temperature` parameter (line ~152 in train.py) does not produce inf after bf16 conversion. If model outputs become NaN in first epoch, add `.float()` cast after the autocast block.

### Why now
This is the only experiment that does not change the algorithmic stack at all — it tests whether the wall-clock cap is the binding constraint. If bf16 gives 50% more epochs for free, it is worth combining with every other winning technique. It is also diagnostic: if throughput does not improve, we learn the model is memory-bound.

---

## Priority ranking by expected_gain / risk

1. **H1 (Cosine LR floor)** — highest ratio: 1-line change, clear mechanism, near-zero risk
2. **H2 (Lookahead on SOAP)** — strong mechanism, medium risk, orthogonal to EMA
3. **H4 (Log-pressure aux loss)** — targets specific OOD weakness, verify normalization first
4. **H5 (bf16 autocast)** — diagnostic value + free epochs, low effort
5. **H3 (SAM on SOAP)** — highest expected gain on OOD but highest cost and implementation complexity

---

## Research state notes

**Current best explanation for remaining gap:** The 30-min wall-clock cap limits the number of effective epochs to ~14. Within those epochs, the EMA checkpoint does not fully converge, and the cosine schedule decays the LR to near-zero before the cap is hit. Secondary bottleneck: the Huber loss treats pressure prediction error in absolute normalized units regardless of the underlying Re-driven dynamic range.

**Ruled out (from prior PRs):** huber_beta coarse sweep (non-0.5 values worse); EMA decay outside 0.99 (prior sweep).

**Open uncertainties:**
1. Is the wall-clock cap the binding constraint (can be tested by H5)?
2. Does the OOD gap come from sharpness (SAM, H3) or from the LR schedule (H1) or from loss formulation (H4)?
3. Is Lookahead beneficial given that SOAP already has EMA-like momentum through its preconditioner?
