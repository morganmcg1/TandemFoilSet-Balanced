# Hypothesis: lookahead-lion (frieren)

## Hypothesis

Wrapping Lion in a Lookahead outer loop (Zhang et al. 2019) should smooth Lion's
inherently discrete sign-update trajectory. Lookahead maintains "slow weights"
that interpolate toward the "fast weights" every `k` steps with coefficient `α`:

```
slow_w ← slow_w + α · (fast_w − slow_w)   # every k steps
fast_w ← slow_w                            # reset fast weights to interpolated point
```

**Why this is a fit for Lion specifically:** Lion's update is `LR · sign(β₁·m + (1-β₁)·g)`
— each step is a discrete `±LR` jump in every coordinate. The magnitude information
is fully discarded. This makes Lion's per-step trajectory inherently more noisy
than Adam/AdamW: the model jitters at constant step-size in random sign patterns,
and the optimizer relies on momentum + LR decay to extract a useful net direction.
Lookahead's slow-weight average is exactly the orthogonal compensation: it adds
back the per-step averaging that Lion removed.

**Why this is orthogonal to other levers in the SOTA stack:**

- **Clip=1.0** (engaging 98%+ of steps): normalizes pre-Lion gradient magnitude.
  Lookahead operates AFTER Lion's sign step — sees the post-update positions, not
  the gradients.
- **EMA d=0.999** (edward in flight on lr=2e-4): operates on parameters *for
  evaluation*. Lookahead changes the actual training trajectory the model sees.
  Different points in the compute graph — they could stack.
- **Cosine T_max=21**: shrinks step size over time. Lookahead changes the
  effective trajectory smoothness at fixed step size. Independent levers.

**Two arms:**

1. **Arm 1 (primary) — Lookahead k=5, α=0.5** (original paper defaults):
   Slow weights update every 5 inner Lion steps. α=0.5 means the slow weight
   moves halfway from its previous position toward the fast weight at each
   sync. These are the Zhang+2019 defaults and have generalized well across
   tasks. Lion has ~150 steps/epoch × 19 epochs = ~2850 steps, so k=5 gives
   ~570 sync points over the run.

2. **Arm 2 — Lookahead k=10, α=0.8** (less frequent, more aggressive sync):
   Tests a coarser averaging window. α=0.8 brings fast weights further into
   the slow-weight basin. Tests whether longer Lion exploration between
   averages is better than tight averaging.

**Predicted improvement:** −0.5 to −2.0 on val_avg/mae_surf_p. The mechanism
should specifically reduce per-step variance of Lion's sign updates without
disturbing the schedule, so the improvement should appear across all val splits.

**Worst case:** Lookahead's slow-weight averaging dampens too aggressively, the
model fails to escape the random-init basin in the 30-min budget, and val regresses.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change any optimizer flags.**

### 2. Implement the Lookahead wrapper

In `target/train.py`, add a Lookahead optimizer wrapper class. It must be
transparent to `optimizer.step()`, `optimizer.zero_grad()`, and the LR
scheduler (`scheduler.step()` should work unchanged).

The minimal Zhang et al. 2019 implementation pattern:

```python
class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        self.base = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        # Store slow weights — initialized as a copy of current params
        self.slow_state = {}
        for group in self.base.param_groups:
            for p in group["params"]:
                self.slow_state[p] = p.data.clone()

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # Inner step: Lion does its thing.
        loss = self.base.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            # Slow update: slow ← slow + α(fast − slow);  fast ← slow
            for group in self.base.param_groups:
                for p in group["params"]:
                    slow = self.slow_state[p]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss

    def state_dict(self):
        return {
            "base": self.base.state_dict(),
            "step_counter": self.step_counter,
            "slow_state": {id(p): v for p, v in self.slow_state.items()},
        }
```

In the optimizer creation block (currently around line 505 of `train.py`):

```python
base_optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
if cfg.lookahead_k > 0:
    optimizer = Lookahead(base_optimizer, k=cfg.lookahead_k, alpha=cfg.lookahead_alpha)
else:
    optimizer = base_optimizer
```

Add to the Config dataclass:

```python
lookahead_k: int = 0       # 0 = disabled
lookahead_alpha: float = 0.5
```

**Critical:** verify the LR scheduler still sees `base_optimizer.param_groups`
(it does via the `@property param_groups`). Log `lr` per epoch as usual to
confirm the schedule is unaffected.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — Lookahead k=5, α=0.5 (paper defaults):**

```bash
cd target/ && python train.py \
    --lookahead_k 5 \
    --lookahead_alpha 0.5 \
    --wandb_group lookahead-lion \
    --wandb_name la-k5-a0p5 \
    --agent willowpai2i24h3-frieren
```

**Arm 2 — Lookahead k=10, α=0.8 (coarser averaging, more aggressive sync):**

```bash
cd target/ && python train.py \
    --lookahead_k 10 \
    --lookahead_alpha 0.8 \
    --wandb_group lookahead-lion \
    --wandb_name la-k10-a0p8 \
    --agent willowpai2i24h3-frieren
```

### 5. Key signals to report

- `val_avg/mae_surf_p` per epoch — does either arm reach below 65.30?
- **Train loss trajectory** vs baseline `3rvfeq4g` — Lookahead should produce a
  visibly smoother training curve (lower stochasticity per epoch) even if val
  is comparable.
- **Best epoch** — does best epoch shift later? Lookahead typically allows
  longer productive training.
- **Per-split breakdown at best checkpoint** vs SOTA — especially watch
  val_geom_camber_rc and val_re_rand. If smoother updates help OOD, these
  splits should improve relatively more.
- **Peak VRAM** — Lookahead adds one fp32 slow-weight copy of the model
  (~21 MB for 2.7M params). Should be negligible.
- **Clip engagement %** — should be unchanged from baseline (~98%) since
  Lookahead operates after Lion, not on gradients.

### 6. Compute nansafe test metrics

```bash
cd target/ && python eval_nansafe.py <arm1_run_id>
cd target/ && python eval_nansafe.py <arm2_run_id>
```

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — your own lion-lr2e4 (PR #3675, merged 2026-05-16 07:30 UTC):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.2991** |
| **test_avg_nansafe/mae_surf_p** | **60.5400** |
| test_single_in_dist | 64.0454 |
| test_geom_camber_rc | 67.5770 |
| test_geom_camber_cruise | 56.1342 |
| test_re_rand | 54.4033 |
| W&B run | `3rvfeq4g` (group: `lion-lr-sweep`) |
| Stack | Lion lr=2e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21 |
| Best epoch | **19** (FINAL — val still descending at timeout) |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

## Why this matters

Your LR-refine PR closed the LR neighborhood (2e-4 / T_max=21 is a tight local
optimum). The natural next question is whether the LIMITATION is the LR or the
**update geometry** — and Lookahead directly attacks that question. The
mechanism is well-motivated specifically for Lion (whose sign-only updates
discard exactly the variance information that Lookahead reintroduces), and the
implementation is small and self-contained.

If Lookahead wins: we have a new SOTA from optimizer geometry, and Lookahead is
also a multiplier on other improvements (per Zhang et al.).
If both arms lose: Lion's sign-noise is benign at our budget — confirming that
Lion's momentum + cosine schedule already does adequate effective averaging,
and the next round should look at loss/data formulation, not optimizer geometry.
