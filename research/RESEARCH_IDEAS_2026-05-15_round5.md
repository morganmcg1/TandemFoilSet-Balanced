<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round-5 Research Ideas — 2026-05-15

Baseline: Huber δ=2.0 (frieren PR #3248), val_avg/mae_surf_p=107.46, test=101.98.
Hard constraint: 30-min wall-clock / ~14 epochs. No new packages without pyproject.toml change.

---

## Idea 1: EMA Model Averaging (Post-Hoc Snapshot Ensemble)

### Mechanism
Exponential moving average of model weights during training acts as an implicit ensemble
of late-training checkpoints. It smooths out the stochastic noise from per-batch parameter
updates without changing the gradient signal or optimizer. In the short 14-epoch regime, the
EMA weights consistently lag 1-3 epochs behind the instantaneous model — exactly the "look
back" needed to avoid overfitting to the noisy LR tail. Note: this was assigned in round-3
(nezuko) but ran against the *MSE* baseline with ema_decay=0.999 (a typical NLP value).
The result was 130.17 — *exactly* the fresh-slate baseline (130.18). That result is not a
falsification; it is a signal that ema_decay=0.999 on a 14-epoch run is too slow to build
meaningful momentum. At 14 epochs, 0.999^14 ≈ 0.986 — the average is dominated by the
initial (random) weights, not the late-training weights. The correct decay for a short run
is 0.99 or even 0.98 so the average converges to recent checkpoints within the budget.
Now we are also stacking on the Huber baseline which removes the noise that was masking
the EMA signal.

### Implementation

In train.py, after line 431 (model = Transolver(...).to(device)):

```python
# EMA model — add after model creation
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema_decay = 0.99  # primary arm; try 0.98 if this doesn't converge fast enough
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
```

Inside the training loop, after optimizer.step() (line 506):

```python
ema_model.update_parameters(model)
```

For validation and test, replace `model.eval()` + evaluate calls with `ema_model.eval()`:

```python
# In the validation block (lines 519-526):
ema_model.eval()
split_metrics = {
    name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device)
    for name, loader in val_loaders.items()
}
```

Also load ema_model weights for test evaluation. The instantaneous model still trains
normally; only validation and test use the EMA weights.

Add a CLI flag to train.py Config:

```python
ema_decay: float = 0.0  # 0.0 = disabled; 0.99 = primary arm; 0.98 = secondary arm
```

And wrap the EMA logic with `if cfg.ema_decay > 0.0:`.

### Expected Impact
Round-3 hint: nezuko's 130.17 ≈ baseline (0.999 too slow). Stacked on Huber + correct
decay (0.99), expect −2 to −5 on val_avg/mae_surf_p. Lower variance across random seeds
is a secondary benefit (EMA should reduce the seed sensitivity seen in the 3-arm Huber sweep).

### Risk / Failure Mode
If ema_decay=0.99 is still too slow at 14 epochs, the EMA average will still underweight
late-training weights. Test: check if `ema_model` val_avg < instantaneous model val_avg.
If not, try 0.98. The failure mode is indistinguishable from the round-3 null — run a
short 3-epoch debug run to check convergence speed of EMA weights before the full run.

---

## Idea 2: log(Re) Fourier Embedding (Physics-Motivated Input Feature Expansion)

### Mechanism
Reynolds number (input feature dim 13, normalized log-scale) is the most important
physics parameter governing pressure distribution. The current preprocess MLP maps
log(Re) as a single scalar through the same linear projection as all 22 fun_dim features.
A Fourier expansion exposes multiple frequency components of Re variation to the attention
mechanism, allowing the slice tokens to specialize by Re regime without requiring the
network to learn these frequency decompositions from scratch. This is analogous to NeRF
positional encoding but applied to the physics parameter rather than spatial coordinates.
NACA Fourier features (thorfinn, round-3) expanded the *geometry* features the same way
and got +5.1% on MSE. Re Fourier expands the *physics* features. The two are orthogonal
and should stack.

Mechanism is distinct from NACA Fourier: NACA features improve OOD *geometry*
generalization (geom_camber splits); Re Fourier should primarily improve OOD *Re regime*
generalization (re_rand split, currently 100.11).

### Implementation

In train.py, before model_config (line 418), add a preprocessing step inside the
data normalization block. The cleanest approach is to modify the input features directly
in the training loop rather than changing the data interface.

Add to Config dataclass:

```python
re_fourier_freqs: int = 0  # 0 = disabled; 4 = primary arm
```

In the training loop, after `x_norm = (x - stats["x_mean"]) / stats["x_std"]` (line 488):

```python
if cfg.re_fourier_freqs > 0:
    # dim 13 is log(Re) in the normalized input
    re_dim = x_norm[:, :, 13:14]  # [B, N, 1]
    freqs = torch.arange(1, cfg.re_fourier_freqs + 1, device=x_norm.device).float()
    # [B, N, 1] * [F] -> [B, N, F]
    re_sin = torch.sin(re_dim * freqs[None, None, :] * math.pi)
    re_cos = torch.cos(re_dim * freqs[None, None, :] * math.pi)
    x_norm = torch.cat([x_norm, re_sin, re_cos], dim=-1)  # [B, N, 24 + 2*F]
```

Also add `import math` at the top if not present. Then update model_config:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2 + 2 * cfg.re_fourier_freqs,  # 22 + 8 for freqs=4
    ...
)
```

The same transform must be applied at eval time (inside evaluate_split) — the cleanest
way is to pass a feature transform callable to evaluate_split, or to add the transform
inside the model's forward pass. Given the read-only data/ constraint, the training loop
approach above is correct, but requires patching evaluate_split to also apply the transform.
The simplest fix: wrap evaluate_split to accept an optional `x_transform` callable.

Primary arm: `--re_fourier_freqs 4` (adds 8 dims → fun_dim=30).

### Expected Impact
Re Fourier should specifically improve val_re_rand (current 100.11). NACA features got
+5.1% on the geometry splits. Analogously, +3% to +8% on val_re_rand is plausible, which
maps to roughly −1 to −3 on the average.

### Risk / Failure Mode
The fun_dim change requires model_config to be updated consistently — mismatch between
train and eval x_norm shapes will crash immediately (easy to detect). The deeper risk is
that log(Re) in normalized space already has small variation within the dataset range and
Fourier encoding adds nothing beyond what a linear projection already captures. The
diagnostic: if val_re_rand doesn't move, the Re signal was already adequate.

---

## Idea 3: PhysicsAttention Temperature Initialization Sweep

### Mechanism
PhysicsAttention has a learnable temperature parameter initialized to 0.5 for all heads:

```python
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
```

This temperature gates the softmax over nodes → slice tokens. Low temperature → sharper
assignment (near hard-assignment, one node per slice); high temperature → smooth assignment
(each slice sees all nodes). The initialization controls which attractor the optimizer
starts from. At init=0.5 and 14-epoch budget, the temperature may not have time to converge
to its optimal value — we are running the optimizer at a point chosen by the paper authors
for a different dataset and epoch budget (50+ epochs). This is a cheap 1-line change that
tests whether the temperature initialization is a bottleneck in the short-epoch regime.

### Implementation

Modify PhysicsAttention.__init__ in train.py (line ~95):

```python
# Original:
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

# Arm 1: warmer init (smoother assignment, broader slice coverage early)
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 1.0)

# Arm 2: colder init (sharper assignment, more specialized slices early)
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.2)
```

Add a CLI flag:

```python
slice_temperature_init: float = 0.5  # default: paper value
```

And in model creation, pass the value to the Transolver constructor, which passes it to
PhysicsAttention. Requires adding a `temperature_init` argument to PhysicsAttention:

```python
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * temperature_init)
```

Run two arms: `--slice_temperature_init 1.0` and `--slice_temperature_init 0.2`.

### Expected Impact
This is a diagnostic as much as an optimization. If temperature=1.0 wins, it suggests
the 14-epoch budget needs softer attention early to explore the slice assignment space.
If temperature=0.2 wins, it suggests specialized slices help even from early training.
Predicted range: −1 to −4 on val_avg/mae_surf_p. If neither moves the metric, temperature
initialization is not a bottleneck and we learn something clean about the optimization.

### Risk / Failure Mode
The temperature parameter is shared across all heads and layers in a given block. Head
diversity relies on different query/key projections learning different patterns, not on
different temperature values. Very low temperature (0.2) may cause early attention collapse
(one node monopolizes each slice), leading to slow convergence in the first few epochs —
which at 14-epoch budget can be fatal. Monitor train loss at epoch 1-3 to detect collapse.

---

## Idea 4: Cosine T_max Fix (Actual LR Annealing)

### Mechanism
This is the most important untested fix in the entire codebase. The training loop uses:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

MAX_EPOCHS=50, but only 11-14 epochs run before timeout. The cosine schedule with T_max=50
means the LR decays from 5e-4 to ~4.7e-4 after 14 epochs (only 3% of the annealing range).
The model effectively trains with a *constant* learning rate equal to the initial lr. This
is a fundamental misconfiguration: we have never seen what a properly annealed LR looks like
in this budget.

Setting T_max=14 (matching the actual epoch budget) would let the LR decay from 5e-4 to
~0 over the actual training run, which is the intended behavior of cosine annealing. The
round-3 warmup-cosine experiment (askeladd, val=109.99) used SequentialLR with a 5-epoch
linear warmup followed by CosineAnnealingLR — but that experiment combined warmup + cosine
+ grad_clip simultaneously. The isolated effect of the T_max fix has never been tested.

The hypothesis: the optimizer is spending all 14 epochs in the high-LR regime, meaning
late-training refinement (the small-step convergence that LR annealing enables) has never
happened. Fixing T_max should provide free improvement.

### Implementation

Estimate the actual epoch count at the start of training and set T_max accordingly.
The cleanest approach: add a CLI flag `lr_T_max` to override the T_max:

```python
lr_T_max: int = 0  # 0 = use MAX_EPOCHS (current behavior); else use this value
```

In the scheduler line (line 436):

```python
t_max = cfg.lr_T_max if cfg.lr_T_max > 0 else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
```

Primary arm: `--lr_T_max 14` (conservative estimate of actual epoch count).
Secondary arm: `--lr_T_max 12` (tighter estimate; LR hits near-zero by epoch 12).

This is a pure CLI change once the flag is added. Zero risk of instability from the code
change itself.

### Expected Impact
This is a first-principles fix. The expected gain is −3 to −8 on val_avg/mae_surf_p.
Askeladd's warmup-cosine (which implicitly fixed annealing via SequentialLR) got 109.99
vs 129.99 baseline — a +15% improvement. Isolating just the T_max fix should capture
a meaningful fraction of that gain without adding the warmup complexity. The open question
is whether warmup or annealing drove most of askeladd's gain.

### Risk / Failure Mode
If T_max is set too short (LR drops to ~0 by epoch 8), early epochs undershoot their
potential and the model enters premature fine-tuning. Use val_avg plateau detection to
choose T_max. The stop condition: if the T_max=12 arm is worse than T_max=14, the model
needs more LR range than estimated. The diagnosis is clear: check the LR curve in W&B.

---

## Idea 5: Per-Node Dropout on Fun Features (Curriculum Regularization)

### Mechanism
The current model has dropout=0.0 everywhere. In the short 14-epoch regime, the model
likely underfits rather than overfits (evidence: val monotonically improves through all
epochs; the best checkpoint is always the last). Adding mild dropout on the fun_dim features
(the 22-dim physics/geometry input to the preprocess MLP) acts as a targeted regularizer
that forces the model to learn redundant representations of the physics inputs, improving
generalization to OOD geometries and Re regimes.

The distinction from standard attention dropout: we are not dropping attention weights
(which would disrupt the slice-token aggregation), but rather randomly zeroing input
feature channels before the preprocess MLP. This is equivalent to feature-level dropout
and has been shown to improve generalization in PDE surrogate settings (FNO, DeepONet
variants) where input features are sparse and physics-informed.

### Implementation

In Transolver.forward (lines ~195-215), the input flow is:

```python
fx = self.preprocess(x) + self.placeholder[None, None, :]
```

Where x = x_norm[:, :, :] (all 24 dims). The fun_dim features are dims 2-23 (space_dim=2,
fun_dim=22). Add a train-only feature dropout:

```python
# In Transolver.__init__, add:
self.input_dropout = nn.Dropout(p=input_dropout)  # new __init__ arg

# In Transolver.forward, before preprocess:
if self.training:
    x = torch.cat([x[:, :, :2], self.input_dropout(x[:, :, 2:])], dim=-1)
fx = self.preprocess(x) + self.placeholder[None, None, :]
```

Add to model_config:

```python
model_config = dict(
    ...
    input_dropout=0.1,  # primary arm; try 0.2 as secondary
)
```

And add `input_dropout=0.0` to the Transolver.__init__ signature.

Alternatively, a simpler approach that requires only train.py changes in the training loop:

```python
# After x_norm = (x - stats["x_mean"]) / stats["x_std"]  (line 488)
if model.training and cfg.input_dropout > 0.0:
    drop_mask = torch.rand(*x_norm.shape[:2], x_norm.shape[2] - 2,
                           device=x_norm.device) > cfg.input_dropout
    x_norm = torch.cat([
        x_norm[:, :, :2],
        x_norm[:, :, 2:] * drop_mask.float()
    ], dim=-1)
```

Add `input_dropout: float = 0.0` to Config. Primary arm: `--input_dropout 0.1`.

### Expected Impact
Mild dropout in the short-epoch regime may hurt rather than help if the model is already
underfitting. The test: val loss should not increase noticeably vs baseline. If val_geom*
splits improve (OOD geometry) while val_single_in_dist holds, the regularization hypothesis
is supported. Expected range: −1 to −4 on val_avg/mae_surf_p, primarily via OOD splits.

### Risk / Failure Mode
The primary risk is that 14 epochs is not enough to recover from the noise injected by
input dropout — the model never reaches the same train loss floor, and the regularization
benefit doesn't manifest within the budget. The falsification criterion: if val_avg is
worse than baseline at the same epoch count, input dropout is not appropriate for this
training budget and should be ruled out.

---

## Idea 6: Gradient Clipping Standalone (Isolating Askeladd's Win)

### Mechanism
Askeladd's round-3 result (val=109.99) bundled three changes: linear warmup, cosine
T_max fix (via SequentialLR), and gradient clipping (grad_clip=1.0). The round-4 warmup-
cosine-stacked experiment re-tests this compound change on the Huber baseline. But we
have never isolated gradient clipping alone. The hypothesis is that gradient clipping is
the dominant driver of askeladd's improvement, not warmup or LR scheduling. Rationale:
Huber already caps *loss-level* gradients at δ, but the backward pass through the network
can still produce large gradient norms due to amplification by the surf_weight=10 factor.
Gradient clipping caps the *parameter-update-level* gradient norm independently of the
loss function, and stacks orthogonally with Huber.

### Implementation

No code change needed. Just add clip_grad to the training loop. But since it requires
a code change in train.py (adding `torch.nn.utils.clip_grad_norm_`), add a CLI flag:

```python
grad_clip: float = 0.0  # 0.0 = disabled; 1.0 = primary arm; 5.0 = secondary arm
```

In the training loop, between loss.backward() and optimizer.step() (lines 505-506):

```python
loss.backward()
if cfg.grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

Primary arm: `--grad_clip 1.0` (same value as askeladd's winning config).
Secondary arm: `--grad_clip 5.0` (looser clip, to check sensitivity).

This is a minimal 3-line code change plus 1 Config field. No architecture changes, no
new dependencies.

### Expected Impact
If grad_clip alone explains askeladd's result, expect the primary arm to reach ~109 on
the MSE baseline and potentially 104-106 on the Huber baseline. If grad_clip has no
isolated effect, that falsifies the "clipping is the driver" hypothesis and points to
warmup or T_max as the dominant mechanism. Either result is informative.

### Risk / Failure Mode
Gradient clipping at clip_norm=1.0 may be too aggressive for the Huber baseline, which
already has reduced gradient variance. A clip that fires on every batch effectively reduces
the learning rate and may slow convergence in the early epochs. Check wandb train/loss
curve: if it's flat in epoch 1-2, the clip is too tight. Use clip_norm=5.0 in that case.

---

## Idea 7: Larger Hidden Dim n_hidden=256 with BF16

### Mechanism
The current model has n_hidden=128, giving 0.66M parameters. The original Transolver
paper uses n_hidden=256 as the default for most benchmarks (2.6M params for the NS2d
task). We tried n_layers=8 (deeper) in round-3; it was undertrained in 14 epochs. But
we have never tried n_hidden=256 (wider). The per-epoch wall-clock at n_hidden=128 is
~132s. At n_hidden=256, attention complexity scales as O(N*n_hidden) — the PhysicsAttention
projection `in_project_slice: n_hidden//n_head → slice_num` scales linearly with hidden
dim, so per-epoch time should approximately double. With bf16 (which should be available
on H100), this fits in the 30-min budget for ~7 epochs vs 14 — not ideal, but still
within the regime where Huber showed rapid early improvement (best epoch was 14/50 meaning
it had already converged by then, suggesting even 7 epochs of a wider model might compete).

The mechanism: n_hidden=256 gives the preprocess MLP (24→256) and each TransolverBlock
more capacity to represent the multi-scale pressure patterns across the 3 mesh domains.

### Implementation

The change is one line in model_config:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,    # was 128
    n_layers=5,
    n_head=8,        # scale heads proportionally (was 4; dim_head stays 32)
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

To keep this within budget, add bf16 autocast. In the training loop, wrap forward+loss
with autocast:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred = model({"x": x_norm})["preds"]
    # ... loss computation
```

The model stays in fp32; only the forward pass matmuls use bf16. Note: alphonse's
round-3 bf16 run crashed — check that PR for the canonical recipe to avoid dtype mismatch.
The safe pattern: model in fp32, autocast wraps forward+loss only, no GradScaler needed
for bf16 (unlike fp16).

Add CLI flags:

```python
n_hidden: int = 128
use_bf16: bool = False
```

Primary arm: `--n_hidden 256 --use_bf16 true`.

### Expected Impact
More capacity (4x params) should help the complex multi-domain pressure prediction.
The risk is that 7 epochs of a 256-width model is strictly worse than 14 epochs of 128.
Predicted range: −3 to −10 if bf16 gives the epoch budget needed; +5 to +15 (worse) if
the model is undertrained at 7 epochs. This is a high-variance bet but tests a lever
(model capacity) that has never been cleanly tested without the undertrained constraint.

### Risk / Failure Mode
The main failure mode is the same as round-3 deeper model: per-epoch cost too high for
the epoch budget. Mitigate by profiling one epoch before committing to a full run. If
one epoch takes >3 min with bf16, the run will complete only 6-7 epochs and likely be
undertrained. In that case, this idea needs to wait for the bf16 throughput to be confirmed
working (alphonse's PR #3282 was assigned this). Do not run n_hidden=256 without first
confirming bf16 works in this codebase.

---

## Idea 8: Log-Cosh Loss (Smooth Huber Alternative)

### Mechanism
Huber loss has a non-smooth elbow at |err|=δ — the gradient changes discontinuously from
quadratic to linear at that point. This kink creates a transition zone where the optimizer
oscillates between quadratic and linear gradient regimes depending on whether a sample's
error straddles δ. Log-cosh loss is a smooth approximation of Huber: it is quadratic near
zero (like MSE) and linear in the tails (like MAE), but the transition is smooth everywhere.

log-cosh: L(e) = log(cosh(e)) ≈ 0.5*e^2 for small e, ≈ |e| for large e

No δ hyperparameter to tune. The smoothness means gradients are continuous, which may
improve optimizer convergence near the Huber elbow. Additionally, log-cosh is scale-free:
unlike Huber where δ must be set relative to the loss scale, log-cosh adapts automatically.
This matters because the normalized loss scale shifts as the model improves — a fixed δ
becomes relatively tighter (more L1-like) as training progresses.

### Implementation

In the training loop, replace the Huber block (lines 491-496):

```python
# Replace:
abs_err = (pred - y_norm).abs()
sq_err = torch.where(
    abs_err < cfg.huber_delta,
    0.5 * abs_err ** 2,
    cfg.huber_delta * (abs_err - 0.5 * cfg.huber_delta),
)

# With:
# Log-cosh loss (no delta hyperparameter needed)
err = pred - y_norm
sq_err = torch.log(torch.cosh(err.clamp(-20, 20)) + 1e-8)
# Note: clamp prevents overflow in cosh for very large errors
```

Add a CLI flag:

```python
loss_fn: str = "huber"  # "huber" | "log_cosh"
```

And wrap the loss computation in a conditional based on cfg.loss_fn.

The `+ 1e-8` inside log prevents log(0) if cosh returns 0 (not possible mathematically,
but defensive). The clamp(-20, 20) prevents cosh overflow: cosh(20) ≈ 2.4e8, well within
fp32 range (max ~3.4e38), so the clamp can be relaxed to (-30, 30) for safety. Test first.

Primary arm: `--loss_fn log_cosh`.

### Expected Impact
Log-cosh should behave similarly to Huber δ=1.0 in terms of tail suppression, but with
smoother optimization dynamics. If the round-4 Huber delta sweep (tanjiro) finds δ=1.0
as the sweet spot, log-cosh is a natural next step: it achieves similar tail behavior
without the δ hyperparameter. Expected improvement: −1 to −4 vs Huber δ=2.0 baseline,
primarily via smoother convergence in the surf_loss term.

### Risk / Failure Mode
Log-cosh penalizes large errors *more leniently* than Huber for very large |err|: for
|err|>>δ, Huber gradient = δ (constant), while log-cosh gradient = tanh(err) → 1 (also
~constant, but approaches limit more slowly). In practice, the difference is in the
transition region. If the surface pressure outliers are very large (|err| >> 5-10 in
normalized space), both converge to approximately linear behavior and there is no
meaningful difference. The falsification test: if log-cosh and Huber δ=2.0 give identical
val curves, the tails are so large that the smooth transition region is irrelevant.

---

## Summary Table

| # | Hypothesis | Level | Key change | Primary risk | Student |
|---|---|---|---|---|---|
| 1 | EMA model averaging (correct decay) | Training | ema_decay=0.99 on Huber baseline | Too slow even at 0.99; re-test from 0.98 | nezuko |
| 2 | log(Re) Fourier embedding | Data/input | fun_dim 22→30, sin/cos of Re*freqs | evaluate_split needs matching transform | alphonse |
| 3 | PhysicsAttention temperature init sweep | Architecture | temperature_init=1.0 vs 0.2 | Low temp → early collapse | edward |
| 4 | Cosine T_max fix (14 vs 50) | Training | lr_T_max=14 (matches actual budget) | T_max too short → premature fine-tuning | askeladd (post-round-4) |
| 5 | Per-node input dropout | Regularization | input_dropout=0.1 on fun features | Hurts underfitting regime; 14 epochs too short | tanjiro (post-round-4) |
| 6 | Gradient clipping standalone | Training | grad_clip=1.0 (isolates askeladd lever) | Already handled by Huber; no marginal effect | fern (post-round-4) |
| 7 | n_hidden=256 + BF16 | Architecture | 4× capacity, bf16 autocast | Undertrained at 7 epochs; needs bf16 working | thorfinn (post-round-4) |
| 8 | Log-cosh loss | Loss | Smooth Huber alternative, no δ tuning | Equivalent to Huber in large-error regime | frieren (post-round-4) |

Ordering priority given round-4 results pending:
- **Highest value now**: Ideas 4 (T_max fix) and 6 (grad clip) are cheap, isolated tests
  of askeladd's compound mechanism. Run these immediately once round-4 slots open.
- **Highest ceiling**: Idea 7 (n_hidden=256 + BF16) — but only after bf16 throughput is
  confirmed working in the codebase (pending alphonse #3282).
- **Orthogonal to all current experiments**: Idea 2 (Re Fourier) and Idea 3 (temperature
  init) test mechanisms that have not been explored in any round.
- **Safe fallback**: Idea 8 (log-cosh) is a clean, isolated substitution once round-4
  Huber delta sweep results are in — it is the natural follow-on to whichever δ wins.
