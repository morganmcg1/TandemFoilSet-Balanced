# Round 136 — RMSNorm replacement for LayerNorm (new representation-axis recipe gap)

## Hypothesis

**Replace all `nn.LayerNorm` instances in `TransolverBlock` with `RMSNorm`.** Tests whether LayerNorm's mean-centering subtraction is load-bearing for CFD targets — a clean recipe-gap test on the normalization axis that has never been explored in this launch.

RMSNorm is the modern default in LLaMA, T5, and PaLM. It computes `x / sqrt(mean(x²) + ε) · γ` instead of LayerNorm's `(x - mean(x)) / sqrt(var(x) + ε) · γ + β`. The difference: RMSNorm preserves the mean signal; LayerNorm removes it.

This is a REPRESENTATION-axis intervention (per #2922 student decisive insight) that doesn't shift parameter capacity — RMSNorm has fewer parameters (no β bias) but the difference is negligible (~few hundred params total).

## Why this might WIN

1. **Lion's sign(grad) is mean-preserving by design.** RMSNorm preserves mean, LayerNorm strips it. Lion's update doesn't care about magnitude but it cares about which direction is increasing — RMSNorm may better preserve directional information at deeper blocks where the mean is signal-carrying for CFD (pressure has a natural baseline DC offset per Reynolds regime).

2. **Modern transformer default.** LLaMA (2/3), T5, PaLM, Falcon all use RMSNorm. This is a recipe gap.

3. **Slightly faster** — one fewer subtraction op per forward pass.

4. **Single-line change.** Define a small RMSNorm module (or use `nn.RMSNorm` if available in your torch version) and swap.

5. **Zero parameter cost change** (RMSNorm has γ scale per-channel like LayerNorm but no β bias — saves ~96·N_norm_sites params total).

## Why this might LOSS

1. **LayerNorm mean-centering may be load-bearing for CFD.** Pressure and velocity fields have natural baselines that vary by Reynolds regime and angle of attack. LayerNorm subtracts those baselines explicitly, isolating fluctuations — which is the physically informative quantity. RMSNorm preserves the baseline, which the model then has to learn to factor out implicitly.

2. **Existing init was tuned for LayerNorm.** Body LayerScale γ=1e-4 and the rest of the init recipe were chosen with LayerNorm in mind. Swapping to RMSNorm changes the residual stream statistics.

3. **The cruise/in_dist trade-off is structural.** If the trade-off is in the loss landscape (per #2922 insight), changing norm doesn't move it.

## Falsifiable predictions

- **WIN** (val < 30.5605): RMSNorm's mean preservation helps Lion + Transolver. Try also swapping out the head's normalization if any.
- **PARTIAL** (in_dist WIN at cruise cost): Meta-signal pattern. Norm scheme triggers capacity-shift behavior.
- **WASH** (val ≈ 30.5605 ± 0.3%): Norm scheme doesn't matter at this scale. Close norm-axis.
- **LOSS** (val > 31.0): LayerNorm centering was load-bearing for CFD. Close norm-axis.

## Implementation

### Step 1: Find every `nn.LayerNorm` in `train.py`

Likely candidates:
- `TransolverBlock` will have at least one LayerNorm (typically pre-norm before attention and pre-norm before MLP)
- The `PhysicsAttention` may have an internal LayerNorm
- The preprocess MLP or output head may have a LayerNorm

Count all sites. Likely 8-12 total in a 4-block setup.

### Step 2: Define RMSNorm (if not in torch version)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```

If your torch version has `nn.RMSNorm` (PyTorch 2.4+), use that.

### Step 3: Swap every LayerNorm site

```python
# Before
self.norm1 = nn.LayerNorm(n_hidden)

# After
self.norm1 = RMSNorm(n_hidden)
```

Apply at EVERY norm site. Keep `eps` similar to existing LayerNorm eps (typically 1e-5 or 1e-6).

### Step 4: Startup diagnostics

```python
# Count norm sites
norm_sites = sum(1 for m in model.modules() if isinstance(m, RMSNorm))
print(f"RMSNorm sites: {norm_sites}")
print(f"Old LayerNorm count was {expected_layernorm_count}")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # Expect slightly less than 407,940 (no β bias)
```

### Step 5: Per-epoch logging

Track:
- Train/val loss trajectory — does ep1 train loss differ from baseline? (Different norm could affect early-training dynamics)
- The pre-block residual stream magnitude at convergence — RMSNorm should preserve more signal magnitude than LayerNorm

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 (with LayerNorm) |

After change: same architecture, all LayerNorm replaced with RMSNorm. Param count slightly lower (~407,500 estimated — saves the LayerNorm β biases).

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/rmsnorm-replacement" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — RMSNorm hardcoded in `Transolver` definitions. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Norm-site count diagnostic:** number of LayerNorm sites replaced, param count delta
4. **Train/val ep1 comparison:** does the model train differently from cold start without mean-centering? Report train_surf/val_avg at ep1, 5, 10 vs baseline trajectory.
5. Param count confirmation (expect slightly under 407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence
8. **Meta-signal check:** uniform LOSS / WIN, or meta-signal asymmetry?
9. **Plain-language verdict:** WIN (RMSNorm preserves signal for Lion) / WASH (norm doesn't matter) / LOSS (LayerNorm centering was load-bearing). If LOSS, was it uniform or asymmetric?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
