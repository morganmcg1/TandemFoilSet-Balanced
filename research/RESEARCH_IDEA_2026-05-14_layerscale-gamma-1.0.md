# Round 132 — LayerScale γ_init=1.0 (remove conservative init for 4-block shallow net)

## Hypothesis

Change the **LayerScale γ initialization from 1e-4 to 1.0** (effectively disabling the cautious warmup) for the 4-block Transolver. Tests whether the conservative small-γ initialization from CaiT — designed for 12-24 block very-deep networks — is over-regularizing block contributions in our shallow 4-block architecture, suppressing useful representations from the earliest training steps.

## Why this might WIN

1. **γ=1e-4 was designed for very deep networks.** Touvron et al. 2021 (CaiT) introduced LayerScale specifically for training VERY DEEP vision transformers (≥24 blocks) where inter-block gradient explosions are a real problem. Our model has 4 blocks — there is no inter-block gradient explosion regime to protect against. The cautious init may be net-harmful rather than net-neutral at this depth.

2. **With γ=1e-4 init, the first ~epochs of training are essentially a single-layer network.** The learnable scalar starts so small that `x + γ * block(x) ≈ x` until γ grows from gradient updates. In a short 30-minute training window on a 60-epoch cosine schedule, this initialization cost is a significant fraction of total compute.

3. **γ is learnable, so γ=1.0 init is still regularized.** Setting γ_init=1.0 does NOT remove LayerScale — the scalar is still learnable and can decrease during training. We're simply starting from a more "trust the blocks" prior rather than the "distrust the blocks" prior from the deep-network recipe.

4. **γ values at convergence may already be near 1.0.** If the learned γ values after 60 epochs are all in [0.5, 2.0], then the 1e-4 init is pure wasted training budget to climb from 1e-4 to the "natural" regime. Starting at 1.0 captures those free epochs.

5. **This is a recipe-gap test at zero cost.** No new parameters. No architecture change. A 2-line change. First test of the initialization axis in 110+ taxa.

## Why this might LOSS

1. **γ=1e-4 init may be load-bearing stability insurance even at 4 blocks.** The FiLM + SE + PhysicsAttention interactions could cause inter-block amplification that the cautious init damps. Removing it could cause early training instability (loss spikes at ep1-3).

2. **If γ has already converged to high values, this test is a wash.** If the learnable γ values at ep60 in the baseline are near 1.0 anyway, γ_init=1.0 vs 1e-4 is identical outcome. But we don't know those values without running.

3. **γ=1.0 init may encourage early over-fitting.** With full block gain from step 0, the model might overfit the in_dist split during early epochs when LR is still near peak, then fail to recover generalization in the cosine tail. The cautious init forces the network to first "learn the residual" before adding block contributions — a form of curriculum.

## Falsifiable predictions

- **WIN** (val < 30.5605): Conservative init was suppressing early-block learning; 4-block depth doesn't need it. Try γ_init=0.1 (intermediate) for confirmation.
- **LOSS with spikes** (train loss diverges at ep1-3): FiLM + PhysicsAttention interaction causes amplification; cautious init was load-bearing. Confirmed: don't change.
- **WASH** (val ≈ 30.5605 ± 0.5%): Learnable γ quickly learns to equivalent values; init doesn't matter. Log γ values at ep5 and ep60 — if ep5 already near 1.0 in baseline, this is expected.
- **LOSS without spikes** (val > 31.0): γ_init=1.0 causes early overfit without training stability issue. Try γ_init=0.1 as middle ground.

## Implementation

### Step 1: Find the LayerScale initialization in `TransolverBlock.__init__`

Currently the code should have something like:
```python
self.gamma_attn = nn.Parameter(1e-4 * torch.ones(hidden_dim))
self.gamma_mlp  = nn.Parameter(1e-4 * torch.ones(hidden_dim))
```

Change the init scalar:
```python
self.gamma_attn = nn.Parameter(1.0 * torch.ones(hidden_dim))
self.gamma_mlp  = nn.Parameter(1.0 * torch.ones(hidden_dim))
```

That is the **only change** — the code that applies LayerScale (e.g., `x = x + self.gamma_attn * attn_out`) remains identical.

### Step 2: Verify forward unchanged

The forward pass remains: `x = x + self.gamma_attn * self.attn(self.ln_1(x))` for attention, same for MLP. The only change is the initialization value.

### Step 3: Startup diagnostics

```python
# Print initial γ values to verify the init change landed
for i, block in enumerate(model.blocks):
    print(f"Block {i} gamma_attn init: {block.gamma_attn.mean().item():.4f}, "
          f"gamma_mlp init: {block.gamma_mlp.mean().item():.4f}")
# Expected: all 1.0000 (vs baseline 0.0001)
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # unchanged ~407,940
```

**Per-epoch:** log `block.gamma_attn.mean()` and `block.gamma_mlp.mean()` for all 4 blocks. Key diagnostic: does γ DECREASE from 1.0 (model learns to down-weight some blocks), stay at 1.0 (neutral), or increase (model wants even MORE block contribution)?

### Step 4: Watch for early instability

Check ep1-ep3 train loss carefully. If train loss at ep1 is ≫ the baseline ep1 loss, there may be amplification issues. In that case, try γ_init=0.1 as a follow-up.

## Baseline (PR #2879)

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

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/layerscale-gamma-1.0" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — γ_init changed in `TransolverBlock.__init__`. Param count unchanged (~407,940). **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test with Δ vs baseline
3. **γ trajectory diagnostic:** table of `gamma_attn.mean()` and `gamma_mlp.mean()` for each block at ep1, ep5, ep30, ep60. Did γ stay near 1.0, drift up, or drift down?
4. Param count confirmation (~407,940 — zero change)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train loss at ep1-3 vs baseline ep1-3 (watch for instability)
7. Train→val loss gap
8. **Plain-language verdict:** did γ_init=1.0 help, hurt, or wash? If WIN → test γ_init=0.1 as well to characterize the axis. If training diverged → log and close. If LOSS → check if γ drifted BELOW 0.1 by ep60 (if so, 1e-4 init was load-bearing). If WASH → γ init doesn't matter for this architecture; close axis.

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
