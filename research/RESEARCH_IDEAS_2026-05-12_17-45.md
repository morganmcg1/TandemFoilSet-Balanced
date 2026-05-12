# Research Ideas — TandemFoilSet Round 5 (Charlie, 30-min cap)
Generated: 2026-05-12 17:45

## Context

Baseline: Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2),
loss = vol_loss + 10*surf_loss, AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR(T_max=epochs),
batch=4, FP32, 30-min wall-clock cap, ~50 epochs typically.
Primary metric: val_avg/mae_surf_p (lower is better).

No prior experiments on this branch. All ideas are first-principles proposals.

---

## Brainstorm (25 candidates, filtered to 12)

Candidates considered and rejected:
- Fourier Neural Operator backbone: heavy, won't converge in 30 min on 240K-node meshes.
- DeepONet: requires sensor-to-query decoupling, incompatible with pad_collate contract.
- Graph attention (GAE): requires edge construction not in current data contract.
- Mixture-of-experts routing: too much extra overhead and unstable in 30 min.
- Diffusion-based surrogate: too slow to sample, no clear physical prior.
- Curriculum by mesh size: can't reorder data loader without modifying data/ (read-only).
- Multi-task learning separate heads: already separate vol/surf in loss; head split unlikely to help.
- Contrastive pre-training: needs unlabelled data preprocessing, multi-phase.
- Physics-residual loss (continuity): requires velocity divergence from gradients; gradient computation over padded mesh is expensive and tricky.
- Separate encoder per zone: mesh has no zone labels in the contract.
- RANS turbulence feature augmentation: Re is already encoded; log(Re) in dim 13.
- KAN (Kolmogorov-Arnold Networks): very new, no CFD evidence, too speculative for short run.
- Neural process meta-learning: needs batching over tasks, incompatible with current loop.

Retained 12 below, ordered roughly by expected risk/reward.

---

## Hypothesis 1: Raised Surface Weight (20 or 30) — Loss Rebalancing

### What it is
Increase `surf_weight` from 10 to 20 or 30, trading volume accuracy for surface pressure accuracy, which directly optimises the primary metric.

### Why it might help
The primary metric is surface pressure MAE only. Volume nodes are ~95% of all nodes but contribute zero to the leaderboard metric. A surf_weight of 10 was not tuned; simply doubling or tripling it should directly improve surface predictions at modest cost to volume accuracy. This is the cheapest, most direct lever available and should be tested first.

### Dataset-specific rationale
Surface nodes are sparse (~5% of mesh nodes on average). With batch_size=4 and meshes of 100K-240K nodes, a single batch has ~5K-12K surface nodes vs ~95K-228K volume nodes. The current 10x weight may still under-emphasise surface nodes relative to their metric importance. The cruise domain has larger meshes (210K nodes) so the imbalance is largest there, and val_geom_camber_cruise is likely the hardest split.

### Code changes in train.py
```python
# In Config dataclass — only change:
surf_weight: float = 20.0  # raise from 10.0; test 20 and 30 as two arms
```

### Predicted delta
surf_weight=20: ~3-8% relative improvement on val_avg/mae_surf_p
surf_weight=30: potentially more, but risk of vol_loss diverging and destabilising training

### Risks
Vol loss quality degrades (less physically interpretable). If the attention mechanism relies on volume nodes to build a good global representation and surf_weight is too high, training may destabilise or surface loss may overfit first.

### CLI
```bash
python train.py --surf_weight 20 --experiment_name surf_weight_20 --agent <student>
python train.py --surf_weight 30 --experiment_name surf_weight_30 --agent <student>
```

### Assigned to: alphonse

---

## Hypothesis 2: Stochastic Weight Averaging (SWA) — Free Ensemble at Zero Extra Cost

### What it is
Apply Stochastic Weight Averaging (SWA, Izmailov et al. 2018) in the last portion of training. SWA averages model weights across several checkpoints in the cosine LR valley, producing a flatter minimum that generalises better — at zero inference cost.

### Why it might help
SWA consistently improves generalisation on OOD test sets in vision and, recently, neural PDE surrogates (NeurIPS 2024 review). Our val_geom_camber splits are geometry-OOD holdouts, which is exactly where weight averaging tends to help most — it escapes sharp minima that fit training geometry well but do not generalise. The 30-min cap means we get ~50 epochs; SWA on the last 20 epochs (from epoch 30 onwards) adds no extra forward passes.

### Dataset-specific rationale
The geometry-OOD splits (val_geom_camber_rc, val_geom_camber_cruise) are the hardest generalisation axes. SWA is known to widen the basin of attraction, producing smoother parameter spaces that generalise across geometric variation. val_re_rand is also affected since Re variation drives sharp loss landscape features.

### Code changes in train.py
```python
# After optimizer definition, add:
from torch.optim.swa_utils import AveragedModel, SWALR

SWA_START_EPOCH = 25  # start averaging at epoch 25 (halfway through 50)
SWA_LR = 5e-5         # SWA LR, ~10x lower than peak
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)
swa_active = False

# Inside the epoch loop, after scheduler.step():
if (epoch + 1) >= SWA_START_EPOCH:
    swa_active = True
    swa_model.update_parameters(model)
    swa_scheduler.step()
    # replace cosine step with SWA LR after SWA_START_EPOCH
    # (keep cosine scheduler stepping before SWA_START_EPOCH; comment out after)

# After training loop, before test eval:
if swa_active:
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    # Evaluate using swa_model instead of model for best checkpoint selection
    # Load swa_model weights into model for test evaluation compatibility
    model.load_state_dict(swa_model.module.state_dict())
```

Note: checkpoint selection during SWA phase should evaluate `swa_model`, not `model`. The swa_model.module holds the averaged weights. After `update_bn`, copy weights back into `model` for the standard test eval path.

### Predicted delta
2-6% relative improvement on val_avg/mae_surf_p, particularly on OOD geometry splits.

### Risks
BN update step (`update_bn`) requires a full pass over the train loader, which costs extra time within the 30-min cap. For Transolver (which uses LayerNorm, not BatchNorm), update_bn is a no-op, so this cost is zero. The main risk is that SWA averaging starts too late or too early — if the model hasn't converged by epoch 25, SWA averages garbage. Test with SWA_START_EPOCH=30 as a safer variant.

### CLI
```bash
python train.py --experiment_name swa_start25 --agent <student>
```
(SWA parameters are hardcoded in the modified train.py for this PR)

### Assigned to: askeladd

---

## Hypothesis 3: Per-Sample Instance Normalisation — Addressing High-Dynamic-Range Targets

### What it is
Before computing the MSE loss, additionally normalise predictions and targets by per-sample standard deviation (instance normalisation). The global y_std is already applied globally; per-sample instance norm corrects for the fact that high-Re samples have ~10x larger target variance than low-Re samples.

### Why it might help
Per-sample y std varies by 1-2 orders of magnitude (val_single_in_dist: avg 458, max 2077; val_geom_camber_cruise: avg 164, max 506). In a global-normalised MSE, high-Re samples dominate the loss gradient by orders of magnitude, making the model over-specialise for high-Re regimes and under-optimise for low-Re regimes. Instance normalisation equalises this, giving every sample equal effective learning signal regardless of Re. This is a standard technique in distribution-shift regression (arXiv:2401.16777).

### Dataset-specific rationale
val_re_rand spans the full Re range (110K-5M). The stratified Re holdout is explicitly testing whether the model generalises across Re regimes. A model that over-fits high-Re samples will fail exactly on the low-Re portion of val_re_rand. Instance norm attacks this directly.

### Code changes in train.py
```python
# In the training loop, inside the per-batch section, after computing sq_err:
# Compute per-sample, per-channel std over REAL nodes only (mask=True)
# y_norm shape: [B, N, 3]; mask shape: [B, N]
with torch.no_grad():
    # per sample, per channel std over valid nodes
    mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
    n_valid = mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
    y_mean_sample = (y_norm * mask_f).sum(dim=1) / n_valid  # [B, 3]
    y_var_sample = ((y_norm - y_mean_sample.unsqueeze(1)) ** 2 * mask_f).sum(dim=1) / n_valid
    y_std_sample = y_var_sample.sqrt().clamp(min=1e-6)  # [B, 3]
    # rescale loss contribution
    instance_scale = 1.0 / y_std_sample  # [B, 3]

# When computing vol_loss and surf_loss, weight sq_err by instance_scale:
sq_err_scaled = sq_err * instance_scale.unsqueeze(1)  # broadcast over N
vol_loss = (sq_err_scaled * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_scaled * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
# NOTE: validation metrics remain in original physical space (scoring.py unchanged)
```

The instance scale should be computed in normalised space so that the loss units remain consistent. An alternative simpler form: normalise each sample's contribution by its total number of valid nodes (already done) — but this doesn't address dynamic range. The above adds dynamic-range correction on top.

### Predicted delta
3-10% relative improvement, primarily on val_re_rand and val_single_in_dist (which spans the widest Re range in training). May slightly hurt OOD geometry splits if those are dominated by a different error source.

### Risks
Computing per-sample stats in the training loop adds a small overhead (~5% per batch). The instance normalisation may destabilise training if implemented carelessly — ensure `clamp(min=1e-6)` to prevent division by near-zero std for nearly-uniform-field samples. Also, this changes the effective loss weighting and may interact with `surf_weight`; keep surf_weight=10 (baseline) in this PR.

### CLI
```bash
python train.py --experiment_name instance_norm_loss --surf_weight 10 --agent <student>
```

### Assigned to: edward

---

## Hypothesis 4: Separate Per-Channel Surface Loss with Pressure Head — Loss Decomposition

### What it is
Decompose the surface loss into three separate terms (Ux, Uy, p) with different weights, giving pressure `p` a higher multiplier than velocity channels (which are less important to the primary metric).

### Why it might help
The current surf_loss is a mean over all 3 channels equally weighted. But the primary metric is `mae_surf_p` only — surface pressure. Velocity channels add noise to the surface loss signal. By weighting the surface pressure channel more heavily, we directly optimise the loss for the leaderboard metric. This is analogous to task-specific loss weighting in multi-task learning.

### Dataset-specific rationale
Pressure coefficient Cp drives the aerodynamic forces. For a CFD surrogate to be physically useful, pressure accuracy matters most. The velocity accuracy on surface nodes is less physically critical (no-slip BC already constraints tangential velocity to near-zero). Giving pressure a 3-5x multiplier within the surface loss focuses the model on what the metric cares about.

### Code changes in train.py
```python
# In the training loop, replace the surf_loss computation:
SURF_P_WEIGHT = 3.0   # extra weight on pressure channel within surface loss
surf_channel_weights = torch.tensor([1.0, 1.0, SURF_P_WEIGHT], device=device)

# sq_err: [B, N, 3]
surf_sq_err_weighted = sq_err * surf_channel_weights[None, None, :]  # broadcast
surf_loss = (surf_sq_err_weighted * surf_mask.unsqueeze(-1)).sum() / (
    surf_mask.sum().clamp(min=1) * surf_channel_weights.mean()
)
# Keep vol_loss unchanged
loss = vol_loss + cfg.surf_weight * surf_loss
```

Try SURF_P_WEIGHT values of 3.0 and 5.0 as two arms.

### Predicted delta
3-8% relative improvement on val_avg/mae_surf_p, with possible slight degradation on vol metrics (acceptable since those aren't the primary metric).

### Risks
The normalisation of surf_loss must be adjusted carefully (divide by mean channel weight to keep the loss on a comparable scale). If not normalised, surf_weight=10 effectively becomes surf_weight=10*avg_channel_weight, changing the volume/surface balance.

### CLI
```bash
python train.py --surf_weight 10 --experiment_name surf_p_weight3 --agent <student>
# SURF_P_WEIGHT=3.0 hardcoded in modified train.py
```

### Assigned to: fern

---

## Hypothesis 5: Wider Model — n_hidden=192 or 256 — Capacity Increase

### What it is
Increase n_hidden from 128 to 192 or 256, widening the Transolver's internal representation. This increases model capacity with minimal architectural change.

### Why it might help
The baseline model at n_hidden=128 has ~2.1M params. The physics attention operates over slice_num=64 tokens per layer; at n_hidden=128, each attention head has dim_head=32 (128/4 heads), which is small. Increasing to n_hidden=192 gives dim_head=48 and ~4.7M params; n_hidden=256 gives dim_head=64 and ~8.3M params. The geometry-OOD splits require interpolating unseen camber shapes — more capacity allows richer intermediate representations. With 96GB VRAM and batch=4, n_hidden=256 is easily feasible (estimated peak ~14GB for the largest meshes).

### Dataset-specific rationale
The tandem configurations (cruise domain, 210K nodes) combine complex near-field interactions between two foils at variable separation. A narrow hidden dim may simply not have enough capacity to represent the interference patterns at unseen camber values (val_geom_camber_cruise).

### Code changes in train.py
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=192,   # raised from 128; try 192 and 256 as two arms
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Note: n_head=4 requires n_hidden divisible by 4. 192 and 256 both satisfy this. dim_head = 192/4 = 48 and 256/4 = 64.

### Predicted delta
n_hidden=192: 2-6% relative improvement (risk: slower per-epoch, may get fewer epochs in 30 min)
n_hidden=256: 3-8% relative improvement (higher risk of timeout with fewer epochs)

### Risks
Training throughput decreases quadratically with n_hidden in attention layers (slice_num x slice_num attention stays fixed; the bottleneck is the linear projections which scale as n_hidden^2). With 30-min timeout and cruise meshes at 210K nodes, n_hidden=256 may only complete 20-25 epochs vs ~50 for n_hidden=128. Start with n_hidden=192.

### CLI
```bash
python train.py --experiment_name nhidden192 --agent <student>
# n_hidden=192 hardcoded in model_config
```

### Assigned to: frieren

---

## Hypothesis 6: More Slices — slice_num=128 — Finer Physics Partitioning

### What it is
Increase slice_num from 64 to 128, giving the PhysicsAttention module twice as many soft tokens to route mesh nodes to. This increases the representational granularity of the physics state space.

### Why it might help
The PhysicsAttention soft-assigns each of N nodes to S=slice_num tokens. With slice_num=64 over 200K nodes, each token represents ~3,100 nodes on average. For tandem foil flows, the two foils have distinct pressure distributions and the wake interaction creates a complex shear layer. More slices allow the model to distinguish finer physical structures (leading edge stagnation, suction peak, wake recovery, ground effect for raceCar) without interference between regions. The original Transolver paper uses slice_num=32-64; a larger value is an unexplored lever in this setting.

### Dataset-specific rationale
The cruise tandem domain (210K nodes) has the largest meshes and the most complex two-foil interactions. With slice_num=64 and 4 heads, each head uses 16 slices — barely enough to represent the distinct flow regions (leading edge, suction peak, pressure side, wake, zone boundaries).

### Code changes in train.py
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,   # raised from 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

The attention step over slice tokens is O(slice_num^2), so 128 vs 64 quadruples the attention FLOPS over tokens. But since slice_num << N (~200K nodes), this is negligible vs the O(N) aggregation/broadcast steps.

### Predicted delta
3-7% relative improvement, especially on geometry-OOD splits where finer physical partitioning helps generalise.

### Risks
Memory increases linearly with slice_num (the slice_token tensor is [B, heads, slice_num, dim_head]). At slice_num=128 with B=4 and n_hidden=128, this is 4*4*128*32 = 65K floats — negligible. No VRAM risk. The main risk is that the orthogonal init on `in_project_slice.weight` may become harder to satisfy with 128 slices (the weight matrix is [dim_head, slice_num] = [32, 128], which is a fat matrix; orthogonal init on columns, not rows, so it remains valid).

### CLI
```bash
python train.py --experiment_name slice128 --agent <student>
```

### Assigned to: nezuko

---

## Hypothesis 7: Gradient Clipping — Training Stability and Better Generalisation

### What it is
Add gradient clipping with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before the optimizer step. The baseline has no gradient clipping.

### Why it might help
Large meshes with high-Re samples produce high-variance gradients. Without clipping, occasional large gradient events can kick the model out of good minima — this is particularly harmful in the cosine LR valley near epoch 40-50 where the model is refining. Gradient clipping is standard practice for transformer training (all LLMs use it). In physics-aware attention models where the softmax temperature is a learnable parameter, un-clipped gradients can cause the temperature to jump sharply, destabilising the slice routing. Clipping max_norm=1.0 is the standard transformer default.

### Dataset-specific rationale
The Re range spans 100K to 5M — a 50x variation in Reynolds number drives large output magnitude differences (per-sample y std varies from ~160 to ~2000). This directly translates to large gradient variance in the loss. Gradient clipping smooths this out.

### Code changes in train.py
```python
# In the training loop, between loss.backward() and optimizer.step():
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Also optionally: log the pre-clip gradient norm to metrics.jsonl to verify clipping is actually firing:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# append grad_norm.item() to epoch metrics
```

### Predicted delta
1-4% relative improvement, primarily through more stable training trajectories. May allow a slightly higher lr as a follow-up experiment.

### Risks
Gradient clipping can slow convergence if the model legitimately needs large gradient steps early in training. The cosine LR already handles this by decaying LR; clipping is most beneficial in the mid-to-late epochs when the model is near convergence. Risk is low given this is a standard transformer practice. Very unlikely to hurt.

### CLI
```bash
python train.py --experiment_name grad_clip_1 --agent <student>
```

### Assigned to: tanjiro

---

## Hypothesis 8: Surface-Conditioned Auxiliary Skip Connection — Physics Branch

### What it is
Add a lightweight auxiliary skip connection from input features directly to the final output prediction for surface nodes only, bypassing the transformer blocks. This gives the model a direct path to use local physical features (saf, dsdf, is_surface) without having to propagate them through 5 transformer layers.

### Why it might help
The PhysicsAttention operates globally over all nodes. For surface pressure, many features are local: signed arc-length (dim 2-3), distance-based shape descriptors (dim 4-11), and the surface indicator (dim 12). A direct linear skip from these local features to the surface pressure prediction provides a fast residual path. This is analogous to the bypass connections in U-Net and the direct forcing terms in physics-informed networks. The Boundary GNN paper (arXiv:2503.18638) found that local physical features alone can explain a significant fraction of surface pressure variation; giving the model a direct path to use them is efficient.

### Dataset-specific rationale
For the geometry-OOD splits (val_geom_camber_rc, val_geom_camber_cruise), the unseen front-foil camber (M=6-8, M=2-4) changes the shape descriptors (dsdf features, dims 4-11) and the arc-length distribution (saf, dims 2-3). A direct skip that leverages these features avoids the model having to learn the mapping through the attention mechanism purely from high-level slice tokens.

### Code changes in train.py
```python
# New module:
class SurfaceSkip(nn.Module):
    """Lightweight auxiliary skip for surface nodes: raw features -> partial output."""
    def __init__(self, in_dim=24, out_dim=3, hidden=32):
        super().__init__()
        # Only use geometry-relevant features: dims 0-11 (position, saf, dsdf)
        # plus dim 13 (log Re), dim 14 (AoA foil 1), dims 15-17 (NACA foil 1)
        self.skip_dims = list(range(12)) + [13, 14, 15, 16, 17]  # 18 dims
        self.net = nn.Sequential(
            nn.Linear(len(self.skip_dims), hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)  # init to zero so model starts at baseline
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_norm, is_surface):
        # x_norm: [B, N, 24]; is_surface: [B, N]
        x_local = x_norm[..., self.skip_dims]  # [B, N, 18]
        skip = self.net(x_local)  # [B, N, 3]
        # Only apply on surface nodes
        return skip * is_surface.float().unsqueeze(-1)

# In Transolver.forward():
def forward(self, data, **kwargs):
    x = data["x"]
    is_surface = data.get("is_surface", None)  # optional
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks:
        fx = block(fx)
    if is_surface is not None and hasattr(self, 'surf_skip'):
        fx = fx + self.surf_skip(x, is_surface)
    return {"preds": fx}
```

In the training/eval loops, pass `is_surface` to the model:
```python
pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
```

Note: model contract is preserved (input dict, output dict with "preds").

The skip module is initialised with zero output weights so the model starts exactly at the baseline; the skip activates gradually as training proceeds.

### Predicted delta
2-7% relative improvement on geometry-OOD splits. Risk of marginal gain if the transformer already learns this implicitly.

### Risks
Requires modifying the model's forward call signature. The evaluation loop (`evaluate_split`) also needs to pass `is_surface` to the model. Ensure both training and validation paths are updated consistently. Zero init on skip output guarantees no regression at initialisation.

### CLI
```bash
python train.py --experiment_name surf_skip_branch --agent <student>
```

### Assigned to: thorfinn

---

## Hypothesis 9: Deeper Model — n_layers=7 — More Propagation Steps

### What it is
Increase n_layers from 5 to 7, giving the model 2 additional transformer blocks. This increases the depth of feature propagation across slice tokens.

### Why it might help
The PhysicsAttention is a global operation over S=64 slice tokens. With 5 layers, information can propagate across the entire mesh 5 times. For tandem foils where the wake from foil 1 interacts with foil 2 several chord lengths downstream, deeper propagation may be needed to capture multi-scale interactions. Each additional layer adds one more round of global slice-space attention followed by per-node MLP refinement. Increasing depth from 5 to 7 adds ~40% more compute but should still fit in the 30-min window.

### Dataset-specific rationale
The gap and stagger parameters (dims 22-23) define the foil-to-foil distance in tandem configurations. At large gap/stagger values, the foil interaction is weaker and the flow physics decouple — but at small gaps (ground effect raceCar domain), the interaction is strong. Deeper networks are better at representing these conditional, non-local correlations.

### Code changes in train.py
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=7,   # raised from 5
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

May benefit from a slightly lower LR due to deeper residual stack; optionally set lr=3e-4:
```bash
python train.py --lr 3e-4 --experiment_name nlayers7 --agent <student>
```

### Predicted delta
2-5% relative improvement. Risk of slower convergence requiring more epochs (mitigated by the fact that the cosine schedule adapts to T_max=epochs automatically).

### Risks
Two extra layers increase per-epoch training time ~40%. With a 30-min cap, this means ~35 epochs instead of ~50. On the positive side, Transolver architectures typically benefit from depth up to ~8 layers per the original paper.

### CLI
```bash
python train.py --lr 3e-4 --experiment_name nlayers7 --agent <student>
```

---

## Hypothesis 10: Warmup + Cosine LR Schedule — Better Early Training

### What it is
Replace the bare CosineAnnealingLR with a warmup + cosine schedule: linearly ramp LR from 0 to peak over the first 5 epochs, then cosine anneal to 0 over the remaining epochs.

### Why it might help
The baseline immediately uses lr=5e-4 from epoch 1. With large meshes (up to 242K nodes) and highly variable gradient magnitudes across domains, the first few training steps can destabilise the attention temperature parameters and the orthogonal slice routing. A warmup phase prevents large-gradient early updates from locking the model into a bad basin. This is universal practice in transformer training and is an obvious missing ingredient in the baseline.

### Dataset-specific rationale
The WeightedRandomSampler ensures balanced domain sampling, but in the first few batches (before the model has seen all domains), the gradient signal is unrepresentative of the final distribution. Warmup prevents those first steps from having disproportionate impact. The cruise domain (210K nodes) produces very different gradient scales than the raceCar single domain (85K nodes).

### Code changes in train.py
```python
# Replace the scheduler definition:
WARMUP_EPOCHS = 5
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / WARMUP_EPOCHS
    # cosine from warmup end to 0
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

import math
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# Remove: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
```

Note: `scheduler.step()` is already called once per epoch at the end of the training loop — no change needed there.

### Predicted delta
1-4% relative improvement, primarily through more stable early training and avoidance of bad early optima.

### Risks
The LambdaLR formulation must be verified: at epoch 0 (first call), lr_lambda(0) = 1/WARMUP_EPOCHS, not 0 — because `scheduler.step()` is called after the first epoch completes, so the optimizer starts at the initial lr. The ramp is: epoch 0 after step → lr_lambda(0) = 1/5 = 0.2; epoch 4 after step → 1.0; epoch 5 onward → cosine decay. This gives peak LR at end of warmup, which is the desired behaviour. Also add `import math` at the top of train.py.

### CLI
```bash
python train.py --experiment_name warmup5_cosine --agent <student>
```

---

## Hypothesis 11: Higher Batch Size + BF16 Mixed Precision — More Samples Per Step

### What it is
Increase batch_size from 4 to 8 (better gradient estimates per step) and enable BF16 mixed precision training via `torch.amp.autocast`. On A100/H100 class GPUs with 96GB VRAM, BF16 halves memory bandwidth pressure and enables larger effective batch.

### Why it might help
With batch_size=4 and meshes of 74K-242K nodes, each batch provides only 4 gradient examples. Increasing to batch_size=8 doubles the number of samples contributing to each gradient estimate, reducing noise particularly for the balanced domain sampler (which draws from 3 domains — with batch=4, some batches may draw 3 samples from one domain and 1 from another; with batch=8, the balance is more reliable). BF16 autocast reduces memory ~40%, making batch=8 feasible even for the 242K-node meshes.

### Dataset-specific rationale
The WeightedRandomSampler draws samples proportional to domain weights. With batch_size=4 and 3 domains equally weighted, the effective per-domain samples per batch is ~1.3 — very noisy. With batch_size=8, it's ~2.7 — still small but significantly better. This should most benefit val_re_rand (which requires generalising across domains and Re regimes) and the tandem OOD splits.

### Code changes in train.py
```python
# Change Config default:
batch_size: int = 8   # raised from 4

# Add autocast:
# In training loop:
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (y - stats["y_mean"]) / stats["y_std"]
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred - y_norm) ** 2
    vol_loss = ...
    surf_loss = ...
    loss = vol_loss + cfg.surf_weight * surf_loss

# In eval loop (evaluate_split), also wrap with autocast:
with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    pred = model({"x": x_norm})["preds"]
```

Note: do NOT use GradScaler with BF16 (only needed for FP16). BF16 has wider dynamic range than FP16 and does not require scaling.

The `pred * stats["y_std"] + stats["y_mean"]` denormalization in evaluate_split should cast pred to float32 before denorm to avoid precision loss in metric accumulation (scoring.py uses float64):
```python
pred_orig = pred.float() * stats["y_std"] + stats["y_mean"]
```

### Predicted delta
2-5% relative improvement from better gradient estimates. BF16 may cause 0.5-1% precision degradation which could offset gains — test carefully.

### Risks
The padded variable-mesh batches (up to 242K nodes) with batch_size=8 require ~14-18GB peak VRAM. This is well within 96GB. The main risk is BF16 precision for the loss and gradient computation — the LayerNorm in Transolver accumulates in FP32 by default (PyTorch autocast keeps LN in FP32), so this should be safe.

### CLI
```bash
python train.py --batch_size 8 --experiment_name bs8_bf16 --agent <student>
```

---

## Hypothesis 12: Transolver++ Local Adaptive Mechanism — Physics State Refinement

### What it is
Implement the core Transolver++ (ICML 2025) local adaptive mechanism: after each global slice-space attention step, add a local MLP that conditions the per-node update on the node's local neighbourhood statistics (e.g., the node's own input features as a residual bias). This bridges global attention outputs with local geometric context.

### Why it might help
Transolver++ reports a 13% relative improvement over vanilla Transolver across multiple PDE benchmarks (ICML 2025). The key insight is that the global slice attention is expressive for smooth flow regions but misses local corrections near boundaries and sharp gradients (leading edge stagnation, separation bubbles). Adding a local adaptive residual — essentially a position-conditioned bias added after each attention step — allows the model to correct global predictions with local context. This is the most directly relevant architectural improvement from the recent literature.

### Dataset-specific rationale
Surface pressure on airfoils has sharp spatial gradients near the leading edge (suction peak) and near the trailing edge (pressure recovery / separation). The global slice attention averages these into slice tokens, potentially blurring them. A local adaptive correction restores this fine-scale resolution.

### Code changes in train.py

Add a `LocalAdaptiveMLP` to each `TransolverBlock`:

```python
class LocalAdaptiveMLP(nn.Module):
    """Local correction term from Transolver++.
    Conditions per-node output on the node's own input features."""
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # init to small values so model starts near baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_input, fx_global):
        """
        x_input: original pre-processed node features [B, N, hidden_dim]
        fx_global: output from global attention [B, N, hidden_dim]
        returns: fx_global + local_correction [B, N, hidden_dim]
        """
        return fx_global + self.net(x_input)
```

Modify `TransolverBlock.__init__` to accept `input_dim` and add a `LocalAdaptiveMLP`:
```python
class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 use_local_adapt=True):
        ...
        if use_local_adapt:
            self.local_adapt = LocalAdaptiveMLP(hidden_dim, hidden_dim, dropout)
```

Modify `TransolverBlock.forward` to accept original input `x0` and apply local correction:
```python
def forward(self, fx, x0=None):
    fx = self.attn(self.ln_1(fx)) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    if hasattr(self, 'local_adapt') and x0 is not None:
        fx = self.local_adapt(x0, fx)
    if self.last_layer:
        return self.mlp2(self.ln_3(fx))
    return fx
```

In `Transolver.forward()`, store the preprocessed input and pass as `x0`:
```python
def forward(self, data, **kwargs):
    x = data["x"]
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    x0 = fx  # preserve preprocessed features as local conditioning signal
    for block in self.blocks:
        fx = block(fx, x0=x0)
    return {"preds": fx}
```

### Predicted delta
5-15% relative improvement based on Transolver++ paper results. This is the highest-expected-impact architectural change in this list, but also the most complex to implement correctly.

### Risks
The local adaptive mechanism in Transolver++ uses the node's coordinates/features, not the hidden state, as the conditioning signal. The implementation above uses the preprocessed hidden state as x0 — this is a reasonable approximation but may differ from the exact paper formulation (which conditions on raw input coordinates). Zero init on the local_adapt output layer ensures the model starts at the baseline and trains the adaptation from scratch. Adding this to all 5 layers approximately doubles the number of MLP parameters; check that training still fits in 30 min (likely fine since the local MLP is [128,128,128] = small).

### CLI
```bash
python train.py --experiment_name transolver_plus_plus --agent <student>
```

---

## Assignment Summary

| Student | Hypothesis | Key Change | Expected Risk |
|---|---|---|---|
| alphonse | H1: Raised Surface Weight | surf_weight=20/30 | Low |
| askeladd | H2: SWA | Weight averaging from epoch 25 | Low-Medium |
| edward | H3: Instance Norm Loss | Per-sample loss normalisation | Medium |
| fern | H4: Per-Channel Surface Loss | P channel gets 3x weight in surf_loss | Low |
| frieren | H5: Wider Model (n_hidden=192) | Capacity increase | Low-Medium |
| nezuko | H6: More Slices (slice_num=128) | Finer physics partitioning | Low |
| tanjiro | H7: Gradient Clipping | max_norm=1.0 before optim step | Very Low |
| thorfinn | H8: Surface Skip Branch | Local bypass for surface nodes | Medium |

**Reserve hypotheses** (assign to students that report early, or as follow-ups):
- H9: Deeper Model (n_layers=7) — assign if H5/H6 succeeds and more capacity helps
- H10: Warmup + Cosine LR — assign if H7 (gradient clipping) shows the baseline is unstable early
- H11: Batch=8 + BF16 — assign if throughput is a bottleneck after seeing epoch timings
- H12: Transolver++ local adaptive — assign as a medium-risk architectural PR after cheapest wins are merged

---

## Recommended Priority Order

1. H1 (surf_weight) — cheapest, most direct, test first
2. H7 (grad clipping) — zero downside, likely 1-4% free gain
3. H4 (per-channel loss) — direct metric alignment
4. H2 (SWA) — free improvement at end of training
5. H3 (instance norm) — addresses core Re dynamic range issue
6. H6 (slice_num=128) — clean architectural extension
7. H5 (wider model) — more compute, higher expected gain
8. H8 (surface skip) — more engineering, medium risk
9. H9-H12 — follow-up based on early results

---

## Research State

**Current best explanation of what limits progress:** The primary metric is surface pressure MAE but (a) the loss function weights all 3 channels equally within the surface term, (b) the high Re dynamic range causes gradient dominance by a few extreme samples, and (c) the model has not been tested with any ensemble/averaging technique. All three are addressable cheaply before touching architecture.

**Ruled out before starting:** FNO backbone (too slow), graph attention (edge construction outside data contract), diffusion model (no physical prior fit), curriculum by mesh size (read-only loader), BN-based normalization (model uses LN), separate zone encoders (no zone labels).

**Open uncertainties:**
1. How many epochs does the baseline complete in 30 min? This determines whether depth increases (H9) are feasible or simply run fewer epochs.
2. What is the baseline val_avg/mae_surf_p? Needed to judge if any hypothesis "wins". First student to finish establishes this floor.
3. Are the OOD geometry splits (camber holdouts) harder because of capacity limits or because the model lacks local geometry sensitivity? This determines whether capacity (H5/H6) or skip connections (H8) are the right lever.
