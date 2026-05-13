# Research Ideas — 2026-05-13 18:00
# Tag: charlie-pai2g-48h-r5 | No W&B

**Current baseline**: `val_avg/mae_surf_p = 42.3455` (PR #2307)
**Per-split**: in_dist=35.4776, camber_rc=60.8311 (OOD bottleneck), camber_cruise=27.6517, re_rand=45.4214

**In-flight**: #2439 frieren GeGLU FFN; #2472 tanjiro split-surf-vol-heads

---

## Idea A — Auxiliary Camber Prediction Head (nezuko ASSIGNED)

**Mechanism**: Add a small auxiliary prediction head on the intermediate latent that
predicts the NACA camber parameter of the active foil (features 15 and 19, already in
x). Pool the per-node latent via masked mean over all real nodes, then project to 1
scalar. MSE auxiliary loss weighted at λ=0.1 added to the main L1 loss. Forces the
internal representation to explicitly encode geometric camber identity, which gives the
model a stronger inductive signal for the held-out camber range M=6–8 in camber_rc.

The key distinction from FiLM (closed 14th taxon): FiLM injects conditioning into the
residual stream as a learned affine transform of the *input* features. Auxiliary head
supervision pushes the *intermediate latent* to encode geometry explicitly via a gradient
signal from the decoder side — a pull rather than a push. FiLM adds parameters to every
layer; this adds ~300 parameters total (one Linear + scalar head).

**Distinctiveness**: No overlap with closed taxa 1–16. In-flight #2439 and #2472 touch
FFN structure and output projections respectively; this touches the intermediate pooling
path only. FiLM closes taxon 14 (residual-stream-input-perturbation conditioning), which
operates before attention. This operates after attention on the post-norm latent.

**Predicted direction**: Primarily targets camber_rc −5% to −10%. May be neutral on
other splits. If camber_rc improves enough the avg metric wins despite minor regression
elsewhere (camber_rc contributes ~34% of total avg). Stop condition: camber_rc improves
by < 2% OR any split regresses > 8%.

**Counter-argument**: If the model has already saturated geometric encoding capacity in
its 96-hidden latent, adding a supervision signal on a pooled projection may not add new
representational content. Also, NACA camber is already present as a raw input feature —
the model could in principle already attend to it. The auxiliary loss helps only if the
current optimization fails to route geometry through to surface predictions.

**Param delta**: ~+300 params (<0.1%). No architectural change to main path.

**Implementation (train.py)**:
```python
# In Model.__init__ (after blocks):
self.camber_head = nn.Linear(hidden_dim, 1)

# In Model.forward (after blocks loop, before final decoder):
# fx shape: [B, N, hidden_dim]; mask: [B, N] bool
node_latent = fx  # pre-decoder latent
pooled = (node_latent * mask.unsqueeze(-1).float()).sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1)
camber_pred = self.camber_head(pooled).squeeze(-1)  # [B]

# In training loop loss block:
# camber target = mean of foil-1 and foil-2 camber from raw x (dim 15 and 19):
# normalize by x_std/x_mean (already done if using x_norm) or use raw feature
camber_gt = x[:, :, 15]  # raw (not normalized), take first real node
# Since all nodes share the same global condition, any valid node index works:
camber_gt_val = (x * mask.unsqueeze(-1).float()).sum(1)[:, 15] / mask.float().sum(1).clamp(min=1)
# Normalize to roughly [0,1] range (NACA camber M=0-9, feature is [0,9]/normalizer)
camber_loss = F.mse_loss(camber_pred, camber_gt_val / 9.0)
loss = vol_loss + cfg.surf_weight * surf_loss + cfg.aux_camber_weight * camber_loss
```

Add `aux_camber_weight: float = 0.1` to Config dataclass. Use `--aux_camber_weight 0.1`.

**CLI**: `python train.py --experiment_name aux-camber-head-lam01 --aux_camber_weight 0.1 --epochs 70`

---

## Idea B — Per-Channel Surface Loss Weight (fern ASSIGNED)

**Mechanism**: The primary metric is `mae_surf_p` (surface pressure MAE only). Yet the
training loss sums L1 error across all 3 channels (Ux, Uy, p) equally before applying
surf_weight=10. The model has no incentive to prioritize pressure over velocity at the
surface. A channel weight vector applied to `sq_err` before masked summation lets the
optimizer focus more gradient signal on the pressure channel specifically.

Current loss path (train.py lines 539–545):
```
sq_err = F.l1_loss(pred, y_norm, reduction='none')  # [B, N, 3]
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum()
```
The `.unsqueeze(-1)` broadcasts surf_mask [B,N] to [B,N,1] — all 3 channels are summed
equally. Inserting a channel weight tensor `w = [1.0, 1.0, w_p]` broadcast to [1,1,3]
before summation is a one-line change that applies differential pressure emphasis.

**Distinctiveness**: surf_weight axis is FULLY CLOSED at scalar 10 (PRs #2394, #2387,
earlier). Per-channel ratio is orthogonal: it changes the relative gradient between
channels within each surface node, not the surface-vs-volume trade-off. No closed taxon
covers this. Orthogonal to in-flight #2439 (FFN structure) and #2472 (decoder routing).

**Predicted direction**: Should improve surface pressure MAE specifically. Velocity MAE
may increase slightly as gradient is redistributed. Net effect depends on whether pressure
and velocity share representational capacity or compete for it at the surface nodes.

**Counter-argument**: y_norm is already in normalized space (per-channel mean/std removed
via stats.json). If normalization is well-calibrated, the 3 channels already have similar
loss scales and uniform weighting IS the right inductive bias. Differential weighting may
just over-fit to pressure without improving physical consistency.

**Param delta**: 0 parameters. Pure loss change.

**Implementation (train.py lines 539–545)**:
```python
# Add to Config:
# p_channel_weight: float = 3.0

# In training loop, replace surf_loss computation:
ch_w = torch.tensor([1.0, 1.0, cfg.p_channel_weight], device=pred.device, dtype=pred.dtype)
sq_err = F.l1_loss(pred, y_norm, reduction='none')  # [B, N, 3]
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * ch_w.unsqueeze(0).unsqueeze(0) * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Note: vol_loss keeps uniform weighting to not disturb volume training. Only surf_loss gets
differential channel weighting. Keep surf_weight=10.0 (closed axis — don't change it).

**Sweep**: try `w_p` in {2.0, 3.0, 5.0}. Recommend starting at 3.0 as single arm.

**CLI**: `python train.py --experiment_name per-channel-surf-p3 --p_channel_weight 3.0 --epochs 70`

---

## Idea C — Layer-wise Learning Rate Decay (edward ASSIGNED)

**Mechanism**: Deep transformers exhibit a well-known gradient flow asymmetry: earlier
layers receive smaller gradients and often under-adapt relative to later layers. LLRD
(layer-wise LR decay) assigns each layer group a different LR: the output decoder gets
the base LR, and each earlier block gets base LR × decay^k for some decay ∈ (0,1).

For our 4-block Transolver with n_layers=4:
- Preprocess MLP (embedding): lr × decay^4 = lr × 0.24
- Block 0 (earliest):          lr × decay^3 = lr × 0.34
- Block 1:                     lr × decay^2 = lr × 0.49
- Block 2:                     lr × decay^1 = lr × 0.70
- Block 3 + decoder:           lr × 1.0

This is fundamentally distinct from the closed lr-DOWN taxon (PR #2325 etc.) which
uniformly scaled ALL parameters. LLRD preserves the later-layer LR while reducing
early-layer LR — it makes optimization geometry-aware.

The most recent LayerScale diagnostics show gamma_mlp >> gamma_attn across all blocks,
meaning MLP branches dominate the learned signal. LLRD combined with this suggests the
optimizer may be under-adapting the early attention layers. Slower LR in early layers
reduces interference while faster convergence of later layers is preserved.

**Distinctiveness**: lr-DOWN (global uniform) is closed as taxon 1 bimodal. Per-layer
differential is orthogonal — no closed taxon covers it. In-flight PRs use standard flat
LR. Distinct from warmup (duration axis), β1 (momentum axis), and amsgrad (statistic axis).

**Predicted direction**: Improvement if early layers are currently over-corrected by the
adaptive optimizer. May be neutral if the current flat LR is already optimal across all
depths. Stop condition: any split regresses by > 5% vs baseline.

**Counter-argument**: With only 4 layers and ~328K params, gradient flow pathology may
not be severe enough to make LLRD meaningful. The benefit of LLRD is most pronounced for
deep networks (12+ layers) where early gradient attenuation is large. For 4 layers the
decay effect between first and last block is at most 4× — arguably not enough to matter.

**Param delta**: 0 parameters. Pure optimizer change.

**Implementation (train.py lines 483–496)**:
```python
# Access submodules through compile wrapper:
_model = getattr(model, "_orig_mod", model)

decay = 0.7
base_lr = cfg.lr  # 5e-4

# Build param groups from output→input order
param_groups = []
# 1. Decoder + final norm
param_groups.append({
    "params": list(_model.fc.parameters()) + list(_model.blocks[-1].ln_3.parameters() if hasattr(_model.blocks[-1], 'ln_3') else []),
    "lr": base_lr * 1.0,
    "name": "decoder"
})
# Note: check actual attribute names. mlp2 and ln_3 are in the last block:
# From train.py ~line 157: self.ln_3, self.mlp2 are inside TransolverBlock, NOT on Model.
# Actually the final decoder (mlp2) IS inside the last block? Re-check...
# From train.py line ~175-185: Model has self.blocks (list of TransolverBlock).
# Each TransolverBlock has ln_3 + mlp2 as the final decoder per-block.
# The FINAL output layer is blocks[-1].mlp2. Confirm in code.

# Simpler approach — iterate blocks in reverse:
num_blocks = len(_model.blocks)
for i, block in enumerate(_model.blocks):
    depth_from_output = num_blocks - 1 - i  # 0 for last block
    lr_i = base_lr * (decay ** depth_from_output)
    param_groups.append({"params": list(block.parameters()), "lr": lr_i, "name": f"block_{i}"})

# Preprocess (embedding):
param_groups.append({
    "params": list(_model.preprocess.parameters()),
    "lr": base_lr * (decay ** num_blocks),
    "name": "preprocess"
})

optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
```

After building param_groups, the scheduler must use `optimizer.param_groups[0]['lr']` for
logging — already the case in the existing code. No other changes needed.

Add `llrd_decay: float = 0.7` to Config dataclass.

**CLI**: `python train.py --experiment_name llrd-decay07 --llrd_decay 0.7 --epochs 70`

---

## Idea D — Attention Temperature Sweep (alternative for edward if LLRD is assigned)

**Mechanism**: PhysicsAttention uses a per-head scalar temperature initialized at 0.5,
applied as `softmax(in_project_slice(x_mid) / temperature)`. Lower temperature sharpens
the slice routing (harder token assignment), higher temperature softens it. The current
init=0.5 was carried over from the Transolver paper without probing the sensitivity.

Temperature axis has NEVER been probed in any round of this research program. Given that:
(a) slice_num=24 is optimal (concave: 16 and 32 both worse), and (b) camber_rc is the OOD
bottleneck, attention sharpness is a plausible lever. OOD generalization often benefits
from softer attention (wider context, less commitment to training-distribution slots).

**Implementation**: Change temperature init from 0.5 to target value. Since temperature
is `nn.Parameter`, it remains learnable — the init determines where optimization starts.
Alternatively, fix temperature (no_grad) to test the effect without allowing it to drift.

**Sweep**: try init values {0.25, 0.5 (baseline), 1.0}. Recommend init=0.25 (sharper) as
first arm to test if harder routing helps camber_rc. If it hurts, try init=1.0 (softer).

Code change: `train.py line 95`:
```python
# Change: torch.ones([1, heads, 1, 1]) * 0.5
# To:     torch.ones([1, heads, 1, 1]) * cfg.temp_init
```

Add `temp_init: float = 0.5` to Config dataclass.

**Param delta**: 0 parameters (temperature is already a parameter; this changes init only).

---

## Idea E — Separate Volume/Surface Normalization Statistics

**Mechanism**: The current normalization uses global y_mean and y_std computed over ALL
nodes. Surface nodes and volume nodes have fundamentally different pressure distributions:
surface nodes have boundary-layer-dominated flow with potentially sharper gradients;
volume nodes span the far-field where values approach free-stream. A single normalization
conflates these two regimes.

Split normalization: compute (surf_mean, surf_std) from surface nodes only and
(vol_mean, vol_std) from volume nodes only. Apply surf normalization to surface-node
predictions and vol normalization to volume-node predictions during training. The model
predicts in a single output space but the normalization target is regime-specific.

**Distinctiveness**: No prior experiment touched normalization statistics. Not related to
surf_weight (closed), not related to FiLM (closed 14th taxon), not related to per-channel
weighting (Idea B). Orthogonal to all in-flight PRs.

**Counter-argument**: stats.json is read-only per data contract. But normalization is
applied in train.py directly (`y_norm = (y - stats["y_mean"]) / stats["y_std"]`). Could
compute supplementary stats in train.py by scanning the training data once before the
training loop, WITHOUT modifying data/. This respects the read-only boundary.

Complexity risk: requires two sets of stats, careful indexing by is_surface mask during
prediction and loss. Not zero cost. May also hurt volume predictions if surf stats diverge
too far from global stats.

---

## Idea F — Decoder Residual Skip Connection

**Mechanism**: The final per-block decoder in TransolverBlock is a 2-layer MLP (ln_3 →
Linear→GELU→Linear → out_dim). There is no skip connection from the transformer latent
to the output. Adding a linear skip `x_skip = Linear(hidden_dim, hidden_dim)` with output
`mlp2(ln_3(fx)) + x_skip(fx)` provides a direct gradient path from the latent to the
output. This is analogous to the residual connections in the main blocks and to the
"pre-activation residual" design used in ResNet-v2.

For a regression task on smooth physical fields, a skip connection may reduce the burden
on the GELU activation to pass near-linear relationships through, which could help for the
pressure channel (which is approximately linear in the far-field).

**Distinctiveness**: No closed taxon covers decoder architecture. Tanjiro in-flight #2472
splits the decoder into two heads (surf/vol) but does not add skip connections. These are
composable if both succeed.

**Param delta**: +hidden_dim×hidden_dim ≈ +9216 params (~+2.8%). Acceptable.

---

## Idea G — Gradient-Norm Clipping with Per-Param Clip (not global norm)

**Mechanism**: The current stack has NO gradient clipping (SAM taxon 12 closed, but SAM
is perturbation-based; vanilla clipping is orthogonal). Global gradient norm clipping
(`clip_grad_norm_`) with max_norm=1.0 is standard but the closed bimodal taxon list
includes early experiments. However, PER-PARAMETER clipping via `clip_grad_value_` clips
each individual gradient tensor independently rather than normalizing by global norm.
`clip_grad_norm_` can uniformly shrink gradients of all parameters if one parameter has a
large gradient (e.g., a late-layer weight during high-loss early training), silently
starving early layers. Per-value clipping prevents this.

**Distinctiveness**: gradient clip was part of round-35/36 tuning but that used global
norm. Per-parameter value clipping has a different effect on early vs. late layers.

**Counter-argument**: Per-value clipping can change the effective update direction. If
gradients are consistently small everywhere, it's a no-op. The benefit only materializes
when there are large-gradient outlier parameters.

---

## Student Assignments

### ASSIGNED: nezuko → Idea A (aux-camber-head retry-1)

Priority: HIGH. PR #2411 was closed stale_wip (pod stall, never ran). Mechanistically
sound and distinct from all closed taxa. ReLU² experiment (PR #2418) showed camber_rc is
the split most responsive to changes — it was the only split that improved with ReLU²
(−2.50%) even as overall LOSS was +7.76%. This motivates giving the model an explicit
geometry signal anchored to camber.

Recommended config: `--aux_camber_weight 0.1 --epochs 70`

### ASSIGNED: fern → Idea B (per-channel surf_weight)

Priority: HIGH. Zero param change, directly targets primary metric (surf_p). The surf_weight
axis is closed at the SCALAR level; the per-channel ratio is entirely untested. Simplest
possible change with a direct mechanistic tie to the evaluation criterion.

Recommended config: `--p_channel_weight 3.0 --epochs 70`

### ASSIGNED: edward → Idea C (layer-wise LR decay)

Priority: MEDIUM-HIGH. Distinct from closed lr-DOWN bimodal taxon (which was uniform
global). LLRD has strong evidence from large-scale transformer fine-tuning literature
(BERT, GPT fine-tuning papers, ViT fine-tuning). The application to a 4-layer network
reduces prior confidence but the experiment is zero-cost on parameters. If the first arm
result is ambiguous, edward can pivot to Idea D (temperature sweep) as retry-1.

Recommended config: `--llrd_decay 0.7 --epochs 70`

---

## Hypothesis Priority Ranking

1. Idea B (per-channel surf_weight) — direct mechanism, zero params, highest info/compute
2. Idea A (aux-camber-head) — direct OOD geometry signal, tiny param overhead, never ran
3. Idea C (LLRD) — standard technique, no parameter cost, plausible for 4 layers
4. Idea D (temperature sweep) — novel axis never probed, high upside, simple change
5. Idea F (decoder skip) — reasonable but tanjiro (#2472) already probing decoder region
6. Idea E (split normalization) — highest complexity, data-contract edge risk
7. Idea G (per-param clip) — weak prior evidence for improvement in this regime
