<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 1 Research Ideas — TandemFoilSet CFD Surrogate
**Date:** 2026-05-12  
**Branch context:** icml-appendix-willow-pai2g-48h-r5 (no prior experiment PRs)  
**Timeout cap:** SENPAI_TIMEOUT_MINUTES=30 wall clock per run  
**Primary metric:** val_avg/mae_surf_p — lower is better

## Baseline Configuration (train.py, as-read)

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0
n_layers=5, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2
scheduler: CosineAnnealingLR(T_max=MAX_EPOCHS), no warmup
loss: vol_mse + 10 * surf_mse, equal channel weighting within each term
NO gradient clipping, NO dropout, NO per-channel p weighting
```

---

## Hypothesis 1: Gradient Clipping

### What it is
Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` between `loss.backward()` and `optimizer.step()` on line 499-500 of train.py.

### Why it should work
The baseline has no gradient clipping. High-Re samples produce pressure values up to ±29K (physical units). Even after global normalization, sample-to-sample variance in y_std is an order of magnitude within a single domain split. Without clipping, a single bad high-Re batch can produce a gradient spike that corrupts all learned weights — the optimizer will take an arbitrarily large step. Gradient clipping is a standard stabilization technique for regression on heavy-tailed targets (well-documented in NLP and physics simulation literature). It costs near-zero compute and cannot hurt a well-behaved run; it can only help a run that was otherwise unstable.

### Exact code change
In train.py, add one line after `loss.backward()` (line 499):
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ADD THIS
optimizer.step()
```

### Predicted delta
-2% to -8% on val_avg/mae_surf_p. Larger effect visible in val_re_rand and val_geom_camber_rc (high-Re tandem) than val_single_in_dist.

### Risk
Low. If the baseline was already stable the clip threshold of 1.0 never fires and the result is identical. If the run was unstable it helps materially. Worth 30 minutes.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/grad-clip-1.0" --epochs 30
```
(Student adds the one-line clip; no new CLI flag needed unless they want to parameterize threshold.)

---

## Hypothesis 2: Linear LR Warmup (5 epochs)

### What it is
Replace the bare `CosineAnnealingLR` with a `SequentialLR` that linearly warms LR from 5e-5 to 5e-4 over 5 epochs, then cosine-decays from 5e-4 to 0 over the remaining epochs.

### Why it should work
AdamW's momentum buffers start cold. Without warmup, the optimizer takes maximum-magnitude steps on the very first batches — exactly when gradient estimates are worst. With meshes of 74K-242K nodes and global normalization that is imperfect for extreme Re, those first-epoch large steps can land in a poor basin from which cosine annealing cannot recover. Linear warmup is the single most consistent training improvement found across all transformer-style models since BERT and continues to be confirmed in physics-ML settings (Transolver++, UPT, geometry-agnostic solvers). The cost is zero extra parameters.

### Exact code change
In train.py, replace lines 434-435:
```python
# BEFORE
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# AFTER
WARMUP_EPOCHS = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS)
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
)
```

### Predicted delta
-3% to -10% on val_avg/mae_surf_p. Effect most visible in training loss curve — a clean monotonic decrease vs. the current likely-noisy early epochs. May compound with gradient clipping (H1).

### Risk
Medium-low. The main risk is that 5 warmup epochs is too long given the 30-minute wall clock, leaving less time in the cosine phase. If epochs run fast (~2-3 min each) this is fine. If epochs are slow (~5 min), reduce WARMUP_EPOCHS to 3.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/lr-warmup-5ep" --epochs 30
```

---

## Hypothesis 3: Increase n_layers from 5 to 8

### What it is
Change `n_layers=5` to `n_layers=8` in the model_config dict at line 423 of train.py.

### Why it should work
The original Transolver paper used L=8 layers as their default and tested L∈{1,2,4,8,16}. Performance on the benchmark tasks saturated at L=8; going to 16 gave marginal gains. The baseline uses only L=5 — which is 37% fewer layers than the paper's recommended default. This directly limits representational depth: the tandem geometry introduces a second foil with its own boundary layer and wake interactions that likely require more attention layers to disentangle. The parameter count increase is modest (~60% more params) and still fits well within 96GB VRAM.

### Exact code change
In train.py, change line 423:
```python
# BEFORE
n_layers=5,
# AFTER
n_layers=8,
```

### Predicted delta
-5% to -15% on val_avg/mae_surf_p. Expected gains concentrated in the tandem OOD splits (val_geom_camber_rc, val_geom_camber_cruise) where richer foil-foil interaction modelling helps most.

### Risk
Medium. More layers = slightly longer epoch time. The 30-minute cap will complete fewer epochs. If the model needs more than ~15 epochs to converge at this depth, the cap could truncate before the best checkpoint. Mitigate by starting LR slightly higher or running fewer epochs with an aggressive LR schedule.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/n-layers-8" --epochs 30
```

---

## Hypothesis 4: Increase slice_num from 64 to 96

### What it is
Change `slice_num=64` to `slice_num=96` in the model_config dict at line 424 of train.py.

### Why it should work
In Transolver, each "slice" is a physics-aware token representing a learned grouping of mesh nodes into a coherent physical state (e.g., attached boundary layer, separated wake, pressure side, suction side). With `slice_num=64`, the model has 64 such tokens to describe the entire flow field. The TandemFoilSet has significantly more structural complexity than single-foil benchmarks: 3 overset mesh zones, up to 2 foil surfaces with independent wakes, and strong foil-foil interference. The original paper's ablation showed performance still improving at M=96-128 on complex geometries; the plateau at M=64 was reported on simpler single-geometry datasets. Going from 64 to 96 adds 50% more physics tokens at O(M²) attention cost — from 64²=4096 to 96²=9216 token-pair computations — which is negligible relative to the full N=74K-242K node count.

### Exact code change
In train.py, change line 424:
```python
# BEFORE
slice_num=64,
# AFTER
slice_num=96,
```

### Predicted delta
-2% to -8% on val_avg/mae_surf_p. Higher benefit on tandem splits than single-foil.

### Risk
Low-medium. O(M²) cost increase is small. Memory increase: M×C×head matrices grow by 50%, still far from OOM. Risk is that the model needs more training steps to learn good slice assignments for 96 tokens; within 30-minute cap this may not fully converge.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/slice-num-96" --epochs 30
```

---

## Hypothesis 5: Increase mlp_ratio from 2 to 4

### What it is
Change `mlp_ratio=2` to `mlp_ratio=4` in the model_config dict at line 425 of train.py.

### Why it should work
`mlp_ratio` controls the hidden dimension multiplier in the feed-forward block of each Transformer layer: FFN hidden dim = `mlp_ratio * n_hidden`. The baseline uses `mlp_ratio=2`, giving FFN hidden dim = 256. The standard Vision Transformer (ViT) and all BERT/GPT-class models use `mlp_ratio=4` (FFN dim = 4×model_dim). This is a well-established default derived from expressiveness analysis of two-layer MLPs: the 4× expansion provides enough capacity to compose residual corrections without redundancy. With `mlp_ratio=2`, each attention layer's output is processed by an underpowered FFN that may not have enough capacity to transform the physics-aware tokens into good node-level representations. This is a pure capacity increase with no novel mechanism risk.

### Exact code change
In train.py, change line 425:
```python
# BEFORE
mlp_ratio=2,
# AFTER
mlp_ratio=4,
```

### Predicted delta
-3% to -10% on val_avg/mae_surf_p. Uniform improvement across all splits expected.

### Risk
Low. Parameter count and memory increase moderately (~20-30% more FFN params). Still very far from OOM at 96GB. Slightly longer per-epoch time. No algorithmic risk.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/mlp-ratio-4" --epochs 30
```

---

## Hypothesis 6: Per-Channel Surface Pressure Weighting

### What it is
Add an extra loss weight on the pressure channel `p` (channel index 2) within the surface loss term, since the primary ranking metric is surface pressure MAE, not equal-weight surface MAE.

### Why it should work
The primary metric is `val_avg/mae_surf_p` — surface pressure MAE on held-out samples. The current loss is:
```
loss = vol_mse(Ux, Uy, p) + 10 * surf_mse(Ux, Uy, p)
```
All three channels contribute equally to `surf_mse`. But the model is ranked only on the `p` channel. Ux and Uy have velocity units (m/s) while p has pressure units (m²/s²) — the normalized variances may also be mismatched. By upweighting the p channel in the surface loss term, we align the optimization objective directly with the evaluation objective. This is a classic task-loss alignment trick: if the metric is not the loss, make the loss closer to the metric.

### Exact code change
In train.py, modify the loss computation (around lines 490-496):
```python
# BEFORE
sq_err = (pred - y_norm) ** 2
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss

# AFTER
sq_err = (pred - y_norm) ** 2
# Upweight pressure channel (index 2) on surface nodes
p_surf_weight = torch.ones(3, device=sq_err.device)
p_surf_weight[2] = 3.0  # 3x weight on p at surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1) * p_surf_weight).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Note: the `p_surf_weight` multiplier of 3.0 means the effective surf weight on p becomes `10 * 3 = 30` vs. `10 * 1 = 10` for Ux/Uy. Start at 3.0; if that destabilizes, try 2.0.

### Predicted delta
-3% to -12% on val_avg/mae_surf_p specifically. Small regression possible on Ux/Uy surface MAE (acceptable). Volume metrics unaffected.

### Risk
Medium. Aggressive channel upweighting can cause the model to ignore velocity channels entirely if ratio is too high, which may harm Ux/Uy surface metrics and indirectly hurt p recovery (they are coupled through the momentum equations). Start at 3.0. If val shows Ux/Uy surface MAE degrading >30% relative, reduce to 2.0.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/p-surf-weight-3x" --epochs 30
```

---

## Hypothesis 7: Huber Loss Instead of MSE

### What it is
Replace the MSE (squared error) loss with Huber loss (smooth L1) for both vol and surf terms. Huber loss uses L2 for small errors and L1 for large errors, controlled by a `delta` parameter.

### Why it should work
The primary challenge in this dataset is the extreme range of target values: within a single split, per-sample y_std varies by an order of magnitude (high-Re samples produce pressure values in the tens of thousands, low-Re in the hundreds). MSE penalizes large errors quadratically — which means a handful of extreme high-Re training samples dominate the gradient signal and the model over-specializes on those extremes. Huber loss clamps the influence of extreme samples, acting as a natural per-sample robustness mechanism without requiring explicit sample reweighting. L1-regime behavior for large errors also matches the primary MAE evaluation metric more closely than MSE does. Huber loss is standard in regression tasks with heavy-tailed residuals (robust statistics literature) and has been validated in CFD surrogate settings.

### Exact code change
In train.py, replace `sq_err` computation and vol/surf loss (lines 490-496):
```python
# BEFORE
sq_err = (pred - y_norm) ** 2
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# AFTER
HUBER_DELTA = 1.0  # in normalized space; tune if needed
huber_err = torch.nn.functional.huber_loss(
    pred, y_norm, reduction='none', delta=HUBER_DELTA
)
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
```

`huber_loss` with `reduction='none'` returns elementwise values identical in shape to `pred`. `delta=1.0` in normalized space means residuals beyond 1 std are handled with L1 — reasonable given the dataset's heavy tails.

### Predicted delta
-3% to -10% on val_avg/mae_surf_p. Stronger effect on Re-stratified holdout (val_re_rand) where high-Re and low-Re samples mix within the same split.

### Risk
Low-medium. If `delta` is set too low, the loss becomes effectively L1 everywhere and training can become noisy (L1 has discontinuous gradients at zero). `delta=1.0` in normalized space is a safe starting point. No new packages needed — `torch.nn.functional.huber_loss` is in core PyTorch.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/huber-delta-1.0" --epochs 30
```

---

## Hypothesis 8: Dropout for OOD Generalization

### What it is
Add dropout (p=0.1) inside the Transolver model, applied after each attention block and after each FFN block.

### Why it should work
The baseline uses `dropout=0.0` throughout. Three of four validation splits test OOD generalization: unseen camber ranges (geometry OOD) and stratified Re holdout (regime OOD). Zero dropout means the model can fully memorize the training distribution's specific geometry-Re combinations — exactly what OOD splits penalize. Dropout is a classic regularization technique that prevents co-adaptation of features and improves generalization, especially under data scarcity (1,499 training samples spread across 3 domains). Applied after attention and FFN blocks at rate 0.1, it is mild enough not to impede convergence but strong enough to reduce overfitting on the geometry and Re dimensions.

The Transolver model class likely accepts a `dropout` parameter. If not, dropout layers can be inserted in the forward pass after the attention output projection and after the second linear layer in each FFN.

### Exact code change
In train.py, add `dropout=0.1` to model_config (around line 417-428):
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    dropout=0.1,   # ADD THIS
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

**Important implementation note:** The student must first check whether the `Transolver` class in the local implementation accepts a `dropout` argument. Run `grep -n "dropout" /workspace/senpai/target/*.py` and inspect the Transolver class definition. If it does not accept `dropout`, the student must add it to the `__init__` and `forward` method — adding `nn.Dropout(p)` after each attention output and after each FFN second linear, using `self.training` to ensure it is disabled at eval time.

### Predicted delta
-2% to -8% on val_avg/mae_surf_p, concentrated in the OOD splits (val_geom_camber_rc, val_geom_camber_cruise, val_re_rand). Small regression possible on val_single_in_dist (in-dist performance may slightly worsen as expected with regularization).

### Risk
Medium. If the Transolver class does not accept `dropout`, the student must modify the model class — which adds implementation risk. The bigger risk is that 0.1 is too aggressive for a 30-minute cap (the model may not converge). If val loss is clearly not improving after 15 epochs, the student should reduce to 0.05 and note this in the PR.

### Reproduce command
```bash
cd target && python train.py --agent willow --wandb_name "willow/dropout-0.1" --epochs 30
```

---

## Priority Ranking for Assignment

Ranked by expected signal clarity, implementation simplicity, and alignment with primary metric:

| Rank | Hypothesis | Expected delta | Risk | Implementation effort |
|------|-----------|---------------|------|-----------------------|
| 1 | H1: Gradient clipping | -2% to -8% | Low | 1 line |
| 2 | H5: mlp_ratio 2→4 | -3% to -10% | Low | 1 line |
| 3 | H7: Huber loss | -3% to -10% | Low-med | 4 lines |
| 4 | H2: LR warmup | -3% to -10% | Low-med | 8 lines |
| 5 | H6: p-channel surf weight | -3% to -12% | Medium | 4 lines |
| 6 | H3: n_layers 5→8 | -5% to -15% | Medium | 1 line |
| 7 | H4: slice_num 64→96 | -2% to -8% | Low-med | 1 line |
| 8 | H8: Dropout 0.1 | -2% to -8% | Medium | check + possibly modify model |

H1 (gradient clipping) should go first as a pure stabilizer — it costs nothing and may reveal that the baseline is unstable, which would explain many potential failures. H1+H2+H5+H7 are the four safest changes. H3, H4, H6, H8 have slightly higher risk or implementation effort.

## Combination Suggestions (after round 1 results)

If H1 wins: combine with H2 next (clipping + warmup is the standard "safe training" recipe).  
If H5 wins: combine with H3 next (mlp_ratio=4 + n_layers=8 is the full "standard transformer" config).  
If H7 wins: check whether H6 also wins; if both win, combine them (Huber + p-weighting = direct metric alignment).  
If H4 wins: try slice_num=128 next.  
If H8 wins: try dropout=0.05 or 0.15 to calibrate.
