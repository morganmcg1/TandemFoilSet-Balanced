<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research Ideas — 2026-05-16 09:30

Generated after reviewing all closed and in-flight experiments (H6–H38).
Current baseline: val_avg/mae_surf_p = 67.64 (H24, PR #3540).
Per-split: single=80.32, geom_camber_rc=81.81, geom_camber_cruise=44.46, re_rand=63.96.

---

## H39 — SWA: Stochastic Weight Averaging over final checkpoints

**Hypothesis (1–2 sentences).**
Average the model weights from the last 3 saved checkpoints (epochs 10, 11, 12, already in low-LR territory under the truncated OneCycleLR schedule) to approximate the SWA flat-minima effect without requiring a dedicated constant-LR tail or EMA momentum. Unlike EMA (H8, closed +29%), SWA reads already-written checkpoint files, so it is immune to the schedule-truncation problem and adds zero training cost.

**Mechanism.**
EMA failed because β=0.999 needs ~1000 stable steps at low LR — the 30-min wall-clock cap means the schedule is truncated at ~11–13 epochs and those steps never arrive. SWA bypasses this entirely: save `checkpoint_ep10.pt`, `checkpoint_ep11.pt`, `checkpoint_ep12.pt` during the normal training loop, then at the very end of training load all three, average their `state_dict` parameter tensors, and evaluate the averaged model. The averaging smooths out the stochastic noise of the final SGD iterates without requiring a separate LR phase.

**Predicted delta on val_avg/mae_surf_p.**
−2 to −5 (conservative; SWA gains tend to be 1–3% on benchmark tasks with comparable schedule compression). Upside if the last few epochs are high-variance.

**Exact CLI args / code changes.**
No new CLI arg needed. After the training loop ends:

```python
# Collect the last-N checkpoint paths (saved during training)
import copy, glob
ckpt_paths = sorted(glob.glob(f"models/{cfg.experiment_name}/checkpoint_ep*.pt"))[-3:]
if len(ckpt_paths) >= 2:
    swa_state = copy.deepcopy(model.state_dict())
    for path in ckpt_paths[1:]:
        other = torch.load(path, map_location=device)["model"]
        for k in swa_state:
            swa_state[k] = swa_state[k] + other[k]
    n = len(ckpt_paths)
    for k in swa_state:
        swa_state[k] = swa_state[k] / n
    model.load_state_dict(swa_state)
```

During the training loop, add `torch.save({"model": model.state_dict()}, f"models/{cfg.experiment_name}/checkpoint_ep{epoch}.pt")` for the final 3 epochs. The regular best-checkpoint logic is unchanged; SWA averaging happens in a post-training block.

**Papers.**
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," UAI 2018. https://arxiv.org/abs/1803.05407
- Huang et al., "Snapshot Ensembles," ICLR 2017. https://arxiv.org/abs/1704.00109

**Risks.**
If the training run is cut before epoch 10, fewer than 3 checkpoints exist; gracefully fall back to the single best checkpoint. The SWA average may perform worse if the last few epochs still have meaningfully high LR (pct_start=0.3 with 12 realized epochs means LR is decaying from ~epoch 4 onwards, so epochs 10–12 are in deep decay territory — this should be fine).

---

## H40 — Domain-type indicator: explicit is_tandem / domain_id feature

**Hypothesis (1–2 sentences).**
Add an explicit 3-class domain indicator (raceCar-single=0, raceCar-tandem=1, cruise=2) derived deterministically from the existing input features, and concatenate it to each node's feature vector before the `preprocess` linear layer. The val_single split is the worst (80.32) because single-foil samples have dims 18–23 = all zeros — structurally identical to a tandem sample with zero gap/stagger — so the model has no clean route to distinguish them at inference.

**Mechanism.**
Currently the only discriminator between single-foil and tandem-foil is that dims 18–23 are all exactly zero for single-foil. In practice the model must learn this as a corner case of the continuous tandem geometry. Making the distinction explicit with a learned embedding (3 → n_hidden via `nn.Embedding(3, n_hidden)`, added as a bias to the preprocess output) gives the model a direct routing token. The domain label is derived purely from the inputs: `domain_id = 0` if `x[b, 0, 22] == 0` (gap=0, single-foil), `domain_id = 2` if any `x[b, 0, 14] > 0` and `x[b, 0, 22] == 0` is false and `x[b, 0, 14]` is near cruise AoA range, else `domain_id = 1`. A simpler binary `is_tandem = (x[b, 0, 22] != 0).long()` is sufficient as a first pass and requires only 2-class embedding.

**Predicted delta on val_avg/mae_surf_p.**
−3 to −8. The val_single asymmetry (80.32 vs. 44–63 for the others) suggests the model conflates single-foil with degenerate tandem — a dedicated token should sharply separate these modes.

**Exact CLI args / code changes.**

```python
# In Transolver.__init__
self.domain_embed = nn.Embedding(2, n_hidden)  # is_tandem: 0=single, 1=tandem

# In Transolver.forward, after fx = self.preprocess(x_rff) + self.placeholder:
is_tandem = (x[:, 0, 22].abs() > 1e-6).long()  # [B]
domain_bias = self.domain_embed(is_tandem).unsqueeze(1)  # [B, 1, n_hidden]
fx = fx + domain_bias
```

No new CLI arg. Zero extra compute cost.

**Papers.**
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017 (segment/type embeddings). https://arxiv.org/abs/1706.03762
- Devlin et al., "BERT," NAACL 2019 (token type IDs as domain indicator). https://arxiv.org/abs/1810.04805

**Risks.**
Low. The domain label is perfectly deterministic from existing features, so no data leakage. If the single/tandem gap-threshold fires incorrectly for any sample, the embedding learns the wrong bias — check that `x[:, 0, 22].abs() > 1e-6` cleanly partitions samples (it should given the data spec).

---

## H41 — Auxiliary physics head: multi-task prediction of Re and AoA

**Hypothesis (1–2 sentences).**
Add a lightweight auxiliary head that predicts `log(Re)` and `AoA_foil1` from the global mean-pooled latent after the final TransolverBlock, with a small auxiliary loss weight (0.1); the shared encoder must then form representations that are explicitly Re- and AoA-aligned. This directly attacks the `val_re_rand` generalization gap (63.96) by supervising the encoder with the exact scalar that drives cross-regime variation.

**Mechanism.**
`log(Re)` (dim 13 of x) and `AoA_foil1` (dim 14) vary continuously and are the primary axes of physical variation. The main decoder sees them only through the input embedding; there is no explicit gradient signal that says "your latent must encode Re." Adding `aux_head = MLP(n_hidden → 2)` applied to `fx.mean(dim=1)` (masked mean-pool over real nodes) and supervising it with MSE against the normalized dim-13 and dim-14 values creates a direct gradient path. This is Caruana-style multi-task learning where the auxiliary task is a strict subset of the input signal (ground truth is available for free — it is just the input feature).

**Predicted delta on val_avg/mae_surf_p.**
−2 to −6, concentrated on `val_re_rand`. Marginal or neutral on `val_single`.

**Exact CLI args / code changes.**

```python
# In Transolver.__init__
self.aux_head = MLP(n_hidden, n_hidden // 2, 2)  # predict [log_Re, AoA1] normalized

# In Transolver.forward, after the block loop:
fx_mean = fx.mean(dim=1)  # [B, n_hidden]  (use mask for proper average)
aux_pred = self.aux_head(fx_mean)  # [B, 2]

# In training loss, after main loss computation:
# aux targets: normalized dim 13 and 14 from x_norm
aux_target = x_norm[:, 0, 13:15]  # [B, 2]
aux_loss = F.mse_loss(aux_pred, aux_target)
loss = loss + cfg.aux_weight * aux_loss
```

Add `aux_weight: float = 0.1` to `Config`. Use masked mean-pool:
```python
real_mask = mask.float().unsqueeze(-1)  # [B, N, 1]
fx_mean = (fx * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
```

**Papers.**
- Caruana, "Multitask Learning," Machine Learning 1997.
- Kendall, Gal, Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018. https://arxiv.org/abs/1705.07115

**Risks.**
The auxiliary loss competes with the main loss during the high-LR warmup phase. If `aux_weight` is too large it degrades the primary metric. Start at 0.1; the auxiliary task is extremely easy (Re is literally encoded in the input), so the head should saturate quickly. Note: `x_norm[:, 0, 13:15]` uses only batch-element 0's features — correct for scalar per-sample quantities since all nodes share the same Re/AoA.

---

## H42 — Gradient accumulation: effective batch size 4 → 8

**Hypothesis (1–2 sentences).**
Use gradient accumulation with `accum_steps=2` to double the effective batch size from 4 to 8, stabilizing the gradient signal during the high-LR warmup phase of OneCycleLR without increasing VRAM or reducing gradient steps per epoch. Every previous closed experiment that affected gradient variance (EMA, SAM, weight decay) operated on optimizer dynamics; this targets the upstream gradient noise source directly.

**Mechanism.**
With batch_size=4 and 1499/4 ≈ 375 optimizer steps per epoch, gradient variance from single-sample noise is high, especially in the warmup where LR is climbing. Accumulating 2 micro-batches halves that variance at zero VRAM cost. The OneCycleLR scheduler step must be tied to optimizer steps (every 2 forward passes), not micro-batch steps. Training throughput drops ~10% due to doubled forward passes, but the quality of each optimizer step improves.

**Predicted delta on val_avg/mae_surf_p.**
−1 to −4. Likely modest but combinable with other changes. The key test is whether val loss at epoch 4–6 (peak LR) is more stable.

**Exact CLI args / code changes.**

Add `accum_steps: int = 1` to `Config`. In the training loop:

```python
optimizer.zero_grad()
for step, (x, y, is_surface, mask) in enumerate(train_loader):
    # ... forward pass, compute loss ...
    loss_scaled = loss / cfg.accum_steps
    loss_scaled.backward()
    if (step + 1) % cfg.accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

CLI: `--accum_steps 2`. The `total_steps` for OneCycleLR must be `len(train_loader) // accum_steps * MAX_EPOCHS`.

**Papers.**
- Chen et al., "Revisiting Few-Sample BERT Fine-Tuning," ICLR 2021 (gradient accumulation for small-batch stability). https://arxiv.org/abs/2006.05987

**Risks.**
If `len(train_loader)` is not divisible by `accum_steps`, the last incomplete accumulation must be flushed at epoch end. The OneCycleLR total_steps must be recomputed correctly or the scheduler will go out of sync. The throughput hit (~10%) further reduces realized epochs under the 30-min cap — at 199s/epoch (n_layers=5 baseline) this is negligible.

---

## H43 — Flow-condition MixUp: interpolate Re/AoA/NACA features, keep geometry fixed

**Hypothesis (1–2 sentences).**
Apply MixUp exclusively on the flow-condition subspace (dims 13–23: `log(Re)`, AoA, NACA params, gap, stagger) and correspondingly on the target y, while keeping node positions (dims 0–1) and shape descriptors (dims 2–11) fixed from the primary sample. This creates synthetic "unseen (geometry, flow condition)" combinations without introducing physically inconsistent node positions.

**Mechanism.**
Standard MixUp on the full feature vector would interpolate node positions between two meshes — physically nonsensical for this overset multi-zone structure. By mixing only the 11-dimensional flow-condition subspace, we generate plausible new conditions on existing geometries. The target y is mixed with the same lambda: `y_mix = λ·y_a + (1-λ)·y_b`. This is directly relevant to the `val_geom_camber` splits (unseen camber M=6–8 and M=2–4) because it creates training samples that explore the flow-condition manifold more densely. It also addresses the small dataset size (1499 samples) by synthetic augmentation.

**Predicted delta on val_avg/mae_surf_p.**
−2 to −6, concentrated on `val_geom_camber_rc` and `val_geom_camber_cruise`. The geom splits test unseen camber but the MixUp creates intermediate flow conditions; it won't fully close the geometry gap but smooths the boundary.

**Exact CLI args / code changes.**

Add `mixup_alpha: float = 0.0` to `Config`. In the training loop, after loading batch:

```python
if cfg.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
    lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
    # Shuffle within batch
    idx = torch.randperm(x.size(0), device=x.device)
    # Mix only flow-condition dims 13–23 (11 dims)
    x_mix = x.clone()
    x_mix[:, :, 13:24] = lam * x[:, :, 13:24] + (1 - lam) * x[idx, :, 13:24]
    y_mix = lam * y + (1 - lam) * y[idx]
    x, y = x_mix, y_mix
```

CLI: `--mixup_alpha 0.4` as starting point (standard alpha for MixUp).

**Papers.**
- Zhang et al., "MixUp: Beyond Empirical Risk Minimization," ICLR 2018. https://arxiv.org/abs/1710.09412
- Guo et al., "MixUp as Locally Linear Out-of-Manifold Regularization," AAAI 2019.

**Risks.**
The mixed target y_mix requires corresponding mixed mesh inputs — since both samples in the mini-batch have different mesh sizes (padded to max), the mixed y may have padding at positions where the primary sample has real nodes. Use only `mask` from the primary sample (x) for loss computation after MixUp. If mixing same-geometry samples (same NACA, different Re), the effect is simply data augmentation — benign. Cross-geometry mixing (different camber) creates physically questionable intermediate states; the benefit is empirical.

---

## H44 — Pressure-specific output head: separate MLP branch for p channel

**Hypothesis (1–2 sentences).**
Branch the final block output into two separate output MLPs — one for `(Ux, Uy)` velocity and one for `p` pressure — with independent weight matrices, giving the pressure prediction dedicated capacity rather than splitting it with the velocity channels. This is architecturally orthogonal to H36's channel-weighted loss and targets the same channel-imbalance bottleneck from the output side.

**Mechanism.**
The current architecture has a single `mlp2` (SwiGLUMLP) that maps `n_hidden → 3` for all three output channels simultaneously. The shared linear layer must compromise between Ux/Uy gradients and p gradients. A separate `p_head = MLP(n_hidden, n_hidden//2, 1)` gives pressure its own gradient pathway at the cost of ~n_hidden²/4 extra parameters (negligible). The primary metric is p-only (`mae_surf_p`), so giving p dedicated capacity is directly aligned with the ranking objective.

**Predicted delta on val_avg/mae_surf_p.**
−2 to −5 if channel conflict is a real bottleneck. If the shared head is not the bottleneck (i.e., the features are already separable and the head is just a linear projection), gain will be near zero.

**Exact CLI args / code changes.**

In `Transolver.__init__`, replace:
```python
# existing: last block has mlp2
```
with:
```python
# Keep velocity output in last block's mlp2 (n_hidden → 2)
# Add separate pressure head
self.p_head = MLP(n_hidden, n_hidden // 2, 1)
# Modify output_dims: last block mlp2 outputs [Ux, Uy] only
```

In `Transolver.forward`, after the block loop:
```python
vel_pred = last_block_output[:, :, :2]   # [B, N, 2]
p_pred   = self.p_head(last_block_input) # [B, N, 1]  -- input before mlp2
preds = torch.cat([vel_pred, p_pred], dim=-1)  # [B, N, 3]
```

Implementation note: the `last_layer=True` TransolverBlock currently calls `self.mlp2(self.ln_3(fx))` inside `forward`; the cleanest approach is to set `last_layer=False` for the final block, then apply the two heads externally in `Transolver.forward`.

**Papers.**
- Cipolla, Gal, Kendall, "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018. https://arxiv.org/abs/1705.07115
- Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021 (separate output heads per field). https://arxiv.org/abs/2010.08895

**Risks.**
If H36 (channel-weighted loss) already lands and solves the channel-imbalance bottleneck, H44 may add complexity for marginal gain. Run H44 regardless — the mechanism (output head capacity) is independent of loss weighting (gradient magnitude). The two are combinable.

---

## H45 — Foil-relative coordinate encoding: distance and angle to nearest foil surface

**Hypothesis (1–2 sentences).**
Augment the input features with two scalars per node: approximate distance to the nearest foil surface and angle relative to the foil chord, computed from the existing `is_surface` flag and node positions already in the input. This gives the model a structured inductive prior for boundary-layer gradients — the dominant physics near the foil surface where pressure error concentrates.

**Mechanism.**
The input already has `is_surface` (dim 12) and RFF-encoded position (dims 0–1). However, interior nodes have no feature that tells them how far they are from the foil. The surface pressure gradient decays roughly as 1/r in potential flow; without explicit distance encoding, the model must infer this from the geometry features (dsdf, dims 4–11) indirectly. Augmenting with `log(1 + dist_to_surface)` gives a direct handle. The distance can be approximated cheaply during the forward pass using `x[:, :, :2]` (node positions) and the `is_surface` mask: for each batch element, find the mean position of surface nodes and compute Euclidean distance from each node to that centroid (a crude approximation). A more accurate approach is to find the minimum distance to any surface node, but that is O(N²) per sample. The centroid approximation costs O(N) and provides a monotone signal.

**Predicted delta on val_avg/mae_surf_p.**
−1 to −4. The dsdf features (dims 4–11) already encode some distance information; this is a redundancy check — if dsdf is already capturing this, the gain is zero.

**Exact CLI args / code changes.**

In `Transolver.forward`, before `self.preprocess`:
```python
# Compute approximate log-distance to foil surface centroid
surf_mask = x[:, :, 12:13]  # [B, N, 1], is_surface flag (unnormalized)
surf_pos = (x[:, :, :2] * surf_mask).sum(dim=1, keepdim=True) / surf_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 2]
dist = (x[:, :, :2] - surf_pos).norm(dim=-1, keepdim=True)  # [B, N, 1]
log_dist = torch.log1p(dist)  # [B, N, 1]
x_aug = torch.cat([x, log_dist], dim=-1)  # [B, N, 25]
```
Adjust `fun_dim = X_DIM - 2 + 1` in model_config (25 total dims, preprocess input grows by 1). No new CLI arg.

**Papers.**
- Lino et al., "Multi-scale rotation-equivariant graph neural networks for unstructured physics simulations," PoF 2022. https://arxiv.org/abs/2107.07503
- Jiang et al., "MeshGraphNets," ICML 2020 (distance-based edge features for mesh GNNs). https://arxiv.org/abs/2010.03409

**Risks.**
The centroid approximation breaks for tandem foils (two foils at different positions). A per-foil centroid (using dims 18–23 to distinguish foil 1 vs. foil 2 surfaces) would be more accurate but requires more careful bookkeeping. Start with the single centroid; if the idea shows promise, refine with per-foil distances. Also, if the dsdf features (dims 4–11) already capture this, the added feature is redundant and may slightly increase noise.

---

## Summary of Rankings

| ID  | Hypothesis                          | Predicted delta | Complexity | Priority |
|-----|-------------------------------------|-----------------|------------|----------|
| H40 | Domain-type indicator (is_tandem)   | −3 to −8        | Minimal    | 1st      |
| H39 | SWA checkpoint averaging            | −2 to −5        | Low        | 2nd      |
| H41 | Auxiliary Re/AoA prediction head    | −2 to −6        | Low        | 3rd      |
| H43 | Flow-condition MixUp (dims 13–23)   | −2 to −6        | Low        | 4th      |
| H44 | Pressure-specific output head       | −2 to −5        | Medium     | 5th      |
| H42 | Gradient accumulation (accum×2)     | −1 to −4        | Low        | 6th      |
| H45 | Foil-relative coordinate encoding  | −1 to −4        | Low        | 7th      |

**Do not assign:** H40 + H39 to the same student (independent, give each its own slot). H41 + H43 are also independent. H44 should wait until H36 (channel-weighted loss) resolves to avoid redundancy.
