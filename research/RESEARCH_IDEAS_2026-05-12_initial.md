<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — TandemFoilSet CFD Surrogate — 2026-05-12 (Round 1)

Branch: `icml-appendix-charlie-pai2g-24h-r2`
Generated: 2026-05-12, fresh track, zero prior experiments.

---

## Context and bottleneck diagnosis

The baseline is a Transolver (Wu et al., ICML 2024) with:
- 1.4M parameters (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)
- AdamW lr=5e-4, weight_decay=1e-4
- CosineAnnealingLR over MAX_EPOCHS, no warmup
- MSE loss in normalized space: `vol_loss + 10.0 * surf_loss`
- batch_size=4, no AMP, no gradient clipping
- 96 GB VRAM, ~30 min wall clock cap → 10-20 realistic epochs

Primary metric: `val_avg/mae_surf_p` (equal-weight surface pressure MAE across 4 splits, physical units).

Key causal candidates for the bottleneck at this starting point:
1. **Training efficiency**: no AMP, tiny batch, no warmup — most of the 30 min budget is probably in data movement and small-batch noise, not useful gradient updates.
2. **Loss formulation**: per-channel weighting is flat; p (pressure) is the target metric but gets equal gradient weight to Ux and Uy; surf_weight=10 was not searched.
3. **Architecture capacity**: 1.4M params on a 96GB GPU is severely underparameterized for 74K–242K node meshes with multi-domain flow.
4. **Generalization to OOD camber**: no curriculum, no geometry augmentation — the model sees NACA M=0-5 and M=9 for raceCar but must generalize to M=6-8.
5. **Optimization trajectory**: flat cosine from epoch 1 into a complex loss landscape; no gradient clipping means occasional large step corruption.

---

## Hypothesis 1 — Larger model: n_hidden=256, n_layers=8

**Category:** Architecture capacity

**Hypothesis:** The 1.4M-param Transolver is severely underparameterized for meshes with up to 242K nodes and three physically distinct domains. Increasing n_hidden from 128→256 and n_layers from 5→8 grows the model to ~8.8M params (6.3x), still well within 96GB VRAM at bs=4. This provides representational capacity that the current model structurally cannot achieve regardless of optimization quality.

**Mechanism:** Transolver's PhysicsAttention operates on slice_num=64 tokens of dimension n_hidden. Doubling n_hidden widens both the per-slice representation and the FFN, allowing more expressive slice assignments. Adding 3 more layers deepens the re-aggregation. The per-epoch cost scales roughly linearly with params at fixed bs and mesh size — estimated ~1.4x slower, which is acceptable within 30 min.

**Predicted delta on val_avg/mae_surf_p:** -10% to -20% relative improvement.

**Exact code change in train.py (model_config dict):**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,      # was 128
    n_layers=8,        # was 5
    n_head=8,          # was 4 — keep dim_head=32, consistent
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Keep all other settings at baseline: lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0.

**CLI reproduce command:**
```bash
python train.py --experiment_name H1-larger-model --agent <student>
```
(model_config must be edited in train.py; all other flags remain default)

**Risk/failure mode:** Per-epoch time may push past timeout before the model converges, delivering worse val metrics than baseline despite better asymptotic capacity. Mitigation: if epoch time >3 min, reduce n_layers back to 6.

**Complexity:** Low (edit 3 lines in model_config dict).

**Citation/rationale:** Transolver paper (Wu et al., ICML 2024) showed clear scaling with n_hidden on their benchmarks. On industrial-scale PDE problems (FNO-variant benchmarks), 5-10M param models typically outperform 1-2M param models significantly.

---

## Hypothesis 2 — AMP (bf16) + gradient clipping + larger effective batch via accumulation

**Category:** Training efficiency

**Hypothesis:** The baseline runs fp32 with no gradient clipping and bs=4. Enabling AMP (bf16) roughly halves memory bandwidth pressure and allows ~2x faster forward/backward passes. Adding gradient clipping (max_norm=1.0) prevents the occasional large update that corrupts optimization in early epochs on high-dynamic-range targets. Together these allow getting ~2x more gradient updates within the 30 min budget, which for early-epoch-limited training is equivalent to training twice as long.

**Mechanism:** At bs=4 with up to 242K nodes, each forward/backward is memory-bandwidth bound for the large attention tensors. bf16 halves the bandwidth for activations. Gradient clipping targets a known failure mode for MSE+high-variance targets (Ux/p at Re=5M can exceed 30,000 in physical units — the normalized variance is still ~1 by design, but occasional outlier batches in early epochs can produce large gradients). This is a training-quality fix, not a representational fix.

**Exact code changes in train.py:**

After the optimizer definition:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scaler = torch.cuda.amp.GradScaler()
```

In the training loop, replace the forward/backward block:
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (y - stats["y_mean"]) / stats["y_std"]
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred.float() - y_norm) ** 2

    vol_mask = mask & ~is_surface
    surf_mask = mask & is_surface
    vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
    surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    loss = vol_loss + cfg.surf_weight * surf_loss

optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Note: cast pred to float32 before sq_err to avoid bf16 precision loss in the loss accumulation.

**CLI reproduce command:**
```bash
python train.py --experiment_name H2-amp-gradclip --agent <student>
```

**Predicted delta on val_avg/mae_surf_p:** -5% to -15% relative. The effect is purely through getting more gradient updates in the same wall-clock budget. If the baseline already completes all 50 epochs within 30 min, this is a smaller win.

**Risk/failure mode:** bf16 can cause loss spikes on untested codepaths. If NaN appears, switch dtype=torch.float16 with scaler, or revert to fp32 with only gradient clipping.

**Complexity:** Low-medium (add ~10 lines to training loop).

**Citation/rationale:** Standard practice for transformer training on large inputs since 2022. The 30-min wall-clock cap makes this particularly high-leverage — every speedup directly translates to more training signal.

---

## Hypothesis 3 — Linear warmup scheduler + higher peak LR

**Category:** Optimization trajectory

**Hypothesis:** The baseline uses CosineAnnealingLR starting from lr=5e-4 at epoch 1 with no warmup. With ~10-20 epochs in 30 min, the first 2-3 epochs are responsible for shaping the loss landscape that the remaining epochs exploit. Starting at 5e-4 immediately in a randomly initialized transformer causes large, destabilizing updates before the internal representations stabilize. A linear warmup over 3 epochs from lr=1e-5 to lr=1e-3, followed by cosine decay, gives the model a brief stabilization phase and a higher peak LR that enables faster convergence during the productive middle epochs.

**Mechanism:** Targets the training failure mode, not representational capacity. The warmup prevents early-epoch gradient corruption from large initial activations in LayerNorm-normalized residual blocks. The 2x higher peak LR (1e-3 vs 5e-4) increases gradient step magnitude in the convergent phase. For short training regimes, peak LR matters more than the final LR — cosine to near-zero is wasteful if training stops at epoch 12-15.

**Exact code change in train.py:**

Replace the scheduler line:
```python
# Remove:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# Replace with:
warmup_epochs = 3
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS - warmup_epochs, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.02, end_factor=1.0, total_iters=warmup_epochs
        ),
        scheduler_cosine,
    ],
    milestones=[warmup_epochs],
)
```

Also change in Config: `lr: float = 1e-3`

**CLI reproduce command:**
```bash
python train.py --lr 1e-3 --experiment_name H3-warmup-lr --agent <student>
```
(scheduler code change must be edited in train.py)

**Predicted delta on val_avg/mae_surf_p:** -5% to -12%. Warmup is well-validated in transformer fine-tuning and short training regimes.

**Risk/failure mode:** If 1e-3 is too high and causes divergence in early epochs, warmup is insufficient. Fallback: use lr=5e-4 with just warmup and no LR increase.

**Complexity:** Low (edit ~8 lines in train.py).

**Citation/rationale:** OneCycleLR/warmup superiority over flat cosine for short training documented in Salimath et al. 2025, and standard in ViT/BERT training (Dosovitskiy et al. 2021, Devlin et al. 2019).

---

## Hypothesis 4 — Per-channel loss weighting: upweight pressure p relative to Ux/Uy

**Category:** Loss reformulation

**Hypothesis:** The primary metric is `val_avg/mae_surf_p` (surface pressure MAE). The current MSE loss weights Ux, Uy, and p equally in normalized space. Since the three channels are independently normalized (each by its own std), they enter the loss at approximately equal scale by construction — but their physical relevance to the paper-facing metric is not equal. Explicitly upweighting p by a factor of 3-10x relative to Ux/Uy in the loss directly aligns gradient signal with the evaluation objective.

**Mechanism:** This is a proxy-metric alignment intervention, not a representational change. The normalized space loss is an average over 3 channels: if `surf_weight=10` already emphasizes surface nodes, adding a p-channel weight of 5x means the model receives 5x more gradient from surface pressure errors than surface velocity errors. This is independent of and orthogonal to surf_weight tuning.

**Exact code change in train.py:**

In the training loop, replace the loss computation:
```python
sq_err = (pred - y_norm) ** 2

# Channel weights: [Ux=1.0, Uy=1.0, p=5.0]
chan_w = torch.tensor([1.0, 1.0, 5.0], device=device, dtype=sq_err.dtype)
sq_err_w = sq_err * chan_w[None, None, :]

vol_loss = (sq_err_w * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_w * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Apply the same chan_w in `evaluate_split` for the loss metric (MAE computation is unchanged — it must remain unweighted for apples-to-apples comparison with the organizer scorer).

Also add `chan_weight_p: float = 5.0` to the Config dataclass.

**CLI reproduce command:**
```bash
python train.py --chan_weight_p 5.0 --experiment_name H4-chan-weight-p --agent <student>
```

**Predicted delta on val_avg/mae_surf_p:** -8% to -18%. This is one of the highest-confidence hypotheses because it directly targets a metric mismatch that is structural, not empirical.

**Risk/failure mode:** Over-weighting p can degrade Ux/Uy predictions enough that the model struggles to learn the correct velocity boundary conditions, which in turn corrupts the pressure field via learned co-dependencies. Mitigation: if val_avg/mae_surf_p degrades vs baseline, try p-weight=2.0.

**Complexity:** Very low (add ~5 lines to loss computation).

**Citation/rationale:** Adaptive PINN loss weighting (Farea et al., arXiv 2509.14437, 2025). The principle that loss weights should align with evaluation metrics is universal across competitive ML (common in Kaggle CFD/regression tasks).

---

## Hypothesis 5 — Surface-only loss (drop vol_loss from training signal)

**Category:** Loss reformulation

**Hypothesis:** The current loss combines vol_loss and surf_loss. But the primary metric is `mae_surf_p` only. Volume nodes (interior + background mesh) vastly outnumber surface nodes and may be dominating the gradient signal. An extreme version of surf_weight=10 is to drop vol_loss entirely and train purely on surface MSE, forcing the model to specialize entirely on the surface. This is analogous to the AB-UPT (Alkin et al., 2025) observation that surface-volume decoupling improves both surface and volume prediction independently.

**Mechanism:** The mechanism is gradient concentration: 100% of the training signal flows from surface prediction error. This is aggressive but may improve surface pressure prediction even if volume fields degrade. The risk is that volume context, which the model learns from vol_loss, helps it predict surface pressure correctly (since pressure is a global field). This experiment directly tests that assumption.

**Exact code change in train.py:**

```python
# In training loop, replace:
loss = vol_loss + cfg.surf_weight * surf_loss

# With:
loss = surf_loss  # surf_weight becomes irrelevant
```

Also try an intermediate: `loss = 0.1 * vol_loss + surf_loss` as a second arm.

**CLI reproduce command:**
```bash
python train.py --surf_weight 0.0 --experiment_name H5-surf-only-loss --agent <student>
```
(Requires adding surf_weight=0 branch: `loss = surf_loss if cfg.surf_weight <= 0 else vol_loss + cfg.surf_weight * surf_loss`)

**Predicted delta on val_avg/mae_surf_p:** Uncertain — could be -15% to +10%. High variance hypothesis. The experiment is discriminating: if surface loss specialization helps, it tells us the vol signal was hurting. If it hurts, vol context is genuinely necessary.

**Risk/failure mode:** The model may learn to output any prediction for interior nodes (since those receive no gradient), which could destabilize training through the LayerNorm statistics. Mitigation: add the 0.1 vol_loss arm to preserve some interior gradient.

**Complexity:** Very low (add 3 lines).

**Citation/rationale:** AB-UPT (Alkin et al., NeurIPS 2025) shows surface-volume decoupling in universal physics transformers. General principle: align training objective with evaluation objective.

---

## Hypothesis 6 — Increase slice_num from 64 to 128

**Category:** Architecture (physics attention resolution)

**Hypothesis:** The core mechanism of Transolver is soft assignment of N mesh nodes to slice_num "physics tokens", followed by attention over those tokens. With slice_num=64 and meshes up to 242K nodes, each slice averages ~3,780 nodes. This is a heavy compression that may lose spatial structure critical for predicting surface pressure gradients (the primary metric). Doubling slice_num to 128 gives each token half the territory, at the cost of 4x more attention computation (O(slice_num²) for the attention matrix over 64 tokens → 128 tokens).

**Mechanism:** More slices = more fine-grained physics token decomposition = better ability to represent localized flow features (separation bubbles, pressure spikes on foil leading/trailing edges). The VRAM overhead is in the attention matrix [B, n_head, slice_num, slice_num] = [4, 4, 128, 128] = 4×4×128×128 floats ≈ negligible compared to node-level tensors [4, 242K, 128].

**Exact code change in train.py:**
```python
model_config = dict(
    ...
    slice_num=128,      # was 64
    ...
)
```

**CLI reproduce command:**
```bash
python train.py --experiment_name H6-slice128 --agent <student>
```
(Edit slice_num in model_config)

**Predicted delta on val_avg/mae_surf_p:** -5% to -12%. Directly increases physics token resolution.

**Risk/failure mode:** Negligible VRAM impact. Slight increase in per-epoch training time (~5-10% from attention compute). May not help if the bottleneck is not slice resolution.

**Complexity:** Minimal (1 line edit).

**Citation/rationale:** Transolver ablation (Wu et al., ICML 2024) showed slice_num sensitivity; the paper used slice_num=64 on their benchmarks but those meshes are 2-3x smaller than the largest TandemFoil meshes.

---

## Hypothesis 7 — Geometry augmentation: random flip of AoA sign for raceCar domain

**Category:** Data/sampling

**Hypothesis:** The raceCar domain uses only negative AoA (inverted airfoil, -10° to 0°). The model sees the same AoA direction throughout, which limits its understanding of pressure asymmetry. Flipping the sign of AoA during training (mirroring the geometry) and correspondingly flipping Ux and the sign of Uy creates physically valid augmented samples at no additional data cost. This is analogous to horizontal-flip augmentation in image classification. The cruise domain (+/- AoA) may benefit less, but both will see more AoA diversity.

**Mechanism:** AoA is stored in dims 14 (foil 1) and 18 (foil 2) of x. Uy is stored in y channel 1. For a pure AoA flip: `x[:, 14] *= -1; x[:, 18] *= -1; y[:, 1] *= -1` (Uy changes sign; Ux and p magnitudes are invariant to vertical flip). The node positions in dims 0-1 (x,z) are NOT flipped — this is a flow-condition augmentation, not a mesh transform.

**Exact code change in train.py (augmentation applied on GPU during training loop):**
```python
# After moving tensors to device, before normalization:
if torch.rand(1).item() < 0.5:  # 50% probability
    # Flip AoA sign — valid only if is_symmetric_flip is safe
    # x dims: 14=AoA foil1, 18=AoA foil2
    x = x.clone()
    x[:, :, 14] = -x[:, :, 14]
    x[:, :, 18] = -x[:, :, 18]
    # Correspondingly flip Uy (channel 1 of y)
    y = y.clone()
    y[:, :, 1] = -y[:, :, 1]
```

Note: camber sign (dims 15-17, 19-21) should NOT be flipped. Only AoA and Uy.

**CLI reproduce command:**
```bash
python train.py --experiment_name H7-aoa-aug --agent <student>
```
(Augmentation code added in training loop in train.py)

**Predicted delta on val_avg/mae_surf_p:** -3% to -10% on the OOD camber splits (val_geom_camber_rc, val_geom_camber_cruise). Uncertain effect on val_single_in_dist (already in-distribution).

**Risk/failure mode:** The flip is only valid if we correctly handle the normalization pass. Since augmentation happens BEFORE normalization, the augmented x values may fall slightly outside the normalized training distribution for AoA dims. Mitigation: confirm the AoA feature dims are correctly identified. The x stats are mean/std computed over the training set; a sign flip around a non-zero mean (raceCar AoA mean ≈ -5°) would create out-of-distribution inputs. This is actually the desired OOD stress-test behavior, but may also cause unstable early training. Consider using only 50% flip probability (already included above) rather than 100%.

**Complexity:** Low (add ~8 lines in training loop).

**Citation/rationale:** Physics-preserving data augmentation for CFD is standard practice (see any aerodynamics ML paper post-2020). The specific AoA flip for 2D airfoil is a textbook-valid augmentation.

---

## Hypothesis 8 — Pressure-head output decoupling: separate surface/volume output heads

**Category:** Architecture

**Hypothesis:** The Transolver uses a single output MLP head (the `mlp2` in the last TransolverBlock) to predict all 3 channels (Ux, Uy, p) for all nodes. But surface nodes and volume nodes live on fundamentally different parts of the solution manifold — surface nodes encode boundary conditions while volume nodes encode the interior field. Giving the model separate output projections for surface vs. volume nodes (at inference time, choose which head to apply based on is_surface) may improve surface pressure prediction without degrading volume field quality. Each head gets cleaner gradient signal for its own domain.

**Mechanism:** Replace the single `mlp2` in the last TransolverBlock with two heads: `mlp2_surf` and `mlp2_vol`. At forward pass: `pred = torch.where(is_surface.unsqueeze(-1), surf_head(fx_last), vol_head(fx_last))`. The shared backbone (all 5 layers of PhysicsAttention + FFN) is unchanged — only the final projection is split. Total parameter increase: ~2x the last linear layer ≈ +6K params, negligible.

**Exact code changes in train.py:**

In TransolverBlock.__init__ (last_layer branch):
```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    self.mlp2_surf = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    self.mlp2_vol = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
```

In TransolverBlock.forward (last_layer branch):
```python
if self.last_layer:
    fx_ln = self.ln_3(fx)
    return fx_ln  # return pre-output features; final projection done in Transolver.forward
```

In Transolver.forward: modify to accept is_surface and apply dual-head projection. Pass is_surface from the data dict:
```python
def forward(self, data, **kwargs):
    x = data["x"]
    is_surface = data.get("is_surface", None)
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks[:-1]:
        fx = block(fx)
    fx = self.last_block_features(self.blocks[-1], fx)  # get pre-head features
    if is_surface is not None:
        surf_pred = self.mlp2_surf(fx)
        vol_pred = self.mlp2_vol(fx)
        pred = torch.where(is_surface.unsqueeze(-1), surf_pred, vol_pred)
    else:
        pred = self.mlp2_surf(fx)  # fallback
    return {"preds": pred}
```

Update training loop to pass is_surface to model: `pred = model({"x": x_norm, "is_surface": surf_mask})["preds"]`

**Predicted delta on val_avg/mae_surf_p:** -5% to -15%. The hypothesis is grounded in the AB-UPT paper's observation that surface and volume predictions benefit from decoupled processing.

**Risk/failure mode:** Architecture change is more invasive than other hypotheses. The forward refactor needs careful testing to confirm mask alignment. Fallback: if dual heads don't help, the surface head alone (applied to all nodes) as a single-head variant is a clean ablation.

**Complexity:** Medium (edit ~20 lines across 2 classes in train.py).

**Citation/rationale:** AB-UPT (Alkin et al., NeurIPS 2025) — anchored-branched universal physics transformer; surface-volume decoupling is their central contribution. GNN+ODE hybrid (Quattromini et al., 2024) also reports improved surface field accuracy from specialized surface processing.

---

## Hypothesis 9 — surf_weight hyperparameter search: try 30 and 50

**Category:** Loss reformulation (hyperparameter search)

**Hypothesis:** The current surf_weight=10.0 was not searched. The primary metric is surface pressure MAE — a higher surf_weight means the model receives proportionally more gradient from surface errors. Given that surface nodes are a small fraction of total nodes (especially for the dense background mesh in Zone 0), surf_weight=10 may still be leaving surface precision on the table. Two quick arms: surf_weight=30 and surf_weight=50.

**Mechanism:** Same as baseline but with 3-5x more gradient emphasis on surface predictions. At surf_weight=10, if there are 100x more volume nodes than surface nodes, the effective per-node weighting for surface is only 10/100 = 0.1 relative weight — still dominated by volume signal.

**CLI reproduce command:**
```bash
python train.py --surf_weight 30 --experiment_name H9a-sw30 --agent <student>
python train.py --surf_weight 50 --experiment_name H9b-sw50 --agent <student>
```

**Predicted delta on val_avg/mae_surf_p:** -5% to -15%. High confidence — surf_weight was not searched and direct alignment with primary metric is the strongest prior.

**Risk/failure mode:** If surf_weight is too high, vol_loss contribution becomes negligible and the model may fail to learn the correct far-field flow, which can degrade surface predictions indirectly. A very high surf_weight also increases gradient variance (surface nodes are sparse and not uniformly distributed).

**Complexity:** Zero (CLI flag only).

**Citation/rationale:** Loss weight hyperparameter search is standard in multi-task learning. The principle is well-documented in GradNorm (Chen et al., 2018) and later works.

---

## Hypothesis 10 — Increase batch size to 8 or 16 (if VRAM allows)

**Category:** Training efficiency

**Hypothesis:** The baseline batch_size=4 is chosen conservatively for a 96GB GPU. At bs=4 with the largest meshes (242K nodes × 24 features × 4 bytes = ~23MB per sample in fp32), a batch consumes ~92MB of node data, far below VRAM limits. The actual VRAM usage is dominated by activations through the 5-layer network. For n_hidden=128, the activation tensor per layer is [bs, N_max, 128] = [4, 242K, 128] × 4 bytes ≈ 500MB per layer × 5 = 2.5GB — still far below 96GB. Batch size 8-16 should be safely feasible and would halve or quarter the gradient noise, enabling more stable convergence in the limited epoch budget.

**Mechanism:** Larger batches reduce gradient variance, which matters for unstable early training. Also increases GPU utilization significantly.

**CLI reproduce command:**
```bash
python train.py --batch_size 8 --experiment_name H10-bs8 --agent <student>
```
For bs=16 arm: `--batch_size 16 --experiment_name H10-bs16`

**Predicted delta on val_avg/mae_surf_p:** -3% to -8%. Smaller effect than architecture/loss changes, but zero code change required.

**Risk/failure mode:** OOM. The pad_collate strategy pads each batch to the largest sample — with bs=8, there is some probability of getting 8 large (242K node) samples in one batch. Monitor VRAM usage in the first epoch. If OOM, fallback to bs=6.

**Complexity:** Zero (CLI flag only).

**Citation/rationale:** Batch size scaling is universally well-understood. For neural operator training, batch size=4 is a known bottleneck (FNO papers typically use bs=16-32 when VRAM allows).

---

## Hypothesis 11 — Frequency feature augmentation: add Fourier encoding of position

**Category:** Architecture (input representation)

**Hypothesis:** The model currently receives raw (x,z) node positions in dims 0-1, which are normalized globally. The signed arc-length (dims 2-3) and dsdf descriptors (dims 4-11) encode local surface geometry, but the raw 2D position lacks multi-scale frequency content. Adding a Fourier positional encoding (sin/cos at K frequencies) of (x,z) before feeding into the preprocess MLP gives the model explicit multi-scale spatial awareness. This is the NeRF/Fourier features insight applied to PDE surrogate inputs.

**Mechanism:** Replace dims 0-1 with a Fourier embedding: for K=4 frequency bands, the 2D position becomes 2×2K=16 features. Total input dim changes from 24 to 24-2+16=38. The preprocess MLP fun_dim must be updated. This is one of the key ideas from Fourier Neural Operators (Li et al., 2021) and implicit neural representations.

**Exact code change in train.py:**

Add Fourier encoding in Transolver.forward before preprocess:
```python
def _fourier_encode(self, pos, num_freqs=4):
    # pos: [B, N, 2]
    freqs = 2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)
    x_freq = pos.unsqueeze(-1) * freqs  # [B, N, 2, K]
    enc = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)  # [B, N, 2, 2K]
    return enc.reshape(*pos.shape[:-1], -1)  # [B, N, 4K]
```

In forward, replace the positional dims:
```python
pos = x[..., :2]            # raw (x,z)
features = x[..., 2:]       # remaining 22 features
pos_enc = self._fourier_encode(pos, num_freqs=4)  # [B, N, 16]
x_aug = torch.cat([pos_enc, features], dim=-1)    # [B, N, 38]
fx = self.preprocess(x_aug) + self.placeholder[None, None, :]
```

Update preprocess MLP input dim: `fun_dim = X_DIM - 2 + 14 = 36` (replacing 2D pos with 16D Fourier enc → net +14 dims). Update model_config: `fun_dim=36`.

**CLI reproduce command:**
```bash
python train.py --experiment_name H11-fourier-pos --agent <student>
```
(Architecture change in train.py)

**Predicted delta on val_avg/mae_surf_p:** -5% to -12%. Speculative but grounded in the broad success of Fourier features for spatial PDE problems.

**Risk/failure mode:** The dsdf (dims 4-11) already encodes multi-scale distance information. If dsdf already captures the same spatial information as Fourier encoding, this may not add signal. Risk is low (no degradation expected, just may be neutral). The main implementation risk is getting the input dim arithmetic correct.

**Complexity:** Low-medium (add ~15 lines; update model_config fun_dim).

**Citation/rationale:** Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (Tancik et al., NeurIPS 2020). FNO (Li et al., ICLR 2021). Used in nearly all neural field / implicit representation work.

---

## Hypothesis 12 — Residual connection from input to output: physics prior for far-field flow

**Category:** Architecture (physics prior)

**Hypothesis:** In CFD, far from the airfoils, the flow tends toward the freestream: Ux → Re-dependent freestream velocity, Uy → 0, p → 0 (in normalized pressure). A model that predicts the delta from freestream (rather than the absolute value) only needs to learn the perturbation field. This is equivalent to a residual connection from the global flow condition (Re, AoA) to the output, analogous to the "physics-informed baseline" pattern in PINN literature. Concretely: initialize the output prediction to a freestream estimate (derived from Re and AoA features in x), and train the model to predict the residual.

**Mechanism:** Compute freestream prediction from the scalar flow features (dims 13-14 of x: log_Re, AoA), then add this as a learned baseline to the model output. The model only needs to learn deviations from freestream, which are smaller in magnitude and spatially concentrated near the foils.

**Exact code change in train.py:**

Add a freestream MLP to Transolver:
```python
self.freestream_head = nn.Sequential(
    nn.Linear(2, n_hidden // 2), nn.GELU(),
    nn.Linear(n_hidden // 2, out_dim),
)
```

In forward, extract flow conditions (broadcast over nodes) and add baseline:
```python
flow_cond = x[:, 0, 13:15]  # [B, 2] — log_Re and AoA, same for all nodes
baseline = self.freestream_head(flow_cond).unsqueeze(1)  # [B, 1, 3]
return {"preds": fx + baseline}
```

**CLI reproduce command:**
```bash
python train.py --experiment_name H12-freestream-residual --agent <student>
```

**Predicted delta on val_avg/mae_surf_p:** Speculative — -5% to +0%. The value is highest for OOD splits (val_re_rand) where the model must generalize across Re regimes.

**Risk/failure mode:** If the freestream baseline is poorly learned, it can hurt rather than help. The implementation assumes all nodes in a sample share the same log_Re and AoA (true for single-foil; for tandem, foil 2 has a different AoA in dim 18). Should use dim 14 (foil 1 AoA) and dim 13 (shared log_Re).

**Complexity:** Low (add ~10 lines to Transolver class).

**Citation/rationale:** Residual-from-prior approaches for PDE surrogates: Brandstetter et al. (2022), Herde et al. (2024). Standard physics-informed shortcut. The AB-UPT paper uses a similar "anchor" concept for their surface representation.

---

## Priority ranking (top 8 for immediate assignment)

Ranked by: mechanistic grounding × expected impact × execution cost, given the 30-min cap and 8 idle students.

| Rank | ID | Hypothesis | Predicted delta | Complexity | Confidence |
|------|-----|-----------|-----------------|------------|------------|
| 1 | H4 | Per-channel loss weight (p × 5x) | -8% to -18% | Very low | High — direct metric alignment |
| 2 | H9 | surf_weight search (30, 50) | -5% to -15% | Zero | High — unsearched hyperparameter |
| 3 | H1 | Larger model (256 hidden, 8 layers) | -10% to -20% | Low | Medium-high — capacity bottleneck |
| 4 | H2 | AMP bf16 + gradient clipping | -5% to -15% | Low | High — more updates within 30 min |
| 5 | H3 | Linear warmup + lr=1e-3 | -5% to -12% | Low | High — standard transformer recipe |
| 6 | H6 | Increase slice_num 64→128 | -5% to -12% | Minimal | Medium — validated in Transolver paper |
| 7 | H5 | Surface-only loss | -15% to +10% | Very low | Low-medium — high variance, discriminating |
| 8 | H8 | Dual surface/volume output heads | -5% to -15% | Medium | Medium — grounded in AB-UPT |

Hypotheses H7 (AoA augmentation), H10 (batch size), H11 (Fourier pos), H12 (freestream residual) are reserved for round 2 based on round 1 outcomes.

---

## Decision tree for round 2

```
If H4 wins AND H9 wins:
  → Combine H4 + H9 best surf_weight + H1 larger model (three-way combination)
  → Also try H4 + H2 (AMP + channel weight)

If H4 wins, H9 does NOT:
  → Keep H4 as baseline, try H1 + H4 combined
  → Diagnose whether surf_weight=10 is already near-optimal via H9 arms

If H1 wins:
  → Try H1 + H2 (larger model + AMP for speed recovery)
  → Try H1 + H6 (more slices in larger model)

If H3 wins:
  → Apply warmup to H1 (larger model + warmup)

If H5 (surface-only) wins:
  → Major signal: vol gradient is hurting. Next: try extreme surf_weight=100 instead of dropping vol entirely
  → Combine with H4 (channel weighting on surface-only loss)

If H8 (dual heads) wins:
  → Try H8 + H4 (dual heads + pressure channel weight)
  → Try H8 with larger backbone (H1 architecture)

If all 8 round 1 experiments lose to baseline:
  → Plateau protocol: escalate to architecture tier
  → Consider FNO-style spectral layers, U-Net skip connections over the Transolver blocks, or full model replacement with a PointTransformer / AB-UPT variant
  → Run worst-case error analysis on val splits to identify failure mode pattern
```

---

## Stop conditions

- **H4 or H9**: Stop if val_avg/mae_surf_p increases by >10% vs baseline (means channel/surf weighting is destabilizing training).
- **H1**: Stop if epoch time >4 min (means we won't get enough epochs for meaningful convergence).
- **H5**: Accept any result — this is a diagnostic; even failure is informative (tells us vol gradient is helpful).
- **H8**: Stop if training crashes on mask alignment — the architecture change is the highest implementation-risk hypothesis.
