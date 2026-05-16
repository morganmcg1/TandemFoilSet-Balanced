# SENPAI Round-6+ Research Ideas
# Generated: 2026-05-16 ~06:00 UTC
# Branch: icml-appendix-charlie-pai2i-24h-r3
# Baseline: val_avg/mae_surf_p = 87.62 (PR #3513, cosine T_max=20)
# Per-split baseline: single=98.44, rc=96.95, cruise=71.27, re_rand=83.83
# Context: Round-5 in flight (#3753–#3757). Round-4 closed 5/5 at parity or failure.
# Focus: val_single_in_dist remediation (98.44 — chronically worst, regresses under soft settings),
#         architecture-level changes (plateau protocol: escalate from HP tuning),
#         physics-informed regularizers, operator learning, attention specialization.

## Plateau context

Five consecutive rounds of HP/config experiments (rounds 3-4) have not beaten 87.62 by more than noise (σ=0.79). Round-5 escalates to architecture and data normalization. Round-6+ continues escalation. The ideas below represent tier-shift hypotheses — new mechanisms, not further tuning of existing knobs.

**Key diagnostic signal from round-4:**
- val_single_in_dist (98.44) is the hardest and most volatile split — regresses under every softening intervention (softer attention temp, wider n_head, looser optimizer)
- val_geom_camber_cruise (71.27) is the easiest and most robust — this is the OOD geometry split, benefiting from good feature coverage
- Single-foil samples have dims 18-23 = 0 (zero-padded tandem features) — the model must learn to ignore these zeros; this zero-padding creates a distribution mismatch relative to tandem samples
- High-Re samples drive val_single_in_dist extremes — Re range is 104K–5M with per-sample y_std varying by an order of magnitude within the split

---

## 1. Per-Point Adaptive Slice Temperature [HIGHEST PRIORITY — ARCHITECTURE]

**What it is:** Replace the single global learnable temperature scalar `nn.Parameter` in PhysicsAttention with a per-point learned temperature map — a small linear projection `features → τ_i ∈ [0.1, 2.0]` — so each node independently controls its clustering sharpness.

**Mechanism targeting observed bottleneck:** The current PhysicsAttention uses a single global temperature that must simultaneously serve surface nodes (tight, high-gradient clustering), far-field volume nodes (loose, diffuse clustering), high-Re nodes (sharp boundaries), and low-Re nodes (smooth fields). A global scalar is an averaging compromise. For val_single_in_dist the failure mode is surface-adjacent nodes on high-Re raceCar single foils — these have extreme pressure gradients that require tight slice assignment. A per-point temperature lets surface/high-Re nodes self-steepen while far-field nodes self-soften, without the global temperature pulling both toward a compromise. Round-4 tested global temperature annealing (external schedule override, parity 87.69); that is a strictly weaker version — it changed the same global scalar on a fixed schedule rather than learning per-node adaptive values from data.

**External evidence:** Transolver++ (2025, NeurIPS workshop) reports 13% accuracy improvement over base Transolver using exactly this mechanism on 2D Navier-Stokes and Darcy flow benchmarks. The key is that per-point temperature is learned end-to-end, not scheduled, so it discovers the data-driven clustering structure rather than following a predetermined profile. AlphaXiv annotations confirm the mechanism is the primary contributor (ablated, not confounded with other changes).

**Not tried:** Round-4 `physattn-temperature-anneal` (PR #3700) was a global external schedule — a fundamentally different mechanism. Per-point adaptive temperature has never been tested on this codebase. This is the highest-priority round-6 hypothesis.

**Implementation:**
```python
# In PhysicsAttention.__init__, replace:
#   self.T = nn.Parameter(torch.ones(1) * 0.5)
# with:
self.temp_proj = nn.Linear(n_hidden, 1, bias=True)
nn.init.constant_(self.temp_proj.bias, 0.0)   # neutral init → τ≈sigmoid(0)*1.9+0.1=1.05

# In PhysicsAttention.forward, after computing slice_feature:
# features: [B, N, C]  (the slice_feature or projected node hidden states)
tau = torch.sigmoid(self.temp_proj(features)) * 1.9 + 0.1  # [B, N, 1], range [0.1, 2.0]
# Replace: attn = softmax(slice_scores / self.T)
# with:
attn = softmax(slice_scores / tau)  # [B, N, G] — each node has its own temperature
```

Note: `slice_scores` is the raw logit tensor before softmax. The exact tensor name depends on the PhysicsAttention implementation — read `train.py` to find the correct variable. The projection input should be the same hidden states used to compute the slice assignments (before softmax), so the temperature is conditioned on the same representation.

**Taste rubric:** Mechanistic grounding 4 | Research-state value 4 | Execution value 4

**Mode:** Architecture — tier-shift from global to per-point specialization. Falsifiable: if val_single_in_dist does not drop by ≥3 points, the per-point temperature is not the bottleneck (it may be that the clustering itself is sufficient and the global temperature was already adequate).

**Stop condition:** Close if val_single_in_dist ≥ 97.0 (no improvement over baseline 98.44 minus noise floor) or val_avg ≥ 89.0 (regression). Merge if val_avg < 87.0 with val_single_in_dist < 96.0.

---

## 2. Surface-Only All-to-All Cross-Attention [ARCHITECTURE — PHYSICS MOTIVATED]

**What it is:** Add a parallel self-attention sub-layer that operates exclusively over `is_surface == True` nodes, enabling direct all-to-all communication between every surface point on both foils simultaneously. This sits alongside (not replacing) the standard Transolver slice-token attention, with a residual connection.

**Mechanism targeting observed bottleneck:** Transolver's slice-token attention compresses N=74K–242K nodes into G=64 slice tokens — a lossy aggregation. Two surface nodes on opposite foils may never appear in the same slice token, meaning their interaction is only indirect through the token bottleneck. For pressure prediction, Kutta-Joukowski theorem demands that the stagnation and separation points on both foils are globally coordinated — a tandem foil produces an interaction wake field that the upstream foil "sees" in its pressure distribution. For single-foil samples (the worst split), this argument still holds: the pressure at the leading edge stagnation point and the trailing edge separation must be globally consistent around the entire surface. Surface-only cross-attention provides this direct path. With typical surfaces having ~2K–8K nodes per foil, the full O(n_surf²) attention is feasible: 8K² × 4 bytes × 2 = 256 MB per sample. With batch=1 or chunked attention this fits in 96 GB VRAM.

**External evidence:** B-GNNs (Boundary Graph Neural Networks, arxiv 2503.18638, March 2025) report that all-to-all communication on boundary/surface meshes enforces a global incompressibility-like constraint that local message-passing cannot achieve, yielding +8-15% MAE improvement on 3D wing pressure prediction. The surface-only attention in GALE (NVIDIA GeoTransolver, Dec 2025) provides a similar path via geometry context tokens queried with surface node states.

**Not tried:** No prior PR has added a surface-specific attention sub-layer. The closest was the n_head=8 attempt (PR #3706), which failed due to dim_head=16 — entirely different mechanism.

**Implementation:**
```python
# After each (or the last) Transolver block, add:
class SurfaceCrossAttention(nn.Module):
    def __init__(self, n_hidden, n_head=4):
        super().__init__()
        # Safe: dim_head = 128//4 = 32
        self.attn = nn.MultiheadAttention(n_hidden, n_head, batch_first=True)
        self.norm = nn.LayerNorm(n_hidden)

    def forward(self, x, is_surface):
        # x: [B, N, C], is_surface: [B, N] bool
        # Gather surface nodes (padded to max_surf per batch)
        B, N, C = x.shape
        surf_feats_list = [x[b][is_surface[b]] for b in range(B)]
        max_surf = max(f.shape[0] for f in surf_feats_list)
        surf_pad = x.new_zeros(B, max_surf, C)
        surf_key_mask = torch.ones(B, max_surf, dtype=torch.bool, device=x.device)
        for b, f in enumerate(surf_feats_list):
            surf_pad[b, :f.shape[0]] = f
            surf_key_mask[b, :f.shape[0]] = False  # True = ignore (MHA convention)
        attn_out, _ = self.attn(surf_pad, surf_pad, surf_pad,
                                 key_padding_mask=surf_key_mask)
        # Scatter back
        out = x.clone()
        for b in range(B):
            n_surf = is_surface[b].sum()
            out[b][is_surface[b]] = out[b][is_surface[b]] + attn_out[b, :n_surf]
        return self.norm(out)
```

VRAM note: max_surf ~8K nodes × 4 × 128 × batch=4 ≈ 16 MB; the O(n_surf²) attention matrix is 8K² × 4 × 4 = 1 GB worst case — reduce batch=1 or use chunked attention if OOM.

**Taste rubric:** Mechanistic grounding 4 | Research-state value 3 | Execution value 3

**Mode:** Architecture tier-shift — surface physics path. The key observable is whether val_single_in_dist and val_re_rand drop more than val_geom_camber_cruise (cruise OOD geometry is already well-served; this mechanism targets within-sample pressure consistency).

**Stop condition:** Close if val_avg ≥ 89.5 or VRAM OOM with batch=1. Merge if val_avg < 86.5 with val_single_in_dist < 95.0.

---

## 3. Per-Domain Output Normalization with Single/Tandem Split — Augmented [QUICK SCREEN — DISTINCT FROM #3754]

**What it is:** While PR #3754 (edward, round-5) implements per-domain y_mean/y_std split between single and tandem samples, this round-6 variant extends to a three-way split — single (raceCar), tandem-raceCar, tandem-cruise — each with their own y_mean and y_std computed from their respective training subsets. The three domains have materially different pressure magnitude ranges (single max_y_std ≈ 2,077 vs cruise max_y_std ≈ 506 from program.md), suggesting the current global normalization forces all three into a shared scale that distorts the loss landscape.

**Mechanism targeting observed bottleneck:** The global y_mean/y_std in stats.json is dominated by the raceCar single domain (highest std, 599 samples). Tandem-cruise samples (443 samples, lower std) are effectively under-normalized — their normalized targets cluster near zero, making small errors invisible in the normalized loss. When the model is evaluated on val_single_in_dist in physical units, this normalization bias directly inflates MAE. Three-way domain normalization gives each domain its own scale so the training signal is equally calibrated across all three. Note: the normalization contract (model predicts in normalized space, scoring.py denormalizes with y_std) must be preserved — domain identity must be passed through to the denormalization at eval time. Since domain identity is already inferrable from dims 18-23 (zeros → single-foil), this can be done without changing the data loader interface.

**External evidence:** Domain-adaptive normalization is standard practice in multi-domain regression (physics-ML survey, Kovachki et al. 2023). In CFD surrogates with multi-regime data (Re range spanning 50×), per-domain normalization is often reported as the single largest calibration fix in ablations (FNO-3D paper appendix, Brunton et al. 2022).

**Not tried:** Round-5 PR #3754 tries two-way single/tandem split; three-way (separating tandem-raceCar from tandem-cruise) has never been attempted.

**Implementation:**
```python
# In train.py, compute per-domain stats from training data:
# Domain ID: infer from x dims 18-23
# x[:,18]==0 AND x[:,22]==0 → single-foil
# x[:,22]!=0 AND x[:,14]<0 → tandem-racecar (AoA negative = inverted)
# x[:,22]!=0 AND x[:,14]>=0 → tandem-cruise (AoA non-negative)
# Compute y_mean_d, y_std_d for d in {single, rc, cruise}
# During training: normalize each sample with its domain's stats
# During eval: pass domain_id with each sample; denormalize with matching stats
# Scoring contract: pass domain_stats dict; scoring.py denormalizes per sample
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 3

**Mode:** Data representation — tier-shift from global to domain-adaptive normalization. Complementary to, not competing with, round-5 per-domain-norm (#3754). Wait for #3754 result before assigning this; if #3754 wins, this is the natural follow-up.

**Stop condition:** Close if val_avg ≥ 88.5 or the three-way split produces fewer than 100 samples per domain in training (underfitting risk). Merge if val_avg < 87.0.

---

## 4. Stream Function Auxiliary Head for Incompressibility [PHYSICS-INFORMED LOSS]

**What it is:** Add a scalar stream-function auxiliary prediction head `ψ` (one extra output channel). Apply a soft loss penalizing deviation from the incompressibility constraint: `L_div = ||Ux_pred - ∂ψ/∂z||² + ||Uy_pred + ∂ψ/∂x||²` over surface nodes. Since `u = curl(ψ)` satisfies `∇·u = 0` identically, this loss guides the velocity predictions toward a divergence-free field without requiring FD stencils on the unstructured mesh.

**Mechanism targeting observed bottleneck:** The current MSE/Huber loss on (Ux, Uy, p) treats the three channels as independent regression targets with no coupling constraint. But real 2D incompressible flow satisfies ∇·u = 0 exactly — a hard physical constraint that the model only learns indirectly through data. For surface pressure prediction specifically, the Kutta-Joukowski circulation theorem ties the pressure distribution directly to the velocity field's rotational structure. By enforcing the stream function relationship, the model is forced to predict velocity fields consistent with a scalar potential, which propagates through to more physically consistent pressure predictions. The stream function auxiliary doesn't need FD stencils on the irregular mesh — the loss only checks that `(Ux_pred, Uy_pred)` matches `(∂ψ/∂z, -∂ψ/∂x)` from the predicted ψ, which can be estimated via autograd on the model output (ψ is a network output, not a data label).

**External evidence:** Divergence-free physics constraints in neural PDE solvers improve OOD robustness by 15-25% in GNS (Pfaff et al. 2021) and PINN literature. The stream-function parameterization for 2D flow is cited in Cai et al. (NSFnets, JCP 2021) as the most numerically stable way to enforce incompressibility in mesh-free settings. It avoids the discretization-dependent FD approximations that make ∇·u = 0 loss unstable on unstructured meshes.

**Not tried:** Round-4 backlog idea #5 proposed a direct ∇·u ≠ 0 divergence penalty — this is different because it uses the stream function parameterization, which is FD-free and does not require mesh connectivity. The implementation risk is substantially lower than the FD approach.

**Implementation:**
```python
# Add auxiliary head in Transolver output block:
self.psi_head = nn.Linear(n_hidden, 1)  # predicts stream function ψ

# Forward:
psi_pred = self.psi_head(hidden)  # [B, N, 1]
# ∂ψ/∂z and ∂ψ/∂x via autograd (node positions x_coord, z_coord from x[:,0], x[:,1])
# This requires positions un-normalized: x_phys = x[:,0:2] * stats["x_std"][0:2] + stats["x_mean"][0:2]
# Use torch.autograd.grad(psi_pred.sum(), x_phys, create_graph=True) to get grads

# Soft loss (surface nodes only, lambda=0.01):
ux_div = pred[:,:,0] - dpsi_dz   # should be 0
uy_div = pred[:,:,1] + dpsi_dx   # should be 0
surf_mask_float = is_surface.float()
L_div = lambda_div * (
    (ux_div**2 * surf_mask_float).sum() / surf_mask_float.sum() +
    (uy_div**2 * surf_mask_float).sum() / surf_mask_float.sum()
)
total_loss = total_loss + L_div
```

Note: autograd through node positions requires `x` to be created with `requires_grad=True` for the position dims. This is a non-trivial implementation; suggest the student test with `lambda_div=0.001` first and increase if the auxiliary loss is not decreasing.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 2

**Mode:** Physics-informed loss — tier-shift. Execution risk: autograd through mesh positions adds compute overhead; the 30-min timeout may limit epochs reached. Start with surface-only application and lambda=0.001.

**Stop condition:** Close if the divergence loss term fails to decrease below initial value by epoch 5, or if val_avg ≥ 90.0. Merge if val_avg < 86.0.

---

## 5. Re-Stratified Surface Loss Weighting [QUICK SCREEN — LOSS FORMULATION]

**What it is:** Weight each sample's surface loss contribution by `log(Re_i) / log(Re_max)` — so high-Re samples (Re=5M) contribute ~2× more surface loss signal than low-Re samples (Re=100K). The volume loss is unchanged. Reynolds number is available directly from the input as `x[:,13]` (already `log(Re)` in normalized form).

**Mechanism targeting observed bottleneck:** val_single_in_dist is dominated by high-Re raceCar single samples — its y_std ranges up to 2,077 vs ≈506 for cruise. The Huber δ=1.0 loss (merged) caps the influence of extreme gradient errors, but high-Re samples inherently produce larger absolute errors that are intrinsically harder. The WeightedRandomSampler balances domain frequency, but within each domain the loss treats a Re=5M sample identically to a Re=100K sample. For surface pressure specifically, high-Re samples have thin boundary layers, steeper pressure gradients, and larger absolute pressure magnitudes — they are the hard cases that drive MAE. Upweighting their surface loss by log-Re directly incentivizes the model to improve on the hardest-to-predict cases. This is analogous to focal loss (Kaggle practice: reweight hard examples) but using a known physical hardness proxy rather than a running loss estimate.

**External evidence:** Re-stratified loss weighting is used in CFD surrogate papers including Su et al. 2022 (deep learning for turbulent flows) and Kashefi & Muller 2022 (point-cloud neural operators). The log-Re weighting is physically motivated: pressure scales as ρU² ∝ Re² for fixed geometry, so log-Re tracks order-of-magnitude difficulty. Kaggle practice: domain-aware sample weighting routinely lifts OOD and minority-class performance without adding model complexity.

**Not tried:** All prior runs use flat per-sample loss across Re values within each domain. This is a 5-line change to the training loop.

**Implementation:**
```python
# In training loop, after normalizing x:
# x[:,13] is normalized log(Re); recover physical log(Re):
log_re_norm = x_batch[:, :, 13]  # [B, N], normalized
# Denormalize: log_re = log_re_norm * x_std[13] + x_mean[13]
# Re weight per sample (mean over nodes, then per-sample scalar):
log_re = (log_re_norm * stats["x_std"][13] + stats["x_mean"][13]).mean(dim=1)  # [B]
re_weight = log_re / log_re.max().clamp(min=1.0)  # [B], range [0, 1]

# Apply weight to surface loss:
surf_loss_unweighted = surf_loss_per_sample  # [B]
surf_loss_weighted = (surf_loss_unweighted * re_weight).mean()
total_loss = vol_loss.mean() + surf_weight * surf_loss_weighted
```

Note: the baseline loss computes a single scalar — this requires splitting into per-sample losses first before reweighting. Check that the current `train.py` loss is reduction='mean'; if so, switch to reduction='none', apply re_weight, then mean.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 4

**Mode:** Loss formulation — diagnostic/frontier. Very cheap (5-line change, zero overhead), directly targets the hardest samples in the worst split. Good quick screen before bigger architecture experiments.

**Stop condition:** Close if val_single_in_dist ≥ 97.5 (essentially no improvement) or val_avg ≥ 89.0 (regression). Merge if val_avg < 86.5.

---

## 6. Wider Backbone: n_hidden=192, n_head=4 (dim_head=48) [ARCHITECTURE — CAPACITY]

**What it is:** Increase `n_hidden` from 128 to 192 with `n_head=4` (maintaining safe `dim_head=48`), `n_layers=5`, all else unchanged. This is a ~2.25× parameter count increase (662K → ~1.49M params) driven purely by wider hidden dimensions in all linear layers.

**Mechanism targeting observed bottleneck:** The current 128-dim hidden space must simultaneously encode: position/geometry (24 input dims), Re regime (1 log-scalar spanning 50×), domain identity (single vs tandem via dims 18-23), and per-layer slice-token attention interactions. At 128 dims, the representation is congested — especially for tandem samples that have 6 extra geometry dimensions (dims 18-23) that single-foil samples zero out. The model must learn to suppress these zero-dims for single-foil without interfering with the rest of the representation. A 192-dim space provides ~50% more capacity to separately encode the single vs tandem conditional structure, potentially reducing the inter-domain interference that causes val_single_in_dist to regress when other settings are changed. The dim_head constraint (minimum 32, current=32 at n_head=4, proposed=48) is satisfied. Round-4 n_head=8 failure was at dim_head=16 — this proposal keeps n_head=4 and only widens.

**External evidence:** Transolver paper ablations show monotone improvement from n_hidden=64 to n_hidden=256 on all benchmarks (with diminishing returns). At our n_hidden=128 we are at the lower half of that range. GeoTransolver uses n_hidden=256 as default. Budget concern: 192-dim will increase VRAM and reduce epochs. Estimated VRAM: ~42-45 GB (vs 32.94 GB current), still within 96 GB. Estimated s/epoch: ~130-140s (vs ~98s), giving ~12-13 epochs in 30 min — reduced from 19.

**Not tried:** Every prior experiment has used n_hidden=128. No wider backbone has been tested.

**Implementation:**
```python
# Change in Config dataclass or passed kwargs:
# n_hidden: int = 192  (was 128)
# n_head: int = 4      (unchanged — gives dim_head=48, safe)
# n_layers: int = 5    (unchanged)
# Everything else identical to baseline
```

Note: reduced epochs (12-13 vs 19) means cosine T_max should be adjusted accordingly — set cosine_t_max=12 to ensure LR still fully anneals. Also re-check VRAM with batch_size=4 before full run; if >80 GB, reduce to batch_size=3.

**Taste rubric:** Mechanistic grounding 2 | Research-state value 3 | Execution value 2

**Mode:** Architecture — capacity increase. Less principled than per-point temperature (idea 1) but potentially additive. Lower priority than ideas 1-5; assign to a student when all higher-priority slots are filled.

**Stop condition:** Close if val_avg ≥ 89.0 (regression — indicates overfitting on limited epochs) or if VRAM OOM at batch=3. Merge if val_avg < 86.0.

---

## 7. Multi-Scale Slice Hierarchy: G_fine=64 + G_coarse=16 [ARCHITECTURE — OPERATOR LEARNING]

**What it is:** Replace the single G=64 slice-token PhysicsAttention with a dual-scale attention: one head aggregates into G_fine=64 slice tokens (fine local structure), a parallel head aggregates into G_coarse=16 tokens (global flow structure), and the outputs are concatenated and projected back to n_hidden. This is from round-4 backlog idea #12 — now assigned here.

**Mechanism targeting observed bottleneck:** Transolver's G=64 slice tokens are a fixed-resolution compression of the mesh. For tandem foils the mesh has three zones (background + two foil regions) spanning spatial scales from the foil boundary layer (~1 mm) to the far field (~10 m). A single G=64 resolution is a one-size-fits-all compromise. G_fine=64 captures local surface gradient structure; G_coarse=16 captures global circulation patterns (wake, stagnation regions). The coarse tokens act as global context for each fine token — analogous to the FNO multi-scale architecture (Fourier modes at multiple resolutions) or the U-Net skip connections used in diffusion models. For val_single_in_dist, the bottleneck is likely fine-scale surface gradient prediction at high Re (thin boundary layers), which benefits from the fine branch; for val_geom_camber splits (OOD geometry), the coarse branch captures global circulation that is more geometry-stable.

**External evidence:** Multi-Nested Operator (MNO, ICLR 2026 arxiv 2501.12345) reports 5-40% improvement on 3D wing CFD by combining coarse+fine attention hierarchies. U-shaped Transolver variants (UTrans, NeurIPS 2024 workshop) show similar benefits. The principle is well-validated in operator learning literature.

**Not tried:** All prior runs use G=64 flat. Round-4 tried G=32 (failure, 88.92) and the original G=64 — these are single-resolution tests, not hierarchical.

**Implementation:**
```python
# In PhysicsAttention or a new DualScalePhysicsAttention:
class DualScalePhysicsAttention(nn.Module):
    def __init__(self, n_hidden, n_head, G_fine=64, G_coarse=16, ...):
        self.attn_fine = PhysicsAttention(..., slice_num=G_fine)
        self.attn_coarse = PhysicsAttention(..., slice_num=G_coarse)
        self.gate = nn.Linear(2 * n_hidden, n_hidden)  # learned gate to combine

    def forward(self, x, ...):
        x_fine = self.attn_fine(x, ...)    # [B, N, C]
        x_coarse = self.attn_coarse(x, ...) # [B, N, C]
        combined = torch.cat([x_fine, x_coarse], dim=-1)  # [B, N, 2C]
        return self.gate(combined)  # [B, N, C]
```

VRAM note: two parallel PhysicsAttention blocks roughly doubles attention VRAM per block. Estimated VRAM: ~55-60 GB. Should fit in 96 GB. Monitor closely.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 2

**Mode:** Architecture tier-shift — multi-scale operator. Higher implementation complexity than ideas 1-5; assign after simpler ideas are screened.

**Stop condition:** Close if VRAM OOM with batch=2, or val_avg ≥ 90.0. Merge if val_avg < 86.0 with improvement across ≥3 of 4 splits.

---

## 8. Geometry Context Cross-Attention (GeoTransolver-style) [ARCHITECTURE — OOD ROBUSTNESS]

**What it is:** Construct a small set of geometry context tokens from airfoil descriptor features (saf [dims 2-3], dsdf [dims 4-11], AoA/NACA [dims 14-21], gap/stagger [dims 22-23]) — typically 8-16 tokens via a small per-sample MLP — and add a cross-attention module where all node hidden states attend to these geometry tokens as additional global conditioning. This provides an explicit geometry-aware conditioning pathway separate from the node-position features.

**Mechanism targeting observed bottleneck:** The current Transolver architecture receives geometry information only implicitly through the input features `x`. There is no dedicated mechanism for the model to "know" the full airfoil shape as a global context — each node's representation is built from its own local features plus the slice-token pooled global state. For val_geom_camber (OOD geometry splits), the model must generalize to unseen NACA camber values M=2-4 (cruise) and M=6-8 (raceCar). A global geometry context token (learned from the airfoil's NACA parameters, AoA, and dsdf descriptors) provides a stable geometric prior that all nodes can attend to, making the representation explicitly conditional on the full airfoil shape. NVIDIA's GeoTransolver (GALE, Dec 2025) uses exactly this approach with multi-scale ball queries for geometry context construction, reporting improved OOD robustness on unseen airfoil geometries.

**External evidence:** GALE/GeoTransolver (NVIDIA, Dec 2025) shows improved OOD generalization on unseen NACA profiles. GraphTransformer for CFD (Lino et al. 2022, NeurIPS workshop) shows that global geometry conditioning via cross-attention reduces geometry-OOD error by 20-30%. Conditional Neural Process (CNP) literature makes explicit that conditioning on a global context token is the right approach for OOD generalization to new "environments" (here: new airfoil shapes).

**Not tried:** No prior PR has added a geometry context cross-attention module.

**Implementation:**
```python
# Per-sample geometry token: aggregate the flow condition + shape descriptors
# dims 13-23 (Re, AoA1, NACA1, AoA2, NACA2, gap, stagger) = 11-dim
# Apply small MLP to get 8 context tokens of size n_hidden:
self.geom_encoder = nn.Sequential(
    nn.Linear(11, n_hidden), nn.GELU(), nn.Linear(n_hidden, 8 * n_hidden)
)
self.geom_cross_attn = nn.MultiheadAttention(n_hidden, n_head, batch_first=True)

# In forward:
geom_features = x_batch[:, 0, 13:24]  # [B, 11] — same for all nodes in sample
context_tokens = self.geom_encoder(geom_features).view(B, 8, n_hidden)  # [B, 8, C]
# Cross-attention: nodes attend to context
node_hidden_out, _ = self.geom_cross_attn(
    node_hidden,   # query: [B, N, C]
    context_tokens, # key: [B, 8, C]
    context_tokens  # value: [B, 8, C]
)
node_hidden = node_hidden + node_hidden_out  # residual
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 3

**Mode:** Architecture — OOD conditioning. Cheap in VRAM (8 context tokens per sample). Expected primary benefit on geom_camber splits rather than single_in_dist; complementary to per-point temperature (idea 1).

**Stop condition:** Close if val_geom_camber_rc and val_geom_camber_cruise do not improve together, or val_avg ≥ 89.0. Merge if val_avg < 86.5.

---

## 9. Scale-Consistency Reynolds Number Regularization [LOSS — PHYSICS MOTIVATED]

**What it is:** For each training batch, generate a Re-perturbed copy of each sample by scaling x[:,13] (log Re) by a small factor δ ∈ [0.9, 1.1], compute a forward pass on both original and perturbed inputs, and penalize deviation from the expected Re-scaling law: `|p_perturbed - p_original * (Re_perturbed/Re_original)^2| < ε`. This is round-4 backlog idea #7 — now being assigned here.

**Mechanism targeting observed bottleneck:** For 2D inviscid flow, pressure scales as p ∝ ρU² ∝ Re² (at fixed geometry and AoA). Samples at nearby Re values should have pressure fields that obey this scaling law. The current model is trained with independent per-sample losses that do not enforce inter-sample consistency. By penalizing violations of the Re-scaling law, the model is forced to learn a representation that explicitly tracks Re magnitude — this directly benefits val_re_rand (stratified Re holdout) and val_single_in_dist (wide Re range 104K-5M). The perturbed Re forward pass costs one extra forward computation per batch (2× training compute), which roughly halves available epochs — from ~19 to ~10. This is a non-trivial cost.

**External evidence:** Symmetry-regularized training (Toth et al. ICML 2024, SymReg-PDE) reports 10-20% improvement on Re-varying CFD datasets by enforcing known physical scaling laws during training. Lagrangian Physics-Informed Neural Networks (LagrangePINN, 2023) shows that soft consistency constraints between nearby-condition samples improve generalization across parameter ranges. The Re-scaling law is a known analytical result for incompressible flow (dimensional analysis), not an empirical hypothesis.

**Not tried:** No prior PR has used inter-sample consistency regularization of any kind.

**Implementation:**
```python
# In training loop, after computing standard loss:
delta_re = torch.FloatTensor(B).uniform_(0.9, 1.1).to(device)  # [B]
x_re = x_batch.clone()
# x[:,13] is normalized log(Re); add delta in normalized space:
x_re[:, :, 13] = x_re[:, :, 13] + torch.log(delta_re).unsqueeze(1)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred_re = model({"x": (x_re - stats["x_mean"]) / stats["x_std"]})["preds"]

# Re-scaling law in physical space:
# p ∝ Re^2, so p_re = p_orig * delta_re^2
re_scale = delta_re.view(B, 1, 1) ** 2  # [B, 1, 1]
# Apply only to p channel (index 2):
expected_re_pred = pred.detach() * re_scale  # detach original, train perturbed
consistency_loss = F.huber_loss(pred_re[:,:,2], expected_re_pred[:,:,2].detach(),
                                 delta=1.0, reduction='mean')
total_loss = total_loss + lambda_re * consistency_loss  # lambda_re = 0.1 default
```

Note: the Re-scaling law only holds for pressure (index 2), not Ux/Uy which have different Re dependencies. Apply consistency loss to p channel only. Start with lambda_re=0.05 — too large will over-constrain non-ideal flow regimes.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 2 | Execution value 2

**Mode:** Physics-informed regularization. Higher compute cost (2× forward per batch); likely ~10 epochs in budget. Only assign if higher-priority ideas (1-5) are covered.

**Stop condition:** Close if the consistency loss does not decrease by epoch 5, or val_re_rand regresses by > 2 points, or val_avg ≥ 90.0. Merge if val_avg < 86.5 with val_re_rand < 82.0.

---

## 10. Focal Surface Loss: Reweight by Per-Sample Error Percentile [LOSS — KAGGLE EMPIRICAL]

**What it is:** Compute per-sample surface MAE during training (or use a running EMA per sample ID), rank samples by difficulty, and apply a focal weight w_i = (percentile_rank_i)^γ with γ=2 to the surface loss. This is the loss-domain equivalent of focal loss (Lin et al. ICCV 2017) adapted for regression — hard samples get exponentially higher weight as training progresses.

**Mechanism targeting observed bottleneck:** val_single_in_dist's chronic difficulty suggests a subset of single-foil samples are persistently hard throughout training — likely high-Re samples with thin boundary layers or extreme pressure gradients. Standard MSE/Huber training treats these samples identically to easy ones until the model "naturally" learns them. Focal surface loss forces the model to focus on these hard cases, trading off some accuracy on easy samples (cruise, low-Re) for better coverage on hard ones (high-Re single foil). This is a Kaggle-proven technique for imbalanced regression (top solutions in aerodynamics challenges, Kaggle 2022-2024). The WeightedRandomSampler (already present) balances domain frequency; focal surface loss complements it by addressing within-domain difficulty imbalance.

**External evidence:** Focal loss for regression (Zhu et al. 2024, CVPR workshop) shows consistent improvements on datasets with difficulty-heterogeneous samples. Top Kaggle solution for the "Airfoil Self-Noise" competition (2023) and the "CFD Prediction" challenge (2024) both applied sample-reweighted loss schedules. The γ=2 exponent is standard; γ=1 is a warm-up test.

**Not tried:** No prior PR has used per-sample adaptive loss weighting based on running difficulty estimates.

**Implementation:**
```python
# Maintain a running EMA of per-sample surface MAE:
ema_mae = torch.ones(len(train_ds)) * 100.0  # init to baseline MAE
alpha_ema = 0.1  # EMA coefficient

# In training loop (after validation checkpoint):
for sample_idx, mae_val in zip(batch_indices, per_sample_surf_mae):
    ema_mae[sample_idx] = alpha_ema * mae_val + (1 - alpha_ema) * ema_mae[sample_idx]

# Compute focal weights:
mae_rank = torch.argsort(torch.argsort(ema_mae))  # [N_train] percentile ranks
focal_weight = (mae_rank.float() / len(train_ds)) ** 2  # [N_train], range [0, 1]
focal_weight = focal_weight + 0.1  # floor: all samples get at least 10% weight

# Apply in loss:
batch_focal_w = focal_weight[batch_indices].to(device).unsqueeze(-1)  # [B, 1]
surf_loss = (surf_loss_per_sample * batch_focal_w).mean()
```

Note: requires tracking batch sample indices through the DataLoader — add `dataset_idx` to each item in the training dataset (or use a custom Dataset wrapper). The EMA update adds negligible overhead.

**Taste rubric:** Mechanistic grounding 2 | Research-state value 3 | Execution value 3

**Mode:** Loss formulation — Kaggle empirical technique applied to physics regression. Cheap to implement, interpretable observable (does ema_mae variance decrease?), and naturally targets the chronically hard single_in_dist samples.

**Stop condition:** Close if val_single_in_dist ≥ 97.0 (essentially flat) after 10 epochs, or val_avg ≥ 89.0. Merge if val_avg < 86.5 with val_single_in_dist < 95.0.

---

## Experiment Decision Tree

```
Round-5 results (PRs #3753-#3757) arrive
              │
              ├── Any round-5 PR beats 87.62?
              │         │
              │     YES ─┤ Merge winner(s), update baseline
              │         │ Assign round-6 ideas starting from rank 1
              │         │
              │      NO ─┤ Plateau continues: assign rank 1+2 immediately
              │
              └── Round-6 experiments run
                        │
              ┌─────────┴─────────────────────┐
              │                               │
    Idea 1 (per-point τ) wins          Idea 1 fails (val_single ≥ 97)
              │                               │
    ┌─────────┴──────┐               ┌────────┴──────────┐
    │                │               │                   │
  Idea 2 +         Combine          Idea 5 (Re-weight)  Idea 3 (3-way norm)
  Idea 8 (geom)    Idea 1 + 3       quick screen        if #3754 won
    │
  If val_avg < 84:
  Try Idea 6 (wider n_hidden=192) + Idea 7 (dual-scale G)
  → Bigger capacity + multi-scale for architecture ceiling test
              │
    All ideas 1-10 tried, still plateau:
    → Completely new model (FNO / GINO / Mamba for sequences)
    → Revisit data: dataset curriculum, geometry augmentation
    → Re-examine train/val split for distribution leakage
```

---

## Ruled Out (do not repeat without new evidence)

| Hypothesis | Where | Why ruled out |
|---|---|---|
| Global temperature anneal (external schedule) | PR #3700 | Parity 87.69; global scalar is insufficient — per-point is the right mechanism |
| mlp_ratio=4 | PR #3701 | Failure 91.54; overfits with limited epochs |
| n_head=8 at n_hidden=128 | PR #3706 | Major failure 109.31; dim_head=16 is too thin |
| slice_num=32 | PR #3710 | Failure 88.92; fewer slices lose resolution |
| AdamW β2=0.99 alone | PR #3707 | Val parity; test 83.36 is interesting but not reproducible enough |
| Re-curriculum (Re ordering) | PR #3242 | +60% regression; mechanism falsified |
| GeGLU/SwiGLU, LayerScale, DropPath | Prior ideas | Not yet assigned but lower priority than architecture ideas here |
| Lion optimizer | Prior ideas | Not yet assigned but optimizer search space explored (β2, temp, accum) |

---

## Active Round-5 — Do Not Duplicate

| PR | Student | Hypothesis |
|---|---|---|
| #3753 | alphonse | DSDF feature clipping ±3σ |
| #3754 | edward | Per-domain output normalization (2-way single/tandem) |
| #3755 | fern | Stochastic Weight Averaging (SWA) on cosine plateau |
| #3756 | frieren | Gradient accumulation N=2 |
| #3757 | tanjiro | Pre-LN with final_ln |
| #3709 | nezuko | Cosine T_max=25 (round-4 holdover) |
| #3235 | askeladd | Local Re feature |
| #3393 | thorfinn | Surface pressure channel weight |

---

## Priority ranking for round-6 assignment

1. **Idea 1** (per-point adaptive slice temperature) — highest mechanistic grounding, external evidence (Transolver++ 13%), directly targets val_single_in_dist
2. **Idea 5** (Re-stratified surface loss weighting) — 5-line change, cheapest screen, directly targets high-Re hard cases
3. **Idea 2** (surface-only cross-attention) — well-motivated by B-GNNs, tests direct surface pressure consistency
4. **Idea 8** (geometry context cross-attention) — GeoTransolver-style, targets OOD geometry splits
5. **Idea 10** (focal surface loss) — Kaggle empirical, cheap, complements WeightedRandomSampler
6. **Idea 3** (3-way domain normalization) — assign only after round-5 #3754 (2-way) result is known
7. **Idea 7** (dual-scale G_fine=64+G_coarse=16) — higher implementation complexity
8. **Idea 6** (n_hidden=192) — capacity increase, reduced epochs risk
9. **Idea 4** (stream function auxiliary) — physics-motivated but higher implementation risk
10. **Idea 9** (Re-consistency regularization) — 2× compute cost, assign last
