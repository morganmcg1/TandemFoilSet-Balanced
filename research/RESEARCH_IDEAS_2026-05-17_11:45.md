# Research Ideas — geom_camber_rc Plateau Attack
**Generated:** 2026-05-17 11:45 UTC  
**Branch:** icml-appendix-willow-pai2i-48h-r4  
**Current plateau target:** test_geom_camber_rc = 53.02 (PR #4550 baseline)  
**Primary stack:** `--n_hidden 176 --epochs 14 --use_bf16 --use_lion --lion_lr 1e-4 --lion_wd 1e-3 --use_qk_norm --use_per_foil_coords`

## Context: Why geom_camber_rc is uniquely hard

The `geom_camber_rc` split is racecar-tandem + camber OOD: training NACA M=3-5, test M=6-8 (high-camber).
High camber moves the pressure peak forward (near LE) and sharpens the suction spike. The model has never
seen this shape regime. Per-foil-coords (#4550) gave no gain here (unlike cruise −4.33 and re_rand −4.70),
confirming that translation-invariance alone is insufficient. The remaining gap is almost certainly one or
more of: (a) insufficient geometric representation of camber curvature, (b) inability to extrapolate the
strong LE suction spike at M=6-8, (c) lack of orientation invariance (AoA interacts with high camber in
nonlinear ways), or (d) slice-attention collapse preventing specialisation on high-camber nodes.

---

## In-flight (do NOT duplicate)

| PR | Mechanism |
|---|---|
| #4583 | Per-foil AoA rotation |
| #4584 | Per-node 2D RoPE BEFORE slice aggregation |
| #4586 | Slice-routing diversity loss (Switch-style) |
| #4568 | Adaptive surface focal loss (γ=0.5) |
| #4567 | Camber-M jittering (σ=0.5 on NACA M param) |
| #4548 | LE-emphasis-only loss (w_le=3.0) |
| #4551 | Stokes incompressibility aux loss |
| #4535 | LinearNO linear attention |

---

## Hypothesis 1: NACA Analytic Camber Line as Input Feature

**Title (≤70 chars):** NACA camber-line y_c(x) as input feature (4 values)

**Hypothesis:** The model currently receives NACA-M as a scalar global parameter (feature indices 7-9 for
foil-0: `NACA0 = [M, P, T]`, indices 11-13 for foil-1). It never sees the *spatial* shape of the camber
line at each node's chordwise position. At OOD camber values (M=6-8), the spatial profile changes
nonlinearly — the suction peak moves forward and sharpens. Appending the analytically-computed NACA camber
line value `y_c` and slope `dy_c/dx` at each node's chordwise position gives the model a direct
"where-on-the-camber-geometry-am-I" signal, enabling it to extrapolate to unseen M values whose spatial
profiles differ strongly from M=3-5.

**Mechanism:** For each node with chord-relative coordinate `x_chord` (already computed by #4550), evaluate
the NACA 4-series analytic camber line:
```python
M = x[:, 7]  # NACA-M parameter (range 0..1 in the dataset, so M_real = M * 10 * 0.01)
P = x[:, 8]  # NACA-P parameter
# NACA 4-series camber line and slope (per-node, vectorised)
p = P * 0.1  # chordwise position of max camber (0..1)
m = M * 0.1  # max camber as fraction of chord (0..1)
xc = x_chord  # chordwise position of each node (from per_foil_coords, already in features)
# y_c at each node:
# if xc <= p: y_c = m/p^2 * (2p*xc - xc^2)
# else:       y_c = m/(1-p)^2 * ((1-2p) + 2p*xc - xc^2)
# slope dy_c/dx similarly
# Append y_c (1 dim) and dy_c/dx (1 dim) → 2 new features
```
Implementation in `train.py::_build_features()` before normalization, using `x_chord` already computed for
`--use_per_foil_coords`. Requires foil-id routing (already present). Zero infra overhead.

**Expected effect on geom_camber_rc:** Direct geometric grounding for the shape extrapolation. During
inference on M=6-8, the analytic formula correctly computes the deeper camber profile, so the model sees
a qualitatively different spatial signal rather than relying on global NACA-M scalar to generalise.
Cruise and re_rand should also benefit modestly (less OOD impact). Single_in_dist may be neutral.

**Falsification:** If geom_camber_rc does not improve by >1% (>0.53 units on 53.02), the model is not
limited by camber-spatial-grounding; the bottleneck is elsewhere (e.g., slice-attention collapse,
orientation, LE pressure amplitude).

**Implementation effort:** Low (10-15 lines of vectorised numpy/torch in build_features, uses existing
x_chord and NACA params from input).

**Compounds with #4550?** Yes — uses `x_chord` already computed; entirely additive 2-dim feature append.

**Reference:** NACA 4-series analytic formula (Abbott & von Doenhoff 1959, Theory of Wing Sections, p.113);
applied to neural surrogates in Kashefi & Mukerji 2022 (PointNet++ for airfoil) and Li et al. 2023 ICLR
(FNO airfoil).

---

## Hypothesis 2: LE Radius as Explicit Input Feature

**Title (≤70 chars):** LE-radius r_LE as per-foil explicit input feature

**Hypothesis:** For NACA 4-series foils, the leading-edge radius `r_LE = 1.1019 * T^2` (where T is the
thickness parameter). At high camber (M=6-8), the LE region experiences a dramatically sharper suction
spike (#4511 diagnostic: LE error is 2× larger than TE). The model currently has NACA-T as a scalar but
not `r_LE` directly. Adding `r_LE` as a per-foil scalar (one value broadcast to all nodes of that foil)
gives the model a direct proxy for "how sharp is this LE", which correlates directly with the magnitude
of the suction peak — the dominant source of error on geom_camber_rc.

**Mechanism:** In `_build_features()`, compute `r_LE_f0 = 1.1019 * NACA0[2]^2` and `r_LE_f1 = 1.1019 *
NACA1[2]^2`, then append as foil-id-split broadcast features (broadcast to all nodes of the respective
foil). Combined with `x_chord`, this gives the model both "where on the chord" and "how sharp is the tip
I'm near" simultaneously.

**Expected effect on geom_camber_rc:** Should reduce LE-prediction error specifically. At OOD camber, the
actual LE radius is unchanged from training distribution (thickness T is in-distribution), but the camber
line geometry changes the effective local curvature at the LE. Still, the explicit r_LE signal may help
the model to condition its output magnitude on LE geometry.

**Falsification:** If LE-region MAE does not decrease on geom_camber_rc (best measured by per-region
val/test breakdown), the bottleneck is not LE-shape encoding but something else (perhaps orientation or
amplitude extrapolation).

**Implementation effort:** Very low (3-5 lines). Can be combined with Hypothesis 1 as a single PR.

**Compounds with #4550?** Yes — purely additive feature.

**Reference:** LE-radius geometry literature: Dacles-Mariani & Zilliac 1995; applied to neural surrogates
in Chen et al. 2021 (airfoil inverse design with LE sensitivity). LE dominance diagnostic from askeladd
#4511.

---

## Hypothesis 3: Camber-Augmentation via NACA Scalar Interpolation (Beyond σ=0.5 Jitter)

**Title (≤70 chars):** Camber OOD bridge: NACA-M uniform-sample in [0.03, 0.09] at train time

**Hypothesis:** fern's in-flight #4567 tests Gaussian jitter σ=0.5 on NACA-M (meaning ≈ ±0.5 NACA units
change). The training distribution has M=3-5 and the OOD test has M=6-8. A Gaussian with σ=0.5 centered
at the training values barely reaches M=6 (≈ 2σ away from M=5). Instead, using **uniform sampling from
[0.03, 0.09]** (the full M range, scaled consistently with how M is stored in the dataset's 24-dim
feature vector) would provide direct in-distribution training examples spanning the OOD camber range.
This is strictly stronger exposure than jitter and avoids the problem that Gaussian jitter concentrates
probability mass in [3.5, 4.5] not [6, 8].

**Mechanism:** In `train.py`, during augmentation, for a fraction `aug_frac` of each batch, replace
`x[:, 7]` (NACA0_M) and/or `x[:, 11]` (NACA1_M) with a uniform sample in [0.03, 0.09] (the full feasible
camber range in feature-space units). Since the mesh coordinates do NOT change (unique topology constraint
#4530), the target y stays fixed — this is "conditional augmentation" where we relabel the condition but
keep the field. This is an explicit extrapolation probe: does training with stated M=6-8 conditions (on
M=3-5 bodies) teach the model to map high-M scalars to higher-suction outputs?

The key difference from #4567 (in-flight) is:
1. Uniform vs Gaussian → guarantees OOD coverage
2. [0.03, 0.09] full span vs ±0.5 small jitter
3. Explicit flag `--camber_aug_mode uniform --camber_aug_frac 0.3`

**Expected effect on geom_camber_rc:** If the model can learn to extrapolate from the camber-conditioning
scalar to the right pressure field shape (even though the mesh doesn't change), this directly builds
bridges to the test distribution. Expected: geom_camber_rc improves 2-5%; other splits neutral.

**Falsification:** If geom_camber_rc does not improve with full-range uniform sampling, then the OOD gap
is NOT due to insufficient exposure to high-M scalars — it is architectural (the model can't extrapolate
the camber→pressure relationship from scalar alone).

**Implementation effort:** Low (differs from #4567 only in sampling strategy; same code path).

**Compounds with #4550?** Yes.

**Reference:** Scalar-condition augmentation in CFD surrogates: Thuerey et al. 2020 (DeepFlowPrediction,
Re-augmentation); Li et al. 2023 (FNO condition broadening). Out-of-distribution coverage via uniform
augmentation vs Gaussian: Chen et al. 2024 ICLR (DomainPrism).

---

## Hypothesis 4: Per-Node Chord-Normalised Curvature Signal (κ·c)

**Title (≤70 chars):** Per-node surface curvature κ×chord as input feature

**Hypothesis:** The dataset provides DSDF features (8-dim, approximate distance-to-surface-function). These
encode proximity to surfaces but NOT curvature. The `geom_camber_rc` split has higher-camber foils where
the LE and maximum-camber regions have qualitatively different curvature profiles. A per-node curvature
estimate — computed from the DSDF gradient or from finite-differencing along ordered surface nodes —
could give the model a local shape signal that scales with M. Note: the earlier curvature-proxy loss
(#4110) was based on DSDF-norm and failed at the LOSS level; this hypothesis is at the INPUT level (a
feature, not a loss), and uses curvature as a local geometric signal to help the encoder rather than as
a loss weight.

**Mechanism:** For surface nodes (is_surface=1), compute approximate surface curvature via the 8-dim DSDF
features. One approach: take the Laplacian of the signed-distance function ∇²SDF, which equals the mean
curvature of the level set. From the 8-directional DSDF gradient `dsdf[:, :8]`, approximate curvature:
`kappa ≈ (dsdf_x_plus + dsdf_x_minus - 2·dsdf_center) / dx^2` using pairs from the 8-directional sample.
Append `kappa * chord_length` (chord-normalised) as 1 new feature. For interior nodes, set kappa=0.

**Expected effect on geom_camber_rc:** Curvature is highest at the LE (the bottleneck region) and at the
camber arc peak. At OOD M=6-8, the camber arc has higher curvature than training distribution. The
explicit curvature signal lets the model "know" it's at a higher-curvature configuration and map to the
expected stronger suction spike.

**Falsification:** If val loss on geom_camber_rc (and LE-region MAE specifically) does not improve, the
model is not limited by local shape signal from curvature — the bottleneck is global field amplitude.

**Implementation effort:** Medium (requires finite-diff from DSDF columns; validate that DSDF orientation
is consistent across meshes). ~30-40 lines including numerical safety.

**Compounds with #4550?** Yes — additive feature that doesn't touch x_chord logic.

**Reference:** DSDF features documented in program.md as 8-directional samples. Surface curvature from
SDF Laplacian: Carr et al. 2001 (implicit surface reconstruction); applied to mesh-free PDE surrogates
in Yin et al. 2023 (SHRED, curvature-conditioned).

---

## Hypothesis 5: AoA-Conditioned Output Scaling via FiLM Layer

**Title (≤70 chars):** FiLM: AoA0/AoA1 condition output projection (1 FiLM per block)

**Hypothesis:** Transolver's output is produced via `mlp2(Sigma(slice_output))` where `Sigma` aggregates
slice tokens back to nodes. The aggregation and output projection are AoA-agnostic (the model must learn
to embed AoA globally through the attention). On `geom_camber_rc`, which involves racecar tandems with
high AoA (aggressive angles) + high camber — a nonlinearly interacting pair — the model may not have
enough capacity to modulate its output magnitude by AoA. FiLM (Feature-wise Linear Modulation) adds a
per-block affine transform `y = γ(c) * x + β(c)` where `c` is the flow condition vector `[log_Re, AoA0,
AoA1, gap, stagger, NACA0, NACA1]`. This is parameter-cheap and does not change the network depth.

**Mechanism:** After each TransolverBlock's output (before residual add), apply `FiLMLayer(n_hidden, dim_c)`:
```python
class FiLMLayer(nn.Module):
    def __init__(self, n_hidden, dim_cond=12):
        super().__init__()
        self.gamma_net = nn.Linear(dim_cond, n_hidden)
        self.beta_net  = nn.Linear(dim_cond, n_hidden)
    def forward(self, x, c):
        return self.gamma_net(c).unsqueeze(1) * x + self.beta_net(c).unsqueeze(1)
```
Condition vector c = concat(log_Re, AoA0, AoA1, gap, stagger, NACA0_M, NACA0_P, NACA0_T, NACA1_M,
NACA1_P, NACA1_T) [dim=11]. Broadcast from sample-level to all nodes. Start with FiLM on the LAST block
only (cheapest probe, most output-relevant).

**Expected effect on geom_camber_rc:** The (high-camber, high-AoA) interaction is precisely what FiLM
should capture. At high camber, the optimal pressure coefficient for a given AoA is qualitatively
different (suction peak shifts forward, higher magnitude). FiLM gives the model a fast pathway to
modulate global output scale/shift by condition, without relying purely on learned embeddings.

**Falsification:** If geom_camber_rc does not improve by >1% with FiLM on last block, either (a) the
bottleneck is local-geometry shape representation (not global conditioning), or (b) the model already
learns equivalent conditioning through its embedding pathway.

**Implementation effort:** Medium (new FiLMLayer class, insert into TransolverBlock; ~40 lines).

**Compounds with #4550?** Yes — orthogonal to per-foil-coords.

**Reference:** FiLM conditioning: Perez et al. 2018 NeurIPS (visual question answering); applied to
physics surrogates in Kochkov et al. 2021 (Machine Learning Accelerated CFD, condition modulation); used
in Poseidon 2024 (FNO + FiLM for multi-physics).

---

## Hypothesis 6: Multi-Scale Slice Attention (Hierarchical Slice Pyramid)

**Title (≤70 chars):** Hierarchical slice_num pyramid: layers 0-1=128, 2-3=64, 4=32

**Hypothesis:** Current Transolver uses uniform slice_num=64 across all 5 layers. Physics near the foil
surface (LE pressure spike, wake structure) requires fine-grained spatial specialisation (many small
slices), while global field structure (pressure far-field, turbulent wake) requires coarse spatial
representation (few large slices). For geom_camber_rc, the error is dominated by the LE region where
slice_num=64 may be too coarse to capture the sharp suction spike at M=6-8. A pyramid schedule
(128→64→32 across depth) gives early layers fine spatial resolution and later layers global context.

Note: slice_num=96 and 128 were tried (CLOSED: #4140, #4092) but uniformly across ALL layers. The key
difference here is the HETEROGENEOUS schedule — coarser in later layers saves compute while giving early
layers more geometric resolution. slice_num=128 uniform ADDED 30% compute; pyramid 128/128/64/64/32
adds ~15%.

**Mechanism:** Add `--slice_num_schedule "128,128,64,64,32"` flag parsed in `train.py::TransolverConfig`;
pass per-layer slice_num to each `TransolverBlock`. The slice attention projections `in_project_slice`
and slice-to-node aggregation `out_project` are already per-block — only the integer `slice_num` needs to
be passed per block. Cross-layer compatibility (slice outputs feed into next block's node features via
residual, not slice-to-slice) means no reshaping issue.

**Expected effect on geom_camber_rc:** Early layers with slice_num=128 can specialise slices to the small
LE zone (high-camber suction region). Layers 2-4 aggregate global context. Expected improvement in LE
MAE → cascades to geom_camber_rc improvement. Other splits: neutral or modest gain (they have less
severe LE spike OOD).

**Falsification:** If val/test geom_camber_rc does not improve vs uniform slice_num=64, the bottleneck is
not slice granularity in early layers — it may be slice COLLAPSE (addressed by #4584/#4586 in-flight).

**Implementation effort:** Medium (per-block slice_num plumbing; ~20 lines in train.py config + block
forward). No new weight shapes needed beyond re-parameterising existing projections.

**Compounds with #4550?** Yes. Note: wait for #4586 slice-diversity loss results first — if slice collapse
is confirmed fixed, pyramid scheduling becomes higher-value.

**Reference:** Hierarchical attention in vision transformers: Swin Transformer v2 (Liu et al. 2022, CVPR);
applied to PDE surrogates: Hao et al. 2023 ORCA (multi-scale attention operators). Pyramid schedule
concept from ResNet-style conv: He et al. 2016.

---

## Hypothesis 7: Separable Per-Foil Normalization (running stats per foil-domain)

**Title (≤70 chars):** Separate running norm stats for foil-0 vs foil-1 nodes

**Hypothesis:** Current normalization (`x = (x - mean) / std`) uses global statistics over all nodes. On
tandem configurations, foil-0 and foil-1 experience qualitatively different flow conditions (foil-1 is
in the wake of foil-0 → lower effective AoA, different pressure levels). On geom_camber_rc (racecar
tandem), the gap/stagger geometry is aggressive — foil-1 is deeply in foil-0's wake. Global normalization
means the model sees a mixed distribution. Separate running statistics per foil (foil_id = 0/1, already
available from #4550) lets each foil's features sit in a well-normalized range conditioned on their
typical values.

**Mechanism:** In `data/loader.py` (read-only), normalization stats are precomputed. In `train.py`, after
loading, apply per-foil normalization split:
```python
mask_f0 = (x[:, :, foil_id_idx] < 0.5)  # foil_id feature column
mask_f1 = ~mask_f0
# Separate mean/std for f0 and f1, applied at input time
```
Alternatively, add `--use_per_foil_norm` flag: during forward pass, split input tensor by foil_id,
normalize each half with foil-specific statistics computed from training set, recombine.

**Expected effect on geom_camber_rc:** More numerically stable conditioning for the in-wake foil on
racecar geometry. Should disproportionately help geom_camber_rc (racecar tandem with large wake
interaction) vs cruise (parallel chord). geom_camber_cruise may also improve; single_in_dist likely
neutral.

**Falsification:** If geom_camber_rc does not improve, global normalization is adequate; the bottleneck
is shape representation, not feature-range calibration.

**Implementation effort:** Low-Medium (requires splitting stats computation by foil_id in training loop;
~20-30 lines). Must validate that foil_id heuristic from #4550 is 100% accurate for all train/test
samples in racecar domain.

**Compounds with #4550?** Yes — uses foil_id feature already computed.

**Reference:** Domain-specific normalization in multi-domain surrogates: Dolean et al. 2023 (PINNSFORMER,
per-domain batch norm); per-group normalization in transformers: Wu & He 2018 (Group Normalization); in
mesh PDE: Herde et al. 2024 Poseidon (domain-aware feature preprocessing).

---

## Hypothesis 8: High-Camber Prototype Loss (Triplet Margin on NACA-M Distance)

**Title (≤70 chars):** Camber-prototype metric loss: pull M=5 toward M=6-8 repr space

**Hypothesis:** The model's latent representation organizes samples by the features it sees. For
geom_camber_rc, the test samples have NACA-M=6-8 while training has M=3-5. If the model learns a latent
space where representation distance increases with NACA-M distance, it will be forced to EXTRAPOLATE
rather than interpolate. A prototype-metric auxiliary loss (softNN: pull samples with similar M closer
in latent space) would regularize the latent to be smooth in the M dimension, reducing the extrapolation
gap.

**Mechanism:** Compute a per-sample camber-prototype embedding by average-pooling the Transolver block
outputs: `z_i = mean_pool(TransolverOutput_i)`. Apply a smooth-NACA-M soft contrastive loss:
```
L_proto = mean_i[ mean_j[ (M_dist(i,j) / tau) * ||z_i - z_j||^2 ] ]
```
where M_dist = |M_i - M_j| / 0.1 (normalized distance in NACA-M space), tau = 0.5. This is similar to
regression metric learning: samples with similar M should have similar embeddings. λ=0.01 weighting.

**Expected effect on geom_camber_rc:** If the latent representation is smooth in M, the model can
linearly interpolate/extrapolate representations from M=5 to M=6 instead of relying on pure parameter
generalization. Should specifically help the M-extrapolation gap.

**Falsification:** If geom_camber_rc does not improve by >1%, the bottleneck is not latent M-smoothness
but rather the output decoder's inability to map smooth representations to correct high-M pressure
fields.

**Implementation effort:** Medium (~40 lines for prototype pooling + contrastive loss).

**Compounds with #4550?** Yes — operates on latent representations, independent of input features.

**Reference:** Metric learning for regression: Yao et al. 2022 (ECCV OrdinalCLIP); contrastive
regression for distribution shift: Cui et al. 2023 (CVPR BalancedContrastiveRegression). Applied to
CFD: Meng et al. 2022 (contrastive pre-training for aerodynamic surrogates).

---

## Hypothesis 9: Separate Surface-Interior Pathways (Dual-Branch Architecture)

**Title (≤70 chars):** Dual-branch: separate Transolver blocks for surface vs interior

**Hypothesis:** All 5 TransolverBlocks currently process surface and interior nodes together. The geom_camber_rc
error is dominated by LE surface pressure (diagnostic #4511). The pressure prediction at surface nodes
(which are the target for mae_surf_p) requires fine-grained geometric awareness; interior nodes contribute
to the global flow field but not to the primary metric. A dual-branch architecture gives surface nodes
their own specialised attention pathway (possibly with higher slice_num or different conditioning),
while interior nodes share a lighter pathway. The branches share a common trunk for the first 2 layers
then split at layer 3.

**Mechanism:** After block 2, split node features into surface/interior subsets (via `is_surface` mask,
already in the batch). Apply block 3-4 as "surface branch" (on surface nodes only, with a separate
smaller-capacity block for interior). Re-merge before the output projection. Approximate implementation:
share block weights 0-2, then:
```python
x_surf = x[is_surface]     # [N_surf, n_hidden]
x_int  = x[~is_surface]    # [N_int, n_hidden]
x_surf = surface_block_3(x_surf)   # full TransolverBlock on surface only
x_int  = interior_block_3(x_int)   # half-capacity block (n_hidden//2)
x[is_surface] = x_surf
x[~is_surface] = x_int
```
This modestly increases surface-path parameters (~+15%) while reducing interior-path total compute.

**Expected effect on geom_camber_rc:** Surface path can specialise to the LE pressure gradient structure
without being pulled toward interior-node representations. Should reduce LE error on OOD camber.

**Falsification:** If geom_camber_rc surface LE error does not decrease, the bottleneck is not insufficient
surface-specific capacity — it may be global conditioning (FiLM) or geometric encoding.

**Implementation effort:** High (~60-80 lines; careful handling of variable surface counts per sample,
mask broadcasting, padding compatibility). Recommend implementing after cheaper input-feature hypotheses
are tested.

**Compounds with #4550?** Yes — orthogonal.

**Reference:** Surface/volume split in CFD surrogates: Hsieh et al. 2019 (learnable surface encoders);
dual-branch attention in mesh learning: Cao et al. 2022 (SurfaceFormer); sparse transformer for
surface-only prediction: Lienen et al. 2023 (BSMS-GNN).

---

## Hypothesis 10: Thin-Airfoil Theory Baseline Residual (Physics-Prior Output)

**Title (≤70 chars):** Predict Δp from thin-airfoil theory baseline (residual target)

**Hypothesis:** Thin-airfoil theory gives an analytic first-order approximation for the surface pressure
coefficient Cp(x) for a cambered profile at small angles:
`Cp(x) ≈ (2/U_inf) * sum_{n=0}^{N} A_n * (cos(n*θ)) / sin(θ)` (Fourier-series expansion of camber
slope dz/dx). For the NACA 4-series specifically, this integral can be evaluated analytically.
Instead of predicting the raw pressure `p`, train the model to predict `Δp = p_CFD - p_thin_airfoil`.
This residual is smaller in magnitude and has a well-defined zero crossing at the thin-airfoil regime
(M=3-5) — the model only needs to learn corrections. At OOD M=6-8, thin-airfoil theory still provides
a reasonable leading-order estimate, so the residual is smaller and smoother than the raw pressure.

**Mechanism:**
1. Precompute per-node thin-airfoil Cp baseline using NACA parameters and AoA (fast analytic formula).
2. At training time, compute residual target `y_residual = y_CFD - y_theory`.
3. Train the model to predict `y_residual` (same architecture, same loss, same normalization — but
   different target).
4. At inference time, `y_pred = model(x) + y_theory(x)`.

Thin-airfoil theory formula for pressure (from NACA analytic):
```python
def thin_airfoil_cp(x_chord, M, P, T, alpha_rad):
    # Fourier decomposition of NACA camber slope
    # Returns Cp(x_chord) per-node analytic estimate
    ...  # ~30 lines
```

**Expected effect on geom_camber_rc:** The residual from M=6-8 thin-airfoil theory is MORE similar to
the residual from M=3-5 (both are corrections on top of the same analytic baseline) than the raw
pressures are. The model sees a smoother training signal and extrapolates better across M values.
geom_camber_rc should improve substantially; other splits should be neutral or slightly worse (they don't
need the physics prior as much).

**Falsification:** If the model trained on residuals does NOT improve geom_camber_rc vs training on raw
p, then the OOD difficulty is not explained by pressure magnitude extrapolation — the model already
handles that through normalization/scaling.

**Implementation effort:** Medium-High (analytic thin-airfoil formula implementation, vectorized per node
and per sample; validation against known Cp distributions). ~60-80 lines. High scientific value even if
it fails (reveals whether physics-prior output targeting is useful for this dataset).

**Compounds with #4550?** Yes — completely output-side change; orthogonal to all input features.

**Reference:** Thin-airfoil theory: Glauert 1926, NACA TR-84 (Theodorsen formulation); residual
prediction in physics surrogates: Kochkov et al. 2021 (ML-accelerated simulation residuals); applied
to pressure prediction: Bhatnagar et al. 2019 (DeepFluid, surrogate correctors); Seidel et al. 2022
(residual prediction for wing lift RANS correction).

---

## Hypothesis 11: NACA-M Extrapolation via Latent Interpolation with Synthetic Targets

**Title (≤70 chars):** Extrapolation via linear latent blend of adjacent-M embeddings

**Hypothesis:** Since the mesh topology is unique per sample (ruling out direct node-level interpolation,
#4530), interpolation in LATENT space is still feasible. Given training samples A (M=3) and B (M=5),
compute their latent embeddings after block-2: `z_A = T2(x_A)`, `z_B = T2(x_B)`. A synthetic latent
for M=7 would be `z_C = 2*z_B - z_A` (linear extrapolation in latent space, or `z_C = z_B + λ*(z_B -
z_A)` with λ tuned). This is "latent extrapolation augmentation": the decoder (blocks 3-5 + output) sees
an extrapolated latent and can learn to produce the correct high-M output. Note: we cannot supervise with
a true M=7 target (none in training set), so this is UNSUPERVISED extrapolation augmentation — the target
`y_C` is itself linearly extrapolated from `y_A` and `y_B`.

**Mechanism:** Within each batch, identify same-domain sample pairs with different M values. For pairs
(A, B) with M_A < M_B, compute synthetic M_C = 2*M_B - M_A with latent z_C and supervised by
y_C = 2*y_B - y_A (linear extrapolation of ground-truth targets — valid if y is locally smooth in M).
Apply for pairs within the racecar-tandem domain only (geom_camber_rc domain). Use α=0.2 probability
of using extrapolated pair.

**Expected effect on geom_camber_rc:** Direct training signal in the M=6-8 extrapolation regime.

**Falsification:** If the latent extrapolation produces degenerate y_C targets (non-physical pressure
fields), or if geom_camber_rc does not improve, then linear interpolation in latent space is not a valid
approximation for the camber dimension — the mapping is too nonlinear.

**Implementation effort:** Medium-High (in-batch pair finding, latent extraction from middle block, y
extrapolation, careful pairing within same-domain racecar-tandem samples). ~50 lines.

**Compounds with #4550?** Yes.

**Reference:** Latent interpolation augmentation: Upchurch et al. 2017 (deep feature interpolation);
CutMix in latent space: Yun et al. 2019 ICCV; applied to PDE surrogates: Yin et al. 2022 (VIDON latent
mixing for PDEs across parameters).

---

## Hypothesis 12: Per-Region Attention Masking (LE-Zone Priority Attention)

**Title (≤70 chars):** Attention soft-mask: LE-region (x_chord<0.15) gets 3x bias

**Hypothesis:** Transolver's slice attention aggregates ALL nodes in the batch with equal weight (no
positional bias). For geom_camber_rc, LE nodes (x_chord < 0.15 ≈ LE zone) are the dominant error source
(#4511 diagnostic). Injecting a learned or fixed attention bias toward LE-zone nodes increases the model's
effective "resolution" in that region during slice routing, causing LE-zone slice centroids to specialize.
This is distinct from LE-emphasis loss (#4548 in-flight) which upweights the LOSS, not the ATTENTION.

**Mechanism:** In `PhysicsAttention.forward()`, after computing slice routing weights `attn =
softmax(linear_project(x_mid) / temp)`, add a fixed bias for nodes where `x_chord < 0.15`:
```python
le_mask = (x_chord < 0.15).float()  # [B, N]
attn = attn + le_bias_scale * le_mask.unsqueeze(-1)  # broadcast to [B, N, slice_num]
attn = softmax(attn, dim=-2)  # re-normalise
```
Test le_bias_scale ∈ {0.5, 1.0} (one arm each). `x_chord` is already available from `--use_per_foil_coords`.
Requires passing x_chord into PhysicsAttention, which is currently only accessible as a derived feature.

**Expected effect on geom_camber_rc:** LE-zone nodes are routed more consistently to specific slices
rather than being uniformly distributed. Slice specialization in the LE region → better LE pressure
prediction. geom_camber_rc should benefit disproportionately (most extreme LE suction spike).

**Falsification:** If geom_camber_rc LE-region MAE does not decrease, the bottleneck is not LE-zone
routing specificity — it is the model's inability to predict the MAGNITUDE of the spike (amplitude
extrapolation problem).

**Implementation effort:** Medium (requires plumbing x_chord into PhysicsAttention; ~25 lines including
config flag). Should wait for #4586 slice-diversity and #4584 pre-agg RoPE results — if those solve
slice collapse generally, this becomes complementary rather than redundant.

**Compounds with #4550?** Yes — uses x_chord already computed.

**Reference:** Spatial attention bias in ViT: Srinivas et al. 2021 (Bottleneck Transformers, positional
bias); applied to mesh transformers: Herde et al. 2024 Poseidon (geometry-conditional attention);
LE-focus in aerodynamic surrogates: Bouhlel et al. 2020 (LE-dense sampling in kriging surrogates).

---

## Priority Ranking

| Rank | ID | Title | Estimated geom_camber_rc gain | Effort | Compounds #4550 | Mode |
|---|---|---|---|---|---|---|
| 1 | H1 | NACA analytic camber-line y_c(x) as feature | 2-4% (direct geometric grounding) | Low | Yes | Frontier refinement |
| 2 | H5 | FiLM: AoA0/AoA1 condition output projection | 2-5% (AoA-camber interaction modulation) | Medium | Yes | Tier shift |
| 3 | H10 | Thin-airfoil theory baseline residual prediction | 3-7% (physics prior reduces OOD amplitude gap) | Medium-High | Yes | Tier shift |
| 4 | H3 | NACA-M uniform augmentation [0.03,0.09] | 2-4% (full OOD coverage) | Low | Yes | Frontier refinement |
| 5 | H6 | Multi-scale slice pyramid 128/128/64/64/32 | 1-3% (LE granularity) | Medium | Yes | Frontier refinement |
| 6 | H2 | LE radius r_LE as per-foil feature | 1-2% (LE conditioning) | Very Low | Yes | Frontier refinement |
| 7 | H7 | Per-foil running normalization stats | 1-3% (tandem in-wake recalibration) | Low-Medium | Yes | Frontier refinement |
| 8 | H12 | Per-region LE attention bias | 1-3% (conditional on #4586 outcome) | Medium | Yes | Diagnostic |
| 9 | H4 | Surface curvature κ×c from DSDF Laplacian | 1-2% (shape signal) | Medium | Yes | Frontier refinement |
| 10 | H8 | Camber-prototype metric loss (triplet on M) | 1-3% (latent M-smoothness) | Medium | Yes | Diagnostic |
| 11 | H9 | Dual-branch surface/interior pathways | 2-4% (surface capacity) | High | Yes | Tier shift |
| 12 | H11 | Latent extrapolation augmentation | 2-5% (direct OOD exposure) | Medium-High | Yes | Tier shift |

## Decision tree for sequencing

```
H1 (NACA camber-line feature) → runs cheap in 14 epochs
  ├─ geom_camber_rc improves >1%: MERGE; next → H5 (FiLM) in parallel with H3 (uniform aug)
  │    ├─ H5 improves: compound H1+H5 → then H10 (residual)
  │    └─ H5 fails: H3 more likely to help (data vs architecture)
  └─ H1 fails (no geom_camber_rc improvement):
       Bottleneck is NOT spatial camber encoding → pivot to H10 (physics prior output)
       AND H5 (FiLM conditioning) simultaneously
         ├─ H10 improves: merge; pivot to H6 (pyramid) and H12 (LE attention)
         └─ H10+H5 both fail: bottleneck is slice collapse (wait for #4584/#4586 results)
              → H6 (pyramid) contingent on slice-diversity results
              → H12 (LE attention bias) as complementary
```
