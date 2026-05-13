# SENPAI Research Ideas — 2026-05-13 22:00

Generated for idle students: **askeladd**, **nezuko**, **thorfinn**
Branch: `icml-appendix-charlie-pai2g-48h-r5`
Baseline: val_avg/mae_surf_p = 42.3455 | test_avg = 38.5059 (PR #2307)
Per-split baseline: in_dist=35.4776, camber_rc=60.8311, cruise=27.6517, re_rand=45.4214

Ranked by expected impact × novelty (descending).

---

## Hypothesis 1 (RANK 1) — Assign to: thorfinn

### What it is

FiLM (Feature-wise Linear Modulation) conditioning: inject regime-identity scalars (log(Re), AoA, NACA camber, gap, stagger) into each TransolverBlock via learned affine transforms γ/β applied to hidden activations after LayerNorm.

### Mechanism

The current Transolver applies attention over node features, but the global flow scalars (log(Re) dim 13, AoA foil1/foil2 dims 14/18, NACA foil1/foil2 dims 15-17/19-21, gap dim 22, stagger dim 23) are embedded identically to spatial features — as part of x. This means the model must discover regime identity implicitly through attention, with no explicit pathway to modulate internal representations by physics regime.

FiLM adds a small conditioning MLP that takes the broadcast scalars as input and outputs (γ, β) vectors of dimension n_hidden=96. Each TransolverBlock applies: `h_out = γ * LayerNorm(h) + β` before the attention or MLP sub-layer. This gives the network an explicit, cheap pathway to condition ALL feature transformations on flow regime without changing the spatial attention structure.

Causal target: the averaging-style bimodal empirical law (8+ confirmed experiments) shows the model systematically trades OOD for in-dist whenever exploration is reduced. FiLM addresses a separate hypothesis — that the model's internal representations are regime-agnostic, forcing it to generalize across dramatically different physics regimes (camber_rc M=6-8 OOD) without a clear identity signal. FiLM doesn't add regularization; it adds conditioning expressiveness.

### Why it's fresh

All 83 prior PRs operate on the attention, loss, schedule, capacity, or augmentation axes. FiLM is a distinct level of intervention: it modifies the information flow pathway itself rather than how attention attends or how loss is computed. It has been proven in image generation, neural rendering, climate models, and protein structure prediction as the canonical way to condition a backbone on global properties while keeping spatial processing intact. Zero overlap with any closed axis.

### Predicted delta

Primary hypothesis: camber_rc improvement (-5% to -15%) from explicit geometry conditioning; cruise improvement likely (-3% to -8%); in_dist stable or slight improvement; re_rand mixed. Net val_avg improvement: -3% to -8% (1.3–3.4 absolute on 42.3455). FiLM's track record in physics-conditioned networks is strong — the mechanism directly addresses the camber_rc bottleneck.

### Code complexity

~45 lines in train.py. Add a `FiLMConditioner` module: 2-layer MLP (24 scalar dims → 48 → n_hidden*2) outputting (γ, β) per block. Modify `TransolverBlock.forward` to accept (γ, β) as optional arguments and apply `h = γ.unsqueeze(1) * norm(h) + β.unsqueeze(1)` after LayerNorm. Extract broadcast scalars from x[:, 0, [13,14,15,16,17,18,19,20,21,22,23]] (any node works since these are constant per sample) before feeding to conditioner.

### Failure mode

If the model is already learning implicit regime conditioning through attention, FiLM adds capacity without benefit. The falsifying signal: camber_rc stays flat or gets worse while in_dist improves — same bimodal pattern as the averaging-style experiments. In that case the mechanism is wrong (not a conditioning failure) and the next hypothesis to test is that camber_rc requires structural changes to the attention geometry.

### Reproduce command

```bash
python train.py --epochs 70 --agent thorfinn --experiment_name film_conditioning_broadcast_scalars
```

Expected wall-clock: ~35 min at ~30 s/epoch. Within SENPAI_TIMEOUT_MINUTES=30 at ~58 epoch effective budget; use 70 epochs as target with cosine converging around epoch 58.

---

## Hypothesis 2 (RANK 2) — Assign to: nezuko

### What it is

Auxiliary multi-task loss: add a small prediction head that predicts front-foil NACA camber (dim 15 of x) from the final TransolverBlock's hidden representations, pooled over surface nodes. The auxiliary MSE loss is weighted λ=0.1 and summed with the primary L1 loss.

### Mechanism

The primary L1 loss only supervises field predictions (Ux, Uy, p). The model receives no explicit signal that camber identity should be encoded in its hidden representations — this must emerge implicitly through the gradient path from surface pressure errors. For the camber_rc OOD split (M=6-8, held out during training), the model has only seen M=2-5 and M=9+ during training, and must interpolate across a gap in camber space.

An auxiliary camber prediction task creates a direct gradient signal forcing the model to maintain camber-discriminative representations in the final hidden layer. This is a well-established multi-task learning technique: the secondary task acts as an inductive bias for the primary task when they share a relevant bottleneck feature. In this case, the bottleneck is "does the hidden representation encode what camber value this airfoil has?"

Architecture: after the final TransolverBlock output `z` (shape [B, N, 96]), apply `mask`-weighted mean pooling over surface nodes → `z_surf` (shape [B, 96]) → 2-layer MLP head (96 → 32 → 1) → scalar camber prediction. Auxiliary loss = λ * MSE(pred_camber, true_camber_normalized). The true_camber for each sample is `x[:, 0, 15]` (dim 15, constant per sample). Total loss = L1_field + 0.1 * MSE_camber.

### Why it's fresh

This is the first auxiliary objective class proposed in this research programme. All 83 prior PRs supervised only the field predictions (Ux, Uy, p). Multi-task learning with auxiliary geometry objectives is well-studied in mesh-based physics surrogates (e.g., Meshgraphnets with auxiliary force prediction, PointNet++ with auxiliary normal prediction) and in aerodynamics (RANS surrogates with auxiliary lift/drag heads). The mechanism is completely orthogonal to all capacity, schedule, and attention axes explored so far.

Critically: the auxiliary task target is already available as an input feature (dim 15) — zero labeling cost, no data changes, the head just needs to recover the input identity from the internal representation. This is a self-supervised geometry consistency constraint.

### Predicted delta

Primary hypothesis: camber_rc improvement (-3% to -12%) from enforced camber-discriminative representations; cruise moderate improvement (-2% to -5%); in_dist may see small benefit or be neutral; re_rand neutral. The risk is that λ=0.1 is too aggressive and hurts primary metric — but at 0.1 the auxiliary loss is approximately 10% of typical primary L1, which should not dominate. If camber_rc improves without harming in_dist, this would be the first result that breaks the bimodal pattern because the mechanism is additive (more supervision) not regularizing (less exploration).

### Code complexity

~25 lines in train.py. Add `CamberHead` module: 2-layer MLP (96→32→1). After computing primary L1 loss, extract `z_surf = (z * mask_surf.unsqueeze(-1)).sum(1) / mask_surf.sum(1, keepdim=True)`, pass through head, compute MSE against `x_norm[:, 0, 15]` (already normalized), scale by λ=0.1, add to total loss. Detach the auxiliary gradient from affecting the LayerScale parameters to avoid interference — compute auxiliary loss only through the main body weights.

### Failure mode

Two failure modes. (1) λ too high: auxiliary task dominates and hurts primary metric uniformly — check if all splits degrade vs. only in_dist. (2) Auxiliary task too easy or already implicitly satisfied: no gradient signal added. The falsifying signal is camber_rc flat or worse while all other splits improve. If that happens, the geometry identity is already represented and the bottleneck is the decoder's use of that representation, not its presence.

### Reproduce command

```bash
python train.py --epochs 70 --agent nezuko --experiment_name aux_camber_head_lambda01
```

Expected wall-clock: ~35 min (auxiliary head adds <1% compute). SENPAI_TIMEOUT_MINUTES=30 governs hard wall-clock.

---

## Hypothesis 3 (RANK 3) — Assign to: askeladd

### What it is

SAM (Sharpness-Aware Minimization) optimizer: replace the standard AdamW gradient step with a two-step procedure — perturb weights to the locally worst-case point on the loss surface (ε=0.05), recompute the gradient there, restore original weights, then apply the AdamW update using the perturbed gradient.

### Mechanism

SAM minimizes both loss value AND loss sharpness simultaneously. The update rule is: `w_{t+1} = w_t - η * ∇L(w_t + ε * ∇L(w_t)/‖∇L(w_t)‖)`. The perturbation ε=0.05 is the "neighborhood radius" — how far to step toward the locally worst weights before computing the actual gradient direction.

Why this might help the bimodal law: the 8-experiment empirical law shows that any technique reducing late-stage exploration (EMA, lr decay, grad-clip, etc.) consistently hurts OOD. SAM's mechanism is different — it does not reduce exploration; it changes the optimization target from "find any low-loss basin" to "find a flat low-loss basin." Flatter minima are strongly associated with better generalization in the literature (Hochreiter & Schmidhuber 1997, Foret et al. 2021). For OOD generalization specifically, flat minima are the correct inductive bias because they are less sensitive to the distribution shift between train (M=2-5,9+) and test (M=6-8) geometry.

SAM is mechanistically the most distinct candidate from all 83 prior experiments: it operates at the optimization geometry level rather than the model or loss level.

### Why it's fresh

SAM has been proven in image classification, NLP, and graph neural networks (Dai et al. 2021 show SAM improves GNN generalization). No CFD surrogate paper in the search space has combined SAM with physics-aware attention over irregular meshes. More importantly, SAM specifically targets the generalization gap the bimodal law reveals — the current model is likely finding sharp minima that fit the training distribution (M=2-5,9+) but do not generalize across the camber gap. SAM is the canonical technique for this mechanism.

### Predicted delta

Primary hypothesis: camber_rc improvement (-5% to -15%) from flatter minima with better OOD generalization; in_dist neutral to -3%. Net val_avg: -3% to -7%. The cost: SAM requires 2 forward+backward passes per step. At ~30.8 s/epoch currently, SAM epochs will be ~55-60 s/epoch → approximately 29-30 effective epochs in the SENPAI_TIMEOUT_MINUTES=30 wall-clock budget. This halves training duration. Use `--epochs 70` so that the timeout governs (30 min ÷ 55 s/epoch ≈ 33 epochs actual). The risk is insufficient training — 33 epochs may be too few for convergence. The cosine schedule should be set relative to actual epochs, not requested epochs; the student should monitor convergence.

### Code complexity

~20 lines in train.py. SAM implementation: `class SAM:` wraps AdamW. `first_step(zero_grad=True)`: compute ∇L, normalize, perturb `w += ε * g/‖g‖`, store perturbation. `second_step(zero_grad=True)`: restore w (subtract perturbation), apply AdamW update using the perturbed gradient. Training loop: `loss.backward(); optimizer.first_step(zero_grad=True); loss2 = model(...); loss2.backward(); optimizer.second_step(zero_grad=True)`. Keep `torch.compile(dynamic=True)` on the model — the two-pass structure is outside the compiled region.

Key detail: SAM requires the loss to be computed **twice per step** with the same batch. Store the batch before the first forward pass and reuse it.

### Failure mode

If the model's generalization gap is caused by insufficient representation capacity (camber_rc needs different model activations that do not exist) rather than sharp minima, SAM will not help. The falsifying signal is val_avg flat or worse with the halved epoch budget. Also watch for SAM + cosine schedule mismatch — if the schedule runs 70 epochs but only 33 complete, the learning rate will be too high throughout (never reaching the low-lr convergence phase). The student should log the actual number of completed epochs and compare to a 33-epoch AdamW baseline if SAM fails.

### Reproduce command

```bash
python train.py --epochs 70 --agent askeladd --experiment_name sam_optimizer_eps005
```

Note: wall-clock will cap at SENPAI_TIMEOUT_MINUTES=30 (~33 actual epochs with SAM's 2× per-step cost). Monitor convergence vs. baseline at epoch 33.

---

## Assignment Summary

| Student    | Hypothesis                         | Rank | Expected delta on val_avg |
|------------|-----------------------------------|------|---------------------------|
| thorfinn   | FiLM conditioning                 | 1    | -3% to -8%               |
| nezuko     | Auxiliary camber head (λ=0.1)     | 2    | -2% to -6%               |
| askeladd   | SAM optimizer (ε=0.05)            | 3    | -3% to -7%               |

All three hypotheses are mechanistically orthogonal to each other and to all 83 prior experiments. All three target the camber_rc OOD bottleneck (60.83) through distinct causal paths: (1) explicit regime conditioning, (2) supervised geometry representation, (3) flat-minima optimization geometry. All comply with read-only data/, no new packages, no W&B, and SENPAI_TIMEOUT_MINUTES=30.
