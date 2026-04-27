# Willow-R5 Round 1 Research Ideas

- **Date**: 2026-04-27 15:30
- **Track**: `willow-r5` (advisor branch `icml-appendix-willow-r5`)
- **Dataset**: TandemFoilSet
- **Primary metric**: `val_avg/mae_surf_p` (4-split equal-weight surface-pressure MAE, lower better)
- **Constraints**: 1 GPU per student, `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`
- **Starting baseline**: Bare Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, MSE loss, `surf_weight=10.0`, AdamW lr=5e-4 wd=1e-4, batch_size=4, cosine 50 epochs)

## Strategy

Willow-r5 begins from the bare baseline; none of the kagent_v_students winners are in. The cleanest, highest-EV first round is to (a) **independently re-verify the strongest single-variable winners** from the parallel track so we own un-aliased ablations for the appendix, and (b) **try a small handful of fresh ideas** that the parallel track did not test. Each hypothesis below is one controllable variable so attribution stays clean. Where a winner is well-supported on the parallel branch, we re-run it solo here as the first node of the willow-r5 dependency graph; later rounds will compound them.

Predicted deltas are from the parallel-track signal where applicable; the magnitudes will scale roughly with their parallel-branch effect on the same baseline.

---

## H1 — `huber-loss-delta-sweep`

- **Family**: loss
- **Change**: replace MSE with Huber, sweep `delta ∈ {0.05, 0.1, 0.2, 0.5, 1.0}` in normalized space; same `surf_weight=10.0`.
- **Predicted delta on `val_avg/mae_surf_p`**: −10 to −20%.
- **Reasoning**: parallel track PR #3 was a clear MERGED win — Huber/L1 closes the well-known gap between MSE training and MAE-graded test. With y already z-scored, residuals are in roughly comparable units across channels, so a single δ sweep is sufficient.
- **CLI**: add `--loss huber --huber_delta {0.05,0.1,0.2,0.5,1.0}` flags; group runs `--wandb_group r5-h1-huber`. One run per δ.
- **Risk**: too small a δ may underfit volume regions and degrade `mae_surf_p` indirectly. Mitigation: keep `surf_weight=10.0` so surface gradient still dominates; report all five δ values.
- **Why high-EV given prior signal**: the single biggest win on the parallel track. Re-establishing it as PR #1 on willow-r5 immediately moves the baseline forward and unlocks compound experiments.

---

## H3 — `fourier-pe-film-re`

- **Family**: data representation + conditioning
- **Change**: add Fourier positional encoding on `(x, z)` (Gaussian-projected, m=80 frequencies, σ ≈ 0.6) concatenated to input features; FiLM (γ, β) modulator conditioned on `log(Re)` injected after the preprocess MLP.
- **Predicted delta on `val_avg/mae_surf_p`**: −15 to −25%.
- **Reasoning**: parallel track PR #7 was MERGED and is the second-largest single win after Huber. The σ ≈ 0.6 / m=160 setting was further verified by PRs #20 and #24.
- **CLI snippet**: in `train.py`, after `x_norm`, build a fixed Gaussian random matrix `B ∈ R^{80×2}`, append `[sin(2π x_xz B^T), cos(2π x_xz B^T)]`. After `self.preprocess`, apply `fx = (1 + γ(logRe)) * fx + β(logRe)` where γ, β are tiny MLPs from log(Re) (dim 1) to dim `n_hidden`.
- **Risk**: adding 160 features to x widens the preprocess MLP and adds ~200K parameters; minor throughput hit. Mitigation: keep n_hidden=128 baseline, just widen the input layer.
- **Why high-EV given prior signal**: well-validated win, with σ already known. High-confidence move; provides a hook for many follow-ups (per-block FiLM later, FiLM on AoA, etc.).

---

## H4 — `slice-num-down-sweep`

- **Family**: capacity / architecture
- **Change**: sweep `slice_num ∈ {16, 24, 32, 48, 64}` (baseline=64). One variable, 5 runs.
- **Predicted delta on `val_avg/mae_surf_p`**: −5 to −10% at the optimum (sn=16–24).
- **Reasoning**: parallel track PRs #27 and #34 both MERGED with sn lowered from 64 toward 16–24; the parallel track found the sweet spot well below 64. Fewer slices means each slice is responsible for a larger physical region — likely a regularizer against per-slice overfitting on small training sets.
- **CLI**: `--slice_num {16,24,32,48,64}`. Group `r5-h4-slicenum`.
- **Risk**: sn too low (<16) may underfit complex tandem flows. Mitigation: include sn=32 and sn=48 as middle-ground anchors so the curve is well-resolved.
- **Why high-EV given prior signal**: low-cost (one flag), and the parallel track's curve suggests sn=64 default is suboptimal. Independent re-verification is cheap and cleanly slots into the appendix.

---

## H5 — `n-layers-down-sweep`

- **Family**: capacity / architecture
- **Change**: sweep `n_layers ∈ {3, 4, 5, 6}` (baseline=5). 4 runs.
- **Predicted delta on `val_avg/mae_surf_p`**: −3 to −8% at nl=3 or nl=4.
- **Reasoning**: parallel track PR #35 (nezuko) MERGED with nl=3 winning over deeper variants; PR #39 verified the compound nl=3 × sn=16 effect. With 1499 training samples, deeper = more overfit risk; surface MAE on the camber-OOD splits should benefit from a smaller model.
- **CLI**: `--n_layers {3,4,5,6}`. Group `r5-h5-nlayers`.
- **Risk**: nl=3 might lose enough representational depth to hurt the cruise high-mesh-density samples. Mitigation: include nl=4 as a middle anchor.
- **Why high-EV given prior signal**: validated MERGED winner, complements H4 — both reduce capacity in different axes. Independent attribution requires a solo run.

---

## H6 — `swiglu-feedforward`

- **Family**: architecture
- **Change**: replace the GELU MLP inside each Transolver block with SwiGLU (gated linear unit with SiLU activation), keeping `mlp_ratio=2` so parameter count is comparable.
- **Predicted delta on `val_avg/mae_surf_p`**: −3 to −7%.
- **Reasoning**: parallel track PR #20 (fern) MERGED with SwiGLU wins. SwiGLU has consistently outperformed GELU FFN in the LLaMA-style literature; the gating provides multiplicative expressivity at small parameter cost.
- **CLI snippet**: in `MLP` class or in `TransolverBlock`, replace the FFN with `SwiGLU(d) = (W1 x ⊙ silu(W2 x)) W3` with `W1, W2: d → d*ratio`, `W3: d*ratio → d`. Add `--mlp_type {gelu,swiglu}`.
- **Risk**: extra params (~33% inside FFN) may push VRAM up and slow training. Mitigation: drop `mlp_ratio` from 2 → 4/3 to match parameter budget exactly if throughput drops.
- **Why high-EV given prior signal**: validated MERGED winner; clean single-variable ablation.

---

## H7 — `amp-bf16-throughput`

- **Family**: optimizer / systems
- **Change**: enable `torch.amp.autocast(dtype=bfloat16)` around the forward pass. No grad accumulation — just AMP.
- **Predicted delta on `val_avg/mae_surf_p`**: −2 to −6% indirectly, by enabling more epochs to complete within the 30-min cap.
- **Reasoning**: parallel track PR #12 MERGED and meaningfully improved metrics by trading wall-clock for more epochs. With 1499 train samples and 4×100 val samples to run every epoch, more wall-budget per run is one of the highest-leverage knobs available.
- **CLI**: `--amp` flag (default off). Group `r5-h7-amp`.
- **Risk**: bf16 numerics may degrade the slice softmax temperature gradient. Mitigation: cast slice softmax and final loss to fp32 explicitly; keep optimizer states in fp32.
- **Why high-EV given prior signal**: solid throughput win on the parallel branch. Critical to verify early on willow-r5 because future capacity-increase experiments depend on speed headroom.

---

## H8 — `surf-weight-sweep`

- **Family**: loss
- **Change**: sweep `surf_weight ∈ {2, 5, 10, 20, 40}` (baseline=10) with MSE loss. One variable.
- **Predicted delta on `val_avg/mae_surf_p`**: −3 to −8% at the optimum (likely surf_weight ∈ [20, 40]).
- **Reasoning**: parallel track PR #11 (frieren) MERGED a fine surf_weight sweep on L1 loss; the equivalent under MSE has not been re-tested on willow-r5. Surface nodes are ~1–2% of total nodes per sample, so even at sw=10 the volume term dominates the gradient norm. Pushing sw higher should pull more capacity onto the primary metric.
- **CLI**: `--surf_weight {2,5,10,20,40}`. Group `r5-h8-surfweight`.
- **Risk**: too-high surf_weight may collapse the volume prediction quality and indirectly hurt the surface (since the same trunk processes both). Mitigation: cover a wide range so the elbow is visible.
- **Why high-EV given prior signal**: known to matter; cheap to run; one variable. Establishes the surface/volume trade-off curve on the un-decorated baseline.

---

## H9 — `domain-aware-conditioning-tokens`

- **Family**: conditioning / data representation (NEW — not tried on parallel track)
- **Change**: add a 4-dim one-hot domain indicator (`single`, `raceCar_tandem`, `cruise_tandem`, `unknown`) computed from existing features (e.g., `gap == 0` ↔ single; `is_freestream` from raceCar/cruise can be derived from feature magnitudes), concatenated to x as 4 extra dims, and fed through the preprocess MLP.
- **Predicted delta on `val_avg/mae_surf_p`**: −5 to −12%.
- **Reasoning**: the three training domains (raceCar single, raceCar tandem, cruise tandem) have boundary-condition differences (slip-wall ground vs. freestream) and AoA / Re ranges that the model currently has to discover from interactions of features {18..23}. An explicit one-hot lifts this from inference-time pattern-matching to a direct input. The parallel track did NOT test this — its FiLM conditioning is on `log(Re)` only.
- **CLI snippet**: in `train.py` after `x_norm`, derive `domain = compute_domain(x)`. Append a one-hot embedding to the input. Add `--domain_token {none,onehot,learned_embed}` flag (3 options).
- **Risk**: domain leakage between train and test is fine (domain is a known input at inference), but a hand-coded heuristic may misclassify edge cases. Mitigation: use simple `gap==0 ∧ stagger==0 ↔ single` and the cruise/raceCar distinction from absolute AoA range or boundary feature.
- **Why high-EV given prior signal**: untested family, plausibly orthogonal to FiLM-on-Re, and the appendix benefits from at least one fresh idea.

---

## Reserved for round 2 (H2, H10–H12)

- **H2 `l1-vs-huber-vs-mse-headtohead`** — clean 3-bar appendix table after H1 lands the best Huber δ.
- **H10 `learnable-output-scale-per-channel`** — 6-param learnable α/β per-channel rescale on the output head.
- **H11 `dropout-attention-only`** — attention-only dropout sweep, distinct from parallel-track dropout-everywhere.
- **H12 `gradient-clip-only`** — solo grad-clip ablation, distinct from parallel-track EMA+clip combo.

## Hypothesis assignment priority (top-8 for round 1)

1. **H1 huber-loss-delta-sweep** — biggest expected delta, parallel-validated, foundation for compounds.
2. **H3 fourier-pe-film-re** — second-biggest expected delta, parallel-validated, gateway to spatial-encoding follow-ups.
3. **H4 slice-num-down-sweep** — easy variable, MERGED on parallel branch, complements H5.
4. **H5 n-layers-down-sweep** — easy variable, MERGED on parallel branch, complements H4.
5. **H7 amp-bf16-throughput** — unlocks more epochs for everything later; high systems-leverage.
6. **H6 swiglu-feedforward** — MERGED on parallel branch, clean architectural ablation.
7. **H9 domain-aware-conditioning-tokens** — fresh idea, untested family, orthogonal to FiLM.
8. **H8 surf-weight-sweep** — low-cost calibration, important for the loss-axis appendix table.

## Notes on compound design (for later rounds)

After round 1 lands, the compound chain that the parallel track converged toward is:
`Huber + Fourier-PE+FiLM + slice_num=16 + n_layers=3 + SwiGLU + AMP`. We should NOT assign a compound recipe in round 1 — single-variable purity is essential for the ICML appendix tables. Round 2 will apply the parallel track's known-good compound directly on top of round-1 winners and report each compound's marginal gain.
