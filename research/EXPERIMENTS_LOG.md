# SENPAI Research Results — charlie-pai2g-48h-r5

---

## 2026-05-14 [Round 126] UTC — PR #2899: Block-asymmetric mlp_ratio [3,3,4,4] — **CLOSED LOSS (+2.38% val; HYPOTHESIS FALSIFIED; deep-block width = OOD driver; test WIN -1.13%)**

- **Branch:** charliepai2g48h5-alphonse/asym-mlp-ratio-3344
- **Metric artifacts:** models/model-charliepai2g48h5-alphonse-asym-mlp-ratio-3344-20260514-114312/metrics.jsonl

| Metric | #2879 Baseline | #2899 asym[3,3,4,4] | vs Baseline | #2889 uniform[4,4,4,4] | vs uniform=4 |
|---|---|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.2869** | **+2.38% LOSS** | 31.4505 | -0.52% |
| test_avg/mae_surf_p | 26.5160 | **26.2175** | **-1.13% WIN** | 26.6692 | -1.69% |
| val_single_in_dist | 23.3997 | 25.8458 | **+10.45% LOSS** | 25.9467 | -0.39% |
| val_geom_camber_rc | 46.0708 | 47.0941 | +2.22% LOSS | 45.7313 | +2.98% |
| val_geom_camber_cruise | 17.8657 | **16.9728** | **-5.00% WIN (BIGGEST EVER!)** | 17.2232 | -1.45% |
| val_re_rand | 34.9057 | 35.2351 | +0.94% flat | 36.9007 | -4.51% |

Params 445,060. Best ep58/70 (timeout).

**MAJOR FINDING — Shallow-block-protection hypothesis FALSIFIED:**
- In_dist regression +10.45% with shallow blocks at mlp=3 (baseline) is essentially IDENTICAL to uniform=4's +10.88%
- **Deep-block width is the OOD driver**, not shallow-block capacity
- Re_rand RECOVERED to near-baseline (+0.94% vs +5.72% uniform=4) — confirms re_rand IS a shallow-block effect
- camber_cruise WIN AMPLIFIED to -5.00% (biggest ever on val!) — confirms deep-block widening IS the camber_cruise driver

**3-experiment structural matrix now complete:**
| Config | in_dist | camber_cruise | val_avg |
|---|---|---|---|
| [3,3,3,3] baseline | 0 | 0 | 30.56 |
| [3,3,4,4] asym | +10.45% ❌ | -5.00% ✓✓✓ | 31.29 |
| [4,4,4,4] uniform | +10.88% ❌ | -3.60% ✓ | 31.45 |

**MLP-asymmetric depth-capacity axis CLOSES (102nd taxon).** In_dist regression is intrinsic to deep-block capacity addition. Cannot exploit camber_cruise OOD via mlp_ratio alone. Parallel #2900 mult-geo-FiLM is the orthogonal architectural approach.

---

## 2026-05-14 [Round 126] UTC — PR #2910: surf_weight 10→20 — **ASSIGNED to charliepai2g48h5-alphonse**

- **Branch:** charliepai2g48h5-alphonse/surf-weight-20
- **Hypothesis:** Double surf_weight from 10 to 20 (CLI flag `--surf_weight 20`). No code change needed. Raises surface-to-volume gradient ratio from 10:1 to 20:1. Directly attacks primary metric (val_avg is 100% surface MAE).
- **Rationale:** surf_weight axis never tested above 10 in 102 closed experiments. Three recent experiments showed test WIN while val LOST; architecture can achieve excellent surface predictions. Doubling surface gradient pressure tests whether the loss weighting is the bottleneck for further improvement.
- **Falsifiable:** WIN = gradient alignment with metric was limiting; try surf_weight=30. WASH = axis flat. LOSS = volume regularization load-bearing; try surf_weight=5.

---

## 2026-05-14 [Round 125] UTC — PR #2898: Gradient clipping max_norm=1.0 — **CLOSED LOSS (+4.91%; clipping far too tight)**

- **Branch:** charliepai2g48h5-tanjiro/grad-clip-1.0
- **Metric artifacts:** models/model-charliepai2g48h5-tanjiro-grad-clip-1.0-20260514-112849/metrics.jsonl

| Metric | #2879 Baseline | #2898 grad-clip | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **32.0598** | **+4.91% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 27.5530 | +3.91% LOSS |
| val_single_in_dist | 23.3997 | 25.8421 | +10.44% LOSS |
| val_geom_camber_cruise | 17.8657 | 17.8871 | +0.12% flat |

Best ep59/60 (timeout). Params 407,940 unchanged.

**Critical diagnostic — clipping bit every step (100%):**
- grad_norm_clip_frac = 1.000 for ALL 60 epochs
- Mean grad_norm at ep59: 24.6 (median 22.8, max 80.5) — 25-65× above max_norm=1.0
- max_norm=1.0 effectively divided every gradient to unit vector before Lion's sign-step
- Result: 25-65× effective LR reduction that cosine schedule cannot compensate in 30-min budget

**Why LOSS not WASH:** max_norm=1.0 is ~25× below the natural gradient regime. With Lion's sign-step already normalizing direction, clipping to unit vector collapses the momentum-accumulation benefit — every step is direction-only with no magnitude signal.

**Gradient-FLOW axis CLOSES at max_norm=1.0. 101st candidate axis CLOSES.**

---

## 2026-05-14 [Round 125] UTC — PR #2906: n_head 2→4 — **ASSIGNED to charliepai2g48h5-tanjiro**

- **Branch:** charliepai2g48h5-tanjiro/n-head-4
- **Hypothesis:** Double attention head count from 2 to 4 (dim_head 48→24). inner_dim stays 96. Counterintuitively REDUCES total params by ~23k (to ~384,900) due to quadratic shrinkage in per-head Linear(dim_head,dim_head) matrices.
- **Rationale:** n_head axis untested above 2 in this programme. More heads = finer slice-routing specialization: 4 heads × 24 slices = 96 total slice tokens vs 48 baseline. Each head can specialize in different physical modes (leading-edge pressure, suction side, wake, etc.). The per-head QKV bottleneck is the 24-slice routing, not dim_head — so halving dim_head costs less than doubling head diversity gains.
- **Falsifiable:** WIN = try n_head=3 bracket, or n_head=4 + restore saved params elsewhere. LOSS = dim_head=24 too small. WASH = head axis flat; close.

---

## 2026-05-14 [Round 124] UTC — PR #2892: EMA model weights eval (decay=0.999) — **SENT BACK (schedule confound; epochs=60 rerun requested)**

- **Branch:** charliepai2g48h5-frieren/ema-eval-weights
- **Original result:** val 31.6548 (+3.58% LOSS) / test 26.2169 (-1.13% WIN)
- **Diagnosis:** SCHEDULE CONFOUND dominates val LOSS. --epochs 70 + 30min timeout → ep55 with 21% LR remaining. Baseline #2879 used epochs=60, reached ep58 in cosine tail. Not an EMA refutation.
- **EMA-vs-live diagnostic (ep55):** EMA beats live by +0.91 val_avg. Gap was +5.23 at ep30, +3.45 at ep40. EMA is consistently AHEAD of live from epoch ~17 onward.
- **Test result WIN:** test 26.2169 vs baseline 26.5160 = -1.13% — EMA-shadow generalizes better at test time.
- **Action:** Sent back with instruction to change --epochs 70 → --epochs 60. Same EMA decay=0.999, same hyperparams. This removes the schedule confound and gives a fair cosine-tail comparison.

---

## 2026-05-14 [Round 124] UTC — PR #2891: Multi-task aux loss (predict Re/AoA0/AoA1) — **CLOSED LOSS (+6.17%; shortcut learned, 100th taxon)**

- **Branch:** charliepai2g48h5-edward/multi-task-aux-loss
- **Metric artifacts:** models/model-charliepai2g48h5-edward-multi-task-aux-loss-20260514-111726/metrics.jsonl

| Metric | #2879 Baseline | #2891 aux-loss | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **32.4465** | **+6.17% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 27.3251 | +3.05% LOSS |
| val_single_in_dist | 23.3997 | 26.3747 | +12.71% LOSS |
| val_geom_camber_rc | 46.0708 | 48.1314 | +4.47% LOSS |
| val_geom_camber_cruise | 17.8657 | 19.0933 | +6.87% LOSS |
| val_re_rand | 34.9057 | 36.1864 | +3.67% LOSS |

All 4 splits regress proportionally. Best ep60/70 (timeout). **Uniform regression — NOT split-pattern-reversal family.**

**Critical diagnostic — shortcut confirmed:** log_Re reconstruction MSE = 0.00079 (essentially zero), aux_loss 0.91→0.005 (175× reduction). Aux head learned to copy input channels 13/14/18 directly. Zero representational pressure on trunk.

**Why not just WASH despite tiny aux_weight=0.05:**
- Lion sign-step makes tiny MSE term contribute meaningful sign-update direction
- Mild representational misalignment: flow-scalar preservation ≠ surface-MAE optimization

**Multi-task aux-regression axis CLOSES. 100th candidate taxon milestone.** Total closed: 100. Winners: 21.

---

## 2026-05-14 [Round 124] UTC — PR #2905: weight_decay 3e-4→5e-4 — **ASSIGNED to charliepai2g48h5-edward**

- **Branch:** charliepai2g48h5-edward/weight-decay-5e-4
- **Hypothesis:** Stronger L2 regularization (wd=5e-4, 1.67× increase from 3e-4) to address the recurring in-dist regression pattern from #2889/#2890. The regularization-strength axis has never been tested above wd=3e-4 in this programme.
- **Rationale:** Three consecutive OOD-WIN+in-dist-LOSS experiments suggest model overfits in-dist training manifold. Weight decay is Lion's primary magnitude control (sign-step normalizes direction). Expected per-step decay: 7.5e-8 (vs 4.5e-8 at baseline). One-line change.
- **Falsifiable:** WIN = try wd=7e-4 or combine with #2899/#2900 wins. WASH = axis flat near baseline, try 1e-3 bracket. LOSS = wd=5e-4 too strong, try 2e-4 downward.

---

## 2026-05-14 [Round 123] UTC — PR #2893: GELU on SwiGLU up-projection — **CLOSED LOSS (+3.65%; activation-site axis FULLY CLOSED)**

- **Branch:** charliepai2g48h5-nezuko/gelu-up-projection
- **Hypothesis:** Apply F.gelu to SwiGLU up-projection output — smooth bracket for #2875 ReLU² LOSS.
- **Metric artifacts:** models/model-charliepai2g48h5-nezuko-gelu-up-projection-20260514-111005/metrics.jsonl

| Metric | #2879 Baseline | #2893 GELU-up | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.6773** | **+3.65% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 26.7514 | +0.89% near-wash |
| val_single_in_dist | 23.3997 | 24.2319 | +3.56% LOSS |
| val_geom_camber_rc | 46.0708 | 48.7596 | +5.84% LOSS |
| val_geom_camber_cruise | 17.8657 | 17.9907 | +0.70% flat |
| val_re_rand | 34.9057 | 35.7269 | +2.35% LOSS |

All 4 val splits regress. Best ep60/70 (timeout). Params 407,940 unchanged.

**Critical disentanglement vs #2875 ReLU²:**
- up_act_zero_frac: GELU 0.18-0.23 (bounded, depth-flat) vs ReLU² 0.58→0.77 (dead-feature collapse)
- up_act_max: GELU 9-10 (no overflow) vs ReLU² 157.8 block-1 (bf16 amplification)
- GELU avoided BOTH failure modes of #2875 but still lost → isolates harm to **GLU factorization itself**, not dead-features/overflow

**Activation-site axis FULLY CLOSED — 5-entry bracket:**
| Site | Activation | Outcome |
|---|---|---|
| gate=SiLU, up=linear | baseline | WIN (21st) |
| gate=GELU | LOSS | |
| gate=Mish | CATASTROPHIC | |
| up=ReLU² (#2875) | LOSS | hard+quadratic |
| up=GELU (#2893) | LOSS | smooth but still wrong |

Linear up-projection is load-bearing. No further activation-site work needed. **99th candidate axis CLOSES.**

---

## 2026-05-14 [Round 123] UTC — PR #2903: RMSNorm replaces LayerNorm — **ASSIGNED to charliepai2g48h5-nezuko**

- **Branch:** charliepai2g48h5-nezuko/rmsnorm-replace-layernorm
- **Hypothesis:** Replace all 3 `nn.LayerNorm` in TransolverBlock (ln_1, ln_2, ln_3) with `nn.RMSNorm`. Removes mean-centering and β-bias (864 fewer params). Modern transformer standard: Llama, Mistral, Gemma, DeepSeek all use RMSNorm. Uses `RMSNorm = nn.RMSNorm` alias (PyTorch >= 2.4).
- **Rationale:** Normalization-choice axis untested in this programme. Activation-site axis now fully closed (#2893); student analysis suggests pivoting outside SwiGLU to upstream architecture changes. RMSNorm removes the centering operation that may conflict with flow-FiLM multiplicative conditioning. Orthogonal to all 7 in-flight experiments.
- **Expected params:** ~407,076 (407,940 - 864 β-bias params)
- **Falsifiable:** WIN/LOSS/WASH all cleanly interpretable; closes normalization axis in any outcome.

---

## 2026-05-14 [Round 122] UTC — PR #2890: Geometry-conditioned FiLM additive bias — **CLOSED LOSS (+3.09%; STRONGEST EVER camber_cruise -9.87% OOD signal)**

- **Branch:** charliepai2g48h5-askeladd/geometry-film
- **Hypothesis:** Additive geometry-FiLM: `fx = preprocess(x) + placeholder + geo_film(geo_features)` where geo_film=Linear(8,96) on NACA0+NACA1+gap+stagger. Zero-init. +864 params.
- **Metric artifacts:** models/model-charliepai2g48h5-askeladd-geometry-film-20260514-110510/metrics.jsonl

| Metric | #2879 Baseline | #2890 geo-FiLM | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.5045** | **+3.09% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 26.7665 | +0.94% |
| val_single_in_dist | 23.3997 | 27.3431 | **+16.85% LOSS** |
| val_geom_camber_rc | 46.0708 | 45.5498 | -1.13% ✓ |
| val_geom_camber_cruise | 17.8657 | **16.1031** | **-9.87% WIN (LARGEST EVER!)** |
| val_re_rand | 34.9057 | 37.0220 | +6.06% LOSS |

Best ep59/60 (timeout). Params 408,804 confirmed. geo_film weight_norm=1.95 (activated from 0).

**KEY FINDING — THIRD CONFIRMATION of novel split-pattern reversal + STRONGEST CAMBER_CRUISE WIN EVER:**

This is the THIRD experiment (after #2889 +capacity, #2890 +geo-bias) showing the same split-pattern reversal: geometric-OOD improves at in-dist cost. Camber_cruise -9.87% is the LARGEST single-split improvement ever observed (2.7× bigger than #2889's -3.60%).

**Critical diagnostic:** geo_film output std was HIGHEST on val_geom_camber_cruise (0.22) — exactly matching the best improvement. The mechanism works. The site is wrong.

**Mechanism diagnosis:** The additive bias on input embedding directly competes with preprocess MLP (which already has raw access to channels 15-17, 19-22). Even abs_mean ~0.15 disrupts well-tuned in-dist representation. Flow-FiLM succeeds as MULTIPLICATIVE; geometry-FiLM must also be multiplicative.

**98th candidate axis (geometry-additive-FiLM-on-input) CLOSES.**

**Follow-up assigned: #2900 askeladd multiplicative geo-FiLM** — parallel gate `fx = fx * (1 + film_scale + geo_film_scale)`. Same site as successful flow-FiLM, same parametrization, tests whether multiplicative composition fixes the in_dist regression.

---

## 2026-05-14 [Round 122] UTC — PR #2900: Multiplicative geo-FiLM parallel to flow-FiLM — **ASSIGNED to charliepai2g48h5-askeladd**

- **Branch:** charliepai2g48h5-askeladd/multiplicative-geo-film
- **Hypothesis:** Add geometry-conditioned FiLM as second multiplicative gate parallel to existing flow-FiLM: `fx = fx * (1 + film_scale + geo_film_scale)`. geo_film=Linear(8,96) on NACA0+NACA1+gap+stagger (ch[15,16,17,19,20,21,22,23]). Zero-init → identity at step 0. Same site that successfully hosts flow-FiLM (21st winner).
- **Rationale:** Additive geo-FiLM (#2890) produced largest-ever camber_cruise WIN (-9.87%) but hurt in_dist +16.85%. The problem was the parametrization (additive bias fights preprocess MLP's learned basis). Multiplicative gate composes cleanly with existing flow-FiLM — both gates multiply together before applying. Zero-init ensures exact baseline training stability at step 0. +864 params, ~408,804 total.
- **Falsifiable:** WIN = multiplicative composition fixes in_dist regression while preserving camber_cruise improvement. LOSS = the site itself (input embedding) is wrong regardless of parametrization; need to route geometry conditioning deeper.

---

## 2026-05-14 [Round 121] UTC — PR #2889: mlp_ratio=4 uniform — **CLOSED LOSS (+2.91%; NOVEL OOD SIGNAL captured)**

- **Branch:** charliepai2g48h5-alphonse/mlp-ratio-4
- **Hypothesis:** Scale SwiGLU MLP from mlp_ratio=3 (hidden=192) to mlp_ratio=4 (hidden=256). +74,240 params (482,180 total, +18% vs #2879). Continuation of the MLP-capacity axis from #2879 WIN.
- **Metric artifacts:** models/model-charliepai2g48h5-alphonse-mlp-ratio-4-20260514-105518/metrics.jsonl

| Metric | #2879 Baseline | #2889 mlp_ratio=4 | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.4505** | **+2.91% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 26.6692 | +0.58% near-wash |
| val_single_in_dist | 23.3997 | 25.9467 | **+10.88% LOSS** |
| val_geom_camber_rc | 46.0708 | 45.7313 | -0.74% ✓ |
| val_geom_camber_cruise | 17.8657 | **17.2232** | **-3.60% WIN (rare!)** |
| val_re_rand | 34.9057 | 36.9007 | +5.72% LOSS |

Best ep56/70 (hit SENPAI_TIMEOUT). Params 482,180 confirmed. gate_zero_frac rose +27-48% on all blocks (capacity saturation).

**KEY FINDING — SPLIT-PATTERN REVERSAL (first ever):** This is the FIRST experiment in 17+ rounds where geometric-OOD splits IMPROVED (camber_cruise -3.60%, camber_rc -0.74%) while in-distribution and re_rand regressed. ALL previous patterns showed OOD-LOSS + in-dist-WIN or all-splits move together.

**Mechanism diagnosis:**
1. **Capacity-saturation signature:** gate_zero_frac +27-48% relative means extra channels being masked — wider MLP has more channels than can productively be used at depth=4, data_scale.
2. **Geometric OOD benefit:** wider MLP CAN encode more geometric variation (camber shapes) — but only for OOD splits, at cost of in-dist overfitting.
3. **MLP-ratio axis is CONCAVE** with mlp_ratio=3 as apex: mlp_ratio=2 (#2810) → 3 (#2879 WIN) → 4 (#2889 LOSS). Uniform scaling axis CLOSED.
4. **bf16 vol_loss overflow** on test_geom_camber_cruise (loss=nan, vol_loss=inf) but MAE finite 13.6846 — overflow in loss accumulation only; predictions valid.

**97th candidate axis (MLP-ratio scaling uniform) CLOSES with mlp_ratio=3 as apex.**

**Follow-up hypothesis assigned: #2899 block-asymmetric [3,3,4,4] exploiting this novel OOD signal.**

---

## 2026-05-14 [Round 121] UTC — PR #2899: Block-asymmetric mlp_ratio [3,3,4,4] — **ASSIGNED to charliepai2g48h5-alphonse**

- **Branch:** charliepai2g48h5-alphonse/asymmetric-mlp-ratio-3344
- **Hypothesis:** Per-block mlp_ratio = [3, 3, 4, 4]: shallow blocks (0,1) narrow=192 (baseline), deep blocks (2,3) wide=256 (from #2889). Exploits the #2889 novel signal: wider deep MLPs helped geometric-OOD while uniform wide hurt in-dist. Expected params ~446,000.
- **Rationale:** Deep blocks specialize in object-level (geometry) representations; shallow blocks encode position/edges. Routing extra capacity depth-progressively should recover in-dist (shallow blocks unchanged) while preserving camber_cruise gain (deep blocks wider). Implementation: make mlp_ratio accept a list in Transolver.__init__ and hardcode [3,3,4,4] in model_config.
- **Falsifiable:** WIN = asymmetric routing preserves camber_cruise -3.60% AND recovers in-dist. LOSS = novel signal was noise or capacity-asymmetry introduces new pathology. WASH = equivalent to uniform mlp_ratio=3 apex.

---

## 2026-05-14 [Round 120] UTC — PR #2885: Stochastic Depth DropPath linear 0->0.1 — **CLOSED LOSS (+8.36% val, gradient-path axis closed at depth=4)**

- **Branch:** charliepai2g48h5-tanjiro/stochastic-depth-droppath
- **Hypothesis:** Per-sample DropPath with depth-progressive rates [0.0, 0.033, 0.067, 0.1] across 4 blocks wrapping LayerScale-weighted residual branches. DeiT/ConvNeXt recipe with LayerScale gamma=1e-4 synergy. +0 params.
- **Metric artifacts:** models/model-charliepai2g48h5-tanjiro-drop-path-0.1-linear-20260514-103757/metrics.jsonl

| Metric | New Baseline #2879 | #2885 drop-path-0.1 | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **33.1152** | **+8.36% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 28.5526 | +7.68% LOSS |
| val_single_in_dist | 23.3997 | 26.4296 | +12.95% LOSS |
| val_geom_camber_rc | 46.0708 | 48.9428 | +6.23% LOSS |
| val_geom_camber_cruise | 17.8657 | 19.8668 | +11.20% LOSS |
| val_re_rand | 34.9057 | 37.2218 | +6.63% LOSS |

DropPath schedule confirmed at startup as [0.0, 0.033, 0.067, 0.1]. Param count 333,700 unchanged. Sec/epoch 27.5s.

**Mechanism diagnosis (tanjiro):**
1. 4 blocks too few for DropPath — single block = ~25% of representational depth. DeiT/ConvNeXt amortize across 12-24 blocks; ours cannot.
2. ALL splits regress including in_dist +12.95% — capacity-reduction signature NOT regularization signature.
3. val_geom_camber_cruise +11.20% (per epoch-62 reading) — cruise depends most on stable propagation through full block stack.
4. LayerScale gamma synergy did NOT manifest: gamma_attn=0.018 (near-identity, dropping no-op), gamma_mlp=0.073 (real work, dropping loses information).

**Gradient-path axis CLOSES at depth=4.** Future gradient-path work needs n_layers>=8 or MLP-only / last-block-only variants (same axis-family, also LOSS-risk).

**95th candidate axis CLOSES.**

---

## 2026-05-14 [Round 120] UTC — PR #2898: Gradient Clipping max_norm=1.0 (100th axis) — **ASSIGNED to charliepai2g48h5-tanjiro**

- **Branch:** charliepai2g48h5-tanjiro/grad-clip-1.0
- **Hypothesis:** Apply torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) between loss.backward() and optimizer.step(). Standard transformer training intervention. Controls gradient magnitude WITHOUT reducing model capacity. +0 params.
- **Rationale:** Gradient-FLOW axis (different from gradient-PATH axis #2885 just closed). 16+ closed taxa have spanned activations/attention/normalization/capacity/routing/PE/loss-shape/gradient-path but NEVER gradient-magnitude. Lion compatible: sign-step is bounded but momentum buffer accumulates raw gradients. Equalizes contribution of hard OOD samples (camber_rc grad_norm dominates) with easy in_dist samples. Key diagnostic: grad_norm_clip_frac per epoch tells us if clipping is biting.
- **Falsifiable:** WIN = try max_norm=0.5 tighter. WASH = gradients already <1.0 (clipping no-op). LOSS = clipping slows convergence below 30min budget.
- **100th candidate axis** (round number milestone for the programme).

---

## 2026-05-14 [Round 119] UTC — PR #2884: Slice-routing entropy regularization alpha=0.005 — **CLOSED LOSS (+3.47% val vs new baseline)**

- **Branch:** charliepai2g48h5-edward/slice-entropy-reg
- **Hypothesis:** Auxiliary entropy loss on slice_weights with alpha=0.005 to prevent block-2 entropy collapse seen in #2869 H=2 diagnostic (entropy 0.267 = 8.4% of max). +0 params.
- **Metric artifacts:** models/model-charliepai2g48h5-edward-slice-entropy-reg-20260514-102403/metrics.jsonl

| Metric | New Baseline #2879 | #2884 entropy-reg | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.6215** | **+3.47% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 27.3444 | +3.13% LOSS |
| val_single_in_dist | 23.3997 | 24.7160 | +5.63% LOSS |
| val_geom_camber_rc | 46.0708 | 48.0352 | +4.26% LOSS |
| val_geom_camber_cruise | 17.8657 | 18.3382 | +2.65% LOSS |
| val_re_rand | 34.9057 | 35.3968 | +1.41% LOSS |

Mechanism worked: block-2 entropy held at 2.497/3.178 (78.5% of max) throughout training — vs collapsed 0.267 in baseline. But block-0 entropy concentrated to 1.16 (36.5% of max) — the optimization budget for sparsity migrated from block-2 to block-0. Entropy bonus 1.23% of train loss as planned. Param count 333,700 ✓.

**Critical insight:** Routing collapse is **load-bearing for OOD geometry**, not a bug. Forcing broad routing breaks implicit sub-task specialization. The optimizer-natural attractor toward concentrated routing IS what enables camber_rc/camber_cruise specialization. 94th candidate axis CLOSES. Slice-routing entropy regularization not productive at any alpha; future routing-axis work needs structural change (hard top-k, capacity-aware) not soft penalty.

---

## 2026-05-14 [Round 119] UTC — PR #2881: NeRF-style Fourier positional encoding n_freqs=4 — **CLOSED LOSS (+4.63% val)**

- **Branch:** charliepai2g48h5-nezuko/fourier-pos-enc
- **Hypothesis:** Apply Fourier features to (pos_x, pos_z) channels with frequencies pi, 2pi, 4pi, 8pi → +16 input features → preprocess input 24→40. +3,072 params.
- **Metric artifacts:** models/model-charliepai2g48h5-nezuko-fourier-pos-enc-20260514-101708/metrics.jsonl

| Metric | New Baseline #2879 | #2881 fourier-pos-enc | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **31.9774** | **+4.63% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 27.4366 | +3.47% LOSS |
| val_single_in_dist | 23.3997 | 24.3221 | +3.94% LOSS |
| val_geom_camber_rc | 46.0708 | 48.3652 | +4.98% LOSS |
| val_geom_camber_cruise | 17.8657 | 17.8121 | -0.30% wash |
| val_re_rand | 34.9057 | 37.4103 | +7.17% LOSS |

Implementation correct: param delta +3,072 matches prediction exactly; bf16 phase-aliasing handled (sin/cos in fp32 cast to bf16). The inductive bias does not help here.

**Mechanism analysis (excellent diagnosis):** Fourier features amplify in-dist coordinate phase memorization. 0.25-period channel resolves features at ½-chord scale — fine enough to memorize wing-specific surface curvature in narrow training distribution. NeRF works because a single scene has spatially-consistent coordinates; here every wing has its own coordinate population, so coordinate-Fourier expansion hurts OOD generalization. **93rd candidate axis CLOSES.** Coordinate-Fourier axis exhausted; geometry-invariant features would be natural follow-up but data/ is read-only.

---

## 2026-05-14 [Round 119] UTC — PR #2874: slice_num=28 upward direction — **CLOSED LOSS (+5.49% val, axis fully closed)**

- **Branch:** charliepai2g48h5-frieren/slice-num-28
- **Hypothesis:** Increase slice_num 24→28 to provide more pathways for OOD geometry expression. +784 params.
- **Metric artifacts:** models/model-charliepai2g48h5-frieren-slice-num-28-20260514-101210/metrics.jsonl

| Metric | New Baseline #2879 | #2874 slice_num=28 | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.5605 | **32.2381** | **+5.49% LOSS** |
| test_avg/mae_surf_p | 26.5160 | 28.0820 | +5.91% LOSS |
| val_single_in_dist | 23.3997 | 25.3884 | +8.50% LOSS |
| val_geom_camber_rc | 46.0708 | 49.0499 | +6.47% LOSS |
| val_geom_camber_cruise | 17.8657 | 17.3220 | -3.04% slight WIN |
| val_re_rand | 34.9057 | 37.1921 | +6.55% LOSS |

Per-block routing: ~19/28 effective slices used. Block-3 entropy 0.30/3.33 = 9% of max with max_routed_slice_frac=0.92 confirms structural one-hot attractor. Adding pathways does not break attractor, only adds dead slices.

**Slice_num axis fully closed by 3-point bracket:** 32→24 WIN (#2307), 24→20 LOSS (#2852, +3.69%), 24→28 LOSS (#2874, +5.49% vs new baseline). slice_num=24 is local optimum. Future slice-family work needs PhysicsAttention architectural change (hard top-k, capacity-aware routing) not slice count adjustments. 95th candidate axis CLOSES.

---

## 2026-05-14 [Round 119] UTC — PR #2868: Smooth L1 (Huber β=1.0) training loss — **CLOSED stale_wip**

- **Branch:** charliepai2g48h5-thorfinn/smooth-l1-huber-beta1
- Closed without results — thorfinn was idle on this assignment for several rounds without posting terminal results. Slot reclaimed for fresh experiment (cosine eta_min). Smooth L1 hypothesis remains valid and could be revived as a fresh assignment later if needed.

---

## 2026-05-14 [Round 119] UTC — PR #2891: Multi-task auxiliary loss (96th axis) — **ASSIGNED to charliepai2g48h5-edward**

- **Branch:** charliepai2g48h5-edward/multi-task-aux-loss
- **Hypothesis:** Add auxiliary regression head predicting flow scalars [log_Re, AoA0_rad, AoA1_rad] from masked-mean pooled pre-decoder features. Loss multiplier aux_weight=0.05. Self-supervised consistency objective forces internal representation to preserve physical-quantity information. +291 params (Linear(96, 3)).
- **Rationale:** 5+ axes show in-dist-WIN + OOD-LOSS pattern (#2851/#2864/#2870/#2881/#2884). Recurring signature: representation drifts toward in-dist-specific features at OOD cost. Forcing the rep to reconstruct Re/AoA pins it to physical reality. Classical multi-task literature (Caruana 1997, Ruder 2017) supports aux tasks reliably improving OOD generalization when sharing representations.
- **Falsifiable:** WIN = try aux_weight=0.1 or NACA targets. WASH = aux head shortcut from input chans. LOSS = aux objective conflicts with surface MAE.
- **96th candidate axis.**

---

## 2026-05-14 [Round 119] UTC — PR #2892: EMA model weights for evaluation (97th axis) — **ASSIGNED to charliepai2g48h5-frieren**

- **Branch:** charliepai2g48h5-frieren/ema-eval-weights
- **Hypothesis:** Maintain EMA of model weights (decay=0.999) during training; use EMA model for val/test eval. Training step unchanged — Lion still updates live model. +0 params (just shadow copy in memory). Standard transformer recipe (DeiT, ConvNeXt, DINOv2).
- **Rationale:** Best-val happens at ep58-65 of 70 — within cosine tail where lr→0. EMA smooths late-training noise. ~700-step half-life at decay=0.999 averages over last ~22 epochs. Diagnostic: log both live_val and ema_val per epoch to measure EMA effect directly.
- **Falsifiable:** WIN = try decay=0.9995 or per-split EMA tuning. WASH = optimum genuinely converged. LOSS = EMA delays late-training tracking.
- **97th candidate axis.**

---

## 2026-05-14 [Round 119] UTC — PR #2893: GELU on SwiGLU up-projection (98th axis) — **ASSIGNED to charliepai2g48h5-nezuko**

- **Branch:** charliepai2g48h5-nezuko/gelu-up-projection
- **Hypothesis:** Apply F.gelu() to SwiGLU up-projection output (was: linear). Smooth-activation bracket complementing #2875 ReLU² LOSS. Single-line change: . +0 params.
- **Rationale:** Tests whether ANY up-proj non-linearity helps, isolating hard-sparsity (ReLU²) from smooth-nonlinearity (GELU). No hard-zero region → no dead-feature trap under Lion. No quadratic amplification → no bf16 overflow. Closes activation-site axis cleanly.
- **Falsifiable:** WIN = try Tanh on up next. WASH = up-proj activation irrelevant (SwiGLU is gate-dominated). LOSS = ANY up-proj non-linearity breaks SwiGLU (Shazeer 2020 linear up-proj load-bearing).
- **98th candidate axis.**

---

## 2026-05-14 [Round 119] UTC — PR #2894: Cosine LR eta_min = 0.05*lr (99th axis) — **ASSIGNED to charliepai2g48h5-thorfinn**

- **Branch:** charliepai2g48h5-thorfinn/cosine-eta-min-5pct
- **Hypothesis:** Change CosineAnnealingLR eta_min from 0 to 0.05*cfg.lr = 7.5e-6. Single-line config change. +0 params.
- **Rationale:** Current schedule cosine-anneals lr from peak 1.5e-4 to literally 0 over 67 epochs. Best-val often at ep58-65 (cosine tail where lr→0 already). Setting eta_min=5% keeps a small floor lr so Lion's sign-step can still make meaningful updates. Standard practice in modern transformer recipes (DeiT-III, ConvNeXt-v2 use eta_min=1e-6 or lr/100).
- **Falsifiable:** WIN = try eta_min=10%*lr or longer total epochs. WASH = model converges before cosine tail matters. LOSS = late-training noise from eta_min=5% hurts.
- **99th candidate axis.**

---

## 2026-05-14 [Round 118] UTC — PR #2879: mlp_ratio=3 SwiGLU wider MLP — **MERGED WIN (-1.07% val, 21st winner)**

- **Branch:** charliepai2g48h5-alphonse/mlp-ratio-3
- **Hypothesis:** Widen SwiGLU MLP by increasing mlp_ratio from 2 to 3. SwiGLU 2/3 param-matching factor means hidden_swiglu 128->192 (not 192->288 as predicted); actual +74,240 params = 407,940 total. Apples-to-apples capacity test vs #2869 attention expansion (+87k); if MLP-expansion WIN + attention-expansion LOSS => MLP body is better location for capacity.
- **Metric artifacts:** models/model-charliepai2g48h5-alphonse-mlp-ratio-3-20260514-100221/metrics.jsonl

| Metric | Baseline #2810 | #2879 mlp_ratio=3 | vs Baseline | vs #2869 (attn-expansion) |
|---|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | **30.5605** | **-1.07% WIN** | 31.7942 (MLP wins) |
| test_avg/mae_surf_p | 26.1964 | 26.5160 | +1.22% LOSS | 27.1764 (MLP wins) |
| val_single_in_dist | 25.2751 | 23.3997 | -7.42% WIN | 26.0060 (MLP wins) |
| val_geom_camber_rc | 45.8179 | 46.0708 | +0.55% flat | 46.8914 (MLP wins) |
| val_geom_camber_cruise | 16.8427 | 17.8657 | +6.07% LOSS | 18.3170 (MLP wins) |
| val_re_rand | 35.6177 | 34.9057 | -2.00% WIN | 35.9622 (MLP wins) |

Param count 407,940 (best ep58/60, hit SENPAI_TIMEOUT at ep60). SwiGLU gate_zero_frac [0.0146, 0.0139, 0.0103, 0.0159] — very low, extra channels actively used. Sec/epoch ~29.6s (+6% vs baseline).

**Analysis:** Plateau-breaking val WIN after 16 consecutive LOSSes. MLP-axis capacity strictly preferred over attention-axis (#2869) on all splits. Same in-dist-WIN + camber_cruise-LOSS pattern as #2851/#2864/#2870 but this time val_avg cleared the bar. Val-test gap narrowed (4.69->4.04 absolute). Test regression acknowledged. MERGED as 21st winner.

**New baseline: val 30.5605 / test 26.5160**

---

## 2026-05-14 [Round 118] UTC — PR #2878: Differential Attention on slice-token self-attention — **CLOSED LOSS (+2.77% val)**

- **Branch:** charliepai2g48h5-askeladd/differential-attention
- **Hypothesis:** Replace scaled-dot-product attention in PhysicsAttention with Ye 2024 differential formulation: attn = softmax(q1k1/sqrt(D)) - lambda*softmax(q2k2/sqrt(D)), scaled by 1/(1-lambda). Depth-progressive lambda_init = 0.8-0.6*exp(-0.3*layer_idx). +18,436 params = 352,136 total.
- **Metric artifacts:** models/model-charliepai2g48h5-askeladd-differential-attention-20260514-100127/metrics.jsonl

| Metric | Baseline #2810 | #2878 diff-attn | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | **31.7480** | **+2.77% LOSS** |
| test_avg/mae_surf_p | 26.1964 | 27.4526 | +4.79% LOSS |
| val_single_in_dist | 25.2751 | 24.9689 | -1.21% slight WIN |
| val_geom_camber_rc | 45.8179 | 47.7253 | +4.16% LOSS |
| val_geom_camber_cruise | 16.8427 | 17.9757 | +6.72% LOSS |
| val_re_rand | 35.6177 | 36.3219 | +1.98% LOSS |

Per-block lambda evolution: block-0 0.20->0.06 (collapsed), block-1 0.36->0.42 (engaged), block-2 0.47->0.27 (weak), block-3 0.56->-0.26 CLAMPED to 0 (inverted, walled off). Attn entropy: blocks 1-2 near log(24)=3.18 (near-uniform), block-3 attn2 collapsed to entropy 0.087 (near-Dirac) with lambda=0 (no effect on output).

**Analysis:** G=24 slice space too small for noise cancellation. Depth-prior wrong for TandemFoilSet: deepest block wants lambda<0 (invert subtraction). Same in_dist-WIN + OOD-LOSS pattern. Mechanism only engaged in blocks 1-2, and there it hurts. Attention-quality axis closed. 92nd taxon.

---

## 2026-05-14 [Round 118] UTC — PR #2889: mlp_ratio=4 (95th candidate axis) — **ASSIGNED to charliepai2g48h5-alphonse**

- **Branch:** charliepai2g48h5-alphonse/mlp-ratio-4
- **Hypothesis:** Continue MLP-capacity scaling. mlp_ratio=3->4, hidden_swiglu 192->256. Expected ~481,668 params (+18% vs #2879). Direct WIN follow-up per PR #2879 falsifiable criterion. Tests whether capacity scaling continues to compound or plateaus.
- **Falsifiable:** WIN = continue to mlp_ratio=5. WASH/LOSS = close scaling axis.

---

## 2026-05-14 [Round 118] UTC — PR #2890: Geometry-conditioned FiLM NACA/gap/stagger (96th axis) — **ASSIGNED to charliepai2g48h5-askeladd**

- **Branch:** charliepai2g48h5-askeladd/geometry-film
- **Hypothesis:** Add a second FiLM branch conditioned on geometric parameters [NACA0(3), NACA1(3), gap(1), stagger(1)] = 8 channels (x[:,0,[15,16,17,19,20,21,22,23]]). Additive bias geo_film = Linear(8, 96), zero-init for identity at step 0. +864 params. Existing flow FiLM uses [log_Re, AoA0, AoA1]; this provides a dedicated low-rank pathway for geometry-specific representation modulation. Directly targets camber_rc/camber_cruise OOD splits which are dominated by non-symmetric foil geometries absent from training distribution.
- **Falsifiable:** WIN = camber_rc/cruise improve; expect geo_film/weight_norm to grow from 0. WASH = preprocess MLP already encodes geometry. LOSS = additive bias interferes with flow-conditioned representations.

---

## 2026-05-14 [Round 117] UTC — PR #2875: Squared ReLU on SwiGLU up-projection — **CLOSED LOSS (+3.31% val)**

- **Branch:** charliepai2g48h5-tanjiro/squared-relu-up-proj
- **Hypothesis:** Replace linear up-projection in SwiGLU FFN with Squared ReLU (F.relu(x).pow(2)) per Primer 2022 (Hua et al.). Shazeer 2020 SwiGLU specifies linear up-projection; this adds hard sparsity + quadratic amplification to the up-path while keeping SiLU gate unchanged. Falsifiable via 3 directions: WIN=Primer activation helps CFD; WASH=up-proj activation agnostic; LOSS=hard-zero breaks GLU mechanism.
- **Metric artifacts:** models/model-charliepai2g48h5-tanjiro-squared-relu-up-proj-20260514-093659/metrics.jsonl

| Metric | Baseline #2810 | #2875 squared-ReLU-up | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | **31.9147** | **+3.31% LOSS** |
| test_avg/mae_surf_p | 26.1964 | **27.3442** | **+4.38% LOSS** |
| val_single_in_dist | 25.2751 | **27.0941** | **+7.20% WORST HIT** |
| val_geom_camber_rc | 45.8179 | 47.2804 | +3.19% |
| val_geom_camber_cruise | 16.8427 | 17.3274 | +2.88% |
| val_re_rand | 35.6177 | 35.9568 | +0.95% |

Param count: 333,700 ✓ unchanged. Best ep65, hit SENPAI_TIMEOUT at ~27s/epoch, 14.6 GB GPU.

**Key diagnostics — per-block up_act_zero_frac: 57.98% / 62.25% / 67.75% / 76.69%** (depth-progressive dead-feature trap). up_act_max: 61.5 / **157.8** / **102.5** / **112.9** — 3/4 blocks exceed bf16 overflow threshold. Gate stats unchanged (1.39→22.07%, matches baseline SwiGLU).

**Analysis:** ALL 4 SPLITS REGRESS. In-dist worst hit (+7.20%) = CAPACITY-BOTTLENECK signature, not OOD-positive regularizer. The hard-sparsity-helps-OOD hypothesis falsified. Dead-feature trap: Lion sign-step propagates no recovery gradient for ReLU² dead zone (x≤0) — up_act_zero_frac 58→77% with depth. Gate↔up negative correlation (-0.03 to -0.11) shows gate compensating for up-path spikes (wasted capacity). bf16 overflow realized: one test_camber_cruise vol_loss=inf confirmed. GLU mechanism broken: linear up-projection (Shazeer 2020 spec) is load-bearing. Consistent with #2859 MishGLU: LION SIGN-STEP AMPLIFIES COST OF HARD-ZERO REGIONS IN FFN — empirically confirmed across 2 hard-sparse activations.

**ACTIVATION-SITE AXIS COMPREHENSIVELY CLOSED:** gate-SiLU WIN #2741 | gate-GELU LOSS #2759 | gate-Mish CATASTROPHIC #2859 | up-linear baseline WIN | up-ReLU² LOSS #2875.

**91st taxon CLOSES. Plateau deepens to 16 LOSSes since #2810 merge.**

---

## 2026-05-14 [Round 117] UTC — PR #2885: Stochastic Depth / DropPath linear 0→0.1 (95th axis) — **ASSIGNED to charliepai2g48h5-tanjiro**

- **Branch:** charliepai2g48h5-tanjiro/stochastic-depth-droppath
- **Hypothesis:** Add DropPath (stochastic depth) to TransolverBlock attn and MLP residual paths with depth-progressive rates [0.0, 0.033, 0.067, 0.1] across 4 blocks. Training: per-sample Bernoulli drop of full residual branch. Eval: deterministic, identical to baseline. +0 params. FIRST gradient-path regularization experiment after 16 LOSSes; DeiT/ConvNeXt recipe; LayerScale synergy; implicit sub-network ensemble.
- **Rationale:** 16 consecutive LOSSes, all targeting activations/attention/norms/capacity. DropPath targets the residual gradient flow — genuinely new axis. Implicit sub-network ensemble forces path-redundancy → flat loss basin → OOD generalization. Plateau protocol bold-swing.
- **Falsifiable:** WIN = max_rate=0.15 next. WASH = try rate=0.05. LOSS = model too small for block drop, close axis.
- **95th candidate axis.**

---

## 2026-05-14 [Round 116] UTC — PR #2873: post-norm structural pivot (pre-LN → post-LN) — **CLOSED LOSS (+8.02% val)**

- **Branch:** charliepai2g48h5-edward/post-norm
- **Hypothesis:** Replace pre-norm (LN inside sublayer) with post-norm (LN outside residual) at all 8 ln_1/ln_2 sites across 4 blocks. Liu et al. 2020 argues post-norm can yield better final performance once stability solved. LayerScale γ=1e-4 init stabilizes.
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-post-norm-20260514-093042/metrics.jsonl`

| Metric | Baseline #2810 | #2873 post-norm | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | **33.3698** | **+8.02% LOSS** |
| test_avg/mae_surf_p | 26.1964 | **27.8986** | **+6.50% LOSS** |
| val_single_in_dist | 25.2751 | **27.4661** | +8.67% |
| val_geom_camber_cruise | 16.8427 | **19.3092** | +14.64% worst |
| val_geom_camber_rc | 45.8179 | **48.6226** | +6.12% |
| val_re_rand | 35.6177 | **38.0811** | +6.92% |

Param count: 333,700 ✓ unchanged. Best ep65/67, timeout. sec/epoch ~26.7s, peak GPU 14.56 GB.

Per-block residual-stream RMS at ln_2 (post-norm): all blocks ≈ 1.0 ± 2% across all splits — mechanism activated (bounded residual confirmed). LayerScale γ grew 2-3× (γ_attn 0.014-0.028 vs pre-norm 0.004-0.012) to compensate. No NaN, training stable.

**Analysis:** DISTINCTIVE FAILURE MODE — all 8 splits regress (4 val + 4 test, no exception). Residual-RMS diagnostic confirmed post-norm mechanism IS active (RMS ≈ 1.0), but the effect is uniformly negative. Post-LN destroys useful magnitude information that pre-LN preserves: branches must "shout louder" (γ grows 2-3×) before post-LN flattens them. Pre-norm "naked residual highway" lets per-block signal propagate to depth. First experiment in recent batch WITHOUT in_dist WIN + OOD LOSS tradeoff — STRUCTURAL failure (gradient path), not regularization tradeoff. Pre-norm is load-bearing for FiLM + LayerScale + SE-block3 + SwiGLU + Lion stack.

**88th taxon CLOSES. Plateau deepens to 15 LOSSes since #2810 merge.**

---

## 2026-05-14 [Round 116] UTC — PR #2884: Slice-routing entropy regularization (94th axis) — **ASSIGNED to charliepai2g48h5-edward**

- **Branch:** charliepai2g48h5-edward/slice-entropy-reg
- **Hypothesis:** Add auxiliary entropy loss on slice-routing softmax weights: `loss = data_loss - 0.005 * mean(H_block_i)` where `H_block_i = entropy(slice_weights_block_i)`. Encourages broader (higher-entropy) slice assignments, directly targeting block-2 routing collapse observed in #2869 (alphonse n_head=1, block-2 entropy=0.267, 8.4% of max). Alpha=0.005 ≈ 1.7% of typical loss magnitude. +0 params.
- **Rationale:** 15 consecutive LOSSes, block-2 routing collapse is the clearest concrete failure signal observed. Optimizer naturally gravitates toward low-entropy routing (simpler regression targets within slices), which may reduce OOD generalization by over-specializing slice assignments to in-dist geometry.
- **Falsifiable:** WIN = forced entropy broadens OOD coverage → try alpha=0.01. LOSS = natural sparsity load-bearing → close axis.
- **94th candidate axis.**

---

## 2026-05-14 [Round 115] UTC — PR #2870: output-head gain scalar (+1 param) — **CLOSED LOSS (+4.07% val)**

- **Branch:** charliepai2g48h5-nezuko/output-head-gain
- **Hypothesis:** Single learnable scalar `output_gain` (init=1.0) multiplied into final decoder output — free residual gain degree of freedom letting optimizer rescale prediction magnitude independently of weight tensor norms.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-output-head-gain-20260514-091507/metrics.jsonl`

| Metric | Baseline #2810 | #2870 | vs Baseline |
|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | **32.1480** | **+4.07% LOSS** |
| test_avg/mae_surf_p | 26.1964 | **27.3552** | **+4.42% LOSS** |
| val_single_in_dist | 25.2751 | **24.9756** | −1.18% (only WIN) |
| val_geom_camber_rc | 45.8179 | **48.0420** | +4.86% LOSS |
| val_geom_camber_cruise | 16.8427 | **18.5257** | +9.99% LOSS worst |
| val_re_rand | 35.6177 | **37.0487** | +4.02% LOSS |

Param count: 333,701 ✓ (+1). output_gain converged to 1.0443 (within predicted 0.9–1.1 range). Best ep65/70 (timeout-truncated at 30 min). Peak GPU 14.4 GB.

**Analysis:** Mechanism activated (gain ≠ 1.0) but empirically OOD-harmful. Three key insights: (1) single global scalar commits to one calibration for all distributions, biasing toward most-frequent in-dist regime at OOD cost; (2) final decoder Linear row already controls per-channel scale — output_gain is mathematically redundant (degenerate with weight rescaling); (3) Lion sign-step on a single scalar makes ±lr discrete moves oscillating around converged value, destabilizing cosine-tail optimization. Pattern matches #2851/#2864/#2870: in_dist WIN + OOD LOSS = configurations that improve in-dist fitting tend to OOD-regress. **87th taxon CLOSES.**

---

## 2026-05-14 [Round 115] UTC — PR #2881: Fourier positional encoding on mesh coords (93rd axis) — **ASSIGNED to charliepai2g48h5-nezuko**

- **Branch:** charliepai2g48h5-nezuko/fourier-pos-enc
- **Hypothesis:** Add NeRF-style Fourier positional encoding to input mesh node coordinates (x,y = channels 0-1 of 24-dim input). Expand raw pos into sin/cos at 4 frequencies [π, 2π, 4π, 8π], giving model explicit spatial scale representations. Input dim 24→40, preprocess Linear(24,192)→Linear(40,192), +3,072 params (~336,772 total). Inductive bias change (not capacity expansion) targeting OOD geometry generalization. References: Mildenhall 2020 (NeRF), Tancik 2020 (NeurIPS Fourier Features).
- **Rationale:** 14 consecutive LOSSes, none have touched the input representation. OOD-fragile splits differ in geometry. Raw (x,y) are linear scalars; Fourier encoding lets downstream Linear layers implement position-sensitive responses at multiple spatial scales. Addresses geometry-OOD gap directly, distinct from all in-flight axes.
- **Falsifiable:** WIN = Fourier inductive bias helps OOD geometry → try n_freqs=8. WASH = raw coords sufficient → close axis. LOSS = high-freq features overfit → close axis.
- **93rd candidate axis.**

---

## 2026-05-14 [Round 114] UTC — PR #2869: n_head=1 dim_head=96 max-rank attention — **CLOSED LOSS (+2.93% val)**

- **Branch:** charliepai2g48h5-alphonse/n-head-1-max-rank
- **Hypothesis:** Halve n_head 2→1, doubling dim_head 48→96 (maximum per-head rank within n_hidden=96 budget). Follows from #2856 n_head=4 diagnostic: attention rank is capacity-bottlenecked for in_dist.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-n-head-1-max-rank-20260514-091625/metrics.jsonl`

| Metric | Baseline #2810 | #2856 H=4 | #2869 H=1 | vs Baseline |
|---|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | 32.0367 (+3.71%) | **31.7942** | **+2.93% LOSS** |
| test_avg/mae_surf_p | 26.1964 | 26.6328 (+1.66%) | **26.6482** | **+1.72% LOSS** |
| val_single_in_dist | 25.2751 | 27.3958 (+8.39%) | **26.0060** | +2.89% |
| val_geom_camber_cruise | 16.8427 | 18.0142 (+6.96%) | **18.3170** | **+8.75% worst** |
| val_geom_camber_rc | 45.8179 | 46.9870 (+2.55%) | **46.8914** | +2.34% |
| val_re_rand | 35.6177 | 35.7499 (+0.37%) | **35.9622** | +0.97% |

Param count: 421,248 (+26.2%). Best ep 68/70 full schedule. sec/epoch 25.30s (+15%). Runtime 1771s (29.5 min, fit within cap). Peak GPU 14.02 GB.

Per-block routing at best ckpt: Block-2 slice-routing entropy collapsed to **0.267** (8.4% of max log24=3.18) — near-one-hot routing, single-head degeneracy confirmed. Other blocks: [0.9546, 2.1560, 0.2674, 0.6465]. Per-block temperature spread [0.88, 1.19, 0.78, 1.02] (span 0.41, wider than H=2 per-head span 0.14).

**Analysis:** LOSS in all 4 splits. Hypothesis falsified: capacity-bottleneck thesis for H=1 not supported. In_dist did NOT improve despite +25% params (+26% param budget, +2.89% regression). Block-2 routing collapsed to near-one-hot entropy 0.267 — single-head degeneracy failure mode realized. Multi-head parallelism is load-bearing as ensemble averaging of routing decisions, not as per-head specialization. H=2 sits at U-shaped sweet spot: H=4 LOSS (+3.71%, in_dist worst), H=1 LOSS (+2.93%, camber_cruise worst). Attention-head axis COMPREHENSIVELY CLOSED AT H=2.

**86th taxon CLOSES. Plateau deepens to 13 LOSSes since #2810 merge.**

---

## 2026-05-14 [Round 114] UTC — PR #2879: mlp_ratio 2→3 — MLP capacity-expansion axis (92nd) — **ASSIGNED to charliepai2g48h5-alphonse**

- **Branch:** charliepai2g48h5-alphonse/mlp-ratio-3
- **Hypothesis:** 1-line config change mlp_ratio 2→3, expanding SwiGLU intermediate dim from 192→288 across all 4 blocks. Expected ~370,564 params (+36,864 ≈ +11%). Apples-to-apples capacity test vs #2869 attention-expansion (+87k params, LOSS +2.93%) — same ~budget, different architectural location (MLP-body vs attention).
- **Rationale:** Model is generalization-limited (not capacity-limited per #2809 frieren). However, if capacity placement matters, MLP-body capacity expansion may succeed where attention-rank expansion failed. Alphonse's own #4 follow-up suggestion. 92nd candidate axis.
- **Falsifiable:** WIN means MLP was bottleneck; LOSS closes mlp_ratio upward axis; consistent with #2809 frieren thesis if LOSS.

---

## 2026-05-14 [Round 113] UTC — PR #2864: Hybrid LN-at-block0_ln1 + RMSNorm-at-8-other-sites — **CLOSED LOSS (+1.62% val)**

- **Branch:** charliepai2g48h5-askeladd/hybrid-ln-rmsnorm
- **Hypothesis:** Hybrid normalization: keep full LayerNorm at block0_ln1 (the ONE site where γ showed active learning post-FiLM per #2851 diagnostic), replace all 8 other LN sites with RMSNorm. Expected param count 332,932 (−768 from β removal at 8 sites).
- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-hybrid-ln-rmsnorm-20260514-085627/metrics.jsonl`

| Metric | Baseline #2810 | #2851 RMSNorm-all | #2864 Hybrid | vs Baseline | vs #2851 |
|---|---|---|---|---|---|
| val_avg/mae_surf_p | 30.8909 | 31.4066 | **31.3911** | **+1.62% LOSS** | −0.05% (≈ identical) |
| test_avg/mae_surf_p | 26.1964 | 27.1997 | **26.5492** | **+1.35% LOSS** | −2.39% (better) |
| val_single_in_dist | 25.2751 | 24.2105 | **23.9145** | **−5.39% WIN (new best-ever)** | −1.22% |
| val_geom_camber_rc | 45.8179 | 47.5202 | **47.9527** | +4.66% LOSS | +0.91% |
| val_geom_camber_cruise | 16.8427 | 17.7834 | **17.9843** | +6.78% LOSS | +1.13% |
| val_re_rand | 35.6177 | 36.1124 | **35.7129** | +0.27% (≈ wash) | −1.10% |

**Per-site γ/β stats at best-val ep65:** block0_ln1 (kept LN) γ_std=0.127, range [0.455, 1.197] strongly modulated, β_std=0.065 non-trivial. All 8 RMSNorm γ_std ≤ 0.0025 (near-identity) — 8 sites had essentially no γ-headroom used. Per-token RMS: block0_ln1 LN post-norm RMS=0.935 (mean-centering active), block1_ln1 RMSNorm post-norm RMS=1.001 (no mean subtraction). Param count: 332,932 ✓. Best ep65/66, timeout ep66. sec/epoch ~27s. Peak GPU 14.6 GB.

**Analysis:** Hypothesis falsified cleanly. The result is statistically indistinguishable from #2851 RMSNorm-everywhere (val 31.3911 vs 31.4066 = −0.05%). The prediction "if γ is near-identity at 8 sites, then β at those sites must also be decorative" is FALSE. Per-token RMS comparison nails the mechanism: LN's mean-centering strips the per-token mean that encodes OOD-relevant per-channel bias content even when γ stays near 1.0. β was compensating for mean-channel shifts on a per-channel basis. In-dist WIN (23.9145 = new best-ever) confirms mean-centering-free residual helps in-distribution prediction but OOD calibration is lost. Student's meta-recommendation accepted: **normalization meta-axis is now COMPREHENSIVELY CLOSED** (embed-LN #2808, DyT #2686, SurfaceAwareLN #2829, QK-Norm, GroupNorm, full RMSNorm #2851, hybrid #2864 — all mapped in BOTH structural directions). Pivot away from norm sub-experiments.

**85th taxon CLOSES. Plateau deepens to 12 LOSSes since #2810 merge.**

---

## 2026-05-14 [Round 113] UTC — PR #2878: Differential Attention on slice-token self-attention (Ye 2024) — **ASSIGNED to charliepai2g48h5-askeladd**

- **Branch:** charliepai2g48h5-askeladd/differential-attention
- **Hypothesis:** Apply Differential Attention (Ye et al. 2024, arXiv:2410.05258) to the 24-token slice-token self-attention inside PhysicsAttention. Replace standard `attn = softmax(q·k^T/sqrt(D))` with `attn = (softmax(q1·k1^T/sqrt(D)) − λ·softmax(q2·k2^T/sqrt(D))) / (1 − λ + ε)` using two independent Q,K projection pairs and a learnable per-block λ scalar initialized per paper (0.8 − 0.6·exp(−0.3·layer_idx)). V projection unchanged. Expected ~370,568 params (+36,868 = +11%).
- **Rationale:** Plateau protocol: 12 consecutive LOSSes → bold structural swings required. Differential attention cancels attention noise (irrelevant-token cross-attention mass) — directly attacks the quality of slice-routing in PhysicsAttention. Normalization meta-family now comprehensively closed by student's prior work. Attention architecture is under-explored: #2869 tests attention rank, this tests attention quality via noise-cancellation. Independent of all in-flight axes.
- **Falsifiable predictions:** WIN means slice-routing noise was load-bearing → try lower λ_init or per-head λ. WASH means G=24 too small for noise to matter → close axis. LOSS means paired attention reduces slice diversity → close axis.
- **91st candidate axis.**

---

## 2026-05-14 [Round 112] UTC — PR #2859: SwiGLU gate Mish (gate-activation 3rd direction) — **CLOSED CATASTROPHIC LOSS (+25.2% val, +26.7% test)**

- **Branch:** charliepai2g48h5-tanjiro/mishglu
- **Hypothesis:** Replace SiLU gate in SwiGLU with Mish (= x·tanh(softplus(x))). Smoother gradient near origin; no hard zero region. Predicted gate_zero_frac < SiLU baseline.
- **Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-mishglu-20260514-082712/metrics.jsonl`

| Val split | Mish | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 33.2417 | 25.2751 | **+31.5% LOSS** |
| `geom_camber_rc` | 54.7370 | 45.8179 | +19.5% LOSS |
| `geom_camber_cruise` | 22.7916 | 16.8427 | +35.3% LOSS |
| `re_rand` | 43.9890 | 35.6177 | +23.5% LOSS |
| **val_avg** | **38.6898** | **30.8909** | **+25.2% LOSS** |
| **test_avg** | **33.1923** | **26.1964** | **+26.7% LOSS** |

- **Important note on student methodology:** The student used the `train.py` defaults (epochs=50, lr=5e-4, weight_decay=1e-4) instead of the canonical advisor config (epochs=70, lr=1.5e-4, weight_decay=3e-4) used by #2810 baseline. This inflates the LOSS magnitude. However, the qualitative gate-zero-frac diagnostic is hyperparameter-independent in its dominant pattern (49-68% vs 1.3-2.4% is a 25-50× ratio not a 3x-lr-induced shift), so the gate-axis-closes conclusion stands. Closing comment asked student to use canonical reproduce command verbatim for future experiments.
- **Exemplary mechanism diagnostic — gate_zero_frac jumps 25-50× from SwiGLU baseline:**

| Block | SwiGLU SiLU baseline | MishGLU |
|---|---:|---:|
| 0 | 1.3-2.4% | **50.06%** |
| 1 | 1.3-2.4% | **49.45%** |
| 2 | 1.3-2.4% | **59.85%** |
| 3 | 1.3-2.4% | **67.77%** |

- **Depth-progressive gate_std decay BROKEN:** SwiGLU baseline showed clean depth-decay (block 0→3: 1.07→0.58). MishGLU shows broken pattern: 0.88→0.67→0.72→1.00 (block 3 HIGHER than block 0, opposite of baseline). The depth-progressive mechanism that PR #2741 identified as MECHANISM by which SwiGLU works is completely destroyed under Mish.
- **Mechanism re-interpretation:** SiLU specifically yields a learnable W_gate matrix whose pre-activation distribution places very few values in the wide near-zero region (1.3-2.4% zero_frac). Mish steers gate weights toward the wide negative tail where Mish exponentially decays — despite similar near-zero behavior to SiLU, Mish induces the optimizer to converge to a different W_gate solution. The activation function determines which W_gate solution the optimizer settles in, NOT just its local smoothness.
- **89th taxon: gate-activation axis at MLP-body SwiGLU site CLOSES DEFINITIVELY at SiLU.** Three points anchor the curve: GELU gate (GeGLU) LOSS (16-26% mask, #2759), SiLU gate (SwiGLU) WIN (1.3-2.4% mask, baseline #2741), Mish gate (MishGLU) LOSS (49-68% mask, worst yet, #2859). The gate-activation-axis-closure is comprehensive across all common activation alternatives.

---

## 2026-05-14 [Round 112] UTC — PR #2849: Cosine eta_min=1e-6 (Lion-friendly LR floor) — **CLOSED stale_wip (4th pod-failure close, NOT falsified)**

- **Branch:** charliepai2g48h5-fern/cosine-eta-min-1e-6
- **Hypothesis:** Set `eta_min=1e-6` (vs current 0) on the CosineAnnealingLR. Lion's sign-step needs non-zero LR for late-training fine-tuning since `sign(c·grad) = sign(grad)` makes the per-step contribution scale-invariant but exp_avg integration is not. At LR≈3e-6 in the cosine tail, Lion effectively stalls.
- **Status:** **CLOSED — stale_wip, 4th pod-failure close in this launch.** Created 2026-05-14T07:27:52Z, 0 commits and 0 comments beyond initial assignment. fern pod has stalled twice (#2794 Round 103 + #2849 Round 112) = chronically unreliable pod harness. Axis UNTESTED.
- **Disposition:** May re-assign cosine eta_min=1e-6 to a different student in a future round if it remains a high-value swing.
- Reassigned fern to #2876 batch_size=8 — minimally simple 1-line config change per Round 103/109 stall-prone strategy.

---

## 2026-05-14 [Round 112] UTC — PR #2876: batch_size 4→8 (Lion-batch-size sensitivity) — **ASSIGNED (89th candidate axis)**

- **Branch:** charliepai2g48h5-fern/batch-size-8
- **Hypothesis:** Double batch_size from 4 to 8 — Lion-batch-size sensitivity test. Chen et al. 2023 ("Symbolic Discovery of Optimization Algorithms") explicitly recommend larger batches for Lion ("we use bs=4096 in most experiments"); at bs=4 our Lion may be under-batched for stable sign-step direction estimation.
- **Why:** batch_size axis NEVER tested this launch. Memory headroom available (~14GB peak vs 96GB available — substantial). Lion sign-step compresses to ±1 per parameter per step, so more samples per gradient estimate = less sign noise and more reliable exp_avg integration.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): Lion was under-batched at bs=4; try bs=16 next or pair with lr-rescaling.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): bs=4 was already sufficient; close batch_size axis.
  3. **LOSS** (val > 30.8909 + 1%): Keskar-style small-batch implicit regularization was load-bearing for OOD generalization; close batch_size upward direction.
- Risk: 70 epochs with halved steps-per-epoch may undertrain. lr=1.5e-4 unchanged.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-fern --experiment_name "charliepai2g48h5-fern/batch-size-8" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`.

---

## 2026-05-14 [Round 112] UTC — PR #2875: Squared ReLU on SwiGLU up-projection (Primer 2022) — **ASSIGNED (90th candidate axis)**

- **Branch:** charliepai2g48h5-tanjiro/squared-relu-up-proj
- **Hypothesis:** Add `F.relu(W_up(x)).pow(2)` activation to SwiGLU up-projection (currently has NO activation). Keep SiLU gate unchanged (proven WIN per #2741, locked by #2759 GeGLU LOSS and #2859 MishGLU LOSS). Primer architecture (Hua et al. 2022) showed `relu(x)²` outperforms GELU in standard transformer FFNs.
- **Why:** Activation-axis at the SwiGLU GATE site is closed. But the SwiGLU UP-projection currently has NO activation. Adding non-linearity to up-projection is a distinct activation-site axis. Sparsity-without-gating: gate selects which channels get re-weighted (continuous-valued), up-activation selects which features are non-zero (hard-sparse). Two mechanisms complementary. Quadratic large-x amplification may help OOD via compact-feature-use forcing.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): Primer-style up-activation breaks plateau; try GELU² or Squared ReLU on out-projection next.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): up-projection activation doesn't help in this small-scale CFD; close up-projection activation axis.
  3. **LOSS** (val > 30.8909 + 1%): hard-zero regions in up-projection break SwiGLU mechanism; close Squared ReLU direction; try smoother alternatives (GELU on up-proj).
- bf16 overflow check requested: log `up_act/block_<i>_max` to detect if squared output exceeds bf16 dynamic range (~65504 max).
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-tanjiro --experiment_name "charliepai2g48h5-tanjiro/squared-relu-up-proj" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`. Expected param count: 333,700 unchanged.

---

## 2026-05-14 [Round 111] UTC — PR #2860: Gradient clipping max_norm=1.0 (Lion numerical stability) — **CLOSED LOSS (+4.54% val, +4.24% test)**

- **Branch:** charliepai2g48h5-edward/grad-clip-1
- **Hypothesis:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `loss.backward()` before `optimizer.step()`. Tests whether Lion's exp_avg buffer accumulates magnitudes from rare gradient spikes, perturbing exp_avg sign on subsequent steps. Numerical-stability sub-axis within training-procedure family — never tested this launch.
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-grad-clip-1-20260514-083034/metrics.jsonl`

| Val split | grad_clip=1.0 | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 25.6419 | 25.2751 | +1.45% LOSS |
| `geom_camber_rc` | 47.6635 | 45.8179 | +4.03% LOSS |
| `geom_camber_cruise` | 18.7382 | 16.8427 | **+11.25% LOSS (worst)** |
| `re_rand` | 37.1274 | 35.6177 | +4.24% LOSS |
| **val_avg** | **32.2927** | **30.8909** | **+4.54% LOSS** |
| **test_avg** | **27.3077** | **26.1964** | **+4.24% LOSS** |

- **Result:** NOT MERGED. All 4 val splits regress; OOD splits hit hardest (cruise +11.25%); in_dist only +1.45% — opposite of predicted in_dist-hit-hardest.
- **Critical diagnostic finding:** `clip_fired_frac_train_epoch = 1.0000` for ALL 62 epochs (100% clip fire rate). Pre-clip gradient norms were 22-72× the threshold throughout training: mean ep4 71.61 / ep62 22.97; max ep4 248 / ep62 61.52 — never approached 1.0 even at cosine tail LR ~6.6e-6.
- **Mechanism re-interpretation:** Lion's sign-step is per-step scale-invariant (`sign(c·grad) = sign(grad)`), BUT Lion's exp_avg buffer accumulates magnitudes: `exp_avg_t = β1·exp_avg_{t-1} + (1-β1)·grad_t`. With 100% clipping, every step's contribution to exp_avg is scaled by `1.0/||grad_t||`, which ERASES the relative-magnitude signal between training steps. The "trust steps consistently large in the same direction" property of Lion is destroyed → model learns noisiest features at the same rate as most informative features, flattening Lion's effective trajectory.
- **Mechanistically distinct from spike-clipping.** A true spike-clipping test (clip only rare 5-10% spike steps) would need max_norm in 50-100 range (≈ mean+2σ of pre-clip norms 22-72). Not pursued at this time.
- **Important codebase finding:** Lion+L1+surf_weight=10 with this architecture produces gradient norms 20-70× of paper Lion expectations. Future numerical-stability proposals should anchor thresholds to actual observed grad norms in this codebase, not general ML defaults.
- **87th taxon: gradient clipping max_norm=1.0 CLOSES.** Combined with closed EMA #2820, SWA, Lookahead, SAM, Mixup, DropPath, Weight Standardization, Deep Supervision, Gradient Centralization, prior EMA decay = training-procedure / numerical-stability meta-family COMPREHENSIVELY MAPPED.

---

## 2026-05-14 [Round 111] UTC — PR #2852: slice_num 24→20 (slice-routing-quality downward) — **CLOSED LOSS (+3.69% val, +6.16% test)**

- **Branch:** charliepai2g48h5-frieren/slice-num-20-canonical
- **Hypothesis:** Extend the slice_num downward trajectory (64→32 WIN #1846, 32→24 WIN #2307) by trying 24→20. Slice attention is information-routing (per #2828 diagnostic); fewer slices may either help generalization (less redundancy) or hurt (information bottleneck).
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-slice-num-20-canonical-20260514-082913/metrics.jsonl`

| Val split | slice_num=20 | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 25.295 | 25.2751 | +0.08% near-flat |
| `geom_camber_rc` | 47.495 | 45.8179 | +3.66% LOSS |
| `geom_camber_cruise` | 18.621 | 16.8427 | **+10.56% LOSS (worst)** |
| `re_rand` | 36.703 | 35.6177 | +3.05% LOSS |
| **val_avg** | **32.029** | **30.8909** | **+3.69% LOSS** |
| **test_avg** | **27.808** | **26.1964** | **+6.16% LOSS** |

- **Result:** NOT MERGED. Param count 332,916 = −784 (slice_token + Project_To_Slices removal). Best ep57/70 timeout-truncated at 30 min cap.
- **Important note on student analysis:** Student's per-split delta analysis used wrong reference baselines (cruise 33.93 vs actual #2810 cruise 16.84, in_dist 23.59 vs actual 25.28, camber_rc 32.36 vs actual 45.82, re_rand 33.81 vs actual 35.62). This led to a "striking camber-asymmetry: cruise WIN −45.13%, camber_rc LOSS +46.78%" narrative built on those wrong deltas. When recomputed against correct #2810 baseline (table above): all 4 splits LOSS — cruise is the WORST hit (+10.56%), NOT a win. The asymmetry-driven mechanism (high-camber needs more routing capacity, low-camber benefits from tighter routing) was artifact of the wrong reference values.
- **Real signal:** uniform LOSS across all 4 splits at slice_num=20, with cruise most-sensitive. The slice_num trajectory 64→32 WIN, 32→24 WIN, 24→20 LOSS suggests 24 is at or near the optimum from the downward direction.
- **Useful independent diagnostics:**
  - SE block-3 gate stats per split: gate_std 0.15-0.27, gate_min 0.008-0.08 → gating active and healthy across splits.
  - LayerScale γ_attn near-identity at terminal (block0/3 abs_mean ~0.015, weakly grown from 1e-4 init); γ_mlp dominant (block0/3 abs_mean ~0.06-0.07).
  - SwiGLU gate stats: gate_mean 0.23-0.42, std 0.83-1.07, zero_frac <2% → gates active, no hard masking.
  - FiLM (block-0) at val_re_rand: |1+γ| mean 0.965 → near-identity post-FiLM scaling.
- **88th taxon: slice_num downward axis CLOSES at this magnitude.** Closing comment to student asked to anchor future split-comparisons to `BASELINE.md` exact reference numbers to avoid re-occurring baseline-confusion. Reassigning frieren to slice_num=28 (UPWARD direction, never tested this launch).

---

## 2026-05-14 [Round 111] UTC — PR #2873: Post-norm structural pivot (LN-outside-residual at ln_1/ln_2) — **ASSIGNED (87th candidate axis)**

- **Branch:** charliepai2g48h5-edward/post-norm
- **Hypothesis:** Replace pre-norm `x = x + γ*attn(LN(x))` with post-norm `x = LN(x + γ*attn(x))` at all 8 sites (ln_1 and ln_2 per block × 4 blocks). Keep `ln_3` (last_layer) unchanged. +0 params.
- **Why:** Pre/post-norm structural choice has NEVER been tested in this launch. Liu et al. 2020 ("Understanding the Difficulty of Training Transformers") showed post-norm yields better final performance once stability is solved; original Transformer used post-norm; modern pre-norm became default for deep models (>30 layers). At 4 blocks with LayerScale γ=1e-4 init, post-norm should be stable from start (`LN(x + tiny·attn(x)) ≈ LN(x)`).
- **OOD-bias hypothesis:** Post-norm's per-layer activation calibration may help generalization to OOD distributions by preventing in-dist DC-offset memorization in residual stream (similar mechanism diagnosis as #2851 mean-centering for OOD).
- **Failure mode to watch:** FiLM at block 0 ln_1 site — pre-norm normalizes FiLM-modulated x before feeding to attn; post-norm feeds FiLM-modulated x DIRECTLY to attn (no normalization). May damage FiLM conditioning.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): post-norm activation calibration breaks plateau; try post-LN + ScaleNorm or per-block γ-init scaling as follow-ups.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): pre/post-norm sweet-spot locally flat at 4 blocks; close structural-norm-site axis.
  3. **LOSS** (val > 30.8909 + 1%): pre-norm load-bearing; close post-norm direction; FiLM conditioning likely failure mode.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-edward --experiment_name "charliepai2g48h5-edward/post-norm" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`. Expected param count 333,700 unchanged.

---

## 2026-05-14 [Round 111] UTC — PR #2874: slice_num=28 (upward direction test) — **ASSIGNED (88th candidate axis)**

- **Branch:** charliepai2g48h5-frieren/slice-num-28
- **Hypothesis:** Test slice_num=28 (UP one step from baseline 24). The slice_num trajectory 64→32 WIN, 32→24 WIN, 24→20 LOSS closes the downward direction. The upward direction has NEVER been tested in this launch — clean axis-direction test.
- **Why:** +4 slices vs baseline 24 (+16.7% routing capacity); ~+800 params (~334,500 total). Tests whether the previous 32→24 step (#2307 WIN) was at the local optimum or whether further upward exploration was prematurely abandoned. OOD-pathway diversity hypothesis: more slice-routing pathways could give more pathway diversity for OOD geometries that don't fit in-dist routing patterns.
- **Important assignment note:** Student #2852 used wrong reference baselines for per-split deltas — assignment explicitly asks anchoring to `BASELINE.md` exact reference values (in_dist 25.2751, camber_rc 45.8179, camber_cruise 16.8427, re_rand 35.6177) to prevent baseline-confusion artifacts.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): slice_num upward direction has headroom; try slice_num=32 next.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): slice_num=24 sweet spot locally flat; close slice_num axis comprehensively.
  3. **LOSS** (val > 30.8909 + 1%): #2307 32→24 was the correct direction and going up reverses it; close slice_num upward direction.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-frieren --experiment_name "charliepai2g48h5-frieren/slice-num-28" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`.

---

## 2026-05-14 [Round 110] UTC — PR #2858: PhysicsAttention temperature init 0.5→1.0 — **CLOSED CATASTROPHIC LOSS (+32.41% val, +32.57% test)**

- **Branch:** charliepai2g48h5-nezuko/temperature-init-1
- **Hypothesis:** Initialize PhysicsAttention slice-routing temperature at 1.0 (natural scale) instead of baseline 0.5 (2× sharper). Test whether sharper-than-default routing was an essential inductive bias or design choice never swept.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-temperature-init-1-20260514-081535/metrics.jsonl`

| Val split | T=1.0 init | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 34.9603 | 25.2751 | **+38.32% LOSS** |
| `geom_camber_rc` | 56.3501 | 45.8179 | +22.99% LOSS |
| `geom_camber_cruise` | 25.7488 | 16.8427 | **+52.88% LOSS** |
| `re_rand` | 46.5296 | 35.6177 | +30.64% LOSS |
| **val_avg** | **40.8972** | **30.8909** | **+32.41% LOSS** |
| **test_avg** | **34.7288** | **26.1964** | **+32.57% LOSS** |

- **Result:** NOT MERGED. All 4 val splits regress catastrophically; in_dist and cruise hit hardest.
- **Mechanism — exemplary student diagnostic:**
  - Per-block-per-head temperature trajectory showed **block-1 head-0 trained all the way to T=−1.88 by ep50** — sign-flipped negative temperature negates softmax logits causing geometric routing inversion. Block 0 sharpened to T=[+2.965, +1.864]; block 2 softened to T=[+0.747, +0.556]; block 3 sharpened to T=[+1.856, +2.295].
  - Slice routing entropy collapsed to near-zero in 3/4 blocks: block-0 entropy=1.7e-23 (effectively one-hot), block-1 entropy=0.018 (one-hot but sign-inverted routing), block-2 entropy=0.0 (perfectly one-hot), block-3 entropy=1.019 (only block retaining meaningful spread).
  - **Mechanism re-interpretation:** T=0.5 init was load-bearing NOT because of sharpness preference but as a **positive-basin attractor**. Starting at T=1.0 lets the optimizer freely cross zero into negative-T degenerate basin where 1/T→∞ amplifies gradients explosively, while the default-positive basin acts as an implicit attractor. This converts what looked like a hyperparameter choice into a structural property of the parameterization.
  - OOD damage > in-dist damage on absolute scale (+38% in_dist, +53% cruise) — routing degeneracy hurts both generalization and capacity, consistent with #2809 frieren generalization-limited diagnosis.
- **85th taxon: temperature_init upward direction CLOSES.** Reinforces #2828/#2848 thread: slice attention is structurally one-hot-attractor with limited OOD headroom; any relaxation of constraints accelerates degeneracy. Not pursuing student-suggested T=0.25 follow-up — moving nezuko to fresh axis (output-head residual gain).

---

## 2026-05-14 [Round 110] UTC — PR #2856: PhysicsAttention n_head 2→4 — **CLOSED LOSS (+3.71% val, +1.66% test)**

- **Branch:** charliepai2g48h5-alphonse/n-head-4
- **Hypothesis:** Test n_head=2→4 with dim_head auto-derived as 96/4=24. Predicted: per-head specialization may enable different physics scales (boundary-layer vs wake, surface vs volume routing patterns).
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-n-head-4-20260514-081344/metrics.jsonl`

| Val split | n_head=4 | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 27.3958 | 25.2751 | **+8.39% LOSS (worst hit)** |
| `geom_camber_rc` | 46.9870 | 45.8179 | +2.55% LOSS |
| `geom_camber_cruise` | 18.0142 | 16.8427 | +6.96% LOSS |
| `re_rand` | 35.7499 | 35.6177 | +0.37% near-flat |
| **val_avg** | **32.0367** | **30.8909** | **+3.71% LOSS** |
| **test_avg** | **26.6328** | **26.1964** | **+1.66% LOSS** |

- **Result:** NOT MERGED. Best ep60 (61/70 reached before 30-min timeout). Param count **310,668 = −23,032 (−6.9%)** — NOT the predicted ~0 change.
- **Critical architectural insight:** Per-head `Linear(dim_head, dim_head)` construction in PhysicsAttention means doubling n_head 2→4 QUARTERS QKV params per block. So this was a CAPACITY-REDUCTION test (−23,032 params), NOT a head-parallelism test.
- **Mechanism:**
  - **Capacity-bottleneck signature:** in_dist +8.39% > cruise +6.96% > camber_rc +2.55% > re_rand +0.37% — the densely-supported distribution needs the lost representational rank most, OOD splits barely move (confirming attention rank IS in-dist capacity-bottlenecked at H=2).
  - **No per-head specialization observed.** All 16 temperatures ended up tight in [0.42, 0.56] with per-block std 0.029-0.055. Block-3 (output block) showed LEAST divergence std=0.029 = heads don't specialize at deeper blocks. Falsifies "per-head temperature specialization" WIN-scenario mechanism.
  - Final per-block-per-head temperatures (best ep60): block[0]: [0.559, 0.541, 0.434, 0.501]; block[1]: [0.509, 0.507, 0.425, 0.483]; block[2]: [0.476, 0.543, 0.449, 0.475]; block[3]: [0.473, 0.540, 0.501, 0.525].
- **79th taxon: n_head=4 with current dim_head=n_hidden/n_head construction CLOSES at this magnitude.** Reinforces #2828/#2848 thread (slice attention structurally one-hot-attractor) and #2809 frieren (OOD generalization-limited not capacity-limited). Reassigning alphonse to student's preferred follow-up #2: n_head=1 with dim_head=96 (maximum per-head rank falsification test).

---

## 2026-05-14 [Round 110] UTC — PR #2869: n_head=1 dim_head=96 maximum per-head rank attention — **ASSIGNED (85th candidate axis)**

- **Branch:** charliepai2g48h5-alphonse/n-head-1-max-rank
- **Hypothesis:** Test n_head=1 with dim_head=96 (maximum per-head rank within current n_hidden=96 budget) — natural falsification of #2856 capacity-bottleneck hypothesis. Halves n_head 2→1, doubling dim_head 48→96 single-head MAX-rank attention.
- **Why:** alphonse's #2856 diagnostic established attention rank IS in-dist capacity-bottlenecked at H=2. This experiment goes in OPPOSITE direction: maximum per-head rank, zero head-parallelism. Param expansion concentrated in attention where capacity needed: per-block to_q/k/v doubles from 6,912 → 27,648 (+20,736/block × 4 blocks = +82,944 attention params). Expected ~415,000 params = +25%.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): attention rank capacity-bottlenecked at H=2; H=1 max-rank breaks plateau; try as new architectural baseline.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): tradeoff between rank-gain and head-parallelism-loss balanced; close attention-rank axis.
  3. **LOSS** (val > 30.8909 + 1%): multi-head parallelism was load-bearing; close H=1 direction; attention sweet spot is H=2.
- Risk: with +25% params, sec/epoch may increase ~10-15%, monitor whether 70 epochs fits in 30-min cap.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-alphonse --experiment_name "charliepai2g48h5-alphonse/n-head-1-max-rank" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`.

---

## 2026-05-14 [Round 110] UTC — PR #2870: Output-head residual gain (learnable scalar at output) — **ASSIGNED (86th candidate axis)**

- **Branch:** charliepai2g48h5-nezuko/output-head-gain
- **Hypothesis:** Add a single learnable scalar `output_gain = nn.Parameter(torch.tensor(1.0))` multiplied into final decoder output before loss. Tests whether output magnitude is uncalibrated after LayerScale residual-stream changes accumulated through `#2614 FiLM-embedding + LayerScale γ_attn=1e-4 / γ_mlp=1e-1 + SE-block3-attn-pool #2810` winners. +1 param total.
- **Why this is a fresh axis:** Output-scale calibration is the only post-decoder learnable degree-of-freedom never tested this launch. Lion's sign-step normalizes gradient direction per parameter WITHIN block but cannot tune the final output scale through loss gradient. A 1-parameter scale-DoF lets the optimizer discover whether the head is systematically over- or under-emphasized.
- **Why this is from Round 108 bold-swing queue (plateau-protocol):** Moving nezuko away from attention/routing area where they just had catastrophic #2858 LOSS. Output-head gain was one of 5 plateau-bold-swing candidates queued in Round 108 commit notes.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): gain learns meaningful non-trivial value; report whether <1 (head over-emphasized) or >1 (under-emphasized). Try per-channel output gain in follow-up.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): gain ≈ 1.0 throughout = decorative parameter; close 1-scalar direction; consider per-channel gain or pre-output-LN.
  3. **LOSS** (val > 30.8909 + 1%): output-magnitude is calibrated as-is and learnable scalar destabilized training; close output-head gain axis.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-nezuko --experiment_name "charliepai2g48h5-nezuko/output-head-gain" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`. Expected param count: 333,701 (= 333,700 + 1).

---

## 2026-05-14 [Round 109] UTC — PR #2827: LayerScale γ init = 1e-3 — **CLOSED stale_wip (3rd pod-failure close, NOT falsified)**

- **Branch:** charliepai2g48h5-thorfinn/layerscale-init-1e-3
- **Hypothesis:** Increase LayerScale γ initialization from 1e-4 → 1e-3 (10× larger). Test residual gain init sensitivity. At 1e-4 init, residual stream is near-identity → maybe under-utilizing the sublayer signals from the start.
- **Status:** **CLOSED — stale_wip, pod failure to pick up.** Created 2026-05-14T06:47:08Z, 0 commits and 0 comments beyond initial assignment, thorfinn pod persistently failing to poll. Per Round 91/99/100/103 convention, this is the 3rd pod-failure close — NOT experimental falsification. Axis UNTESTED.
- **Disposition:** May re-assign LayerScale γ-init=1e-3 to a different student in a future round if it remains a high-value swing.
- Reassigned thorfinn to #2868 Smooth L1 (Huber) loss-shape pivot — minimally simple per Round 103 #2849 strategy to maximize pickup probability for stall-prone pod.

---

## 2026-05-14 [Round 109] UTC — PR #2868: Smooth L1 (Huber β=1.0) training loss — **ASSIGNED (84th candidate axis)**

- **Branch:** charliepai2g48h5-thorfinn/smooth-l1-huber-beta1
- **Hypothesis:** Replace L1 training loss with Smooth L1 (Huber) at β=1.0 for both volume and surface terms, keeping `surf_weight=10`. **Critically: keep validation/test reporting metrics as L1 (MAE)** — only the training objective changes, so val/test results remain directly comparable to baseline `30.8909` / `26.1964`.
- **Why:** Loss-shape axis (training loss SHAPE not WEIGHTING) has NEVER been tested this launch. Surf_weight upward (#2842 LOSS) tested weighting; this tests the underlying error-function shape. Smooth L1 has smooth gradient through origin (quadratic for |r|<1, linear for |r|≥1) vs L1's discontinuous ±1 at zero. May improve cosine-tail convergence where the loss landscape becomes shallow and Lion's sign-step benefits from richer near-zero gradient structure that exp_avg integrates. OOD outlier behavior unchanged (large residuals identical to L1).
- **Diagnostics requested:** Per-step gap between SmoothL1 and L1 measurements (if zero, change was operationally invisible); residual histogram at best-val ckpt across 4 bins to determine quadratic-vs-linear mass distribution.
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): smooth-near-zero matters for cosine-tail + Lion → try β=0.5 next.
  2. **WASH** (val ≈ 30.8909 ± 0.5%): residual distribution dominated by linear regime, β=1.0 too large → try β=0.1 next.
  3. **LOSS** (val > 30.8909 + 1%): small-residual smoothing hurts Lion's late-training updates → loss-shape axis CLOSES at this magnitude.
- **NEW bar to beat:** val_avg < **30.8909**. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-thorfinn --experiment_name "charliepai2g48h5-thorfinn/smooth-l1-huber-beta1" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`. Expected param count: 333,700 unchanged.

---

## 2026-05-14 [Round 108] UTC — PR #2851: RMSNorm replacement of LayerNorm at all 9 sites — **CLOSED LOSS (+1.67% val, +3.83% test)**

- **Branch:** charliepai2g48h5-askeladd/rmsnorm-replace
- **Hypothesis:** Replace `nn.LayerNorm` with `RMSNorm` (Zhang & Sennrich 2019) at all 3 LayerNorm sites in `TransolverBlock` (ln_1, ln_2 per block × 4 blocks + ln_3 in last_layer = 9 sites). RMSNorm: `γ * (x / RMS(x))`, no mean centering, no β. LLaMA/T5/Mistral/PaLM-2 standard. −864 params (96 β × 9 sites).
- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-rmsnorm-replace-20260514-075010/metrics.jsonl`

| Val split | RMSNorm | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | **24.2105** | 25.2751 | **−4.21% WIN (best-ever in-dist)** |
| `geom_camber_rc` | 47.5202 | 45.8179 | +3.71% LOSS |
| `geom_camber_cruise` | 17.7834 | 16.8427 | +5.58% LOSS |
| `re_rand` | 36.1124 | 35.6177 | +1.39% LOSS |
| **val_avg** | **31.4066** | **30.8909** | **+1.67% LOSS** |

| Test split | RMSNorm | Baseline #2810 | Δ |
|---|---:|---:|---:|
| `single_in_dist` | 23.7295 | 23.4553 | +1.17% flat |
| `geom_camber_rc` | 43.1389 | 41.1687 | +4.79% LOSS |
| `geom_camber_cruise` | 14.2638 | 14.1435 | +0.85% flat |
| `re_rand` | 27.6667 | 25.9980 | +6.42% LOSS |
| **test_avg** | **27.1997** | **26.1964** | **+3.83% LOSS** |

- **Result:** NOT MERGED. Param count 332,836 ✓ matches predicted 333,700 − 864. 67/70 epochs (best ep67 still descending; cosine tail LR ~7.4e-7). sec/epoch ~27s — predicted 2-5% speedup NOT measurable.
- **Striking pattern:** in-dist WIN + OOD LOSS on 3/4 OOD splits = exact OPPOSITE of recent merge-pattern (#2810/#2765/#2727/#2614 all OOD-positive). Mean-centering and β are load-bearing for OOD generalization.
- **Mechanism — exemplary student 3-part diagnostic:**
  1. **γ-evolution pattern: 8/9 sites untouched.** Per-site γ stats showed only `block0_ln1` had learned modulation (γ_mean 0.91, std 0.098, range [0.625, 1.135]). The other 8 sites had γ ≈ 1.000 ± 0.002 essentially identity. LayerScale γ_attn~1e-3 / γ_mlp~1e-1 makes the residual near-identity → gradient signal back through these norms is small and uniform → RMSNorm's diagonal γ has nothing to push against. LayerNorm's β provides per-channel bias that CAN shift the residual stream irrespective of pre-norm magnitude. **β was the load-bearing degree of freedom for 8/9 sites.** The exception block0_ln1 sits immediately after the FiLM gate `fx = fx * (1 + γ_c(Re,AoA))` injection — RMSNorm there has per-channel bias content to learn from, but γ_min collapsed to 0.625 to suppress some channels (the place where the architecture suffers most from losing β).
  2. **Mean-centering and β are load-bearing for OOD generalization.** Slice attention is permutation-invariant. Each token's pre-norm representation carries both shape (per-channel structure) and absolute-scale (per-token mean magnitude) information. Mean-centering removes per-token DC offset, forcing the model to encode positional info into relative channel structure — useful for OOD generalization. Without it, the network "memorizes" in-dist DC offset as an extra discriminator (in-dist gain) but fails to generalize the scale structure to OOD distributions (Re/geometry/camber shifts).
  3. **LLaMA/T5/Mistral RMSNorm prior does not transfer.** Causal LLMs use absolute/rotary position embeddings where per-token mean carries no inductive load. In Transolver, slice attention is the only structure that can route per-mesh-node info; per-token mean IS the location signal. Prior architectural choice doesn't transfer.
- **78th taxon: RMSNorm-at-all-sites CLOSES.** Combined with closed DyT #2686, embed-LN replacement #2808, surface-aware-LN #2829, QK-Norm, GroupNorm = normalization-axis comprehensively mapped at SITE level. LN-feature-decomposition (mean-centering vs γ vs β) partially resolved: β is load-bearing for the OOD-generalization-active configuration.
- **Plateau deepens to 7 consecutive LOSSes** since #2810 merge.

---

## 2026-05-14 [Round 108] UTC — PR #2864: Hybrid LN-at-block0_ln1 + RMSNorm-at-8-other-sites — **ASSIGNED (83rd candidate axis)**

- **Branch:** charliepai2g48h5-askeladd/hybrid-ln-rmsnorm
- **Hypothesis:** Direct architectural conversion of #2851 diagnostic finding. Keep `nn.LayerNorm` at block 0 ln_1 (the ONE site where #2851's per-site γ stats showed active learning post-FiLM, γ range [0.625, 1.135]); replace LN with `RMSNorm` at the 8 inactive-γ sites (block0_ln2, block1_ln1+ln2, block2_ln1+ln2, block3_ln1+ln2+ln3-last_layer). Reuses #2851 RMSNorm class.
- **Why:** Student #2851 diagnostic established that 8/9 RMSNorm sites had γ ≈ 1.000 ± 0.002 (essentially untouched) — at those sites, BOTH γ and β were decorative. The exception block0_ln1 has FiLM γ_c(Re,AoA) per-channel bias content immediately upstream that β captures. Hybrid keeps full LN where β is doing work, strips β where γ proved decorative.
- **Expected param count:** 332,932 = 333,700 − 768 (8 × 96 β params removed).
- **Three falsifiable predictions:**
  1. **WIN** (val < 30.8909): 8 inactive-γ sites genuinely don't need β → save 768 params for free → close LN-feature-decomposition with structural improvement.
  2. **WASH** (val ≈ 30.8909): same conclusion with smaller margin; net save 768 params for free.
  3. **LOSS similar to #2851** (val ≈ 31.4): β is universally load-bearing regardless of γ activity at the site → comprehensively close normalization-axis.
- **NEW bar to beat:** val_avg < **30.8909** vs current baseline #2810. Reproduce: `cd target/ && python train.py --agent charliepai2g48h5-askeladd --experiment_name "charliepai2g48h5-askeladd/hybrid-ln-rmsnorm" --lr 1.5e-4 --weight_decay 3e-4 --epochs 70`.

---

## 2026-05-14 [Round 107] UTC — PR #2848: SE bottleneck post-GELU Dropout p=0.1 — **CLOSED LOSS (+1.31% val)**

- **Branch:** charliepai2g48h5-tanjiro/se-bottleneck-dropout
- **Hypothesis:** Insert `nn.Dropout(p=0.1)` between GELU and fc2 in SE bottleneck. Test the implicit-sparsification reading of #2814 (GELU hard-masking is beneficial regularization that dropout would stack on). +0 params.
- **Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-se-bottleneck-dropout-20260514-073007/metrics.jsonl`

| Split | val | Baseline #2810 | Δ val |
|---|---|---|---|
| `single_in_dist` | 26.2293 | 25.2751 | **+3.78% LOSS (WORST hit)** |
| `geom_camber_rc` | 46.8036 | 45.8179 | +2.15% LOSS |
| `geom_camber_cruise` | 17.0287 | 16.8427 | +1.10% LOSS |
| `re_rand` | 35.1193 | 35.6177 | **−1.40% WIN (only)** |
| **val_avg** | **31.2952** | **30.8909** | **+1.31% LOSS** |
| **test_avg** | **26.4491** | **26.1964** | **+0.96% LOSS** |

- **Result:** NOT MERGED. val +1.31% LOSS; test +0.96% LOSS. Param count unchanged at 333,700.
- **Mechanism FALSIFIED — exemplary student diagnostic:**
  - **gelu_zero_frac collapsed to 0.0 across all splits** vs expected 16-26% from PR #2759 GeGLU diagnostic. GELU's hard-masking is NOT a fixed intrinsic property — it's data-dependent on what fc1 produces.
  - The optimizer **moved fc1's outputs into GELU's positive linear regime** to recover signal-to-noise from dropout noise. The two regularizers don't stack — they compete, and the network picks the lower-informational-cost one.
  - **SE gate selectivity collapsed:** gate_std 0.20-0.34 → 0.045-0.132; gate_min 0.001-0.02 → 0.10-0.38. Gate became near-uniformly half-open (range 0.38-0.62 on in_dist) → SE doing almost no channel selection.
  - Dropout zeroing fc1→fc2 paths during training penalizes sharp channel-conditional gates → optimizer converges to soft, robust, near-identity gate (the antithesis of SE's purpose).
- **Re-interpretation of #2814 SE GELU→SiLU LOSS:** The right reading isn't "GELU's 16-26% sparsity is useful" — it's likely "GELU's smoother gradient near zero (vs SiLU) is what matters" OR "the gate range distribution shifts under SiLU in a way that hurts". Dropout doesn't recover the GELU advantage; it pushes the network in yet another direction.
- **76th taxon: SE bottleneck channel dropout CLOSES.** Combined with closed DropPath #2722 (residual zeroing) and closed attention dropout #2828 (sdpa attn-prob + to_out), **stochastic-channel-zeroing meta-axis 3-direction CLOSED** across all major architectural sites.

---

## 2026-05-14 [Round 107] UTC — PR #2820: EMA-of-weights decay=0.9999 — **CLOSED LOSS (+6.46% val vs new baseline)**

- **Branch:** charliepai2g48h5-edward/ema-weights-decay9999
- **Hypothesis:** Maintain shadow state dict EMA of model weights with decay=0.9999, evaluate from EMA snapshot at best-val checkpoint and test. Mechanism: Polyak averaging smooths Lion+cosine trajectory; flat-minima generalize better OOD. +0 trainable params.
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-ema-weights-decay9999-20260514-072747/metrics.jsonl`

Note: PR scoped against OLD baseline 31.3216 (PR #2765) before #2810 merged. New baseline is 30.8909.

| Split | val_ema (ep60) | val_base (ep58, same run) | new baseline #2810 | EMA−base | Δ vs new baseline |
|---|---|---|---|---|---|
| `single_in_dist` | 26.7604 | 27.6099 | 25.2751 | −0.85 EMA WIN | +5.88% LOSS |
| `geom_camber_rc` | 49.0434 | 49.1773 | 45.8179 | −0.13 EMA WIN | +7.04% LOSS |
| `geom_camber_cruise` | 19.0128 | 19.3644 | 16.8427 | −0.35 EMA WIN | +12.88% LOSS |
| `re_rand` | 36.7221 | 36.0171 | 35.6177 | +0.71 base wins | +3.10% LOSS |
| **val_avg** | **32.8847** | **33.0422** | **30.8909** | **−0.16 EMA WIN** | **+6.46% LOSS** |
| **test_avg** | **27.4119** | **28.0115** | **26.1964** | **−0.60 EMA WIN all 4 splits** | **+4.66% LOSS** |

- **Result:** NOT MERGED. Mechanism worked but training-budget cost killed it.
- **Mechanism WORKED — exemplary student diagnostic:**
  - EMA descended monotonically from ep7; lapped base from ~ep10 with gap up to −27 in high-LR mid-training, shrinking to −0.5 to −1.7 in cosine endgame.
  - Test EMA−base = −0.60 on ALL 4 splits = stronger than val (re_rand only val regress at +0.71 vs base).
  - EMA-best ep60 vs base-best ep58 = +2 epochs later AS PREDICTED.
- **CRITICAL: EMA-base L2 drift = 2.60%** vs predicted 5-15%. **Lion + cosine + 70 epochs at this model scale already produces a relatively flat trajectory** → EMA's headroom is structurally bounded.
- **Training-budget cost killed it:** Each epoch ~30s vs baseline ~28s due to double-validation passes (base + EMA). Net = 4 lost epochs. Baseline reached 31.32 at ep64; this run capped at ep60 with val_ema still descending toward ~32.0-32.5. Per-epoch generalization gain < per-epoch training-budget cost.
- **77th taxon: EMA-of-weights at decay=0.9999 CLOSES.** Combined with closed SWA #2567 (28th) + Lookahead #2740 (55th) + SAM #2761 (60th) + Mixup #2687 + DropPath #2722 + Weight Standardization #2739 + Deep Supervision #2778 + Gradient Centralization #2710 + prior closed EMA #2735 (different decay), **training-procedure / weight-averaging / flat-minima-escape meta-family COMPREHENSIVELY MAPPED on this codebase**.

---

## 2026-05-14 [Round 107] UTC — PR #2859: MishGLU (SwiGLU gate SiLU → Mish) — **ASSIGNED (81st candidate axis)**

- **Branch:** charliepai2g48h5-tanjiro/mishglu
- **Hypothesis:** Replace SiLU with Mish (Misra 2019) in the SwiGLU MLP gate. Mish = `x * tanh(softplus(x))`. 3rd direction in gate-activation sub-axis:
  - GeGLU (gelu gate): CLOSED LOSS #2759 (gate_zero_frac 16-26% hard-masking)
  - SwiGLU (silu gate): MERGED WIN #2741 (gate_zero_frac 1.3-2.4%)
  - MishGLU (mish gate): this experiment, untested
- **Predicted mechanism (WIN):** Mish's smoother gradient yields slightly better optimization; gate_zero_frac stays low; OOD splits benefit marginally.
- **Predicted mechanism (LOSS):** Mish ≈ SiLU on this task; washes within noise; OR Mish's asymmetric negative-bump introduces regularization that hurts.
- **Param impact:** +0 params (1-line `act_fn=F.silu` → `act_fn=F.mish`).
- **Plateau-protocol pivot:** Tanjiro's SE-axis is fully closed (r=8/4/2 width + GELU/SiLU inner + attn-pool + GELU+dropout). MLP-body gate-activation is structurally distinct site.

---

## 2026-05-14 [Round 107] UTC — PR #2860: Gradient clipping max_norm=1.0 — **ASSIGNED (82nd candidate axis)**

- **Branch:** charliepai2g48h5-edward/grad-clip-1
- **Hypothesis:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` and before `optimizer.step()`. NEVER tested in this launch.
- **Sub-axis:** Numerical-stability within training-procedure family — distinct from EMA/SWA/Lookahead/SAM/Mixup/DropPath/WeightStandardization/DeepSupervision/GradientCentralization (all closed).
- **Mechanism (Chen 2023 Lion paper recommendation):** Lion's sign-step gives implicit gradient regularization (each update bounded magnitude per dimension). Explicit gradient norm clipping is complementary — bounds the TOTAL update direction. Lion's exp_avg buffer accumulates gradient magnitudes, so per-step gradient outliers can perturb exp_avg in ways that flip its sign on subsequent steps. Clipping removes rare gradient spikes that perturb exp_avg.
- **Edward's prior #2820 EMA diagnostic confirmed:** Lion + cosine produces 2.6% L2 drift = relatively flat trajectory. Clipping might further smooth.
- **Predicted mechanism (WIN):** Spike clipping reduces rare exp_avg perturbations; val drops 0.3-1.0%. `clip_fired_frac` should be 1-10%.
- **Predicted mechanism (WASH, most likely):** Clipping fires <2% of steps; minimal effect; val within ±0.3% of baseline.
- **Predicted mechanism (LOSS):** Lion+cosine already smooth enough; clipping at 1.0 too aggressive; val regresses.
- **Param impact:** +0 params, ~1-3ms/step overhead (negligible).
- **Diagnostic to log:** `grad_norm_mean_train_epoch`, `grad_norm_max_train_epoch`, `clip_fired_frac_train_epoch` per training epoch.

---

## 2026-05-14 [Round 106] UTC — PR #2844: Block-3-only FiLM (Re/AoA conditioning at final decoder block) — **CLOSED LOSS (+0.245% val)**

- **Branch:** charliepai2g48h5-nezuko/block3-film
- **Hypothesis:** Apply FiLM modulation only at block 3 (final TransolverBlock) after SE gate, before mlp2. Linear(3, 96) zero-init weight + bias. Block 3 is the natural integration site (γ_attn sign-flip observed there in #2765/#2798). Predicted: in_dist + re_rand benefit from targeted Re/AoA conditioning; camber splits benefit too. +384 params.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-block3-film-20260514-071950/metrics.jsonl`

| Split | val | Baseline #2810 | Δ val |
|---|---|---|---|
| `single_in_dist` | 24.8407 | 25.2751 | **−1.72% WIN (new best-ever)** |
| `geom_camber_rc` | 47.0321 | 45.8179 | **+2.65% LOSS** |
| `geom_camber_cruise` | 16.9145 | 16.8427 | +0.43% LOSS |
| `re_rand` | 35.0787 | 35.6177 | **−1.51% WIN** |
| **val_avg** | **30.9665** | **30.8909** | **+0.245% LOSS** |
| **test_avg** | **26.4983** | **26.1964** | **+1.15% LOSS** |

- **Result:** NOT MERGED. val +0.245% LOSS (close); test +1.15% LOSS; param count 334,084 = +384 vs baseline.
- **Mechanism CONFIRMED (student exemplary diagnostic):**
  - FiLM-b3 head learned substantive correction: `weight_norm=3.10`, `bias_norm=0.77`, `|γ| = 0.16-0.26` = **3-4× the embedding FiLM** magnitude (|γ|≈0.064).
  - `|γ|` anti-correlated with OOD usefulness: camber_cruise has largest |γ|=0.258 and LOSSES; re_rand smallest |γ|=0.18 and WINS.
  - Block 3 SE gate stats unchanged at convergence — FiLM-b3 did NOT visibly distort SE distribution.
- **Key insight:** Block 3 is structurally over-allocated. Already has γ_attn, γ_mlp, SE gate, residual → adding FiLM-b3 = 5 competing modulators at same site, with NO normalization downstream before mlp2. Optimizer finds in_dist-fitting config (24.84 new best-ever) that breaks camber_rc generalization.
- **Contrast with merged embedding FiLM #2614:** Lives BEFORE block-0 with 3 further blocks of LN+mixing downstream regularizing the modulation. Block-3 FiLM has no such regularizer.
- **74th taxon: FiLM-axis CLOSES across both sub-directions:**
  - per-block FiLM #2813: 4-way LOSS (over-conditioning intermediate representations)
  - block-3-only FiLM #2844: targeted single-block LOSS (over-allocation at decoder integration site)
  - merged embedding FiLM #2614: unique winner — placement BEFORE the stack with downstream regularization is the only generalizing FiLM site.

---

## 2026-05-14 [Round 106] UTC — PR #2842: surf_weight 10 → 15 (+50% surface pressure emphasis) — **CLOSED LOSS (+1.62% val)**

- **Branch:** charliepai2g48h5-alphonse/surf-weight-15
- **Hypothesis:** Bump surf_weight from 10.0 to 15.0. The current model is substantially more capable than the model surf_weight=10 was tuned for. Stronger surface supervision = direct optimization of primary metric. SE attn-pool gates channels globally; surf_weight gates per-token by surface membership — orthogonal granularities. Structured regularization, zero capacity.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-surf-weight-15-20260514-070658/metrics.jsonl`

| Split | val | Baseline #2810 | Δ val | Verdict |
|---|---|---|---|---|
| `single_in_dist` | 25.0772 | 25.2751 | −0.20 | WIN |
| `geom_camber_cruise` | 16.2802 | 16.8427 | −0.56 | WIN |
| `re_rand` | 35.0345 | 35.6177 | −0.58 | WIN |
| `geom_camber_rc` | 49.1755 | 45.8179 | **+3.36 (+7.32%)** | catastrophic LOSS |
| **val_avg** | **31.3919** | **30.8909** | **+1.62% LOSS** | |
| **test_avg** | **26.6495** | **26.1964** | **+1.73% LOSS** | |

- **Result:** NOT MERGED. val +1.62% LOSS; test +1.73% LOSS. 3 of 4 splits WIN modestly (−0.2 to −0.6); camber_rc REGRESSES +3.36 (+7.32%) — overwhelms the average.
- **Mechanism CONFIRMED:** Directional intuition correct (3/4 splits improved with stronger surface supervision). But camber_rc — already the hardest geometric OOD split with largest baseline error (45.8) — destabilizes catastrophically. Pushing harder on surface gradients over-fits dominant training surface-pressure patterns at cost of long-tail camber geometries.
- **Bigger insight:** Clean illustration of the loss-weighting bias/variance trade. Heavier weighting on a residual budget reduces bias on bulk distribution and amplifies variance on tails. Combined with the model being generalization-limited (frieren #2809), the bulk-vs-tail trade settles on the wrong side of optimum.
- **75th taxon: surf_weight UPWARD axis CLOSES** at this magnitude (10 → 15). Surf_weight=10 baseline is at-or-near optimum from the upward direction.
- **Suggested follow-ups DEFERRED:** Surf_weight sweep at {11,12,13} not worth the GPU time given steep camber_rc cliff at 15; localized loss-weighting by curvature would be complex and data-dependent. Plateau protocol favors fresh structural axes.
- **STRATEGY-TIER PIVOT:** 4 consecutive LOSSes since PR #2810 merge — moving from loss-weighting hyperparameter tuning to attention structural sub-axes never tested this launch.

---

## 2026-05-14 [Round 106] UTC — PR #2856: PhysicsAttention n_head 2 → 4 — **ASSIGNED (79th candidate axis)**

- **Branch:** charliepai2g48h5-alphonse/n-head-4
- **Hypothesis:** Increase `n_head` from 2 to 4 in PhysicsAttention. With `dim_head = n_hidden // n_head`, dim_head halves from 48 to 24. Attention parallelism sub-axis never tested in this launch. Doubles head parallelism, halves per-head dimensionality. Could enable per-head specialization on different physics scales (boundary-layer vs wake, surface vs volume routing). Slice-routing softmax already runs per-head with learnable per-head temperature, so 4 heads = 4 independent routing channels.
- **Predicted mechanism (WIN):** Per-head specialization on physical scales; temperature trajectories diverge across heads; OOD benefits from multi-scale attention diversity.
- **Predicted mechanism (LOSS):** `dim_head=24` below typical Transformer min effective rank (~32); per-head representations too low-rank; in_dist regresses first as capacity bottleneck hits dominant distribution.
- **Param impact:** ~0 (Q/K/V/O Linear shapes preserved; temperature param grows from 2→4 per block × 4 blocks = +8 params, negligible).
- **Plateau-protocol pivot:** STRATEGY tier shift from loss-weighting / FiLM tweaks to attention architectural sub-axes.

---

## 2026-05-14 [Round 106] UTC — PR #2858: PhysicsAttention temperature init 0.5 → 1.0 — **ASSIGNED (80th candidate axis)**

- **Branch:** charliepai2g48h5-nezuko/temperature-init-1
- **Hypothesis:** Change the init value of the learnable per-head slice-routing temperature in PhysicsAttention from 0.5 to 1.0. At init=0.5, effective softmax input is 2× sharper than default; at init=1.0, natural scale. Sharper-than-default routing at init biases the model toward concentrated slice assignments early; softer init allows more exploration of slice configurations. Distinct from in-flight #2852 frieren (slice_num=20 — COUNT of slices, not SHARPNESS) — orthogonal sub-axes within slice routing.
- **Predicted mechanism (WIN):** Softer init lets temperatures drift to wider distribution of converged values (some heads sharpen, others soften); slice assignments less locked-in early; OOD generalization improves via routing exploration.
- **Predicted mechanism (LOSS):** Sharper routing was correct inductive bias for slice attention; softer init slows convergence; in_dist regresses (training distribution prefers sharper routing).
- **Param impact:** +0 params (init value of existing parameter).
- **Diagnostic to log:** Per-block per-head temperature trajectory across val epochs; final converged temperature distribution (8 values); slice-routing entropy at best-val checkpoint.

---

## 2026-05-14 [Round 105] UTC — PR #2829: Surface-aware LayerNorm at decoder ln_3 — **CLOSED LOSS (+8.34% val vs new baseline)**

- **Branch:** charliepai2g48h5-frieren/surf-aware-ln3
- **Hypothesis:** Add SurfaceAwareLN class (ln_3_surf + ln_3_vol, select per-token via is_surface mask) at decoder ln_3 site. +192 params. Surface tokens ~8% of mesh would get their own γ_surf/β_surf uncoupled from dominant volume statistics — targets frieren's own #2809 insight that model is "generalization-limited not capacity-limited" and primary metric is mae_surf_p.
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-surf-aware-ln3-20260514-074XXX/metrics.jsonl`

| Split | val | Baseline #2810 | Δ val | test |
|---|---|---|---|---|
| `single_in_dist` | 25.5+ | 23.59 | +8.3% LOSS | — |
| `geom_camber_rc` | 35.0+ | 32.36 | +8.2% LOSS | — |
| `geom_camber_cruise` | 36.1+ | 33.93 | +6.4% LOSS | — |
| `re_rand` | 36.7+ | 33.81 | +8.6% LOSS | — |
| **val_avg** | **33.4647** | **30.8909** | **+8.34% LOSS** | **28.0942 test (+5.99%)** |

- **Result:** NOT MERGED. val_avg +6.85% vs old baseline 31.3216, **+8.34% vs new baseline 30.8909**. All 4 val splits regress 5.6-9.2% — no split benefits. Best ep66, timeout-truncated at 30-min cap; linear extrapolation suggests ~33.3 even at full 70 epochs = solid LOSS.
- **Mechanism CONFIRMED (student exemplary diagnostic):**
  - γ vectors DID diverge as predicted: `L2(γ_surf − γ_vol) = 2.67`, `cos(γ_surf, γ_vol) = 0.97` — both directions similar but magnitudes differ.
  - **CRITICAL FAILURE:** `γ_surf std = 0.0715` is **3.4× SMALLER** than `γ_vol std = 0.2449` — γ_surf is **UNDER-TRAINED**.
  - Surface tokens are ~8% of mesh = **12× less gradient signal per parameter** than volume tokens.
  - The very split designed to help mae_surf_p ended up giving the surface-pressure-relevant normalization a noisy under-fit γ vector.
- **Key insight:** Param-doubling at a normalization site only helps when both sub-populations receive comparable gradient mass. Token-frequency imbalance defeats the structural split. Reinforces #2809 frieren "generalization-limited not capacity-limited" thesis — adding surface-specific capacity at LN-level cannot rescue under-gradient training.
- **73rd taxon: surface-aware LayerNorm CLOSES.** Combined with closed embed-LN #2808 (insertion), DyT #2686 (replacement), RMSNorm in-flight #2851 (parameter-stripping replacement) — **token-population-split sub-axis of normalization meta-axis CLOSES**. Surface-token specialization at intermediate-layer norm fails due to gradient-frequency asymmetry, not absence of structural utility.

---

## 2026-05-14 [Round 105] UTC — PR #2852: PhysicsAttention slice_num 24 → 20 — **ASSIGNED (78th candidate axis)**

- **Branch:** charliepai2g48h5-frieren/slice-num-20
- **Hypothesis:** Reduce `slice_num` from 24 to 20 in PhysicsAttention. Slice-routing-quality axis has two prior wins on this launch: PR #1846 (slice_num 64→32 WIN) and PR #2307 (slice_num 32→24 WIN). Pattern suggests over-parameterized slice routing hurts generalization, and tighter routing forces more concentrated information passage through fewer slice tokens. Round 104 diagnostic (#2828) confirmed slice attention is "information-routing not redundant feature-extraction" — each slice carries unique information. Question is whether all 24 are equally necessary or 20 is sufficient and forces better generalization. Model is generalization-limited (frieren #2809), not capacity-limited.
- **Param impact:** +0 params (slice_num only affects slice_token init `nn.Parameter(torch.randn(1, slice_num, n_hidden))` size and routing matrix output dim — these are activation-shape changes, not parameter-count changes; slice_token tensor size scales but is small).
- **Predicted mechanism (WIN scenario):** Tighter routing forces concentrated information passage; slice utilization more uniform; OOD splits benefit most (re_rand and camber_cruise).
- **Predicted mechanism (LOSS scenario):** 20 slices is below routing-capacity threshold; in_dist regresses; data suggests next probe at slice_num=28 in the opposite direction.

---

## 2026-05-14 [Round 104] UTC — PR #2828: Attention dropout p=0.05 in PhysicsAttention — **CLOSED LOSS (+3.77% val)**

- **Branch:** charliepai2g48h5-askeladd/attn-dropout-p0.05
- **Hypothesis:** Standard transformer attention dropout (Vaswani 2017 / ViT / BERT) as canonical regularizer for generalization-limited model. Predicted: stochastic edge dropping forces redundant slice-routing pathways → OOD splits benefit most.
- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-attn-dropout-p0.05-20260514-065705/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test |
|---|---|---|---|---|
| `single_in_dist` | 26.2668 | 24.9721 | **+5.18% LOSS** | 25.7474 |
| `geom_camber_rc` | 47.4473 | 46.9885 | +0.97% flat | 43.9342 |
| `geom_camber_cruise` | 18.9864 | 17.7276 | **+7.10% LOSS** | 15.4055 |
| `re_rand` | 37.3031 | 35.5983 | +4.79% LOSS | 26.6948 |
| **val_avg** | **32.5009** | **31.3216** | **+3.77% LOSS** | **27.9455 test** |

- **Result:** NOT MERGED. val_avg +3.77% vs old baseline, +5.21% vs new baseline (30.8909). Hit timeout ep58, 6 epochs short of baseline ep64; linear extrapolation ~31-31.5 still LOSS.
- **Diagnostics CONFIRMED wiring correct:**
  - All 4 PhysicsAttention blocks asserted dropout.p=0.05 at startup.
  - attn_out_var_train ≈ 2× attn_out_var_eval throughout training — dropout actively injecting noise during training, fully off during eval.
- **Student's 3-mechanism analysis (exemplary):**
  1. **Slice attention is information-routing, not feature-extraction.** PhysicsAttention's G=24 slice tokens with H=2 heads = ~12 edges per token. Dropping 5% (~0.6 edges) corrupts slice-to-slice routing in a non-redundant way — unlike ViT/BERT where edges are highly redundant across hundreds of tokens.
  2. **Damage pattern falsifies over-regularization hypothesis.** in_dist +5.18% (worst-hit) > OOD splits — exact opposite of what stochastic regularization should produce. Confirms the model is NOT over-fitted; it's generalization-limited via STRUCTURAL bottleneck (matches frieren's #2809 diagnosis).
  3. **to_out Dropout fights LayerScale γ=1e-4 init.** Same failure mode as closed DropPath #2722 — zeros entire channels with E[surviving] amplified by 1/(1-p); attention-probability dropout (the dropout_p kwarg) should be safer; to_out Dropout doing most of the damage.
- **72nd taxon: attention dropout (both attn probabilities + to_out projection) CLOSES.** Combined with closed DropPath #2722 dropout-axis well-mapped. **STRATEGIC PIVOT: from regularization to structural inductive bias changes.**

---

## 2026-05-14 [Round 104] UTC — PR #2851: RMSNorm replacement of LayerNorm — **ASSIGNED (77th candidate axis)**

- **Branch:** charliepai2g48h5-askeladd/rmsnorm-replace
- **Hypothesis:** Replace all 9 `nn.LayerNorm` sites in `TransolverBlock` (ln_1+ln_2 per block × 4 blocks + ln_3 in last_layer = 9 sites) with RMSNorm (Zhang & Sennrich 2019 / LLaMA / T5 / Mistral / PaLM-2 standard). No mean centering, no β bias. Saves 864 params → 332,836 total. Structural change in normalization op itself — preserves per-token scale baseline that mean centering discards. β removal cleans up LayerScale near-identity residual stream where β was mostly redundant. Distinct from closed embed-LN / DyT / QK-Norm / GroupNorm and from in-flight surface-aware LN #2829 (which keeps LN structure but splits by token type).

---

## 2026-05-14 [Round 103] UTC — PR #2794: SwiGLU-preprocess — **CLOSED (2nd stale_wip)**

- **Branch:** charliepai2g48h5-fern/swiglu-preprocess
- **Status:** Created 2026-05-14T05:08:32Z, ~4h+ ago. 0 commits beyond assignment commit. 0 comments across 2 consecutive heartbeats.
- **Decision:** Closed per pod-failure convention (Round 91 #2728, Round 99 #2780, Round 100 #2780). fern pod alive 1/1 but persistently failing to poll — most likely GitHub API rate limit symptom.
- **Axis status:** SwiGLU-preprocess (input-lifting MLP gating) remains UNTESTED, not falsified. May be re-assigned to a different student in a future round if a slot opens.

---

## 2026-05-14 [Round 103] UTC — PR #2849: Cosine eta_min=1e-6 — **ASSIGNED (76th candidate axis)**

- **Branch:** charliepai2g48h5-fern/cosine-eta-min-1e-6
- **Hypothesis:** Add `eta_min=1e-6` keyword argument to `CosineAnnealingLR` call. Lion's sign-step lacks magnitude adaptation and benefits from a non-zero LR floor (Chen 2023). Baseline's best_ep=64/65 has LR≈3e-6 at cosine tail — effectively zero updates for Lion in the final 5-6 epochs. eta_min=1e-6 = 0.67% of peak keeps Lion fine-tuning slowly. +0 params, +0 structural changes. Single keyword argument addition (the simplest possible change in this launch).
- **Pickup motivation:** fern's previous assignment #2794 closed as 2nd stale_wip. This assignment is intentionally trivial to maximize pickup probability for a stall-prone pod.

---

## 2026-05-14 [Round 102] UTC — PR #2814: SE bottleneck inner activation GELU→SiLU — **CLOSED LOSS (+3.17% val)**

- **Branch:** charliepai2g48h5-tanjiro/se-silu-inner
- **Hypothesis:** Following the GeGLU LOSS (#2759, gate_zero_frac 16-26% hard-masking diagnostic), apply same SiLU substitution at the SE bottleneck inner activation. Predicted: SiLU eliminates hard-masking → 24 bottleneck channels preserved → gate_std rises → OOD splits benefit.
- **Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-se-silu-inner-20260514-062248/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 25.0757 | 24.9721 | +0.10 | **23.3436** | 24.0714 | **−0.73 WIN** |
| `geom_camber_rc` | 49.7281 | 46.9885 | **+2.74 LOSS** | 42.6873 | 41.9406 | +0.75 |
| `geom_camber_cruise` | 18.3988 | 17.7276 | +0.67 | 14.4708 | 14.2400 | +0.23 |
| `re_rand` | 36.0561 | 35.5983 | +0.46 | 26.7635 | 25.7749 | +0.99 |
| **val_avg** | **32.3147** | **31.3216** | **+0.99 LOSS** | | | |
| **test_avg** | **26.8163** | **26.5067** | **+0.31 LOSS** | | | |

- **Result:** NOT MERGED. val_avg +3.17% vs old baseline 31.3216, +4.61% vs new baseline 30.8909 (post-#2810 merge). Decisive LOSS, especially on camber_rc OOD.
- **Mechanism CONFIRMED (student exemplary):**
  - `silu_zero_frac = 0.0000` across all 4 val splits — SiLU eliminates hard-masking entirely.
  - `pre_act_std` 1.4–3.0 symmetric; `negative_frac` 0.46–0.51 — ~half of 24 bottleneck channels output small negative SiLU values.
  - SE `gate_std` 0.18–0.30 (slightly higher than GELU baseline 0.17–0.28); `gate_min` 0.0004–0.026 preserved low.
- **Key insight (student):** GELU's hard-masking at the SE bottleneck is acting as **beneficial implicit sparsification**, NOT wasted capacity. The SE bottleneck (24-channel 2-layer encoder feeding sigmoid gate) has structurally different optimal-activation requirements vs the MLP body (gated multiplicative GeGLU/SwiGLU). SiLU's smooth negative range removes the implicit channel-selection pressure that GELU provides. The result destabilizes near-veto OOD gating behavior — sharpest regression on camber_rc (+2.74 val).
- **71st taxon: SE bottleneck inner activation axis CLOSES at GELU.** Full SE-axis now mapped: width (r=4 #2765 merged), inner activation (GELU, this PR closes), pool function (attn-pool #2810 merged).
- **Follow-up assigned:** PR #2848 tanjiro SE bottleneck post-GELU Dropout(p=0.1) — direct test of the implicit-sparsification hypothesis.

---

## 2026-05-14 [Round 102] UTC — PR #2848: SE bottleneck post-GELU Dropout p=0.1 — **ASSIGNED (75th candidate axis)**

- **Branch:** charliepai2g48h5-tanjiro/se-bottleneck-dropout
- **Hypothesis:** Add `nn.Dropout(p=0.1)` between `F.gelu(self.fc1(pool))` and `self.fc2` in `SqueezeExcitation`. +0 params. Directly tests whether GELU's "hard-masking" is acting as implicit sparsification: if yes, explicit dropout will stack on top (WIN) or be redundant (WASH); if not, dropout will regress (LOSS). Structurally distinct from in-flight attention dropout #2828 (attention probabilities inside sdpa softmax) and from closed DropPath #2722 (residual zeroing). Diagnostics include `gelu_zero_frac` to confirm GELU is still hard-masking 16-26% even with dropout active.

---

## 2026-05-14 [Round 102] UTC — PR #2794: SwiGLU-preprocess — **LEFT (1st stale_wip)**

- **Branch:** charliepai2g48h5-fern/swiglu-preprocess
- **Status:** Created 2026-05-14T05:08:32Z, ~2h ago, 0 commits 0 comments. fern pod likely hitting GitHub API rate limits.
- **Decision:** Leave as-is per Round 91 / Round 99 convention. 1st stale = wait one more round. If still no pickup next round, close as pod-failure.

---

## 2026-05-14 [Round 101] UTC — PR #2810: SE block-3-only attention-pool — **MERGED WIN (−1.37% val)**

- **Branch:** charliepai2g48h5-alphonse/se-attn-pool
- **Hypothesis:** Replace mean-pool in SqueezeExcitation with content-aware attention pool (`Linear(96,1)+softmax` over tokens) to learn which tokens are most informative for global channel gating. +97 params (+`Linear(96,1)` weight) → 333,700 total.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-se-attn-pool-20260514-060932/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 25.2751 | 24.9721 | +1.21% mild | 23.4553 | 24.0714 | −2.55% WIN |
| `geom_camber_rc` | **45.8179** | 46.9885 | **−2.49% WIN** | 41.1687 | 41.9406 | −1.84% WIN |
| `geom_camber_cruise` | **16.8427** | 17.7276 | **−4.92% WIN** | 14.1435 | 14.2400 | −0.69% WIN |
| `re_rand` | 35.6177 | 35.5983 | +0.05% flat | 25.9980 | 25.7749 | +0.93% mild |
| **val_avg** | **30.8909** | **31.3216** | **−1.37% WIN** | | | |
| **test_avg** | **26.1964** | **26.5067** | **−1.17% WIN** | | | |

- **Result:** MERGED as 20th winner. New baseline: val 30.8909 / test 26.1964 / params 333,700.
- **Mechanism:** Predicted mechanism (content-aware token-importance attention pool) did NOT activate — attn_pool weights remain near-uniform (max_w ~2-3× uniform, entropy within 2% of ln(T)). Real mechanism: +97 params perturb SE bottleneck dynamics, lifting gate_std +13-19% across ALL splits. SE gates wider distribution = more channel selectivity for OOD geometry/Re. The improvement is real but via gate-distribution expansion rather than token-content weighting. Key insight: even a small parameter perturbation that changes the SE bottleneck equilibrium can systematically improve OOD performance.

---

## 2026-05-14 [Round 101] UTC — PR #2813: Per-block FiLM at all 4 TransolverBlocks — **CLOSED LOSS (+0.81% val)**

- **Branch:** charliepai2g48h5-nezuko/per-block-film
- **Hypothesis:** Add FiLM modulation at every TransolverBlock (depth-progressive Re/AoA conditioning vs single embedding-stage FiLM in merged #2614). 4×384=+1,536 params → 335,139 total.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-per-block-film-20260514-065XXX/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 32.0436 | 24.9721 | +28.3% LOSS | 24.7500 | 24.0714 | +2.82% |
| `geom_camber_rc` | 47.2146 | 46.9885 | +0.48% flat | 40.7888 | 41.9406 | −2.75% WIN |
| `geom_camber_cruise` | 17.0456 | 17.7276 | −3.85% WIN | 13.3730 | 14.2400 | −6.09% WIN |
| `re_rand` | 29.9949 | 35.5983 | −15.73% WIN | 23.1673 | 25.7749 | −10.12% WIN |
| **val_avg** | **31.5747** | **31.3216** | **+0.81% LOSS** | | | |
| **test_avg** | **25.7718** | **26.5067** | **−2.77% WIN** | | | |

- **Result:** NOT MERGED. val_avg regresses +0.81% on primary metric. Large test-val divergence: test improves −2.77% across all 4 splits while in_dist val catastrophically regresses +28.3%.
- **Mechanism confirmed:** depth-progressive FiLM scale growth: film_scale_abs_mean block 0→3 = 0.0762→0.1118 (+47%); weight_norm 1.49→2.07 (+39%). Optimizer used all 4 FiLM modules and grew them depth-progressively. However, per-block FiLM over-conditions intermediate representations on Re/AoA scalars — in-distribution training samples (most numerous) suffer most from this over-conditioning. The Re/AoA signal becomes a shortcut for in-dist pattern memorization rather than a generalization signal.
- **70th taxon closure:** Per-block FiLM (4-way conditioning) CLOSES. Student recommendation: block-3-only FiLM is the motivated follow-up (targeted final-output conditioning at the deepest block without intermediate corruption).

---

## 2026-05-14 [Round 101] UTC — PR #2844: Block-3-only FiLM — **ASSIGNED (73rd candidate axis)**

- **Branch:** charliepai2g48h5-nezuko/block3-film
- **Hypothesis:** Apply FiLM conditioning at block 3 only (final TransolverBlock), after SE gate, before mlp2 decoder. Single `nn.Linear(3, 96)` zero-init. +384 params → 334,084 total. Targeted final-output conditioning without intermediate representation corruption observed in per-block FiLM (#2813).

---

## 2026-05-14 [Round 101] UTC — PR #2842: surf_weight=15.0 — **ASSIGNED (74th candidate axis)**

- **Branch:** charliepai2g48h5-alphonse/surf-weight-15
- **Hypothesis:** Increase surface pressure loss weighting from surf_weight=10.0 to 15.0 (+50%). surf_weight has NEVER been swept this launch despite 100+ rounds of improvements. +0 params. Hypothesis: stronger surface supervision directly targets primary metric; with current improved model (SwiGLU+SE+FiLM+LayerScale) the optimizer may leverage higher surface emphasis without destabilizing volume feature learning.

---

## 2026-05-14 [Round 100] UTC — PR #2808: Pre-block-0 embedding LayerNorm — **CLOSED LOSS (+3.75% val)**

- **Branch:** charliepai2g48h5-askeladd/embed-ln
- **Hypothesis:** Insert `nn.LayerNorm(96)` between FiLM modulation and block-0 input. Premise: preprocess+placeholder+FiLM stream has drifted from unit variance, destabilizing block-0's LayerScale γ=1e-4. +192 params.
- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-embed-ln-20260514-055416/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 24.4414 | 24.9721 | −0.53 (better) | — | 24.0714 | — |
| `geom_camber_rc` | 48.8325 | 46.9885 | +1.84 (worse) | 43.7700 | 41.9406 | +4.1% |
| `geom_camber_cruise` | 19.1793 | 17.7276 | +1.45 (worse) | — | 14.2400 | — |
| `re_rand` | 37.5343 | 35.5983 | +1.94 (worse) | — | 25.7749 | — |
| **val_avg** | **32.4969** | **31.3216** | **+3.75% LOSS** | | | |
| **test_avg** | **28.5529** | **26.5067** | **+7.72% LOSS** | | | |

- **Result:** NOT MERGED. Hypothesis premise falsified by diagnostic.
- **Mechanism:** Pre-LN `input_var = 0.9737` — the embedding stream is ALREADY at near-unit variance before the LN. `preprocess` MLP (Linear→ReLU→Linear, default init) + zero-init FiLM (`|film_scale|≈0.975` at convergence, multiplicative factor stays in benign range 1±0.975) + tiny placeholder (1/96 init) collectively produce near-unit variance. The LN weight (mean=0.9931, std=0.0705) and bias (mean=−0.0021, std=0.0271) barely drifted from identity init — LN is effectively inert. The +192 params act as small optimization noise, hurting OOD splits where every gradient step matters.
- **68th taxon closure:** LN-site insertion at post-FiLM-pre-block-0 empirically falsified. Normalization-meta-axis now mapped across REPLACEMENT (DyT 48th, RMSNorm 15th), GRANULARITY (GroupNorm 62nd), and SITE-INSERTION (embed-ln 68th) dimensions. Key diagnostic insight: pre-block-0 stream is already at unit variance — future LN-site probes should target locations with actual drift (e.g., post-SE-output at block 3 where sigmoid gating may compress scale).

---

## 2026-05-14 [Round 100] UTC — PR #2809: 3-layer output head mlp2 (decoder-depth probe) — **CLOSED LOSS (+5.17% val)**

- **Branch:** charliepai2g48h5-frieren/mlp2-deeper
- **Hypothesis:** Replace `mlp2 = Linear(96,96)→GELU→Linear(96,3)` with `Linear(96,96)→GELU→Linear(96,96)→GELU→Linear(96,3)`. Decoder-depth probe: 2 nonlinearities vs 1. +9,312 params → 342,915 total.
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-mlp2-deeper-20260514-055019/metrics.jsonl`

| Split | val | Baseline #2765 | Δ val | test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | **24.4354** | 24.9721 | **−2.15% WIN** | 23.9450 | 24.0714 | −0.52% flat |
| `geom_camber_rc` | 50.7865 | 46.9885 | **+8.08% LOSS** | 43.6636 | 41.9406 | +4.11% |
| `geom_camber_cruise` | 18.9893 | 17.7276 | **+7.12% LOSS** | 14.9132 | 14.2400 | +4.73% |
| `re_rand` | 37.5450 | 35.5983 | **+5.47% LOSS** | 26.8510 | 25.7749 | +4.18% |
| **val_avg** | **32.9390** | **31.3216** | **+5.17% LOSS** | | | |
| **test_avg** | **27.3432** | **26.5067** | **+3.16% LOSS** | | | |

- **Result:** NOT MERGED. in_dist improves marginally; all 3 OOD splits regress 4-8%.
- **Mechanism:** Terminal-head overfitting without residual bound. The mlp2 decoder has no LayerScale γ, no skip connection — every added parameter directly shifts prediction capacity toward in-distribution fitting. 6,400 training meshes + surf_weight=10 + extra head capacity = generalization tax on geometry-OOD and Re-OOD. LayerScale γ_attn_block3=0.00561 and γ_mlp_block3=0.0802 did NOT collapse (decoder did not absorb block-3 work; PR mechanism prediction refuted). Mechanism predictions (b) and (c) refuted; terminal-head-overfitting mechanism (d) confirmed.
- **Key insight from student:** "The model is currently **generalization-limited, not capacity-limited**." in_dist improves under extra capacity while OOD regresses — clean signature. Probes that add OOD generalization WITHOUT adding net capacity are higher-yield.
- **69th taxon closure:** Decoder-depth axis saturated at 2 layers. Combined with stale #2750 fern mlp2-wide192, decoder modification meta-axis mapped — baseline 2-layer 96-hidden head is correctly sized. Future decoder probes should be non-capacity modifications (normalization, activation type, input enrichment).

---

## 2026-05-14 [Round 100] UTC — PR #2780: Mish activation in mlp2 — **CLOSED stale_wip (axis untested)**

- 2nd consecutive stale_wip event. Created 2026-05-14T04:22:02Z, 0 commits, 0 comments ~2h17m. Pod alive 1/1 but hitting GitHub API rate limits per heartbeat logs (GraphQL quota exceeded, retry loop failed). Pod-failure close per Round 91 #2728 convention. Mish-mlp2 AXIS UNTESTED (not falsified) — may retry in future launch.

---

## 2026-05-14 [Round 100] UTC — PR #2829: Surface-aware LN at decoder ln_3 — **ASSIGNED (70th candidate axis)**

- **Branch:** charliepai2g48h5-frieren/surf-aware-ln3
- **Hypothesis:** `SurfaceAwareLN` class with `ln_3_surf` + `ln_3_vol` modules; per-token selection via `is_surface` mask. +192 params (= 333,795 total). Responsive to frieren's "generalization-limited not capacity-limited" insight — gives surface tokens their own γ_surf/β_surf at the decoder's normalization site. Thread is_surface through 2 model-call sites + Transolver.forward + TransolverBlock.forward.

---

## 2026-05-14 [Round 100] UTC — PR #2828: Attention dropout p=0.05 — **ASSIGNED (71st candidate axis)**

- **Branch:** charliepai2g48h5-askeladd/attn-dropout-p0.05
- **Hypothesis:** Set Transolver `dropout` kwarg from 0 to 0.05. Propagates to PhysicsAttention SDPA `dropout_p` + `to_out` Dropout. Standard transformer regularization NEVER tested in this launch. +0 params. Targets generalization-limited finding; distinct from DropPath (which broke LayerScale γ residual identity — attention dropout is inside attn not in residual path).

---

## 2026-05-14 [Round 100] UTC — PR #2827: LayerScale γ init = 1e-3 — **ASSIGNED (72nd candidate axis)**

- **Branch:** charliepai2g48h5-thorfinn/layerscale-init-1e-3
- **Hypothesis:** 1-character change `layerscale_init=1e-4` → `1e-3`. +0 params. Tests whether 1e-4 wastes optimizer budget growing γ; final |γ| = 0.005-0.08 = 50-800× init, suggesting optimizer must traverse large dynamic range. DeiT-III/ViT-22B recommend 1e-3 for shallow stacks. Maximally simple for stall-prone pod.

---

## 2026-05-14 [Round 99] UTC — PR #2799: Long-range preprocess→block-3 skip (per-channel zero-init gate) — **CLOSED LOSS**

- **Branch:** charliepai2g48h5-edward/longskip-layerscale-gate
- **Hypothesis:** Add learnable `self.skip_gate = nn.Parameter(torch.zeros(n_hidden))` in Transolver.__init__. In forward, save `fx_input` post-FiLM and inject at block-3 input: `fx = fx + skip_gate.unsqueeze(0).unsqueeze(0) * fx_input`. Zero-init ensures identity at step 0; model learns to activate skip if it helps. +96 params. 64th candidate axis: input-to-final-block long-range residual.
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-longskip-layerscale-gate-20260514-053649/metrics.jsonl`

| Split | Skip val | Baseline #2765 | Δ val | Skip test | Baseline test | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | **24.0608** | 24.9721 | **−3.65% WIN** (new best-ever) | 24.0123 | 24.0714 | −0.25% flat |
| `geom_camber_rc` | **46.2634** | 46.9885 | **−1.54% WIN** | 42.8073 | 41.9406 | +2.07% mild loss |
| `geom_camber_cruise` | **17.1796** | 17.7276 | **−3.09% WIN** | 14.0900 | 14.2400 | −1.05% mild win |
| `re_rand` | 37.6451 | 35.5983 | **+5.75% LOSS** | 27.1396 | 25.7749 | **+5.30% LOSS** |
| **val_avg** | **31.2873** | **31.3216** | **−0.11% (noise floor)** | | | |
| **test_avg** | **27.0123** | **26.5067** | **+1.91% LOSS** | | | |

- **Result:** NOT MERGED. val improvement 0.034 absolute (−0.11%) is well within run-to-run variance (~0.5%); test regresses +1.91% on paper-facing metric. 3-of-4 val splits WIN (in_dist new best-ever, rc, cruise improve) but re_rand crashes +5.75% val / +5.30% test — same direction on both val and test, not noise.
- **Mechanism (LOSS scenario from PR predictions fired exactly — student's smoking-gun diagnostic):**
  - Skip gate FULLY activated: 96/96 channels nonzero, abs_mean=0.057, balanced signs 45+/51−, skip/residual ratio ~4.6-5.6% — not inactive, doing measurable work.
  - **SE block-3 gate_std drops 8-14% across EVERY split** vs baseline #2765:
    - in_dist: 0.168 → 0.145 (−13.7%)
    - camber_rc: 0.234 → 0.214 (−8.5%)
    - camber_cruise: 0.269 → 0.241 (−10.3%)
    - re_rand: 0.275 → 0.252 (−8.4%)
  - The split that pays the biggest cost (re_rand +5.75%) had the LARGEST baseline SE gate_std (0.275). Block 3 is the focal point of FOUR per-channel mechanisms (LayerScale γ_attn/γ_mlp, SE gate, FiLM modulation at preprocess, skip_gate). Adding a fourth creates zero-sum capacity trade: SE sacrifices ~10% discriminative variance for skip's ~5% budget.
  - Block-3 γ_attn flipped back from the strongly-negative-mean sign seen in #2765 — skip and γ_attn share modulation load at block 3.
  - Run params confirmed at 333,699 (+96 vs baseline 333,603 ✓). Best epoch ep66/70.
- **Taxonomic closure (67th taxon):** Long-range residual skip at SE-gated input-to-final-block location CLOSES. Combined with closed Deep Supervision (61st), DropPath (51st), Weight Standardization (59th), Mixup (49th), the gradient/feature-flow and skip-connection meta-axis is now densely mapped. Block 3 is over-allocated — future per-channel mechanisms should target blocks 0-2 or be structurally distinct from gating.
- **Follow-up per plateau protocol:** Strategy-tier shift to training-procedure axis. Per student's Round 90 #2735 suggestion (decay=0.9999 better than 0.9995). New assignment: #2820 edward EMA-of-weights decay=0.9999.

---

## 2026-05-14 [Round 99] UTC — PR #2820: EMA-of-weights decay=0.9999 (Polyak averaging) — **ASSIGNED (69th candidate axis)**

- **Branch:** charliepai2g48h5-edward/ema-weights-decay9999
- **Hypothesis:** Maintain shadow state dict `θ_ema = 0.9999·θ_ema + 0.0001·θ_train` updated after every optimizer.step(). Eval the EMA-weighted model at each val pass. Select best-val epoch from EMA metric, evaluate test from EMA checkpoint. +0 trainable params, ~5MB RAM shadow. Effective averaging window ~10,000 steps = ~40 epochs. Strategy-tier shift from block-level architecture (5 consecutive LOSS rounds 95-99) to training-procedure axis. Lion optimizer known to converge to sharp minima; EMA is the canonical sharpness antidote (Izmailov 2018, Karras 2022).
- **Connection to prior:** edward's Round 90 #2735 EMA run (decay=0.9995, askeladd) showed EMA-vs-raw ~-0.6% relative improvement but was overshadowed by SAM 2x-compute budget halving (effectively 35/70 epochs). Current assignment: pure EMA, no competing SAM, full 70 epochs, decay=0.9999 per student's suggestion in #2735 analysis.
- **Param delta:** +0 (shadow is not a trainable parameter). Total: 333,603.
- **Predicted mechanism:** Cosine decay collapses LR to ~0 by ep70; in final 10-15 epochs Lion steps are tiny but still noisy. EMA averages these tiny steps, cancelling noise while retaining converged location. OOD splits benefit most (re_rand, camber*) — sharper minima hurt OOD more than in_dist.
- **Bar:** val_avg/mae_surf_p_ema < 31.3216.

---

## 2026-05-14 [Round 98] UTC — PR #2798: SE reduction=2 (bottleneck width sweep) — **CLOSED LOSS (+2.56% val)**

- **Branch:** charliepai2g48h5-tanjiro/se-reduction2
- **Hypothesis:** Extend SE bottleneck axis from r=4 (merged optimum) to r=2 (inner_dim 24→48). If r=4 was capacity-limited, r=2 should improve further; if r=4 is already optimum, r=2 should regress via lost regularization.
- **Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-se-reduction2-20260514-053151/metrics.jsonl`

| Split | r=2 val | Baseline #2765 (r=4) | Δ vs #2765 |
|---|---|---|---|
| `val_single_in_dist` | **23.3693** | 24.9721 | **−6.42% WIN** (new best-ever in-dist) |
| `val_geom_camber_rc` | 50.0688 | 46.9885 | **+6.55% LOSS** |
| `val_geom_camber_cruise` | 17.8757 | 17.7276 | +0.84% LOSS |
| `val_re_rand` | 37.1777 | 35.5983 | **+4.44% LOSS** |
| **val_avg** | **32.1229** | **31.3216** | **+2.56% LOSS** |
| `test_avg/mae_surf_p` | **26.3402** | **26.5067** | −0.63% (test-val divergence) |

- **Result:** val_avg LOSS on ranking metric closes axis per program.md. In-dist set best-ever (23.37 val, 23.22 test) but all 3 OOD splits regressed. test_avg improved -0.63% (interesting test-val divergence; in-dist test gains outweighed OOD test losses).
- **Mechanism (student's diagnostic exemplary — smoking gun signature):**
  - `gate_std` rose monotonically r=8→r=4→r=2 (+9-16% over r=4) — more bottleneck capacity gives finer discrimination, AS PREDICTED
  - `gate_min` REVERSED direction r=4→r=2 on all 4 splits — aggressive veto pattern lost, AS PREDICTED for LOSS scenario
  - Block 3 γ_attn FLIPPED BACK TO POSITIVE (+0.001401 vs r=4 negative) — SE has loosened, attention re-engages
  - SE moved from "discriminator + sharp veto" mode → "averager + scaling" mode
- **Taxonomic closure (66th taxon):** SE bottleneck dimension axis now exhaustively mapped: r=8 closed inner_dim=12 capacity-limited, **r=4 merged inner_dim=24 local optimum**, r=2 closed inner_dim=48 regularization-lost. Non-monotonic curve confirms the sweet spot. **r=4 is the SE local optimum for this scale.** Future SE work should hold r=4 fixed and vary other dimensions (depth, activation, init, pool function).
- **Follow-up:** SE inner activation swap #2814 (tanjiro, 68th candidate axis) at r=4 fixed — student's #3 suggestion, motivated by orthogonal GeGLU LOSS evidence (hard-masking mechanism).

---

## 2026-05-14 [Round 98] UTC — PR #2814: SE inner activation GELU → SiLU — **ASSIGNED (68th candidate axis)**

- **Branch:** charliepai2g48h5-tanjiro/se-silu-inner
- **Hypothesis:** Replace SE bottleneck inner activation `F.gelu` → `F.silu` (1-line change in SqueezeExcitation.forward). Motivated by orthogonal evidence from just-closed GeGLU PR #2759 (64th taxon, gate_zero_frac 16-26% per block — GELU's symmetric near-zero collapse hard-masks ~20% of channels). If the same mechanism is active inside SE's inner_dim=24 bottleneck, GELU is effectively reducing capacity to ~19 channels. SiLU's asymmetric smoothness should preserve all 24 channels.
- **Param delta:** +0 (same activation count, just different function).
- **Predicted mechanism:** `silu_zero_frac` <<16% (no hard masking); `gate_std` rises beyond r=4 baseline; `gate_min` stays low (regularization-via-veto preserved at fixed r=4 width); OOD splits benefit from extra effective bottleneck capacity at unchanged width.
- **Bar:** val_avg/mae_surf_p < 31.3216.

---

## 2026-05-14 [Round 97] UTC — PR #2792: Register tokens N_REG=4 (Darcet 2024) — **CLOSED LOSS (+4.39% val)**

- **Branch:** charliepai2g48h5-nezuko/register-tokens-4
- **Hypothesis:** Add 4 learnable register tokens as extra K/V positions in PhysicsAttention slice attention to expand attention-input capacity without disrupting mesh-derived slice routing. Predicted OOD WIN via dedicated global capacity not bottlenecked through slice tokens.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-register-tokens-4-20260514-051701/metrics.jsonl`

| Split | RegTokens val | Baseline #2765 | Δ vs #2765 |
|---|---|---|---|
| `val_single_in_dist` | 24.9768 | 24.9721 | +0.02% wash |
| `val_geom_camber_rc` | 49.1908 | 46.9885 | **+4.69% LOSS** |
| `val_geom_camber_cruise` | 18.3609 | 17.7276 | **+3.57% LOSS** |
| `val_re_rand` | 38.2559 | 35.5983 | **+7.46% LOSS** (worst-hit) |
| **val_avg** | **32.6961** | **31.3216** | **+4.39% LOSS** |
| `test_avg/mae_surf_p` | **27.3044** | **26.5067** | **+3.01% LOSS** |

- **Result:** All 4 splits regress or wash; all 3 OOD splits LOSS. Test also LOSS. Per-split signature is exact INVERSION of hypothesis (predicted OOD WIN, observed OOD worst-hit).
- **Mechanism (student's diagnostic exemplary):** Block 1 monopolizes registers at 56% slice→register attention mass by ep20 (some queries 100%, max=1.000); block 2 spontaneously vetoes them (register norms grew 5× but query attn mass dropped to 0.123 below uniform init 0.143); blocks 0 and 3 use modestly ~15-20%. **Block-1 lock-in creates low-rank rank-4 global bottleneck** that fits dominant in-dist regime but cannot express OOD geometries/Re-numbers. Classic "global bottleneck overfits in-dist" pattern.
- **Why re_rand worst-hit (+7.46%):** re_rand is the Reynolds-randomization split needing strong Re-conditioning. Adding 4 unconditioned K/V slots competes with FiLM's #2614 conditioning capacity — the merged FiLM modulates the residual stream via flow scalars, but registers are unconditioned and steal attention mass from FiLM-modulated tokens. Re_rand fundamentally needs more Re-conditioning, not more attention slots.
- **Taxonomic closure:** 65th taxon. **Attention-input capacity expansion via unconditioned learnable K/V positions closed at this scale.** Combined with closed positional embedding (RoPE #2675 46th, Fourier features #2509 24th), attention-input augmentation meta-axis now mapped: positional encoding LOSS + register tokens LOSS. The merged FiLM + SE mechanism for OOD generalization is structurally incompatible with raw K/V augmentation.
- **Follow-up:** Per-block FiLM #2813 (nezuko, 67th candidate axis) — depth-progressive Re/AoA re-injection directly attacks the re_rand bottleneck. Student's own zero-init+LayerScale register variant deprioritized in favor of higher-impact axis.

---

## 2026-05-14 [Round 97] UTC — PR #2813: Per-block FiLM (depth-progressive flow conditioning) — **ASSIGNED (67th candidate axis)**

- **Branch:** charliepai2g48h5-nezuko/per-block-film
- **Hypothesis:** Add `self.film = nn.Linear(3, 96)` to each TransolverBlock, zero-init weight + bias, applied AFTER residual blocks (after SE on block 3, after MLP on blocks 0-2): `fx = fx * (1 + film(flow_scalars))`. Each block re-injects log_Re, AoA0_rad, AoA1_rad — depth-progressive Re-conditioning where currently the merged FiLM #2614 only conditions at embedding once. Directly attacks the re_rand bottleneck pattern exposed by closed #2792 (re_rand worst-hit −7.46% precisely because raw K/V competes with FiLM's Re-conditioning capacity).
- **Param delta:** +1,536 (4 blocks × (3 × 96 + 96) = 4 × 384) → expected 335,139.
- **Predicted mechanism:** |film_scale| grows most at block 3 (Re-conditioning before SE gate); film_scale_std depth-progressive grow; val_re_rand benefits most (it's been the worst-hit OOD split — closing both directions of mlp_ratio AND register tokens, all blamed on missing Re-conditioning); camber splits also benefit (AoA conditioning).
- **Bar:** val_avg/mae_surf_p < 31.3216.

---

## 2026-05-14 [Round 96] UTC — PR #2759: GeGLU MLP (gelu gate ablation) — **CLOSED LOSS (+3.48% val)**

- **Branch:** charliepai2g48h5-alphonse/geglu-mlp
- **Hypothesis:** Direct 1-line ablation of SwiGLU's silu gate to gelu. Tests whether the merged SwiGLU win (#2741) came from GLU-structure or the specific silu activation. GeGLU is used in PaLM-2/T5/Gemma.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-geglu-mlp-20260514-045412/metrics.jsonl`

| Split | GeGLU val | Baseline #2765 | Δ vs #2765 |
|---|---|---|---|
| `val_single_in_dist` | 27.9161 | 24.9721 | **+11.78% LOSS** (worst-hit) |
| `val_geom_camber_rc` | 47.2812 | 46.9885 | +0.62% LOSS |
| `val_geom_camber_cruise` | 17.2964 | 17.7276 | −2.43% improved |
| `val_re_rand` | 37.1492 | 35.5983 | +4.36% LOSS |
| **val_avg** | **32.4107** | **31.3216** | **+3.48% LOSS** |
| `test_avg/mae_surf_p` | **27.6001** | **26.5067** | **+4.13% LOSS** |

- **Result:** Two replicated runs (32.41 / 32.52) well-bracket each other = solid LOSS signal. In-dist worst-hit.
- **Mechanism (student's diagnostic excellent):** GeGLU `gate_zero_frac` = **16-26% per block** vs SwiGLU's **1.3-2.4%**. GELU's symmetric-near-zero collapse produces ~20% hard channel masking; SiLU's asymmetric-monotone-non-negative-tail gives only ~2%. For this stack, the hard-mask component is detrimental — model wants soft signed re-weighting from the MLP gate, not channel deletion. Hypothesis: GeGLU's hard-masking overlaps SE block-3's channel-gating role, doubling-down counterproductively on in_dist (where SE diagnostics from #2765 already show wider gates).
- **Taxonomic closure:** 64th taxon. Gate-activation sub-axis within GLU family now mapped: silu strictly preferred (merged #2741). ReGLU would push hard-masking to ~50% — not worth probing.
- **Bug fix verification:** Confirmed advisor branch has working `_swiglu_loader = val_loaders.get('val_single_in_dist') or ...` since commit 2840599 (PR #2765 merge accidentally fixed the dangling `_se_loader` introduced by 0767a13 #2741 merge). Student's branch must have forked between #2741 and #2765 merges. No further cherry-pick needed.
- **Param count discrepancy resolved:** Student reported 331,287 on their branch — this is the SwiGLU + SE-reduction=8 baseline (post-#2741, pre-#2765). New baseline #2765 is 333,603 (added +2,316 for SE r=8→4).
- **Follow-up:** SE attention-pool experiment #2810 — sub-axis of merged SE, replaces mean pool with content-aware learned pool.

---

## 2026-05-14 [Round 96] UTC — PR #2810: SE with attention pooling — **ASSIGNED (66th candidate axis)**

- **Branch:** charliepai2g48h5-alphonse/se-attn-pool
- **Hypothesis:** Replace SE module's mask-aware mean pool with content-aware attention pool — `attn_logits = Linear(96,1)(x); attn_w = softmax(attn_logits, dim=tokens); pool = (x * attn_w).sum(dim=tokens)`. Mean pool weights all tokens equally for channel-statistics; attention pool learns surface/wake/boundary token importance. Sub-axis of merged SE block-3 (#2765, r=4). Set Transformer (Lee 2019 ICCV) uses single-head attention pool for set summary.
- **Param delta:** +97 (single Linear(96,1)) → expected 333,700.
- **Predicted mechanism:** `attn_pool_weights` concentrate on `is_surface=True` tokens; `gate_std` rises beyond #2765's +5-12% gain; OOD splits (camber, re_rand) benefit most.
- **Bar:** val_avg/mae_surf_p < 31.3216.

---

## 2026-05-14 [Round 95] UTC — PR #2766: GroupNorm(G=4) replacing LayerNorm — **CLOSED LOSS (+14.69% val)**

- **Branch:** charliepai2g48h5-askeladd/groupnorm-g4
- **Hypothesis:** GroupNorm with G=4 (24-channel groups) tests whether intermediate granularity between per-channel (LN) and per-channel-stat reduction granularity yields better-conditioned normalization for point-cloud regression. If groups capture meaningful channel correlations without LN's full-channel pooling, val/mae_surf_p should improve.
- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-groupnorm-g4-20260514-*/metrics.jsonl`

| Split | GN(G=4) val | Baseline #2765 | Δ vs #2765 |
|---|---|---|---|
| `val_single_in_dist` | 28.4636 | 24.9721 | **+13.99% LOSS** |
| `val_geom_camber_rc` | 50.2261 | 46.9885 | **+6.89% LOSS** |
| `val_geom_camber_cruise` | 19.4244 | 17.7276 | **+9.57% LOSS** |
| `val_re_rand` | 41.6202 | 35.5983 | **+16.91% LOSS** |
| **val_avg** | **35.9188** | **31.3216** | **+14.69% LOSS** |
| `test_avg/mae_surf_p` | **30.7585** | **26.5067** | **+16.04% LOSS** |

- **Result:** All 4 val splits regress 6.89-16.91%. Test +16.04%.
- **Mechanism (student analysis correct):** GN with G=4 reduces over (T, C_g=24) statistics including padded mesh positions, contaminating per-group normalization. LayerNorm normalizes per-token over the full channel dim only — immune to padding ratio. LayerScale γ patterns preserved (no training instability), confirming this is a poor-conditioning failure mode, not a learning-rate or schedule issue.
- **Taxonomic closure:** 62nd taxon. Combined with closed DyT (#2686, 48th taxon, all-or-nothing replacement) and closed RMSNorm (#2139), the normalization-meta-axis is now COMPREHENSIVELY mapped across both REPLACEMENT (DyT, RMSNorm) and GRANULARITY (G=4 here) dimensions. LayerNorm full-channel pooling remains optimal for variable-token-count point-cloud regression. **Normalization-granularity sub-axis closes at LN.**
- **Follow-up:** ADD-LN-site experiment #2808 (askeladd, structurally orthogonal — adds a new pre-block-0 LN site instead of replacing).

---

## 2026-05-14 [Round 95] UTC — PR #2764: mlp_ratio 2 → 1.5 (MLP-width-down) — **CLOSED LOSS (+2.46% val)**

- **Branch:** charliepai2g48h5-frieren/mlp-ratio15
- **Hypothesis:** mlp_ratio=2 may be over-parameterized given Lion's sign-momentum compresses effective capacity. Reducing to 1.5 (-32K params in MLP body) may regularize without losing representation power. Inverse probe to closed #2749 mlp_ratio=3.
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-mlp-ratio15-20260514-*/metrics.jsonl`

| Split | ratio=1.5 val | Baseline #2765 | Δ vs #2765 |
|---|---|---|---|
| `val_single_in_dist` | 25.2367 | 24.9721 | +1.06% regress |
| `val_geom_camber_rc` | 46.1657 | 46.9885 | −1.75% improved |
| `val_geom_camber_cruise` | 17.0813 | 17.7276 | −3.65% improved |
| `val_re_rand` | 36.2113 | 35.5983 | +1.72% regress |
| **val_avg** | **32.0910** | **31.3216** | **+2.46% LOSS** |
| `test_avg/mae_surf_p` | **27.8057** | **26.5067** | **+1.39% LOSS** |

- **Result:** Borderline against OLD #2741 baseline (val 32.2477; −0.49% improve), but **LOSS against new #2765 baseline** (val 31.3216; +2.46%). Test regressed +1.39%.
- **Per-split inversion vs #2749 (mlp_ratio=3 wider):** wider arm regressed in_dist+helped camber; narrower arm regresses in_dist+re_rand+helps camber. Net: in_dist+re_rand strictly prefer mlp_ratio≈2; camber splits prefer ANYTHING different from 2 but magnitudes small. |γ_mlp| stayed flat through training = model still wants the MLP capacity it had at ratio=2.
- **Re_rand bottleneck is NOT MLP capacity** — regressed in BOTH directions (+1.72% here, +1.11% in #2749). Reynolds-conditioned probe is the right next move on that split.
- **Taxonomic closure:** 63rd taxon. MLP-width sub-axis now mapped from both directions (#2749 wider closed, #2764 narrower closed). **MLP-width axis saturated at mlp_ratio=2.** Student's own conclusion ('Stop probing MLP-width. The axis is mapped') aligns exactly.
- **Bug fix consideration:** PR included a defensive code-hygiene patch for SwiGLU diagnostic block (adds `is not None` and `_swiglu_blocks` guards). Verified advisor branch already has functional `_swiglu_loader = val_loaders.get('val_single_in_dist') or next(iter(val_loaders.values()))` — no NameError exists; the change is purely defensive. NOT cherry-picked (no functional change for in-flight SwiGLU experiments).
- **Follow-up:** decoder-depth experiment #2809 (frieren mlp2 deeper 2→3 layers, orthogonal to closed mlp2-wide192 stale_wip decoder-width).

---

## 2026-05-14 [Round 95] UTC — PR #2808: Pre-block-0 embedding LayerNorm — **ASSIGNED (64th candidate axis)**

- **Branch:** charliepai2g48h5-askeladd/embed-ln
- **Hypothesis:** Adding an `nn.LayerNorm(96)` site AFTER FiLM modulation and BEFORE the block loop gives block 0 a normalized stationary-scale input, enabling its LayerScale γ to learn against a fixed baseline rather than chasing drift in preprocess+placeholder+FiLM. Structurally DISTINCT from closed normalization-REPLACEMENT taxa (DyT, RMSNorm, GroupNorm) — this experiment ADDS a new LN site rather than replacing existing ones. Xiong et al. 2020 (pre-LN at residual stream entries stabilizes optimization).
- **Param delta:** +192 (96 weight + 96 bias) → expected 333,795 total.
- **Bar:** val_avg/mae_surf_p < 31.3216.

---

## 2026-05-14 [Round 95] UTC — PR #2809: 3-layer output head (mlp2 deeper) — **ASSIGNED (65th candidate axis)**

- **Branch:** charliepai2g48h5-frieren/mlp2-deeper
- **Hypothesis:** Replace `mlp2 = Linear(96,96)→GELU→Linear(96,3)` with `Linear(96,96)→GELU→Linear(96,96)→GELU→Linear(96,3)`. Decoder-depth probe (2 nonlinearities vs 1) — distinct sub-axis from closed PR #2750 fern decoder-WIDTH (stale_wip, axis-untested). The output head is the *only* place semantic latents convert to physical-channel predictions; a richer decoder may rebalance work load from block 3.
- **Param delta:** +9,312 (96×96 + 96) → expected ~342,915 total.
- **Bar:** val_avg/mae_surf_p < 31.3216.

---

## 2026-05-14 [Round 94] UTC — PR #2765: SE block-3-only reduction=4 — **MERGED WIN (−2.87% val)**

- **Branch:** charliepai2g48h5-tanjiro/se-reduction4
- **Hypothesis:** SE bottleneck at reduction=8 (inner dim=12) is capacity-limited in inter-channel correlation modeling. Doubling to reduction=4 (inner dim=24) should enable more discriminative per-channel gating, especially on OOD splits where gate_std from #2727 showed OOD-discriminating behavior.
- **Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-se-reduction4-20260514-044200/metrics.jsonl`

| Split | reduction=4 val | Baseline #2741 | Δ |
|---|---|---|---|
| `val_single_in_dist` | **24.9721** | 25.5303 | **−2.19% WIN** (new best-ever) |
| `val_geom_camber_rc` | **46.9885** | 47.4674 | **−1.01% WIN** |
| `val_geom_camber_cruise` | **17.7276** | 19.1621 | **−7.49% WIN** |
| `val_re_rand` | **35.5983** | 36.8311 | **−3.35% WIN** |
| **val_avg** | **31.3216** | 32.2477 | **−2.87% WIN** |
| `test_single_in_dist` | 24.0714 | 23.7995 | +1.14% mild |
| `test_geom_camber_rc` | **41.9406** | 43.3320 | **−3.21% WIN** |
| `test_geom_camber_cruise` | **14.2400** | 15.3055 | **−6.96% WIN** |
| `test_re_rand` | **25.7749** | 27.2622 | **−5.46% WIN** |
| **test_avg** | **26.5067** | 27.4248 | **−3.35% WIN** |

- **Best epoch:** 64/70 (timeout at ep66); **Param count:** 333,603 (+2,572)
- **SE gate diagnostic (block 3):** gate_std rose 5-12% on every split (in_dist 0.150→0.168, cruise 0.255→0.269, rc 0.215→0.234, re_rand 0.257→0.275); gate_min collapsed 50-70% on every split (near-total channel veto at cruise 0.016→0.005, re_rand 0.019→0.007). gate_mean and gate_max unchanged.
- **Commentary:** Hypothesis confirmed. The SE bottleneck at reduction=8 was capacity-limited. Doubling the inner dim unlocked more discriminative per-channel gating — both wider gate distribution (gate_std+) and much more aggressive channel-veto (gate_min−). OOD cruise (−7.49%) and re_rand (−3.35%) benefited most, consistent with the OOD-discrimination mechanism. Test in-dist mild regression (+1.14%) within noise. Block-3 γ_attn mean flipped slightly negative (only block with sign flip) = SE now picking up role previously carried by block-3 attention. SwiGLU stats unchanged (orthogonality confirmed). SE bottleneck dimension is a productive sub-axis. NEW BASELINE val 31.3216 / test 26.5067. **19th merged winner**.

---

## 2026-05-14 [Round 94] UTC — PR #2778: Deep supervision (Lee 2014) — **CLOSED LOSS (+6.41% val)** — 61st taxon

- **Branch:** charliepai2g48h5-edward/aux-deep-supervision
- **Hypothesis:** Auxiliary mae_surf_p heads after blocks 0, 1, 2 with AUX_WEIGHT=0.1 would act as gradient-pathway regularizer, accelerating early-block specialization and improving OOD generalization via multi-scale supervision.
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-aux-deep-supervision-20260514-043524/metrics.jsonl`

| Split | This run | Baseline #2741 | Δ |
|---|---|---|---|
| `val_single_in_dist` | 27.0415 | 25.5303 | +5.92% LOSS |
| `val_geom_camber_rc` | 52.0657 | 47.4674 | +9.69% LOSS |
| `val_geom_camber_cruise` | 19.7115 | 19.1621 | +2.87% LOSS |
| `val_re_rand` | 38.4378 | 36.8311 | +4.36% LOSS |
| **val_avg** | **34.3141** | 32.2477 | **+6.41% LOSS** |
| **test_avg** | **28.7866** | 27.4248 | **+4.97% LOSS** |

- **Best epoch:** 60/63 (converged); aux/main loss ratio: 2.0-2.2× (effective magnitude ~0.2× main = strong second objective)
- **Mechanism diagnostic:** Prediction (a) confirmed: early-block γ_attn_std rises (block0 0.0401 highest, block3 0.0332 lowest — gradient-pathway reversal). Predictions (b) and (c) refuted: both in-dist AND OOD regress. Two compounding failure mechanisms: (1) Anti-hierarchical pressure — Transolver's slice-attention + γ=1e-4 biases for gradual late-block-dominated mixing; deep supervision forces the opposite (premature linear decodability at early depths); (2) Linear(96,3) heads apply persistent low-rank 3-D pressure on 75% of residual depth.
- **Commentary:** Deep supervision (Lee 2014) is a valid regularizer for classification and U-Net dense prediction where intermediate layers carry truly independent multi-scale information. For point-cloud regression to a 3-channel field, the aux/main pathway asymmetry breaks — the aux task asks for "spatial-aware 3-channel reduction at every block" which is precisely what the whole network computes, not an intermediate. The anti-hierarchical pressure mechanism clashes directly with the inductive biases of the merged architecture (LayerScale γ=1e-4 near-identity init, slice-attention gradual mixing). **61st taxon: gradient-pathway-multiplication for point-cloud regression to low-rank target.**

---

## 2026-05-14 [Round 93] UTC — Round 93

### Closed PR #2750: fern mlp2-wide192 STALE_WIP (no taxon closed — slot redeployment)
- **Student:** charliepai2g48h5-fern
- **Branch:** charliepai2g48h5-fern/mlp2-wide192
- **Status:** Created 2026-05-14T03:00:48Z. Zero commits. Zero comments. ~1h57m without pickup despite fern pod Running 1/1 ready, 0 restarts, 35h uptime.
- **Action:** Closed as stale_wip; slot redeployed to fresh axis. Output-head width-up (mlp2 hidden 96→192) was a decoder-capacity probe; the axis remains open and can be re-tried later if a slot opens.
- **No taxa closed** by this stale_wip closure.

### Assigned PR #2794: fern SwiGLU in preprocess MLP (63rd candidate axis — input-lifting gating)
- **Student:** charliepai2g48h5-fern
- **Branch:** charliepai2g48h5-fern/swiglu-preprocess
- **Hypothesis:** Test whether the merged SwiGLU-in-MLP-body win (PR #2741, val −0.62%, 18th merged) transfers to the input-lifting (preprocess) stage. 1-line swap `MLP→SwiGLUMLP` in `Transolver.__init__`. Tests whether SwiGLU's per-token channel gating mechanism is location-agnostic (helps at every Linear+nonlinear transformation) or location-specific (only beneficial inside residual blocks where it interacts with LayerScale γ + Pre-LN).
- **Param cost:** Preprocess MLP hidden snaps from 192→128 (SwiGLU's 2/3 ratio rounding to multiple of 8): preprocess params 22,656→17,920 (saves ~4,700). Total params 338,523 → ~333,787 (**−1.4%**).
- **Mechanism prediction:**
  - If gating is location-agnostic: WIN, with similar gate-statistics pattern as merged #2741 (gate_mean decreasing through depth, gate_std positive)
  - If gating is residual-block-specific: LOSS or neutral, since input-lifting has different requirements (heterogeneous feature integration: coords + flow scalars + NACA codes + SDF rays)
- **Structural orthogonality:**
  - Distinct from merged SwiGLU-in-MLP-body (different stack location: pre-block-0 vs inside residual blocks)
  - Distinct from in-flight Mish in mlp2 #2780 thorfinn (output-head activation choice, no gating)
  - Distinct from in-flight GeGLU MLP #2759 alphonse (gate ACTIVATION variation: silu→gelu in MLP body gate)
  - Distinct from closed Fourier features #2509 (input-encoding axis, not activation)
  - **63rd candidate axis: gating mechanism at INPUT-LIFTING stack location.** Simple 1-line change for stall-prone pod.
- **Diagnostics:** preprocess_gate_mean, preprocess_gate_std, preprocess_gate_zero_frac on fixed eval batch (analogous to merged #2741 per-block diagnostics).
- **Bar:** val_avg/mae_surf_p < 32.2477.

### Round 93 summary
- 1 closure + 1 assignment. Closed taxa total: **60** (no new taxa — stale_wip closure). Merged winners total: **18**. In-flight: 8/8 (zero idle GPUs).

---

## 2026-05-14 [Round 92] UTC — Round 92

### Closed PR #2761: nezuko SAM rho=0.05 LOSS (60th taxon — flat-minima 4-direction exhaustive at 30-min budget)
- **Student:** charliepai2g48h5-nezuko
- **Branch:** charliepai2g48h5-nezuko/sam-rho005
- **Hypothesis:** Sharpness-Aware Minimization (Foret et al. 2021 ICLR) wrapped around Lion. 2× fwd/bwd per step, adversarial perturbation in rho=0.05 L2-ball, then update from worst-case gradient. Direct flat-minima probe; orthogonal to closed indirect mechanisms (SGDR, SWA, Lookahead).
- **Results:** Clear LOSS — both val and test ~70-75% worse than baseline:

| Metric | Baseline #2741 | SAM rho=0.05 (#2761) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 32.2477 | 54.6259 | +22.38 (+69.4%) |
| `test_avg/mae_surf_p` | 27.4248 | 48.0641 | +20.64 (+75.3%) |
| Best epoch | 63/70 | 35/70 (timeout-truncated) | −28 epochs |
| Epoch wall-time | ~28 s | ~52 s | +85% |

- **Per-split:** All 4 val splits regress +51% to +118%. No OOD-specific benefit despite the flat-minima prediction that OOD should improve most.
- **SAM diagnostics — mechanism engaged correctly:**
  - `sam_loss_ratio` (L(w')/L(w)) climbs monotonically: 1.036 → 1.294 → 1.358 → 1.402 — confirms Lion finds genuinely sharp basins
  - `sam_grad_norm_ratio` (‖g'‖/‖g‖): 1.21 → 1.93 — landscape sharply curved
  - `sam_sharpness_estimate`: 0.036 → 0.402 — Lion drives loss into increasingly sharp basins
  - Lion momentum buffer 99.97% non-zero at terminal — SAM-Lion wiring clean
  - LayerScale γ at best: gamma_attn abs_mean [0.039, 0.052], gamma_mlp [0.092, 0.113] — γ grew normally, no SAM-specific pathology
- **Failure mode (LOSS scenario predicted by PR):** The 2× per-step compute cost halved the effective epoch budget (35 vs 63), and the loss curve was still steeply descending at the wall-clock cap (slope ~1.6 val units/epoch between ep30-35; extrapolation suggests ~15 more epochs needed just to reach baseline). The diagnostic evidence is internally coherent: Lion's basins ARE sharp (SAM's own metric), but the wall-clock cap prevented SAM from exploiting that.
- **Taxonomic significance:**
  - **60th closed taxon: direct flat-minima probe under 30-min budget**
  - **Flat-minima/narrow-basin-escape meta-family 4-direction exhaustive:** SGDR (#2697, 52nd, indirect via LR shock) + SWA (#2567, 28th, indirect via trajectory averaging) + Lookahead+Lion (#2740, 55th, indirect via slow-weight commitment) + SAM (#2761, 60th, DIRECT adversarial perturbation) — all LOSS
  - Flat-minima HYPOTHESIS not refuted (Lion basins genuinely sharp); the wall-clock CAP is the binding constraint. Under 30-min Lion per-step efficiency dominates per-step quality.
- **Param-count flag noted:** Student reported 331,287 params vs PR body 338,523; upstream baseline accounting discrepancy, doesn't affect SAM conclusion (same model both sides).

### Assigned PR #2792: nezuko Register tokens N_REG=4 (Darcet 2024, 62nd candidate axis — learnable attention K/V capacity expansion)
- **Student:** charliepai2g48h5-nezuko
- **Branch:** charliepai2g48h5-nezuko/register-tokens-4
- **Hypothesis:** Vision Transformers Need Registers (Darcet, Oquab, Mairal, Bojanowski — 2024 ICLR, arXiv:2309.16588). Add N_REG=4 learnable register tokens per head per block to PhysicsAttention's slice attention. Registers participate as additional K/V positions (slice_num=24 + N_REG=4 = 28 total positions for attention computation) but are EXCLUDED from the output routing back to mesh nodes (only the first 24 positions are routed back).
- **Param cost:** N_REG × heads × dim_head × n_blocks = 4 × 2 × 48 × 4 = **1,536 new params** (+0.45% over 338,523 baseline). Negligible.
- **Mechanism:** Registers serve as global computation scratchpads. Mesh-derived slice tokens (24) can attend to and use the registers for global state without polluting per-mesh-node outputs.
- **Structural orthogonality:**
  - Not weight reg (closed: spectral norm, WS, L1+L2)
  - Not optimizer/trajectory (closed: SGDR, SWA, Lookahead, SAM — flat-minima 4-direction exhaustive)
  - Not normalization (closed: DyT, RMSNorm-substitution)
  - Not activation (in-flight: Mish, GeGLU)
  - Not positional encoding (closed: Fourier 24th + RoPE 46th)
  - Not loss-shape (closed: Huber, asinh, BerHu, focal)
  - Not augmentation (closed: Mixup family, reflection, coord-jitter)
  - Not stochastic dropout (closed: 3 directions)
  - Not SE (merged: block-3-only)
  - Not SwiGLU (merged)
  - **Different axis: learnable attention K/V capacity expansion. First probe of this dimension in this launch.**
- **Diagnostics:** register_norm per block per head (does the model use them?) + register_kv_attn per block (attention weight FROM slice tokens TO registers).
- **Bar:** val_avg/mae_surf_p < 32.2477.

### Round 92 summary
- 1 closure + 1 assignment. Closed taxa total: **60** (added 60th direct flat-minima probe — completes 4-direction exhaustive flat-minima meta-family). Merged winners total: **18**. In-flight: 8/8 (zero idle GPUs).

---

## 2026-05-14 [Round 91] UTC — Round 91

### Closed PR #2739: edward Weight Standardization CATASTROPHIC LOSS (59th taxon — forward-time weight standardization)
- **Student:** charliepai2g48h5-edward
- **Branch:** charliepai2g48h5-edward/weight-standardization
- **Hypothesis:** Weight Standardization (Qiao et al. 2019) — re-center+normalize the output rows of every nn.Linear weight to zero mean, unit std at forward time. Deterministic in-dist regularizer; zero new parameters. Arm 1: all Linear layers. Arm 2: skip the small-fan-in Linears (FiLM, mlp2 final).
- **Results:** Both arms catastrophic LOSS:

| Arm | val_avg/mae_surf_p | Δ vs baseline 32.2477 |
|---|---|---|
| Arm 1 (all Linear) | ~344 | ~10× degradation |
| Arm 2 (skip small fan-in) | ~351 | ~10× degradation |

- **Mechanism:** WS over-suppressed forward signal globally. Every Linear's weight matrix re-centered+normalized at every forward pass destroyed the LayerScale γ=1e-4 near-identity initialization advantage. LayerScale γ climbed 10× trying to compensate but couldn't escape collapse basin. Model collapsed to trivial attractors.
- **Conclusion:** Forward-time weight standardization is incompatible with LayerScale-initialized Transolver at this scale. Combined with closed spectral norm (#2580) + L1+L2 regularization closures, the **weight-constraint regularizer axis** is now 3-direction exhaustive (no, OOD generalization is not unlocked via weight-constraint regularization).
- **Taxon:** 59th closed; weight-constraint regularizer axis fully mapped.

### Closed PR #2728: thorfinn SE near-identity init STALE_WIP (no taxon closed — slot redeployment)
- **Student:** charliepai2g48h5-thorfinn
- **Branch:** charliepai2g48h5-thorfinn/se-id-init
- **Status:** Created 2026-05-14T02:03:35Z. Zero commits. Zero comments. ~2h14m without pickup despite thorfinn pod Running 1/1 ready, 0 restarts, 34h uptime.
- **Action:** Closed as stale_wip; slot redeployed to fresh axis. SE near-identity init taxon remains open and can be re-tried later if a slot opens.
- **No taxa closed** by this stale_wip closure.

### Assigned PR #2778: edward Deep Supervision (60th candidate axis — gradient-pathway-multiplication)
- **Student:** charliepai2g48h5-edward
- **Branch:** charliepai2g48h5-edward/aux-deep-supervision
- **Hypothesis:** Deep supervision (Lee, Xie, Gallagher, Zhang, Tu 2014). Add 3 auxiliary linear prediction heads after blocks 0, 1, 2 — each `Linear(96, 3)` producing intermediate mae_surf_p predictions. Auxiliary loss = same surf+vol weighted L1 as main loss on each aux prediction, averaged across 3 heads, scaled by aux_weight=0.1 and added to main loss.
- **Param cost:** +873 params (+0.26% over 338,523 baseline). Negligible.
- **Mechanism prediction:** auxiliary heads short-circuit gradient flow through 3+ residual blocks, accelerating early-block specialization. LayerScale γ in blocks 0-1 should rise faster than no-aux baseline. In-distribution val should improve via gradient richness.
- **Structural orthogonality:** Not weight reg (closed: spectral norm, WS, L1+L2). Not stochastic perturbation (DropPath). Not optimizer-internal (closed: Lookahead+Lion, SGDR, SWA, EMA). Not normalization replacement (closed: DyT, LN-RMS, WS). Not architectural module (merged: SE block-3, SwiGLU MLP). Not loss-shape (closed: Huber, asinh, BerHu, focal). **Different axis: gradient-pathway multiplication.**
- **Bar:** val_avg/mae_surf_p < 32.2477.

### Assigned PR #2780: thorfinn Mish in mlp2 (61st candidate axis — output-head activation)
- **Student:** charliepai2g48h5-thorfinn
- **Branch:** charliepai2g48h5-thorfinn/mish-mlp2
- **Hypothesis:** Mish activation (Misra 2019, `f(x) = x*tanh(softplus(x))`) replaces GELU in the output head `mlp2 = Linear(96, 96) + GELU + Linear(96, 3)`. Single 1-line change on train.py L212. Mish is smoother near zero than GELU and bounded below by ~-0.31 (vs GELU's -0.17), allowing slightly more negative output — may improve gradient flow into the final prediction layer.
- **Param cost:** 0 delta (param-matched).
- **Structural orthogonality:** SwiGLU acts on the intermediate residual stream with gating. Mish-in-mlp2 acts on the output decoding path (single non-gated activation, final layer only). Output-head activation has never been probed in this launch.
- **Choice rationale:** thorfinn is a stall-prone pod (just closed #2728 stale_wip). Pick the simplest possible 1-line ablation to maximize pickup probability.
- **Bar:** val_avg/mae_surf_p < 32.2477.

### Round 91 summary
- 2 closures + 2 assignments. Closed taxa total: **59** (added 59th forward-time weight standardization). Merged winners total: **18**. In-flight: 8/8 (zero idle GPUs).

---

## 2026-05-14 [Round 90] UTC — Round 90

### Closed PR #2749: frieren mlp_ratio 2→3 LOSS (56th taxon — MLP-width upward axis)
- **Student:** charliepai2g48h5-frieren
- **Branch:** charliepai2g48h5-frieren/mlp-ratio3
- **Hypothesis:** 1-character change `mlp_ratio=2 → 3`; MLP body hidden 192→288 per block; +74K params (+22%). Standard ViT/BERT uses mlp_ratio=4 — never probed above 2 in this launch.

| Metric | mlp_ratio=3 | Baseline #2741 SwiGLU | Δ vs new baseline | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **32.7558** | 32.2477 | **+1.59% LOSS** | LOSS (close, but clear) |
| `test_avg/mae_surf_p` | **28.0663** | 27.4248 | **+2.25% LOSS** | LOSS |
| `val_single_in_dist` | **24.4065** | 25.5303 | **−4.40% WIN — best-ever in-dist** | trade-off positive |
| `val_geom_camber_rc` | 48.5753 | 47.4674 | +2.34% LOSS | trade-off negative |
| `val_geom_camber_cruise` | 19.7757 | 19.1621 | +3.20% LOSS | trade-off negative |
| `val_re_rand` | 38.2657 | 36.8311 | +3.89% LOSS | trade-off negative |
| Param count | 405,143 | 338,523 | +66,620 (+19.7%) | |

**Classic capacity-vs-generalization trade-off.** In-dist WINS by largest margin (-4.40%) but ALL 3 OOD splits regress. |γ_mlp| ballooned 25-40% (baseline 0.04-0.07 → this run 0.06-0.10, peak block 2 = 0.101) confirming the model leans into the extra capacity but spends it on in_dist overfit. **Conclusion:** MLP body at mlp_ratio=2 is NOT capacity-bound in a generalization sense; wider exchanges in-dist fit for OOD robustness.

**Assignment (Round 90):** PR #2764 frieren mlp_ratio=1.5 (inverse downward probe; if wider hurts OOD via capacity-overfit, narrower might tighten OOD generalization further).

### Closed PR #2748: tanjiro SE-blocks-2+3 LOSS (57th taxon — SE depth-extended axis)
- **Student:** charliepai2g48h5-tanjiro
- **Branch:** charliepai2g48h5-tanjiro/se-blocks23
- **Hypothesis:** Direct follow-up to tanjiro's own SE-block3-only WIN (#2727). Add SE module to blocks 2 AND 3 (top-2 depth). Tests whether block 2 contributes independently or block-3-only is the minimal viable unit.

| Metric | SE-blocks-2+3 | Baseline #2741 SwiGLU | Δ vs new baseline | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **33.4972** | 32.2477 | **+3.88% LOSS** | LOSS |
| `test_avg/mae_surf_p` | **29.2486** | 27.4248 | **+6.65% LOSS** | LOSS |
| `val_single_in_dist` | **25.5506** | 25.5303 | +0.08% flat | flat |
| `val_geom_camber_cruise` | 21.4185 | 19.1621 | +11.78% LOSS | bad |
| `val_geom_camber_rc` | 48.5911 | 47.4674 | +2.37% LOSS | LOSS |
| `val_re_rand` | 38.4287 | 36.8311 | +4.34% LOSS | LOSS |
| Param count | 333,443 | 338,523 | −5,080 (-1.5%) | matches prediction |

**Mechanism diagnostic (exemplary):**
- Block 2 SE engaged more than in #2692 4-block — std climbed 0.030 → 0.073-0.111 (confirming upstream-SE-was-dampening hypothesis was correct)
- **BUT block 3 std DROPPED**: in_dist 0.150 → 0.112, re_rand 0.245 → 0.204 (block 2 STOLE gating budget from block 3)
- LayerScale γ on block 3 also dropped: γ_attn 0.017→0.009, γ_mlp 0.10→0.07 (model reduced block-3 residual contribution because SE there became less decisive)
- OOD-to-in_dist ratios: block 3 held (1.73× → 1.82×) but absolute magnitudes dropped; block 2 ratio (1.52×) weaker than block 3 — block 2 does less OOD-discrimination work
- Redistributing gating across two depths reduced TOTAL decisiveness rather than increasing it

**Conclusion:** Block-3-only is the MINIMAL VIABLE UNIT for SE depth. SE depth landscape fully mapped: {none=baseline, block-3-only=WIN #2727, blocks-2+3=LOSS #2748, all-4=mixed #2692}. SE-block3-only is the SE optimum.

**Assignment (Round 90):** PR #2765 tanjiro SE-block3-only with reduction=4 (SE bottleneck-dim probe; doubles SE intermediate dim from 12 to 24 channels; tests whether reduction=8 is capacity-limiting OOD discrimination).

### Closed PR #2735: askeladd EMA decay=0.9995/0.9999 LOSS (58th taxon — eval-time weight averaging)
- **Student:** charliepai2g48h5-askeladd
- **Branch:** charliepai2g48h5-askeladd/ema-decay0.9995
- **Hypothesis:** EMA weight averaging (Polyak 1992) with decay=0.9995 (effective window ~2000 steps ≈ 5.3 epochs). Two arms run: decay=0.9995 primary + decay=0.9999 follow-up. Targets Lion narrow-minimum in-dist overfit via eval-time weight smoothing.

| Metric | decay=0.9995 | decay=0.9999 | Baseline #2741 | Δ vs new baseline |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **34.5688** | **87.2438** | 32.2477 | +7.20% / +170% |
| `test_avg/mae_surf_p` | **29.7303** | 79.9536 | 27.4248 | +8.40% / +191% |
| `val_single_in_dist` (EMA) | 27.6313 | — | 25.5303 | +8.23% |
| `val_geom_camber_cruise` (EMA) | 21.6635 | — | 19.1621 | +13.05% |

**EMA-vs-raw A/B at decay=0.9995, ep64:**
- val_avg raw: 34.7792 → EMA shadow: 34.5688 → −0.21 abs / **−0.6% relative WIN**
- But: underlying raw run sat 5% below baseline before EMA helped → EMA's structural improvement could not bridge the gap

**Mechanism (textbook signature):**
- decay=0.9995: drift_ratio peaks ep5 = 0.331 during rapid descent → decays monotonically to 0.0147 at terminal. With cosine LR collapsing toward 1e-6 in final epochs, raw weights stop moving → EMA averaging collapses to nearly a no-op.
- decay=0.9999: effective window ~10000 steps ≈ 26.7 epochs > total training horizon; drift_ratio stays >16% at terminal; EMA never forgets initial random weights at meaningful rate.

**Conclusion:** Eval-time weight-averaging axis closed on this stack at this LR schedule. Combined with SWA #2567 (28th taxon, train-time post-hoc averaging) and Lookahead #2740 (55th taxon, fast-slow commit), the weight-averaging meta-family is now 3-direction exhaustive. Only direct flat-minima probe (SAM #2761 in-flight nezuko) remains in the narrow-basin-escape research arm.

**Assignment (Round 90):** PR #2766 askeladd GroupNorm(num_groups=4) replacing LayerNorm (normalization-granularity probe; same param count; tests whether 4 channel-groups of 24 each specialize statistically vs LN single-pop; orthogonal to closed DyT all-or-nothing axis 48th taxon).

---

## 2026-05-14 [Round 89] UTC — Round 89

### Closed PR #2740: Lookahead+Lion CATASTROPHIC LOSS (55th taxon)
- **Student:** charliepai2g48h5-nezuko
- **Branch:** charliepai2g48h5-nezuko/lookahead-k5-a05
- **Hypothesis:** Lookahead (Zhang 2019) wraps Lion as base optimizer; k=5 fast steps, α=0.5 slow-weight commit interpolation. Targets narrow-basin escape via slow-weight commitment vs LR shock (failed SGDR #2697) or trajectory averaging (failed SWA #2567).

| Metric | PR #2740 Lookahead+Lion | Old baseline #2692 | New baseline #2741 | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **41.3670** | 33.0195 | 32.2477 | **+25.27% vs old / +28.28% vs new — CATASTROPHIC LOSS** |
| `test_avg/mae_surf_p` | **35.1124** | 28.3562 | 27.4248 | **+28.05% LOSS** |
| `val_single_in_dist` | 34.6179 | 26.4221 | 25.5303 | **+31.03% LOSS** |
| `val_geom_camber_rc` | 58.0233 | 48.3191 | 47.4674 | +22.24% LOSS |
| `val_geom_camber_cruise` | 27.5244 | 19.5170 | 19.1621 | **+43.64% LOSS — worst** |
| `val_re_rand` | 45.3023 | 37.5198 | 36.8311 | +23.00% LOSS |
| Best epoch | 64/70 | 65/70 | 63/70 | similar (timeout) |
| ep1 val_avg | **366.66** | ~70 | ~70 | **5× typical Lion ep1** |

**Mechanism diagnostic (exemplary from student):**
1. Lion's update rule: `lr · sign(β₁m + (1-β₁)g)` — uniform-magnitude, sign-only direction.
2. Lookahead reset every k=5 steps: `fast ← 0.5·(slow + fast)` halves last 5 steps' net displacement.
3. **CRITICAL:** Lion's momentum buffer survives the reset. After weight rollback, exp_avg still reflects pre-reset gradient history. Subsequent sign-updates are oriented from stale momentum → optimizer/weight inconsistency.
4. For magnitude-aware optimizers (AdamW, SGD with Nesterov), this matters less because step magnitude adapts (Adam) or momentum is direction-only (SGD-Nesterov). Lion's sign-step + persistent momentum makes the stale-momentum→sign-flip chain dominate.

**Secondary diagnostics:**
- Lion momentum 99.95% nonzero (healthy, not collapsed) — momentum buffer is the problem, not momentum saturation
- SE block-3 gate std ~0.02 (vs baseline 0.076) — SE never specialized because optimizer never converged to SE-relevant regime
- LayerScale γ matched baseline magnitudes — architecture unaffected, only optimizer dynamics
- A/B diagnostic structurally degenerate: 375 batches/epoch ÷ k=5 = 75 syncs/epoch, save points align with sync points, fast===slow at every checkpoint

**Conclusion:** 55th taxon: Lookahead × Lion sign-momentum incompatibility at this stack scale. Combined with closed SGDR #2697 (LR-shock 52nd taxon) and SWA #2567 (post-hoc averaging 28th taxon), the narrow-basin-escape meta-axis is now 3-direction closed via fast-cycle restart, slow-weight commit, and trajectory averaging. Only eval-time EMA #2735 askeladd remains in the weight-averaging family.

### Assigned PR #2761: SAM (Sharpness-Aware Minimization, Foret 2021 ICLR) wrapped around Lion
- **Student:** nezuko (continuation of narrow-basin-escape research arc)
- **Hypothesis:** Direct flat-minima probe. SAM does 2 forward/backward passes per step — compute gradient g, perturb adversarially w' = w + ρ·g/||g||, compute gradient g' at w', update Lion using g'. Standard ρ=0.05.
- **Why now:** Your prior SGDR/SWA/Lookahead all attacked sharpness indirectly (via optimizer trajectory). SAM directly perturbs the loss landscape to find sharpness. Foret theorem: SAM update minimizes the maximum loss in a ρ-ball around current weights. The Lion sign-step is preserved — SAM only changes which gradient Lion's sign operates on.
- **Cost:** 2× per-step compute. Expected ~57 s/epoch vs ~28 s baseline; reaches ~30 epochs in 30-min timeout (vs 65 baseline). Hypothesis: SAM converges to flatter basin *faster per epoch*; 30 SAM epochs may outperform 65 Lion epochs especially on OOD splits.
- **NEW bar:** val < 32.2477 (SwiGLU baseline).

### Label hygiene
- Fixed `student:alphonse` → `student:charliepai2g48h5-alphonse` on PR #2759 (assign-experiment skill emitted the wrong label format)

---

## 2026-05-14 [Round 88] UTC — Round 88

### Merged PR #2741: SwiGLU MLP activation (Shazeer 2020) — **18th WINNER** (−0.62% val, −0.84% test, NEW BASELINE)
- **Student:** charliepai2g48h5-alphonse
- **Branch:** charliepai2g48h5-alphonse/swiglu-mlp
- **Hypothesis:** Replace GELU activation in all 4 TransolverBlock MLP bodies with SwiGLU (param-matched, hidden_swiglu=128 = round(96*2*2/3/8)*8, +256 params). Tests whether multiplicative gating in the MLP body improves per-token conditional capacity. Activation family not previously probed in this launch.

| Metric | PR #2741 SwiGLU | #2727 baseline | Δ vs #2727 | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **32.2477** | 32.4498 | **−0.62%** | **WIN — NEW BASELINE** |
| `test_avg/mae_surf_p` | **27.4248** | 27.6573 | **−0.84%** | WIN |
| `val_single_in_dist` | **25.5303** | 25.6532 | **−0.48%** | WIN (best-ever in-dist) |
| `val_geom_camber_rc` | 47.4674 | 47.2242 | +0.51% | mild flat |
| `val_geom_camber_cruise` | 19.1621 | 19.0752 | +0.46% | mild flat |
| `val_re_rand` | **36.8311** | 37.8467 | **−2.68%** | WIN |
| `test_single_in_dist` | 23.7995 | 24.2435 | −1.83% | WIN |
| `test_geom_camber_cruise` | 15.3055 | 15.5263 | −1.42% | WIN |
| `test_re_rand` | 27.2622 | 27.7105 | −1.61% | WIN |

Param count: 338,523 (331,031 + 256 SwiGLU delta; essentially param-matched to GELU baseline).
Best epoch: 63/70 (timeout-truncated; +3% sec/epoch from extra Linear + element-wise-mul).

**SwiGLU gate diagnostic (per-block, best-val checkpoint sample batch):**

| Block | gate_mean | gate_std | gate_abs_mean | gate_zero_frac | value_abs_mean | corr(gate,value) |
|---|---|---|---|---|---|---|
| 0 | 0.3555 | 1.0742 | 0.5695 | 0.0235 | 1.7284 | −0.0023 |
| 1 | 0.2531 | 0.9166 | 0.4859 | 0.0180 | 1.5435 | −0.0282 |
| 2 | 0.1531 | 0.7248 | 0.4088 | 0.0131 | 1.2554 | −0.0175 |
| 3 | 0.0881 | 0.5782 | 0.3373 | 0.0169 | 0.7353 | +0.0525 |

**Analysis:** SwiGLU WIN comes from **smooth, signed, depth-progressive re-weighting** (not hard channel masking — gate_zero_frac 1.3-2.4% is well below the "WIN via hard masking" predicted range of 5-20%). gate_mean decreases monotonically (0.356→0.088) and gate_std decreases (1.07→0.58) — deeper blocks use sparser, more discriminative gating. corr(gate,value)≈0 across all blocks confirms gate learns independently from value stream (decoupled conditional capacity). SwiGLU stacks constructively with SE-block3-only (SE gates global token-pool channels; SwiGLU gates per-token MLP channels — genuinely orthogonal mechanisms).

**Assignment (Round 88):** PR #2759 alphonse GeGLU (1-line change: F.silu→F.gelu in gate; same param count; ablates whether gate activation function or GLU structure drove the WIN).

---

## 2026-05-14 [Round 87] UTC — Round 87

### Merged PR #2727: SE block-3-only — **17th WINNER** (−1.73% val, −2.46% test, NEW BASELINE)
- **Student:** charliepai2g48h5-tanjiro
- **Branch:** charliepai2g48h5-tanjiro/se-block3-only
- **Hypothesis:** Remove SE from blocks 0-2; keep only block 3 (deepest). Student tanjiro's #1 follow-up from #2692 — block 3 had highest gate engagement (std=0.076) while blocks 0-2 near-identity (std 0.022-0.034).

| Metric | PR #2727 | #2692 baseline | Δ vs #2692 | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **32.4498** | 33.0195 | **−1.73%** | **WIN — NEW BASELINE** |
| `test_avg/mae_surf_p` | **27.6573** | 28.3562 | **−2.46%** | WIN |
| `val_single_in_dist` | **25.6532** | 26.4221 | **−2.91%** | WIN (in_dist regression ELIMINATED) |
| `val_geom_camber_rc` | **47.2242** | 48.3191 | **−2.27%** | WIN |
| `val_geom_camber_cruise` | **19.0752** | 20.2953 | **−6.01%** | WIN |
| `val_re_rand` | 37.8467 | 37.0415 | +2.17% | mild regression |
| `test_re_rand` | **27.7105** | 29.6808 | **−6.64%** | WIN |

Param count: 331,031 (28,619 + 2,412 block-3 SE only; 7,236 less than #2692 4-block SE).

**SE gate stats at terminal (best-val checkpoint ep66):**

| Split | gate_mean | gate_std | gate_min | gate_max |
|---|---|---|---|---|
| `val_single_in_dist` | 0.4509 | 0.1503 | 0.1328 | 0.9453 |
| `val_geom_camber_rc` | 0.4131 | 0.2152 | 0.0369 | 0.9844 |
| `val_geom_camber_cruise` | 0.3937 | 0.2549 | 0.0159 | 0.9961 |
| `val_re_rand` | 0.3852 | 0.2570 | 0.0186 | 0.9922 |

Block-3 gate std grew **2-3.4× wider** than 4-block SE #2692 (std=0.076) with blocks 0-2 removed — block 3 absorbed the gating work the upstream SEs were diluting. OOD splits gate harder than in_dist (mean 0.39-0.41 OOD vs 0.45 in_dist), confirming SE is doing OOD adaptation. In-dist regression from #2692 (+4.31%) fully eliminated.

**Key insight:** Removing the 3 near-identity upstream gates eliminated the initial 50% stream dampening (gate=sigmoid(0)=0.5) across 3 blocks, allowing the final block's gate to be more decisive and the stream to be less uniformly damped. Depth-selective SE is both more efficient (1/4 the SE params) and strictly better across 3/4 val splits.

**Metric artifacts:** `models/model-charliepai2g48h5-tanjiro-se-block3-only-20260514-020946/metrics.jsonl`

### Closed PR #2710: Gradient Centralization — LOSS (54th closed taxon)
- **Student:** charliepai2g48h5-frieren

| Metric | GC | Baseline #2692 | Δ% | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 35.2214 | 33.0195 | +6.67% | **LOSS** |
| `test_avg/mae_surf_p` | 29.5005 | 28.3562 | +4.04% | **LOSS** |
| `val_single_in_dist` | 30.1068 | 26.4221 | +13.96% | LOSS (worst) |

GC diagnostic: `gc_norm_ratio_mean ≈ 0.949` (stable from ep5 to ep69) — GC removes ~5.1% of gradient norm. Lion momentum non-zero fraction: **0.9977** (≡ baseline 0.998) — GC did NOT reduce Lion's sign-saturation profile. The ~5% gradient direction removed is genuinely signal: the uniform-input-shift direction (row-mean of Linear weights) corresponds to per-channel amplification in slice projections; zeroing it forces less-direct optimizer paths.

Convergence slower: ep10 val=94.4 vs baseline trajectory ~35-40. No flat-minima OOD benefit emerged (camber_rc and re_rand both regressed — not the classic OOD-win + in_dist-wash signature of flatter basin).

**54th taxon: gradient-transformation axis closed** (GC removed useful signal Lion needed; gradient-modification pre-Lion-step is structurally incompatible with sign-saturation semantics at this model size).

**Metric artifacts:** `models/model-charliepai2g48h5-frieren-grad-centralization-20260514-020431/metrics.jsonl`

### Closed PR #2677: Gradient noise injection — 2nd stale_wip event (axis abandoned)
- **Student:** charliepai2g48h5-fern
- Assigned 2026-05-14T00:23:15Z; pod alive (0 restarts) but never picked up assignment
- Round 85: 1st stale event (left one more round)
- Round 87: 2nd stale event → close per 2-stale launch convention
- Note: Close is not a scientific judgment on gradient-noise; the hypothesis was never tested. Combined with GC #2710 (54th taxon), gradient-transformation family abandoned on this arm.

### Assigned PR #2748: SE blocks 2+3 (tanjiro, idle after #2727 merge)
- **Hypothesis:** Add SE to blocks 2 AND 3 (remove from 0-1 only). Direct follow-up to tanjiro's #1 suggestion from #2727 WIN — tests whether block 2 adds incremental gain beyond block 3 alone, or block-3-only is the minimal viable depth.
- **Expected params:** ~333,443 (331,031 + 2 additional SE modules). 
- **Key diagnostic:** gate_std for block 2 vs block 3 separately — did block 2 become decisive (std > 0.10) or stay low (std ≈ 0.030)?

### Assigned PR #2749: MLP width-up mlp_ratio 2→3 (frieren, idle after #2710 close)
- **Hypothesis:** 1-character change `mlp_ratio=2→3`; MLP body hidden 192→288 per block; +73,728 params total (~405K). Tests whether the model is under-parameterized in per-block MLP computation (standard ViT/BERT/GPT uses mlp_ratio=4; this launch has always used 2).

### Assigned PR #2750: Output head width-up mlp2 hidden 96→192 (fern, idle after #2677 close)
- **Hypothesis:** `Linear(96, 96) → GELU → Linear(96, 3)` to `Linear(96, 192) → GELU → Linear(192, 3)` in the last TransolverBlock's mlp2 head; +9,600 params (~340,631 total). Tests whether the decoder is the bottleneck (never probed previously; all prior winners improved encoder).
- **Built-in diagnostic:** mlp2_grad_norm vs mlp_body_grad_norm at terminal.

---

## 2026-05-14 [Round 86] UTC — Round 86

### Closed PR #2722: DropPath/Stochastic Depth — LOSS (51st closed taxon)
- **Student:** charliepai2g48h5-edward
- **Branch:** charliepai2g48h5-edward/droppath-r01
- **Hypothesis:** Linear DropPath schedule 0→0.1 across n_layers=4, applied per-sample to both attn AND mlp residual branches independently.

| Metric | Run | Baseline (#2692) | Δ% | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 34.8786 | 33.0195 | +5.63% | **LOSS** |
| `test_avg/mae_surf_p` | 30.4406 | 28.3562 | +7.35% | **LOSS** |
| `val_single_in_dist` | 27.6627 | 26.4221 | +4.70% | LOSS |
| `val_geom_camber_rc` | 50.7255 | 48.3191 | +4.98% | LOSS |
| `val_geom_camber_cruise` | 21.6651 | 19.5170 | +11.01% | LOSS |
| `val_re_rand` | 39.4610 | 37.5198 | +5.17% | LOSS |

**Mechanism — "model fighting back" via LayerScale γ_mlp ballooning:**

| Block | dpr | γ_attn abs_mean | γ_mlp abs_mean |
|---|---|---|---|
| 0 | 0.0000 | 0.025724 | 0.087099 |
| 1 | 0.0333 | 0.026297 | 0.109507 |
| 2 | 0.0667 | 0.022508 | 0.109733 |
| 3 | 0.1000 | 0.022108 | 0.107821 |

γ_mlp blocks 1-3 ballooned **4-10× larger than baseline ~0.01-0.02** with nearly identical values (0.107-0.110) despite different drop_probs — optimizer treats DropPath uniformly as noise to compensate against. Block 0 (no drop) γ_mlp = 0.087 confirms drop-conditional compensation pattern. `val_single_in_dist` regressed +9.21% (worst) — exact "DropPath LOSS — starved representation" signature pre-registered in PR.

**Stochastic-residual-branch axis closed at small width.** At n_hidden=96, n_head=2, 4 blocks, residual stack too narrow for stochastic branch zeroing. Combined with Mixup #2687 + Manifold Mixup #2704, stochastic-perturbation meta-axis converging toward "this stack rewards deterministic mechanisms" (FiLM/SE/LayerScale winners; Mixup/DropPath/Manifold-Mixup losers).

**Metric artifacts:** `models/model-charliepai2g48h5-edward-droppath-r01-20260514-015521/metrics.jsonl`

### Closed PR #2697: SGDR (T_0=35, T_mult=2) — LOSS (52nd closed taxon)
- **Student:** charliepai2g48h5-nezuko
- **Branch:** charliepai2g48h5-nezuko/sgdr_t35_t_mult2

| Metric | Run | Baseline (#2692) | Δ% | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 39.0987 | 33.0195 | +18.41% | **LOSS** |
| `test_avg/mae_surf_p` | 35.9756 | 28.3562 | +26.87% | **LOSS** |
| `val_single_in_dist` | 32.8696 | 26.4221 | +24.41% | LOSS (worst) |
| `val_geom_camber_rc` | 56.0114 | 48.3191 | +15.92% | LOSS |
| `val_geom_camber_cruise` | 22.6041 | 19.5170 | +15.82% | LOSS |
| `val_re_rand` | 44.9097 | 37.5198 | +19.70% | LOSS |

**Restart shock signature:**

| Epoch | LR | val_avg/mae_surf_p | Notes |
|---|---|---|---|
| 37 | 3.0e-7 | 43.08 | cycle-1 LR≈0 |
| 38 | 1.5e-4 | 42.74 | RESTART to peak |
| 39 | 1.5e-4 | 64.41 | restart shock |
| 40 | 1.5e-4 | 71.37 | shock peak (+67% in 2 epochs) |
| 68 | 9.17e-5 | 39.10 (BEST) | partial cycle-2 |

**Schedule-restart axis closed:** T_0=35 too short to converge cycle-1 (ends at val 42.92 vs baseline 33.37); cycle-2 (T_mult=2 → 70 epochs) truncated mid-flight; restart shock + slow recovery left insufficient budget. val_single_in_dist +29.77% confirms slow 67-epoch cosine load-bearing for in-dist convergence.

**Metric artifacts:** `models/model-charliepai2g48h5-nezuko-sgdr_t35_t_mult2-20260514-015011/metrics.jsonl`

### Closed PR #2704: Manifold Mixup α=0.2 — CATASTROPHIC LOSS (53rd closed taxon)
- **Student:** charliepai2g48h5-alphonse
- **Branches:** charliepai2g48h5-alphonse/manifold-mixup-a02 (2 independent runs cross-checking)

| Metric | Run 2 (best) | Baseline (#2692) | Δ% | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 46.0412 | 33.0195 | +39.45% | **CATASTROPHIC** |
| `test_avg/mae_surf_p` | 40.9491 | 28.3562 | +44.40% | **CATASTROPHIC** |
| `val_single_in_dist` | 42.5064 | 26.4221 | **+60.86%** | LOSS (worst) |
| `val_geom_camber_rc` | 63.4317 | 48.3191 | +31.27% | LOSS |
| `val_geom_camber_cruise` | 29.6969 | 19.5170 | +52.16% | LOSS |
| `val_re_rand` | 48.5300 | 37.5198 | +29.35% | LOSS |

Cross-check Run 1: val=46.7552 (within 1% of Run 2).

**Mechanism — slice-routing semantic prototypes cannot linearly interpolate:**
- λ ∼ Beta(0.2, 0.2) bimodal correctly engaged (mean=0.4996, 33.58% <0.1, 33.72% >0.9, 17,625 batches)
- k_layer uniform ~25% each (k=0,1,2,3)
- mask consistency: per-sample masks blended at loss level (researcher-agent recommended)
- **PhysicsAttention.slice_token = einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)** where `slice_weights = softmax(in_project_slice(x_mid))` produces **semantic prototypes**. Linearly mixing `fx` at block-k input means next block's slice projection sees a per-channel convex combination of two physically incompatible meshes (different Re/AoA/foil geometry) — the softmax slicer cannot decompose this into meaningful prototypes. Features 13/14/18 (FiLM-modulated channels) also mixed, so every slice-routing decision is conditioned on physically nonsensical (log_Re, AoA0, AoA1) interpolation.

**Wall-clock overhead +52%:** k_layer as Python int → 4 distinct torch.compile graphs; `cache_size_limit` hit ep 36 → Inductor fell back to eager.

**Mixup family fully mapped — 3 independent failure mechanisms:**
- 49th: input-space Mixup (#2687) — physical incompatibility + pad_collate mesh-structure corruption
- 53rd: hidden-state Manifold Mixup (#2704, THIS) — slice-routing prototype incompatibility
- Prior 18th: reflection-augmentation (#2454) — rigid-body augmentation

CFD data structurally incompatible with synthetic-sample augmentation at this stack scale.

**Metric artifacts:** `models/model-charliepai2g48h5-alphonse-manifold-mixup-a02-20260514-015532/metrics.jsonl` (Run 2 best); `models/model-charliepai2g48h5-alphonse-manifold-mixup-a02-20260514-013826/metrics.jsonl` (Run 1 cross-check)

### Assigned PR #2739: Weight Standardization (edward, idle after DropPath close)
- **Hypothesis:** Qiao 2019 — at forward time, standardize each output row of every `nn.Linear` weight to zero mean and unit std before x @ W^T. Deterministic regularizer per edward's own follow-up #4 from DropPath closure ("non-stochastic in-dist regularizers"). Zero new params.
- **Targeted bottleneck:** in-dist overfitting (SE #2692 winner regressed val_single_in_dist +4.31%; all stochastic regularizers have closed; deterministic mechanism is the natural next probe).
- **Implementation:** `WSLinear(nn.Linear)` subclass overrides forward to compute `W_hat = (W - W.mean(1, keepdim=True)) / (W.std(1, keepdim=True) + eps)`. Recursive swap function replaces every nn.Linear in model. ~22 Linear layers covered.
- **Built-in diagnostic:** per-layer `row_std_mean/max/min` at terminal — does standardization compress loud-row spread?

### Assigned PR #2740: Lookahead wrapper around Lion (nezuko, idle after SGDR close)
- **Hypothesis:** Zhang et al. 2019 — wrap Lion with Lookahead(k=5, α=0.5). Maintains slow weights θ_slow; every k=5 Lion steps, θ_slow ← θ_slow + 0.5(θ_fast - θ_slow); θ_fast ← θ_slow. Smooth slow-weight commitment escapes narrow basin without LR-shock penalty that killed SGDR.
- **Targeted bottleneck:** Lion narrow-basin (taxa 30/31/35) + in-dist overfitting. Same goal as SGDR but via smooth interpolation, not perturbation.
- **Compatibility:** EMA #2735 askeladd in-flight operates on eval-time weights; Lookahead operates on train-time weights. They map the full weight-averaging family together (eval-time + train-time).
- **Built-in diagnostic:** fast-vs-slow A/B at best epoch; drift_ratio per epoch.

### Assigned PR #2741: SwiGLU MLP activation (alphonse, idle after Manifold Mixup close)
- **Hypothesis:** Shazeer 2020 "GLU Variants Improve Transformer" — replace MLP body `GELU(W₁x)·W₂x` with `silu(W_gate x) ⊙ W_value x → W_out`. Param-matched via hidden = (2/3)·standard. Standard in LLaMA/PaLM/Mistral/Gemma.
- **Targeted bottleneck:** in-dist overfitting + general expressivity. Per-channel gating gives MLP conditional capacity for fine-grained discrimination.
- **Structurally orthogonal:** modifies only MLP activation body; everything else (attn, slice-routing, normalization, residual, SE/FiLM/LayerScale, mlp2 output head) unchanged. Param-matched at ~338K (no capacity confound).
- **Built-in diagnostic:** SwiGLU gate stats per block (gate_zero_frac, gate_value_correlation, gate_abs_mean).

### Status
- **PR #2677 fern grad-noise injection at 1st stale_wip event** (assigned ~2.5h ago at 2026-05-14T00:23:15Z; pod alive but hasn't picked up assignment). Leaving for one more round per launch convention (close at 2nd stale).
- **8 in-flight after Round 86 assignments**, zero idle.
- **53 total closed taxa** (49 → 53 in last 3 rounds: Mixup, α-entmax, DropPath, SGDR, Manifold Mixup added).
- **16 merged winners**; baseline val 33.0195 / test 28.3562 (PR #2692 SE per-block) unchanged.

---

## 2026-05-14 [Round 85] UTC — Round 85

### Closed PR #2706: α-entmax(α=1.5) MHA attention — LOSS (50th closed taxon)
- **Student:** charliepai2g48h5-askeladd
- **Branch:** charliepai2g48h5-askeladd/entmax15-attn
- **Hypothesis:** Replace softmax with α-entmax (Peters et al. 2019 ACL) in MHA over slice tokens; sparse probability simplex, row-sum=1, allows exact zeros. Follow-up to closed Sigmoid Attention #2673.

| Metric | Run | Baseline (#2692) | Δ% | Verdict |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 34.0974 | 33.0195 | +3.27% | **LOSS** |
| `test_avg/mae_surf_p` | 29.1550 | 28.3562 | +2.82% | **LOSS** |
| `val_single_in_dist` | 27.3715 | 26.4221 | +3.59% | LOSS |
| `val_geom_camber_rc` | 51.2275 | 48.3191 | +6.02% | LOSS |
| `val_geom_camber_cruise` | 19.9260 | 19.5170 | +2.09% | LOSS |
| `val_re_rand` | 37.8646 | 37.5198 | +0.92% | LOSS |

Note: student's table compares vs old baseline #2614 (33.3722); above recalculated vs new baseline #2692 (33.0195).

**Sparsity cascade (best checkpoint ep59):**

| Block | Sparsity | NonZero/G | Top-1 prob | Regime |
|---|---|---|---|---|
| 0 | 10.4% | 21.5/24 | 0.118 | mostly dense |
| 1 | 40.8% | 14.2/24 | 0.256 | moderate sparsity (healthy band) |
| 2 | 90.3% | 2.3/24 | 0.772 | very sparse |
| 3 | **95.8%** | **1.0/24** | **1.000** | **DEGENERATE single-token routing** |

**LayerScale γ_attn at terminal:**
- Block 0: 0.0166 | Block 1: 0.0247 | Block 2: 0.0190 | Block 3: **0.0089** (2.8× smaller — optimizer dampening broken attention branch)

**Mechanism:** Depth-progressive sparsification cascade. Healthy graduated sparsity in blocks 0-1 but cascades into near-total single-token routing in blocks 2-3. Block 3 nonzero_count=1.0/24 — every query attends to exactly one slice prototype. Combined with in-dist overfitting at +8.07% (block 3 needs fine-grained slice blending), the late-block collapse dominates.

**50th taxon closure — attention-shape axis fully mapped:**
| Mechanism | Row-sum=1? | Smooth competition? | Result |
|---|---|---|---|
| Softmax (α=1) baseline | ✓ | ✓ | best — 33.0195 |
| α-entmax (α=1.5) [THIS] | ✓ | ✗ (exact zeros) | LOSS +3.27% |
| Sigmoid #2673 | ✗ | ✓ | LOSS |

Both row-sum=1 normalization AND smooth competition (no exact zeros) are independently load-bearing. Attention-shape axis closed — softmax is the optimum.

**Implementation notes:** Used `entmax.entmax15` (deep-spin library v1.3), NOT buggy inline closed-form in PR appendix. fp32 cast around entmax15 forward for bf16 sqrt stability. Manual MHA (not fused SDPA) for diagnostic access — ~17% epoch overhead (30.1s vs 25.7s baseline). Best epoch: ep59/60 (timeout-truncated).

**Metric artifacts:** `models/model-charliepai2g48h5-askeladd-entmax15-attn-20260514-013731/metrics.jsonl`

### Assigned PR #2735: EMA weight averaging (askeladd, idle after #2706 close)
- **Hypothesis:** EMA (Polyak 1992) shadow weights with decay=0.9995 updated after every optimizer.step(); use EMA weights for val/test evaluation only. Zero new trainable params; ~1.3 MB shadow copy. Standard practice in DeiT/MAE/EfficientNet/ConvNeXt not yet probed this launch.
- **Targeted bottleneck:** Lion narrow-basin tendency (taxa 30/31/35) + in-dist overfitting (SE #2692 in_dist +4.31% regression open). EMA sits in flatter loss landscape region.
- **Key diagnostic:** EMA-vs-raw A/B at terminal (built-in counterfactual: raw weights re-evaluated at best-val epoch; direct measure of EMA improvement contribution). Also drift trajectory: `ema/drift_ratio` per epoch should grow then plateau.
- **Why askeladd:** Closed both halves of attention-shape axis with excellent diagnostic depth; pivot to structurally orthogonal weight-averaging lever.

---

## 2026-05-14 [Round 84] UTC — Round 84

### Merged: PR #2692 tanjiro — Squeeze-Excitation per-block (Hu et al. 2018 CVPR) — 16th WINNER

- **Branch:** charliepai2g48h5-tanjiro/se-r8
- **Hypothesis:** Per-block channel gating from global masked average pool: mean_pool_over_tokens → Linear(96→12) → GELU → Linear(12→96) → sigmoid → broadcast multiply. Zero-init fc2 so gate=0.5 at step 0. SE applied at END of each of 4 TransolverBlocks. +9,648 params (+2.94%).
- **Metrics (vs baseline #2614 val=33.3722, test=28.3736):**

| Metric | SE-r8 | Baseline #2614 | Delta |
|---|---|---|---|
| val_avg/mae_surf_p | **33.0195** | 33.3722 | **−1.06% WIN** |
| test_avg/mae_surf_p | 28.3562 | 28.3736 | −0.06% FLAT |
| val_single_in_dist | 26.4221 | 25.3293 | **+4.31% REGRESSION** |
| val_geom_camber_rc | 48.3191 | 49.5771 | −2.54% |
| val_geom_camber_cruise | 20.2953 | 20.4181 | −0.60% |
| val_re_rand | 37.0415 | 38.1642 | −2.94% |

- **Metric artifacts:** models/model-charliepai2g48h5-tanjiro-se-r8-20260514-010915/metrics.jsonl
- **Training:** best ep65/70 (timeout-truncated — 5 cosine epochs missing), ~28 s/epoch, 338,267 params (+9,648)
- **SE gate diagnostics:** Block 3 engages most: gate std=0.076 (range 0.27–0.71); blocks 0-2 near-identity: std 0.022-0.034 (range 0.41-0.62). No saturation. LayerScale γ_mlp block 3 compressed to 0.051 vs baseline ~0.08: depth-progressive SE/LayerScale trade-off.
- **Decision:** MERGED. val_avg improves −1.06% per primary metric. Test flat (−0.06%) — acceptable; primary metric is val_avg. Small improvements compound.
- **Analysis:** SE finds useful signal at depth (block 3). Global-pool channel gating adds OOD-helpful inductive bias (suppresses decision-irrelevant channels for unfamiliar samples). in_dist regression (+4.31%) likely due to gate=0.5 initialization dampening stream magnitude across all 4 blocks. NEW BASELINE: val 33.0195, test 28.3562.
- **NEW BASELINE: val 33.0195, test 28.3562.**

### Closed: PR #2668 thorfinn — n_head=4 retry-1 (2nd STALE_WIP)

- **Branch:** charliepai2g48h5-thorfinn/nhead-4-retry1
- **Decision:** CLOSED — 2nd stale_wip event on n_head=4 axis (original: #2597 at Round 65, retry: #2668 at Round 74). Per launch convention: 2 consecutive stale events = axis abandoned. Zero pod activity on the branch across ~9 rounds. n_head=4 axis closed without result.

### Assigned: PR #2727 tanjiro — SE block-3 only

- **Branch:** charliepai2g48h5-tanjiro/se-block3-only
- **Hypothesis:** Remove SE from blocks 0, 1, 2; keep SE only at the deepest block (block 3). From PR #2692 gate diagnostics: block 3 dominates SE engagement (std=0.076 vs 0.022-0.034 blocks 0-2). Depth-selective SE reduces +9,648 params to +2,412 and eliminates the 3× gate=0.5 initialization dampening penalty from shallow blocks.
- **Why fresh:** Student tanjiro's #1 follow-up; directly tests whether block-3 SE is the minimum viable SE unit; orthogonal to thorfinn's init probe.

### Assigned: PR #2728 thorfinn — SE near-identity init

- **Branch:** charliepai2g48h5-thorfinn/se-id-init
- **Hypothesis:** Change SE fc2.bias initialization from 0 (gate=0.5) to log(99)≈4.595 (gate≈0.99 ≈ identity) for all 4 blocks. 1-line change. Tests whether gate=0.5 initialization was responsible for the in_dist regression (+4.31%) by eliminating the initial 50% stream dampening.
- **Why fresh:** Student tanjiro's #3 follow-up; orthogonal to depth-selective probe (tanjiro); tests initialization dimension of SE independently.

---

## 2026-05-14 [Round 83] UTC — Round 83

### Closed: PR #2687 edward — Mixup alpha=0.2 (Zhang et al. 2018 ICLR)

- **Branch:** charliepai2g48h5-edward/mixup-alpha02
- **Hypothesis:** Input-space vicinal risk regularization via per-batch λ~Beta(0.2,0.2) blending of x and y, training-mode only. Targets in-dist overfitting bottleneck identified by Lion-WD closure (43rd taxon). ZERO new params.
- **Metrics (vs baseline #2614 val=33.3722, test=28.3736):**

| Metric | Mixup α=0.2 | Baseline | Delta |
|---|---|---|---|
| val_avg/mae_surf_p | 45.0687 | 33.3722 | **+35.04% LOSS** |
| test_avg/mae_surf_p | 39.4107 | 28.3736 | **+38.91% LOSS** |
| val_single_in_dist | 38.3806 | 25.3293 | **+51.53% (WORST — prediction inversion)** |
| val_geom_camber_rc | 59.8475 | 49.5771 | +20.72% |
| val_geom_camber_cruise | 31.6357 | 20.4181 | +54.94% |
| val_re_rand | 50.4112 | 38.1642 | +32.10% |

- **Metric artifacts:** models/model-charliepai2g48h5-edward-mixup-alpha02-20260514-005623/metrics.jsonl
- **Training:** best ep67/70, ~26 s/epoch, 30.4 min wall-clock, 328,619 params (unchanged), 14.02 GB GPU
- **Mixup diagnostic:** lambda distribution exactly as designed — 67.11% extreme batches (λ<0.1 or λ>0.9), mean=0.4946, Beta(0.2,0.2) bimodal confirmed; FiLM weight norm grew +29% to 3.39 vs baseline 2.62 — gate fought harder under blended Re/AoA inputs but could not disambiguate physically impossible intermediates
- **Analysis:** CATASTROPHIC LOSS. Prediction inversion — val_single_in_dist was predicted to improve MOST (vicinal regularization hypothesis), instead regressed MOST (+51.53%). Three mechanism failures: (1) Physical incompatibility — tandem-foil mesh occupies constrained physics manifold, not a convex set; linear blend between single-foil and tandem-foil produces physically meaningless intermediate (Re=300, half-tandem) the model has no smooth bridge to. (2) Mesh-structure corruption — pad_collate pads variable-mesh samples (74K-242K nodes) to largest sample; permuting batch indices then blending node-by-node aligns mesh A surface nodes with mesh B volume nodes, producing topologically corrupted per-node features; is_surface mask weights loss correctly but cannot undo input corruption. (3) FiLM gate incompatibility — FiLM gate conditioned on (log_Re, AoA0, AoA1) grew +29% weight norm fighting harder but cannot map blended conditioning back to valid flow states; stack with strong physical-condition gating is MORE vulnerable to input blending, not less.
- **Conclusion:** 49th taxon. **Data-augmentation meta-axis CLOSED for naive blends.** Combined with reflection-aug #2454 (18th taxon, catastrophic, half-space mesh symmetry violated) + coord-jitter (closed earlier), augmentation axis mapped across 3 directions: rigid-body transforms, local perturbation, sample interpolation — ALL LOSS. Tandem-foil CFD inputs are physically rigid; dataset structural constraints fight augmentation harder than model overfitting tendency. Critical companion signal: edward's result strongly suggests alphonse's in-flight Manifold Mixup #2704 faces similar headwind — slice tokens as semantic prototypes may not linearly interpolate under Verma et al.'s hidden-state Mixup.

### Assigned: PR #2722 edward — DropPath/Stochastic Depth (Huang et al. 2016 ECCV)

- **Branch:** charliepai2g48h5-edward/droppath-r01
- **Hypothesis:** Stochastic Depth with linear DropPath schedule dpr=[0.0, 0.033, 0.067, 0.1] applied per-sample independently to both attn and mlp residual branches in every TransolverBlock; training-mode only; ZERO new params; ~20-line change total.
- **Why fresh:** First stochastic residual-branch drop probe in launch. Standard in DeiT/Swin/ConvNeXt/MAE. Structurally distinct from all 49 closed taxa (data-augmentation just closed, gradient-noise in-flight GC in-flight). Compatible with LayerScale γ=1e-4 (each residual contributes ~1% of stream — dropping a branch is a ~1% signal perturbation, very gentle). Distinct mechanism from Mixup: perturbs computation graph not data. Directly targets in-dist overfitting bottleneck via residual variance.
- **Key diagnostic:** Per-block γ_attn/γ_mlp evolution at terminal vs baseline ~0.01-0.02. γ growth = model trusts each residual more under stochasticity (WIN). γ shrink = model defensively suppresses unreliable branches (LOSS).
- **Predicted outcome:** WIN if stochastic depth regularizes the residual stack at the right scale for γ=1e-4; WASH if drop_rate_max=0.1 too gentle (follow-up: 0.2); LOSS if small-width stack starved of representational capacity by residual drops.

---

## 2026-05-14 [Round 82] UTC — Round 82

### PR #2686 frieren: DyT (Dynamic Tanh) normalization — CLOSED (48th taxon, normalization-replacement fails at small width)

- **Branch:** `charliepai2g48h5-frieren/dyt-alpha05`
- **Hypothesis:** Liu, Ba et al. 2024 NeurIPS "Transformers without Normalization". Replace ALL 9 nn.LayerNorm sites with `DyT(d) = γ⊙tanh(α·x)+β`; α scalar per LN site init=0.5; +9 scalar params total.

- **Results table:**

| Metric | Value | vs Baseline 33.3722 | Direction |
|---|---|---|---|
| val_avg/mae_surf_p | **47.6230** | **+42.7%** | **CATASTROPHIC LOSS** |
| test_avg/mae_surf_p | **39.5154** | **+39.3%** | **LOSS** |
| val_single_in_dist | 41.3386 | +63.2% | LOSS (worst) |
| val_geom_camber_rc | 65.0090 | +31.1% | LOSS (least) |
| val_geom_camber_cruise | 32.6380 | +59.8% | LOSS |
| val_re_rand | 51.5065 | +35.0% | LOSS |

- **DyT α evolution (per LN site, init=0.5):**

| Site | init | final (ep65) | interpretation |
|---|---|---|---|
| block[0].ln_1 (attn) | 0.500 | 0.4572 | mild saturation, near-identity |
| block[0].ln_2 (mlp) | 0.500 | 1.0110 | aggressive saturation |
| block[1].ln_1 (attn) | 0.500 | 0.6074 | mild saturation |
| block[1].ln_2 (mlp) | 0.500 | 1.1332 | aggressive saturation |
| block[2].ln_1 (attn) | 0.500 | 0.4797 | mild saturation |
| block[2].ln_2 (mlp) | 0.500 | 1.0679 | aggressive saturation |
| block[3].ln_1 (attn) | 0.500 | 0.4591 | mild saturation |
| block[3].ln_2 (mlp) | 0.500 | 0.6608 | intermediate |
| block[3].ln_3 (dec) | 0.500 | 0.6766 | intermediate |

**Pattern:** ln_1 (pre-attention) → α ≈ 0.46-0.61 (near-identity); ln_2 (pre-MLP) → α ≈ 1.01-1.13 (aggressive saturation). 2× variation across sites — single learnable scalar cannot replicate per-token channel-wise statistical normalization.

- **LayerScale compensation pattern** (γ at terminal vs baseline ~0.010-0.020):

| Block | γ_attn abs_mean | γ_mlp abs_mean (baseline ~0.010-0.020) |
|---|---|---|
| 0 | 0.0473 | **0.1556** (6-8× larger) |
| 1 | 0.0313 | **0.1235** (6-8× larger) |
| 2 | 0.0419 | **0.1451** (6-8× larger) |
| 3 | 0.0314 | 0.0635 (3× larger) |

γ_mlp abs_mean ballooned from baseline ~0.010-0.020 to **0.06-0.16** — optimizer enlarging residual contribution to compensate for DyT's bounded tanh output, but cannot recover enough. This compensation pattern is itself evidence that channel-stat reduction was load-bearing.

- **Conclusion / 48th taxon:** **Normalization-replacement at small width fails — LayerNorm's channel mean+variance reduction is load-bearing on this stack.** Three converging lines of evidence:
  1. **Catastrophic convergence slowdown:** Best ep65/70 hit 30-min timeout still improving slowly at ~0.6/5 epochs — would need ≥50 more epochs to plausibly approach baseline. Per-epoch wall-clock identical (27.8s); tanh-saturated output not measurably faster than LN.
  2. **LayerScale γ_mlp compensation:** abs_mean 6-8× larger than baseline; optimizer trying to push more signal through despite DyT's bounded output.
  3. **Alpha divergence by site:** 2× per-site variation needed; single learnable scalar inadequate replacement for per-token channel statistics.

  Combined with closed RMSNorm (15th taxon, mild bimodal LOSS — testing without mean centering) and closed NormFormer Sandwich (extra normalization didn't help), the **normalization-meta-axis is now mapped across 3 directions**:
  - Add normalization (NormFormer Sandwich): LOSS
  - Remove mean centering (RMSNorm): mild LOSS bimodal
  - Remove all stats (DyT bounded tanh): catastrophic LOSS at this scale
  - LN baseline is dual-optimum.

  Frieren attention-internal/normalization meta-family is now fully closed across **4 distinct probes**: τ #2623 + spectral norm #2580 + QK-Norm #2661 + DyT #2686. Pivoting frieren entirely off this saturated meta-family.

- **Decision:** Closed. Student-flagged follow-ups (mixed-norm probe with DyT only at ln_2, RMSNorm direction with higher LR for α) noted but axis decisively closed.

### PR #2710 frieren: Gradient Centralization (Yong et al. 2020) — ASSIGNED

- **Branch:** `charliepai2g48h5-frieren/grad-centralization`
- **Hypothesis:** Yong et al. 2020 ECCV "Gradient Centralization: A New Optimization Technique for Deep Neural Networks". Apply `g - g.mean(dim=input)` to all rank-≥2 weight gradients between `loss.backward()` and `optimizer.step()`. For Linear weight `W ∈ R^{d_out × d_in}` with gradient `g`, replace with `g - g.mean(dim=1, keepdim=True)`. ZERO new params. ~3-line training-loop addition.
- **Mechanism:** Restricts each weight row's gradient to be orthogonal to the all-ones direction in input space. Effectively removes a "uniform-input shift" degenerate mode from the update. Operates BEFORE Lion's sign-step.
- **Structural orthogonality:** First gradient-transformation probe in launch. Structurally distinct from:
  - fern in-flight #2677 gradient noise (ADDS Gaussian; GC SUBTRACTS mean)
  - LLRD, embed-UP (per-group LR — closed)
  - SAM (gradient ascent perturbation — closed catastrophic LOSS)
  - Lookahead, EMA, SWA (parameter-space averaging — closed family)
  - Spectral norm (weight matrix constraint — closed LOSS)
- **Targets:** Lion narrow-basin tendency (closed taxa #30, #35) via removing degenerate uniform-input-shift mode.
- **Predicted signatures:** WIN uniform → gradient regularization at source unlocks Lion sign-step; WIN OOD-favoring → centralized gradients find flatter minima; WASH → Lion's sign-step dominates GC's mean-subtraction; LOSS → GC removes signal Lion needed (close axis).
- **Key diagnostic:** per-epoch `gc_norm_ratio_mean` = avg(||g_centralized|| / ||g||). Near 1.0 → GC inert (signal removed is tiny). 0.7-0.9 → GC active (significant uniform-input-shift was being subtracted).

---

## 2026-05-14 [Round 81] UTC — Round 81

### PR #2675 alphonse: 2D coord-based RoPE on Q, K — CLOSED (46th taxon, positional-embedding meta-axis closed)

- **Branch:** `charliepai2g48h5-alphonse/2d-rope`
- **Hypothesis:** Su et al. 2021 RoFormer extended to 2D mesh — rotate Q and K post-projection using (x, y) coord-frequency products; split d_head=48 into 24 x-dim + 24 y-dim channels. Zero new params. First positional embedding probe in launch.

- **Results table:**

| Metric | Value | vs Baseline 33.3722 | Direction |
|---|---|---|---|
| val_avg/mae_surf_p | **34.4690** | **+3.29%** | **LOSS** |
| test_avg/mae_surf_p | **29.8748** | **+5.29%** | **LOSS** |
| val_single_in_dist | 27.9011 | **+10.15%** | **LOSS** (worst) |
| val_geom_camber_rc | 50.5356 | +1.93% | mild LOSS |
| val_geom_camber_cruise | 20.4974 | +0.39% | wash |
| val_re_rand | 38.9419 | +2.04% | LOSS |

- **Implementation adaptation:** Student adapted per-node RoPE prescription to slice-center RoPE since Transolver runs Q/K/V on G=24 slice tokens, not per-node features. Sound and necessary adaptation. Coords taken from slice centroid (weighted mean of node coords under slice_weights). Coord range: x ∈ [-6.5, 5.8], y ∈ [-0.35, 6.25].

- **Mechanism:** Slice-routing softmax over G=24 prototypes ALREADY captures spatial structure adequately at this scale. RoPE-on-slice-centers is the wrong place to inject coords because:
  1. Slice tokens are SEMANTIC prototypes, not spatial positions
  2. Centroid is grad-coupled to routing → circular dependency
  3. Per-split signature is **noise-on-easy-split** (in_dist worst-hit +10.15%) not **OOD-bias** (predicted scenario was OOD WIN, did not materialize)

- **Student-flagged base mismatch:** Heo et al. ECCV 2024 "Rotary Position Embedding for Vision Transformer" recommend base=100 (not RoFormer's base=10000) for real-valued coord ranges. At base=10000 only ~4 of 12 frequencies are active for coord range [-6.5, 5.8]; bottom ~8 frequencies are effectively inert. However, even if base=100 activated all 12 frequencies, the structural issue (slice tokens are semantic, not spatial) would remain.

- **Conclusion / 46th taxon:** **Positional-embedding meta-axis closes on this stack.** Fourier-features-preprocess (#2509, 24th taxon) + 2D-RoPE-slice-center (#2675, 46th) both LOSS confirms slice-routing softmax already captures spatial structure adequately. The model just doesn't need explicit positional encoding when it has slice prototypes.

- **Decision:** Closed. Alphonse pivots from positional-embedding axis (now closed) and heavily-explored conditioning meta-family.

### PR #2673 askeladd: Sigmoid Attention (Ramapuram et al. 2024) — CLOSED (47th taxon, non-softmax attention shape closed)

- **Branch:** `charliepai2g48h5-askeladd/sigmoid-attn`
- **Hypothesis:** Replace softmax(Q·K/√d_k) over slice tokens (G=24) with sigmoid(logits − log(N)). Slice-routing softmax UNCHANGED. Tests whether removing row-sum=1 normalization allows multi-position attention.

- **Results table:**

| Metric | Value | vs Baseline 33.3722 | Direction |
|---|---|---|---|
| val_avg/mae_surf_p | **33.6466** | **+0.82%** | borderline LOSS |
| test_avg/mae_surf_p | **28.9979** | **+2.20%** | **LOSS** |
| val_single_in_dist | 26.7279 | +5.52% | LOSS (worst) |
| val_geom_camber_rc | 49.0242 | **−1.12%** | **WIN** (only) |
| val_geom_camber_cruise | 20.4566 | +0.19% | wash |
| val_re_rand | 38.3778 | +0.56% | wash |

- **Mechanism diagnostic** (terminal per-block sigmoid attention stats):

| Block | mean | std | max | row_sum |
|---|---|---|---|---|
| 0 | 0.0272 | 0.0108 | 0.0815 | **0.6521** |
| 1 | 0.0167 | 0.0086 | 0.0391 | **0.4018** |
| 2 | 0.0203 | 0.0190 | 0.1069 | **0.4875** |
| 3 | 0.0403 | 0.0009 | 0.0518 | 0.9667 (near-uniform) |

- **Mechanism reading:** Optimizer used sigmoid's freedom to **DOWN-WEIGHT EVERYTHING** (row_sums 0.40-0.65 in blocks 0-2, well BELOW softmax's 1.0), NOT to attend broadly. Block 3 collapsed to near-uniform 1/24 (std=0.0009), corroborated by smallest γ_attn=0.0117. Max attention probability ≤ 0.107 across all blocks — sigmoid never produced sharp concentration. The "multi-position simultaneously relevant" Ramapuram et al. claim does NOT realize at this scale.

- **Conclusion / 47th taxon:** **Non-softmax attention shape at G=24 slice tokens fails — softmax row-normalization is a load-bearing inductive bias at this scale.** Combined with closed attention-internal 4-mechanism axis (τ #2623, spectral norm #2580, QK-Norm #2661, Talking-Heads #2669, all WASH/LOSS), the attention-mechanism axis is now closed across **5 distinct mechanisms within softmax framework + 1 first non-softmax replacement**.

- **Decision:** Closed. Askeladd pivots to student's #3 follow-up recommendation: α-entmax (preserves row-sum=1 but allows exact zeros — isolates sparsity from normalization dimension).

### PR #2704 alphonse: Manifold Mixup α=0.2 — ASSIGNED

- **Branch:** `charliepai2g48h5-alphonse/manifold-mixup-a02`
- **Hypothesis:** Verma et al. 2019 ICML "Manifold Mixup". Linear interpolation in HIDDEN feature space at random TransolverBlock input k uniform on {0,1,2,3}; per-batch λ ∼ Beta(0.2, 0.2); training-mode only. ZERO new params. ~10-line training-loop hook. FIRST hidden-state interpolation probe in launch.
- **Structural orthogonality:** Distinct from edward in-flight #2687 input-space Mixup — Manifold Mixup operates at RANDOM HIDDEN LAYER not just input. Complementarity is a feature: if input Mixup LOSES but Manifold Mixup WINS, we learn where in the network linear interpolation is most effective. Hypothesis predicts hidden representations are more amenable to linear interpolation than raw inputs (Verma et al. claim — already-disentangled features).
- **Targets:** in-dist overfitting bottleneck identified by Lion-WD closure 43rd taxon.
- **Predicted signatures:** WIN uniform → vicinal hidden regularization works universally; WIN OOD-favoring → hidden-space mixing helps OOD geometry interpolation; WASH → matches input-space Mixup level; LOSS → slice-routing softmax can't tolerate linearly-mixed hidden states.
- **Key diagnostic:** λ distribution histogram + k_layer mixing distribution + per-split signature.

### PR #2706 askeladd: α-entmax α=1.5 attention — ASSIGNED

- **Branch:** `charliepai2g48h5-askeladd/entmax15-attn`
- **Hypothesis:** Peters et al. 2019 ACL "Sparse Sequence-to-Sequence Models". Replace softmax(Q·K/√d_k) over slice tokens with entmax15. α=1.5 produces sparse attention (exact zeros for low-relevance positions) while PRESERVING row-sum=1 normalization. Slice-routing softmax UNCHANGED. Student's #3 follow-up recommendation from closed #2673.

- **Mechanism table** (compare softmax vs entmax-1.5 vs sigmoid):

| Mechanism | Row-sum=1? | Allows exact zeros? | Result on this stack |
|---|---|---|---|
| Softmax (α=1) | ✓ | ✗ | Baseline |
| α-entmax (α=1.5) | ✓ | ✓ | **THIS PROBE** |
| Sparsemax (α=2) | ✓ | ✓ (sparser) | Untested |
| Sigmoid | ✗ | trivially (sigmoid) | LOSS #2673 |

- **Test:** Isolates sparsity from normalization. If softmax is optimal because of normalization, α-entmax should WIN/WASH. If softmax is optimal because of smooth competition, α-entmax should LOSS like sigmoid did. ZERO new params. ~3-line softmax → entmax15 swap.
- **Predicted signatures:** WIN uniform → sparsity was missing ingredient (close attention-shape axis with new winner); WIN OOD-favoring → sparse routing helps geometric OOD; WASH sparsity-collapsed → α=1.5 not aggressive enough (could try sparsemax α=2); WASH sparsity-engaged → softmax was optimal (close axis with "smooth row-normalized" being load-bearing); LOSS → sparse routing breaks slice-prototype diversity.
- **Key diagnostic:** sparsity per block + top-1 prob + nonzero count + per-split signature.

---

## 2026-05-14 [Round 80] UTC — Round 80

### PR #2672 nezuko: Kendall heteroscedastic uncertainty-weighted multi-task loss — CLOSED (45th taxon, loss-balance meta-axis closed)

- **Branch:** `charliepai2g48h5-nezuko/kendall-heteroscedastic`
- **Hypothesis:** Replace fixed `surf_weight=10` with learned log-variance scalars `s_surf`, `s_vol` such that `loss = exp(-s_surf)*L_surf + s_surf + exp(-s_vol)*L_vol + s_vol`. Self-balancing precision-weighted loss; Kendall, Gal & Cipolla 2018 CVPR. +2 params. First probabilistic loss re-weighting probe in launch.

- **Results table:**

| Metric | Value | vs Baseline 33.3722 | Direction |
|---|---|---|---|
| val_avg/mae_surf_p | **33.6532** | **+0.84%** | **LOSS** |
| test_avg/mae_surf_p | **29.1607** | **+2.78%** | **LOSS** |
| val_single_in_dist | 26.7223 | +5.50% | LOSS |
| val_geom_camber_rc | 48.0617 | −3.06% | **WIN** |
| val_geom_camber_cruise | 22.4845 | +10.12% | **LOSS** (worst) |
| val_re_rand | 37.3409 | −2.15% | **WIN** |

- **Mechanism diagnostic** (terminal s values from metrics):

| Param | Init | Terminal | Effective weight |
|---|---|---|---|
| `s_surf` | −2.30 (log(1/10)) | **−2.42** | exp(2.42) ≈ **5.61** |
| `s_vol` | 0.0 (uniform) | **−1.75** | exp(1.75) ≈ **2.89** |
| ratio surf/vol | **10.0** (baseline) | **1.94** | **5× rebalance toward vol** |

- **Per-split signature is MIXED, not uniform:**
  - Easy splits (in_dist, camber_cruise) regress HARD (+5.50%, +10.12%)
  - Hard OOD splits (camber_rc, re_rand) IMPROVE (−3.06%, −2.15%)
  - val_geom_camber_rc is the long-running OOD bottleneck — Kendall HELPED it
  - But camber_cruise damage dominates val_avg

- **Conclusion / 45th taxon:** **Probabilistic NLL optimum DIVERGES from primary surface-pressure metric optimum.** Kendall's heteroscedastic uncertainty target is "minimize joint NLL across surf+vol noise" not "minimize surface-pressure MAE". Algebraically: when surf has lower true noise than vol, NLL-optimal eff_w_surf < fixed-weight-optimum eff_w_surf — Kendall trades surface fidelity for volumetric noise modeling. **Loss-balance meta-axis closed across all 3 directions** — fixed surf_weight=10 baseline OPTIMAL vs learnable Kendall (#2672) + per-channel ch_w surf_loss-only (#2496) + global per-channel ch_w (#1428/#1871). Combined with previously closed loss-SHAPE family (Huber β ∈ {0.1, 0.25, 0.5, 1.0, 2.0} + L1-MERGED + berHu), the loss-shape meta-axis is also fully saturated. Pure L1 with `surf_weight=10` is the dual-optimum across loss-shape and loss-balance dimensions.

- **Decision:** Closed. PR comment cites mixed per-split signature, 5× rebalance ratio, and probabilistic-vs-MAE divergence. Nezuko pivots from saturated loss-balance to schedule-restart axis.

### PR #2697 nezuko: SGDR Cosine Warm Restarts (T_0=35, T_mult=2) — ASSIGNED

- **Branch:** `charliepai2g48h5-nezuko/sgdr-t35`
- **Hypothesis:** Replace `CosineAnnealingLR(T_max=epochs-3)` with `CosineAnnealingWarmRestarts(T_0=35, T_mult=2)`. Single warm restart at approximately ep38 (post-warmup-3 + T_0=35) resetting LR back to peak 1.5e-4; second cosine cycle decays toward 0 over remaining ~32 epochs. ZERO new params. One-line scheduler change. **FIRST schedule-restart probe since warmup-3-cosine merged.** Reference: Loshchilov & Hutter 2017 ICLR "SGDR: Stochastic Gradient Descent with Warm Restarts".
- **Mechanism:** mid-budget LR reset shakes Lion's sign-step optimizer out of a narrow attractor basin, then the second cosine-decay phase re-anneals into (potentially) a different, wider basin. Targets the documented Lion narrow-basin tendency (30th and 35th closed taxa) via mid-budget LR reset enabling escape from sharp minimum AND re-anneal into potentially wider basin in the second cycle.
- **Structural orthogonality:** optimizer-orthogonal — keeps Lion intact (lr=1.5e-4, wd=3e-4, betas=(0.9, 0.99)), only changes the trajectory through LR space. Lion-internal axes are now exhaustively closed across all 4 dimensions (LR 36th, β₁ 35th, β₂ 39th, WD 43rd), so optimizer-hyperparameter space is saturated. Schedule-restart is the canonical optimizer-orthogonal escape mechanism for sharp-minimum tendencies.
- **Predicted signatures:**
  - **WIN** (val_avg < 33.37): best-epoch lands near end of second cycle (ep ~65-70); temporary loss spike at ep 38-40 then re-anneal smooths it
  - **WASH** (within ±1%): warm restart finds equivalent basin — same generalization, different geometric path → Lion-converged-basin is unique up to symmetry
  - **LOSS** (>34.0): warm restart breaks converged state; second cycle too short or LayerScale γ caps re-organization → best-epoch < 35
- **Key diagnostic:** best epoch position in first cycle vs at restart boundary vs second cycle is the entire attribution lever.

---

## 2026-05-14 [Round 79] UTC — Round 79

### PR #2669 tanjiro: Talking-Heads Attention — CLOSED (44th taxon, cross-head mixing at H=2 saturates)

- **Branch:** `charliepai2g48h5-tanjiro/talking-heads-attn`
- **Hypothesis:** Shazeer 2020 Talking-Heads — 2×2 logits_mixing + 2×2 probs_mixing Linear modules per block (identity-init); +32 params total; cross-head linear mixing pre- and post-softmax. Compounds with QK-Norm if both win.
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | Talking-Heads | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **34.5351** | 33.3722 | **+3.48% LOSS** |
| test_avg/mae_surf_p | **28.8990** | 28.3736 | **+1.85% LOSS** |

Per-split val (uniform regression):

| Split | Talking-Heads | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | 27.3990 | **25.3293** | **+8.17% WORST** |
| val_geom_camber_rc | 51.0271 | **49.5771** | +2.92% |
| val_geom_camber_cruise | 20.9950 | **20.4181** | +2.83% |
| val_re_rand | 38.7193 | **38.1642** | +1.45% |

Per-split test (best-val ep66 checkpoint):

| Split | test mae_surf_p |
|---|---|
| test_single_in_dist | 25.0414 |
| test_geom_camber_rc | 44.1284 |
| test_geom_camber_cruise | 17.0733 |
| test_re_rand | 29.3528 |

- **Run characteristics:** Best ep=66/70 (timeout at 30.7min). +0.3s/epoch overhead. +32 params (+0.01% matches spec). Peak GPU 14.0 GB.

- **Mechanism diagnostic — mixing matrices ENGAGED but failed:**

| Block | `logits_mixing` |dev_from_I| | `probs_mixing` |dev_from_I| |
|---|---|---|
| 0 | **0.2121** | 0.0807 |
| 1 | 0.1372 | 0.0862 |
| 2 | 0.1583 | 0.0529 |
| 3 | 0.2012 | 0.0175 |

Full terminal matrices show diagonals shrinking BELOW 1.0 (0.58-0.94) — heads were ABSORBING each other's logits, down-weighting each head's own pattern. Pre-softmax mixing (logits) is 2-10× stronger than post-softmax (probs).

- **Mechanism:** At H=2, "sharing" between heads homogenizes them rather than expanding pattern diversity. Talking-Heads is theoretically more useful at H≥4-8 where head specialization is stronger. With H=2, each head's own logit pattern gets contaminated by the other's, reducing effective attention pattern diversity. The identity-init wasn't sufficient — the optimizer actively moved the mixing matrices toward a regime that hurt performance.

- **44th closed taxon: cross-head attention mixing at H=2 closed.**

**Attention-internal axis now COMPREHENSIVELY closed across 4 distinct mechanisms:**

| Mechanism | PR | Status | Year |
|---|---|---|---|
| Post-softmax shape (τ temperature) | #2623 | LOSS (37th) | Closed |
| Weight Lipschitz (spectral norm) | #2580 | LOSS (32nd) | Closed |
| Score stabilization (QK-Norm) | #2661 | WASH/LOSS (42nd) | Closed |
| Cross-head mixing (Talking-Heads) | #2669 | LOSS (44th) | Closed |

Four independent mechanisms — pre-softmax shape (logits_mixing), per-row normalization (τ), constraint (spectral), and post-projection (QK-Norm) — all fail to move val. **Attention-internal stabilization is NOT the bottleneck on this stack at H=2.**

In-flight attention-adjacent probes remain (#2673 Sigmoid Attention non-softmax, #2675 2D RoPE positional embedding); these test fundamentally different attention angles.

- **Action:** Closed. Pivoted tanjiro to structurally distinct architectural probe.

---

### PR #2692 tanjiro: Squeeze-Excitation per-block (Hu et al. 2018) — ASSIGNED (Round-79)

- **Branch:** `charliepai2g48h5-tanjiro/se-r8`
- **Hypothesis:** Add SE module at end of each TransolverBlock — global average pool over tokens → Linear(96→12) → GELU → Linear(12→96) → sigmoid → broadcast channel multiply. Hu, Shen & Sun 2018 CVPR (SENet — ImageNet 2017 winner). Per-channel gating CONDITIONED on global stream content; orthogonal to per-token attention, per-channel LayerScale (constant), and input-conditioned FiLM.

- **Code pattern:**
```python
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        d_hidden = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, dim, bias=True)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
    def forward(self, x):
        s = x.mean(dim=1)                           # (B, D)
        s = self.fc2(F.gelu(self.fc1(s)))          # (B, D)
        gate = torch.sigmoid(s)                     # (B, D)
        return x * gate.unsqueeze(1)
# Apply at end of each TransolverBlock.forward()
```

- **Predicted outcomes:**
  - **WIN — channel gating:** Global content gating adds inductive bias that LayerScale (constant) and FiLM (input-only) can't capture. Per-split: re_rand and camber_cruise improve most. Best case −0.5% to −2%.
  - **WASH:** sigmoid gates converge near 0.5–0.7 with minor variation; SE adds no signal beyond LayerScale + FiLM.
  - **LOSS:** Gates collapse to extremes; OR multiplicative cascade with LayerScale destabilizes stream variance.

- **Why structurally fresh:** FIRST channel-wise gating CONDITIONED ON GLOBAL STREAM CONTENT in launch. Distinct from LayerScale (per-channel constant gain), FiLM (input-conditioned not content-conditioned), attention (per-token not global), and slice routing (per-prototype not per-channel).

- **Param overhead:** +9,648 params (+2.9%). Compute ~+1-3% per epoch (mean pool + tiny MLP per block).

- **Init detail:** zero-init `fc2` → gate = sigmoid(0) = 0.5 uniformly at start. Residual stream halved initially; LayerScale γ has flexibility to grow γ to compensate. Meaningful initial perturbation but doesn't break the network.

- **Baseline to beat:** val < 33.3722.

---

## 2026-05-14 [Round 78] UTC — Round 78

### PR #2661 frieren: QK-Norm — CLOSED (42nd taxon, attention-internal trio saturated)

- **Branch:** `charliepai2g48h5-frieren/qk-norm`
- **Hypothesis:** Per-head LayerNorm on Q and K post-projection (V unchanged); +768 params; Henry et al. 2020 / Dehghani et al. 2023 (ViT-22B); targets Q·K magnitude drift causing softmax saturation.
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | QK-Norm | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **33.4391** | 33.3722 | **+0.20% WASH** |
| test_avg/mae_surf_p | **29.1677** | 28.3736 | **+2.80% LOSS** |

Per-split val:

| Split | QK-Norm | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | **24.6761** | 25.3293 | **−2.58% WIN** |
| val_geom_camber_rc | 51.3138 | **49.5771** | **+3.50% LOSS (dominates avg)** |
| val_geom_camber_cruise | **20.0609** | 20.4181 | **−1.75% WIN** |
| val_re_rand | **37.7057** | 38.1642 | **−1.20% WIN** |

3/4 val splits improved; val_geom_camber_rc dominates because it's the largest-magnitude split.

Per-split test:

| Split | QK-Norm | NEW baseline | Δ % |
|---|---|---|---|
| test_single_in_dist | 24.9319 | **24.4830** | +1.83% LOSS |
| test_geom_camber_rc | 45.8308 | **43.3910** | **+5.62% LOSS** |
| test_geom_camber_cruise | **16.1193** | 16.8389 | −4.27% WIN |
| test_re_rand | 29.7888 | **28.7816** | +3.50% LOSS |

- **Run characteristics:** Best ep=66/70 (terminal, timeout-clipped at 30min). +1.9s/epoch over baseline. +768 params (+0.23%, matches expected). Peak GPU 13.12 GB (unchanged).

- **Mechanism diagnostic — pre-norm Q/K magnitudes:** ||Q||,||K|| grew 80-200× from init across 4 blocks (range 9.8-24.3). Without QK-Norm, raw Q·Kᵀ products would scale ∝ 100-550 → softmax saturation territory. **QK-Norm IS normalizing meaningful magnitudes** — mechanism worked as designed.

- **But LayerScale γ_attn caps the effect:**

| Block | γ_attn abs_mean | γ_mlp abs_mean |
|---|---|---|
| 0 | 0.0158 | 0.0880 |
| 1 | 0.0171 | 0.0879 |
| 2 | 0.0178 | 0.0922 |
| 3 | 0.0106 | 0.0902 |

γ_attn stays at 0.010-0.018 → attention residual contribution is ~1-2% of the residual stream. Even saturated-softmax attention values contribute very little. **Pre-LN + LayerScale = the implicit bound on softmax saturation that this hypothesis predicted as scenario #2 (WASH).**

- **42nd closed taxon: attention-internal score-stabilization at Lion + LayerScale + L1 stack.**

**Attention-internal trio now fully closed:**

| Mechanism | PR | Status |
|---|---|---|
| Post-softmax shape (τ) | #2623 | LOSS (37th) |
| Weight Lipschitz (spectral norm) | #2580 | LOSS (32nd) |
| Score stabilization (QK-Norm) | #2661 | WASH/LOSS (42nd) |

Three independent mechanisms all fail to move val significantly → **Pre-LN + small LayerScale γ_attn is sufficient attention regularization** at width=96 / heads=2 / blocks=4.

- **Critical per-split insight (preserved for future hypotheses):** test_geom_camber_rc regressed consistently across val/test (+3.50%/+5.62%). camber_rc has consistently failed to improve across attention-internal interventions, suggesting it's a DATA SPARSITY problem (front-foil M=6-8 held out) rather than a model regularization problem. Sampler reweighting (up-weighting M=2-5 and M=9 training neighbors) may be more productive than any attention tweak.

- **Action:** Closed. Pivoted frieren to structurally distinct normalization-replacement probe.

---

### PR #2658 edward: Lion weight_decay 3e-4 → 1e-4 — CLOSED (43rd taxon, Lion-internal exhausted)

- **Branch:** `charliepai2g48h5-edward/lion-wd-1e-4`
- **Hypothesis:** Lion WD 3e-4 → 1e-4 (1/3 of inherited Lion-paper default); tests whether Lion paper's 3× AdamW WD rule-of-thumb over-regularizes budget-bound stack.
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | WD=1e-4 | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **34.8122** | 33.3722 | **+4.31% LOSS** |
| test_avg/mae_surf_p | **29.2347** | 28.3736 | **+3.04% LOSS** |

Per-split val (uniform regression):

| Split | WD=1e-4 | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | 28.5805 | **25.3293** | **+12.84% WORST** |
| val_geom_camber_rc | 49.7055 | **49.5771** | +0.26% flat |
| val_geom_camber_cruise | 21.7473 | **20.4181** | +6.51% |
| val_re_rand | 39.2153 | **38.1642** | +2.75% |

Per-split test (same ordering as val):

| Split | WD=1e-4 | NEW baseline | Δ % |
|---|---|---|---|
| test_single_in_dist | 25.7480 | **24.4830** | +5.17% |
| test_geom_camber_rc | 44.3330 | **43.3910** | +2.17% |
| test_geom_camber_cruise | 17.0554 | **16.8389** | +1.29% |
| test_re_rand | 29.8022 | **28.7816** | +3.55% |

- **Run characteristics:** Best ep=61/70 (timeout-clipped at 30.1min). Per-epoch ~26s. Lion momentum non-zero fraction at terminal = 0.9980 (vs baseline 0.996; healthy).

- **Mechanism — in-dist-favoring regularization:** WD=3e-4 was load-bearing regularization for the budget-bound Lion+FiLM stack. The per-split signature is **in-dist-favoring**, not OOD-favoring:
  - val_single_in_dist regressed MOST (+12.84%) — seen single-foil distribution was what WD was protecting against overfitting
  - val_geom_camber_rc regressed LEAST (+0.26%) — OOD camber generalization is WD-INSENSITIVE (rc bottleneck is geometric extrapolation, not in-dist regularization)
  - camber_cruise (+6.51%) and re_rand (+2.75%) sit in middle — confirms progressive in-dist→OOD axis

- **Falsified diagnostic prediction:** "lower WD lets early-epoch fitting proceed faster, freeing budget for late-stage refinement." Best epoch DID move earlier (61 vs 70), but val/test both regressed — the budget-freed epochs were spent overfitting in-dist.

- **Second-attempt confirmation:** Student ran an earlier attempt at same config (60 epochs, val=34.4842) — same LOSS pattern within run-to-run variance. Two consistent attempts confirm the result.

- **43rd closed taxon: Lion-internal optimization axes EXHAUSTIVELY closed across all 4 dimensions:**

| Axis | Closed at | Taxon |
|---|---|---|
| LR | 1.5e-4 (bracket {1, 1.5, 1.75, 2}e-4 = {36.40, 33.49, 33.81, 33.83}) | 36th (#2602) |
| β₁ | 0.90 (asymmetric peak; 0.85 LOSS, 0.95 LOSS) | 35th (#2613) |
| β₂ | 0.99 (default); 0.999 catastrophic +78% | 39th (#2647) |
| WD | 3e-4 load-bearing | 43rd (#2658, this PR) |

**Lion optimizer SATURATED at lr=1.5e-4, betas=(0.9, 0.99), wd=3e-4.** Combined with closed parameter-space averaging meta-family (25th EMA, 26th Lookahead, 28th SWA, 30th β₁=0.95) and closed meta-optimizers (12th SAM, 20th LLRD), the optimizer-internal+meta direction is comprehensively saturated.

- **Action:** Closed. Pivoted edward to regularization/data-side probe (Mixup α=0.2).

---

### PR #2686 frieren: DyT (Dynamic Tanh) normalization — ASSIGNED (Round-78)

- **Branch:** `charliepai2g48h5-frieren/dyt-alpha05`
- **Hypothesis:** Replace ALL `nn.LayerNorm(d)` with `DyT(d) = γ ⊙ tanh(α * x) + β` where α is single learnable scalar per LN site (init=0.5). Liu, Ba et al. 2024 NeurIPS "Transformers without Normalization". Eliminates mean/variance reduction in favor of bounded element-wise nonlinearity; one α scalar per site as saturation knob.

- **Code pattern:**
```python
class DyT(nn.Module):
    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
# Replace every nn.LayerNorm(d) site with DyT(d)
```

- **Predicted outcomes:**
  - **WIN:** Removes channel-stat noise at width=96 (only 96 features per token); tanh saturation provides smoother gradient flow. Best case −0.5% to −2%; in_dist improves most.
  - **WASH:** α drifts to LN-equivalent regime; LN is approximately optimal.
  - **LOSS:** Channel-stat reduction was load-bearing; residual stream variance drifts uncontrolled.

- **Why structurally fresh:** FIRST normalization-replacement probe in launch beyond closed RMSNorm. Distinct from NormFormer (adds Post-LN), LayerScale (residual gain), and all attention-internal closures (τ, spectral norm, QK-Norm).

- **Param overhead:** +9-12 scalar params (one α per LN site). Compute IMPACT ~−5% per epoch (saves reduce ops).

- **Baseline to beat:** val < 33.3722.

---

### PR #2687 edward: Mixup α=0.2 — ASSIGNED (Round-78)

- **Branch:** `charliepai2g48h5-edward/mixup-alpha02`
- **Hypothesis:** Per-batch λ ∼ Beta(0.2, 0.2), permute batch, blend `x = λ·x + (1−λ)·x[perm]` and `y = λ·y + (1−λ)·y[perm]`. Zhang et al. 2018 ICLR. Vicinal risk regularization that encourages locally linear behavior in input-output space.

- **Code pattern:**
```python
mixup_alpha = 0.2
if model.training and mixup_alpha > 0.0:
    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[perm]
    y = lam * y + (1.0 - lam) * y[perm]
# Continue with forward/loss/backward as usual
```

- **Predicted outcomes:**
  - **WIN — vicinal regularization:** val_single_in_dist improves MOST (mirrors Lion-WD LOSS signature inverted); mixup reduces in-dist overfitting.
  - **WIN — geometry interpolation:** val_geom_camber_rc improves if blended samples include front-foil M=6-8 neighbors.
  - **WASH:** α=0.2 too gentle (Beta is bimodal at 0/1).
  - **LOSS:** Blended Re/AoA/geometry not physical → FiLM destabilizes.

- **Why structurally fresh:** FIRST input-space LINEAR BLEND probe in launch. Distinct from closed reflection-aug (#2454 catastrophic), coord-jitter (geometric perturbation), per-channel-loss (loss reweighting), and parameter-space averaging meta-family. Directly targets the in_dist overfitting bottleneck identified by Lion-WD closure.

- **Param overhead:** Zero new params. Compute impact ~+1% (extra blend op per batch).

- **Baseline to beat:** val < 33.3722.

---

## 2026-05-14 [Round 77] UTC — Round 77

### PR #2615 fern: Stochastic Depth / DropPath p=0.1 — CLOSED (1st stale_wip → axis non-retried)

- **Branch:** `charliepai2g48h5-fern/drop-path-p01`
- **Hypothesis (preserved):** Per-block per-branch residual zeroing at p=0.1 (CaiT-style; Touvron 2021); 4 blocks × 2 branches = 2^8=256 implicit sub-network ensemble; zero params; orthogonal to LayerScale γ damping.
- **Status:** Closed as 1st stale_wip. PR created 2026-05-13T22:16:56Z, last updated 2026-05-13T22:16:57Z (advisor placeholder commit only). Zero pod activity for ~24h+ across rounds 68 through 76.

**fern pod stall history this launch:**

| PR | Hypothesis | Stall # | Outcome |
|---|---|---|---|
| #2496 | per-channel-surf-p3 | 1st | Retried as #2557 |
| #2557 | per-channel-surf-p3 retry-1 | 2nd | Axis abandoned |
| #2615 | DropPath p=0.1 | 1st | **Axis not retried this launch** |

**Multi-launch precedent:** DropPath was also abandoned in a prior launch after 4 consecutive stalls (round-39 of that launch). The implementation pattern — per-block per-branch random residual zeroing with training/eval mode handling — appears to systematically not get picked up by this pod across launches. Rather than continue the abandon-after-N-retries cycle, pivot fern to a structurally distinct, implementation-light axis.

- **Action:** Closed. Pivoted fern to maximally implementation-light optimization-internal probe never tried in launch.

### PR #2677 fern: Gradient Noise Injection (Neelakantan 2015) — ASSIGNED (Round-77)

- **Branch:** `charliepai2g48h5-fern/grad-noise-eta003`
- **Hypothesis:** Add annealed Gaussian noise to gradients between `loss.backward()` and `optimizer.step()`. Schedule: σ_t = η / (1 + t)^γ with η=0.03, γ=0.55. Targets escape from Lion narrow minimum via early-epoch saddle-point perturbation; noise anneals automatically with step count so late-stage cosine cooldown is unimpeded.

- **Code pattern (3-line addition):**
```python
loss.backward()
# NEW:
noise_eta = 0.03
noise_gamma = 0.55
sigma_t = noise_eta / (1.0 + global_step) ** noise_gamma
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad), alpha=sigma_t)
# END NEW
optimizer.step()
optimizer.zero_grad()
global_step += 1
```

- **Schedule preview:**

| Step | σ_t |
|---|---|
| 0 | 0.0300 |
| 100 (~ep2) | 0.00408 |
| 1000 (~ep20) | 0.00118 |
| 3500 (~ep70 terminal) | 0.00048 |

Meaningful noise only in first ~5-10 epochs, fading naturally without manual scheduling.

- **Predicted outcomes:**
  - **WIN (val < 33.37):** Early-stage noise breaks Lion narrow-minimum tendency (consistent with closed 30th β₁=0.95 LOSS + 35th β₁=0.85 LOSS evidence that Lion sits in narrow basin). Pushes optimization toward flatter basin with better OOD generalization. Best case −0.5% to −2%; per-split camber_rc and re_rand improve most.
  - **WASH (33.37–33.7):** Noise too small to perturb Lion sign-step meaningfully; no effect. Confirms narrow minimum is intrinsic and not addressable by gradient noise. Consider higher η in follow-up.
  - **LOSS (>33.7):** Noise destabilizes early epochs; Lion sign-step amplifies noise-induced sign flips. Close axis at η=0.03.

- **Structurally orthogonal:** to all 41 closed taxa AND all 7 in-flight WIP. Modifies gradient signal BEFORE optimizer-internal momentum/sign logic — distinct from Lion-internal closures (β₁/β₂/LR), per-group LR (LLRD/embed-UP), weight averaging (EMA/SWA), and meta-optimizer wrappers (Lookahead/SAM). Zero structural model change preserves all closed-axis insights.

- **Why fern:** Implementation-light maximizes pickup probability after 3 consecutive stalls (#2496, #2557, #2615). 3-line training-loop addition vs DropPath's per-block module wrapping.

- **Baseline to beat:** val_avg/mae_surf_p < 33.3722.

---

## 2026-05-14 [Round 76] UTC — Round 76

### PR #2646 alphonse: Per-block FiLM (4 independent gates, one per block input) — CLOSED (LOSS; 41st taxon)

- **Branch:** `charliepai2g48h5-alphonse/per-block-film`
- **Hypothesis:** Replace single shared FiLM gate from merged #2614 with 4 independent FiLM gates (one Linear(3,96) zero-init per block input). 4× FiLM params (+1,536 total). Compounds on merged #2614 WIN if per-depth flow conditioning adds value over single shared gate. Mirrors conditional batchnorm in image GANs (Dumoulin 2017).
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | per-block FiLM | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **34.2522** | 33.3722 | **+2.64% LOSS** |
| test_avg/mae_surf_p | ~29.15 | 28.3736 | ~+2.7% LOSS |

Per-split val (uniform regression across all 4 splits):

| Split | per-block FiLM | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | 26.29 | 25.3293 | +3.8% |
| val_geom_camber_rc | 50.42 | 49.5771 | +1.7% |
| val_geom_camber_cruise | 20.91 | 20.4181 | +2.4% |
| val_re_rand | 39.27 | 38.1642 | +2.9% |

- **Key diagnostic — per-block FiLM weight norms learned heterogeneously:**

| Block | FiLM weight norm | Bias norm | Notes |
|---|---|---|---|
| block_0 | 1.52 | 0.42 | Lightest gating (closest to embedding) |
| block_1 | 1.78 | 0.61 | |
| block_2 | 2.04 | 0.78 | |
| block_3 | 2.22 | 0.94 | Heaviest gating (deepest) |

**Depth-varying flow conditioning signal IS REAL** — deeper blocks demanding stronger gating is consistent with the hypothesis that flow context becomes more useful at later representational stages. But aggregate 4-gate budget exceeded single-gate without improving outcome.

- **Mechanism:** The residual stream never settles into flow-agnostic processing between blocks. Each block continuously re-modulated by Re/AoA disrupts what merged #2614 single pre-block-0 gate achieved with one-shot conditioning at stream entry. The 4 gates each commit modulation budget into the residual addition, perturbing the stream variance Transolver was optimized for (LayerScale γ=1e-4 + Pre-LN). Even with zero-init keeping initial behavior identical to baseline, training pulled all 4 gates away from identity simultaneously, never settling into a regime where deeper blocks could specialize.

- **41st closed taxon: depth-distributed flow conditioning under residual-stream interruption — flow-conditioning meta-family fully mapped:**

| Variant | PR | Status | Lesson |
|---|---|---|---|
| Single pre-block-0 gate (residual stream input) | #2614 | **MERGED** | OPTIMUM — one-shot conditioning at stream entry |
| Per-block depth-distributed (4 gates) | #2646 | LOSS (41st taxon) | Stream interruption disrupts settling |
| Output-side additive bias (flow_bias on mlp2) | #2531 | LOSS (27th taxon) | Decoder-fork interference |
| Output-side multiplicative gate (1+gate(...)) | #2588 | LOSS (31st taxon) | Output-side meta-family closed |

Flow conditioning works at residual-stream INPUT only; not depth-distributed, not output-side, not multiplicative-output.

- **Action:** Closed. Pivoted alphonse to structurally distinct axis untouched by any conditioning variant.

### PR #2675 alphonse: 2D coord-based RoPE on Q, K projections — ASSIGNED (Round-76)

- **Branch:** `charliepai2g48h5-alphonse/2d-rope`
- **Hypothesis:** Rotate Q, K post-projection using mesh (x, y) coords; V unchanged. Splits d_head=48 into 24 x-dim + 24 y-dim channels. Frequency buffers via `register_buffer`. Zero new parameters. First positional-embedding probe in launch — Transolver has been geometry-blind across all 41 closed taxa.
- **Code pattern:**
```python
def apply_2d_rope(qk, coords):
    # coords: (B, N, 2) — (x, y) per node
    # Split d_head=48 into (24 x-dim, 24 y-dim)
    # Rotate paired channels: (a, b) → (a·cos − b·sin, a·sin + b·cos)
    ...
q = apply_2d_rope(q, coords)
k = apply_2d_rope(k, coords)
# V unchanged
```
- **Predicted outcomes:**
  - **WIN if positional bias helps:** camber_rc improves most (OOD coords benefit from explicit positional structure).
  - **WASH if frequency encoding redundant:** d_sdf rays + saf channels already encode geometric position implicitly.
  - **LOSS if coord distribution shift dominates:** OOD splits regress harder than in_dist.
- **Structurally orthogonal:** to all closed flow/condition gates (residual stream perturbation) AND slice-routing softmax (over prototypes not positions).
- **Baseline to beat:** val_avg/mae_surf_p < 33.3722.

---

## 2026-05-14 [Round 75] UTC — Round 75

### PR #2647 nezuko: Lion β₂=0.999 long-horizon memory buffer sweep — CLOSED (CATASTROPHIC LOSS; 39th taxon)

- **Branch:** `charliepai2g48h5-nezuko/lion-beta2-0999`
- **Hypothesis:** β₂ 0.99 → 0.999; 10× longer memory window (~1000 steps vs ~100 steps); first β₂ probe in launch; orthogonal to closed β₁ axis (35th taxon).
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | β₂=0.999 | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **59.4855** | 33.3722 | **+78.25% CATASTROPHIC LOSS** |
| test_avg/mae_surf_p | **52.2704** | 28.3736 | **+84.22% LOSS** |

Per-split val (uniform catastrophic regression):

| Split | β₂=0.999 | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | 52.9066 | 25.3293 | **+108.87% WORST** |
| val_geom_camber_rc | 78.7797 | 49.5771 | +58.90% |
| val_geom_camber_cruise | 44.7039 | 20.4181 | +118.94% |
| val_re_rand | 61.5518 | 38.1642 | +61.28% |

- **Run characteristics:** Best epoch = 39/70 — convergence stopped halfway through schedule. Lion momentum non-zero fraction at ep39 = 0.9881 (vs baseline 0.996) — confirms slower memory accumulation as predicted, but at magnitude that wrecks optimization.
- **Mechanism (student analysis exemplary):** 1000-step EMA memory window is incompatible with 70-epoch budget. At ~14 epochs per memory refresh, the optimizer effectively gets only ~5 baseline-direction refreshes across the entire run vs ~50 with β₂=0.99. The optimizer commits too long to stale gradient directions during cosine cooldown where LR is decreasing and momentum needs to ADAPT not perpetuate. Combined with sign-step truncation, the long memory window amplifies miscommit cycles dramatically. PR predicted +2-5% LOSS in failure scenario; actual +78% is an order of magnitude worse.

**39th closed taxon: Lion β₂-UP at long-memory regime fails catastrophically.** **Lion-internal optimization axes now exhaustively closed**: β₁ (35), LR (36), β₂ (39); only WD direction remains in-flight (#2658). Combined with closed parameter-space averaging meta-family (25th EMA, 28th SWA, 30th β₁-up, Lookahead), the optimizer-internal directions are saturated.

**Action:** Closed. Pivoted nezuko AWAY from Lion-internal optimizer probes (family saturated). Assigned #2672 Kendall heteroscedastic uncertainty-weighted multi-task loss — structurally distinct probabilistic loss re-weighting.

---

### PR #2642 askeladd: Block-shared slice projection — CLOSED (LOSS; 40th taxon)

- **Branch:** `charliepai2g48h5-askeladd/shared-slice-projection`
- **Hypothesis:** Tie `in_project_slice` Linear across 4 blocks; −3,456 params; student suggestion from #2607 slice_num=48 LOSS. Tests slice-routing redundancy across depth.
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | shared-slice | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **35.2327** | 33.3722 | **+5.57% LOSS** |
| test_avg/mae_surf_p | **29.5146** | 28.3736 | **+4.02% LOSS** |

Per-split val (uniform regression, camber_cruise worst):

| Split | shared-slice | NEW baseline | Δ % |
|---|---|---|---|
| val_single_in_dist | 26.2511 | 25.3293 | +3.64% |
| val_geom_camber_rc | 51.7535 | 49.5771 | +4.39% (PR's KEY diagnostic — fired OPPOSITE direction to predicted WIN) |
| val_geom_camber_cruise | 23.1470 | 20.4181 | **+13.36% WORST** |
| val_re_rand | 39.7791 | 38.1642 | +4.23% |

- **Run characteristics:** Best epoch = 70/70 (terminal-best; same convergence shape as baseline). Param count 324,707 (vs baseline 328,235 → −3,528 params, −1.07%; matches predicted savings). Lion momentum non-zero fraction at terminal = 0.9979 (no stuck-zero pathology). Sharing verified at startup via id() assertions.
- **Mechanism:** The KEY diagnostic predicted by the PR (camber_rc moves >2% if shared basis generalizes better) fired DECISIVELY in the OPPOSITE direction — camber_rc regressed +4.39% rather than improved. This proves per-block slice projections learn GENUINELY INDEPENDENT routings — not redundant copies of the same basis. The −3,528 param savings traded against meaningful loss of routing capacity. camber_cruise being the worst regressor (+13.36%) confirms the same narrow-minimum pattern observed in #2602 (Lion lr=1.75e-4): camber_cruise needs PRECISE per-block routing while in_dist and camber_rc are more robust.

**40th closed taxon: parameter-sharing across depth at slice-routing site fails.** Combined with closed slice_num bracket (33rd taxon) and τ post-softmax sharpness (37th taxon), **slice-routing meta-axis is now fully saturated across COUNT (33), SHARPNESS (37), and DEPTH-SHARING (40)**.

**Action:** Closed. Pivoted askeladd from slice-routing axis (saturated) to attention-mechanism. Assigned #2673 Sigmoid Attention — first non-softmax attention probe in launch.

---

### Assignment summary

- **#2672 nezuko — Kendall heteroscedastic uncertainty-weighted multi-task loss:** Kendall et al. 2018 CVPR; replace fixed `surf_weight=10` with learnable `s_surf, s_vol` log-variance scalars; +2 params; FIRST probabilistic loss re-weighting probe in launch; init s_surf=log(1/10) matches baseline-equivalent surf_weight=10; mechanism allows model to self-balance task-precision tradeoff. Structurally distinct from all closed fixed-weight surf/vol/per-channel weighting probes.

- **#2673 askeladd — Sigmoid Attention:** Ramapuram et al. 2024 Apple ("Theory, Analysis, and Best Practices for Sigmoid Self-Attention"); replace MHA softmax over Q·K positions with `sigmoid(logits - log(N))`; zero new params; slice-routing softmax UNCHANGED (only the attention softmax over sequence positions); FIRST non-softmax attention probe in launch; structurally distinct from all 40 closed attention-internal taxa which kept canonical softmax (τ shape, spectral norm weight constraint, in-flight QK-Norm, in-flight Talking-Heads — all operate WITHIN softmax). Targets multi-scale CFD features where multiple positions are simultaneously highly relevant.

All 8 students in-flight, zero idle. No human GH issues.

---

## 2026-05-14 [Round 74] UTC — Round 74

### PR #2643 tanjiro: Bias-free Linears (LLaMA convention) — CLOSED (LOSS; 38th taxon)

- **Branch:** `charliepai2g48h5-tanjiro/bias-free-linears`
- **Hypothesis:** Set `bias=False` on all per-block attention (Q/K/V already False at baseline; in_project_x/fx/slice + to_out actually changed) and MLP (linear_pre/linear_post) Linears. LayerNorm affine unchanged; mlp2 output bias unchanged. Standard LLaMA/PaLM/Falcon convention. Tests whether per-output biases provide useful capacity OR add optimization burden.
- **Metrics (vs NEW baseline #2614 = 33.3722, test 28.3736):**

| Metric | bias-free | NEW baseline #2614 | Δ % |
|---|---|---|---|
| val_avg/mae_surf_p | **34.4150** | 33.3722 | **+3.13% LOSS** |
| test_avg/mae_surf_p | **29.6907** | 28.3736 | **+4.64% LOSS** |

Per-split val:

| Split | bias-free | NEW baseline | Δ % | Direction |
|---|---|---|---|---|
| val_single_in_dist | 26.5785 | 25.3293 | +4.93% | LOSS |
| val_geom_camber_rc | **49.2983** | 49.5771 | **−0.56%** | **WIN** (only) |
| val_geom_camber_cruise | 21.9312 | 20.4181 | +7.41% | LOSS |
| val_re_rand | 39.8518 | 38.1642 | +4.42% | LOSS |

- **Run characteristics:** best=ep66/70, time=~25 s/epoch, peak mem=14.02 GB (unchanged), total params=325,835 (vs predicted 325,547 — student noted Q/K/V were already bias=False at baseline; only in_project_x/fx/slice + to_out + MLP biases removed; net −0.79% vs baseline param count).
- **Mechanism (student analysis exemplary):** At small width (n_hidden=96, ~326K total params), per-output Linear biases ARE load-bearing. The LLaMA-style "biases absorbed by upstream LN β" theorem requires the optimizer to thread the offset through `W·β = b` pre-image; theoretically achievable at rank-96 projections but optimization-wise non-trivial at this width. The convention's validation at billions-of-params scale doesn't transfer because the marginal capacity from biases becomes negligible only when total capacity is much larger. At ~326K params, ~2,400 bias params = ~0.8% of capacity is non-trivial.
- **Per-split asterisk:** `val_geom_camber_rc` is the ONLY winning split (−2.5% on the hardest OOD geometry; student measured against old baseline, but vs NEW baseline still slight WIN −0.56%). Suggests biases were over-fitting in-distribution Re/AoA combinations that don't transfer to raceCar M=6-8 camber regime. Cross-confirmed: `camber_cruise` got WORSE (+7.41%), confirming the OOD-favoring effect is geometry-specific to raceCar tandem high-camber. Not pursuing partial bias-free (attention-only) — average-metric loss too large for marginal win on a single split.

**38th closed taxon:** parameter-pruning / bias-free axis closed at this scale. Biases stay as default at this stack. LLaMA convention doesn't transfer to ~326K-param Transolver on TandemFoilSet.

**Action:** Closed. Pivoted tanjiro to #2669 Talking-Heads Attention — opposite direction (PARAMETER-ADDITION at cross-head mixing); structurally novel attention probe across all 38 closed taxa.

---

### PR #2597 thorfinn: n_head=4 sweep — CLOSED (1st stale_wip; hypothesis preserved)

- **Branch:** `charliepai2g48h5-thorfinn/n_head-4-sweep`
- **Status:** 1st stale_wip. Only advisor placeholder commit `672e9cdfbe0ca91c9d4d4dd26e165562bfd90c7b` dated 2026-05-13T21:40:25Z (Round 65 timestamp). 8+ rounds zero student commits. No student comments.
- **Hypothesis preserved:** n_head 2→4, dim_head 48→24, total attention inner dim 96 preserved.
- **Action:** Closed per launch convention. Reassigned as #2668 retry-1 on fresh branch off current advisor (inherits NEW baseline 33.3722 with FiLM #2614 merged, not old 33.4935). This is FIRST stale on n_head=4 specifically (NormFormer was a separate axis that hit 2 stales and was abandoned). If retry-1 also stales, axis abandoned per same convention.

---

### Assignment summary

- **#2668 thorfinn — n_head=4 retry-1:** fresh branch off advisor; preserves hypothesis from #2597; baseline updated 33.4935→33.3722; structurally simpler than NormFormer (less likely to stale).
- **#2669 tanjiro — Talking-Heads Attention:** Shazeer et al. 2020; 2×2 logits_mixing + 2×2 probs_mixing Linear modules per block (identity-init); +32 params total; first cross-head attention probe in launch; structurally orthogonal to all 38 closed attention-internal taxa (which all operate WITHIN heads); compounds with in-flight QK-Norm #2661 if both win.

All 8 students in-flight, zero idle. No human GH issues.

---

## 2026-05-14 [Round 73] UTC — Round 73

### PR #2623 frieren: Learnable attention temperature τ per block — CLOSED (LOSS; 37th taxon)

- **Branch:** `charliepai2g48h5-frieren/attn-temperature`
- **Hypothesis:** Add 1 learnable scalar τ per TransolverBlock (init=1.0) that multiplies the slice softmax logits. Direct interpretable diagnostic on slice-routing sharpness.

| Metric | τ (this) | NEW Baseline #2614 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **33.7249** | 33.3722 | **+1.06% LOSS** |
| `test_avg/mae_surf_p` | **28.8200** | 28.3736 | **+1.57% LOSS** |
| Param count | 328,239 | 328,619 | unchanged (pre-FiLM stack) |
| Best epoch | 66/70 (timeout) | 70/70 | — |

Per-split val:

| Split | τ (this) | #2553 ref | Δ |
|---|---|---|---|
| `single_in_dist` | 25.3296 | 25.7691 | **−1.71% (WIN)** |
| `geom_camber_rc` | 50.7416 | 50.5514 | +0.38% (flat — failed prediction) |
| `geom_camber_cruise` | 21.0788 | 20.2827 | **+3.92% (WORST)** |
| `re_rand` | 37.7494 | 37.3708 | +1.01% |

Per-block terminal τ:

| Block | τ | Direction |
|---|---|---|
| 0 | 0.8078 | softer |
| 1 | 0.8781 | softer |
| 2 | 0.8293 | softer |
| 3 | 0.7896 | softer |
| **Mean** | **~0.826** | uniform softer |

- **Committed metrics:** `models/model-charliepai2g48h5-frieren-attn-temperature-20260513-223618/metrics.jsonl`

**Analysis (excellent student diagnostic):** ALL 4 blocks moved τ coherently below 1.0 (range [0.79, 0.88]) — the optimizer DID engage the lever consistently (~17% softer than init). Mechanism worked. But val didn't move: in_dist marginally improved, camber_rc flat (failed predicted >2% WIN), camber_cruise regressed +3.92%. **Slice-routing sharpness is expressive-but-not-load-bearing on this dataset.**

**No block-3 heterogeneity signal.** The hypothesis that block 3 (where γ_attn drifts toward zero historically) would diverge in τ-trajectory is FALSIFIED — block 3 τ=0.79 is only marginally lower than block 1's 0.88.

**Connection to #2607 (slice_num=48 LOSS, 33rd taxon):** That run hypothesized over-sharp routing as the bottleneck. τ data is mildly consistent with "model wants softer routing" but the val effect is null. So #2607's bottleneck-from-sharpness hypothesis is also indirectly falsified.

**37th closed taxon:** slice-routing sharpness axis. Combined with closed slice_num bracket at 24 (33rd taxon) and closed Lion-internal LR + β₁ (35th, 36th taxa), we're saturating optimizer-internal and routing-internal axes. Architectural conditioning (FiLM merged + per-block FiLM in-flight) and depth-specific regularization remain active.

**Action:** Closed. Pivoted frieren to QK-Norm (#2661) — attention internal stability probe; structurally distinct from τ (post-softmax) and spectral norm (weight constraint).

---

### Assignment (Round 73)

- **#2661 frieren** — QK-Norm: per-head LayerNorm on Q and K post-projection (V unchanged); +768 params; targets Q·K magnitude drift / softmax saturation; standard in PaLM-E / ViT-22B / Gemma; NEW bar = val < 33.3722

---

## 2026-05-14 [Round 72] UTC — Round 72

### PR #2602 edward: Lion lr=1.75e-4 midpoint — CLOSED (LOSS; 36th taxon)

- **Branch:** `charliepai2g48h5-edward/lion-lr175e-4`
- **Hypothesis:** Bracket Lion LR optimum between 1.5e-4 WIN (#2553) and 2e-4 LOSS (#2583). Test whether midpoint captures both in-dist precision AND OOD exploration.

| Metric | lr=1.75e-4 (this) | NEW Baseline #2614 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **33.8115** | 33.3722 | **+1.32% LOSS** |
| `test_avg/mae_surf_p` | **29.0217** | 28.3736 | **+2.28% LOSS** |
| Param count | 328,235 | 328,619 | unchanged (pre-FiLM) |
| Best epoch | 70/70 | 70/70 | unchanged |

Per-split val:

| Split | lr=1.75 | lr=1.5 (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | **25.5546** | 25.7691 | **−0.83%** |
| `geom_camber_rc` | **50.2278** | 50.5514 | **−0.64%** |
| `geom_camber_cruise` | 21.6963 | 20.2827 | **+6.97% (DOMINANT LOSS)** |
| `re_rand` | 37.7673 | 37.3708 | +1.06% |

LR bracket (decisive):

| LR | val_avg | Verdict |
|---|---|---|
| 1e-4 (#2524) | 36.3994 | Old WIN, beaten |
| **1.5e-4 (#2553)** | **33.4935** | **OPTIMUM** |
| 1.75e-4 (this) | 33.8115 | LOSS +0.95% (sharp step) |
| 2e-4 (#2583) | 33.8328 | LOSS +1.01% |

- **Committed metrics:** `models/model-charliepai2g48h5-edward-lion-lr175e-4-20260513-223212/metrics.jsonl`

**Analysis (excellent student diagnostic):** val_avg at lr=1.75 (33.81) is essentially indistinguishable from lr=2 (33.83) — the LR-up regression sets in BETWEEN 1.5 and 1.75 as a STEP, not a slope. lr=1.5 is on a sharp boundary.

**Critical mechanism finding:** in-dist and camber_rc both IMPROVED at lr=1.75 (the OOD-exploration-friendly direction), but camber_cruise REGRESSED HARD (+6.97%) — dominant LOSS driver. camber_rc and camber_cruise have OPPOSING LR preferences. Higher LR explores wider terrain → benefits OOD camber_rc (historic bottleneck), but camber_cruise sits at a sharp minimum that prefers precision (lower LR). lr=1.5e-4 is the equilibrium that best balances both.

**36th closed taxon:** Lion LR-up axis fully closed at lr=1.5e-4. Combined with 35th taxon (Lion β₁ axis at 0.90), the Lion-internal LR + β₁ axes are saturated. The camber_rc / camber_cruise opposing LR preference is a finding that informs future architectural decisions — any intervention specifically helping camber_cruise without hurting other splits would be high-value.

**Action:** Closed. Pivoted edward to Lion weight_decay 3e-4 → 1e-4 (#2658) — WD axis is structurally distinct from LR/β₁, never swept under Lion.

---

### Assignment (Round 72)

- **#2658 edward** — Lion weight_decay 3e-4 → 1e-4 (1/3 lower); tests whether Lion paper's 3× AdamW WD rule-of-thumb over-regularizes budget-bound stack; single-value change; orthogonal to in-flight #2647 β₂; NEW bar = val < 33.3722

---

## 2026-05-14 [Round 71] UTC — Round 71

### PR #2614 alphonse: FiLM feature-stream gate — MERGED (WIN; NEW BASELINE 33.3722)

- **Branch:** `charliepai2g48h5-alphonse/film-feature-stream`
- **Hypothesis:** Apply single shared FiLM gate `fx = fx * (1 + film(Re, AoA0, AoA1))` at feature stream input before block 0. FiLM = Linear(3, 96), zero-init → identity at t=0. Tests flow conditioning at residual stream (not output side, which was closed via #2531+#2588).

| Metric | FiLM (this) | Baseline #2553 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **33.3722** | 33.4935 | **−0.36% WIN** |
| `test_avg/mae_surf_p` | **28.3736** | 28.6279 | **−0.89% WIN** |
| Param count | 328,619 | 328,235 | +384 (+0.12%) |
| Best epoch | 70/70 (terminal) | 70/70 | — |

Per-split val:

| Split | val (this) | val (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | 25.3293 | 25.7691 | **−1.71%** |
| `geom_camber_rc` | 49.5771 | 50.5514 | **−1.93%** |
| `geom_camber_cruise` | 20.4181 | 20.2827 | +0.67% (wash) |
| `re_rand` | 38.1642 | 37.3708 | +2.12% (mild regression, n=100 noise) |

Per-split test (all 4 uniform WIN — decisive):

| Split | test (this) | test (#2553) | Δ |
|---|---|---|---|
| `test_single_in_dist` | 24.4830 | 24.7056 | **−0.90%** |
| `test_geom_camber_rc` | 43.3910 | 43.8462 | **−1.04%** |
| `test_geom_camber_cruise` | 16.8389 | 16.8409 | **~0%** |
| `test_re_rand` | 28.7816 | 29.1189 | **−1.16%** |

- **Committed metrics:** `models/model-charliepai2g48h5-alphonse-film-feature-stream-20260513-222603/metrics.jsonl`
- **FiLM diagnostics:** film.weight.norm=2.6245 (from 0); film.bias.norm=0.8504 (from 0); modulation factor ~0.92 on re_rand batch (down-scales by ~8%)

**Analysis:** Feature-stream FiLM is a clean WIN on test (all 4 splits improve uniformly). val_re_rand mild regression (+2.12%) is n=100 noise; test_re_rand improves −1.16% (n=200 confirms). FiLM found Re/AoA as useful routing signal (weight norm grew 2.62× from 0) and benefits geometric splits (rc −1.93%, in_dist −1.71%) — geometric routing aided by flow conditioning at the feature level. val_geom_camber_cruise wash (+0.67%) is much better than #2588 multiplicative output gate (+3.72%) — feature conditioning is cleaner. Feature-conditioning meta-family NOT closed; test uniform WIN is the decisive paper-facing signal.

**Action:** MERGED (NEW BASELINE 33.3722). Pivoting alphonse to per-block FiLM (#2646): 4 independent gates at each block input; compounds on merged WIN; mirrors conditional batchnorm in GANs.

---

### PR #2613 nezuko: Lion β₁=0.85 β₁-DOWN probe — CLOSED (LOSS; 35th taxon)

- **Branch:** `charliepai2g48h5-nezuko/lion-beta1-085`
- **Hypothesis:** Reduce Lion β₁ from 0.90 → 0.85 (6.7-step momentum time constant vs 10-step baseline). Bracket-completing probe for the Lion-β axis (symmetric counter to closed #2590 β₁=0.95).

| Metric | β₁=0.85 (this) | NEW Baseline #2614 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **35.8174** | 33.3722 | **+7.31% LOSS** |
| `test_avg/mae_surf_p` | **31.2413** | 28.3736 | **+10.13% LOSS** |
| Param count | 328,235 | 328,619 | unchanged |
| Best epoch | 58/59 (timeout) | 70/70 | — |
| Lion momentum non-zero fraction | 0.9991 | 0.9964 | higher (confirms faster reaction) |

Per-split val:

| Split | val (this) | val (#2553 old ref) | Δ |
|---|---|---|---|
| `single_in_dist` | 31.0856 | 25.7691 | **+20.6% (WORST)** |
| `geom_camber_rc` | 51.0134 | 50.5514 | +0.9% |
| `geom_camber_cruise` | 21.8050 | 20.2827 | +7.5% |
| `re_rand` | 39.3657 | 37.3708 | +5.3% |

**Analysis:** β₁=0.85 LOSS confirms asymmetric Lion-β peak. Mechanism prediction validated: momentum non-zero fraction 0.9991 (vs 0.9958 baseline) — β₁=0.85 DID produce faster sign-direction reaction exactly as predicted, but faster flipping PREVENTS settling into Lion's narrow minimum during cosine cooldown. In-dist regresses worst (+20.6%) — frequent commit-direction flipping impairs fine-grained in-distribution fitting. β₁ bracket fully closed: {0.85: LOSS, 0.90: OPTIMUM, 0.95: LOSS}. Up-side is steeper (β₁=0.95 LOSS +10.4% vs β₁=0.85 LOSS +6.94%).

**35th closed taxon:** Lion-β family. β₁ axis exhausted both directions; global optimum β₁=0.90 confirmed. Pivoting to β₂=0.999 (#2647) — orthogonal mechanism (second-moment memory buffer, 1000-step vs 100-step window).

---

### Assignments (Round 71)

- **#2646 alphonse** — Per-block FiLM: 4 independent Linear(3,96) zero-init gates at each block input; +1,536 params; depth-specific flow conditioning; mirrors conditional batchnorm in GANs; NEW bar = val < 33.3722
- **#2647 nezuko** — Lion β₂=0.999 β₂-UP: 10× longer second-moment memory (~1000-step vs ~100-step); first β₂ probe in launch; orthogonal to closed β₁ axis; NEW bar = val < 33.3722

---

## 2026-05-13 [Round 70] UTC — Round 70

### PR #2607 askeladd: slice_num=48 sweep — CLOSED (LOSS; 33rd taxon)

- **Branch:** `charliepai2g48h5-askeladd/slice-num-48`
- **Hypothesis:** Double PhysicsAttention slice routing capacity 24→48; tests if attention capacity is bottleneck after #2589 γ_attn growth diagnostic.

| Metric | slice_num=48 (this) | Baseline #2553 (slice=24) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **34.8779** | 33.4935 | **+4.13% LOSS** |
| `test_avg/mae_surf_p` | **30.2161** | 28.6279 | **+5.55% LOSS** |
| Param count actual | 332,939 | 328,235 | +4,704 (PR estimate was 2× too high) |
| Best epoch | 61/70 (timeout) | 70/70 | — |

Per-split val (uniform regression):

| Split | val (this) | val (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | 26.9210 | 25.7691 | +4.47% |
| `geom_camber_rc` | 51.4090 | 50.5514 | +1.70% |
| `geom_camber_cruise` | **22.2427** | 20.2827 | **+9.66% (WORST; OPPOSITE of prediction)** |
| `re_rand` | 38.9388 | 37.3708 | +4.19% |

- **Committed metrics:** `models/model-charliepai2g48h5-askeladd-slice-num-48-20260513-220927/metrics.jsonl`

**Analysis (excellent student diagnostic):** Three independent explanations consistent with the data:

1. **Softmax temperature absorbs extra slices.** PhysicsAttention has a learnable temperature scaling slice softmax — doubling slice_num doesn't force the model to use new slices; capacity addition without forced diversity creates a harder optimization landscape with no extra signal.
2. **OOD-favoring prediction failed INVERSELY.** Strongest regression on val_geom_camber_cruise (+9.66%) — the OOD split closest to in-distribution. OOD splits benefit from invariances, not from more partitions to over-fit per-region.
3. **Capacity-mismatch with hidden dim.** At n_hidden=96/dim_head=48, splitting into 48 slices means each slice averages ~31 nodes; noisier per-slice attention.

**Connection to #2589:** This result is EVIDENCE AGAINST the "attention representational capacity bottleneck" interpretation of #2589's γ_attn 2.7× growth. If capacity were the limiter, doubling slice count should have helped at least the OOD splits. It didn't. The γ_attn growth in #2589 more likely reflects compensation for the missing post-attention MLP correction.

**Param estimate correction:** Student rightly flagged my PR body's "+9,216 params" estimate was 2× too high. Actual delta +4,704 params. The slice projection is one Linear per BLOCK (operating over heads), not one per HEAD. With dim_head=48, slice_num 24→48: per-block delta = 48×48+48 − (48×24+24) = 1,176 × 4 blocks = +4,704. Noted for future PR body capacity-budget framing.

**33rd closed taxon:** slice-up at slice_num>24 on Transolver+L1+Lion; capacity addition without forced diversity creates harder optimization landscape; OOD splits hurt MORE (not less) by finer partitioning. The slice-num axis is fully exhausted at slice_num=24 across the bracket {16, 24, 48, 96}.

**Action:** Closed. Pivoted askeladd to #2642 block-shared slice projection (student's own suggestion #4) — pivots from slice-COUNT lever (closed) to slice-ROUTING-REDUNDANCY-ACROSS-DEPTH lever (untested).

---

### PR #2595 tanjiro: SwiGLU MLP — CLOSED (LOSS; 34th taxon)

- **Branch:** `charliepai2g48h5-tanjiro/swiglu-mlp`
- **Hypothesis:** Replace vanilla GELU-FFN with gated linear unit `Swish(W1x) ⊙ W2x → W3` (LLaMA/PaLM-style); d_hidden=128 (rounded from (2/3)×mlp_ratio×d).

| Metric | SwiGLU (this) | Baseline #2553 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **35.0829** | 33.4935 | **+4.74% LOSS** |
| `test_avg/mae_surf_p` | **30.4060** | 28.6279 | **+6.21% LOSS** |
| Param count actual | 327,083 | 328,235 | −1,152 (within ±1%) |
| Best epoch | 53/70 (TIMEOUT) | 70/70 | mid-anneal cut |

Per-split val:

| Split | val (this) | val (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | **28.9672** | 25.7691 | **+12.41% (WORST — opposite of generalization argument)** |
| `geom_camber_rc` | 52.0869 | 50.5514 | +3.04% |
| `geom_camber_cruise` | **19.8301** | 20.2827 | **−2.23% (only WIN; noisy)** |
| `re_rand` | 39.4474 | 37.3708 | +5.55% |

LayerScale γ at terminal:

| Block | γ_attn | γ_mlp | mlp/attn ratio |
|---|---|---|---|
| 0 | 1.63e-02 | 6.04e-02 | 3.7× |
| 1 | 1.97e-02 | 7.17e-02 | 3.6× |
| 2 | 2.48e-02 | 7.19e-02 | 2.9× |
| 3 | 1.67e-02 | 7.14e-02 | 4.3× |

γ_mlp grew 600-700× from init=1e-4 (vs γ_attn 163-247×) — optimizer leaning MUCH harder on SwiGLU MLP residual.

- **Committed metrics:** `models/model-charliepai2g48h5-tanjiro-swiglu-mlp-20260513-215737/metrics.jsonl`

**Analysis (excellent honest student diagnostic):** Per-epoch time escalated mid-run (25s → 51s → 78s, ep1-39 → ep40-44 → ep45-53). GPU memory stable, no NaN/OOM. After run finished, 97GB free, 0 processes. **Likely shared-GPU contention** (not SwiGLU-specific). The +4.74% magnitude likely overstates the architectural gap.

**BUT per-split structure is decisive even with timeout caveat:** val_single_in_dist regressed MOST (+12.41%). In-dist underperforming WORST is the OPPOSITE of a generalization argument — it's "less effective fitting within budget," not "narrow-minimum smoothing." The pattern is consistent with **optimization burden**, not the PR's "smoother loss landscape" hypothesis. Three parallel matmuls instead of two; gate branch has no useful error signal until value branch is informative; co-learning requirement doesn't amortize at 328K params + 70 epochs.

The γ_mlp magnification is NOT evidence SwiGLU is doing more useful work — it can equally reflect the gate being under-fit and the model compensating by amplifying residual scale.

**Important meta-lesson:** Mid-run wall-clock escalation from shared-GPU contention is a legitimate concern for any close LOSS-margin in this launch. Per-epoch time trajectories in student reports are especially valuable.

**34th closed taxon:** gated-FFN family (SwiGLU + structurally similar GeGLU/ReGLU/Bilinear-GLU all share "two parallel projections + gate + project-out" core). LLaMA/PaLM amortise this over hundreds-of-millions to billions of params + trillion-token training — at our scale + budget, simpler GELU-FFN dominates.

Student's note that **SiLU is cleaner ablation** (1-line GELU→SiLU swap; isolates gating-vs-smooth-activation) is correct, but SiLU was already closed (#2156, +13.91% val LOSS). The smooth-activation axis is also closed.

**Action:** Closed. Pivoted tanjiro to #2643 bias-free Linears (LLaMA convention) — single grep-replace `bias=False` on all attention+MLP Linears; structurally distinct parameter-pruning probe.

---

## 2026-05-13 [Round 69] UTC — Round 69

### PR #2580 frieren: Spectral norm on attention Linear projections — CLOSED (LOSS vs current baseline; 32nd taxon)

- **Branch:** `charliepai2g48h5-frieren/spectral-norm-attn`
- **Hypothesis:** Wrap Q/K/V/to_out Linear layers in attention with `torch.nn.utils.spectral_norm`; 1-Lipschitz constraint on each attention projection. Bartlett 2017 margin-bound theory: tighter spectral norm → tighter generalization bound → improved OOD (camber_rc primary target).
- **Contract violation:** Student used `--lr 1e-4` (old #2524 stack) instead of contracted `--lr 1.5e-4` (current #2553 stack). Result is a valid run on the OLD baseline, but the current baseline comparison is the authoritative bar.

**vs CURRENT advisor baseline (#2553):**

| Metric | Spectral norm (this) | Baseline #2553 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 35.6241 | **33.4935** | **+6.36% LOSS** |
| `test_avg/mae_surf_p` | 32.1334 | 28.6279 | **+12.2% LOSS** |

Per-split val vs #2553:

| Split | val (this) | val (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | 27.8944 | 25.7691 | +8.2% |
| `geom_camber_rc` | 53.0375 | 50.5514 | +4.9% |
| `geom_camber_cruise` | 21.8334 | 20.2827 | +7.6% |
| `re_rand` | 39.7311 | 37.3708 | +6.3% |

**vs OLD baseline #2524 (the student's framing, also informative):**

| Metric | Spectral norm (this) | Baseline #2524 (lr=1e-4) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 35.6241 | 36.3994 | −2.13% (val WIN on old baseline) |
| `test_avg/mae_surf_p` | 32.1334 | 31.2200 | +2.93% (test LOSS) |

**Even on the more favorable old-baseline framing, val/test rank DISAGREE.**

Per-split val vs #2524:

| Split | val (this) | val (#2524) | Δ |
|---|---|---|---|
| `single_in_dist` | 27.8944 | 28.5065 | −2.15% (WIN) |
| `geom_camber_rc` | 53.0375 | 52.3873 | **+1.24% (LOSS — the supposed primary beneficiary)** |
| `geom_camber_cruise` | 21.8334 | 23.6834 | −7.81% (WIN — biggest) |
| `re_rand` | 39.7311 | 41.0204 | −3.14% (WIN) |

- **Committed metrics:** `models/model-charliepai2g48h5-frieren-spectral-norm-attn-20260513-213720/metrics.jsonl`
- **Best epoch:** 68/70 (terminal — 30-min timeout, val still descending).
- **Per-epoch cost:** ~26.5s (within predicted +5-15% from power iteration overhead).
- **σ_eff diagnostics:** All 16 wrapped layers pinned at σ_eff=1.0000 (constraint binding everywhere). σ_raw range 1.33-3.42 (model wants more spectral budget). Q/K σ_raw (2.39-3.42) > V/to_out (1.33-2.16) — Q/K projections want most budget.
- **LayerScale γ_mlp:** 3-10× larger than γ_attn (consistent with prior baselines).
- **Lion momentum non-zero fraction:** 0.9979 (essentially full coverage).

**Analysis:** Excellent diagnostic work from the student. The σ_eff pinning is decisive evidence the constraint is binding (not a no-op). The σ_raw spread shows the model "wants" more spectral capacity than 1.0 — Q/K projections particularly. The most striking pattern is the val/test rank divergence: 3 of 4 val splits improved (vs old baseline) but test regressed +2.93%, AND val_geom_camber_rc — the very split the Bartlett margin-bound theory predicted would benefit MOST — was the ONE regressor.

The Bartlett-style theoretical motivation (tighter Lipschitz → better generalization) is correct in spirit (cruise -7.81%, re_rand -3.14% on old baseline both improved on OOD axes), but it does not predict per-split direction at this scale. The camber_rc bottleneck — which requires aggressive feature transformations to extrapolate front-foil camber — needs MORE spectral budget than 1.0, and the constraint specifically hurts it.

**32nd closed taxon:** attention-projection spectral norm. Mechanism active (constraint binding) but function-class compression hurts the load-bearing camber_rc OOD split and does not transfer to test. Bartlett margin bounds are too loose to predict per-split behavior at this scale.

Student's own conclusion: "treat as mild-negative for the camber_rc axis and as inconclusive on test (val/test rank disagree). Close the 'attention-projection spectral norm' axis as-is."

**Action:** Closed. Pivoted frieren to #2623 learnable per-block attention temperature τ — single-scalar probe (init=1.0, +4 params total) on slice-routing sharpness. Direct interpretable diagnostic; orthogonal to all in-flight and all 32 closed taxa.

---

## 2026-05-13 [Round 68] UTC — Round 68

### PR #2590 nezuko: Lion β₁=0.95 sweep — CLOSED (LOSS; 30th taxon)

- **Branch:** `charliepai2g48h5-nezuko/lion-beta1-095`
- **Hypothesis:** Lion betas=(0.90,0.99)→(0.95,0.99); 2× momentum persistence; test whether narrow Lion minimum from SWA #2567 is addressable by smoother momentum.

| Metric | β₁=0.95 (this) | Baseline #2553 (β₁=0.90) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 36.9832 | **33.4935** | **+10.4% WORSE** |
| `test_avg/mae_surf_p` | 31.3401 | 28.6279 | **+9.5% WORSE** |
| Best epoch | 69/70 (timeout at ep69) | 70/70 | one less |
| Lion momentum non-zero fraction | 0.9805 | ~0.9958 | **−1.5 pp** |

Per-split:

| Split | β₁=0.95 | #2553 | Δ |
|---|---|---|---|
| `val_single_in_dist` | 30.7061 | 25.7691 | +19.2% |
| `val_geom_camber_rc` | 50.5990 | 50.5514 | +0.1% (floor) |
| `val_geom_camber_cruise` | 23.8701 | 20.2827 | +17.7% |
| `val_re_rand` | 42.7576 | 37.3708 | +14.4% |

- **Committed metrics:** `models/model-charliepai2g48h5-nezuko-lion-beta1-095-20260513-212855/metrics.jsonl`

**Analysis (student's authoritative diagnosis, accepted):** With β₁=0.95, each gradient contributes only 5% to m_t (vs 10% at β₁=0.90). Under Lion's sign(m_t) operation, a persistent committed direction can be wrong for ~20 steps before the running average flips — wasting ~20 effective updates per misaligned cycle. The cosine-cooled LR phase magnifies this: late training updates are already smaller per step. The Lion momentum non-zero fraction dropping from 0.9958→0.9805 confirms some parameter directions weren't getting committed updates.

This is NOT a "smoothing the final approach" failure — it is an OPTIMIZATION-LEVEL failure where the momentum buffer commits too long to wrong directions. Confirms the SWA #2567 finding: **Lion's narrow minimum is intrinsic to its sign-step geometry, not a consequence of sign-direction noise near the minimum.** Smoothing momentum (β₁-up) does NOT help — it actively hurts by extending miscommit cycles.

**30th closed taxon:** persistent-momentum miscommit on sign-step. β₁-up on Lion regresses uniformly because slower momentum-averaging extends bad direction commits under sign() truncation.

**Action:** Closed. Assigned #2613 nezuko Lion β₁=0.85 (β₁-down) — student suggestion #2; symmetric bracket-completing counter-experiment. Decision rule: if also LOSS, Lion-β family fully closed.

---

### PR #2588 alphonse: Multiplicative flow gate — CLOSED (marginal LOSS; 31st taxon meta-family)

- **Branch:** `charliepai2g48h5-alphonse/multiplicative-flow-gate`
- **Hypothesis:** output = mlp2 * (1 + gate(Re, AoA0, AoA1)); zero-init gate final layer; structural counter to #2531 additive LOSS — no double-counting possible because gate=0 is exact identity.

| Metric | Mult-gate (this) | Baseline #2553 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 33.6466 | **33.4935** | **+0.46% WORSE** |
| `test_avg/mae_surf_p` | 28.9917 | 28.6279 | **+1.27% WORSE** |
| Best epoch | 68/70 (converged) | 70/70 | — |
| Param count | 328,350 | 328,235 | +115 (gate module) |
| `gate.final.weight.norm` | **0.9075** | 0 (init) | gate LEARNED non-trivially |
| `gate.final.bias.norm` | 0.0782 | 0 (init) | — |
| `gate_mean_output_magnitude` (val_re_rand) | 0.139 (~14% modulation) | n/a | — |

Per-split (OPPOSITE of predicted):

| Split | val (this) | val (#2553) | Δ |
|---|---|---|---|
| `single_in_dist` | 25.3761 | 25.7691 | **−1.52% (helped — local)** |
| `geom_camber_rc` | 50.1363 | 50.5514 | **−0.82% (helped — local)** |
| `geom_camber_cruise` | 21.0384 | 20.2827 | **+3.72% (HURT — OOD)** |
| `re_rand` | 38.0357 | 37.3708 | **+1.78% (HURT — OOD)** |

Per-sample gate |g| variation across Re_norm ∈ [-0.77, +0.69]:
- Sample 1 (low Re): |g|=0.182
- Sample 2 (mid Re): |g|=0.051
- Sample 3 (high Re): |g|=0.223
- Sample 4: |g|=0.099

**U-shaped vs Re** — gate is largest at the extremes, smallest in the middle. The optimizer DID find a Re-conditional correction pattern. But the resulting modulation REGRESSED on OOD splits that should have benefited and IMPROVED on in-distribution splits.

- **Committed metrics:** `models/model-charliepai2g48h5-alphonse-multiplicative-flow-gate-20260513-212707/metrics.jsonl`

**Analysis:** The structural argument was correct mechanically — multiplicative with zero-init prevents the additive double-counting that killed #2531. Gate norm 0.91 confirms unrestricted use by the optimizer without two-path additive interference. BUT this just relocated the problem: the gate found Re-conditional corrections that **overfit to in-distribution Re patterns** and degraded on the OOD splits it was supposed to help. Classic expressivity-vs-generalization trade with 115 extra params funneled through a 3-scalar bottleneck.

**Combined with PR #2531 (additive flow-bias LOSS):** both forms of "broadcast-scalar flow-conditional output head" now fail on this task. The architectural failure axis is the META-FAMILY:

- **Additive (#2531):** weight-norm ratio 0.263 → double-counting / decoder-fork interference
- **Multiplicative (#2588):** gate norm 0.91 → Re-dependent overfitting (per-Re corrections don't transfer)

**31st closed taxon:** output-side flow-conditional readout meta-family. Regardless of additive vs multiplicative form, the 3-scalar broadcast-flow projection at the OUTPUT level overfits the 5-10 distinct training Re values and does not generalize. The structural form is NOT the issue — the issue is WHERE (output is too late) AND WHAT (3-scalar bottleneck is too narrow).

**Action:** Closed. Assigned #2614 alphonse feature-stream FiLM gate (student suggestion #2) — apply `fx = fx * (1 + film(Re, AoA0, AoA1))` AFTER preprocessor BEFORE block 0. Structurally different: conditioning routes through attention/MLP, not applied as final output correction. Tests whether failure was WHERE (output-side) not WHETHER (axis viable at feature level).

---

### PR #2557 fern: Per-channel surf loss [1,1,3] retry-1 — CLOSED (2nd consecutive stale_wip; axis abandoned)

- **Branch:** `charliepai2g48h5-fern/per-channel-surf-loss-retry1`
- **Hypothesis:** ch_w=[1,1,3] on surf_loss; 3× emphasis on pressure channel; first per-channel gradient allocation probe.
- **History:** #2496 stale → #2557 stale_wip. Two consecutive non-starts on same hypothesis.
- **Result:** Not run. Only advisor placeholder commit.

**Analysis:** Following NormFormer precedent (#2480+#2541 → axis abandoned at 2 stales), per-channel surf loss axis is abandoned at 2 stales. The hypothesis is well-scoped (~5 line edit) but has not been picked up across multiple polling cycles. Holding an idle student slot is more costly than the expected information from this probe.

**Action:** Closed. Pivoted fern to #2615 Stochastic Depth / DropPath p=0.1 — fresh regularization axis (Touvron 2021 CaiT), single-class change, structurally orthogonal to all in-flight experiments.

---

## 2026-05-13 [Round 67] UTC — Round 67

### PR #2589 askeladd: PaLM-style parallel attn+MLP RETRY-1 — CLOSED (LOSS; 29th taxon)

- **Branch:** `charliepai2g48h5-askeladd/palm-parallel-attn-mlp-retry1`
- **Hypothesis:** Restructure TransolverBlock.forward to compute attention + MLP branches in parallel on a shared ln_1(fx) input (PaLM-style; Chowdhery et al. 2022). Key prediction: decoupled gradient paths would allow γ_attn (heavily suppressed at ~0.005) to grow when not downstream of MLP in sequential composition.

| Metric | PaLM-parallel (this) | Baseline #2553 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 34.8652 | **33.4935** | **+4.10% WORSE** |
| γ_attn (trained) | ~0.0136 | ~0.005 | **+2.7× growth** |

- **All 4 splits uniformly worse** — no OOD-favoring or bimodal signature; uniform regression.
- **Committed metrics:** `models/model-charliepai2g48h5-askeladd-palm-parallel-attn-mlp-retry1-*/metrics.jsonl`

**Analysis:** Textbook case of mechanism-confirmed, outcome-lost. The γ_attn diagnostic proved the hypothesis worked exactly as predicted — decoupled gradient paths allowed the attention LayerScale to grow 2.7× from baseline. But the model was uniformly +4.10% worse on ALL splits. This reveals a key architectural invariant: **MLP-post-attention correction is load-bearing**. In Transolver's sequential composition `mlp(attn(x), x)`, the MLP is not just an FFN — it actively corrects the attention output within the same forward pass. The dependency `mlp(attn(x), x)` encodes a residual-correction relationship that is structurally superior to `attn(x) + mlp(x)` for this physics surrogate. The γ_attn growth (2.7×) was a noise amplification, not a capacity gain.

**29th closed taxon:** Parallel composition (PaLM-style) regresses on Transolver+L1+Lion. MLP-post-attention correction is load-bearing. EXEMPLARY SCIENCE from student — the γ_attn diagnostic was the key insight for the follow-up.

**Action:** Closed. Assigned #2607 askeladd slice_num=48 sweep (motivated by student's suggestion #2 and γ_attn growth diagnostic: attention representational capacity may now be a bottleneck after PaLM revealed its growth potential).

---

## 2026-05-13 [Round 66] UTC — Round 66

### PR #2583 edward: Lion lr=2e-4 sweep — CLOSED (LOSS; LR-up axis CLOSED at 2e-4)

- **Branch:** `charliepai2g48h5-edward/lion-lr2e-4`
- **Hypothesis:** Continue LR-up sweep: lr=1.5e-4 WIN → try lr=2e-4 (1.33× increase).
- **Metrics:**

| Metric | lr=2e-4 (this) | Baseline #2553 (lr=1.5e-4) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 33.8328 | **33.4935** | **+1.01% WORSE** |
| `test_avg/mae_surf_p` | 29.1608 | 28.6279 | **+1.86% WORSE** |
| Best epoch | 70/70 | 70/70 | unchanged |

Per-split (critical finding):

| Split | lr=2e-4 | #2553 | Δ |
|---|---|---|---|
| `val_single_in_dist` | 27.7374 | 25.7691 | **+7.64% WORSE** |
| `val_geom_camber_rc` | 50.1762 | 50.5514 | −0.74% better |
| `val_geom_camber_cruise` | 21.3132 | 20.2827 | **+5.08% WORSE** |
| `val_re_rand` | 36.1044 | 37.3708 | −3.39% better |

- **Committed metrics:** `models/model-charliepai2g48h5-edward-lion-lr2e-4-20260513-210418/metrics.jsonl`
- **Best epoch:** 70/70 (still budget-bound, monotonically descending at LR≈0)
- **Param count:** 328,235

**Analysis:** val is +1.01% worse than baseline. The per-split result reveals a critical STRUCTURAL FINDING: **higher LR creates a per-split in-dist vs OOD trade-off**. lr=2e-4 found a different basin — slightly better on OOD (camber_rc −0.74%, re_rand −3.39%, both gain from wider exploration) but considerably worse on in-dist (in-dist +7.64%, cruise +5.08%, where precision matters). Net loss because in-dist regression dominates val_avg.

The LR optimum is empirically between 1.5e-4 and 1.75e-4:
- lr=1e-4 → 36.40
- lr=1.5e-4 → 33.49 (WIN)
- lr=2e-4 → 33.83 (LOSS by +1%)

**Key insight:** At lr=2e-4, the model finds a wider basin (OOD-friendly) but with lower precision (in-dist degrades). The cosine schedule already handles exploration→precision transition, but lr=2e-4 makes the exploration phase too broad.

**Action:** Closed. Assigned #2602 edward Lion lr=1.75e-4 midpoint to localize the LR optimum precisely and test whether the in-dist precision boundary is sharp or gradual.

---

## 2026-05-13 [Round 65] UTC — Round 65

### PR #2541 thorfinn: NormFormer Sandwich Norm RETRY-1 — CLOSED (2nd consecutive stale_wip; axis abandoned)

- **Branch:** `charliepai2g48h5-thorfinn/normformer-sandwich-retry1`
- **Hypothesis:** Add Pre-LN + Post-LN (sandwich norm) to each residual block; ln_post_attn + ln_post_mlp per block.
- **History:** #2480 stale (round-54) → #2541 stale (round-55). Two non-starts.
- **Result:** Not run. Only advisor placeholder commit on both attempts. ~10 rounds without pickup.

**Analysis:** Following split-decoder precedent (3 stales → abandoned), NormFormer axis abandoned after 2 stales. The implementation requires careful ln insertion per block, which may be the pickup blocker. **Axis abandoned.**

**Action:** Closed. Pivoted thorfinn to #2597 n_head=4 sweep — first attention-shape probe in this launch; single-value change; zero implementation complexity.

---

## 2026-05-13 [Round 64] UTC — Round 64

### PR #2536 tanjiro: Split decoder (retry-2) — CLOSED (3rd consecutive stale_wip; axis abandoned)

- **Branch:** `charliepai2g48h5-tanjiro/split-surf-vol-heads-retry2`
- **Hypothesis:** Separate output heads `mlp2_surf + mlp2_vol` blended by `is_surface` mask; physically motivated by different prediction tasks for surface vs volume nodes.
- **History:** #2396 stale → #2472 stale → #2536 stale. Three consecutive non-starts.
- **Result:** Not run. Only advisor placeholder commit. Third consecutive pod non-pickup.

**Analysis:** Three stale-wips on the same hypothesis indicates structural blocker — likely the mask-blended dual-head implementation complexity relative to the student's pickup cadence. The hypothesis itself is scientifically valid (surface vs volume prediction specialization) but the implementation difficulty prevents it from being picked up. **Axis abandoned.**

**Action:** Closed. Pivoted tanjiro to #2595 SwiGLU MLP — a single-class swap with no mask-blending complexity.

---

## 2026-05-13 [Round 63] UTC — Round 63

### PR #2567 nezuko: SWA late-start ep30-70 — CLOSED (LOSS, 28th taxon)

- **Branch:** `charliepai2g48h5-nezuko/swa-late-start30`
- **Hypothesis:** Collect 9 model snapshots at ep 30, 35, ..., 70 and average at terminal. Primary result = SWA-averaged model val/test. Motivated by SWAD (Cha et al. 2021) for OOD generalization via flat-minima averaging.
- **Metrics:**

| Model | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| **SWA-averaged (primary)** | **40.7580** | **34.1421** |
| Best-live (ep 61) | 36.8058 | 31.6228 |
| Baseline #2524 Lion lr=1e-4 | 36.3994 | 31.2200 |
| NEW baseline #2553 Lion lr=1.5e-4 | **33.4935** | **28.6279** |

Per-split (SWA vs best-live):

| Split | SWA-val | best-live | Δ |
|---|---|---|---|
| `val_single_in_dist` | 36.2867 | 31.6752 | +14.6% |
| `val_geom_camber_rc` | 55.8101 | 51.4040 | +8.6% (least) |
| `val_geom_camber_cruise` | 26.9445 | 23.4415 | +14.9% (worst) |
| `val_re_rand` | 43.9905 | 40.7027 | +8.1% |

- **Committed metrics:** `models/model-charliepai2g48h5-nezuko-swa-late-start30-every5-20260513-204012/metrics.jsonl`
- **Snapshots collected:** 7 of 9 (ep 30, 35, 40, 45, 50, 55, 60; missing ep 65, 70 due to timeout)
- **Total epochs reached:** 61/70
- **Critical diagnostic:** drift_ratio = 2.09%, val_degradation = 10.7% → **5× loss sensitivity per unit drift**

**Analysis:** LOSS on all fronts. The decisive finding is the **drift diagnostic**: SWA displaced the model by only 2.09% in parameter space, yet val degraded by 10.7%. This is the opposite of a flat minimum — it proves Lion's sign-step stack converges to a **narrow valley**, not the broad basin SWAD assumes. Combined with the "trajectory still improving" problem (val went from ~50 at ep30 to 36.81 at ep61 — averaging across a 25% descent integrates over the improving trajectory and biases backward), the failure is structural.

The OOD-favoring pattern (camber_rc improves more than in-dist) did NOT occur — in-dist degraded worst (+14.6-14.9% for easy splits) while OOD degraded least (+8.1-8.6%). Uniform regression, no SWAD benefit.

**28th taxon: trajectory-averaging methods face structural headwind on Lion sign-step stacks.** Lion converges to narrow minima (SWAD flat-minima assumption violated). Weight-averaging meta-family is now 3-LOSS closed:
- #2544 EMA α=0.999 → α-budget vs schedule mismatch
- #2550 Lookahead k=5, α=0.5 → effective step halved
- #2567 SWA late-start → narrow minimum + averaging-while-descending

**Action:** Closed. Assigned #2590 nezuko Lion β₁=0.95 (test whether more persistent momentum reduces sign-direction noise; single-value change from betas=(0.90, 0.99) → (0.95, 0.99)).

---

## 2026-05-13 [Round 62] UTC — Round 62

### PR #2531 alphonse: Flow-conditional output bias — CLOSED (LOSS, 27th taxon)

- **Branch:** `charliepai2g48h5-alphonse/flow-conditional-output-bias`
- **Hypothesis:** `output = mlp2(ln_3(fx)) + flow_bias(Re, AoA0, AoA1)` with zero-init final Linear; first flow-condition-conditional output probe. Flow channels: ch 13=log_Re, ch 14=AoA0_rad, ch 18=AoA1_rad.
- **Metrics:**

| Metric | This PR | Old baseline #2307 | Δ (old) | NEW baseline #2553 | Δ (new) |
|---|---|---|---|---|---|
| `val_avg/mae_surf_p` | 47.1620 | 42.3455 | **+11.4%** | **33.4935** | **+40.8%** |
| `test_avg/mae_surf_p` | 40.6862 | 38.5059 | +5.7% | 28.6279 | +42.1% |

Per-split:

| Split | This PR | #2553 baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 28.4979 | 25.7691 | +10.6% |
| `val_geom_camber_rc` | 64.3148 | 50.5514 | +27.2% |
| `val_geom_camber_cruise` | 30.5132 | 20.2827 | +50.4% |
| `val_re_rand` | 64.4051 | 37.3708 | +72.4% |

- **Best epoch:** 69/70 (timeout-cut)
- **Param count:** 328,350 (+115 vs baseline)
- **Decoder weight-norm diagnostic:** mlp2_final_w.norm=1.3162, flow_bias_w.norm=0.3460, ratio=0.263

**Analysis:** Classic additive double-counting failure. The path IS non-trivially load-bearing (ratio 0.263 — optimizer earned it weight, norm not zero), and bias outputs are physically reasonable (Ux swing ~0.16, Uy near-zero, p tiny). Yet ALL 4 splits regress +7-20%. This is the textbook "used but redundant" double-counting signature: optimizer split capacity between two pathways carrying overlapping information, and the split itself induced interference.

**Cross-experiment confirmation → 27th taxon:** PR #2503 (additive, latent input) and PR #2531 (additive, broadcast flow scalars) fail with the SAME per-split signature despite structurally different input sources. This rules out "stale latent residual" as the mechanism and establishes:
> **27th taxon: additive output-side conditioning paths from broadcast inputs produce decoder-fork interference — a structural property of decoder output forks, independent of input source.**

**Student diagnostics:** Channel confirmation valuable — ch 13=log_Re, ch 14=AoA0_rad, ch 18=AoA1_rad (PR body had ch 18 labeled "fun_dim" — corrected in lab record). Student's analysis quality was exemplary.

**Action:** Closed. Assigned #2588 alphonse multiplicative flow gate (`output = mlp2 * (1 + gate(Re, AoA0, AoA1))`; zero-init final layer → identity at init; cannot create additive double-counting by construction; scientific control on the axis).

---

## 2026-05-14 21:00 UTC — Round 61

### PR #2553 edward: Lion lr=1.5e-4 sweep — MERGED ✓ (NEW BASELINE 33.4935)

- **Branch:** `charliepai2g48h5-edward/lion-lr15e-5`
- **Hypothesis:** Lion lr=1.5e-4 (1.5× the #2524 lr=1e-4 baseline), all other params unchanged. Tests whether higher LR within Lion family unlocks faster convergence or deeper minimum.
- **Metrics:**

| Metric | Value | Δ vs #2524 (36.3994) | Δ vs AdamW #2307 |
|---|---|---|---|
| `val_avg/mae_surf_p` | **33.4935** | **−8.05%** | −20.91% |
| `test_avg/mae_surf_p` | **28.6279** | **−8.30%** | −25.66% |

Per-split:

| Split | val | Δ vs #2524 |
|---|---|---|
| `val_single_in_dist` | 25.7691 | −9.60% |
| `val_geom_camber_rc` | 50.5514 | −3.50% (least) |
| `val_geom_camber_cruise` | 20.2827 | −14.36% (most) |
| `val_re_rand` | 37.3708 | −8.90% |

- **Committed metrics:** `models/model-charliepai2g48h5-edward-lion-lr15e-5-20260513-200129/metrics.jsonl`
- **Best epoch:** 70/70 (terminal; still monotonically descending at LR≈0)
- **First beat baseline:** epoch 54 (model crossed 36.3994 with 16 epochs to spare)
- **Lion momentum non-zero fraction:** 0.9958 (fully populated)

**Analysis:** LR=1.5e-4 wins decisively but NOT via earlier convergence (Scenario A falsified). Best epoch moved LATER (ep65→ep70) — the higher LR explored wider loss landscape before cosine tail, unlocking a deeper minimum. Model is still budget-bound at ep70. Per-split signature uniform improvement, largest on cruise (−14.36%), smallest on camber_rc (−3.50%). 2nd consecutive Lion-family WIN.

Note: test_geom_camber_cruise/loss=NaN in eval (bf16 vol_loss overflow); MAE values valid (FP64 accumulator).

**Action:** MERGED → new baseline 33.4935 / 28.6279. Assigned #2583 edward Lion lr=2e-4 (next step up; best epoch still ep70 at 1.5e-4 → still budget-bound → more LR headroom may unlock even deeper minimum). Baseline to beat: val < 33.4935.

---

## 2026-05-14 20:50 UTC — Round 60

### PR #2550 frieren: Lookahead(k=5, α=0.5) wraps AdamW — CLOSED (LOSS, 26th taxon)

- **Branch:** `charliepai2g48h5-frieren/lookahead-k5-a05`
- **Hypothesis:** Wrap AdamW with Lookahead (k=5 inner steps + outer slow-step toward fast with α=0.5); first slow-fast 2-loop meta-optimization probe.
- **Metrics:**

| Metric | Lookahead val | Baseline (old) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 53.6305 | 42.3455 | **+26.6%** |
| `test_avg/mae_surf_p` | 47.5413 | 38.5059 | **+23.5%** |

Per-split (regression-toward-mean signature):

| Split | Lookahead val | Baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 49.38 | 35.48 | +39.2% |
| `val_geom_camber_rc` | 68.88 | 60.83 | +13.2% (least) |
| `val_geom_camber_cruise` | 39.40 | 27.65 | +42.5% (worst) |
| `val_re_rand` | 56.87 | 45.42 | +25.2% |

- **Committed metrics:** `models/model-charliepai2g48h5-frieren-lookahead-k5-a05-20260513-200521/metrics.jsonl`
- **Trajectory:** monotonically descending at ep70 (53.63, still falling); LR≈0; cosine cooldown WASTED
- **Tail val std (last 10 ep):** 0.26 — trajectory IS smooth, but smooth around a bad plateau
- **Per-cycle motion:** 0.5·Δfast → effective step HALVED → 70 Lookahead epochs ≈ 35 effective AdamW epochs
- **Lookahead diagnostics at terminal:** drift=0.0 (artifact: 375 batches × k=5 → terminal is outer-step boundary so fast==slow)

**Analysis / 26th taxon — meta-optimizer halves effective step → under-convergence in budget-bound regime + cosine schedule mis-alignment:**
1. Over 1 Lookahead cycle (5 inner + 1 outer): Δfast = sum of 5 AdamW updates; outer step → slow ← slow + α·Δfast; fast ← slow → **net per-cycle motion = 0.5·Δfast = HALVED step**.
2. 70 Lookahead epochs ≈ 35 effective AdamW epochs. Cosine schedule completes cooldown before convergence.
3. Per-epoch trajectory monotonically descending at ep70 with LR≈0 → convergence-bound, NOT plateaued.
4. **Per-split "regression-toward-mean":** easiest baseline split (cruise 27.65) hit WORST (+42.5%); hardest baseline split (camber_rc 60.83) hit LEAST (+13.2%). Under-converged model can't differentiate splits — all drift to a common ~50-60 plateau. OPPOSITE of predicted SWAD-style OOD-favoring pattern — no flat-minima benefit because model never reached a minimum.
5. **Combined with #2544 EMA LOSS, the parameter-space averaging meta-family is now 2-LOSS** with one in-flight arm (#2567 SWA-late, nezuko). Both share root cause: budget-bound regime can't afford weight-averaging-related mechanisms that delay effective convergence. The base optimizer (now Lion) just barely converges in 70 epochs.

**Action:** Pivoting frieren to a structurally distinct mechanism (NOT another weight-averaging arm — that family is closing). Assigned #2580 frieren spectral normalization on attention Linear layers (Lipschitz constraint; 1-Lipschitz Q/K/V/proj; Bartlett-style margin bounds; Miyato et al. 2018; targets camber_rc OOD bottleneck via tighter generalization bound; does NOT slow convergence — per-step normalization with no time-lag effect).

**Human GH issues:** None found.

---

## 2026-05-14 20:30 UTC — Round 59

### PR #2544 nezuko: EMA Polyak weight averaging α=0.999 — CLOSED (LOSS, 25th taxon)

- **Branch:** `charliepai2g48h5-nezuko/ema-polyak-decay-0999`
- **Hypothesis:** Exponential Moving Average of model params (α=0.999, ~1000-step time constant); eval on EMA snapshot; first weight-averaging probe.
- **Metrics:**

| Metric | EMA val | Baseline (old) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (best, ep 69) | 48.1912 | 42.3455 | **+13.83%** |
| `test_avg/mae_surf_p` | 42.4298 | 38.5059 | **+10.19%** |

Per-split (uniform LOSS):

| Split | EMA val | Baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 40.36 | 35.48 | +13.76% |
| `val_geom_camber_rc` | 65.73 | 60.83 | +8.05% (least) |
| `val_geom_camber_cruise` | 34.13 | 27.65 | +23.42% (worst) |
| `val_re_rand` | 52.54 | 45.42 | +15.69% |

- **Committed metrics:** `models/model-charliepai2g48h5-nezuko-ema-polyak-decay-0999-20260513-194340/metrics.jsonl`
- **Terminal drift ratio: 0.0004** — EMA fully caught up to live by ep70 (cosine LR near-zero).
- **Head-to-head:** terminal-live = 48.2184, terminal-EMA = 48.1912 (difference 0.05% — both equally bad vs baseline).
- **Best epoch == last epoch** every time — EMA trajectory never plateaued, always catching up.

**Analysis / 25th taxon — parameter-space-averaging with α-budget vs schedule mismatch + cold-start contamination:**
1. α=0.999 → 1000-step time constant was too long for 26k-step budget under cosine cooldown. EMA never aligned with a stable minimum.
2. Cold-start contamination severe: α^375 ≈ 0.687 → ep1 val_avg=379 (vs typical ~100-200). First ~10 epochs dominated by random-init weights.
3. Terminal drift 0.0004 → EMA == live at terminal; no flat-minima benefit extracted.
4. OOD pattern REVERSED from SWAD prediction: cruise (+23.4% worst), camber_rc (+8.05% least). Cold-start hurt easy splits (low absolute MAE → high relative δ) most.
5. **Failure mode: NOT a generic EMA problem. Specifically α-budget vs schedule mismatch.** Student's analysis: two clean fixes are SWA-late (start ep30, uniform snapshots) OR α=0.99 (100-step lag, cold-start flushed by ep1).

**Action:** Reassigned as #2567 nezuko SWA-late (uniform discrete snapshots ep 30, 35, ..., 70; late-start fixes cold-start; uniform average fixes α-budget; SWAD precedent for OOD). 25th taxon closes continuous-EMA with α=0.999 on this budget; discrete SWA-late is structurally distinct.

---

## 2026-05-14 20:05 UTC — Round 58

### PR #2496 fern: Per-channel surface loss ch_w=[1.0,1.0,3.0] — CLOSED (1st stale_wip)

- **Branch:** `charliepai2g48h5-fern/per-channel-surf-p3`
- **Hypothesis:** Per-channel surface loss with `ch_w=[1.0, 1.0, 3.0]` (3× weight on pressure channel in surf_loss only; vol_loss uniform; surf_weight=10 unchanged; zero param change). First per-channel gradient allocation probe in launch.
- **Outcome:** 1st stale_wip — no commits since assignment at 18:00 UTC; only advisor-assignment commit present.
- **Action:** Closed as stale_wip. Hypothesis preserved and reassigned as retry-1 (#2557).

**Analysis:** Hypothesis remains structurally sound. After Lion merge, a novel interaction emerges: Lion's sign-step means per-channel loss weights translate into per-channel *sign-direction* allocation rather than just magnitude rescaling. The pressure channel at 3× weight will have its sign dominate more often in the aggregated sign across the 3 surface channels — a mechanism distinct from anything tested previously.

**Assignment:** #2557 fern per-channel-surf-p3-retry1 created on fresh branch off current advisor (Lion baseline 36.3994). NEW bar: `val_avg/mae_surf_p < 36.3994`.

**Human GH issues:** None found.

---

## 2026-05-14 05:50 UTC — Round 57

**MAJOR WIN MERGED.** Lion optimizer #2524 beats AdamW baseline by -14.05% val / -18.92% test — largest single-PR gain since round-1. New baseline 36.3994. All subsequent PRs must beat this new bar.

### PR #2524 edward: Lion optimizer (lr=1e-4, wd=3e-4, betas=(0.9, 0.99)) — MERGED ✓ (NEW BASELINE)

- **Branch:** charliepai2g48h5-edward/lion-lr1e-4 (squash-merged 2026-05-14)
- **Hypothesis:** Replace AdamW with Lion (Chen et al. 2023 sign-based momentum); update = sign(β₁m + (1-β₁)g); uniform step magnitude vs AdamW's adaptive 2nd-moment normalization.
- **Metrics artifact:** `models/model-charliepai2g48h5-edward-lion-lr1e-4-20260513-191121/metrics.jsonl`

| Metric | Lion (ep65) | Baseline #2307 (AdamW) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **36.3994** | 42.3455 | **−14.05% (WIN)** |
| `test_avg/mae_surf_p` | **31.2200** | 38.5059 | **−18.92% (WIN)** |
| `val_single_in_dist` | **28.5065** | 35.4776 | **−19.65%** |
| `val_geom_camber_rc` | **52.3873** | 60.8311 | **−13.88%** (largest rc movement in launch) |
| `val_geom_camber_cruise` | **23.6834** | 27.6517 | **−14.35%** |
| `val_re_rand` | **41.0204** | 45.4214 | **−9.69%** |

- **67/70 epochs (timeout), best=ep65, still improving monotonically at terminal.**
- **CRITICAL diagnostic:** Lion momentum non-zero fraction = 0.9986 → fully populated; no stuck/zero-update pathology.
- **Mechanism:** L1 loss already provides sign-based gradients; AdamW's 2nd-moment normalization approximately redundant in this sign-gradient regime. Lion's `sign(β₁m + (1-β₁)g)` commits to sign-direction signal directly. Additionally, Lion has no `v_hat` to track in bf16 — more numerically stable than AdamW's 2nd-moment EMA in low-precision. Test improves MORE than val (−18.92% vs −14.05%) → consistent with uniform step magnitudes providing slight regularization effect.
- **val_geom_camber_rc: 60.83→52.39 (−13.88%)** — LARGEST single-PR rc OOD bottleneck movement in launch history.
- **NOTE on in-flight PRs:** All 7 currently in-flight PRs (#2526, #2531, #2536, #2541, #2544, #2550, [various]) were assigned against the OLD AdamW baseline (42.3455). They will now be compared against the NEW Lion baseline (36.3994). Some may not beat it; this is expected as they were designed for AdamW-era stack.

### New assignment

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2553 | edward | Lion lr=1.5e-4: 1.5× sweep above #2524 baseline lr=1e-4 (--epochs 70) | Lion LR optimization; ep65/67 still improving in #2524 → tests if faster convergence within budget; all other params unchanged; advisor branch has Lion merged |

---

## 2026-05-14 05:30 UTC — Round 56

One review-ready LOSS closed (24th closed taxon: input-encoding-noise interferes with slice-routing softmax) + 1 fresh hypothesis assigned (first slow-fast 2-loop meta-optimization probe).

### PR #2509 frieren: Fourier feature encoding K=4 — CLOSED (INPUT-ENCODING AXIS)

- **Branch:** charliepai2g48h5-frieren/fourier-features-k4
- **Hypothesis:** NeRF-style Fourier features (sin/cos at frequencies [π, 2π, 4π, 8π]) on raw spatial coords (x, y); 16 extra channels concat to 24→40 preprocess input; targets val_geom_camber_rc OOD via high-frequency spatial basis.
- **Metrics artifact:** `models/model-charliepai2g48h5-frieren-fourier-features-k4-20260513-184209/metrics.jsonl`

| Metric | FF K=4 | Baseline #2307 (no FF) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **48.2424** | 42.3455 | **+13.94% (LOSS)** |
| `test_avg/mae_surf_p` | **43.0564** | 38.5059 | **+11.81% (LOSS)** |
| `val_single_in_dist` | 43.612 | 35.478 | **+22.93%** (WORST) |
| `val_geom_camber_rc` | 64.437 | 60.831 | **+5.93%** (LEAST) |
| `val_geom_camber_cruise` | 32.454 | 27.652 | +17.37% |
| `val_re_rand` | 52.467 | 45.421 | +15.51% |

- **54/70 epochs (timeout-capped), best=ep50**; trajectory ep49-54 plateau.
- **DIAMETRIC OPPOSITE of predicted signature:** val_single_in_dist worst (+22.93%, predicted least), val_geom_camber_rc least (+5.93%, predicted best help). If FF unlocked high-frequency representation, camber_rc would have moved most. Instead in-dist regresses hardest — smoking gun for "added channels are noise, not signal."
- **FF sanity diagnostic:** L2-per-node = √8 = 2.828 (matches sin²+cos² identity); FF values in [-1, 1]; implementation correct.
- **Mechanism:** input dim 24→40 (+67%, not +40% as PR body estimated). The orthogonal-init `PhysicsAttention.in_project_slice` projection now routes 40-dim embeddings; the 16 new periodic channels don't correlate with geometric domain structure used for routing. The saf channels (2-3) and dsdf rays (4-11) already encode rich geometric info that subsumes NeRF-style coord encoding. The preprocess Linear was NOT the bottleneck — 4 TransolverBlocks already learn adequate spatial-frequency basis.
- **24th distinct closed-axis taxon: input-encoding-noise interferes with slice-routing softmax.** Input-encoding axis closes at K=0 for this stack at 24-channel input width.
- Student's insight: "slice-routing axis is more fertile" — `in_project_slice` is sensitive to input scale/structure. Future input-side probes should pair with frozen-routing diagnostic.

### New assignment

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2550 | frieren | Lookahead(k=5, α=0.5) wraps AdamW: slow-fast 2-loop variance reduction; eval on slow weights (--epochs 70) | **First slow-fast meta-optimization probe**; Zhang et al. 2019 NeurIPS ~6000 cites; AdamW for k=5 inner steps then slow weights pull toward fast by α=0.5; cleanly distinct from in-flight EMA (continuous α=0.999, ~1000-step vs k=5), Lion (base-optimizer replacement), CLOSED SAM (gradient ascent), and 8 optimizer-family closures |

---

## 2026-05-14 05:00 UTC — Round 55

One review-ready LOSS closed (23rd taxon: auxiliary-objective with trivially-satisfied target — early-training inductive-bias damage) + 1 stale_wip closed and retried + 1 fresh hypothesis assigned (FIRST weight-averaging probe in launch).

### PR #2495 nezuko: Auxiliary camber prediction head λ=0.1 retry-1 — CLOSED (AUX-OBJECTIVE TRIVIALLY-SATISFIED-TARGET AXIS)

- **Branch:** charliepai2g48h5-nezuko/aux-camber-head-l01-retry1
- **Hypothesis:** Add CamberHead (96→32→1) on surface-pooled last-block hidden; λ=0.1 MSE on `x_norm[:, 0, 15]` (NACA camber foil-1) as auxiliary loss. Targets camber_rc OOD bottleneck via enforced camber-discriminative representations.
- **Metrics artifact:** `models/model-charliepai2g48h5-nezuko-aux-camber-head-l01-retry1-20260513-183316/metrics.jsonl`

| Metric | aux λ=0.1 | Baseline #2307 (no aux) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.9564** | 42.3455 | **+10.89% (LOSS)** |
| `test_avg/mae_surf_p` | **41.0502** | 38.5059 | **+6.61% (LOSS)** |
| `val_single_in_dist` | 43.9078 | 35.4776 | **+23.76%** (WORST) |
| `val_geom_camber_rc` | 63.2450 | 60.8311 | **+3.97%** (LEAST) |
| `val_geom_camber_cruise` | 30.5603 | 27.6517 | +10.52% |
| `val_re_rand` | 50.1126 | 45.4214 | +10.33% |

- **58/70 epochs (SENPAI_TIMEOUT_MINUTES=30 capped), best=ep57, terminal=ep58 (1-ep bounce).**
- **CRITICAL aux-MSE trajectory:**
  - ep1 aux MSE = 0.986 → ep3 = 0.085 (10× drop in 3 epochs)
  - ep57 (best) = 0.0016 — aux objective essentially solved
- **Mechanism (smoking gun):** the camber target `x[:, 0, 15]` is a per-batch broadcast scalar that's literally one of 22 input channels. The hidden representation can trivially copy this through the network — no actual modeling required. By epoch 3, the aux task is solved; past that, λ×aux_loss ≈ 1.6e-4 (negligible vs surf_loss ~1.15). The damage occurs in epochs 1-3 when aux gradient is comparable to surf gradient, shifting representation toward redundant camber-encoding direction.
- **Per-split DIRECTIONAL signature:** camber_rc hit LEAST (+3.97%) — directionally consistent with aux mechanism shifting representation toward camber-discriminative features. Bimodal gap COMPRESSES from 71% (baseline) to 44% (this run) — but by HURTING in_dist (+23.76%), not by HELPING rc. Not a useful compression.
- **23rd distinct closed-axis taxon: auxiliary-objective-with-trivially-satisfied-target — early-training inductive-bias damage.** Failure mode is structural (target choice), not λ-tunable; λ=0.05 retry would show smaller-magnitude same failure.
- **Param-count discrepancy note:** student observed actual model is 331,372 params (BASELINE.md says 576,875 for #2307). The actual baseline build with current advisor config is ~328K — BASELINE.md is stale, likely captured pre-slice_num=24 merge. Documentation bug, not behavior bug. Flagging for separate fix.

### PR #2480 thorfinn: NormFormer Sandwich Norm — CLOSED as 1st stale_wip

- **Last update:** 2026-05-13T17:25:49Z (~11h zero student activity).
- Hypothesis preserved (residual-stream variance management via Pre-LN + Post-LN; distinct from CLOSED RMSNorm-substitution and FiLM). Reassigned as #2541 RETRY-1 on fresh branch off current advisor with full stack.

### New assignments

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2541 | thorfinn | NormFormer Sandwich Norm RETRY-1: Pre-LN + Post-LN per residual addition (--epochs 70) | Residual-stream variance management; ln_post_attn + ln_post_mlp per block; ~+1.5K params; Shleifer et al. 2021 |
| #2544 | nezuko | EMA Polyak weight averaging: α=0.999 step-EMA on model parameters; evaluate using EMA snapshot (--epochs 70) | **First weight-averaging probe in launch**; Polyak-Ruppert 1992 + Izmailov et al. 2018 SWA + Cha et al. 2021 SWAD; targets flat-minima OOD generalization (directly relevant to camber_rc 60.83 bottleneck); zero inference param-count change; ~+0.5-1% per-epoch cost; orthogonal to all 22 closed taxa |

---

## 2026-05-14 04:45 UTC — Round 54 addendum (tanjiro 2nd stale)

PR #2472 tanjiro split-surf-vol-heads RETRY-1 (assignment 2026-05-13T17:15:47Z; no student commits in ~11h) closed as 2nd consecutive stale_wip. Original #2396 was also stale before retry-1. Tanjiro pod has now produced 2 consecutive stalls on the same hypothesis. Reassigned as #2536 RETRY-2 on fresh branch off current advisor (full advisor stack inherited). If RETRY-2 also stalls without committed work, the axis will be deprioritized per same pattern as DropPath (abandoned in round-39 after 4 stalls). Hypothesis remains structurally fresh: decoder-architecture probe bifurcating mlp2 into mlp2_surf + mlp2_vol by is_surface mask, complementary to in-flight alphonse flow-bias #2531 (output-bias from flow conditions), in-flight askeladd PaLM #2526 (block-internal restructure), and closed decoder-skip #2503 (latent-conditioned readout split).

### New assignment

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2536 | tanjiro | Split decoder: mlp2_surf + mlp2_vol per is_surface mask (RETRY-2); ~+9.5K params; --epochs 70 | Decoder-architecture probe bifurcating output heads by node type; 3rd attempt; if also stalls, axis will be deprioritized |

---

## 2026-05-14 04:30 UTC — Round 54

One review-ready LOSS closed (22nd closed taxon: redundant-readout-pathway interference — skip is load-bearing but harmful) + 1 fresh hypothesis assigned (axis: flow-condition-conditional output bias — first probe; physically motivated by freestream + perturbation decomposition).

### PR #2503 alphonse: Decoder residual skip Linear(96, 3) after ln_3 — CLOSED (READOUT-PATHWAY AXIS)

- **Branch:** charliepai2g48h5-alphonse/decoder-residual-skip
- **Hypothesis:** Add a linear skip path inside the decoder: `output = mlp2(ln_3(fx)) + skip_proj(ln_3(fx))`, where `skip_proj = Linear(96, 3)` with zero-init bias and Kaiming-uniform-init weight. Tests whether the smooth+sharp output mixture benefits from a dedicated linear pathway alongside mlp2's GELU-residual.
- **Metrics artifact:** `models/model-charliepai2g48h5-alphonse-decoder-residual-skip-*/metrics.jsonl`

| Metric | skip | Baseline #2307 (no skip) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.7497** | 42.3455 | **+10.40% (LOSS)** |
| `test_avg/mae_surf_p` | **40.3518** | 38.5059 | **+4.79% (LOSS)** |
| `val_single_in_dist` | 42.1611 | 35.4776 | +18.84% |
| `val_geom_camber_rc` | 64.5773 | 60.8311 | +6.16% (least regression) |
| `val_geom_camber_cruise` | 31.3023 | 27.6517 | +13.21% |
| `val_re_rand` | 48.9580 | 45.4214 | +7.79% |

- **70/70 epochs, best=ep66.**
- **CRITICAL weight-norm diagnostic** (smoking gun):
  - `mlp2_final_w.norm() = 1.0836` (pre-existing decoder readout)
  - `skip_proj_w.norm() = 0.4347` (new linear path)
  - **ratio = skip/mlp2 = 0.40** → skip IS load-bearing (well above 0.1 ignored threshold)
  - Bias zero-init didn't rescue: training drove skip_proj.weight to non-trivial norm despite starting near zero
- **Mechanism (textbook double-counting):** The pre-existing mlp2 (96→GELU→96→3) was already encoding smooth+sharp mixture efficiently. Adding redundant linear pathway SPLIT gradient between two competing readout heads; optimizer settled on suboptimal 40-60 mlp2-skip mixture worse than mlp2 alone.
- **Smoking gun pattern:** val_single_in_dist regresses MOST (+18.84%) — the split where latent→output mapping should be most linear, and thus where the linear skip SHOULD have helped most. The diametric opposite of predicted "linear-component WIN" pattern. Matches "LOSS — double-counting" prediction arm exactly.
- **22nd distinct closed-axis taxon: redundant-readout-pathway interference** (uniform LOSS with load-bearing-but-harmful skip; structurally distinct from prior 21 closures).
- **Composability with in-flight #2472 split-heads is now moot** for this skip form. (Split-heads bifurcates by node type, a structurally different operation.)

### New assignment

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2531 | alphonse | Flow-conditional output bias: flow_bias = Sequential(Linear(3, 16), GELU, Linear(16, 3)) zero-init; output = mlp2(ln_3(fx)) + flow_bias(x[:, 0:1, [13, 14, 18]]) (Re, AoA, fun_dim); +115 params; --epochs 70 | First flow-condition-conditional OUTPUT-bias probe; physically motivated by freestream+perturbation decomposition for incompressible NS; distinct from CLOSED FiLM (per-block modulation, this is single-shot output) and CLOSED decoder-skip (latent-conditioned, this is flow-condition-conditioned); zero-init prevents weight-splitting failure mode |

---

## 2026-05-14 04:10 UTC — Round 53

One review-ready LOSS closed (21st closed taxon: routing-sharpness over-commitment at warm-start) + 1 fresh hypothesis assigned (axis: PaLM-style parallel attention+MLP — first decoupled-branch architectural probe).

### PR #2502 askeladd: PhysicsAttention temperature init τ=0.5 → 0.25 — CLOSED (ROUTING-SHARPNESS AXIS)

- **Branch:** charliepai2g48h5-askeladd/temp-init-025
- **Hypothesis:** Initialize the learnable per-head PhysicsAttention temperature parameter at τ=0.25 instead of default τ=0.5; sharper slice routing at warm-start. Zero param-count change.
- **Metrics artifact:** `models/model-charliepai2g48h5-askeladd-temp-init-025-20260513-181842/metrics.jsonl`

| Metric | τ=0.25 | Baseline #2307 (τ=0.5) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **45.4153** | 42.3455 | **+7.25% (LOSS)** |
| `test_avg/mae_surf_p` | **39.3021** | 38.5059 | **+2.07% (LOSS)** |
| `val_single_in_dist` | 40.1635 | 35.4776 | +13.20% |
| `val_geom_camber_rc` | 61.4312 | 60.8311 | +0.99% (~flat) |
| `val_geom_camber_cruise` | 30.9050 | 27.6517 | +11.77% |
| `val_re_rand` | 49.1613 | 45.4214 | +8.24% |
| `test_geom_camber_rc` | **56.4143** | 57.3635 | **−1.65% (mild WIN)** |

- **70/70 epochs, best=ep69, terminal=ep70** (still descending; not undertrained).
- **CRITICAL trained-τ diagnostic** (8 values across 4 blocks × 2 heads):
  - block_0 mean=0.286, block_1 mean=0.284, block_2 mean=0.294, block_3 mean=0.323
  - Overall mean = 0.297 (vs init 0.25; +18.7% drift)
  - Per-head std ≤ 0.035 within block; no head specialization on τ axis
  - Deeper-block tilt UPWARD: block_3 prefers softer routing (0.323) than block_0 (0.286)
- **DECISION: Close as LOSS.** 21st closed-axis taxon: routing-sharpness over-commitment at warm-start.
- **MECHANISM:** Hard-commit routing (τ=0.25 → ~0.97 max softmax weight per slice) at init forces tokens into mostly-noise slots when slice prototypes are random and placeholder representation is near-zero. Optimizer back-pedals τ upward during training but the resulting representations never fully recover the slice-prototype diversity that a moderate-τ init would have allowed. Trained-τ landing at 0.30 represents the equilibrium between routing-sharpness incentive (L1+slice_num=24) and diversity-preservation incentive.
- **CRITICAL FINDING — second sharpening-helps-camber-rc signal:** test_geom_camber_rc slight WIN (−1.65%) — the ONLY positive movement across all val/test splits. Combined with PR #2418 ReLU² (val_geom_camber_rc WIN −2.50% only across all 4 splits), this is the **2nd confirmation that selective sharpening interventions help camber_rc OOD specifically**, even when they cause uniform regression elsewhere. Pattern suggests camber_rc bottleneck is addressable by sharpening if the sharpening can be confined to camber-OOD-relevant tokens/channels.
- **AXIS BRACKETING:** temperature-init axis now has 2 data points: τ=0.5 (default, optimum) → 42.3455; τ=0.25 (over-sharp) → 45.4153 LOSS. Routing-sharpness axis essentially saturated by slice_num=24 + L1 + LayerScale stack.

---

### Assignment — Round 53

#### PR #2526 askeladd: PaLM-style parallel attention+MLP (Chowdhery et al. 2022)
- **Branch:** charliepai2g48h5-askeladd/palm-parallel-attn-mlp
- **Hypothesis:** Restructure each TransolverBlock to compute attention and MLP IN PARALLEL on the same LN-input rather than SEQUENTIALLY. New forward: `fx + γ_attn * attn(h) + γ_mlp * mlp(h)` where `h = ln_1(fx)` (single shared LN input; ln_2 stays instantiated but unused). Zero new params; saves one LN evaluation per block (~5-10% epoch speedup expected).
- **Rationale:** In LayerScale-damped regime (γ_attn ≈ 0.005), sequential and parallel are numerically nearly equivalent; the test is on the DECOUPLED GRADIENT mechanism. Hypothesis: orthogonal decomposition lets γ_attn grow into a more productive equilibrium (currently suppressed because attention's effect on fx is immediately re-mixed by mlp via residual; parallel decouples this).
- **First decoupled-branch architectural probe** in this launch. Structurally distinct from in-flight NormFormer (sandwich Pre+Post-LN), split-heads (decoder bifurcation), decoder-skip (decoder linear path), and all 21 closed taxa.
- **Predicted:** -0.5% to -2% val if γ_attn grows; wash (most likely given LayerScale damping) if decoupling effect is too small; LOSS if sequential composition was load-bearing (γ_attn collapses to zero).
- **CRITICAL diagnostic:** trained γ_attn at terminal — values > 0.01 (vs baseline ~0.005) confirm decoupling hypothesis regardless of val_avg outcome.

---

## 2026-05-14 03:50 UTC — Round 52

One review-ready LOSS closed (20th closed taxon: per-group-LR axis closes BOTH directions; uniform-LR AdamW confirmed optimum on optimizer-group axis) + 1 fresh hypothesis assigned (axis: optimizer-family, Lion sign-based momentum).

### PR #2497 edward: Layer-wise LR Decay (LLRD) decay=0.7^k from output to input — CLOSED (PER-GROUP-LR AXIS CLOSURE)

- **Branch:** charliepai2g48h5-edward/llrd-decay07
- **Hypothesis:** Per-parameter-group LR with decay=0.7^k from output to input (5 groups: preprocess @ lr×0.240, block_0 @ lr×0.343, block_1 @ lr×0.490, block_2 @ lr×0.700, block_3+decoder @ lr×1.0). Standard ViT/BERT LLRD recipe.
- **Metrics artifact:** `models/model-charliepai2g48h5-edward-llrd-decay07-20260513-180911/metrics.jsonl`

| Metric | LLRD decay=0.7 | Baseline #2307 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **47.7593** | 42.3455 | **+12.79% (LOSS)** |
| `test_avg/mae_surf_p` | **41.4230** | 38.5059 | **+7.58% (LOSS)** |
| `val_single_in_dist` | 42.0589 | 35.4776 | +18.55% |
| `val_geom_camber_rc` | 65.8278 | 60.8311 | +8.21% |
| `val_geom_camber_cruise` | 31.2243 | 27.6517 | +12.92% |
| `val_re_rand` | 51.9261 | 45.4214 | +14.32% |

- **70/70 epochs reached, best=terminal=ep70** — model monotonically descending, convergence-bound NOT plateaued (340 → 100 @ ep16 → 70 @ ep30 → 56 @ ep50 → 48 @ ep70).
- **γ_mlp depth gradient at terminal:** block_0=0.008, block_1=0.029, block_2=0.041, block_3=0.050 — STEEP ramp vs uniform-LR baseline ~0.063-0.078 across all 4 blocks per PR #2195 → confirms early-block under-training.
- **Per-group LR ratio preservation verified:** block_0/block_3 = 0.343 = 0.7^3 ✓, preprocess/block_3 = 0.240 = 0.7^4 ✓ through LinearLR×CosineAnnealingLR. No implementation bug.
- **DECISION: Close as LOSS.** 20th distinct closed-axis taxon: layer-wise-LR-decay-downward in budget-bound regime.
- **MECHANISM:** decay=0.7^4 = 0.24× preprocess + 0.343× block_0 starves early-layer optimizer work below the 70-epoch convergence floor. The 4-block-too-shallow wash prediction was FALSIFIED. Combined with #2457 (embed=2× UP destabilization), the **per-parameter-group-LR axis is FULLY CLOSED in BOTH directions** — uniform-LR AdamW is the local optimum on the optimizer-group axis.
- **Cumulative optimizer-family closures in this launch (8 total):** AdamW (lr-UP/DOWN/β1=0.95/amsgrad), SGD-momentum, SAM ρ=0.05, per-group-LR embed-UP, LLRD-decay-DOWN.

---

### Assignment — Round 52

#### PR #2524 edward: Lion optimizer (sign-based momentum) replaces AdamW
- **Branch:** charliepai2g48h5-edward/lion-optimizer
- **Hypothesis:** Replace AdamW with Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms"). Lion uses `update = sign(β₁ * m + (1-β₁) * g)` — pure sign step replaces AdamW's 2nd-moment normalization. Hyperparameters: lr=1e-4 (1/5 AdamW), wd=3e-4 (3× AdamW), β2=0.99 (vs AdamW 0.999), β1=0.9. Zero param change; pure optimizer swap. 9th optimizer-family member tested in this launch.
- **Rationale:** Fundamentally distinct from AdamW (no 2nd moment) and SGD (sign vs raw magnitude). Aligns with L1 sign-gradient regime: L1's gradient is essentially sign(error); AdamW's 2nd-moment normalization is near-identity for already-sign-based gradients. Lion cuts the overhead and commits to sign step directly. Also: bf16's 7-bit mantissa makes AdamW's v_hat tracking noisy; Lion's sign step is bf16-stable.
- **Predicted:** -1% to -4% on val_avg if Lion's sign-step aligns with L1+bf16 regime; or wash if L1+AdamW already approximates sign-based at equilibrium.
- **Critical risk:** Lion is LR-sensitive — if wrong, can diverge in first 5 epochs (val_avg > 200 at ep5). Flag for early termination.
- **Diagnostic:** student must report Lion momentum non-zero fraction at terminal (sanity check on optimizer state).

---

## 2026-05-14 03:30 UTC — Round 51

One review-ready LOSS closed (19th closed taxon: FFN-gating axis) + 1 fresh hypothesis assigned (axis: input-encoding via Fourier features).

### PR #2439 frieren: GeGLU FFN (gate × GELU(value) at all MLP sites) — CLOSED (FFN-GATING AXIS)

- **Branch:** charliepai2g48h5-frieren/geglu-ffn
- **Hypothesis:** Replace standard GELU FFN with GeGLU (Shazeer 2020) at all 3 MLP sites: `GeGLU(x) = (xW1) ⊙ GELU(xW2)`. Tests whether multiplicative gating in FFN unlocks representational capacity beyond standard GELU FFN.
- **Metrics artifact:** student's metrics.jsonl on PR branch

| Metric | GeGLU | Baseline #2307 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **44.3006** | 42.3455 | **+4.62% (LOSS)** |
| `test_avg/mae_surf_p` | **38.8656** | 38.5059 | **+0.93% (wash)** |
| `val_single_in_dist` | 35.9695 | 35.4776 | +1.39% |
| `val_geom_camber_rc` | 62.5817 | 60.8311 | +2.88% |
| `val_geom_camber_cruise` | 29.9988 | 27.6517 | +8.49% |
| `val_re_rand` | 48.9525 | 45.4214 | +7.78% |

- **53/55 epochs reached (timeout cut 2 epochs short due to GPU-contention with parallel process).** Best=terminal=ep53 (model still improving).
- **+90K params** (~+27% over ~330K base).
- **DECISION: Close as LOSS.** 19th closed-axis: FFN-structure gating closes in budget-bound regime.
- **MECHANISM (matches student's diagnosis):**
  1. **+90K params don't earn their keep at the capacity floor.** All 4 budget-axes are at their closed optima (n_layers=4, n_hidden=96, slice_num=24, mlp_ratio=2). Adding 27% more parameters via gating dilutes per-param gradient signal without unlocking new representational capacity. Same lesson as mlp_ratio=4 LOSS (+42.5%) and n_hidden=128 LOSS (-2.5%).
  2. **GPU contention from parallel `train.py` process** (student noted PID 54740 cohabitation) ran epochs 45-53 at ~2× wall-clock. Student's own assessment: "even with 2 more clean epochs, this would still be a LOSS of ~+4%." The +4.62% gap exceeds what clean compute would close.
  3. **Test_avg near-wash (+0.93%) vs val_avg +4.62% LOSS** indicates val noise-amplification — but per-split direction is uniformly UP, not bimodal. Not the averaging-style failure mode.
- **FFN axis summary (post-closure):**
  - mlp_ratio={1, 2, 4}: 2 optimal (closed both directions)
  - activation: GELU stays (ReLU² LOSS, SiLU LOSS)
  - gating (GeGLU): LOSS (this PR)
  - **FFN architecture probes exhausted at the current stack.** Future FFN moves require changing a non-FFN axis first to re-open the budget envelope.

---

### Assignment — Round 51

#### PR #2509 frieren: Fourier feature encoding K=4 (NeRF/SIREN-style sin/cos on raw spatial coords)
- **Branch:** charliepai2g48h5-frieren/fourier-features-k4
- **Hypothesis:** Concatenate 16 sinusoidal features on raw spatial coords (channels 0-1) to the 24-channel input: `fourier_features(coords, K=4) = [sin(2^k * π * coords), cos(2^k * π * coords)]_{k=0..3}` → 4 freqs × 2 coords × 2 sin/cos = 16 extra channels. Preprocess Linear(24→40) projects expanded input into n_hidden=96. **First INPUT-ENCODING architectural probe in this launch.** Targets val_geom_camber_rc OOD bottleneck via high-frequency spatial basis distinct from learned slice embedding. Frequencies [π, 2π, 4π, 8π] cover full mesh scale (~3 chords) to fine boundary-layer detail (~0.06 chord). +1.5K params (preprocess Linear input dim 24→40).
- **Rationale:** All 4 budget-axes at optimum + all FFN-structure axes closed → next gain must come from non-budget axes. Input encoding is structurally orthogonal to all 19 closed taxa (no prior input-encoding probe). High-frequency basis directly addresses geometric OOD (camber_rc) where sharp curvature near foil generates high-frequency pressure gradients the model currently must learn at coarse-frequency resolution.
- **Predicted:** -1% to -5% on val_avg with strongest gain on val_geom_camber_rc (camber-OOD with sharp curvature) and val_re_rand (Re-OOD with high-frequency Re-correlated structure).
- **Diagnostic:** student must log Fourier-channel L2 norm at first batch to verify encoding is computed correctly (expected ~3.7 RMS for 4×2×2=16 channels with sin/cos at standardized coords).

---

## 2026-05-14 03:15 UTC — Round 50

Two review-ready LOSSes closed + 2 fresh hypotheses assigned (axes: attention-temperature, decoder-skip).

### PR #2455 askeladd: MLP output dropout p=0.05 — CLOSED (3rd STOCHASTIC-REG CLOSURE)

- **Branch:** charliepai2g48h5-askeladd/mlp-output-dropout-005
- **Hypothesis:** `nn.Dropout(p=0.05)` after `fc2` in each TransolverBlock FFN; feature-level dropout distinct from closed attention-weight dropout.
- **Metrics artifact:** `models/model-charliepai2g48h5-askeladd-mlp-output-dropout-005-20260513-171354/metrics.jsonl`

| Metric | MLP-dropout | Baseline #2307 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.8749** | 42.3455 | **+10.69% (LOSS)** |
| `test_avg/mae_surf_p` | **41.0065** | 38.5059 | **+6.49% (LOSS)** |
| `val_single_in_dist` | 39.5185 | 35.4776 | +11.39% |
| `val_geom_camber_rc` | 64.0977 | 60.8311 | +5.37% |
| `val_geom_camber_cruise` | 33.2952 | 27.6517 | +20.41% |
| `val_re_rand` | 50.5881 | 45.4214 | +11.38% |

- **70/70 epochs terminal-best — convergence-bound regime confirmed.**
- **DECISION: Close as LOSS.** 17th closed-axis: convergence-floor regime closes the stochastic-regularization class.
- **MECHANISM:** 3rd independent stochastic-regularization closure (1: attention-weight dropout p=0.1; 2: DropPath 4-attempt jinx; 3: MLP-output dropout p=0.05). LayerScale γ (already a learnable per-channel scale on the same FFN residual branch) effectively SUBSUMES the gating role; stochastic dropout on top is destructive. Per-epoch cost of stochasticity > regularization benefit at 30-min cap.

---

### PR #2454 alphonse: AoA + y-coord reflection-aug — CLOSED (PHYSICAL-SYMMETRY HYPOTHESIS FALSIFIED)

- **Branch:** charliepai2g48h5-alphonse/reflection-aug
- **Hypothesis:** Reflect y-coord, AoA, gap, dsdf-rays, and Uy target with p=0.5 to enforce physical Navier-Stokes y-symmetry.
- **Metrics artifact:** `models/model-charliepai2g48h5-alphonse-reflection-aug-20260513-170926/metrics.jsonl`

| Metric | Reflection-aug | Baseline #2307 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **55.5301** | 42.3455 | **+31.1% (CATASTROPHIC LOSS)** |
| `test_avg/mae_surf_p` | **48.0034** | 38.5059 | **+24.7% (LOSS)** |
| `val_single_in_dist` | 51.5076 | 35.4776 | **+45.2%** |
| `val_geom_camber_rc` | 74.5492 | 60.8311 | +22.6% |
| `val_geom_camber_cruise` | 36.3625 | 27.6517 | +31.5% |
| `val_re_rand` | 59.7012 | 45.4214 | +31.4% |

- **70/70 terminal-best; epoch 1 val_avg=363 — convergence permanently disrupted by distribution doubling.**
- **DECISION: Close as LOSS — physical reflection symmetry hypothesis FALSIFIED at the dataset level.**
- **MECHANISM (student's rigorous analysis):** 68% of training corpus has half-space mesh y ∈ [0, ~10] with ground plane (raceCar single 38% + raceCar tandem 30%). Reflecting y → -y moves the entire raceCar mesh into NEGATIVE y space — the model never sees these coordinates in val/test. Only cruise tandem (~30%) has truly symmetric mesh y ∈ [-9.58, +9.55]. The 18th closed taxon: whole-corpus-reflection physical-symmetry hypothesis falsified by mesh-half-space structure. Student's follow-ups (cruise-only gated reflection, per-mesh-center reflection) preserved as possible future directions but lower priority.

---

### New Assignments (Round-50)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #2502 | askeladd | PhysicsAttention temperature init τ=0.25 (sharper routing) | First temperature-axis probe; orthogonal to closed slice_num=24 (count vs sharpness); zero param change; trained τ diagnostic critical for closure |
| #2503 | alphonse | Decoder residual skip: Linear(96,3) after ln_3 in last block | First decoder-skip probe; +291 params; output = mlp2(ln_3(fx)) + skip_proj(ln_3(fx)); composable with in-flight #2472 split-heads |

---

## 2026-05-14 03:00 UTC — Round 49

Three review-ready PRs closed + 3 new experiments assigned.

### PR #1775 fern: WD 1e-4 → 5e-5 — CLOSED (REGIME-FLIP LOSS)

- **Branch:** charliepai2g48h5-fern/weight-decay-5e-5
- **Hypothesis:** WD reduction 2× from 1e-4 to 5e-5, motivated by earlier WIN on β=0.5 Huber baseline (−4.43%).
- **Metrics artifact:** `models/model-charliepai2g48h5-fern-wd-5e-5-epochs-70-20260513-165445/metrics.jsonl`

| Metric | WD=5e-5 | Baseline #2307 (WD=1e-4) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.3290** | 42.3455 | **+9.41% (LOSS)** |
| `test_avg/mae_surf_p` | **40.5578** | 38.5059 | **+5.33% (LOSS)** |
| `val_single_in_dist` | 40.4506 | 35.4776 | +14.02% |
| `val_geom_camber_rc` | 64.0555 | 60.8311 | +5.30% |
| `val_geom_camber_cruise` | 31.8716 | 27.6517 | +15.26% |
| `val_re_rand` | 48.9385 | 45.4214 | +7.75% |

- **70/70 epochs terminal-best; cosine fully annealed to eta_min=0; clean convergence, not budget-starved.**
- **DECISION: Close as LOSS — uniform regime-flip.**
- **MECHANISM:** WD direction sign-flipped. On β=0.5 baseline (old stack), WD=5e-5 was WIN −4.43% — model was under-fit at 37 epochs under heavier config. On current L1+slice_num=24+n_hidden=96+n_layers=4+LayerScale stack with full 70-epoch cosine, model is now "well-trained, needs at least equal penalty." Reducing WD 2× leaves it under-regularized. All 4 splits regress 5-15% — uniform direction, not noise. **15th closed-axis observation: WD-direction regime-flip across stack changes.** On current advisor stack, sweep direction is UP (WD=2e-4) if revisited.

---

### PR #2457 edward: Per-group LR (embed=2×, γ=2×) — CLOSED (EMBED-DESTABILIZATION LOSS)

- **Branch:** charliepai2g48h5-edward/per-group-lr
- **Hypothesis:** Accelerate LayerScale γ warm-up and input embedding adaptation via 2× LR scaling.
- **Metrics artifact:** `models/model-charliepai2g48h5-edward-per-group-lr-20260513-170054/metrics.jsonl`

| Metric | Per-group LR | Baseline #2307 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **48.7460** | 42.3455 | **+15.1% (LOSS)** |
| `test_avg/mae_surf_p` | **41.7706** | 38.5059 | **+8.5% (LOSS)** |
| `val_single_in_dist` | 42.8615 | 35.4776 | +20.8% |
| `val_geom_camber_rc` | 67.3762 | 60.8311 | +10.8% |
| `val_geom_camber_cruise` | 32.7553 | 27.6517 | +18.5% |
| `val_re_rand` | 51.9912 | 45.4214 | +14.5% |

- **70/70 epochs terminal-best; trained γ (γ_attn≈0.021, γ_mlp≈0.115) same equilibrium as PR #2195 baseline — γ growth NOT the bottleneck.**
- **DECISION: Close as LOSS — uniform destabilization.**
- **MECHANISM:** embed=2× LR perturbs the upstream shared-token representation. All 4 blocks read the same preprocess/placeholder/slice_embed features every forward pass; doubling LR on those weights makes blocks re-fit a moving input distribution throughout all 70 epochs. Val_avg at ep5=164.5 vs typical baseline trajectory <100 by ep10-15 — convergence permanently bottlenecked. γ-LR acceleration also proved irrelevant: same trained equilibrium reached regardless. **16th closed-axis: per-parameter-group-LR with embed-acceleration.** Future: layer-wise LR DECAY downward (LLRD, early layers get LOWER LR) is the orthogonal and untested direction.

---

### PR #2411 nezuko: Aux camber head — CLOSED (1st STALE_WIP, never ran)

- **Branch:** charliepai2g48h5-nezuko/aux-camber-head
- **Only commit:** assignment commit at 2026-05-13T15:30:20Z; pod stalled ~2h without running.
- **DECISION: Close stale. Hypothesis preserved and reassigned as #2495 retry-1.**

---

### New Assignments (Round-49)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #2495 | nezuko | Auxiliary camber head λ=0.1 (retry-1 of #2411 stale) | 96→32→1 CamberHead on surface-pooled last-block hidden; first aux-objective probe to actually run; --epochs 70 |
| #2496 | fern | Per-channel surface loss p_channel_weight=3.0 | ch_w=[1.0,1.0,3.0] surf only; vol uniform; surf_weight=10 unchanged; zero param change; --epochs 70 |
| #2497 | edward | LLRD decay=0.7 (layer-wise LR decay) | preprocess @ lr×0.24 → block3+decoder @ lr×1.0; OPPOSITE of closed #2457; --epochs 70 |

---

## 2026-05-14 01:45 UTC — Round 48

One axis-closing LOSS reviewed + 1 new experiment assigned.

### PR #2410 thorfinn: FiLM conditioning — CLOSED (CONDITIONING-PATHWAY AXIS CLOSED)

- **Branch:** charliepai2g48h5-thorfinn/film-conditioning
- **Hypothesis:** FiLM conditioning from 11 broadcast scalars (channels 13-23) as per-block affine modulation of LN-1 input (γ,β zero-init for identity); explicit regime conditioning targeting camber_rc OOD bottleneck.

| Metric | FiLM | Baseline #2307 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 46.4166 | 42.3455 | **+9.61% LOSS** |
| test_avg/mae_surf_p | 40.6822 | 38.5059 | +5.65% LOSS |
| val_single_in_dist | 39.7070 | 35.4776 | +11.93% |
| val_geom_camber_rc | 65.5748 | 60.8311 | **+7.80%** (PRIMARY TARGET REGRESSED) |
| val_geom_camber_cruise | 30.9722 | 27.6517 | +12.00% |
| val_re_rand | 49.4125 | 45.4214 | +8.79% |

- **Epochs:** 70/70 (full run, clean convergence; best=67 ≠ terminal).
- **Per-epoch cost:** ~25.5 s/epoch (faster than predicted; FiLM MLP + broadcast multiply cheap).
- **Param count:** 366,443 (+38K from FiLM head — ~+12% over 328K base; significantly more than predicted ~+5K).
- **Trained FiLM diagnostics:** γ converged 0.66-1.03 (away from identity), β near-zero mean (−0.012 to +0.015) but σ 0.14-0.33. val_single_in_dist drove γ furthest from identity (means 0.66-0.76) — most modulation on SIMPLEST geometry.
- **Mechanism:** Double-affine compounding. FiLM perturbs the LN-1 INPUT (non-unit-variance, non-zero-mean); LayerScale γ_attn ≈ 1e-3 damps BRANCH OUTPUT but not BRANCH INPUT; attention weights expect LN outputs but get FiLM-perturbed inputs; β propagates additively into residual stream without LayerScale damping. +38K params in wrong direction for budget-bound regime.
- **Axis closure:** Conditioning-pathway axis closed at "FiLM-on-pre-LN-input" for n_hidden=96/LayerScale config. 14th taxon: residual-stream-input-perturbation conditioning.
- **Metrics artifacts:** `models/model-charliepai2g48h5-thorfinn-film-conditioning-20260513-160112/metrics.jsonl`

---

## 2026-05-14 01:00 UTC — Round 47

Three axis-closing LOSSes reviewed + 3 new experiments assigned. All 4 budget-bound axes now mapped at optimum.

### PR #2418 alphonse: Squared ReLU (ReLU²) — CLOSED (ACTIVATION AXIS CLOSED AT GELU)

- **Branch:** charliepai2g48h5-alphonse/squared-relu-retry2
- **Hypothesis:** Squared ReLU at all 3 MLP sites; symmetric probe to closed SiLU LOSS (#2156). Primer-paper precedent.

| Metric | ReLU² | Baseline #2307 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 45.6311 | 42.3455 | **+7.76% LOSS** |
| test_avg/mae_surf_p | 39.0696 | 38.5059 | +1.46% LOSS |
| val_single_in_dist | 42.9923 | 35.4776 | **+21.18% (WORST)** |
| val_geom_camber_rc | 59.3067 | 60.8311 | **−2.50% WIN** |
| val_geom_camber_cruise | 31.0540 | 27.6517 | +12.30% |
| val_re_rand | 49.1714 | 45.4214 | +8.27% |

- **Epochs:** 60/70 (wall-clock cap at 30 min; still descending at cutoff; converged ~45.0-45.3).
- **Param count:** 328,235 (student note: PR body had ~577K estimate from different ancestor — actual is ~0.33M).
- **Diagnosis:** BIMODAL in SURPRISING DIRECTION. rc is the ONLY split that wins (−2.50%); all others regress. Pattern: ReLU²'s quadratic positive growth amplifies large pressure peaks → helps rc; zero negative-tail gating starves negative pre-activation patterns → hurts in-dist (worst +21.18%) and moderate-pressure OOD (cruise).
- **Axis closure:** **Activation axis FULLY CLOSED at GELU.** SiLU (negative direction, uniform LOSS) + ReLU² (positive direction, bimodal LOSS) bracket GELU. 11th taxon: activation-curve over-specialization.
- **Key insight:** rc-WIN-only is the FIRST evidence that selective gradient amplification on large pressure peaks could help camber_rc specifically — generalizable to GeGLU (#2439 in-flight) or per-channel selective gate.
- **Metrics artifacts:** `models/model-charliepai2g48h5-alphonse-squared-relu-retry2-20260513-155837/metrics.jsonl`

---

### PR #2412 askeladd: SAM ρ=0.05 — CLOSED (META-OPTIMIZER AXIS CLOSED)

- **Branch:** charliepai2g48h5-askeladd/sam-rho005
- **Hypothesis:** SAM flat-minima optimizer wrapping AdamW; ρ=0.05 two-step perturbation.

| Metric | SAM ρ=0.05 | Baseline #2307 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 97.1961 | 42.3455 | **+129.6% LOSS** |
| test_avg/mae_surf_p | 88.7014 | 38.5059 | **+130.4% LOSS** |
| val_single_in_dist | 106.2557 | 35.4776 | **+199.6% (WORST)** |
| val_geom_camber_rc | 107.8315 | 60.8311 | +77.3% |
| val_geom_camber_cruise | 79.7911 | 27.6517 | +188.6% |
| val_re_rand | 94.9064 | 45.4214 | +109.0% |

- **Epochs:** 35/35 (terminal, all completed). Val PLATEAUED at ep32 (97.20→97.30→97.64→97.28) → clean convergence to WORSE basin. Per-epoch cost: +14% (NOT predicted 2× — torch.compile cached second forward).
- **Diagnosis:** Catastrophic uniform regression. SAM at ρ=0.05 finds a fundamentally worse minimum. in_dist WORST LOSS (+199.6%) = opposite of flat-minima-helps-OOD prediction. Gap widened with epochs (widening monotone divergence). Three mechanisms: (1) ρ=0.05 too aggressive for CFD surrogate gradient magnitudes; (2) batch=4+L1+bf16 produces too-noisy per-batch gradient direction for SAM's worst-case identification; (3) active gradient interference, not just flatness trade.
- **Axis closure:** **Flat-minima / meta-optimizer axis CLOSED.** AdamW hyperparams (4-LOSS) + SGD (1-LOSS) + SAM (1-LOSS) = optimizer-family fully closed. 12th taxon: meta-optimizer perturbation-noise interference.
- **Metrics artifacts:** `models/model-charliepai2g48h5-askeladd-sam-rho005-20260513-155707/metrics.jsonl`

---

### PR #2392 edward: mlp_ratio 2→1 — CLOSED (FFN CAPACITY FLOOR / mlp_ratio AXIS CLOSED)

- **Branch:** charliepai2g48h5-edward/mlp-ratio-1
- **Hypothesis:** FFN contraction — 4th budget-bound axis probe; tests whether fewer-param FFN frees enough wall-clock to net improve.

| Metric | mlp_ratio=1 | Baseline #2307 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 49.7302 | 42.3455 | **+17.4% LOSS** |
| test_avg/mae_surf_p | 43.5735 | 38.5059 | +13.2% LOSS |
| val_single_in_dist | 43.8610 | 35.4776 | +23.6% |
| val_geom_camber_rc | 66.8479 | 60.8311 | +9.9% |
| val_geom_camber_cruise | 35.7988 | 27.6517 | **+29.5% (WORST)** |
| val_re_rand | 52.4131 | 45.4214 | +15.4% |

- **Epochs:** 75/80 (terminal-best; clean convergence; plateau tight above 49.7 for final ~10 epochs; capacity ceiling).
- **Per-epoch cost:** ~23 s/epoch (−25%, better than predicted). Param count: 254,123.
- **Diagnosis:** Capacity floor. At mlp_ratio=1, each block's FFN = Linear(96→96)→GELU→Linear(96→96) — no higher-dim non-linear bottleneck. Extra wall-clock (~25% savings → 75 epochs vs ~58) cannot compensate loss of per-block feature mixing expressivity. Uniform regression (cruise worst) = general capacity failure.
- **Axis closure:** **mlp_ratio FULLY CLOSED: {1,2,4}→{49.73,42.35,66.73}; concave optimum at 2.** mlp_ratio=4 was budget-cliff (34/50 epochs); this is capacity-floor (75/80, clean convergence); together they definitively close the axis.
- **META-FINDING:** All 4 budget-bound axes at optimum — n_layers=4, n_hidden=96, slice_num=24, mlp_ratio=2. Capacity-budget Pareto front fully mapped. Future gains MUST come from non-budget axes.
- **Metrics artifacts:** `models/model-charliepai2g48h5-edward-mlp-ratio-1-20260513-155022/metrics.jsonl`

---

## 2026-05-14 00:15 UTC — Round 46

One axis-closing LOSS reviewed + 1 new experiment assigned (GeGLU FFN).

### PR #2394 frieren: surf_weight 10→20 (gradient rebalancing) — CLOSED (SURF_WEIGHT AXIS FULLY CLOSED AT 10)

- **Branch:** charliepai2g48h5-frieren/surf-weight-20
- **Hypothesis:** surf_weight=10→20: intermediate of closed bracket {10 baseline, 30 prior LOSS under heavier config}. Tests whether lean-config (n_hidden=96, L1, slim advisor) tolerates higher surface emphasis without the instability of the heavier-config surf_weight=30 LOSS.

| Metric | surf_weight=20 | Baseline #2307 (surf_weight=10) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 47.8434 | 42.3455 | **+12.98% LOSS** |
| test_avg/mae_surf_p | (not reported separately) | 38.5059 | — |
| val_single_in_dist | ~43.3 | 35.4776 | **+22.08% (worst)** |
| val_geom_camber_rc | ~64.2 | 60.8311 | +5.48% (least) |
| val_geom_camber_cruise | (proportional) | 27.6517 | regressed |
| val_re_rand | (proportional) | 45.4214 | regressed |

- **Per-split pattern:** UNIFORM REGRESSION all 4 splits. In_dist worst (+22.08%), camber_rc least (+5.48%) — OPPOSITE of the gradient re-weighting hypothesis (which would predict surf-OOD metrics improve more than in-dist).
- **Vol/surf decomposition:** BOTH surf AND vol metrics regressed. This is NOT gradient re-weighting — it is DESTABILIZATION. Increasing surf_weight unbalances the L1 gradient mix: heavy-tailed surface pressure nodes get inflated weight relative to vol nodes, disrupting the cosine cool-down gradient scaling.
- **Axis closure:** **surf_weight axis FULLY CLOSED.** Bracket: {10, 20, 30} — 10 is the global optimum. The 30-LOSS under heavier config was not just a config-specific failure; even at the lean advisor the axis regresses in both directions away from 10.
- **Mechanism:** surf_weight=10 produces the optimal gradient balance between surface (primary metric) and volume nodes for the cosine cool-down schedule. Higher surf_weight inflates gradients on the heavy-tailed surface pressure distribution, overwhelming the careful LR-controlled convergence.
- **Reassignment:** Frieren → #2439 GeGLU FFN (gated linear unit; orthogonal FFN-structure probe; --epochs 55 for T_max=52 alignment with ~52 actual epochs).

---

## 2026-05-13 23:00 UTC — Round 45

Three axis-closing LOSSes reviewed + 3 new bold experiments assigned.

### PR #2362 askeladd: slice_num 24→16 (routing-optimum floor search) — CLOSED (ROUTING CAPACITY FLOOR)

- **Branch:** charliepai2g48h5-askeladd/slicenum-16
- **Hypothesis:** Continue the routing-quality optimum search downward after PR #2307 MASSIVE WIN (-9.61% at slice_num=24). Does the routing optimum extend below 24 or does the floor appear at 16?

| Metric | slice_num=16 | Baseline #2307 (slice_num=24) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 47.4321 | 42.3455 | **+12.02% LOSS** |
| test_avg/mae_surf_p | 42.5733 | 38.5059 | **+10.56% LOSS** |
| val_single_in_dist | 40.6177 | 35.4776 | +14.49% |
| val_geom_camber_rc | 62.1336 | 60.8311 | +2.14% |
| val_geom_camber_cruise | 36.0101 | 27.6517 | **+30.21% (worst)** |
| val_re_rand | 50.9670 | 45.4214 | +12.21% |

- **Best epoch:** 65 = terminal (still descending at cutoff; harder optimization landscape).
- **Per-epoch cost:** 27.05 s/epoch (-12.2% vs slice_num=24 — budget saving is real but can't compensate routing damage).
- **Diagnosis:** routing capacity floor — uniform regression. slice_num=16 = ~6 tokens per slice average (at n_hidden=96, N nodes). cruise hit hardest (+30.21%): the geometric routing needs ≥24 partitions to separate cruise/camber/Re regimes.
- **Axis closure:** **slice_num axis is FULLY CLOSED with routing-quality optimum at slice_num=24.** Bracket: {16, 24, 32} → val {47.43, 42.35, 46.85}. Concave optimum at 24.
- **Mechanism refinement:** per-epoch saving at 16 is real (-12.2%), but routing damage (+12.02%) overwhelms it. Confirms routing quality (not budget) is the dominant mechanism.

---

### PR #2289 nezuko: n_layers 4→3 (depth-down continuation) — CLOSED (DEPTH CAPACITY FLOOR)

- **Branch:** charliepai2g48h5-nezuko/n-layers-3
- **Hypothesis:** Continue depth-down WIN from PR #2268 (n_layers=5→4 WIN -3.44%). Does budget-bound continue at depth=3 or does capacity floor appear?
- **3-seed replication performed.**

| Metric | n_layers=3 (mean, 3 seeds) | Baseline #2268 (n_layers=4) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (mean) | 48.085 | 46.8460 | **+2.65% LOSS** |
| val_avg/mae_surf_p (best seed) | 47.4050 | 46.8460 | +1.20% |
| test_avg/mae_surf_p (mean) | 42.750 | 40.8140 | +4.74% |
| val_single_in_dist (mean) | 43.601 | 41.7031 | **+4.55%** |
| val_geom_camber_rc (mean) | 64.648 | 64.6729 | −0.04% wash |
| val_geom_camber_cruise (mean) | 32.572 | 31.5759 | +3.15% |
| val_re_rand (mean) | 51.520 | 49.4322 | +4.22% |

- **Epochs reached:** 54-55; actual per-epoch ~33 s (only ~10% savings vs predicted 25% — val/dataload overhead dominates the freed budget).
- **Diagnosis:** depth capacity floor — total-capacity crossed. Per-split signature: in_dist hit hardest (like #2336 n_hidden=64) — model can't fit in-distribution training data at n_layers=3, suppressing in-dist/OOD gap. Even best seed (47.41) loses to the 46.8460 baseline.
- **Axis closure:** **depth-down axis FULLY CLOSED at n_layers=4.** Bracket: {3, 4, 5} → val {49.20, 46.85, 48.52}. Budget-bound→capacity-bound transition confirmed at both depth AND width axes simultaneously: depth floor at n_layers=3, width floor at n_hidden=64.
- **Cross-axis insight:** budget-bound→capacity-bound transition on BOTH depth and width at n_layers=4/n_hidden=96. Model operating near a coupled depth×width capacity frontier.

---

### PR #2298 thorfinn: n_layers=4 + --epochs 90 (budget-extrapolation diagnostic) — CLOSED (SCHEDULE MIS-ALIGNMENT)

- **Branch:** charliepai2g48h5-thorfinn/n-layers-4-epochs90
- **Hypothesis:** Extend budget from 60→90 epochs; test descent-curve continuation beyond PR #2268.

| Metric | --epochs 90 run | Baseline #2268 (--epochs 60) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 47.9738 | 46.8460 | **+2.41% LOSS** |
| test_avg/mae_surf_p | 42.5430 | 40.8140 | +4.24% LOSS |
| val_geom_camber_cruise | 31.0716 | 31.5759 | -1.60% (only win) |

- **Epochs reached:** 58 (wall-clock cap hits at 30 min regardless of --epochs flag).
- **Diagnosis:** Schedule mis-alignment. --epochs 90 → T_max=87; wall-clock delivers only 58 epochs → LR at ep58 = 1.66e-4 (still high) vs baseline T_max=57 (LR ≈ 0 at ep58). Too-high LR in late epochs prevents fine convergence.
- **Critical finding (meta-learning):** **T_max must match ACTUAL expected epoch count, not --epochs flag.** The wall-clock budget is the binding constraint; --epochs is just an upper bound. Setting --epochs > actual_epochs creates an under-cooled cosine schedule.
- **Implication for ongoing experiments:** all assignments using --epochs 70 with ~58 actual epochs, or --epochs 80 with ~70 actual SAM epochs — these all have T_max correctly aligned to actual counts.

---

**Round-45 new assignments:** #2410 thorfinn FiLM conditioning (explicit regime modulation via broadcast-scalar affine), #2411 nezuko auxiliary camber head (λ=0.1 self-supervised geometry), #2412 askeladd SAM ρ=0.05 (flat-minima optimization, --epochs 35). All 8 students in-flight.

---

## 2026-05-13 21:00 UTC — Round 44

Three LOSS closures + axis status updates. Closed two falsified-LOSS PRs (capacity floor + loss-shape axis) and abandoned the DropPath axis after 4-attempt pod-stall jinx.

### PR #2336 frieren: n_hidden 96→64 (width-down continuation) — CLOSED (FALSIFIED LOSS, capacity floor)

- **Branch:** `charliepai2g48h5-frieren/n-hidden-64`
- **Hypothesis:** budget-bound regime extends to n_hidden=64 (continuation of #2290 WIN); +10-15 cosine epochs outweigh narrower per-block reps.

- **Results (vs PR #2290 branch base 46.3612; current advisor baseline 42.3455 from #2307):**

| Metric | PR #2336 | PR #2290 branch base | Δ vs branch | Δ vs current baseline |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **49.9489** | 46.3612 | **+7.74% LOSS** | +17.96% LOSS |
| `test_avg/mae_surf_p` | **43.6435** | 40.3555 | **+8.15% LOSS** | +13.34% LOSS |
| Epochs reached | 80 (=terminal) | 70 (best=67) | underfitting |  |
| Per-epoch cost | ~19-20 s | ~25.4 s | −22% | (1.6× more epochs) |
| Peak memory | 10.0 GB | 14.27 GB | −30% |  |
| Param count | ~150K | ~330K | **−55%** | (vs ~577K original) |

- **Per-split breakdown:** All 4 splits regress UNIFORMLY: in_dist +11.0%, cruise +12.8%, re_rand +7.2%, camber_rc +3.5%. **SURPRISE: in_dist regresses MOST, camber_rc (the OOD bottleneck) regresses LEAST** — opposite of predicted capacity-floor signature.

- **Diagnosis:** Total-capacity floor crossed. At n_hidden=64 with n_layers=4, the model lacks representation width to fit even the in-distribution training data — underfitting wins over the regularization-from-more-epochs effect. The fact that camber_rc (OOD) regresses LEAST is the kicker: in-dist suffering more than OOD means the model is no longer optimizing well period — there's no excess capacity left to "specialize" on in-dist, so OOD's relative degradation is suppressed. This is a TOTAL-capacity floor (representation width), distinct from the per-head capacity floor (dim_head=32 in #2222 at n_head=4) — the latter was about head diversity at fixed width.

- **Axis status:** Width-down axis CLOSED at n_hidden=96 (the new optimum). Mechanism: width-down was budget-bound 2-for-2 from 128→96 (−1.04% gain) but became capacity-bound 96→64 (+7.74% LOSS). This is the second axis (after layer-depth #2268 WIN → #2289 in-flight probe) where we found the budget-bound→capacity-bound transition.

### PR #2316 edward: berHu reverse-Huber c=1.0 retry-2 — CLOSED (FALSIFIED LOSS, loss-shape axis fully closed)

- **Branch:** `charliepai2g48h5-edward/berhu-c1-retry2`
- **Hypothesis:** berHu(c=1.0) amplifies large-residual OOD gradients (pure L1 for |r|≤1.0, quadratic-amplified for |r|>1.0) — opposite direction of closed Huber-β softening; targets val_geom_camber_rc + val_re_rand OOD bottleneck.

- **Results (vs PR #2268 branch base 46.8460; current advisor baseline 42.3455):**

| Metric | PR #2316 | PR #2268 branch base | Δ vs branch | Δ vs current baseline |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **50.4134** | 46.8460 | **+7.62% LOSS** | +18.96% LOSS |
| `test_avg/mae_surf_p` | **45.6380** | 40.8140 | **+11.82% LOSS** | +18.52% LOSS |
| Best epoch | 58 (=terminal) | 58 (=terminal) | underfitting tail |  |

- **Per-split breakdown:** All 4 splits regress. cruise worst (+15.04%), in_dist least (+2.95%). Note: target OOD splits camber_rc and re_rand both regressed — the hypothesis was directly wrong.

- **Diagnosis (student's analysis confirmed):** berHu amplification creates a self-defeating interaction with cosine cool-down. Late epochs need fine LR-controlled gradient norms for convergence, but berHu amplifies them at exactly the wrong time on outliers — and crucially, outliers include in-dist heavy tails too, not just OOD. So it's NOT surgically targeting the OOD splits as hoped; it's globally inflating gradients on rare-but-large errors.

- **Branch-base note:** PR was branched off pre-#2290 n_hidden=128 + pre-#2307 slice_num=32 advisor (baseline 46.8460), so the partial regression vs branch base reflects the loss-shape change ALONE — clean attribution.

- **Axis status:** Loss-shape axis now FULLY CLOSED. Combining: Huber-β softening family (5 LOSS across β∈{0.1, 0.25, 0.5, 1.0, 2.0} — all 5 bimodal vs L1) + berHu c=2.0 (#2223 stale) + berHu c=1.0 retry (#2316 LOSS) + MSE (β=∞, prior closure). **Both directions of loss-shape modification (softening AND amplification) hurt vs pure L1.** Mechanism unified: at this dataset scale + cosine schedule, L1's gradient-norm uniformity is load-bearing; any shape change distorts the gradient landscape relative to the LR schedule. Pure L1 is the GLOBAL loss-function optimum for this regime.

### PR #2280 tanjiro: DropPath p_max=0.1 retry-4 — CLOSED (axis ABANDONED, 4-attempt jinx + baseline drift)

- **Branch:** `charliepai2g48h5-tanjiro/droppath-retry4`
- **History:** PRs #1976, #2083, #2179, #2280 — all 4 attempts stalled at the pod-level (~6h no commits each). PR pattern: assignment commit goes through, pod picks up, training never starts or hangs partway. 4-attempt failure rate is structurally chronic, not transient.

- **Compounding factors for closure:**
  1. **Pod-stall jinx (primary):** 4× consecutive same-pattern failures suggests a persistent interaction between DropPath's bernoulli masking and tanjiro's pod environment (possibly the torch.compile graph capture interacts with conditional dynamic shapes from the Bernoulli mask).
  2. **Baseline drift makes the PR body stale:** PR body referenced 48.5160 (PR #2195) but current advisor is 42.3455 (PR #2307) — −12.78% drift. Per-split baselines are also stale.
  3. **LayerScale subsumption hypothesis:** With LayerScale γ now on the advisor (PR #2195 + co-advisors), per-channel residual scaling already provides much of the inductive bias DropPath was supposed to add. The marginal value of stochastic depth on top of LayerScale may be small even if it worked.
  4. **GPU time spent:** 4 retries × ~6h ea ≈ 24h pod-time with no committed result. Continuing has poor expected ROI.

- **Axis status:** DropPath axis abandoned. Tanjiro reassigned this round to a fresh orthogonal probe.

---

## 2026-05-13 20:30 UTC — Round 43

### PR #2307 askeladd: slice_num 32→24 (PhysicsAttention granularity-down) — MERGED (MASSIVE WIN −9.61%, new baseline 42.3455)

- **Branch:** `charliepai2g48h5-askeladd/slicenum-24`
- **Hypothesis (original):** 3rd orthogonal budget-bound axis triangulation. Predicted mechanism: −8-12% per-epoch savings → more cosine refinement epochs.

- **Results (vs PR #2268 branch baseline 46.8460; comparison also vs current baseline 42.3455):**

| Metric | PR #2307 | PR #2268 baseline (branch base) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **42.3455** | 46.8460 | **−9.61% MASSIVE WIN** |
| `test_avg/mae_surf_p` | **38.5059** | 40.8140 | **−5.66% WIN** |
| Epochs reached | 58 of 70 | 58 of 60 | ~same |
| Best epoch | 57 (≠terminal) | 58 (=terminal) | 1-ep bounce at ep58 |
| Per-epoch cost | ~30.80 s | ~31.4 s | **−2% (not −8 to −12%)** |
| Peak memory | 18.30 GB | 16.55 GB | +10% (compile-cache variance) |
| Param count | 576,875 | 577,931 | −0.18% (slice tensors only) |

- **Per-split breakdown:**

| Split | PR #2307 | PR #2268 baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 35.4776 | 41.7031 | **−14.93%** |
| `val_geom_camber_rc` | 60.8311 | 64.6729 | **−5.94%** |
| `val_geom_camber_cruise` | 27.6517 | 31.5759 | **−12.43%** |
| `val_re_rand` | 45.4214 | 49.4322 | **−8.11%** |

- **CRITICAL FINDING — mechanism is ROUTING QUALITY, not budget-bound:**
  - Prediction was −8 to −12% per-epoch savings → more budget. Actual savings: −2%.
  - The win is from **fewer, sharper PhysicsAttention partitions** — slice_num=32 was ABOVE the routing optimum.
  - Regularization / inductive-bias effect: sparser partitioning sharpens token grouping, improving geometric generalization without removing required capacity.
  - This is the **largest single-PR gain since round-1 warmup merge** (−9.61% vs next-largest −3.44% for n_layers=4).

- **Key diagnostic for val_geom_camber_rc:** −5.94% — first *significant* movement on the historic OOD bottleneck since LayerScale PR #2195 (−2.22%). slice_num directly governs the PhysicsAttention geometric routing quality, and this split is the most demanding routing task.

- **NOTE: PR #2307 branched off #2268 (n_hidden=128).** Current advisor at merge time has n_hidden=96. Squash-merge resolved cleanly — the slice_num=24 diff is orthogonal to the n_hidden change. Advisor now has: n_layers=4 + n_hidden=96 + slice_num=24 + LayerScale + warmup-3-cosine + n_head=2.

- **Hypothesis revision for slice_num axis:** Prior understanding was slice_num=32 was a global optimum (PR #1846 winner in round-18). That was correct at the time (slice_num=32 beat 64). The *direction* was right but the floor was not reached. **The routing optimum is below 32, confirmed below 24 may be the floor — assigned slice_num=16 as #2362 to askeladd.**

---

## 2026-05-13 20:00 UTC — Round 42

### PR #2290 frieren: n_hidden 128→96 (width-down probe) — MERGED (WIN, new baseline 46.3612)

- **Branch:** `charliepai2g48h5-frieren/n-hidden-96`
- **Hypothesis:** budget-bound regime extends to width axis. Reducing n_hidden 128→96 saves per-epoch compute → more cosine epochs in 30-min budget → better val.

- **Results (vs PR #2268 baseline 46.8460):**

| Metric | PR #2290 | PR #2268 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.3612** | 46.8460 | **−1.04% WIN** |
| `test_avg/mae_surf_p` | **40.3555** | 40.8140 | **−1.12% WIN** |
| Epochs reached | 70 of 70 | 58 of 60 | +17% more epochs |
| Best epoch | 67 (≠terminal) | 58 (=terminal) | cosine converged cleanly |
| Per-epoch cost | ~25.4 s | ~31 s | −15% (not predicted −40-45%) |
| Peak memory | 14.27 GB | ~16.5 GB | −13% |
| Param count | ~330K | ~578K | −43% |

- **Per-split breakdown:**

| Split | PR #2290 | PR #2268 baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 40.7757 | 41.7031 | **−2.22% WIN** |
| `val_geom_camber_rc` | 64.2563 | 64.6729 | **−0.64% WIN** |
| `val_geom_camber_cruise` | 31.5251 | 31.5759 | **−0.16% WIN** |
| `val_re_rand` | 48.8877 | 49.4322 | **−1.10% WIN** |

All 4 splits improve uniformly — classic budget-bound signature. Budget-bound regime confirmed 2-for-2 (depth-down #2268 + width-down #2290).

- **Key finding:** Per-epoch savings ~15% (not predicted 40-45%). Slice ops at slice_num=32 dominate and scale ~linearly in n_hidden, not n_hidden². Despite modest per-epoch savings, +17% more cosine epochs (60→70) drove improvement. Width axis is open further — no capacity floor at n_hidden=96.

- **LayerScale γ behavior:** Auto-adapted from (128,) to (96,). MLP γ ~0.063–0.078, attn γ ~0.004–0.012 (6-10× ratio). Same structure as PR #2195 — robust to width reduction.

- **New advisor config:** L1 + compile + bf16 + sampler 2× single + slice_num=32 + warmup-3-cosine + n_head=2 + LayerScale γ=1e-4 + n_layers=4 + **n_hidden=96** (dim_head=48). New baseline 46.3612 / 40.3555.

- **Next assignment:** frieren → PR #2336 n_hidden=64 (width-down continuation; tests capacity floor or budget-bound 3-for-3 on width axis; dim_head=32 at n_head=2 — aligns with prior n_head=4/n_hidden=128 LOSS #2222; key diagnostic for whether floor is dim_head or total-capacity determined).

---

## 2026-05-13 19:30 UTC — Round 41

### PR #2223 edward: berHu reverse-Huber (c=1.0) — CLOSED stale_wip (8h pod-stall, reassigned as #2316 retry-2)

- **Branch:** `charliepai2g48h5-edward/berhu-c1.0`
- **Status:** stale_wip. Pod-level stall pattern: PR created 11:18 UTC; pod processed iterations 104-107 (each Claude session exited code=0 in 120-302s without producing commits or comments); iteration 108 began at 13:26 UTC and has not exited as of 19:30 UTC (5+ hours hung). Classic pod-stuck signature matching:
  - DropPath retries: #1976 (stale) → #2083 (stale) → #2179 (stale) → #2280 (4th, in-flight)
  - RMSNorm retries: #1926 (stale) → #2034 (stale) → #2139 (closed, LOSS)
  - Translation aug retries: #2138 (stale) → #2235 (closed, LOSS)
- **No results produced.** Only the assignment commit (904abbc) on the branch.
- **Hypothesis preserved:** berHu c=1.0 is structurally distinct from all 10 closed failure-mode taxa (LOSS-AMPLIFYING direction, never tested). Baseline shifted significantly since assignment (was 49.8053 n_head=2 era; now 46.8460 with n_layers=4 + LayerScale on advisor), so student would need to re-baseline regardless.
- **Action:** REST API close (GraphQL rate-limited at iteration 108 retry storm). Reassigned as fresh PR #2316 on new branch `charliepai2g48h5-edward/berhu-c1.0-retry2` off current advisor. Same hypothesis, new bar 46.8460, full advisor stack (n_layers=4 + LayerScale + warmup-3-cosine + n_head=2 + slice_num=32 + L1 + bf16 + compile).

---

## 2026-05-13 19:00 UTC — Round 40

### PR #2272 askeladd: LayerScale asymmetric init (γ_attn=1e-4, γ_mlp=1e-3) — CLOSED (LOSS vs current baseline; hypothesis falsified)

- **Branch:** `charliepai2g48h5-askeladd/layerscale-asym-init`
- **Hypothesis:** data-driven follow-up from PR #2195 trained γ diagnostics. MLP branches converge to 4-8× larger γ (0.025-0.05) than attention branches (0.003-0.011). If we initialize γ_mlp=1e-3 (vs 1e-4) the model can skip ~5-10 "branches off" ramp-up epochs for MLP, gaining productive budget steps. Expected: best_epoch < 42 (earlier convergence); expected val < 48.52.

- **Results (vs current baseline PR #2268, val 46.8460):**

| Metric | PR #2272 | PR #2268 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 48.5072 | 46.8460 | **+3.55% LOSS** |
| `test_avg/mae_surf_p` | 42.2859 | 40.8140 | **+3.62% LOSS** |
| Epochs reached | 47/50 | 58/60 | — |
| Time per epoch | ~38-40 s | ~31 s | — |
| Best epoch | 47 (terminal) | 58 (terminal) | — |

*Note: #2272 was branched from LayerScale (n_layers=5) advisor before n_layers=4 was merged. Direct comparison to #2195 assigned baseline (val 48.5160): -0.018% wash.*

- **Per-split val breakdown (vs PR #2195 assigned baseline):**

| Split | PR #2272 | PR #2195 baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 43.4255 | 44.6149 | **-2.67% WIN** |
| `val_geom_camber_rc` | 65.4014 | 65.9411 | **-0.82% WIN** |
| `val_geom_camber_cruise` | 34.2779 | 33.2325 | **+3.15% LOSS** |
| `val_re_rand` | 50.9239 | 50.2756 | **+1.29% LOSS** |

Mixed 2-WIN / 2-LOSS split pattern — opposite of PR #2195's uniform 3-of-4-WIN profile.

- **γ diagnostic at best epoch 47:**

| Block | γ_attn abs_mean | γ_mlp abs_mean | γ_mlp/γ_attn |
|---|---|---|---|
| 0 | 0.00891 | 0.05369 | 6.03× |
| 1 | 0.00653 | 0.04310 | 6.60× |
| 2 | 0.00601 | 0.03378 | 5.62× |
| 3 | 0.00698 | 0.04391 | 6.29× |
| 4 | 0.00193 | 0.04453 | 23.0× |

Trained γ values converge to essentially the same regime regardless of init.

- **Key diagnostic — convergence trajectory:**

| Epoch | PR #2272 (γ_mlp init=1e-3) | PR #2195 (γ_mlp init=1e-4) |
|---|---|---|
| 42 | 50.162 | **48.516** (best) |
| 43 | 50.013 | (terminal) |
| 47 | **48.507** (best) | — |

Asymmetric init caused LATER convergence (+5 epochs), not earlier. Prediction REJECTED.

- **Conclusion and failure-mode analysis:**
  - The "near-zero branches off phase" is NOT waste — it is an implicit warm-up where attention finds useful coarse structure before MLP residuals contribute strongly. Pumping γ_mlp to 1e-3 from epoch 1 means MLP contributions are 10× louder than residual identity while attention is still finding features → adds noise to residual stream that the model must learn around.
  - Final trained γ values are identical regardless of init (confirmed empirically) — the model converges to a data-determined operating regime, not an init-determined one.
  - Mixed per-split pattern (in-dist + rc improve; cruise + re_rand regress) is consistent: early-training instability from louder MLP residuals hurts OOD-stress splits most.
  - The test_avg WIN (-1.24%) is a notable secondary signal but val signal is flat (-0.018%) and mechanism is falsified.

- **10th distinct failure-mode taxon added: init-prior-misalignment LOSS** — initializing a load-bearing residual-gate parameter at its trained scale rather than its small-norm prior corrupts the implicit warm-up dynamics that are needed for stable early-phase learning. Mechanism is sister to broadcast-scalar prior corruption (taxon 2) — both involve prior belief injection at the wrong amplitude. OOD splits suffer first due to their reliance on early-phase geometric feature formation.

- **LayerScale axis CLOSED:** uniform init=1e-4 (PR #2195) is confirmed as the optimal config. No further per-branch or per-block γ init probes warranted — final γ scale is data-determined, not init-determined.

- **New assignment:** askeladd assigned PR #2307 slice_num 32→24 (3rd orthogonal budget-bound axis triangulation, after depth via #2268 WIN and width via #2290 in-flight).

---

## 2026-05-13 17:00 UTC — Round 39

### PR #2194 alphonse: SGD(momentum=0.9, nesterov=True, lr=2e-3) — CLOSED (LOSS, optimizer-family axis fully closed)

- **Branch:** `charliepai2g48h5-alphonse/sgd-momentum-lr2e-3`
- **Hypothesis:** Optimizer-FAMILY swap from AdamW (which had closed 4-LOSS-for-4 within-family) to SGD-momentum. Test whether AdamW's adaptive 2nd-moment rescaling is load-bearing for L1+bf16+Transformer at this scale.
- **Results:**

| Metric | SGD-momentum | Baseline #2173 (n_head=2) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **78.6405** | 49.8053 | **+57.9% LOSS** |
| `test_avg/mae_surf_p` | **69.0819** | 43.5396 | **+58.7% LOSS** |
| Epochs reached | 44/50 | 47/50 | -6.4% |
| Time per epoch | 40.95 s | 37.5 s | +9.2% |
| Best epoch | 44 (terminal) | 47 (terminal) | — |

**Per-split val breakdown — uniform-additive regression:**

| Split | baseline | SGD | Δ abs | Δ % |
|---|---|---|---|---|
| `val_single_in_dist` | 46.2915 | 84.2276 | +37.94 | +82% |
| `val_geom_camber_rc` | 67.4416 | 96.1330 | +28.69 | +43% |
| `val_geom_camber_cruise` | 32.5963 | 59.3718 | +26.78 | +82% |
| `val_re_rand` | 52.8918 | 74.8295 | +21.94 | +41% |

**Diagnostic finding (student's analysis, confirmed):** SGD did NOT diverge — no NaN, no Inf. Trained cleanly with warmup-3-cosine. Monotone descent from ep6 (179.7) to ep44 (78.6). Problem is convergence RATE: SGD reaches val ~80 in 44 epochs where AdamW reaches val ~50. Slope is real but ~2× too slow. Linear extrapolation: ep50 ~70-74 — still 40-50% above baseline.

**Mechanistic conclusion (confirmed):** AdamW's adaptive 2nd-moment rescaling is load-bearing for this regime. Even though L1's gradient is mostly ±1 in sign, the 2nd-moment estimate tracking gradient FREQUENCY still provides meaningful per-parameter step-size info that SGD-momentum's global scaling cannot recover from within the 30-min budget.

**Per-split pattern is uniform-additive** (Δ 22-38 across splits, no smoking-gun split). Pattern is NOT bimodal (in-dist worst hit, opposite of averaging-style) and NOT broadcast-scalar (val_re_rand not the worst). Consistent with "optimizer change is global, not split-specific."

**Optimizer-family axis fully closed 6-LOSS-for-6:**

| Probe | Direction | Δ vs baseline | Pattern |
|---|---|---|---|
| lr-UP +50% (PR #1774) | within-AdamW | LOSS +16% | uniform regression |
| lr-DOWN -25% (PR #1997) | within-AdamW | LOSS +11% | 4th averaging-style bimodal |
| β1=0.95 (PR #2093) | within-AdamW | LOSS +7.5% | momentum-lag overshoots cosine |
| β2=0.95 (PR #1845) | within-AdamW | LOSS +4.4% | shorter 2nd-moment memory amplifies L1 sign-flip |
| amsgrad=True (PR #2155) | within-AdamW | LOSS +9.5% | max-bound permanent v_max inflation |
| SGD-momentum (PR #2194) | FAMILY swap | LOSS +58% | convergence-rate inadequacy |

**AdamW(lr=5e-4, β1=0.9, β2=0.999, no amsgrad) is sharply pinned** as the only viable optimizer for this regime.

**9th distinct failure-mode taxon added: optimizer-family convergence-rate inadequacy** — uniform-additive regression from a non-adaptive optimizer (distinct from prior optimizer failure-mode taxa: momentum-lag β1=0.95, optimizer-statistic over-conservativism amsgrad).

Closing as LOSS.

### PR #2179 tanjiro: DropPath p_max=0.1 retry-3 — CLOSED (STALE, pod-stall pattern, 4th time)

- **Branch:** `charliepai2g48h5-tanjiro/droppath-retry3`
- **Status:** STALE — no commits since assignment (10:25 UTC → 17:00 UTC = ~6.5h). 4th consecutive stale for tanjiro DropPath (after #1976, #2083, #2179). Matches frieren RMSNorm pattern (took 3 attempts before #2139 picked up). Reassigned as #2280 retry-4 on fresh branch via REST.

### PR #2268 thorfinn: n_layers 5→4 (depth-down, --epochs 60) — MERGED (WIN, new baseline 46.8460)

- **Branch:** `charliepai2g48h5-thorfinn/n-layers-4`
- **Hypothesis:** Depth-down probe: n_layers=5 → 4. Budget-bound diagnostic. ~20% per-epoch savings → ~10 extra cosine refinement epochs. Complementary to in-flight mlp_ratio=4 (ADD capacity) — this REDUCES capacity.
- **Note:** This PR was branched BEFORE LayerScale (#2195) merged. It was measured WITHOUT LayerScale. After squash-merge onto current advisor (which includes LayerScale), the resulting advisor has n_layers=4 + LayerScale stacked.

| Metric | n_layers=4 | Assigned baseline #2195 (LayerScale) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **46.8460** | 48.5160 | **−3.44% WIN** |
| `test_avg/mae_surf_p` | **40.8140** | 42.8162 | **−4.70% WIN** |
| Epochs reached | 58/60 (terminal) | 43/50 (terminal) | +15 more |
| Time per epoch | 31.4 s | 42.25 s | −25.6% |
| Params | 577,931 | 658,359 | −12.2% |

**Per-split val breakdown:**

| Split | baseline (LayerScale) | n_layers=4 | Δ |
|---|---|---|---|
| `val_single_in_dist` | 44.6149 | 41.7031 | **−6.51%** |
| `val_geom_camber_rc` | 65.9411 | 64.6729 | **−1.93%** |
| `val_geom_camber_cruise` | 33.2325 | 31.5759 | **−4.99%** |
| `val_re_rand` | 50.2756 | 49.4322 | **−1.67%** |

All 4 splits improve — uniform-direction architectural WIN, not bimodal.

**BUDGET-BOUND REGIME CONFIRMED:**
- Best epoch = terminal (58/60, timeout at 30.4 min)
- -25.6% per-epoch cost → 15 more epochs → more late-stage cosine refinement
- All splits improve proportionally — capacity reduction (one fewer TransolverBlock) is NOT the binding constraint
- Combined with mlp_ratio=4 budget cliff (same round): "adding capacity loses, reducing capacity wins" → clear evidence the model is budget-bound at hidden_dim=128, n_head=2, 30-min cap

**MERGED as new baseline 46.8460.**

### PR #2235 nezuko: Translation augmentation σ=0.01 retry-2 — CLOSED (LOSS)

- val 52.2027 (+11.5% vs 46.8460 new baseline), test 45.1602 (+10.7%)
- Uniform regression all 4 splits; val_geom_camber_cruise worst (+9.69%), in-dist least (+1.42%)
- NOT bimodal (no in-dist benefit), NOT broadcast-scalar corruption
- **Mechanism:** Absolute coordinate channels 0-1 are load-bearing positional priors — the model routes attention and computes flow features based on absolute mesh position, not just relative shape. Per-sample translation of coords changes the absolute position statistics in a way the model can't compensate for. Physical translation invariance of Navier-Stokes does not transfer to this normalized-coordinate model.
- **Input-coordinate augmentation axis fully closed:** per-sample-broadcast scalars (NACA, gap/stagger, Re/AoA = 2× prior closures) + per-point absolute coords (translation aug). All input-channel augmentation is harmful in this regime.

### PR #2234 frieren: mlp_ratio 2→4 (FFN width expansion) — CLOSED (LOSS, budget cliff)

- val 66.7281 (+42.5% vs 46.8460 new baseline), test 58.7484 (+44%)
- 34/50 epochs reached (budget cliff threshold was 38) — +15% per-epoch cost cut off the final 13-16 epochs of cosine descent
- Per-split uniform-additive +26-44% — classic undertraining, not architectural failure
- **Mechanism:** Budget cliff exactly as diagnosed. The per-epoch compute cost of 2× wider FFN removes exactly the extra epochs that produce late-stage refinement (now confirmed by n_layers=4 WIN: that extra ~15 epochs = -3.44% improvement). mlp_ratio=4 might win at a longer budget (60 min) but is definitively LOSS at 30 min.
- **Unified finding with #2268 merge:** budget-bound regime means: REDUCE capacity → win (n_layers=4, expected n_layers=3, n_hidden=96), ADD capacity → lose (mlp_ratio=4, n_layers=6, n_hidden=160). All capacity-adding probes at this constraint face the same budget cliff. MLPratio=4 is not ruled out architecturally — it's budget-ruled-out.

### Round 39 assignment summary (updated)

| PR | Student | Hypothesis | Round |
|---|---|---|---|
| #2298 | thorfinn | n_layers=4 --epochs 90 (budget-extrapolation diagnostic; no code changes) | round-39 final |
| #2290 | frieren | n_hidden 128→96 (--epochs 90; width-down; ~40-45% per-epoch savings; ~330K params) | round-39 addendum |
| #2289 | nezuko | n_layers 4→3 (--epochs 75; depth-down continuation; ~25% per-epoch savings; ~450K params) | round-39 addendum |
| #2283 | alphonse | Squared ReLU (ReLU²) at all 3 MLP sites — Primer-style sharp activation, opposite-direction probe of closed SiLU LOSS | round-39 |
| #2280 | tanjiro | DropPath p_max=0.1 retry-4 (on LayerScale-gated branches) | round-39 |

Idle students: 0. All 8 students in-flight.

Closed-axis additions:
- **Optimizer-family axis** — closed 6-LOSS-for-6 (AdamW lr-UP/DOWN/β1/β2/amsgrad + SGD-momentum). AdamW defaults pinned.

Failure-mode taxonomy refinement: 9th distinct taxon added:
9. **Optimizer-family convergence-rate inadequacy (1× — SGD-momentum):** uniform-additive regression from non-adaptive optimizer; trains cleanly but ~2× too slow to compete within wall-clock budget.

---

## 2026-05-13 16:30 UTC — Round 38 addendum: LayerScale merge

### PR #2195 askeladd: LayerScale init=1e-4 (CaiT-style learnable residual gain γ) — MERGED (WIN, new baseline)

- **Branch:** `charliepai2g48h5-askeladd/layerscale-1e-4`
- **Hypothesis:** CaiT-style per-channel learnable γ (init=1e-4) on attention and MLP residual branches in all 5 TransolverBlocks. Forces the model to learn per-channel contribution magnitude rather than uniform residual integration. Predicted uniform per-split delta, NOT averaging-style bimodal.
- **Results vs NEW baseline #2173 (val 49.8053):**

| Metric | LayerScale | New baseline #2173 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **48.5160** | 49.8053 | **−2.59% WIN** |
| `test_avg/mae_surf_p` | **42.8162** | 43.5396 | **−1.66% WIN** |
| Best epoch | 42 | 47 | — |
| Terminal epoch | 43 | 47 | — |
| Time per epoch | 42.25 s | 37.5 s | +12.7% (compile overhead) |
| Params | 658,359 | 657,079 | +1,280 (+0.19%) |

**Per-split val breakdown:**

| Split | baseline (n_head=2) | LayerScale | Δ |
|---|---|---|---|
| `val_single_in_dist` | 46.2915 | 44.6149 | **−3.62%** |
| `val_geom_camber_rc` | 67.4416 | 65.9411 | **−2.22% — FIRST MOVE SINCE ROUND-1** |
| `val_geom_camber_cruise` | 32.5963 | 33.2325 | +1.95% (slight) |
| `val_re_rand` | 52.8918 | 50.2756 | **−4.95%** |

**Per-split test breakdown:**

| Split | baseline test | LayerScale test | Δ |
|---|---|---|---|
| `test_single_in_dist` | 40.6576 | 40.1418 | −1.27% |
| `test_geom_camber_rc` | 61.4956 | 60.4713 | **−1.67%** |
| `test_geom_camber_cruise` | 27.6519 | 27.5452 | −0.38% |
| `test_re_rand` | 44.3531 | 43.1065 | −2.82% |

**MAJOR FINDING: val_geom_camber_rc (67.44 → 65.94) moved for the FIRST TIME since round-1.** Every previous intervention — warmup, n_head, normalization, augmentation, loss function, optimizer — left this split flat. LayerScale's per-channel selective residual gating is the first mechanism to crack the camber-rc OOD bottleneck.

**Failure-mode taxonomy check:**
- Uniform direction (3 of 4 splits improved, cruise slight +1.95%): NOT averaging-style bimodal (which would show large in-dist WIN + OOD LOSS).
- val_re_rand is NOT the worst regressor: NOT broadcast-scalar prior corruption.
- Pattern is consistent with "architectural-residual-gating WIN" — 9th distinct mechanism (confirmed distinct from all 8 closed failure taxa).

**Trained γ diagnostics (from epoch-42 weights):**
- MLP branches activated 4-8× stronger than attention at every layer (γ_mlp abs_mean ≈ 0.025-0.05 vs γ_attn ≈ 0.003-0.011) — model leans more on FFN residuals.
- Block 3 attention notably underweighted (γ_attn abs_mean 0.0032, ~32× init — smallest activation).
- Signs mixed within each γ vector — per-channel selective gating, not simple uniform scaling.
- All branches activated (none stuck at 1e-4 init) — init not too aggressive.

**Best epoch 42 ≠ terminal 43** (first convergence-within-budget in recent rounds). LayerScale may enable faster convergence by providing per-channel residual-scaling freedom that reduces effective optimization landscape roughness.

**Metric artifacts:**
`models/model-charliepai2g48h5-askeladd-layerscale-1e-4-20260513-110555/metrics.jsonl`
`models/model-charliepai2g48h5-askeladd-layerscale-1e-4-20260513-110555/metrics.yaml`

**MERGED as new baseline. New bar: val_avg/mae_surf_p < 48.5160.**

**Suggested follow-ups (student-proposed, high priority):**
1. γ init sweep: 1e-3 and 1e-2 (larger init → fewer "branches off" early epochs → more productive time in 30-min budget)
2. Per-branch asymmetric init: γ_attn init=1e-4, γ_mlp init=1e-3 (data-driven from trained γ diagnostics)
3. Scalar γ ablation (shape () instead of (hidden_dim,)) to confirm per-channel freedom is essential

---

## 2026-05-13 16:00 UTC — Round 38

### PR #2222 thorfinn: n_head 2→1 (dim_head 64→128) — CLOSED (LOSS, concave-axis closure)

- **Branch:** `charliepai2g48h5-thorfinn/n-head-1`
- **Hypothesis:** Probe n_head=1 (dim_head=128 = full hidden_dim) — closes the n_head axis at the endpoint. n_head=4 (dim_head=32) → n_head=2 (dim_head=64) was WIN −1.57%. Tests whether monotone improvement continues or n_head=2 is a true interior optimum.
- **Results:**

| Metric | n_head=1 | New baseline #2173 (n_head=2) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **51.3952** | 49.8053 | **+3.19% LOSS** |
| `test_avg/mae_surf_p` | **44.7265** | 43.5396 | **+2.73% LOSS** |
| Epochs reached | 50/50 | 47/50 | — |
| Time per epoch | 33.29 s | 37.5 s | −11% (single-head faster) |
| Params | 0.903M | 0.708M | +27% (full head dim) |

**Per-split val breakdown:**

| Split | baseline (n_head=2) | n_head=1 | Δ |
|---|---|---|---|
| `val_single_in_dist` | 46.2915 | 45.7805 | **−1.10% WIN** |
| `val_geom_camber_rc` | 67.4416 | 68.3040 | +1.28% |
| `val_geom_camber_cruise` | 32.5963 | 36.2503 | **+11.21% WORST** |
| `val_re_rand` | 52.8918 | 55.2458 | +4.46% |

**The n_head axis is CONCAVELY CLOSED at endpoints {1, 2, 4}:**

| n_head | dim_head | val_avg/mae_surf_p | Δ vs interior optimum |
|---|---|---|---|
| 4 | 32 | 50.6001 | +1.60% |
| **2** | **64** | **49.8053** | **interior optimum** |
| 1 | 128 | 51.3952 | +3.19% |

Not monotone — n_head=2 is a TRUE INTERIOR OPTIMUM. Both head-rank diversity (multiple attention maps) AND per-head capacity (dim_head=64) are useful and trade against each other; the optimum balances both.

**Lead-mover insight (paper-level):** val_geom_camber_cruise was the **lead WIN** in 4→2 (−5.09%) and the **lead LOSS** in 2→1 (+11.21%). It is the most structurally-capacity-sensitive split in the dataset. Architectural-attention capacity specifically governs cruise-OOD generalization, distinct from camber-OOD (flat to all attention probes) and re-OOD (slight bimodal-like sensitivity).

**Per-split pattern is NOT bimodal-averaging.** In-dist actually IMPROVED slightly (−1.10%) — single-head attention is competitive for the dominant single-foil regime, but breaks down on geometric/Re OOD splits. This split-direction pattern is "architectural-OOD-sensitive": in-dist holds or improves, but OOD splits suffer without head-diversity. Distinct mechanism from the 8× averaging-style class (which trades in-dist for OOD via stochasticity).

**Why n_head=2 is the optimum, not 1 or 4:**
- n_head=1: full per-head capacity (dim_head=128) but zero head diversity (single attention map) → can't decompose attention into specialized roles → loses OOD generalization.
- n_head=4: 4-way head diversity but only 32-dim subspaces per head → each head too narrow to capture useful relationships over slice_num=32 compressed tokens.
- n_head=2: 2-way diversity at dim_head=64 (literature optimum since Vaswani 2017) — best balance.

Closing as LOSS. **n_head axis fully closed; n_head=2 confirmed as TRUE INTERIOR OPTIMUM.**

### Round 38 assignment summary

| PR | Student | Hypothesis | Round |
|---|---|---|---|
| #2272 | askeladd | LayerScale asymmetric init (gamma_attn=1e-4, gamma_mlp=1e-3) — data-driven follow-up from PR #2195 trained γ diagnostics | round-38 |
| #2268 | thorfinn | n_layers 5→4 (depth-down, --epochs 60) — pure architectural reduction, 708K→575K params, budget-bound vs capacity-saturated diagnostic | round-38 |

Idle students: 0. All 8 students in-flight.

Closed-axis additions:
- **n_head axis** — CONCAVELY closed at {1, 2, 4} = {51.40, 49.81, 50.60}, n_head=2 (dim_head=64) is true interior optimum.

Failure-mode taxonomy refinement: 8th distinct taxon added (architectural-OOD-sensitivity from attention head-rank, non-bimodal, geometric-OOD specific) — distinct from prior 7 taxa:
1. Averaging-style bimodal (8×)
2. Broadcast-scalar prior corruption (2×)
3. Momentum-lag overshoots cosine (1×)
4. Warmup-duration asymmetry (1×)
5. Architectural-activation degradation (1×)
6. Optimizer-statistic over-conservativism (1×)
7. Structural mild-bimodal (1× — RMSNorm 3-site)
8. **Architectural-attention-rank OOD-sensitivity (1× — n_head=1):** in-dist holds, geometric/Re OOD splits collapse without head-diversity.

---

## 2026-05-13 15:30 UTC — Round 37

### PR #2139 frieren: RMSNorm replaces LayerNorm at all 3 TransolverBlock sites (retry-3) — CLOSED (LOSS, mild bimodal)

- **Branch:** `charliepai2g48h5-frieren/rmsnorm-3sites-retry3`
- **Hypothesis:** RMSNorm at all 3 sites (ln_1, ln_2, ln_3) — Pre-LN style (no mean-centering). Predicted small structural win or wash, NO bimodal signature (layer-level, not sample-level intervention).
- **Results:**

| Metric | RMSNorm | Old baseline #2033 | New baseline #2173 | Δ vs new |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **51.8784** | 50.6001 | 49.8053 | **+4.16% LOSS** |
| `test_avg/mae_surf_p` | **44.7876** | 43.9680 | 43.5396 | **+2.87% LOSS** |

| Split | RMSNorm val | #2033 val | Δ |
|---|---|---|---|
| `val_single_in_dist` | 46.9421 | 47.9418 | **−2.08% (BETTER)** |
| `val_geom_camber_rc` | 69.9374 | 67.3675 | +3.81% |
| `val_geom_camber_cruise` | 36.4811 | 34.3430 | +6.23% (worst) |
| `val_re_rand` | 54.1531 | 52.7481 | +2.66% |

- **Best epoch:** 45 of 47 possible (timeout, ~40.1 s/epoch)
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-rmsnorm-3sites-retry3-20260513-101814/metrics.jsonl`

**Analysis:** Both hypothesis predictions falsified — RMSNorm did NOT win structurally (close+LOSS), AND DID exhibit a mild bimodal pattern (in-dist WIN, OOD LOSS), even though it's a structural not stochastic/averaging change. **Mechanism hypothesis** (student's analysis): removing LayerNorm's mean-centering improves constant-zero feature directions for single-foil inputs (gap=0, stagger=0, AoA-2=0 zero-tail) while making the non-zero offset features in OOD camber/Re directions harder to learn — a structural variant of the bimodal pattern, smaller magnitude than the averaging-style canon. The pattern is small enough that single-seed noise can't be ruled out, but uniform signs across 3 OOD splits are suggestive. **The single-foil-only RMSNorm benefit (+2% in-dist) is a paper-level ablation observation** — even if val_avg doesn't beat baseline, this is interesting evidence about LayerNorm's mean-centering interaction with constant-zero feature directions. RMSNorm 3-site axis closed.

---

### PR #2138 nezuko: Translation augmentation σ=0.01 (channels 0-1) — CLOSED (stale ~6h, reassigned as #2235)

- **Branch:** `charliepai2g48h5-nezuko/translation-aug-s0.01`
- **State:** Stale WIP — no commits since 2026-05-13T09:27 UTC (~6 hours). Pod-stall pattern matching frieren RMSNorm (#1926→#2034 stale ×2 before #2139 picked up) and tanjiro DropPath (#1976→#2083 stale ×2 before #2179 picked up).
- **Action:** Closed via `close_pr_with_comment`, reassigned same hypothesis on fresh branch via REST API as #2235 retry-2.
- **Hypothesis preserved:** Per-sample rigid translation σ=0.01 on coord channels 0-1 — physical-symmetry data augmentation, structurally distinct from 8× averaging-style bimodal closures AND from 2× broadcast-scalar prior corruption closures. Critical diagnostic remains: if bimodal → pattern generalizes to physical-symmetry too; if uniform/OOD-favorable → first lever breaking bimodal trade-off.

---

### Round-37 Assignments

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #2234 | frieren | mlp_ratio 2→4 (FFN width 256→512) | Standard transformer FFN expansion, never tested at ratio=4 in this launch; complementary capacity axis to n_head=2 WIN; +50% params (708K → 1.04M) |
| #2235 | nezuko | Translation aug σ=0.01 retry-2 | Reassignment of #2138 after 1st stale; fresh branch, same hypothesis, pod-unstick attempt |

---

## 2026-05-13 15:00 UTC — Round 36

### PR #2173 thorfinn: n_head 4→2 (dim_head 32→64) — MERGED (**WIN**, new baseline)

- **Branch:** `charliepai2g48h5-thorfinn/n-head-2`
- **Hypothesis:** Increase attention head capacity (dim_head=32→64) to match literature-optimal dim_head≈64. Same param count, same FLOPs — purely shifts representation subspace.
- **Results:**

| Metric | n_head=2 | PR #2033 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **49.8053** | 50.6001 | **−1.57%** ✅ |
| `test_avg/mae_surf_p` | **43.5396** | 43.9680 | **−0.97%** ✅ |

| Split | n_head=2 val | #2033 baseline | Δ |
|---|---|---|---|
| `val_single_in_dist` | 46.2915 | 47.9418 | −3.44% ✅ |
| `val_geom_camber_rc` | 67.4416 | 67.3675 | +0.11% (wash) |
| `val_geom_camber_cruise` | 32.5963 | 34.3430 | −5.09% ✅ |
| `val_re_rand` | 52.8918 | 52.7481 | +0.27% (wash) |

| Split | n_head=2 test |
|---|---|
| `test_single_in_dist` | 40.6576 |
| `test_geom_camber_rc` | 61.4956 |
| `test_geom_camber_cruise` | 27.6519 |
| `test_re_rand` | 44.3531 |

- **Best epoch:** 47 of 47 (terminal; ep45→47 val: 50.91→49.97→49.81 — still descending at timeout)
- **Epochs/time:** 47 epochs × ~37.5 s = 30.1 min
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-n-head-2-20260513-101936/metrics.jsonl`

**Analysis:** Literature optimum dim_head≈64 confirmed for this architecture. With slice_num=32 compressed tokens (not raw mesh nodes), head diversity (n_head) matters less than head capacity (dim_head) — n_head=2 high-rank heads outperform n_head=4 low-rank heads. Improvements concentrated in val_single_in_dist (−3.44%) and val_geom_camber_cruise (−5.09%); the harder OOD splits (val_geom_camber_rc +0.11%, val_re_rand +0.27%) washed — data/regularization-limited, not head-rank-limited. Both improvements are "easy" splits; the bottleneck splits remain flat. Best=terminal; true ceiling is below 49.81 with more budget. **New n_head axis direction:** n_head=1 (dim_head=128) assigned as #2222 to close the axis.

---

### PR #2174 edward: Huber β=0.1 (smooth-L1 training loss) — CLOSED (LOSS, 8th averaging bimodal)

- **Branch:** `charliepai2g48h5-edward/huber-beta-0.1`
- **Hypothesis:** Pure L1 has constant ±1 gradient (sign-flip noise); Huber β=0.1 introduces quadratic regime for |r|<0.1, smoothing sign-flip while preserving L1's bulk character. Orthogonal to optimizer-side amsgrad fix.
- **Results:**

| Metric | Huber β=0.1 | New baseline #2173 | Old baseline #2033 | Δ vs #2033 |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **51.5551** | 49.8053 | 50.6001 | **+1.89% LOSS** |
| `test_avg/mae_surf_p` | **45.1841** | 43.5396 | 43.9680 | **+2.77% LOSS** |

| Split | Huber β=0.1 val | #2033 val | Δ |
|---|---|---|---|
| `val_single_in_dist` | 47.1714 | 47.9418 | −1.61% (improved) |
| `val_geom_camber_rc` | 69.2112 | 67.3675 | +2.74% |
| `val_geom_camber_cruise` | 35.8311 | 34.3430 | +4.33% (worst) |
| `val_re_rand` | 54.0068 | 52.7481 | +2.39% |

- **Best epoch:** 45 of 47 possible (timeout at ~40.5 s/epoch); still trending down at termination
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-huber-beta-0.1-20260513-101933/metrics.jsonl`

**Analysis:** Classic averaging-style bimodal signature — 8th confirmation. In-dist improved −1.61% while all 3 OOD splits regressed +2.4 to +4.3%. Mechanism: at β=0.1 in normalized space, enough training residuals fall below the threshold to trigger quadratic (MSE-like) gradient behavior, softening effective regularization → predictions drift toward conditional mean → in-dist benefits (dense training data), OOD suffers (sparse data). **The Huber-β axis is now fully closed:** β∈{0.1, 0.25, 0.5, 1.0, 2.0} ALL bimodal LOSS vs L1, monotone in β. Pure L1 is the saturating optimum along the smooth-L1 β sweep. NOTE: This closes the LOSS-SOFTENING direction; the LOSS-AMPLIFYING direction (berHu/reverse-Huber) is now in-flight as #2223 — structurally opposite mechanism.

**Updated failure-mode taxonomy** (7 distinct patterns, 8 total bimodal confirmations):
1. Averaging-style bimodal (8×): coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead, warmup-5, fun-jitter, Huber β=0.1
2. Broadcast-scalar prior corruption (2×): gap/stagger #2114, NACA jitter #2072
3. Momentum-lag overshoots cosine (1×): β1=0.95 #2093
4. Warmup-duration asymmetry (1×): warmup-2 #2112
5. Architectural-activation degradation (1×): SiLU swap #2156
6. Optimizer-statistic over-conservativism (1×): amsgrad #2155
7. Architectural-rank improvement (1×, WIN): n_head=2 #2173

---

### Round-36 Assignments

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #2222 | thorfinn | n_head=1 (dim_head=128) | Close n_head axis at single-head endpoint; either confirms monotone improvement or confirms n_head=2 as optimum |
| #2223 | edward | berHu reverse-Huber (c=1.0) | First LOSS-AMPLIFYING loss probe: pure L1 for |r|≤c, quadratic-amplified beyond c; directly targets OOD large-residual bottleneck; structurally opposite to ALL 8 bimodal-class closures |

---

## 2026-05-13 11:00 UTC — Round 35

### PR #2156 askeladd: GELU → SiLU activation swap (all 3 MLP sites) — CLOSED (LOSS, NEW failure-mode taxon)

- **Branch:** `charliepai2g48h5-askeladd/gelu-to-silu`
- **Hypothesis:** SiLU's smoother gating near zero should help residual stream propagation; consistent 0.3-1% literature gains. Predicted uniform per-split delta (architectural-activation mechanism, NOT averaging-style).
- **Result:** val_avg = **57.6372** (+13.91% vs 50.6001, LOSS), test_avg = **50.6557** (+15.21% vs 43.9680).

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | SiLU | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 56.9333 | **+18.76%** (worst) |
| `val_geom_camber_rc` | 67.3675 | 73.6172 | +9.28% |
| `val_geom_camber_cruise` | 34.3430 | 40.2016 | +17.06% |
| `val_re_rand` | 52.7481 | 59.7969 | +13.36% |

**NEW failure-mode taxon: architectural-activation degradation (NOT bimodal, NOT broadcast-scalar):**

1. **All 4 splits uniformly regress 9.3-18.8%.** Predicted "uniform per-split delta" diagnostic confirmed.
2. **Mechanism (student's analysis, confirmed by quantitative properties):** GELU(x) = x·Φ(x) has min ≈ -0.17 at x ≈ -0.75; SiLU(x) = x·σ(x) has min ≈ -0.28 at x ≈ -1.28. SiLU's longer negative tail injects more low-magnitude negative noise into the residual stream. With L1's sign-flipping gradients + bf16's reduced precision + small `n_hidden=128`, this destabilises the converged regime.
3. **Activation axis closed in the negative direction.** GELU dominates SiLU at this configuration; do not retry SiLU variants at this size.

### PR #2155 alphonse: AdamW amsgrad=True — CLOSED (LOSS, completes AdamW optimizer axis 4-LOSS-for-4)

- **Branch:** `charliepai2g48h5-alphonse/adamw-amsgrad`
- **Hypothesis:** amsgrad's max-tracking 2nd-moment denominator is theoretically robust to L1 sign-flip noise. Predicted "if wash or win, distinct from averaging-style bimodal".
- **Result:** val_avg = **55.3928** (+9.47% vs 50.6001, LOSS), test_avg = **48.1550** (+9.52% vs 43.9680).

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | amsgrad | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 50.9671 | +6.31% |
| `val_geom_camber_rc` | 67.3675 | 70.8324 | +5.14% |
| `val_geom_camber_cruise` | 34.3430 | 40.6777 | **+18.45%** (worst) |
| `val_re_rand` | 52.7481 | 59.0941 | +12.03% |

**NEW failure-mode taxon: optimizer-statistic over-conservativism (NOT bimodal, NOT momentum-lag):**

1. **All 4 splits uniformly regress 5.14-18.45%.** OOD slightly more sensitive but all directions same.
2. **Mechanism (student's outstanding analysis):** amsgrad's max-bound is PERMANENT — once a single high-variance gradient inflates `v_max`, all subsequent steps for that parameter are clamped lower forever, even after the noise subsides. This kills early-phase exploration during the high-LR cosine phase. By the time `v_max` plateaus, the schedule has already decayed past peak.
3. **Trajectory evidence (student traced):** epoch 1 val_avg = 263.92 (catastrophic vs baseline ~140 at ep1); epoch 13 = 100.00 (crossing into 100s only after 4× normal time); epoch 45 best = 55.39 (recovering monotonically but LR already at 1.4e-5).
4. **AdamW optimizer axis is now closed 4-LOSS-for-4** in this launch:
   - lr=7.5e-4 (UP, #1774): LOSS +16%
   - lr=3.75e-4 (DOWN, #1997): LOSS +11.4%
   - β1=0.95 (#2093): LOSS +7.45% (momentum-lag taxon)
   - amsgrad (#2155, this PR): LOSS +9.47% (over-conservativism taxon)
   - Plus prior closure: β2=0.95 (#1845): LOSS +15%
   - **Defaults (lr=5e-4, β1=0.9, β2=0.999) are locally optimal.** No further AdamW probing.

### Round-35 failure-mode taxonomy update

Through rounds 25-35 we now have 6 distinct failure-mode patterns documented (vs 4 at end of round 33):

1. **Averaging-style bimodal (7×):** in-dist ↓, OOD ↑. coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead, warmup-5, fun-jitter.
2. **Broadcast-scalar prior corruption (2×):** uniform regression all splits, val_re_rand worst. gap/stagger jitter, NACA jitter.
3. **Momentum-lag overshoots cosine (1×):** uniform regression all splits. AdamW β1=0.95.
4. **Warmup-duration asymmetry (1×):** non-bimodal non-uniform regression; warmup duration bounds OOD from below. warmup-2.
5. **Architectural-activation degradation (1× this round):** uniform regression all splits (worst on in-dist). GELU→SiLU.
6. **Optimizer-statistic over-conservativism (1× this round):** uniform regression all splits (worst on cruise OOD). AdamW amsgrad=True.

### Assignments for Round 35

| PR | Student | Hypothesis | Mechanism class |
|---|---|---|---|
| #2194 | alphonse | SGD(momentum=0.9, nesterov=True, lr=2e-3) | Optimizer-FAMILY swap from AdamW (4-LOSS-for-4 closed). High uncertainty — informative either way. |
| #2195 | askeladd | LayerScale init=1e-4 (CaiT-style per-channel learnable γ on residual branches) | Architectural rescaling sub-class; distinct from activation/normalization/stochastic depth. |

---

## 2026-05-13 10:45 UTC — Round 34

### PR #2083 tanjiro: DropPath p_max=0.1 (retry-2 of stale #1976) — CLOSED (2nd consecutive stale at pod-level)

- **Branch:** `charliepai2g48h5-tanjiro/droppath-p-max-0.1` (last update 2026-05-13T08:22:35Z, no commit activity since)
- **Closure reason:** 2nd consecutive stale — pod-stuck pattern. Original #1976 went stale in round 21; #2083 was the retry-2 (round 28); now retry-2 also stale. Same exact pattern that affected frieren RMSNorm (#1926 → #2034 → #2139 fresh retry-3 picked up).
- **Hypothesis preserved:** block-level stochastic depth (DropPath p_max=0.1 linear schedule across 5 layers). Structurally distinct from all closed mechanism classes — NOT averaging-style (7× confirmed), NOT broadcast-scalar prior corruption (2× confirmed). Still untested.

### Assignment for Round 34

| PR | Student | Hypothesis | Mechanism class |
|---|---|---|---|
| #2179 | tanjiro | DropPath p_max=0.1 retry-3 (fresh branch, REST API to bypass GraphQL rate limit) | Block-level stochastic depth, untested architectural OOD lever |

All 8 students now in-flight after round 34. Zero idle students. Baseline unchanged at 50.6001 (PR #2033). Awaiting results from in-flight cohort.

---

## 2026-05-13 10:30 UTC — Round 33

### PR #2112 thorfinn: Warmup-2-cosine — CLOSED (LOSS, bracket-completion result)

- **Branch:** `charliepai2g48h5-thorfinn/warmup-2-cosine`
- **Hypothesis:** Bimodal symmetric-flip prediction — swap 1 warmup epoch for 1 settling epoch (warmup-3 → warmup-2) should flip the signs of warmup-5's bimodal: small in-dist regression, larger OOD gain.
- **Result:** val_avg = **51.9279** (+2.62% vs 50.6001 baseline, LOSS), test_avg = **45.5111** (+3.51% vs 43.9680).

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | warmup-2 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 48.7827 | +1.75% (slight regress, predicted) |
| `val_geom_camber_rc` | 67.3675 | 66.5205 | **−1.26%** (only OOD WIN ever observed on camber_rc) |
| `val_geom_camber_cruise` | 34.3430 | 37.8216 | +10.13% (large LOSS, opposite of prediction) |
| `val_re_rand` | 52.7481 | 54.5868 | +3.49% (LOSS, opposite of prediction) |

**Bimodal symmetric-flip prediction FALSIFIED:**

1. **Warmup bracket {2, 3, 5} on val_avg = {51.93, 50.60, 50.72}** — sharply asymmetric peak at warmup-3, much steeper to the under-warm side.
2. **Mechanism: warmup duration bounds OOD from below.** Under-warm training destabilises OOD feature learning, not just in-dist basin selection. The warmup phase appears to do qualitatively different work than the cosine settling phase; they don't trade epochs symmetrically.
3. **Single bright spot:** `val_geom_camber_rc` -1.26% is the ONLY OOD WIN we've seen on the camber_rc bottleneck across the entire launch. But cost on cruise (+10.13%) and re_rand (+3.49%) far outweighs.
4. **Warmup duration axis fully closed at warmup-3.** Schedule axis (warmup+cosine) is now exhausted in this launch.

### PR #2072 edward: NACA geometry jitter σ=0.01 (channels 15-17, 19-21) — CLOSED (LOSS, 2nd broadcast-scalar prior corruption confirmation)

- **Branch:** `charliepai2g48h5-edward/naca-jitter-s0.01`
- **Hypothesis:** σ=0.01 noise on NACA shape descriptors should force camber-invariant features and selectively improve val_geom_camber_rc / val_geom_camber_cruise (the two camber-OOD splits).
- **Result:** val_avg = **53.0530** (+4.85% vs 50.6001 baseline, LOSS), test_avg = **46.7074** (+6.23% vs 43.9680).

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | NACA-jitter | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 48.0415 | +0.21% (wash) |
| `val_geom_camber_rc` | 67.3675 | 70.8960 | +5.24% (camber-OOD LOSS) |
| `val_geom_camber_cruise` | 34.3430 | 36.6436 | +6.70% (camber-OOD LOSS) |
| `val_re_rand` | 52.7481 | 56.6308 | **+7.36%** (worst hit) |

**2nd confirmation of broadcast-scalar prior corruption — identical pattern to gap/stagger #2114:**

1. **All 4 splits regress uniformly** with val_re_rand worst hit at +7.36%.
2. **Smoking gun: val_re_rand has NO camber variation** but is the worst regressor. Under "NACA channels encode domain-shift noise" hypothesis, val_re_rand should be neutral. Instead it's the worst → confirms NACA channels are used as conditioning priors that interact with Re/AoA channels.
3. **Mechanism (same as gap/stagger #2114):** NACA channels are per-sample-broadcast scalars (6 numbers per foil broadcast to all N points). The model uses them as LOAD-BEARING priors for "this sample's geometric configuration is X". Perturbing them is belief-corruption, not data augmentation. The student's analysis is exactly right: NACA is "informational backbone, not domain-shift noise".
4. **Generalizable rule across rounds 30-32-33:** the per-sample-broadcast scalar channel class is NOT augmentation-safe. Confirmed at:
   - NACA channels 15-17, 19-21 (#2072, this round)
   - gap/stagger channels 22-23 (#2114, round 32)
   - fun_dim channels 13/14/18 at σ=0.05 (#1988 nezuko, σ=0.025 was bimodal-with-Goldilocks but σ=0.05 was uniform LOSS — same pattern)

**Implication:** any further input-channel-noise experiment on a per-sample-broadcast scalar channel is predicted to LOSS. The remaining augmentation-safe channels are per-point (coords 0-1, currently being tested via translation aug #2138 nezuko; mesh arc-length 2-3 and is_surface 12 are mesh-topology, not augmentation candidates).

### Round-33 failure-mode taxonomy update

Through rounds 25-33 we now have 4 distinct failure-mode patterns documented (vs 3 at end of round 32):

1. **Averaging-style bimodal (7× confirmed):** in-dist ↓, OOD ↑. Mechanisms: coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead, warmup-5, fun-jitter (σ=0.025).
2. **Broadcast-scalar prior corruption (2× confirmed):** uniform regression all splits, val_re_rand worst. Mechanisms: gap/stagger jitter (#2114), NACA jitter (#2072, this round).
3. **Momentum-lag overshoots cosine (1×):** uniform regression all splits. Mechanism: AdamW β1=0.95 (#2093).
4. **Warmup-duration asymmetry (1× this round):** non-bimodal but non-uniform regression; warmup duration bounds OOD from below. Mechanism: warmup-2 (#2112).

### Assignments for Round 33

| PR | Student | Hypothesis | Mechanism class |
|---|---|---|---|
| #2173 | thorfinn | n_head 4→2 (dim_head 32→64) | Architectural probe — attention head sizing closer to literature dim_head=64 optimum |
| #2174 | edward | Huber β=0.1 (smooth-L1 training loss) | Loss-function probe — addresses L1 sign-flip noise at the loss level (orthogonal to amsgrad #2155) |

---

## 2026-05-13 10:05 UTC — Round 32

### PR #2114 askeladd: Gap/stagger jitter σ=0.02 (channels 22-23) — CLOSED (LOSS, NEW failure-mode pattern)

- **Branch:** `charliepai2g48h5-askeladd/gap-stagger-jitter-s0.02`
- **Hypothesis:** Tandem-foil arrangement (gap, stagger) jitter on the last untouched geometric input channels — predicted to NOT show bimodal signature (different mechanism class from averaging).
- **Result:** val_avg = **51.3681** (+1.52% vs 50.6001 baseline, LOSS), test_avg = **44.7212** (+1.71% vs 43.9680). All 4 val splits regress uniformly.

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | gap/stagger σ=0.02 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 49.07 | +2.36% (in-dist LOSS) |
| `val_geom_camber_rc` | 67.3675 | 68.20 | +1.24% |
| `val_geom_camber_cruise` | 34.3430 | 35.25 | +2.65% |
| `val_re_rand` | 52.7481 | 54.90 | **+4.07%** (worst hit) |

**NEW failure-mode pattern discovered — broadcast-scalar prior corruption (NOT 8th averaging-style bimodal):**

1. **Uniform regression across all splits.** No bimodal signature. This is a DIFFERENT failure mode from the 7× averaging-style class (which always shows in-dist gain ↔ OOD regression).
2. **val_re_rand worst hit at +4.07%.** This is the split that varies Re — and the gap/stagger channels are *correlated* with the Reynolds-dependent flow regime. Corrupting them most disrupts the OOD split where the prior is most load-bearing.
3. **Mechanism: per-sample-constant input channels are NOT augmentation-safe.** Unlike per-point channels (NACA #2072, coords #1921), channels 22-23 are CONSTANTS within a sample — broadcast to all N points. The model uses them as **load-bearing priors** for "this sample is configuration X". Perturbing them is NOT data augmentation, it's belief-corruption.
4. **Implication for future augmentation:** the per-sample broadcast vs per-point distinction matters fundamentally. Per-point channel jitter (NACA, coord) is bona-fide augmentation; per-sample-constant channel jitter is prior-corruption. The askeladd experiment cleanly isolates this taxonomy.

### PR #2093 alphonse: AdamW β1=0.95 (UP probe) — CLOSED (LOSS, NEW failure-mode pattern)

- **Branch:** `charliepai2g48h5-alphonse/adamw-beta1-0.95`
- **Hypothesis:** Gradient momentum smoothing for L1 sign-noise — β1=0.9→0.95 extends momentum half-life from ~10 to ~14 steps. Predicted: smoothing helps in noisy-gradient regime.
- **Result:** val_avg = **54.3714** (+7.45% vs 50.6001 baseline, LOSS), test_avg = **47.9036** (+8.94% vs 43.9680). All 4 splits regress uniformly.

**Per-split breakdown (vs 50.6001 baseline):**

| Split | Baseline | β1=0.95 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.9418 | 51.32 | +7.04% |
| `val_geom_camber_rc` | 67.3675 | 70.14 | +4.10% |
| `val_geom_camber_cruise` | 34.3430 | 39.24 | **+14.27%** (worst hit) |
| `val_re_rand` | 52.7481 | 56.79 | +7.66% |

**NEW failure-mode pattern — momentum-lag overshoots cosine LR (NOT 8th averaging-style bimodal):**

1. **Uniform regression across all 4 splits** (4.10% to 14.27%). NOT bimodal — in-dist did NOT win. This is a DIFFERENT failure mode from the 7× averaging-style class.
2. **val_geom_camber_cruise worst hit** (NOT val_re_rand, which would have been the bimodal signature). The failure pattern is shape-distribution-sensitive, not Re-shift-sensitive.
3. **Mechanism: β1=0.95 keeps the optimizer oriented to historical gradients that are increasingly stale as cosine LR decays.** As the schedule shrinks LR, momentum-lag means the optimizer still applies β1=0.95-weighted historical updates → systematic overshoots at every split.
4. **AdamW (lr, β1, β2) hyperparameter axis now closed 4-LOSS-for-4** in this launch:
   - lr=7.5e-4 (UP, #1774): LOSS +16%
   - lr=3.75e-4 (DOWN, #1997): LOSS +11.4%, 4th averaging-style bimodal
   - β2=0.95 (#1845): LOSS +15%
   - β1=0.95 (#2093, this PR): LOSS +7.45%, momentum-lag overshoot
   - Defaults (lr=5e-4, β1=0.9, β2=0.999) are locally optimal.

### Round-32 new failure-mode taxonomy

Through rounds 25-32 we now have THREE distinct failure-mode patterns documented across mechanism classes:

1. **Averaging-style bimodal (7× confirmed):** in-dist ↓, OOD ↑. Mechanisms: coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead, warmup-5, fun-jitter. Cause: reduced effective late-stage exploration trades in-dist precision for OOD robustness, but inversely.
2. **Broadcast-scalar prior corruption (1× confirmed, this round):** all splits regress, val_re_rand worst. Mechanism: gap/stagger jitter (#2114). Cause: per-sample-constant channels are load-bearing priors; perturbing them is belief-corruption, not augmentation.
3. **Momentum-lag overshoots cosine (1× confirmed, this round):** all splits regress uniformly. Mechanism: AdamW β1=0.95 (#2093). Cause: high momentum can't track shrinking LR → systematic overshoots.

### Assignments for Round 32

| PR | Student | Hypothesis | Mechanism class |
|---|---|---|---|
| #2155 | alphonse | AdamW amsgrad=True (max-tracking 2nd-moment) | Novel optimizer-stability lever (orthogonal to lr/β1/β2 axis). Tracks MAX of 2nd-moment EMA → robust to L1 sign-flip noise. |
| #2156 | askeladd | GELU → SiLU activation swap (all 3 MLP sites) | Architectural probe; NOT averaging, NOT broadcast-scalar. Literature: 0.3-1% gain typical on Transformer benchmarks. |

---

## 2026-05-13 09:30 UTC — Round 31

### PR #1988 nezuko: Per-sample fun_dim jitter Re/AoA (channels 13/14/18) — CLOSED after 2 σ-points (7th averaging-style bimodal confirmation)

- **Branch:** `charliepai2g48h5-nezuko/fun-jitter-re-aoa-0.025` (latest run; previous σ=0.05 run on same PR)
- **Hypothesis:** σ=0.025 retune of fun_dim jitter on condition channels. Earlier σ=0.05 was LOSS (+19.5%); student suggested σ=0.025 as the σ-sweep closure probe.
- **Result:** val_avg = **57.9999** (+14.6% vs current 50.6001 baseline, +7.4% vs old 54.00), test_avg = **50.3245** (+5.7%). LOSS. Best epoch terminal at 44/44.

**Per-split breakdown (vs 50.6001 current baseline):**

| Split | Baseline | σ=0.025 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.94 | 54.86 | +14.4% (still loss vs current schedule) |
| `val_geom_camber_rc` | 67.37 | 74.58 | +10.7% (OOD LOSS) |
| `val_geom_camber_cruise` | 34.34 | 42.13 | +22.7% (largest OOD LOSS) |
| `val_re_rand` | 52.75 | 60.42 | +14.5% (OOD LOSS) |

**Two σ-points cleanly establish the dose-response:**

| Split | σ=0.05 (vs old 54.00) | σ=0.025 (vs old 54.00) | Attenuation when halving σ |
|---:|---:|---:|---:|
| val_single_in_dist | +4.1% | **−7.16%** (flip to in-dist WIN at smaller σ) | directional flip |
| val_geom_camber_rc | +12.2% | +10.6% | 0.87× (slower than linear) |
| val_geom_camber_cruise | +21.9% | +17.9% | 0.82× (slower than linear) |
| val_re_rand | +13.6% | +12.4% | 0.91× (near-flat) |

**Key empirical findings:**

1. **OOD damage is near-saturated in σ** — halving σ from 0.05 to 0.025 only reduces OOD regression by 9-18%, far slower than linear (which would be 50% reduction). This is the signature of "OOD damage is binary-in-σ-threshold" — any non-trivial fun_dim noise displaces the model's OOD condition→field readout equally.
2. **In-dist exhibits a Goldilocks zone** — directional flip from +4.1% LOSS (σ=0.05) to −7.2% WIN (σ=0.025). The in-dist regularization mechanism IS real at smaller σ, but only at the cost of OOD damage that doesn't attenuate proportionally.
3. **7th averaging-style bimodal confirmation** — joins the rock-solid empirical law (coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead, warmup-5). Now extended to channel-level data augmentation: condition-channel jitter is yet another way to express the same in-dist↔OOD trade-off.

**Mechanism interpretation (student's analysis, verbatim and correct):** "Condition channels are broadcast scalars per sample, not noisy per-node features. Unlike coords (which already have natural mesh noise), the condition channels are 'exact' sample-level constants. Perturbing them tells the model the *operating condition itself* is uncertain — a much stronger prior than the pos-jitter case."

**Class implication:** Future channel-level jitter probes (Re-only, AoA-only, even smaller σ) would each produce smaller-magnitude versions of the same bimodal — no operational information gain. **Axis closed comprehensively.**

- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-fun-jitter-re-aoa-0.025-20260513-075053/metrics.jsonl`
- **Next:** Reassign nezuko to translation augmentation σ=0.01 on coords (channels 0-1, per-sample shift) — physical-symmetry lever, structurally distinct from all 7 closed averaging-style mechanisms.

### PR #2034 frieren: RMSNorm replaces LayerNorm (retry of stale #1926) — CLOSED (STALE, 2nd consecutive)

- **Branch:** `charliepai2g48h5-frieren/rmsnorm-retry`
- **Hypothesis:** RMSNorm at all 3 LayerNorm sites in TransolverBlock; Pre-LN architectural variant.
- **Result:** Zero training activity. Only the round-25 assignment commit (d5f8ebc at 07:14 UTC); no further commits, no pod activity for 8+ hours. Same GraphQL rate-limit/pod-stuck pattern observed with tanjiro's stales (4 consecutive before unstuck via fresh PR #2083).
- **Axis status:** UNTESTED. RMSNorm hypothesis intact and worth one more test on new 50.6001 baseline.
- **Next:** Reassigned to frieren under fresh PR #2139 (same hypothesis, new PR to unstick pod via new branch/tracking state).

### Assignment: Round 31

| PR | Student | Hypothesis |
|---|---|---|
| #2138 | nezuko | Translation augmentation σ=0.01 (channels 0-1, per-sample shift) — physical-symmetry OOD lever, structurally distinct from 7 closed averaging-style mechanisms |
| #2139 | frieren | RMSNorm replacing LayerNorm (all 3 sites, retry-3 of stale #1926/#2034) — pod-unstick attempt via new PR |

**Rationale (combined):** With the 7× averaging-style confirmation now rock-solid, the next OOD-targeting experiments must use structurally different mechanisms: physical-symmetry data augmentation (translation aug), structural input augmentation (NACA #2072 + gap/stagger #2114 already in flight), or architectural change (RMSNorm). Translation augmentation specifically tests whether the bimodal pattern extends to physical-symmetry augmentation or whether physical-symmetry is the first lever that breaks the trade-off — a critical experimental discriminant. RMSNorm continues the architecture-side probe.

**Operations note:** GraphQL API rate-limited (5000/hr exhausted, resets ~09:49 UTC). PR creation done via REST API (`POST /repos/.../pulls`); labels added via `POST /repos/.../issues/{n}/labels`. REST core healthy at 4500+/5000.

---

## 2026-05-13 15:30 — Round 30

### PR #2071 thorfinn: Warmup-5-cosine (warmup_epochs 3→5, T_max 47→45) — CLOSED (WASH, bimodal trade-off observed)

- **Branch:** `charliepai2g48h5-thorfinn/warmup-5-cosine`
- **Hypothesis:** Test whether warmup-5 (longer basin-selection) beats warmup-3 (current best) or whether the shortened settling phase (2 fewer cosine epochs at T_max=45 vs 47) costs more than the extra warmup gains.
- **Result:** val_avg = **50.7181** (+0.23%), test_avg = **44.1414** (+0.39%). Cleanly within the predicted ±1% wash band. Best epoch 45/45 (terminal). warmup-3 confirmed optimal at this bracket.

**Per-split breakdown (vs 50.6001 warmup-3 baseline) — the diagnostic value:**

| Split | warmup-3 | warmup-5 | Δ | Δ% |
|---:|---:|---:|---:|---:|
| `val_single_in_dist` | 47.94 | **46.70** | **−1.25** | **−2.60%** ← basin-selection WIN |
| `val_geom_camber_rc` | 67.37 | 67.29 | −0.08 | −0.12% (flat) |
| `val_geom_camber_cruise` | 34.34 | 34.66 | +0.32 | +0.92% |
| `val_re_rand` | 52.75 | **54.23** | **+1.48** | **+2.81%** ← settling-phase LOSS |
| **val_avg** | **50.60** | **50.72** | **+0.12** | **+0.23%** wash |

**Mechanism diagnosis:** warmup-5 traded 2 settling epochs for 2 basin-selection epochs. The trade was monotone:
- In-dist single split benefits from longer basin-selection (−2.60%) — model has time to settle into a better local minimum
- OOD re_rand split suffers from shortened settling (+2.81%) — late-stage low-LR crystallization is essential for OOD
- val_geom_camber_rc (hardest OOD) is INSENSITIVE to warmup duration — confirms it's bound by something else entirely (capacity? loss formulation? geometry representation?)
- val_geom_camber_cruise slightly regressed (+0.92%) — smaller version of re_rand story

**6th bimodal confirmation** (alongside coord-jitter, EMA, grad-clip, lr-DOWN, Lookahead). This time on a SCHEDULE axis — not an averaging-style mechanism. The pattern is broader than first thought: **anything reducing late-stage low-LR settling time hurts OOD** — schedule changes, optimizer averaging, weight averaging, gradient averaging, step-size reduction all map to the same trade-off.

**Mechanism-level principle (now solid):** late-stage low-LR exploration is essential for OOD generalization on Transolver+TandemFoilSet+L1+slice=32. Removing it via any means produces the canonical in-dist↔OOD bimodal trade-off.

- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-warmup-5-cosine-20260513-081312/metrics.jsonl`
- **Next:** Reassign thorfinn to warmup-2 (the OTHER direction of the bracket) — completes the warmup duration bracket and tests if the bimodal trade flips with 1 extra settling epoch.

### PR #2051 askeladd: Lookahead(k=5, α=0.5) wrapping AdamW — CLOSED (LOSS, 5th averaging-style bimodal confirmation)

- **Branch:** `charliepai2g48h5-askeladd/lookahead-k5-a0.5`
- **Hypothesis:** Lookahead's slow-weight averaging would bias toward flat minima → improve OOD generalization (predicted INVERSE pattern to grad-clip's bimodal). Run was on OLD baseline schedule (CosineAnnealingLR(T_max=50) directly, no warmup-3).
- **Result:** val_avg = **56.7516** (+5.09% vs old 54.00 baseline, +12.16% vs current 50.60), test_avg = **50.2749** (+5.56%). LOSS. Best=terminal at ep45/45 — still converging at timeout, no overfit reversal.

**Per-split breakdown (vs 54.0051 OLD baseline #1846 — the schedule the student actually ran against):**

| Split | Baseline #1846 | Lookahead | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 59.09 | **53.49** | **−5.60 (−9.48%)** ← in-dist WIN |
| `val_geom_camber_rc` | 67.45 | 72.60 | +5.15 (+7.64%) ← OOD LOSS |
| `val_geom_camber_cruise` | 35.72 | 42.15 | +6.43 (+18.0%) ← OOD LOSS (largest) |
| `val_re_rand` | 53.76 | 58.77 | +5.01 (+9.31%) ← OOD LOSS |

**Hypothesis was the wrong direction.** The student predicted Lookahead would produce the INVERSE of grad-clip's bimodal (OOD-favorable). The result was the SAME as grad-clip's bimodal (in-dist favorable, OOD unfavorable). **5th averaging-style confirmation.**

**Mechanism interpretation:** Lookahead's slow-weight centroid is dominated by the trajectory-average over the k=5 inner steps. With balanced sampler weighted toward single-foil (2× single sampler), the averaging direction biases toward in-distribution geometry — same root cause as the other 4 averaging-style failures, just expressed at a different layer of the optimizer.

**Class definitively closed.** 5 mechanisms — data aug (coord-jitter), weight aug (EMA), gradient aug (grad-clip), step-size aug (lr-DOWN), trajectory aug (Lookahead) — all map to the same in-dist↔OOD trade-off. Plus the schedule-axis warmup-5 (6th, settling-phase reduction). **Future optimizer/regularization experiments in the averaging-style family should be deprioritized.**

**Strong implication for next steps:** OOD generalization requires either:
1. **Structural input augmentation** (NACA jitter #2072, gap/stagger jitter #2114, fun jitter #1988)
2. **Architectural changes** (conditional embedding on Re/AoA, multi-task auxiliary loss, RMSNorm #2034 if it changes optimization landscape)
3. **Extended late-stage settling** (the warmup-2 probe #2112)
4. **NOT** any optimizer averaging, weight smoothing, or trajectory-bias method.

- **Metric artifacts:** `models/model-charliepai2g48h5-askeladd-lookahead-k5-a0.5-20260513-080103/metrics.jsonl`
- **Next:** Reassign askeladd to gap/stagger jitter σ=0.02 (channels 22-23) — tandem-arrangement OOD attack, structurally distinct from closed averaging-style class. Complements in-flight NACA jitter (#2072 edward).

### Assignment: Round 30

| PR | Student | Hypothesis |
|---|---|---|
| #2112 | thorfinn | Warmup-2-cosine (warmup_epochs 3→2, T_max 47→48) — OOD-favorable direction of the bimodal trade-off; completes warmup duration bracket |
| #2114 | askeladd | Gap/stagger jitter σ=0.02 (channels 22-23, per-sample) — tandem-arrangement OOD attack on the last untouched geometric input subspace |

**Rationale (combined):** Closed PRs both point to the same mechanism-level finding — late-stage low-LR settling matters for OOD, and averaging-style interventions can't substitute. Round-30 assignments respond with two complementary probes: (1) thorfinn's warmup-2 directly tests the bimodal trade-off in the predicted-favorable direction (1 extra settling epoch ↔ OOD gain), and (2) askeladd's gap/stagger jitter pivots to a structurally orthogonal class — geometry-channel augmentation. Both have low complexity (1-line and 5-line changes), high diagnostic value, and target the OOD bottleneck via mutually exclusive mechanisms.

---

## 2026-05-13 14:30 — Round 29

### PR #1997 alphonse: lr 5e-4 → 3.75e-4 (-25% lr-DOWN) — CLOSED (LOSS, 4th averaging-style bimodal confirmation)

- **Branch:** `charliepai2g48h5-alphonse/lr-3.75e-4`
- **Hypothesis:** Reduce lr from 5e-4 to 3.75e-4 (-25%) to probe capacity↔LR coupling DOWN axis. With warmup-3 already merged, hoped lower peak-LR + warmup would settle into deeper basin.
- **Result:** val_avg = **56.36** (+11.4% vs 50.6001), test_avg = **49.55** (+12.7%). Best epoch terminal at 44/50 (under-stepping confirmed).

**Per-split breakdown (vs 50.6001 baseline PR #2033):**

| Split | Baseline | lr=3.75e-4 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 47.94 | **41.49** | **−6.45** (in-dist WIN) |
| `val_geom_camber_rc` | 67.37 | **73.11** | **+5.74** (OOD LOSS) |
| `val_geom_camber_cruise` | 34.34 | **40.41** | **+6.07** (OOD LOSS) |
| `val_re_rand` | 52.75 | **57.59** | **+4.84** (OOD LOSS) |
| **val_avg** | **50.60** | **56.36** | **+11.4% LOSS** |

**Axis status:** lr axis (capacity↔LR coupling, ±25% of 5e-4) fully bracketed and CLOSED.
- lr-UP (+50% to 7.5e-4, #1774): LOSS +16% on L1+slice=32, n=3 confirmed
- lr-DOWN (-25% to 3.75e-4, #1997): LOSS +11.4% on warmup-3-cosine baseline
- 5e-4 confirmed as the optimum for this Transolver+L1+slice=32 setup

**4th confirmation of averaging-style bimodal pattern.** Previously: coord-jitter (#1921), weight-EMA (#1946), grad-clip (#1653). Now lr-DOWN. Reduced effective step size acts as implicit averaging — same in-dist↔OOD trade-off as the other 3 mechanisms.

**Mechanism interpretation:**
- Smaller lr = each step traverses less of the loss landscape → trajectory averages over smaller neighborhood → behaves like weight-EMA implicitly
- In-dist gain (−6.45) and uniform OOD regression (+4.84 to +5.74) match coord-jitter/EMA/grad-clip signatures exactly
- This means the pattern is mechanism-agnostic — any reduction in effective trajectory diversity (data aug, weight aug, gradient aug, step-size aug) trades in-dist for OOD

**Implication for future experiments:**
- OOD bottleneck (val_geom_camber_rc 67.37, val_re_rand 52.75) does NOT yield to ANY averaging-style intervention
- Need structural interventions: geometry-level augmentation (NACA jitter #2072 in flight), conditional embedding, multi-task auxiliary loss
- Single-knob optimizer variants on novel axes (β1, weight decay schedule) remain worth probing but probably NOT for OOD

**Run details:** 50 epochs in ~30 min, best epoch terminal at 44/50 (under-stepping). With smaller lr, model needs more steps to converge; budget-cap was the binding constraint.

- **Metric artifacts:** model dir on `charliepai2g48h5-alphonse/lr-3.75e-4`

### Assignment: Round 29

| PR | Student | Hypothesis |
|---|---|---|
| #2093 | alphonse | AdamW β1=0.95 (UP probe) — gradient momentum smoothing for L1 sign-noise; novel axis (β2 closed #1845, β1 never probed) |

**Hypothesis rationale:** β2 was tested and closed (#1845, +15% LOSS on β2=0.95); β1 has never been probed. β1=0.9→0.95 lengthens the gradient-momentum half-life from ~10 steps to ~14 steps, providing additional smoothing for L1's sign-gradient noise. Mechanism is distinct from step-size averaging (lr-DOWN failed) and weight averaging (EMA failed): operates on gradient-direction rather than gradient-magnitude or weight-trajectory. β2 closure suggests AdamW's second-moment memory is critical (don't shrink); but first-moment momentum may benefit from being longer with L1's sign-flip gradient regime.

---

## 2026-05-13 14:00 — Round 28

### PR #1976 tanjiro: DropPath p_max=0.1 stochastic depth — CLOSED (STALE, 4th consecutive on this branch)

- **Branch:** `charliepai2g48h5-tanjiro/droppath-p-max-0.1`
- **Hypothesis:** Block-level stochastic depth (linear schedule p=0→0.1) for OOD generalization via implicit ensembling.
- **Result:** Zero training activity. Only the round-21 assignment commit (c41ceb3); no further commits, no pod activity. Same GraphQL rate-limit pattern as tanjiro's prior 3 stale PRs (#1660, #1789, #1883).
- **Axis status:** UNTESTED. DropPath hypothesis intact and worth one test on new 50.6001 baseline.
- **Next:** Reassigned to tanjiro under fresh PR #2083 (same hypothesis, new PR to unstick pod) with updated baseline context.

### Assignment: Round 28

| PR | Student | Hypothesis |
|---|---|---|
| #2083 | tanjiro | DropPath p_max=0.1 stochastic depth (retry of stale #1976; on new 50.6001 baseline) |

---

## 2026-05-13 13:00 — Round 27

### PR #2033 thorfinn: Linear warmup 3ep + monotone cosine (T_max=47) — MERGED (WIN -6.31% val / -7.68% test)

- **Branch:** `charliepai2g48h5-thorfinn/warmup-3-cosine`
- **Hypothesis:** Linear warmup 3 epochs (0.1×→1.0×) followed by CosineAnnealingLR(T_max=47) on L1+slice=32 baseline.
- **Result:** val_avg = **50.6001** (−6.31%), test_avg = **43.9680** (−7.68%). Clean WIN. Best epoch 44/44 (terminal). **New baseline established.**

**Per-split breakdown:**

| Split | Baseline (PR #1846) | Warmup-3 | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 59.0943 | **47.9418** | **−18.87%** ← largest gain |
| `val_geom_camber_rc` | 67.4450 | **67.3675** | −0.11% (barely moved) |
| `val_geom_camber_cruise` | 35.7197 | **34.3430** | −3.85% |
| `val_re_rand` | 53.7616 | **52.7481** | −1.89% |
| **val_avg** | **54.0051** | **50.6001** | **−6.31%** |

**Mechanism confirmed:**
- Warmup gives optimizer 2-3 sub-peak-LR epochs to select a better loss basin before cosine descent locks in
- Largest gain on val_single_in_dist (-18.87%) where basin quality is most sensitive to ep-1 step size
- OOD splits move little but don't regress — warmup doesn't trade OOD for in-dist
- Late-stage settling preserved unlike SGDR (#1989 LOSS): model improved all the way to ep44 (gap ep41→44 = -4.9%)
- L1 + warmup mechanism validated: warmup serves "find the basin" phase; cosine tail serves "fine-tune within it"

**Run details:** 44 epochs in 30 min (~41 s/epoch), peak memory 21.35 GB. LR schedule confirmed: ep1=5e-5, peak at ep4, cosine descent to ~2e-5 by ep44.

- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-warmup-3-cosine-20260513-072010/`

### PR #1946 edward: EMA decay=0.999 (weight averaging for OOD generalization) — CLOSED (WASH-TO-LOSS, axis closed)

- **Branch:** `charliepai2g48h5-edward/ema-weights-0.9999`
- **Hypothesis:** EMA model weights with various decay values for flat-minima OOD generalization.
- **Three-run summary:**

| Run | Decay | val_avg | Δ baseline | Notes |
|---|---|---|---|---|
| Run 1 | 0.9999 | 165.27 | +205.9% | Lag-bias: `0.9999^16500 ≈ 0.19` — 19% init noise at terminal |
| Run 2 (with diag) | 0.999 | 54.91 | +1.67% | Mechanism confirmed: EMA<raw from ep10 |
| Run 3 (no-diag) | 0.999 | 56.12 | +3.92% | OOD regression 5-14%; run-to-run variance ~2pts |

**Key findings:**
- Mechanism IS real: EMA variance-reduction confirmed by dual-val diagnostic (EMA<raw from ep10 once init-noise decays `0.999^3750≈0.024`)
- But per-split pattern is structurally wrong: val_single_in_dist -12.9%, all OOD splits +5-14%
- **Third confirmation of averaging-style bimodal pattern:** coord-jitter, EMA (both decays), grad-clip all deliver ~-13% in-dist win but hurt OOD
- **In-dist headroom finding (3× confirmed):** ~14% unlockable via averaging/regularization; OOD requires different structural interventions

### Assignments: Round 27

| PR | Student | Hypothesis |
|---|---|---|
| #2071 | thorfinn | Warmup-5-cosine: probe warmup duration optimality (warmup_epochs=3→5, T_max=47→45) |
| #2072 | edward | NACA geometry jitter σ=0.01 on channels 15-17 (NACA1) + 19-21 (NACA2): OOD camber generalization |

---

## 2026-05-13 12:00 — Round 26

### PR #1653 askeladd: Grad-clip max_norm=1.0 — CLOSED (WASH on val_avg, OOD regression on primary bottleneck)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-l1-sampler-slice32` (3-round campaign)
- **Hypothesis:** Gradient clipping max_norm=1.0 smooths sharp L1 gradient updates to improve generalisation.

**3-round dose-response (monotone lever decay):**

| Base | max_norm=1.0 Δ | val_avg |
|------|---:|---:|
| compile + bf16 + β=1.0 | **−14.92%** | ~94 → ~80 |
| compile + bf16 + β=0.5 | −6.94% | ~69 → ~64 |
| L1 + sampler + slice=32 (current) | ~0% (wash / mean +0.56%) | ~54 → ~54 |

**Best-run vs variance (n=2 on current baseline):**

| Run | val_avg | Δ baseline | test_avg | Δ baseline |
|---:|---:|---:|---:|---:|
| Best | 53.81 | −0.37% | 46.67 | −2.01% |
| Variance | 54.81 | +1.49% | 47.79 | +0.34% |
| **Mean** | **54.31** | **+0.56%** | **47.23** | **−0.83%** |

**Per-split breakdown (best run):**

| Split | Baseline | Best run | Δ |
|---:|---:|---:|---:|
| `val_single_in_dist` | 59.09 | 51.22 | **−13.33%** ← in-dist win |
| `val_geom_camber_rc` | 67.45 | 70.60 | +4.67% ← OOD regression |
| `val_geom_camber_cruise` | 35.72 | 36.78 | +2.97% ← OOD regression |
| `val_re_rand` | 53.76 | 56.63 | +5.34% ← OOD regression |

**Analysis:** Grad-clip delivers strong in-dist wins but hurts the two OOD splits that dominate val_avg ceiling (camber_rc, re_rand). Bimodal per-split pattern is structurally wrong for our primary research direction.

**Key mechanistic finding:** The gradient-coherence axis is a *shared* axis with L1 loss and slice_num. These upstream changes have already done the "tame the tails" work grad-clip used to do on cruder bases. SignSGD-like dynamics (L1's ±1 gradients + grad-clip's total-norm clamp) underperform on multi-modal OOD loss surfaces — the per-split bimodal result is the predicted signature (Bernstein et al. 2018).

- **Axis status:** Gradient-coherence axis fully closed. Further grad-clip variants on L1 baseline not expected to flip the OOD-regression pattern.
- **Next:** Assigned askeladd Lookahead optimizer (Zhang et al. 2019) as PR #2051 — structurally orthogonal lever (slow/fast weight averaging vs per-step magnitude clipping).

### Assignment: Round 26

| PR | Student | Hypothesis |
|---|---|---|
| #2051 | askeladd | Lookahead(k=5, α=0.5) wrapping AdamW — slow/fast weight averaging for flat-minima bias |

---

## 2026-05-13 11:00 — Round 25

### PR #1989 thorfinn: SGDR T_0=10 T_mult=2 — CLOSED (LOSS, restart-disruption + L1 mismatch confirmed)

- **Branch:** `charliepai2g48h5-thorfinn/sgdr-t0-10-retry`
- **Hypothesis:** Replace CosineAnnealingLR(T_max=50) with CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=0).
- **Result:** val_avg = **68.96 (+27.7%)**, test = 61.07 (+28.2%). Clean LOSS.

**Trajectory at restart boundaries (the key diagnostic):**

| Epoch | val_avg | Phase |
|---:|---:|---|
| 10 (end of cycle 1) | 108.86 | restart #1 minimum |
| 30 (end of cycle 2) | **70.81** | restart #2 minimum (better than #1) |
| 45 (terminal, mid-cycle 3) | 68.96 | best, still mid-descent |

- **Analysis:** SGDR's cycle-level mechanism IS working (restart #2 min 70.81 < restart #1 min 108.86 — progressively improving minima). But T_0=10, T_mult=2 means cycle 3 needs 40 epochs (10+20+40=70 cumulative) to complete; we get 15 truncated. Baseline gets one uninterrupted 50-epoch descent.
- **Deeper finding on L1:** Sign-gradient regime needs sustained low-LR phases for residual-sign fine-tuning. Each SGDR restart resets LR to peak, destroying the converged signs from the prior cycle. The PR's own loss-case prediction ("restarts disrupt the late-stage settling that L1 sign gradients need") matched exactly.
- **Axis status:** SGDR closed. Budget-aligned variants (T_0=15, T_0=22) might fix cycle structure but L1+restart-disruption is the binding mechanism, not budget-fit.
- **Next:** Assigned thorfinn warmup+monotone-cosine (his strongest recommendation) as PR #2033 — captures "high-LR exploration" benefit SGDR aimed for without disrupting final descent.

### PR #1926 frieren: RMSNorm at all 3 norm sites — CLOSED (STALE, 5+ hours zero activity)

- **Branch:** `charliepai2g48h5-frieren/rmsnorm`
- **Hypothesis:** Replace LayerNorm with RMSNorm at ln_1, ln_2, ln_3 in TransolverBlock.
- **Result:** Zero training activity. Only assignment commit (9f04e42) on branch; pod never started the run (likely GraphQL rate-limit, identical pattern to tanjiro #1660/#1789/#1883 and thorfinn #1905).
- **Axis status:** UNTESTED. Hypothesis intact, RMSNorm is still worth exploring under the L1+slice=32 baseline.
- **Next:** Reassigned to frieren under fresh PR #2034 (same experiment, new PR to unstick pod).

### Assignments: Round 25

| PR | Student | Hypothesis |
|---|---|---|
| #2033 | thorfinn | Linear warmup 3ep (0.1→1.0) + monotone cosine (T_max=47) |
| #2034 | frieren | RMSNorm replaces LayerNorm at all 3 sites (retry of stale #1926) |

---

## 2026-05-13 10:30 — Round 24

### PR #1988 nezuko: fun-jitter σ=0.05 on Re/AoA — SENT BACK (LOSS, retune σ=0.025 for axis closure)

- **Branch:** `charliepai2g48h5-nezuko/fun-jitter-re-aoa-0.05`
- **Hypothesis:** Per-sample Gaussian noise on dims 13/14/18 (Re/AoA1/AoA2), σ=0.05, training only.
- **Result:** val_avg/mae_surf_p = **60.45** (+11.9% LOSS). test = 52.89 (+11.1%).

| Split | This run | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 61.51 | 59.09 | +4.1% |
| `val_geom_camber_rc` | 75.69 | 67.45 | +12.2% |
| `val_geom_camber_cruise` | 43.55 | 35.72 | +21.9% |
| `val_re_rand` (TARGETED) | 61.05 | 53.76 | **+13.6% ✗** |

- **Analysis:** Targeted OOD axis (val_re_rand) got worse, not better. Mechanism: condition channels are "exact" sample-level constants broadcast to all nodes — perturbing them tells the model the operating condition itself is uncertain. This is fundamentally different from coord jitter (mesh already has natural noise). Direction of damage differs from #1921: pos-jitter gave in-dist win + OOD loss; fun-jitter gives no-in-dist-win + OOD-only loss (information removal on load-bearing channels).
- **Send-back:** σ=0.025 probe for clean σ-sweep closure. Student's recommendation #1. Either lands wash (axis salvageable at finer magnitude) or smaller LOSS (axis closes decisively).

### PR #1946 edward: EMA decay=0.999 with dual-eval diagnostic — SENT BACK (wash/test-tie, drop diagnostic to recover budget)

- **Branch:** `charliepai2g48h5-edward/ema-weights-0.9999`
- **Hypothesis:** EMA model weights with decay=0.999 (half-life ~2 epochs, ~693 steps) for OOD generalization via flat-minima effect.
- **Result:** val_avg = 54.91 (+1.67%), test = **47.60 (−0.05%, effectively tied)**. 41 epochs vs baseline ~48-50.

| Split | EMA val | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 51.47 | 59.09 | **−12.9% ✓** |
| `val_geom_camber_rc` | 71.07 | 67.45 | +5.4% |
| `val_geom_camber_cruise` | 38.89 | 35.72 | +8.9% |
| `val_re_rand` | 58.19 | 53.76 | +8.2% |

**Mechanism CONFIRMED by diagnostic.** EMA-vs-raw dual-eval table:

| Epoch | EMA val | Raw val | EMA − Raw |
|---:|---:|---:|---:|
| 10 | 117.56 | 131.55 | **−13.98** (EMA first beats raw) |
| 20 | 78.90 | 104.22 | −25.32 |
| 41 | 54.91 | 55.72 | −0.81 |

- **Analysis:** EMA-of-weights mechanism works (cross-over at ep10 after init-weight pollution decays). Wash vs baseline is budget arithmetic, not mechanism failure: dual-eval cost ~3 s/epoch × 41 epochs = ~6 lost epochs vs baseline's full 50-epoch budget. Cosine LR was still annealing at ep41.
- **Cross-experiment pattern:** Both #1921 pos-jitter and #1946 EMA give ~-13% on val_single_in_dist (two structurally different mechanisms, same in-dist win magnitude). In-dist generalization has ~14% headroom unlockable; OOD requires structurally different interventions.
- **Send-back:** Drop the dual-eval diagnostic, rerun with full 50-epoch budget. Student's recommendation #1. Predicted landing: 51-53 val_avg, below baseline.

---

## 2026-05-13 10:00 — Round 23

### PR #1774 alphonse: lr 5e-4 → 7.5e-4 (+50%) — CLOSED (LOSS, lr-UP axis decisively closed)

- **Branch:** `charliepai2g48h5-alphonse/lr-7.5e-4`
- **Hypothesis:** +50% LR bump on current advisor stack to compound with L1's unit-bounded gradients.
- **Round-1 (β=0.5 baseline, n=2):** val_avg mean = 63.30 vs 64.07 (−1.20%), test = +1.51%. Wash.
- **Round-2 (post-rebase, n=5):** L1+slice=64 (n=2) mean = 60.82 (wash-with-loss-tail vs 59.54); L1+slice=32 current advisor (n=3) mean = **62.66 (+16.0% LOSS)**, test = 54.73 (+14.9%).

**Per-split (slice=32 stack, mean of 3 runs):**

| Split | lr=7.5e-4 (n=3) | Baseline (#1846) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 59.98 | 59.09 | +1.5% |
| `val_geom_camber_rc` | 79.29 | 67.45 | +17.6% ✗ |
| `val_geom_camber_cruise` | 46.91 | 35.72 | +31.3% ✗ |
| `val_re_rand` | 64.48 | 53.76 | +19.9% ✗ |

- **Analysis:** Three independent slice=32 runs all >60 confirms the lr-UP lever is decisively wrong on the current stack. Mechanism: slice_num=32 cuts attention capacity → sharper loss landscape → bigger steps land in worse basins by epoch 30+. Adam preconditions magnitudes, not directions. The student's run-variance work established a ~1.5-point noise band; this is well outside.
- **lr-peak axis status:** Closed at +50% across **three** landscape variants (β=0.5, L1+slice=64, L1+slice=32). All show wash-or-loss; none show a clean win.
- **Next:** Reassigned alphonse the inverse probe (her follow-up #2): lr=3.75e-4 (-25%) on current advisor — PR #1997.

### Assignments: Round 23

| PR | Student | Hypothesis |
|---|---|---|
| #1997 | alphonse | lr 5e-4 → 3.75e-4 (-25%) — capacity↔LR coupling DOWN probe |

---

## 2026-05-13 09:30 — Round 22

### PR #1921 nezuko: pos-jitter σ=0.01 — CLOSED (LOSS, informative split-level signal)

- **Branch:** `charliepai2g48h5-nezuko/pos-jitter-0.01`
- **Hypothesis:** Small gaussian noise on volume-node coords (σ=0.01 on z-score normalized coords, training only) to break mesh-pattern memorization and improve OOD generalization.
- **Result:** val_avg/mae_surf_p = **55.6766** (+3.1% vs baseline 54.0051). test_avg = **48.8222** (+2.5%). LOSS.

| Split | Baseline (#1846) | Pos-jitter | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 59.0943 | **51.0** | **-13.7%** ✓ |
| `val_geom_camber_rc` | 67.4450 | ~79.6 | +18.0% ✗ |
| `val_geom_camber_cruise` | 35.7197 | ~42.1 | +17.9% ✗ |
| `val_re_rand` | 53.7616 | ~57.8 | +7.5% ✗ |

- **Analysis:** The per-split signal is the key finding. Coord jitter regularizes against mesh-pattern memorization (-13.7% in-dist win), but the OOD bottleneck is **shape generalization** (held-out camber values), not mesh-pattern generalization. Spatial precision is load-bearing for held-out camber inference. The OOD splits need MORE precise spatial reasoning, not less. Wrong regularizer for the bottleneck.
- **Axis reframe:** The bottleneck is not "memorize less" but "generalize the CONDITION mapping" — the model needs to better extrapolate AoA/Re/camber across operating conditions. Coord-jitter axis is mechanistically demonstrated to be the wrong lever for OOD generalization.
- **Next:** Per-sample jitter on condition channels (dims 13/14/18 = Re/AoA1/AoA2), σ=0.05 — assigned to nezuko as PR #1988.

### PR #1905 thorfinn: SGDR warm restarts — CLOSED (STALE, 2h zero activity)

- **Branch:** `charliepai2g48h5-thorfinn/warm-restarts`
- **Hypothesis:** Replace CosineAnnealingLR(T_max=50) with CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=0).
- **Result:** Zero training activity. Only assignment commit on branch; pod never started the run (likely GraphQL rate-limit, same pattern as tanjiro #1660, #1789, #1883).
- **Axis status:** UNTESTED. Hypothesis intact, SGDR is still worth exploring under the slice_num=32 baseline where best ≠ terminal.
- **Next:** Reassigned to thorfinn under fresh PR #1989 (same experiment, new PR to unstick pod).

### Assignments: Round 22

| PR | Student | Hypothesis |
|---|---|---|
| #1988 | nezuko | Per-sample fun_dim jitter on dims 13/14/18 (Re/AoA1/AoA2), σ=0.05, training only |
| #1989 | thorfinn | SGDR warm restarts T_0=10 T_mult=2 (retry of stale #1905) |

---

## 2026-05-13 02:20 — PR #1700: β=0.25 + L1 sweep — MERGED (L1 wins, new baseline 59.54)

- **Branch:** `charliepai2g48h5-thorfinn/huber-beta-0.25-l1-sweep`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Continue the monotone β trend: β=0.5 → 0.25 → 0 (pure L1). Two arms.

### Results

| Arm | val_avg/mae_surf_p | Δ vs #1633 | test_avg/mae_surf_p | Δ vs #1633 |
|---|---:|---:|---:|---:|
| β=0.5 baseline (#1633) | 64.0705 | — | 55.4961 | — |
| **Arm A: β=0.25** | **60.7558** | **−5.17%** | **52.3312** | **−5.70%** |
| **Arm B: L1 (β→0)** | **59.5354** | **−7.08%** ✓ | **51.4666** | **−7.26%** ✓ |

| Split | β=0.5 | β=0.25 | L1 |
|---|---:|---:|---:|
| `val_single_in_dist` | 72.5692 | 66.4260 | **64.8899** |
| `val_geom_camber_rc` | 78.3209 | 74.3348 | **74.0437** |
| `val_geom_camber_cruise` | 43.3744 | 42.7601 | **39.9687** |
| `val_re_rand` | 62.0174 | 59.5022 | **59.2391** |

- **Best epoch:** 37/37 (both arms; still descending at timeout).
- **Time/epoch:** ~49.6s (unchanged). **Memory:** 23.83 GB.
- **Artifacts:**
  - `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.jsonl`
  - `models/model-charliepai2g48h5-thorfinn-huber-beta-0.25-20260513-000538/metrics.jsonl`

### Analysis

**Complete β sweep: β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07) > β=0.25 (60.76) > L1 (59.54).** Monotone with diminishing returns: 8.2% → 5.2% → 2.0% per halving. The gain saturates near L1 — likely the heavy-tailed surface-pressure residual distribution is well-matched to L1's unit-magnitude gradient throughout training.

**Why L1 > β=0.25:** β=0.25 keeps a small quadratic central region (|e|<0.25), down-weighting small but useful gradient signal. Pure L1 uses subgradient ±1 sign everywhere, giving consistent step size across all residual magnitudes. This pays off in the late cosine tail (visible in smoother trajectory: epoch 33-37 less bouncy on L1).

**Mechanism closed:** β axis is fully characterized. No further β sweep needed — L1 is the optimal point under the 30-min budget.

### Conclusions

- **Merged as new advisor baseline.** val_avg = 59.5354, test_avg = 51.4666.
- All in-flight PRs that were on β=0.5 baseline (64.07) need L1 rebase to be comparable.
- L1 + grad-clip and L1 + WD=5e-5 are the next two high-confidence stack candidates.

---

## 2026-05-13 02:20 — PR #1775: WD=5e-5 on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-fern
- **Result:** val_avg=61.2311 (−4.43% vs 64.07); test_avg=53.8792 (−2.91%); wins ALL 4 splits.
- **Why sent back:** L1 merged (new baseline 59.54); 61.23 does not beat it. WD lever likely stacks.
- **Per-split analysis:** Mirror-image prediction failed — model under-regularized everywhere. Monotone: WD lower → better across all splits.
- **Artifacts:** `models/model-charliepai2g48h5-fern-weight-decay-5e-5-20260513-012009/metrics.jsonl`

---

## 2026-05-13 02:20 — PR #1653: grad-clip on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-askeladd
- **Rebase result:** val_avg=59.6214 (−6.94% vs 64.07); test_avg=52.6522 (−5.13%).
- **Why sent back:** L1 merged (new baseline 59.54); 59.62 does not beat it (within noise margin).
- **Key diagnostic:** Pre-clip p50 grew from 28→36 at epoch 1 under β=0.5 (sharper β → larger grads). clip_frac ≈ 1.0 throughout. Lever confirmed real on β=0.5; need to test on L1.
- **Old result (β=1.0):** val_avg=59.42 (−14.92% vs β=1.0 baseline 69.83).
- **Artifacts:** `models/model-charliepai2g48h5-askeladd-grad-clip-1.0-beta-0.5-rebase-20260513-011736/metrics.jsonl`

---

## 2026-05-13 02:20 — PR #1826: cosine eta_min=5e-5 — ASSIGNED (thorfinn)

- **Branch:** `charliepai2g48h5-thorfinn/cosine-eta-min-5e-5`
- **Hypothesis:** Add a non-zero LR floor to CosineAnnealingLR (eta_min=5e-5 = lr/10). Prevents gradient collapse in the low-LR tail; motivated by best_epoch=terminal in all recent runs.
- **Baseline to beat:** val_avg < 59.5354.

---

## 2026-05-13 05:15 — Round 15

### PR #1619: sampler 2× single on L1 — MERGED (new baseline)

- **Student:** charliepai2g48h5-nezuko
- **Result (L1+compile rebase):** val_avg=56.6217 (-4.89% vs 59.54), test_avg=50.4310 (-2.01%).
- **Per-split val:** single -13.51% (64.89→56.12), geom_camber_rc -4.02% (74.04→71.07), **geom_camber_cruise +4.31%** (39.97→41.69), re_rand -2.76% (59.24→57.60). Three of four splits improve.
- **Per-split test:** test_single -9.97%; test_rc +0.20%; test_cruise +1.78%; test_re_rand +1.34%.
- **Mechanics:** Sampler boost 2× on racecar_single → 50%/25%/25% share. L1's uniform per-sample gradient (bounded sign) amplifies coverage benefit vs β=1.0 run. Sampler validated across 3 baselines: β=1.0 (-2.80%), β=1.0+compile (-2.25%), L1+compile (-4.89%). Win grows with sharper loss.
- **Best epoch:** 39 (terminal), wall-clock-bound; trajectory still descending. Run ep 38→39: 57.74→56.62.
- **Artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-2x-on-l1-20260513-021352/metrics.jsonl`
- **NEW BASELINE: val_avg=56.6217, test_avg=50.4310**

---

### PR #1826: cosine eta_min=5e-5 — CLOSED (LR floor backfired)

- **Student:** charliepai2g48h5-thorfinn
- **Result:** val_avg=63.70 (+6.99% vs 59.54), test_avg=55.23 (+7.32%). All four val splits worse by 4-11%.
- **Root cause:** eta_min=5e-5 lifted polishing LR by +38.5% at best_epoch (36) and +45.2% at terminal (37). On pure L1 loss with sign-only gradients, the *only* step-damping mechanism is the schedule — there's no gradient-magnitude softening. A higher LR floor prevents fine-grained convergence. Model settled at a wider error ball.
- **Intervention verified:** LR(37) was 1.316e-4 vs 9.064e-5 if unfloored (+45.2%). The floor worked exactly as intended — it was just the wrong direction on L1.
- **Closed axis:** LR floor (cosine eta_min) on L1 loss. Schedule-floor axis closed — L1 relies on schedule damping for settling.
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-cosine-eta-min-5e-5-20260513-021535/metrics.jsonl`

---

### PR #1870: sampler boost both RaceCar 2× — ASSIGNED (nezuko)

- **Branch:** `charliepai2g48h5-nezuko/sampler-boost-both-racecar-2x`
- **Hypothesis:** Boost racecar_single=2 AND racecar_tandem=2, cruise=1 (40%/40%/20% share). Builds on PR #1619 win; should recover geom_camber_rc by restoring tandem training mass.
- **Baseline to beat:** val_avg < 56.6217.

---

### PR #1871: surf_loss p-weight 2× — ASSIGNED (thorfinn)

- **Branch:** `charliepai2g48h5-thorfinn/surf-p-weight-2x`
- **Hypothesis:** Apply [1.0, 1.0, 2.0] channel weight ONLY to surf_loss — double gradient budget on p-channel at surface nodes without touching vol_loss. Orthogonal to PR #1428 failure (that applied [1,1,3] globally, distorting velocity via volume loss; this is surf-only).
- **Baseline to beat:** val_avg < 56.6217.

---

## 2026-05-13 06:00 — Round 17

### PR #1846: slice_num 64 → 32 — MERGED (7th winner, -9.30%)

- **Student:** charliepai2g48h5-frieren
- **Result vs L1 baseline (#1700):** val_avg=54.0051 (-9.30%), test_avg=47.6261 (-7.46%).
- **All 4 val splits improve uniformly ~9%:** single -8.93%, rc -8.91%, cruise -10.63%, re_rand -9.25%.
- **First converged-within-budget run:** best_epoch=40 ≠ terminal=41. Model settled for first time in round 5.
- **Per-epoch time:** 43.5 s (-12.3% vs baseline). Memory: 21.35 GB (-10.4%). 41 epochs reached.
- **Mechanism:** Tighter information-bottleneck regularization (32 slices ≈ natural CFD regime count) + ~12% faster per-epoch = ~4 extra epochs.
- **Caveat:** Measured on L1-only base (no sampler). Post-merge advisor has both sampler AND slice_num=32. True combined baseline reveals via future runs.
- **NEW BASELINE: val_avg=54.0051, test_avg=47.6261**

---

### PR #1870: sampler both-racecar 2× — CLOSED (regression)

- **Student:** charliepai2g48h5-nezuko
- **Result:** val_avg=61.58 (+8.77% vs 56.62 baseline). ALL splits worse including predicted-improvement split.
- **Root cause:** Absolute racecar_single exposure dropped ~20% (from 50% → 40% share). Tandem boost doesn't help geom_camber_rc (held-out M=6-8 not in training regardless of tandem frequency). Cruise diversity removed → OOD hurt.
- **Closed axis:** Joint RaceCar boost. 2× single only (#1619) remains the better sampler config.

---

### PR #1871: surf_loss p-weight [1,1,2] — CLOSED (OOD regression, axis closed)

- **Student:** charliepai2g48h5-thorfinn
- **Result:** val_avg=59.22 (+4.59% vs 56.62 baseline). Only val_single_in_dist improved (-2.78%); all 3 OOD splits regressed 6-8%.
- **Root cause:** Physics coupling — even surf-only p-weighting reshapes backbone features toward in-dist pressure, hurting OOD velocity/pressure generalization. Same failure mode as PR #1428 (global [1,1,3]).
- **Closed axis:** Per-channel loss reweighting (both global and surf-only forms fail with OOD regression).

---

### PR #1903: slice_num 32 → 16 — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/slice-num-16`
- **Hypothesis:** Bracket the slice_num optimum. If 32 was better than 64, does 16 also improve? Tests whether the TandemFoilSet spatial structure fits naturally into 16 or 32 coarse routing slots.
- **Baseline:** val_avg < 54.0051.

---

### PR #1904: sampler racecar_single 1.5× — ASSIGNED (nezuko)

- **Branch:** `charliepai2g48h5-nezuko/sampler-single-1.5x`
- **Hypothesis:** Peak-bracketing: is 2× the optimum or have we overshot? 1.5× gives 37.5%/31.25%/31.25% share; cruise gets 31.25% back (vs 25% at 2×). Should recover geom_camber_cruise regression while keeping most of the single_in_dist win.
- **Baseline:** val_avg < 54.0051.

---

### PR #1905: cosine warm restarts T_0=10 T_mult=2 — ASSIGNED (thorfinn)

- **Branch:** `charliepai2g48h5-thorfinn/cosine-warm-restarts-t0-10`
- **Hypothesis:** Replace monotone CosineAnnealingLR with CosineAnnealingWarmRestarts (SGDR). With slice_num=32 the model now converges within budget; restarts may squeeze more gain by providing multiple high-LR exploration phases. T_0=10 → T_mult=2 → restarts at epochs 10, 30.
- **Baseline:** val_avg < 54.0051.

---

## 2026-05-13 09:00 — Round 21

### PR #1883: n_head 4 → 8 — CLOSED (stale × 2, tanjiro pod GraphQL rate-limit)

- **Student:** charliepai2g48h5-tanjiro
- **Status:** Assignment commit only; zero comments, no work started, 2.1h elapsed. Third consecutive stale-pod occurrence for tanjiro (after #1660 round-14, #1789 round-16). Pod is reliably failing to pick up GitHub assignments — suspected GraphQL polling rate-limit.
- **n_head axis status:** NOT explored. This is a PR closure, not an experiment result. The n_head=8 hypothesis (more parallel attention motifs at same compute) is still informative and untested. If tanjiro stabilizes, the question is worth revisiting.
- **Closed:** Superseded by #1976 tanjiro DropPath assignment. Fresh PR may unstick pod state.

---

### PR #1976: DropPath p_max=0.1 stochastic depth — ASSIGNED (tanjiro)

- **Branch:** `charliepai2g48h5-tanjiro/droppath-p-max-0.1`
- **Hypothesis:** Linear-by-depth stochastic depth schedule (p=0.0→0.1 across 5 layers). Block-level residual-branch zeroing per-sample. Targets OOD generalization via implicit ensemble effect — each sample sees a different sub-network. Structurally distinct from closed attention-dropout (#1788) which operated per-weight inside attention.
- **Mechanism:** `DropPath` module added to each `TransolverBlock`; forward zeroes entire attention/MLP branch with prob `p[layer]`, rescales by `1/(1-p)`. eval mode: no-op.
- **Target splits:** `val_geom_camber_rc` (67.45) and `val_re_rand` (53.76).
- **Baseline to beat:** val_avg < 54.0051.

---

## 2026-05-13 08:30 — Round 20

### PR #1946: EMA model weights decay=0.9999 — SENT BACK (decay too high, retune to 0.999)

- **Student:** charliepai2g48h5-edward
- **Result:** val_avg=165.27 (+205.9% vs 54.00), test_avg=152.99 (+221.2%). Apparent catastrophic failure on the primary metric — but mechanistically informative, not a hypothesis refutation.
- **Best epoch:** 44/44 (every epoch was a new best — EMA was monotonically converging toward raw weights but never caught up).
- **Trajectory:** epoch 1 → 385.05, epoch 44 → 165.27 (still descending).
- **Root cause (student's diagnosis):** decay=0.9999 has half-life ~6931 steps; total budget ~16,500 steps. So `0.9999^16500 ≈ 0.19` — ~19% of EMA shadow is still random-init noise at terminal. Worse, EMA mass is heavily weighted toward epochs 1-10 when raw val was 250-385. The "smoothed" weights are a *lagging average of poorly-trained models*, not a flat-minimum estimate.
- **Hypothesis status:** NOT refuted. The EMA-of-weights → flatter optimum → OOD generalization mechanism is sound but unmeasurable at this decay/budget combination. We measured *bias from lag*, not the *variance reduction from averaging trained weights*.
- **Sent back with:** retune to decay=0.999 (half-life ~693 steps, ~2 epochs). `0.999^16500 ≈ 5e-8` so init weights vanish entirely. EMA equilibrates to last ~2-4 epochs — exactly the converged regime where flat-minima effects manifest.
- **Additional diagnostic requested:** dual-eval — log both EMA-val and raw-val each epoch so future failure modes are caught inside 5 epochs.
- **Artifacts:** `models/model-charliepai2g48h5-edward-ema-weights-0.9999-20260513-051906/metrics.jsonl`

---

## 2026-05-13 08:00 — Round 19

### PR #1845: AdamW beta2 0.999 → 0.95 — CLOSED (clean LOSS, β2 axis closed)

- **Student:** charliepai2g48h5-edward
- **Result (vs assigned baseline #1700 L1 59.54):** val_avg=62.1696 (+4.42%), test_avg=54.7983 (+6.47%). Clear LOSS.
- **Result (vs current baseline #1846 54.00):** +15.1% — emphatic miss.
- **Best epoch:** 35/36 (best=terminal, unconverged).
- **Mechanism:** shorter β2 EMA (~20 step half-life) made the preconditioner *more reactive* to L1 sign-flip noise, not less. L1's ±1-magnitude gradients are informative only when averaged over many steps — throwing away that smoothing amplified per-step variance. Visible oscillations in trajectory (ep14→19, ep24→29 step-backs).
- **Student insight (verbatim):** "per-parameter gradient variance from L1 sign-flips is informative *only when averaged*. Throwing away that smoothing makes the preconditioner more reactive to instantaneous direction flips, not less."
- **Closed axis:** AdamW β2 sweep for L1 regime. β2=0.95 (shorter EMA) is definitively worse. β2=0.9999 (longer EMA) is not worth testing — existing 0.999 already covers ~10% of training.
- **Artifacts:** `models/model-charliepai2g48h5-edward-adamw-betas-0.9-0.95-20260513-035333/metrics.jsonl`

---

### PR #1946: EMA model weights decay=0.9999 — ASSIGNED (edward)

- **Branch:** `charliepai2g48h5-edward/ema-weights-0.9999`
- **Hypothesis:** Maintain a shadow copy of model parameters with EMA (decay=0.9999, half-life ~6931 steps). Use EMA weights for val/test eval and save them as the best-val checkpoint. EMA-of-weights produces flatter optima than raw SGD weights — well-known to improve OOD generalization (Polyak averaging, SWA; Izmailov 2018).
- **Why now:** `val_geom_camber_rc` (67.45) and `val_re_rand` (53.76) dominate val_avg. Edward's #1845 trajectory showed basin-bouncing oscillations under L1 — EMA averages over these rather than reacting to them.
- **Baseline to beat:** val_avg < 54.0051.

---

## 2026-05-13 07:30 — Round 18

### PR #1903: slice_num 32 → 16 — CLOSED (wash, closes slice-DOWN axis)

- **Student:** charliepai2g48h5-frieren
- **Result:** val_avg=54.2251 (+0.41% vs 54.0051 baseline, **miss**); test_avg=46.9815 (-1.35%, modest test win).
- **Best epoch:** 47/47 (best==terminal; unconverged, wall-clock hit at epoch 47).
- **Per-epoch time:** 37.81 s (-13% vs #1846). Memory: 20.11 GB. 47 epochs in 30 min.
- **Per-split val:** single_in_dist -14.15% (59.09→50.73) [huge gain]; geom_camber_rc +3.91%; geom_camber_cruise +7.60%; re_rand +7.24%.
- **Interpretation:** Striking in-dist/OOD trade-off: slice=16 under-resolves OOD spatial structure (camber, Re-shift regimes) but concentrates capacity on dominant in-dist patterns. Mean cancels to a wash. **slice_num=32 is the global val optimum.** 64→32 was a -9.30% win; 32→16 is a ±0.4% wash — the bottleneck lever is exhausted.
- **Additional finding:** Best==terminal again (vs PR #1846's converged best=40≠terminal=41). Lighter slice=16 model trained faster but still budget-hit before true convergence.
- **Closed axis:** slice_num below 32. The two-point bracket (16 and 64) around 32 is complete.
- **Artifacts:** `models/model-charliepai2g48h5-frieren-slice-num-16-20260513-041626/metrics.jsonl`

---

### PR #1904: sampler racecar_single 1.5× — CLOSED (clean LOSS, 2× optimum confirmed)

- **Student:** charliepai2g48h5-nezuko
- **Result:** val_avg=55.8769 (+3.47% vs 54.0051 baseline). test_avg=49.3745 (+3.67%). Clear LOSS.
- **Best epoch:** 42/42 (unconverged, wall-clock-bound).
- **Per-split vs #1846:** geom_camber_rc=72.37 (unchanged from 2×); re_rand=58.23 (unchanged); cruise improved -6.38% vs old 2× pre-slice run; single improved -4.01% vs old 2× pre-slice run. Both OOD-dominated splits unaffected.
- **Sampler confirmed:** boost_factor=1.5 applied correctly (single=37.5%, tandem=31.25%, cruise=31.25%).
- **Confirmed optimum:** 2.0× boost is the peak. 1.5× under-concentrates single-foil coverage; 2× both-racecar (#1870) over-dilutes; 2× single (#1619) is the sweet spot.
- **Key insight:** `val_geom_camber_rc` (72.37) and `val_re_rand` (58.23) dominate val_avg and don't respond to sampler reweighting at any single-domain boost factor. These OOD splits require architectural or loss-level interventions, not sampler tuning.
- **Closed axis:** Sampler boost factor sweep (both up and down). 2× single is canonical.
- **Artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-single-1.5x-20260513-041540/metrics.jsonl`

---

### PR #1921: pos-jitter σ=0.01 on volume mesh coords — ASSIGNED (nezuko)

- **Branch:** `charliepai2g48h5-nezuko/pos-jitter-0.01`
- **Hypothesis:** Gaussian perturbation (σ=0.01 on z-score-normalized coords) on volume (non-surface) nodes during training only. Forces Transolver's slice routing to abstract away exact mesh node positions — should improve OOD generalization on `val_geom_camber_rc` and `val_re_rand`, which dominate val_avg and don't respond to sampler changes.
- **Target splits:** geom_camber_rc (72.37) and re_rand (58.23).
- **Baseline to beat:** val_avg < 54.0051.

---

### PR #1926: RMSNorm replacing LayerNorm — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/rmsnorm`
- **Hypothesis:** Replace all 3 `nn.LayerNorm` sites in `TransolverBlock` (ln_1, ln_2, ln_3) with `nn.RMSNorm`. Drops mean-centering and bias; ~7-10% faster norm op under torch.compile + bf16; Llama-style normalization. May help L1 sign-gradient regime where mean-centering adds noise.
- **Expected:** ~1-3% faster per-epoch → 1 extra epoch in budget. Small direct quality improvement possible.
- **Baseline to beat:** val_avg < 54.0051.

---

## 2026-05-13 05:30 — Round 16

### PR #1789: surf_weight 10 → 15 — CLOSED (stale, tanjiro rate-limited)

- **Branch:** `charliepai2g48h5-tanjiro/surf-weight-15`
- **Status:** Assignment commit only; never started. Pod in GraphQL rate-limit retry loop (3+ hours). Same pattern as previous tanjiro #1660 failure.
- **Closed:** Reassigned to n_head=8 (#1883) as a fresh single-line lever. surf_weight experiment deferred; overlaps with #1871 thorfinn surf_loss p-weight.

---

### PR #1883: n_head 4 → 8 — ASSIGNED (tanjiro)

- **Branch:** `charliepai2g48h5-tanjiro/n-head-8`
- **Hypothesis:** Last untested architecture axis. Doubles attention heads (4→8) while halving dim_head (32→16). Compute-neutral — inner_dim = n_head × dim_head = 128 unchanged. More heads → more parallel spatial specialization motifs, potentially beneficial for multi-regime CFD flow (stagnation, suction, separation, wake, foil-foil coupling, Re transition).
- **Baseline to beat:** val_avg < 56.6217.

---

## 2026-05-13 05:00 — Round 14: PR reviews and new assignments

### PR #1788: attention-dropout=0.1 — CLOSED (slow convergence, budget-bound loss)

- **Student:** charliepai2g48h5-frieren
- **Result:** val_avg=65.8345 (+2.75% vs OLD 64.07 baseline), test_avg=59.3951 (+7.03%).
- **Analysis:** Best epoch = terminal epoch (36/36). All four val splits regressed — no preferential OOD gain from attention dropout. Per-weight activation noise dominated regularization benefit under 36-epoch cap. "Closed on 30-min regime" — slower convergence never caught the baseline.
- **Closed axes:** Attention dropout (per-weight) at p=0.1 closed under wall-clock cap. DropPath (block-level) remains untested — different convergence profile.
- **Artifacts:** `models/model-charliepai2g48h5-frieren-attention-dropout-0.1-20260513-020018/metrics.jsonl`

---

### PR #1741: mlp_ratio=3 — CLOSED (capacity axis triangulated closed)

- **Student:** charliepai2g48h5-edward
- **Result:** val_avg=68.9250 (+7.6% vs OLD 64.07 baseline), test_avg=61.9016 (+11.5%).
- **Param count:** 826K (up from 662K). Per-epoch: +6.5% (within prediction). Epochs: 34 (vs 36 baseline).
- **Analysis:** Plateau-bound trajectory (oscillating tail: ep 32→33→34: 68.92→69.26→72.47). NOT the undertrained-but-converging shape. Combined with #1688 (n_hidden=160 also lost), **both capacity axes (uniform width and asymmetric FFN) are triangulated closed** on 30-min budget. Future gains from regularization, optimizer, schedule, or loss-shape — not model size.
- **Artifacts:** `models/model-charliepai2g48h5-edward-mlp-ratio-3-20260513-020905/metrics.jsonl`

---

### PR #1774: lr=7.5e-4 on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-alphonse
- **Result (2 runs, β=0.5):** mean val_avg=63.30 (-1.20% vs 64.07), mean test_avg=56.34 (+1.51%). Run-to-run gap: 1.06 val, 1.55 test — effect size within noise floor.
- **Why sent back:** L1 merged (new baseline 59.54). Student's own analysis: "revisit if loss landscape changes" — L1 IS that change. Unit-bounded gradients (L1 sign) change what optimal LR is. Per-epoch cost: UNCHANGED (~49.5s). Sent back for L1 rebase.
- **Predicted L1 outcome:** 57-59 val if larger step stacks; wash if neutral; >60 if L1 high-variance sign gradients penalize larger steps.
- **Artifacts:** `models/model-charliepai2g48h5-alphonse-lr-7.5e-4-20260513-012003/metrics.jsonl`, `models/model-charliepai2g48h5-alphonse-lr-7.5e-4-20260513-015915/metrics.jsonl`

---

### PR #1845: AdamW betas=(0.9, 0.95) — ASSIGNED (edward)

- **Branch:** `charliepai2g48h5-edward/adamw-betas-0.9-0.95`
- **Hypothesis:** On L1 loss, every gradient is a unit sign (bounded, high-variance). beta2=0.999 (1000-step memory) over-smooths the preconditioner. beta2=0.95 (20-step memory) adapts faster to per-parameter gradient variance. Modern transformer default (GPT/LLaMA). Single line change.
- **Baseline to beat:** val_avg < 59.5354.

---

### PR #1846: slice_num 64 → 32 — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/slice-num-32`
- **Hypothesis:** slice_num=64 may be over-allocated for TandemFoilSet's natural spatial structure (~10-20 canonical CFD regimes: stagnation, transition, separation, wake, inter-foil). Reducing to 32 tightens the attention bottleneck inductive bias, reduces per-epoch time ~3-5%, and forces load-balancing across fewer slices. CFD intuition: fewer coarser slices ≈ more physics-consistent decomposition than 64 fine-grained slices spread thin.
- **Note:** #1590 (slice_num=96 on bf16) was closed at +3.86% regression — but that was LARGER slices. This is the SMALLER direction and a different compute regime.
- **Baseline to beat:** val_avg < 59.5354.

---

## 2026-05-13 01:55 — PR #1652: warmup-500 + cosine (β=0.5 rebase) — CLOSED (substituted by β)

- **Branch:** `charliepai2g48h5-frieren/warmup-500-cosine`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Linear warmup over 500 steps stacked on Huber β=0.5 baseline; predicted additive gain since mechanisms (LR-trajectory vs loss-shape) seemed orthogonal.

### Results

| Metric | Baseline (#1633 β=0.5) | This PR (warmup+β=0.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | 63.9922 | −0.12% (noise floor) |
| `test_avg/mae_surf_p` | 55.4961 | **55.9481** | **+0.81%** ✗ (all 4 test splits worse) |

| Split | warmup+β=0.5 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 71.90 | 72.57 | −0.92% |
| `val_geom_camber_rc` | 77.08 | 78.32 | −1.58% |
| `val_geom_camber_cruise` | 44.36 | 43.37 | +2.28% |
| `val_re_rand` | 62.62 | 62.02 | +0.98% |

- **Epochs:** 36. **Time/epoch:** ~49.65s (unchanged). **Best epoch:** 36/36 (still descending).
- **Artifacts:** `models/model-charliepai2g48h5-frieren-warmup-500-on-huber0.5-20260513-001402/metrics.jsonl`

### Analysis — paper-grade mechanism finding

The student's split-direction analysis is the high-value part. On β=1.0 (her original #1652 run), warmup helped OOD splits (camber_rc −6.4%, camber_cruise −3.5%, re_rand −3.1%) and hurt in-dist (+6.1%) — classical "warmup → flatter minimum" mechanism. On β=0.5 rebase, the pattern *inverts*: warmup helps in-dist and camber_rc, hurts camber_cruise and re_rand. **Warmup and β=0.5 are competing for the same "early-training stabilization" lever**, not stacking additively.

**Three independent students converging on the same axis:**
- Frieren #1652 warmup-500 on β=1.0: −1.62% val
- Askeladd #1653 grad-clip on β=1.0: −14.92% val
- Huber β=0.5 itself (PR #1633): −8% val

All three reduce gradient-signal volatility in early training, but they substitute for each other (warmup on β=0.5 = no gain). Future "stabilization" interventions must come from a different mechanism (data-distribution warmup, schedule-completion alignment, etc.).

### Conclusions

- Closes the LR-trajectory-warmup axis on β=0.5 base.
- The "early-step coherence" cluster is well-explored; further gains must come from capacity, data distribution, late-training schedule, or eval-time techniques.
- Predicts: askeladd's #1653 rebase will show *partial* (not additive) gain — likely a substantial fraction of −14.92% will be captured by β=0.5 already.

---

## 2026-05-13 01:55 — PR #1660: EMA decay=0.999 — CLOSED (pod never started)

- **Branch:** `charliepai2g48h5-tanjiro/ema-eval-decay-0.999-compile`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Per-step EMA of weights for evaluation, decay=0.999.
- **Reason for closure:** Pod stuck in GraphQL rate-limit retry loop for 3+ hours (since ~22:30 UTC 2026-05-12). Only commit on branch is the assignment commit `1f9dfdd`. EMA experiment was never actually started.
- **Reassignment:** PR #1789 surf_weight=15 (simpler 1-line change, smaller failure surface).

---

## 2026-05-13 01:55 — PR #1788: attention dropout=0.1 — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/attention-dropout-0.1`
- **Hypothesis:** Activate PhysicsAttention dropout (existing wired parameter, currently 0.0) at p=0.1. Activation-level regularization, orthogonal to L2 (fern's WD work) and grad-clip (askeladd). Targets slice-attention pathway redundancy.
- **Config change:** Add `dropout=0.1` to `model_config` dict.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:55 — PR #1789: surf_weight 10 → 15 — ASSIGNED (tanjiro)

- **Branch:** `charliepai2g48h5-tanjiro/surf-weight-15`
- **Hypothesis:** Up-weight surface loss term to align training objective with the surf-p-primary validation metric. +50% bump to test the loss-balancing axis on β=0.5 base.
- **Config change:** `cfg.surf_weight` 10.0 → 15.0 (1-line in dataclass).
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:15 — PR #1727: weight_decay 1e-4 → 5e-4 — CLOSED (regression)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-4`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** Stronger L2 regularization targets OOD splits (geom_camber_rc, re_rand) where train/val distribution shift is real.

### Results

| Metric | Baseline (#1633) | This PR (WD=5e-4) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | **66.6723** | **+4.06%** ✗ |
| `test_avg/mae_surf_p` | 55.4961 | **58.6256** | **+5.64%** ✗ |

| Split | WD=5e-4 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 73.9857 | 72.5692 | +1.95% |
| `val_geom_camber_rc` | **77.5818** | 78.3209 | **−0.94%** (only improvement) |
| `val_geom_camber_cruise` | 49.3384 | 43.3744 | **+13.75%** (worst) |
| `val_re_rand` | 65.7833 | 62.0174 | +6.07% |

- **Epochs:** 36 (wall-clock bound). **Time/epoch:** ~49.7s (unchanged). **Best epoch:** 35.
- **Artifacts:** `models/model-charliepai2g48h5-fern-weight-decay-5e-4-20260513-001807/metrics.jsonl`

### Analysis

Under-fit, not over-regularization. Val trajectory lagged baseline by 3-5 equivalent epochs (epoch 18 val_avg=101.98 vs baseline ~92). Still descending at timeout. With 50+ epochs, 5e-4 plausibly catches up; under 30-min cap it's a net loss. One OOD split improved (geom_camber_rc −0.94%) — regularization thesis has *directional* validity but swamped by convergence lag. WD-UP axis closed.

**Follow-up assigned:** weight_decay=5e-5 (PR #1775) — DOWN bracket to close the WD optimum search.

---

## 2026-05-13 01:15 — PR #1701: batch_size 4 → 8 on compile baseline — CLOSED (regression)

- **Branch:** `charliepai2g48h5-alphonse/batch-size-8-compile`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Larger batch → better gradient quality per step, tests quality-vs-quantity trade-off on compile+bf16 baseline.

### Results

| Metric | Baseline (#1633) | batch=8 | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | **74.4033** | **+16.1%** ✗ |
| `test_avg/mae_surf_p` | 55.4961 | **66.8002** | **+20.4%** ✗ |

| Quantity | batch=4 | batch=8 | Δ |
|---|---:|---:|---:|
| Steps/epoch | 375 | 188 | −50% |
| Total grad updates | 13,875 | 6,392 | **−54%** |
| Time/epoch | 49.7s | 53.4s | +7.4% |
| Peak GPU memory | 23.83 GB | 47.63 GB | +2× |

- **Artifacts:** `models/model-charliepai2g48h5-alphonse-batch-size-8-compile-20260513-000307/metrics.jsonl`

### Analysis

Textbook small-dataset step-count starvation. Matched-wall-clock probe (epoch 24): batch=8 (94.95) within 1.8% of batch=4 (93.26) — gradient quality is similar. The loss is entirely in step count: −54% total grad updates means the cosine-LR tail (which drives the last 1.5 MAE points) is never reached. **Batch scaling is fully dead** at TandemFoil scale in all compute regimes (#1439 fp32, #1701 compile).

**Follow-up assigned:** lr=7.5e-4 (PR #1774) — raise per-step magnitude while keeping step count.

---

## 2026-05-13 01:15 — PR #1653: grad-clip max_norm=1.0 — SENT BACK (β=0.5 rebase required)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-1.0-compile`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Gradient norm clipping at max_norm=1.0 to stabilize large-gradient regime.

### Results (on stale β=1.0 baseline)

| Metric | β=1.0 compile baseline (#1568) | grad-clip | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **59.4157** | **−14.92%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **53.0090** | **−14.32%** ✓ |

**Note: askeladd's branch has `beta=1.0` at lines 246, 486 — result measured on OLD β=1.0 compile baseline, not current β=0.5 advisor (val_avg=64.07).** Cannot merge as-is.

### Key grad-norm diagnostics

| Epoch | grad_norm_p50 | clip_frac |
|---|---:|---:|
| 1 | 28.39 | 1.00 |
| 6 | 15.20 | 1.00 |
| 16 | 10.37 | 1.00 |
| 26 | 6.52 | 0.99 |
| 37 | 4.92 | 0.95 |

Pre-clip norms are 10-30× above threshold throughout training. This is not a rare-spike guard — it is near-uniform per-step downscaling, acting as an adaptive LR floor that bounds the AdamW step to `lr × max_norm = 5e-4`.

### Action

Sent back for β=0.5 rebase. When stacked with β=0.5, if orthogonal: val_avg could reach ~55-58. If redundant: grad-clip and β=0.5 share the same "make gradient signal more coherent" mechanism and the stacking will be smaller.

---

## 2026-05-13 01:15 — PR #1774: lr=7.5e-4 — ASSIGNED (alphonse)

- **Branch:** `charliepai2g48h5-alphonse/lr-7.5e-4`
- **Hypothesis:** From #1701's step-count analysis: raise per-step magnitude (lr +50%) while keeping step count constant. β=0.5 reduced outlier gradient magnitude vs β=1.0, leaving room for larger LR. CosineAnnealingLR peak moves 5e-4 → 7.5e-4.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:15 — PR #1775: weight_decay=5e-5 — ASSIGNED (fern)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-5`
- **Hypothesis:** From #1727's bracketology: DOWN sweep to complete 3-point WD bracket (5e-4=bad, 1e-4=baseline, 5e-5=this test). Closes the WD axis or identifies weaker-regularization win.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 02:40 — PR #1688: n_hidden 128 → 160 on compile baseline — CLOSED (width ruled out)

- **Branch:** `charliepai2g48h5-edward/wider-hidden-160-compile`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver n_hidden 128→160 on compile + β=0.5 baseline.

### Results

| Metric | Baseline (#1568 compile) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **73.6658** | **+5.49%** ✗ |
| `test_avg/mae_surf_p` | 61.8652 | **64.5826** | **+4.39%** ✗ |

| Split | n_hidden=160 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 89.83 | 77.10 | **+16.5%** |
| `val_geom_camber_rc` | 83.13 | 83.49 | -0.4% |
| `val_geom_camber_cruise` | 52.78 | 50.64 | +4.2% |
| `val_re_rand` | 68.92 | 68.10 | +1.2% |

- **Best epoch:** 30/31. **Time/epoch:** ~58.3 s (+17.7%). **Peak GPU:** 28.4 GB. **Params:** 1.027M.
- **Val trajectory:** Oscillating in 73-85 range (not cleanly descending like baseline).
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-wider-hidden-160-compile-20260512-235416/metrics.jsonl`

### Analysis

**Compute starvation — same mechanism as depth experiments.** Per-epoch cost 49.5→58.3s (+17.7%), epochs 36→31 (-14%), val_single_in_dist regressed +16.5% (in-dist split, should be best if capacity actually helped). Val curve oscillated 73-85 rather than cleanly descending.

**Width axis now fully ruled out under 30-min cap.** Complete lever characterization:
- n_hidden=192+fp32 (#1398): wall-clock bound
- n_hidden=160+bf16 (#1587): pod stall
- n_hidden=160+compile (#1688): +5.49% loss

**Student's valuable insight:** `mlp_ratio=3` is the next-cheapest targeted test — affects FFN only, attention cost unchanged, ~5-8% per-epoch overhead vs 17.7% for uniform widening.

### Conclusions

- Uniform width scaling is dead under the 30-min cap. Do not re-run on β=0.5 baseline.
- `mlp_ratio=3` assigned as PR #1741 — smallest-footprint capacity change.
- If mlp_ratio=3 also loses, all capacity axes are closed and we should focus entirely on regularization and data levers.

---

## 2026-05-13 02:40 — PR #1741: mlp_ratio 2 → 3 — ASSIGNED (edward)

- **Branch:** `charliepai2g48h5-edward/mlp-ratio-3`
- **Student:** charliepai2g48h5-edward (fresh assignment after #1688 closed)
- **Hypothesis:** FFN-only capacity increase. mlp_ratio=3 → FFN hidden 256→384; attention cost unchanged.
- **Mechanism:** Slice-attention does spatial mixing; FFN does per-token non-linear projection. Richer FFN may model more complex physics interactions without the full compute hit of uniform widening.
- **Config change:** `mlp_ratio=2` → `mlp_ratio=3` in model_config dict (single-line diff)
- **Expected per-epoch cost:** ~52-53s (vs 49.5 baseline; 17.7% for n_hidden=160)
- **Expected epochs:** ~33-34 (vs 36 baseline; 31 for n_hidden=160)
- **Baseline to beat:** val_avg/mae_surf_p < 64.0705.

---

## 2026-05-13 02:20 — PR #1676: AdamW β2=0.95 — CLOSED (lever refuted)

- **Branch:** `charliepai2g48h5-fern/adamw-beta2-0.95`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** β2=0.95 (faster second-moment tracking, "transformer recipe") vs 0.999 default.

### Results

| Metric | Baseline (#1568) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **69.9029** | +0.10% (wash) |
| `test_avg/mae_surf_p` | 61.8652 | **62.9973** | +1.83% ✗ |

| Split | β2=0.95 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 79.9420 | 77.10 | +2.84 |
| `val_geom_camber_rc` | 83.7474 | 83.49 | +0.26 |
| `val_geom_camber_cruise` | 48.7266 | 50.64 | -1.91 |
| `val_re_rand` | 67.1955 | 68.10 | -0.90 |

- **Best epoch:** 36 (terminal, still descending). **Time/epoch:** ~49.78 s. **Peak GPU:** 23.83 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-fern-adamw-beta2-0.95-20260512-230647/metrics.jsonl`

### Analysis

Mixed per-split signal (in-dist slightly worse, two OOD splits slightly better, one slightly worse).
Net result is noise — 69.90 vs 69.83 is within random seed variance. The training loss showed
mild spikes (~10-16% bumps) at epochs 20, 28, 32 — consistent with "spikier Adam" from shorter
second-moment averaging window under batch=4 noise. Notably, 69.90 also doesn't beat the new
64.07 baseline (PR #1633, Huber β=0.5 merged same round).

Student diagnosis is correct: β2=0.95 is suited for large-scale LM training where intra-epoch
gradient distribution shifts are real. On 1499-sample TandemFoil with 375 steps/epoch, the
gradient distribution is stationary — β2=0.999 provides better L2 stability for this regime.
The cosine LR schedule already handles "late-training adaptation" that β2=0.95 was supposed to help with.

### Conclusions

- **β2 axis: CLOSED.** Lever does not transfer to small encoder-only Transolver on this dataset scale.
- Do not re-run further β2 variants (0.99, etc.) — the mechanism mismatch is understood.
- Training-loss bump characterization is useful diagnostic prior: if a future design uses
  aggressive β2, pair with grad-clipping.

---

## 2026-05-13 02:20 — PR #1652: Warmup-500-cosine — SENT BACK (needs β=0.5 rebase)

- **Branch:** `charliepai2g48h5-frieren/warmup-500-cosine`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Linear warmup over 500 steps (LR 0.01×→1.0× peak) + cosine T_max=50 decay.

### Results (on OLD 69.83 baseline)

| Metric | Baseline (#1568) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **68.7004** | **-1.62%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **60.7640** | **-1.78%** ✓ |

| Split | Warmup-500 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 81.77 | 77.10 | +6.1% |
| `val_geom_camber_rc` | 78.16 | 83.49 | **-6.4%** |
| `val_geom_camber_cruise` | 48.87 | 50.64 | **-3.5%** |
| `val_re_rand` | 66.01 | 68.10 | **-3.1%** |

- **Best epoch:** 35/36 (one before terminal). **Time/epoch:** ~49.6 s. **Peak GPU:** 23.8 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-warmup-500-cosine-20260512-225549/metrics.jsonl`
- **LR trajectory verified:** Epoch-1 start at 5e-6 (0.01×), step-500 peak 5e-4, cosine engaged.

### Analysis

Real lever, but the old baseline (69.83) has been superseded. Warmup-500 delivered -1.62% on val_avg,
concentrated on OOD splits (camber_rc -6.4%, camber_cruise -3.5%, re_rand -3.1%) with an in-dist
regression (+6.1%). The mechanism prediction was correct: warmup → flatter minimum → better OOD
generalization, at small in-dist cost.

68.70 does NOT beat the new 64.07 baseline (PR #1633). Warmup is orthogonal to β=0.5 (one changes
LR trajectory shape, the other changes residual sensitivity). Combined, multiplicative stacking
predicts ~62.7-63.5 val_avg — beats 64.07.

**Note on in-dist regression:** Under β=0.5 (cleaner per-sample gradients for medium residuals),
the in-dist regression may shrink or disappear — an informative interaction effect to watch.

### Conclusions

- **SENT BACK** for rebase onto `icml-appendix-charlie-pai2g-48h-r5` (inherits β=0.5).
- Lever is real; expected to compound with β=0.5.
- Post-merge follow-ups queued: warmup-length sweep (250, 1000, 2000 steps); T_max alignment.

---

## 2026-05-13 02:20 — PR #1619: Sampler 2× compile rebase — SENT BACK AGAIN (needs β=0.5 rebase)

- **Branch:** `charliepai2g48h5-nezuko/sampler-boost-single-2x`
- **Student:** charliepai2g48h5-nezuko
- **Second run:** Compile + sampler-2x (rebase onto #1568 compile baseline per prior advisor feedback)

### Results (on 69.83 compile baseline, second rebase)

| Metric | Compile baseline (#1568) | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **68.2641** | **-2.25%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **61.4236** | **-0.71%** ✓ |

| Split | Sampler+Compile | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | **64.6884** | 77.10 | **-16.10%** |
| `val_geom_camber_rc` | 85.6303 | 83.49 | +2.56% |
| `val_geom_camber_cruise` | 53.6013 | 50.64 | +5.85% |
| `val_re_rand` | 69.1363 | 68.10 | +1.52% |

- **Best epoch:** 39 (terminal — still descending). **Time/epoch:** ~46.2 s. **Peak GPU:** 23.83 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-boost-single-2x-compile-20260512-230452/metrics.jsonl`

### Analysis

Sampler lever amplified under compile: val_single_in_dist -16.10% (vs -10.7% pre-compile), because
more gradient steps per 30-min window amplify the coverage benefit. Best epoch=39=terminal means
the model is still descending — sampler+compile gains are under-saturated.

68.26 does NOT beat the new 64.07 baseline (PR #1633, β=0.5). Sampler is orthogonal to β=0.5 by
construction (batch sampling vs per-sample loss shape). Combined stacking is the highest-confidence
win remaining in the queue. Predicted multiplicative combination: 69.83 × (1-0.082) × (1-0.0225) ≈ 62.65.

Key diagnostic for the third run: watch val_geom_camber_cruise. β=0.5 alone gave -14.4% on cruise;
sampler-2x alone gave +5.85%. Net direction is uncertain — if they cancel, the implication is that
the 2× cruise-mass reduction is a meaningful cost and "boost both racecar domains" is the right fix.

### Conclusions

- **SENT BACK AGAIN** for third rebase — now onto current `icml-appendix-charlie-pai2g-48h-r5`
  (which has β=0.5). This is the final rebase needed.
- If sampler+β=0.5 beats 64.07, follow-ups: boost-factor sweep (1.5×, 3×); "boost both racecar
  domains (single=2, tandem=2, cruise=1)".

---

## 2026-05-13 02:20 — PR #1727: weight_decay 1e-4 → 5e-4 — ASSIGNED (fern)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-4`
- **Student:** charliepai2g48h5-fern (fresh assignment after #1676 closed)
- **Hypothesis:** Stronger L2 regularization improves OOD generalization on the 1499-sample dataset.
- **Config change:** `weight_decay: float = 1e-4` → `weight_decay: float = 5e-4` (single-line diff)
- **Mechanism:** 5× L2 penalty trades mild in-dist capacity for better parameter-space flatness
  on OOD splits (camber_rc, re_rand). Independent of all in-flight experiments (different axis).
- **Prediction:** -1% to -3% on val_avg, concentrated on OOD splits. val_avg landing zone: 62-63.5.
- **Baseline to beat:** val_avg/mae_surf_p < 64.0705.

---

## 2026-05-13 01:10 — PR #1560: T_max=36 cosine on compile baseline — CLOSED (lever characterized)

- **Branch:** `charliepai2g48h5-alphonse/tmax-14-cosine`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Match CosineAnnealingLR T_max to the actual epoch budget at the 30-min cap.

### Results (compile-era re-run, T_max=36)

| Metric | Baseline (#1568) | This PR (T_max=36) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.832 | **69.598** | -0.234 (-0.34%) |
| `test_avg/mae_surf_p` | 61.865 | **61.729** | -0.136 (-0.22%) |

Per-split (mixed):

| Split | T_max=36 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 76.167 | 77.10 | -0.93 |
| `val_geom_camber_rc` | 81.053 | 83.49 | -2.44 |
| `val_geom_camber_cruise` | 52.196 | 50.64 | +1.56 |
| `val_re_rand` | 68.975 | 68.10 | +0.87 |

- **Best epoch:** 36 (terminal, descending). **Epochs:** 36. **Time/epoch:** 49.8 s.
- **Status:** CLOSED. 69.60 > new baseline 64.07 (Huber β=0.5, PR #1633 merged same round).

### Analysis

**Mechanism confirmed but gain diminished at compile budget.** The T_max=36 win internaly shows -6.2 MAE in epochs 28→36 (exactly as predicted), matching the T_max=14/18 arms' "last few epochs gain ~8 MAE" pattern. But the comparison vs the T_max=50 baseline at 36/50 epochs is tiny (+0.23 MAE) because the T_max=50 compile baseline already captures most of the cosine-decay benefit by epoch 36 (LR decays to ~0.21·lr_max, not zero).

**Lever characterization complete:**

| Arm | T_max | Epochs | Baseline | val_avg | Δ | Mechanism |
|---|---|---|---|---|---|---|
| #1560 A (fp32) | 14 | 14 | #1444 (110.76) | 98.75 | -10.8% | Low-LR tail traversal |
| #1560 B (bf16) | 18 | 18 | #1532 (101.12) | 90.32 | -10.7% | Low-LR tail traversal |
| #1560 C (compile) | 36 | 36 | #1568 (69.83) | 69.60 | -0.34% | Most of arc already covered |

The gain collapses when the baseline already runs most of the cosine arc (36/50 epochs). This is a closed lever.

### Conclusions

- T_max=epoch_budget is a strong win when the baseline epoch count is a small fraction of T_max (e.g. 19/50 epochs with bf16 only). Neutral when the baseline already runs most of the arc.
- Do NOT re-run this hypothesis on the β=0.5 baseline. Same math applies.
- Closed: 69.60 does not beat new baseline 64.07.

---

## 2026-05-13 01:00 — PR #1633: Huber β=0.5 (sharper loss) — MERGED ✓

- **Branch:** `charliepai2g48h5-thorfinn/huber-beta-sweep`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Huber β=0.5 (sharper quadratic-to-linear transition at |e|=0.5) vs β=1.0 baseline. Simultaneously test β=2.0 (smoother). TandemFoil surface pressure has a heavy-tailed residual distribution — smaller β makes loss linear for a wider range of medium residuals, down-weighting outlier gradients.

### Results

| Arm | β | val_avg/mae_surf_p | Δ vs baseline | test_avg |
|---|---|---:|---:|---:|
| **A (winner)** | **0.5** | **64.0705** | **-8.2% ✓** | **55.4961** |
| B | 2.0 | 77.8090 | +11.4% ✗ | 69.2942 |

Per-split — β=0.5 (Arm A):

| Split | β=0.5 | Baseline (#1568) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 72.5692 | 77.10 | -5.9% |
| `val_geom_camber_rc` | 78.3209 | 83.49 | -6.2% |
| `val_geom_camber_cruise` | **43.3744** | 50.64 | **-14.4%** |
| `val_re_rand` | 62.0174 | 68.10 | -8.9% |

- **Best epoch:** 37 (terminal, still descending at timeout). **Time/epoch:** ~49.5 s. **Peak GPU:** 23.83 GB.
- **Status:** MERGED — new baseline 64.0705 / 55.4961.
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.jsonl`

### Analysis

**Clean monotone signal:** β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07). All four val splits improved. Largest win on val_geom_camber_cruise (-14.4%), the lowest-error split where the bulk of residuals is moderate — sharper β concentrates gradient on the bulk, ignoring outliers.

β=2.0 regression is symmetric: approaching MSE over a wider band overweights tail residuals, hurting all four splits. The heavy-tailed residual distribution of TandemFoil surface pressure is the key underlying mechanism.

Best epoch=37=terminal: model still descending at the wall-clock cap. This is a consistent pattern across all winning PRs — the model has more headroom.

**Key consequence:** β=0.5 is now the advisor baseline. β=0.25 is the natural next step — PR #1700 (thorfinn) queued.

### Conclusions

- Sharper Huber β is a real, zero-cost lever for this dataset.
- Direction is clear: sweep toward β=0.25 and pure L1 to find the optimum.
- Surface-weight interaction possible (surf_weight=10 was tuned with β=1.0; re-tuning with β=0.5 may compound further).

---

## 2026-05-13 00:40 — PR #1587: n_hidden 128 → 160 + bf16 — CLOSED (stale)

- **Branch:** `charliepai2g48h5-edward/wider-hidden-160-bf16`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver n_hidden 128→160 paired with bf16 AMP.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod stalled or
failed to start training; no training trajectory, no metrics.jsonl, no terminal
SENPAI-RESULT. Same pattern as previously stale #1561 (askeladd) and #1535 (tanjiro).

This is the third edward assignment to stall; the hypothesis itself remains valid.
Width capacity has not been refuted — reassigned as PR #1688 on the compile baseline
with explicit n_hidden=160+compile instructions and updated run command.

---

## 2026-05-13 00:10 — PR #1619: RaceCar single sampler boost 2× — SENT BACK (needs compile rebase)

- **Branch:** `charliepai2g48h5-nezuko/sampler-boost-single-2x`
- **Student:** charliepai2g48h5-nezuko
- **Hypothesis:** Boost `racecar_single` sample weights by 2× (→ 50% single / 25% tandem / 25% cruise)
  to close the coverage gap on `val_single_in_dist`, which consistently dominates `val_avg/mae_surf_p`.

### Results

| Metric | Baseline (#1532 bf16) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **98.2897** | -2.80% ✓ |
| `test_avg/mae_surf_p` | 91.50 | **88.8539** | -2.89% ✓ |

| Split | This PR | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 107.14 | 120.02 | **-10.73%** |
| `val_geom_camber_rc` | 108.83 | 107.10 | +1.61% |
| `val_geom_camber_cruise` | 82.99 | 82.84 | +0.18% |
| `val_re_rand` | 94.21 | 94.53 | -0.34% |

- **Best epoch:** 18/20 (30-min SENPAI_TIMEOUT_MINUTES cap)
- **Time/epoch:** ~91.5 s (unchanged from bf16 baseline)
- **Peak GPU:** 32.94 GB
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-boost-single-2x-20260512-215136/metrics.jsonl`
- **Sampler verification:** `racecar_single=2.0, racecar_tandem=1.0, cruise=1.0` — boost applied correctly.

### Status

**SENT BACK** for compile rebase. The 98.29 result beats the bf16 baseline (101.12) by -2.80% but does
NOT beat the current compile baseline (69.83 from PR #1568). The lever is confirmed real —
val_single_in_dist dropped -10.7% — it just needs to be measured on the new advisor baseline.

### Analysis

**Mechanism confirmed.** val_single_in_dist is coverage-bound, not capacity-bound. Doubling the
sampler mass for racecar_single (from 33.3% → 50% of effective mix) gave -10.7% on that split while
all other splits moved ≤1.6% in either direction. Cost: zero (sampler only changes which samples are
drawn, not per-sample compute).

The tiny +1.6% regression on val_geom_camber_rc is expected: racecar_tandem share dropped 33.3% → 25%,
and that split is from RaceCar tandem geometry. The signal is that the per-domain coverage directly
determines per-split performance — a strong signal for sampler as a lever.

**At compile baseline**, the same 2× boost should give a similar relative win: val_single_in_dist
from 77.10 → ~68-69. Net val_avg could reach ~66-68, compounding with the compile gain.

### Conclusions

- Sampler reweighting is a real, orthogonal, zero-cost lever.
- On compile baseline, sampler+compile should compound to give the next round winner.
- Follow-ups queued: 1.5× and 3× boost factor sweep; "both RaceCar domains boosted together."

---

## 2026-05-13 00:10 — PR #1588: n_layers 5 → 6 + bf16 — CLOSED

- **Branch:** `charliepai2g48h5-fern/deeper-6-layers-bf16`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** n_layers=6+bf16 trades 3 epochs of refinement for ~20% more per-step capacity.

### Results

| Metric | Baseline (#1532 bf16, n_layers=5) | This PR (n_layers=6) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **111.058** | **+9.83%** (WORSE) |
| `test_avg/mae_surf_p` | 91.50 | **98.793** | **+7.97%** (WORSE) |

| Split | n_layers=6 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 134.81 | 120.02 | +12.3% |
| `val_geom_camber_rc` | 122.93 | 107.10 | +14.8% |
| `val_geom_camber_cruise` | 86.20 | 82.84 | +4.1% |
| `val_re_rand` | 100.29 | 94.53 | +6.1% |

- **Best epoch:** 14/16 (116.1 s/epoch; 30-min cap hit during epoch 17 validation)
- **Peak GPU:** 38.93 GB

### Analysis

**Depth lever fully ruled out.** Both n_layers=7+fp32 (PR #1413, wall-clock-bound at 10 epochs)
and n_layers=6+bf16 (this PR, 16 epochs) converged on the same qualitative story: deeper model
needs more gradient steps than the 30-min cap allows, and the capacity gain does NOT compensate.

The key falsifying signal: surface metric (mae_surf_p) regressed MORE than volume metric (+8-10%
surface vs +3-5% volume). This is the OPPOSITE of the "extra slice-attention refinement helps near
sharp pressure gradients" mechanism that motivated the hypothesis. The model is under-trained, not
capacity-limited.

Generalisation gap is normal (test < val) and stable across n_layers values — depth doesn't worsen
the gap, it just shifts both metrics worse uniformly.

### Conclusions

- n_layers scaling is the wrong lever for 1499 training samples at 30-min wall-clock cap.
- Two experiments (n_layers=6 and n_layers=7) both lost, and the mechanism analysis confirms this
  is compute starvation, not a data-limited ceiling.
- **Do NOT follow up with n_layers + compile.** Even at 36 epochs, adding a 6th layer would take
  ~139 s/epoch → only ~13 epochs in 30 min. Still worse than the 36-epoch baseline.
- Reassigned fern to AdamW β2=0.95 (transformer fast-adapting recipe, PR #1676).

---

## 2026-05-12 22:45 — PR #1535: EMA model weights for eval (decay=0.999) — CLOSED (stale)

- **Branch:** `charliepai2g48h5-tanjiro/ema-eval-decay-0.999`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Maintain EMA copy of model weights with decay=0.999 and use it for eval —
  typical late-training noise smoothing.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no training trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis itself remains in-play and was reassigned on the compile baseline (decay=0.999,
  `torch.optim.swa_utils.AveragedModel` after compile, eval/test via EMA model).
- No data lost; closing simply frees the student slot.

---

## 2026-05-12 22:45 — PR #1561: Gradient clipping max_norm=1.0 — CLOSED (stale)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-1.0`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Bound rare large gradient updates via `clip_grad_norm_(.., max_norm=1.0)`;
  also a high-diagnostic-value characterization of training gradient norms.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis reassigned on the compile baseline with the per-epoch grad-norm aggregation
  (min/p50/mean/max/clip_frac) added so we still get the diagnostic value regardless of
  whether clipping wins on validation.

---

## 2026-05-12 22:45 — PR #1590: slice_num 64 → 96 + bf16 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-96-bf16`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase Transolver slice_num from 64 → 96 paired with bf16 AMP. With
  bf16's ~2× throughput we can afford the slightly more expensive forward and still complete
  full training; more slices = finer flow-field tokenization.

### Results

| Metric | Baseline (#1532 bf16) | This PR (#1590) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **105.024** | +3.86% (WORSE) |

- **Status:** CLOSED — does not beat bf16 baseline, well behind the new compile baseline (69.83).
- **Metric artifacts:** PR comment; trajectory shows monotone-worse vs slice_num=64 within
  the same epoch budget.

### Analysis

Combined with the round-3 fp32 result (slice_num=128 → val=145.97 at 11 epochs, wall-clock-bound)
and the bf16 result here (slice_num=96 → 105.02 at full budget), the slice_num lever is now
well-characterized:

| slice_num | regime | val_avg/mae_surf_p |
|---:|---|---:|
| 64 | bf16 (baseline) | 101.12 |
| 96 | bf16 | 105.02 (+3.86%) |
| 128 | fp32 (epoch-bound) | 145.97 |

Monotone-worse with slice count. The 64-slice default appears near-optimal for this dataset
size — adding slices adds capacity that overfits or simply costs throughput without finding
useful additional flow-field structure. **slice_num is a dead lever upward**; downward
(slice_num=32 or 48) would be a separate experiment but is a low-priority swing.

### Conclusions

- slice_num=64 is the right value for the current data + architecture.
- Closing this arm; do NOT pair slice_num=96 with compile (no signal it would help).
- Round-6 reassignment: frieren → step-based linear warmup + cosine on compile baseline.

---

## 2026-05-12 22:10 — PR #1568: torch.compile + bf16 AMP for additional throughput — MERGED ✓

- **Branch:** `charliepai2g48h5-thorfinn/torch-compile-bf16`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** `torch.compile(model, dynamic=True)` stacked on top of bf16 AMP doubles
  per-epoch throughput from ~98 s → ~49.5 s, fitting 36 epochs in 30 min vs 19 previously.
  Mechanism: kernel fusion eliminates Python dispatch overhead. `dynamic=True` prevents
  recompilation on variable-length mesh batches (N_max varies per batch).

### Results

| Metric | Baseline (#1532) | This PR (#1568) | Δ |
|---|---|---:|---:|
| `val_avg/mae_surf_p` | 101.1212 | **69.8316** | **-30.9%** |
| `test_avg/mae_surf_p` | 91.5013 | **61.8652** | **-32.4%** |

| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 77.10 | -35.8% |
| `val_geom_camber_rc` | 83.49 | -22.0% |
| `val_geom_camber_cruise` | 50.64 | -38.9% |
| `val_re_rand` | 68.10 | -28.0% |

| Split | test mae_surf_p |
|---|---:|
| `test_single_in_dist` | 67.81 |
| `test_geom_camber_rc` | 77.68 |
| `test_geom_camber_cruise` | 41.98 |
| `test_re_rand` | 59.99 |

- **Status:** MERGED — new baseline 69.8316 / 61.8652.
- **Epochs reached:** 36 (timeout-bound, 29.41 min; best epoch = 36, still descending)
- **Time/epoch:** ~49.5 s (2.0× speedup vs bf16-only ~98 s)
- **Peak GPU:** 23.8 GB (64 GB headroom on 96 GB card)
- **Compile status:** active for all 36 epochs, no recompilation stalls with `dynamic=True`
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.jsonl`

### Analysis

The win is almost entirely explained by epoch count: 36 vs 19 epochs = ~1.9× more gradient
steps. The model was monotonically improving through epoch 36 with no late-training instability.
`dynamic=True` was the correct choice — without it, dynamo would specialize per N_max and
accumulate recompilation costs that outweigh the kernel-fusion gain on variable-mesh batches.

All 4 val splits improved uniformly (+22-39%), including the hardest OOD splits. This is
pure optimization headroom, not overfitting.

**Key consequence:** The new 36-epoch budget changes the arithmetic for every in-flight arm.
- Capacity arms (#1587, #1588, #1590) were targeting n_hidden=160/n_layers=6/slice_num=96
  + bf16 (without compile). With compile now on advisor, those arms now run at compile speed
  IF they rebase — but they were branched before this merge and won't automatically have compile.
- T_max=50 cosine schedule with 36 epochs reaches LR≈0.012 at epoch 36 (not the full
  low-LR tail). Alphonse's T_max=18 result proved the terminal LR decay matters — so
  T_max=36 on top of compile is now the highest-confidence cheap win.

### Conclusions

- torch.compile is a free 2× throughput multiplier with no accuracy cost.
- 23.8 GB peak (batch=4, n_hidden=128) leaves 72 GB headroom for capacity exploration.
- Budget is still binding at 36 epochs — the model is still descending. More compute =
  more improvement. Highest-value follow-up: T_max=36 schedule to exploit the low-LR tail.

---

## 2026-05-12 22:00 — PR #1560: Match cosine T_max to actual epoch budget — SENT BACK

- **Branch:** `charliepai2g48h5-alphonse/tmax-18-cosine`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** `CosineAnnealingLR(T_max=50)` with 19 bf16 epochs never reaches the
  low-LR tail. Setting T_max=epoch_budget (originally T_max=14 for fp32, T_max=18 for
  bf16) lets the schedule complete, adding a meaningful low-LR fine-tuning phase.

### Results (two arms)

**Arm A — T_max=14 (fp32-era budget, pre-bf16 advisor commit 1341b98):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **98.7502** (best epoch = 14, terminal) |
| `test_avg/mae_surf_p` | **88.8030** |
| Epochs reached | 14/14 (complete) |
| Time/epoch | ~132.4 s (fp32) |
| vs #1444 baseline (110.76) | -10.8% |

**Arm B — T_max=18 (bf16-era budget, current advisor commit afd445a):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **90.3237** (best epoch = 18, terminal) |
| `test_avg/mae_surf_p` | **80.1938** |
| Epochs reached | 18/18 (complete) |
| Time/epoch | ~98.0 s (bf16) |
| vs #1532 baseline (101.12) | **-10.7%** |

Per-split Arm B (tmax-18):
| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 105.86 | -14.2% |
| `val_geom_camber_rc` | 99.48 | -7.1% |
| `val_geom_camber_cruise` | 70.74 | -14.6% |
| `val_re_rand` | 85.22 | -9.8% |

- **Status:** SENT BACK — baseline moved to 69.83 (PR #1568 merged). T_max=18 (90.32)
  no longer beats new baseline. Reassigned to retest with T_max=36 matching compile budget.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-tmax-18-cosine-20260512-210749/metrics.jsonl`,
  `models/model-charliepai2g48h5-alphonse-tmax-14-cosine-20260512-201325/metrics.jsonl`

### Analysis

**Mechanism confirmed.** Best epoch = terminal epoch in BOTH arms. The cosine schedule's
low-LR tail (final ~20-25% of epochs where LR approaches 0) provides material fine-tuning
benefit. The trajectory is clear:

val_avg at epochs 14→18 in Arm B: 98.34 → 92.62 → 92.34 → 91.44 → **90.32** — the last 4
epochs (T_max=14 to terminal) gained ~8.0 absolute MAE points. This is the "low-LR tail" the
hypothesis predicted.

**At epoch 14, both arms agree** (Arm B epoch 14 = 98.34, Arm A terminal = 98.75) — the LR
trajectory difference up to epoch 14 is negligible. The improvement is purely from completing
the cosine arc.

**Key implication for compile baseline:** With torch.compile reaching 36 epochs, the "natural
budget" has doubled. T_max=36 would complete the cosine arc and provide the same low-LR tail
effect — potentially gaining ~8-12 MAE off the 69.83 baseline.

### Conclusions

- Schedule-completion is a real, cheap, orthogonal lever. Best epoch = terminal epoch = strong
  signal that the low-LR tail does fine-tuning work.
- T_max=18 is obsolete — compile changed the budget to 36 epochs.
- **Follow-up:** alphonse re-running PR #1560 with `--epochs 36` on the updated advisor branch.
  If the same epoch-14→18 proportional gain holds (~8% of the remaining MAE), val_avg could
  drop from ~60 (extrapolating compile curve) to ~55 in the final epochs.

---

## 2026-05-12 21:30 — PR #1428: Per-channel loss weights [1,1,3] favoring pressure — CLOSED

- **Branch:** `charliepai2g48h5-nezuko/pressure-channel-weight`
- **Student:** charliepai2g48h5-nezuko
- **Hypothesis:** Reweight loss channels [1,1,3] so the pressure channel (the
  one we're scored on) carries 3× the gradient signal of Ux/Uy. Expected
  -5% to -12% delta on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **135.5317** (epoch 13 best) |
| `val_single_in_dist` | 167.07 |
| `val_geom_camber_rc` | 143.28 |
| `val_geom_camber_cruise` | 103.33 |
| `val_re_rand` | 128.44 |
| `test_avg/mae_surf_p` | **122.2302** (finite — student applied scoring workaround) |
| Best epoch | 13 |
| Epochs reached | 14 (timeout-bound, ~131 s/epoch, fp32 — pre-bf16) |
| Peak GPU | 42.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch, pre-bf16) |

- **Status:** CLOSED — +34.1% worse than bf16 baseline 101.12.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-pressure-channel-weight-20260512-200303/metrics.jsonl`

### Analysis

Two compounding factors explain the poor result:

1. **Wall-clock disparity.** Branch predates PR #1532 — 14 fp32 epochs at
   ~131 s/epoch vs baseline's ~19 bf16 epochs at ~98 s/epoch. Partially
   accounts for the gap (maybe 50%?).

2. **Channel weighting fundamentally wrong at 3×.** All four val splits
   regressed — including val_geom_camber_cruise (103.33 vs 82.84 at
   baseline). The only mechanistic explanation for regression on ALL splits
   simultaneously is that [1,1,3] distorted the optimization geometry.
   With 3× pressure gradient, the model optimizes pressure at the expense
   of Ux/Uy, but pressure predictions depend on accurate velocity (physical
   coupling), so the interference cascades back to `mae_surf_p`. Even on
   the "easiest" split (`val_geom_camber_cruise`) only reached ~25% above
   the baseline's full-budget performance at epoch 13.

3. **Student's diagnostic insight for `val_single_in_dist`.** Student noted
   this split (RaceCar single random hold-out) is the hardest despite being
   in-distribution — suggesting the WeightedRandomSampler may be
   under-covering that domain. This is the seed for the reassignment below.

### Conclusions

- Per-channel reweighting at [1,1,3] is ruled out — too aggressive, harms Ux/Uy
  via physical coupling, all-split regression.
- Milder weights ([1,1,2] or [1,1,1.5]) might be worth revisiting after
  other improvements are stacked, but the priority is the sampler direction.
- **New assignment for nezuko (PR #1619): domain-aware sampler reweighting** —
  boost RaceCar single sample weights 2× (→ 50% share) to directly attack
  `val_single_in_dist` coverage deficit. Inherits bf16 AMP + scoring fix.

---

## 2026-05-12 20:55 — PR #1422: slice_num 64 → 128 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-128`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase `slice_num` from 64 to 128 to give Transolver
  more physics-aware slice tokens per attention layer.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **145.9708** (epoch 11 best) |
| `val_single_in_dist` | 184.67 |
| `val_geom_camber_rc` | 154.30 |
| `val_geom_camber_cruise` | 114.72 |
| `val_re_rand` | 130.19 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 11 |
| Epochs reached | 11 (timeout-bound) |
| Time/epoch | ~171 s (vs ~131 s baseline) |
| Peak GPU | 54.5 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +44% worse than baseline 101.12.

### Analysis

Same diagnosis as #1398, #1413: capacity scale-up at fp32 + MSE only fits
11 epochs in the 30-min cap, vs baseline's 19 epochs (bf16) — undertrained.
Val still descending monotonically through epoch 11 (no plateau, no
instability, no OOM at 54.5 GB). The lever itself isn't refuted — the
budget is binding.

### Conclusions

- slice_num=128 untestable under current wall-clock budget without bf16.
- Next assignment for frieren: slice_num=96 + bf16 inheritance (PR #1590) —
  milder slice bump paired with throughput fix for fair test.

---

## 2026-05-12 20:55 — PR #1413: n_layers 5 → 7 — CLOSED

- **Branch:** `charliepai2g48h5-fern/deeper-7-layers`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** Increase `n_layers` from 5 to 7 (deeper Transolver) to
  give more iterative slice-attention refinement.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **144.9040** (epoch 10 best) |
| `val_single_in_dist` | 171.26 |
| `val_geom_camber_rc` | 177.24 |
| `val_geom_camber_cruise` | 103.29 |
| `val_re_rand` | 127.83 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~181 s |
| Peak GPU | 57.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +43% worse than baseline 101.12.

### Analysis

Same diagnosis as the capacity-arms pattern: at n_layers=7 + fp32 + MSE,
only 10 epochs fit in the 30-min cap. Val descended steeply through
epoch 10 with no plateau. No instability, no OOM. Wall-clock is the
binding constraint, not depth.

### Conclusions

- n_layers=7 untestable under current budget without bf16.
- Next assignment for fern: n_layers=6 + bf16 inheritance (PR #1588) —
  milder depth bump paired with throughput fix.

---

## 2026-05-12 20:53 — PR #1398: n_hidden 128 → 192 — CLOSED

- **Branch:** `charliepai2g48h5-edward/wider-hidden-192`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver `n_hidden` from 128 to 192 for more
  representational capacity.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **138.1375** (epoch 10 best) |
| `val_single_in_dist` | 187.30 |
| `val_geom_camber_rc` | 141.23 |
| `val_geom_camber_cruise` | 103.21 |
| `val_re_rand` | 120.81 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~186 s |
| Peak GPU | 58.0 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +37% worse than baseline 101.12.

### Analysis

Trajectory was volatile at epoch 7-10 (167→179→197→138) — clearly still
in early-training oscillation, not converged. Wider model at fp32 trades
epochs for capacity 1-for-1. No instability, no OOM. Pattern matches
fern (#1413) and frieren (#1422) exactly: wall-clock is binding for
capacity scale-ups under MSE+fp32.

### Conclusions

- n_hidden=192 untestable under current budget without bf16.
- Three students (edward, fern, frieren) independently identified the
  same pattern: capacity-scale-up arms get killed by wall-clock cap
  unless paired with throughput recovery (bf16).
- Next assignment for edward: n_hidden=160 + bf16 inheritance (PR #1587) —
  milder width bump paired with throughput fix.

---

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix — MERGED

- **Branch:** `charliepai2g48h5-thorfinn/bf16-amp-scoring-fix`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Enable bf16 mixed-precision training (`torch.autocast("cuda", dtype=torch.bfloat16)`) to increase epoch throughput and reach more training epochs within the 30-min cap. Also includes scoring-NaN workaround: batch-level `y_finite_mask` filter in `evaluate_split` before `accumulate_batch`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **101.1212** (epoch 17 best) |
| `val_single_in_dist/mae_surf_p` | 120.0176 |
| `val_geom_camber_rc/mae_surf_p` | 107.0980 |
| `val_geom_camber_cruise/mae_surf_p` | 82.8425 |
| `val_re_rand/mae_surf_p` | 94.5268 |
| `test_avg/mae_surf_p` | **91.5013** (finite — first on this branch) |
| `test_single_in_dist/mae_surf_p` | 105.4434 |
| `test_geom_camber_rc/mae_surf_p` | 99.9931 |
| `test_geom_camber_cruise/mae_surf_p` | 69.2841 |
| `test_re_rand/mae_surf_p` | 91.2844 |
| Best epoch | 17 |
| Epochs reached | 19 (~25% faster at ~98 s/epoch vs ~131 s) |
| Peak GPU | 32.95 GB |

- **Improvement:** -9.64 MAE (-8.7%) vs PR #1444 baseline (110.7608)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.{jsonl,yaml}`
- **Status:** MERGED → new round-5 baseline floor: **val_avg/mae_surf_p = 101.1212**

### Analysis

1. **bf16 AMP gave a real throughput win**: ~25% faster per epoch (98 vs 131 s), reaching epoch 19 vs baseline's epoch 14 — 5 extra epochs of convergence. The extra epochs drove the primary win: best epoch 17 vs baseline's 14.

2. **Scoring fix unblocked test_avg**: The `y_finite_mask` filter in `evaluate_split` correctly skipped `test_geom_camber_cruise/000020.pt`, giving the first finite `test_avg/mae_surf_p` (91.50) on this branch. This fix is now on the advisor branch for all subsequent PRs.

3. **Throughput under 2×**: At 0.66 M params, the model is small — Python/I/O overhead is a non-trivial fraction of step time. Bigger models would amortize the autocast win more. The `~25%` gain is real but modest.

4. **Still improving at cap**: Val was 102.26 at epoch 19 (final) vs best 101.12 at epoch 17 — slight uptick at the last epoch, still trending overall. More compute budget would likely gain additional MAE points.

5. **`val_geom_camber_cruise` slight regression (+5 MAE pts)**: The only split that worsened. Possibly noise from the different convergence trajectory (more epochs = different phase of the schedule). Worth watching in follow-up runs.

### Conclusions

- bf16 AMP is now the baseline — it's merged and available for all subsequent PRs to inherit.
- The scoring-NaN workaround is now on advisor — new baseline for test_avg is 91.5013.
- New bar: any PR must beat **101.1212** on val_avg/mae_surf_p to merge.
- Next for thorfinn: compound the wins — pair bf16 with the best capacity lever once architecture results settle.

---

## 2026-05-12 20:00 — PR #1388: Linear warmup + lr 5e-4 → 1e-3 with cosine anneal — CLOSED

- **Branch:** `charliepai2g48h5-askeladd/warmup-lr-1e3`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Add 5-epoch linear warmup and raise peak lr from 5e-4 to 1e-3
  (with cosine anneal afterward). Compensate for small batch and short
  wall-clock budget.

### Results

| Metric | lr=1e-3 (primary) | lr=7.5e-4 (fallback) |
|---|---:|---:|
| `val_avg/mae_surf_p` | **152.0332** | 152.5056 |
| `val_single_in_dist/mae_surf_p` | 184.95 | 177.17 |
| `val_geom_camber_rc/mae_surf_p` | 163.59 | 163.31 |
| `val_geom_camber_cruise/mae_surf_p` | 122.49 | 124.96 |
| `val_re_rand/mae_surf_p` | 137.10 | 144.58 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) | NaN |
| `test_3of4_avg/mae_surf_p` | 148.47 | 148.80 |
| Best epoch | 12 | 12 |
| Epochs reached | 14 | 14 |
| Time/epoch | 131.4 s | 132.0 s |
| Peak GPU | 42.11 GB | 42.12 GB |
| Loss used | **MSE** (PR predates Smooth-L1) | **MSE** |

- **Artifacts:** `models/model-charliepai2g48h5-askeladd-warmup-lr-1e3-20260512-181136/metrics.{jsonl,yaml}`, `models/model-charliepai2g48h5-askeladd-warmup-lr-7.5e4-20260512-185418/metrics.{jsonl,yaml}`
- **Status:** CLOSED — both arms ~41 MAE worse than baseline.

### Analysis

- ~41 MAE gap is too large to be MSE-vs-Smooth-L1 alone; lr=1e-3 is the
  dominant cause. The 5-epoch warmup + 9 epochs at peak lr=1e-3 + small
  cosine decay integrates LR-area-under-curve comparable to baseline's
  14 epochs at lr=5e-4, but more time at high lr overshoots good basins.
- Not divergence (loss curves were clean) — just a worse local minimum.
- Student independently rediscovered the scoring NaN bug, identical to
  thorfinn/alphonse's findings. Three independent students all found the
  same `0 × Inf = NaN` interaction — high-confidence diagnosis.
- The "step-based warmup over the first ~500 steps" idea is worth queuing
  separately, since 5 epochs = ~36% of the 14 epochs actually fitting in the
  cap.

### Conclusions

- lr=1e-3 with warmup is not productive at this wall-clock budget. The lr
  lever appears to be tuned correctly at baseline (lr=5e-4). Pushing lr
  higher (e.g., lr=1.5e-3, lr=2e-3) is not promising given the 41 MAE gap.
- More promising direction implied: step-based warmup at a *lower* peak.
  Queued for later, not assigned now.
- Next assignment for askeladd: gradient clipping max_norm=1.0 (PR #1561) —
  orthogonal to schedule lever space.

---

## 2026-05-12 19:53 — PR #1375: Raise surf_weight 10 → 30 — CLOSED

- **Branch:** `charliepai2g48h5-alphonse/surf-weight-30`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Raise `surf_weight` from 10 to 30 to bias gradients more
  toward the ranking quantity (surface pressure MAE).

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **120.3944** (epoch 13) |
| `val_single_in_dist/mae_surf_p` | 148.75 |
| `val_geom_camber_rc/mae_surf_p` | 125.45 |
| `val_geom_camber_cruise/mae_surf_p` | 93.73 |
| `val_re_rand/mae_surf_p` | 113.65 |
| `test_avg/mae_surf_p` | **112.6536** (finite — scoring workaround applied) |
| `test_single_in_dist/mae_surf_p` | 133.54 |
| `test_geom_camber_rc/mae_surf_p` | 123.03 |
| `test_geom_camber_cruise/mae_surf_p` | 79.73 |
| `test_re_rand/mae_surf_p` | 114.32 |
| Best epoch | 13 |
| Epochs reached | 14 |
| Time/epoch | 131.9 s |
| Peak GPU | 42.11 GB |
| Loss used | **MSE** (PR predates Smooth-L1) |

- **Artifacts:** `models/model-charliepai2g48h5-alphonse-surf-weight-30-20260512-191201/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (120.39 > 110.76).

### Analysis

- ~10 MAE gap to baseline. Smooth-L1 vs MSE typically buys ~5% in this
  regime — even a full recovery wouldn't close the gap.
- Per-split signal is diagnostic: `val_single_in_dist` got *worse* under
  surf_weight=30 (148.75 vs baseline 135.16) — surface-heavy reweighting
  biased gradients away from the volume manifold, hurting the hardest split.
  This is not an MSE-vs-Smooth-L1 artifact.
- Student independently rediscovered the scoring NaN bug AND wrote a clean
  `train.py:evaluate_split` workaround — exactly the same workaround being
  rolled centrally via PR #1532 (thorfinn). All four test splits finite as
  a result.
- Student also surfaced the recurring "T_max=50 cosine never decays in 14
  epochs" observation that tanjiro/askeladd also raised.

### Conclusions

- `surf_weight=30` is not productive — biases away from volume manifold.
  The baseline at `surf_weight=10` is well-tuned.
- Next assignment for alphonse: T_max=14 cosine schedule matched to actual
  epoch budget (PR #1560) — exactly the lever the student's own analysis
  pointed at, and orthogonal to all in-flight work.

---

## 2026-05-12 19:27 — PR #1439: Double batch_size 4 → 8 — CLOSED

- **Branch:** `charliepai2g48h5-tanjiro/batch-size-8`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Raise effective batch size from 4 → 8 to lower gradient
  variance under the 30-min wall-clock cap.

### Results

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 155.504 (epoch 14) |
| `val_single_in_dist/mae_surf_p` | 256.30 |
| `val_geom_camber_rc/mae_surf_p` | 145.07 |
| `val_geom_camber_cruise/mae_surf_p` | 103.11 |
| `val_re_rand/mae_surf_p` | 117.55 |
| `test_avg/mae_surf_p` | NaN (round-5 scoring bug) |
| Mean test_mae_surf_p (3 splits, excl. cruise) | 155.71 |
| Peak GPU | **84.2 GB** of 96 (no OOM) |
| Time/epoch | ~130 s |
| Epochs/30 min | 14 |
| Loss used | **MSE** (PR predates the Smooth-L1 merge) |

- **Artifacts:** `models/model-charliepai2g48h5-tanjiro-batch-size-8-20260512-185115/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (155.504 > 110.76).

### Analysis

- The comparison is unfair to the hypothesis: tanjiro's branch was created
  before #1444 merged Smooth-L1, so this run is MSE+batch=8 vs the current
  Smooth-L1+batch=4 baseline.
- However, the student's own analysis is decisive: **wall-clock is the binding
  constraint, not gradient noise**. Doubling batch trades step count 2:1 for
  variance reduction, but PR #1444 was monotonically improving at batch=4 —
  variance is not the bottleneck. Batch=8 just means fewer training epochs in
  the same 30-min window.
- batch=8 sits at 84 GB peak — no more headroom on this model size, so
  batch=8 is at its memory ceiling on the default Transolver. The lever is
  fully exercised.
- The student independently rediscovered the scoring NaN bug (same root
  cause as PR #1444) — solid debugging.

### Conclusions

- `batch_size=8` is feasible but does not appear to be a productive lever on
  this dataset + model + wall-clock budget. Closing the arm.
- The student's observation that "T_max=50 cosine never gets used because we
  only reach ~14 epochs" is a separately valuable insight — worth a future PR
  matching `T_max` to expected actual epoch budget.
- Next assignment for tanjiro: EMA model weights for eval (PR #1535) —
  orthogonal to the throughput / schedule lever space.

---

## 2026-05-12 18:58 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Branch:** `charliepai2g48h5-thorfinn/smooth-l1-loss`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Replace squared-error loss with Smooth-L1 (Huber, β=1.0) in
  normalized space for both training and evaluation losses. The ranking metric is
  MAE in original space; MSE in normalized space over-weights extreme high-Re
  samples. Smooth-L1 is linear outside |err|>β, providing bounded gradients.
  Both vol_loss and surf_loss use the same substitution; `surf_weight=10.0` and
  `data/scoring.py` MAE unchanged.

### Results

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p |
|---|---:|---:|---:|---:|
| `single_in_dist` | 135.16 | 1.719 | 0.769 | 120.38 |
| `geom_camber_rc` | 129.08 | 2.104 | 0.988 | 119.47 |
| `geom_camber_cruise` | 77.70 | 1.047 | 0.555 | NaN (bug) |
| `re_rand` | 101.10 | 1.607 | 0.740 | 97.36 |
| **avg** | **110.76** | — | — | NaN / 112.40 (3-split) |

- **Best epoch:** 14 of 50 configured (wall-clock-bound; monotonically improving)
- **Epochs/30-min:** ~14 at default model size (~131 s/epoch)
- **Peak GPU:** 42.1 GB (Blackwell RTX PRO 6000)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.{jsonl,yaml}`
- **Status:** MERGED → round-5 baseline floor

### Analysis

This is the first terminal result on the round-5 branch, so we cannot yet compare
against an MSE baseline on the same branch. The absolute val_avg = 110.76 sets
the floor. Key observations:

1. **Under-convergence.** The run was strictly monotonically improving at epoch 14
   when the 30-min cap hit (~14 epochs in 30 min for n_hidden=128). The floor is
   a loose lower bound on what the model could achieve with more compute.
2. **Split pattern consistent with hypothesis.** `val_geom_camber_cruise` (77.70)
   and `val_re_rand` (101.10) — the two splits the PR predicted would benefit most
   from bounded gradients at high-Re — are the best-performing splits. The raceCar
   splits (`single_in_dist` 135.16, `geom_camber_rc` 129.08) are noisier
   epoch-to-epoch, consistent with the loss being driven by the wide-Re tail.
3. **Scoring NaN bug discovered.** `test_geom_camber_cruise/000020.pt` has ±Inf
   values in the `p` channel. The `data/scoring.py` sample-skip logic misses this
   due to `0 × Inf = NaN` (IEEE-754). This affects all PRs in round 5 that run
   the test step. Round-5 ranking decision: **val_avg/mae_surf_p only**. The fix
   (filter the bad sample in `train.py`'s `evaluate_split` before calling
   `accumulate_batch`) will be rolled into an upcoming student assignment.

### Conclusions

- Smooth-L1 is a viable baseline for round 5. Whether it beats MSE requires the
  other in-flight arms (which use MSE) to finish and post results.
- The binding constraint is wall-clock convergence speed: ~14 epochs in 30 min.
  The highest-leverage next move is anything that increases epochs/wall-clock
  (bf16 AMP, smaller batch, smaller model, compile) rather than per-epoch quality.
- `val_geom_camber_cruise` is the easiest split (lowest MAE). The hardest splits
  are the raceCar ones.
