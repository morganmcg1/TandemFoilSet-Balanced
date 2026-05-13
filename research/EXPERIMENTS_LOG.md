# SENPAI Research Results — Charlie pai2g 24h r3

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3`.
Records of reviewed and merged experiment PRs. Add a new section under the
appropriate heading whenever an experiment terminal-completes.

---

## Round 2 — build on merged stack (ongoing)

All experiments in this round must rebase on `icml-appendix-charlie-pai2g-24h-r3` (includes grad_clip=1.0, wd=1e-3, OneCycleLR, EMA=0.999, AoA+NACA augment, Huber δ=0.5, surf_weight curriculum 1→20) before running. Current baseline: **val_avg/mae_surf_p = 97.620** (PR #1686, thorfinn curriculum) / test 91.947 (safe 4-split).

---

### 2026-05-12 19:53 — PR #1520: OneCycleLR + EMA weights (fern)
**Branch:** `charliepai2g24h3-fern/onecycle-lr-ema` | **Status: MERGED** ⭐

- **Hypothesis:** Replace CosineAnnealingLR (T_max=50, only ~5% annealed in 14 epochs) with OneCycleLR (auto-adapts to actual steps). Add EMA(0.999) weights for evaluation to eliminate checkpoint-selection jitter.
- **val_avg/mae_surf_p: 112.546** (epoch 14/14) — **−2.5% vs baseline 115.403**.
- **Per-split:** single=125.10, rc=136.04, **cruise=86.31** (best camber), re_rand=102.73.
- **Test (3-split proxy):** single=113.89, rc=118.86, re_rand=99.84 → **110.862** (−3.7% vs baseline 115.13).
- **Analysis:** Val curve was strictly monotone decreasing (epoch 14 still the best at cap). EMA eliminated the regression fern's own baseline saw at epochs 13–14 (115.40 → 119.37 → 126.42). OneCycleLR peak LR = 5e-3 reached at epoch 3; lr=4.3e-3 at cap (barely in cosine anneal phase). val_re_rand improved 6.4% (109.76 → 102.73) — EMA smoothing particularly helps the Re-generalization split. Both effects (OneCycleLR + EMA) are compounding.
- **Artifacts:** `models/model-charliepai2g24h3-fern-onecycle-ema-decay999-20260512-191518/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-13 00:25 — PR #1484 v2: Huber loss δ=0.5 on merged stack (alphonse) — **MERGED ⭐ NEW BASELINE**
**Branch:** `charliepai2g24h3-alphonse/huber-pressure-loss` | **Status: MERGED**

- **Hypothesis:** Huber loss with δ=0.5 (more aggressive than δ=1.0) clips high-Re gradient extremes; on the merged stack (grad_clip, wd, OneCycle, EMA, augment), the gradient distribution is well-conditioned enough that the round-1 "δ=0.5 hurts single_in_dist" failure mode is no longer present.
- **val_avg/mae_surf_p (Arm A, δ=0.5):** **99.879** — **−3.13% vs PR #1495 baseline (103.10)**.
- **test_avg/mae_surf_p (Arm A, safe 4-split):** **93.596** — **−1.23% vs baseline (94.757)**.
- **Per-split val (Arm A):** single=123.26, rc=118.37, cruise=69.27 (−11.2%!), re_rand=88.62.
- **Per-split test (Arm A, safe re-eval):** single=111.92, rc=101.71, cruise=75.54, re_rand=85.21.
- **Arm B (δ=1.0):** val 109.59 / test 102.40 — beats #1520 but loses to #1495; uniformly inferior to Arm A.
- **Analysis:** On the merged stack, δ=0.5 dominates δ=1.0 on **every per-split surface pressure metric including val_single_in_dist (123.3 vs 127.2)**. The round-1 fear was an artifact of the un-augmented, un-grad-clipped base. With aug + grad_clip + EMA, residual magnitudes are better behaved so more aggressive clipping consistently helps. Both arms still descending at the 14-epoch cap → ceiling is timeout-limited, not δ-limited.
- **P4 reconciliation:** Arm A used `--use_onecycle True --epochs 50` — the exact "broken" config per P4. Yet it produced the new best result. Hypothesis: Huber loss is itself a gradient-clipping mechanism, interacting beneficially with the truncated OneCycleLR anneal (it does not need the long-tail anneal to converge). The "OneCycleLR broken" claim from PR #1574 was tested under MSE+EMA, not Huber+EMA. P4 needs revisiting under different loss formulations.
- **Reusable artifact:** `safe_test_eval.py` — committed at repo root, does per-node skip of non-finite `y`. All future PRs should run it on best-val checkpoint for paper-facing 4-split test metrics.
- **Artifacts:** `models/model-huber-d0p5-onecycle-ema-20260512-225607/{metrics.jsonl,metrics.yaml,test_safe_eval.json}`, `safe_test_eval.py`.

---

### 2026-05-12 19:54 — PR #1484 v1: Huber loss delta=0.5 and delta=1.0 (alphonse) — SENT BACK
**Branch:** `charliepai2g24h3-alphonse/huber-pressure-loss` | **Status: SUPERSEDED by v2 above**

- **Hypothesis:** Huber loss (delta=1.0 in normalized space) clips high-Re gradient extremes and improves `val_avg/mae_surf_p` by 3–8%.
- **val_avg/mae_surf_p:** delta=0.5: **108.097**, delta=1.0: **108.104** (both epoch 14; tie within noise).
- **Per-split (delta=0.5 best):** single=146.2, rc=113.3, cruise=79.0, re_rand=93.8.
- **Test (3-split proxy):** delta=0.5: single=124.8, rc=98.4, re_rand=90.4 → ~104.55. delta=1.0: single=111.3, rc=105.3, re_rand=102.3 → ~106.27.
- **Analysis:** Best raw number of all round-1 runs (108.10 vs 115.40 baseline), but ran on pre-merge base (no grad_clip + wd, no OneCycleLR + EMA). Merge conflict blocked direct merge. Key concern: delta=0.5 helps cruise/re_rand but hurts single_in_dist (146.2 vs 122.6 at d=1.0) — aggressive clipping removes signal needed for high-Re single-foil. Bug analysis of cruise NaN matches tanjiro's independent trace.
- **Why sent back:** Merge conflicts (DIRTY state). Pre-merge base comparison unfair to new baseline 112.546. Instructed: rebase, run both arms (d=0.5, d=1.0) with full merged stack (OneCycleLR + EMA + clip + wd), pass criterion val_avg < 112.546.
- **Artifacts:** `models/model-huber-delta-{1p0,0p5}-20260512-*/metrics.yaml`

---

### 2026-05-13 02:30 — PR #1745: Huber δ=0.5 × curriculum 1→20 composition (thorfinn) — **MERGED ⭐ NEW BASELINE**
**Branch:** `charliepai2g24h3-thorfinn/huber-plus-curriculum-compose` | **Status: MERGED**

- **Hypothesis:** Huber δ=0.5 (from #1484) and surf_weight curriculum 1→20 (from #1686) are orthogonal mechanisms — Huber clips per-node gradient magnitude, curriculum steers surface/volume gradient share over time. Composing them in one run should be super-additive.
- **val_avg/mae_surf_p: 91.507** — **−6.27% vs #1686 baseline (97.620)**.
- **test_avg/mae_surf_p (safe 4-split): 85.611** — **−6.91% vs #1686 baseline (91.947)**.
- **Per-split val:** single=110.04 (−4.65), rc=100.44 (**−10.62** — super-additive!), cruise=71.16 (−2.83), re_rand=84.38 (−6.36).
- **Per-split test:** single=96.26, rc=88.65, cruise=77.18, re_rand=80.36.
- **Super-additivity analysis:** Huber alone had rc=118.37 (#1484), curriculum alone had rc=111.06 (#1686). Sum-of-deltas predicts ~104.5; actual = **100.44**. Non-additive gain of ~4 points on camber_rc. Mechanism: Huber stabilises per-node gradient distribution → curriculum's surf_weight ramp can steer gradients more precisely → both objectives (geometry-OOD generalisation + surface emphasis) are satisfied together, where neither alone was sufficient.
- **Bottleneck remaining:** val_single_in_dist = 110.04 (WORST split, 26+ points above avg 91.507). Architecture changes (SwiGLU, deeper model) and domain over-sampling are the active next-steps for this split.
- **Composability note (alphonse #1736):** alphonse's Huber-δ sweep (0.25, 0.1) ran on the pre-#1745 stack (before curriculum merge). Should rebase on #1745 to test whether even smaller δ further improves on the Huber+curriculum baseline.
- **Artifacts:** `models/model-huber-d0p5-curriculum-1to20-cosine14-20260513-010447/{metrics.jsonl,metrics.yaml,test_safe_eval.json}`

---

### 2026-05-13 02:20 — PR #1709: Focal per-sample loss weighting (askeladd) — **CLOSED (P3 disproved)**
**Branch:** `charliepai2g24h3-askeladd/focal-per-sample-loss-weighting` | **Status: CLOSED**

- **Hypothesis:** Per-sample focal weighting `w_i = (mse_i / mean_mse)^γ` would compound with augmentation by amplifying gradients on hard samples (mechanistic complement to P1/P2).
- **val_avg/mae_surf_p (γ=2.0 best):** 112.404 — **+9.0% regression vs #1495 baseline (103.10)**. γ=1.0 was 113.78 (+10.4%).
- **test_avg/mae_surf_p (γ=2.0):** 105.516 — **+11.3% regression vs baseline (94.757)**.
- **Per-split regression:** UNIFORM — all four splits got worse, including val_single_in_dist (+6.5-8% for both γ values). The predicted "hardest split improves" signal never appeared.
- **Focal weight diagnostics (key):** At γ=2.0, effective batch size ≈ 1.65/4 (≈41% utilisation). At γ=1.0, eff_bs ≈ 2.25/4. The focal weighting functions as a stochastic batch-size reducer at B=4.
- **Root causes (student's analysis, accepted):**
  1. **Eff_bs collapse.** B=4 is too small for focal weighting — one outlier sample captures disproportionate gradient mass. Uniform regression is the expected failure mode.
  2. **Hard samples are noisy targets, not under-fit signal.** val_single_in_dist residuals are large because y-range is large (±29k vs ±2.5k median) — amplifying their gradient amplifies label noise, not learnable signal.
  3. **Gradient-budget competition with augmentation.** surf_weight + augmentation already route gradient to hard channels; focal weighting is a third lever on the same axis.
- **New principle (P3 revised):** Focal per-sample reweighting fails at B≤4 with high-y-variance regression targets. Per-domain data curriculum (askeladd #1822) avoids eff_bs collapse by changing which samples appear in a batch (not within-batch loss weights).
- **Assigned follow-up:** askeladd PR #1822 domain over-sampling (racecar_single 2x/3x) — orthogonal approach to same single_in_dist bottleneck.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-focal-gamma{1,2}-cosine14-20260513-*/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-13 02:00 — PR #1662 v2: Fourier mesh PE — both arms PASS, **SENT BACK for v3 verification on merged stack**
**Branch:** `charliepai2g24h3-nezuko/fourier-mesh-positional-encoding` | **Status: SENT BACK v3**

- **v2 ran with the v1→v2 prescription:** Arm A = uniform L=2 + cosine T_max=14 + no EMA + augment; Arm B = surface-only L=4 + cosine T_max=14 + no EMA + augment. Both 14 epochs, both within 30-min cap.
- **Both arms PASS the original criterion** (val < 103.10 AND test < 94.76):
  - Arm A (L=2): val_avg **95.727**, test_avg **89.989** (safe 4-split).
  - Arm B (surface-only L=4): val_avg **95.598**, test_avg **89.895**.
- **Against NEW #1686 baseline (97.620 / 91.947):** Arm B beats by **−2.07% val / −2.23% test** — still a winner under the post-#1484/#1686 merged stack.
- **Per-split Arm B vs #1686:** single 111.06 (−3.16%), rc 106.97 (−3.68%), cruise 72.07 (−2.59%), re_rand 92.30 (+1.72%). Three of four splits cleanly improved.
- **Per-arm comparison:** A vs B within 0.13% val / 0.10% test — capacity-vs-location distinction does NOT load-bear at this scale. Both fixes work; A simpler, B mechanistically nicer (gates high-freq basis to surface where boundary-layer detail lives).
- **Mechanism (student write-up):** v1's failure was the OneCycle@11ep scheduler pathology (PR #1574 anneal-truncation), NOT capacity or scope. Once a fair recipe is used, ALL FOUR splits cleanly improve including the OOD geometry splits that v1 regressed on. Fourier features expose mesh-coordinate periodicity directly to the input projection so the network doesn't have to learn it from scratch via low-frequency MLP basis.
- **Why sent back NOT merged:** Two reasons.
  1. **Dirty merge state.** v2 ran on pre-#1484/#1686 stack — merge conflicts in Config/forward expected after rebase. Rebase is straightforward (orthogonal Config fields).
  2. **Composability with Huber + curriculum + EMA untested.** v2 ran with `--ema_decay 0.0` (no EMA), no Huber, no curriculum — three of four key training-loop ingredients changed since the run. We need to confirm Fourier-PE compounds with the merged stack before merging.
- **v3 instructions (single arm):** surface-only L=4 (Arm B, the recommended arm) on the FULL merged stack — Huber δ=0.5 + curriculum 1→20 + EMA=0.999 + cosine T_max=14 + augment. NO `--ema_decay 0.0` this time. Single 14-epoch run, then `safe_test_eval.py`.
- **Pass criterion (v3):** val_avg/mae_surf_p < 97.620 AND test_avg/mae_surf_p (safe 4-split) < 91.947. If v3 ≈ v2 (95-96 val), Fourier compounds; if v3 ≈ baseline (97.6 val), Fourier was a substitute for something in the merged stack and we close cleanly.
- **Artifacts (v2):** `models/model-fourier-L2-cosine14-20260513-002214/{metrics.jsonl,test_safe_eval.jsonl}`, `models/model-fourier-L4-surface-only-cosine14-20260513-010015/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-13 00:18 — PR #1662 v1→v2 send-back: Fourier mesh positional encoding (nezuko) — SENT BACK
**Branch:** `charliepai2g24h3-nezuko/fourier-mesh-positional-encoding` | **Status: SUPERSEDED by v2 entry above**

- **v1 hypothesis:** Replace raw (x, y) mesh coords with sinusoidal Fourier features γ(x) = [sin(2π·2ᵏ·x), cos(2π·2ᵏ·x)] for k=0..5 (L=6). NeRF-style positional encoding for boundary-layer / wake / stagnation high-frequency structure.
- **v1 results: val_avg/mae_surf_p = 107.19 (+3.97% vs 103.10 baseline), test_avg = 101.67 (+7.30%) — FAIL.**
- **Per-split val:** `val_single_in_dist = 118.03 vs 125.91 = −6.26% IMPROVED` ✅ — the only substantial improvement on the historically worst split we've seen. But `rc = +13.7%`, `re_rand = +7.2%`, `cruise = +2.2%` all regressed.
- **Two distinct fixable causes identified by student:**
  1. **Over-capacity at L=6** — high-freq basis absorbed into memorising training-set detail; OOD splits suffer.
  2. **Wrong spatial scope** — high-freq PE applied uniformly to bulk-flow nodes is wasted; only boundary-layer-adjacent nodes need it.
- **Schedule confound:** v1 used `--use_onecycle True --epochs 11` (broken per PR #1574). Not a fair comparison to baseline.
- **Why sent back, not closed:** val_single_in_dist −6.26% is exceptional; the failure modes are mechanistically clear; cosine T_max=14 + fixes will give a fair test.
- **v2 instructions (2 arms):**
  - Arm A: L=2 + cosine T_max=14 (lower capacity test)
  - Arm B: L=4 surface-only (gated by is_surface mask) + cosine T_max=14 (location-restriction test)
- **Pass criterion (v2):** val_avg < 103.10 AND test < 94.76 for either arm.
- **Artifacts (v1):** `models/model-fourier-pos-encode-L6-20260512-231202/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-13 00:02 — PR #1709: Focal per-sample loss weighting (askeladd) — WIP (assigned)
**Branch:** `charliepai2g24h3-askeladd/focal-per-sample-loss-weighting` | **Status: WIP**

- **Hypothesis:** Per-sample loss weighting `w_i = (mse_i / mean_mse)^γ` (Lin et al. 2017 focal style, stop-grad on weight). Amplifies gradient on hard samples — the MECHANISTIC OPPOSITE of log-cosh's saturation (which caps gradient on hard samples and conflicts with augment per PR #1543).
- **Why this should compound with augment:** Augmentation creates hard samples; focal weighting upweights them. Both interventions push the model to fit hard cases. The loss CURVE stays quadratic (no saturation), only per-sample contribution is reweighted multiplicatively.
- **Arms:** Arm A γ=1.0 (moderate); Arm B γ=2.0 (aggressive). Both cosine T_max=14, full augment stack.
- **Pass criterion:** val_avg < 103.10 AND test_avg (4-split safe re-eval) < 94.76.
- **Predicted Δ:** −2 to −5% on val_avg, with largest gain on `val_single_in_dist` (currently 125.91, the worst split — high-Re samples have largest residuals → most focal upweight).
- **Diagnostic:** log batch-wise weight.std() at epochs 7/14 — too low (<0.1) means too gentle to matter; too high (>5) means one sample dominates.
- **Artifacts:** TBD

---

### 2026-05-13 00:00 — PR #1488 v3 Arm C: surf_weight_p=20 alone (askeladd) — CLOSED (entanglement disproved)
**Branch:** `charliepai2g24h3-askeladd/decoupled-channel-heads` | **Status: CLOSED**

- **Hypothesis (Arm C isolation):** Remove head decoupling, keep per-channel surface weights [w_uv=10, w_p=20] on shared mlp2. Tests whether per-channel pressure weighting is the active ingredient in Arm B's 102.12 val win.
- **val_avg/mae_surf_p: 105.72** (epoch 14/14, cosine fully annealed) — **+2.62% over baseline 103.10**.
- **test_avg/mae_surf_p (safe 4-split):** **99.29** — **+4.53% over baseline 94.76**.
- **Per-split (Arm C vs Arm B):** single 128.48 (+3.68), rc 120.59 (+7.29), cruise 77.59 (+0.05 tie), re_rand 96.23 (+3.37). C is WORSE than B on every split — opposite of "B was helped by per-channel weighting."
- **Cleanest interpretation:** Arm B's val 102.12 was single-seed noise from the cosine schedule, NOT signal from either head decoupling or per-channel weighting. Signed test−val gap (B=−5.30, C=−6.43, #1495=−8.34) suggests val/test variance at this scale is too large to detect 1-point val improvements reliably. Neither change above noise.
- **Why closed:** All three arms (A: full stack, B: cosine+decoupled+weighted, C: cosine+weighted-only) fail pass criterion. Hypothesis exhausted.
- **Generalizable principle (logged for paper):** "Per-channel pressure weighting [10,10,20] and AoA+NACA augmentation are also substitutes." Combined with PR #1543's log-cosh finding: a class of "weight the loss + diversify the data" stacking patterns fails when both target the same channel/failure mode. Stack-compatible alternatives: per-sample reweighting (orthogonal to per-residual curve), per-channel splits where saturation is confined to where residuals are large.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-per-channel-surf-weight-cosine-v3-20260512-231009/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-13 01:00 — PR #1698: Test-time augmentation (fern) — **CLOSED (signal-perturbing TTA fails for regression)**
**Branch:** `charliepai2g24h3-fern/test-time-augmentation` | **Status: CLOSED**

- **Hypothesis:** At test time, evaluate the model on N AoA-jittered copies of each input and average the predictions. A model trained with AoA jitter (PR #1495) has been taught small AoA perturbations should yield similar flow fields — TTA cashes in that invariance at inference.
- **Arms:** Arm A (N=5, jitter=0.5°); Arm B (N=9, jitter=0.75°).
- **Baseline reproduction (Step 1):** val 103.518 vs PR #1495's 103.100 (+0.4% drift inside 1% tolerance). Test (safe re-eval) 95.437 vs #1495's 94.757 (+0.68 absolute run-to-run drift — useful noise floor).
- **Arm A (N=5):** test_avg/mae_surf_p (safe) = **95.837 (+0.40)** — fails.
- **Arm B (N=9):** test_avg/mae_surf_p (safe) = **95.728 (+0.29)** — fails.
- **Per-split:** uniform regression across all 4 splits, both arms. Not a noisy false-negative — TTA is genuinely neutral-to-mildly-harmful here.
- **Pred-std diagnostic:** model produces 10-30 m²/s² variation across N jittered passes (vs MAE 85-110 m²/s²) → confirmed model IS responsive to jitter, so averages are pointing in the wrong direction, not noop.
- **Mechanism (fern's analysis):** TTA's classification record relies on label invariance under augmentation. Here the target y(θ_AoA) MOVES with augmentation — averaging predictions of nearby AoAs pulls toward a smoothed neighborhood mean of the actual signal, which is exactly what regression-on-augmented-signal does NOT want. Training-time augmentation regularizes the model toward AoA-smooth representations (loss-shaped), but pred-time averaging blurs the signal itself (output-shaped). The two regularizations operate on different objects.
- **Universal principle (P6 — see Universal principles section):** TTA fails when augmentation perturbs the target signal, not just nuisance variables. Mechanistic dual of P1: there gradient capping defeats augmentation's hard-sample injection; here output averaging defeats the model's task-relevant input sensitivity.
- **Artifacts (preserved on branch):** `target/tta_eval.py` (generic reusable TTA wrapper — with N=1, jitter=0 functions as a safe re-eval). Not cherry-picked to advisor branch since `safe_test_eval.py` from #1484 already covers that.

---

### 2026-05-13 04:30 — PR #1822: Domain over-sampling 2×/3× racecar_single (askeladd) — **CLOSED (P9: per-domain sampling is gradient zero-sum)**
**Branch:** `charliepai2g24h3-askeladd/domain-oversample-racecar-single` | **Status: CLOSED**

- **Hypothesis:** Over-sample racecar_single domain 2× (Arm A) / 3× (Arm B) via WeightedRandomSampler to direct more gradient at val_single_in_dist (the bottleneck split). Orthogonal to focal weighting (#1709) — changes WHICH samples appear per batch rather than within-batch loss weights.
- **Pass criterion FAIL on both arms:**

| Arm | val_avg | test_avg (safe) | Δ vs #1745 val | Δ vs #1745 test |
|---|---:|---:|---:|---:|
| Baseline (#1745) | **91.507** | **85.611** | — | — |
| A (2×) | 96.794 | 91.026 | +5.29 | +5.41 |
| B (3×) | 98.001 | 93.368 | +6.49 | +7.76 |

- **The bottleneck mechanism DID work — partially.** Arm B val_single_in_dist **99.55** (−10.5 vs #1745's 110.04) and test_single_in_dist **89.33** (−6.93) — the **largest improvements ever seen on this split** in this research track.
- **Off-domain regression dominates net val_avg:**

| Split | #1745 val | Arm A val (Δ) | Arm B val (Δ) |
|---|---:|---:|---:|
| val_single_in_dist (target) | 110.04 | 106.80 (**−3.24**) ✅ | 99.55 (**−10.49**) ✅ |
| val_geom_camber_rc | 100.44 | 109.39 (+8.95) ❌ | 115.81 (+15.37) ❌ |
| val_geom_camber_cruise | 71.16 | 77.91 (+6.75) ❌ | 79.96 (+8.80) ❌ |
| val_re_rand | 84.38 | 93.08 (+8.70) ❌ | 96.69 (+12.31) ❌ |

- **Sampler verified working** via diagnostic: empirical per-domain fractions match expected (Arm A 0.500/0.250/0.250; Arm B 0.600/0.200/0.200). No diagnostic-level bug.
- **Mechanism (student's analysis, accepted):** Sampler displacement is zero-sum at the gradient level for fixed total training steps. Shifting more samples to one domain starves the others by exactly that amount. The 3 non-target domains each lose >5.8 pts of test MAE — no free generalization lunch from this manipulation alone.
- **New universal principle (P9):** **Per-domain SAMPLING-level oversampling is zero-sum at the gradient level for fixed compute.** Boosting one domain's exposure linearly trades against others' generalization. The mechanism (gradient share for racecar_single) IS active — single_in_dist improvement is monotone in factor — but off-domain regression is larger than in-domain gain at every factor tested. This rules out the entire family of "reweight-the-data-distribution" approaches as a STANDALONE fix for a single bottleneck split. Compatible with P3 (focal-weight failure): both are data-level rebalancing under fixed compute. P9 generalizes P3 to *deterministic* per-domain rebalancing.
- **Follow-up assignment:** askeladd → #1912 per-domain LOSS weighting (in-batch, λ ∈ {0.3, 1.0}). Tests whether the off-domain regression was sampler-specific (sample-exclusion failure mode) or gradient-share-fundamental (P9 generalization). If λ=1.0 ALSO regresses off-domain at matched gradient share, P9 extends from sampling to loss weighting — and the bottleneck needs an architecture/representation fix instead.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-racecar-single-oversample-{2x,3x}-20260513-*/{metrics.jsonl,test_safe_eval.json}`

---

### 2026-05-13 04:00 — PR #1827: surf_weight=30/50 sweep on #1745 stack (thorfinn) — **CLOSED (curriculum axis past optimum at sw=20)**
**Branch:** `charliepai2g24h3-thorfinn/surf-weight-sweep-30-50` | **Status: CLOSED**

- **Hypothesis:** Curriculum's surf_weight ramp wins at sw=20 (#1686 vs #1488's sw=10); push the plateau higher (30 / 50) on top of Huber+curriculum stack (#1745) to further accelerate surface convergence.
- **Both arms FAIL pass criterion** (val < 91.507 AND test < 85.611):

| Arm | sw | val_avg | test_avg (safe) | Δ vs #1745 val | Δ vs #1745 test |
|---|---:|---:|---:|---:|---:|
| Baseline (#1745) | 20 | **91.507** | **85.611** | — | — |
| A | 30 | 96.668 | 90.847 | +5.6% | +6.1% |
| B | 50 | 95.218 | 89.945 | +4.1% | +5.1% |

- **Per-split val (every split regressed):** single_in_dist 110.04→116.80/115.54 (+6.1%/+5.0%), camber_rc 100.44→107.28/104.78 (+6.8%/+4.3%), camber_cruise 71.16→73.02/72.29 (+2.6%/+1.6%), re_rand 84.38→89.57/88.27 (+6.2%/+4.6%). Predicted "single_in_dist benefits most" inverted — it regressed *more* than other splits at Arm A.
- **Volume regression (predicted, occurred):** test_avg/mae_vol_p +7.5% (Arm A) / +16.0% (Arm B). Past 5% threshold the student called out beforehand.
- **Non-monotonic curve:** Arm B (sw=50) is slightly *better* than Arm A (sw=30) — likely single-seed noise (±1 val_avg from prior runs); the safe interpretation is a flat-bottom landscape with optimum around sw=18-22.
- **Train surf_loss curves:** Both arms still descending at epoch 14 — the surface objective is *not* failing to optimize, but the surface-vs-volume gradient *balance* shifts so volume representation degrades, and surface MAE follows volume down.
- **New universal principle (P8):** **Two-stage curriculum benefits plateau at a moderate final value (~20× base).** Beyond this, gradient-balance failure dominates: surface convergence does not accelerate, volume regresses, surface MAE follows volume down. The 1→N curriculum is best understood as steering the optimizer toward a balanced surface/volume solution, NOT as a unidirectional "push surface harder" dial. Mechanistically aligned with Huber×curriculum super-additivity from #1745: Huber stabilises per-node gradient distribution, curriculum steers gradient share — they work together in a Goldilocks regime that sw>20 disrupts.
- **Follow-up assignment (incoming):** thorfinn → surf_weight_warmup_epochs sweep at fixed sw=20. The 5-epoch ramp from #1686 was arbitrary; testing 3 vs 8 epochs decouples ramp shape from plateau height.
- **Artifacts:** `models/model-surf-weight-30-cosine14-20260513-021547/{metrics.jsonl,test_safe_eval.json}`, `models/model-surf-weight-50-cosine14-20260513-031707/{metrics.jsonl,test_safe_eval.json}`

---

### 2026-05-13 03:10 — PR #1770: n_layers depth scaling 5→6/7 (fern) — **CLOSED (budget-cap constraint, not capacity)**
**Branch:** `charliepai2g24h3-fern/n-layers-depth-scaling` | **Status: CLOSED**

- **Hypothesis:** Depth scaling (n_layers 5→6/7) would improve surface-pressure MAE by giving more sequential receptive-field passes over slice-attention, especially on val_single_in_dist (sharpest pressure gradients). n=5 is at the smaller end of the Transolver paper's sweep range.
- **Arms:** Arm A (n_layers=6, +20% compute); Arm B (n_layers=7, +40% compute).
- **All other knobs at #1686 winning values:** cosine T_max=14, augment, grad_clip, wd=1e-3, surf_weight curriculum 1→20, EMA=0.999.
- **Results — both arms FAIL:**

| Arm | n_layers | sec/epoch | epochs completed | val_avg/mae_surf_p | Δ vs #1745 |
|---|---|---:|---:|---:|---:|
| Baseline (#1745) | 5 | ~130 | 14/14 | **91.507** | — |
| Arm A | 6 | ~157 | **12/14** | **103.644** | +13.3% |
| Arm B | 7 | ~181 | **10/14** | **111.923** | +22.3% |

- **Per-split test (Arm A vs #1745):** single=115.20 (+19.7%), rc=102.92 (+16.2%), cruise=82.48 (+6.9%), re_rand=90.83 (+13.0%). **The predicted "single_in_dist improves most" inverted** — it regressed most.
- **Val trajectory smoking gun:** Arm A was still descending at >5 pt/epoch when cap fired at epoch 12. Arm B LR at epoch 10 termination was 9.4e-5 — well into the steep-descent phase, not in the polish tail. Both models were under-trained relative to a 14-epoch cosine schedule.
- **Mechanism (budget cap binding):** Adding layers increases sec/epoch by ~20-40%, reducing completed epochs within the 30-min cap. cosine T_max=14 never finishes annealing → model is left in early-training LR territory → terminal epoch too high for surface precision. The quality ceiling is NOT the model's capacity — it is the schedule completeness.
- **New universal principle (P7):** Under a binding wall-clock cap with cosine T_max=N, architectural changes that increase sec/epoch (depth, wide attention, mesh resolution) trade against schedule completion. The optimal axis is one that keeps sec/epoch ≈ baseline (width, FFN gating, loss reparameterisation) rather than one that increases it linearly.
- **Follow-on assignment:** fern #1850 — slice_num sweep (96, 128). slice_num scales only the in_project_slice linear layer (tiny), keeping sec/epoch ≈ baseline. Tests attention granularity rather than depth.
- **Artifacts:** `models/model-charliepai2g24h3-fern-n-layers-6-cosine14-20260513-011742/{metrics.jsonl,test_safe_eval.json}`, `models/model-charliepai2g24h3-fern-n-layers-7-cosine14-20260513-015555/{metrics.jsonl,test_safe_eval.json}`

---

### 2026-05-12 23:33 — PR #1543 v2: Log-cosh + augment (fern) — CLOSED (substitutes, not complements)
**Branch:** `charliepai2g24h3-fern/logcosh-loss` | **Status: CLOSED**

- **Hypothesis:** Log-cosh loss (v1 won −5.21% on pre-augment stack #1520) should compound additively with AoA+NACA augmentation on the merged baseline #1495.
- **val_avg/mae_surf_p: 106.93** (epoch 14/50, cap) — **+3.71% over baseline 103.10**.
- **test_avg/mae_surf_p (safe re-eval, 4-split):** **100.61** — **+6.18% over baseline 94.76**.
- **Per-split (val):** single=127.26 (+1.07%), **rc=129.82 (+13.5% WORSE — the killer)**, cruise=75.22 (−3.55%), re_rand=95.42 (+1.35%).
- **The v2 − v1 delta is ~0** (augmentation added nothing on top of log-cosh, vs +9.4 units on MSE) — clinching evidence that the two interventions are SUBSTITUTES, not complements.
- **Mechanism (student-diagnosed):** Augmentation broadens distribution → samples have larger residuals → MSE responds with larger gradients → model is pushed to fit harder geometries. Log-cosh saturates the gradient at `tanh(r) ≈ ±1` for `|r| > 2` → augmented samples receive the *same* learning signal as easy ones → augmentation's purpose defeated. Both intervene on the same failure mode (high-magnitude pressure residuals dominating MSE gradients); they cannot stack because the second's mechanism defeats the first.
- **Generalizable principle (logged for paper):** "Loss saturation and data augmentation are not additive in this regime — they are substitutes targeting the same gradient-dominance failure mode."
- **Why closed:** Mechanism is decisive; tuning the saturation threshold won't fix the ordering conflict.
- **Student's suggestion #3 (per-channel log-cosh on pressure only) noted but not picked up:** velocity channels have residuals in 1-3 range where log-cosh ≈ MSE; so per-channel-on-pressure is functionally equivalent to full log-cosh and would face the same compositional conflict.
- **Artifacts:** `models/model-logcosh-full-stack-v2-20260512-225228/{metrics.jsonl,metrics.yaml,safe_eval.json,config.yaml}`

---

### 2026-05-13 01:15 — PR #1693 v1: SwiGLU FFN (tanjiro) — **SENT BACK (massive raw signal, needs rebase + verification)**
**Branch:** `charliepai2g24h3-tanjiro/swiglu-ffn` | **Status: WIP (v2 in progress)**

- **Hypothesis:** Replace 2-layer GELU MLP in each TransolverBlock with SwiGLU (gated linear unit, Shazeer 2020). Content-dependent feature selection via gating.
- **v1 result (extraordinary):** val_avg/mae_surf_p = **87.278** (best epoch 12/14), test_avg/mae_surf_p (safe 4-split) = **82.237**. Beats new baseline #1686 (97.62/91.95) by **−10.5% val / −10.6% test**.
- **Uniform gain:** 12-16% improvement across ALL 4 val and test splits. The largest test gains on `re_rand` (−15.8%) and `single_in_dist` (−13.9%) — the highest-Re, most numerically extreme splits. Consistent with content-dependent gating being most useful where features span the widest range.
- **Per-split val (v1):** single=106.39, rc=95.50, cruise=67.41, re_rand=79.81 (all below thorfinn's #1686 numbers).
- **Per-split test (v1, safe):** single=90.52, rc=88.08, cruise=74.72, re_rand=75.63.
- **Param count:** 827K (+7% over baseline 770K — small overhead).
- **Config:** v1 explicitly disabled EMA (`--ema_decay 0.0`); used cosine T_max=14, no Huber (predated #1484), no curriculum (predated #1686).
- **Why sent back, not merged:**
  - (1) Merge conflicts (`mergeable_state=dirty`) — branch predates #1484/#1686 stack merges; rebase needed.
  - (2) Result is large enough (15% MAE improvement from one architectural change) that one verification rerun on the merged stack is justified — distinguishes "SwiGLU is the dominant signal" from "v1's EMA-disable was load-bearing" or "v1's pre-rebase context hid an artifact."
  - (3) v2 will simultaneously answer: does SwiGLU compose with curriculum + Huber + EMA? Strongest possible test in one run.
- **v2 instructions:** rebase, run `--use_swiglu True` with the full merged-stack defaults (curriculum + Huber + EMA + augment + cosine T_max=14). Pass criterion: val < 97.62 AND test (safe 4-split) < 91.947.
- **Artifacts (v1):** `models/model-charliepai2g24h3-tanjiro-swiglu-ffn-cosine14-20260513-000123/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-12 23:20 — PR #1494 v3: FiLM on log(Re) (tanjiro) — CLOSED (regression on fair comparison)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: CLOSED**

- **Hypothesis:** Feature-wise Linear Modulation per TransolverBlock, conditioned on log(Re), gives the model an explicit route to specialize per Reynolds regime. Predicted −5 to −12% on val_avg, especially on val_re_rand split.
- **Arm A (full stack, OneCycle ep=50 + EMA + augment + FiLM):** val_avg = **118.27** (+14.7% over baseline 103.10), test_avg = **110.02**. OneCycle still at 88% of peak LR at e13 (mismatched to 13-epoch budget) — same scheduling bug as PR #1574.
- **Arm B (cosine T_max=14 + augment + FiLM — FAIR COMPARISON to #1495):** val_avg = **104.98** (+1.8% over baseline 103.10), test_avg = **98.59** (+4.0% over baseline 94.76).
- **Per-split (Arm B):** single=125.67 (tie), rc=118.70 (+3.8%), cruise=78.05 (tie), **re_rand=97.51 (+3.6% worse — opposite of predicted direction!)**.
- **Root cause (student diagnosed):** (1) `log(Re)` already at input dim 13, processed by same input MLP as coords — FiLM adds redundant route to same signal. (2) Augmentation + per-block FiLM compete on small (1499-train) dataset; FiLM gives Re-specialisation shortcut that fights augment's regularising effect. FiLM weights saturated cleanly at ~4.7 norm by e7 — conditioning IS being learned, it just doesn't add value on top of augmentation.
- **v2 vs v3 reconciliation:** v2 headline 100.99 (no augment) was dominated by cosine-fix + rebase, NOT by FiLM itself. v3 Arm B factors that out cleanly.
- **Why closed (not sent back):** Pre-registered pass criterion failed under the exact #1495 protocol with FiLM as the only delta. Hypothesis cleanly disproved.
- **Bonus:** Arm A independently confirms PR #1574 OneCycleLR scheduling bug. Student's FiLM weight-norm growth diagnostic is the cleanest write-up of mis-scheduled OneCycle.
- **Suggested follow-ups (logged, not picked up):** Drop log(Re) from input + FiLM (test info-overlap hypothesis); input-only FiLM (smaller hypothesis); wider conditioning vector (Re, AoA1, AoA2, gap, stagger).
- **Artifacts:** `models/model-film-full-stack-v3-20260512-215615/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log}}` and `models/model-film-augment-cosine-v3-20260512-223010/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log}}`

---

### 2026-05-13 00:56 — PR #1686: Two-stage surf_weight curriculum 1→20 over 5 epochs (thorfinn) — **MERGED ⭐ NEW BASELINE**
**Branch:** `charliepai2g24h3-thorfinn/two-stage-surf-weight-curriculum` | **Status: MERGED**

- **Hypothesis:** Linearly ramp `surf_weight` from 1.0 → 10.0 (Arm A) or 1.0 → 20.0 (Arm B) over the first 5 epochs, then hold. Lets the volume field converge first before surface objective dominates the loss. Standard curriculum-learning rationale.
- **Why this:** Adjacent signal from PR #1488 v2 Arm B (askeladd) suggested `surf_weight_p=20` *alone* was harmful (#1488 v3 disproved that — substitutes with augment). A *gradual* ramp avoids the early-epoch overemphasis that breaks static surf_weight=20.
- **val_avg/mae_surf_p (Arm B, surf_weight 1→20):** **97.620** — **−2.26% vs PR #1484 baseline (99.879)**.
- **test_avg/mae_surf_p (Arm B, safe 4-split):** **91.947** — **−1.65% vs #1484 (93.596)**.
- **Arm A (1→10):** val 99.48 / test 93.26 — also beats #1484 baseline but loses to Arm B.
- **Per-split val Arm B:** single=114.69 (best on this split ever!), rc=111.06, cruise=73.99, re_rand=90.74.
- **Per-split test Arm B (safe):** single=102.62, rc=98.62, cruise=79.69, re_rand=86.86.
- **Critical finding:** Biggest gain is on `val_single_in_dist` (−4.36 vs Arm A; first substantial improvement on this historically WORST split via curriculum). Second instance of a single_in_dist improvement on this branch (after nezuko's Fourier PE v1 −6.26%). The single-foil high-Re split responds to early-epoch protection of volume backbone.
- **Composability question raised:** thorfinn used **MSE loss** (predates #1484 Huber merge). Huber+curriculum stacking is UNTESTED — possible follow-up. Also schedule: cosine T_max=14 + curriculum was the proven recipe; how curriculum × OneCycleLR composes is open.
- **Volume vs surface trade:** Arm B's test_avg/mae_vol_p is 101.36 vs Arm A's 96.08. Pushing surf_weight to 20 trades off some volume MAE for surface MAE — net win on the primary metric.
- **Artifacts:** `models/model-charliepai2g24h3-thorfinn-curriculum-armB-1to20-5ep-20260512-235512/{metrics.jsonl,metrics.yaml,test_safe_eval.jsonl}`

---

### 2026-05-12 23:06 — PR #1574: augment + OneCycleLR + EMA composability (thorfinn) — CLOSED (regression + scheduling bug)
**Branch:** `charliepai2g24h3-thorfinn/augment-onecycle-ema-stack` | **Status: CLOSED**

- **Hypothesis:** Verify that augment (#1495) + OneCycleLR + EMA (#1520) compose cleanly on the merged stack, and test an EMA warmup ramp (linear 0→0.999 over first 5 epochs).
- **Arm A (full stack, EMA constant 0.999):** val_avg = **115.51** (+12.0% over baseline 103.10) — **FAIL**.
- **Arm B (full stack, EMA warmup ramp 0→0.999 over 5 ep):** val_avg = **110.74** (+7.4% over baseline) — **FAIL**.
- **Root cause (student diagnosed):** OneCycleLR with `pct_start=0.05` and `--epochs 50` reaches peak LR at step 187/3750. Because the wall-clock cap is 30 min ≈ 14 epochs of actual training, the remaining 97% of cosine anneal tail never executes. The merged stack default `use_onecycle=True` is actively hurting any 14-epoch run that uses it.
- **Implications:** (a) PR #1495 won at 103.10 only because it ran with cosine T_max=14, not OneCycleLR. (b) Any in-flight PR using `--use_onecycle True --epochs 50` is similarly stuck at peak LR. (c) Going forward, default cosine T_max=epochs unless OneCycleLR's `pct_start * epochs` matches the actual epoch budget.
- **Why closed:** Clean negative result on the specific composability question. The scheduling bug is the actionable insight; rerunning with the same args wouldn't help. EMA warmup arm could be tested separately later under a corrected schedule.
- **Artifacts:** `models/model-charliepai2g24h3-thorfinn-augment-onecycle-ema-{constant,warmup}-*/metrics.jsonl`

---

### 2026-05-12 22:56 — PR #1488 v2: Decoupled heads + surf_weight_p=20 (askeladd) — SENT BACK (entangled)
**Branch:** `charliepai2g24h3-askeladd/decoupled-channel-heads` | **Status: SENT BACK**

- **Hypothesis:** Three independent Linear heads (Ux, Uy, p) + per-channel surface weights [10, 10, 20].
- **Arm A (OneCycle + EMA + augment):** val_avg = **108.06** (epoch 14/50) — **FAIL** (+4.8% over baseline 103.10). OneCycleLR pct_start=0.05 on epochs=50 ramped to peak by ep3 and never annealed in the 14 epochs that ran.
- **Arm B (cosine T_max=14 + augment):** val_avg = **102.12** (epoch 14/14, fully annealed) — **MARGINAL VAL PASS** (-0.95% under baseline 103.10) BUT test 4-split safe re-eval = **96.82 vs baseline 94.76 (+2.18% on paper-facing metric)**.
- **Per-split (Arm B):** single=123.42, rc=131.24, cruise=78.04, re_rand=99.54.
- **Why sent back, not merged:** Val win is noise-level; test regression is meaningful. Two changes are entangled (head decoupling AND surf_weight_p=20 vs baseline uniform=10). Need isolation: run surf_weight_p=20 with shared mlp2 (no head decoupling) to identify the active ingredient. The decoupled-heads architecture itself only removes ~16K params with no inductive bias change — likely the per-channel weighting is doing the work.
- **Useful negative finding:** Arm A demonstrates OneCycleLR + EMA + augment + decoupled heads + heavy pressure weight do NOT compose well at the 14-epoch budget — relevant to thorfinn's #1574 composability test.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-decoupled-heads-{full-stack-v2-20260512-205745,augment-v2-20260512-215411}/{metrics.jsonl,test_safe_eval.jsonl}`

---

### 2026-05-12 22:17 — PR #1493 v2: slice_num 64→128 (nezuko) — CLOSED (regression)
**Branch:** `charliepai2g24h3-nezuko/more-slices-128` | **Status: CLOSED**

- **Hypothesis:** Doubling PhysicsAttention slice_num gives the model more abstract "physics tokens" for soft node-set assignment. Predicted −5 to −10% on val_avg.
- **val_avg/mae_surf_p: 121.354** (epoch 11/11, fully annealed) — **+17.70% over baseline 103.10**.
- **Per-split:** single=135.46, rc=144.30, cruise=94.89, re_rand=110.76. Uniform regression across all 4 splits.
- **Test (4-split safe re-eval):** **113.686** vs PR #1495's 94.757 → **+20.0%**.
- **Analysis:** Clean run on full merged stack (be35472: PR #1491+#1520+#1495). Conclusion: at 11-epoch budget, doubling slice_num adds ~40K params to `in_project_slice` layers (5×4×32×64 extra) that have to learn useful routing patterns from scratch — undertrained → noise. The baseline slice_num=64 is already converged for its size within 11 epochs. Transolver paper's slice-num sensitivity result was at a different architecture/budget regime where slice_num was actually bottlenecked.
- **Why closed:** >5% regression on every metric; large margin; root cause well-understood by student. Not worth a v3.
- **Credit:** Nezuko's analysis of the cruise NaN trace (boolean→0.0→Inf*0=NaN mechanics) is the canonical explanation. Used the safe re-eval pattern correctly for paper-facing test metric.
- **Artifacts:** `models/model-more-slices-128-v2-20260512-205438/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log}}`

---

### 2026-05-12 22:25 — PR #1662: Fourier mesh positional encoding (nezuko) — WIP (assigned)
**Branch:** `charliepai2g24h3-nezuko/fourier-mesh-positional-encoding` | **Status: WIP**

- **Hypothesis:** Replace raw (x, y) mesh coordinates with sinusoidal Fourier features γ(x) = [sin(2π·2ᵏ·x), cos(...)] for k=0..5. Standard NeRF-style positional encoding. Gives attention direct access to high-frequency spatial signals (boundary layers, wake structure). Predicted −3 to −8% on val_avg.
- **Expected per-split:** Largest gains on boundary-layer-dominated splits (single_in_dist, re_rand). Zero parameter cost.
- **Artifacts:** TBD

---

### 2026-05-12 20:55 — PR #1543: Log-cosh loss on merged stack v1 (fern) — SENT BACK
**Branch:** `charliepai2g24h3-fern/logcosh-loss` | **Status: SENT BACK**

- **Hypothesis:** Log-cosh loss is a smooth, threshold-free Huber alternative. Expected −3 to −8% on val_avg.
- **val_avg/mae_surf_p: 106.682** (epoch 14/14) — **−5.21% vs PR #1520 (112.55)**, but **+3.5% over current merged baseline 103.10 (PR #1495)**.
- **Per-split:** single=124.05, rc=129.81, **cruise=75.92** (best split — gradient saturation effect on high-Re), re_rand=96.95.
- **Test (4-split safe re-eval):** **100.373**.
- **Analysis:** Effect shape matches hypothesis exactly. Cruise (high-Re heavy-tail) gets the biggest gain (−12.0%), single_in_dist (mid-magnitude residuals) barely moves (−0.8%). Gradient saturation via `tanh(r)` is doing what Huber does, without the δ knob. BUT run was on `git_commit=29893da` (post-#1520, pre-#1495) — 6 min after #1495 merged. Stale base.
- **Why sent back:** Result doesn't beat current baseline 103.10. Log-cosh effect likely orthogonal to augmentation (different mechanism — loss-curvature vs data-OOD). Rebase + re-run with augmentation default ON should land near 97-100 if the effects compound. Single-arm rerun instructed.
- **Artifacts:** `models/model-logcosh-onecycle-ema-20260512-200805/{metrics.jsonl,metrics.yaml,safe_eval.json,config.yaml}`

---

### 2026-05-12 19:59 — PR #1495: AoA + NACA camber jitter v2 (thorfinn) — rebase
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: MERGED** ⭐

- **Hypothesis:** Online ±0.5° AoA jitter + ±0.002 NACA camber jitter on training inputs should improve OOD camber generalization.
- **val_avg/mae_surf_p: 103.100** (epoch 14/14) — **−8.4% vs PR #1520 baseline 112.546**.
- **Per-split:** single=125.91, rc=114.35, **cruise=77.99** (best split, exactly as hypothesized), re_rand=94.15.
- **Test (safe re-eval, 4-split):** single=105.14, rc=100.58, cruise=83.48 (199/200), re_rand=89.83 → **94.757**.
- **Test (3-split proxy):** 98.520.
- **Analysis:** Augmentation behaves as predicted on camber-OOD: cruise (M=2-4 held out) interpolates cleanly at 77.99 (best of all splits). val curve was monotone descending (244 → 195 → 183 → ... → 103). Ran with cosine T_max=14 (no OneCycleLR/EMA), so the merged baseline number 103.10 is for this specific config. Composability with OneCycleLR+EMA is untested. Thorfinn also wrote a reusable safe re-eval side script (`safe_re_eval.py`) that is now canonical for paper-facing test reporting.
- **Artifacts:** `models/model-geom-aoa-augment-r2-20260512-190924/{metrics.jsonl,metrics.yaml,test_safe_eval.{jsonl,log},safe_re_eval.py}`

---

### 2026-05-12 20:02 — PR #1494: FiLM conditioning v2 (tanjiro) — SENT BACK (best raw result)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: SENT BACK**

- **Hypothesis:** FiLM (γ·h + β) per TransolverBlock conditioned on log(Re) for cross-regime generalization.
- **val_avg/mae_surf_p: 100.987** (epoch 14/14) — best raw number this track has produced. 12.6% better than #1520, 2.1% better than current #1495 baseline.
- **Per-split:** single=122.17, rc=112.23, cruise=76.64, **re_rand=92.90** (best split, exactly as hypothesized).
- **Test (safe re-eval):** single=108.58, rc=99.88, cruise=63.97 (199/200), re_rand=88.70 → **90.281**.
- **FiLM diagnostics:** Block0 weight norm 0 → 4.38, block4 0 → 2.09 over 14 epochs. Conditioning is being learned (monotonic growth). Earlier blocks acquire larger modulation. Train grad_clip_fire_rate stayed at 1.0 (max_norm=1.0 actively binding).
- **Why sent back:** mergeStateStatus=DIRTY/CONFLICTING. Branch rebased on c7f371c (pre-#1520); missing OneCycleLR+EMA (#1520) AND augmentation (#1495). Need v3 rebase onto post-#1495 base + run two arms: (A) full stack including augmentation+OneCycleLR+EMA, (B) FiLM + augmentation cosine T_max=14 matching #1495 setup.
- **Artifacts:** `models/model-re-film-conditioning-v2-20260512-191851/{metrics.jsonl,metrics.yaml,test_safe_eval.log}`

---

### 2026-05-12 19:58 — PR #1488: Decoupled per-channel heads + surf_weight_p=20 (askeladd)
**Branch:** `charliepai2g24h3-askeladd/decoupled-channel-heads` | **Status: SENT BACK**

- **Hypothesis:** Replace shared mlp2 with three independent linear heads (Ux, Uy, p) + per-channel surface weights `[10, 10, 20]` to amplify pressure gradient signal.
- **val_avg/mae_surf_p: 132.340** (epoch 13/14) — 28% WORSE than current baseline 103.10.
- **Per-split:** single=158.49, rc=152.18, cruise=99.93, re_rand=118.75.
- **Test (3-split proxy):** ~119.81. Cruise NaN per usual.
- **Analysis:** Ran on pre-#1491 base (no grad_clip+wd, no OneCycleLR+EMA, no augmentation). Cosine T_max=50 mismatch caused noisy val curve (oscillating around minimum: 202 → 152 → 132 → 163). Architecture itself (decoupled heads, −16K params from removing mlp2) is fine; the bad result is the missing optimization stack + scheduler mismatch, not the head decoupling. Askeladd correctly identified the literal /3 normalization in instructions would have given effective `[3.33, 3.33, 6.67]` weights and made the right judgment call to drop /3 — implementation was `[10, 10, 20]` as intended. Also included a non-finite-y pre-filter in evaluate_split (parallel to thorfinn's safe re-eval).
- **Why sent back:** Need rebase + re-run with full merged stack (two arms: A=full stack, B=cosine T_max=14 matching #1495). Behavior of decoupled heads cannot be assessed until the optimization stack is matched.
- **Artifacts:** `models/model-charliepai2g24h3-askeladd-decoupled-heads-surf-p20-20260512-190233/{metrics.jsonl,metrics.yaml}`

---

## Round 1 — broad coverage (assigned 2026-05-12)

Hypotheses sourced from `/research/RESEARCH_IDEAS_2026-05-12_18:00.md`.

**Cross-round findings (apply to all round 1 results):**
- All 5 reviewed runs hit the 30-min timeout. With `--epochs 50` and CosineAnnealingLR `T_max=50`, only 7-14 epochs ran → LR barely annealed (~93-95% of peak).
- `test_geom_camber_cruise/mae_surf_p` is NaN for all runs. Root cause traced by tanjiro (PR #1494): `splits_v2/.test_geom_camber_cruise_gt/000020.pt` contains 761 `+Inf` values in `y[:, 2]`. In `data/scoring.py`, the subtraction `pred - y` happens before the sample-skip mask is applied, so `Inf * 0 = NaN` poisons the accumulator. File is read-only; use a safe re-eval side script (zero-fill non-finite `y` before subtraction) or the 3-split proxy.
- grad_clip=1.0 fires on 100% of training batches (real norms 41-115). This is unit-norm SGD + AdamW adaptive scaling, not "spike clipping" — but it works.

---

### 2026-05-12 18:56 — PR #1491: Gradient clipping + weight_decay tuned (fern)
**Branch:** `charliepai2g24h3-fern/grad-clip-adamw-tuned` | **Status: MERGED** ⭐

- **Hypothesis:** grad_clip=1.0 + weight_decay 1e-4→1e-3 would stabilize training on high-Re outliers.
- **val_avg/mae_surf_p: 115.403** (epoch 12/14)
- **Per-split:** single=133.09, rc=129.76, **cruise=88.99** (best), re_rand=109.76
- **Test (3-split proxy):** single=116.98, rc=119.26, re_rand=109.15 → proxy avg ~115.1
- **Analysis:** Clipping fired on 100% of batches (norms 41-115). Produces the smoothest val trajectory of round 1 (249 → 115 over 12 epochs, nearly monotone). The wd=1e-3 + clip combination outperforms all other round-1 variants. This is the new baseline.
- **Artifacts:** `models/model-grad-clip-wd1e-3-20260512-181000/metrics.jsonl`

---

### 2026-05-12 18:56 — PR #1495: AoA + NACA camber jitter augmentation (thorfinn)
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: SENT BACK**

- **Hypothesis:** Online jitter of AoA (±0.5°) and NACA camber (±0.002) improves OOD camber splits.
- **val_avg/mae_surf_p: 129.694** (epoch 12/14)
- **Per-split:** single=155.30, rc=141.33, **cruise=102.93** (OOD best), re_rand=119.22
- **Test (3-split proxy):** single=139.75, rc=129.04, re_rand=122.90 → ~130.56
- **Analysis:** 12% worse than fern's run, but same epoch budget. Camber OOD splits are NOT the worst — single-foil in-dist (155.3) is worst, possibly because extreme high-Re raceCar pressures dominate. Cannot isolate augmentation effect without equal-budget no-aug control. Cosine T_max mismatch (same as all round 1). Sent back: rebase on #1491 baseline + fix T_max.
- **Artifacts:** `models/model-geom-aoa-augment-20260512-181104/metrics.jsonl`

---

### 2026-05-12 18:53 — PR #1492: mlp_ratio 2→4 wider FFN (frieren)
**Branch:** `charliepai2g24h3-frieren/mlp-ratio-4-wider-ffn` | **Status: SENT BACK**

- **Hypothesis:** Restoring mlp_ratio to the paper's default (4) improves FFN capacity.
- **val_avg/mae_surf_p: 144.334** (epoch 11/13)
- **Per-split:** single=183.46, rc=153.62, cruise=105.23, re_rand=135.03
- **Test (3-split proxy):** single=155.33, rc=139.70, re_rand=125.49 → ~140.18
- **Analysis:** 25% worse than fern's run. Same 30-min timeout issue. mlp_ratio=4 is ~21% slower per epoch, so fewer epochs completed. Without proper cosine annealing the comparison is unfair. Sent back: rebase on #1491 + set --epochs 12 to match actual budget.
- **Artifacts:** `models/model-mlp-ratio-4-20260512-180817/metrics.jsonl`

---

### 2026-05-12 19:09 — PR #1494: FiLM conditioning on log(Re) (tanjiro)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: SENT BACK**

- **Hypothesis:** Inject FiLM (γ·h + β) per TransolverBlock conditioned on log(Re); should help cross-Re generalization (val_re_rand).
- **val_avg/mae_surf_p: 129.94** (epoch 12/14) — 12.6% worse than #1491 baseline.
- **Per-split:** single=156.91, rc=140.57, cruise=106.23, **re_rand=116.04 (best)**.
- **Test (safe re-eval):** single=138.96, rc=123.33, cruise=90.24 (199/200 samples), re_rand=120.40 → **test_avg=118.23**.
- **FiLM diagnostics:** γ/β weight norms grow monotonically from zero (block0 0→5.97, block4 0→3.01 over 12 epochs). Conditioning IS being learned. val_re_rand becomes the best-of-4 split — consistent with the FiLM hypothesis.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch. Need rebase + --epochs 14 re-run.
- **Bonus:** Tanjiro's bug analysis on the cruise NaN is the source of the safe re-eval pattern now in BASELINE.md.
- **Artifacts:** `models/model-re-film-conditioning-20260512-182128/{metrics.jsonl,metrics.yaml,test_safe_eval.log}`

---

### 2026-05-12 19:22 — PR #1493: PhysicsAttention slice_num 64→128 (nezuko)
**Branch:** `charliepai2g24h3-nezuko/more-slices-128` | **Status: SENT BACK**

- **Hypothesis:** Doubling slice_num gives PhysicsAttention more token capacity to represent surface vs. volume regions in 74-242K-node meshes.
- **val_avg/mae_surf_p: 138.317** (epoch 10/11) — 19.9% worse than #1491 baseline.
- **Per-split:** single=175.88, rc=147.04, cruise=108.51, re_rand=121.83.
- **Test (3-split proxy):** single=146.80, rc=135.49, re_rand=123.74 → ~135.01.
- **Memory:** Peak 54.5 / 96 GB — slice_num=128 is cheap. Room for slice_num=192 or 256 in a later round.
- **Cruise NaN trace:** Independently identified the same 761-Inf bug as tanjiro (PR #1494). Clearest write-up of the boolean→float cast mechanics. Credited alongside tanjiro.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch (11 epochs only). Need rebase on #1491 + --epochs 11 re-run.
- **Artifacts:** `models/model-more-slices-128-20260512-180855/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-12 18:53 — PR #1490: Scale model n_hidden=256, n_head=8 (edward)
**Branch:** `charliepai2g24h3-edward/scale-model-256` | **Status: SENT BACK**

- **Hypothesis:** n_hidden 128→256, n_head 4→8 (~2.54M params) improves capacity.
- **val_avg/mae_surf_p: 172.262** (epoch 6/7; 30-min cap after 7 epochs)
- **Per-split:** single=199.15, rc=194.97, cruise=131.46, re_rand=163.47
- **Test:** NaN overall; 3-split proxy: single=191.41, rc=186.78, re_rand=159.10 → ~179.09
- **Analysis:** Severely under-budgeted — ~260 s/epoch means only 7 epochs in 30 min. Model trending down (172 → 176 at epoch 7) but far from converged. Also: model is 2.54M not the predicted ~6M (mlp_ratio=2 was not changed). OOM risk (83.9GB peak). Sent back: scale down to n_hidden=192 + set --epochs 10 + rebase on #1491.
- **Artifacts:** `models/model-scale-model-256-20260512-180850/metrics.jsonl`
