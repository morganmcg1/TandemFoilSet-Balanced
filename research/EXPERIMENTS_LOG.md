# SENPAI Research Results — `willow-pai2i-48h-r4`

## 2026-05-17 00:35 — PR #4129 CLOSED + #4238 assigned

### #4129 askeladd AdamW beta2 sweep (0.95, 0.98) at nh=176+bf16+ep18 — **CLOSED** (hypothesis rejected)

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/adamw-beta2-sweep`)
- **Hypothesis:** Lower beta2 (0.95 or 0.98 vs default 0.999) increases optimizer responsiveness in small-batch / short-training regimes, improving OOD convergence.

#### Results

| Metric | Old baseline #4082 | Arm B: beta2=0.98 | Arm A: beta2=0.95 |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 50.9008 | 51.2010 (+0.59%) | 51.9042 (+1.97%) |
| test_avg/mae_surf_p | 43.8989 | 44.4595 (+1.28%) | 44.5655 (+1.52%) |
| geom_camber_cruise | 28.27 | 29.1355 (+3.07%) | 28.9069 (+2.26%) |

- **W&B runs:** `zkzw00of` (beta2=0.98), `x6wye37p` (beta2=0.95)
- **Note vs NEW baseline #4106:** gap is even larger — Arm B val=51.20 vs new baseline val=48.84 (+4.8%)

#### Decision: CLOSE
- Both arms regress on val AND all per-split test metrics.
- Monotonic ordering: baseline (0.999) < Arm B (0.98) < Arm A (0.95) — moving away from 0.999 strictly worsens, in both directions.
- Axis confirmed exhausted: beta2<0.999 closed. Upper side (>0.999) also likely not beneficial per ordering.
- **Mechanistic insight:** 1499 train samples × ~27k optimizer steps — second-moment EMA at beta2=0.999 has effective 1000-step averaging window, well-filled for a 6750-step trajectory. "Small-batch regime" intuition for lower beta2 doesn't apply here.
- **Student-identified geom_camber_cruise vulnerability:** consistently most-penalized by optimizer noise across all beta2 values. Flagged for future OOD stability analyses.

#### Follow-up: #4238 askeladd beta1 sweep (0.85, 0.95) at nh=192+bf16+ep20

---

## 2026-05-17 00:20 — PR #4106 MERGED (new baseline) + #4190 CLOSED + #4232/#4233 assigned

### #4106 fern Push wider: n_hidden=192 + bf16 + ep20 retest — **MERGED** (new baseline val=48.84/test=42.59)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/fern-nhidden192-bf16-ep18`)
- **Hypothesis (v2 retest):** n_hidden=192 was compute-starved at ep18 (val=50.92 borderline in v1). Push to ep20 to give the wider model fair cosine completion.

#### Results (vs prior baseline #4082)

| Metric | New baseline #4106 (nh=192, ep20) | Prior #4082 (nh=176, ep18) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **48.8400** | 50.9008 | **−4.05%** |
| **test_avg/mae_surf_p** | **42.5895** | 43.8989 | **−2.98%** |
| best_epoch | 20/20 | 18/18 | — |
| total_train_minutes | 43.6 | 39.0 | +4.6 |
| Peak GPU memory | 47.6 GB | 44.6 GB | +3.0 GB |
| Params | 1.47M | 1.23M | +18% |

| Split | New #4106 | Prior #4082 | Δ |
|---|---:|---:|---:|
| single_in_dist | 46.4089 | 48.97 | **−5.24%** |
| geom_camber_rc | 55.5071 | 55.45 | +0.10% (flat) |
| geom_camber_cruise | 27.1443 | 28.27 | **−3.99%** |
| re_rand | 41.2976 | 42.91 | **−3.76%** |

- **W&B run:** `or5uq1id` (group `willow-r8-width-push`)
- **Val trajectory last 5 ep:** ep16=54.92, ep17=55.18, ep18=50.67, ep19=49.32, ep20=48.84. Δ ep19→ep20=−0.97% (vs ep18→ep19=−2.67%) — **curve still descending but decelerating** at cut.

#### Decision per criteria
- **Merge** if val < 50.9008 AND test < 43.8989 → ✅ both met (val −4.05%, test −2.98%)
- **Send back** if 50.9 ≤ val < 51.5 → N/A
- **Close** if val ≥ 51.5 → N/A

#### Key mechanistic findings (from student writeup + advisor review)

1. **Compound width+epochs win confirmed.** Going from nh=176/ep18 → nh=192/ep20 wins on every primary metric. Width frontier still unsaturated.
2. **Mild-overfitting hypothesis from v1 (ep18) refuted.** Every OOD split that regressed at ep18 now improves or holds flat at ep20: rc +0.19% → +0.10% (flat), cruise +1.85% → −3.99%, re_rand +1.44% → −3.76%. The wider model was *compute-starved*, not overfitting — exactly as the still-descending val curve suggested.
3. **geom_camber_rc is the structural hard split** (~55 across all variants: nh=176/ep18=55.45, nh=192/ep18=55.55, nh=192/ep20=55.51). Width and epochs don't move it. Moving this split requires something other than width/budget — likely architectural or feature-engineering.
4. **Curve decelerating at ep20** (final-epoch Δ shrunk from −2.67% to −0.97%) — the wider model is nearly converged. Marginal returns past ep20 likely small.
5. **VRAM headroom remains:** 47.6 GB peak with 96 GB available → ~50 GB margin. Width can grow more without memory constraint.

#### New training recipe (post-#4106)

```bash
cd "target/" && SENPAI_TIMEOUT_MINUTES=50 python train.py --n_hidden 192 --epochs 20 --use_bf16
```

#### Round-9 backlog additions
- **n_hidden=208 + ep18 at fern's 50-min cap** (assigned as #4232 — already in flight)
- **Investigating geom_camber_rc directly** — dedicated PR for shape/curvature features or augmentation; needs custom hypothesis
- Stochastic depth retest **at width=192** (closed at nh=160; might unlock at higher capacity)
- n_layers=6 retest **at nh=192** (closed at nh=176; might fit if compute frontier holds)

### #4190 tanjiro n_hidden=144 (capacity-vs-epochs at 30-min budget) — **CLOSED** (val=57.05 +12.1% regress)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/capacity-vs-epochs-nh144`)
- **Hypothesis:** At 30-min budget, smaller model (nh=144, ~0.84M params) completes more of the cosine schedule than nh=176 cut at ep14. Capacity-vs-schedule-completion tradeoff.

#### Results

| Metric | This run (nh=144) | Baseline #4082 (nh=176) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **57.0497** | 50.9008 | **+6.15 (+12.1%)** |
| test_avg/mae_surf_p | **49.5109** | 43.8989 | **+5.61 (+12.8%)** |
| single_in_dist | 54.4835 | 48.97 | +5.51 |
| geom_camber_rc | 61.3161 | 55.45 | +5.87 |
| geom_camber_cruise | 34.5220 | 28.27 | +6.25 |
| re_rand | 47.7219 | 42.91 | +4.81 |
- **W&B run:** `6qzim3wc`
- **Cut at ep16/18** (per-epoch throughput 116.9 s/ep vs predicted 95-105; total 31.18 min over 30-min cap by 1.2 min)
- **Peak VRAM:** 38.9 GB (13% reduction vs nh=176) — memory unused as compute is the binding constraint

#### Decision per criteria
- **Close** if val > 53.5 → ✅ val=57.05 ≫ 53.5

#### Key strategic finding

**Capacity dominates schedule completion at 30-min budget.** Even at 87.5% cosine completion (lr_factor=0.038 at cut), val descent slowed to −0.5/ep — the remaining 2 epochs would have shaved another ~1-2 points, landing val ~55-56 — still well above 50.90 baseline. **Future 30-min budget experiments should default to nh≥176.** Tanjiro's pod role pivots to "screening student" — bold-idea viability tests at nh=176+ep14 with cut at ep13.

### #4232 fern Push width frontier: n_hidden=208 + bf16 + ep18 — **ASSIGNED**

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/nh208-push-width-frontier`)
- **Hypothesis:** Width scaling unsaturated at nh=192 (per #4106's still-descending val curve). Push n_hidden=208 (+8% params over 192) at ep18 — fits 50-min cap (~46 min projected). If wins, width frontier continues; if regresses, capacity has peaked at nh=192 for this depth.
- **Run:**
  ```bash
  cd "target/" && python train.py \
    --n_hidden 208 --use_bf16 --epochs 18 \
    --wandb_group willow-r8-width-frontier \
    --wandb_name fern-nh208-bf16-ep18
  ```
- **Decision criteria:** Merge if val<48.84 AND test<42.59; send back if val<50.5 (improves over #4082 but not #4106); close if val≥51.5.

### #4233 tanjiro AGC (Adaptive Gradient Clipping) screening — **ASSIGNED**

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/agc-screening-test`)
- **Hypothesis:** Replace global `grad_clip=1.0` with per-parameter adaptive clipping (Brock et al., NFNets ICML 2021). AGC clips each tensor's gradient relative to its parameter L2 norm — large parameters get larger gradient headroom, small parameters get tighter clipping. Particularly relevant for bf16 training where gradient magnitudes vary across layers. Orthogonal to all in-flight regularization/optimizer work.
- **Why screening at smaller config:** Tanjiro's 30-min cap can't fit nh=192+ep20 (~44 min) or nh=176+ep18 (~39 min). Test at nh=176+ep14 (cuts at ep13 in budget). If signal positive, promote to longer-budget student for full-schedule comparison.
- **Run:**
  ```bash
  cd "target/" && python train.py \
    --n_hidden 176 --use_bf16 --epochs 14 \
    --use_agc --agc_clip_factor 0.01 \
    --wandb_group willow-r8-agc \
    --wandb_name tanjiro-agc-nh176-bf16-ep14
  ```
- **Decision criteria (screening):** Promote-worthy if val at ep13 < 65 (≥3% below #4082's ep13 trajectory ~67); close if val at ep13 ≥ 70 OR diverges in first 3 epochs.

---

## 2026-05-16 23:55 — PR #4187 CLOSED + #4227 assigned — frieren pmag-weight closes; pivot to AdaBelief optimizer

### #4187 frieren Pressure-magnitude weighted L1 — **CLOSED** (val=53.17 +4.5%, all OOD splits regress)

- **Student:** willowpai2i48h4-frieren (branch: `willowpai2i48h4-frieren/pressure-magnitude-weight`)
- **Hypothesis:** Reweight L1 loss to put 1.82× weight on top-decile |p_true| nodes (per-batch quantile q=0.90, alpha=1.0), mean-normalized so total gradient mass is preserved. Tests whether the model under-fits high-|p| surface points.

#### Results

| Metric | Baseline (#4082) | This arm (pmag q90 α1) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **50.9008** | **53.1674** | **+2.27 (+4.5%)** |
| test_avg/mae_surf_p | **43.8989** | **45.2689** | **+1.37 (+3.1%)** |
| best_epoch | 18/18 | 18/18 | — |
| total_train_minutes | ~39.0 | 39.4 | +0.4 |

| Split | Baseline | pmag q90 α1 | Δ |
|---|---:|---:|---:|
| single_in_dist | 48.97 | 48.72 | **−0.25** |
| geom_camber_rc | 55.45 | 56.73 | +1.27 |
| geom_camber_cruise | 28.27 | 30.90 | **+2.63** |
| re_rand | 42.91 | 44.73 | +1.82 |

- **W&B run:** `w42jjrgd` (group `willow-r8-pmag-weight`)
- **Weighting diagnostics verified active:** `pmag_hi_frac=0.1000` (exactly 10% by design), `pmag_weight_ratio=1.818` (theoretical max for q=0.90/α=1.0 = 2/1.1 = 1.818). Loss was applied as designed.
- **Val trajectory:** ep01=223.3, ep18=53.17, monotonically descending, Δ ep17→ep18=−0.94 (similar to baseline's curve still descending).

#### Decision per criteria
- **Merge** if val < 50.9008 AND test < 43.8989 → ❌ both miss
- **Send back** if val < 52.5 (mild signal worth pmag_weight_alpha=0.5 retest) → ❌ val=53.17 ≥ 52.5
- **Close** if val ≥ 52.5 → ✅ pmag-weight closes for this baseline

#### Mechanistic interpretation (student's analysis, with advisor confirmation)

1. **Mean-normalization down-weights 90% of surface nodes.** With α=1.0/q=0.90, bottom-90% gets effective weight 1/1.1 ≈ 0.91 (9% gradient cut), top-10% gets 2/1.1 ≈ 1.82 (82% boost). The vast majority of points lose gradient — but they were already information-bearing. Model loses broad surface fitting capacity in exchange for emphasis it didn't need.

2. **The "underfit-the-tail" premise doesn't hold at this baseline.** Earlier curvature-weight close (#4042/#4110) was attributed to a *proxy* issue, but this test goes after the *target* directly — and still loses. Combined with prior negative results, the operative conclusion is **the baseline's residual error is not concentrated at the high-|p| tail — it's distributed across surface nodes**.

3. **OOD splits suffer most.** `geom_camber_cruise` (smallest |p| range across splits) regresses MOST (+2.63). Reweighting toward per-batch top-decile in training distribution does not transfer to OOD geometry/Re regimes — the "important" nodes in train batches aren't aligned with what matters in held-out splits. **Deeper insight: the loss-shape axis assumes a static notion of 'important node' that doesn't generalize across geometric distribution shift.**

#### Closed axes (do not retry on this stack)
- Surface loss reweighting by **target magnitude** (this PR)
- Surface loss reweighting by **DSDF-derived proxy** (#4110)
- Together: **both surface-loss-reweighting axes (target-based + proxy-based) are now closed for this baseline.**

#### Suggested follow-ups for round-9 (deferred)
- Gradient-magnitude weighted L1 — focus on prediction-error magnitude not target magnitude (orthogonal mechanism)
- Channel reweighting (Ux/Uy vs p surface loss balance) — current surf_weight=10 was tuned pre-bf16+nh176
- Per-node uncertainty-weighted L1 with learned per-vertex weights (high complexity, in principle better-aligned)

### #4227 frieren AdaBelief optimizer — **ASSIGNED** (replace AdamW with AdaBelief)

- **Student:** willowpai2i48h4-frieren (branch: `willowpai2i48h4-frieren/adabelief-optimizer`)
- **Hypothesis:** Swap AdamW → AdaBelief (Zhuang et al., NeurIPS 2020). AdaBelief replaces AdamW's second-moment EMA of `g²` with EMA of `(g - m)²` — variance of gradient *relative to its EMA mean*. Large steps when gradient consistent (g ≈ m, high "belief"), small steps when gradients erratic.
- **Why this could help:**
  1. AdaBelief's published wins are strongest on regression/small-batch supervised tasks — fits batch=4 pressure surface MAE regression.
  2. Orthogonal to in-flight regularization (edward RMSNorm, nezuko WD+eta_min, askeladd AdamW beta2, thorfinn EMA). Structurally different optimizer, not hyperparameter variant. Should compound with any regularization win.
  3. Surface-loss closures (#4042+#4110+#4187) point to *distributed* residual error — AdaBelief's per-param adaptive step based on gradient *consistency* is structurally better-fit than AdamW's magnitude-based scaling.
- **Run:**
  ```bash
  cd "target/" && python train.py \
    --n_hidden 176 --use_bf16 --epochs 18 \
    --optimizer adabelief \
    --wandb_group willow-r8-adabelief \
    --wandb_name frieren-adabelief-nh176-bf16-ep18
  ```
- **Decision criteria:** Merge if val < 50.9008 AND test < 43.8989; send back if 50.9 ≤ val < 51.5 (lr retune); close if val ≥ 53.

---

## 2026-05-16 23:15 — PR #4039 CLOSED + #4205 assigned — edward multi-scale PE doesn't stack with width; pivot to RMSNorm

### #4039 edward Multi-scale Fourier PE stacked retest — **CLOSED** (val=63.29 +24.3%, every split regress; width absorbs PE benefit)

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/edward-multi-scale-fourier-pe`)
- **Hypothesis (v2):** Multi-scale Fourier PE (num_freq=8, freq_min_exp=-2, freq_max_exp=5) — Arm B winner from n_hidden=160 baseline — should stack with n_hidden=176 width since PE operates on input embeddings and width on hidden dim (orthogonal axes).

#### Results

| Setup | val_avg | test_avg | epochs | best_ep | runtime |
|---|---:|---:|---:|---:|---:|
| Baseline #4082 (nf=4, nh=176, bf16, ep18) | **50.9008** | **43.8989** | 18/18 | 18 | 2361 s (~39.4 min, 45-min cap) |
| **#4039(v2): nf=8 wide + nh=176** | **63.2904** | **55.5211** | **14/18 (cut)** | **14** | **1843 s (~30.7 min, 30-min cap)** |
| Δ vs baseline | **+24.3%** | **+26.5%** | | | |

| Split | Baseline | #4039(v2) | Δ |
|---|---:|---:|---:|
| single_in_dist | 48.97 | 70.98 | **+45.0%** |
| geom_camber_rc | 55.45 | 69.46 | **+25.3%** |
| geom_camber_cruise | 28.27 | 31.21 | +10.4% |
| re_rand | 42.91 | 50.44 | **+17.6%** |

- **W&B runs:** `fieqyozr` (#4039 v2 stack)
- **Prior v1 W&B runs (n_hidden=160 baseline, Arm B winner):** `t1seach2` (nf=8 wide, val=51.47), `t9mewkgq` (nf=8 default), `5k9mu16u` (nf=6 default)
- **Schedule cut analysis:** ep14 trajectory was already +3.6% behind baseline ep14 (val=63.29 vs baseline ep14=61.07). Extrapolating cosine decay over the missing ep15→ep18 (baseline drops 10.2 units) → run would land ~53.1 val at ep18, still +4.3% worse than baseline. **Gap is not a budget artifact, it's the arm under-performing at every matched epoch.**

#### Decision per criteria
- **Merge** if val < 50.9008 AND test < 43.8989 → ❌ both miss by wide margin
- **Send back** if val < 51.4683 but doesn't beat #4082 → ❌ val=63.29 >> 51.47
- **Close** if val ≥ 51.4683 → ✅ multi-scale PE does NOT stack with width

#### Mechanistic interpretation (excellent student writeup)

The hypothesis predicted multi-scale PE and width were orthogonal — fine PE provides finer input features, width provides hidden expressivity. **They are not orthogonal here.** Mechanism:

- At n_hidden=160 (prior result), the model was *input-bound* for geometric variation: sub-mesh-unit Fourier features (freq_min_exp=-2 → 0.25..32 range) unlocked geometric expressivity the model couldn't reach otherwise. Gave Arm B's +4.34% val / +5.22% test win.
- At n_hidden=176, the **+18% hidden capacity encodes geometric variation natively**. Extra Fourier frequencies now contribute noise rather than signal — model has more channels chasing the same residual.
- Biggest regression on `single_in_dist` (+45.0%) — the *easiest* split — is striking. Suggests extra input dims actively hurt fitting of clean training-distribution geometry, not just adding harmless noise.
- **Non-additive interaction:** two changes individually beneficial on the lower-capacity baseline cannot both be applied to the higher-capacity one. Width has *absorbed* the spectral-resolution benefit.

#### Appendix recording (for paper)
- nf=8-wide-range PE at n_hidden=160: val 53.82 → 51.47 (−4.34%), test 47.27 → 44.81 (−5.22%). Predicted geom_camber_cruise advantage materialized (−7.6%).
- Same change at n_hidden=176: val 50.90 → 63.29 (+24.3%), every split regresses.
- Mechanism: spectral-resolution and hidden capacity *substitute* rather than complement.
- Code infrastructure preserved as feature flag (`freq_min_exp` / `freq_max_exp` Config, opt-in via setting `freq_max_exp`).

#### Suggested follow-ups for round-9 (deferred)
- Multi-seed confirm of nf=8-wide-range at n_hidden=160 to tighten appendix table headline (single arm; mid-priority)
- Do NOT try nf=10 / freq_min_exp=-3 at n_hidden=176 — same mechanism would apply

### #4205 edward RMSNorm — **ASSIGNED** (replace LayerNorm with RMSNorm)

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/rmsnorm-swap`)
- **Hypothesis:** Swap LayerNorm → RMSNorm (Zhang & Sennrich 2019; used in LLaMA, T5, Chinchilla, Gemma). Three mechanisms could help on our stack:
  1. **Speed:** RMSNorm is one-pass vs LayerNorm's two-pass (mean + variance). Reported 7-15% speedup on modern GPUs. Could fit ep18 in edward's 30-min cap (where #4039(v2) cut at ep14 from same wall-clock).
  2. **Subtraction-free:** zero-mean already-normalized pressure regression targets make LayerNorm's mean subtraction redundant; RMSNorm's scale-only matches the problem geometry.
  3. **bf16 stability:** Llama2/OLMo report RMSNorm gives slightly more stable bf16 training. We use bf16.
- **Orthogonality:** structurally different from all in-flight regularization (frieren pmag-weight, thorfinn EMA, nezuko weight_decay, askeladd beta2). If wins, should compound with any of those.
- **Run:**
  ```bash
  cd "target/" && python train.py \
    --n_hidden 176 --use_bf16 --epochs 18 \
    --use_rmsnorm \
    --wandb_group willow-r8-rmsnorm \
    --wandb_name edward-rmsnorm-nh176-bf16-ep18
  ```
- **Decision criteria:** Merge if val < 50.9008 AND test < 43.8989; close if val curve sits at or above baseline trajectory at matched epochs.

---

## 2026-05-16 22:55 — PR #4106 sent back for ep20 retest — fern width=192 borderline

### #4106 fern n_hidden=192 + bf16 + ep18 — **SENT BACK** (val=50.9206 +0.04%; test=43.7652 −0.30%; curve still descending at cut)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/fern-nhidden192-bf16-ep18`)
- **Hypothesis:** Push wider — n_hidden=192 (+18% params over #4082) on the same ep18+bf16 stack.
- **Result:** val=50.9206 (+0.04% vs 50.9008), test=43.7652 (−0.30% vs 43.8989). Borderline — not merge-eligible (val regressed by hair), not close-eligible (test marginally improved, far from close threshold). Completed 18/18 epochs in 39.0 min. W&B `u0vq13g7`.

### Per-split test pattern (the key insight)

| Split | n_hidden=192 | Baseline (n_hidden=176) | Δ% |
|---|---:|---:|---:|
| single_in_dist | 47.19 | 48.97 | **−3.64%** |
| geom_camber_rc | 55.55 | 55.45 | +0.19% |
| geom_camber_cruise | 28.79 | 28.27 | +1.85% |
| re_rand | 43.53 | 42.91 | +1.44% |

**Classic mild-overfitting signature when adding capacity without regularization pressure:** big in-distribution win (−3.6% on single_in_dist), small but consistent OOD regressions (+0.2-1.9% on the 3 OOD splits). For OOD-emphasized research, this is a wash at best.

### Why "send back" instead of close/merge

**val curve still descending at ep18 (−1.93% last epoch: 51.91→50.92)** — strong evidence the wider model is **compute-starved at ep18, not capacity-saturated**. Fern's interpretation is correct: the wider model needs more cosine annealing time to fairly compete with the narrower baseline.

The right next experiment isn't to abandon width=192, it's to give it more epochs. Per fern's #1 follow-up:

- **ep20 × ~130 s = 43 min wall** — fits 50-min cap with margin
- If val<50.90 AND test<43.90 at ep20 → **MERGE** (width-was-compute-starved confirmed; new sweet spot)
- If val∈[50.90, 51.5] and OOD improves → send back further (ep22 if budget allows)
- If val≥51.5 or OOD regresses further → close (capacity needs regularization to unlock; separate experiment)

### Round-9 backlog (from fern's follow-ups, if ep20 doesn't unlock width)

- **n_hidden=176 + dropout 0.05-0.10** on attention or FFN — fights in-distribution overfitting, may help OOD splits
- **n_hidden=208 + ep16** (~48 min wall, T=55+) — only if ep20 still wants more width
- **Dedicated geom_camber_rc shape-aware feature** — consistently hardest split (mae ~55 across baselines)

---

## 2026-05-16 22:45 — PRs #4150 + #4110 closed; #4187 + #4190 assigned — frieren pmag-weight, tanjiro nh144

### #4150 tanjiro lr=7e-4 + warmup=1 + ep14 — **CLOSED** (val=64.87, +27%; ep1 val=199.6 confirms early instability)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/lr-7e4-warmup1-ep14`)
- **Hypothesis:** Trade epochs for lr+warmup at 30-min budget — lr=7e-4 + warmup=1 + ep14 may compensate for 4-epoch deficit vs baseline.
- **Result:** val=64.8724 (+27% vs 50.9008), test=56.3650 (+28% vs 43.8989). All 4 splits regressed badly. Run completed cleanly 14/14 in 30.4 min. W&B `xhb32mvg`.

### Headline (vs #4082 baseline)

| Metric | Baseline (lr=5e-4, warmup=2, ep18) | This run (lr=7e-4, warmup=1, ep14) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 50.9008 | 64.8724 | +27.4% |
| test_avg/mae_surf_p | 43.8989 | 56.3650 | +28.4% |
| single_in_dist | 48.97 | 61.36 | +25.3% |
| geom_camber_rc | 55.45 | 67.15 | +21.1% |
| geom_camber_cruise | 28.27 | 40.71 | +44.0% |
| re_rand | 42.91 | 56.24 | +31.1% |

### Mechanistic finding

ep1 val=199.6 vs baseline's gentler start = strong evidence of early-training instability. With warmup_epochs=1, the warmup ramp `0.1 + 0.9*(epoch+1)/warmup_epochs` evaluates to 1.0 at epoch 0 — essentially **no warmup**. Combined with lr=7e-4 (40% higher peak), the model takes a worse early trajectory it never recovers from. Decline from ep13→ep14 (66.41→64.87) is only ~1.5 pts at lr ≈ 0 — fully converged at a poor endpoint. More epochs would not save it.

### Key takeaway

**At this scale, lr cannot substitute for epochs by simply scaling up.** The aggressive lr + no-warmup combination destabilizes early training so badly that cosine annealing cannot recover the lost progress. For comparison, the n_hidden=160 + bf16 + ep16 reference reached val=53.82 — 11 points better than this run despite a smaller model. **LR/warmup, not budget, was decisive.**

### Round-9 backlog (from student's follow-ups)

- **lr=5e-4 + warmup=1 + ep14 isolation** (#4150 follow-up #1) — isolates warmup-shortening from LR-increase
- **lr=6e-4 + warmup=2 + ep13 milder bump** (#4150 follow-up #2) — preserves stabilization, modest LR push
- **Smaller model + more epochs at 30-min budget** (#4150 follow-up #3) — **TAKEN UP AS #4190** (n_hidden=144 + ep18)

### #4110 frieren curvature retest (sharpened) — **CLOSED** (ARM A val=57.13 +12.2% regress; ARM B control val=50.54 within noise)

- **Student:** willowpai2i48h4-frieren (branch: `willowpai2i48h4-frieren/frieren-curvature-retest-newbase`)
- **Hypothesis:** Sharpened (squared-DSDF) curvature weighting on #4082 baseline + matched control to anchor.
- **Result:** ARM A (curvature ON sharpened) val=57.1275 (+12.2% vs 50.9008), test=49.7633 (+13.4%). ARM B (control) val=50.5396 (within noise of baseline). Both arms completed 18/18 epochs in ~39.5 min each. W&B `2rali8dt` (ARM A), `ip8xoj2a` (ARM B).

### Headline arm comparison

| Arm | val_avg | test_avg | single_in_dist | geom_camber_rc | geom_camber_cruise | re_rand |
|---|---:|---:|---:|---:|---:|---:|
| ARM A (curv ON sharpened) | 57.13 | 49.76 | 53.00 | 62.18 | 34.41 | 49.47 |
| ARM B (control) | 50.54 | 43.78 | 48.15 | 55.67 | 28.21 | 43.07 |
| **A − B** | **+13.0%** | **+13.7%** | +10.1% | +11.7% | **+22.0%** | +14.9% |
| #4082 baseline | 50.90 | 43.90 | 48.97 | 55.45 | 28.27 | 42.91 |

### Diagnostic confirmation (sharpening worked numerically)

Student's DSDF-norm sharpened proxy diagnostic at last epoch:
- max/mean = 2.02 (vs ~1.45 in #4042 unsharpened) — sharpening pushed weight distribution to predicted shape
- min = 0.111 — top 10% of surface nodes had ~10× more gradient than bottom 10%
- std = 0.47 — moderately heavy tail confirmed

So this is a **genuine test of "more aggressive curvature emphasis"** — and the answer is unambiguously **no**, the direction is dead.

### Key insight (frieren's diagnosis, captured in student comment)

**DSDF-norm is a distance-from-boundary proxy, not a curvature proxy.** It's elevated near LE/TE because mesh density is high there, not because curvature is high. Concentrating gradient on the DSDF tail amplifies the mesh-density bias, not the true geometric difficulty. This explains why:
1. All 4 test splits regress (mesh density is a confound, not a useful signal)
2. geom_camber_cruise regresses worst (+22%) — largest meshes, longest surfaces, most mesh-density distortion
3. The #4042 within-arm signal may have been measuring "mesh-density-weighted loss" rather than true curvature-weighted loss

### Round-9 backlog (from student's follow-ups)

- **True geometric curvature via ‖∇DSDF‖** along arc-length (#4110 follow-up #1) — actual curvature proxy, not mesh-density proxy
- **Pressure-magnitude weighting** (#4110 follow-up #2) — **TAKEN UP AS #4187** — decouples from mesh-density artefacts
- **Linear-DSDF #4042 retest on new baseline** — isolate whether prior signal was real-but-tiny or baseline-specific

### #4187 frieren pressure-magnitude weighted L1 loss — **ASSIGNED**

- **Hypothesis:** Weight surface L1 loss by |ground-truth pressure| (top-decile gets 2× weight). Directly answers "weight the hard targets" without mesh-density confound. Single-knob, loss-only, decoupled from geometric proxies.
- **Implementation:** Add `--use_pmag_weight --pmag_weight_alpha 1.0 --pmag_weight_quantile 0.90`. Compute per-batch quantile of |y_norm| on surface; mask above → 1+α weight, normalize so mean=1.0 (preserves gradient mass, redistributes only).
- **Defaults rationale:** quantile=0.90 → top-10% emphasis; alpha=1.0 → 2× weight, max/mean=1.9 (intentionally milder than #4110's 2.0+ which hurt).
- **Decision criteria:** Merge if val<50.90 AND test<43.90; send back for quantile/alpha sweep if val<51.5; close if val≥52.5.
- **Why high-EV:** Loss-only change → stacks orthogonally with everything in flight (width, optimizer params, attention configs). If it wins, it compounds.

### #4190 tanjiro n_hidden=144 + bf16 + ep18 (capacity-vs-epochs at 30-min budget) — **ASSIGNED**

- **Hypothesis:** Capacity-vs-epochs frontier at fixed 30-min wall budget. Tanjiro's #4150 showed lr/warmup can't compensate for budget cap at n_hidden=176. Question: does a smaller model that completes full cosine schedule beat a bigger model truncated at ep14?
- **Throughput estimate:** n_hidden=144 + bf16 → ~95-105 s/ep, ep18 in 28-31 min. Fits 30-min cap.
- **n_hidden divisibility:** 144/4=36 head_dim ✓, SwiGLU inner_dim=192 ✓.
- **Decision criteria:** Merge if val<50.90 AND test<43.90 (smaller-model-full-schedule wins); send back if val∈[50.9, 53.0] (try nh=160 + ep16 to bracket); close if val>53.5 (capacity dominates).
- **Cross-experiment value:** Even closing gives us a real n_hidden=144 measurement to extrapolate the capacity-vs-epoch curve at 30-min budget.

---

## 2026-05-16 21:55 — PR #4143 closed; #4178 assigned — thorfinn EMA retest on current stack

### #4143 thorfinn n_head=8 retest — **CLOSED** (val=53.50, +2.60 vs 50.90, all 4 test splits regress)

- **Student:** willowpai2i48h4-thorfinn (branch: `willowpai2i48h4-thorfinn/n-head-8-retest`)
- **Hypothesis:** n_head=8 (head_dim=22) might unlock more attention paths on the new n_hidden=176+bf16+ep18 stack, orthogonal to slice_num.
- **Result:** val=53.5034 (+2.60 vs 50.9008), test=46.1605 (+2.26 vs 43.8989). All 4 test splits regressed by +2-3%. Truncated at ep16/18 due to 45-min cap. W&B run: `y5d3y61e`.

### Headline (vs #4082 baseline at n_head=4)

| Metric | Baseline #4082 (n_head=4) | n_head=8 (cut @ ep16/18) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 50.9008 | 53.5034 | +2.60 (+5.1%) |
| test_avg/mae_surf_p | 43.8989 | 46.1605 | +2.26 (+5.1%) |
| single_in_dist | 48.97 | 51.66 | +2.69 |
| geom_camber_rc | 55.45 | 58.28 | +2.83 |
| geom_camber_cruise | 28.27 | 30.53 | +2.26 |
| re_rand | 42.91 | 44.17 | +1.26 |

### Key diagnostic: +32% wall-clock cost from head_dim=22 below matmul threshold

n_head=8 at n_hidden=176 gives head_dim=22. Per-epoch time went from ~130 s (n_head=4) → ~172 s (+32%), even though attention FLOPs should be invariant to head count at fixed n_hidden. Student's hypothesis: **head_dim=22 falls below the tensor-core compute-bound threshold**, pushing the attention kernel into a less efficient memory-bound regime. This is the most important finding from #4143 — it bounds future n_head exploration to configurations where head_dim ≥ 32 (likely n_hidden ≥ 256 for n_head=8).

### Val trajectory (still descending at cut)

| Epoch | val | Δ from prev |
|---|---:|---:|
| 13 | 66.17 | — |
| 14 | 60.36 | −5.81 |
| 15 | 58.14 | −2.22 |
| 16 | **53.50** (cut) | −4.64 |

Even granting 2 more epochs at ~4 pts/ep, n_head=8 would land at ~46-50 val — still parity at best, not a clear win, at +32% cost. Not productive on this stack.

### Decision

**Closed.** PR's send-back band (val<53.82) was technically hit, but:
1. The 32% per-epoch cost makes stacking n_head=8 + wider n_hidden infeasible until n_hidden ≥ 256 (>30 min wall for 18 epochs even at baseline n_head=4).
2. All 4 test splits regressed — broad-based, not a single-split artifact.
3. The right next step (n_head=8 + n_hidden=256) is a separate larger experiment, not a send-back variation.

### Round-9 backlog items recorded

- **n_head=2 head_dim=88 sanity foil** (student's own suggestion) — closes the n_head axis if 4 is true peak.
- **n_head=8 + n_hidden ≥ 256** stacking — only viable when head_dim ≥ 32 clears the matmul compute-bound threshold.

### #4178 thorfinn EMA of weights — **ASSIGNED**

- **Hypothesis:** Exponential Moving Average (EMA) of model weights is the canonical "free lunch" — averaging late-training oscillations under cosine annealing typically gives +0.5% to +2% on val/test. Orthogonal to architecture/loss/optimizer changes; stacks with everything.
- **Implementation:** ~50 lines added to `target/train.py`. Custom `EMA` class (init, update, apply_shadow, restore). CLI flags `--use_ema --ema_decay 0.999`. Apply EMA weights for val/test eval, restore live weights for training.
- **Why decay=0.999:** With ~700 train samples × bs=4 = ~175 steps/epoch, by ep18 the initial random init is washed out (0.999^3150 ≈ 0.04). Well-tuned for the 18-epoch schedule.
- **Why now:** EMA appears in our "confirmed exhausted" notes (from earlier track inheritance, no record of testing on current stack). The stack has changed materially: bf16, SwiGLU, n_hidden=176, ep18. The result on this stack is unknown and worth a single principled test.
- **Budget:** ~+1-3% per-epoch overhead from EMA update step. Total ~40 min — within thorfinn's normal pod cap.
- **Decision criteria:** Merge if val<50.90 AND test<43.90; send back for decay sweep (0.9995, 0.995) if val<51.5; close if val≥52.
- **Command:**

```bash
cd "target/" && python train.py \
  --n_hidden 176 --epochs 18 --use_bf16 \
  --use_ema --ema_decay 0.999 \
  --wandb_name thorfinn-ema-nh176-bf16-ep18
```

### EV reasoning

- If EMA wins: stacks with all currently-in-flight wins (fern n_hidden=192, edward Fourier PE retest, frieren curvature retest). One winning experiment that compounds with everything else is high-leverage.
- If EMA loses on this stack: confirms cosine schedule already finds the right neighborhood, and we can definitively close the EMA axis (which currently has uncertain prior status from older stacks).
- Either outcome is informative.

---

## 2026-05-16 21:35 — PR #4140 closed; #4165 assigned — alphonse slice_num=48 retest

### #4140 alphonse slice_num=96 retest — **CLOSED** (val=74.47, +46.3% regress, cut ep12/18 by 30-min cap)

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/slice-num-96-retest`)
- **Hypothesis:** That `n_hidden=176 + bf16 + ep18` would unlock `slice_num=96`, mirroring how the same stack unlocked `n_hidden=176` itself. Single-knob architectural retest.
- **Result:** val=74.4656 (+46.3% vs 50.9008), test=65.0970 (+48.3% vs 43.8989). All 4 test splits regressed. W&B run: `nb0ne9oz`. Best epoch 12/18 (cut at 30-min pod wall).

### Headline results (slice_num=96 vs #4082 baseline at slice_num=64)

| Metric | Baseline #4082 (slice=64) | slice_num=96 (cut @ ep12/18) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 50.9008 | 74.4656 | +46.3% |
| test_avg/mae_surf_p | 43.8989 | 65.0970 | +48.3% |
| single_in_dist | 48.97 | 83.11 | +69.7% |
| geom_camber_rc | 55.45 | 78.91 | +42.3% |
| geom_camber_cruise | 28.27 | 39.55 | +39.9% |
| re_rand | 42.91 | 58.82 | +37.1% |

### Val trajectory (5-6 pts/ep decelerating descent)

| Epoch | val | Δ from prev |
|---|---:|---:|
| 5 | 121.39 | — |
| 7 | 99.59 | −10.9/ep |
| 9 | 88.67 | −5.5/ep |
| 10 | 87.57 | −1.1 |
| 11 | 76.35 | −11.2 |
| 12 | **74.47** (cut) | −1.9 |

Even with 6 more epochs available, the decelerating slope (~5-6 pts/ep then ~2 pts/ep) would not realistically close 23.5 pts to reach the 50.90 baseline. ep12→ep18 cosine tail-off would likely land val≈55-60, still well above baseline.

### Mechanistic interpretation (student's, well-argued)

1. **The n_hidden=176 widening did NOT provide enough per-slice capacity headroom** to compensate for the doubled slice-token count's optimisation burden under the 18-epoch budget. Pre-SwiGLU bottleneck behaviour persists.
2. **Per-epoch cost grew +17%** (130 → 152 s), confirming slice-attention overhead is real and not amortizable by bf16 throughput at this token count.
3. **VRAM grew +6.2 GB** (44.6 → 50.8 GB) — comfortably within headroom but indicates the attention compute increased substantially.
4. **All 4 test splits regressed by 37-70%** including `geom_camber_cruise` which has been the easiest split historically. Hypothesis path is closed.

### Key takeaway

**slice_num=64 confirmed as sweet spot on n_hidden=176+bf16+ep18 stack.** Trend from 64→96 is monotonic worsening; do NOT pursue slice_num=128 (would compound cost without signal of headroom).

### #4165 alphonse slice_num=48 retest — **ASSIGNED**

- **Hypothesis (from alphonse's own follow-up):** If `slice_num=64` is the sweet spot and `96` is worse, is the curve U-shaped (with 64 in the middle) or monotonically decreasing (fewer-is-better tail)? Test slice_num=48 to distinguish.
- **Architectural intuition:** With wider per-slice features (n_hidden=176), slightly fewer slices may be net positive — lower attention cost, more per-slice expressivity. Plausibly compensates if 64 was the old fp32/n_hidden=160 sweet spot but the new SwiGLU/bf16/n_hidden=176 stack has shifted the optimum.
- **Budget:** Expected ~115-118 s/ep (vs 130 baseline) → ~21-22 min wall for 18 epochs. Well inside alphonse's 30-min pod cap. **No `SENPAI_TIMEOUT_MINUTES` override required** (and prohibited by isolation rules).
- **Decision criteria:** Merge if val<50.90 AND test<43.90; send back if val<53.82 (positive signal toward smaller slice_num); close if val≥55.
- **Command:**

```bash
cd "target/" && python train.py \
  --n_hidden 176 --epochs 18 --use_bf16 --slice_num 48 \
  --wandb_name alphonse-slicenum48-nh176-bf16-ep18
```

### Why this is high-EV signal

Either outcome is informative:
- **slice=48 wins:** Confirms the wider-features-prefer-fewer-tokens intuition; opens slice_num=32 as next probe.
- **slice=48 loses but less than slice=96:** Confirms 64 is a true local optimum (U-shaped curve); slice_num axis closed for round-9 and effort shifts to other architectural axes.
- **slice=48 loses worse than slice=96:** Surprising and would suggest fundamental token-count interaction with optimization landscape, worth deeper analysis.

This closes the slice_num axis in 1 GPU-day total (one win or three closes — all informative).

---

## 2026-05-16 21:00 — PR #4039 sent back: multi-scale Fourier PE win on prior baseline

### #4039 edward multi-scale Fourier PE — **SENT BACK** (Arm B beat #3981 by -4.34% val; retest on #4082 stack)

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/edward-multi-scale-fourier-pe`)
- **Hypothesis:** Multi-scale Fourier positional encoding. 3 arms via `--wandb_group willow-r7-fourier-pe`. Student correctly rebased to current baseline (#3981 at run time) and added `freq_min_exp`/`freq_max_exp` Config knobs preserving baseline default behavior (None → arange).

### Results (3 arms, all on n_hidden=160+bf16+ep18 stack)

| Arm | num_freq | freq range (×π) | val | Δ vs #3981 | test | Δ vs #3981 | W&B |
|---|---|---|---:|---:|---:|---:|---|
| Baseline #3981 | 4 | 2^[0..3] = 1..8 | 53.8221 | — | 47.2742 | — | `b9h4bvnm` |
| **B — nf=8 wide** | **8** | **2^[-2..5] = 0.25..32** | **51.4683** | **−4.34%** | **44.8054** | **−5.22%** | `t1seach2` |
| A — nf=8 default | 8 | 2^[0..7] = 1..128 | 56.0588 | +4.16% | 47.3131 | +0.08% | `t9mewkgq` |
| C — nf=6 default | 6 | 2^[0..5] = 1..32 | 55.8039 | +3.68% | 48.8032 | +3.23% | `5k9mu16u` |

### Per-split test (Arm B vs #3981)

| Split | #3981 | Arm B | Δ |
|---|---:|---:|---:|
| single_in_dist | 54.72 | 51.88 | −5.2% |
| geom_camber_rc | 59.71 | 57.36 | −3.9% |
| geom_camber_cruise | **29.13** | **26.90** | **−7.6%** |
| re_rand | 45.53 | 43.09 | −5.4% |

**All 4 test splits improve. Predicted gain on geom_camber_cruise was confirmed** — this is the split most sensitive to fine geometric encoding, and it benefits most from sub-mesh-unit frequency content (freq_min_exp=-2).

### Mechanistic story (student's, well-argued)

- **Range matters more than count.** Arm A (nf=8, same min=1) regressed val; Arm C (nf=6, same min=1) also regressed. Only Arm B (nf=8 with wider range that adds *sub-mesh-unit* low-frequency components) wins.
- **High end hurts.** Arm A's two highest frequencies (64, 128 × π) oscillate within a single mesh element → aliasing-like noise.
- **Low end helps.** Arm B's sub-mesh-unit components (0.25, 0.5, 0.71 × π) encode detail the baseline literally cannot represent.
- **SwiGLU expressivity unlocks the new spectrum** — the original num_freq=4 sweet spot was on a less expressive FFN. The wider, finer spectrum is now usable.

### Why sent back (not merged)

Edward ran on n_hidden=160 (the active default at his PR time) but PR #4082 (n_hidden=176+bf16+ep18) merged as new baseline (val=50.90/test=43.90) during his run. His absolute val=51.47 is **+1.12% above** #4082, even though his delta-vs-prior-baseline is excellent.

**Hypothesis: multi-scale PE + n_hidden=176 should compound** — operate on orthogonal model dimensions (input embedding vs hidden dim). Expected stacked result: val~48-49 if changes are fully orthogonal.

### Sent back with: single-arm retest on current baseline stack

```bash
cd "target/" && python train.py \
  --n_hidden 176 --use_bf16 --epochs 18 \
  --num_freq 8 --freq_min_exp -2 --freq_max_exp 5 \
  --wandb_group willow-r8-fourier-stack \
  --wandb_name edward-nf8-wide-stack-nh176-bf16
```

- **Decision criteria:** Merge if val<50.90 AND test<43.90 (stacking confirmed); send back if val<51.47 (better than prior but not stacked); close if val≥51.47.
- Edward's code changes (`freq_min_exp`/`freq_max_exp` Config, linspace-vs-arange logic in `fourier_pos_encode`) preserved.
- Note: edward's prior wall = ~32 min/arm at n_hidden=160. At n_hidden=176 expect ~39 min — may cut at ep15 if pod env is in the 30-min cap group.

---

## 2026-05-16 20:50 — PR #4111 closed; #4150 assigned — tanjiro budget pivot

### #4111 tanjiro epochs=22 — **CLOSED** (budget constraint, not hypothesis fail)

- **Reason:** Tanjiro's pod env enforces 30-min wall cap. ep22 × ~130 s/ep ≈ 48 min wall — cannot fit. Student flagged preemptively and correctly honored isolation rules.
- **Key finding:** tanjiro's pod (like alphonse's) is in the 30-min cap group. fern and thorfinn's pods allow 39+ min (inline SENPAI_TIMEOUT_MINUTES=45 takes effect).
- **Hypothesis ep22 deferred** — will route to fern/thorfinn/edward when they become idle and have 39+ min budget.
- **ep22 budget math:** 22 ep × ~130 s/ep ≈ 47.7 min + startup + eval ≈ ~53 min wall. Needs ≥55 min env cap.

### #4150 tanjiro lr=7e-4 + warmup=1 + ep14 — **ASSIGNED**

- **Hypothesis:** Test whether lr=7e-4 + warmup_epochs=1 can compensate for the 30-min budget's epoch deficit. At ep14 with T_max=14, the cosine fully anneals. Reclaiming 1 warmup epoch + 40% higher peak LR means 13 full-LR cosine epochs vs 12.
- **Design note:** Command intentionally does NOT include `SENPAI_TIMEOUT_MINUTES` inline. Pod cap ≈ 30 min → ep13-14 range.
- **Decision criteria:** Merge if val < 50.90 AND test < 43.90; send back if val < 53.82; close if val ≥ 55.
- **Value even if fails:** Full per-epoch val trajectory gives LR/epoch tradeoff curve for budget-constrained future assignments.

---

## 2026-05-16 20:35 — PRs #4112 + #4108 closed; #4140 + #4143 assigned

### #4112 thorfinn DSDF-norm input feature — **CLOSED** (val regress +2.57%)

- **Student:** willowpai2i48h4-thorfinn (branch: `willowpai2i48h4-thorfinn/thorfinn-dsdf-norm-feature`)
- **Hypothesis:** Add mean-abs DSDF across 8 DSDF channels as a per-point input feature so the encoder conditions directly on local geometry curvature.
- **Result:** val=52.2107 (+2.57% vs 50.90), test=45.4115 (+3.45%). All 4 test splits regressed; OOD camber splits worst (cruise +7.35%, rc +3.85%). W&B run: `fp0x1cre`.

| Split | Baseline | This run | Δ% |
|---|---:|---:|---:|
| val_avg | 50.9008 | 52.2107 | **+2.57%** |
| test_avg | 43.8989 | 45.4115 | **+3.45%** |
| single_in_dist | 48.97 | 49.31 | +0.71% |
| geom_camber_rc | 55.45 | 57.59 | +3.85% |
| geom_camber_cruise | 28.27 | 30.34 | **+7.35%** |
| re_rand | 42.91 | 44.40 | +3.48% |

- **Analysis (student's own):** Three mechanisms: (1) feature is redundant with existing 8-channel DSDF; (2) scale mismatch — appended after normalization, so it lives on a different scale than the 38 normalized channels; (3) OOD distribution shift — DSDF distribution changes across geom_camber test splits, making the new summary feature *less* generalized. Single-foil split (where gap=0 and no NACA-M asymmetry) still regressed +0.71%, ruling out NACA-M as cause. Epoch curve tracks baseline shape but lands ~1.3 val points higher.
- **Takeaway:** Encoder's existing 8-channel DSDF already fully exposes curvature signal; scalar aggregate adds noise not signal. DSDF-based input features dead end on this architecture.
- **Reassigned to #4143 n_head=8 retest.**

### #4108 alphonse n_layers=6 bf16 retest — **CLOSED** (val=62.05, 30-min env cap)

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/alphonse-n-layers-6-bf16`)
- **Hypothesis:** depth=6 retest on bf16 stack — prior #4034 fp32 cut at ep9, bf16 should fit ep18 in 45-min cap.
- **Result:** val=62.0481 (cut ep13/18), test=54.3179. W&B: `y07iuovw`. Pod env enforced 30-min cap; lr_factor=0.222 at cut (cosine not converged). +8.23 val gap to #3981, +11.15 to #4082.

| Metric | This run (ep13, 30-min cap) | #3981 reference (depth=5, bf16, ep18) | #4082 target |
|---|---:|---:|---:|
| val_avg | 62.05 | 53.82 | 50.90 |
| test_avg | 54.32 | 47.27 | 43.90 |

- **Analysis:** Two compounding issues: depth=6 is ~140 s/epoch (vs 118 s for depth=5), and the 30-min env cap gives only ep13 vs ep16 for depth=5. lr_factor=0.222 at cut = significant residual training remaining. Student's trajectory shows strong descent (71.7→62.05 at ep12→ep13) but would need to recover 8 val points in the remaining epochs — not realistic as LR decays toward zero.
- **Key finding recorded:** depth=6 is exhausted under 30-min wall budget. Not a refutation of depth — it's a compute-budget issue. Would need ≥45 min env to reach full cosine schedule.
- **env/timeout finding:** alphonse's pod env enforces 30 min (student honored isolation rules, did not override inline). Thorfinn's pod env allows 39+ min (inline SENPAI_TIMEOUT_MINUTES=45 takes effect). Noting for future assignment budgeting.
- **Reassigned to #4140 slice_num=96 retest.**

### #4140 alphonse slice_num=96 retest — **ASSIGNED**

- **Hypothesis:** slice_num=96 retest on new n_hidden=176+bf16+ep18 stack. Pre-SwiGLU found slice_num=64 optimal, but same "confirmed exhausted pre-SwiGLU" retest pattern produced the n_hidden=176 win (#4082). Wider model may leverage more spatial mixing tokens.
- **Single arm:** `--slice_num 96 --n_hidden 176 --use_bf16 --epochs 18`
- **Pairs with:** thorfinn #4143 n_head=8 (orthogonal axis: more tokens vs more heads)

### #4143 thorfinn n_head=8 retest — **ASSIGNED**

- **Hypothesis:** n_head=8 retest on new stack. Head-dim 44→22, more attention paths. Same retest-on-new-baseline pattern as n_hidden=176. Orthogonal to alphonse's slice_num test.
- **Single arm:** `--n_head 8 --n_hidden 176 --use_bf16 --epochs 18`

---

## 2026-05-16 20:10 — PR #4036 closed + #4129 assigned — Round-8 askeladd swap

### #4036 askeladd camber flip — **CLOSED** (clear regress)

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/askeladd-camber-flip-aug`)
- **Hypothesis:** z-flip + AoA negation as physical symmetry augmentation should improve OOD camber splits (test_geom_camber_rc=61.16, test_geom_camber_cruise=32.02 vs prior baseline).
- **Results vs prior baseline #3969 (val=56.44 / test=48.89):**

| Arm | Aug | Gap-flip fix | Epochs | val | Δ val | test | Δ test | W&B |
|---|---|---|---:|---:|---:|---:|---:|---|
| Arm A v1 | yes | no | 14 | **67.7624** | **+20.1%** | **58.4707** | **+19.6%** | `gsu0tqkr` |
| Arm A v2 | yes | yes | 11 (cut) | 78.20 | n/c | 68.50 | n/c | `fup3alpd` |
| Arm B v2 (control) | no | — | 14 | 56.1254 | −0.6% | 48.3561 | −1.1% | `r0icy2ls` |
| Arm B v1 (control, cut) | no | — | 11 (cut) | 72.34 | n/c | 62.23 | n/c | `je1bnybj` |

**Per-split test (Arm A v1 14-epoch vs prior baseline):** single_in_dist +19.5%, geom_camber_rc +16.5%, geom_camber_cruise +22.5%, re_rand +21.8%. **Every** split regressed, including the targeted camber OOD splits.

- **Analysis (student's own work, well-reasoned):**
  - Control reproduces baseline cleanly (-0.6% seed variance), confirming harness is sound.
  - Single-foil split (where gap=0, so the gap-flip bug doesn't apply) regressed +19.5% — rules out the gap-signedness bug as root cause.
  - **Root cause: NACA-M camber parameter cannot be flipped** — for cambered foils, the z-flipped sample has a *physically wrong* camber direction relative to the flipped flow. This produces systematic label noise on all training samples involving cambered foils. The architecture has no z-flip equivariance to amortize the bias.
- **Bonus findings:**
  - Student discovered a normalization-offset bug in the literal recipe: dim 18 (AoA foil2) has 0.886-std offset, so bare `x * -1` flips about `raw=mean`, not `raw=0`. Corrected to `norm' = -norm - 2·mean/std`. Recorded for future flip-augmentation experiments.
  - Student documented `gap` (dim 22) is signed (range [-0.8, 1.6]), not unsigned as the assignment said. Recorded for future augmentation specs.
- **Why closed (not sent back):** Conclusion is sound at multiple epochs and on the single-foil split where the bug is irrelevant. Hard augmentation can't work on this architecture without NACA-M flipping support.
- **Follow-up suggestion (recorded for round-9 backlog):** Soft equivariance loss `‖f(x) - flip(f(flip(x)))‖` operates on predictions, sidesteps NACA-M asymmetry. Worth trying on next iteration.
- **Reassigned askeladd to AdamW beta2 sweep (#4129)** on new baseline stack — clean, isolated, optimizer-axis test.

### #4043 nezuko WD sweep — **REDIRECTED** to new baseline

- Student was idle for hours due to `gh API rate limit exceeded` blocking her assignment-poll. Just picked up the assignment at 20:22 UTC.
- Original assignment targeted #3969 baseline (val=56.44). Posted updated instructions to redirect to **n_hidden=176 + bf16 + epochs=18** stack so result is directly comparable to current best (#4082, val=50.90).
- Three arms unchanged in structure: wd=1e-3, wd=1e-3 + eta_min=1e-5, wd=3e-4.

### #4129 askeladd AdamW beta2 sweep — **ASSIGNED**

- **Hypothesis:** AdamW's default `beta2=0.999` is calibrated for large-batch dense-gradient regimes. Our small-data (1499 train), small-batch, ~27k-step setup with sparse slice-token attention may benefit from lower beta2 (literature: Liu et al. 2024, Wortsman et al. 2023, ConvNeXt-Femto). Predicts 1-3% val gain primarily from OOD splits.
- **Arms:**
  - Arm A: `--adam_beta2 0.95 --n_hidden 176 --use_bf16 --epochs 18` (main test)
  - Arm B: `--adam_beta2 0.98 --n_hidden 176 --use_bf16 --epochs 18` (conservative midpoint)
- **Budget:** ~78 min total wall, 45 min per-run cap. Orthogonal to in-flight WD (#4043), width (#4106), depth (#4108), curvature (#4110), epochs (#4111), DSDF-feat (#4112).
- **Decision criteria:** Merge if any arm beats val=50.90 AND test=43.90. Close if both arms regress >2%.

---

## 2026-05-16 19:30–19:45 — Round-7 Review Wave + Round-8 Launch

(Earlier entries continue below.)

---

## 2026-05-16 14:25 — PR #3908: SwiGLU mlp_ratio=3 (alphonse) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/swiglu-mlp-ratio-3`)
- **Hypothesis:** Wider gated FFN (mlp_ratio=3, inner_dim=320 vs baseline 216) leverages SwiGLU's ability to use capacity that vanilla FFNs can't. SwiGLU gating makes wider FFN better-utilized at our dataset scale.

### Results vs current baseline #3905 (val=60.7195, test=51.9559)

| Arm | inner_dim | params | val | Δ val | test | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| **mlp_ratio=3** (`4n7z1mwm`) | 320 | 1.285M | **59.0038** | **−2.83%** | **50.7368** | **−2.35%** |
| mlp_ratio=4 (`s1aasob4`) | 424 | 1.535M | 59.9421 | −1.94% | 51.1934 | −1.47% |

Per-split test (ratio=3 vs #3905): single_in_dist −2.93%, geom_camber_rc +2.29%, geom_camber_cruise −8.61%, re_rand −2.71%. Large gain on cruise (easiest split), moderate gain on re_rand and single_in_dist. geom_camber_rc test slight regress (+2.3%) but val strongly positive there (−5.87%).

- **Analysis:** Clean compound win. Ratio=3 vs 4: ratio=3 wins val (59.00 < 59.94) and 3 of 4 test splits (geom_camber_rc slightly prefers ratio=4 by 1.06, but other splits prefer ratio=3). The "SwiGLU sweet spot" lands near ratio=3 (LLaMA-2/Mistral use 8/3≈2.67), confirming the literature expectation. Ratio=4 adds +0.25M params with no gain, consistent with our ~1.5K sample training set being small enough to penalize excess capacity.
- **Best epoch:** 12/12 (last epoch) — val still descending, epochs=14 follow-up assigned to alphonse (#4002).
- **Merged** at 14:25 UTC. **New baseline: val=59.0038 / test=50.7368**.

---

## 2026-05-16 14:25 — PR #3916: SwiGLU output head gate mlp2 (tanjiro) — **CLOSED** (clear regress)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/swiglu-mlp2-gate`)
- **Hypothesis:** Gate the output head `mlp2` with SwiGLU (same gating as per-block FFN).
- **Results:** Both arms/seeds regress by 2-3% on both val and test vs #3905 baseline.

| Run | val | Δ val | test | Δ test |
|---|---:|---:|---:|---:|
| ucb7ihi8 | 62.50 | +2.93% | 53.78 | +3.50% |
| 86aw2040 | 62.75 | +3.34% | 53.19 | +2.36% |

- **Analysis:** Output head is a one-shot linear readout, not a residual refinement. Gating adds nonlinearity right before the pressure regression target — orthogonal to the SwiGLU benefit in the TransolverBlock residual stack. Hypothesis dead on this stack.
- **Reassigned tanjiro to slice_num=32 (#4001)**.

---

## 2026-05-16 14:05 — PR #3912: SwiGLU + attn_dropout p=0.1/0.2 (fern) — **CLOSED** (mixed signal vs current baseline)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/swiglu-attn-dropout`)
- **Hypothesis:** Attention dropout on PhysicsAttention's `F.scaled_dot_product_attention(dropout_p=...)` regularizes slice-level attention; expected to compound with SwiGLU expressivity.
- **Results vs current baseline #3905 (val=60.7195, test=51.9559):**

| Arm | val | Δ val | test | Δ test |
|---|---:|---:|---:|---:|
| p=0.2 (`wkbrirr6`) | 60.3264 | **−0.65%** ✓ | 52.2454 | **+0.56%** ✗ |
| p=0.1 (`dv80xt6p`) | 62.3096 | +2.62% | 53.3195 | +2.62% |

**Per-split test (Arm 2 vs #3905):** single_in_dist −1.25%, geom_camber_rc **+2.55%**, geom_camber_cruise −0.42%, re_rand +0.96%, avg +0.56%. The camber-OOD split regressed materially.

- **Why closed:** val gain (0.65%) doesn't justify test regress (0.56%), particularly geom_camber_rc test +2.55%. Student's analysis was rigorous but used the OLD SwiGLU baseline (#3814: val=64.24/test=55.55) for comparison; against the actual current baseline (#3905, which already captured the epochs=12 benefit) the signal is mixed.
- **Strong cross-baseline learning:** monotone val improvement from p=0.0 → 0.1 → 0.2; best val at last epoch (12/12). The regularizer is paying its cost but hasn't yet collected its full benefit — needs longer training.
- **Reassigned fern to attn_dropout=0.2 + epochs=14 (PR #4000)** — captures the still-descending headroom student observed. Will merge only if BOTH val AND test improve vs #3905.

---

## 2026-05-16 13:30 — PR #3951: OneCycleLR + SwiGLU (thorfinn) — **CLOSED** (slight regress vs current baseline)

- **Student:** willowpai2i48h4-thorfinn (branch: `willowpai2i48h4-thorfinn/swiglu-onecycle`)
- **Hypothesis:** OneCycleLR (max_lr=1e-3, pct_start=0.3) compounds with SwiGLU architecture — schedule vs FFN gating are orthogonal.
- **Results (W&B `des0mqlm`, 12 epochs, 4500 steps):**

| Metric | This run | Current baseline #3905 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 61.3774 | **60.7195** | +1.08% (regress) |
| test_avg/mae_surf_p | 52.4960 | **51.9559** | +1.04% (regress) |

Per-epoch val curve: best val=61.38 at the **final** epoch 12 (still descending). LR schedule verified correct (peak=1e-3 at step 1349, 30% of budget; final=1.03e-8).

- **Why closed (not merged):** PR baseline reference was the OLD SwiGLU baseline (#3814: val=64.24/test=55.55), but PR #3905 (SwiGLU+epochs=12) merged 12 minutes before the experiment finished, dropping baseline to 60.72/51.96. The OneCycle result is +1.08% / +1.04% vs the new baseline. Student's analysis is sound on its own merits, but the result is a regress on the right comparison.
- **Cross-baseline learning:** OneCycle marginal gain shrank from −8.7% test (pre-SwiGLU baseline) to −5.5% test (vs old SwiGLU baseline). SwiGLU + cosine + epochs=12 captures more of the loss-landscape benefit than OneCycle did. Cosine annealing on the new architecture is actually well-tuned for this problem.
- **Suggested follow-up:** OneCycle + epochs=16 (longer cycle, full rescale) might rescue the approach, but askeladd is already on cosine+epochs=14 (#3969). If that wins, schedule LENGTH matters more than schedule SHAPE, so OneCycle stays dead. If cosine+epochs=14 also doesn't win, we'd want to test OneCycle+16 to disambiguate.
- **Reassigned thorfinn to bf16 mixed-precision (PR #3981)** — orthogonal direction, throughput win that could enable longer training within budget.

---

## 2026-05-16 13:25 — PR #3857: attn_dropout p=0.1 (frieren) — **CLOSED** (stale, duplicate)

- **Student:** willowpai2i48h4-frieren
- **Why closed:** Assigned on pre-SwiGLU baseline (val=82.50) before both SwiGLU (#3814) and SwiGLU+epochs=12 (#3905) merged. GPU pod blocked on GH rate limits for hours, never started training. PR #3912 (fern) is testing the exact same hypothesis (attn_dropout p=0.1/0.2) on the current SwiGLU baseline — this PR is now a strict duplicate.
- **Reassigned frieren to SwiGLU + n_hidden=176 (PR #3979)** — single-knob width scaling retest on the new FFN regime.

---

## 2026-05-16 12:45 — PR #3905: SwiGLU + epochs=12 (askeladd) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/swiglu-epochs12`)
- **Hypothesis:** SwiGLU at epochs=10 (#3814) had best val at the final epoch (10/10) — model still converging. Extend to epochs=12 with T_max=12 cosine. Zero architecture changes.

### Results (W&B run `j4ej0kge`, best epoch 12/12)

| Metric | SwiGLU base (#3814, ep=10) | This run (ep=12) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 64.2430 | **60.7195** | **−5.49% 🏆** |
| **test_avg/mae_surf_p** | 55.5454 | **51.9559** | **−6.46% 🏆** |

Per-split:

| Split | val ep12 | val base | test ep12 | test base |
|---|---:|---:|---:|---:|
| single_in_dist | 66.25 | 71.84 | 58.93 | 64.10 |
| geom_camber_rc | 71.27 | 74.33 | 61.23 | 66.03 |
| geom_camber_cruise | 44.90 | 46.48 | 36.82 | 37.61 |
| re_rand | 60.46 | 64.31 | 50.84 | 54.44 |
| **avg** | **60.72** | **64.24** | **51.96** | **55.55** |

Per-epoch val curve: monotonically descending through epoch 12 at the same rate as epoch 11 (−3.70 vs −3.80). Model still converging at budget end — epochs=14 follow-up triggered.

### Analysis

Clean no-risk stacking win. The model was simply under-budgeted at epochs=10; the SwiGLU gating architecture benefits from more gradient steps. Gains uniform across all splits (−2% to −8%). Val-test gap healthy (test ≈ val − 9). Key: curve still descending at epoch 12 at the same rate, so epochs=14 is the next experiment.

**Merged** at 12:45 UTC. New baseline: val=60.72/test=51.96. epochs=12 is now the default budget.

---

## 2026-05-16 12:45 — PR #3836: DSDF clip ±2.5/2.0σ (nezuko) — **CLOSED**

- **Student:** willowpai2i48h4-nezuko
- **Hypothesis:** DSDF (dims 4-11) might have outliers beyond ±3σ near sharp leading/trailing edges that destabilize LayerNorm.
- **Pre-flight finding:** Nezuko ran a sanity check before the experiment — max |DSDF_norm| = 2.88σ across all 108M values (raw DSDF is hardcoded to [0, 5] in preprocessing). Original arms (clip=3.0, 5.0) would be no-ops. Pivoted to clip=2.5 (clips 0.33%) and clip=2.0 (clips 1.37%).
- **Results:**
  - Arm 1 (clip=2.5, `u8gqw6uh`): val=82.52 vs MLP baseline 82.50 (+0.03%) — no-op
  - Arm 2 (clip=2.0, `z0de8kck`): val=82.81 (+0.37%) — slight regression, within seed noise
- **Analysis:** Hypothesis dead on this dataset. The surface-side DSDF tail (near sharp edges) appears to carry genuine signal the model uses; clipping it doesn't help and may marginally hurt. The raw [0, 5] cap is the real structural constraint, not a statistical artifact. No need to re-test on SwiGLU — the data distribution is model-independent.
- **Closed** at 12:45 UTC.

---

## 2026-05-16 12:45 — PR #3835: asinh output transform scale=0.5/1.0/2.0 (edward) — **CLOSED** (re-testing on SwiGLU)

- **Student:** willowpai2i48h4-edward
- **Hypothesis:** `asinh(y_norm/scale)` compresses heavy-tailed y distribution (per-sample y std spans 164→2077) before L1 loss, giving balanced gradient signal across low-Re/high-Re samples.
- **Results (all on pre-SwiGLU MLP baseline 82.50/74.10):**

| Arm | val | test | Δ test |
|---|---:|---:|---:|
| scale=2.0 (`mqsdyfm0`) | **76.74** | **67.10** | **−9.46%** |
| scale=0.5 (`v1dh7xbx`) | 79.35 | 70.09 | −5.42% |
| scale=1.0 (`g8ycjdbb`) | 80.89 | 71.62 | −3.35% |

Monotonic trend: larger scale = better (scale=2.0 most linear, scale=0.5 most aggressive compression). Per-split: single_in_dist −12.5%, geom_camber_cruise −11.0%, re_rand −8.2%, geom_camber_rc −6.4%.

- **Analysis:** Clear winning direction but on the wrong baseline. Current baseline is 60.72/51.96 (SwiGLU+epochs=12). asinh is orthogonal to SwiGLU (target transform vs architecture). Re-testing asinh_scale=2.0 AND 3.0/4.0 on the SwiGLU stack — potentially transformative (if −9.5% test carries over: 51.96 × 0.905 ≈ test=47).
- **Closed** at 12:45 UTC. Re-test assigned as PR to edward (swiglu-asinh).

---

## 2026-05-16 12:15 — PR #3833: OneCycleLR schedule (thorfinn) — **CLOSED** (re-testing on SwiGLU)

- **Student:** willowpai2i48h4-thorfinn
- **Hypothesis:** OneCycleLR (max_lr=1e-3, pct_start=0.3, cycle_momentum=False) with super-convergence on 12-epoch budget.
- **Results (on pre-SwiGLU MLP baseline 82.50/74.10):**
  - Arm 1 (max_lr=1e-3, `z3nj8xpe`): val=77.52/test=67.68 — **−8.7% test** ✓
  - Arm 2 (max_lr=5e-4, `ou3tbyhc`): val=80.15/test=70.53 — −4.8% test
- **LR curve Arm 1:** initial=1e-4 → peak=1e-3 (step ~1338, 30% of budget) → final ~9e-5. Scheduler correctly per-batch.
- **Analysis:** Clear win; max_lr=1e-3 engages super-convergence. Gains uniform across all 4 splits. OneCycleLR is orthogonal to SwiGLU (schedule vs architecture). Re-testing on SwiGLU stack — thorfinn assigned #3951 (swiglu-onecycle).
- **Closed** at 12:15 UTC.

---

## 2026-05-16 11:30 — PR #3814: SwiGLU FFN in TransolverBlock (askeladd) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/askeladd-swiglu-ffn`)
- **Hypothesis:** Replace the standard 2-layer Linear-GELU-Linear FFN (`self.mlp`) in each `TransolverBlock` with a SwiGLU gated unit: `silu(W1x) * W2x → W3`. The gate lets each token selectively suppress or amplify feature directions — a natural fit for CFD where adjacent nodes live in very different physical regimes (boundary-layer vs. wake vs. freestream). Parameter count is matched via `inner_dim = round_to_mult(hidden_dim * mlp_ratio * 2/3, 8) = 216`. `mlp2` (output head) was correctly left as standard MLP (changing its shape would break output dim 160→3).

### Results (W&B run `dvcj6w25`, best epoch 10/10)

| Metric | Baseline (#3691, 82.50) | SwiGLU (dvcj6w25) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 82.4997 | **64.2430** | **−18.26 (−22.1%) 🏆** |
| **test_avg/mae_surf_p** | 74.1023 | **55.5454** | **−18.56 (−25.0%) 🏆** |

Per-split:

| Split | val SwiGLU | val baseline | test SwiGLU | test baseline |
|---|---:|---:|---:|---:|
| single_in_dist | 71.8437 | 90.995 | 64.1005 | 83.128 |
| geom_camber_rc | 74.3348 | 91.548 | 66.0329 | 82.735 |
| geom_camber_cruise | 46.4804 | 65.790 | 37.6118 | 56.332 |
| re_rand | 64.3132 | 81.665 | 54.4363 | 74.215 |
| **avg** | **64.2430** | **82.4997** | **55.5454** | **74.1023** |

Every split improved substantially. Largest gains on `single_in_dist` and `geom_camber_rc` — the two splits with highest baseline error, consistent with gate-based FFN helping most where features from different physical regimes must be distinguished.

Model: 1,035,703 params (+0.7% vs baseline 1.03M). Reproducibility: second seed `msnk1t8p` gave val=64.55/test=55.84 (within 0.3 — not a seed fluke). Best val still at last epoch (10/10) — model still converging at budget end; gains are likely underestimated.

### Analysis

**Largest single-experiment gain in the programme history.** SwiGLU's gated activation (`silu(W1x) * W2x → W3`) directly addresses the multi-regime challenge in CFD surrogates: the same FFN must simultaneously handle boundary-layer nodes, far-field freestream nodes, and wake nodes. The gating mechanism lets each token suppress irrelevant feature directions, which a standard GELU FFN cannot do. The biggest improvements landing on the high-error, geometry-varied splits (single_in_dist, geom_camber_rc) are fully consistent with this interpretation.

Student's smart deviation: correctly identified that `mlp2` is an output head (shape 160→3), not a per-block FFN, and left it as-is. Replacing it with SwiGLU shape 160→3 would have broken the output contract.

**Next steps triggered:** (1) SwiGLU + epochs=12 — val still descending at epoch 10; (2) SwiGLU + mlp_ratio=3 — gated FFNs often tolerate wider inner dims than vanilla; (3) SwiGLU + attn_dropout=0.1 — combine the two strongest unexplored regularization axes.

**Merged** at 11:30 UTC. New baseline: val=64.24/test=55.55. SwiGLU FFN is now default.

---

## 2026-05-16 11:30 — PR #3838: Per-domain output normalization (alphonse) — **CLOSED**

- **Student:** willowpai2i48h4-alphonse
- **Hypothesis:** Domain-specific y_mean/y_std normalization to equalize gradient scale across domains with wildly different y distributions (raceCar single std ~2077, cruise ~506).
- **Results:** val=89.28 (vs baseline 82.50 → +6.78; vs new SwiGLU baseline 64.24 → +25.0). FAIL on both metrics.
- **Analysis:** The gap/aoa2 heuristic for domain detection couldn't cleanly separate raceCar tandem from cruise. Domain-specific stats added non-stationarity to the gradient signal without a clear boundary condition that generalizes. Not worth iterating given the magnitude of the SwiGLU win — GPU better spent on SwiGLU follow-ups.
- **Closed** at 11:30 UTC.

---

## 2026-05-16 11:30 — PR #3741: eta_min=1e-5 cosine LR floor (fern) — **CLOSED**

- **Student:** willowpai2i48h4-fern
- **Hypothesis:** Setting eta_min=1e-5 (non-zero cosine floor) keeps meaningful gradients in the final epochs rather than decaying LR to zero.
- **Results:** 3 seeds, all regress. Best: `u0nphp8l` val=86.21 (vs baseline 82.50 → +3.71; vs new SwiGLU baseline 64.24 → +21.9). Mean across 3 seeds ~86.7. FAIL.
- **Analysis:** The cosine floor is not the binding constraint. SwiGLU just delivered −22% by changing the FFN architecture. The LR schedule horizon is now correctly co-designed via `--epochs 12` (merged in #3691). Additional eta_min tuning is not a productive direction. Closing to free the GPU for SwiGLU follow-ups.
- **Closed** at 11:30 UTC.

---

## 2026-05-16 05:20 — Round-3 retry closures (#3633, #3634, #3636, #3638, #3479)

After PR #3632 (coord noise) merged as new baseline (val=83.50/test=73.79), all 5 remaining round-3 retries finished and regressed vs the new baseline:

| PR | Student | Hypothesis | Final run | val | test | Δ val vs 83.50 |
|---|---|---|---|---:|---:|---:|
| #3633 | askeladd | Learnable Fourier freqs | `z2kg48ty` | 87.97 | 77.25 | +5.4% |
| #3634 | fern | slice_num=96 | `fagaonns` | 89.10 | 78.29 | +6.7% |
| #3636 | nezuko | num_freq=2 | `2fnr2k1z` | 88.51 | 77.51 | +6.0% |
| #3636 | nezuko | num_freq=6 | `xvtmrakm` | FAILED | — | crashed at startup (20s) |
| #3638 | alphonse | p_weight=3 | `fort2r4i` | 85.35 | 75.70 | +2.2% (best of round) |
| #3479 | frieren | per-channel + lr=1e-3 | `5lcpht9s` | 88.55 | 79.33 | +6.1% |

### Analysis

**The coord noise merge raised the bar significantly.** All techniques that were promising vs the OLD baseline (88.24) failed to beat the NEW baseline (83.50). Several lessons:

1. **Learnable Fourier freqs (askeladd)** — Same as fixed (+5.4% worse). The fixed log-spaced freqs were near-optimal; the few extra trainable params didn't get enough gradient signal in 10 epochs.

2. **slice_num=96 (fern)** — +6.7% worse. Combined with slice_num=128 (#3092, val=106.82), confirms slice_num=64 is the sweet spot. More slice tokens fragment attention compute too thin without enough epochs.

3. **num_freq=2 (nezuko)** — +6.0% worse. Fewer frequencies = less representational granularity. num_freq=4 confirmed as the optimum. num_freq=6 crashed at startup (likely dim mismatch in preprocess MLP; not worth debugging since num_freq=2 already eliminates the lower bracket).

4. **p_weight=3 (alphonse)** — +2.2% worse. BEST of the round; closest to baseline. The pressure channel emphasis competes with the L1 + Fourier PE inductive biases. surf_weight (global, not per-channel) is the cleaner mechanism — assigned to alphonse next.

5. **Per-channel heads + lr=1e-3 (frieren)** — +6.1% worse. The technique that helped on pre-Fourier-PE stacks (~val 95 → 95.6) didn't transfer. Both per-channel heads and coord noise target output-side optimization; they don't compose cleanly.

### Cross-cutting takeaway

**Coord noise augmentation absorbed much of the upside that round-3 architecture/loss experiments would have provided.** The post-coord-noise baseline (83.50) is a harder target than the post-Fourier-PE baseline (88.24). Round-4 must target genuinely orthogonal axes — augmentation variants, FFN capacity, attention head diversity, surface weighting. Architecture width/depth scaling is exhausted at this budget.

All 5 PRs closed. 5 students reassigned to round-4 hypotheses.

---

## 2026-05-16 04:30 — PR #3637: Width n_hidden=176 (thorfinn) — CLOSED

- **Student:** willowpai2i48h4-thorfinn
- **Hypothesis:** n_hidden=176 is the sweet spot between the working 160 and the failing 192, giving +10% width with +20% params.

### Results (W&B run `7zjst4wu`)

| Metric | Baseline (#3632, 83.50) | n_hidden=176 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 83.4954 | 88.4539 | **+5.0 (vs old baseline 88.24: +0.21)** |
| test_avg/mae_surf_p | 73.7918 | 79.2911 | **+5.50** |

Ran against the OLD baseline at val=88.24 (submitted before #3632 coord noise was merged). Against THAT baseline: val=88.45 (+0.21, essentially noise-level regression). Against current 83.50: +5.0 (clearly worse).

### Analysis

Width scaling is confirmed plateaued at n_hidden=160 for this 30-min budget. n_hidden=176 (+10% width) gives both val and test regressions. Both sub-192 widths (176, 192) have been tested and both regress. The model is not capacity-limited in width — it's under-trained in time. Depth also fails at budget. The winning lever going forward is **data/augmentation** (coord noise proved this) and **loss/pe engineering** (Fourier PE proved this).

**Closed** — width scaling exhausted at n_hidden=160 for current budget.

---

## 2026-05-16 04:30 — PR #3635: Depth n_layers=6 on current stack (edward) — CLOSED

- **Student:** willowpai2i48h4-edward
- **Hypothesis:** n_layers=6 on the full current stack (Fourier PE + L1 + n_hidden=160) might give gains that were masked in the stale #3469 experiment.

### Results (W&B run `4vmya3cn`)

| Metric | Baseline (#3372, 88.24) | n_layers=6 (8ep) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 88.2442 | 94.5011 | **+6.26 ✗** |
| test_avg/mae_surf_p | 77.0880 | 83.5150 | **+6.43 ✗** |

Per-split: single_in_dist+14.95 (worst), geom_camber_cruise +1.31 (best), re_rand +2.97. Every split regressed.

### Analysis

Depth=6 at --epochs 8 (budget constraint) is under-converged: the extra block needs more gradient steps to learn meaningful higher-order cross-slice interactions. geom_camber_cruise nearly held neutral (+1.31) — the extra depth might help if training epochs could increase. Not viable at 30-min budget. Depth scaling would require 20+ epochs or a curriculum/pretraining approach.

**Closed** — confirms depth scaling is budget-constrained at 30min window. Consistent with #3469.

---

## 2026-05-16 04:30 — PR #3632: Coordinate noise augmentation std=0.01 (tanjiro) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-tanjiro
- **Hypothesis:** Gaussian jitter (std=0.01) on normalized (x,z) coords during training only gives richer geometry variation each epoch, improving OOD generalization.

### Results (W&B run `0q6t1hpc`)

| Metric | Old baseline (#3372, 88.24) | Coord noise (0q6t1hpc) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 88.2442 | **83.4954** | **−4.75 (−5.38%) 🏆** |
| **test_avg/mae_surf_p** | 77.0880 | **73.7918** | **−3.30 (−4.28%) 🏆** |

Per-split test: single_in_dist 83.77 (−4.68%), geom_camber_rc 80.55 (−2.60%), geom_camber_cruise 55.20 (−7.08%), re_rand 75.64 (−3.47%). Improvement on every split.

Config: n_hidden=160, n_layers=5, Fourier PE num_freq=4, L1 loss, coord_noise_std=0.01 (train only), lr=5e-4 (Config default — note: NOT the lr=1e-3 used in #3372; testing lr=1e-3 with coord noise is an open opportunity).

### Analysis

Second-largest single-experiment gain in the track (+5.38% val, after Fourier PE +8.2%). Coord noise acts as implicit mesh-topology augmentation: the model sees slightly different geometry each epoch, forcing it to learn the physics rather than memorize mesh coordinates. The cruise split gained most (−7.08% test) — consistent with cruise shapes having highest geometry variability.

Key insight: lr=5e-4 (default) was used, NOT lr=1e-3 (the prev baseline lr). Testing lr=1e-3 + coord noise is an open compounding experiment.

**Merged** at 04:30 UTC. New baseline: val=83.50/test=73.79. coord_noise_std=0.01 is now default.

---

## 2026-05-16 02:25 — PR #3372: Fourier PE 4-freq on (x,z) coords (askeladd) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-askeladd (branch: `askeladd/fourier-pos-encoding`)
- **Hypothesis:** Replace raw (x, z) coordinates with NeRF-style log-spaced sinusoidal Fourier positional encoding (num_freq=4) to give the model multi-scale geometric receptive field beyond linear interpolation.

### Results (W&B run `qyc68z5k`)

| Metric | Old baseline (#3507, 96.10) | This run (Fourier PE 4-freq) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 96.0997 | **88.2442** | **−7.86 (−8.2%) 🏆** |
| **test_avg/mae_surf_p** | 85.5256 | **77.0880** | **−8.44 (−9.9%) 🏆** |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 87.8840 |
| geom_camber_rc | 82.7020 |
| geom_camber_cruise | 59.4070 |
| re_rand | 78.3590 |
| **avg** | **77.0880** |

Config: L1 loss, warmup 2ep, cosine T_max=10, lr=1e-3, n_hidden=160, n_layers=5, n_head=4, slice_num=64, Fourier PE num_freq=4. `fun_dim` grows from 22 to 38 (4×num_freq sinusoidal features per coord pair). Per-epoch ~168s (unchanged — PE is just a preprocess step). ENCODED_X_DIM=38.

### Analysis

Fourier PE gave the **largest single-experiment gain on the track** — larger than L1 loss (−8.1% val) and larger than width-160 (−4.4% val). The improvement is concentrated in the cruise and OOD splits: `test_geom_camber_cruise` went from 61.38 → 59.41 (−3.2%), `test_re_rand` from 84.55 → 78.36 (−7.3%), `test_single_in_dist` from 103.75 → 87.88 (−15.3%). This pattern is consistent with the hypothesis: Fourier features encode geometry at multiple scales simultaneously, helping the model generalize to unseen geometries (OOD) rather than memorizing the training mesh topology.

The gain was earned cheaply: no architecture change, no extra parameters (only the input layer of the preprocess MLP grows by 14 weights), no training time overhead.

**Merged** at 02:25 UTC as new baseline. num_freq=4 is now `Config` default. All in-flight students need to rebase to get the new `ENCODED_X_DIM` computation.

---

## 2026-05-16 02:30 — PRs #3490/#3508/#3524/#3552/#3288: Round-2 experiment sweep — all CLOSED

Five PRs closed in this batch, all regressing vs the new baseline val=88.24. Summary:

| PR | Student | Hypothesis | Best val | Best test | Δ vs 88.24 |
|---|---|---|---:|---:|---:|
| #3490 | nezuko | L1 LR sweep {3e-4, 2e-3, 4e-3} | 98.88 (lr=2e-3) | 87.75 | +10.64 |
| #3508 | fern | Cosine warm restarts SGDR T_0=4 | 100.79 | 90.63 | +12.55 |
| #3524 | thorfinn | Huber loss β=1.0 | 101.44 (`oj7zwn3z`) | 90.14 | +13.20 |
| #3552 | alphonse | Width n_hidden=192 (--epochs 8) | 102.73 | 92.16 | +14.49 |
| #3288 | edward | Scoring fix + lr default verify | 96.53 | 86.62 | +8.29 (superseded) |

**Conclusions:**
- **lr=1e-3 is optimal**: both lower (3e-4) and higher (2e-3, 4e-3) LR with L1 are worse. No exploration needed here.
- **SGDR warm restarts hurt**: LR resets disrupt the still-descending loss curve at 10 epochs. Cosine-to-zero stays canonical.
- **Huber β=1.0 worse than L1**: L1 directly optimizes MAE, Huber doesn't. L1 locked as canonical.
- **Width 192 over-parameterized at --epochs 8**: Too much capacity for the 30-min budget. n_hidden=176 is the next candidate.

---

## 2026-05-16 01:38 — PR #3469: Deeper model n_layers 5→6 (tanjiro) — CLOSED

- **Student:** willowpai2i48h4-tanjiro (branch: `tanjiro/depth-6layers`)
- **Hypothesis:** One additional Transolver block (n_layers 5→6) provides higher-order cross-slice interactions on the airfoil geometry; predicted improvement on val_avg/mae_surf_p, especially on cross-regime splits.

### Results (W&B run `5y4w4b45`)

| Metric | Stale ref (#3091, 109.42) | This run (n_layers=6) | Δ stale | Δ new baseline (96.10) |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 109.4166 | **108.4452** | −0.97 ✓ | **+12.34 ✗** |
| test_avg/mae_surf_p (3-split workaround) | 107.4694 | 105.2823 | −2.19 | — |
| test_avg/mae_surf_p (full) | NaN | NaN | — | — |

Per-split val (n_layers=6 vs old baseline 109.42):
- val_re_rand: −10.51 (big win); val_geom_camber_*: −3.51/−4.17 (modest gain); val_single_in_dist: +14.29 (regression).

### Analysis

The run was completed against the STALE pre-#3089 codebase (MSE loss, n_hidden=128). The reported −0.97 win against the #3091 baseline (109.42) is real but the experiment was never validated on the post-#3507 advisor (val=96.10, with L1 + n_hidden=160). The depth-6 result (108.45) is +12.34 above the current baseline, so even a generous re-run would need to gain >12 to land on the merge curve — vs an observed +1 gain in the stale ablation, this is implausible.

Useful signals captured for future depth work on the new baseline:
- depth-6 reliably helps `val_re_rand` and `val_geom_camber_*` on tandem-cruise OOD tracks
- depth-6 hurts `val_single_in_dist` by +14 — capacity overfits single-foil distribution
- ~158s/epoch at n_hidden=128 + n_layers=6 (vs ~168s for n_hidden=160 + n_layers=5) — depth is cheaper than width per epoch
- Loss curves still descending at epoch 10 → under-trained

**Closed** at 01:38 UTC as the stale-baseline regression is too large to bridge. tanjiro reassigned to a fresh experiment on the current advisor tip.

---

## 2026-05-16 00:30 — PR #3507: Width scaling n_hidden 128→160 (alphonse) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/alphonse-width-160`)
- **Hypothesis:** Increasing Transolver hidden width from 128 to 160 (+25% capacity, +56% params) on top of the L1 + warmup + clip stack will improve val_avg/mae_surf_p further; cosine schedule still fully anneals at the slightly slower ~168s/epoch.

### Results (W&B run `7vxhbv8o`)

| Metric | Old baseline (#3089, 100.53) | This run (n_hidden=160) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 100.5275 | **96.0997** | **−4.40 (−4.4%) 🏆** |
| **test_avg/mae_surf_p** | 90.1489 | **85.5256** | **−4.62 (−5.1%) 🏆** |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 103.7483 |
| geom_camber_rc | 92.4243 |
| geom_camber_cruise | 61.3787 |
| re_rand | 84.5510 |
| **avg** | **85.5256** |

Config: L1 loss (carry-over from #3089), warmup 2 ep, cosine to 0 (T_max=10), grad-clip 1.0, lr=1e-3, batch=4, surf_weight=10. Width 128→160; params 662k → 1.03M. Per-epoch ~168s (↑ from ~134s); peak VRAM 50.1 GB (53% of 96 GB envelope).

### Analysis

Width-160 composes cleanly with the merged optimization stack and delivers the expected gain on both val and test. Improvement is broadly distributed across all 4 test splits (no per-split regression). Val curves still descending at epoch 10 → continued width scaling is likely net-positive, but with diminishing returns expected past ~192 given the budget-constrained 10-epoch annealing.

**Merged** at 00:30 UTC as new advisor baseline. All in-flight students need to rebase to inherit `Config.n_hidden = 160`.

---

## 2026-05-15 22:35 — PR #3095: Higher surf_weight + per-channel p weighting (nezuko) — CLOSED

- **Student:** willowpai2i48h4-nezuko (branch: `willowpai2i48h4-nezuko/surface-weight`)
- **Hypothesis:** Increasing `surf_weight` from 10 to 20-30 pushes the optimizer to focus more on surface pressure prediction; adding per-channel `p` weighting (3×) further boosts the primary metric.

### Results — rebased confirmation arm (surf_weight=20, W&B run `6amjj7jr`)

| Metric | Value | Δ vs baseline (109.42) | Δ vs new baseline (100.53) |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 111.92 | +2.3% ✗ | +11.3% ✗ |
| test_avg/mae_surf_p | 97.70 | (was NaN) | +8.4% ✗ |

Earlier arm sweep (stale pre-#3091 code):

| Arm | Config | val_avg/mae_surf_p |
|---|---|---:|
| A (surf_weight=30) | surf_w=30 | 131.08 |
| B (pchan=3) | surf_w=10, p_w=3 | 133.31 |
| C (combined) | surf_w=30, p_w=3 | 146.42 |

### Analysis

All arms regressed vs both old and new baselines. surf_weight=20 on the rebased (warmup+clip+lr=1e-3+L1) code gave val=111.92 — nominally +2.3% above the 109.42 old baseline, and decisively worse (+11.3%) against the current 100.53 baseline. surf_weight=30 and the pchan knob are both clearly worse. The student's per-split analysis shows the regression concentrated in `single_in_dist` while `re_rand` and `geom_camber_rc` actually improve — suggesting surf_weight tuning shifts a per-split tradeoff rather than moving the aggregate down.

The train.py NaN fix in `evaluate_split` was correctly implemented; subsumed by alphonse #3089 merge.

**Closed** at 22:35 UTC. surf_weight hypothesis exhausted at {10, 20, 30}; optimum appears at ≤10. Nezuko reassigned to L1 LR sweep.

---

## 2026-05-15 22:31 — PR #3089: L1 loss + NaN scoring fix (alphonse) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/l1-loss`)
- **Hypothesis:** Replacing MSE loss with L1 in the normalized-prediction space aligns the training objective with the MAE evaluation metric; predicted −5% to −10% on `val_avg/mae_surf_p`.

### Results — final rebased confirmation arm (W&B run `14w7wdyb`, `alphonse-l1-rebased`)

| Metric | Value | Δ vs baseline (109.42) |
|---|---:|---:|
| **val_avg/mae_surf_p** | **100.5275** | **−8.1% ✓** |
| **test_avg/mae_surf_p** | **90.1489** | first clean finite number |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 112.07 |
| geom_camber_rc | 98.04 |
| geom_camber_cruise | 64.21 |
| re_rand | 86.28 |
| **avg** | **90.15** |

Config: L1 loss (`Config.loss_type = "l1"`, default flipped) + warmup + grad-clip + lr=1e-3 + 10 epochs (fully annealed cosine). Also includes `_pointwise_loss` helper for MSE/L1/Huber dispatch and `torch.isfinite` per-sample mask in `evaluate_split` (canonical scoring NaN fix).

### Analysis

L1 loss clearly improves val_avg/mae_surf_p (−8.1%) and delivers the first clean test metric. The composition of L1 with warmup+clip+lr=1e-3 works well — the levers are orthogonal (L1 addresses objective mismatch; warmup+clip+lr addresses optimization stability). Single best experiment so far by absolute Δ.

The scoring NaN fix is particularly valuable — `test_avg/mae_surf_p = 90.15` is now a reliable paper-facing metric for all future runs.

**Merged** at 22:31 UTC as new baseline. All in-flight students need to rebase to get L1 default + scoring fix.

---

## 2026-05-15 21:37 — PR #3414: SWA (stochastic weight averaging) over last K checkpoints

- **Student:** willowpai2i48h4-tanjiro (branch: `tanjiro/swa-checkpoint-averaging`)
- **Hypothesis:** Averaging the weights of the last K checkpoints (K=3 or K=5) produces a smoother loss landscape than any single checkpoint, reducing overfitting and improving val_avg/mae_surf_p.

### Results

| Arm | SWA window | raw val_avg/mae_surf_p (best ckpt) | swa_val_avg/mae_surf_p | Δ vs 109.42 baseline | W&B run |
|---|---|---:|---:|---:|---|
| A — SWA last 5 | K=5 (epochs 6–10) | **103.72** | 111.09 | +1.5% ✗ | `gduowc1p` |
| B — SWA last 3 | K=3 (epochs 8–10) | **108.01** | 109.48 | ~flat (+0.06%) | `udfmekyw` |

**SENPAI-RESULT (terminal):** `swa_val_avg/mae_surf_p = 109.48`, `swa_test_avg 3-split = 106.34` (Arm B).

### Analysis

SWA did NOT improve the primary metric on either arm. The SWA-averaged checkpoint was consistently **worse** than the best raw checkpoint in both arms. The mechanism is clear: with `--epochs 10` and cosine annealing to 0, the loss is still descending at the final epoch. Averaging the last K checkpoints includes sub-optimal earlier states from the middle of descent, which drags the average above the best single checkpoint.

The Arm A raw val (103.72) is better than baseline but that's just run variance — it's an unintended observation from a re-run of the baseline config. The proposed feature (SWA averaging) consistently regressed both arms.

**Conclusion:** SWA is only beneficial when the training curve has plateaued — which requires more epochs than our 30-min budget allows at the current batch size. The experiment correctly identified this limitation in the writeup.

**Closed as dead end** at 21:37 UTC. New hypothesis assigned to tanjiro: depth n_layers=5→6 (#3469).

## 2026-05-15 14:07 — PR #3092: More physics-attention slice tokens (slice_num 64→128, 192)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/more-slices`)
- **Hypothesis:** Doubling `slice_num` from 64 to 128 raises the resolution of Transolver's physics decomposition over 74K–242K node meshes; predicted −3% to −7% on `val_avg/mae_surf_p`.

### Results

| Arm | slice_num | n_params | best val_avg/mae_surf_p | best epoch | total epochs | peak VRAM | epoch time | W&B run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (winner) | 128 | 672,919 | **150.26** | 9 | 10 | 54.5 GB | 171 s | `yiiy92uj` |
| B | 192 | 683,479 | 153.71 | 9 | 10 | 68.4 GB | 213 s | `l7nnvr53` |

Per-split val surface pressure MAE (best ckpt, epoch 9):

| Split | Arm A (128) | Arm B (192) |
|---|---:|---:|
| val_single_in_dist | 185.70 | 183.15 |
| val_geom_camber_rc | 157.16 | 179.11 |
| val_geom_camber_cruise | 127.68 | **115.40** |
| val_re_rand | 130.51 | 137.20 |
| **val_avg/mae_surf_p** | **150.26** | 153.71 |

Per-split test: `test_geom_camber_cruise/mae_surf_p = None / NaN` on **both arms** (vol_loss=Infinity), poisoning `test_avg/mae_surf_p`. The student reported a 3-split mean (excl. cruise) of 144.76 (A) / 152.56 (B), but this is not the contract metric.

### Analysis & Verdict — sent back (not merged)

- Arm A (slice_num=128) beats Arm B (slice_num=192) by 3.45 on `val_avg/mae_surf_p` (−2.2% absolute), and is significantly cheaper (−20% VRAM, −20% epoch time). Higher slice_num does NOT help in this short-training regime — likely the optimization burden of more slice assignments to learn outweighs the gain.
- **No baseline number on this branch** — we cannot establish that `slice_num=128` improves on the actual baseline `slice_num=64`. The PR documents what beats what *internally* but not against the actual reference.
- `test_avg/mae_surf_p` is NaN — fails the full-metric-fidelity contract from CLAUDE.md.

### Critical cross-cutting finding: LR schedule is mis-tuned for the wall-clock budget

The student's most valuable observation: with `SENPAI_TIMEOUT_MINUTES=30` and ~170s/epoch, only ~10 epochs of the configured 50-epoch cosine schedule complete. `T_max=50` means LR is still at ~80% of peak when training stops — **no experiment on this branch is getting LR annealing**. This affects every other in-flight PR (#3089, #3090, #3091, #3093, #3095, #3096, #3097). Future PRs should pass `--epochs 10` (or whatever matches actual completed-epoch count) so `T_max` matches budget.

### NaN on `test_geom_camber_cruise/mae_surf_p`

The model emits inf/NaN predictions on at least one sample in the cruise test split when evaluated from a partial-training checkpoint. Identical across both arms, so doesn't affect this PR's A-vs-B comparison. Likely fixable by training to convergence (with proper LR annealing), gradient clipping, or `torch.nan_to_num` band-aid. Edward's PR #3091 (grad clip) and alphonse's PR #3089 (L1 loss) may both address this naturally.

### Follow-up (sent back to fern as comment on #3092)

Run 2-arm comparison at `--epochs 10` to fully anneal cosine `T_max`:
- Arm A: `slice_num=64` (establishes the branch baseline)
- Arm B: `slice_num=128` (confirms with proper schedule)

Merge if Arm B beats Arm A on `val_avg/mae_surf_p` AND `test_avg/mae_surf_p` is finite on Arm B.

---

## 2026-05-15 14:38 — PR #3091: LR warmup + gradient clipping + higher peak LR (edward) — **MERGED**

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/warmup-clip`)
- **Hypothesis:** Adding 2-epoch linear warmup and gradient clipping (max_norm=1.0) stabilizes training and enables higher peak LR. Arm B tested lr=1e-3 (2× baseline 5e-4). Predicted delta: −3% to −8%; actual win was >10%.

### Results

| Arm | lr | best epoch | val_avg/mae_surf_p | test (3-split workaround) | W&B run |
|---|---|---|---|---|---|
| A (warmup+clip+5e-4) | 5e-4 | 13 | 121.54 | 124.19 | `qm3lqtwz` |
| **B (warmup+clip+1e-3)** | 1e-3 | 14 (last) | **109.42** | **107.47** | `0ez1sqmi` |

Per-split val surface pressure MAE (best ckpt, epoch 14):

| Split | Arm A | Arm B |
|---|---:|---:|
| val_single_in_dist | 184.40 | 119.58 |
| val_geom_camber_rc | 115.04 | 119.40 |
| val_geom_camber_cruise | 88.03 | 88.57 |
| val_re_rand | 104.43 | 110.12 |
| **val_avg/mae_surf_p** | 121.54 | **109.42** |

Test: NaN on `test_geom_camber_cruise` for both (scoring bug). 3-split workaround: 124.19 (A) / 107.47 (B).

### Analysis & Decision — MERGED

- **Decisive win.** Arm B beats Arm A by 12.1 on val_avg (−10%) and by 16.7 on test 3-split (−13%). Pre-clip grad norm was 160 vs 14 at the last step — clipping is doing real work.
- Arm B's best epoch = 14/14 (last): model was still strictly improving when the timeout cut it, indicating significant headroom at longer training.
- `warmup_epochs=2` over 14 effective epochs = ~14% warmup, higher than intended. Short warmup is still the right call at high LR — doesn't hurt.
- Code change is minimal: 20 lines, adds warmup lambda scheduler + clip + grad_norm logging. Clean, composable with all other experiments.
- **Merged as new branch baseline: val_avg/mae_surf_p = 109.42** (lr=1e-3 + warmup + clip).

### Follow-up (edward)

Assigned edward a bug-fix + consolidation PR:
- Unblock `test_avg/mae_surf_p` by nan_to_num fix in `evaluate_split` (avoids `0 * NaN = NaN` propagation in accumulate_batch)
- Bump Config.lr default from 5e-4 to 1e-3 to lock in winning config for all future students

---

## 2026-05-15 15:30 — PR #3089: L1 loss vs Huber β=1.0 (alphonse) — **SENT BACK** (close to merge)

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/l1-loss`)
- **Hypothesis:** Replace MSE with L1 loss in normalized space; align training objective with MAE metric. Predicted −8% to −15%.

### Results

| Arm | Loss | best epoch | val_avg/mae_surf_p (W&B-verified) | test_avg/mae_surf_p (claim) | W&B run |
|---|---|---|---|---|---|
| **A (winner)** | L1 | 13 | **102.37** | 89.67 (offline re-eval) | `lb2ly5g3` |
| B | Huber β=1.0 | 13 | 117.47 | 106.03 | `9gh0e13m` |

Per-split val surface pressure MAE (Arm A, best epoch 13, alphonse's report):

| Split | Arm A (L1) | Arm B (Huber) |
|---|---:|---:|
| val_single_in_dist | 133.71 | 138.99 |
| val_geom_camber_rc | 108.91 | 118.50 |
| val_geom_camber_cruise | 76.50 | 102.26 |
| val_re_rand | 90.37 | 110.13 |
| **val_avg/mae_surf_p** | **102.37** | 117.47 |

W&B verification (subagent):
- Arm A: val_avg/mae_surf_p = 102.37 (best_val_avg) ✓ VERIFIED — beats baseline (109.42) by **−6.4%**.
- Arm A: test_avg/mae_surf_p = `None` in W&B summary; alphonse's claimed 89.67 came from offline re-eval after adding the fix post-training.
- Arm A: val_geom_camber_cruise/mae_surf_p = 84.79 in W&B (real number) ← alphonse's nan_to_num/sub-select fix DOES work for val.
- 3-split test mean (excl. cruise): test_re_rand=86.10, test_geom_camber_rc=96.20, test_single_in_dist=111.43 → mean ≈ **97.91** vs edward's 107.47.

### Bug-fix included

Alphonse correctly identified the `0 * NaN = NaN` propagation in `accumulate_batch` (same as edward's #3288 but more robust — handles both NaN and Inf in y via `torch.isfinite` + sub-select fully finite samples).

### Decision — Sent back with two specific asks

1. **Flip Config default `loss_type: str = "mse"` → `"l1"`** so future runs compose on L1 automatically.
2. **Push a clean W&B-logged eval with the fix in place** so `test_avg/mae_surf_p` is verifiable (not just offline re-eval). Quick `--debug --epochs 1` pass is sufficient.

After: transition to `status:review`, mark ready, merge.

### Composition with #3091

Alphonse trained on `lr=5e-4` + no warmup + no clip (pre-#3091 advisor branch). When merged into post-#3091 branch, future runs get: `lr=1e-3 + warmup + clip + L1`. Likely further headroom — these changes are orthogonal.

### Coordination with #3288

Alphonse's scoring fix (sub-select + torch.where) is more robust than edward's (`nan_to_num`). When alphonse's PR merges first, edward's #3288 should drop the duplicate scoring fix and only keep the lr default bump.

---

## 2026-05-15 17:30 — PR #3096: x-axis reflection symmetry augmentation (tanjiro) — **SENT BACK** (regression, conditional re-run)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/xflip-aug`)
- **Hypothesis:** Per-sample x-flip aug with Ux/AoA/stagger negation; predicted gains on geom_camber OOD splits.
- **W&B run:** `a7kc6xxi` (verified)

### Results

| Arm | val_avg/mae_surf_p | test 3-clean-split | best epoch | total epochs |
|---|---:|---:|---:|---:|
| Single arm (xflip aug) | **161.54** | 162.46 | 12 | 14 |

Compared to current baseline (109.42 from PR #3091): **+47% regression**. But branch was forked pre-#3091 (lr=5e-4, no warmup, no clip), so most of the gap is the stale-branch infrastructure. On the same pre-#3091 code, fern's slice_num=128 baseline (#3092) landed at val=150.26 — tanjiro is ~7% worse than that with augmentation.

Per-split val surface MAE (best epoch 12):

| Split | Tanjiro xflip | fern slice_num=128 (same code) |
|---|---:|---:|
| val_single_in_dist | **203.61** | 185.70 |
| val_geom_camber_rc | 173.37 | 157.16 |
| val_geom_camber_cruise | 125.17 | 127.68 |
| val_re_rand | 143.99 | 130.51 |
| **val_avg/mae_surf_p** | **161.54** | 150.26 |

### Three concerning signals

1. **Model peaked at epoch 12 and rose for epochs 13–14** (163.0 → 167.0). The wall clock didn't cut mid-improvement; the model was overfitting. With higher LR (lr=1e-3 in current advisor) it'll likely overfit even earlier.
2. **`val_single_in_dist = 203.61` is the WORST split** — the easiest split (in-distribution) is being hurt by augmentation. xflip is making in-dist samples harder while only marginally helping OOD.
3. **`val_geom_camber_cruise` ≈ identical to fern's number** (125.17 vs 127.68). The predicted OOD gain isn't showing up in absolute numbers; the relative-easier-than-in-dist signal is plausible but not symmetry-specific.

### Bug-fix analysis

Tanjiro independently identified the same `0 * NaN = NaN` propagation in `accumulate_batch` that edward and alphonse flagged. Same root cause, same path (read-only `data/scoring.py`).

### Decision rule for the rebased confirmation arm

- val < 109.42 → merge
- val ∈ [109.42, 115] → merge only if geom_camber_cruise is clearly the best split (OOD-aug story still holds)
- val > 115 → close. Hypothesis empirically unsupported at this scale.

### Notes

- Augmentation halves effective gradient signal per orientation; could benefit from longer schedule, but within 30-min budget the unaugmented baseline gets twice the effective per-orientation samples.
- Symmetry aug is theoretically sound; the result here is most likely an interaction with: (a) stale code, (b) wall-clock cap, (c) MSE loss (L1 might compose better with aug). Worth re-investigating in round 2 stacked with alphonse's L1 + edward's warmup.

---

## 2026-05-15 17:30 — PR #3097: Deeper Transolver n_layers 5→8 + DropPath

- **Student:** willowpai2i48h4-thorfinn (branch: `willowpai2i48h4-thorfinn/deeper-droppath`)
- **Hypothesis:** n_layers 5→8 + DropPath p=0.1 for regularized depth scaling; predicted −5% to −10% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Arm | n_layers | drop_path | params | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p | epoch time | VRAM | W&B run |
|-----|----------|-----------|--------|--------|------------------------|---------------------|-----------|------|---------|
| Baseline 5L | 5 | 0.0 | 0.66M | 14 | 132.73 | 121.78 | 132 s | 42.1 GB | `p1m774ow` |
| deep8-dp005 | 8 | 0.05 | 1.03M | 9 | **152.30** | **137.34** | 218 s | 64.5 GB | `qyyxx33r` |
| deep8-dp01 | 8 | 0.10 | 1.03M | 9 | 161.58 | 149.86 | 218 s | 64.5 GB | `jgaksniq` |

Current advisor baseline (from PR #3091): val_avg/mae_surf_p = **109.42**.

### Decision: CLOSED

Both deep arms are 40–48% worse than the advisor baseline (109.42). Vs student's own stale-code 5L reference (132.73), deep8-dp005 is still 15% worse. Student's bug-fix (cruise NaN workaround) is redundant with alphonse's #3089 fix.

### Analysis

Root cause: **wall-clock-budget undertraining, not capacity**. 8L is ~65% slower per epoch; within SENPAI_TIMEOUT_MINUTES=30 the deeper model completes only 9 epochs vs baseline's 14. Both deep arms peaked on their final epoch (still descending), classic signature of truncated training. Per-epoch comparison: wider model is actually better through epochs 5–8, but never reaches the post-convergence regime that baseline hits around epoch 12–14.

The hypothesis is **not refuted** — it's compute-bound. Under a 2× wall-clock budget, 8L+DropPath might win. Under the current budget, it cannot.

### Follow-up

Depth scaling is viable if we combine with a per-step speedup. Frieren's #3093 (bf16+bs=8, ~2× more epochs) could unlock this. Student reassigned to: **EMA of weights** (PR #3371) — addresses the late-epoch drift that affects all runs.

---

## 2026-05-15 17:30 — PR #3093: bf16 autocast + batch_size 4→8

- **Student:** willowpai2i48h4-frieren (branch: `willowpai2i48h4-frieren/bf16-amp`)
- **Hypothesis:** bf16 mixed precision + bs=8 for 1.5–2× wall-clock speedup → more epochs within 30-min budget; predicted −3% to −8% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Run | bs | precision | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p | epoch time | VRAM | W&B run |
|-----|----|-----------|--------|------------------------|---------------------|-----------|------|---------|
| frieren-bf16-bs8-v3 | 8 | bf16 | 18/50 | **128.70** (ep 15) | **117.22** | 104 s | 88.8 GB | `hxslyna3` |

Current advisor baseline (from PR #3091): val_avg/mae_surf_p = **109.42**.

### Decision: SENT BACK for rebased confirmation arm

128.70 is 17.5% worse than the 109.42 baseline, BUT this run used **stale code (lr=5e-4, no warmup, no clip)**. The speed unlock is genuine: 18 epochs vs ~14 at fp32+bs=4 stale, and training appears stable throughout (no overflow, no divergence).

The comparison is apples-to-oranges. Asked student to rebase onto current advisor tip (f3a71a2 = #3091 warmup+clip+lr=1e-3) and run `--epochs 10` for composed-config benchmark. Decision rule: val < 109.42 → merge; val ∈ [109.42, 115] → TBD; val > 115 → close.

### Per-split test MAE (best ckpt, stale-code run)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| test_single_in_dist | 142.48 | 2.18 | 0.86 |
| test_geom_camber_rc | 124.92 | 2.70 | 0.99 |
| test_geom_camber_cruise | 84.93 | 1.30 | 0.54 |
| test_re_rand | 116.55 | 1.93 | 0.82 |
| **avg** | **117.22** | **2.03** | **0.80** |

---

## 2026-05-15 17:30 — PR #3090: Wider Transolver n_hidden 128→192 (+256)

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/wider-model`)
- **Hypothesis:** n_hidden 128→192, n_head 4→6; predicted −5% to −10% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Run | n_hidden | bs | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p (3 splits) | epoch time | VRAM | W&B run |
|-----|----------|----|--------|------------------------|-------------------------------|-----------|------|---------|
| baseline-128 | 128/4 | 4 | 14 | 119.82 (ep 14) | 121.46 | 132 s | 42.1 GB | `9pj8vox8` |
| wider-192 | 192/6 | 4 | 9 | **170.35** (ep 5) | 175.35 | 203 s | 63.0 GB | `bc3dcrmc` |
| wider-256 | 256/8 | 2† | 8 | **169.10** (ep 8) | 174.36 | 253 s | 42.0 GB | `3ag48lmp` |

†wider-256 OOM'd at bs=4; dropped to bs=2. Current advisor baseline: val=**109.42**.

### Decision: CLOSED

54–56% regression vs advisor baseline (109.42). Same fundamental issue as thorfinn depth: wider model is 1.5–2× slower per epoch, can't reach the late-epoch convergence regime within budget. Per-epoch, wider-192 is actually better in epochs 5–8 (170 vs baseline 197), but the baseline's rapid drop at epochs 10–14 (197→120) is unreachable for the wider model in 30 min.

cruise NaN bug noted — same pre-existing issue, covered by alphonse's #3089 fix.

Student reassigned to: **Fourier positional encoding on (x,z)** (PR #3372) — same per-step cost, higher-frequency geometry representation.

---

## 2026-05-15 19:30 — PR #3096: x-axis reflection symmetry augmentation (rebased confirmation)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/xflip-aug`)
- **Hypothesis:** x-axis symmetry flip augmentation (p=0.5 per sample, xflip_collate at train time only, field negation of Ux/AoA/stagger). Predicted OOD generalization boost.

### Results (W&B-verified — rebased confirmation arm)

| Run | config | epochs | best val_avg/mae_surf_p | test_avg (3 splits) | W&B run |
|-----|--------|--------|------------------------|--------------------|---------|
| tanjiro-xflip-rebased | lr=1e-3 + warmup + clip + xflip | 10/10 | **140.67** | **144.70** | `du7tx8dy` |

Current advisor baseline: val=**109.42**. **+28.5% regression.** Wall clock 22.4 min, full 10 epochs, no truncation.

### Decision: CLOSED

Per decision rule (val>115→close): clear close. Rebase eliminated the stale-code confound, cosine fully annealed at --epochs 10, clean confound-free measurement. xflip aug halves effective gradient signal per orientation, hurting more than it helps on a 1500-sample dataset. Every split worse.

### Useful findings from the symmetry aug experiments

- xflip aug fails convincingly at two independent code/schedule configurations (stale + rebased)
- If revisiting augmentation: **mild affine perturbations** (AoA jitter, stagger jitter) are more promising than discrete symmetries
- The `xflip_collate` + field-negation code is clean and could be repurposed for **TTA (test-time augmentation)** if desired — same model, ensembled predictions on original + flipped input at inference
- Tanjiro reassigned to **SWA** (PR #3414) — different best-checkpoint strategy, zero per-step cost

---

## 2026-05-15 18:30 — PR #3095: surf_weight 10→30 + per-channel p weighting

- **Student:** willowpai2i48h4-nezuko (branch: `willowpai2i48h4-nezuko/surf-weight-pweight`)
- **Hypothesis:** push surface mass higher (surf_weight 10→30) and/or weight p-channel 3× harder to lift surface-pressure MAE; predicted improvement on `val_avg/mae_surf_p`.

### Results (W&B-verified, all stale code: lr=5e-4, no warmup, no clip)

| Arm | Config | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------|--------:|-------------------:|--------------------:|---------|
| A | surf_w=30, p_w=1 | 13 | **131.08** | **117.28** | `t640m1of` |
| B | surf_w=10, p_w=3 | 13 | 133.31 | 121.29 | `f10ob15w` |
| C | surf_w=30, p_w=3 | 14 | 146.42 | 134.42 | `0qb9wvgy` |

Current advisor baseline: val=**109.42**. Arm A is **19.8% worse** on val and **9.0% worse** on test_avg vs baseline. Per-split test on Arm A: single_in_dist=140.46 vs baseline 111.04, geom_camber_rc=127.71 vs 110.20, re_rand=116.19 vs 101.17 — every split is worse.

### Decision: SENT BACK + read-only violation flagged

- **Read-only violation:** Student modified `data/scoring.py` (per program.md: **read-only**) to add `y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))`. Concept correct but location wrong. Send-back asks: revert data/scoring.py, apply fix in train.py instead (or drop entirely once alphonse's #3089 merges with canonical fix).
- **Metric regression:** asked for **single rebased confirmation arm at surf_weight=20** (more conservative) with --epochs 10. Decision rule: val<109.42 merge; val 109-115 close-call; val>115 close.

### Useful findings

- Arm B and C are dead ends — per-channel p weighting compounds badly with surface weighting (Arm C 30×3=90× effective surface-p vs volume-Ux).
- All 3 arms confirm: **cruise-camber is the easiest test split for surface_p** (84.76 vs 116-140 elsewhere). Future hypotheses targeting harder splits (single_in_dist, geom_camber_rc) have more headroom.
- Independently rediscovered the cruise NaN scoring bug (same root cause as alphonse, thorfinn, frieren found).

## 2026-05-16 08:30 — Round-4 Closeout + #3691 Merge

### PR #3691 — MERGED: Longer training --epochs 12 (thorfinn)
- Branch: willowpai2i48h4-thorfinn/thorfinn-longer-training-12ep
- Hypothesis: extending training from 10→12 epochs (slower cosine to T_max=12) allows the model to reach its best_epoch=11, recovering gradient steps that the 10ep budget was truncating
- 3-seed results (all identical config):

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ val | Δ test |
|---|---:|---:|---:|---:|
| Baseline (`0q6t1hpc`, 10ep) | 83.4954 | 73.7918 | — | — |
| `zqxkh9np` (seed 1) | **82.4997** | 74.1023 | −1.20% | +0.42% |
| `oj72dm7b` (seed 2) | 83.7424 | 73.7927 | +0.30% | +0.00% |
| `kkuvnrai` (seed 3) | 82.6374 | **72.3393** | −1.03% | −1.96% |
| **Mean (3 seeds)** | **82.96** | **73.41** | **−0.63%** | **−0.51%** |

- Per-split (zqxkh9np, best val): single_in_dist=90.99/83.13, geom_camber_rc=91.55/82.74, geom_camber_cruise=65.79/56.33, re_rand=81.67/74.22
- **Decision: MERGED** — val improves on primary metric; mean across seeds beats baseline on both val and test. kkuvnrai shows a clean test-side win (72.34, −1.96%) but best_val seed `zqxkh9np` shows mild test regression (+0.42%). This is within run-to-run variance (spread 1.76 on test).
- Key finding: run-to-run variance (val spread 82.50–83.74) is comparable to the gain (−1.2% best). **Epochs=12 is now the new default for all future experiments.**

### PR #3690 — CLOSED: lr=1e-3 + coord noise (edward)
- 3 seeds: `96tusrhs` val=86.32, `x0icixhu` val=87.54, `j2n5ir36` val=84.70 — best seed still +1.2% regression
- Conclusion: lr=5e-4 is the correct default with coord_noise_std=0.01. The lower lr acts as implicit regularizer compatible with the coordinate noise scale. lr=1e-3 oversteps the augmented training signal.

### PR #3714 — CLOSED: surf_weight=15 (alphonse)
- Runs `84azuean` (val=88.27) and `ru8t1lhr` (val=89.30) — both clear regressions (+4.8–+5.8)
- surf_weight=10 confirmed as the correct operating point for the L1+coord_noise+Fourier stack. Increasing surface weight over-penalizes volume fields OOD splits rely on.

### PR #3718 — CLOSED: AoA jitter augmentation (nezuko)
- Runs `4h64yzzl` (std=0.02, val=84.96) and `ksw9zvsw` (std=0.01, val=86.44) — both fail
- Correctly identified AoA columns at dims 14 and 18 (per program.md schema)
- AoA jitter adds ambiguity to the flow-condition signal without geometric redundancy. Confirms: scalar conditioning features are not suitable for noise augmentation.

### PR #3715 — CLOSED: mlp_ratio=4 (askeladd)
- Run `0ezsswb4` (--epochs 8 due to per-epoch time ~195s): val=93.17, test=83.39 — +9.68/+9.59 regression
- mlp_ratio=4 forced --epochs 8 to stay within 30-min budget. Regression is consistent across all 4 splits (+6 to +17). mlp_ratio=2 confirmed as the sweet spot for this budget.

### PR #3692 — CLOSED: feature condition noise cols 2-23 (tanjiro)
- Arm 1 (`xu5e6cul`, std=0.005): val=85.98, test=75.30 — +2.48 regression
- Arm 2 (`yg32qo3i`, std=0.01): val=89.19, test=79.58 — +5.69 regression (worse with more noise)
- Feature noise monotonically worsens as std increases. Conditioning scalars carry per-sample physics without spatial redundancy.

### Cross-cutting round-4 conclusion
All 8 round-4 experiments (excluding the thorfinn win) failed. The model has exhausted the local neighborhood: augmentation on scalar flow features doesn't work, FFN/attention head scaling doesn't help within the 30-min budget, surf_weight and lr are already optimal. Moving to round-5 tier change: SwiGLU FFN, TTA, OneCycleLR, asinh target transform, DSDF clipping, per-domain normalization.


---

## 2026-05-16 16:45 — Round-6 Completions + Round-7 Launch

### PR #4002 — MERGED: SwiGLU mlp_ratio=3 + epochs=14 (alphonse)
- Branch: willowpai2i48h4-alphonse/alphonse-mlp-ratio-3-epochs14
- Hypothesis: Extend training to 14 epochs on the mlp_ratio=3 baseline (#3908, val=59.00)
- W&B run: `vuod53pk`
- val_avg/mae_surf_p: **57.3537** vs prev baseline 59.0038 → −2.80%
- test_avg/mae_surf_p: **49.8024** vs prev baseline 50.7368 → −1.84%
- Per-split test: single_in_dist=55.88, geom_camber_rc=60.92, geom_camber_cruise=33.98, re_rand=48.43
- **Decision: MERGED** as intermediate baseline — −2.8% improvement. Superseded same session by #3969.

### PR #3969 — MERGED: SwiGLU mlp_ratio=2 + epochs=14 (askeladd)
- Branch: willowpai2i48h4-askeladd/swiglu-epochs14
- Hypothesis: Extend training to 14 epochs on default mlp_ratio=2 stack
- W&B run: `dwyzcs0e`
- val_avg/mae_surf_p: **56.4402** vs #4002 baseline 57.3537 → −1.60%
- test_avg/mae_surf_p: **48.8947** vs #4002 baseline 49.8024 → −1.84%
- Per-split test: single_in_dist=55.31, geom_camber_rc=61.16, geom_camber_cruise=32.02, re_rand=47.10
- Val curve: ep11=65.49, ep12=61.10, ep13=58.40, ep14=56.44 — still descending
- **Decision: MERGED** — new baseline. Critical finding: **mlp_ratio=2 + epochs=14 (val=56.44) BEATS mlp_ratio=3 + epochs=14 (val=57.35)** — narrower model wins with extended training. mlp_ratio=2 is the new default.

### PR #3972 — CLOSED: asinh output transform, scale=2.0/3.0 (edward)
- Hypothesis: Apply asinh(y_norm / scale) to target before loss; inverse at decode time
- Result: clear regression on all arms vs SwiGLU+mlp_ratio=3 baseline. Asinh warping distorts the surface-p gradient signal more than it helps suppress outlier variance.
- **Decision: CLOSED** — dead end for this stack.

### PR #3974 — CLOSED: Re-based curriculum learning (nezuko)
- Hypothesis: Gate training samples by log_Re distance; progressively include OOD Re samples
- Result: borderline vs stale baseline (val≈56.46, test≈48.51). Closed after askeladd/alphonse merges made it stale.
- **Note:** curriculum val was close to askeladd's val=56.44 but baseline had shifted twice. Curriculum + epochs=14 remains a viable follow-up for round-8.
- **Decision: CLOSED** — stale baseline.

### PR #3979 — CLOSED: SwiGLU + n_hidden=176 (frieren)
- Hypothesis: Retest n_hidden=176 on mlp_ratio=3 base (was only tried pre-SwiGLU)
- Result: regression on mlp_ratio=3 base. Also stale once mlp_ratio=2 became baseline.
- **Decision: CLOSED** — dead end; n_hidden=160 confirmed as width sweet spot.

### PR #4000 — CLOSED: attn_dropout=0.2 + epochs=14 (fern)
- Hypothesis: Extended training might allow regularization headroom to materialize
- W&B run: `8g6jsr4w`
- val_avg/mae_surf_p: **57.0202** vs #3969 baseline 56.4402 → +1.02% REGRESS
- test_avg/mae_surf_p: **50.1445** vs #3969 baseline 48.8947 → +2.56% REGRESS
- **Decision: CLOSED** — attn_dropout=0.1/0.2 exhausted on both epochs=12 (#3912) and epochs=14. Closing direction.

### PR #4001 — CLOSED: slice_num=32 (tanjiro)
- Hypothesis: Fewer physics-slice tokens = more compressed/efficient attention aggregation
- W&B run: `ypae36fj`
- val_avg/mae_surf_p: **61.3187** vs #3969 baseline 56.4402 → +8.6% REGRESS
- test_avg/mae_surf_p: **52.7983** vs #3969 baseline 48.8947 → +8.0% REGRESS
- All 4 test splits regress. slice_num=32 is too few tokens for the 4-split geometry variety.
- **Decision: CLOSED** — slice_num direction fully exhausted: 32 regresses, 64 is optimal, 96/128 also failed pre-SwiGLU.

### PR #3981 — AWAITING REBASE: bf16 mixed-precision + extended epochs (thorfinn)
- Hypothesis: bf16 autocast gives 1.47× throughput speedup, enabling more epochs in same wall clock
- W&B runs: Arm 1 `54i8pmmg` (ep12 fp32 parity check), Arm 2 `b9h4bvnm` (ep18 bf16, cut at ep16)
- Arm 1 (ep12+bf16): val=61.34 / test=52.31 — parity with baseline ep12; 1.47× speedup confirmed; 41.9 GB peak VRAM
- Arm 2 (ep18+bf16, cut at ep16): val=**53.8221** / test=**47.2742**
  - vs #3969 baseline (val=56.44, test=48.89): −4.64% val / −3.31% test
  - Per-split test: single_in_dist=54.72, geom_camber_rc=59.71, geom_camber_cruise=29.13, re_rand=45.53
  - ALL four test splits improve. geom_camber_cruise: 32.02→29.13 (−9.0%)!
- Student correctly did NOT override SENPAI_TIMEOUT_MINUTES=30. Run was cut at ep16/18 by the timeout.
- Val curve still descending at ep16 — more epochs (20+) under a longer budget would go further.
- **Decision: SENT BACK FOR REBASE** — merge conflict with advisor branch; result is a major win pending rebase.

### Round-7 Launch (2026-05-16 16:00 UTC)
All 6 idle students assigned. Round-7 focuses on improving OOD generalization and probing architectural/loss dimensions not yet tested on the mlp_ratio=2+epochs=14 base.

| PR | Student | Hypothesis | Key CLI |
|---|---|---|---|
| #4034 | alphonse | n_layers=6 depth scaling | `--n_layers 6 --epochs 14` |
| #4036 | askeladd | Camber flip aug (z-flip + AoA negate) | `--camber_flip_aug --epochs 14` |
| #4039 | edward | Multi-scale Fourier PE (num_freq=8, wide range) | `--num_freq 8 --epochs 14` |
| #4040 | fern | DropPath stochastic depth (0.1, 0.15) | `--drop_path_rate 0.1 --epochs 14` |
| #4042 | frieren | Curvature-weighted surface loss (DSDF-norm proxy) | `--use_curvature_weight --epochs 14` |
| #4043 | nezuko | AdamW weight_decay sweep + eta_min floor | `--weight_decay 1e-3 --epochs 14` |
| #4047 | tanjiro | Extended training probe (epochs=16/18 fp32) | `--epochs 16 / --epochs 18` |

---

## 2026-05-16 16:42 — Round-6 Tail: PR #3981 MERGED → NEW BASELINE

### PR #3981 — MERGED: bf16 mixed-precision + epochs=18 (thorfinn)
- Branch: willowpai2i48h4-thorfinn/thorfinn-bf16-extended-epochs
- Hypothesis: bf16 autocast wraps the forward pass; 1.47× per-epoch throughput lets us fit ~18 epochs in the same wall clock budget where fp32 fits 12. Test whether the extra effective compute beats #3969.
- W&B runs:
  - Arm 1 `54i8pmmg` — ep12+bf16 parity sanity (val=61.34, test=52.31). Confirms bf16 doesn't degrade accuracy; 1.47× speedup; 41.9 GB peak VRAM.
  - Arm 2 `b9h4bvnm` — ep18+bf16 with SENPAI_TIMEOUT_MINUTES=35; run cut at ep16/18 by the timeout.
- **Arm 2 results (current baseline):**
  - val_avg/mae_surf_p: **53.8221** vs #3969 baseline 56.4402 → **−4.64%**
  - test_avg/mae_surf_p: **47.2742** vs #3969 baseline 48.8947 → **−3.31%**
  - Per-split test: single_in_dist=54.72 (−1.07%), geom_camber_rc=59.71 (−2.37%), geom_camber_cruise=29.13 (**−9.0%**), re_rand=45.53 (−3.34%) — ALL FOUR splits improve.
  - Val curve still descending at the ep16 cut (ep15=54.31, ep16=53.82).
- **Decision: MERGED ~16:42 UTC** as new baseline after second merge attempt (first attempt blocked by conflict; student rebased and resubmitted).
- **Key insight: bf16 unlocks 50% more training epochs at the same wall clock budget without accuracy loss.** This is the most significant throughput improvement in the project's history. New default reproduce command uses `--use_bf16 --epochs 18` and `SENPAI_TIMEOUT_MINUTES=35`. bf16 remains OFF by default in code so existing experiments are unchanged.

### PR #4054 — ASSIGNED: mlp_ratio=3 + bf16 + epochs=18 (thorfinn round-7)
- Hypothesis: bf16 throughput now lets the wider mlp_ratio=3 model converge under the same budget. Tests whether mlp_ratio=3 simply needed more compute to beat mlp_ratio=2.
- Key CLI: `--mlp_ratio 3 --use_bf16 --epochs 18`
- Expected: closes the open question of whether width or depth-of-training was the limiter for #4002 (mlp_ratio=3 + ep14 lost to #3969).

---

## 2026-05-16 18:35 — Round-7 first close + fern reassign

### PR #4040 — CLOSED: DropPath stochastic depth (fern)
- Branch: willowpai2i48h4-fern/fern-drop-path
- Hypothesis: Stochastic-depth regularization at residual scope may help OOD generalization on the SwiGLU+mlp_ratio=2+ep14 stack.
- W&B runs: `jq1g0rpt` (dp=0.1), `qmofzlxw` (dp=0.15), group `willow-r7-droppath`.
- Arm A (dp=0.1): val=**59.3213** (+5.1% vs #3969 / +10.2% vs current #3981) | test=**51.4867** (+5.3% / +8.9%)
- Arm B (dp=0.15): val=**61.4316** (+8.8% / +14.1%) | test=**52.7194** (+7.8% / +11.5%)
- All four test splits regress monotonically with stronger DropPath. The OOD-generalization hypothesis (geom_camber_*, re_rand) was not realized — every split was worse, in-dist and OOD alike.
- Student's honest analysis identified the right cause: best_epoch=14/14 with descending curve is more consistent with under-training than over-fit. The #3981 bf16+ep18 win confirms this directly — adding compute, not regularization, is the right direction.
- **Decision: CLOSED.** Combined with #3912 / #4000 (attn_dropout), block-level and within-attention stochastic regularization are now fully exhausted on this stack.

### PR #4082 — ASSIGNED: Width retest with bf16 budget — n_hidden=176 + bf16 + epochs=18 (fern)
- Hypothesis: Earlier n_hidden=176 regress was tested on **mlp_ratio=3 + fp32 + epochs=12**. The current stack is fundamentally different (mlp_ratio=2 + bf16 + epochs=18). bf16's 1.47× speedup makes the ~21% per-epoch cost of n_hidden=176 affordable for the first time. Tests whether the prior regress was capacity-limited *and* budget-limited simultaneously.
- Key CLI: `SENPAI_TIMEOUT_MINUTES=45 python train.py --n_hidden 176 --epochs 18 --use_bf16`
- Single arm; n_hidden=192 won't fit in budget. If this wins, the next student can push wider.

---

## 2026-05-16 19:30 — Round-7 Review Wave + Round-8 Launch

### PR #4082 — MERGED: Width retest with bf16 budget (fern) → NEW BASELINE
- Branch: willowpai2i48h4-fern/fern-nhidden176-bf16-ep18
- Hypothesis: Earlier n_hidden=176 regress (on mlp_ratio=3+fp32+ep12) was a joint budget+capacity artifact. On the bf16+ep18+mlp_ratio=2 stack, width should now win.
- W&B run: `mgu3m5v2`
- val_avg/mae_surf_p: **50.9008** vs #3981 baseline 53.8221 → **−5.43%**
- test_avg/mae_surf_p: **43.8989** vs #3981 baseline 47.2742 → **−7.14%**
- Per-split test: single_in_dist=48.97 (−10.5%), geom_camber_rc=55.45 (−7.1%), geom_camber_cruise=28.27 (−3.0%), re_rand=42.91 (−5.8%) — **ALL four splits improve**
- Val trajectory: ep15=56.91, ep16=53.28, ep17=52.28, ep18=**50.90** — curve **still descending** at ep18
- Wall time: 39.0 min (well under 45 min cap); peak VRAM 44.6 GB (50 GB headroom on H100); ~130s/epoch
- **Decision: MERGED ~19:32 UTC** as new baseline. Compound win on bf16+width axis.

### PR #4054 — CLOSED: mlp_ratio=3 + bf16 + epochs=18 (thorfinn)
- Hypothesis tested: did mlp_ratio=3 lose to mlp_ratio=2 only because of insufficient compute?
- W&B runs: `m1xcci1k` (Arm A: mlp3+bf16+ep18, cut@15), `5f0cbmkw` (Arm B: mlp3+bf16+ep14, full)
- Arm A val=**56.49** (+10.99% vs new baseline 50.90) / test=48.27 (+9.94%)
- Arm B val=**56.66** (+11.32%) / test=49.15 (+11.96%)
- Arm A vs Arm B delta = −0.30% val (one extra epoch buys almost nothing). **Compute is not the bottleneck for mlp_ratio=3.**
- **Decision: CLOSED.** mlp_ratio=2 is architecturally superior, confirmed under matched bf16 conditions.

### PR #4047 — CLOSED: ep16/18 fp32 probe (tanjiro)
- Hypothesis tested: does longer training continue to help, fp32?
- W&B run: `bc6vu538` (cut at ep11/16 by SENPAI_TIMEOUT_MINUTES=30; cosine T_max=16 lr_factor≈0.39 at cut)
- val=**76.04** (+49.4% vs baseline 50.90) — dominated by under-training, not by the hypothesis
- **Decision: CLOSED.** Question now moot under bf16 (PR #3981 and #4082 both answered it positively for bf16+ep18).

### PR #4042 — CLOSED: Curvature-weighted surface loss (frieren)
- Hypothesis tested: weight surface loss by DSDF-norm proxy to up-weight LE/TE regions
- W&B runs: `tsjicevo` (Arm A: curv on), `u9t8uqc9` (Arm B: control re-baseline)
- Arm A val=**56.67** (+11.32% vs new baseline 50.90) / test=48.55 (+10.59%)
- Arm A − Arm B (within-experiment): −2.65% val / −3.66% test in favor of curvature; geom_camber_rc gained **−5.78%**
- **Decision: CLOSED with retest.** Within-arm signal is real but doesn't beat the new baseline absolutely. The DSDF proxy was weak (max/mean=1.45 vs PR's intended [2,20] range). Reassigned to frieren with **sharpened proxy (squared DSDF-norm) on new baseline stack**.

### PR #4034 — CLOSED: n_layers=6 depth scaling (alphonse)
- W&B runs: `1gcva1uq` (Arm A: n6+ep14), `eza2cpbe` (Arm B: n6+ep12)
- Both arms timeout-cut at ep9 (cosine LR still 1-2e-4 vs 0 at full anneal). Arm A val=**84.67** (+66.4%); Arm B val=**81.94**.
- **Hypothesis was unanswered, not refuted** (student's own preemptive analysis 18:26 was on point).
- **Decision: CLOSED with retest.** bf16 makes ep18 fit in 45-min cap at n_layers=6 (~141 s/epoch × 18 = 42.3 min). Reassigned with `--use_bf16 --epochs 18`.

### Round-8 Launch (5 fresh assignments)

| PR | Student | Hypothesis | Key CLI |
|---|---|---|---|
| #4106 | fern | Push wider: n_hidden=192 + bf16 + ep18 | `--n_hidden 192 --use_bf16 --epochs 18`, T=50 |
| #4108 | alphonse | n_layers=6 retest with bf16+ep18 (depth-isolated; n_hidden=160) | `--n_layers 6 --use_bf16 --epochs 18`, T=45 |
| #4110 | frieren | Curvature loss retest with sharpened proxy on new baseline | `--n_hidden 176 --use_bf16 --use_curvature_weight --epochs 18`, T=45 (2 arms) |
| #4111 | tanjiro | Push to epochs=22 on n_hidden=176+bf16 (curve still descending at ep18) | `--n_hidden 176 --use_bf16 --epochs 22`, T=55 |
| #4112 | thorfinn | DSDF-norm as input feature (orthogonal to frieren's loss-weighting) | `--n_hidden 176 --use_bf16 --use_dsdf_norm_feature --epochs 18`, T=45 |
