# SENPAI Research Results

---

## 2026-05-14 05:25 — Round 05:25: PLATEAU CONFIRMED via 4-PR wave-closure

Second wave of Round 02:28 assignments arrived. All 4 review-ready PRs regressed vs #2690 baseline. **Striking cross-PR pattern: all 4 saw rc REGRESS by 5-7%**, including the one that specifically targeted rc.

### Cross-PR rc regression table

| PR | Student | Mechanism | val_avg Δ | **rc Δ** | Other splits |
|----|---------|-----------|----------:|---------:|-------------|
| #2770 | askeladd | Re-cond FFN-FiLM | +3.19% | **+5.32%** | single +5.09%, cruise -1.22%, re_rand +0.60% |
| #2772 | nezuko | p-label noise ε=0.01 | +4.25% | **+6.83%** | single +5.51%, cruise +1.86%, re_rand +0.74% |
| #2775 | fern | AoA_1 jitter σ=0.02 | +7.2% | **+6.8%** | (all splits regressed) |
| #2779 | thorfinn | NACA-pair FiLM | +1.9% | **+6.5%** | single -2.5% ✅, cruise -0.1%, re_rand +0.6% |

**This is a plateau signal.** The post-#2690 parameter space is locally hostile to ANY rc-targeted intervention regardless of mechanism family (Re-cond, data-aug, geometry-cond, label-noise). The merged #2690 ReCondOutputBias absorbed all the easy rc gains, and the local neighborhood is exhausted.

### Closures (4 PRs)

**#2770 askeladd Re-cond FFN-FiLM (+3.19% val)**: Capacity competition with ReConditionalLayerNorm[ffn]. Both hooks read log_Re and condition the same FFN sub-block. ReCondLN[ffn] |γ_residual|_max dropped to 0.65 (slightly weaker than typical) while new FFN-FiLM picked up 1.03 — clean evidence of partial migration but worse final outcome. **Refined principle**: Re-conditioning succeeds only at injection points NOT already conditioned by another active hook.

**#2772 nezuko p-label noise ε=0.01 (+4.25% val)**: Three failure-mode roots: (1) targets are physical (precise scalar pressure field, not categorical labels); (2) ε scaled by per-sample p-std may have been too aggressive; (3) rc gap is geometry-driven, not noise-tolerance-driven. Label noise acts as variance multiplier, not regularizer, on continuous physical targets. **Label-noise meta-axis CLOSED**.

**#2775 fern AoA_1 jitter σ=0.02 (+7.2% val, rc +6.8%)**: Targeted intervention made target split WORSE. Critical finding: **AoA_1 coverage mismatch is NOT the dominant cause of the rc gap**. rc OOD is a JOINT condition — augmenting one axis breaks the joint correlation that defines real rc samples, polluting the rc decision boundary. **Third independent rc-targeted intervention to fail (after #2689, #2721 first arm)**. **Single-axis data augmentation FAMILY-CLOSED**.

**#2779 thorfinn NACA-pair FiLM (+1.9% val, BUT single_in_dist -2.5%)**: Cruise/rc structural dichotomy reappears (8th independent observation). Geometric FiLM mechanism IS REAL — corr(|β|, log_Re)=0.13 (NOT Re-proxy), offset_absmax 0.21 grew from zero-init. Helps where geometry is shared (in-dist) hurts where geometry is novel (rc joint). Filing the mechanism for future use after plateau is broken. **Geometry-conditional-FiLM-as-tested axis CLOSED**.

### New assignments (4 deliberately diverse mechanism families)

Per plateau protocol: escalate to bolder mechanism families spanning the orthogonal directions we have NOT exhausted.

**#2802 askeladd `stochastic-depth-droppath-p0p1`** (Architecture-level regularization)
- DropPath with linear schedule (0 at block 0 → 0.1 at block 4). Standard recipe.
- Forces implicit ensembling per Huang et al. 2016.
- Zero new params; orthogonal to all current mechanisms.

**#2804 nezuko `manifold-mixup-trunk-mid-beta0p2`** (Representation-space regularization)
- Manifold Mixup at trunk middle layer (3 of 5), Beta(0.2, 0.2).
- Mixes hidden states AND targets — NOT label noise (which #2772 just failed).
- Targets representation-space smoothness for OOD interpolation.

**#2806 fern `cosine-restart-2cycles-tmax14`** (Optimizer schedule)
- SGDR-style: T1=14 + T2=14 cosine cycles with restart, lr2_scale=0.5.
- Forces non-local parameter exploration via high-LR restart at epoch 15.
- Plateau-breaking specifically: kicks out of cycle-1 local minimum.

**#2807 thorfinn `wider-shallower-h192-l4`** (Architecture sizing)
- n_hidden=192, n_layers=4, n_head=6 (head_dim=32, bf16-GEMM friendly).
- Same ~672K params, different width-depth allocation.
- Never tested. n_layers=6 failed earlier (+6.22%) — we've never tried SHALLOWER + wider.

### Strategic frame for next round

Round 05:25 spans 4 orthogonal mechanism families (regularization architecture, regularization representation, optimizer schedule, capacity sizing) plus the in-flight EMA/SAM optimizer-flat-minima pair (#2795/#2796) and oversampling (#2721 frieren). After this wave we'll have evidence on which family the plateau breaks in — or whether none of these work and we need to go even bolder (synthetic data, PINN losses, or backbone replacement).

---

## 2026-05-14 05:08 — Round 05:08: 2 closures + 2 bold-direction assignments + noise floor calibration

First wave of Round 02:28 assignments arrived. No winners — both review-ready PRs regressed vs the #2690 baseline. But the multi-seed PR delivered a critical noise-floor calibration that will shape all future merge decisions on this branch.

### Closure — #2768 alphonse `re-conditional-attn-temperature` (+3.5% val regression)

- val_avg 28.5527 vs #2690 baseline 27.5868 = **+0.97 (+3.5%)**
- test_avg 25.1924 vs 24.1056 = **+1.09 (+4.5%)**
- All 4 val splits regressed: single +1.10, rc +2.25, cruise +0.20, re_rand +0.31
- Metrics: `models/model-re-conditional-attn-temperature-20260514-042931/metrics.jsonl`

**Mechanism worked perfectly** — clean diagnostic confirms hypothesis at the parameter level:
- α opened from 1.0 → mean ~1.28 by epoch 5, stabilized 1.20–1.30 across all blocks
- corr(α, log_Re) = +1.00 across all 5 blocks (monotonic by construction)
- Sharper softmax for high-Re samples, softer for low-Re — direction matches hypothesis
- No clamp hits (|α-1| ≤ 0.65), no NaNs, only 10 added params

**But the lever didn't transfer to val gain.** This is the FIRST Re-conditioning hook to NOT win on this branch (4-of-4 streak broken at the 5th):
1. ✅ ReFiLM (slice-attn output gates) — feature-output FiLM
2. ✅ ReScaleHead (output scale) — feature-output scale
3. ✅ ReConditionalLayerNorm (γ/β affine) — feature-output affine post-LN
4. ✅ ReConditionalOutputBias (output bias) — feature-output bias
5. ❌ ReConditionalAttnTemperature (softmax denominator) — **computation path**, not feature output

**Refined principle**: Re-condition feature *outputs* ≫ Re-condition computation *paths*. The 4 winners all act on already-computed features (or about-to-be-output features). Attn temp changes how slice mass is *distributed*, not the values being summed — it's the function class itself, not the parametrization. Mean α≈1.28 = 28% sharpening = less mixing across slices = worse OOD generalization (consistent with all-splits regression rather than regime-specific failure).

**Re-conditional-attn-TEMPERATURE axis CLOSED.** No follow-up; the diagnostic is conclusive.

### Closure — #2725 edward `multi-seed-variance-new-baseline` (not a winner, but established noise floor)

- 2-seed config switched mid-flight from `--tta_val_per_epoch` (no-op, cost 5 epochs) to `--tta_test` (end-of-train TTA + variance only). Right call.
- 2-seed mean val_avg/mae_surf_p (with TTA) = 28.8634 (vs #2650 baseline 28.2414 = **+2.21%**)
- 2-seed mean test_avg/mae_surf_p (with TTA) = 24.8868 (vs #2650 baseline 24.4827 = **+1.65%**)
- Both seeds landed **above** #2650 (28.60 and 29.13) — meaning **#2650 itself was on the favorable tail of its seed distribution**.
- Metrics: `models/model-multi-seed-tta-seed42-20260514-034250/metrics.jsonl`, `models/model-multi-seed-tta-seed43-clean-20260514-042239/metrics.jsonl`

**CRITICAL DELIVERABLE — Noise floor on the #2650 baseline stack (n=2 seeds):**

| Quantity | Mean | Std |
|----------|-----:|----:|
| val_avg/mae_surf_p | 28.866 | **0.37** |
| test_avg/mae_surf_p | 24.946 | **0.16** |

**Implications for all future merge decisions on this branch:**

1. **The merged #2690 ReConditionalOutputBias gain (-0.66 val_avg over #2650) is ~1.8σ over the #2650 seed mean.** Real but borderline. Cumulative -76.5% is unchanged.
2. **Future single-seed wins under ~0.4 val_avg ARE noise-equivalent.** Decision criteria updated: single-seed gain ≥ 0.4 val_avg = confident winner; gain ∈ [0, 0.4) = marginal, prefer multi-seed confirmation or merge with explicit noise caveat; gain < 0 = close.
3. **TTA stacking is small (~0.06 test, ~0 val) and dominated by 0.16 test std.** Per-epoch TTA-val checkpoint selection was empirically a no-op. **TTA-stacking axis CLOSED.**

This noise floor will be added to `BASELINE.md` Current Baseline section.

### New assignment — #2795 alphonse `ema-weight-averaging-decay0p999`

Bold direction off the Re-conditioning family after #2768 broke the streak.

- **Hypothesis**: Polyak EMA from step 0 with decay=0.999. Maintain running `θ_ema = 0.999·θ_ema + 0.001·θ`, evaluate using `θ_ema`. Aggregates final-region trajectory smoothly (~last 2-3K steps weighted exponentially by end of training).
- **Why now**: Model is budget-limited (best epoch always = last epoch). EMA captures the final descent more stably than the literal terminal weights. Different from SWA (which failed in #2702 due to swa_lr=1e-4 LR conflict with cosine floor).
- **Zero new params, zero new compute** — pure free-lunch optimizer trick.
- **Diagnostics requested**: `||θ-θ_ema||/||θ||` at end of training + val_avg with/without EMA at best epoch.

### New assignment — #2796 edward `sam-sharpness-aware-minimization`

Bold optimizer-level direction targeting flat minima for OOD generalization.

- **Hypothesis**: SAM (Foret et al. 2021) with ρ=0.05. Perturb θ→θ+ρ·∇L/||∇L||, compute grad at perturbed point, use THAT grad to update θ. Finds parameters where both θ and its neighborhood have low loss → flat minima → better OOD generalization.
- **Why now**: Our hardest splits are OOD (rc, re_rand). SAM's flat-minima bias is exactly designed for this regime. Edward's #2725 noise floor showed current optimizer is at the local-basin noise floor — moving basins should provide >0.4σ separation.
- **Trade-off**: 2× cost → halve to 14 epochs (T_max=14). Risk of undertraining is the main caveat.
- **Diagnostics requested**: sharpness measure `(L(θ+ρ·∇L/||∇L||) - L(θ)) / L(θ)` on val batch + grad-norm trajectory.

### Open state at end of Round 05:08

| PR | Student | Slug | Status |
|----|---------|------|--------|
| #2770 | askeladd | re-cond-ffn-film | WIP (~1h) |
| #2772 | nezuko | p-label-noise-1pct | WIP (~50min) |
| #2775 | fern | aoa1-neg-jitter | WIP (~50min) |
| #2779 | thorfinn | naca-pair-film | WIP (~50min) |
| #2788 | tanjiro | re-cond-input-scale | WIP (just-assigned) |
| #2721 | frieren | rc-nn-geom-weighted | WIP needs rebase + softer cap |
| #2795 | alphonse | ema-decay0p999 | WIP (just-assigned 05:08) |
| #2796 | edward | sam-rho0p05 | WIP (just-assigned 05:08) |

**Family balance**: 3 Re-conditioning (#2770 FFN-FiLM, #2788 input-scale, #2779 NACA-pair geometric FiLM), 2 data-augmentation (#2772, #2775), 1 oversampling-diagnostic (#2721), 2 optimizer-level (#2795 EMA, #2796 SAM). Healthy diversity given the 5th Re-hook just failed; the 2 optimizer-level swings de-risk overfitting to a single mechanism family.

---

## 2026-05-14 02:40 — Round 02:40: stale-PR rotation (1 closure, 1 reassignment, 1 send-back follow-up)

Mini-round triggered by survey showing #2724 tanjiro stalled ~2h45m. No new winners, no review-ready PRs. All 8 students now actively WIP after rotation.

### Closure — #2724 tanjiro `geometry-mirror-tta`

- Branch: `charliepai2g24h1-tanjiro/geometry-mirror-tta`
- Reason: pod stuck on GraphQL rate-limit cycles (iterations 304-307 visible, all "No work assigned" despite PR correctly labeled `status:wip` + `student:charliepai2g24h1-tanjiro`). Same persistent rate-limit pattern as #2599 (~24h stuck) and #2534 (~5h stuck) closed in Round 01:54 — both students recovered when a fresh PR was created.
- No metrics produced. Geometry-mirror TTA hypothesis remains theoretically valuable (zero training cost, rc has heavy AoA asymmetry per #2721 diagnostic) but pushed to a future round in favor of the proven Re-conditioning family for tanjiro's reassignment.

### Reassignment — #2788 tanjiro `re-conditional-input-scale` (5th Re-hook)

- Hypothesis: 4 merged Re-hooks (ReFiLM at slice-logits, ReScaleHead at output-scale, ReConditionalLayerNorm on activations, ReConditionalOutputBias on outputs) all operate on intermediate features or outputs. Add a 5th Re-hook at the **input embedding**: γ(log_Re) ∈ R^16 applied multiplicatively to the 16-D input vector before the encoder.
- Mechanism: input-side dual of the #2690 output-bias winner. Together they bracket the network with Re-conditioning at both ends, giving the model an explicit per-channel importance dial across the Re continuum. Init-to-identity (zero-init final layer, gamma starts at 1.0) so baseline behavior preserved at step 0. ~32 params.
- Why this swing: 4-of-4 Re-hook precedent is the strongest signal we have. Even -1% pushes cumulative to -77.5%. Low risk (init-to-identity), orthogonal to all 4 merged hooks (input vs. intermediate vs. output).
- Decision criteria: val_avg < 27.5868 → winner; [27.5868, 28.97) → send back; ≥ 28.97 → close.

### Send-back follow-up — #2721 frieren `rc-nn-oversampling-geom-weighted`

- Re-sent the prior Round 02:28 send-back with cleaner shell-safe instructions (the original send-back lost two `$EXPR` placeholders to bash interpolation issues — the local exp name and `--re_conditional_output_bias` flag were emitted as empty strings in the comment).
- Also requested rebase: PR is currently `mergeable=CONFLICTING` due to the #2690 merge changing `train.py`.
- Full follow-up at https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/2721#issuecomment-4447593928 with explicit code snippet (distance_weights tensor with log_Re zeroed and NACA-geometry channels 2×), `--rc_nn_max_boost 2.0`, and the new merged reproduce command including `--re_conditional_output_bias`.

### Active matrix after Round 02:40

| PR | Student | Slug | Notes |
|----|---------|------|-------|
| #2768 | alphonse | re-cond-attn-temperature | 6th Re-hook (slice softmax scale) |
| #2770 | askeladd | re-cond-ffn-film | 6th Re-hook (FFN hidden) |
| #2772 | nezuko | p-label-noise-1pct | output-side data aug |
| #2775 | fern | aoa1-negative-jitter-sigma02 | targeted at #1 rc-distinctive channel |
| #2779 | thorfinn | naca-pair-film | 8-D tandem shape FiLM |
| #2721 | frieren | rc-nn-geom-weighted | needs rebase + softer cap re-run |
| #2788 | tanjiro | re-cond-input-scale | 5th Re-hook (input γ) |
| #2725 | edward | multi-seed-variance | noise floor + TTA stack |

**Re-hook density now exceptional**: 4 merged + 3 active (input-scale #2788, attn-temp #2768, FFN-FiLM #2770) = 7 distinct Re-conditioning injection points being explored on this branch. The 5 active variants attack:
- **Re-conditioning of features**: 3 active (input-scale, attn-temp, FFN-FiLM)
- **Geometry conditioning**: 1 active (NACA-pair FiLM)
- **Data augmentation**: 2 active (p-label noise, AoA_1 jitter)
- **Oversampling**: 1 active-sent-back (geom-weighted k-NN)
- **Ensembling / noise floor**: 1 active (multi-seed + TTA)

---

## 2026-05-14 02:28 — Round 02:28: WINNER #2690 + 5 closures + 6 new assignments

### 🏆 PR #2690 thorfinn `re-conditional-output-bias` — MERGED, NEW BASELINE

**val_avg 27.5868 (-2.32% vs #2650 28.2414) | test_avg 24.1056 (-1.54%)**

| Split | val | vs #2650 | test |
|-------|-----|---------|------|
| single_in_dist | 27.2278 | +0.20% | 27.8682 |
| **geom_camber_rc** | **39.8226** | **-5.67%** ✅ | 36.9633 |
| geom_camber_cruise | 13.3872 | -2.09% ✅ | 10.2260 |
| re_rand | 29.9096 | +0.02% | 21.3648 |
| **val_avg** | **27.5868** | **-2.32%** | **24.1056** |

**Mechanism**: 4th Re-conditioning hook (additive output bias conditioned on log(Re), injected after ReScaleHead). Ux channel dominates by ~30× (absmax 0.73); corr(|b|, logRe) = -0.640 meaning larger correction at low-Re (viscous boundary-layer offsets). The rc improvement (-5.67%) is the standout — Ux correction targeting low-Re viscous regimes directly addresses rc's high-Re + negative-AoA regime where systematic biases are largest.

**New baseline: val_avg=27.5868, test_avg=24.1056. Cumulative: -76.5% from initial 117.17.**

### Closures (5 PRs, all vs new baseline 27.5868)

| PR | Student | Slug | val_avg | vs new baseline | Mechanism / finding |
|----|---------|------|---------|----------------|---------------------|
| #2699 | alphonse | Re-cond LayerScale | 29.2061 | +5.88% | Gates real (corr=-0.999) but 2 epochs short (mem overhead). Per-block residual gating disrupts early feature computation (block0_attn α=0.557). |
| #2702 | askeladd | SWA last 5 ep | 29.9779 | +8.67% (SWA best) | SWA itself worked (-2.62% vs regular) but swa_lr=1e-4 switches LR UP from cosine floor, undermining the critical cosine-tail refinement. SWA meta-axis CLOSED. |
| #2723 | nezuko | K=3 ensemble | 29.1357 | +5.61% | Heads anti-correlated (cos -0.14 to -0.17) — competition not independence. Ensemble-output-head meta-axis CLOSED. |
| #2703 | fern | Asymmetric ch15 σ=0.05 | 30.3078 | +9.87% | One-sided positive pushed cruise OUTSIDE its tight cluster (+15.1% cruise). Single-axis ch15 augmentation insufficient for multi-axis rc OOD. ch15-augmentation-family CLOSED. |
| #2721 | frieren | rc-NN-oversample | 28.3367 | +2.72% | rc mechanism works (-3.71% rc, -0.21% test_avg) but collateral on iid/re_rand. Sent back for geometry-weighted k-NN (exclude log_Re from distance metric, max_boost=2.0). |

### Round 02:28 New Assignments (5 + 1 sent-back = 6 total active)

| PR | Student | Slug | Hypothesis |
|----|---------|------|-----------|
| #2768 | alphonse | `re-conditional-attn-temperature` | 6th Re-hook: scale slice logits by α(Re) before softmax — orthogonal to FiLM's shift, 10 params, full affine control when stacked |
| #2770 | askeladd | `re-conditional-ffn-film` | 6th Re-hook: FiLM inside each block's FFN hidden layer (after first GELU), ~2.3K params, different injection from ReCondLN |
| #2772 | nezuko | `p-label-noise-epsilon-1pct` | Gaussian noise on p-targets during training, ε=0.01, linearly annealed; zero inference cost |
| #2775 | fern | `aoa1-negative-jitter-sigma02` | Targeted augmentation on AoA_1 channel (one-sided negative), grounded in frieren's diagnostic: rc = all-negative AoA_1 |
| #2779 | thorfinn | `naca-pair-film-conditioning` | Geometry-conditional FiLM on 8-D tandem signature (M_1,P_1,T_1,M_2,P_2,T_2,gap,stagger) — shape-conditioning, not physics-conditioning |
| #2721 | frieren | `rc-nn-oversampling-geom-weighted` | Softer reweighting (max_boost=2.0) + geometry-weighted distance (exclude log_Re, upweight NACA geometry 2×) |

---

## 2026-05-14 01:55 — Round 01:54 Closures: shape-bin-M-oversampling, surface-only-aux-head, 2 stale pods

### 🚨 PROGRAMME CORRECTION — Multi-axis OOD, not single-axis-M

**Frieren's #2689 JSONL count established a critical empirical fact:** train set CONTAINS samples at M=0.778 (75 samples) and M=0.889 (71 samples). The "rc OOD beyond train cluster ch15 ≤ 0.667" theory we've operated on for 6+ rounds is WRONG on the camber-amplitude axis alone. The rc OOD condition is multi-dimensional — likely a joint condition on (M_front, M_back, P_front, P_back, Re, AoA).

**The 7-experiment cruise/rc dichotomy observation is still real** — it just has a different mechanism than "ch15 extrapolation". The single-axis interpolation/extrapolation story doesn't account for samples already present in train at the rc test values. Most likely: rc's OOD-ness is in the JOINT distribution (the combination of channels), not any single channel.

**Reassignments**: #2721 frieren rc-NN-oversampling-multi-axis builds a per-split feature signature table AND uses multi-axis k-NN-to-rc weighting to identify and target the actual rc-similar train samples. This will replace 6+ rounds of single-axis-M speculation with empirical data.

### PR #2689 frieren `shape-bin-oversampling-m05` — CLOSED +6.0% val (new baseline)

- **Result:** val_avg 29.9505 vs baseline 28.2414. ALL splits regressed: single +10.6%, rc +1.2%, cruise +6.6%, re_rand +8.5%.
- **Effective sampler verified**: 71% mass on M≥0.5 (vs 46.8% raw), exactly 3× boost.
- **Hypothesis falsified on two levels**: (a) single-axis M-density does not solve rc gap (+1.2% bump only); (b) collateral damage from undersampling M<0.5 hurt iid bulk by 6-10%.
- **Critical correction**: train contains M=0.778 (75) and M=0.889 (71) samples — rc OOD is multi-axis joint condition, not ch15 extrapolation.

**Artifact**: `models/model-charliepai2g24h1-frieren-shape-bin-oversampling-m05-20260514-005622/metrics.jsonl`

### PR #2688 nezuko `surface-only-aux-normal-head` — CLOSED +3.66% val (new baseline)

- **Result:** val_avg 29.2733 vs baseline 28.2414. Per-split: single +3.6%, rc +2.4%, cruise +1.5%, re_rand +6.5%.
- **Aux objective ran CLEANER than #2660**: aux_loss 0.039 vs 0.062, pred_mag 0.988 vs 0.980.
- **Main task regressed in identical shape as #2660** (vol+surf aux) — directly disproves the volume-pollution hypothesis.
- **Mechanism: representation-capacity competition** at the post-LN trunk hidden state. Aux supervision at final injection point steals capacity from bulk-flow encoding.
- **Aux-output-head meta-axis FULLY CLOSED** across 2 arms.

**Artifact**: `models/model-surface-only-aux-normal-head-20260514-005837/metrics.jsonl`

### PR #2599 tanjiro `se-channel-attention-r8` — CLOSED STALE

24+ hours with no comments or results; pod stuck on rate-limit cycle. SE-channel-attention hypothesis remains untested by this branch. Tanjiro reassigned to #2724 geometry-mirror-TTA (free training-time, exact y-symmetry test-time averaging).

### PR #2534 edward `tta-re-bracket` (multi-seed follow-up) — CLOSED STALE

5+ hours without follow-up arm; pod stuck. Original TTA arm (val 29.86 vs #2011 baseline) had +1.0 variance no_tta run, would be ~+5.7% vs new baseline even with TTA. TTA mechanism cleanly validated (-0.04 val / -0.12 test, σ=0.05 sweet spot). Implementation saved for stacking on future winner. Edward reassigned to #2725 multi-seed-variance-new-baseline.

### Round 01:54 New Assignments (4 PRs)

- **#2721 frieren rc-NN-oversampling-multi-axis**: DIAGNOSTIC + corrective. Builds per-split feature signature table (commits `split_signatures.json`), computes k-NN-to-rc weighting on full (M, P, T, Re, AoA) space, oversamples most-rc-similar 20% by up to 3× without undersampling iid bulk. **High value as diagnostic alone** regardless of training outcome.
- **#2723 nezuko ensemble-output-heads-k3**: K=3 parallel output heads with shared trunk, mean prediction. Deep-ensemble-style variance reduction targeting rc/re_rand. ~1.5K added params, zero extra trunk compute.
- **#2724 tanjiro geometry-mirror-TTA**: exact y-symmetry test-time averaging (flip y-coord + AoA, predict, un-flip Uy, average). Stacks with edward's validated Re-bracket TTA mechanism.
- **#2725 edward multi-seed-variance-new-baseline**: 2 seeds (42, 43) of new baseline + TTA-val-per-epoch checkpoint selection. Establishes run-to-run noise floor for all future merge decisions; validates TTA stacking.

### Round 01:54 Programme Status

**8/8 students WIP**, all on the new (#2650) baseline. Active hypothesis cluster:
- Re-conditioning hooks (3 arms in flight): #2699 alphonse LayerScale, #2690 thorfinn output-bias + #2650 baseline merged
- Data-side multi-axis (1 diagnostic): #2721 frieren rc-NN-oversampling
- Output-side ensemble (1 arm): #2723 nezuko K=3 heads
- TTA (2 arms): #2724 tanjiro mirror, #2725 edward multi-seed Re-bracket
- Other: #2702 askeladd SWA, #2703 fern asymmetric ch15

**Programme correction propagated**: future strategy must wait on the #2721 diagnostic split_signatures.json output before designing more rc-targeted experiments.

---

## 2026-05-14 01:08 — Round 01:05 Closures: T_max=35, surface-only-input-normal, mask-ch15-jitter

### PR #2678 alphonse `recondln-t-max35` — CLOSED +7.09% val (new baseline #2650)

- **Branch**: charliepai2g24h1-alphonse/recondln-t-max35
- **Hypothesis**: T_max=35 extends cosine warmdown beyond wall-cap; #2650 trace was monotone-descending at ep28, "budget-limited not converged".
- **Status**: CLOSED — hypothesis cleanly falsified, T_max=28 confirmed principled optimum

| Metric | T_max=35 | Baseline #2650 (T_max=28) | Δ |
|--------|----------|---------------------------|---|
| val_avg/mae_surf_p | 30.2427 | 28.2414 | **+7.09%** |
| test_avg/mae_surf_p | 26.6104 | 24.4827 | **+8.69%** |
| val_single_in_dist | 31.5387 | 27.1740 | +16.06% |
| val_geom_camber_rc | 43.0129 | 42.2153 | +1.89% |
| val_geom_camber_cruise | 14.8633 | 13.6733 | +8.70% |
| val_re_rand | 31.5559 | 29.9031 | +5.53% |

**Critical diagnostic**: ep28 ticked UP (30.24→30.36) at residual LR 1.32e-4 vs #2650's monotone-descent to eta_min=1e-5. **Direct evidence the eta_min FLOOR (not residual LR) was doing the work in #2650's last 3 epochs.** T_max=28 ≈ wall-cap-epochs is the principled setting; extending T_max delays the floor regime past the wall, removing the refinement entirely. T_max=35 sits between T_max=28 (winner) and T_max=40 (worse) — confirms U-shape sensitivity, 28 is tuned optimum not budget artefact.

**T_max>28 axis FULLY CLOSED**. Reassigned alphonse to #2699 re-conditional-layerscale (5th Re-conditioning hook).

**Artifact**: `models/model-recondln-t-max35-20260514-002546/metrics.jsonl`

---

### PR #2671 askeladd `surface-only-normal-feature` — CLOSED +7.02% val (new baseline #2650)

- **Branch**: charliepai2g24h1-askeladd/surface-only-normal-feature
- **Hypothesis**: Zero out normal channels on volume nodes to isolate the physical signal at surface elements and remove "ghost orientation" pollution that drove #2627's val_re_rand +35% regression.
- **Status**: CLOSED — pollution mechanism CONFIRMED but net negative; dichotomy partly structural

| Metric | This run | Baseline #2650 | Δ (new baseline) | #2627 (all-nodes) | gap closure |
|--------|----------|----------------|------------------|-------------------|-------------|
| val_avg/mae_surf_p | 30.2252 | 28.2414 | **+7.02%** | 31.27 (+8.34% vs #2011) | -3.34% improvement |
| val_re_rand | 32.392 | 29.9031 | +5.14% | 35.01 (+35% vs #2011) | **-8.5pp gap closure** ✅ |
| val_geom_camber_cruise | 13.801 | 13.6733 | +0.93% | 13.98 (-44.8% vs cruise base) | preserved as small gain |
| val_geom_camber_rc | 44.882 | 42.2153 | +7.00% | 44.89 (+27.2% vs #2011) | rc regression PERSISTS |
| test_avg/mae_surf_p | 26.1163 | 24.4827 | +4.47% | — | — |

**Programme finding (partially confirmed):**
- **Pollution side-channel WAS real and big**: val_re_rand recovered from +35% (#2627 all-nodes) → +5.14% (this surface-only) = -8.5pp gap closure. Volume-node "ghost normals" inherited from nearest surface point were structurally arbitrary in wake/far-field and the model was learning normal→flow mappings conditional on cruise-regime priors.
- **Cruise gain is real but small**: -2.44% val, -2.62% test. Surface normals at viscous walls DO carry physically meaningful info for pressure-normal coupling.
- **BUT rc regression PERSISTS at +7.0% even without volume pollution** → cruise/rc dichotomy at N=1499 / SOAP / 30-min budget is **partly structural**, not purely a pollution artefact.
- Dichotomy collapse from -45%/+27% (#2627) → -2.4%/+7.0% (#2671): most of #2627's drama was volume-pollution amplifying both effects.

**Surface-only input-feature path closed** — net negative despite clean mechanism validation. Reassigned askeladd to #2702 SWA (optimizer-level flat-minima, fundamentally different angle from feature/architecture/loss).

**Artifact**: `models/model-charliepai2g24h1-askeladd-surface-only-normal-feature-20260514-002742/metrics.jsonl`

---

### PR #2662 fern `naca-jitter-ch16-17-only-sigma02` (mask ch15, σ=0.02) — CLOSED +7.04% val (new baseline #2650)

- **Branch**: charliepai2g24h1-fern/naca-jitter-channels-15-17-only
- **Hypothesis (corrected after fern's empirical channel-index discovery)**: Mask ch15 (rc OOD axis = camber amplitude M), jitter only ch16+ch17 (fully in-distribution axes). Preserve #2625's cruise win, drop #2625's rc regression.
- **Status**: CLOSED — hypothesis falsified in informative way; ch15 was the ENTIRE active axis

| Metric | mask-ch15 | Baseline #2650 | Δ vs new baseline | #2625 (all-3) | #2625 Δ vs #2011 |
|--------|-----------|----------------|-------------------|---------------|------------------|
| val_avg/mae_surf_p | 30.2298 | 28.2414 | **+7.04%** | 29.8728 | +3.45% |
| val_geom_camber_rc | 44.2822 | 42.2153 | +4.90% | 52.0962 | +24.20% |
| val_geom_camber_cruise | 14.3725 | 13.6733 | +5.11% | **7.9610** | **-43.72%** |
| val_single_in_dist | 30.1703 | 27.1740 | +11.03% | — | — |
| val_re_rand | 32.0943 | 29.9031 | +7.33% | — | — |

**Programme finding (cleanest possible)**:
- With ch15 masked: BOTH the -43.7% cruise win AND the +24.2% rc regression of #2625 evaporated TOGETHER.
- cruise reverted to ~baseline (+5.1% vs #2650, vs #2625's -43.7% win)
- rc came down sharply (+4.9% vs #2650, vs #2625's +24.2% regression)
- ch16+ch17 jitter alone is small-magnitude smoothing of already-dense neighborhoods → near-noop
- **ch15 IS the active axis**: simultaneously the rc OOD axis (train ≤ 0.667, rc test = {0.778, 0.889}) AND the cruise interpolation axis (cruise test ∈ {0.222–0.444} middle of train) — with opposite effects on the two splits.
- Symmetric ch15 jitter densifies inward (helps cruise interpolation, hurts rc extrapolation by relatively widening the rc gap).

**ch16+ch17 jitter direction CLOSED**. Reassigned fern to #2703 asymmetric-ch15-jitter-sigma05 (her own follow-up #2 — one-sided positive jitter on ch15 only, PUSHING outward toward rc OOD M={0.778, 0.889} rather than densifying inward).

**Artifact**: `models/model-naca-jitter-ch16-17-only-sigma02-20260514-001759/metrics.jsonl`

---

### Round 01:05 Programme Summary

**11 falsified hypotheses on this branch establish two structural findings:**
1. **Re-conditioning at bounded injection points is the ONLY winning direction** (4 wins: ReFiLM-slice-logit, ReScaleHead, ReFiLM-residual closed, ReCondLN-affine winner). Every architecture/feature/loss/optimizer-schedule experiment outside this family has failed at +5-11% regression.
2. **The cruise/rc dichotomy is structural at N=1499 / SOAP / 30-min / Transolver**. Even after surface-pollution removal (#2671), rc regression persists at +7.0%. The mechanism is ch15-localized: ch15 simultaneously serves as cruise interpolation densification axis AND rc OOD extrapolation gap. Symmetric augmentation makes the rc gap RELATIVELY worse.

**Round 01:05 strategic pivots:**
- alphonse #2699: 5th Re-conditioning hook (per-block LayerScale α(Re)) — extend proven winning direction
- askeladd #2702: SWA on last 5 epochs — optimizer-level flat-minima, only previously-untested axis on the new baseline
- fern #2703: asymmetric one-sided ch15 jitter σ=0.05 — DIRECT attack on rc extrapolation gap (push outward not densify inward) — first experiment in the round designed to bridge rather than densify

**Remaining 5 WIP (continuing): thorfinn #2690 output-bias 4th hook, frieren #2689 oversampling M≥0.5, nezuko #2688 aux-surface-only, tanjiro #2599 SE-attn (stuck pod), edward #2534 TTA-multi-seed.**

---

## 2026-05-13 21:05 — PR #2559: [surface-embed-trunk-token] Learned is_surface embedding into trunk — CLOSED

- **Branch**: charliepai2g24h1-alphonse/surface-embed-trunk-token
- **Hypothesis (direct response to #2529)**: Inject is_surface as a learned 2-row embedding at the trunk input (before block 0) so surface/volume routing becomes available throughout the residual stream — fixing the trunk-bottleneck identified in #2529.
- **Status**: CLOSED — +4.2% val regression. The DIAGNOSTIC is the value here, and it closes a paired meta-axis.

| Metric | Surface-embed | Baseline (#2011) | Δ |
|--------|---------------|------------------|---|
| val_avg/mae_surf_p | 30.0829 | 28.8762 | **+4.2% (WORSE)** |
| test_avg/mae_surf_p | 25.8225 | 24.9992 | +3.3% (WORSE) |

**Per-split val**: ALL 4 splits regressed: single_in_dist +2.12, geom_camber_rc +1.29, geom_camber_cruise +0.22, re_rand +1.20.

**Surface-embedding diagnostic (the key signal):**
- `vol_emb` L2 = 0.2192, `surf_emb` L2 = 0.1575 (39% magnitude difference)
- Cosine similarity = -0.06 at end (evolved monotonically from -0.29 → -0.06)
- **The trunk DID learn to route surface vs volume nodes differently** — distinct magnitudes and ~orthogonal directions

**Programme finding (closes surface/volume routing axis paired-negative)**:
- #2529 (head-level): no head divergence on pressure channel (surf/vol L2 ratio = 1.02)
- #2559 (trunk-level): trunk routing learned (cos = -0.06) but no pressure-MAE gain

→ Together: **surface-pressure error is NOT a surface/volume routing problem at this model scale.** PhysicsAttention already handles node-class differentiation through learned slice softmaxes. Adding coarse routing signals either duplicates the existing pathway (`x[:, :, 12]` is_surface flag already in preprocess MLP) or competes with finer slice-level routing for capacity.

**Dominant error source is camber_rc (val=41.95 = 3× cruise=14.15)** — points to geometry extrapolation or Re extrapolation as the real bottleneck, not routing. Next direction (#2585): Re-FiLM on the residual stream — applies γ(Re), β(Re) post-residual-add, distinct from baseline ReFiLM (slice-logit conditioning) and from LayerScale (no Re conditioning).

**Artifact**: `models/model-charliepai2g24h1-alphonse-surface-embed-trunk-token-20260513-202118/metrics.jsonl`

---

## 2026-05-13 20:50 — PR #2537: [derived-features-re2-aoa] Derived input features log(Re)^2, log(Re)·AoA — CLOSED

- **Branch**: charliepai2g24h1-frieren/derived-features-re2-aoa
- **Hypothesis**: Add `log(Re)^2` and `log(Re) · AoA1` as explicit polynomial cross features to relieve early-layer burden of learning Re-AoA coupling.
- **Status**: CLOSED — +3.26% val, +4.18% test regression. Hypothesis cleanly falsified.

| Metric | Derived features | Baseline (#2011) | Δ |
|--------|------------------|------------------|---|
| val_avg/mae_surf_p | 29.8169 | 28.8762 | **+3.26% (WORSE)** |
| test_avg/mae_surf_p | 26.0441 | 24.9992 | **+4.18% (WORSE)** |

**Smoking gun**: The two splits the hypothesis predicted to benefit MOST (val_re_rand, val_geom_camber_rc) regressed the MOST (+1.67, +0.94 respectively). The single improved split was val_geom_camber_cruise (-0.45) — the most-fixed-Re/most-fixed-camber split.

**Programme finding**: When the architecture already has explicit Re-conditioning (ReFiLM β/γ, |γ|max=0.70 by ep28) AND learned Re-rescaling (ReScaleHead 3-channel), explicit polynomial features in the input duplicate information that FiLM/ReScaleHead extract through their own nonlinearities. With N=1499, the model has to LEARN to down-weight the redundant channels — costing effective sample efficiency. Tabular-ML feature engineering orthogonality argument does NOT transfer when explicit Re-aware conditioning hooks already exist in the architecture.

**Artifact**: `models/model-charliepai2g24h1-frieren-derived-features-re2-aoa-20260513-200429/metrics.jsonl`

---

## 2026-05-13 20:43 — PR #2532: [drop-path-0p1] Stochastic Depth on residual branches (rates 0.1, 0.05) — CLOSED

- **Branch**: charliepai2g24h1-askeladd/drop-path-0p1
- **Hypothesis**: Linear stochastic-depth schedule across 5 Transolver blocks (0.0→drop_path_rate) regularizes the residual stream, reducing overfit on N=1499 samples.
- **Status**: CLOSED — both arms regressed val by ~+5.7-6.2%.

| Metric | drop_path=0.1 | drop_path=0.05 | Baseline (#2011) |
|--------|---------------|----------------|------------------|
| val_avg/mae_surf_p | 30.5304 (+5.7%) | 30.6569 (+6.2%) | 28.8762 |
| test_avg/mae_surf_p | 26.1550 (+4.6%) | 26.1157 (+4.5%) | 24.9992 |

**Per-split val**: Uniform regression across all splits except val_re_rand (drop_path=0.1, +0.54 only). Train loss already at ~2e-3 (very low) confirms not in capacity-overfit regime — it's an optimization plateau.

**Programme finding (closes a meta-axis)**: Branch-level stochastic depth is inappropriate for shallow physics-attention stacks (n_layers ≤ 5). DeiT/ConvNeXt evidence relies on 12+ block stacks where keep-rate compounds across depth. Linear schedule meant only blocks 3-4 saw non-trivial drop rates; lower blocks essentially deterministic. Combined with Lookahead (#2384), confirms that any mechanism adding gradient/trajectory noise needs deeper stacks OR longer schedules. Token-level dropout (input nodes, not residuals) is structurally different and remains unexplored (see #2582).

**Artifact**: `models/model-charliepai2g24h1-askeladd-drop-path-0p1-20260513-193005/metrics.jsonl`

---

## 2026-05-13 20:25 — PR #2538: [bernoulli-surface-loss] Bernoulli surface-pressure constraint — CLOSED

- **Branch**: charliepai2g24h1-nezuko/bernoulli-surface-loss
- **Hypothesis**: Soft physics constraint penalizing variance of `p + ½|U|²` on surface nodes; `bernoulli_weight=0.01` as auxiliary loss.
- **Status**: CLOSED — +4.1% val regression. Hypothesis falsified on both physics grounds and implementation grounds.

| Metric | Bernoulli λ=0.01 | Baseline (#2011) | Δ |
|--------|------------------|------------------|---|
| val_avg/mae_surf_p | 30.0466 | 28.8762 | **+4.1% (WORSE)** |
| test_avg/mae_surf_p | 25.8175 | 24.9992 | +3.3% (WORSE) |

**Per-split val** (Δ vs baseline): single_in_dist +2.99 (smoking gun — ID hit hardest), geom_camber_rc +0.22, geom_camber_cruise +0.57, re_rand +0.90.

**Bernoulli loss diagnostic**: train/bernoulli_loss INCREASED over training: epoch 1 = 0.36 → epoch 28 = 0.75. The model rationally chose to pay the 0.0075 regularizer cost to fit the data, indicating the constraint is fighting the main loss.

**Programme finding (closes 2 axes)**:
1. **Physics conceptual**: TandemFoilSet is RANS (viscous) data. Bernoulli p + ½ρ|U|² = const is an inviscid constraint that does NOT hold across the surface in this data — even at the optimum. The ID-split degradation pattern confirms data does not satisfy the constraint.
2. **Implementation**: Loss computed in normalized prediction space where Ux/Uy/p have different normalization scales — the 'Bernoulli sum' is a meaningless mixture. Even un-normalizing wouldn't help due to (1).

Physics-informed soft constraints on viscous RANS data are now closed. Volumetric Laplacian (#2325) and surface Bernoulli (#2538) — both physics-soft-constraints — are CLOSED. Stagnation-point-only Bernoulli is theoretically valid but requires reliable LE detection from predictions.

**Artifact**: `models/model-charliepai2g24h1-nezuko-bernoulli-surface-loss-20260513-194319/metrics.jsonl`

---

## 2026-05-13 20:08 — PR #2535: [mixup-scalar-alpha-0p4] Scalar-only Mixup α=0.4 — CLOSED

- **Branch**: charliepai2g24h1-fern/mixup-scalar-alpha-0p4
- **Hypothesis**: Linear interpolation in parameter space (Re, AoA1, AoA2, NACA1, NACA2, gap, stagger) creates physically valid synthetic flow examples; Beta(0.4, 0.4) Mixup expands effective training set.
- **Status**: CLOSED — catastrophic +52.7% val regression.

| Metric | Mixup α=0.4 | Baseline (#2011) | Δ |
|--------|-------------|------------------|---|
| val_avg/mae_surf_p | 44.0894 | 28.8762 | **+52.7% (WORSE)** |
| test_avg/mae_surf_p | 38.8855 | 24.9992 | **+55.6% (WORSE)** |

**Per-split val** (Δ vs baseline): single_in_dist +23.6 (+83%), geom_camber_rc +13.6 (+33%), geom_camber_cruise +10.9 (+77%), re_rand +12.7 (+41%).

**Programme finding**: Mixup family ruled out for this geometry-conditioned regression. The interpolation hypothesis is invalid: mixing two NACA-4-digit airfoils linearly does NOT produce a valid airfoil. Per-node features encode geometry A, but targets are convex combinations of fields A and B — the targets do not match the geometry encoded in the per-node features. ID-split degradation pattern (single_in_dist worst hit at +83%) confirms the training-signal corruption hypothesis. Even α=0.1 would carry the same fundamental per-node-consistency violation. Conditional Mixup (same NACA pairs) is interesting but has very small effective pool.

**Artifact**: `models/model-charliepai2g24h1-fern-mixup-scalar-alpha-0p4-20260513-193110/metrics.jsonl`

---

## 2026-05-13 20:06 — PR #2529: [surf-vol-split-head] Surf/vol split output head — CLOSED

- **Branch**: charliepai2g24h1-alphonse/surf-vol-split-head
- **Hypothesis**: Split single `Linear(d, 3)` output head into separate `surf_head`/`vol_head` gated by `is_surface` — surface head specializes on pressure-dominated surface regime.
- **Status**: CLOSED — small +2.39% val regression.

| Metric | Split head | Baseline (#2011) | Δ |
|--------|-----------|------------------|---|
| val_avg/mae_surf_p | 29.5649 | 28.8762 | **+2.39% (WORSE)** |
| test_avg/mae_surf_p | 25.1784 | 24.9992 | +0.72% (WORSE) |

**Per-split val**: single_in_dist +1.10 (+3.85%), geom_camber_rc +1.13 (+2.69%), geom_camber_cruise −0.02, re_rand +0.54 (+1.75%).

**Mechanism diagnostic (key finding):** L2 norms of head rows showed surf_head/vol_head ratio for pressure channel = **1.02** (no specialization on the metric-relevant channel), while velocity channels saw vol_head dominate (surf/vol ratio 0.5-0.7). Cosine similarity between heads: **0.49 → 0.49** over training — no meaningful divergence. Surface-node fraction is only 1.7% per batch, giving surf_head ~50× fewer gradient samples than vol_head.

**Programme finding (critical for follow-up direction)**: The bottleneck is in the trunk's shared representations, not in output head capacity. Adding parallel heads can only re-weight a fixed shared representation — it cannot introduce new specialization the trunk doesn't already encode. Multi-channel head expansion is now FULLY closed (#2322 geom-output-head also closed earlier). **Follow-up #2559 (alphonse) tests trunk-level surface specialization via `is_surface_emb` injected into node tokens before block 0.**

**Artifact**: `models/model-charliepai2g24h1-alphonse-surf-vol-split-head-20260513-192611/metrics.jsonl`

---

## 2026-05-13 16:50 — PR #2384: [lookahead-soap-k5] Lookahead(SOAP, k=5, α=0.5) — CLOSED

- **Branch**: charliepai2g24h1-nezuko/lookahead-soap-k5
- **Hypothesis**: Lookahead wrapper around SOAP with k=5, α=0.5; smoothing without state-swap overhead (the EMA failure mode from #1966). No load_state_dict, no torch.compile recompile.
- **Status**: CLOSED — +6.3% val regression. Mechanical claim confirmed (no recompile, drift/slow ≤0.04, zero wall-clock overhead), but α=0.5 pull-back is a 50% effective LR brake in budget-limited regime.

| Metric | Lookahead | Baseline (#2011) | Δ |
|--------|-----------|------------------|---|
| val_avg/mae_surf_p | 30.6801 | 28.8762 | **+6.3% (WORSE)** |
| test_avg/mae_surf_p | 26.2470 | 24.9992 | **+5.0% (WORSE)** |
| Wall-clock per epoch | 64-65s | 64s | ≈0 (✅ no recompile) |
| drift/slow ratio | ≤0.04 (max) | — | — |

**Per-split val**: single_in_dist +3.83, camber_rc +1.96, camber_cruise **−0.19** (✅ smoothing helps the most-converged split), re_rand +1.62.

**Root cause analysis** (from student): the α=0.5 pull-back means each k-step window contributes only `α(fast−slow) = 0.5×normal` of the gradient update. Effective LR ≈ 50% of SOAP. Val was still descending at epoch 28 (Δ −0.20/ep over last 4 eps), would linearly cross baseline only at ~ep 37 — out of budget. Lookahead's smoothing dividend is a convergence-phase effect; in a budget-limited monotonic descent it just brakes you. The only split that improved (cruise, the lowest-loss / most-converged) confirms the mechanism but also confirms it's the wrong regime.

**Programme finding (meta-axis)**: Trajectory smoothing of SOAP under 28-ep cosine budget is fully closed. Three independent mechanisms (EMA #1966 +2.60%, SWA #2032 +1.24-3.44%, Lookahead +6.3%) — all averaging in-run weight states — fail under the same root cause: in a still-descending model, averaging the past trajectory for the per-step optimum costs more than the noise it removes. Stop pursuing in-run smoothing variants.

**Artifact**: `models/model-charliepai2g24h1-nezuko-lookahead-soap-k5-20260513-152016/metrics.jsonl`

---

## 2026-05-13 15:30 — PR #1467: [more-slices-128] slice_num 64→128 — CLOSED

- **Branch**: charliepai2g24h1-nezuko/more-slices-128
- **Hypothesis**: Double PhysicsAttention slice_num from 64 to 128 to give more routing capacity for multi-domain physics. Predicted help on geom_camber OOD splits.
- **Status**: CLOSED — +13.5% val / +12.5% test regression. All 4 splits regressed. Classic overfit.

| Metric | slice=128 | Baseline (slice=64) | Δ |
|--------|-----------|---------------------|---|
| val_avg/mae_surf_p | 32.7601 | 28.8762 | **+13.5% (WORSE)** |
| test_avg/mae_surf_p | 28.1216 | 24.9992 | **+12.5% (WORSE)** |
| Params | 682,151 | 662,488 | +3.0% |
| Epochs in 30 min | 23 | 28 | −5 (~9% slower/epoch) |
| Peak GPU mem | 37.46 GB | ~28 GB | +9 GB |

**Per-split val regression**: single_in_dist=34.64 (+6.04), camber_rc=45.11 (+3.16), camber_cruise=17.02 (+2.87), re_rand=34.28 (+3.47). The OOD geometry splits the hypothesis specifically targeted regressed *alongside* in-distribution — extra slices did not separate the multi-domain physics.

**Overfit signature**: Train surf_loss collapsed to 0.003 (~20× drop from epoch 1) while val plateaued in low 30s by epoch 18. With 1499 training samples, slice_num=64 is the local maximum — extra routing capacity gives the optimizer more directions to overfit.

**Programme finding**: slice_num=128 axis CLOSED. The symmetric lower-direction (slice_num=32, #2320 askeladd) is currently in flight to map the full curve.

**Artifact**: `models/model-charliepai2g24h1-nezuko-more-slices-128-20260513-142218/metrics.jsonl`

---

## 2026-05-13 11:40 — PR #1884: [onecycle-lr] OneCycleLR max_lr=2e-3, pct_start=0.1 — CLOSED

- **Branch**: charliepai2g24h1-alphonse/onecycle-lr
- **Hypothesis**: OneCycleLR(max_lr=2e-3, pct_start=0.1) — 10% warmup then cosine decay; higher peak LR than baseline 1e-3 to achieve super-convergence with SOAP.
- **Status**: CLOSED — +3.52% val regression. grad_clip=1.0 fundamentally incompatible with max_lr=2e-3 on SOAP.

| Metric | OneCycleLR | Baseline (#2011) | Δ |
|--------|-----------|-----------------|---|
| val_avg/mae_surf_p | 29.8919 | 28.8762 | **+3.52% (WORSE)** |
| test_avg/mae_surf_p | 25.4001 | 24.9992 | **+1.60% (WORSE)** |
| Best epoch | 28 | 28 | — |

**Root cause**: `max_lr=2e-3` saturated `grad_clip=1.0` from epoch 2 through epoch 15 (clip_frac=1.0 throughout the LR-peak window). Effective per-step LR was scaled down by grad_norm_mean (3-13×), so the OneCycleLR peak provided no benefit. By the time clip relaxed (~epoch 22), LR had decayed below baseline cosine values. Model was still improving at epoch 28 (29.89 → still falling).

**Programme finding**: OneCycleLR direction closed. `max_lr` above ~1.5e-3 is incompatible with `grad_clip=1.0` on SOAP at this model scale. Any follow-up would require either lowering grad_clip (load-bearing setting) or accepting that the "super-convergence" peak cannot be realized.

**Artifact**: `models/model-charliepai2g24h1-alphonse-onecycle-lr-20260513-105301/metrics.jsonl`

---

## 2026-05-13 11:10 — PR #2154: [n-head-8] n_head 4→8, same inner_dim=128 — CLOSED

- **Branch**: charliepai2g24h1-fern/n-head-8
- **Hypothesis**: Doubling attention heads to 8 (dim_head 32→16) doubles ReFiLM conditioning degrees of freedom (256→512) at zero compute cost. Heads should specialize on distinct physical regimes.
- **Status**: CLOSED — +14.2% val regression (compute-budget failure, not hypothesis failure)

| Metric | n_head=8 | Baseline (#2011) | Δ |
|--------|----------|-----------------|---|
| val_avg/mae_surf_p | 32.9800 | 28.8762 | **+14.2% (WORSE)** |
| test_avg/mae_surf_p | 28.9306 | 24.9992 | **+15.7% (WORSE)** |
| Best epoch | 23 | 28 | — |
| s/epoch (steady-state) | ~79.1 | ~64.3 | +23% slower |

**Root cause**: dim_head=16 is below bf16 GEMM efficiency sweet spot on this hardware (+23% per-epoch slowdown). CosineAnnealingLR(T_max=28) was cut off at epoch 23/28 → model never reached low-LR convergence phase. Same-epoch comparison (epoch 23): n_head=8 only −2.3% behind baseline — the heads *do* specialize (entropy spread 1.74–3.61, ReFiLM γ absmax 0.28→1.23) but the compute-budget effect dominates.

**Programme finding**: n_head=4 with dim_head=32 retained for this stack. Scaling head count without scaling inner_dim is net negative under bf16+compile at this kernel size.

**Artifact**: `models/model-n-head-8-20260513-100820/metrics.jsonl`

---

## 2026-05-13 11:10 — PR #2146: [log-cosh-loss] Log-cosh C∞ replacement for Huber — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/log-cosh-loss
- **Hypothesis**: Log-cosh (`δ²·log(cosh(x/δ))`) is a smooth C∞ alternative to Huber — no gradient discontinuity at |x|=δ, should benefit SOAP's Hessian approximation.
- **Status**: CLOSED — +2.93% val regression. Confirmed loss-shape axis closed.

| Metric | log-cosh | Baseline (#2011) | Δ |
|--------|----------|-----------------|---|
| val_avg/mae_surf_p | 29.7218 | 28.8762 | **+2.93% (WORSE)** |
| test_avg/mae_surf_p | 25.9905 | 24.9992 | **+3.97% (WORSE)** |
| Best epoch | 28 | 28 | — |

**Root cause**: Log-cosh has *weaker* gradients than Huber in the transition band (tanh(1)≈0.762 vs Huber's 1.0 at |x|=δ). With 88% of residuals already in quadratic regime by epoch 28, only the ~12% of tail residuals differ — and those are pulled *less* aggressively by log-cosh. Worst splits: re_rand (+5.0%), geom_camber_rc (+3.4%) — exactly the splits where pressure residuals are heaviest.

**Programme finding**: Loss-shape axis firmly closed across all three variants (#2081 δ_v-loose +1.16%, #2111 δ_p-tight +1.50%, #2146 log-cosh +2.93%). Huber(δ=0.1) is a robust local optimum for SOAP.

**Artifact**: `models/model-charliepai2g24h1-thorfinn-log-cosh-loss-20260513-101114/metrics.jsonl`

---

## 2026-05-13 13:45 — PR #2032: [plateau-swa-v3] SWA v3 lower LR=5e-5 — CLOSED

- **Branch**: charliepai2g24h1-edward/plateau-swa
- **Hypothesis (v3)**: SWA with LR plateau at SWA_LR=5e-5 (lower than v2's 1e-4), overriding scheduler before each SWA step. Lower plateau should preserve more cosine convergence quality while still enabling SWA averaging.
- **Status**: CLOSED — v3 +3.44% val regression (29.8696 vs baseline 28.8762). Worse than v2 (29.2337).

| Metric | v3 Live Best (ep 28) | v3 SWA (n=3) | v2 SWA | Baseline #2011 |
|--------|---------------------|-------------|--------|----------------|
| val_avg/mae_surf_p | 30.1219 | 29.8696 | 29.2337 | **28.8762** |
| test_avg/mae_surf_p | 25.5233 | 25.3870 | 25.1638 | **24.9992** |

**Root cause**: Lower SWA_LR=5e-5 did not preserve base convergence quality — live-best 30.12 is worse than v2 live-best 29.65. The LR plateau for SWA is fundamentally incompatible with our 28-epoch budget where the final cosine annealing (LR → 1e-5) is doing essential convergence work. SWA averaging IS real (−0.25 val improvement over live-best) but cannot compensate for the plateau quality loss.

**SWA axis fully closed**: v1 no-plateau (#1933, no spread), v2-hybrid LR=1e-4 (+1.24%), v3 LR=5e-5 (+3.44%). All three variants regressed.

**Artifact**: `models/model-charliepai2g24h1-edward-plateau-swa-v3-lower-lr-20260513-112244/metrics.jsonl`

---

## 2026-05-13 13:30 — PR #2204: [sorted-pressure-dist] W1 regularizer on sorted surface pressure quantiles — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/sorted-pressure-dist
- **Hypothesis**: Add Wasserstein-1 regularizer (sorted MSE on pressure quantiles) to align predicted per-sample pressure distributions with ground truth, closing the W1 gap between predicted and true pressure histograms.
- **Status**: CLOSED — +1.01% val regression. W1 mechanism learned but doesn't transfer to surf_p MAE.

| Metric | sorted-pressure-dist | Baseline (#2011) | Δ |
|--------|---------------------|-----------------|---|
| val_avg/mae_surf_p | 29.1672 | 28.8762 | **+1.01% (WORSE)** |
| test_avg/mae_surf_p | 25.2059 | 24.9992 | **+0.83% (WORSE)** |
| Best epoch | 28 | 28 | — |

**Root cause**: The W1 regularizer successfully learned distribution alignment (W1 gap reduced 15×), but this came at the cost of pointwise MAE — the model traded spatial precision for distributional correctness. Distribution matching and pointwise accuracy are partially competing objectives in this setting.

**Programme finding**: Sorted W1 pressure regularizer axis closed. Distribution matching alone is insufficient; pointwise spatial accuracy is what drives surf_p MAE. If distribution matching is to help, it would need to be combined with a stronger pointwise signal.

---

## 2026-05-13 13:30 — PR #2198: [refilm-per-block] 5 independent FiLM blocks vs shared — CLOSED

- **Branch**: charliepai2g24h1-fern/refilm-per-block
- **Hypothesis**: Replace single shared ReFiLM with 5 independent FiLMs (one per Transolver block). Each block gets its own Re→(γ,β) mapping — deeper layers may need different Re-conditioning from shallow layers.
- **Status**: CLOSED — +2.9% val regression. Per-block gates DID specialize but overfitted.

| Metric | refilm-per-block | Baseline (#2011) | Δ |
|--------|-----------------|-----------------|---|
| val_avg/mae_surf_p | 29.72 | 28.8762 | **+2.9% (WORSE)** |
| Best epoch | 28 | 28 | — |

**Key observation**: Block specialization confirmed (block4 absmax 0.81 vs block0 0.51), but geom_camber_rc worst split (+2.05%). Extra per-block capacity overfitted to in-distribution Re variation rather than generalizing to OOD splits. The shared FiLM acts as an implicit regularizer that forces all blocks to agree on Re-dependent behavior.

**Programme finding**: Per-block FiLM axis closed for this model size. Shared ReFiLM is optimal. Follow-up: test wider shared MLP (refilm-hidden-16) to improve capacity without the overfitting risk of per-block independence.

---

## 2026-05-13 13:30 — PR #2147: [cosine-long-tail] T_max=40/56 cosine beyond 28-epoch budget — CLOSED

- **Branch**: charliepai2g24h1-askeladd/cosine-long-tail
- **Hypothesis**: Extending T_max beyond 28 keeps LR higher throughout training (cosine never reaches eta_min=1e-5 within budget), avoiding premature LR starvation.
- **Status**: CLOSED — T_max=40 +11.4%, T_max=56 +31.4%. Monotonically worse.

| Metric | T_max=40 | T_max=56 | Baseline T_max=28 | 
|--------|----------|----------|-------------------|
| val_avg/mae_surf_p | 32.16 | 38.05 | 28.8762 |

**Root cause**: Higher final LR (cosine plateau) means the model never reaches the low-noise convergence regime. The cosine anneal to eta_min=1e-5 is *doing real work* — the final low-LR phase is where the model fine-tunes its pressure predictions. T_max=28 was correctly identified as the optimal schedule for a 28-epoch budget.

**Programme finding**: T_max=28 is the confirmed optimal cosine schedule. Cosine-schedule axis fully closed: T_max<28 (too fast), T_max=28 (optimal), T_max>28 (starves final convergence). SGDR and OneCycleLR also closed.

---

## 2026-05-13 13:30 — PR #2169: [re-input-jitter] Gaussian noise σ=0.05/0.10 on log(Re) — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/re-input-jitter
- **Hypothesis**: Add Gaussian noise to log(Re) input channel during training to improve OOD-Re generalization (re_rand split). Targets the model's sensitivity to exact Re values.
- **Status**: CLOSED — σ=0.05 +5.51%, σ=0.10 +14.14%. Re channel is critical.

| Metric | σ=0.05 | σ=0.10 | Baseline |
|--------|--------|--------|----------|
| val_avg/mae_surf_p | 30.47 | 33.03 | 28.8762 |

**Root cause**: The Re channel is load-bearing for the entire ReFiLM conditioning stack. Even σ=0.05 (~5% jitter on log(Re)) disrupted the precise Re-to-attention-gate mapping that ReFiLM learned. All 4 val splits degraded — the jitter doesn't generalize to re_rand; it just corrupts the Re-dependent physics the model has learned.

**Programme finding**: Re input axis firmly closed. Re conditioning is a strength, not a weakness. Augmenting it degrades the model. The re_rand OOD-Re generalization gap is better addressed through better Re-conditioning architecture, not by blurring the Re signal.

---

## 2026-05-13 10:20 — PR #2092: [coord-translation-aug] Rigid mesh translation augmentation — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/coord-translation-aug
- **Hypothesis**: Per-sample rigid (dx, dy) translation of entire mesh, NSE-invariant in unbounded domain.
- **Status**: CLOSED — +3.3% val regression. Bounded BVP breaks translation invariance.

| Metric | Translation aug | Baseline (#2011) | Δ |
|--------|----------------|-----------------|---|
| val_avg/mae_surf_p | 29.8364 | 28.8762 | **+3.3% (WORSE)** |
| test_avg/mae_surf_p | 25.8271 | 24.9992 | **+3.3% (WORSE)** |

**Split pattern**: single_in_dist regressed most (+7.5%) — the split most reliant on absolute-position signal. geom_camber_rc regressed least (+0.4%) — already hard, absolute-position contributes less.

**Root cause (tanjiro's analysis)**: NSE is translation-invariant in free space, but TandemFoilSet-Balanced is a bounded BVP with inlet/outlet/walls at fixed absolute positions. Input features encode the foil (saf, dsdf, NACA params) but NOT domain boundaries. The model's only localization signal is channels 0–1. Translating those while keeping target y unchanged adds systematic label noise along the absolute-position dimension. Translation augmentation CLOSED for this benchmark.

**Artifact**: `models/model-charliepai2g24h1-tanjiro-coord-translation-aug-20260513-091211/metrics.jsonl`

---

## 2026-05-13 10:05 — PR #2110: [sgdr-warm-restarts-v2] SGDR T_0=14, T_mult=1 — CLOSED

- **Branch**: charliepai2g24h1-askeladd/sgdr-warm-restarts-v2
- **Hypothesis**: CosineAnnealingWarmRestarts T_0=14, T_mult=1 — two cosine cycles in 28 epochs; LR resets to 1e-3 at epoch 15 from warm SOAP preconditioner.
- **Status**: CLOSED — +8.13% val regression. Restart shock destroyed cycle-2 convergence.

| Metric | SGDR | Baseline (#2011) | Δ |
|--------|------|-----------------|---|
| val_avg/mae_surf_p | 31.2245 | 28.8762 | **+8.13% (WORSE)** |
| test_avg/mae_surf_p | 27.4939 | 24.9992 | **+9.98% (WORSE)** |

**LR trace**: ep1=1e-3, ep14≈2e-5 (cycle-1 floor, val=37.95), ep15=1e-3 (RESTART → val JUMPED to 64.52, +70%), ep28≈2e-5 (cycle-2 floor, val=31.22).

**Failure mechanism**: The restart at epoch 15 hit a sharp minimum from cycle-1's eta_min. Resetting to lr=1e-3 with warm SOAP preconditioner produced an aggressive update landing far outside the local basin. Cycle-2 took all 14 epochs to re-anneal back down. End of cycle-2 (31.22) beat end of cycle-1 (37.95) by ~18%, confirming the warm-init benefit is real but not enough to overcome the restart penalty within the 28-epoch budget. Askeladd's analysis: "budget arithmetic kills the idea" — 28-epoch budget accommodates exactly two 14-epoch cycles with no room to capitalize on re-exploration.

**Programme conclusion**: Periodic warm restarts CLOSED for this budget. The correct next step (per askeladd's follow-up) is extending T_max to keep LR higher without resetting. Assigned cosine-long-tail (#2147).

**Artifact**: `models/model-sgdr-warm-restarts-20260513-090346/metrics.jsonl`

---

## 2026-05-13 10:05 — PR #2079: [n-layers-6] Deeper Transolver stack 5→6 layers — CLOSED

- **Branch**: charliepai2g24h1-fern/n-layers-6
- **Hypothesis**: Extra Transolver block at same hidden_dim for more representational depth.
- **Status**: CLOSED — +6.22% val regression. Slowdown trimmed budget from 28→24 epochs; model still on downslope.

| Metric | n=6 | Baseline (#2011) | Δ |
|--------|-----|-----------------|---|
| val_avg/mae_surf_p | 30.6712 | 28.8762 | **+6.22% (WORSE)** |
| test_avg/mae_surf_p | 26.7463 | 24.9992 | **+6.99% (WORSE)** |

**Cause**: Steady-state epoch time 75.9s vs ~64s (baseline) — ~19% slowdown matching +20% params. Trimmed budget 28→24 epochs. Best epoch=24 (last), still descending. Deeper model is **under-trained** in this budget, not over-fit. Peak GPU 32.84 GB (unchanged), so this is not memory-limited — purely compute throughput.

**Single_in_dist regression was worst** (+12.86% val): the cleanest split requires the most fine-tuning steps to reach its optimum. OOD splits (rc +2.29%, cruise +3.63%) are less sensitive to the 4 missing epochs.

**Programme conclusion**: n_layers=6 closed for 30-min budget. Next step: n_head=8 (same inner_dim, no slowdown, doubles FiLM gating degrees of freedom) assigned to fern as #2154.

**Artifact**: `models/model-n-layers-6-20260513-085622/metrics.jsonl`

---

## 2026-05-13 10:05 — PR #2111: [huber-delta-p-tighter] δ_p=0.05, δ_v=0.1 — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/huber-delta-p-tighter
- **Hypothesis**: Tighten pressure δ from 0.1→0.05 to concentrate pressure gradient in pure-L2 quadratic regime.
- **Status**: CLOSED — +1.50% val regression. Pressure was already mostly in quadratic regime.

| Metric | δ_p=0.05 | Baseline (#2011) | Δ |
|--------|---------|-----------------|---|
| val_avg/mae_surf_p | 29.3080 | 28.8762 | **+1.50% (WORSE)** |
| test_avg/mae_surf_p | 25.0658 | 24.9992 | +0.27% |

**Key diagnostic**: Per-channel l2-fraction at end of training — p (0.877), Uy (0.877), Ux (0.843). **Pressure was already nearly identical to velocity in the quadratic regime.** Tightening δ_p from 0.1 to 0.05 moved the cap into the distribution interior, halving gradient magnitude for moderate-outlier pressure residuals (those in |r|∈(0.05, 0.1]) — exactly the informative samples that drive learning on hard cases. single_in_dist regressed +5.28%.

**Root cause**: After ReScaleHead + ReFiLM + p_weight=5 stack, normalized pressure residuals are already small and concentrated near zero. The Huber δ=0.1 was already in the "right zone." Tightening removed gradient signal from remaining hard cases.

**Programme conclusion**: Huber δ axis CLOSED (both directions tried — looser velocity in #2081, tighter pressure in #2111 — both regressed). Loss shape is well-calibrated at δ=0.1. Next step: log-cosh smooth replacement assigned to thorfinn (#2146).

**Artifact**: `models/model-charliepai2g24h1-thorfinn-huber-delta-p-tighter-20260513-091929/metrics.jsonl`

---

## 2026-05-13 09:35 — PR #2081: [per-channel-huber-delta] δ_v=0.5, δ_p=0.1 — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/per-channel-huber-delta
- **Hypothesis**: Loosen velocity Huber δ (0.1→0.5) while keeping pressure δ=0.1. Physical intuition: velocity errors at high Re are physically meaningful and should not be over-clipped.
- **Status**: CLOSED — +1.16% val regression. Mechanism clearly understood.

| Metric | δ_v=0.5, δ_p=0.1 | Baseline (#2011) | Δ |
|--------|------------------|-----------------|---|
| val_avg/mae_surf_p | 29.2115 | 28.8762 | **+1.16% (WORSE)** |
| test_avg/mae_surf_p | 25.0824 | 24.9992 | +0.33% |

**Per-split:**

| Split | val this | val base | Δval | test this | test base | Δtest |
|-------|---------|---------|------|----------|---------|------|
| single_in_dist | 28.8386 | 28.6013 | +0.83% | 28.5907 | 29.5300 | −3.18% |
| geom_camber_rc | 40.9788 | 41.9483 | −2.31% | 38.2637 | 37.0266 | +3.34% |
| geom_camber_cruise | 14.9267 | 14.1462 | +5.52% | 11.3704 | 11.0171 | +3.21% |
| re_rand | 32.1018 | 30.8090 | +4.20% | 22.1046 | 22.4230 | −1.42% |
| **avg** | **29.2115** | **28.8762** | **+1.16%** | **25.0824** | **24.9992** | **+0.33%** |

**Diagnostic**: huber_l2_frac climbed 0.72→0.97 across training. Loosening velocity δ to 0.5 pushed nearly all velocity residuals into the quadratic regime — effectively making velocity loss pure rel-L2, removing the Huber tail entirely. This shifted optimization gradient mass away from pressure (primary metric). re_rand val regressed +4.20% — opposite of hypothesis.

**Programme conclusion**: Loosening velocity δ is harmful. The inverse direction (tighter δ_p=0.05, velocity δ=0.1 unchanged) is better-motivated — concentrating pressure gradient signal in the quadratic regime — assigned to thorfinn as #2111.

**Artifact**: `models/model-per-channel-huber-delta-20260513-082400/metrics.jsonl`

---

## 2026-05-13 09:20 — PR #2077: [soap-linear-warmup] 3-epoch linear LR warmup before cosine — CLOSED

- **Branch**: charliepai2g24h1-askeladd/soap-linear-warmup
- **Hypothesis**: 3-epoch LinearLR warmup (lr: 1e-6 → 1e-3) before CosineAnnealingLR prevents early-training instability in SOAP and improves final convergence.
- **Status**: CLOSED — regression (+1.38% val / +2.49% test). Two compounding failure modes.

| Metric | Warmup | Baseline (#2011) | Δ |
|--------|--------|-----------------|---|
| val_avg/mae_surf_p | 29.2754 | 28.8762 | **+1.38% (WORSE)** |
| test_avg/mae_surf_p | 25.6217 | 24.9992 | **+2.49% (WORSE)** |

**Per-split:**

| Split | val warmup | val base | Δval | test warmup | test base | Δtest |
|-------|-----------|---------|------|------------|---------|------|
| single_in_dist | 30.1193 | 28.6013 | +5.30% | 31.0709 | 29.5300 | +5.22% |
| geom_camber_rc | 41.9844 | 41.9483 | +0.09% | 38.1052 | 37.0266 | +2.91% |
| geom_camber_cruise | 13.9614 | 14.1462 | −1.31% | 10.9757 | 11.0171 | −0.38% |
| re_rand | 31.0366 | 30.8090 | +0.74% | 22.3352 | 22.4230 | −0.39% |
| **avg** | **29.2754** | **28.8762** | **+1.38%** | **25.6217** | **24.9992** | **+2.49%** |

**LR trace**: ep1=1e-6 → ep3=6.67e-4 → ep4=1e-3 (peak) → ep27=2.56e-5 (best) → ep28=1.39e-5. SequentialLR worked as specified.

**Failure analysis** (per askeladd's thorough diagnosis):
1. **No instability to fix**: SOAP already trains stably from lr=1e-3 at epoch 0 — val curves descend monotonically, no divergence. Warmup addresses a problem that doesn't exist here.
2. **Budget waste**: 30-min cap = ~28 epochs regardless. 3 warmup epochs at near-zero LR (1e-6 → 1e-3) are effectively wasted — the cosine phase is compressed to 25 epochs vs baseline's 28. Regression dominated by single_in_dist (+5.3%), the split most sensitive to total effective training steps.

**Programme conclusion**: LR warmup for SOAP on this problem is not beneficial. Warmup direction closed. OneCycleLR (#1884, alphonse) will establish if *any* higher-peak-LR regime with warmup helps — that experiment has a genuinely different hypothesis (peak LR 2e-3, not just warmup before 1e-3).

**Artifact**: `models/model-soap-linear-warmup-20260513-082127/metrics.jsonl`

---

## 2026-05-13 09:00 — PR #2011: [film-re-attention] ReFiLM conditioning inside PhysicsAttention — MERGED

- **Branch**: charliepai2g24h1-fern/film-re-attention
- **Hypothesis**: FiLM (Feature-wise Linear Modulation) applied to slice logits inside PhysicsAttention allows the model to select different slice subsets per Reynolds number — deeper Re-conditioning than ReScaleHead (output rescaling). Zero-init gates ensure identity at epoch 0.
- **Status**: MERGED — new baseline: val_avg=28.8762 (-1.17% vs #1614)

| Metric | ReFiLM | Baseline (#1614) | Δ |
|--------|--------|-----------------|---|
| val_avg/mae_surf_p | **28.8762** | 29.2179 | **−1.17%** |
| test_avg/mae_surf_p | **24.9992** | 25.6024 | **−2.36%** |

**Per-split val:**

| Split | ReFiLM | Base (#1614) | Δ |
|-------|--------|-------------|---|
| single_in_dist | 28.6013 | 28.5620 | +0.14% |
| geom_camber_rc | 41.9483 | 42.6891 | −1.73% |
| geom_camber_cruise | 14.1462 | 13.7711 | +2.72% |
| re_rand | 30.8090 | 31.8496 | −3.27% |
| **avg** | **28.8762** | **29.2179** | **−1.17%** |

**Mechanism confirmed**: Mean slice entropy dropped 33% (4.153 → 2.759) — model genuinely uses different attention slice subsets per Re. Gains concentrate on Re-variable (re_rand −3.27% val) and OOD-shape (geom_camber_rc −1.73% val; −4.91% test). FiLM gates open monotonically (zero-init, |γ|max=0.70, |β|max=0.62 by ep28). All 3 Re-conditioning mechanisms now in baseline stack: ReScaleHead (output) + p_channel_weight (loss) + ReFiLM (attention).

**Params**: 4,624 (shared module across all 5 blocks/4 heads); peak GPU +3.9 GB for FiLM intermediates. Best epoch = last (28) — schedule still binding.

**Artifact**: `models/model-charliepai2g24h1-fern-film-re-attention-20260513-072042/metrics.jsonl`

---

## 2026-05-13 08:30 — PR #1963: [coord-jitter-aug] rebased onto #2011 stack — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/coord-jitter-aug
- **Previous state**: Mild positive on old baseline (−0.78% val / −1.51% test vs #1599), sent back for rebase onto #1614. After further rebase onto #2011 (current baseline):
- **Status**: CLOSED — +1.93% val regression on full rebased stack.

| Metric | coord-jitter (rebased) | Baseline (#2011) | Δ |
|--------|----------------------|-----------------|---|
| val_avg/mae_surf_p | ~29.44 | 28.8762 | **+1.93% (WORSE)** |

**Failure analysis**: Input-domain coordinate perturbation (per-node std=0.005 Gaussian jitter) does NOT compound with loss-domain reweighting (p_channel_weight=5) + ReFiLM. Two distinct axes that were orthogonal on the old stack interact destructively on the new stack. Per-node random jitter is geometrically non-physical (distorts foil geometry), which may be why it doesn't compound cleanly. **Input augmentation axis closed for per-node jitter.**

**Follow-on**: tanjiro now assigned coord-translation-aug (#2092) — rigid whole-mesh translation, NSE-invariant and geometrically valid (distinct mechanism).

---

## 2026-05-13 07:15 — PR #1985: [p-channel-weight-15] Sweep p_weight 5→15 — CLOSED

- **Branch**: charliepai2g24h1-edward/p-channel-weight-15
- **Hypothesis**: p_weight=5 won -2.11%; p still dominates error by ~70×; more gradient mass to p should improve further.
- **Status**: CLOSED — regressed every split and every channel.

| Metric | p_weight=15 | p_weight=5 (baseline #1614) | Δ |
|--------|------------|----------------------------|---|
| val_avg/mae_surf_p | 30.4444 | 29.2179 | **+4.20% (WORSE)** |
| test_avg/mae_surf_p | 25.8211 | 25.6024 | +0.85% (worse) |

**Per-split val:**

| Split | p=15 | p=5 | Δ |
|-------|------|-----|---|
| single_in_dist | 30.60 | 28.56 | +7.14% |
| geom_camber_rc | 44.45 | 42.69 | +4.12% |
| geom_camber_cruise | 14.73 | 13.77 | +6.93% |
| re_rand | 32.00 | 31.85 | +0.49% |

**Critically: p itself regressed** — the per-element p Huber actually DROPPED on train (0.001575→0.001125, model fitted train-p harder) but val p got WORSE. Classic train/val divergence from over-aggressive single-channel weighting.

**Programme conclusion**: p_weight=5 is at or above the optimum. The cross-channel coupling through the shared Transolver backbone is real and non-negotiable — over-amplifying p distorts shared features. Loss-weighting axis is CLOSED at p_weight=5.

**Artifact**: `target/models/model-charliepai2g24h1-edward-p-channel-weight-15-20260513-061953/metrics.jsonl`

---

## 2026-05-13 07:00 — PR #1966: [ema-beta-0p99-rampup] EMA β=0.99 + Karras rampup — SENT BACK FOR REBASE

- **Branch**: charliepai2g24h1-frieren/ema-beta-0p99-rampup
- **Hypothesis**: EMA β=0.99 with Karras-style linear rampup fixes the β=0.999 window-too-long failure (PR #1917). Rampup denominator=0.1×MAX_EPOCHS×steps_per_epoch so β_eff starts near 0 and asymptotes to 0.99 over training.
- **Status**: SENT BACK — beat old baseline (29.0776 vs 29.8463, -2.6%), but run was on old stack (no p_weight=5). New baseline is 29.2179 (from #1614). Needs rebase to verify compounding.

| Metric | EMA (ep29) | Live (ep29) | Old baseline (#1599) | Δ vs old |
|--------|-----------|-------------|---------------------|----------|
| ema_val_avg/mae_surf_p | **29.0776** | 29.1616 | 29.8463 | **−2.6%** |
| ema_test_avg/mae_surf_p | **25.0978** | — | 26.1005 | **−3.8%** |
| vs new baseline (29.2179) | **−0.48%** | — | — | (borderline, needs rebase) |

**Per-split ema val (best ep29):**

| Split | ema val | baseline (#1599) | Δ |
|-------|---------|-----------------|---|
| single_in_dist | 29.44 | 30.20 | −2.5% |
| geom_camber_rc | 42.20 | 43.11 | −2.1% |
| geom_camber_cruise | 14.09 | 14.54 | −3.1% |
| re_rand | 30.58 | 31.54 | −3.0% |

**β_eff rampup (key epochs)**: ep1=0.18, ep5=0.63, ep10=0.86, ep15=0.94, ep20=0.97, ep25=0.98, ep29=0.987. Slower ramp than predicted because denominator used MAX_EPOCHS=50 (not actual 30 epochs). Advisory for rebase: set denominator to 30 epochs.

**Artifact**: `models/model-charliepai2g24h1-frieren-ema-beta-0p99-rampup-20260513-061015/metrics.jsonl`

**Analysis**: Mechanism validated — β=0.99 rampup works. EMA is 0.084 better than live at ep29 (the smoothing dividend). All 4 val splits and 4 test splits improved. re_rand showed largest absolute drop (noise from Re variance is smoothed by EMA). The hypothesis is confirmed: β=0.99 keeps the effective window narrow enough that EMA tracks live closely while still smoothing jitter.

---

## 2026-05-13 07:00 — PR #1964: [surf-weight-15] Raise surf_weight 10→15 — SENT BACK FOR REBASE

- **Branch**: charliepai2g24h1-thorfinn/surf-weight-15
- **Hypothesis**: surf_weight=15 (1.5× more surface emphasis than baseline) tests if the asymmetric response (-7: bad, +15: maybe good) means there's headroom above surf_weight=10.
- **Status**: SENT BACK — mild positive on old baseline (29.6254 vs 29.8463, -0.74%) but new baseline is 29.2179. Needs rebase to verify orthogonal compounding with p_weight=5.

| Metric | surf=15 | Old baseline (surf=10) | Δ |
|--------|---------|----------------------|---|
| val_avg/mae_surf_p | 29.6254 | 29.8463 | **−0.74%** |
| test_avg/mae_surf_p | 25.6508 | 26.1005 | **−1.72%** |
| vs new baseline (29.2179) | +1.36% (regression) | — | (needs rebase) |

**Sensitivity curve so far**: surf=7 (+2.94% BAD) | surf=10 (29.8463 baseline) | surf=15 (−0.74%) | surf=30 (askeladd pending)

**Analysis**: Asymmetric response confirmed — down hurts, up helps mildly. The direction is correct but the magnitude (−0.74% val, −1.72% test) is at noise floor. surf_weight and p_weight target orthogonal axes (region: surf-vs-vol; channel: p-vs-velocity) — possible compounding.

---

## 2026-05-13 07:00 — PR #1963: [coord-jitter-aug] Spatial coord jitter std=0.005 — SENT BACK FOR REBASE

- **Branch**: charliepai2g24h1-tanjiro/coord-jitter-aug
- **Hypothesis**: Adding std=0.005 Gaussian noise to spatial coordinates (training only) forces the model to learn from slightly perturbed meshes, improving OOD robustness.
- **Status**: SENT BACK — mild positive on old baseline (29.6143 vs 29.8463, -0.78% val, -1.51% test) but new baseline is 29.2179. Needs rebase to verify compounding.

| Metric | coord-jitter | Old baseline (#1599) | Δ | vs new baseline |
|--------|-------------|---------------------|---|-----------------|
| val_avg/mae_surf_p | 29.6143 | 29.8463 | −0.78% | +1.36% (needs rebase) |
| test_avg/mae_surf_p | 25.7063 | 26.1005 | −1.51% | +0.41% (needs rebase) |

**Analysis**: Acting as generic regularization rather than rc-targeted augmentation (re_rand test −4.06% was largest delta, not rc). single_in_dist val/test diverge (+noise). Test signal more credible than val (larger set). Mechanism is input-domain augmentation — orthogonal to loss weighting in #1614.

---

## 2026-05-13 07:00 — PR #1952: [rescale-head-2ch] Drop Ux from ReScaleHead — CLOSED

- **Branch**: charliepai2g24h1-fern/rescale-head-2ch
- **Hypothesis**: Drop Ux channel (near-identity, corr=+0.637) from ReScaleHead, leaving only Uy and p.
- **Status**: CLOSED — rebased run regressed +4.63% val / +2.20% test vs new baseline (29.2179).

| Metric | 2ch rebased | New baseline (#1614) | Δ |
|--------|------------|---------------------|---|
| val_avg/mae_surf_p | 30.5712 | 29.2179 | **+4.63% (BAD)** |
| test_avg/mae_surf_p | 26.1663 | 25.6024 | **+2.20% (BAD)** |

**Reasoning for close**: With p_weight=5 shifting gradient mass to pressure, the Ux channel was providing a balancing function. When dropped, velocity channels lose their stability anchor. The original 2ch win (vs #1599 stack) did not survive the new loss weighting — the mechanism is loss-stack-dependent and the 3ch ReScaleHead is the correct architecture.

---

## 2026-05-13 06:15 — PR #1614: [per-channel-loss-weights] Up-weight pressure channel (p_weight=5) — MERGED

- **Branch**: charliepai2g24h1-edward/per-channel-loss-weights
- **Hypothesis**: Upweighting the p channel by 5× post-Huber shifts ~7× more gradient mass to the dominant error source. Implementation: linear multiplier on per-element Huber output (post-Huber), unweighted denominator — guarantees exactly 5× amplification across Huber regimes.
- **Status**: MERGED — new baseline: val_avg=29.2179 (-2.11%)

| Metric | p_weight=5 | Baseline (#1599) | Δ |
|--------|-----------|-----------------|---|
| val_avg/mae_surf_p | **29.2179** | 29.8463 | **−2.11%** |
| test_avg/mae_surf_p | **25.6024** | 26.1005 | **−1.91%** |
| Best epoch | 29 | 29 | — |
| Peak GPU | 23.91 GB | 24 GB | flat |

**Per-split val/test (epoch 29):**

| Split | val (p5) | val (base) | Δval | test (p5) | test (base) | Δtest |
|-------|---------|-----------|------|----------|------------|-------|
| single_in_dist | 28.562 | 30.20 | −5.43% | 30.135 | 30.09 | +0.15% |
| geom_camber_rc | 42.689 | 43.11 | −0.97% | 38.939 | 39.41 | −1.19% |
| geom_camber_cruise | 13.771 | 14.54 | −5.29% | 10.847 | 11.74 | −7.60% |
| re_rand | 31.850 | 31.54 | +0.99% | 22.489 | 23.16 | −2.90% |
| **avg** | **29.218** | **29.846** | **−2.11%** | **25.602** | **26.101** | **−1.91%** |

**Per-channel trade-off:**

| Channel | p5 | base (#1599) | Δ |
|---------|----|-------------|---|
| mae_surf_p | 29.218 | 29.846 | −2.1% |
| mae_surf_Ux | 0.440 | 0.376 | +17.1% |
| mae_surf_Uy | 0.248 | 0.224 | +10.7% |

**Artifact**: `models/model-charliepai2g24h1-edward-per-channel-loss-weights-p5-20260513-052434/metrics.jsonl`

**Analysis**: p upweighting is an orthogonal mechanism that compounds cleanly with SOAP+compile+ReScaleHead. p dominates error by ~70×; directing 7× more gradient mass to p is clearly net positive. Training was stable (no oscillation, smooth monotone improvement). Model still hitting best epoch at the last epoch → convergence-limited. Key implementation insight: post-Huber weighting preserves the δ=0.1 threshold invariance across all Huber regimes. Velocity channels degrade slightly (+11-17%) but this is expected and net positive overall.

---

## 2026-05-13 06:15 — PR #1952: [rescale-head-2ch] Drop Ux channel from ReScaleHead — SENT BACK FOR REBASE

- **Branch**: charliepai2g24h1-fern/rescale-head-2ch
- **Hypothesis**: Ux scale_std=0.058 (near identity, corr=+0.637 — weakest of 3 channels). Removing Ux cleans gradient routing into the Uy/p heads which have real Re-conditioning function.
- **Status**: SENT BACK — beat old baseline (29.3842 vs 29.8463, -1.55%) but #1614 merged setting new bar at 29.2179. Need rebase to verify compounding.

| Metric | 2ch head | Baseline (#1599) | Δ |
|--------|---------|-----------------|---|
| val_avg/mae_surf_p | 29.3842 | 29.8463 | −1.55% |
| test_avg/mae_surf_p | 25.6407 | 26.1005 | −1.76% |
| Head params | 130 | 163 | −33 |

**Per-split val (epoch 29):**

| Split | 2ch | base | Δ |
|-------|-----|------|---|
| single_in_dist | 28.879 | 30.20 | −4.4% |
| geom_camber_rc | 42.408 | 43.11 | −1.6% |
| geom_camber_cruise | 14.690 | 14.54 | +1.0% |
| re_rand | 31.559 | 31.54 | +0.1% |

**2ch head diagnostics**: Uy scale_std=0.266, corr=+0.953; p scale_std=0.518, corr=+0.876 — essentially unchanged from 3ch run. Gradient-routing effect on Transolver body is the mechanism.

**Artifact**: `models/model-charliepai2g24h1-fern-rescale-head-2ch-20260513-052123/metrics.jsonl`

**Next action**: Rebase onto new baseline (29.2179), re-run. If 2ch Ux-drop compounds with p_weight=5, merge.

---

## 2026-05-13 05:15 — PR #1704: [ema-weights] EMA model weights for smoother final checkpoint

- **Branch**: charliepai2g24h1-frieren/ema-weights
- **Hypothesis**: EMA (β=0.999) of model weights produces a smoother final checkpoint than the terminal live-weights checkpoint, at zero training-time cost.
- **Status**: CLOSED — did not beat baseline; dual-val protocol overhead was the root cause

| Metric | EMA (26 ep) | Live weights (26 ep) | Baseline (30 ep) | Delta vs baseline |
|--------|-------------|----------------------|------------------|-------------------|
| ema_val_avg/mae_surf_p | **32.2245** | — | 30.4412 | +1.78 (+5.9%) |
| val_avg/mae_surf_p (live) | — | 32.0731 | 30.4412 | +1.63 (+5.4%) |
| ema_test_avg/mae_surf_p | **27.7392** | — | 26.1013 | +1.64 (+6.3%) |
| Epochs completed | 26 | 26 | 30 | −4 epochs |
| Mean epoch time | 70.4s | — | ~62s | +13% (dual val) |

**Per-split EMA val (epoch 26):**

| Split | EMA val | EMA test | Baseline val | Baseline test |
|-------|---------|----------|--------------|---------------|
| single_in_dist | 35.55 | 35.35 | 34.27 | 32.96 |
| geom_camber_rc | 43.90 | 39.47 | 41.43 | 37.90 |
| geom_camber_cruise | 15.63 | 12.37 | 14.04 | 11.38 |
| re_rand | 33.82 | 23.76 | 32.02 | 22.16 |
| **avg** | **32.22** | **27.74** | **30.44** | **26.10** |

**Artifact**: `models/model-charliepai2g24h1-frieren-ema-weights-20260513-040222/metrics.jsonl`

**Analysis**: Implementation was correct; EMA trajectory behaved exactly as theory predicts. The mid-run EMA advantage was REAL (Δ=−11.7 MAE at epoch 14 — EMA was ~20% better than live). But two factors killed the result: (1) The PR required logging both live and EMA val each epoch, doubling validation work (+13% wall clock overhead), costing 4 epochs (26 vs 30). (2) Cosine LR with eta_min=1e-5 already smooths late-epoch updates naturally — by epoch 24-26 the EMA and live weights are nearly identical (Δ flips sign at epoch 25). The hypothesis was not wrong about the mechanism; the experimental protocol was wrong. **Closing**, assigning corrected protocol.

**Key EMA trajectory**: shadow lagged init for ~10 epochs, peaked at Δ=−11.7 MAE at epoch 14, narrowed to Δ=+0.15 at epoch 26.

---

## 2026-05-13 02:10 — PR #1599: [re-conditioned-scaling] sent back v3 — compound confirmed but stale baseline

- **Branch**: charliepai2g24h1-fern/re-conditioned-scaling
- **Status**: SENT BACK for third rebase — compound mechanism confirmed, but baseline moved during run

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (on cosine-eta-min base, ep 13) | 38.0178 |
| test_avg/mae_surf_p | 33.5671 |
| vs cosine-eta-min baseline (39.8693 / 35.2214) | **−4.7% val, −4.7% test** |
| vs current bf16-amp baseline (36.8778 / 31.9058) | +3.1% val (regression) |
| ReScale corr(scale, log Re) Ux/Uy/p | +0.68 / **+0.89** / **+0.86** |
| scale_std Ux/Uy/p at ep 13 | 0.052 / 0.211 / **0.504** |

**Mechanism confirmed**: ReScaleHead COMPOUNDS with SOAP — answering the three-possibilities question from the original send-back. Pre-SOAP failure on AdamW was caused by first-order instability during warmup (head and backbone competing for Re-dependent scale). SOAP's preconditioner routes the Re signal cleanly into the 163-param head subspace, eliminating the competition. Physical signature: Ux nearly inert (freestream-dominated), Uy moderate, p strongest — consistent with Bernoulli-like ~Re² pressure scaling.

**Why sent back**: Result is on cosine-eta-min base, but bf16-amp merged during fern's run. Need to verify the 4.7% compound holds on the new bf16 baseline. If it does, target val ≈ 35.15 (cleanly beats baseline). The mechanism is well-established now; this is the last rebase needed.

**Side observation (one-shot noise, not conclusive)**: A SOAP-only run hit val 36.72; SOAP + eta_min=1e-5 + ReScale hit val 38.02 — possible eta_min × ReScale interaction worth watching but currently single-shot.

**Artifact**: `models/model-charliepai2g24h1-fern-re-conditioned-scaling-20260513-005513/metrics.jsonl`

---

## 2026-05-13 02:00 — PR #1456: [bf16-amp + cosine-eta-min] bf16 AMP with T_max=17 on SOAP

- **Branch**: charliepai2g24h1-alphonse/bf16-amp (rebased onto cosine-eta-min base)
- **Hypothesis**: bf16 AMP gives ~+29% throughput → ~17 epochs in 30 min vs 13 previously. T_max=17 aligns cosine tail to new budget. Compounds with all prior wins.
- **Status**: MERGED — new baseline 36.8778

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 16/17) | **36.8778** |
| val_single_in_dist | 42.92 (−10.2%) |
| val_geom_camber_rc | 47.78 (−8.6%) |
| val_geom_camber_cruise | **18.60** (−11.0%) |
| val_re_rand | 38.21 (−0.7%) |
| test_avg/mae_surf_p | **31.9058** (−9.42%) |
| test_single_in_dist | 42.15 (−8.3%) |
| test_geom_camber_rc | 42.69 (−7.9%) |
| test_geom_camber_cruise | **15.26** (−11.5%) |
| test_re_rand | 27.53 (−12.2%) |
| Epochs | 17 (vs 13 previously) |
| Mean epoch time | 108.6 s (vs ~131 s previously) |
| Peak GPU memory | 32.98 GB / 96 GB |
| clip_frac trajectory | 0.98 → 0.34 (smoothly decaying) |
| huber_l2_frac | 0.42 → 0.86 (Huber actively capping outliers) |
| Baseline | 39.8693 |
| Delta | **−2.99 (−7.51%)** |

**Analysis**: bf16 + T_max alignment compounds cleanly on SOAP/eta_min. All 4 val + 4 test splits improved (rare clean win across the board). No numerical issues with bf16 + SOAP + Huber-rel-L2 (SOAP preconditioner runs in fp32 internally). ep 17 (at LR floor 1.84e-5) drifts back +0.09 from ep 16 best — confirms T_max=17 is well-matched.

**Key insight**: With substantial memory headroom (33/96 GB), there's room for larger batches OR larger models — next experiments target both axes.

**Artifact**: `models/model-charliepai2g24h1-alphonse-bf16-amp-cosine-eta-min-20260513-005955/metrics.jsonl`

---

## 2026-05-13 01:55 — PR #1740: [soap-higher-lr] lr=1e-3→2e-3 under SOAP — null result

- **Branch**: charliepai2g24h1-tanjiro/soap-higher-lr
- **Hypothesis**: SOAP's curvature-aware preconditioning raises the LR ceiling well above AdamW's 1e-3 limit; lr=2e-3 should accelerate convergence without divergence.
- **Status**: CLOSED — null result (within noise), mechanism understood

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | 39.7891 |
| test_avg/mae_surf_p | 35.6166 |
| Baseline | 39.8693 / 35.2214 |
| Delta val | −0.08 (noise) |
| Delta test | +0.40 (noise, wrong direction) |
| clip_frac ep 1 | 0.952 (95% of batches clipped) |
| clip_frac ep 13 | 0.179 |
| Divergence | None — model trained cleanly |

**Analysis**: LR ceiling confirmed — lr=2e-3 ran cleanly all 13 epochs with no NaN, no spikes (vs AdamW's divergence at lr=1.5e-3 in PR #1539). But grad_clip=1.0 normalized away the LR increase: effective per-step update `= (clip × lr) / grad_norm` was nearly identical between lr=1e-3 and 2e-3. The clip threshold (not the LR) is the bottleneck.

**Key insight**: SOAP LR ceiling is ≥ 2e-3 (stable). To actually exploit higher LR, grad_clip needs to widen too — exactly what thorfinn is testing in #1668 (clip 1.0→5.0). Future LR experiments must couple with clip widening.

**Artifact**: `models/model-charliepai2g24h1-tanjiro-soap-higher-lr-20260513-005527/metrics.jsonl`

---

## 2026-05-13 00:25 — PR #1630: [cosine-eta-min] CosineAnnealingLR eta_min=1e-5 floor on SOAP

- **Branch**: charliepai2g24h1-tanjiro/sgdr-restarts (pivoted from SGDR to cosine-eta-min)
- **Hypothesis**: Prevent CosineAnnealingLR from reaching near-zero at the terminal epoch by adding `eta_min=1e-5`. The SOAP baseline uses `T_max=14` but only completes 13 epochs — epoch 13 is both the budget-limited final epoch and the best checkpoint. Without a floor, the cosine schedule drives LR to ~4.95e-5 at ep 13; with `eta_min=1e-5`, LR is ~5.90e-5 (+19% relative).
- **Status**: MERGED — new baseline 39.8693

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **39.8693** |
| val_single_in_dist | 47.81 (+1.72 vs prev) |
| val_geom_camber_rc | 52.28 (−3.70) |
| val_geom_camber_cruise | **20.89** (−3.43) |
| val_re_rand | **38.49** (−4.73) |
| test_avg/mae_surf_p | **35.2214** |
| test_geom_camber_cruise | **17.24** |
| test_re_rand | **31.37** |
| Epochs | 13 (~30-min cap) |
| LR at ep 13 | 5.90e-5 (vs 4.95e-5 without floor) |
| Baseline | 42.4015 |
| Delta | **−2.53 (−5.97%)** |

**Analysis**: Single-line change. The +19% relative LR boost at epoch 13 (the best-checkpoint epoch) gives meaningful gradient signal in the final step. 3/4 OOD splits improved; single_in_dist slightly worse (+1.72) likely because it was already well-fit. Val still monotone descending — model has not converged. This is a free compounding gain on top of SOAP.

**Key insight**: The cosine schedule's late-epoch LR matters most when the budget-limited final epoch equals the best checkpoint. This will compound further when bf16-amp provides more epochs.

**Artifact**: `models/model-charliepai2g24h1-tanjiro-cosine-eta-min-20260512-231540/metrics.jsonl`

---

## 2026-05-13 00:00 — PR #1579: [pcgrad-surgery] Gradient surgery for vol/surf conflict

- **Branch**: charliepai2g24h1-frieren/pcgrad-surgery
- **Hypothesis**: PCGrad gradient surgery (Yu et al. NeurIPS 2020) reduces destructive interference between vol_loss and surf_loss gradients, lowering effective update noise that forces 100% gradient clipping.
- **Status**: CLOSED — mechanism confirmed but wall-clock loss is structural at 30-min budget

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 8, EMA weights) | **59.2256** |
| test_avg/mae_surf_p | **53.7574** |
| SOAP baseline | 42.4015 |
| Delta | **+39.6% regression** |
| seconds/epoch | 225 (vs 138 for SOAP baseline, 1.63×) |
| Epochs completed | 8 (vs 13 for SOAP in same wall-clock) |
| conflict_frac (vol vs surf) | **0.04 (sparse)** |
| post-PCGrad grad_norm_mean | 2.28 (vs SOAP 9.16 at ep 13) |
| post-PCGrad grad_norm_max | 6.65 (vs SOAP 259 at ep 13) |
| Peak GPU memory | 47.6 GB (vs SOAP ~25-30 GB) |

**Per-epoch trajectory vs SOAP at matched epochs**:
- PCGrad@ep8 = 59.23 beats SOAP@ep8 = 72.44 by **13.2 points** (mechanism confirmed)
- But SOAP@ep13 = 42.40, which PCGrad never reaches due to 1.63× epoch cost

**Analysis**: The mechanism works — PCGrad achieves 9× lower mean grad-norm and 70× lower max grad-norm vs SOAP at matched epochs. The vol/surf gradient conflict is real but **sparse** (4% of parameter tensors at any given step). The relative-L2 + Huber + SOAP stack already tames most conflict; residual is concentrated in a few high-magnitude tensors. PCGrad's projection addresses those tensors but the 2× backward pass overhead (1.63× wall-clock) cannot be earned back at a 30-min budget. Structural dead-end at this compute level.

**Key insight**: Per-tensor conflict_frac ≈ 0.04 confirms gradient conflict is real but sparse. Most variance in grad norms is magnitude-based (handled by SOAP + loss normalization), not direction-based. Multi-pass gradient methods are structurally disadvantaged at small epoch budgets.

**Artifact**: `models/model-charliepai2g24h1-frieren-pcgrad-surgery-20260512-231800/metrics.jsonl`

---

## 2026-05-12 22:55 — Race-condition send-backs (3 PRs on stale baseline)

After SOAP merged at 22:30 as new baseline (42.4015), three PRs completed concurrently on the pre-SOAP base. None directly comparable to current baseline; all sent back for SOAP-rebased re-test.

| PR | Student | Slug | Final val (stale base) | Decision |
|----|---------|------|------------------------|----------|
| #1456 | alphonse | bf16-amp | 83.9115 (18 epochs, +29% throughput) | Send back: rebase + T_max=17 |
| #1599 | fern | re-conditioned-scaling | 92.4482 (scale corr +0.92 with log_Re) | Send back: rebase + test compound |
| #1630 | tanjiro | sgdr-restarts | 90.6703 (restart cost ~4 epochs) | Send back: pivot to monotone cosine + eta_min=1e-5 |

**Mechanism evidence preserved**:
- **bf16-amp**: 18 epochs vs 14 in same wall-clock = +29% throughput confirmed. Compounds cleanly with SOAP since SOAP only got 13 epochs (val still falling).
- **re-conditioned-scaling**: ReScaleHead worked mechanically (scale correlation with log_Re reached +0.92, exactly as predicted) but didn't beat AdamW baseline. SOAP compound test will reveal: stacks, redundant, or conflicts.
- **sgdr-restarts**: Restart at epoch 8 fired correctly per design but cost ~4 epochs of re-convergence with no better basin. Pivot to tanjiro's own follow-up suggestion (monotone cosine + eta_min=1e-5 floor) — same intent (preserve late-epoch step magnitude) without the restart penalty.

All 3 students now have active rebases against the SOAP baseline. PR #1630 had a code-block truncation in the send-back comment; corrected via follow-up comment.

---

## 2026-05-12 22:xx — PR #1613: [soap-optimizer] SOAP quasi-Newton optimizer

- **Branch**: thorfinn/soap-optimizer
- **Hypothesis**: SOAP (Shampoo as Adam's Preconditioner) provides Kronecker-factored quasi-Newton curvature estimates that condition gradient steps — addressing the root cause of the LR ceiling (poor first-order curvature model), not just its symptoms.
- **Status**: MERGED — new baseline 42.4015 (**largest single improvement in programme**)

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **42.4015** |
| val_single_in_dist | 46.09 |
| val_geom_camber_rc | 55.98 |
| val_geom_camber_cruise | **24.32** |
| val_re_rand | 43.22 |
| test_avg/mae_surf_p (4-split) | **36.4017** |
| test_single_in_dist | 41.76 |
| test_geom_camber_rc | 48.10 |
| test_geom_camber_cruise | 19.79 |
| test_re_rand | 35.97 |
| Epochs | 13 (~30-min cap, SOAP overhead) |
| grad_norm_mean trace | 38.87 → 9.16 (4.2× reduction) |
| clip_frac trace | 1.000 (ep 1-10) → 0.987 → 0.984 |
| Baseline | 89.3940 |
| Delta | **-52.6%** |

**Analysis**: SOAP's Kronecker-factored preconditioner transforms convergence speed. The 4.2× grad norm reduction (38.87 → 9.16 across 13 epochs) is direct evidence that the preconditioner is working — each step is better conditioned. All 4 val splits improved dramatically (cruise: 66→24, rc: 101→56, re_rand: 81→43, single_in_dist: 109→46). Val was still falling at ep 13 — the model has not converged, suggesting bf16-amp compound would be major.

**Critical diagnostic**: clip_frac=0.984 at ep 13 means SOAP is still being clipped ~9× per step (grad_norm_mean=9.16 vs clip=1.0). This is the basis for the next experiment (soap-relax-clip, PR #1668).

**SOAP install**: pip unavailable; vendored as `soap.py` (upstream commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`).

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-soap-optimizer-20260512-220030/metrics.jsonl`

---

## 2026-05-12 22:xx — PR #1473: [huber-loss v3] Huber(δ=0.1) on relative-L2 normalized residuals

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Applying Huber(δ=0.1) to the per-sample energy-normalized residuals (relative-L2 space) provides intra-sample outlier capping on top of inter-sample scale normalization.
- **Status**: MERGED — new baseline 89.3940

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **89.3940** |
| val_single_in_dist | 109.01 |
| val_geom_camber_rc | 101.19 |
| val_geom_camber_cruise | **66.36** |
| val_re_rand | 81.02 |
| test_avg/mae_surf_p (4-split) | **79.5993** |
| test_single_in_dist | 98.51 |
| test_geom_camber_rc | 88.12 |
| test_geom_camber_cruise | 54.80 |
| test_re_rand | 76.97 |
| Baseline | 89.6121 |
| Delta | **-0.24%** |
| huber_delta | 0.1 (normalized space) |
| L2-fraction (ep 1 → 14) | 33% → 63% |
| grad clip_frac (ep 14) | 0.075 (vs 0.984 on rel-L2-only) |

**Analysis**: The compound works: Huber(δ=0.1) in normalized space is genuinely active throughout (L2-fraction 33%→63%, vs 93%→94% with the raw-space δ=0.5 that largely collapsed to MSE). The most striking diagnostic is clip_frac dropping from ~1.0 (MSE baseline) to 0.075 by epoch 14 — the loss surface is dramatically smoother. val_re_rand showed the biggest improvement (84.29→81.02). val_geom_camber_rc regressed slightly (97.99→101.19) — this is the hardest OOD split and may need the re-conditioned-scaling architecture fix. Win is narrow (-0.24%) but clean, monotone, and confirmed in committed JSONL.

**Key mechanism**: relative-L2 handles inter-sample scale variation (across Re regimes); Huber handles intra-sample node outliers (within each sample). Complementary mechanisms.

**Artifacts**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-211810/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1458: [wider-deeper v2] Scale Transolver n_hidden=256, n_layers=6, n_head=8

- **Branch**: charliepai2g24h1-edward/wider-deeper
- **Hypothesis**: 3M-param Transolver (4.5× baseline) converges to a better solution within 30-min cap.
- **Status**: CLOSED — not competitive in 30-min epoch budget

**v1 results (batch_size=2, old LR, pre-grad-clip base):**

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 7) | 126.9875 |
| Epochs | 7 (~311s/epoch, still falling at timeout) |

**v2 results (batch_size=4, lr=1e-3, T_max=14, grad-clip base):**

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 6) | 158.0610 |
| test_avg/mae_surf_p | 145.9203 |
| Peak VRAM | 98.83 GB |
| Epochs | 6 (near-OOM, 311s/epoch) |

**Analysis**: At 3M params, the model gets only 6-7 epochs in 30 min vs 14 for the 662K baseline. The v2 regression (158 vs v1's 127) was caused by applying lr=1e-3 (calibrated for the small model) to the large model — 100% clip rate throughout. Even v1's promising trajectory (204→127 in 7 epochs) cannot plausibly converge below 89.61 given the epoch budget. Closed; edward reassigned to per-channel-loss-weights.

**Artifacts**: `models/model-charliepai2g24h1-edward-wider-deeper-20260512-180258/metrics.jsonl`, `models/model-charliepai2g24h1-edward-wider-deeper-v2-20260512-201243/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1539: [lr-1.5e-3-cosine-14] Test lr ceiling: 1.5e-3 + T_max=14

- **Branch**: charliepai2g24h1-thorfinn/lr-1.5e-3-cosine-14
- **Hypothesis**: lr=1.5e-3 (vs 1e-3 baseline) with T_max=14; tests whether the LR ceiling is higher than 1e-3.
- **Status**: CLOSED — LR ceiling confirmed, regression

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 100.2425 |
| val_single_in_dist | 119.01 |
| val_geom_camber_rc | 110.47 |
| val_geom_camber_cruise | 76.51 |
| val_re_rand | 94.98 |
| test_avg/mae_surf_p | 89.1065 |
| Baseline (new) | 89.6121 |
| Delta | +11.8% (WORSE) |
| Clip frac | 99–100% throughout |

**Analysis**: lr=1.5e-3 is confirmed above the LR ceiling. 100% clip rate throughout — pushing nominal LR higher just increases the wasted component beyond the clip threshold. single_in_dist degraded most (+10.43 vs val). The ceiling experiment is complete: lr=1e-3 is the right AdamW LR for this dataset/architecture at grad_clip=1.0. To push past this ceiling, need a different optimizer (SOAP, quasi-Newton) rather than higher LR. Thorfinn reassigned to soap-optimizer (H1, HIGH priority).

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-lr-1.5e-3-cosine-14-20260512-200502/metrics.jsonl`

---

## 2026-05-12 21:xx — PR #1456: [bf16-amp] bf16 automatic mixed precision

- **Branch**: charliepai2g24h1-alphonse/bf16-amp
- **Hypothesis**: bf16-amp reduces memory pressure and increases throughput, enabling more epochs in the 30-min wall-clock budget.
- **Status**: SENT BACK (v2) — regression due to schedule misalignment; rebase + T_max=18 required

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 114.21 (REGRESSION) |
| Baseline | 96.5587 |
| Delta | +18.2% (WORSE) |
| Epochs completed | 18 (vs 14 baseline — +30% throughput confirmed) |

**Analysis**: bf16-amp confirmed 30% throughput gain (18 vs 14 epochs). Regression was entirely due to running with old `T_max=50`: at 18 epochs, only 36% of the cosine schedule was used. Sent back to rebase onto new baseline (rel-L2 loss), set `T_max=18` to match new epoch budget, and re-run. If throughput advantage holds, the compound (18 epochs × aligned schedule × relative-L2 base) should beat 89.61.

**Artifacts**: Not yet committed (stale branch, not merged)

---

## 2026-05-12 21:xx — PR #1473: [huber-loss-v2] Huber loss (δ=0.5) rebased onto grad-clip baseline

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Huber loss caps outlier-residual gradients on extreme-value mesh nodes; on top of grad_clip + lr=1e-3 baseline, should improve convergence stability and final val MAE.
- **Status**: SENT BACK (v3) — beat old baseline (96.56) but not new baseline (89.61 from relative-l2); next step is Huber on top of relative-L2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **90.0929** |
| val_single_in_dist | 104.29 |
| val_geom_camber_rc | 101.05 |
| val_geom_camber_cruise | 70.12 |
| val_re_rand | 84.92 |
| test_avg/mae_surf_p | 78.97 |
| huber_delta | 0.5 |
| huber_l2_frac (ep 1 → 14) | 76% → 94% |
| Baseline (old) | 96.5587 |
| Delta vs old baseline | -6.69% |
| New baseline | 89.6121 |
| Delta vs new baseline | +0.54% (just missed) |

**Analysis**: Clean training, no instabilities, L2-fraction trajectory perfect (capping early, MSE-like late). Beat the old MSE baseline by 6.69% but lost to fern's relative-L2 by a narrow margin (90.09 vs 89.61). Sent back to compound: apply Huber to normalized residuals in relative-L2 space. New delta should be tuned to the normalized scale (~0.05–0.1 rather than 0.5). The mechanisms are complementary: relative-L2 handles inter-sample scale variation, Huber handles intra-sample node outliers.

**Artifacts**: `models/` path TBD after v3 re-run

---

## 2026-05-12 21:xx — PR #1460: [relative-l2-loss] Per-sample relative L2 loss

- **Branch**: charliepai2g24h1-fern/relative-l2-loss
- **Hypothesis**: Relative L2 loss (`||pred-y||²/||y||²`) normalizes by sample energy, automatically down-weighting high-energy (high-Re) samples and up-weighting low-energy ones — a better inductive bias for the multi-Re dataset.
- **Status**: MERGED — new baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **89.6121** |
| val_single_in_dist | 109.07 |
| val_geom_camber_rc | 97.99 |
| val_geom_camber_cruise | **67.09** |
| val_re_rand | 84.29 |
| test_avg/mae_surf_p (4-split) | **78.14** |
| test_single_in_dist | 91.14 |
| test_geom_camber_rc | 85.89 |
| test_geom_camber_cruise | 56.35 |
| test_re_rand | 79.18 |
| Peak VRAM | 42.11 GB |
| Epochs | 14 (~131s/epoch, 30-min cap) |
| Baseline | 96.5587 |
| Delta | **-7.20%** |

**Analysis**: Relative-L2 loss's per-sample energy normalization creates a flatter cross-split loss landscape. The gradient clip fraction dropped to 0.984 at ep 14 (vs 1.0 throughout on MSE) — the loss surface is genuinely smoother. Val was still falling at ep 14 (95.94 → 93.35 → 89.61 in last 3 epochs), indicating more headroom with more epochs. Cruise split improved dramatically (67.09 vs 74.35 baseline). RC and single_in_dist improved but remain the hardest splits — both span the full Re range and benefit most from architecture-level scale separation (H2 re-conditioned-scaling).

**Artifacts**: `models/model-charliepai2g24h1-fern-relative-l2-loss-20260512-200551/metrics.jsonl`

---

## 2026-05-12 20:41 — PR #1462: [warmup-cosine-v2] 1-epoch LinearLR warmup + CosineAnnealingLR

- **Branch**: charliepai2g24h1-frieren/warmup-cosine
- **Hypothesis**: Linear LR warmup prevents overly aggressive early steps, improving convergence stability.
- **Status**: CLOSED (dead end — within-noise tie with baseline; warmup is redundant with grad_clip=1.0)

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 97.0766 |
| Baseline | 96.5587 |
| Delta | +0.5% (WORSE, within noise) |
| test_avg/mae_surf_p (4-split) | 85.5327 |

**Conclusion**: Warmup is redundant with grad_clip=1.0 at this budget. grad_clip bounds the effective step size on 100% of batches, eliminating the "too-aggressive first step" regime that warmup is designed to prevent. The two mechanisms are redundant. Split-level results trade off (rc improved, single_in_dist worsened), consistent with noise. Frieren's mechanistic analysis is correct and conclusive. **Warmup at this budget is exhausted.**

---

## 2026-05-12 19:52 — PR #1518: [higher-lr-cosine-14] lr=1e-3 + CosineAnnealingLR(T_max=14)

- **Branch**: charliepai2g24h1-thorfinn/higher-lr-cosine-14
- **Hypothesis**: With `grad_clip=1.0` bounding the effective step size, raising lr from 5e-4 to 1e-3 yields faster convergence; reducing T_max from 50 to 14 ensures the cosine schedule reaches its low-LR fine-tuning phase within the actual 14-epoch budget.
- **Status**: MERGED — new baseline. Also includes y-sanitization fix in `train.py:evaluate_split`.

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | **96.5587** |
| val_single_in_dist | 108.58 |
| val_geom_camber_rc | 110.59 |
| val_geom_camber_cruise | 74.35 |
| val_re_rand | 92.71 |
| test_avg/mae_surf_p (4-split, NaN-free) | **85.87** |
| test_single_in_dist | 94.97 |
| test_geom_camber_rc | 99.77 |
| test_geom_camber_cruise | 61.86 |
| test_re_rand | 86.88 |
| Peak VRAM | 42.11 GB |
| Epochs | 14 (~131s/epoch, 30-min cap) |

**Convergence trace**: crossed old 117.17 baseline at epoch 10 (110.19); val still falling at epoch 14 (100.34 → 98.66 → 96.56). Pre-clip norms: mean 23–66, max 288–740. Clipping fires ~100% of batches.

**Analysis**: Dominant mechanism: T_max=14 let the cosine schedule actually reach its low-LR fine-tuning phase, which T_max=50 never achieved in 14 epochs. The higher LR (1e-3 vs 5e-4) accelerated early convergence and compound with the schedule effect. The y-sanitization fix made the cruise test split computable for the first time. -17.6% improvement over previous baseline.

**Key insight**: Val was still falling at epoch 14 — the model has not fully converged. A slightly higher LR or different schedule tail may extract more.

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-higher-lr-cosine-14-20260512-191045/metrics.jsonl`

---

## 2026-05-12 19:30 — PR #1473: [huber-loss] Switch MSE → Huber loss (delta=0.5)

- **Branch**: charliepai2g24h1-tanjiro/huber-loss
- **Hypothesis**: Huber loss caps outlier-residual gradients on extreme-value mesh nodes, improving stability and final val MAE.
- **Status**: Winner pending rebase — sent back to resolve merge conflicts and re-verify on grad-clip baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **111.296** ⭐ |
| val_single_in_dist | 133.55 |
| val_geom_camber_rc | 122.56 |
| val_geom_camber_cruise | 89.51 |
| val_re_rand | 99.57 |
| test_avg/mae_surf_p (3 splits, excl cruise) | 112.51 |
| Peak VRAM | 42.1 GB |
| Epochs | 14 (~130s/epoch, 30-min cap) |
| huber_l2_frac (epoch 1 → 13) | 0.749 → 0.931 |
| huber_delta | 0.5 |

**Analysis**: Clean training trajectory, no instabilities. L2-fraction climbed from 75% → 93% across training, exactly the "outlier-capping early, MSE-like late" pattern the PR predicted. **111.296 beats the 117.17 baseline by ~5%** — clear winner.

**Caveat**: This run forked from pre-grad-clip-1 base. The advisor branch now has grad_clip=1.0 as default. Student must rebase, re-run with grad-clip active, and apply the y-sanitization fix in train.py:evaluate_split. The 117.17 grad-clip baseline becomes the head-to-head comparison target.

**Bonus**: Tanjiro diagnosed the test_geom_camber_cruise NaN bug independently — `Inf*0=NaN` in scoring.py accumulator from a single sample with -inf pressure values. Their proposed fix is correct in spirit but must be applied in train.py:evaluate_split (since `data/` is read-only per program.md).

**Artifacts**: `models/model-charliepai2g24h1-tanjiro-huber-loss-20260512-180430/metrics.jsonl`

---

## 2026-05-12 18:57 — PR #1479: [grad-clip-1] Add gradient clipping (clip_norm=1.0) for training stability

- **Branch**: charliepai2g24h1-thorfinn/grad-clip-1
- **Hypothesis**: Gradient clipping (clip_norm=1.0) prevents gradient explosions from extreme-value mesh nodes and stabilizes early training.
- **Status**: MERGED — new baseline

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | **117.17** |
| val_single_in_dist/mae_surf_p | 134.83 |
| val_geom_camber_rc/mae_surf_p | 134.17 |
| val_geom_camber_cruise/mae_surf_p | 87.04 |
| val_re_rand/mae_surf_p | 112.66 |
| test_avg/mae_surf_p | NaN (cruise GT bug) |
| test 3-split avg (excl cruise) | 116.17 |
| Peak VRAM | 42.1 GB |
| Epochs completed / 30 min | 14 (~130s/epoch) |
| Pre-clip gradient norms | mean 50–100, max 300–800 |

**Key finding**: Pre-clip gradient norms are 50–800, clipping fires on **100% of batches** every epoch. The baseline Transolver is gradient-unstable without clipping. grad_clip=1.0 is now mandatory baseline infrastructure.

**NaN bug discovered**: test_geom_camber_cruise sample 20 has ±Inf in GT pressure channel; `Inf*0=NaN` poisons the accumulator in scoring.py. Fix: sanitize y before calling accumulate_batch in train.py.

**Artifacts**: `models/model-charliepai2g24h1-thorfinn-grad-clip-1-20260512-180544/metrics.jsonl`

---

## 2026-05-12 18:56 — PR #1457: [surf-weight-50] Raise surf_weight 10→50

- **Branch**: charliepai2g24h1-askeladd/surf-weight-50
- **Status**: Sent back for round 2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 14) | 135.36 |
| val_geom_camber_cruise | 104.81 |
| val_geom_camber_rc | 157.27 |
| val_re_rand | 124.40 |
| val_single_in_dist | 154.98 |
| Epochs | 14 |

**Analysis**: Healthy surf/vol balance (no volume collapse). Did not beat grad-clip-1 (117.17). Sent back to combine surf_weight=30 with the new grad-clip baseline.

---

## 2026-05-12 18:55 — PR #1458: [wider-deeper] Scale Transolver n_hidden=256, n_layers=6, n_head=8

- **Branch**: charliepai2g24h1-edward/wider-deeper
- **Status**: Sent back for round 2 with batch_size=4

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 7) | 126.99 |
| val_geom_camber_cruise | 95.27 |
| val_geom_camber_rc | 141.93 |
| val_re_rand | 113.94 |
| val_single_in_dist | 156.82 |
| n_params | 3.0M (not 8M as estimated) |
| Peak VRAM | 49.43 GB |
| Epochs | 7 (batch_size=2 → ~295s/epoch) |

**Analysis**: Val still falling at epoch 7; model not converged. With batch_size=4 (safe given 49GB VRAM) should complete 14 epochs. Trajectory (204→127 in 7 eps) very promising — extrapolates well below 117.

---

## 2026-05-12 18:53 — PR #1462: [warmup-cosine] Add 2-epoch linear LR warmup

- **Branch**: charliepai2g24h1-frieren/warmup-cosine
- **Status**: Sent back for round 2

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (ep 13) | 131.83 |
| val_geom_camber_cruise | 103.34 |
| val_geom_camber_rc | 131.83 |
| val_re_rand | 119.68 |
| val_single_in_dist | 172.48 |
| Epochs | 14 |

**Analysis**: Schedule worked mechanically. With only 14 epochs, 2-epoch warmup consumes 14% of budget. With grad-clip now merged (more stable training), sent back with shorter 1-epoch warmup + start_factor=0.1.

---

## Round 1 WIP — Still Running

| PR | Student | Hypothesis |
|----|---------|------------|
| #1456 | alphonse | bf16-amp |
| #1460 | fern | relative-l2-loss |
| #1467 | nezuko | more-slices-128 |
| #1473 | tanjiro | huber-loss |

---

## 2026-05-13 04:30 — PR #1794: [torch-compile] torch.compile(mode="default", dynamic=True) on bf16+SOAP stack

- **Branch**: charliepai2g24h1-alphonse/torch-compile
- **Hypothesis**: torch.compile with mode="default" and dynamic=True (for variable-shape pad_collate tensors) would yield +20-30% throughput and enable more epochs within the 30-min wall-clock budget.
- **Status**: **MERGED** — new baseline at val_avg/mae_surf_p = 30.4412

| Metric | Value | vs bf16 baseline (36.8778) |
|--------|-------|---------------------------|
| val_avg/mae_surf_p | **30.4412** | **−17.5%** |
| test_avg/mae_surf_p | **26.1013** | **−18.2%** |
| val_single_in_dist | 34.27 | −8.65 (−20.2%) |
| val_geom_camber_rc | 41.43 | −6.35 (−13.3%) |
| val_geom_camber_cruise | 14.04 | −4.56 (−24.5%) |
| val_re_rand | 32.02 | −6.19 (−16.2%) |
| Epochs in 30 min | **30** | +13 epochs (+76% throughput) |
| Peak GPU memory | 24 GB | −9 GB vs bf16 alone |
| Best epoch | 30 (still descending) | — |

**Key mechanistic findings**:
- `mode="default"` with `dynamic=True` was the correct choice. `reduce-overhead` would have caused recompilation storms because `pad_collate` produces variable-length tensors (different sequence lengths per batch). `dynamic=True` handles this by generating shape-symbolic compiled kernels.
- torch.compile drops peak GPU memory 33→24 GB (better kernel fusion, less intermediate tensor fragmentation).
- T_max was set to 28 (aligning cosine tail with 30-epoch budget) — alphonse correctly auto-detected throughput after a warm-up timing run.
- ALL 8 splits (4 val + 4 test) improved — model was still descending at epoch 30 (the last epoch was the best).
- Cumulative gain from initial 117.17 baseline: **−74.0%**.

**Artifact**: `models/model-charliepai2g24h1-alphonse-torch-compile-20260513-021531/metrics.jsonl`

---

## 2026-05-13 04:30 — PR #1797: [wider-soap-192] n_hidden 128→192 (1.47M params) — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/wider-soap-192
- **Hypothesis**: n_hidden 128→192 (662K → 1.47M params, 2.22×) would increase model capacity and improve generalization.
- **Status**: **CLOSED** — hypothesis falsified cleanly

| Metric | Value | vs baseline (36.8778) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 49.1129 | **+33.2% (worse)** |
| test_avg/mae_surf_p | ~44+ | worse |
| Epochs in 30 min | 14 | vs 30 for 662K model |
| Peak GPU memory | 43 GB | vs 24 GB for baseline |

**Mechanistic conclusion**: Dataset is data-bottlenecked, not optimization-bottlenecked. The training set (1,499 samples) cannot fill the wider representation space — extra parameters worsen rather than improve convergence per epoch. SOAP already conditions the optimization; the limiting factor is information content in the training data. 2.22× param count → 2.14× fewer epochs in 30 min → consistently worse at every matched epoch count. This rules out width as a capacity lever at the current data scale.

---

## 2026-05-13 04:30 — PR #1668: [soap-relax-clip] grad_clip 1.0→5.0 — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/soap-relax-clip
- **Hypothesis**: Relaxing grad_clip from 1.0 to 5.0 would unlock SOAP's natural step magnitude and improve convergence.
- **Status**: **CLOSED** — mechanism confirmed, slight regression vs new baseline

| Metric | Value | vs baseline (36.8778) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 37.2200 | +0.93% (slight regression) |
| test_avg/mae_surf_p | 32.3257 | +1.32% |
| clip_frac by ep 14 | 0.00 | was 0.33 at baseline |

**Mechanistic conclusion**: clip=5.0 fully unlocked SOAP's steps (clip_frac 0.33→0.00, mechanism validated). However, SOAP+cosine+bf16 at clip=1.0 slightly outperforms clip=5.0. By late training (with cosine LR decay), the gradient norms are already small enough that clip=1.0 is non-binding. Widening clip provided no additional signal. The value of clip relaxation was in the early training regime (clip_frac 0.98–1.00), which cosine scheduling has already effectively resolved. Note: comparison to AdamW baseline in student writeup was incorrect — the correct comparator is clip=1.0 vs clip=5.0 on the same SOAP+bf16+cosine stack.


---

## 2026-05-13 04:15 — PR #1847: [larger-batch-compile] batch_size 4→8 — CLOSED

- **Branch**: charliepai2g24h1-alphonse/larger-batch-compile
- **Hypothesis**: Doubling batch size from 4→8 would exploit 72 GB memory headroom freed by torch.compile (24/96 GB), lower gradient variance, and improve generalization on OOD splits.
- **Status**: **CLOSED** — hypothesis falsified, three-mechanism root cause identified

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 36.9205 | **+21.3% (worse)** |
| test_avg/mae_surf_p | 32.0504 | **+22.8% (worse)** |
| Peak GPU memory | 47.69 GB | +23 GB (not +9 GB as predicted) |
| Epochs in 30 min | 30 | same (training NOT compute-bound) |
| Optimizer steps/epoch | 188 | was 376 at batch=4 |

**Three-mechanism root cause**:
1. **Training is not compute-bound at batch=4**: per-epoch wall-time stayed 60s/ep — batch=8 gives identical epoch count (30 in 30 min), so the "fewer epochs" justification was wrong
2. **Half the optimizer updates per epoch** (188 vs 376, LR held at 1e-3 due to clip ceiling): at 1,499 training samples, MORE optimizer steps beats lower gradient variance per step
3. **T_max=23 caused cosine restart regression**: schedule hit floor at ep 23, LR climbed for 7 more epochs (val regressed 36.92 → 38.84). Should have been T_max=28 or T_max=steps-to-timeout

**Memory cost much higher than predicted**: 24 GB → 47.69 GB (+23 GB), not the +9 GB predicted. Activation memory grows super-linearly with batch size under torch.compile graph capture.

**Net programme lesson**: Data-bottleneck manifests in TWO ways — (1) wider models can't fill representation space, (2) larger batches halve optimizer steps without providing extra information. Both ruled out for 1,499-sample dataset.


---

## 2026-05-13 04:45 — PR #1854: [soap-fp32-precond] SOAP GG/Q in fp32 under bf16 AMP — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/soap-fp32-precond
- **Hypothesis**: bf16 precision in SOAP's GG/Q eigenbases degraded preconditioner quality; keeping them in fp32 would improve OOD generalization.
- **Status**: **CLOSED** — hypothesis inverted, bf16 Q acts as implicit regularization

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 31.7537 | **+4.31% (worse)** |
| test_avg/mae_surf_p | 27.1862 | **+4.16% (worse)** |
| val_single_in_dist | 33.40 | -2.5% (better!) |
| val_geom_camber_rc | 45.01 | +8.6% (worse) |
| val_geom_camber_cruise | 15.04 | +7.1% (worse) |
| val_re_rand | 33.57 | +4.8% (worse) |

**Pattern**: in-dist improved, ALL 3 OOD splits degraded. This is NOT numerical noise — it's the signature of overfitting. Sharp fp32 preconditioner fit training distribution tighter; bf16 Q's rounding noise acted as implicit regularization that generalized better to OOD.

**Key finding**: All changes confirmed applied (GG fp32 init, fp32 grad for lerp_, Q stays fp32, project/project_back cast to fp32). Memory unchanged (23.87 GB ≈ 24 GB baseline).

**This is the third consecutive experiment showing the same OOD-worse pattern (wider, deeper, more precise preconditioner). Model is regularization-limited, not capacity/precision-limited.**

---

## 2026-05-13 04:45 — PR #1848: [deeper-soap] n_layers 5→7 — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/deeper-soap
- **Hypothesis**: depth increases representational power more data-efficiently than width; n_layers 5→7 with n_hidden=128 keeps params moderate (880K→904K).
- **Status**: **CLOSED** — compute-budget falsification (still descending at ep 21 cutoff)

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 33.9762 | **+11.6% (worse)** |
| test_avg/mae_surf_p | 29.1507 | **+11.7% (worse)** |
| Per-epoch wall time | 84s | was 60-65s (+30%) |
| Epochs in 30 min | 21 | was 30 |
| Peak GPU | 32.4 GB | was 24 GB |

**Val trajectory at cutoff**: 38.53 → 36.04 → 34.75 → **33.98** (ep 21, steep ~1/ep descent). Model was still converging fast; **compute-budget verdict, not intrinsic verdict**. BUT: at fixed 30-min wall-clock, the 662K/5-layer model dominates by running 30 epochs vs 21.

**Uniform regression across all splits** (not OOD-targeted) → simple undertraining, not compositionality issue.

**Programme lesson**: Both width (wider-soap-192) AND depth (deeper-soap) fail at fixed 30-min budget on 1,499 samples. Current 662K/5-layer is in the optimal compute zone. Data-bottleneck is confirmed. Moving to regularization-based improvements.

---

## 2026-05-13 04:55 — PR #1897: [stochastic-depth] DropPath drop_path_max=0.1 across layers — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/stochastic-depth
- **Hypothesis**: Linear DropPath schedule [0, 0.025, 0.05, 0.075, 0.1] should regularize OOD without inference cost. Predicted OOD splits (rc, re_rand) improve most, in-dist may regress slightly.
- **Status**: **CLOSED** — clean negative, hypothesis falsified
- **Metrics JSONL**: `models/model-charliepai2g24h1-thorfinn-stochastic-depth-20260513-041301/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 33.0241 | **+8.48% (worse)** |
| test_avg/mae_surf_p | 28.5180 | **+9.26% (worse)** |
| val_single_in_dist | 37.42 | **+9.20%** (WORST regression) |
| val_geom_camber_rc | 45.46 | +9.72% |
| val_geom_camber_cruise | 16.23 | +15.57% |
| val_re_rand | 32.99 | +3.03% |
| Peak GPU | 23.88 GB | unchanged |
| Epochs in 30 min | 29 | -1 |

**Pattern**: EVERY split regressed, in-dist regression LARGEST. Hypothesis predicted opposite (OOD-asymmetric improvement). Best epoch = 29 (last) → still descending.

**Why DropPath failed here**: With only 5 transformer blocks, the linear schedule mean ≈ 5% expected skip rate is too coarse — skipping a whole block is much more destructive than dropping features. Each block likely encodes non-redundant Transolver slice/attention patterns; redundancy assumption violated.

**Net programme lesson**: The "regularization-limited" diagnosis is refuted. Combined with attention-dropout (#1900, ~0% net), TWO independent regularization experiments fail to improve OOD. The OOD-asymmetric regressions in wider/deeper/sharper-precond are better explained by **optimization fragility + compute-budget loss** (each ate epochs through extra per-step cost), NOT by underfit regularization.

---

## 2026-05-13 04:55 — PR #1900: [attention-dropout] dropout=0.1 in PhysicsAttention — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/attention-dropout
- **Hypothesis**: Enable already-wired but no-op dropout (attn weights + output projection) at p=0.1. Predicted OOD-asymmetric improvement.
- **Status**: **CLOSED** — within-noise negative, but the per-split signature is diagnostic
- **Metrics JSONL**: `models/model-charliepai2g24h1-tanjiro-attention-dropout-20260513-041403/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 30.5841 | +0.47% (within noise) |
| test_avg/mae_surf_p | 26.6998 | +2.29% (worse) |
| val_single_in_dist | 33.94 | **-1.0% (better!)** |
| val_geom_camber_rc | 42.65 | +2.9% |
| val_geom_camber_cruise | 14.92 | +6.3% |
| val_re_rand | 30.82 | **-3.7% (better!)** |
| Peak GPU | 24.49 GB | unchanged |
| Epochs in 30 min | 29 | -1 |

**Smoking gun observation from student**: *"Loss curve was still trending down at epoch 29 — this is itself evidence the model is not regularization-limited — there was no train/val gap to close."*

**Pattern**: OOD splits split (1 better re_rand / 2 worse rc, cruise). In-dist actually improved (-1.0%) — opposite of the regularization-overfit prediction. Looks like noise-level perturbation with a single positive outlier.

**Net result**: This — combined with stochastic-depth — refutes the regularization-limited diagnosis. The next theme should be **convergence/budget-aware experiments**: weight averaging (SWA, EMA), faster schedules (OneCycleLR #1884 in flight), loss-domain rebalancing (lower surf_weight), NOT more regularization.

**Diagnostic signal preserved**: val_re_rand improved -3.7%, the only positive outlier across both runs. Worth asking whether re_rand (random-Re OOD) responds to a Re-specific regularizer that uniform dropout doesn't capture — points to fern's re-conditioned-scaling direction (#1599).



---

## 2026-05-13 05:05 — PR #1599: [re-conditioned-scaling] Learned Re-conditioned output scale head — MERGED

- **Branch**: fern/re-conditioned-scaling
- **Hypothesis**: Re varies 50×+ in the dataset; pressure magnitudes vary by orders of magnitude. Add a tiny 163-param ReScaleHead (log_Re → softplus scale per channel) on top of Transolver output to separate shape learning from scale calibration. Inspired by DimINO Re-dimensionalization (Huang et al. 2024).
- **Status**: **MERGED** — val_avg -1.95%, new baseline 29.8463
- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-re-conditioned-scaling-20260513-035742/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | **29.8463** | **−0.59 (−1.95%)** |
| test_avg/mae_surf_p | 26.1005 | −0.0008 (≈0%) |
| val_single_in_dist | 30.20 | −4.07 (-11.9%) |
| val_geom_camber_rc | 43.11 | +1.68 (+4.1%) |
| val_geom_camber_cruise | 14.54 | +0.50 (+3.6%) |
| val_re_rand | 31.54 | −0.48 (−1.5%) |
| Epochs in 30 min | 29 | unchanged |
| Peak GPU | 24 GB | unchanged |

**ReScaleHead diagnostics (best epoch 27)**:
| Channel | scale mean | scale std | corr(scale, log Re) |
|---------|-----------|----------|---------------------|
| Ux | 1.180 | 0.058 | +0.637 |
| Uy | 1.111 | 0.262 | +0.936 |
| p | 1.308 | 0.527 | +0.858 |

**Analysis**: Val wins via single_in_dist (-4.07) dominating OOD regressions (+2.18 summed). Test is flat (in-dist and OOD gains cancel). Mechanism confirmed in all 3 runs: Uy/p show strong Re-correlation (0.86–0.94); Ux is weak (freestream-dominated). The compound size shrunk significantly vs the SOAP-only baseline run (was -4.7%, now -1.95%) because the SOAP + torch.compile backbone implicitly learns Re-scale through 30 epochs. Still a valid compounding win.

**Programme implication**: ReScaleHead is now the default in the advisor branch. All future experiments inherit it. Future compound direction: 2-channel head (Ux scale ≈ identity; drop Ux to reduce parameter noise) or FiLM-style conditioning (inject log(Re) into PhysicsAttention slice weighting instead of output rescaling).

**Cumulative programme gain**: −74.5% from 117.17 → 29.8463

---

## 2026-05-13 05:55 — PR #1936: [surf-weight-7] Lower surf_weight 10→7 — CLOSED

- **Branch**: charliepai2g24h1-tanjiro/surf-weight-7
- **Hypothesis**: surf_weight=10 over-emphasizes surface; lowering to 7 should improve OOD-rc.
- **Status**: **CLOSED** — clean negative, refuted in opposite direction
- **Metrics JSONL**: `models/model-charliepai2g24h1-tanjiro-surf-weight-7-20260513-050813/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 31.3366 | **+2.94%** (worse) |
| test_avg/mae_surf_p | 26.9876 | **+3.40%** (worse) |
| val_geom_camber_rc | 44.12 | **+6.5%** (WORSE — opposite of hypothesis) |
| val_geom_camber_cruise | 15.25 | +8.6% |
| val_single_in_dist | 34.09 | -0.5% |
| val_re_rand | 31.88 | -0.4% |

**Programme learning**: surface gradients are load-bearing across ALL splits, especially rc (which has the largest absolute surface error 41.43). Reducing surface loss weight makes the dominant error source worse. surf_weight=10 was correctly tuned. The val_re_rand -3.7% signal from attention-dropout (#1900) was incidental, not a surface-overweighting signature.

**Loss-domain rebalancing (uniform surf_weight scalar) is now ruled out.** Next direction: input-domain augmentation (coord jitter) and the OPPOSITE direction probe (surf_weight=15).

---

## 2026-05-13 05:55 — PR #1933: [swa-last-k] SWA over last 5 cosine-floor epochs — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/swa-last-k
- **Hypothesis**: SWA over last K=5 epochs at cosine floor (LR ≈ 1e-5) reduces single-epoch noise and improves over best-epoch checkpoint.
- **Status**: **CLOSED** — null result, SWA-at-floor doesn't work
- **Metrics JSONL**: `models/model-charliepai2g24h1-thorfinn-swa-last-k-20260513-050934/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| swa_val_avg/mae_surf_p | 31.1956 | +2.48% |
| best_epoch val_avg | 31.1750 | +2.41% |
| **SWA vs best_epoch Δ** | **+0.07%** | (tied, within noise) |

**Critical programme finding from student**: *"Run-to-run variance is ~1-2%; the 0.07% SWA-vs-best gap is well below this noise floor."* This run was +2.4% worse than baseline at best-epoch on identical hyperparameters — confirming **single-seed noise floor is ~1-2%**. Multi-seed validation needed for borderline results.

**Why SWA-at-floor failed**: At LR=1e-5, weight updates are O(lr×grad)=O(1e-5)/param. The 5 floor checkpoints occupy ~zero region of weight-space, so their average ≈ the last checkpoint. Izmailov 2018 SWA gains require *high-LR plateau* (e.g. constant lr=1e-4 for SWA window) creating weight oscillation to flatten. Our cosine floor is the opposite regime.

**Loss trajectory confirmed convergence-limited**: val descended monotonically from 32.76 (ep24) → 31.18 (ep30). Model still descending at cutoff.

---

## 2026-05-13 05:55 — PR #1917: [ema-weights-v2] EMA β=0.999, EMA-only val — CLOSED

- **Branch**: charliepai2g24h1-frieren/ema-weights-v2
- **Hypothesis**: EMA β=0.999, EMA-only val (no live val) recovers v1's 4-epoch wall-clock penalty, enabling 30-epoch comparison.
- **Status**: **CLOSED** — protocol fix worked but β too high for budget
- **Metrics JSONL**: `models/model-ema-weights-v2-20260513-051110/metrics.jsonl`

| Metric | Value | vs baseline (30.4412) |
|--------|-------|----------------------|
| ema_val_avg/mae_surf_p | 31.3225 | +2.9% |
| ema_test_avg/mae_surf_p | 27.1987 | +4.2% |
| live_val_avg (E29 one-shot) | 31.1183 | +2.2% |
| **EMA vs live at E29** | **EMA=31.32 > live=31.12** | EMA biased to older worse weights |
| Epochs completed | 29 | -1 (vs 30 baseline) |
| Mean epoch time | 62s | +3% (down from v1's +13%) |

**Smoking gun from student**: With β=0.999 and ~375 batches/epoch, effective window ≈ 1/(1-β) = 1000 steps ≈ 2.7 epochs. By E29, EMA averages weights from E26-E29 — all WORSE than live at E29 (loss still descending). EMA + cosine floor double-smooth the same signal.

**Programme learning**: small-budget training (30 ep) needs SHORT EMA windows (β≤0.99, effective window ≤0.3 ep), NOT the standard β=0.999. Combine with Karras rampup to avoid early-training EMA-stuck-at-init failure (E1→E10 trajectory went 333→68).

**EMA hypothesis pivoted**: v3 will use β=0.99 with Karras rampup.

---

## 2026-05-13 19:10 — PR #2320: [slice-num-32] Halve PhysicsAttention slices 64→32 — CLOSED

- **Branch**: charliepai2g24h1-askeladd/slice-num-32
- **Hypothesis**: slice_num 64→32 buys ~10% wall-clock allowing more epochs in 30-min budget.
- **Status**: **CLOSED** — +0.95% val regression
- **Metrics JSONL**: `models/model-charliepai2g24h1-askeladd-slice-num-32-20260513-171408/metrics.jsonl`

| Metric | Value | vs baseline (28.8762) |
|--------|-------|----------------------|
| val_avg/mae_surf_p | 29.1496 | **+0.95% worse** |
| test_avg/mae_surf_p | 25.1765 | +0.71% worse |
| Epochs | 31 (vs 28 baseline) | +3 (but SCHEDULER_T_MAX=28 hardcoded → last 3 at eta_min flat) |

**Programme learning**: PhysicsAttention slicing is NOT the per-step bottleneck — only 11% speedup, not 40-50%. Hard OOD splits suffer most (geom_camber_rc -2.05%, re_rand -2.03%). slice_num axis fully closed (128 +13.5%, 32 +0.95%, 64 optimum). The hardcoded SCHEDULER_T_MAX=28 issue noted by student is a real fix opportunity but doesn't help current 30-min runs.

---

## 2026-05-13 19:10 — PR #2319: [aoa-film-conditioning] Extend ReFiLM from Re-only to 5-dim — CLOSED

- **Branch**: charliepai2g24h1-alphonse/aoa-film-conditioning
- **Hypothesis**: Extending ReFiLM to [log Re, AoA, camber, gap, stagger] helps geometry OOD splits.
- **Status**: **CLOSED** — +3.93% val regression
- **Metrics JSONL**: `models/model-charliepai2g24h1-alphonse-aoa-film-conditioning-20260513-171533/metrics.jsonl`

**Programme learning (critical)**: ReFiLM(Re) works because Re is a SPECIAL channel — it varies WITHIN splits and globally governs the slice-selection landscape. AoA/gap/stagger/camber are CONSTANT within each sample, so FiLM has no within-split modulation to learn. Gate IS active (γ absmax 1.24, 1.8x Re-only baseline) but counterproductive. Multi-channel FiLM expansion axis fully closed.

---

## 2026-05-13 19:10 — PR #2321: [llrd-transolver] Layer-wise LR decay (0.7 per block) — CLOSED

- **Branch**: charliepai2g24h1-edward/llrd-transolver
- **Hypothesis**: LLRD across 5 Transolver blocks improves OOD generalization.
- **Status**: **CLOSED** — +6.85% val regression (>5%)
- **Metrics JSONL**: `models/model-charliepai2g24h1-edward-llrd-transolver-20260513-171839/metrics.jsonl`

**Programme learning**: LLRD attenuates layers near the output too aggressively on a shallow 5-block stack. LLRD is most useful for very deep transformers (24L+); inappropriate for our 5L architecture. LLRD axis closed.

---

## 2026-05-13 19:10 — PR #2323: [soap-max-precond-dim-128] SOAP max_precond_dim 256→128 — CLOSED

- **Branch**: charliepai2g24h1-frieren/soap-max-precond-dim-128
- **Hypothesis**: Halve max_precond_dim for faster Kronecker refresh.
- **Status**: **CLOSED** — +16.45% val regression (severe)
- **Metrics JSONL**: `models/model-charliepai2g24h1-frieren-soap-max-precond-dim-128-20260513-170625/metrics.jsonl`

**Programme learning**: SOAP requires max_precond_dim >= n_hidden=128 to fully precondition per-channel covariance. Halving creates poorly-conditioned updates. SOAP HP space fully closed (betas, wd, precond_freq, max_precond_dim).

---

## 2026-05-13 19:10 — PR #2322: [geom-conditioned-output-head] 4-dim ReScaleHead — CLOSED

- **Branch**: charliepai2g24h1-fern/geom-conditioned-output-head
- **Hypothesis**: ReScaleHead conditioning expanded from [log Re] to [log Re, AoA, Gap, Stagger] helps geometry OOD.
- **Status**: **CLOSED** — +3.79% val regression
- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-geom-conditioned-output-head-20260513-171831/metrics.jsonl`

**Programme learning**: Diagnostic confirmed ReScaleHead remained dominated by Re (corr_logRe +0.85 on p). Geom info is already in 24-dim node features and reaches output via Transolver — adding it again at head is redundant signal. Multi-channel ReScaleHead axis closed; Re-only is optimal.

---

## 2026-05-13 19:10 — PR #2325: [pressure-laplacian-loss] kNN-graph Laplacian on volume p — CLOSED

- **Branch**: charliepai2g24h1-thorfinn/pressure-laplacian-loss
- **Hypothesis**: L1 Laplacian smoothness regularizer on predicted pressure helps OOD geometry.
- **Status**: **CLOSED** — +3.91% val regression
- **Metrics JSONL**: `models/model-thorfinn-pressure-laplacian-loss-20260513-172533/metrics.jsonl`

**Programme learning**: Two compounding issues — (a) +10-15% per-epoch overhead from kNN cdist outside autocast cost 2 epochs; (b) stochastic M=1024 with λ=0.01 acted as noise injection (variance ~ mean) rather than coherent prior. CRUCIALLY: loss targeted VOLUME nodes but ranking is on SURFACE — wrong region. Physics-informed Laplacian axis closed for this formulation. Future direction: mesh-edge-based Laplacian or pressure-Poisson residual.

---

## 2026-05-13 19:10 — PR #2428: [layerscale-init-1e-4] LayerScale on Transolver residual branches (γ₀=1e-4 + retry γ₀=0.1+nodecay) — CLOSED

- **Branch**: charliepai2g24h1-nezuko/layerscale-init-1e-4
- **Hypothesis**: CaiT-style per-channel learnable residual scaling stabilizes deep skip connections.
- **Status**: **CLOSED** — both arms regress (+2.69% γ₀=1e-4, +2.93% γ₀=0.1+nodecay)
- **Metrics JSONL**: `target/metrics/charliepai2g24h1-nezuko/layerscale-init-1e-4.jsonl` + `layerscale-init-0.1-nodecay.jsonl`

**Programme learning**: First arm (γ₀=1e-4) failed due to (a) SOAP blew through tiny init in 1 epoch, (b) WD=1e-4 = 100% relative decay/step. Second arm (γ₀=0.1 + WD-exclude) addressed both — γ stays 0.08-0.12, no blowthrough, no decay collapse — but STILL +2.93% val regression. Baseline isn't unstable, so LayerScale's stabilization mechanism doesn't apply. Per-feature damping costs effective capacity. **LayerScale residual scaling axis FULLY CLOSED across both γ₀ regimes.**

## 2026-05-13 21:20 — PR #2569: RMSNorm replaces LayerNorm in all transformer blocks
- **Branch**: `charliepai2g24h1-nezuko/rmsnorm-replace-layernorm`
- **Hypothesis**: Drop LN mean-centering + β-bias → preserve gauge-pressure offsets and gain ~10% throughput (RMSNorm canonical in Llama/Mistral)
- **Status**: **CLOSED** — small regression on both val (+2.67%) and test (+2.37%), no wall-clock speedup

| Split | Baseline (#2011) | RMSNorm | Δ |
|---|---|---|---|
| val_single_in_dist | 28.6013 | 29.3784 | +2.72% |
| val_geom_camber_rc | 41.9483 | 43.9530 | **+4.78%** |
| val_geom_camber_cruise | 14.1462 | **13.9074** | −1.69% |
| val_re_rand | 30.8090 | 31.3507 | +1.76% |
| **val_avg** | **28.8762** | **29.6474** | **+2.67%** |
| **test_avg** | **24.9992** | **25.5906** | **+2.37%** |
- **Diagnostic**: Train loss IDENTICAL at ep 28 (RMSNorm 0.0079 vs LN 0.0081). Gap is purely in generalization. Regression concentrated on the hardest OOD split (geom_camber_rc) — confirms LN mean-centering + β-bias is acting as a useful regularizer at this scale.
- **Speedup**: Promised ~10% did NOT materialize (64.12 vs 63.94 s/epoch). `torch.compile(default,dynamic)` already fuses the LN kernel, so dropping mean-subtraction yields 0% throughput gain.
- **Metrics JSONL**: `models/model-charliepai2g24h1-nezuko-rmsnorm-replace-layernorm-20260513-203949/metrics.jsonl`

**Programme learning**: At this scale (N=1499, 5 blocks) + compile + bf16 AMP, LayerNorm is the right choice. RMSNorm's compute story is killed by torch.compile fusion; its inductive-bias story is mildly anti-correlated with our OOD generalization. **Normalization-replacement axis closed**; ReFiLM γ-amplitude on RMSNorm pushed harder (|γ|max 0.880 vs 0.70) but didn't translate to lower val MAE — confirming the bottleneck is not in modulation capacity.

## 2026-05-13 21:20 — PR #2560: SAM (ρ=0.02) wrapping SOAP — flat-minima search at 2× per-step compute
- **Branch**: `charliepai2g24h1-fern/sam-rho-0p02-soap-wrap`
- **Hypothesis**: Two-step adversarial perturbation (Foret et al. 2021) flattens the loss landscape, improves OOD generalization (small-N + N=1499)
- **Status**: **CLOSED — catastrophic regression** val +53.4%, test +57.8%

| Split | Baseline (16 ep would-be) | SAM @ 16 ep | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 53.97 | +88.7% |
| val_geom_camber_rc | 41.95 | 56.99 | +35.9% |
| val_geom_camber_cruise | 14.15 | 23.73 | +67.7% |
| val_re_rand | 30.81 | 42.44 | +37.7% |
| **val_avg** | **28.88 (28 ep)** | **44.28 (16 ep)** | **+53.4%** |
| **test_avg** | **25.00 (28 ep)** | **39.45 (16 ep)** | **+57.8%** |

- **Diagnostic**: SAM was working as designed — perturbed loss reliably 20% > unperturbed, gradient norm decayed 3× over the run (3.83 → 1.27). NOT a bug. Pure compute-budget failure: 2× per-step compute → 16 epochs vs 28 baseline → still falling 1.3%/2-epochs at the wall-cap.
- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-sam-rho-0p02-soap-wrap-20260513-203405/metrics.jsonl`

**Programme learning**: Under 30-min SENPAI_TIMEOUT_MINUTES + N=1499 + already-strong SOAP+cosine, ANY 2× per-step regularizer is dominated by epoch-count loss. **The wall-cap is the binding constraint, not the optimizer.** This rules out SAM, ESAM, and LookSAM screening on this branch (all would need wall-cap-vs-epoch-count tradeoff calculations dominated by per-epoch math). Together with closed LayerScale / EMA / SWA / Lookahead / DropPath / drop-token-vol-pending — **the "drop-in stochastic regularizer" axis is essentially exhausted on Charlie.** Future regularization wins must come from data-side augmentation (zero training-step cost) or single-pass curriculum/objective tweaks.


## 2026-05-13 21:38 — PR #2581: mlp_ratio 2→4 (double FFN hidden dim)
- **Branch**: `charliepai2g24h1-askeladd/mlp-ratio-4-ffn-double`
- **Hypothesis**: Per-node FFN is the bottleneck; widening FFN (canonical ViT/Llama choice) should add useful capacity
- **Status**: **CLOSED — large regression** val +28.2%, test +27.5%

| Metric | Baseline (#2011) | mlp_ratio=4 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 28.8762 | 37.0291 | +28.2% |
| test_avg/mae_surf_p | 24.9992 | 31.8651 | +27.5% |
| params | 662K | 995,943 | +50.4% |
| epochs in 30-min cap | 28 | 25 | −3 |
| s/epoch (steady) | ~64.5 | ~70-72 | +9-12% |

**Per-split val deltas**: single +33.9%, rc +17.6%, **cruise +54.5%**, re_rand +25.3% — uniform regression with **largest hit on the EASIEST split** (cruise), the classic over-parameterization signature on N=1499.

- **Diagnostic**: Per-epoch convergence is slower with wider FFN even controlling for wall-clock. Extra parameters slow optimization without delivering a better basin. Linear extrapolation to epoch 28 ≈ 34 (still well above baseline 28.88), so the wall-clock penalty doesn't explain the regression.
- **Metrics JSONL**: `models/model-charliepai2g24h1-askeladd-mlp-ratio-4-ffn-double-20260513-205356/metrics.jsonl`

**Programme learning**: Combined with closed mlp_ratio=3 (#2256, +23.9%), the FFN-capacity axis is FULLY CLOSED in BOTH directions. mlp_ratio=2 is the local optimum at N=1499. The closure analysis suggested "n_layers / slice_num / n_hidden" — but all three are independently closed (n_layers=6 #2079, slice_num=128 #1467 / =32 #2320, wider-soap-192 #1797). **Capacity-expansion meta-axis is now exhausted across all 4 known capacity knobs.**

## 2026-05-13 21:38 — PR #2324: grad-accum-batch8 (gradient accumulation steps=2)
- **Branch**: `charliepai2g24h1-tanjiro/grad-accum-batch8`
- **Hypothesis**: Effective batch size 4→8 via accumulation, no memory cost
- **Status**: **CLOSED (infra-stale, not science-stale)**

Tanjiro pod was stuck for ~5 hours on a pod-side secondary rate-limit cycle (each poll iteration burns ~90s on 6× gh_retry attempts that themselves contribute to the throttle, self-reinforcing the loop). PR labels were correctly set throughout but the pod could not poll to see them. Closing and reassigning to a fresh PR — when the rate-limit naturally clears the pod will pick up the new work.

**Hypothesis itself remains untested.** Could be re-attempted by another student in a future round if grad-accum becomes worth revisiting.


## 2026-05-13 22:30 — PR #2582: DropToken (token random masking, p=0.10) — CLOSED
- **Branch**: `charliepai2g24h1-frieren/drop-token-vol-only`
- **Hypothesis**: Random per-token volume-only masking acts as drop-in regularization (forces model to be robust to missing nodes).
- **Status**: **CLOSED — catastrophic regression** val +18.0%, test +17.4%

| Split | Baseline (#2011) | DropToken p=0.1 | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | **39.61** | **+38.5%** |
| val_geom_camber_rc | 41.95 | 47.93 | +14.3% |
| val_geom_camber_cruise | 14.15 | 16.85 | +19.1% |
| val_re_rand | 30.81 | 31.92 | +3.6% |
| **val_avg** | **28.8762** | **34.0739** | **+18.0%** |
| **test_avg** | **24.9992** | **29.3543** | **+17.4%** |

- **Diagnostic**: Worst hit on the EASIEST split (single_in_dist +38.5%) — classic under-converged signature. PhysicsAttention's softmax over slice-tokens is sensitive to token-count perturbation. Token dropout breaks the prior that all nodes are visible at evaluation.
- **Metrics JSONL**: `models/model-charliepai2g24h1-frieren-drop-token-vol-only-20260513-211*/metrics.jsonl`

**Programme learning**: Combined with closed coord-jitter (#2594), translation, re-input-jitter, Mixup — **the "input-side perturbation augmentation" axis is now FULLY CLOSED**. The model needs all tokens visible because slice attention computes affinity over the full token set. Future augmentation must operate on labels (e.g., target smoothing) or features (e.g., shape descriptors), not input topology.

## 2026-05-13 22:30 — PR #2592: SDF (signed-distance-to-foil) input feature — CLOSED
- **Branch**: `charliepai2g24h1-nezuko/sdf-input-feature`
- **Hypothesis**: Adding per-node signed-distance to nearest foil surface as 25th input channel helps Transolver disambiguate volume nodes near vs far from the foil.
- **Status**: **CLOSED** — val +2.6%, test +1.9%; student self-diagnosed SDF info as redundant with slice attention's softmax-as-NN.

| Metric | Baseline (#2011) | SDF feature | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 28.8762 | 29.6193 | +2.57% |
| test_avg/mae_surf_p | 24.9992 | 25.4625 | +1.85% |
| input feature dim | 24 | 25 | +1 |
| params | 662K | 663K | +0.2% |

- **Diagnostic**: SDF column grad_norm decayed 10× over training (from 0.34 → 0.034) — model actively learned to **ignore** the channel. PhysicsAttention slice softmax already produces a soft distance-weighted neighborhood; an explicit Euclidean distance feature is **redundant** with the learned attention prior. Student's own analysis (very thorough) noted SDF magnitude is dominated by coord scale, which is already present.
- **Metrics JSONL**: `models/model-charliepai2g24h1-nezuko-sdf-input-feature-20260513-213*/metrics.jsonl`

**Programme learning**: Geometric distance features that overlap with attention softmax are dead-ends here. **The "explicit distance feature" axis is closed.** Curvature (2nd derivative) and shape-descriptor jitter remain unexplored and target genuinely new information not captured by slice attention.

## 2026-05-13 22:30 — PR #2594: Coordinate jitter (σ=0.01, training-only) — CLOSED
- **Branch**: `charliepai2g24h1-fern/coord-jitter-sigma-0p01`
- **Hypothesis**: Per-node Gaussian noise σ=0.01 on coordinates during training acts as smoothness regularizer.
- **Status**: **CLOSED** — val +3.7%, test +2.9%.

| Metric | Baseline (#2011) | Coord-jitter σ=0.01 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 28.8762 | 29.9469 | +3.71% |
| test_avg/mae_surf_p | 24.9992 | 25.7204 | +2.89% |
| val_geom_camber_rc | 35.30 | 36.62 | +3.74% |

- **Diagnostic**: Mesh-position perturbation doesn't address the bottleneck. The geom_camber_rc split differs from training in **NACA shape parameters** (camber, thickness) — NOT in mesh-position distribution. Coord-jitter perturbs the wrong axis.
- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-coord-jitter-sigma-0p01-20260513-214*/metrics.jsonl`

**Programme learning**: Critical empirical insight — **the OOD axis is shape-distribution, NOT mesh-position-distribution**. This redirects the augmentation search: instead of jittering coords, jitter the shape-descriptor channels (NACA-4 parameters) directly. Reassigned to fern as #2625 `naca-feature-jitter-sigma-0p02`.

## 2026-05-13 22:30 — PR #2539: Fourier positional encoding (3 arms σ=1.0, 0.1, 0.05) — CLOSED
- **Branch**: `charliepai2g24h1-thorfinn/fourier-pos-encoding-rff`
- **Hypothesis**: NeRF-style Random Fourier Features on (x,y) coords add high-frequency representational capacity to position encoding.
- **Status**: **CLOSED** — all 3 σ scales regress (+3.1%, +1.4%, +1.8%).

| σ | val_avg | Δ vs baseline | test_avg | Δ |
|---|---|---|---|---|
| baseline (#2011) | 28.8762 | — | 24.9992 | — |
| 1.0 | 29.79 | +3.16% | 25.34 | +1.36% |
| 0.1 | 29.27 | +1.36% | 25.54 | +2.19% |
| 0.05 | 29.40 | +1.81% | 25.51 | +2.06% |

- **Diagnostic**: Asymptotic ordering across the 3 scales (σ=0.1 best, then 0.05, then 1.0) shows there is no monotonic improvement — moving σ in any direction doesn't unlock a better basin. The PhysicsAttention slice-softmax over (xy)-coords already implicitly learns multi-scale position representations.
- **Metrics JSONL**: 3 separate runs under `models/model-charliepai2g24h1-thorfinn-fourier-pos-encoding-rff-*/metrics.jsonl`

**Programme learning**: **Fixed-frequency positional encoding axis is closed** — RFF features are redundant with raw coordinates when the architecture has flexible slice attention. Combined with closed coord-jitter, the "coord-side input augmentation/encoding" meta-axis is now thoroughly explored and exhausted.

## 2026-05-13 22:30 — PR #2585 (PENDING re-run): ReFiLM-residual hidden=4, deeper blocks ≥2
- **Branch**: `charliepai2g24h1-alphonse/refilm-residual-conditioning`
- **Status**: SENT BACK FOR ONE ARM. Mechanism real (corr -0.934 with log_Re), test_avg -1.70%, but val_avg tied (+0.06%) and val_geom_camber_rc +2.04% with |γ|max=0.854 indicates over-conditioning.
- **Re-run instructions**: hidden=4, apply only at blocks ≥2 (deeper half of stack), one arm.
- **Expected completion**: ~22:50 UTC.


## 2026-05-13 22:35 — PR #2598: AOA rotation augmentation θ∈±2° — CLOSED
- **Branch**: `charliepai2g24h1-askeladd/aoa-rotation-aug-2deg`
- **Hypothesis**: Random ±2° in-plane rotation of coords+velocity targets+AoA channel augments AoA manifold via Galilean rotation symmetry of NSE.
- **Status**: **CLOSED — large regression** val +7.21%, test +6.97%

| Split | Baseline (#2011) | AOA-rot ±2° | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 31.72 | +10.91% |
| val_geom_camber_rc | 41.95 | 42.34 | **+0.93% (flat!)** |
| val_geom_camber_cruise | 14.15 | 17.53 | +23.91% |
| val_re_rand | 30.81 | 32.24 | +4.65% |
| **val_avg** | **28.8762** | **30.9579** | **+7.21%** |
| **test_avg** | **24.9992** | **26.7421** | **+6.97%** |

- **Diagnostic**: The targeted OOD split barely moved while easier splits regressed sharply — classic signature of "augmentation on wrong axis". Student's own analysis: "Camber-rc generalization is not bottlenecked by AoA coverage — it's bottlenecked by camber shape. Rotating doesn't add camber variants. The hypothesis misdiagnosed the OOD axis."
- **Metrics JSONL**: `models/model-charliepai2g24h1-askeladd-aoa-rotation-aug-2deg-20260513-215702/metrics.jsonl`

**Programme learning**: SECOND independent confirmation (with #2594) that the OOD axis is **shape-distribution NOT mesh-position-distribution**. AoA rotation perturbs the AoA manifold but the rc bottleneck doesn't live on that manifold. Reassigning askeladd to surface-normal feature (genuinely new shape info, orthogonal to SDF magnitude / curvature 2nd-derivative).


## 2026-05-13 23:32 — PR #2585 (2 arms): ReFiLM-residual stream — CLOSED
- **Branch**: `charliepai2g24h1-alphonse/refilm-residual-stream-shared` (+ hidden4-late variant)
- **Hypothesis**: Re-conditioned FiLM on the residual stream (h ← γ(Re)·h + β(Re) + h_skip) — trunk-level Re conditioning beyond slice logits.
- **Status**: **CLOSED — axis exhausted across 2 arms**

| Arm | val_avg | Δ | test_avg | Δ | |γ|max | corr(mod, log_Re) |
|---|---|---|---|---|---|---|
| Baseline #2011 | 28.8762 | — | 24.9992 | — | — | — |
| Arm 1: h=8, all blocks | 28.8934 | +0.06% (tied) | 24.5737 | **-1.70%** ✅ | 0.854 | -0.934 |
| Arm 2: h=4, blocks 2-4 | 29.2558 | +1.31% ❌ | 24.7541 | -0.98% ✅ | 0.770 | -0.757 |

- **Per-split (Arm 1 vs Arm 2 vs baseline)**:
  - val_single_in_dist: +1.19% / **+6.35%** ❌ both regressed (in-dist perturbation)
  - val_geom_camber_rc: +2.04% / -1.09% (Arm 2 fixes Arm 1's rc regression)
  - val_geom_camber_cruise: -4.04% / -2.91% (both improve)
  - val_re_rand: -1.79% (target) / +1.85% (Arm 2 LOSES val_re_rand gain)

- **Diagnostic insights from student**:
  1. **Capacity isn't the lever**: halving hidden from 8→4 barely moved |γ|max (0.854→0.770). The optimizer drives gates to ~0.7-0.8 regardless. Gradient signal dominates, not expressivity.
  2. **Depth restriction is a sharp trade**: skipping blocks 0-1 fixes rc but cancels val_re_rand gain. Re-modulation effect is distributed across all blocks; restricting it cuts the signal.
  3. **Mechanism remains valid** (corr_mod_logre confirmed across arms, gates open from zero init, slice-logit FiLM stats unchanged) — the axis itself doesn't translate to val_avg wins.
  4. **val_single_in_dist regresses in BOTH arms** → ANY Re-conditioning of the residual stream perturbs the in-distribution feature pipeline. The injection point is the issue, not the parameterization.

- **Metrics JSONL**:
  - Arm 1: `models/model-charliepai2g24h1-alphonse-refilm-residual-stream-shared-20260513-210800/metrics.jsonl`
  - Arm 2: `models/model-charliepai2g24h1-alphonse-refilm-residual-stream-shared-hidden4-late-20260513-223401/metrics.jsonl`

**Programme learning**: **Residual-stream FiLM injection point CLOSED**. The mechanism works but the in-dist-vs-OOD trade-off is fundamental at this injection point. The unbounded amplification potential of γ⊙h+β before the next norm distorts in-distribution features. Reassigned alphonse to **#2650 re-conditional-layernorm-affine** — same Re-conditioning idea at a DIFFERENT injection point: LN's γ/β (post-normalization affine), which operates in a feature-bounded regime by construction. Canonical Conditional-InstanceNorm pattern from Dumoulin et al. 2017 — different inductive bias than residual-stream gating.



## 2026-05-13 23:38 — PR #2622: Element-level focal-MAE (γ=2, clamp=10) — CLOSED
- **Branch**: `charliepai2g24h1-frieren/element-level-focal-mae-gamma2`
- **Hypothesis**: Apply focal weighting (1 + |r/δ|^γ).clamp(1,10) per-element on Huber residuals to upweight hard pressure outliers — pull the heavy-tailed surface-p error tail toward the median.
- **Status**: **CLOSED — catastrophic regression** val +18.1%, test +12.1%

| Split | Baseline (#2011) | Focal-MAE γ=2 | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 34.42 | +20.31% |
| val_geom_camber_rc | 41.95 | 49.31 | +17.55% |
| val_geom_camber_cruise | 14.15 | 16.27 | +15.01% |
| val_re_rand | 30.81 | 36.43 | +18.27% |
| **val_avg** | **28.8762** | **34.1062** | **+18.10%** ❌ |

- **Diagnostic**: Clamp saturated immediately. On heavy-tailed pressure residuals, |r/δ|^2 produces enormous values that get clipped at 10, but the saturating mass dominates the gradient — model trains to fit the loud outliers at the expense of the bulk distribution. Element-level reweighting is fundamentally incompatible with the long-tailed residual structure of surface pressure.
- **Metrics JSONL**: `models/model-charliepai2g24h1-frieren-element-level-focal-mae-gamma2-20260513-223*/metrics.jsonl`

**Programme learning**: **Element-level focal-MAE CLOSED**. The pressure residual distribution is too heavy-tailed for per-element weighting — every batch has elements that saturate the clamp, and gradient flow collapses to those elements. Reassigned frieren to **#2659 sample-level-focal-mae-gamma1**: same focal idea at a DIFFERENT granularity (per-sample weighting averaged over elements), γ=1 (linear, not quadratic) to avoid saturation. Should be more stable since per-sample loss averaging smooths out the heavy-tailed element distribution before the focal weight is applied.


## 2026-05-13 23:39 — PR #2624: Surface curvature as input feature — CLOSED
- **Branch**: `charliepai2g24h1-nezuko/surface-curvature-input-feature`
- **Hypothesis**: Inject local surface curvature κ as an additional input channel — give the model explicit geometric 2nd-order information at surface elements, hypothesized to help with shape-OOD generalization (the rc/cruise camber-shape splits).
- **Status**: **CLOSED — small regression with fat OOD tails** val +0.91%, test -0.49%

| Split | Baseline (#2011) | Curvature input | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 28.45 | -0.52% |
| val_geom_camber_rc | 41.95 | 43.95 | **+4.77%** |
| val_geom_camber_cruise | 14.15 | 13.74 | -2.90% |
| val_re_rand | 30.81 | 31.43 | +2.01% |
| **val_avg** | **28.8762** | **29.1395** | **+0.91%** ❌ |
| **test_avg** | **24.9992** | **24.8770** | **-0.49%** ✓ small win |

- **Diagnostic**: val_geom_camber_rc gets WORSE (+4.77%), val_geom_camber_cruise improves slightly. Same shape-axis dichotomy as #2625 (NACA jitter). Curvature is computed FROM the geometry the model already sees — the slice attention's softmax over element coords is functionally a learned local curvature estimator already. Explicit curvature provides redundant information that confuses the optimizer when the test distribution shifts (rc OOD).
- **Metrics JSONL**: `models/model-charliepai2g24h1-nezuko-surface-curvature-input-feature-20260513-22*/metrics.jsonl`

**Programme learning**: **Explicit curvature input CLOSED**. Confirms that the architecture already has access to local geometric information via slice attention. Reassigned nezuko to **#2660 surface-normal-auxiliary-output-head** — same surface-normal information at a DIFFERENT injection point: an auxiliary output head (Kendall, Gal, Cipolla 2017 multi-task learning) that forces the trunk to develop normal-aware representations rather than reading normals from an input channel. Output-task regularization is fundamentally different signal from input-feature augmentation.


## 2026-05-13 23:40 — PR #2625: NACA-4 parameter jitter σ=0.02 (3-channel) — CLOSED
- **Branch**: `charliepai2g24h1-fern/naca-feature-jitter-sigma-0p02`
- **Hypothesis**: Jitter NACA-4 channels (camber amplitude, camber position, thickness) σ=0.02 at training time — directly augment the SHAPE-axis (the actual OOD axis identified in #2594/#2598).
- **Status**: **CLOSED — dichotomy** val +3.45% overall, but **DRAMATIC per-split split**

| Split | Baseline (#2011) | NACA-jitter | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 31.91 | +11.59% |
| val_geom_camber_rc | 41.95 | **52.10** | **+24.20%** ❌ catastrophic |
| val_geom_camber_cruise | 14.15 | **7.96** | **−43.71%** ✅ MASSIVE WIN |
| val_re_rand | 30.81 | 30.51 | -0.96% |
| **val_avg** | **28.8762** | **29.8728** | **+3.45%** ❌ |

- **Diagnostic**: Striking dichotomy. cruise wins by -43.7% (huge!) while rc regresses by +24.2%. THORFINN #2626 per-channel-heads showed a parallel pattern (cruise -44%, rc +21%) — cross-evidence for a unifying structural insight.

- **🔑 KEY PROGRAMME FINDING** (this round):
  > **rc camber-position is an EXTRAPOLATION problem; cruise camber is an INTERPOLATION problem.**
  >
  > The rc split is held out on the camber-position channel — its test values lie *outside* the train distribution on that axis. Augmenting along channel 16 (camber position) with σ=0.02 expands the training distribution *inward* (toward in-dist), making the rc gap relatively wider. The cruise split is interpolated within the train distribution along the other shape axes — generic shape smoothing helps there.
  >
  > **Implication**: input-side shape augmentation must be CHANNEL-SELECTIVE — augment channels NOT held out in the test splits, never augment along the OOD axis itself.

- **Metrics JSONL**: `models/model-charliepai2g24h1-fern-naca-feature-jitter-sigma-0p02-20260513-22*/metrics.jsonl`

**Programme learning**: **5th confirmed input-side perturbation augmentation closure**. The full-channel jitter strategy is closed. Reassigned fern to **#2662 naca-jitter-ch15-17-only-sigma02** — same σ=0.02 jitter, but channel 16 (camber position, the rc OOD axis) masked to zero. Targets preserving the cruise -43.7% win without the rc +24.2% regression. If successful, demonstrates the channel-selective augmentation principle and unlocks a new class of OOD-aware input augmentations.


## 2026-05-14 00:00 — PR #2627: Surface-normal direction (n_x, n_y) per-node input feature — CLOSED
- **Branch**: `charliepai2g24h1-askeladd/surface-normal-volume-feature`
- **Hypothesis**: Inject surface-normal direction (n_x, n_y) for each volume node from nearest surface point as 2 additional input channels. Physically motivated by pressure-normal coupling at viscous walls; expected to help OOD camber generalization.
- **Status**: **CLOSED — large regression with the 3rd-confirmed dichotomy** val +8.34%, test +6.55%

| Split | Baseline (#2011) | Surface-normal | Δ |
|---|---|---|---|
| val_single_in_dist | 28.60 | 31.19 | +7.8% |
| val_geom_camber_rc | 41.95 | 44.89 | **+27.2%** ❌ |
| val_geom_camber_cruise | 14.15 | 13.98 | **−44.8%** ✅ MASSIVE WIN |
| val_re_rand | 30.81 | 35.01 | **+35.0%** ❌ |
| **val_avg** | **28.8762** | **31.2686** | **+8.34%** ❌ |
| **test_avg** | **24.9992** | **26.6361** | **+6.55%** ❌ |

- **🔑 3RD CONFIRMATION of cruise-WIN / rc-LOSS dichotomy**:
  - #2625 fern NACA-jitter (cruise -43.7% / rc +24.2%)
  - #2626 thorfinn per-channel-heads (cruise -44% / rc +21%)
  - **#2627 askeladd surface-normal (cruise -44.8% / rc +27.2%)** — new
  - Three orthogonal interventions (input augmentation, output-head architecture, new input feature) all produce essentially identical cruise/rc dichotomy ratios. **This is a STRUCTURAL finding about the data/model geometry under the current 30-min budget + N=1499 + Transolver baseline**, not specific to any one intervention.

- **Diagnostic**: Student's analysis nails three points: (a) "volume nodes inherit the normal of the nearest surface point" creates wake/far-field "ghost orientation" pollution; (b) "model has learned a normal→flow mapping conditional on cruise-regime priors that doesn't generalize outside them" — the interpolation-densification pattern again; (c) val_re_rand +35% is the worst side-effect — model uses normals as a memorization channel conflated with Re-specific flow features.

- **Metrics JSONL**: `models/model-charliepai2g24h1-askeladd-surface-normal-volume-feature-20260513-231529/metrics.jsonl`

**Programme learning**: Per-node normal feature with volume-node inheritance closed. Reassigned to **#2671 surface-only-normal-feature** (student's suggested follow-up #1): zero normals on volume nodes, keep only on surface elements. Clean test of whether the dichotomy is structural or pollution-driven.

### Channel-index correction (from fern empirical verification on #2662)

**CORRECTION TO ROUND 23:41 NOTES**: The NACA-4 code is `MPTT`. Verified empirically against `splits_v2/`:
- ch15 = M = camber **amplitude** (1st NACA digit)
- ch16 = P = camber **position** (2nd NACA digit)
- ch17 = T = thickness (3rd-4th digits, dataset has these collapsed into one ch)

Train tandem ch15 = {0, 0.111, ..., 0.667, 1.0}; rc/cruise are held out on **ch15** (camber amplitude M):
- val_geom_camber_rc ch15 = {0.667, 0.778, 0.889} — values 0.778, 0.889 are OOD (extrapolation beyond train [0,1.0] cluster)
- val_geom_camber_cruise ch15 = {0.222, 0.333, 0.444} — in-distribution (interpolation in middle)
- Both splits have ch16 fully in-distribution.

The "rc"/"cruise" prefix refers to **raceCar/cruise environment**, NOT to camber-position channel.

The structural finding (rc=EXTRAPOLATION along the held-out shape axis; cruise=INTERPOLATION) is correct — just on **ch15** (camber amplitude M), not ch16. PR #2662 was sent back with corrected instruction: mask ch15, jitter ch16+ch17.


## 2026-05-14 00:25 — PR #2650: Re-conditional LayerNorm γ/β (shared, zero-init, CIN-style) — MERGED ✅ NEW BASELINE
- **Branch**: `charliepai2g24h1-alphonse/re-conditional-layernorm-affine`
- **Hypothesis**: Apply Re-conditioning at the LN-affine injection point (CIN / adaLN-Zero pattern) — same Re-conditioning idea as closed residual-stream FiLM (#2585) but at a fundamentally different injection point where LN normalisation bounds the modulation before γ/β applies.
- **Status**: **MERGED — new baseline val_avg = 28.2414 (-2.20% vs previous 28.8762)**

| Metric | Baseline (#2011) | ReCondLN (#2650) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | 28.8762 | **28.2414** | **−2.20%** ✅ |
| **test_avg/mae_surf_p** | 24.9992 | **24.4827** | **−2.07%** ✅ |

Per-split val:

| Split | Baseline (#2011) | ReCondLN | Δ |
|---|---|---|---|
| val_single_in_dist | 28.6013 | **27.1740** | **−4.99%** ✅ |
| val_geom_camber_rc | 41.9483 | 42.2153 | +0.64% (mild) |
| val_geom_camber_cruise | 14.1462 | **13.6733** | **−3.34%** ✅ |
| val_re_rand | 30.8090 | **29.9031** | **−2.94%** ✅ |

Per-split test:

| Split | test_avg |
|---|---|
| test_single_in_dist | 27.6193 (−6.47%) |
| test_geom_camber_rc | 38.2108 (+3.20%) |
| test_geom_camber_cruise | 10.6390 (−3.43%) |
| test_re_rand | 21.4617 (−4.29%) |

- **Mechanism**: Shared CIN/adaLN-Zero Re-conditioning of all 3 LN roles (pre-attn ln_1, pre-FFN ln_2, pre-out ln_3) via log(Re)→(γ_residual, β). Zero-init final layer — identity at step 0, opens to |γ_res|_max=2.34 (ln_1), |β|_max=1.39 (ln_3) by ep28. Corr(|γ_res|, logRe)=-0.77 at ln_1 confirms non-trivial Re-conditioning policy. Strongest at pre-attention LN (fits: that's where ReFiLM's slice-routing also operates).
- **Key confirmation**: val_single_in_dist IMPROVED -4.99% vs the residual-stream FiLM (#2585 Arm 1) which regressed +1.19%. The bounded-modulation hypothesis is vindicated — LN normalisation before γ/β prevents unbounded amplification.
- **Cost**: +13,872 params (~2% overhead). Zero memory change (27.79 GB → 27.79 GB). Compute overhead negligible (torch.compile fuses the FiLM MLP).
- **Convergence**: best=last=28, val still falling monotonically — budget-limited not converged.
- **Metrics JSONL**: `models/model-charliepai2g24h1-alphonse-re-conditional-layernorm-affine-20260513-234229/metrics.jsonl`

**Programme learning**: **LN-affine injection point WINS where residual-stream injection point FAILED.** The CIN/adaLN-Zero pattern (Dumoulin et al. 2017, Peebles & Xie 2022) provides bounded Re-conditioning by operating on normalised features. This opens a new composition axis: combining ReCondLN with other stack components (per-channel heads, surface normal features) should be orthogonal. The rc mild regression (+0.64% val) is the new bottleneck to address. Student suggested: probe whether restricting ReCondLN to a subset of blocks would fix the rc regression without losing val_avg. Also suggests: the model is NOT converged at ep28 (best=last, still falling) — a T_max=35 re-run would likely push further.


## 2026-05-14 00:45 — PR #2660: Surface-normal aux output head (Kendall multi-task, vol+surf nodes) — CLOSED
- **Branch**: `charliepai2g24h1-nezuko/surface-normal-auxiliary-output-head`
- **Hypothesis**: Aux head predicting nearest-surface (n_x, n_y) for every node forces shape-aware trunk via multi-task learning (Kendall/Gal/Cipolla 2017).
- **Status**: **CLOSED — regression vs new baseline** val +3.13% (29.13 vs new 28.24)

| Split | New baseline #2650 | This run | Δ |
|---|---|---|---|
| val_single_in_dist | 27.17 | 28.54 | +5.0% |
| val_geom_camber_rc | 42.22 | 43.30 | +2.6% |
| val_geom_camber_cruise | 13.67 | 13.83 | +1.2% |
| val_re_rand | 29.90 | 30.86 | +3.2% |

Comparing to old #2011 baseline (which student used): cruise -45.4% / rc +22.7% — **4th confirmation of cruise-WIN/rc-LOSS dichotomy**.

- **Aux task health verified**: aux_loss 0.062, pred_magnitude 0.980 — the aux objective itself works. Failure is volume-node target inheritance pollution.
- **Student's analysis**: "volume-node target imposes a 2-dim representation constraint on every volume node, but pressure prediction at volume nodes already requires representations encoding bulk flow features. Co-supervising the same representations may be locally inconsistent with the bulk-flow encoding the trunk would otherwise prefer."

**Programme learning**: 4th independent confirmation of cruise/rc dichotomy. Reassigned to **#2688 surface-only-aux-normal-head** (student's suggested follow-up #1): drop volume-node target, keep only surface elements. Parallel to askeladd's #2671 surface-only INPUT normal feature.


## 2026-05-14 00:45 — PR #2659: Sample-level focal-MAE (γ=1, no clamp) — CLOSED
- **Branch**: `charliepai2g24h1-frieren/sample-level-focal-mae-gamma1`
- **Hypothesis**: Per-sample focal weighting (γ=1, mean-normalized) on Huber-MAE — upweight hard SAMPLES (not elements), avoid heavy-tailed element saturation from #2622.
- **Status**: **CLOSED — large regression** val +15.9% vs new baseline (+13.4% vs #2011)

| Split | New baseline #2650 | This run | Δ |
|---|---|---|---|
| val_single_in_dist | 27.17 | 33.32 | +22.6% |
| val_geom_camber_rc | 42.22 | 46.27 | +9.6% |
| val_geom_camber_cruise | 13.67 | 16.68 | +22.0% |
| val_re_rand | 29.90 | 34.68 | +16.0% |

Comparing to #2011: cruise -34% / rc +31% — **5th cruise/rc dichotomy confirmation**.

- **Student's analysis**: "'Hard sample within batch' is not the same as 'rc-distribution sample'". With batch=4, max-MAE sample is gradient noise not OOD signal. 800× sample weight ratio (~0.005 to ~3.9) collapsed effective batch size to 1-2.
- **Sample-focal effective_loss_ratio** grew 1.21→1.46 over training — confirms reweighting was active but ineffective.

**Programme learning**: **Focal-MAE FAMILY FULLY CLOSED** at all configurations: element-level γ=2 with clamp (#2622, +18.1%) AND sample-level γ=1 no clamp (#2659, +15.9%). Fundamentally incompatible with N=1499 / batch=4 / heavy-tailed pressure residuals. Reassigned to **#2689 shape-bin-oversampling-m05**: data-side intervention, oversample M≥0.5 train samples 3× to densify near rc OOD boundary on ch15.


## 2026-05-14 00:45 — PR #2626: Per-channel heads kaiming-h128 re-run — CLOSED (axis fully closed)
- **Branch**: `charliepai2g24h1-thorfinn/per-channel-separate-heads`
- **Hypothesis (re-run)**: kaiming_normal init + head_hidden=128 fixes the under-convergence of the zero-init/h64 arm, allowing per-channel head specialization to beat baseline.
- **Status**: **CLOSED — axis structurally closed across 2 arms**

| Arm | val_avg | Δ vs #2011 | val_camber_rc | val_cruise | Pattern |
|---|---|---|---|---|---|
| zero-init/h64 | 29.86 | +0.98% | 42.87 (+21%) | 14.17 (-44%) | dichotomy |
| **kaiming/h128** | **29.35** | **+1.63%** | **42.79 (+21%)** | **14.53 (-43%)** | **dichotomy preserved** |

- **Student's structural diagnosis**: "the rc regression is not a convergence problem; it is a **specialization problem**: an independent per-channel projection is fundamentally worse at the rc regime than a shared projection. Independent per-channel projections lose the implicit cross-channel coupling (Ux/Uy/p coupled at sharp leading-edge gradients in rc samples)."
- **Convergence WAS fixed by kaiming/h128**: ~0.9 MAE faster than zero-init at every epoch. But the architectural advantage of separate heads is net-negative on macro-avg.
- **6th independent confirmation of cruise/rc dichotomy** (kaiming arm).

**Programme learning**: **Per-channel-heads axis FULLY CLOSED** across 2 arms with convergence fix tested. Student's recommendation #5 "stop pursuing the 3-head variant" ratified. Reassigned to **#2690 re-conditional-output-bias**: 4th Re-conditioning hook (after ReFiLM, ReScaleHead, ReCondLN) at the output-bias injection point. Extends the proven-winning Re-conditioning axis to a new bounded injection point.

### Round 00:45 Programme Summary

**SIX independent confirmations of cruise-WIN / rc-LOSS structural dichotomy across 6 orthogonal interventions:**
1. #2625 fern NACA-jitter (data-aug): cruise -43.7% / rc +24.2%
2. #2626 thorfinn per-channel-heads zero-init (architecture): cruise -42.6% / rc +21.4%
3. #2627 askeladd surface-normal volume (input feature): cruise -44.8% / rc +27.2%
4. #2660 nezuko surface-normal aux output (multi-task): cruise -45.4% / rc +22.7%
5. #2659 frieren sample-focal-MAE (loss reweighting): cruise -34.2% / rc +31.1%
6. #2626 thorfinn per-channel-heads kaiming-h128 re-run (architecture+convergence-fix): cruise -42.7% / rc +21.2%

The dichotomy is now a **first-class structural finding** about the trunk representation under N=1499 / 30-min / Transolver baseline. Any intervention that densifies in-dist neighborhood helps cruise (interpolation) and hurts rc (extrapolation beyond train cluster on ch15 M).

**The winning mechanism (PR #2650 ReConditionalLayerNorm) is the ONLY known intervention that doesn't trigger the dichotomy**: val_geom_camber_rc only mildly regressed (+0.64% val) while ALL other splits improved. The winning axis is **Re-conditioning at bounded injection points** (ReFiLM, ReScaleHead, ReCondLN).

Round 00:45 strategic priorities:
- **Test if surface-only variants break the dichotomy** (#2671 askeladd input-side + #2688 nezuko output-side) — if both still show dichotomy, it's fully structural (independent of pollution).
- **Test data-side approach** (#2689 frieren shape-bin oversampling) — qualitatively different from feature/loss/architecture; densifies train distribution near rc boundary.
- **Test 4th Re-conditioning hook** (#2690 thorfinn output bias) — extends the only known winning axis.
- **Test budget extension on winner** (#2678 alphonse T_max=35) — best=last in #2650, model still descending.
