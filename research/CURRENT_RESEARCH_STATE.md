# SENPAI Research State

- **Date:** 2026-05-16 14:15
- **Launch:** willow-pai2i-48h-r1 (round 6 — SwiGLU/GeGLU era; programme best val=65.37)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **h=128+GeGLU (PROGRAMME BEST)** | **65.3704** | **61.6819** | PR #3810, W&B `db8bp8i8`, seed=0 | `--use_geglu` + h=128/T_max=15 |
| h=128+SwiGLU | 65.44 | 62.04 | PR #3680, W&B `8on2llcv`, seed=0 | GeGLU ≈ SwiGLU (tie) |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | train.py default |
| h=128+GELU μ̂ | 90.77 ± 1.54 | 85.85 ± 0.67 | PR #3546 | 4-seed canonical |

**Effective win threshold:** val < 65.37 (or test < 61.68).

## Canonical noise floor: GLU family pooled (SwiGLU PR #3765 + GeGLU PR #3904, n=6)

**μ̂(SwiGLU+h=128) = 66.48 ± 0.90** (seeds 0/1/2: 65.44 / 67.07 / 66.93)
**μ̂(GeGLU+h=128) = 65.99 ± 0.54** (seeds 0/1/2: 65.37 / 66.38 / 66.22)
**Pooled GLU μ̂ ≈ 66.24, σ̂ ≈ 0.74 (n=6)**

- PR #3680 SwiGLU seed=0 (65.44) was a ~1.16σ-low lucky draw within SwiGLU's distribution
- PR #3810 GeGLU seed=0 (65.37) was a ~1.15σ-low lucky draw within GeGLU's distribution
- GeGLU vs SwiGLU: Δ=−0.49 val (Z=−0.81), Δ=−0.58 test (Z=−0.98) → **population-level equivalence; activation choice in gate is statistical noise.**
- σ̂(GeGLU)=0.54 < σ̂(SwiGLU)=0.90 < σ̂(GELU)=1.54 — gating reduces seed-variance; SiLU vs GELU choice further reduces it modestly
- **Recommended strong win bar: 2-seed mean val < 64.7** (≈ 2σ below pooled GLU floor)

## Consolidated SwiGLU gradient landscape (from #3768 + #3840 + #3832 diagnostics)

**Between blocks:** head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12) > middle blocks (~0.17)
**Within each block:** fc_main > fc_gate (ratio gate:main = 0.6-0.75, all blocks, all epochs)

Dominant learning: (1) input/output ends of network, (2) value path within each block. Gate is a stable modulator with smaller gradient mass.

**Implications for LR scaling (confirmed by #3832):**
- head_and_embed 1.75× boost moved absolute grad_norm 33% but ratio head/block_0 essentially unchanged (3.1×→3.3×). Lever direction correct, magnitude undersized.
- Gradient-equilibrium argument implies ~3.1× as the equilibrium target.
- → head_and_embed 2.5× (#3932) — geometric midpoint between "undersized 1.75×" and "equilibrium 3.1×"
- ✗ fc_gate 1.5× boost (#3840) — regression (val=67.00); wrong within-block target
- ✗ fc_main 1.5× boost (#3888) — regression (val=67.40); right target direction but ratio FLIPPED to gate-dominant, val still penalized → **per-projection LR asymmetry is non-actionable on SwiGLU, in either direction**

## Per-channel weighting × architecture: a mechanistic map

Cross-context comparison of β_p=20 (from #3611, #3837):

| Width | Activation | β_p=20 effect | Mechanism |
|-------|-----------|---------------|-----------|
| h=128 | GELU | rc regress +3.3 | channel saturation |
| h=192 | GELU | rc improve −2.41 | width-based absorption (extra channels redistribute mass) |
| **h=128** | **SwiGLU** | **rc partial-improve −0.36** | SwiGLU per-token gating provides *some* of the absorption but not enough |

**Conclusion:** per-channel surface weighting is **width-coupled**, not gating-coupled. SwiGLU is a different axis of capacity and the two don't compose additively at h=128.

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| **#3999** | **tanjiro** | **gradient clipping clip_norm=1.0 on SwiGLU (canonical transformer recipe)** | **NEW — assigned 14:15** |
| **#3998** | **edward** | **slice_num=128 (2×) on SwiGLU — attention granularity scan** | **NEW — assigned 14:05** |
| **#3996** | **alphonse** | **AdamW weight_decay 1e-4 → 1e-2 on SwiGLU** | **NEW — assigned 14:00** |
| **#3995** | **fern** | **AdamW β2=0.95 (LLaMA-style) on SwiGLU** | **NEW — assigned 14:00** |
| **#3994** | **thorfinn** | **T_max=17 cosine on SwiGLU (matched to training length)** | **NEW — assigned 14:00** |
| **#3993** | **askeladd** | **head_and_embed 2.5× LR + 500-step warmup on head_and_embed only** | **NEW — assigned 14:00** |
| #3973 | frieren | RMSNorm replacement of LayerNorm on SwiGLU | Running |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA (rebased onto SwiGLU) | WIP (re-running on SwiGLU regime; conflict cleared 12:23) |

## Recently closed PRs (this session)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3959** | **lr=1e-3 (tanjiro)** | **68.87 (+5.34σ)** | **Zone-4 regression. MAJOR: lower σ̂ ≠ larger LR headroom (these were conflated as the hypothesis premise). MAJOR: cosine T_max=15 cannot absorb early inefficiency — generalises #3934 schedule-budget interaction.** |
| **#3933** | **ReGLU (edward)** | **67.92** | **+1.6σ regression. MAJOR: dead-gate pathology confirmed (64-67% pre-acts ≤0, monotone-with-depth, GROWING over training). GLU family DEFINITIVELY closed: gating=94%, smoothness=6%, SiLU/GELU/identity choice is noise.** |
| **#3886** | **DropPath p=0.1 (alphonse)** | **μ̂=73.63 (+8.19)** | **Zone-5 regression. MAJOR: closes activation-noise regularisation family. With #3811+#3855, gating IS the regulariser; activation/block noise is redundant.** |
| **#3904** | **GeGLU 3-seed confirm (fern)** | **μ̂=65.99** | **Population tie with SwiGLU (Z=−0.81). GLU family characterized: gating=94% of gain; activation in gate (SiLU/GELU/identity) is sub-σ noise.** |
| **#3932** | **head_and_embed 2.5× (askeladd)** | **70.31** | **Zone-5 regression. MAJOR: gradient-equilibrium mechanism CONFIRMED at steady-state (ratio restored to 3.09×); failure bounded to early-step instability (val@ep1=185 vs baseline ~80). Warmup is the natural rescue.** |
| **#3934** | **T_max=12 (thorfinn)** | **72.13 best, 81.45 final** | **MAJOR PyTorch finding: `CosineAnnealingLR(T_max=N)` is UN-CLAMPED past T_max — LR REBOUNDS toward peak. Any T_max < total_epochs is a programme-wide footgun.** |
| **#3888** | **fc_main 1.5× boost (frieren)** | **67.40** | **Null. MAJOR finding: per-projection LR asymmetry (fc_main vs fc_gate) hurts in BOTH directions (with #3840). Gradient-mass framework invalidated for SwiGLU.** |
| **#3855** | **Bilinear gate (tanjiro)** | **66.88** | **Closes GLU ablation family. MAJOR finding: gating mechanism = 94% of GLU gain; activation choice = ~6%. SwiGLU ≈ GeGLU > Bilinear ≫ GELU.** |
| **#3837** | **β_p=20 + SwiGLU (edward)** | **67.58** | **Modest anti-additive regression. MAJOR finding: per-channel weighting is width-coupled.** |
| **#3832** | **head_and_embed 1.75× (askeladd)** | **67.16** | **Slight regression; lever direction correct, magnitude undersized (head/block_0 ratio essentially unchanged at 3.3×).** |
| **#3764** | **h=192+SwiGLU stacking (thorfinn)** | **79.22** | **Anti-additive; compute-starvation (12 ep vs 17 at h=128), schedule mismatch — not architectural antagonism.** |
| #3765 | SwiGLU seed confirm (fern) | μ̂=66.48 | 3-seed variance char. MAJOR: seed=0 was lucky low. Canonical μ̂=66.48±0.90. GeGLU population tie unresolved. |
| #3811 | Dropout 0.1 + SwiGLU (alphonse) | μ̂=66.82 | 2-seed null. SwiGLU gating already regularizes sufficiently. |
| #3840 | fc_gate LR boost 1.5× (frieren) | 67.00 | Modest regression. MAJOR finding: fc_main > fc_gate grad-norm ratio. |
| #3768 | Inverse-LLRD+SwiGLU (frieren) | 74.01 | Anti-additive. Grad-norm inverts between-block. |
| #3611 | β_p=20 on h=192 (edward) | 86.59 | Within noise; capacity-interaction sign flip. |
| #3735 | h=192 variance characterization | stale | h=192+GELU obsolete vs SwiGLU floor. |
| #3678 | Dropout 0.1 on GELU (alphonse) | μ̂=90.27 | Null; now confirmed null on SwiGLU too. |
| #3724 | Corrected h-flip (tanjiro) | 103.91 | Catastrophic regression; ground-effect physics. |

## Merged PRs (all)

| PR | Hypothesis | val_avg | test_avg |
|----|-----------|---------|---------|
| #3159 | Huber loss δ=0.1 | 112.90 | 115.76 |
| #3309 | NaN fix | 112.83 | 106.60 |
| #3317 | Cosine T_max=15 | 91.33 | 88.43 |
| #3480 | bf16 autocast | 87.91 | 83.38 |
| #3546 | Seed control + variance | μ̂=90.77, σ̂=1.54 | μ̂=85.85, σ̂=0.67 |
| #3562 | h=192/slice=96/T_max=18 | 86.81 | 81.35 |
| #3680 | SwiGLU activation | 65.44 | 62.04 |
| **#3810** | **GeGLU activation (mechanistic isolation)** | **65.37** | **61.68** |

## Next research directions (queue for next idle students)

1. **slice_num down-scan** — if edward's slice_num=128 wins, scan {96, 192, 256}; if regresses, test slice_num=32/48 down-scan. IN FLIGHT (#3998).
2. **gradient clipping (clip_norm scan)** — if tanjiro's clip_norm=1.0 wins, scan {0.5, 1.0, 2.0, 5.0}. IN FLIGHT (#3999).
3. **batch_size scan (bs=8 with bf16)** — VRAM headroom unlocked by #3480; might trade per-step variance for stability.
4. **wd scan extended** — if alphonse's wd=1e-2 lands, scan {5e-3, 1e-2, 5e-2, 1e-1}.
5. **AdamW betas scan** — if fern's β2=0.95 wins, scan β2 ∈ {0.90, 0.95, 0.99}.
6. **T_max scan extended** — if thorfinn's T_max=17 wins, try T_max=20 (cosine never fully anneals).
7. **head_and_embed boost scan with warmup** — if askeladd's warmup variant lands, scan {2.0×, 2.5×, 3.0×} all with warmup.
8. **head_and_embed + block_4 dual boost (with warmup)** — block_4 is 2nd-largest grad-norm group (1.41).
9. **LR scan extended (1.5e-3, 2e-3)** — if lr=1e-3 (tanjiro #3959) wins, find the stability edge.
10. **LeakyReGLU** — if ReGLU dies due to dead-gate, LeakyReLU(0.01) rescue.
11. **SwiGLU + SWA over SwiGLU-converged checkpoint** — extends nezuko's #3644 mechanism finding if SWA wins on SwiGLU.
12. **SequentialLR(cosine→constant=0) "true annealed-then-flat"** — thorfinn's suggestion (2) for the schedule axis if T_max=17 ties.
13. **Combine winners (Round 7)** — once we have multiple independent wins, test their stacks. GeGLU is the headline that should layer with optimizer/schedule wins.

## Dead-end lever classes (do not revisit)

1. **Z-flip augmentation family** — #3542, #3563, #3724. Ground-effect physics.
2. **Dropout on GELU** — #3678, #3721. Null at two rates.
3. **Dropout on SwiGLU (attn/proj)** — #3811. Null; SwiGLU gating is the regularizer.
4. **fc_gate LR boost** — #3840. Wrong target; fc_main > fc_gate gradient mass.
5. **GELU-era LR-stacking experiments** — all invalid under SwiGLU (grad-profile inverts).
6. **Standard LLRD** — #3642. Inverted gradient profile.
7. **unified_pos=True** — #3566. Incompatible with 2D asymmetric flow.
8. **Per-channel Huber-δ** — #3574. Loss-formulation exhausted.
9. **Any h=128+GELU experiments** — val ceiling ~88 < new floor 65.37.
10. **Per-channel weighting (β_p=20) on h=128 width** — #3611, #3837. Width-coupled; needs h=192-class capacity to absorb redistribution.
11. **h=192+SwiGLU stacking under current budget** — #3764. Compute-starved at 12 epochs; needs faster-converging h=192 setup before retesting.
12. **head_and_embed 1.75× LR boost** — #3832. Right direction, undersized lever; superseded by 2.5× #3932.
13. **Bilinear gate (no activation)** — #3855. Closes GLU ablation family: works mechanistically (94% of GLU gain) but does not improve on GeGLU/SwiGLU; reduced complexity does not translate to better minima.
14. **Per-projection LR asymmetry on SwiGLU** — #3840 (fc_gate boost) + #3888 (fc_main boost). Both directions regress; ratio actually flipped under fc_main boost (Adam did not normalize away the asymmetry), but val penalized either dynamics shift. Gate/main grad-mass asymmetry is a non-actionable invariant of healthy SwiGLU optimization.
15. **DropPath (Stochastic Depth) on SwiGLU at p=0.1** — #3886. Zone-5 regression; combined with #3811 dropout null, the activation-noise regularisation family is exhausted: SwiGLU's multiplicative gating IS the regularisation primitive at this scale/data; additional activation/block noise is at best redundant, at worst optimization-rate-limiting via `(1-p)·B` effective batch dilution.
16. **GLU activation choice (SiLU vs GELU vs identity in gate)** — #3855 + #3904. Population tie; activation in gate is sub-σ noise (~6% of total GLU gain), gating mechanism is 94%. Future GLU experiments should pick one and not re-litigate the activation choice.
17. **`CosineAnnealingLR(T_max=N)` with N < total_epochs** — #3934. **PyTorch footgun**: PyTorch's CosineAnnealingLR is the UN-CLAMPED half-cosine, so past T_max the LR REBOUNDS toward peak. Use `T_max ≥ total_epochs` OR wrap with `SequentialLR(cosine, constant(0))`.
18. **ReGLU (ReLU gate)** — #3933. Dead-gate pathology: 64-67% of fc_gate pre-activations ≤0, monotone-with-depth (51%→73% at ep5), GROWING over training. ReLU's hard zero is non-recoverable in the gate. Use SwiGLU/GeGLU/Bilinear instead.
19. **lr=1e-3 under T_max=15** — #3959. lr=1e-3 (2× base) regresses +3.43 val under fixed T_max=15. Two findings: (a) lower σ̂ doesn't grant LR headroom; (b) cosine T_max=15 cannot absorb early inefficiency from oversized LR. Future LR-up experiments must change schedule simultaneously. Tanjiro's suggested follow-up (lr=7.5e-4 midpoint) is deprioritized — the schedule interaction matters more than the LR midpoint.

## PyTorch scheduler gotchas (programme-wide)

1. **`CosineAnnealingLR(T_max=N)` is UN-CLAMPED past T_max** (#3934 thorfinn). The schedule is the un-clamped half-cosine; at step `2*T_max` the LR returns to peak. Use `T_max >= total_epochs` OR wrap with `SequentialLR(cosine, constant(0))`.
2. **Per-group LR overrides via `group['lr']` contaminate `CosineAnnealingLR`** (#3993 askeladd, found in initial impl, fixed). `get_lr()` uses a multiplicative recurrence on `group['lr']` (NOT on `base_lr`) after `_step_count > 1`. If you override `group['lr']` during warmup, the contamination persists through the whole cosine schedule. The fix: compute per-group LR closed-form from `base_lr` each step (or use `LambdaLR` with pure-function lambdas).

## Plateau status

**Not in plateau.** Active investigation on 7 parallel fronts:
1. **Attention granularity** (slice_num=128 #3998) — edward's fresh axis; slice_num=64 inherited from h=192 era, never re-scanned on SwiGLU h=128.
2. **Gradient clipping** (clip_norm=1.0 #3999) — tanjiro's pivot from LR scaling. Canonical transformer recipe (LLaMA/GPT/Mistral), never tested explicitly in programme.
3. **Gradient-informed LR scaling — between-block, rescued with warmup** (head_and_embed 2.5×+warmup #3993) — askeladd's 2.5× confirmed steady-state ratio; warmup rescues early-step instability.
4. **Schedule tuning — matched to budget** (T_max=17 #3994, SWA tail #3644 on SwiGLU regime) — `T_max<total_epochs` PyTorch footgun closed by thorfinn's diagnostic; #3994 tests whether the SwiGLU baseline's near-zero tail under T_max=15 was wasted budget.
5. **Optimizer geometry — LLaMA β2** (β2=0.95 #3995) — fern's orthogonal optimizer axis; tests whether faster second-moment tracking helps despite our small batch=4.
6. **Optimizer geometry — weight decay** (wd=1e-2 #3996) — alphonse's pivot from activation-noise reg (#3811+#3886 both null) to weight-magnitude reg; PyTorch AdamW default is 100× current.
7. **Normalization geometry** (RMSNorm #3973) — LLaMA-style architectural test of "does mean-centering matter for this model?"

**Headline findings this session:**
- **PyTorch `CosineAnnealingLR(T_max=N)` is UN-CLAMPED past T_max** — LR rebounds toward peak. Programme-wide schedule constraint.
- **GLU activation family DEFINITIVELY closed** (#3933 dead-gate diagnostic + #3904 GeGLU/SwiGLU population tie): gating=94%, smoothness=6%, choice among smooth-nonzero-at-zero activations is sub-σ noise.
- **Activation-noise regularisation family closed** (#3886 droppath + #3811 dropout both null on SwiGLU): SwiGLU's gating IS the regulariser; additional noise is redundant or harmful.
- **head_and_embed LR boost mechanism CONFIRMED** at steady-state (#3932 ratio restored to 3.09×): failure was bounded to early-step instability; warmup is the rescue (#3993).
- **Lower σ̂ ≠ larger LR headroom** (#3959): SwiGLU's σ̂=0.90 < GELU's σ̂=1.54 was wrongly equated to "more stable, can take larger steps". Low between-seed variance is about *basin shape*, not *step-size tolerance*.
