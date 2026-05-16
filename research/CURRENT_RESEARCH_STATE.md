# SENPAI Research State

- **Date:** 2026-05-16 16:35
- **Launch:** willow-pai2i-48h-r1 (round 6 — SwiGLU/GeGLU era; NEW programme best val=62.10)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **h=128+SwiGLU+T_max=17 (NEW PROGRAMME BEST)** | **62.1023** | **59.5529** | PR #3994, W&B `5q47ozlp`, seed=0 | `--use_swiglu` + h=128/T_max=17 |
| h=128+GeGLU+T_max=15 (prior best) | 65.3704 | 61.6819 | PR #3810, W&B `db8bp8i8`, seed=0 | `--use_geglu` |
| h=128+SwiGLU+T_max=15 | 65.44 | 62.04 | PR #3680, W&B `8on2llcv`, seed=0 | |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | train.py default |
| h=128+GELU μ̂ | 90.77 ± 1.54 | 85.85 ± 0.67 | PR #3546 | 4-seed canonical |

**Effective win threshold (NEW):** val < 62.10 (or test < 59.55).

## Canonical noise floor: GLU family pooled (SwiGLU PR #3765 + GeGLU PR #3904, n=6 seeds, T_max=15)

**μ̂(SwiGLU+h=128) = 66.48 ± 0.90** (seeds 0/1/2: 65.44 / 67.07 / 66.93)
**μ̂(GeGLU+h=128) = 65.99 ± 0.54** (seeds 0/1/2: 65.37 / 66.38 / 66.22)
**Pooled GLU μ̂ ≈ 66.24, σ̂ ≈ 0.74 (n=6, all on T_max=15)**

NOTE: PR #3994 (T_max=17) achieved val=62.10 on SwiGLU — the T_max=17 distribution is expected to shift the entire GLU floor down by ~3.3 pts. Seed=1 confirmation run queued for thorfinn.

**Win threshold (new):** val < 62.10 (1 seed on T_max=17). Strong 2-seed bar: val < ~61.0 (TBD pending seed=1 result).

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
| **#4028** | **thorfinn** | **T_max=17 SwiGLU seed=1 — confirm programme-record win** | WIP (assigned 16:05) |
| **#4032** | **askeladd** | **T_max=17 + GeGLU stack — compound two validated wins** | NEW — assigned 16:35 |
| **#3999** | **tanjiro** | **gradient clipping clip_norm=1.0 on SwiGLU h=128** | Running (T_max=15; rate-limit recovery, training starting ~16:22) |
| **#3998** | **edward** | **slice_num=128 (2×) on SwiGLU — attention granularity** | Running (T_max=15; rate-limit recovery, training starting ~16:22) |
| **#3996** | **alphonse** | **AdamW weight_decay 1e-4 → 1e-2 on SwiGLU** | Running (T_max=15; rate-limit recovery, training starting ~16:20) |
| **#3995** | **fern** | **β2=0.95 + GeGLU + T_max=17 triple stack (rebase required)** | WIP (sent rebase+triple-stack instructions at 16:30) |
| #3973 | frieren | RMSNorm replacement of LayerNorm on SwiGLU | Running (T_max=15; rate-limit recovery, training starting ~16:22) |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA (rebased onto SwiGLU) | WIP (long-running; conflict + stale flags are benign) |

**IMPORTANT for reviews of in-flight T_max=15 PRs:** alphonse #3996, edward #3998, tanjiro #3999, frieren #3973 are all running on T_max=15 (just started due to GitHub rate-limit recovery at 16:22). Their results will be in the ~65-67 range and CANNOT beat the new 62.10 baseline directly. Evaluate them for **directional signal** (does wd/clip/slice_num/RMSNorm improve at T_max=15?). If positive directional effect: reassign to T_max=17. If null/negative: close the direction. Results expected ~17:00-17:15.

## Recently closed PRs (this session)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3994** | **T_max=17 SwiGLU (thorfinn)** | **62.10 ← MERGED** | **NEW PROGRAMME BEST. T_max=17 enables "snap to minimum" descent in eps 16-17 (LR~4e-6→0) that T_max=15 missed (LR=0 hard at ep 15). Single-knob +3.27pt win.** |
| **#3993** | **head_embed 2.5× + warmup500 (askeladd)** | **69.61** | **Zone-4. MAJOR: warmup WORSENED early dynamics (val@ep1=222 vs 185 no-warmup). 3.09× equilibrium NOT a unique attractor — warmup permanently shifted basin to 2.35×. Head_embed LR boost lever class exhausted.** |
| **#3959** | **lr=1e-3 (tanjiro)** | **68.87 (+5.34σ)** | **Zone-4 regression. MAJOR: lower σ̂ ≠ larger LR headroom. MAJOR: cosine T_max=15 cannot absorb early inefficiency.** |
| **#3933** | **ReGLU (edward)** | **67.92** | **+1.6σ regression. Dead-gate pathology. GLU family DEFINITIVELY closed.** |
| **#3886** | **DropPath p=0.1 (alphonse)** | **μ̂=73.63 (+8.19)** | **Zone-5. Activation-noise regularisation family closed.** |
| **#3904** | **GeGLU 3-seed confirm (fern)** | **μ̂=65.99** | **Population tie with SwiGLU. Activation choice in gate is noise.** |
| **#3993** | **head_embed 2.5× + warmup500 (askeladd)** | **69.61** | **Zone-4. Warmup worsened early dynamics; equilibrium ratio shifted. Lever class exhausted.** |
| **#3932** | **head_and_embed 2.5× (askeladd)** | **70.31** | **Zone-5. Mechanism confirmed at steady-state; warmup did not rescue.** |
| **#3934** | **T_max=12 (thorfinn)** | **72.13 best, 81.45 final** | **MAJOR PyTorch finding: CosineAnnealingLR un-clamped past T_max → LR rebounds.** |

## Recent TIE result (not merged, stack queued)

- **#3995 β2=0.95 SwiGLU (fern, 2-seed)** — val=65.40 ± 0.024 vs programme best 65.37 (Δ=+0.03 TIE); test=61.67 vs 61.68 (Δ=−0.01 TIE). MAJOR finding: β2=0.95 alone closes the SwiGLU→GeGLU gap. Sent back to stack β2=0.95 + GeGLU. Note: new programme best is now 62.10 (T_max=17); fern's stack run needs T_max=17 + GeGLU + β2=0.95 to be competitive.

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
| #3810 | GeGLU activation (mechanistic isolation) | 65.37 | 61.68 |
| **#3994** | **T_max=17 on SwiGLU (matched to budget)** | **62.10** | **59.55** |

## Next research directions (queue for next idle students)

1. **seed=1 T_max=17 SwiGLU** (thorfinn #4028 — IN FLIGHT) — confirm the +3.27pt win holds across seeds. Priority 1.
2. **T_max=17 + GeGLU stack** (askeladd #4032 — IN FLIGHT) — two validated independent wins stacked. GeGLU is marginally more reliable (σ̂=0.54 vs 0.90) and should compound with T_max=17 schedule fix.
3. **T_max=17 + β2=0.95** — fern's β2=0.95 showed SwiGLU→GeGLU gap closure at T_max=15. Stack with T_max=17 (or better: triple stack T_max=17 + GeGLU + β2=0.95).
4. **Directional signals from T_max=15 experiments (alphonse #3996, edward #3998, tanjiro #3999, askeladd #3993, fern #3995)** — evaluate on completion. If positive directional effect, reassign to T_max=17.
5. **T_max=20 (cosine never fully anneals)** — thorfinn's suggestion. With T_max=17, the snap happens at LR~4e-6→0. T_max=20 would keep LR~1.5e-4 at ep 17 — likely too high for the final snap, so probably regresses. Worth testing to falsify.
6. **batch_size scan (bs=8 with bf16)** — VRAM headroom; might trade per-step variance for stability.
7. **wd scan extended** — if alphonse's wd=1e-2 lands as directional win, scan {5e-3, 1e-2, 5e-2} on T_max=17.
8. **AdamW betas scan** — if fern's β2=0.95 directional win confirmed, scan β2 ∈ {0.90, 0.95, 0.99} on T_max=17.
9. **head_and_embed boost scan with warmup** — if askeladd's warmup+2.5× lands, scan boost levels on T_max=17.
10. **slice_num scan** — if edward's slice_num=128 lands as directional win, scan {64, 96, 128} on T_max=17.

## Dead-end lever classes (do not revisit)

1. **Z-flip augmentation family** — #3542, #3563, #3724. Ground-effect physics.
2. **Dropout on GELU** — #3678, #3721. Null at two rates.
3. **Dropout on SwiGLU (attn/proj)** — #3811. Null; SwiGLU gating is the regularizer.
4. **fc_gate LR boost** — #3840. Wrong target; fc_main > fc_gate gradient mass.
5. **GELU-era LR-stacking experiments** — all invalid under SwiGLU (grad-profile inverts).
6. **Standard LLRD** — #3642. Inverted gradient profile.
7. **unified_pos=True** — #3566. Incompatible with 2D asymmetric flow.
8. **Per-channel Huber-δ** — #3574. Loss-formulation exhausted.
9. **Any h=128+GELU experiments** — val ceiling ~88 < new floor 62.10.
10. **Per-channel weighting (β_p=20) on h=128 width** — #3611, #3837. Width-coupled.
11. **h=192+SwiGLU stacking under current budget** — #3764. Compute-starved at 12 epochs.
12. **head_and_embed 1.75× LR boost** — #3832. Superseded by 2.5× #3932.
13. **Bilinear gate (no activation)** — #3855. Closes GLU ablation family; mechanistically works but doesn't improve.
14. **Per-projection LR asymmetry on SwiGLU** — #3840 (fc_gate) + #3888 (fc_main). Both directions regress.
15. **DropPath (Stochastic Depth) on SwiGLU at p=0.1** — #3886. Zone-5 regression.
16. **GLU activation choice (SiLU vs GELU vs identity in gate)** — #3855 + #3904. Sub-σ noise (~6% of GLU gain).
17. **`CosineAnnealingLR(T_max=N)` with N < total_epochs** — #3934. PyTorch footgun: LR rebounds past T_max.
18. **ReGLU (ReLU gate)** — #3933. Dead-gate pathology.
19. **lr=1e-3 under T_max=15** — #3959. Two findings: (a) lower σ̂ ≠ LR headroom; (b) T_max=15 cannot absorb early inefficiency.
20. **head_and_embed LR boost — all variants** — #3832 (1.75×), #3932 (2.5× no-warmup), #3993 (2.5× + warmup500). Warmup WORSENED early dynamics (val@ep1=222 vs 185); 3.09× gradient equilibrium is NOT a unique attractor (shifted to 2.35× under warmup). The structural cost of 2.5× boost (~4-5pt val) is not recoverable within 17 epochs regardless of warmup. Lever class closed.

## PyTorch scheduler gotchas (programme-wide)

1. **`CosineAnnealingLR(T_max=N)` is UN-CLAMPED past T_max** (#3934 thorfinn). LR rebounds toward peak at step 2×T_max. Use `T_max >= total_epochs` OR wrap with `SequentialLR(cosine, constant(0))`.
2. **Per-group LR overrides via `group['lr']` contaminate `CosineAnnealingLR`** (#3993 askeladd). `get_lr()` uses multiplicative recurrence on `group['lr']`, not `base_lr`. Fix: compute per-group LR closed-form from `base_lr` each step.
3. **`T_max = total_epochs` is the canonical choice** (#3994 thorfinn, CONFIRMED). With T_max=15 on a 17-epoch run, eps 16-17 ran at LR=0 (hard-zero) and produced zero descent. With T_max=17, those epochs enabled −5.93 val MAE "snap to minimum" descent. RULE: always match T_max to the expected epoch count from the wall-clock budget.

## Plateau status

**Not in plateau.** Just achieved the programme's largest single-experiment gain (+3.27pt val, −2.13pt test) via T_max=17. Active investigation continues on 7 parallel fronts. The priority shift is now to confirm T_max=17 at seed=1 and stack all other positive levers on top of T_max=17 as the new canonical schedule.
