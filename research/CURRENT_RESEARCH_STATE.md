# SENPAI Research State

- **Date:** 2026-05-16 20:15
- **Launch:** willow-pai2i-48h-r1 (round 6 — triple-stack era; NEW programme best val=60.43)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 (NEW PROGRAMME BEST)** | **60.4338** | **57.4381** | PR #3995, W&B `insf46p8`, seed=0 | `--use_geglu` + β2=0.95 (hardcoded) + T_max=17 |
| h=128+SwiGLU+T_max=17 (prior best) | 62.1023 | 59.5529 | PR #3994, W&B `5q47ozlp`, seed=0 | 3-seed μ̂=63.06 ± 0.93 |
| h=128+GeGLU+T_max=15 | 65.3704 | 61.6819 | PR #3810, W&B `db8bp8i8`, seed=0 | Superseded |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | train.py default |

**Win threshold:** val < 60.43 (triple-stack seed=0). Seed-variance unknown (3-seed canonical not yet established).

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4089 | nezuko | SWA over final 4 cosine epochs of T_max=17 SwiGLU (no LR kick-out) | Training (~19:26 UTC start, ~19:56 expected) |
| NEW | thorfinn | Triple-stack seed=1 (3-seed canonical) | Assigning now |
| NEW | alphonse | Triple-stack seed=2 (3-seed canonical) | Assigning now |
| NEW | askeladd | β1=0.95 on triple-stack (compound momentum) | Assigning now |
| NEW | frieren | β2 fine scan {0.93, 0.97} on triple-stack | Assigning now |
| NEW | tanjiro | EMA model weights (decay=0.999) on triple-stack | Assigning now |
| NEW | edward | Lion optimizer on triple-stack | Assigning now |
| NEW | fern | mlp_ratio=3 on triple-stack (bigger FFN) | Assigning now |

## Key mechanistic findings this round

### Triple-stack decomposition (from PR #4032 ablation)

| Config | val_avg | Δ |
|---|---|---|
| T_max=17 SwiGLU μ̂ (3-seed) | 63.06 | — |
| T_max=17 + GeGLU (default β2=0.9) | 62.47 | −0.59 |
| T_max=17 + GeGLU + β2=0.95 (triple-stack) | 60.43 | **−2.04 additional** |

**β2=0.95 is the dominant lever in the triple-stack** — not the activation swap. GeGLU alone gives marginal improvement; β2=0.95 stacked on top delivers most of the gain. This suggests the optimization regime (slower squared-gradient EMA for better late-training precision) is the key unlock at our scale/budget.

### Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null (cumulative shrinkage only ~3%) |
| LR=1e-3 under T_max=15 | Divergence |
| Head+embed LR boost (1.5–2.5×) | All null or worse |
| T_max < 17 | Confirmed suboptimal (PyTorch Gotcha #3) |
| RMSNorm (vs LayerNorm) | −5.18 val regression |
| slice_num=128 (2× attention) | −10.92 val regression |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail (T_cosine=10) | Regression vs baseline (but SWA mechanism real, re-testing with corrected budget) |

### SWA mechanism (PR #3644 nezuko)

SWA is **validated real on TandemFoilSet** under SwiGLU:
- swa_tail beat tail_best by −3.52 val (~2σ), **−6.76 test (~3.8σ)**
- Test gains > val gains → SWA preferentially fixes OOD distributions
- Failure mode: cosine-budget mismatch (LR kick-out from constant tail). Fix: SWA over final cosine epochs (PR #4089 nezuko, running).

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds (#3934)
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()` (#3993)
3. **T_max must equal total_epochs** — T_max<total causes hard-zero LR before training ends (#3994)

## Next research directions

### Priority 1 (triple-stack seed confirmation)

The triple-stack baseline is single-seed. σ̂ unknown. Win threshold = val<60.43 is wobbly without 3-seed canonical.
- Assign seed=1 and seed=2 runs immediately.

### Priority 2 (optimizer improvements on triple-stack)

Given that β2=0.95 was the dominant win, more optimizer exploration is warranted:
- **β1=0.95**: slower first-moment EMA — companion to β2 win; may or may not compound
- **β2 fine scan**: β2 ∈ {0.93, 0.97, 0.98} — check if 0.95 is optimal
- **EMA of weights**: continuous EMA at decay=0.999 — a complementary form of basin averaging
- **Lion optimizer**: fixed-LR-sign-based, often outperforms AdamW on smaller batches

### Priority 3 (architecture on triple-stack baseline)

- **mlp_ratio=3**: more FFN capacity (GeGLU hidden goes from 171 → 256 units)
- **SWA on triple-stack**: stack validated SWA mechanism (from #4089) onto triple-stack

### Priority 4 (speculative, if above plateau)

- Physics-informed loss (continuity equation soft constraint)
- SAM (sharpness-aware minimization, 2× compute)
- Layer-wise LR decay across transformer layers
- Cross-slice attention (between-slice heads)
