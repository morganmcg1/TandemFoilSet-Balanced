# SENPAI Research State

- **Date:** 2026-05-16 05:45
- **Launch:** willow-pai2i-48h-r1 (round 5 active; 8/8 students in flight)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~18 epochs achievable in bf16 at bs=4)
- **Latest direction from human team:** None (no open issues as of 05:45)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p` (all 4 splits valid since PR #3309).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p (all-time best)** | **87.9105** | PR #3480, W&B `t00506x1` |
| **test_avg/mae_surf_p (paper-facing)** | **83.3782** | PR #3480 |
| **Canonical 4-seed mean μ̂** | 90.77 ± σ̂=1.54 | PR #3546 |
| **Win threshold (1σ)** | val < 89.2 | = μ̂-σ̂ |
| **Win threshold (2σ, strong)** | val < 87.7 | = μ̂-2σ̂ |

## Merged PRs (all)

| PR | Hypothesis | val_avg | test_avg |
|----|-----------|---------|---------|
| #3159 | Huber loss δ=0.1 | 112.90 | 115.76 |
| #3309 | NaN fix (cruise test) | 112.83 | 106.60 |
| #3317 | Cosine T_max=15 | 91.33 | 88.43 |
| #3480 | **bf16 autocast (bs=4)** | **87.91** | **83.38** |
| #3546 | **Seed control + variance** | μ̂=90.77, σ̂=1.54 | μ̂=85.85, σ̂=0.67 |

## URGENT — askeladd #3562 val=86.81 awaiting SENPAI-RESULT

Run `hzxs6zx9` (h=192/slice=96/T_max=18 under bf16) shows val_avg=86.81 — beats all-time best 87.91 and is ≈2.6σ below canonical mean. Two other completed runs same config: `sv85254i` val=91.06, `fqzs1zk1` val=92.97. Student nudged twice to post terminal SENPAI-RESULT. **Merge pending receipt of terminal marker.**

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3562 | askeladd | Wider h=192, slice=96, T_max=18 under bf16 | **URGENT: val=86.81 best-to-date, awaiting SENPAI-RESULT** |
| #3678 | alphonse | Dropout attn_drop=proj_drop=0.1, 2-seed | WIP (new assignment post-#3546) |
| #3611 | edward | Per-channel surf weight β_p=20 | WIP (nudged for status at 05:30) |
| #3680 | thorfinn | SwiGLU activation in MLP blocks | WIP |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA | **SENT BACK FOR REBASE** (CONFLICTING, needs rebase onto #3546) |
| #3721 | fern | **DropPath / Stochastic Depth (rate=0.1)** | **NEW — assigned 05:40** |
| #3722 | frieren | **Inverse-LLRD (γ_inv=1.176, boost bottom block)** | **NEW — assigned 05:42** |
| #3724 | tanjiro | **Corrected h-flip (flip pos_z+AoA+Uy, preserve NACA camber)** | **NEW — assigned 05:44** |

## Recently closed PRs (round 5)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| #3642 | LLRD γ=0.85 | 92.45 | Inverted gradient profile — bottom block needs highest LR in Transolver, not lowest. LLRD assumption empirically falsified. |
| #3566 | unified_pos=True | 102.63 | Encoding mismatch — flag swaps (x,z) for rotation-symmetric radial, discards directional info. +7.7σ regression. |
| #3574 | Per-channel Huber-δ δ_p=0.05 | 91.78 | Within noise, above μ̂. Loss-formulation lever class exhausted. |
| #3644 | Cosine T_max=10 + constant tail + SWA | — | Needs rebase (CONFLICTING). Sent back. |

## Dead-end lever classes (do not revisit)

1. **Naive horizontal-flip symmetry** — PRs #3542, #3563. Dataset NOT z-symmetric. Camber unsigned, AoA one-sided, pos_z one-sided for raceCar domain.
2. **SWA/EMA on cosine T_max=15 frozen tail** — PRs #3580, #3521. Tail LR≈0 makes SWA≈best-by-val. Only valid on a non-frozen tail schedule.
3. **Uniform surf_weight scan** — PRs #3428, #3174, #3522. All within σ or worse.
4. **Cosine warmup** — PR #3175. Cosine T_max=15 already provides soft warm-up; explicit warmup adds nothing.
5. **High-batch (bs=8)** — PR #3460. -39% optimizer updates per epoch starves convergence.
6. **LLRD standard (top-to-bottom decay)** — PR #3642. Transolver gradients are inverted vs BERT/RoBERTa. Block_0 has largest grad norm.
7. **unified_pos=True** — PR #3566. Incompatible with 2D asymmetric-flow Transolver (discards directional features).
8. **Per-channel Huber-δ tightening** — PR #3574. Full loss-formulation lever class exhausted.

## Key insights (cumulative)

1. **bf16 is a clean orthogonal win** (#3480) — canonical default, 18ep/30min, 32.9GB VRAM.
2. **Noise floor σ̂=1.54** (#3546) — baseline 87.91 is 1.86σ downward outlier. Meaningful win needs val < 89.2.
3. **Dataset NOT z-symmetric** (#3542) — falsifies naive h-flip. Only cruise is z-symmetric.
4. **SWA/EMA dead on frozen tail** (#3580, #3521) — constant-LR tail needed before averaging can help.
5. **LLRD gradient inversion** (#3642) — Transolver's block_0 has highest grad norm; standard LLRD starves it.
6. **unified_pos encoding mismatch** (#3566) — replaces directional coords with radial, incompatible with this 2D fork.
7. **Loss-shape lever exhausted** (#3574, #3305, #3428) — δ and surf_weight scans converge to baseline.

## Active hypotheses and expected outcomes

| PR | Student | Lever class | Expected val | Confidence |
|----|---------|------------|-------------|-----------|
| #3562 | askeladd | Capacity scaling (h=192/slice=96) | **86.81 (observed in W&B!)** | HIGH — already trained |
| #3678 | alphonse | Regularization (dropout 0.1) | 87–90 | Medium-high |
| #3680 | thorfinn | Architecture (SwiGLU FFN) | 88–90 | Medium |
| #3721 | fern | Regularization (DropPath 0.1) | 88–91 | Medium |
| #3722 | frieren | Optimization (inverse-LLRD) | 88–91 | Medium (gradient-evidence backed) |
| #3724 | tanjiro | Data aug (corrected h-flip) | 88–91 | Medium |
| #3611 | edward | Loss weighting (β_p=20) | 88–91 | Medium |
| #3644 | nezuko | Schedule+SWA (T_max=10+tail) | 87–91 | Pending rebase |

## Next research directions (post round 5)

1. **Stack if askeladd #3562 wins (h=192):** SwiGLU + DropPath on h=192 backbone — #3680 and #3721 already target h=128; winner can be re-tested on h=192.
2. **Per-domain normalization** — per-split input/output normalization statistics. Not yet tested.
3. **Hybrid positional encoding** — fern's suggested per-block injection with directional features preserved (the correct version of #3566).
4. **Cruise-only conditional corrected-flip** — if #3724 tanjiro's corrected-flip wins, add conditional application to cruise as a stacking candidate.
5. **β_p scan** — only if #3611 edward wins; scan β∈{15, 25, 30}.
6. **Constant-tail SWA (nezuko #3644)** — after rebase; if it wins, the non-frozen-tail insight unlocks EMA too.
7. **Gradient-norm-proportional LR** — frieren's follow-up from #3642 if inverse-LLRD wins.
8. **LAMB optimizer** — Adam-variant with per-layer normalization, specifically designed for cases where gradient norms vary widely across layers (exactly Transolver's condition).

## Plateau status

Not in plateau. Askeladd #3562 is a potential new all-time best (val=86.81 observed). 6 active orthogonal lever classes in flight across regularization, architecture, optimization, and data augmentation tiers. The round 5 close wave (3 regressions) has mapped the local dead ends and sharpened the remaining search space.
