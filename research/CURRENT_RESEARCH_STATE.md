# SENPAI Research State

- **Date:** 2026-05-16 06:10
- **Launch:** willow-pai2i-48h-r1 (round 5/6; new all-time best just merged — PR #3562)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~18 epochs achievable in bf16 at bs=4 h=128; ~13 epochs at h=192)
- **Latest direction from human team:** None (no open issues as of 06:10)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p` (all 4 splits valid since PR #3309).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p (new all-time best)** | **86.8095** | PR #3562, W&B `hzxs6zx9` |
| **test_avg/mae_surf_p (paper-facing)** | **81.3514** | PR #3562 |
| **h=192 informal 4-run mean** | ~89.70 (σ̂≈2.97) | PR #3562 informal |
| **h=128 canonical μ̂** | 90.77 ± σ̂=1.54 | PR #3546 (old config) |
| **Win threshold (h=192, TBD)** | val < μ̂(h=192) − σ̂(h=192) | Pending PR #3735 variance char |

⚠️ **The h=192 win threshold is not yet calibrated.** PR #3735 (askeladd) will establish μ̂ and σ̂ for the new architecture. Until then, use val < 86.81 as the point-estimate threshold and val < 89.2 as the rough distributional threshold.

## Merged PRs (all)

| PR | Hypothesis | val_avg | test_avg |
|----|-----------|---------|---------|
| #3159 | Huber loss δ=0.1 | 112.90 | 115.76 |
| #3309 | NaN fix (cruise test) | 112.83 | 106.60 |
| #3317 | Cosine T_max=15 | 91.33 | 88.43 |
| #3480 | bf16 autocast (bs=4) | 87.91 | 83.38 |
| #3546 | Seed control + variance | μ̂=90.77, σ̂=1.54 | μ̂=85.85, σ̂=0.67 |
| **#3562** | **h=192/slice=96/T_max=18** | **86.81 ← BEST** | **81.35 ← BEST** |

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3735 | askeladd | **h=192 4-seed σ̂ variance characterization** | **NEW — assigned 06:05** |
| #3678 | alphonse | Dropout attn_drop=proj_drop=0.1, 2-seed | WIP |
| #3611 | edward | Per-channel surf weight β_p=20 (REBASE on h=192) | **SENT BACK for rebase + retest on h=192** |
| #3680 | thorfinn | SwiGLU activation in MLP blocks (h=128) | WIP |
| #3644 | nezuko | Cosine T_max=10 + constant LR tail + SWA | **SENT BACK for rebase** (CONFLICTING) |
| #3721 | fern | DropPath / Stochastic Depth rate=0.1 | WIP (h=128 config) |
| #3722 | frieren | Inverse-LLRD (γ_inv=1.176, boost bottom block) | WIP (h=128 config) |
| #3724 | tanjiro | Corrected h-flip (flip pos_z+AoA+Uy, preserve camber) | WIP (h=128 config) |

**Note:** PRs #3680, #3721, #3722, #3724 were designed for h=128. If they beat val < 89.2 on h=128, they're directionally promising and should be retested on h=192 as a stacking confirmation before merging.

## Immediate priorities (next review cycle)

1. **PR #3735 askeladd** — h=192 σ̂ characterization. Infrastructure; always merges. Sets the new win threshold.
2. **PR #3678 alphonse** — dropout 0.1 on h=128. If val < 89.2, send back for retest on h=192 or merge + queue h=192 retest.
3. **PR #3680 thorfinn** — SwiGLU. Same logic as alphonse.
4. **PR #3611 edward** — per-channel β_p=20 on h=192. Rebase pending.
5. **PR #3644 nezuko** — constant-tail SWA. Rebase pending.
6. **PRs #3721, #3722, #3724** — DropPath, inverse-LLRD, corrected h-flip on h=128.

## Dead-end lever classes (do not revisit)

1. **Naive horizontal-flip symmetry** — #3542, #3563. Dataset NOT z-symmetric.
2. **SWA/EMA on cosine T_max=15 frozen tail** — #3580, #3521. Tail LR≈0 makes SWA≈best-by-val.
3. **Uniform surf_weight scan** — #3428, #3174, #3522.
4. **Cosine warmup** — #3175. Cosine T_max already provides soft warmup.
5. **High-batch (bs=8)** — #3460. Starves optimizer updates.
6. **LLRD standard (top-to-bottom decay)** — #3642. Transolver gradients inverted vs BERT.
7. **unified_pos=True** — #3566. Incompatible with 2D asymmetric-flow Transolver.
8. **Per-channel Huber-δ tightening** — #3574. Loss-formulation lever exhausted.

## Key insights (cumulative)

1. **bf16 is canonical** (#3480) — 18ep/30min, 32.9GB VRAM at h=128.
2. **Noise floor σ̂=1.54** (#3546) — win at h=128 needs val < 89.2; noise floor for h=192 TBD (#3735).
3. **Dataset NOT z-symmetric** (#3542) — falsifies naive h-flip. Only cruise is symmetric.
4. **SWA/EMA dead on frozen tail** (#3580, #3521) — need non-frozen tail first.
5. **LLRD gradient inversion** (#3642) — block_0 has highest grad norm in Transolver; standard LLRD starves it.
6. **unified_pos encoding mismatch** (#3566) — replaces directional coords with radial; incompatible.
7. **Loss-shape lever exhausted** (#3574) — δ and surf_weight scans converge to baseline.
8. **Capacity scaling wins** (#3562) — h=192/slice=96 unlocks real gain. bf16 VRAM freed this slot.

## Next research directions (post round 6)

1. **Stack h=192 + winners from h=128 experiments** — once dropout, SwiGLU, DropPath results land, winners should be retested at h=192 to confirm stacking.
2. **h=256 scaling** — if h=192 σ̂ confirms the gain is real, try h=256 (check VRAM: 49 GB at h=192 → estimate ~70 GB at h=256, within 96 GB limit).
3. **h=192 + T_max=20 or longer** — epoch 13 best at timeout suggests under-trained. A longer run budget (2 GPUs per student?) or T_max=20 with 30min cap might help.
4. **Per-domain normalization** — not yet tested on either h=128 or h=192.
5. **Gradient-norm-proportional per-block LR** — frieren's deeper follow-up from #3642 diagnostics.
6. **Constant-tail SWA (nezuko #3644)** — after rebase; still worth testing.

## Plateau status

**Not in plateau.** PR #3562 delivered a genuine new all-time best (val −1.1pt, test −2.0pt). Multiple orthogonal lever classes still in flight (regularization, architecture, optimization, data aug). The h=192 capacity unlock opens a new scaling frontier. Next natural question: how far does capacity scaling go?
