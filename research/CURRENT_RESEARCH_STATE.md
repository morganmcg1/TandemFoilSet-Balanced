# SENPAI Research State

- **Last updated:** 2026-05-16 05:35 (SwiGLU full mlp_ratio=2 PR #3654 merged — new baseline 75.578; all in-flight PRs notified; frieren assigned GEGLU ablation PR #3727)
- **Most recent research direction from human researcher team:** none (no open issues — verified 05:30Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **75.578** (PR #3654 SwiGLU hidden_inner=192)
- **GH rate-limit status:** ~2200/5000 remaining.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3536 | tanjiro | eta_min=1e-5 (compound retest on SwiGLU) | WIP | Notified new baseline 75.578 |
| #3607 | thorfinn | FFN dropout p=0.05 (SwiGLU, rebased) | WIP | Notified new baseline 75.578 |
| #3639 | alphonse | EMA / Polyak α=0.999 (SwiGLU) | WIP | Notified new baseline 75.578 |
| #3643 | askeladd | n_head=2 (head_dim=48, SwiGLU) | WIP | Notified new baseline 75.578 |
| #3645 | edward | surf_weight=10→5 (SwiGLU) | WIP | Notified new baseline 75.578 |
| #3646 | fern | stochastic depth DropPath p=0.1 (SwiGLU) | WIP | Notified new baseline 75.578 |
| #3655 | nezuko | RFF σ=3 + learnable-σ 2-arm (SwiGLU) | WIP (pod stuck in watchdog loop since 04:22Z) | Notified new baseline 75.578; pod has not committed results |
| #3727 | frieren | GEGLU ablation: SiLU→GELU gate at hidden_inner=192 | WIP (newly assigned) | Gate-function ablation on new baseline |

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=64→96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — 95.808
7. **PR #3608** (SwiGLU FFN, param-matched hidden_inner=128) — 78.407 (−18.2%)
8. **PR #3654** (SwiGLU full mlp_ratio=2, hidden_inner=192) — **75.578** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + n_head=4 + **SwiGLU FFN (SiLU gate, hidden_inner=192, bias-free W1/V/W2)** + n_layers=5 + dropout=0.0 + surf_weight=10.

Total improvement to date: 116.61 → 75.578 = **−35.2%** from original Huber baseline.

## SwiGLU capacity scaling — program summary

After the SwiGLU win (PR #3608, −18.2%), scaling FFN capacity within the gating parameterization continues to help:
- **hidden_inner=128** (param-matched): val 78.407 (PR #3608)
- **hidden_inner=192** (+50% FFN params): val 75.578 (PR #3654, −3.6%)

Key insight from #3654: **Gating does most of the work** — ~83% of the total SwiGLU-stack improvement came from gating alone. Extra capacity within a gated FFN sub-linearly increases overfitting risk (the gate can selectively use or suppress extra hidden units). In contrast, plain GELU+mlp_ratio=4 (#3503) failed because extra params with single-path expansion just memorized.

## Current research focus

### Tier 1 (gate-function ablation and high-value compounds on SwiGLU 192)
1. **GEGLU at hidden_inner=192** — frieren (#3727). Is the win from gating per se or SiLU specifically? Direct head-to-head SiLU vs GELU at matched hidden_inner=192.
2. **eta_min=1e-5 on SwiGLU** — tanjiro (#3536). In isolation on OLD baseline: 95.835. Should compound orthogonally on SwiGLU 192.
3. **SwiGLU + dropout p=0.05** — thorfinn (#3607). Rebase onto SwiGLU stack; test FFN dropout at p=0.05.

### Tier 2 (regularization + optimization axes)
4. **EMA/Polyak α=0.999** — alphonse (#3639). Orthogonal to FFN parameterization.
5. **Drop-path p=0.1** — fern (#3646). Block-level stochastic regularization.
6. **surf_weight=5** — edward (#3645). Loss rebalance for single_in_dist.

### Tier 3 (architecture probes)
7. **n_head=2 (head_dim=48)** — askeladd (#3643). Wider heads on SwiGLU stack.
8. **RFF σ=3** — nezuko (#3655). Low-freq positional encoding (pod stuck — monitoring).

## Open questions (on new SwiGLU 192 baseline)
- Is the SwiGLU win from gating or SiLU specifically? (frieren GEGLU, #3727)
- Does hidden_inner=256 continue the capacity scaling trend? (queued, not yet assigned)
- Does eta_min=1e-5 compound orthogonally on SwiGLU 192? (tanjiro #3536)
- Does dropout p=0.05 help geom_cruise without hurting single_in_dist? (thorfinn #3607)
- Does EMA smooth the cosine tail on the SwiGLU 192 model? (alphonse #3639)
- Does drop-path regularize productively on the new FFN? (fern #3646)
- Does surf_weight=5 help single_in_dist on the new baseline? (edward #3645)
- Does n_head=2 interact favorably with SwiGLU 192? (askeladd #3643)
- Does RFF σ=3 provide useful positional information? (nezuko #3655)
- What is the theoretical floor? 75.578 = −35.2% from original baseline. A full 14-epoch run of hidden_inner=192 might reach ~73-74.

## Closed/regressed (cumulative)
- #3579 alphonse lr=1e-3: +2.47%
- #3569 fern wd=5e-4: +1.30%
- #3564 edward n_layers=4: +2.71%
- #3535 askeladd n_head=8: +22.9%
- #3502 alphonse n_hidden=64: +9.20%
- #3503 edward mlp_ratio=4: +5.02% (was GELU; SwiGLU reframes this result)
- #3505 frieren per-channel pressure [1,1,2]: +2.85%
- #3506 thorfinn n_layers=6: +15.0%
- #3534 nezuko RFF σ=10: +2.89%
- #3453 edward T_max=10: +3.55%
- #3301 alphonse width-192: +8.53%
- #3304 frieren surf_weight=20: +4.12%
- #3302 askeladd depth-8: +1.53%
- #3223 thorfinn BF16+batch=8: +34%
- #3295 edward slice_num=128: +20%
- Round-1: #3205, #3179, #3183, #3214, #3216, #3220

## Next research directions (beyond current in-flight)
1. **hidden_inner=256** — probe whether capacity scaling saturates or continues. 128→192 gave −3.6%; 192→256 (+33% more) is the natural bracket. Wall-clock concern: ~160-165 s/epoch limits to 11-12 epochs at 30-min cap.
2. **n_layers=6 on SwiGLU 192** — #3506 failed catastrophically on GELU (−15%), but SwiGLU unlocked capacity. Re-test depth with the new parameterization.
3. **GEGLU result interpretation** — if GEGLU ≈ SwiGLU, gating dominates (simplified design space); if GEGLU < SwiGLU, SiLU's negative region matters for physics surrogates.
4. **eta_min=1e-5 + hidden_inner=192 compound** — if eta_min wins on SwiGLU 128, test compound on 192 (frieren's #3654 trajectory was still descending at E13; non-zero eta_min + 192 may recover lost E14).
5. **Attention dropout** — so far only FFN dropout tested; attention head dropout is orthogonal.
6. **Log-pressure target** — handles dynamic range across splits.
7. **Data augmentation** — physics-aware flip or geometric perturbation.
