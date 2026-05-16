# SENPAI Research State

- **Last updated:** 2026-05-16 03:35 (SwiGLU PR #3608 merged — new baseline 78.407; all in-flight PRs notified to rebase; round-9 assignments underway)
- **Most recent research direction from human researcher team:** none (no open issues — verified 03:20Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **78.407** (PR #3608 SwiGLU FFN)
- **GH rate-limit status:** ~2800/5000 remaining.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3536 | tanjiro | eta_min=1e-5 (compound retest) | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| #3607 | thorfinn | FFN dropout p=0.05 (SwiGLU) | WIP (sent back for rebase + p=0.05) | test −0.54% but val +0.63%; retest on SwiGLU |
| #3606 | nezuko | RFF σ=3 (lower freq) | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| #3639 | alphonse | EMA / Polyak α=0.999 | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| #3643 | askeladd | n_head=2 (head_dim=48) | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| #3645 | edward | surf_weight=10→5 (loss rebalance) | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| #3646 | fern | stochastic depth / drop-path p=0.1 | WIP (rebase needed) | Old baseline — must rebase onto SwiGLU |
| **(queued)** | frieren | SwiGLU mlp_ratio=2 (hidden_inner=192, +24% params) | pending PR creation | Follow-up to winning PR #3608 |

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=64→96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — 95.808
7. **PR #3608** (SwiGLU FFN, param-matched) — **78.407** (current baseline — MASSIVE WIN)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + n_head=4 + **SwiGLU FFN (hidden_inner=128, bias-free W1/V/W2)** + n_layers=5 + dropout=0.0 + surf_weight=10.

## SwiGLU win — program summary

The SwiGLU PR is the largest single-experiment improvement in the program: **−18.2% val, −20.1% test**. All 4 splits improved 15-26%.

Key insights from this result:
- **GELU FFN was the dominant bottleneck** — explains why 8+ rounds of capacity/regularization/schedule experiments made marginal gains. The parameterization itself was holding back the model.
- **Multiplicative gating unlocks adaptive feature selection** — especially for mesh geometry shifts (geom_cruise −26% val, −29% test). SwiGLU's `SiLU(W1·x) ⊙ (V·x)` gives each token a gate-selected subspace that GELU's single-path `Linear → GELU → Linear` cannot express.
- **mlp_ratio=4 failure (PR #3503) was NOT about capacity** — it was about single-path expansion parameterization. SwiGLU re-introduces 1.5× effective expansion at matched params and wins massively.
- **Retrospective**: if we had tested SwiGLU earlier, many rounds of tuning could have been skipped. Architectural changes at the FFN level are higher-leverage than hyperparameter sweeps on an underpowered FFN.

## Current research focus

### Tier 1 (highest priority — test SwiGLU + X compounds)
1. **Full mlp_ratio=2 SwiGLU** (hidden_inner=192, +24% params) — frieren queued. If matched-param version wins so decisively, does unconstrained version win more?
2. **eta_min=1e-5 on SwiGLU** — tanjiro rebase. In isolation on OLD baseline: 95.835. On SwiGLU: should be orthogonal and stack.
3. **SwiGLU + dropout p=0.05** — thorfinn rebase. Old result showed geom_cruise −6.3% test; p=0.05 should avoid slowing single_in_dist fit.

### Tier 2 (regularization + optimization axes on SwiGLU)
4. **EMA/Polyak α=0.999** — alphonse rebase. Orthogonal to FFN parameterization.
5. **Drop-path p=0.1** — fern rebase. Block-level stochastic regularization.
6. **surf_weight=5** — edward rebase. Loss rebalance for single_in_dist.

### Tier 3 (architecture probes on SwiGLU)
7. **n_head=2 (head_dim=48)** — askeladd rebase. Wider heads on SwiGLU stack.
8. **RFF σ=3** — nezuko rebase. Low-freq positional encoding.

## Open questions (on new SwiGLU baseline)
- Does full-size mlp_ratio=2 SwiGLU (hidden_inner=192) give further gains? (frieren queued)
- Does eta_min=1e-5 still compound orthogonally on SwiGLU? (tanjiro rebase)
- Does dropout p=0.05 help geom_cruise without hurting single_in_dist? (thorfinn rebase)
- Does EMA smooth the cosine tail on the SwiGLU model? (alphonse rebase)
- Does drop-path regularize productively on the new FFN? (fern rebase)
- Does surf_weight=5 help single_in_dist on the new baseline? (edward rebase)
- Does n_head=2 interact favorably with SwiGLU? (askeladd rebase)
- Does RFF σ=3 provide useful positional information on the SwiGLU model? (nezuko rebase)
- What is the theoretical floor? SwiGLU at 78.4 is 33% better than the original 116.6 Huber baseline. Total improvement to date: −32.8%.

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

## Next research directions (beyond round-9)
1. **GEGLU** — GELU-gated variant (same structure as SwiGLU, GELU instead of SiLU on gate). Head-to-head: is the win from gating per se, or SiLU specifically?
2. **SwiGLU + larger n_layers** — with SwiGLU unlocking capacity, the n_layers=6 closure on GELU (#3506) may not hold. Worth a re-test with n_layers=6 on SwiGLU.
3. **Attention dropout** — so far only FFN dropout tested; attention head dropout is orthogonal.
4. **Log-pressure target** — handles dynamic range across splits.
5. **Per-layer-class weight decay** — decouple OOD/in-dist trade.
6. **Data augmentation** — physics-aware flip or geometric perturbation.
