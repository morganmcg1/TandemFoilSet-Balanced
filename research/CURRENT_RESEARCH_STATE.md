# SENPAI Research State

- **Last updated:** 2026-05-16 10:30 (round-11 triage: closed #3773 n_head=1, #3655 RFF 2-arm, #3607 FFN-dropout p=0.1 — all clear negatives. 3 fresh assignments queued.)
- **Most recent research direction from human researcher team:** none (no open issues — verified 09:28Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **70.925** (PR #3643 n_head=2 head_dim=48)
- **GH rate-limit status:** ~3000+/5000 remaining; window reset at 10:20 UTC; next reset 11:20 UTC.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3645 | edward | surf_weight=10→5 (SwiGLU) | WIP (stale) | Loss rebalance — needs nudge |
| #3727 | frieren | GEGLU at hidden_inner=192 | WIP (stale, retest) | Was −3.10% standalone win — needs nudge |
| #3849 | alphonse | RMSNorm replacing LayerNorm | WIP (newly assigned) | Budget-positive arch swap |
| #3851 | tanjiro | Warmup 2→1ep + cosine T_max=13 | WIP (newly assigned) | Schedule reallocation |
| #3853 | fern | AdamW β2 0.999→0.95 | WIP (newly assigned) | Optimizer hyper |
| (new) | askeladd | **Mish in SwiGLU gate** | WIP (to assign) | Gate-activation curvature |
| (new) | nezuko | **AdamW β1=0.9→0.95** | WIP (to assign) | Optimizer hyper, complements fern's β2 |
| (new) | thorfinn | **Huber β=1.0→0.5** | WIP (to assign) | Loss shape, first since baseline |

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — 100.811
4. **PR #3399** (slice_num=64→96) — 97.757
5. **PR #3377** (n_hidden=128→96) — 96.667
6. **PR #3314** (weight_decay=1e-4→3e-4) — 95.808
7. **PR #3608** (SwiGLU FFN, param-matched hidden_inner=128) — 78.407
8. **PR #3654** (SwiGLU full mlp_ratio=2, hidden_inner=192) — 75.578
9. **PR #3643** (n_head 4→2, head_dim 24→48) — **70.925** (current baseline)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4, betas=(0.9, 0.999)) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + **n_head=2 (head_dim=48)** + SwiGLU FFN (SiLU gate, hidden_inner=192, bias-free W1/V/W2) + n_layers=5 + LayerNorm + dropout=0.0 + surf_weight=10.

Total improvement to date: 116.61 → 70.925 = **−39.2%** from original Huber baseline.

## Key program insights

### Program finding #1: the 14-epoch / 30-min budget is the binding constraint
After 3 round-10 closures (LayerScale, hidden_inner=256, n_layers=6) all showed "val curve still descending at E14", pattern is unambiguous: any experiment that slows convergence per-step or per-epoch fails under the budget. Implication: future wins require **budget-respecting changes** — modifications that converge as fast or faster than baseline.

### Program finding #2: uniform regularization is decisively past its optimum
**5 textbook-magnitude uniform regularizers have failed** on the post-SwiGLU+n_head=2 stack:
- DropPath p=0.1: +24.6% (#3646)
- EMA α=0.999: +13.5% (#3639)
- attn-dropout p=0.1: +7.8% (#3774)
- **FFN dropout p=0.1: +6.2% (#3607 — round 11)**
- wd=5e-4: +1.3% (#3569)

The SwiGLU+n_head=2 model has substantially lower intrinsic overfitting risk than the GELU+n_head=4 baseline where these were originally evaluated. Mechanism: tighter train-val gap + slower convergence under noise injection. Asymmetric/structured regularization (depth-scheduled DropPath, per-layer wd) remains untested.

### Program finding #3: positional encoding axis dead on new stack
RFF σ=10 and σ=3 (both fixed and learnable) all regress on the post-SwiGLU+n_head=2 stack. Mechanism: multiplicative gating + wider attention heads now provide expressive feature interactions that RFF was previously substituting for. The original RFF win (#3606 σ=3 -1.91%) was on the much weaker GELU+wd baseline.

### Program finding #4: wider-head trend has a knee at n_head=2
- n_head=8 (#3535): +22.9% (much worse)
- n_head=4 (baseline before #3643): 75.578
- n_head=2 (#3643): **70.925 ✓ WON**
- n_head=1 (#3773 round 11): +5.0% (regressed)

The optimal subspace partitioning is **2-head × 48-dim**. Single ultra-wide head loses multiplicative interaction between heads; 4+ narrow heads can't form rich-enough per-head subspaces. Knee is at 2.

### Wins: SwiGLU + n_head=2 — the compounding architectural axes
- **SwiGLU gating (#3608)**: −18.2% val. Multiplicative gating unlocks adaptive feature selection.
- **n_head=2 head_dim=48 (#3643)**: −6.2% val. Wider per-head attention subspace.

Both pushed in the **opposite direction of conventional ML wisdom**. The physics surrogate prefers fewer-wider heads + gated FFN — suggests the relevant features are dense interactions across mesh tokens, not sparse multi-pattern attention.

### Standing surprise: GEGLU > SwiGLU at hidden_inner=192 (pending compound retest)
Frieren's #3727 showed GELU-gated FFN beats SiLU-gated by −3.1% val on the OLD stack. Retest on n_head=2 stack stalled — needs nudge.

## Current research focus

### Tier 1 (compound architectural wins — in flight or stale)
1. **GEGLU + n_head=2** — frieren (#3727 retest, **STALE — needs nudge**).
2. **surf_weight=10→5** — edward (#3645, **STALE — needs nudge**).

### Tier 2 (budget-neutral architectural & schedule alternatives — round 10/11 fresh)
3. **RMSNorm replacement of LayerNorm** — alphonse (#3849).
4. **Warmup 2→1ep + Cosine T_max=13** — tanjiro (#3851). Reallocate compute toward cosine annealing tail.
5. **AdamW β2 0.999→0.95** — fern (#3853). Responsive second-moment EMA.
6. **Mish in SwiGLU gate** — askeladd (newly assigned). Gate activation curvature.
7. **AdamW β1=0.9→0.95** — nezuko (newly assigned). Slower momentum, complements fern.
8. **Huber β=1.0→0.5** — thorfinn (newly assigned). Sharper L1 loss transition.

## Open questions
- Does GEGLU compound with n_head=2 to give an additional −2-3%? (frieren #3727 STALE)
- Does surf_weight=5 rebalance vol/surf losses productively? (edward #3645 STALE)
- Does RMSNorm match or beat LN with possible wall-clock benefit? (alphonse #3849)
- Does shorter warmup + longer cosine fix the "still descending" pattern? (tanjiro #3851)
- Does β2=0.95 give responsive second-moment tracking benefit? (fern #3853)
- Does Mish gate activation differ meaningfully from SiLU? (askeladd — new)
- Does β1=0.95 smooth momentum direction productively? (nezuko — new)
- Does Huber β=0.5 sharpen mid-range error gradients usefully? (thorfinn — new)
- ~~Does n_head=1 extend the wider-head trend?~~ **NO — knee at 2 (+5.0%)**
- ~~Does RFF positional encoding still help on new stack?~~ **NO — both arms +8-9% (#3655 closed)**
- ~~Does FFN dropout p=0.1 regularize productively?~~ **NO — 5th uniform regularizer failure (#3607)**

## Closed/regressed (cumulative)
- **NEW: #3773 askeladd n_head=1 head_dim=96: +5.0% val (wider-head knee at 2)**
- **NEW: #3655 nezuko RFF σ=3 + learnable σ: +9.4% / +8.4% (RFF axis dead on new stack)**
- **NEW: #3607 thorfinn FFN dropout p=0.1: +6.2% val (5th uniform regularizer)**
- #3744 tanjiro hidden_inner=256: +12.9% val
- #3775 fern n_layers=6 on new stack: +7.28% val
- #3819 alphonse LayerScale γ=1e-4: +7.98% val
- #3774 alphonse attn-dropout p=0.1: +7.79% val
- #3639 alphonse EMA α=0.999: +13.5%
- #3646 fern DropPath p=0.1: +24.6%
- #3536 tanjiro eta_min=1e-5: +6.6%
- #3579 alphonse lr=1e-3: +2.47%
- #3569 fern wd=5e-4: +1.30%
- #3564 edward n_layers=4: +2.71%
- #3535 askeladd n_head=8: +22.9%
- #3502 alphonse n_hidden=64: +9.20%
- #3503 edward mlp_ratio=4: +5.02%
- #3505 frieren per-channel pressure [1,1,2]: +2.85%
- #3506 thorfinn n_layers=6: +15.0% (GELU)
- #3534 nezuko RFF σ=10: +2.89%
- #3453 edward T_max=10: +3.55%
- #3301 alphonse width-192: +8.53%
- #3304 frieren surf_weight=20: +4.12%
- #3302 askeladd depth-8: +1.53%
- #3223 thorfinn BF16+batch=8: +34%
- #3295 edward slice_num=128: +20%
- Round-1: #3205, #3179, #3183, #3214, #3216, #3220

## Next research directions (beyond current in-flight)
1. **GEGLU + n_head=2 + RMSNorm** (compound test if individual wins land)
2. **Pre-norm vs Post-norm ordering** — budget-neutral
3. **Log-pressure target** — handles dynamic range; single_in_dist split has highest error 81.93
4. **Physics-aware horizontal flip data augmentation** — doubles training data with symmetry
5. **OneCycleLR vs SequentialLR** — different schedule shape, same compute
6. **Slot temperature τ in slot attention** — learnable scalar (mechanism distinct)
7. **MoE-FFN with 2 experts + token routing** — capacity reallocation (complex)
8. **Depth-scheduled DropPath (0→0.05 linear ramp)** — asymmetric regularization, untested
9. **Mish or GELU activation alongside SwiGLU's V(x) path** (instead of/in addition to gate path)
