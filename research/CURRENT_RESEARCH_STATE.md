# SENPAI Research State

- **Last updated:** 2026-05-16 09:30 (round-10 triage: closed #3819 LayerScale, #3744 hidden_inner=256, #3775 n_layers=6 — all 3 "budget-constrained" negatives. 3 fresh assignments queued.)
- **Most recent research direction from human researcher team:** none (no open issues — verified 09:28Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **70.925** (PR #3643 n_head=2 head_dim=48)
- **GH rate-limit status:** ~2890/5000 remaining; reset 10:20 UTC.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3607 | thorfinn | FFN dropout p=0.05 (SwiGLU) | WIP (rebase) | Old GELU regularizer axis now suspect (per #3646 DropPath result) |
| #3645 | edward | surf_weight=10→5 (SwiGLU) | WIP | Loss rebalance — testing on n_head=2 stack soon |
| #3655 | nezuko | RFF σ=3 + learnable-σ 2-arm | WIP (rebase) | Positional encoding axis |
| #3727 | frieren | GEGLU at hidden_inner=192 | WIP (sent back for retest on n_head=2 stack) | Was −3.10% standalone win |
| #3773 | askeladd | n_head=1 head_dim=96 (extend wider-head trend) | WIP | Follow-up to #3643 win |
| (closed) #3744 | tanjiro | hidden_inner=256 | CLOSED — past knee (+12.9%) | Pivoting to schedule probe |
| (closed) #3775 | fern | n_layers=6 on SwiGLU+n_head=2 | CLOSED — budget-constrained (+7.28%) | Pivoting to optimizer probe |
| (closed) #3819 | alphonse | LayerScale γ=1e-4 | CLOSED — γ too small (+7.98%) | Pivoting to RMSNorm |

After round-10 triage, alphonse/tanjiro/fern are idle; new assignments queued.

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

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + **n_head=2 (head_dim=48)** + SwiGLU FFN (SiLU gate, hidden_inner=192, bias-free W1/V/W2) + n_layers=5 + dropout=0.0 + surf_weight=10.

Total improvement to date: 116.61 → 70.925 = **−39.2%** from original Huber baseline.

## Key program insights

### Major program finding: the 14-epoch / 30-min budget is now the binding constraint
After round-10's 3 simultaneous closures (LayerScale, hidden_inner=256, n_layers=6), the pattern is unambiguous: **any experiment that slows convergence per-step or per-epoch fails under the budget.** Counted:
- LayerScale γ=1e-4: monotonic-still-descending at E14, +7.98% (residuals need ~10 epochs to grow γ)
- hidden_inner=256: hit cap at 12/14 epochs, +12.9% (compute per epoch too high)
- n_layers=6: hit cap at 13/14 epochs, +7.28% (compute per epoch too high)
- DropPath p=0.1, EMA α=0.999, attn-dropout p=0.1, FFN-dropout p=0.1: all noise-injection regularizers slow effective convergence → all regress
- eta_min=1e-5: low-lr tail wastes annealing time → regresses

**Strategic implication:** Future wins will come from **budget-respecting changes** — modifications that converge as fast or faster than baseline at equal or lower per-epoch compute. This rules out: (a) added params/capacity, (b) added per-step noise, (c) tiny-init schemes that need time to grow. This favors: budget-neutral architectural swaps (RMSNorm, GEGLU), schedule reallocations (more cosine annealing, less warmup), optimizer adaptations (β2 tuning), faster-converging initializations.

### SwiGLU + wider-narrower heads — the compounding architectural axes
The two biggest single-experiment improvements:
- **SwiGLU gating (#3608)**: −18.2% val. Multiplicative gating unlocks adaptive feature selection.
- **n_head=2 head_dim=48 (#3643)**: −6.2% val. Wider per-head attention subspace.

Both pushed in the **opposite direction of conventional ML wisdom** (LLMs typically want many narrow heads; SwiGLU is standard but n_head=2 with head_dim=48 is unusual). The physics surrogate prefers fewer-wider heads + gated FFN — suggests the relevant features are dense interactions across mesh tokens, not sparse multi-pattern attention.

### Uniform regularization + tiny-init capacity expansion past their optimum
The post-SwiGLU stack has tight train-val gap. **All uniform regularizers and slow-init capacity expansion now hurt** (all under 14-epoch budget):
- wd=5e-4: +1.3% (#3569 closed)
- FFN dropout p=0.1: +0.6%, retest pending (#3607)
- DropPath p=0.1: +17% (#3646)
- EMA α=0.999: +13.5% (#3639)
- eta_min=1e-5: +6.6% (#3536)
- attn-dropout p=0.1: +7.8% (#3774)
- LayerScale γ=1e-4: +7.98% (#3819 — capacity expansion via tiny-init scheme)
- n_layers=6: +7.3% (#3775 — direct capacity expansion)
- hidden_inner=256: +12.9% (#3744 — direct capacity expansion)

**Implication:** Either find asymmetric/structured approaches (depth-scheduled DropPath, per-layer wd), or shift to budget-neutral architectural+schedule axes.

### Surprise: GEGLU > SwiGLU at hidden_inner=192
Frieren's #3727 showed GELU-gated FFN beats SiLU-gated by −3.1% val. **This contradicts LLM literature** where SwiGLU edges GEGLU slightly. Possible mechanism: on smooth CFD features, GELU's slightly-different negative-region curvature gives the gate better gradient signal. Pending compound retest on n_head=2 stack.

## Current research focus

### Tier 1 (compound architectural wins, all budget-neutral)
1. **GEGLU + n_head=2** — frieren (#3727 retest). Both standalone winners; expect compound −2-3%.
2. **n_head=1 head_dim=96** — askeladd. Does the wider-head trend continue or saturate?

### Tier 2 (budget-neutral architectural alternatives — new direction)
3. **RMSNorm replacement of LayerNorm** — alphonse (newly assigned). Faster forward, fewer params per norm, often matches/exceeds LN in transformers.
4. **Warmup 2→1 epoch + Cosine T_max=13** — tanjiro (newly assigned). Reallocate compute toward cosine annealing tail (which has been the "still descending" zone in closures).
5. **AdamW β2 0.999→0.95** — fern (newly assigned). Responsive second-moment EMA (half-life ~14 steps vs ~700) better tracks mature-stack gradient statistics.

### Tier 3 (other open axes)
6. **surf_weight=5** — edward (#3645). Loss rebalance.
7. **FFN dropout p=0.05** — thorfinn (#3607 rebase). Lower-than-tested p; may avoid the DropPath-style convergence drag.
8. **RFF σ=3** — nezuko (#3655 rebase). Positional encoding.

## Open questions (on new n_head=2 baseline)
- Does n_head=1 push the head-width trend further or hit the knee? (askeladd #3773)
- Does GEGLU compound with n_head=2 to give an additional −2-3%? (frieren #3727)
- Does RMSNorm match or beat LayerNorm on this stack? (alphonse — new)
- Does shorter warmup + longer cosine annealing fix the "still descending at E14" pattern? (tanjiro — new)
- Does AdamW β2=0.95 give responsive second-moment tracking benefit? (fern — new)
- ~~Does n_layers=6 finally work on SwiGLU+n_head=2?~~ **NO — +7.28%, budget-constrained**
- ~~Does LayerScale γ=1e-4 stabilize?~~ **NO — +7.98%, depth=5 doesn't need it**
- ~~Does hidden_inner=256 saturate or continue paying off?~~ **PAST KNEE — +12.9%**
- ~~Does attention dropout reach a different sweet spot?~~ **DEAD axis at p=0.1**
- ~~Does eta_min=1e-5 compound on SwiGLU?~~ **DEAD axis**
- ~~Does DropPath p=0.1 regularize productively?~~ **DEAD axis**
- ~~Does EMA α=0.999 smooth the cosine tail?~~ **DEAD axis**

## Closed/regressed (cumulative)
- **NEW: #3744 tanjiro hidden_inner=256: +12.9% val (past capacity knee, budget-constrained)**
- **NEW: #3775 fern n_layers=6 on new stack: +7.28% val (budget-constrained)**
- **NEW: #3819 alphonse LayerScale γ=1e-4: +7.98% val (depth=5 needs no LayerScale, init too small)**
- #3774 alphonse attn-dropout p=0.1: +7.79% val
- #3639 alphonse EMA α=0.999: +13.5% live val
- #3646 fern DropPath p=0.1: +24.6% val
- #3536 tanjiro eta_min=1e-5 on SwiGLU: +6.6%
- #3579 alphonse lr=1e-3: +2.47%
- #3569 fern wd=5e-4: +1.30%
- #3564 edward n_layers=4: +2.71%
- #3535 askeladd n_head=8: +22.9% (vs current direction; n_head=2 won)
- #3502 alphonse n_hidden=64: +9.20%
- #3503 edward mlp_ratio=4: +5.02% (GELU; SwiGLU reframes)
- #3505 frieren per-channel pressure [1,1,2]: +2.85%
- #3506 thorfinn n_layers=6: +15.0% (GELU; budget-constrained on SwiGLU+n_head=2 too)
- #3534 nezuko RFF σ=10: +2.89%
- #3453 edward T_max=10: +3.55%
- #3301 alphonse width-192: +8.53%
- #3304 frieren surf_weight=20: +4.12%
- #3302 askeladd depth-8: +1.53%
- #3223 thorfinn BF16+batch=8: +34%
- #3295 edward slice_num=128: +20%
- Round-1: #3205, #3179, #3183, #3214, #3216, #3220

## Next research directions (beyond current in-flight)
1. **GEGLU + n_head=1** (combine the two best discoveries if both stack)
2. **Mish vs SiLU in SwiGLU gate** — activation curvature variant
3. **Pre-norm vs Post-norm ordering** — convergence-speed axis
4. **Log-pressure target** — handles dynamic range across splits (single_in_dist is dominant error split)
5. **Physics-aware horizontal flip data augmentation** — doubles training data with symmetry
6. **OneCycleLR vs SequentialLR** — different schedule shape, same compute
7. **Initialization scheme (Xavier vs trunc_normal vs orthogonal)** — convergence-speed axis
8. **Slot temperature τ in slot attention** — learnable scalar (small param, mechanism distinct)
9. **MoE-FFN with 2 experts + token-level routing** — capacity reallocation (complex; risk: budget hit)
