# SENPAI Research State

- **Last updated:** 2026-05-16 07:45 (PR #3643 n_head=2 merged — new baseline 70.925; #3639 EMA + #3646 DropPath closed; #3727 GEGLU sent back; askeladd/alphonse/fern assigned new experiments #3773/#3774/#3775)
- **Most recent research direction from human researcher team:** none (no open issues — verified 06:30Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **70.925** (PR #3643 n_head=2 head_dim=48)
- **GH rate-limit status:** ~3500/5000 remaining.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3607 | thorfinn | FFN dropout p=0.05 (SwiGLU) | WIP (rebase) | Old GELU regularizer axis now suspect (per #3646 DropPath result) |
| #3645 | edward | surf_weight=10→5 (SwiGLU) | WIP | Loss rebalance — testing on n_head=2 stack soon |
| #3655 | nezuko | RFF σ=3 + learnable-σ 2-arm | WIP (rebase) | Positional encoding axis |
| #3727 | frieren | GEGLU at hidden_inner=192 | WIP (sent back for retest on n_head=2 stack) | Was −3.10% standalone win |
| #3744 | tanjiro | hidden_inner=192→256 | WIP | Capacity ceiling probe |
| #3773 | askeladd | n_head=1 head_dim=96 (extend wider-head trend) | WIP (newly assigned) | Follow-up to #3643 win |
| #3774 | alphonse | attention dropout p=0.1 | WIP (newly assigned) | New regularization axis on attention path |
| #3775 | fern | n_layers=6 on SwiGLU + n_head=2 | WIP (newly assigned) | Depth retest on new stack |

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

### SwiGLU + wider-narrower heads — the compounding architectural axes
The two biggest single-experiment improvements:
- **SwiGLU gating (#3608)**: −18.2% val. Multiplicative gating unlocks adaptive feature selection.
- **n_head=2 head_dim=48 (#3643)**: −6.2% val. Wider per-head attention subspace.

Both pushed in the **opposite direction of conventional ML wisdom** (LLMs typically want many narrow heads; SwiGLU is standard but n_head=2 with head_dim=48 is unusual). The physics surrogate prefers fewer-wider heads + gated FFN — suggests the relevant features are dense interactions across mesh tokens, not sparse multi-pattern attention.

### Uniform regularization is past its optimum
The post-SwiGLU stack has a much tighter train-val gap. **All uniform regularizers tested now hurt**:
- wd=5e-4: +1.3% (#3569 closed)
- FFN dropout p=0.1: +0.6% on old baseline, retest pending (#3607)
- DropPath p=0.1: +17% (#3646 just closed)
- EMA α=0.999: +13.5% live, +23.4% EMA (#3639 just closed)
- eta_min=1e-5: +6.6% (#3536 closed)

**Implication:** the next regularization wins (if any) require asymmetric/structured approaches — depth-scheduled DropPath, attention-only dropout, or per-layer-class weight decay. Or maybe regularization is simply not the right axis now.

### Surprise: GEGLU > SwiGLU at hidden_inner=192
Frieren's #3727 showed GELU-gated FFN beats SiLU-gated by −3.1% val. **This contradicts LLM literature** where SwiGLU edges GEGLU slightly. Possible mechanism: on smooth CFD features, GELU's slightly-different negative-region curvature gives the gate better gradient signal. Pending compound retest on n_head=2 stack.

## Current research focus

### Tier 1 (compound architectural wins)
1. **GEGLU + n_head=2** — frieren (#3727 retest). Both standalone winners; expect compound −2-3%.
2. **n_head=1 head_dim=96** — askeladd (new). Does the wider-head trend continue or saturate?
3. **n_layers=6 on new stack** — fern (new). #3506 failed catastrophically on GELU; SwiGLU+n_head=2 may unlock depth capacity.
4. **hidden_inner=256** — tanjiro (#3744). Capacity ceiling probe.

### Tier 2 (asymmetric regularization, never tested)
5. **Attention dropout p=0.1** — alphonse (new). Clean new axis: dropout in attention path, not FFN. Different mechanism than the failed uniform regularizers.

### Tier 3 (other open axes)
6. **surf_weight=5** — edward (#3645). Loss rebalance.
7. **FFN dropout p=0.05** — thorfinn (#3607 rebase). Lower-than-tested p; may avoid the DropPath-style convergence drag.
8. **RFF σ=3** — nezuko (#3655 rebase). Positional encoding.

## Open questions (on new n_head=2 baseline)
- Does n_head=1 push the head-width trend further or hit the knee? (askeladd)
- Does GEGLU compound with n_head=2 to give an additional −2-3%? (frieren #3727)
- Does n_layers=6 finally work on SwiGLU+n_head=2? (fern)
- Does attention dropout reach a different sweet spot than FFN dropout? (alphonse)
- Does hidden_inner=256 saturate or continue paying off? (tanjiro #3744)
- ~~Does eta_min=1e-5 compound on SwiGLU?~~ **DEAD axis**
- ~~Does DropPath p=0.1 regularize productively?~~ **DEAD axis**
- ~~Does EMA α=0.999 smooth the cosine tail?~~ **DEAD axis**

## Closed/regressed (cumulative)
- **NEW: #3639 alphonse EMA α=0.999: +13.5% live val (axis dead at this α)**
- **NEW: #3646 fern DropPath p=0.1: +24.6% val (axis dead at this p)**
- #3536 tanjiro eta_min=1e-5 on SwiGLU: +6.6%
- #3579 alphonse lr=1e-3: +2.47%
- #3569 fern wd=5e-4: +1.30%
- #3564 edward n_layers=4: +2.71%
- #3535 askeladd n_head=8: +22.9% (vs current direction; n_head=2 won)
- #3502 alphonse n_hidden=64: +9.20%
- #3503 edward mlp_ratio=4: +5.02% (GELU; SwiGLU reframes)
- #3505 frieren per-channel pressure [1,1,2]: +2.85%
- #3506 thorfinn n_layers=6: +15.0% (GELU; retesting on SwiGLU+n_head=2 now)
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
2. **Log-pressure target** — handles dynamic range across splits
3. **Depth-scheduled DropPath** (linear ramp 0→0.05 across layers) — asymmetric regularization
4. **Per-layer weight decay** — decouple OOD/in-dist
5. **Data augmentation** — physics-aware flip
6. **Slot attention slice_num scaling** — already at 96; try 128 again on new stack (was −20% on weaker base)
7. **Batch size 8** — try again on faster n_head=2 model (was −34% on weaker base)
