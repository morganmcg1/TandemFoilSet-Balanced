# SENPAI Research State

- **Last updated:** 2026-05-16 12:32 (round-12 triage complete: closed #3872 Mish, #3853 β2=0.95, #3851 warmup catastrophic, #3849 RMSNorm, #3727 frieren GEGLU stale; assigned 5 fresh PRs)
- **Most recent research direction from human researcher team:** none (no open issues — verified 12:32Z).
- **Current best (merged):** `val_avg/mae_surf_p` = **70.925** (PR #3643 n_head=2 head_dim=48)
- **GH rate-limit status:** fresh window (~4100 remaining); next reset 13:20 UTC.

## Active PRs after this triage

| PR | Student | Hypothesis | State | Notes |
|----|---------|-----------|-------|-------|
| #3645 | edward | surf_weight=10→5 (SwiGLU) | WIP (active 11:23 UTC) | Loss rebalance — back from stale |
| #3873 | nezuko | AdamW β1=0.9→0.95 | WIP (training) | Optimizer momentum hyper |
| #3875 | thorfinn | Huber β=1.0→0.5 | WIP (training) | Loss-shape axis |
| **#3935** | askeladd | Pre-norm → Post-norm in TransolverBlock | NEW (assigned 12:25) | Architectural reordering, budget-neutral |
| **#3937** | fern | Learnable slot-attention temperature τ | NEW (assigned 12:28) | 5 params (one per block), init=1.0 |
| **#3938** | tanjiro | trunc_normal init (std=0.02) on all Linear | NEW (assigned 12:30) | Initialization axis, never tested |
| **#3939** | alphonse | GEGLU retest on n_head=2 stack | NEW (assigned 12:31) | Reassigned from frieren #3727 (stale) |
| **#3940** | frieren | Log-magnitude pressure target | NEW (assigned 12:32) | First hypothesis attacking single_in_dist 81.93 worst-split |

All 8 students busy. **#3727 frieren GEGLU closed** after 3 nudges + 6h stale; GEGLU hypothesis preserved by reassignment to alphonse on a fresh pod.

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

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=3e-4, betas=(0.9, 0.999)) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12, eta_min=0) + lr=7e-4 + epochs=14 + slice_num=96 + n_hidden=96 + **n_head=2 (head_dim=48)** + SwiGLU FFN (SiLU gate, hidden_inner=192, bias-free W1/V/W2) + n_layers=5 + LayerNorm pre-norm + dropout=0.0 + surf_weight=10.

Total improvement to date: 116.61 → 70.925 = **−39.2%** from original Huber baseline.

## Key program insights

### Program finding #1: the 14-epoch / 30-min budget is the binding constraint
After multiple round-10/11/12 closures (LayerScale, hidden_inner=256, n_layers=6, β2=0.95, RMSNorm) all showed "val curve still descending at E14" or noisy convergence patterns, the rule is firm: any experiment that slows convergence per-step or per-epoch fails under the budget. Implication: future wins require **budget-respecting changes** — modifications that converge as fast or faster than baseline.

### Program finding #2: uniform regularization is decisively past its optimum
**5 textbook-magnitude uniform regularizers have failed** on the post-SwiGLU+n_head=2 stack:
- DropPath p=0.1: +24.6% (#3646)
- EMA α=0.999: +13.5% (#3639)
- attn-dropout p=0.1: +7.8% (#3774)
- FFN dropout p=0.1: +6.2% (#3607)
- wd=5e-4: +1.3% (#3569)

The SwiGLU+n_head=2 model has substantially lower intrinsic overfitting risk than the GELU+n_head=4 baseline where these were originally evaluated. Mechanism: tighter train-val gap + slower convergence under noise injection. Asymmetric/structured regularization (depth-scheduled DropPath, per-layer wd) remains untested.

### Program finding #3: positional encoding axis dead on new stack
RFF σ=10 and σ=3 (both fixed and learnable) all regress on the post-SwiGLU+n_head=2 stack. Mechanism: multiplicative gating + wider attention heads now provide expressive feature interactions that RFF was previously substituting for. The original RFF win (#3606 σ=3 -1.91%) was on the much weaker GELU+wd baseline.

### Program finding #4: wider-head trend has a knee at n_head=2
- n_head=8 (#3535): +22.9% (much worse)
- n_head=4 (baseline before #3643): 75.578
- n_head=2 (#3643): **70.925 ✓ WON**
- n_head=1 (#3773): +5.0% (regressed)

The optimal subspace partitioning is **2-head × 48-dim**. Single ultra-wide head loses multiplicative interaction between heads; 4+ narrow heads can't form rich-enough per-head subspaces. Knee is at 2.

### Program finding #5: schedule perturbations are first-order
**Warmup 2→1ep + cosine T_max=13 was catastrophic (+17.7%/+19.1%)** — the LR-velocity discontinuity at epoch 1 (warmup→cosine handoff) wrecks Adam moment state. At batch=4 with ~25 steps/epoch, 2-epoch warmup is the minimum viable for Adam moment stabilization. Schedule timing changes are *not* drop-in like activation/init swaps; they perturb optimizer state directly.

### Program finding #6: optimizer-EMA hypers are tightly calibrated
- β2=0.95 (#3853): +1.87% — too responsive at batch=4
- eta_min=1e-5 (#3536): +6.6% — disturbs cosine endpoint
- β1=0.95 (#3873): TBD — pending

At batch=4 the gradient noise floor is high; Adam's default β1=0.9, β2=0.999 are well-suited to this regime. Faster moment EMAs amplify step-size variance.

### Wins: SwiGLU + n_head=2 — the compounding architectural axes
- **SwiGLU gating (#3608)**: −18.2% val. Multiplicative gating unlocks adaptive feature selection.
- **n_head=2 head_dim=48 (#3643)**: −6.2% val. Wider per-head attention subspace.

Both pushed in the **opposite direction of conventional ML wisdom**. The physics surrogate prefers fewer-wider heads + gated FFN — suggests the relevant features are dense interactions across mesh tokens, not sparse multi-pattern attention.

### Standing surprise: GEGLU > SwiGLU at hidden_inner=192 (compound retest now with alphonse #3939)
Frieren's #3727 (old stack) showed GELU-gated FFN beats SiLU-gated by −3.1% val. #3727 closed 12:31 UTC after 6h stale + 3 nudges; hypothesis reassigned to alphonse as #3939 on a fresh pod. If GEGLU compounds with n_head=2, this is a candidate for another architectural axis.

## Current research focus

### Tier 1 (compound architectural wins — in flight)
1. **GEGLU + n_head=2 retest** — alphonse (#3939, reassigned from frieren).
2. **surf_weight=10→5** — edward (#3645, ACTIVE 11:23 UTC).
3. **Log-magnitude pressure target** — frieren (#3940, NEW). First direct attack on single_in_dist worst-split.

### Tier 2 (budget-neutral architectural alternatives — in flight)
4. **Pre-norm → Post-norm in TransolverBlock** — askeladd (#3935). Architectural reorder, never-tested.
5. **Learnable slot-attention temperature τ per block** — fern (#3937). 5 new params init=1.0.
6. **Truncated normal init (std=0.02) on Linear weights** — tanjiro (#3938). Initialization axis untested.

### Tier 3 (optimizer/loss hypers — in flight)
7. **AdamW β1=0.9→0.95** — nezuko (#3873).
8. **Huber β=1.0→0.5** — thorfinn (#3875).

## Open questions
- Does GEGLU compound with n_head=2 to give an additional −2-3%? (frieren #3727 STALE — possibly dead pod)
- Does surf_weight=5 rebalance vol/surf losses productively? (edward #3645 — now active)
- Does β1=0.95 smooth momentum direction productively? (nezuko #3873)
- Does Huber β=0.5 sharpen mid-range error gradients usefully? (thorfinn #3875)
- Does post-norm beat pre-norm at depth 5? (askeladd — pending)
- Does learnable slot temperature converge to a non-1.0 value useful for the model? (fern — pending)
- Does trunc_normal(0.02) outperform Kaiming-uniform default? (tanjiro — pending)
- ~~Does Mish gate activation differ meaningfully from SiLU?~~ **NO — +3.73% val (#3872)**
- ~~Does β2=0.95 give responsive second-moment tracking benefit?~~ **NO — +1.87% val + noisy (#3853)**
- ~~Does shorter warmup + longer cosine fix the "still descending" pattern?~~ **NO — +17.74% catastrophic (#3851)**
- ~~Does RMSNorm match or beat LN with possible wall-clock benefit?~~ **NO — +3.29% + slower (#3849)**
- ~~Does n_head=1 extend the wider-head trend?~~ **NO — knee at 2 (+5.0%)**
- ~~Does RFF positional encoding still help on new stack?~~ **NO — both arms +8-9%**
- ~~Does FFN dropout p=0.1 regularize productively?~~ **NO — 5th uniform regularizer failure**

## Closed/regressed (cumulative)
- **NEW: #3849 alphonse RMSNorm: +3.29% val + slower (LN-bias term load-bearing at small width)**
- **NEW: #3851 tanjiro warmup 2→1 + T_max=13: +17.74% val (catastrophic LR-handoff)**
- **NEW: #3853 fern AdamW β2=0.95: +1.87% val + noisy (too responsive at batch=4)**
- **NEW: #3872 askeladd Mish in SwiGLU gate: +3.73% val (SiLU specifically calibrated to gating)**
- #3773 askeladd n_head=1 head_dim=96: +5.0% val (wider-head knee at 2)
- #3655 nezuko RFF σ=3 + learnable σ: +9.4% / +8.4% (RFF axis dead on new stack)
- #3607 thorfinn FFN dropout p=0.1: +6.2% val (5th uniform regularizer)
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
1. **GEGLU + n_head=2 + log-pressure target** (compound test if individual wins land)
2. **Log-pressure target** — handles dynamic range; single_in_dist split has highest error 87.36
3. **Physics-aware horizontal flip data augmentation** — doubles training data with symmetry
4. **OneCycleLR vs SequentialLR** — different schedule shape, same compute (must keep 2-ep+ warmup per finding #5)
5. **MoE-FFN with 2 experts + token routing** — capacity reallocation (complex)
6. **Depth-scheduled DropPath (0→0.05 linear ramp)** — asymmetric regularization, untested
7. **Mish or GELU activation alongside SwiGLU's V(x) path** (instead of/in addition to gate path)
8. **Mixed n_head per layer** (e.g., layer 0-2 n_head=2, layer 3-4 n_head=1) — heterogeneous head specialization
9. **Attention-only LR multiplier** — separate LR group for slot attention vs FFN
