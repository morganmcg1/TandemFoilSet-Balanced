# SENPAI Research State

- **Updated:** 2026-04-29 01:10 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #775 Round 2 (warmup=0 + clip=0.5 + Huber δ=0.5 + EMA=0.99) — MERGED:**  
`val_avg/mae_surf_p = 96.54`, `test_avg/mae_surf_p = 85.33`  
**−31.5% val / −33.5% test vs unmodified default (140.95/128.32).**

Default flags for ALL future runs: `--warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99`

**Previous milestone (PR #769, Huber δ=0.5 alone):** val=102.86, test=94.83  
**Previous milestone (PR #773, EMA alone):** val=119.35, test=108.79

## Current research focus

Four independent wins are now stacked and confirmed. The core stack is:
1. **Huber δ=0.5 (PR #769):** −13.8% val. Outlier robustness via linear penalty for large residuals.
2. **EMA decay=0.99 (PR #773):** −15.4% vs unmodified default. Weight averaging for flatter minima.
3. **clip=0.5 + warmup=0 (PR #775):** −6.1% on top of Huber alone. Clipping acts as continuous gradient regularizer (100% of steps clipped throughout, median pre-clip ~45-60). Dropping warmup recovers ~5 epochs of optimization budget.

**The current phase is: validating remaining independent directions on the full 4-way stack.**
- All pre-stack experiments (alphonse #881, edward #867, fern #859, thorfinn #860, frieren #862) are being rebased and re-run with the full stack.
- Any win on EMA-alone or Huber-alone baselines is valid signal but not yet confirmed on the full stack.

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #881 | alphonse | Huber δ ∈ {0.1,0.25,0.5} + EMA stack (no clip — pre-stack) | Status:WIP |
| #951 | edward | Per-channel Huber δ: δ_p ∈ {0.1,0.25,0.5,1.0} vs δ_vel=0.5 on full stack | Status:WIP |
| #859 | fern | Surface weight scan sw ∈ {10,15,20,30} + Huber+EMA+clip full stack | Status:WIP (rebase) |
| #860 | thorfinn | OneCycle schedule on full 4-way stack (cosine vs onecycle + full stack) | Status:WIP (rebase) |
| #862 | frieren | Slice scan downward {32,48,64} + full 4-way stack | Status:WIP (rebase) |
| #776 | tanjiro | Deeper model n_layers=8 | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |
| #944 | nezuko | Clip norm fine-scan {0.1,0.25,0.5,1.0} on full 4-way stack | Status:WIP |

## Key pending questions (expected within 2-3 poll cycles)

1. **Does Huber δ < 0.5 improve further with EMA+clip?** (alphonse #881 — running without clip; need clip version). δ trend was monotone downward without EMA; with clip in place the effective gradient scale is reduced, so optimal δ may shift.
2. **Does clip=0.25 outperform clip=0.5 on full stack?** (nezuko #944). Round-2 showed warmup=5/clip=0.25 slightly beat warmup=5/clip=0.5 (97.85 vs 99.47); warmup=0/clip=0.5 beat both at 96.54. Now need warmup=0/clip=0.25.
3. **Does OneCycle schedule help on full 4-way stack?** (thorfinn #860 rebase). OneCycle gave +5.3% over EMA-only; needs retest on full stack. This is likely the most impactful pending question.
4. **Does lower slice_num {32,48} beat 64 on full stack?** (frieren #862 rebase). Monotone regression at higher slices suggests 32 or 48 may win; each step also frees VRAM.
5. **Does surf_weight scan help with full stack?** (fern #859 rebase). Huber may already implicitly upweight surface vs volume (volume residuals are larger → more linear attenuation); optimum sw may shift from 20 toward 10 or 15.
6. **Does per-channel Huber δ (δ_p < δ_vel) help?** (edward #951). Pressure has heaviest residual tails — per-channel δ_p=0.25 vs δ_vel=0.5 should concentrate pressure outlier robustness where it matters most. (**β₂ scan closed as dead end** — β₂=0.999 is already optimal; β₂=0.95 ties at best.)

## Potential next research directions

**Immediate:**
- δ < 0.5 + clip: alphonse's sweep has EMA but not clip; if it shows δ=0.25 helps, request a full-stack clip+δ=0.25 test
- OneCycle + lower peak_lr: thorfinn's result shows 1e-3 works; exploring 1.5×/3× base_lr sensitivity (thorfinn follow-up #1)
- clip=0.25 with warmup=0: nezuko's new PR #944 covers this

**After current wave completes:**
- **Per-channel Huber δ**: δ_p smaller than δ_Ux/Uy (pressure residuals have different tail shape)
- **AdamW β₂ with clip** (low priority): β₂=0.999 is at the optimum without clip (PR #867 closed). β₂=0.95 ties the control, so the interaction with clip is marginal. Defer unless other directions stall.
- **EMA decay scan with full stack**: PR #773 fixed decay=0.99 without clip; clip smooths per-step gradients, so longer EMA half-lives (0.995, 0.999) may benefit
- **OneCycle peak_lr and pct_start tuning**: thorfinn's follow-ups #1–#3

**Architectural (after hyperparameter space exhausted):**
- Deeper model n_layers=8 (tanjiro #776 in flight)
- Surface-aware routing in PhysicsAttention (askeladd #770 in flight)
- Multi-task curriculum (Ux/Uy first, freeze, fine-tune p)
- Galerkin attention swap

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap (budget truncates at ~14 epochs).
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling in `train.py`.
- One hypothesis per PR. Compound only after isolated wins verified.
- **Default flags for all future runs: `--warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99`**
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
