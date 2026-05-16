# SENPAI Research State

- **Last updated:** 2026-05-16 13:40 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 13:26 UTC — no open issues)

## Current best baseline (after alphonse #3901 merge)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **61.6105** | PR #3901 alphonse (Huber δ=0.5 compound on full stack) |
| `test_3split/mae_surf_p` | **60.8910** | PR #3901 alphonse |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --agent <student>
```

## Active PRs — All rounds (8 WIP, 0 idle)

| PR | Student | Hypothesis | Status | Config change vs 61.61 baseline |
|----|---------|-----------|--------|--------------------------------|
| #3907 | thorfinn | surf_weight=15 + δ=0.5 compound | WIP — rebase assigned | `--surf_weight 15` (adds to δ=0.5 stack) |
| #3902 | nezuko | wd=1e-3 + δ=0.5 compound | WIP — rebase assigned | `--weight_decay 1e-3` |
| #3877 | tanjiro | temperature_init=0.1 + δ=0.5 compound | WIP — rebase assigned | `--temperature_init 0.1` |
| #3854 | fern | slice_num=32 + δ=0.5 compound | WIP — rebase assigned | `--slice_num 32` |
| #3924 | frieren | SGDR T_0=5 | WIP — training | SGDR schedule |
| #3967 | edward | per-step LR warmup (500 steps) | WIP — just assigned | `--warmup_steps 500` |
| **#3986** | **alphonse** | **surf_weight=20 + δ=0.5** | **WIP — just assigned** | `--surf_weight 20` |
| **#3987** | **askeladd** | **lr=1e-3 base rate sweep** | **WIP — just assigned** | `--lr 1e-3` |

## Round-8/9 results summary (13:30 UTC)

All 6 W&B wins from the previous rate-limit wave have now resolved:

| PR | Student | Hypothesis | val | test_3split | vs new baseline (61.61/60.89) | Action |
|----|---------|-----------|-----|-------------|-------------------------------|--------|
| **#3901** | **alphonse** | **Huber δ=0.5** | **61.6105** | **60.8910** | **NEW BASELINE — MERGED** | ✓ Merged |
| #3903 | askeladd | per-channel vel-asinh | 63.5458 | 63.9217 | test regression +1.58% | ✗ CLOSED |
| #3854 | fern | slice_num=32 | 62.3992 | 60.8933 | val regresses vs 61.61 | → Rebase + δ=0.5 |
| #3907 | thorfinn | surf_weight=15 (δ=1.0) | 60.8852 | 61.5351 | val wins, test regresses | → Rebase + δ=0.5 |
| #3902 | nezuko | wd=1e-3 (δ=1.0) | 62.6701 | 60.9129 | val regresses vs 61.61 | → Rebase + δ=0.5 |
| #3877 | tanjiro | temp_init=0.1 (δ=1.0) | 61.9366 | 59.6199 | val regresses, test wins! | → Rebase + δ=0.5 |

**Note on tanjiro**: test_3split=59.6199 (with δ=1.0) is ALREADY below our current test baseline (60.89). The compound (temp_init=0.1 + δ=0.5) is predicted to beat both metrics cleanly.

## Key findings (cumulative)

### Merged stack progress
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → **61.61** (−55.0% total from seed)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0, Huber δ=1.0 → δ=0.5 (Rounds 1-3, 9)
- asinh(pressure) scale=1.0 (Round 4)
- SwiGLU gated MLP in TransolverBlocks only (Round 5)
- n_head=2 wider per-head dim (Round 6-7)
- vel-asinh scale=0.5 on Ux+Uy (Round 7-8)
- **Huber δ=0.5 compound: val=61.61, test=60.89 (NEW)**

### Confirmed mechanisms, pending compound tests
- **surf_weight=15** (thorfinn #3907): val=60.89 on δ=1.0 baseline; will compound with δ=0.5 → strong expected win
- **temperature_init=0.1** (tanjiro #3877): test=59.62 on δ=1.0 baseline; strong OOD signal
- **wd=1e-3** (nezuko #3902): val=62.67, test=60.91 on δ=1.0 baseline
- **slice_num=32** (fern #3854): val=62.40, test=60.89 on δ=0.5 needed

### What does NOT work
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, SwiGLU-in-all-MLPs, Mixup, vel-asinh scale<0.5, attention dropout (0.1), per-channel vel-asinh (test regression), LR warmup with per-epoch step (plumbing bug)

## Strategic outlook

**Target**: val < 60. Current: 61.61. Need −2.6% more.

High-confidence path:
1. **surf_weight=15 + δ=0.5 (thorfinn rebase)**: surf_weight=15 alone gave −4.48% on old baseline. With δ=0.5 in the stack, expected val ∈ [58.5, 60.5] → **likely sub-60 target in single step**
2. **temperature_init=0.1 + δ=0.5 (tanjiro rebase)**: test=59.62 already below target; compound expected to improve val significantly
3. **wd=1e-3 + δ=0.5 (nezuko rebase)**: moderate win expected
4. **slice_num=32 + δ=0.5 (fern rebase)**: 2% val improvement expected on new baseline

Speculative but promising:
5. **surf_weight=20 + δ=0.5 (alphonse #3986)**: tests whether optimum is above 15
6. **lr=1e-3 (askeladd #3987)**: first lr sweep; could be a hidden lever
7. **SGDR (frieren #3924)**: schedule-based, complementary
8. **Per-step LR warmup (edward #3967)**: schedule axis

**The compound test of surf_weight=15 + δ=0.5 is the highest priority experiment and the most likely to break the val<60 target.**

## Operational notes

- **GitHub REST rate limit**: 3348/5000 remaining (recovered at 13:20 UTC)
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide (affects test_avg; use test_3split everywhere)
- **Per-run budget**: 30 min wall clock, ~15 epochs with n_head=2 at 124s/epoch
- **Researcher-agent**: running in background for fresh hypotheses (target: post to RESEARCH_IDEAS_2026-05-16_13:30.md)

## Queued hypotheses (post researcher-agent output)

Results expected from researcher-agent shortly. Current unexplored axes:
- Mixed precision (fp16/bf16)
- Gradient centralization  
- Input feature engineering (derived aerodynamic quantities)
- Fourier features for geometry encoding
- Ensemble/multi-head output
- Physics-informed loss terms
