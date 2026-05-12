# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 11)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 98.1642`** — PR #1558 (Huber surface loss delta=0.5), merged 2026-05-12 22:00.
**Test 3-split mean: 98.7537** (cruise still NaN, but all other splits now reportable via #1527 guard).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | **#1558** | **Huber surface loss δ=0.5** | **98.1642** | **−17.72%** |

## Current research focus

**Cycle 11.** Three orthogonal mechanisms are now merged (BIVW + surf-head + Huber) giving 26% total improvement. The question now is what stacks on top of the Huber baseline:

1. **Huber on volume** — surface Huber won by suppressing per-node outlier gradients; same problem exists on volume nodes. Testing now (#1650 tanjiro).
2. **Huber delta sweep** — is δ=0.5 optimal or do δ=0.2/0.3 give further L1-regime benefit? Thorfinn running (#1627).
3. **Grad-clip + higher LR on Huber base** — Huber already reduces gradient scale; does clip also help at batch level? Fern rebasing (#1499).
4. **Capacity expansions rebased onto Huber** — slice_num=128 (#1501 nezuko), wider MLP (#1498 edward), BF16 (#1572 frieren) all need rebase.
5. **LR schedule on Huber base** — warmup + cosine on new optimum (#1497 askeladd, rebasing).
6. **Pressure-channel emphasis** — still recovering from rate-limit stall (#1496 alphonse).

## Key insight from cycle 11

**Per-channel BIVW (PR #1580) failed badly (+29.6% regression).** Root cause: scalar BIVW was *implicitly* a p-variance-driven Re-curriculum (p dominates pooled variance). Per-channel decoupling removed this beneficial coupling and gave conflicting per-channel signals. Lesson: the scalar BIVW coupling across channels is load-bearing for `mae_surf_p` and must be preserved in any future reweighting work.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP (stale) | Baseline updated to 98.16; recovering from rate-limit |
| 1497 | askeladd | warmup-cosine-lr | WIP (stale) | Baseline updated to 98.16; rebasing onto Huber |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP (stale) | Baseline updated to 98.16; rebasing |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP (rebase) | Baseline updated to 98.16; add grad_clip=10.0 arm |
| 1501 | nezuko | more-slices (64 to 128) | WIP | Baseline updated to 98.16; rebasing |
| 1572 | frieren | bf16-mixed-precision | WIP | BF16 arms running |
| 1627 | thorfinn | huber-delta-sweep | WIP | delta=0.2 and 0.3 arms; running |
| 1650 | tanjiro | huber-on-volume-loss | WIP | NEW: vol_huber_delta ∈ {0.3, 0.5, 1.0} |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head** — confirmed (PR #1528, −5.4% additional).
3. **Huber surface loss delta=0.5** — confirmed (PR #1558, **−17.7%**). The largest gain yet.
4. **Per-channel BIVW** — **rejected** (PR #1580, +29.6% regression). Scalar BIVW's p-dominated coupling is beneficial — do not break it.
5. **Smaller Huber delta** — testing (PR #1627). surf_l1_frac high at delta=0.5; smaller delta may push more into L1 regime.
6. **Huber on volume loss** — testing (PR #1650). Orthogonal extension of surface Huber; volume MSE has same outlier-amplification problem.
7. **Grad-clip + higher LR** — rebasing on new 98.16 baseline (#1499).
8. **BF16/AMP** — testing (#1572); primarily for capacity headroom.
9. **Capacity (MLP width, slices)** — still WIP (#1498, #1501); rebasing onto Huber base.
10. **Warmup schedule** — WIP (#1497); rebasing onto Huber base.
11. **Pressure-channel emphasis** — WIP (#1496); rebasing onto Huber base.

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head, no BIVW) — 6.2% worse.
- **PR #1500** (n_hidden=256 at FP32) — budget failure; BF16 unlock in progress (#1572).
- **PR #1580** (per-channel BIVW) — 29.6% regression. Scalar BIVW implicitly p-dominated; decoupling removed the Re-curriculum.

## Potential next directions

- **Huber on surface and volume jointly** — if #1650 confirms, compose both (already in flight)
- **Per-channel Huber delta** — different delta per channel (p vs Ux/Uy channels)
- **Smaller Huber delta** (δ=0.1, 0.2, 0.3) — thorfinn sweeping now (#1627)
- **Compose Huber + clip + LR** — if #1499 rebase beats 98.16
- **Capacity scaling (n_hidden=256) + BF16 + Huber** — full composition when #1572 merges
- **Re-bin stratified sampler** — make BIVW's implicit Re-curriculum explicit at data-loader level (pre-computed p-variance weights frozen over full corpus)
- **surf_weight tuning** — with Huber active, optimal surf_weight may have shifted from 10
- **LR schedule on Huber recipe** — warmup + cosine on the new optimum (#1497)
- **EMA per-channel variance** (tanjiro suggestion) — reduce within-batch noise for any per-channel weighting future attempt

## Known issues

- ~~**test_avg/mae_surf_p = NaN**~~ — Fixed via PR #1527 (merged). 3-split test mean now reportable.
- **Rate limit impact**: GitHub GraphQL rate limits caused multiple student pods to see "No assigned PRs" and idle for 30–60 min. Pods recovering as rate limit resets.
- **Slice-attention VRAM**: n_hidden=256 needs BF16 (#1572) to be fairly evaluated.
- **Multiple WIPs need rebase**: #1496–#1499, #1501 all need rebase onto advisor branch post-Huber-merge (baseline notes sent to each).
