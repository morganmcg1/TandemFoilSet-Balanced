# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 82.77** (W&B run `mfjoux5g`, thorfinn FiLM v2-on-l1, merged PR #815, 2026-04-28)
- **test_avg/mae_surf_p = 72.27**
- Per-split val: single_in_dist=95.54, geom_camber_rc=91.38, geom_camber_cruise=64.90, re_rand=79.26
- Beat-threshold for new PRs: **val_avg < 82.77**

### Prior bests (for reference)
- val_avg/mae_surf_p = 92.63 (tanjiro L1 surface MAE, merged PR #761) — superseded by FiLM+L1
- val_avg/mae_surf_p = 103.13 (askeladd Huber surf loss, merged PR #814) — superseded by L1

## Founding baseline (round 1 reference)

- val_avg/mae_surf_p = 122.15 (W&B run `8cvp4x6r`, unmodified Transolver)
- test_avg/mae_surf_p = 130.90 (W&B run `zaqz12qi`, re-eval via #807)
- Round-1 noise band: 122–146 (single seed, 14-epoch budget)
- PR #807 (NaN-safe masked accumulation) merged — all future runs produce finite `test_avg`

## Progress summary

| PR | Title | Outcome | val_avg |
|----|-------|---------|---------|
| #807 | NaN-safe scoring fix | **MERGED** (infra) | — |
| #814 | Huber surface loss (delta=1.0) | **MERGED** | 103.13 |
| #761 | L1 surface MAE loss | **MERGED** | 92.63 |
| **#815** | **FiLM+L1 (per-block Re conditioning)** | **MERGED — current best** | **82.77** |
| #748 | Transolver 2x capacity | Closed (under-trained) | 203.16 |
| #762 | Boundary-layer features | Closed (−13.3%) | 138.43 |
| #759 | EMA model weights | Closed (wrong-regime) | 124.51 |
| #847 | Huber delta sweep (0.5, 2.0) | Closed — flat in 0.5-2.0; L1 dominates by 9.9% | 102.97 |
| #751 v2 | Dropout 0.05 + drop_path 0.05 on L1 | Closed — within noise (+0.6%, 93.16) | 93.16 |
| #858 | Focal surface loss gamma=0.5/1.0 on L1 | Closed — γ=0.5 within noise, γ=1.0 +13.4% worse | 92.13 |
| #743 | Channel-weighted L1 (v3, rebase pending) | WIP | 99.21 (v2 ref) |
| #750 | LR warmup + cosine v2 (rebase pending) | WIP | 111.12 (v1 ref) |
| #756 | Fourier Re-encoding v2 (rebase pending) | WIP | 120.22 (v1 ref) |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| askeladd | #884 | RevIN output normalization (per-sample y norm for cross-Re amplitude invariance) | WIP |
| thorfinn | #909 | Pre-block FiLM: condition attention input rather than output | WIP — new assignment 2026-04-28 |
| nezuko | #910 | Re-stratified batch sampling: Re-diverse mini-batches for FiLM+L1 | WIP — new assignment 2026-04-28 |
| fern | #902 | Volume L1 (mirror surface L1 success on volume side) | WIP |
| edward | #750 | LR warmup + cosine v2 (rebase pending onto FiLM+L1) | WIP |
| frieren | #756 | Fourier Re-encoding v2 (rebase pending onto FiLM+L1) | WIP |
| alphonse | #743 | Channel-weighted L1 v3 (rebase onto post-#761; now needs rebase onto FiLM+L1 too) | WIP |
| tanjiro | #869 | surf_weight sweep (3.0 and 5.0) | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments include `--epochs 14` so cosine annealing completes.
- **NaN test poisoning FIXED** via PR #807. All future runs produce finite `test_avg/mae_surf_p`.
- **L1 dominates the loss-shape sensitivity curve.** Full ordering confirmed (PRs #761, #814, #847): L1 (92.63) << Huber(0.5) (102.97) ≈ Huber(1.0) (103.13) < Huber(2.0) (106.78). The big lever is the Huber→L1 step (−9.9%).
- **FiLM stacks cleanly with L1** (PR #815 v2-on-l1: −10.6% on top of L1). Orthogonal mechanisms confirmed: loss shape (L1) ⊥ hidden-state Re modulation (FiLM). All 4 val splits improved. FiLM gains biggest on Re-stratified and widest-Re-range splits (re_rand −9.2%, cruise −10.3%).
- **Focal loss falsified on L1 base** (PR #858): high-error surface nodes are convergence-bottlenecked, not gradient-bottlenecked. Focal amplification slows convergence at this budget.
- **Channel weighting stacks with Huber** (alphonse #743 v2: −3.8% on top of Huber). Pending whether it also stacks with FiLM+L1 (v3 in progress).
- **Surface dominates volume ~7:1 at L1 convergence** (tanjiro diagnosis). surf_weight=3.0 (#869) tests whether rebalancing frees volume capacity.
- **Boundary-layer features falsified.** log(Re·|saf|) is redundant; volume-node saf mismatch hurts in-dist.
- **IMPORTANT:** Most WIP PRs (#743 channel-weighted, #750 LR warmup, #756 Fourier) were designed against the L1 baseline (92.63). They now need to beat **82.77**. When they come in for review, if they beat their own baseline but not 82.77, send back for a rebase onto the FiLM+L1 advisor.

## Potential next research directions

1. **Pre-block FiLM** — condition attention Q/K/V on Re (before block, not after). **Assigned → thorfinn PR #909.**
2. **Re-stratified batch sampling** — Re-diverse mini-batches to improve FiLM conditioning signal. **Assigned → nezuko PR #910.**
3. **RevIN output normalization** — per-sample amplitude normalization of y before loss (targets 10× intra-split y_std variation across Re). **Assigned → askeladd PR #884.**
4. **Volume L1** — mirror surface L1 mechanism on the volume loss (currently MSE). **Assigned → fern PR #902.**
5. **surf_weight rebalancing** — rebalance surface/volume gradient ratio after FiLM merge. **Assigned → tanjiro PR #869.** May need re-sweep against new FiLM+L1 baseline.
6. **Capacity scaling on FiLM+L1 base** — the model uses only 44.6GB of 96GB. 2× hidden size on the new best baseline is a fresh, high-EV direction.
7. **Per-channel L1 on p only, MSE on Ux/Uy** — tanjiro follow-up; assign after #869 result.
8. **Low-rank slice attention (LRSA)** — replace S×S slice-token self-attention with rank-16 factored. High EV, higher complexity.
9. **Compound: FiLM + channel weighting + RevIN + Re-stratify** — if any 3+ win independently, round-4 stack.
