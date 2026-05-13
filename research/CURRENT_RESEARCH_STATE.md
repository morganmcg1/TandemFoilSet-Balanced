# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 12:00 (round-26: Closed #1653 askeladd grad-clip WASH — monotone lever decay across 3 baselines: β=1.0 −14.92% → β=0.5 −6.94% → L1+slice=32 +0.56% mean; bimodal per-split with OOD regression confirms gradient-coherence axis saturated by upstream L1/sampler changes. Assigned #2051 askeladd Lookahead(k=5,α=0.5). **Baseline still 54.0051**)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None on this branch.

## Current research focus

**7 merged winners → baseline 54.0051 (-51% from 110.76 at round-1 start).**

New baseline = L1 + compile + bf16 + sampler 2× single + slice_num=32.

**Round-18 findings:**
- **slice_num-DOWN axis closed**: 16 is a val wash (+0.41%) with an in-dist/OOD split trade-off. 32 is the global optimum.
- **Sampler axis closed**: 2.0× single is the confirmed peak. 1.5× (+3.47%) is strictly worse. No further sampler tuning needed.
- **New bottleneck identified**: `val_geom_camber_rc` (72.37) and `val_re_rand` (58.23) now dominate val_avg. Neither responds to sampler reweighting. **OOD generalization** is the highest-leverage remaining research axis.

**Shifted priority to OOD generalization levers:**
1. **Input augmentation** (#1921 nezuko pos-jitter) — targeting mesh-coord overfitting as source of OOD brittleness
2. **Normalization variant** (#1926 frieren rmsnorm) — RMSNorm as structural complement to routing changes
3. **Optimizer/regularization stack** (#1653 askeladd grad-clip, #1775 fern WD=5e-5, #1845 edward betas, #1774 alphonse lr — all on L1 or Huber baselines, need rebase to confirm still useful on 54.00 baseline)
4. **Schedule** (#1905 thorfinn SGDR warm restarts)
5. **Architecture (n_head=8)** — tanjiro pod stale × 3; n_head axis deferred until pod recovers or another student is available. DropPath (#1976) is the current architecture-regularization experiment.

## Merged winners

| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|---|
| #1846 ✓ | frieren | slice_num 64 → 32 | **54.0051** | **47.6261** |
| #1619 ✓ | nezuko | Sampler 2× racecar_single | 56.62 | 50.43 |
| #1700 ✓ | thorfinn | Pure L1 loss | 59.54 | 51.47 |
| #1633 ✓ | thorfinn | Huber β=0.5 | 64.07 | 55.50 |
| #1568 ✓ | thorfinn | torch.compile + bf16 | 69.83 | 61.87 |
| #1532 ✓ | thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 |
| #1444 ✓ | thorfinn | MSE → Huber β=1.0 | 110.76 | NaN |

**Current baseline: val_avg/mae_surf_p = 54.0051, test_avg/mae_surf_p = 47.6261 (PR #1846)**

Per-split baseline (PR #1846):

| Split | val mae_surf_p |
|---|---|
| `val_single_in_dist` | 59.0943 |
| `val_geom_camber_rc` | 67.4450 |
| `val_geom_camber_cruise` | 35.7197 |
| `val_re_rand` | 53.7616 |

> Advisor config: Pure L1 + bf16 AMP + torch.compile(dynamic=True) + scoring-NaN workaround + sampler 2× racecar_single + slice_num=32.
> ~40-43 epochs in 30 min (~43.5 s/epoch vs 49.6 s at slice_num=64). Peak GPU: ~21 GB.
> **Model convergence window narrowing** (best_epoch=40 in #1846 run; re-opened to best=terminal in recent runs suggesting noise/variability near convergence).

## In-flight (WIP)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1988 | nezuko | Per-sample fun_dim jitter Re/AoA — **retune σ=0.025** | **Round-24 send-back** — σ=0.05 LOSS (+11.9%); σ=0.025 probe for clean axis closure |
| #2034 | frieren | RMSNorm replacing LayerNorm (all 3 sites) | **Round-25 retry** — RMSNorm hypothesis; resubmit of stale #1926 |
| #2033 | thorfinn | Linear warmup 3ep + monotone cosine (T_max=47) | **Round-25** — schedule axis follow-up to closed #1989 SGDR; captures exploration without restart disruption |
| #1976 | tanjiro | DropPath p_max=0.1 stochastic depth | **New round-21** — OOD generalization via block-level residual-branch regularization |
| #1946 | edward | EMA decay=0.999 — **drop diagnostic, full-budget rerun** | **Round-24 send-back** — mechanism confirmed by dual-eval (EMA<raw from ep10); test tied at 47.60. Need full 50-epoch budget. |
| #1775 | fern | WD=5e-5 | Proven -4.43% on β=0.5; needs rebase onto 54.00 |
| #1997 | alphonse | lr 5e-4 → 3.75e-4 (-25%) | **Round-23** — capacity↔LR coupling DOWN probe; follow-up to closed #1774 |
| #2051 | askeladd | Lookahead(k=5, α=0.5) wrapping AdamW | **Round-26** — slow/fast weight averaging; flat-minima bias hypothesis for OOD generalization |

## Warning on in-flight rebase

Most long-running in-flight PRs (#1775, #1845, #1883) were assigned on baselines of 56.62-64.07. The current baseline is 54.0051. Only merge if they beat 54.0051.

## Closed axes (comprehensive)

- **Capacity:** width (n_hidden=160), depth (n_layers=6), FFN-only (mlp_ratio=3) — all close, budget-bound.
- **Attention dropout (per-weight):** p=0.1 — convergence cost > benefit at 30-min cap.
- **Batch=8:** step-count starvation.
- **WD up (5e-4):** under-fits on short budget.
- **Warmup:** substituted by β sharpening.
- **LR=1e-3 + warmup:** overshoot.
- **Surf_weight 10→30:** vol-surf imbalance.
- **Per-channel loss [1,1,3] global:** distorts velocity (PR #1428).
- **Per-channel surf_loss [1,1,2] surf-only:** OOD regression (PR #1871) — same physics coupling.
- **LR floor (eta_min=5e-5):** removes step-damping for L1 sign gradients (PR #1826).
- **Sampler both-racecar 2×:** absolute single exposure drops (PR #1870).
- **Sampler racecar_single 1.5×:** under-concentrates single coverage (PR #1904). **2× is the confirmed optimum.**
- **slice_num=96:** worse than 64 (PR #1590).
- **slice_num=16:** val wash (+0.41%) with in-dist/OOD trade-off; 32 is the global optimum (PR #1903). **slice_num axis fully closed.**
- **AdamW β2=0.95:** LOSS on L1 (+4.42% on L1 base, +15% vs current). Shorter second-moment memory amplifies L1 sign-flip noise (PR #1845). β2 axis closed.
- **Grad-clip max_norm=1.0:** WASH on L1+sampler+slice=32 (best run −0.37%, mean +0.56%; n=2 straddles baseline). Monotone dose-response: β=1.0 −14.92% → β=0.5 −6.94% → L1+slice=32 ~0%. Bimodal per-split: in-dist −13.33% but OOD +3-5% — structurally wrong for primary bottleneck. Gradient-coherence axis saturated by upstream L1+sampler changes (PR #1653). Gradient-coherence axis closed.
- **AdamW β2=0.95 (earlier β=0.5 test):** near-wash on β=0.5, no clear signal (PR #1676).
- **lr=7.5e-4 (+50% lr-UP):** LOSS on L1+slice=32 (+16% vs current, n=3 mean 62.66) — closed across all 3 landscape variants tested (β=0.5 wash, L1+slice=64 wash-with-loss-tail, L1+slice=32 LOSS) (PR #1774). lr-UP axis closed; capacity↔LR coupling DOWN probe in flight as #1997.
- **SGDR T_0=10 T_mult=2:** LOSS on L1+slice=32 (+27.7% val, n=1, 68.96 vs 54.00). Mechanism worked at cycle level (restart #2 min < restart #1 min) but L1 sign-gradient regime is fundamentally hostile to LR restarts (signs reset, late settling destroyed). Budget-fit variants deferred — L1+restart is the binding incompatibility (PR #1989). Schedule axis continues with warmup+monotone-cosine in flight as #2033.

## Open questions / next experiments

1. **OOD generalization** — primary bottleneck. Both `val_geom_camber_rc` (67.45) and `val_re_rand` (53.76) are 30-40% higher than geom_camber_cruise. Do NOT respond to sampler changes.
   - **Coord-jitter (#1921) CLOSED (LOSS):** -13.7% in-dist, but all OOD splits degraded. Spatial precision is load-bearing for camber inference.
   - **Fun-jitter Re/AoA σ=0.05 (#1988) LOSS at this magnitude:** +13.6% on targeted val_re_rand. σ=0.025 probe in flight to close the σ-sweep cleanly.
   - **In-dist headroom (~14%) is unlockable but does NOT translate to OOD:** pos-jitter (-13.7%) and EMA (-12.9%) both give big val_single_in_dist wins; neither helps OOD. The in-dist→OOD compounding doesn't exist with simple regularization.
   - Untried structural OOD interventions: domain conditional embedding (architectural), Lookahead optimizer (slow/fast trajectory), AoA reflection.
2. **Normalization** — RMSNorm in flight (#1926). Pre-LN vs Post-LN placement also untested.
3. **Proven-lever stack on new baseline** — grad-clip (#1653), WD=5e-5 (#1775) all need rebasing to current 54.00 baseline. β2 axis is now closed (PR #1845).
4. **Schedule** — SGDR in flight (#1905). Warm restarts may unlock multi-descent gains.
5. **Lookahead optimizer** — slow/fast weight averaging; **in-flight as #2051 askeladd (round-26).** Standard k=5, α=0.5 recipe (Zhang et al. 2019). Hypothesis: implicit flat-minima bias from α-averaging should improve OOD more than in-dist (inverse of grad-clip's pattern).
6. **Stochastic depth (DropPath)** — block-level residual regularization; different from per-weight attention dropout (closed). Untested.
7. **EMA model weights for val/test eval** — in flight (#1946 edward). First try decay=0.9999 lagged catastrophically (val=165); rebracket to decay=0.999 (~2-epoch half-life) is correct window for our budget. Result expected next round.
