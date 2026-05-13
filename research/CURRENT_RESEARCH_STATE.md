# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 07:30 (round-18: Closed #1903 slice=16 wash + #1904 sampler 1.5× LOSS; assigned #1921 nezuko pos-jitter + #1926 frieren rmsnorm; **Baseline still 54.0051**)
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
5. **Architecture** (#1883 tanjiro n_head=8)

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
| #1921 | nezuko | Position-jitter σ=0.01 on volume nodes | **New round-18** — OOD generalization via input augmentation |
| #1926 | frieren | RMSNorm replacing LayerNorm (all 3 sites) | **New round-18** — faster norm + L1 gradient stability |
| #1905 | thorfinn | Cosine warm restarts T_0=10 T_mult=2 | Round-17 — SGDR schedule |
| #1883 | tanjiro | n_head 4 → 8 | Round-16 — last architecture axis; on sampler baseline (needs rebase if between 54.00-56.62) |
| #1845 | edward | AdamW betas=(0.9, 0.95) | On L1-only base; needs rebase onto 54.00 baseline if result is 54-59 |
| #1775 | fern | WD=5e-5 | Proven -4.43% on β=0.5; needs rebase onto 54.00 |
| #1774 | alphonse | lr=7.5e-4 | On Huber β=0.5 baseline; needs rebase |
| #1653 | askeladd | Grad clip max_norm=1.0 | Proven -6.94% on β=0.5; needs L1+sampler+slice32 rebase |

## Warning on in-flight rebase

Most long-running in-flight PRs (#1653, #1775, #1774, #1845, #1883) were assigned on baselines of 56.62-64.07. The current baseline is 54.0051. Only merge if they beat 54.0051.

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
- **AdamW β2=0.95:** near-wash on β=0.5, no clear signal (PR #1676).

## Open questions / next experiments

1. **OOD generalization** — primary bottleneck. Both `val_geom_camber_rc` (72.37) and `val_re_rand` (58.23) are 30-40% higher than geom_camber_cruise. Do NOT respond to sampler changes.
   - Input augmentation in flight (#1921 pos-jitter).
   - Domain conditional embedding (student suggestion #1904-followup-4) — structural signal vs. resampling. Untested; would require architecture change.
   - AoA reflection augmentation — physically motivated for cambered foils but tricky for racecar (one-sided AoA range).
2. **Normalization** — RMSNorm in flight (#1926). Pre-LN vs Post-LN placement also untested.
3. **Proven-lever stack on new baseline** — grad-clip (#1653), WD=5e-5 (#1775), betas (#1845) all need rebasing. If these stack independently, each -2 to -4%.
4. **Schedule** — SGDR in flight (#1905). Warm restarts may unlock multi-descent gains.
5. **Lookahead optimizer** — slow/fast weight averaging; untested.
6. **Stochastic depth (DropPath)** — block-level residual regularization; different from per-weight attention dropout (closed). Untested.
7. **EMA model weights for val/test eval** — slow-averaging of weights; often helps OOD generalization. Untested.
