# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 06:00 (round-17: MERGED #1846 slice_num=32 — **7th winner, new baseline 54.0051** (-9.30% vs L1, -4.59% vs sampler+L1; uniform improvement all 4 splits). Closed #1870 nezuko sampler-both-racecar (+8.77% worse; joint-boost axis closed) and #1871 thorfinn surf-p-weight-2x (+4.59% worse; per-channel surf reweighting axis closed). Assigned #1903 frieren slice_num=16, #1904 nezuko sampler 1.5×, #1905 thorfinn SGDR warm restarts.)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None on this branch.

## Current research focus

**7 merged winners → baseline 54.0051 (-51% from 110.76 at round-1 start).**

New baseline = L1 + compile + bf16 + sampler 2× single + **slice_num=32**.

Key finding from round 17: **slice_num=32 is the largest single-PR win of round 5** (-9.30%). Tighter attention bottleneck regularizes the model globally — ALL 4 splits improved uniformly ~9%. First run where best ≠ terminal (model now converges within budget). Mechanism: forcing mesh routing to 32 slices instead of 64 matches the natural ~10-20 CFD spatial regime count, and reduces per-epoch compute ~12%.

**Priority axes:**
1. **Slice_num bracket (#1903 frieren)** — does 16 beat 32? Most informative next architecture step.
2. **Optimizer/regularization stack** (#1653 askeladd grad-clip, #1775 fern WD=5e-5, #1845 edward betas, #1774 alphonse lr — all proven on older baselines, need rebase/retest)
3. **Schedule** (#1905 thorfinn SGDR warm restarts)
4. **Sampler peak-bracket** (#1904 nezuko 1.5×)
5. **Architecture variant** (#1883 tanjiro n_head=8)

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

> Advisor config: Pure L1 + bf16 AMP + torch.compile(dynamic=True) + scoring-NaN workaround + sampler 2× racecar_single + **slice_num=32**.
> ~40-43 epochs in 30 min (~43.5 s/epoch vs 49.6 s at slice_num=64). Peak GPU: ~21 GB.
> **Model now converges within budget** (best_epoch=40, terminal=41 in last run).

## In-flight (WIP)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1653 | askeladd | Grad clip max_norm=1.0 (L1 rebase) | Proven -6.94% on β=0.5. Will need rebase onto new slice_num=32 baseline if it beats the threshold |
| #1775 | fern | WD=5e-5 (L1 rebase) | Proven -4.43% on β=0.5. Same rebase caveat |
| #1774 | alphonse | lr=7.5e-4 (L1 rebase) | May beat old baseline, need new rebase onto 54.00 |
| #1845 | edward | AdamW betas=(0.9, 0.95) | On L1-only base; if result ≈56-59, needs rebase onto slice_num=32 |
| #1883 | tanjiro | n_head 4 → 8 | Last architecture axis; on new 56.62 sampler baseline |
| #1903 | frieren | slice_num 32 → 16 | **New round-17** — bracket the optimum |
| #1904 | nezuko | Sampler racecar_single 1.5× | **New round-17** — 2× peak bracket |
| #1905 | thorfinn | Cosine warm restarts T_0=10 T_mult=2 | **New round-17** — SGDR schedule |

## Warning on in-flight rebase

Many in-flight PRs were assigned on L1-only baseline (59.54). The current baseline is 54.0051 (L1+sampler+slice_num=32). If they come in between 54.00 and 59.54, they'll need a rebase onto the new advisor. Only merge if they beat 54.0051.

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
- **slice_num=96:** worse than 64 (PR #1590).
- **AdamW β2=0.95:** near-wash on β=0.5, no clear signal (PR #1676).

## Open questions / next experiments

1. **slice_num=16** — in flight (#1903). Answers the optimum question cleanly.
2. **n_head=8** — in flight (#1883). Last architecture axis.
3. **SGDR warm restarts** — in flight (#1905). May unlock model-settling gains.
4. **Proven-lever stack on new baseline** — grad-clip (#1653), WD=5e-5 (#1775), betas (#1845) all need rebasing. If these stack independently, each -2 to -4%.
5. **AoA reflection augmentation** — never tested. Physics symmetry for doubly-symmetric foils; could be powerful for OOD.
6. **Lookahead optimizer** — slow/fast weight averaging; untested.
7. **Layer normalization variant** — RMSNorm or Pre-norm placement; untested.
