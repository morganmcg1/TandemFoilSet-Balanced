# SENPAI Research State — TandemFoilSet (willow-pai2i-24h-r4)

- **As of:** 2026-05-16 10:35 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r4`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced`
- **W&B:** `wandb-applied-ai-team/senpai-v1`
- **Most recent human researcher direction:** None recorded. Launch isolation rules are in force.

## Research programme summary

Predict (Ux, Uy, p) at every node of unstructured 2D CFD meshes (74K–242K nodes) of tandem airfoils. The primary ranking metric is `test_avg/mae_surf_p` — equal-weighted surface pressure MAE across four test splits. Per-run budget is 30 min wall clock × 50 epochs hard cap.

## Current best (BASELINE.md)

**PR #3504 (frieren, merged 2026-05-16 ~08:30 UTC) — Richer FiLM cond_dim=11, film_mid=128**
- `val_avg/mae_surf_p = 67.2955` (W&B `v38snhoe`)
- `test_avg/mae_surf_p = **59.29**` (4-split finite)
- Per-split test: single=69.69, rc=74.16, cruise=36.63, re_rand=56.69

Cumulative path: vanilla 106.23 → #3257 94.35 → #3263 90.06 → #3358 80.08 → #3262 69.27 → #3258 66.87 → #3504 **59.29** (**−44.2% from vanilla in 6 PRs**).

All remaining PRs must beat **test_avg/mae_surf_p < 59.29**.

## Active R3/R4 portfolio (all 8 students WIP)

| # | Student | PR | Hypothesis | Status |
|---|---------|----|-----------|--------|
| 1 | **nezuko**  | **#3825** | Per-block FiLM: one cond_dim=11 head per block (5 heads total vs current 1) | WIP (assigned 08:50) |
| 2 | **frieren** | **#3890** | **p_channel_weight sweep {4, 5}: redirect within-surface gradient toward p** | **WIP (assigned 10:35; fresh)** |
| 3 | **thorfinn** | **#3891** | **LayerNorm on FiLM conditioning input: free-lunch scale normalization** | **WIP (assigned 10:35; fresh)** |
| 4 | tanjiro | #3781 | Re-conditioned per-sample loss reweighting (focal on Re tails, α=1.0) | WIP (assigned 07:22) |
| 5 | alphonse | #3693 | Peak LR sweep {1e-3, 2.5e-4} on full stack | WIP (training, 5h+) |
| 6 | askeladd | #3351 | EMA β=0.99 — fresh decision-point at 09:27 (rebase or request reassign) | WIP (waiting on student) |
| 7 | edward   | #3599 | RFF n_freqs=32 rerun on new baseline (sent back 09:14) | WIP (rebase in flight) |
| 8 | fern     | #3746 | Grad-clip cap sweep {10.0, 100.0} vs winning 1.0 | WIP (assigned 06:40) |

**All 8 students active.**

## R3/R4 closed/merged history

| # | Student | Hypothesis | PR | Status |
|---|---------|-----------|-----|--------|
| ✓ | frieren  | Surface MAE + p-weight 3× + NaN guard | #3257 | **MERGED (R1 winner #1)** — test=94.35 |
| ✓ | thorfinn | FiLM log(Re) conditioning | #3263 | **MERGED (R1 winner #2)** — val=100.24, test=90.06 |
| ✓ | alphonse | Cosine LR T_max=14 | #3358 | **MERGED (R2 winner #1)** — val=90.44, test=80.08 |
| ✓ | edward   | RFF σ=1.0, n_freqs=16 on (x,z) coords | #3262 | **MERGED (R2 winner #2)** — val=79.28, test=69.27 |
| ✓ | fern     | Grad-clip 1.0 + 5-epoch warmup on full stack | #3258 | **MERGED (R3 winner #1)** — val=77.65, test=66.87 (−3.47%) |
| ✓ | frieren  | Richer FiLM cond_dim=11, film_mid=128 | #3504 | **MERGED (R3 winner #2)** — val=67.30, test=59.29 (−11.34%) |
| ✗ | frieren  | Re-stratified loss (1/per_sample_y_std) | #3386 | CLOSED — failed (+1.7% regression) |
| ✗ | nezuko   | Multi-scale slice tokens | #3429 | CLOSED — equal-epoch tie with control |
| ✗ | nezuko   | Surface-biased slice routing | #3260 | CLOSED (paired −0.05%) |
| ✗ | nezuko   | Volume MAE reformulation (L1 on both) | #3550 | CLOSED — failed (+4.7% regression) |
| ✗ | nezuko   | Surface-only decoder head (parallel zero-init) | #3618 | CLOSED — wash on old base, +4.11% regression on new; output-head specialization not a missing DoF given 30× loss bias |
| ✗ | thorfinn | Per-block FiLM v2 on full RFF+cosine+FiLM stack | #3468 | CLOSED — mechanism overlap with RFF on single_in_dist |
| ✗ | tanjiro  | Transolver depth n_layers=6 (budget-limited) | #3658 | CLOSED — test=75.31 (+12.6%); −2 epochs cost > +1 block gain |
| ✗ | tanjiro  | surf_weight sweep re-run (sw=5) on FiLM+RFF base | #3406 | CLOSED — +4.17% regression (old cond_dim=1 base) |
| ✗ | tanjiro  | Huber loss delta=0.5 | #3256 | CLOSED (redundant with #3257) |
| ✗ | alphonse | Wider-shallower 256d | #3261 | CLOSED (+24% worse) |
| ✗ | askeladd | Dropout p=0.1 | #3264 | CLOSED (+6% worse) |
| ✗ | frieren  | surf_weight sweep {7, 5} | #3826 | CLOSED — sw=7 uniform regression +2.69% test; falsified "FiLM substitutes for surf_weight" hypothesis cleanly |
| ✗ | thorfinn | slice_num sweep {96, 128} | #3761 | CLOSED — both arms regression; capacity-vs-compute lost by budget (10-12 epochs vs 14, LR=0 while val still descending) |

## Standings — test_avg/mae_surf_p (lower is better)

| Rank | PR | Hypothesis | test_avg (4-split) | vs baseline | Status |
|------|----|------------|-------------------:|-------------|--------|
| **1** | **#3504 (frieren)** | **Richer FiLM cond_dim=11, film_mid=128** | **59.29** | **NEW BASELINE** | **MERGED** |
| 2 | #3258 (fern) | Grad-clip 1.0 + warmup-5 | 66.87 | +12.8% above | MERGED |
| 3 | #3262 (edward) | RFF σ=1.0, n_freqs=16 | 69.27 | +16.8% above | MERGED |
| 4 | #3358 (alphonse) | cosine T_max=14 | 80.08 | +35.0% above | MERGED |
| 5 | #3263 (thorfinn) | FiLM(log_Re) cond_dim=1 | 90.06 | +51.9% above | MERGED |
| 6 | #3257 (frieren) | Surf-MAE+p-weight 3×+NaN guard | 94.35 | +59.1% above | MERGED |

## Per-split residual errors (current baseline, test_avg/mae_surf_p=59.29)

| Split | Current MAE | Comment |
|-------|------------:|---------|
| `test_geom_camber_cruise` | 36.63 | Tandem-only; gap/stagger conditioning dominant |
| `test_re_rand` | 56.69 | Re-range sampling; log_Re conditioning helps |
| `test_single_in_dist` | 69.69 | Largest absolute error; RFF helps but ceiling remains |
| `test_geom_camber_rc` | 74.16 | Largest absolute error; AoA+NACA helps but rc still highest |

## Active R4 hypotheses and predictions (new target: test < 59.29)

- **nezuko per-block FiLM (#3825):** 5 FiLM(cond_dim=11) heads, one per Transolver block. Current base applies only block-0 conditioning; deeper blocks lose geometry/regime signal through attention transforms. Per-block re-injection should compound cruise+re_rand gains. +71K params, no extra attention FLOPs. Predicted test ~55-58 (hopeful ~55, conservative ~57-58).
- **frieren surf_weight sweep (#3826):** sw=7 primary, sw=5 gated. The 30× implicit surface bias (10×3) was set before FiLM(cond_dim=11) provided explicit geometry conditioning. With explicit conditioning, relaxing surf_weight may free volume loss to improve spatial consistency. Prior sw=5 attempt (#3406) failed on old cond_dim=1 base — different hypothesis now. Predicted test ~57-59 for sw=7, closer to 59 floor if FiLM already absorbed geometry info.
- **thorfinn slice_num sweep (#3761):** {96, 128} vs 64. single_in_dist (69.69) still biggest absolute error on new baseline, spans widest Re range. More slice tokens = more partition diversity. Predicted test ~56-59.
- **tanjiro Re-loss reweighting (#3781):** α=1.0 per-sample weight `1 + α*(log_Re_z)²`. Targets single_in_dist (69.69) and re_rand (56.69). Zero compute overhead. Predicted test ~57-59 conservative.
- **fern grad-clip cap sweep (#3746):** {10.0, 100.0} vs winning 1.0. Tests outlier batch suppression on new FiLM base where gradient landscape has changed.
- **alphonse LR sweep (#3693):** {1e-3, 2.5e-4} bracketing 5e-4. May need rebase onto new base.
- **edward RFF σ sweep (#3599):** σ {0.5, 2.0} × n_freqs {16, 32}. rc split (74.16) and single_in_dist (69.69) show potential. May need rebase.
- **askeladd EMA (#3351):** Stale, needs rebase.

## Key mechanics and constraints

- **Best epoch consistently = 14** (wall-clock limit, not 50-epoch cap). Model still improving at cutoff.
- **VRAM:** Current stack at 44.5 GiB / 46.4% of 96 GiB H100. Significant headroom.
- **NaN guard canonical:** cruise=1 skip, others=0. All active branches inherit automatically.

## Next research directions (post-R4)

1. **Per-block × wider mid sweep**: If per-block FiLM (#3825) wins, try film_mid=192 at per-block scale.
2. **Cosine T_max adjustment**: With richer FiLM merged, optimizer may need different annealing shape.
3. **n_freqs=32 on (x,z)**: Edward covering this in #3599.
4. **n_hidden=192 capacity test**: Bigger trunk — only if per-step budget allows without losing too many epochs.
5. **Loss-based per-split reweighting**: Dynamic per-split loss weights vs static surf_weight.
6. **Architecture: surface-aware slice routing**: Bias slice selection toward surface nodes in PhysicsAttention.
