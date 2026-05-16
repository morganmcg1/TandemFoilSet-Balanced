# SENPAI Research State

- **Updated:** 2026-05-16 17:30 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 68.80**, **test_avg/mae_surf_p = 59.49** (PR #4018,
FiLM-Re+AoA, cond=[log_Re, AoA0, AoA1], best epoch 18).

Per-split val: single=80.63, rc=80.24, cruise=47.81, re_rand=66.50.
Per-split test: single=68.75, rc=72.54, cruise=39.27, re_rand=57.42.

## FiLM conditioning progression (active research front)

Three consecutive FiLM wins, each a one-line input-layer change:

| PR | Conditioning | val_avg | Δ | Key observation |
|----|-------------|--------:|--:|-----------------|
| #4004 | log_Re (1 scalar) | 71.46 | -9.6% | Global improvement all splits; Re dominates |
| #4018 | + AoA0, AoA1 (3 scalars) | 68.80 | -3.7% | val_geom_camber_rc gained least — NACA not yet conditioned |
| #4041 | + NACA0, NACA1, gap, stagger (11 scalars) | **WIP** | ? | Tests whether geometry conditioning helps OOD NACA splits |

**Next probe:** PR #4041 (alphonse in-flight) extends to all 11 broadcast-constant scalars. If NACA conditioning helps val_geom_camber_rc specifically, it validates the geometric conditioning axis. If not, FiLM conditioning is saturated and the axis is closed.

## R14 portfolio strategy (just dispatched, 7 fresh assignments)

The 7 stale pre-FiLM PRs were closed because their baselines were 8-15 pts behind the FiLM regime. Replaced with **7 orthogonal hypotheses all anchored on the FiLM-Re+AoA baseline (68.80)** — chosen to attack different bottlenecks simultaneously.

| Theme | Student | PR | Hypothesis | Expected mechanism |
|-------|---------|----|------------|---------------------|
| Compute | askeladd | #4064 | bf16 autocast | +4-6 epochs in 30-min budget |
| Compute | edward | #4068 | n_layers 5→4 | -20% step time → +4-5 epochs |
| Compute | nezuko | #4069 | torch.compile(dynamic=True) | -10-25% step time, kernel fusion |
| Optim | fern | #4071 | Schedule-Free AdamW | Removes T_max fragility, internal averaging |
| FiLM capacity | tanjiro | #4072 | Wider FiLM head 128→256 | Probe conditioning bottleneck |
| EMA | thorfinn | #4073 | decay 0.997→0.995 | Looser window matches FiLM dynamics |
| Normalization | frieren | #4077 | RMSNorm replaces LayerNorm | -1 op per norm, drop-in for LayerNorm |
| Architecture | alphonse | #4041 | FiLM-full (11 scalars) | Probe NACA conditioning gap |

Three compute experiments simultaneously (bf16, n_layers=4, torch.compile) — these may stack later if any wins. The optim/EMA/norm levers test orthogonal axes. FiLM-full and Wider-FiLM-head test the FiLM saturation hypothesis from two angles (more dims vs. more head capacity).

## Critical lessons (R1–R13)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Both FiLM runs are still monotonically descending at epoch 18 — compute-bound throughout. **Most current dispatches attack this directly.**
2. **FiLM conditioning is the dominant architectural discovery.** Three consecutive wins. -13.9% total from mlp_ratio=1 baseline via FiLM alone.
3. **FiLM conditioning axis still OPEN.** Re (9.6%) > Re+AoA (3.7%). Each extension cheaper than the last. Saturation signal: diminishing returns. Current probes: full 11-scalar conditioning (#4041) and wider FiLM head (#4072).
4. **val_geom_camber_rc diagnostic pointer:** This split (OOD on front-foil NACA shape) improved least with AoA conditioning (-1.8%), directly pointing at NACA shape as the conditioning gap.
5. **slice_num, dropout, surf_weight, n_head, mlp_ratio axes all CLOSED.**
6. **Identity FiLM init is critical pattern:** `nn.init.zeros_(film_head[-1].weight/bias)` ensures epoch-0 = baseline, allowing safe zero-risk experiments.

## Round wins merged (R1–R13)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% | MERGED |
| #3400 | SmoothL1 beta=0.25 | 97.15 | -1.32% | MERGED |
| #3402 | dropout=0.1 | 96.17 | -1.01% | MERGED |
| #3533 | slice_num=64→32 | 90.58 | -5.81% | MERGED |
| #3602 | slice_num=32→16 | 84.44 | -6.78% | MERGED |
| #3601 | EMA 0.999→0.998 | 81.16 | -3.88% | MERGED |
| #3783 | EMA 0.998→0.997 | 80.88 | -0.34% | MERGED |
| #3950 | slice_num 16→12 | 80.60 | -0.34% | MERGED |
| #3982 | mlp_ratio 2→1 | 79.05 | -1.92% | MERGED |
| #4004 | FiLM-on-Re | 71.46 | -9.6% | MERGED |
| #4018 | FiLM-Re+AoA | **68.80** | **-3.7%** | **MERGED — current baseline** |

**Total improvement from calibration baseline:** 143.52 → 68.80 = **-52.1%**

## Currently in flight (8 WIP — all students active on FiLM baseline)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4041 | alphonse  | FiLM-full: all 11 broadcast scalars | architecture | WIP |
| #4064 | askeladd  | bf16 autocast on FiLM | compute | WIP |
| #4068 | edward    | n_layers 5→4 on FiLM | compute | WIP |
| #4069 | nezuko    | torch.compile(dynamic=True) | compute | WIP |
| #4071 | fern      | Schedule-Free AdamW | optim | WIP |
| #4072 | tanjiro   | Wider FiLM head 128→256 | FiLM capacity | WIP |
| #4073 | thorfinn  | EMA decay 0.997→0.995 | EMA | WIP |
| #4077 | frieren   | RMSNorm replaces LayerNorm | normalization | WIP |

**Closed (stale, pre-FiLM baselines):** #3558, #3560, #3572, #3573, #3743, #3769, #3772 — all replaced with FiLM-aligned hypotheses above.

## Potential next research directions (R15+)

1. **Compound the winners.** If multiple of {bf16, n_layers=4, torch.compile} land, they should be merged sequentially and the next round runs on the compound baseline. Compute savings compound linearly with epochs.
2. **FiLM × NACA shape conditioning.** If PR #4041 wins, the next probe is per-NACA expert routing (mixture of FiLM heads conditioned on NACA shape clusters).
3. **FiLM inside PhysicsAttention slice projection.** Apply Re/geometry conditioning to the slice projections within PhysicsAttention, not just the block-level representations. More targeted attention modulation.
4. **Cross-block FiLM with shared parameters.** Currently each block gets its own (γ,β). Try sharing the FiLM head across blocks (cheaper, may regularize).
5. **Second seed of current best (68.80).** Variance confirmation before ICML deadline.
6. **GEGLU/SwiGLU FFN.** mlp_ratio=1 is closed at FFN width axis, but FFN nonlinearity is unexplored.

## Plateau plan

Progress: 11 consecutive wins (R1-R13), -52.1% total from calibration. First time crossing 50% improvement.
FiLM conditioning has been the dominant theme of R11-R13. R14 attacks 8 orthogonal axes simultaneously to find what stacks on top of FiLM.
Next trigger: fires if two consecutive rounds land 0 winners vs the 68.80 baseline.
