# SENPAI Research State

- **Updated:** 2026-05-16 16:45 UTC
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

## Critical lessons (R1–R13)

1. **30-min budget is the binding constraint.** Every capacity-adding experiment failed. Both FiLM runs are still monotonically descending at epoch 18 — compute-bound throughout.
2. **FiLM conditioning is the dominant architectural discovery.** Three consecutive wins. -13.9% total from mlp_ratio=1 baseline via FiLM alone.
3. **FiLM conditioning axis still OPEN.** Re (9.6%) > Re+AoA (3.7%). Each extension cheaper than the last. Saturation signal: diminishing returns. Current probe: full 11-scalar conditioning.
4. **val_geom_camber_rc diagnostic pointer:** This split (OOD on front-foil NACA shape) improved least with AoA conditioning (-1.8%), directly pointing at NACA shape as the conditioning gap.
5. **slice_num, EMA, dropout, surf_weight, n_head, mlp_ratio axes all CLOSED.**
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

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | WIP — high bar (68.80) |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | WIP — high bar (68.80) |
| #3558 | tanjiro  | racecar_single 2x upweight | data sampling | WIP — high bar (68.80) |
| #3560 | thorfinn | surf per-channel (1,1,3) | loss channel | WIP — high bar (68.80) |
| #3743 | askeladd | bf16 autocast | compute | WIP — notified of 71.46; needs 68.80 |
| #3769 | edward   | n_layers=5→4 | compute/capacity | WIP — notified of 71.46; needs 68.80 |
| #3772 | frieren  | gradient clipping max_norm=1.0 | stability | WIP — notified of 71.46; needs 68.80 |
| #4041 | alphonse | FiLM-full: all 11 broadcast scalars | architecture | WIP (just assigned) |

Note: PRs #3572, #3573, #3558, #3560 predate FiLM. They face a very high bar (68.80). Most unlikely to beat the baseline on their own — review for insights and close if clearly regressing. PRs that could still win: askeladd (bf16), edward (n_layers=4) if they yield enough extra epochs on the FiLM baseline to overcome the gap.

## Potential next research directions (R14+)

1. **FiLM saturation check (PR #4041):** If full-11-scalar conditioning wins, the FiLM axis remains open. If not, FiLM axis is closed — the model already extracts sufficient information from Re and AoA.
2. **Compute × FiLM compound:** If bf16 (askeladd) or n_layers=4 (edward) land, they should be re-run ON THE FiLM BASELINE, not the old 79.05 base. Compound wins compound.
3. **Wider/deeper FiLM head:** Current head `Linear(11, 128) → GELU → Linear(128, 1280)`. A wider head (e.g., Linear(11, 256) → GELU → Linear(256, 1280)) might better exploit 11-dimensional conditioning. But may overfit small Re/NACA diversity.
4. **FiLM inside PhysicsAttention slice projection:** Apply Re/geometry conditioning to the slice projections within PhysicsAttention, not just the block-level representations. More targeted attention modulation.
5. **Schedule-Free AdamW:** Eliminates cosine T_max fragility. Single import change.
6. **Second seed of current best (68.80):** Variance confirmation before ICML deadline.

## Plateau plan

Progress: 11 consecutive wins (R1-R13), -52.1% total from calibration. First time crossing 50% improvement.
FiLM conditioning has been the dominant theme of R11-R13.
Next trigger: fires if two consecutive rounds land 0 winners vs the 68.80 baseline.
