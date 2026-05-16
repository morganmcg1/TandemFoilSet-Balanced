# SENPAI Research State

- **Updated:** 2026-05-16 02:35 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 90.58**, **test_avg/mae_surf_p = 81.25** (PR #3533,
slice_num=32, single-seed, best epoch 16).

Per-split val: single=108.12, rc=104.27, cruise=66.77, re_rand=83.15.
Per-split test: (not individually recorded in PR — test_avg=81.25 confirmed).

**Key gap:** `val_single_in_dist=108.12` remains the persistent worst split (+3.7%
above next-worst rc=104.27). slice_num=32 improved single by -7.21%, but it still
leads the degradation. Three in-flight experiments targeting it indirectly:
- edward #3554: wd=5e-4 (regularization; previously showed -4 to -9% on single)
- tanjiro #3558: racecar_single 2x upweight in sampler
- thorfinn #3560: surf per-channel (Ux,Uy,p)=(1,1,3) pressure 3x gradient budget

Single-seed variance ≈ ±5-10 pts. Require ≥3% improvement
(val_avg ≤ ~87.9) for a clear winner, or ≥1% with all-splits consistency.

## Critical lessons (R1–R6)

1. **30-min budget is the binding constraint** — n_hidden=160, mlp_ratio=4, stoch-depth, n_hidden=96 all failed because they changed per-step cost without proportional benefit. No capacity-adding experiments until compute-efficiency win.
2. **slice_num is the key lever** — Reducing 64→32 gave -5.81%, the biggest win since SmoothL1. Two mechanisms compound: O(K²) attention cost reduction (more epochs) + implicit regularization (coarser slices). **Now probing slice_num=16.**
3. **Dropout saturated at 0.1** — dropout=0.2 fails (+47% train loss). Axis closed.
4. **surf_weight saturated at 10** — surf_weight=15 within noise. Axis closed.
5. **EMA window must match training budget** — EMA-0.9995 (window ~2000 steps) was catastrophic (+34.5%); spans entire training run including high-loss early epochs. EMA-0.999 (window ~1000 steps) is current best. Probing 0.998 (loosening direction).
6. **LR schedule effectively flat** — CosineAnnealingLR(T_max=50) over 16 actual epochs → LR only drops from 5e-4 to 4.0e-4. Model never sees low-LR fine-tuning phase. **Probing T_max=16 (matched to budget).**

## Round wins merged (R1–R6)

| PR | Hypothesis | val_avg | Δ vs prior | Decision |
|----|------------|--------:|-----:|----------|
| #3280 | SmoothL1 beta=0.5 | 98.45 | -5.81% vs 104.52 | MERGED |
| #3400 | SmoothL1 beta=0.25 (2-seed mean) | 97.15 | -1.32% vs 98.45 | MERGED |
| #3402 | dropout=0.1 in PhysicsAttention (8/8 consistency) | 96.17 | -1.01% vs 97.15 | MERGED |
| #3533 | slice_num=64→32 (halve attention cost) | **90.58** | **-5.81% vs 96.17** | **MERGED — current baseline** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Round |
|----|---------|------------|-------|-------|
| #3572 | nezuko   | CosineWarmRestarts T_0=4 T_mult=2 | LR schedule | R6 |
| #3573 | fern     | lr 5e-4→7e-4 (2-seed) | optim | R6 |
| #3554 | edward   | weight_decay=5e-4 (2-seed, on dropout=0.1 base) | regularization | R6 |
| #3558 | tanjiro  | racecar_single 2x upweight in sampler | data sampling | R6 |
| #3560 | thorfinn | surf per-channel (Ux,Uy,p)=(1,1,3) | loss channel | R6 |
| #3601 | alphonse | EMA decay 0.999→0.998 (looser window) | EMA tuning | R7 |
| #3602 | askeladd | slice_num=32→16 (continue winning axis) | compute budget | R7 |
| #3603 | frieren  | CosineAnnealingLR T_max=50→16 (budget-matched) | LR schedule | R7 |

**Note:** All in-flight students have been notified of the new baseline (val=90.58). Students with in-flight R6 PRs have been asked to rebase onto the updated advisor branch (slice_num=32 now merged).

## Plateau / saturation map (R1–R7)

**CLOSED axes:**
- **Loss formulation (Huber beta):** saturated at beta=0.25.
- **Throughput / batch size:** H100 memory-bandwidth-bound.
- **Schedule (cosine T_max shrink, pre-slice_num=32):** closed vs old base. Re-probing with matched T_max=16 on new base.
- **Cyclic AoA encoding:** AoA range too narrow. Closed.
- **Capacity up (n_hidden=160, mlp_ratio=4):** timeout-bound. Closed.
- **Capacity down (n_hidden=96):** only 4% per-epoch speedup, +1.50% val regression. Triangulated: n_hidden=128 is optimal.
- **Stochastic depth:** 5-block × 14-epoch budget is wrong regime. Closed.
- **Dropout:** saturated at 0.1. 0.2 underfits (+47% train loss). Closed.
- **surf_weight:** saturated at 10. surf_weight=15 within noise (2-seed). Closed.
- **EMA tighter (0.9995):** catastrophic +34.5% regression. Window too large for training budget. Closed from tighter direction.

**OPEN axes (in-flight):**
- **slice_num (16):** askeladd #3602 — does the improvement continue past 32?
- **LR schedule (WarmRestarts):** nezuko #3572 — multi-cycle LR within 16 epochs.
- **LR schedule (matched T_max):** frieren #3603 — cosine T_max=16 on new budget.
- **LR value:** fern #3573 — first probe at lr=7e-4 (+40%).
- **Regularization (weight_decay):** edward #3554 — wd=5e-4 on dropout=0.1 base, 2-seed.
- **EMA looser (0.998):** alphonse #3601 — shorter window, avoids early-epoch bias.
- **Data sampling (single-foil boost):** tanjiro #3558 — racecar_single 2x upweight.
- **Loss channel weighting:** thorfinn #3560 — (1,1,3) pressure 3x gradient.

## Potential next research directions (R8+)

### After R7 results land

1. **Compound winners** — stack best orthogonal wins from R6+R7. slice_num=32 + matched cosine + any optim win.
2. **slice_num=24** — interpolate between 16 and 32 if slice_num=16 regresses (bracket the optimum).
3. **Re-conditioned FiLM** — log(Re) → (γ, β) per Transolver block. Architecturally novel cross-regime conditioning. Targets val_re_rand and single-foil OOD.
4. **LR search continuation** — if lr=7e-4 wins, try 1e-3. If it regresses, try 6e-4.
5. **Schedule-Free AdamW** (Defazio 2024) — eliminates LR scheduling entirely; good fit for noisy short-run training.
6. **n_layers=4** — drop one block, save compute, fit more epochs. With slice_num=32 already saving attention cost, the FFN blocks become a bigger share of per-epoch time.

### If R7 plateaus (0 wins)

Escalation: move to architecturally different approaches.
- FiLM conditioning per block (log Re / geometry conditioning).
- n_layers=4 (drop one block, save compute, fit more epochs).
- Dual surface/volume decoder heads.
- Completely different architecture (point transformer, GNO).

## Plateau plan

Progress: ~5.81% in R6 (largest since SmoothL1). Current streak reset.
Next trigger fires if R7 AND R8 both land 0 winners vs the 90.58 baseline.
