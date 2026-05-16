# SENPAI Research State

- **Updated:** 2026-05-16 20:00 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 59.08**, **test_avg/mae_surf_p = 51.29** (PR #4064,
bf16 autocast on FiLM-Re+AoA, best epoch 25, still descending at termination).

Per-split val: single=69.49, rc=68.90, cruise=40.32, re_rand=57.60.
Per-split test: single=60.89, rc=63.00, cruise=32.91, re_rand=48.38.

**Total improvement from calibration baseline:** 143.52 → 59.08 = **-58.8%**

## Round wins merged (R1–R14)

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
| #4018 | FiLM-Re+AoA | 68.80 | -3.7% | MERGED |
| **#4064** | **bf16 autocast** | **59.08** | **-14.1%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=12**, mlp_ratio=1 |
| Conditioning | **FiLM head [log_Re, AoA0, AoA1]** (3-scalar → per-block γ,β) |
| Precision | **bf16 autocast** (forward + loss; reductions in fp32) |
| Optim | AdamW, lr=5e-4, weight_decay=1e-4, batch=4, cosine T_max=50 |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint |
| Compute | ~74s/epoch, **25 epochs** in 30-min cap |

## Dominant discovery: compute is the binding constraint

bf16 autocast's -14.1% win came entirely from 7 extra epochs (25 vs 18). The model is STILL descending at the terminal epoch — every extra epoch worth ~1.4 pts. The research program priority is: **give the model more effective training time**.

Three orthogonal ways to get more training:
1. **Fewer seconds per epoch** (bf16 done; torch.compile → nezuko; slice_num=8 → tanjiro)
2. **More informative gradient per step** (batch=8 → askeladd; lr=7.5e-4 → thorfinn)
3. **More expressive per-step update** (GEGLU FFN → frieren; FiLM-two-stage → alphonse)

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4041 v2 | alphonse  | FiLM two-stage (base+geom, is_tandem gate) | FiLM architecture | WIP |
| #4069 rebased | nezuko | torch.compile on bf16 baseline | compute | WIP (sent back to rebase+retest) |
| #4104 | askeladd  | batch 4→8 on bf16 | data parallelism | WIP |
| #4068 | edward    | n_layers 5→4 on bf16 | compute | WIP (old baseline — compare carefully) |
| #4071 | fern      | Schedule-Free AdamW on bf16 | optim | WIP (old baseline — compare carefully) |
| #4105 | frieren   | GEGLU FFN replaces GELU | FFN nonlinearity | WIP |
| #4107 | tanjiro   | slice_num 12→8 on bf16 | compute | WIP |
| #4109 | thorfinn  | lr 5e-4→7.5e-4 on bf16 | optim | WIP |

**Note on edward (#4068) and fern (#4071):** These were assigned against old baseline (68.80). If their results beat 68.80 but not 59.08, they get sent back to re-run on the bf16 merged baseline — the compute wins are orthogonal.

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is the optimum |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck for 3 scalars |
| slice_num (16→12) | MERGED as tiny win — probe 12→8 next |
| mlp_ratio | CLOSED at 1 (width not the lever) |
| dropout (0.1) | CLOSED |
| n_head (4) | CLOSED |
| surf_weight (10.0) | CLOSED |
| FiLM-full naive (11 scalars) | SENT BACK — structural zeros for single-foil; v2 (two-stage) in flight |

## Potential next research directions (post-R14)

1. **Compound compute wins**: if nezuko (torch.compile on bf16) wins, we'll have 28+ epochs. Stack with askeladd batch=8 if both win — potential for ~30-33 epochs.
2. **FiLM two-stage confirmation**: if alphonse v2 wins (is_tandem gating fixes single-foil regression), FiLM conditioning axis reopens with 8-12 new scalars.
3. **n_layers 5→4**: edward's result (if it wins against old baseline) should be re-run on bf16 baseline — n_layers=4 + bf16 may compound.
4. **Schedule-Free AdamW**: fern's result may have interesting optimizer dynamics in longer training regime (25 epochs vs 18).
5. **LR sweep**: thorfinn's lr=7.5e-4 result will calibrate the optimal LR for the 25-epoch bf16 regime. Follow-up: cosine T_max aligned to actual epochs reached.
6. **Multi-seed confirmation**: val_avg variance ±5-10 pts at single seed. Before ICML deadline, run 3 seeds of the best config.
