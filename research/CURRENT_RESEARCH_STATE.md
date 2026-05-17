# SENPAI Research State

- **Updated:** 2026-05-17 04:05 UTC (R24 — 4 axes closed; 4 new assignments; frieren warmup=500 sent back)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 36.13**, **test_avg/mae_surf_p = 31.97** (PR #4282,
mlp_ratio=2 GEGLUBlock dead-code fix, best epoch 37, single-seed).

Per-split val: single=36.67, rc=48.15, cruise=21.37, re_rand=38.34.
Per-split test: single=36.53, rc=44.62, cruise=17.23, re_rand=29.50.

**Total improvement from calibration baseline:** 143.52 → 36.13 = **-74.8%**

**Note:** 3-seed baseline confirmation in flight (askeladd #4342). Single-seed variance ±5-10pt.

## Round wins merged (R1–R23)

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
| #3982 | mlp_ratio 2→1 (dead code) | 79.05 | -1.92% | MERGED |
| #4004 | FiLM-on-Re | 71.46 | -9.6% | MERGED |
| #4018 | FiLM-Re+AoA | 68.80 | -3.7% | MERGED |
| #4064 | bf16 autocast | 59.08 | -14.1% | MERGED |
| #4105 | GEGLU FFN on bf16 | 50.57 | -14.4% | MERGED |
| #4071 | Schedule-Free AdamW | 45.07 | -10.9% | MERGED |
| #4107 | slice_num 12→8 | 43.82 | -2.78% | MERGED |
| #4069 | torch.compile(dynamic=True) | 37.31 | -14.9% | MERGED |
| **#4282** | **mlp_ratio=2 GEGLUBlock fix** | **36.13** | **-3.2%** | **MERGED — current baseline** |

## Key architecture (current baseline)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN | GEGLU gating, **inner_dim=256** (`hidden_dim=int(128 * 2)`) |
| Compile | `torch.compile(model, dynamic=True, mode="default")` |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] |
| Precision | bf16 autocast |
| Optim | Schedule-Free AdamW `lr=5e-4, wd=1e-4, warmup=200` |
| Loss | SmoothL1 (beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997 |
| Compute | ~47.76s/epoch, **37 epochs**, peak VRAM 22.61 GB, **983,871 params** |

## Currently in flight (8 WIP — all students active)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4338 | fern | n_layers=4 + mlp_ratio=3: restore epoch budget at wider FFN | combo | WIP — R24 fresh |
| #4340 | nezuko | dropout sweep {0.05, 0.15} on mlp_ratio=2 stack | regularization | WIP — R24 fresh |
| #4341 | tanjiro | slice_num=6 on mlp_ratio=2 stack | attention | WIP — R24 fresh |
| #4342 | askeladd | 3-seed multi-seed baseline confirmation | validation | WIP — R24 fresh |
| #4260 | frieren | SF warmup_steps=500 retest on mlp_ratio=2 stack | optim | WIP — R24 send-back |
| #4301 | thorfinn | weight_decay sweep {0, 5e-4} on mlp_ratio=2 stack | regularization | WIP — pre-R24 |
| #4314 | edward | lr sweep {3e-4, 7.5e-4} on mlp_ratio=2 stack | optim | WIP — pre-R24 |
| #4281 | alphonse | beta=0.1 confirm + beta=0.05 2-arm on mlp_ratio=2 stack | loss | WIP — pre-R24 |

## Fully closed axes

| Axis | Verdict |
|------|---------|
| **n_layers** | FULLY CLOSED — 3 (floor), 4 (tie-within-noise), 5 (optimal), 6 (undertrained) |
| **mlp_ratio (FFN capacity)** | FULLY CLOSED — 2 optimal; 3 slight regression; 4 OOD overfitting |
| **n_head** | FULLY CLOSED — n_head=4 × dim_head=32 optimal; 8 dim_head too thin; 2 Q/K/V-params waste |
| EMA decay | CLOSED at 0.997 |
| slice_num (prior) | CLOSED at 8 on old stack; re-probing at 6 on new stack (tanjiro) |
| dropout (prior) | CLOSED at 0.1 on old stack; re-sweeping on new stack (nezuko) |
| GEGLU on attention projections | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| batch_size=8 | CLOSED |
| surface-weight ramp (10×) | CLOSED |
| RMSNorm | CLOSED |
| FiLM-full (11 scalars) | CLOSED |
| n_hidden=160 | DEFERRED (tested on broken stack; no retesting yet) |

## Potential next research directions

1. **n_layers=4 + mlp_ratio=3** — IN FLIGHT (fern #4338). Fix the undertraining issue from standalone mlp_ratio=3.
2. **dropout sweep {0.05, 0.15}** — IN FLIGHT (nezuko #4340). Larger model may need different regularization.
3. **slice_num=6** — IN FLIGHT (tanjiro #4341). Revisit on new mlp_ratio=2 stack.
4. **3-seed baseline confirmation** — IN FLIGHT (askeladd #4342). ~90min, 3 seeds.
5. **warmup_steps=500 on mlp_ratio=2** — IN FLIGHT (frieren #4260). Won on old stack; expecting compound.
6. **weight_decay sweep {0, 5e-4}** — IN FLIGHT (thorfinn #4301).
7. **lr sweep {3e-4, 7.5e-4}** — IN FLIGHT (edward #4314).
8. **SmoothL1 beta {0.1, 0.05}** — IN FLIGHT (alphonse #4281).
9. **n_layers=4 + mlp_ratio=2 + warmup_steps=500** — if warmup=500 wins, combine with depth
10. **n_hidden=160 on mlp_ratio=2 stack** — after main hyperparameter axes settle
11. **Stochastic depth (drop_path)** — regularization alternative to dropout
12. **Multi-seed confirmation of future winners** — once strong improvements identified
