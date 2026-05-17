# SENPAI Research State

- **Updated:** 2026-05-17 05:30 UTC (R25 — 4 closures; 4 new assignments; 0 merges this round)
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
| #4360 | frieren | warmup_steps=100 on mlp_ratio=2 — inverse probe; warmup=500 failed, does shorter help? | optim | WIP — R25 fresh |
| #4361 | thorfinn | n_hidden=160 on mlp_ratio=2 — deferred capacity axis; wd evidence says capacity-limited | architecture | WIP — R25 fresh |
| #4363 | tanjiro | slice_num=12 on mlp_ratio=2 — probe upward; slice_num=6 slower, hypothesis: wider FFN wants more slices | attention | WIP — R25 fresh |
| #4364 | fern | surf_weight sweep {15, 20} — bias loss toward primary surface metric; not re-swept since early rounds | loss | WIP — R25 fresh |
| #4342 | askeladd | 3-seed multi-seed baseline confirmation | validation | WIP — pre-R25 |
| #4340 | nezuko | dropout sweep {0.05, 0.15} on mlp_ratio=2 stack | regularization | WIP — pre-R25 |
| #4314 | edward | lr sweep {3e-4, 7.5e-4} on mlp_ratio=2 stack | optim | WIP — pre-R25 |
| #4281 | alphonse | beta=0.1 confirm + beta=0.05 2-arm on mlp_ratio=2 stack | loss | WIP — pre-R25 |

## Fully closed axes

| Axis | Verdict |
|------|---------|
| **n_layers** | FULLY CLOSED — 3 (floor), 4 (tie-within-noise), 5 (optimal), 6 (undertrained); n_layers=4+mlp_ratio=3 combo also tried and regressed |
| **mlp_ratio (FFN capacity)** | FULLY CLOSED — 2 optimal; 3 slight regression; 4 OOD overfitting; combo at n_layers=4+mlp_ratio=3 also fails |
| **n_head** | FULLY CLOSED — n_head=4 × dim_head=32 optimal; 8 dim_head too thin; 2 Q/K/V-params waste |
| **weight_decay** | FULLY CLOSED at 1e-4 — wd=0 helps test but hurts val_re_rand; wd=5e-4 squeezes capacity; model is capacity-limited not reg-limited |
| **SF warmup_steps** | PARTIALLY CLOSED — 50 hurts, 500 hurts on new stack; 200 confirmed optimal from above. Probing 100 (below) |
| **slice_num (current)** | PARTIALLY CLOSED — 6 regresses (+3.1%), 8 optimal; 12 in-flight (upward probe) |
| EMA decay | CLOSED at 0.997 |
| dropout (prior) | CLOSED at 0.1 on old stack; re-sweeping on new stack (nezuko) |
| GEGLU on attention projections | FULLY CLOSED — all regressed |
| Gate-activation axis | CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family | FULLY CLOSED |
| batch_size=8 | CLOSED |
| surface-weight ramp (10×) | CLOSED (ramp; static value axis now open — fern #4364) |
| RMSNorm | CLOSED |
| FiLM-full (11 scalars) | CLOSED |
| n_hidden=160 | IN FLIGHT (thorfinn #4361) |

## Key R25 insights (inform future hypotheses)

1. **Capacity-warmup interaction** (frieren): warmup length scales inversely with model capacity for SF AdamW. mlp_ratio=1 stack needed longer warmup (500 won); mlp_ratio=2 stack's cleaner gradients favor shorter (warmup=200 current, testing 100). Principle applies to future architectural changes.
2. **Model is capacity-limited** (thorfinn wd sweep): wd=0 freed capacity and helped test (−1.2%) but hurt val_re_rand. wd=5e-4 squeezed capacity and regressed everywhere. Direct implication: pursue width/depth increases.
3. **Slice_num kernel alignment** (tanjiro): compile path distinctly favors slice_num=8 (possibly power-of-two). slice_num=6 was actually 10% slower than baseline despite fewer slices. This is a systems finding to keep in mind.
4. **Camber-cruise gain orthogonality**: Multiple experiments (mlp_ratio=3 standalone, n_layers=4+mlp_ratio=3 combo) show camber-cruise (val_geom_camber_cruise) consistently improves with wider FFN, while in-dist/re_rand don't. This suggests the camber-cruise split is FFN-capacity-limited while OOD-Re splits are more depth/routing-limited.
5. **Asymmetric FFN potential** (fern's suggestion): wider FFN on only the last Transolver block could capture the camber-cruise gain without losing the mixing depth that OOD-Re splits need. Promising future hypothesis if current directions exhaust.

## Potential next research directions

1. **warmup_steps=100** — IN FLIGHT (frieren #4360). Inverse probe to failed 500.
2. **n_hidden=160** — IN FLIGHT (thorfinn #4361). Deferred capacity axis, now timely.
3. **slice_num=12** — IN FLIGHT (tanjiro #4363). Upward probe at wider FFN.
4. **surf_weight sweep {15, 20}** — IN FLIGHT (fern #4364). Direct loss bias toward primary metric.
5. **dropout {0.05, 0.15} on new stack** — IN FLIGHT (nezuko #4340).
6. **lr sweep {3e-4, 7.5e-4}** — IN FLIGHT (edward #4314).
7. **SmoothL1 beta {0.1, 0.05}** — IN FLIGHT (alphonse #4281).
8. **3-seed baseline confirmation** — IN FLIGHT (askeladd #4342).
9. **Asymmetric FFN per block** — after current results; wider last-block only.
10. **EMA checkpoint averaging (last-N-epoch)** — alternative checkpoint selection; thorfinn's wd=0 arm showed test wins despite val tie.
11. **Stochastic depth (drop_path)** — only if evidence shifts toward over-parameterized (currently under).
12. **n_hidden=160 + warmup_steps tuning** — compound if n_hidden=160 wins.
13. **surf_weight combination with capacity changes** — after fern's surf_weight result.
