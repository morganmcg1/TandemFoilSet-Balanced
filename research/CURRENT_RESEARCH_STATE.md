# SENPAI Research State

- **Updated:** 2026-05-17 02:15 UTC (R23 — mlp_ratio=2 fix merged as new baseline; 4 new assignments)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 36.13**, **test_avg/mae_surf_p = 31.97** (PR #4282,
mlp_ratio=2 GEGLUBlock dead-code fix, best epoch 37).

Per-split val: single=36.67, rc=48.15, cruise=21.37, re_rand=38.34.
Per-split test: single=36.53, rc=44.62, cruise=17.23, re_rand=29.50.

**Total improvement from calibration baseline:** 143.52 → 36.13 = **-74.8%**

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
| #3982 | mlp_ratio 2→1 (dead code — no effect) | 79.05 | -1.92% | MERGED |
| #4004 | FiLM-on-Re | 71.46 | -9.6% | MERGED |
| #4018 | FiLM-Re+AoA | 68.80 | -3.7% | MERGED |
| #4064 | bf16 autocast | 59.08 | -14.1% | MERGED |
| #4105 | GEGLU FFN on bf16 | 50.57 | -14.4% | MERGED |
| #4071 | Schedule-Free AdamW on bf16+GEGLU | 45.07 | -10.9% | MERGED |
| #4107 | slice_num 12→8 on bf16+GEGLU+SF | 43.82 | -2.78% | MERGED |
| #4069 | torch.compile(dynamic=True) on full stack | 37.31 | -14.9% | MERGED |
| **#4282** | **mlp_ratio=2 GEGLUBlock fix (dead-code bug)** | **36.13** | **-3.2%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, **mlp_ratio=2** |
| FFN width | **inner_dim=256** — `GEGLUBlock(hidden_dim, hidden_dim, hidden_dim=int(hidden_dim * mlp_ratio))` |
| Compile | **`torch.compile(model, dynamic=True, mode="default")`** (PR #4069) |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] (3-scalar → per-block γ,β) |
| Precision | bf16 autocast (forward + loss; reductions in fp32) |
| FFN | **GEGLU gating** with **mlp_ratio=2** (inner_dim=256): `FFN(x) = W2(GELU(W1a(x)) * W1b(x))` |
| Optim | **Schedule-Free AdamW** `lr=5e-4, weight_decay=1e-4, warmup_steps=200` — no LR scheduler |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint |
| Compute | ~47.76s/epoch, **37 epochs** in 30-min cap, peak VRAM 22.61 GB, **983,871 params** |

## Key discovery: mlp_ratio was dead code; FFN capacity axis newly opened

PR #4282 revealed that all prior experiments since #3982 (mlp_ratio 2→1) were running
with `mlp_ratio=1` regardless of config — `GEGLUBlock` was hardcoded `hidden_dim=hidden_dim`.
The fix: `hidden_dim=int(hidden_dim * mlp_ratio)`. mlp_ratio=2 gives inner_dim=256 (vs 128),
adding 33.6% params and delivering val=36.13 (−3.2%). The FFN capacity axis is now live and open.

Previous "mlp_ratio CLOSED at 1" finding (R7, PR #3982) is invalidated — that PR effectively
tested mlp_ratio=1 vs mlp_ratio=1. The axis needs a fresh sweep from 2 upward.

EMA-epoch coupling insight: EMA decay optimum tracks per-step iterate displacement, not optimizer
choice. Compile doubled epoch budget (23→42→37 epochs), shifting optimal EMA back to 0.997.

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4299 | fern | mlp_ratio=3 (inner_dim=384) on mlp_ratio=2 compile stack | FFN capacity | WIP — R23 fresh |
| #4300 | nezuko | mlp_ratio=4 (inner_dim=512) on mlp_ratio=2 compile stack | FFN capacity | WIP — R23 fresh |
| #4301 | thorfinn | weight_decay sweep {0, 5e-4} on mlp_ratio=2 stack | optim | WIP — R23 fresh |
| #4302 | tanjiro | n_layers=6 depth probe on mlp_ratio=2 stack | depth | WIP — R23 fresh |
| #4068 | edward | n_layers=4 + compile retest (on old mlp_ratio=1 stack) | compute | WIP — pre-mlp_ratio fix; must beat 36.13 |
| #4260 | frieren | SF warmup_steps {50, 500} (on old mlp_ratio=1 stack) | optim | WIP — pre-mlp_ratio fix; must beat 36.13 |
| #4281 | alphonse | SmoothL1 beta=0.1 (on old mlp_ratio=1 stack) | loss | WIP — pre-mlp_ratio fix; must beat 36.13 |
| #4290 | askeladd | n_head=2 (dim_head=64) on compile stack | architecture | WIP — pre-mlp_ratio fix; must beat 36.13 |

**Note on pre-fix PRs:** edward #4068, frieren #4260, alphonse #4281, askeladd #4290 are all running
on the old (mlp_ratio=1 dead-code) stack. Their wins need to be measured against 36.13 now.
edward's n_layers=4 predicted val ~35 may still win. Others less certain.

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is optimal on compile stack (42 epochs) |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck for 3 scalars |
| slice_num | CLOSED at 8 (PR #4185 TIE on halving) |
| mlp_ratio axis (prior closure invalid) | RE-OPENED — #3982 was dead code; now sweeping from 2 upward |
| dropout (0.1) | CLOSED |
| surf_weight (10.0) | CLOSED |
| lr peak (5e-4) | SATURATED |
| FiLM-full naive (11 scalars) | CLOSED |
| FiLM-broadcast-scalar axis | CLOSED |
| cosine T_max tuning | SUPERSEDED by Schedule-Free AdamW |
| GEGLU readout head | CLOSED (+3.2%) |
| SwiGLU gate | CLOSED (+4.2%) |
| ReGLU gate | CLOSED (+1.9%) |
| Gate-activation axis | FULLY CLOSED — GEGLU > ReGLU > SwiGLU |
| Per-node geometric FiLM | CLOSED (+9.4%) |
| FiLM family (all variants) | FULLY CLOSED |
| GEGLU on attention projections | FULLY CLOSED — all regressed |
| n_head=8 (dim_head=16) | CLOSED (+4%) — dim_head too thin |
| n_layers=3 | CLOSED — capacity floor at n_hidden=128 |
| batch_size=8 | CLOSED — halved step count, no VRAM gain on this GPU |
| surface-weight ramp (10×) | CLOSED — destabilizes SF AdamW optimizer state |
| n_hidden=160 (old stack) | DEFERRED — tested on broken mlp_ratio=1 stack; revisit if FFN axis saturates |

## Potential next research directions

1. **mlp_ratio=3** — IN FLIGHT (fern #4299). Predicted ~33 epochs; natural next rung.
2. **mlp_ratio=4** — IN FLIGHT (nezuko #4300). Ceiling probe; predicted ~29–30 epochs.
3. **weight_decay sweep {0, 5e-4}** — IN FLIGHT (thorfinn #4301). Larger model may need different regularization.
4. **n_layers=6** — IN FLIGHT (tanjiro #4302). First upward depth probe at mlp_ratio=2.
5. **n_layers=4 + compile** — IN FLIGHT (edward #4068). Localizes depth knee from below.
6. **SF warmup_steps {50, 500}** — IN FLIGHT (frieren #4260). On old stack; may need retest.
7. **SmoothL1 beta=0.1** — IN FLIGHT (alphonse #4281). On old stack; may need retest.
8. **n_head=2 (dim_head=64)** — IN FLIGHT (askeladd #4290). Coarser attention granularity.
9. **mlp_ratio=3 + n_hidden=160** — after FFN axis settles; combined capacity probe.
10. **lr sweep {3e-4, 7e-4}** — optimal lr may shift with larger 983k-param model.
11. **dropout re-tune {0.0, 0.2}** — larger model may need different regularization.
12. **Smooth surface ramp** — gentle 2× cosine curriculum (not abrupt 10×) on mlp_ratio=2 stack.
13. **Multi-seed confirmation** — 3-seed variance on val=36.13 before deadline.
14. **n_hidden=192 + mlp_ratio=2** — high-capacity probe after FFN axis closes.
15. **FP8 precision** — next compute lever after compile saturates.
