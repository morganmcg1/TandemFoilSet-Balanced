# SENPAI Research State

- **Updated:** 2026-05-17 02:45 UTC (R23b — beta=0.1 sent back for mlp_ratio=2 retest; n_layers=4 closed; edward → lr sweep)
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

## Key discoveries this session

1. **mlp_ratio was dead code (R23, PR #4282):** GEGLUBlock was hardcoded `hidden_dim=hidden_dim` regardless of mlp_ratio. Fix: `hidden_dim=int(hidden_dim * mlp_ratio)`. mlp_ratio=2 → inner_dim=256 → val 37.31→36.13. The FFN capacity axis is now live and open.

2. **n_layers=4 axis closed (R23b):** Three retests across shifting baselines showed each baseline improvement consumed n_layers=4's compute headroom. At compile+mlp_ratio=2, it's a tie within noise. Potential future sub-component if mlp_ratio=3/4 is compute-heavy.

3. **SmoothL1 beta=0.1 promising (R23b):** Alphonse's beta=0.1 result (val=35.86) beats both old and new baselines on the pre-fix stack. Sent back for 2-arm retest (beta=0.1 confirm + beta=0.05) on the mlp_ratio=2 stack.

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4299 | fern | mlp_ratio=3 (inner_dim=384) on mlp_ratio=2 stack | FFN capacity | WIP — R23 fresh |
| #4300 | nezuko | mlp_ratio=4 (inner_dim=512) ceiling probe | FFN capacity | WIP — R23 fresh |
| #4301 | thorfinn | weight_decay sweep {0, 5e-4} on mlp_ratio=2 stack | optim | WIP — R23 fresh |
| #4302 | tanjiro | n_layers=6 depth probe on mlp_ratio=2 stack | depth | WIP — R23 fresh |
| #4314 | edward | lr sweep {3e-4, 7.5e-4} on mlp_ratio=2 stack | optim | WIP — R23b fresh |
| #4281 | alphonse | beta=0.1 confirm + beta=0.05 sweep on mlp_ratio=2 stack | loss | WIP — R23b send-back |
| #4260 | frieren | SF warmup_steps {50, 500} (pre-fix stack) | optim | WIP — must beat 36.13 |
| #4290 | askeladd | n_head=2 (dim_head=64) on compile stack (pre-fix stack) | architecture | WIP — must beat 36.13 |

**Note on pre-fix PRs:** frieren #4260 and askeladd #4290 are on the old (mlp_ratio=1) stack.
Their results will be compared against 36.13. Send back for retest if promising but miss new baseline.

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is optimal on compile stack |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck |
| slice_num | CLOSED at 8 |
| mlp_ratio axis (prior closure invalid) | RE-OPENED — sweeping from 2 upward (fern mlp_ratio=3, nezuko mlp_ratio=4) |
| dropout (0.1) | CLOSED |
| surf_weight (10.0) | CLOSED |
| lr peak (5e-4) | UNDER RETEST — edward sweeping {3e-4, 7.5e-4} on mlp_ratio=2 stack |
| FiLM-full naive (11 scalars) | CLOSED |
| FiLM-broadcast-scalar axis | CLOSED |
| cosine T_max tuning | SUPERSEDED by Schedule-Free AdamW |
| GEGLU on attention projections | FULLY CLOSED — all regressed |
| n_head=8 (dim_head=16) | CLOSED (+4%) — dim_head too thin |
| n_layers=3 | CLOSED — capacity floor at n_hidden=128 |
| n_layers=4 | CLOSED — 3-retest tie within noise on current strong baseline |
| batch_size=8 | CLOSED — halved step count, no compute benefit |
| surface-weight ramp (10×) | CLOSED — destabilizes SF AdamW optimizer state |
| n_hidden=160 (old stack) | DEFERRED — revisit if FFN axis saturates |
| Gate-activation axis | FULLY CLOSED — GEGLU > ReGLU > SwiGLU |
| FiLM family (all variants) | FULLY CLOSED |

## Potential next research directions

1. **beta=0.1 (arm A confirm) + beta=0.05 (arm B)** — IN FLIGHT (alphonse #4281). Val=35.86 already beats new baseline on old stack; expecting compound win on mlp_ratio=2 stack.
2. **mlp_ratio=3** — IN FLIGHT (fern #4299). Natural FFN capacity continuation.
3. **mlp_ratio=4** — IN FLIGHT (nezuko #4300). FFN capacity ceiling bracket.
4. **weight_decay sweep {0, 5e-4}** — IN FLIGHT (thorfinn #4301). Regularization with larger 983k model.
5. **n_layers=6** — IN FLIGHT (tanjiro #4302). First upward depth probe on mlp_ratio=2 stack.
6. **lr sweep {3e-4, 7.5e-4}** — IN FLIGHT (edward #4314). LR optimum may shift with 33% more params.
7. **n_layers=4 + mlp_ratio=3** — If nezuko's mlp_ratio=4 wins but is compute-expensive, combine with n_layers=4 to reclaim wall-clock.
8. **dropout re-tune {0.0, 0.2}** — After main axes settle; larger model may need different regularization.
9. **Smooth surface ramp** — Gentle 2× cosine curriculum; revisit after core axes close.
10. **n_hidden=160 + mlp_ratio=2** — After FFN capacity axis closes; combined capacity probe.
11. **Multi-seed confirmation** — 3-seed variance on val=36.13 before deadline.
12. **FP8 precision** — Next compute lever after compile saturates.
