# SENPAI Research State

- **Updated:** 2026-05-17 01:25 UTC (R22 — torch.compile merged, 6 new assignments)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 37.31**, **test_avg/mae_surf_p = 32.81** (PR #4069,
torch.compile on full bf16+GEGLU+SF+slice=8 stack, best epoch 42, still descending at termination).

Per-split val: single=37.19, rc=50.50, cruise=21.48, re_rand=40.09.
Per-split test: single=36.49, rc=46.33, cruise=17.85, re_rand=30.54.

**Total improvement from calibration baseline:** 143.52 → 37.31 = **-74.0%**

## Round wins merged (R1–R22)

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
| #4064 | bf16 autocast | 59.08 | -14.1% | MERGED |
| #4105 | GEGLU FFN on bf16 | 50.57 | -14.4% | MERGED |
| #4071 | Schedule-Free AdamW on bf16+GEGLU | 45.07 | -10.9% | MERGED |
| #4107 | slice_num 12→8 on bf16+GEGLU+SF | 43.82 | -2.78% | MERGED |
| **#4069** | **torch.compile(dynamic=True) on full stack** | **37.31** | **-14.9%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, mlp_ratio=1 |
| Compile | **`torch.compile(model, dynamic=True, mode="default")`** (PR #4069) — fuses FiLM affine + GEGLU gate + QKV ops; `dynamic=True` required |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] (3-scalar → per-block γ,β) |
| Precision | bf16 autocast (forward + loss; reductions in fp32) |
| FFN | **GEGLU gating** (PR #4105): `FFN(x) = W2(GELU(W1a(x)) * W1b(x))` |
| Optim | **Schedule-Free AdamW** (PR #4071): `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — no LR scheduler |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint; EMA built before compile so `ema_model.module` is uncompiled |
| Compile checkpoint load | `load_target = getattr(model, "_orig_mod", model)` before `load_state_dict` |
| Compute | ~42.4s/epoch, **42 epochs** in 30-min cap, peak VRAM 18.88 GB |

## Dominant discovery: compute-bound thesis holds; compile is a free -41% wall-clock; capacity headroom opened

torch.compile on the full bf16+GEGLU+SF+slice=8 stack delivered -41.3% sec/epoch (72.3→42.4s) and +17 epochs (25→42), with val still descending at 0.12 pt/epoch at terminal. VRAM dropped 27% (25.96→18.88 GB), opening 7 GB headroom for capacity experiments.

Key insight from R22: The +20% params on attention projections is the budget ceiling at this 30-min cap. GEGLU wins belong to the FFN only. Gate-activation axis fully closed (GEGLU > ReGLU > SwiGLU). GEGLU-on-attention-projections axis fully closed (readout, to_out, in_project_fx all regressed).

Four orthogonal levers now active:
1. **Fewer seconds per epoch** — n_layers=3 (nezuko), n_layers=4+compile (edward retest)
2. **More capacity within freed VRAM** — n_hidden=160 (thorfinn), n_head=8 (alphonse)
3. **Gradient quality** — batch_size=8 (tanjiro), SF warmup_steps sweep (frieren)
4. **Faster validation (speed up val eager path)** — EMA compile (askeladd)

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| new | nezuko | n_layers=3 on compile+SF+GEGLU+bf16+slice=8 | compute | WIP — R22 fresh |
| new | thorfinn | n_hidden=128→160 on compile stack | capacity | WIP — R22 fresh |
| new | alphonse | n_head=8 (dim_head=16, inner_dim=128) on compile stack | capacity | WIP — R22 fresh |
| new | frieren | SF warmup_steps sweep {50, 500} vs 200 | optim | WIP — R22 fresh |
| new | tanjiro | batch_size=4→8 on compile stack | gradient | WIP — R22 fresh |
| new | askeladd | compile EMA module (speed val eager path) | compute | WIP — R22 fresh |
| #4068 | edward | n_layers=4 + compile retest (rebase onto PR #4069) | compute | WIP — R22 send-back |
| #4177 | fern | EMA decay re-tune {0.995, 0.999} on SF stack | optim | WIP — pre-compile baseline |

**Note on fern #4177:** result will be on the pre-compile stack (val baseline 43.82). If val > 37.31, we will ask for compile retest. If val ≤ 37.31 — immediate merge.

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is the optimum (pre-compile; fern retest in flight) |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck for 3 scalars |
| slice_num (16→12) | MERGED as tiny win |
| mlp_ratio | CLOSED at 1 (width not the lever) |
| dropout (0.1) | CLOSED |
| surf_weight (10.0) | CLOSED |
| lr peak (5e-4 vs 7.5e-4) | SATURATED — closed at 5e-4 |
| FiLM-full naive (11 scalars) | CLOSED |
| **FiLM-broadcast-scalar axis** | CLOSED |
| **cosine T_max tuning** | SUPERSEDED by Schedule-Free AdamW (#4071) |
| **GEGLU readout head** | CLOSED (#4168 +3.2%) |
| **SwiGLU (F.silu gate)** | CLOSED (#4155 +4.2%) |
| **ReGLU (F.relu gate)** | CLOSED (#4209 +1.9%) |
| **Gate-activation axis** | FULLY CLOSED — GEGLU > ReGLU > SwiGLU |
| **Per-node geometric FiLM** | CLOSED (#4186 +9.4%) |
| **FiLM family (all variants)** | FULLY CLOSED |
| **slice_num halving axis** | CLOSED at 8 (#4185 TIE) |
| **GEGLU on attention projections** | FULLY CLOSED — readout (#4168), to_out (#4206), in_project_fx (#4228) all regressed |

## Potential next research directions

1. **n_layers=3 + compile** — IN FLIGHT (nezuko). Predicted ~28s/epoch → 50 epochs (cap), predicted val ≤ 34.
2. **n_layers=4 + compile** — IN FLIGHT (edward retest). Predicted val ≤ 35.
3. **n_hidden=160** — IN FLIGHT (thorfinn). First capacity probe in freed VRAM headroom.
4. **n_head=8 (neutral params)** — IN FLIGHT (alphonse). Clean head granularity test.
5. **batch_size=8** — IN FLIGHT (tanjiro). Freed VRAM allows this; gradient noise experiment.
6. **SF warmup_steps {50, 500}** — IN FLIGHT (frieren). Lightweight optimizer sweep.
7. **Compile EMA module** — IN FLIGHT (askeladd). Speeds val from eager to compiled.
8. **EMA decay re-tune on compile stack** — after fern result; may need reassignment.
9. **n_layers=2 probe** — after n_layers=3 settles; aggressive depth reduction.
10. **n_hidden=192** — after n_hidden=160 settles; capacity ceiling probe.
11. **Geometric attention bias** — bias slice routing by dsdf; more principled than FiLM.
12. **Surface-only loss fine-tune** — final epochs with surf_weight→∞.
13. **Weight decay sweep {0, 1e-5, 1e-3}** — after core axes settle.
14. **Multi-seed confirmation** — 3-seed variance on val=37.31 before deadline.
15. **FP8 precision** — Blackwell-native, next compute lever after compile saturates.
