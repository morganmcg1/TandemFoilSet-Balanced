# SENPAI Research State

- **Updated:** 2026-05-16 23:45 UTC (R21 b — tanjiro slice=6 closed, reassigned)
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 43.82**, **test_avg/mae_surf_p = 38.05** (PR #4107,
slice_num=8 on bf16+GEGLU+SF + FiLM-Re+AoA, best epoch 25, still descending at termination).

Per-split val: single=47.39, rc=55.44, cruise=26.97, re_rand=45.50.
Per-split test: single=42.38, rc=50.51, cruise=22.71, re_rand=36.58.

**Total improvement from calibration baseline:** 143.52 → 43.82 = **-69.5%**

## Round wins merged (R1–R19)

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
| **#4107** | **slice_num 12→8 on bf16+GEGLU+SF** | **43.82** | **-2.78%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, **slice_num=8**, mlp_ratio=1 |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] (3-scalar → per-block γ,β) |
| Precision | bf16 autocast (forward + loss; reductions in fp32) |
| FFN | **GEGLU gating** (PR #4105): `FFN(x) = W2(GELU(W1a(x)) * W1b(x))` |
| Optim | **Schedule-Free AdamW** (PR #4071): `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — no LR scheduler |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint |
| Compute | ~72.3s/epoch, **25 epochs** in 30-min cap, peak VRAM 25.19 GB |

## Dominant discovery: orthogonal wins compound; compute headroom still binding

In R14→R17 we landed three major jumps stacking cleanly: bf16 (#4064) -14.1%; GEGLU (#4105) -14.4%; Schedule-Free AdamW (#4071) -10.9%. Total -35% in 3 PRs. **The model is still descending at termination in ALL wins** — the compute-bound thesis holds strongly.

Key insight from R17: cosine T_max=50 with only 23 effective epochs puts LR at ~59% of peak at the terminal step — late epochs were under-powered. Schedule-Free removes this fragility for all future levers.

Three orthogonal levers still in play:
1. **Fewer seconds per epoch** — torch.compile (nezuko #4069, training completed, result pending GH rate-limit reset), n_layers=4 (edward #4068, pending)
2. **More expressive per-step update** — ReGLU vs GEGLU (frieren #4209, fresh), GEGLU on to_out (alphonse #4206, fresh), slice_num=6 (tanjiro #4185), EMA re-tune (fern #4177)
3. **More capacity / conditioning** — FiLM family now fully CLOSED. Next escalation: attention architecture variants (geometric bias, modified slice assignment).

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4206 | alphonse  | GEGLU gate on attention to_out projection | FFN/attn | WIP — R21, fresh assignment |
| #4209 | frieren   | ReGLU (F.relu in gate) vs GEGLU — completes gate-activation axis | FFN nonlinearity | WIP — R21, fresh assignment |
| #4228 | tanjiro   | GEGLU gate on in_project_fx (attention feature input projection) | attn architecture | WIP — R21b, fresh assignment |
| #4177 | fern      | EMA decay re-tune on SF stack: probe {0.995, 0.999} vs 0.997 | optim | WIP |
| #4068 | edward    | n_layers 5→4 on bf16+GEGLU+SF stack | compute | WIP |
| #4069 | nezuko    | torch.compile(dynamic=True) on bf16+GEGLU+SF stack | compute | WIP — training completed ~23:14; results pending GH API rate-limit recovery |
| #4134 | thorfinn  | Cosine T_max 50→25 (superseded by SF — result informational only) | LR schedule | WIP — training completed; result pending GH rate-limit |
| #4136 | askeladd  | batch=8 + lr=1e-3 (linear scaling) on GEGLU | data parallelism | WIP — training completed ~22:59; result pending GH rate-limit; result against old baseline, needs full-stack retest if positive |

**Operational note (R21):** GitHub API rate limit hit for student pod user account (20516801). All pods stopped posting after ~22:00 UTC. alphonse and frieren got through a brief rate-limit window at 23:26 UTC (posted results for #4186, #4155). Other pods (nezuko, thorfinn, askeladd, edward, fern, tanjiro) have training complete but results not yet posted. Rate limit resets by ~00:20 UTC. New assignments for alphonse (#4206) and frieren (#4209) dispatched.

**Closed this R21:**
- **#4186 CLOSED** (alphonse per-node geometric FiLM): +9.4% regression, uniform across all 4 splits. Three-part diagnosis: redundant pathway with preprocess, compute squeeze (-7 epochs), gradient dilution over ~1500 nodes. **FiLM family fully exhausted.**
- **#4155 CLOSED** (frieren SwiGLU): +4.2% worse mean of 2 seeds. GELU's sharper gate is better feature-selector than SiLU on heavy-tailed pressure fields. **Gate-activation axis: only ReGLU and Bilinear remain to test.**
- **#4185 CLOSED** (tanjiro slice_num=6): TIE at +0.20% (val 43.91). sec/epoch went UP +2.9% — slice projection is no longer the bottleneck. **slice_num halving axis FULLY CLOSED** (8 is the optimum).

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is the optimum |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck for 3 scalars |
| slice_num (16→12) | MERGED as tiny win |
| mlp_ratio | CLOSED at 1 (width not the lever) |
| dropout (0.1) | CLOSED |
| n_head (4) | CLOSED |
| surf_weight (10.0) | CLOSED |
| lr peak (5e-4 vs 7.5e-4) | SATURATED — closed at 5e-4 |
| batch=8 (no LR scaling) | CLOSED — linear-scaling lr=1e-3 in flight #4136 |
| FiLM-full naive (11 scalars) | CLOSED |
| **FiLM-broadcast-scalar axis** | CLOSED — GEGLU's block-level gating subsumes broadcast FiLM disentanglement |
| **mlp_ratio (CLOSED for GEGLU)** | CLOSED — #4137 regression +1.58%; wall-clock saturation |
| **cosine T_max tuning** | SUPERSEDED by Schedule-Free AdamW (#4071) |
| **GEGLU readout head** | CLOSED (#4168 +3.2% — 128→3 projection too narrow for gating) |
| **SwiGLU (F.silu gate)** | CLOSED (#4155 +4.2% — GELU sharper is better for CFD) |
| **Per-node geometric FiLM** | CLOSED (#4186 +9.4% — redundant pathway, compute squeeze, gradient dilution) |
| **FiLM family (all variants)** | FULLY CLOSED — broadcast-scalar, per-node, readout all closed |
| **slice_num halving axis** | CLOSED at 8 — 8→6 TIE (#4185 +0.20%); slice projection no longer the bottleneck |

## Potential next research directions

1. **torch.compile on SF+GEGLU+bf16 stack** — IN FLIGHT (#4069 nezuko). Prior result val=41.20 on pre-SF baseline (-18.5%), full-stack result pending. High-confidence win: predicted val ≤ 38. Will merge immediately on result.
2. **ReGLU vs GEGLU** — IN FLIGHT (#4209 frieren). Completes gate-activation axis (GEGLU won over SwiGLU; ReLU gate is the remaining "harder cutoff" test). 40% win / 45% tie / 15% worse.
3. **GEGLU on to_out projection** — IN FLIGHT (#4206 alphonse). Extends GEGLU win pattern to attention output projection; same shape (128→128) as successful FFN GEGLU.
4. **GEGLU on in_project_fx** — IN FLIGHT (#4228 tanjiro). Extends GEGLU gating to the attention input feature projection; orthogonal to #4206 (to_out).
5. **n_layers 5→4** — IN FLIGHT (#4068 edward). Predicted ~62-68s/epoch → 27-29 epochs in cap.
6. **EMA decay re-tune** — IN FLIGHT (#4177 fern). Probe 0.995 vs 0.999 on SF stack.
7. **batch=8 + lr=1e-3** — IN FLIGHT (#4136 askeladd). Against old baseline; if positive, needs full-stack retest.
8. **SF warmup_steps sweep** — probe {50, 500}; ep1 val spike suggests warmup worth tuning. Assign when student becomes idle.
9. **Bilinear gate (no activation)** — Other end of sharpness axis (GEGLU=GELU, SwiGLU=SiLU, ReGLU=ReLU, Bilinear=none). Assign after ReGLU result.
10. **Multi-seed confirmation** — Before ICML deadline, 3-seed variance on val=43.82 (±5-10 pt noise).
11. **Attention geometric bias** — Bias slice-attention assignment by dsdf (distance-to-surface): surface nodes preferentially routed to dedicated slice groups. More principled than closed FiLM family.
12. **Surface-only loss fine-tune** — After 20 epochs normal, anneal with surf_weight→∞ for final 5 epochs. Refocus gradient signal on primary metric.
