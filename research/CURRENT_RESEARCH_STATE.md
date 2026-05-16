# SENPAI Research State

- **Updated:** 2026-05-16 21:50 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 45.07**, **test_avg/mae_surf_p = 38.58** (PR #4071,
Schedule-Free AdamW on bf16+GEGLU + FiLM-Re+AoA, best epoch 23, still descending at termination).

Per-split val: single=48.79, rc=58.57, cruise=26.72, re_rand=46.21.
Per-split test: single=43.26, rc=51.59, cruise=22.20, re_rand=37.26.

**Total improvement from calibration baseline:** 143.52 → 45.07 = **-68.6%**

## Round wins merged (R1–R17)

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
| **#4071** | **Schedule-Free AdamW on bf16+GEGLU** | **45.07** | **-10.9%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=12, mlp_ratio=1 |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] (3-scalar → per-block γ,β) |
| Precision | bf16 autocast (forward + loss; reductions in fp32) |
| FFN | **GEGLU gating** (PR #4105): `FFN(x) = W2(GELU(W1a(x)) * W1b(x))` |
| Optim | **Schedule-Free AdamW** (PR #4071): `schedulefree.AdamWScheduleFree(lr=5e-4, weight_decay=1e-4, warmup_steps=200)` — no LR scheduler |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint |
| Compute | ~79.2s/epoch, **23 epochs** in 30-min cap, peak VRAM 25.96 GB |

## Dominant discovery: orthogonal wins compound; compute headroom still binding

In R14→R17 we landed three major jumps stacking cleanly: bf16 (#4064) -14.1%; GEGLU (#4105) -14.4%; Schedule-Free AdamW (#4071) -10.9%. Total -35% in 3 PRs. **The model is still descending at termination in ALL wins** — the compute-bound thesis holds strongly.

Key insight from R17: cosine T_max=50 with only 23 effective epochs puts LR at ~59% of peak at the terminal step — late epochs were under-powered. Schedule-Free removes this fragility for all future levers.

Three orthogonal levers still in play:
1. **Fewer seconds per epoch** — torch.compile (nezuko, pending bf16+GEGLU+SF retest), n_layers=4 (edward, pending retest)
2. **More expressive per-step update** — SwiGLU vs GEGLU (frieren #4155), GEGLU in readout head (alphonse #4168), slice_num=8 (tanjiro #4107 rebase)
3. **More capacity / conditioning** — per-node geometric conditioning (signed-distance, surface-normal features) is the natural escalation now that broadcast-scalar FiLM axis is saturated

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4168 | alphonse  | GEGLU in mlp2 readout head (final output projection) | FFN gating | WIP — R17-late |
| #4177 | fern      | EMA decay re-tune on SF stack: probe {0.995, 0.999} vs 0.997 | optim | WIP — R17-late, just assigned |
| #4068 | edward    | n_layers 5→4 on bf16+GEGLU+SF stack | compute | WIP (needs full-stack rebase incl. SF) |
| #4069 | nezuko    | torch.compile(dynamic=True) on bf16+GEGLU+SF stack | compute | WIP (needs full-stack rebase incl. SF) |
| #4107 | tanjiro   | slice_num 12→8 on bf16+GEGLU+SF stack | compute | WIP (needs full-stack rebase) |
| #4134 | thorfinn  | Cosine T_max 50→25 (superseded by SF merge — result interesting but not decision-making) | LR schedule | WIP — R16 |
| #4136 | askeladd  | batch=8 + lr=1e-3 (linear scaling) on GEGLU | data parallelism | WIP — R16 |
| #4155 | frieren   | SwiGLU vs GEGLU (replace `F.gelu` with `F.silu` in gate) | FFN nonlinearity | WIP — R16-late |

**Baseline update comment sent to #4068, #4069, #4107** notifying them of the new val=45.07 target and full-stack (bf16+GEGLU+SF) rebase requirement.

## Closed axes (final state)

| Axis | Verdict |
|------|---------|
| EMA decay (0.997→0.995) | SATURATED — 0.997 is the optimum |
| RMSNorm | REGRESSION — mean-centering matters for CFD pressure |
| Wider FiLM head (128→256) | TIE — FiLM head capacity not the bottleneck for 3 scalars |
| slice_num (16→12) | MERGED as tiny win — probe 12→8 in flight |
| mlp_ratio | CLOSED at 1 (width not the lever) |
| dropout (0.1) | CLOSED |
| n_head (4) | CLOSED |
| surf_weight (10.0) | CLOSED |
| lr peak (5e-4 vs 7.5e-4) | SATURATED — closed at 5e-4 |
| batch=8 (no LR scaling) | CLOSED — needs linear-scaling lr=1e-3 variant (in flight #4136) |
| FiLM-full naive (11 scalars) | CLOSED — two-stage v2 ran; closed below |
| **FiLM-broadcast-scalar axis** | CLOSED (R17) — #4041 v2 on bf16+GEGLU regressed +1.57%; GEGLU's block-level gating subsumes the disentanglement that broadcast FiLM was buying |
| **mlp_ratio (also CLOSED for GEGLU)** | CLOSED — #4137 regression +1.58%; wall-clock-saturation |
| **cosine T_max tuning** | SUPERSEDED — Schedule-Free AdamW (#4071) merged; SF removes T_max fragility by construction; thorfinn's T_max=25 probe (#4134) is still running but outcome is moot for future assignments |

## Potential next research directions (post-R17)

1. **GEGLU in mlp2 readout head** — IN FLIGHT (#4168 alphonse). Single-line change; same gating as block MLPs; +16k params.
2. **SwiGLU vs GEGLU** — IN FLIGHT (#4155 frieren). Replace `F.gelu` with `F.silu`; LLaMA/PaLM choice; smooth pressure fields may favor SiLU.
3. **slice_num=8 on full stack (bf16+GEGLU+SF)** — IN FLIGHT (#4107 tanjiro rebase). Predicted compound target val ≤ 40; if wins → probe slice_num=6/4.
4. **n_layers 5→4 on full stack** — IN FLIGHT (#4068 edward). Predicted sec/epoch ~62-68s → 27-29 epochs; target val ≤ 42.
5. **torch.compile(dynamic=True) on full stack** — IN FLIGHT (#4069 nezuko). Expected ~-15% sec/epoch → +4 epochs; target val ≤ 42.
6. **EMA decay re-tune on SF stack** — SF keeps LR at full strength longer; EMA 0.997 was tuned for cosine. Probe 0.995, 0.997, 0.999 on SF baseline. Cheap 3-arm sweep.
7. **SF warmup_steps sweep** — 200 worked; sharp early epoch (ep1 val=251) suggests probing {50, 500} could improve early-epoch stability.
8. **Per-node geometric conditioning** — Escalation from broadcast-scalar FiLM (now closed). Signed-distance or surface-normal features injected at each FiLM site per mesh node. Bigger architecture step; may need its own paper contribution.
9. **GEGLU in attention output projection** — The `to_out` linear in PhysicsAttention is still vanilla. A similar GEGLU probe there (in_dim=hidden*n_head, out_dim=hidden) could replicate the block-FFN win.
10. **Multi-seed confirmation (3 seeds)** — Before ICML deadline, tighten variance on the current val=45.07 baseline (±5-10 pt single-seed noise).
11. **Batch=8 + lr=1e-3 on SF stack** — In flight as askeladd #4136 (was dispatched before SF merge). The linear-scaling argument still applies; SF may interact differently with larger batches.
12. **fern re-assignment** — fern is idle after #4071 merge. Top candidates: EMA decay re-tune (#6 above), warmup_steps sweep (#7), or attention-output GEGLU (#9).

## Round 16 dispatched (R16)

| PR | Student | Hypothesis | Outcome / Status |
|----|---------|------------|------------------|
| #4134 | thorfinn  | Cosine T_max 50→25 | WIP, training |
| #4136 | askeladd  | batch=8 + lr=1e-3 (linear scaling) | WIP, training |
| #4137 | frieren   | GEGLU + mlp_ratio 1→2 | CLOSED (regression +1.58% val, wall-clock-driven) — frieren reassigned to #4155 |
| #4155 | frieren   | SwiGLU vs GEGLU (replace `F.gelu` with `F.silu` in gate; same params, same sec/epoch) | WIP — R16-late follow-up |

## R17 actions (this iteration)

- **#4071 MERGED** (fern Schedule-Free AdamW): val=45.07, test=38.58, -10.9% win, all 8 splits improved, zero compute overhead. New baseline.
- **#4041 CLOSED** (alphonse FiLM-broadcast-scalar axis): v2 on bf16+GEGLU regressed +1.57%; GEGLU's block-level gating absorbs the disentanglement benefit. FiLM-broadcast-scalar axis closed. Alphonse reassigned to #4168 (GEGLU readout head).
- **#4137 CLOSED** (frieren GEGLU+mlp_ratio=2): regression +1.58%; wall-clock saturation. mlp_ratio axis CLOSED for GEGLU too.
- **#4107 SENT BACK** (tanjiro slice_num=8): won old bf16 baseline (-2.13%) but needs full stack (bf16+GEGLU+SF) rebase. Predicted target val ≤ 40.
- **frieren reassigned** (#4155 SwiGLU).
- **alphonse reassigned** (#4168 GEGLU readout head).
- **fern now idle** — needs next assignment this cycle.
