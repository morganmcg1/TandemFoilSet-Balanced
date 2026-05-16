# SENPAI Research State

- **Updated:** 2026-05-16 20:35 UTC
- **Track:** Charlie local-metrics arm (`charlie-pai2i-48h-r1`)
- **Advisor branch:** `icml-appendix-charlie-pai2i-48h-r1`
- **Target base:** `icml-appendix-charlie`
- **Team:** 8 students × 1 GPU each, 30-min/run cap, 50-epoch cap

## Most recent human-team direction

None on record for this launch. Default goal: drive `test_avg/mae_surf_p`
down vs the baseline Transolver config in `target/train.py`.

## Current best baseline

**val_avg/mae_surf_p = 50.57**, **test_avg/mae_surf_p = 43.94** (PR #4105,
GEGLU FFN on bf16 + FiLM-Re+AoA, best epoch 23, still descending at termination).

Per-split val: single=56.18, rc=63.01, cruise=32.57, re_rand=50.52.
Per-split test: single=49.90, rc=56.89, cruise=26.45, re_rand=42.52.

**Total improvement from calibration baseline:** 143.52 → 50.57 = **-64.8%**

## Round wins merged (R1–R15)

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
| **#4105** | **GEGLU FFN on bf16** | **50.57** | **-14.4%** | **MERGED — current baseline** |

## Key architecture (current baseline configuration)

| Group | Value |
|-------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=12, mlp_ratio=1 |
| Conditioning | FiLM head [log_Re, AoA0, AoA1] (3-scalar → per-block γ,β) |
| Precision | bf16 autocast (forward + loss; reductions in fp32) |
| FFN | **GEGLU gating** (PR #4105): `FFN(x) = W2(GELU(W1a(x)) * W1b(x))` |
| Optim | AdamW, lr=5e-4, weight_decay=1e-4, batch=4, cosine T_max=50 |
| Loss | SmoothL1 (Huber, beta=0.25), surf_weight=10.0 |
| EMA | decay=0.997, applied as val/test checkpoint |
| Compute | ~78.9s/epoch, **23 epochs** in 30-min cap, peak VRAM 25.7 GB |

## Dominant discovery: orthogonal wins compound; compute headroom still binding

In one day (R14→R15) we landed two -14% jumps that stack: bf16 (#4064) cut sec/epoch -27% → +7 epochs → -14.1%. GEGLU (#4105) +6% sec/epoch but per-epoch quality gain dominates → -14.4%. Total -27% in 2 PRs. **The model is still descending at termination in BOTH wins** — the compute-bound thesis still holds.

Three orthogonal levers still in play:
1. **Fewer seconds per epoch** — torch.compile (nezuko, pending bf16+GEGLU retest), n_layers=4 (edward, pending bf16+GEGLU retest)
2. **More informative gradient per step** — Schedule-Free AdamW (fern, pending), cosine T_max=epochs (queued for thorfinn)
3. **More expressive per-step update** — SwiGLU (next probe), GEGLU + mlp_ratio=2 (next probe), FiLM-two-stage (alphonse, pending bf16+GEGLU retest), per-node geometric conditioning (next-axis idea)

## Currently in flight (8 WIP — all students active, zero idle GPUs)

| PR | Student | Hypothesis | Theme | Status |
|----|---------|------------|-------|--------|
| #4041 v2 | alphonse  | FiLM two-stage (base+geom, is_tandem gate) on bf16+GEGLU | FiLM architecture | WIP (sent back for bf16+GEGLU rebase) |
| #4068 | edward    | n_layers 5→4 on bf16+GEGLU | compute | WIP (sent back for bf16+GEGLU rebase) |
| #4069 | nezuko    | torch.compile(dynamic=True) on bf16+GEGLU | compute | WIP (sent back for bf16+GEGLU rebase) |
| #4071 | fern      | Schedule-Free AdamW on bf16+GEGLU | optim | WIP (sent back for bf16+GEGLU rebase) |
| #4107 | tanjiro   | slice_num 12→8 on bf16 (pre-GEGLU) | compute | WIP — needs bf16+GEGLU rebase if wins old baseline |
| #4134 | thorfinn  | Cosine T_max 50→25 (align to actual epochs) | LR schedule | WIP — newly assigned R16 |
| #4136 | askeladd  | batch=8 + lr=1e-3 (linear scaling) on GEGLU | data parallelism | WIP — newly assigned R16 |
| #4137 | frieren   | GEGLU + mlp_ratio 1→2 (double FFN intermediate dim) | FFN capacity | WIP — newly assigned R16 |

**Note on the 4 sent-back PRs:** All four (#4041 v2, #4068, #4069, #4071) had architectural/optimization wins against the OLD baseline (68.80 fp32) but couldn't merge once the new bf16 baseline (59.08), then bf16+GEGLU baseline (50.57), landed underneath them. Each is fully orthogonal to the merged wins, so a compound win is expected on rebase.

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
| batch=8 (no LR scaling) | CLOSED — needs linear-scaling lr=1e-3 variant |
| FiLM-full naive (11 scalars) | CLOSED — sent back to two-stage v2 (in flight) |

## Potential next research directions (post-R15)

1. **GLU-family probes:** SwiGLU (SiLU gate) vs GEGLU comparison. Llama uses SiLU; might edge GEGLU on smooth pressure fields. Single-line change.
2. **GEGLU + mlp_ratio=2:** With +245k FFN params already paying for itself, doubling FFN width is the natural next capacity bump.
3. **GEGLU in mlp2 readout head:** The final readout MLP is still vanilla `Linear → GELU → Linear`. Replicate the win there.
4. **Cosine T_max=epochs_reached (queued for thorfinn):** At epoch 25 the baseline's effective LR is 50% of peak (under-annealed). Shorter T_max would let LR fully decay → sharper minimum on this 25-epoch budget.
5. **Longer training off-policy run:** GEGLU was still descending at 1.7 pts/epoch at terminal epoch 23 within the 30-min cap. A 45-60 min run would test whether the late-training compounds.
6. **Per-node geometric conditioning:** Signed-distance or surface-normal features fed at each FiLM site (not broadcast scalars). Bigger architectural step.
7. **batch=8 + linear-scaling lr=1e-3:** Revisit batch size after GEGLU stabilizes — this time with linear-scaling LR.
8. **pad_collate ceiling investigation:** sec/epoch wall-clock floor identified as `pad_collate(max_n)` + Python/dataloader, not compute. Profiling could surface 10-20% throughput gains.
9. **Multi-seed confirmation of bf16+GEGLU baseline:** Before ICML deadline, 3 seeds of the current config to tighten the variance estimate (currently ±5-10 pts).

## Round 16 dispatched (R16, 3 new assignments)

| PR | Student | Hypothesis | Rationale |
|----|---------|------------|-----------|
| #4134 | thorfinn  | Cosine T_max 50→25 | Surface the under-annealed cosine tail; #4109 confirmed peak-LR saturated, schedule shape is the lever |
| #4136 | askeladd  | batch=8 + lr=1e-3 (linear scaling) | Tests Goyal et al. 2017 linear-scaling on GEGLU regime; corrects #4104's under-training |
| #4137 | frieren   | GEGLU + mlp_ratio 1→2 | Capacity bump on the new FFN axis frieren just opened; mlp_ratio=1 closure was for vanilla GELU only |
