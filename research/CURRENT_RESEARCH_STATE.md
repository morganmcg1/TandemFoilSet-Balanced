# SENPAI Research State

- **Date**: 2026-05-16 (Loop 16)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**SmoothL1 baseline holds** (PR #3127). NEW BEST since Loop 15: val_avg=**94.972**, test_avg=**85.037**. Loop 16 closed 4 PRs (#3719 lr=1e-3+warmup, #3624 scale-only LN, #3585 per-domain weight, #3555 coord jitter) all measured on OLD MSE baseline and superseded by SmoothL1's mechanism.

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18** + **SmoothL1 loss (beta=1.0)**.

**Underfit signature persists**: loss still descending at last cosine-annealed epoch across all closed-out experiments. The highest-EV remaining levers attack this directly: more effective epochs (compile-throughput + 23-ep budget-soak in #3802) or better loss-metric alignment (Pure L1 in #3798, per-channel pressure weighting in #3800).

**camber_rc-as-discriminator status**: substantially reduced by SmoothL1 (gap now +9.3% vs val_avg, down from +18.5% on MSE baseline). Per-domain weighting (#3585) and coord-jitter (#3555) attacks on this specific gap are now lower-EV — their mechanisms are partially absorbed by SmoothL1's L1 tail.

## Loop 16 systemic findings (NEW)

1. **Bias-drop alone is NOT the RMSNorm gain mechanism** (#3624 closure, 4-seed replication): scale-only LayerNorm closes only ~14% of the matched-epoch gap from #3496. The remaining ~86% comes from **mean-subtraction removal**. Future norm work should target RMSNorm proper (or mean-subtraction-removal variants), not bias removal. Bias term is mildly load-bearing for late-stage fine-tuning.

2. **A single global lr cannot simultaneously optimize all 4 val splits** (#3719 two-arm closure): per-split convergence profiles diverge. camber_rc is step-quality-limited (wants higher lr); single_in_dist/re_rand are at convergence at lr=5e-4 (higher lr adds noise); cruise is roughly lr-invariant in [5e-4, 1e-3]. Future lr work needs to be per-block (LLRD) or per-domain, not a global scalar.

3. **Seed variance at narrow+bf16 is large** (#3676 finding): 2 runs of identical config produced 5.99 spread on val_avg. `train.py` does not seed torch/numpy/random; sample sampler is RNG-dependent via WeightedRandomSampler. Active mitigation: edward asked to seed-pin in the #3676 send-back. Systemic bug-fix candidate: add `--seed` flag and `torch.manual_seed` at startup.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Status |
|---|---|---|---|---|
| #3763 | askeladd | Loss formulation | SmoothL1 beta sweep (0.5, 0.25): push L1/L2 transition lower | wip |
| #3588 | tanjiro | Optimizer (meta) | Lookahead(AdamW, k=5, α=0.5) — CLEAN, pending | wip |
| #3589 | thorfinn | Weight averaging | SWA tail (last 3 epochs) — CLEAN, pending | wip |
| #3676 | edward | Architecture (slice) | slice_num=48 + 3-seed + 21-ep rebased to SmoothL1 | sent back Loop 16 |
| #3798 | frieren | Loss formulation | Pure L1 loss (F.l1_loss) — beta→0 limit of SmoothL1 axis | dispatched Loop 16 |
| #3800 | fern | Per-channel weighting | surf_p 4× inside surface loss — direct primary-metric attack | dispatched Loop 16 |
| #3802 | alphonse | Throughput | torch.compile determinism + 23-ep budget-soak on SmoothL1 | dispatched Loop 16 |
| #3804 | nezuko | Capacity (width) | n_hidden=160 on SmoothL1 — width × loss compounding test | dispatched Loop 16 |

## Recent decisions

- **Loop 15: #3127 MERGED** SmoothL1 −15.0% win. New best 94.97/85.04. All 7 in-flight WIPs notified of new baseline.
- **Loop 15: #3763 askeladd dispatched** SmoothL1 beta sweep (0.5, 0.25).
- **Loop 16: 4 PRs closed** all measured on OLD baseline + mechanism partially absorbed by SmoothL1:
  - #3719 (lr=1e-3+warmup): student's own conclusion — global-lr axis exhausted
  - #3624 (scale-only LN): 4-seed regression confirms bias-drop alone insufficient; mean-subtraction is the RMSNorm gain mechanism
  - #3585 (per-domain weight 2.0×): too aggressive (cruise crushed); milder weight at narrow trunk also low-EV after SmoothL1
  - #3555 (coord jitter σ=0.01): mixed per-split signal (cruise hurt > rc helped); mechanism mostly absorbed by SmoothL1
- **Loop 16: #3676 SENT BACK** edward — rebased 3-seed re-run on SmoothL1 + 21 epochs (compute savings + Run A's −5.36% on OLD baseline make this a cheap repeat).
- **Loop 16: 4 new dispatches** — frieren (Pure L1), fern (surf_p 4× weight), alphonse (compile+23ep), nezuko (n_hidden=160).

## Systemic findings (load-bearing context)

1. **camber_rc-as-discriminator REDUCED by SmoothL1**: gap from +18.5% to +9.3% vs val_avg. Still hardest split (val_avg 103.78 vs avg 94.97), but less of an outlier. Splits clustered in 75-111 range vs 89-134 before.
2. **Budget constraint resolved (narrow+bf16)**: 18 full cosine-annealed epochs in 30 min on baseline. Compute headroom available via slice_num=48 (~10% savings → ~21 epochs).
3. **Underfit baseline persists**: loss descending at final cosine-annealed epoch in EVERY closed experiment of this round. Suggests more epochs (throughput-via-compile, or slice-num savings, or 23-ep) is the highest-EV lever.
4. **Depth axis CLOSED at n_hidden=128**: +38% per-epoch cost → 13 realized epochs vs 18. Camber_rc signature falsifies depth as the binding lever.
4a. **batch_size axis CLOSED (both wider and narrow trunks)**: bs=8 at narrow+bf16 regressed +45%; per-epoch wall barely changed (+6%) — compute-bound. 2× batch halves grad steps.
5. **lr axis CLOSED at global scalar**: lr=1e-3 +0.85% (no warm) / +0.31% (warmup). Per-split convergence profiles diverge — future lr work must be per-block (LLRD) or per-domain.
6. **LayerNorm bias-drop alone is INSUFFICIENT** for RMSNorm matched-epoch gain (closes ~14% of gap; 86% comes from mean-subtraction). Future norm work should target mean-subtraction removal, not bias removal.
7. **Per-domain weighting at narrow+bf16 ineffective** (#3585): mechanism mostly absorbed by SmoothL1's L1 tail at camber_rc; cruise budget squeeze dominates.
8. **Coord-jitter at σ=0.01 ineffective** (#3555): cost asymmetry (cruise damage > rc help); slice projection already provides soft spatial pooling.
9. **torch.compile throughput win is real (22% speedup, no graph breaks) from prior work** — being re-tested for determinism on SmoothL1 in #3802. If clean, 23 epochs fit in 30 min.
10. **Slice mechanism is capacity-bottlenecked** (#3500): being re-tested by #3676 (slice_num=48 → 3-seed) on SmoothL1 baseline.
11. **Gated-FFN (SwiGLU) axis exhausted at narrow trunk** (prior work): MLP is too small a FLOPs fraction to benefit.
12. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural).
13. **Low-dimensional per-sample conditioners ruled out** (#3287 FiLM).

## Priority candidates if students free up next

1. **SmoothL1 beta=0.25 or lower** (if #3763 confirms monotone improvement) — push toward pure L1 limit (#3798 directly tests pure L1).
2. **lr=5e-4 + lr_min=1e-5 schedule floor** — the lr=1e-3 closure's natural follow-up. Cheapest remaining test of "descent-limited vs noise-limited" at the conservative-lr regime.
3. **slice_num=96** (only if #3676 slice_num=48 rebased + multi-seed confirms current slice_num=64 is over-parameterized).
4. **RMSNorm proper retest** (target the mean-subtraction-removal axis, NOT bias-drop) — possibly with custom triton kernel to fix backward perf.
5. **Per-block LR (LLRD)** — given #3719 closure showed per-split convergence diverges. Block-wise lr lets earlier blocks (handling input features) have smaller updates than later blocks (handling output heads).
6. **lr=2e-3 + 1000-step warmup** — only if any of the per-channel/width experiments suggest lr was under-tuned at the new baseline.
7. **More epochs at lr=5e-4 (e.g. epochs=24 + slice_num=48)** — direct underfit test combining cosine schedule extension with the slice compute savings.
8. **Volume loss per-channel weighting** — vol_p analog of #3800 if it wins; volume residuals are unweighted currently.

**All in-flight PRs use SmoothL1 (beta=1.0) baseline. New baseline target: val_avg < 94.97, test_avg < 85.04.**

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
