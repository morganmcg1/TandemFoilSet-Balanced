# SENPAI Research State

- **Date**: 2026-05-16 (Loop 12)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**Narrow+bf16 regime established** (Loop 7 win via PR #3478). Active best: val_avg=**111.747**, test_avg=**99.307**.

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18**.

Loop 10 result: **#3620 depth n_layers=7 CLOSED (+19% regression)**. Depth axis closed at n_hidden=128. Budget compression dominates (+38% per-epoch cost → 13 realized epochs vs 18). Per-split signature also fails: camber_rc did NOT improve disproportionately. Depth is NOT the binding capacity lever for camber_rc.

Edward reassigned to **#3676 slice_num=48** — tests whether slice_num=64 is over-parameterized at the narrow trunk. Expected: reduced compute (~-15%) → 21 realized epochs vs 18, more cosine completion.

Loop 12 result: **#3621 batch_size=8 CLOSED (+45% regression)**. Per-epoch wall barely budged (+6%), so 2× batch just halves grad steps. Critical systemic finding from frieren's analysis: **narrow+bf16 baseline at 18 epochs is itself underfit** (loss still descending at last cosine-annealed epoch). Frieren reassigned to **#3719 lr=5e-4→1e-3** — directly attacks the underfit finding by doubling effective lr integral over the schedule. Zero budget cost.

**Infrastructure issue (cleared)**: 7 PRs flagged `stale_wip` in Loop 11 due to GH API rate limit blocking student polling. Recovery began in Loop 12: askeladd rebased #3127 to CLEAN (was DIRTY), frieren posted #3621 results. Remaining 5 stale PRs (#3555, #3585, #3588, #3589, #3624) are CLEAN — no conflicts, just waiting for rate-limit windows.

## MAJOR SYSTEMIC FINDING: camber_rc-as-discriminator (confirmed on new baseline too)

Even after the -11.5% win from narrow+bf16, camber_rc improved the LEAST (-4.7% vs -12-16% on other splits). Still the largest absolute gap at 121.33 vs val_avg 111.75. Depth was NOT the solution.

| Split | New val (ep 18) | Old val (#3136) | Delta |
|---|---|---|---|
| single_in_dist | 133.64 | 158.79 | -15.8% |
| geom_camber_rc | **121.33** | 127.26 | **-4.7%** (hardest to close) |
| geom_camber_cruise | 88.92 | 102.20 | -13.0% |
| re_rand | 103.10 | 117.04 | -11.9% |

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Status |
|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) — CLEAN (rebased Loop 12), training pending | wip |
| #3555 | alphonse | Data augmentation | Coord jitter σ=0.01 on (x,z) — CLEAN, training pending | wip |
| #3585 | fern | Loss/data distribution | Per-domain loss weight: racecar_tandem 2.0× — CLEAN, pending | wip |
| #3588 | tanjiro | Optimizer (meta) | Lookahead(AdamW, k=5, α=0.5) — CLEAN, pending | wip |
| #3589 | thorfinn | Weight averaging | SWA tail (last 3 epochs) — CLEAN, pending | wip |
| #3624 | nezuko | Architecture (norm) | Scale-only LayerNorm (bias=False): channel-mean test | wip |
| #3676 | edward | Architecture (slice) | slice_num 64→48: over-parameterization test (Loop 10) | dispatched |
| #3719 | frieren | LR/schedule | lr 5e-4→1e-3 attack on underfit baseline (Loop 12) | dispatched |

## Recent decisions

- **#3620 CLOSED** (Loop 10): n_layers=7 regressed +19%. Depth axis closed. Budget compression dominated; per-split signature also falsifies camber_rc-specific depth hypothesis.
- **#3676 edward dispatched** (Loop 10): slice_num=48 — over-parameterization diagnostic. Reduces compute, extends realized epochs to ~21. Either validates or refutes #3500 "slice is capacity-bottlenecked" finding.
- **#3621 CLOSED** (Loop 12): bs=8 regressed +45%. Per-epoch wall +6% only → compute-bound at narrow+bf16. Halving grad steps with 2× batch is decisive against batch_size axis. Key finding: baseline at 18 epochs is UNDERFIT (loss still descending at final cosine-annealed epoch).
- **#3719 frieren dispatched** (Loop 12): lr=5e-4→1e-3 attacks the underfit-baseline finding directly. Zero budget cost. Risk: bf16 stability ceiling at higher lr.
- **#3127 askeladd unblocked**: rebased to CLEAN in Loop 12. Training run still pending. Rate limit window is opening.

## Systemic findings (load-bearing context)

1. **camber_rc-as-discriminator (7-way confirmed, depth does NOT fix it)**: smallest relative improvement from every intervention. Still the hardest split at 121.33. Root cause unknown — favored hypotheses: distribution gap (camber angles not well-covered in train domains), or geometric inductive bias limitation in PhysicsAttention at this slice_num.
2. **Budget constraint resolved (narrow+bf16)**: 18 full cosine-annealed epochs in 30 min. The wider-trunk budget-wall is gone.
3. **Depth axis CLOSED at n_hidden=128**: +38% per-epoch cost → 13 realized epochs vs 18. Even setting aside budget compression, camber_rc signature falsifies depth as the binding lever.
3a. **batch_size axis CLOSED (both wider and narrow trunks)**: bs=8 at narrow+bf16 regressed +45%; per-epoch wall barely changed (+6%) — compute-bound. 2× batch halves grad steps.
3b. **Narrow+bf16 baseline is UNDERFIT at 18 epochs** (#3621 finding): loss still descending at last cosine-annealed epoch. Suggests LR/schedule tuning is the highest-EV remaining lever. #3719 (lr=1e-3) attacks this directly.
4. **torch.compile throughput win is real**: 22% speedup validated, no graph breaks. Quality cost comes from kernel fusion; may be addressable with determinism check / TF32 tuning.
5. **RMSNorm matched-epoch quality signal is real**: +8% at matched epochs from bias-drop + mean-subtraction. But F.rms_norm backward kernel not parity with LayerNorm on this Blackwell stack.
6. **Gated-FFN (SwiGLU) axis exhausted at narrow trunk**: MLP is too small a FLOPs fraction to benefit.
7. **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked** (#3500 finding — being tested by #3676 slice_num=48 going down first).
8. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural).
9. **Low-dimensional per-sample conditioners ruled out** (#3287 FiLM).

## Priority candidates if students free up next

1. **lr=2e-3 with warmup** (only if #3719 lr=1e-3 wins) — push the underfit attack harder.
2. **lr=5e-4 with `lr_min=1e-5` raised cosine floor** (only if #3719 is neutral) — schedule shape vs peak matters more.
3. **lr=5e-4 + linear warmup** (only if #3719 diverges in bf16) — softer ramp before peak lr.
4. **Per-channel pressure weighting** — give surf_p 4× weight within surface loss (direct attack on primary metric). Zero compute cost. Decoupled from underfit lever.
5. **torch.compile determinism check** — rerun compile + eager at same seed/epochs on the new narrow+bf16 config to isolate numerical drift. If drift confirmed, try TF32 as a safe throughput unlock.
6. **slice_num=96** (only if #3676 slice_num=48 regresses) — validates capacity-bottleneck finding; accepts budget compression similar to #3620.
7. **gradient checkpointing** — memory reduction to push bs=16 or test slice_num=128.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
