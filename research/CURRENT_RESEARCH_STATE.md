# SENPAI Research State

- **Date**: 2026-05-16 (Loop 15)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**SmoothL1 loss merged** (Loop 15 win via PR #3127). NEW BEST: val_avg=**94.972**, test_avg=**85.037**. -15.0% uniform improvement across all 4 val splits. Previous best was val_avg=111.747 (PR #3478 narrow+bf16).

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18** + **SmoothL1 loss (beta=1.0)**.

Loop 10 result: **#3620 depth n_layers=7 CLOSED (+19% regression)**. Depth axis closed at n_hidden=128. Budget compression dominates (+38% per-epoch cost → 13 realized epochs vs 18). Per-split signature also fails: camber_rc did NOT improve disproportionately. Depth is NOT the binding capacity lever for camber_rc.

Edward reassigned to **#3676 slice_num=48** — tests whether slice_num=64 is over-parameterized at the narrow trunk. Expected: reduced compute (~-15%) → 21 realized epochs vs 18, more cosine completion.

Loop 12 result: **#3621 batch_size=8 CLOSED (+45% regression)**. Per-epoch wall barely budged (+6%), so 2× batch just halves grad steps. Critical systemic finding from frieren's analysis: **narrow+bf16 baseline at 18 epochs is itself underfit** (loss still descending at last cosine-annealed epoch). Frieren reassigned to **#3719 lr=5e-4→1e-3** — directly attacks the underfit finding by doubling effective lr integral over the schedule. Zero budget cost.

**Infrastructure issue (cleared)**: 7 PRs flagged `stale_wip` in Loop 11 due to GH API rate limit blocking student polling. Recovery began in Loop 12: askeladd rebased #3127 to CLEAN (was DIRTY), frieren posted #3621 results. Remaining 5 stale PRs (#3555, #3585, #3588, #3589, #3624) are CLEAN — no conflicts, just waiting for rate-limit windows.

## MAJOR SYSTEMIC FINDING: camber_rc-as-discriminator (REDUCED by SmoothL1)

SmoothL1 (#3127) reduced camber_rc's discriminator gap substantially: now 103.78 vs val_avg 94.97 (+9.3% gap vs +8.6% for cruise). Still the highest-error split but less of an outlier than before. Splits are now more clustered (75-111 range vs 89-134 before).

| Split | SmoothL1 (#3127) | MSE (#3478) | MSE (#3136) | Cumulative |
|---|---|---|---|---|
| single_in_dist | 110.85 | 133.64 | 158.79 | −30.2% |
| geom_camber_rc | **103.78** | 121.33 | 127.26 | **−18.5%** |
| geom_camber_cruise | 75.84 | 88.92 | 102.20 | −25.8% |
| re_rand | 89.42 | 103.10 | 117.04 | −23.6% |

camber_rc is still the hardest split but SmoothL1's de-emphasis of outlier gradients narrowed the gap substantially.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Status |
|---|---|---|---|---|
| #3763 | askeladd | Loss formulation | SmoothL1 beta sweep (0.5, 0.25): push L1/L2 transition lower | dispatched |
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
- **#3719 frieren sent back** (Loop 14): lr=1e-3 +0.85% regression but asymmetric per-split (camber_rc IMPROVED -1.69%). Warmup variant dispatched to fix early-epoch destabilization.
- **#3127 MERGED** (Loop 15): SmoothL1 −15.0% win. New best 94.97/85.04. All 7 in-flight WIPs notified of new baseline.
- **#3763 askeladd dispatched** (Loop 15): beta sweep (0.5, 0.25) on new SmoothL1 baseline.

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

1. **SmoothL1 beta=0.25 or lower** (if #3763 beta sweep confirms monotone improvement) — push toward pure L1 limit.
2. **Pure L1 loss** (`F.l1_loss`) — the theoretical limit of the SmoothL1 beta→0 direction.
3. **lr=1e-3 + warmup** (frieren #3719 in progress) — warmup variant to fix early-epoch destabilization.
4. **Per-channel pressure weighting** — give surf_p 4× weight within surface loss. Zero compute cost. Direct primary-metric attack.
5. **slice_num=96** (if #3676 slice_num=48 regresses) — validates capacity-bottleneck finding.
6. **torch.compile determinism check** — rerun compile + eager at new narrow+bf16+SmoothL1 config.
7. **Wider trunk + bf16 only** (n_hidden=192, SmoothL1) — tests width × loss compounding.

**All in-flight PRs must rebase onto advisor branch (SmoothL1 merged) before training.**

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
