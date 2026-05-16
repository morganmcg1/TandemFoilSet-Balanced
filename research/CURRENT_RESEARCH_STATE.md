# SENPAI Research State

- **Date**: 2026-05-16 (Loop 10)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**Narrow+bf16 regime established** (Loop 7 win via PR #3478). Active best: val_avg=**111.747**, test_avg=**99.307**.

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18**.

Loop 10 result: **#3620 depth n_layers=7 CLOSED (+19% regression)**. Depth axis closed at n_hidden=128. Budget compression dominates (+38% per-epoch cost → 13 realized epochs vs 18). Per-split signature also fails: camber_rc did NOT improve disproportionately. Depth is NOT the binding capacity lever for camber_rc.

Edward reassigned to **#3676 slice_num=48** — tests whether slice_num=64 is over-parameterized at the narrow trunk. Expected: reduced compute (~-15%) → 21 realized epochs vs 18, more cosine completion.

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
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) — DIRTY, needs rebase onto #3478 baseline | stuck/wip |
| #3555 | alphonse | Data augmentation | Coord jitter σ=0.01 on (x,z) — rebasing onto new baseline | wip |
| #3585 | fern | Loss/data distribution | Per-domain loss weight: racecar_tandem 2.0× — rebasing | wip |
| #3588 | tanjiro | Optimizer (meta) | Lookahead(AdamW, k=5, α=0.5) — rebasing | wip |
| #3589 | thorfinn | Weight averaging | SWA tail (last 3 epochs) — rebasing | wip |
| #3621 | frieren | Training dynamics | batch_size 4→8: lower gradient variance (63 GB headroom) | wip |
| #3624 | nezuko | Architecture (norm) | Scale-only LayerNorm (bias=False): channel-mean test | wip |
| #3676 | edward | Architecture (slice) | slice_num 64→48: over-parameterization test (dispatched Loop 10) | dispatched |

## Recent decisions

- **#3620 CLOSED** (Loop 10): n_layers=7 regressed +19%. Depth axis closed. Budget compression dominated; per-split signature also falsifies camber_rc-specific depth hypothesis.
- **#3676 edward dispatched**: slice_num=48 — over-parameterization diagnostic. Reduces compute, extends realized epochs to ~21. Either validates or refutes #3500 "slice is capacity-bottlenecked" finding.
- **#3127 askeladd still stuck**: 3h+ without rebase commit. Pod cycling between GH rate-limit hits. Will close and reassign SmoothL1 if still DIRTY at Loop 11.

## Systemic findings (load-bearing context)

1. **camber_rc-as-discriminator (7-way confirmed, depth does NOT fix it)**: smallest relative improvement from every intervention. Still the hardest split at 121.33. Root cause unknown — favored hypotheses: distribution gap (camber angles not well-covered in train domains), or geometric inductive bias limitation in PhysicsAttention at this slice_num.
2. **Budget constraint resolved (narrow+bf16)**: 18 full cosine-annealed epochs in 30 min. The wider-trunk budget-wall is gone.
3. **Depth axis CLOSED at n_hidden=128**: +38% per-epoch cost → 13 realized epochs vs 18. Even setting aside budget compression, camber_rc signature falsifies depth as the binding lever.
4. **torch.compile throughput win is real**: 22% speedup validated, no graph breaks. Quality cost comes from kernel fusion; may be addressable with determinism check / TF32 tuning.
5. **RMSNorm matched-epoch quality signal is real**: +8% at matched epochs from bias-drop + mean-subtraction. But F.rms_norm backward kernel not parity with LayerNorm on this Blackwell stack.
6. **Gated-FFN (SwiGLU) axis exhausted at narrow trunk**: MLP is too small a FLOPs fraction to benefit.
7. **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked** (#3500 finding — being tested by #3676 slice_num=48 going down first).
8. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural).
9. **Low-dimensional per-sample conditioners ruled out** (#3287 FiLM).

## Priority candidates if students free up next

1. **SmoothL1 reassignment** (if #3127 closed) — hypothesis is valid (val=114.14 proven on old config), just needs a clean rebased run. Reassign to a healthy pod.
2. **torch.compile determinism check** — rerun compile + eager at same seed/epochs on the new narrow+bf16 config to isolate numerical drift. If drift confirmed, try TF32 (`torch.set_float32_matmul_precision('high')`) as a safe throughput unlock.
3. **Per-channel pressure weighting** — give surf_p 4× weight within surface loss (direct attack on primary metric). Zero compute cost.
4. **slice_num=96** (only if #3676 slice_num=48 regresses) — validates capacity-bottleneck finding; accepts budget compression similar to #3620.
5. **Wider trunk + bf16 without compile** (n_hidden=192, bf16) — untested. ~175s/ep → ~10 epochs. Tests width at budget cost.
6. **gradient checkpointing** — memory reduction to push bs=16 or test slice_num=128.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
