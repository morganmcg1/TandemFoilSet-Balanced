# SENPAI Research State

- **Date:** 2026-05-16 14:55 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)
- **PR #3906 (tanjiro clip=0.25) MERGED 14:32 UTC**: baseline 81.660 → **80.893** (−0.94% absolute, −3.42% paired).
- **PR #3758 (fern n_layers=4 R3) CLOSED 14:50 UTC**: paired Δ +3.14% val_avg / +3.34% test 3-split on the clip stack. **Depth axis now fully closed (both directions regress under clip).** Fern reassigned PR #4012 Sobolev gradient loss.
- **Baseline stack flag:** `--grad_clip_norm 0.25` is now the standard. Edward #3985, frieren #3980, fern #4012 all on clip=0.25.

## Current research focus

**Eight axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM + grad_clip_norm=0.25.

**Current best: 80.893 val_avg/mae_surf_p** (tanjiro #3906, clip=0.25 on full FiLM stack, 2026-05-16 14:32)

⚠️ **STACK STALENESS WARNING:** A few in-flight experiments (#3390, #3594, #3492, #3777) are still running with clip=1.0 or no clip. Apply the rebase-if-positive-Δ protocol. Note baseline is now 80.893 with clip=0.25.

### Mechanism map after the clip=0.25 merge

| Role | Mechanism that owns it |
|---|---|
| Loss alignment with eval metric | Huber β=1.0 |
| Compute throughput / epoch count | bf16 AMP |
| Schedule shape over 19-epoch budget | Cosine T_max=15 |
| Trajectory smoothing (long time scale) | EMA decay=0.999 |
| Physics-parameter conditioning | FiLM (single + two-shot) |
| Per-step optimizer geometry (direction normalization) | grad_clip_norm=0.25 |

This carving emerged from the n_layers=4 R3 closure: **two mechanisms for the same role don't compound** — depth=4 was a third path to late-training stability that gradient clipping now subsumes. Future experiments should target an unowned axis, not a redundant one.

### Open mechanism question

Clip=0.25 at 100% clip rate is equivalent to effective_lr = lr × (0.25/||g||) per step. **Is the win from purer direction or smaller effective LR?** Tighter clip sweep (PR #4003) and a dedicated LR-disentangle experiment will resolve.

### Closed this iteration

- **Per-block FiLM (frieren #3829) CLOSED 13:28:** Paired Δ −0.53% val_avg averaged over 2 runs (R1 +0.29%, R2 −1.34%). Confounded design (per-block + per-site γ,β), +19.5% params (13× predicted). FiLM injection-count + per-block-capacity axes now saturated at n_hidden=128.
- **Lookahead (edward #3830) CLOSED 13:35:** Paired Δ +0.42% val_avg / +1.45% test 3-split. Mechanism real (val trajectory ~11% smoother) but redundant with EMA(0.999). Trajectory-smoothing role occupied by EMA.
- **n_layers=4 (fern #3758) CLOSED 14:50:** R3 on clip stack showed +3.14% val / +3.34% test regression. R1/R2 evidence was on the pre-clip stack — the late-training stability that depth=4 was providing is now owned by gradient clipping. **Depth axis fully closed: n_layers=5 is the right depth.**

### Key in-flight experiments (14:55 UTC)

| Student | PR | Hypothesis | Status |
|---------|----|-----------|--------|
| **fern** | **#4012** | **Sobolev / edge-gradient L1 on surface pressure (4-arm weight sweep)** | **Just assigned 14:55 UTC — first physics-aware-loss axis in flight** |
| tanjiro | #4003 | Clip thresh R2: {0.05, 0.1, 0.15, 0.25 control} | Assigned 14:33 UTC; on clip=0.25 baseline |
| edward | #3985 | AGC (per-group adaptive clip) vs global clip=0.25 | Assigned 13:37; notified clip=0.25 control |
| frieren | #3980 | Lion + clip=0.25 vs AdamW + clip=0.25 | Assigned 13:27; notified clip=0.25 update |
| thorfinn | #3390 | T_max=20 on clip=1.0 stack (R2) | Training; on stale clip=1.0 both arms |
| alphonse | #3594 | SF-AdamW + clip R2 | Training; notified clip=0.25 baseline shift |
| nezuko | #3492 | n_hidden=192 + clip=1.0 (R3 rebase) | Rebasing; will be stale on clip=0.25 when done |
| askeladd | #3777 | SDF input features (geometric inputs) | Training (started ~13:43 UTC after pod restart) |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM — attn + MLP per block, shared module, +0 params (frieren) | −3.05% vs FiLM baseline | 89.784 |
| #3511 | Grad clip=1.0 — direction normalization on bf16+FiLM stack (tanjiro) | −9.05% vs two-shot FiLM | 81.660 |
| **#3906** | **Clip=0.25 — tighter direction normalization; 100% clip rate entire run (tanjiro)** | **−3.42% paired vs clip=1.0** | **80.893** |

## Falsified / closed hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs |
| #3758 | n_layers=4 R3 on clip stack | +3.14% val / +3.34% test regression | Depth=4 subsumed by clip; two mechanisms for same role don't compound |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier; test direction +0.86% |
| #3365 | batch_size=6/8 on bf16 | bs=4 best (monotonic regression) | GPU compute-bound; bigger batch cuts SGD steps |
| #3684 | slice_num=32/64/96 on full FiLM stack | 64 best (+3.8-4.9% for 96 and 32) | 64 is knee point at n_hidden=128 capacity |
| #3681 | Three-shot FiLM (preprocess injection) | +4.08% regression | Shared head over-stretched; preprocess is wrong site |
| #3829 | Per-block FiLM heads (per-site γ,β) | Paired Δ −0.53% (noise floor) | FiLM injection-count + per-block-capacity saturated at n_hidden=128 |
| #3830 | Lookahead wrapper (k=6, α=0.5) | Paired Δ +0.42% val / +1.45% test | Trajectory smoothing redundant with EMA(0.999) |

## Cosine schedule axis (active — two parallel attacks in flight)

Current T_max=15 is **suboptimal for bf16's 19-epoch budget** — floors at LR=5e-8 by epoch 16, wasting 3 epochs:
- **Extend: T_max=20 (#3390 R2)** — R1 showed 88.229 on bf16-only
- **Eliminate: SF-AdamW (#3594 R2)** — R1 showed −20.75% on pre-two-shot stack

Whichever has the larger paired Δ on full FiLM+clip stack wins. Both may compose with other hypotheses.

## Depth axis (CLOSED)

Fully mapped after #3758 R3 closure:
- n_layers=3: untested (no longer prioritized — n_layers=4 already regressed under clip)
- **n_layers=4 (#3758 R3): +3.14% val / +3.34% test regression on clip stack** ← closed
- n_layers=5: current baseline ✓
- n_layers=6 (#3595): +2.47% regression

Pre-clip R1/R2 evidence for n_layers=4 was a proxy for the late-training stability that gradient clipping now owns directly. **No further depth sweeps planned.**

## Slice-num axis (CLOSED)

slice_num=64 is the optimum at n_hidden=128 capacity. Revisit sn=96 only if n_hidden=192 (#3492) merges.

## FiLM injection-count + per-block-capacity axes (CLOSED)

- Single-shot: merged (PR #3122, −4%)
- Two-shot: merged (PR #3584, −3.05%)
- Three-shot: closed (PR #3681, +4.08%)
- Per-block heads (per-site γ,β): closed (PR #3829, paired Δ −0.53% at noise floor)
- Both injection-count and per-block-capacity at saturation at n_hidden=128. May unlock at higher capacity if nezuko #3492 R3 merges.

## Potential next hypotheses (not yet assigned)

All future experiments must include `--grad_clip_norm 0.25` in the full stack.

1. **Tighter clip threshold (ASSIGNED #4003, tanjiro)** — {0.05, 0.1, 0.15} vs 0.25 control. Does the monotone curve continue? If yes → tighter is optimal, points toward Lion (sign = clip→0 limit). If plateau/regression → clip=0.25 is optimal.
2. **Lion optimizer (ASSIGNED #3980, frieren)** — Tests Lion's sign projection (L∞ norm) vs AdamW + clip=0.25 (L2 norm).
3. **AGC (ASSIGNED #3985, edward)** — Tests per-parameter-group adaptive clip vs global 0.25.
4. **Sobolev / edge-gradient loss (ASSIGNED #4012, fern)** — First physics-aware-loss axis. Penalizes high-frequency error along the surface contour. Targets a fully unowned axis (loss formulation).
5. **LR vs clip disentangle** — Does clip=0.25 win because of direction purity or smaller effective LR? Test: clip=inf (no clip) + lr=1.25e-4 vs clip=0.25 + lr=5e-4. If they match, it's a LR effect; if clip=0.25 still wins, direction matters independently.
6. **T_max=20 + clip=0.25** — thorfinn #3390 R2 running clip=1.0; if wins, resend with clip=0.25.
7. **SF-AdamW + clip=0.25** — alphonse #3594 R2 in flight; if wins with clip=1.0, resend with clip=0.25.
8. **n_hidden=192 + clip=0.25** — nezuko #3492 R3 running clip=1.0; highest-EV experiment (R2 Δ −8.21%).
9. **Wider FiLM MLP hidden** — `film_mlp_hidden=256`; tests conditioner body capacity. Conditional on #3492 R3 outcome.
10. **Mixup / CutMix data augmentation** — may help cross-geometry split.
11. **Edge-gradient on velocity (Ux/Uy)** — extension of Sobolev once the pressure-only version is decided. Different physical role; orthogonal to surface pressure smoothness.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --grad_clip_norm 0.25`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover. Hit at 14:53 UTC — wait until ~15:30 to resume PR-comment polling cleanly.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Two-mechanisms-for-same-role insight (from #3758 closure):** When R1/R2 evidence for a hypothesis comes from a stack the new baseline has since changed, re-running on the new stack often reveals the original mechanism was a proxy. Pre-clip vs post-clip mechanism check is now a standard test for any axis whose paired-Δ was within ±2% of noise.
- **Cosine T_max=15 wasted-epochs insight:** LR floors at epoch 16 on bf16's 19-epoch budget. Largest unfixed inefficiency. Two parallel attacks in flight.
- **FiLM-owns-velocity insight:** Fourier R4 per-split shows mae_surf_Ux +4.64% with Fourier — FiLM owns velocity channel.
- **slice_num=96 + n_hidden=192 interaction:** student suggested they may compose; hold for post-nezuko evaluation.
- **Stack staleness pattern:** A few PRs are still on the pre-clip=0.25 stack and need rebase verifies (thorfinn #3390, nezuko #3492, alphonse #3594).
- **Seed variance ~±1.5-2%:** Single-seed measurements can be misleading near the noise floor — paired Δ within session is the reliable signal; absolute deltas need confirmation across seeds.
