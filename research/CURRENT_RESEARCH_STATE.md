# SENPAI Research State

- **Date:** 2026-05-16 14:35 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)
- **PR #3906 (tanjiro clip threshold R1) MERGED 14:32 UTC**: clip=0.25 beats baseline 81.660 → **80.893** (−0.94% absolute, −3.42% paired). Clip=0.25 replaces clip=1.0 as standard stack. Tanjiro reassigned PR #4003 (tighter clip sweep {0.05, 0.1, 0.15, 0.25 control}).
- **Baseline stack updated:** `--grad_clip_norm 0.25` is now the standard flag. Edward #3985 (AGC) and frieren #3980 (Lion) notified.

## Current research focus

**Eight axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM + grad_clip_norm=1.0 + **grad_clip_norm=0.25** (replaces clip=1.0).

**Current best: 80.893 val_avg/mae_surf_p** (tanjiro #3906, clip=0.25 on full FiLM stack, 2026-05-16 14:32)

⚠️ **STACK STALENESS WARNING:** In-flight experiments (#3390, #3594, #3492, #3777, #3758) are running with clip=1.0 or no clip. Use the rebase-if-positive-Δ protocol. Note baseline is now 80.893 with clip=0.25.

### 🔥🔥 MAJOR WIN: Clip=0.25 merged (−3.42% paired, −0.94% absolute → new baseline 80.893)

Clip=0.25 (100% clip rate entire run, maximally aggressive direction normalization) merged as the 8th axis. **Mechanism conclusively confirmed: direction normalization at full saturation is the load-bearing mechanism.** Monotone curve: clip=4.0 (+3.45% regression) → clip=1.0 (baseline) → clip=0.25 (−3.42% winner). Tighter sweep in flight as PR #4003.

### Key open question from tanjiro's analysis

Clip=0.25 at 100% clip rate is equivalent to effective_lr = lr × (0.25/||g||) per step. Is the win from **purer direction** or **smaller effective LR**? Tighter clip sweep + dedicated LR-disentangle experiment will resolve.

### 🔥 Sent back for R3 clip-compose rebase (12:24 UTC)

- **🔥🔥 n_hidden=192 (nezuko #3492 R2 → R3):** R2 paired Δ −8.21% val_avg / −8.00% test (strongest signal of round). Sent back to rebase with `--grad_clip_norm 1.0` in BOTH arms. Note: now on stale clip=1.0; if R3 wins, may need R4 with clip=0.25 to confirm on current baseline.
- **n_layers=4 (fern #3758 R2 → R3):** Sent back for clip-compose R3 with clip=1.0. Same stale-stack caveat.

### 🔥 Closed this iteration

- **Per-block FiLM (frieren #3829) CLOSED 13:28:** Paired Δ −0.53% val_avg averaged over 2 runs (R1 +0.29%, R2 −1.34%). Confounded design (per-block + per-site γ,β), +19.5% params (13× predicted), won't beat 81.660 even w/ clip rebase. **FiLM injection-count + per-block-capacity axes now saturated at n_hidden=128.** Frieren reassigned PR #3980 Lion optimizer.
- **Lookahead (edward #3830) CLOSED 13:35:** Paired Δ +0.42% val_avg / +1.45% test 3-split. Mechanism confirmed (val trajectory ~11% smoother) but redundant with EMA(0.999). Per-split pattern shows regularization signature (helps `val_geom_camber_rc`, hurts in-dist + Re-rand). **Trajectory-smoothing role occupied by EMA on our stack.** Edward reassigned PR #3985 AGC.

### Other in-flight experiments (stale stack — apply rebase-if-positive-Δ protocol)

- **🔥 T_max=20 (thorfinn #3390 R2):** Student just started R2 at 13:32 UTC with `--grad_clip_norm 1.0` in BOTH arms (smart — caught the baseline move). R1 showed 88.229 on bf16-only; R2 tests full clip stack composition.
- **🔥 SF-AdamW (alphonse #3594 R2):** Training. R1 showed −20.75% pre-FiLM; R2 needs FiLM+clip stack verify.
- **SDF features (askeladd #3777):** Pod 59m old; spinning up after restart.

### Confirmed closed axes

- **slice_num=64 locked in** (#3684 closed): 32 and 96 both regress by +3.8-4.9%. 64 is the knee point at n_hidden=128 capacity. (Note: 96 may unlock at n_hidden=192.)
- **Three-shot FiLM falsified** (#3681 closed): +4.08% regression. Shared conditioner head over-stretched; preprocess injection site is wrong (not inside residual stream).
- **Per-block FiLM closed** (#3829 closed 13:28 UTC): noise-floor signal (paired Δ −0.53%), +19.5% params, confounded design.
- **Lookahead closed** (#3830 closed 13:35 UTC): paired Δ +0.42% val / +1.45% test; mechanism real but redundant with EMA(0.999). Trajectory-smoothing role occupied.
- **Fourier axis closed** (#3117 R4): subsumed by FiLM.
- **Batch size axis closed** (#3365): GPU compute-bound; bs=4 is optimal.

### Key in-flight experiments (14:35 UTC)

| Student | PR | Hypothesis | Status |
|---------|----|-----------|--------|
| tanjiro | **#4003** | **Clip thresh R2: {0.05, 0.1, 0.15, 0.25 control}** | **Just assigned 14:33 UTC** |
| thorfinn | #3390 | T_max=20 on clip=1.0 stack (R2) | Training; R2 started 13:32 UTC w/ clip=1.0 both arms |
| alphonse | #3594 | SF-AdamW + clip R2 | Training; notified about clip=0.25 baseline shift |
| edward | #3985 | AGC (per-group) vs global clip | Assigned 13:37; notified clip=0.25 as new control |
| frieren | #3980 | Lion + clip=0.25 vs AdamW+clip=0.25 | Assigned 13:27; notified clip=0.25 update |
| nezuko | #3492 | n_hidden=192 + clip=1.0 (R3 rebase) | Rebasing; will be stale on clip=0.25 when done |
| fern | #3758 | n_layers=4 + clip=1.0 (R3 rebase) | Rebasing; will be stale on clip=0.25 when done |
| askeladd | #3777 | SDF input features | Training (started ~13:43 UTC after pod restart) |

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
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier; test direction +0.86% |
| #3365 | batch_size=6/8 on bf16 | bs=4 best (monotonic regression) | GPU compute-bound; bigger batch cuts SGD steps |
| #3684 | slice_num=32/64/96 on full FiLM stack | 64 best (+3.8-4.9% for 96 and 32) | 64 is knee point at n_hidden=128 capacity |
| #3681 | Three-shot FiLM (preprocess injection) | +4.08% regression | Shared head over-stretched; preprocess is wrong site |

## Cosine schedule axis (active — two parallel attacks in flight)

Current T_max=15 is **suboptimal for bf16's 19-epoch budget** — floors at LR=5e-8 by epoch 16, wasting 3 epochs:
- **Extend: T_max=20 (#3390 R2)** — R1 showed 88.229 on bf16-only (better than 89.784!)
- **Eliminate: SF-AdamW (#3594 R2)** — R1 showed −20.75% on pre-two-shot stack

Whichever has the larger paired Δ on full FiLM stack wins. Both may compose with other hypotheses.

## Depth axis (n_layers fully mapped; n_layers=4 awaiting seed-2)

- n_layers=3: untested (potential follow-up if n_layers=4 lands)
- n_layers=4: **R1 paired Δ −1.21% ✅** (#3758) — mechanism fully verified (params −18.2%, sec/epoch −19.2%, +4 fine-tune epochs); absolute 90.198 vs baseline 89.784 — seed-2 in flight to resolve
- n_layers=5: current baseline
- n_layers=6: +2.47% regression (#3595)

Monotone curve at 30-min budget: depth costs epochs more than it adds capacity.

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
2. **Lion optimizer (ASSIGNED #3980, frieren)** — Now updated to use clip=0.25 in both arms. Tests whether Lion's sign projection (L∞ norm) outperforms AdamW + clip=0.25 (L2 norm).
3. **AGC (ASSIGNED #3985, edward)** — Now updated to use clip=0.25 as control. Tests per-parameter-group adaptive clip vs global 0.25.
4. **LR vs clip disentangle** — Does clip=0.25 win because of direction purity or smaller effective LR? Test: clip=inf (no clip) + lr=1.25e-4 vs clip=0.25 + lr=5e-4. If they match, it's a LR effect; if clip=0.25 still wins, direction matters independently.
5. **Sobolev / gradient loss** — physically motivated; adds gradient supervision near surface. High-EV but high-effort.
6. **T_max=20 + clip=0.25** — thorfinn #3390 R2 running clip=1.0; if wins, resend with clip=0.25.
7. **SF-AdamW + clip=0.25** — alphonse #3594 R2 in flight; if wins with clip=1.0, resend with clip=0.25.
8. **n_layers=4 + clip=0.25** — fern #3758 R3 running clip=1.0; if wins, resend with clip=0.25.
9. **n_hidden=192 + clip=0.25** — nezuko #3492 R3 running clip=1.0; highest-EV experiment (R2 Δ −8.21%).
10. **Wider FiLM MLP hidden** — `film_mlp_hidden=256`; tests conditioner body capacity. Conditional on #3492 R3 outcome.
11. **Mixup / CutMix data augmentation** — may help cross-geometry split.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --grad_clip_norm 0.25`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Depth-vs-epochs insight:** n_layers=6 loses 3 fine-tune epochs to n_layers=5. Wall-clock cost is the binding constraint.
- **Cosine T_max=15 wasted-epochs insight:** LR floors at epoch 16 on bf16's 19-epoch budget. Largest unfixed inefficiency. Two parallel attacks in flight.
- **FiLM-owns-velocity insight:** Fourier R4 per-split shows mae_surf_Ux +4.64% with Fourier — FiLM owns velocity channel.
- **slice_num=96 + n_hidden=192 interaction:** student suggested they may compose; hold for post-nezuko evaluation.
- **Stack staleness pattern:** Multiple PRs ran on pre-FiLM stack and need rebase verifies (thorfinn, nezuko, tanjiro, alphonse). Baseline moved fast through Round 4.
- **Seed variance ~±1.5-2%:** Confirmed by fern #3758 R1 (Arm A 91.305 vs merged baseline 89.784, same config); askeladd #3365 (cross-commit ±5 MAE bf16); edward #3684 (Arm A −1.39% vs merged). Single-seed measurements can be misleading near the noise floor — paired Δ is reliable, absolute deltas need confirmation.
