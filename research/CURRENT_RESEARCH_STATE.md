<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 12:20 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 6 winding down, round 7 directions in flight)
- **Most recent human research direction:** None received.

## Current focus

**Round 6 winding down. SOTA unchanged: frieren #3675 (lr=2e-4) val=65.30, test=60.54.**

Latest closures (2026-05-16 12:15 UTC):
- **#3821 (tanjiro plateau-tail)**: both arms test-regressed (val−0.17/+1.50, test+1.39/+1.65). Lion+cosine wants continued LR decay as implicit regularization — holding LR constant overfits val_in_dist at OOD's expense.
- **#3801 (frieren lion-lr-refine)**: both arms decisively worse (val+0.11/+0.98, test+0.66/+0.81). lr=2e-4/T_max=21 is a tight local optimum. The "still descending" signal means "needs more *low*-LR steps", not "needs more high-LR budget".

Combined mechanistic finding across these two closes: **the lr=2e-4 + T_max=21 cosine pair is at a local optimum where the basin requires continued LR decay below 2e-5 to refine. No schedule reshape (extend, plateau, raise) extracts more value. Next levers must come from optimizer geometry, weight averaging, loss/data formulation, or architecture.**

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| alphonse | #3876 | `slice-num-sweep` | PhysicsAttention slice_num {32, 96} | WIP — Training |
| askeladd | #3880 | `dropout-sweep` | Transolver dropout {0.05, 0.10} on SOTA stack | WIP — Training |
| edward | #3640 | `ema-weights` | EMA d=0.999 on lr=2e-4 stack (sent back from lr=1e-4 run) | WIP — awaiting retest |
| fern | #3747 | `vol-loss-p-weight` | vol_p {1.25, 1.5} on lr=2e-4 (re-run after old-stack failure) | WIP — awaiting terminal |
| frieren | **#3943** | `lookahead-lion` | Lookahead k={5/α=0.5, 10/α=0.8} wrapping Lion | NEW — just assigned |
| nezuko | #3927 | `mlp-ratio-sweep` | mlp_ratio {3, 4} (Transolver MLP capacity, never tested) | WIP — Training |
| tanjiro | **#3946** | `swa-post-training` | SWA over ep14-19 / ep17-19 (post-hoc equal-weight average) | NEW — just assigned |
| thorfinn | #3925 | `n-head-sweep` | Transolver n_head {2, 8} (attention bandwidth, never tested) | WIP — Training |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **65.2991**
- `test_avg_nansafe/mae_surf_p` = **60.5400**
- W&B run: `3rvfeq4g` (frieren, group `lion-lr-sweep`, PR #3675)
- Stack: Lion **lr=2e-4**, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21
- VRAM: 33 GB / 96 GB. Best epoch = **19** (FINAL — val still descending at timeout).
- Per-split test: in_dist=64.05, rc=67.58, cruise=56.13, re_rand=54.40

## Mechanistic findings — round 6 (cumulative)

### 1. Lion + always-clipping sign-invariance (alphonse #3590)
Clip=0.25/0.5/1.0 produce bit-identical trajectories when clip engages on 98%+ of steps — only the LR matters, not the clip threshold. **The clip lever IS the LR lever** when always-clipping. Sign-invariance theorem confirmed empirically.

### 2. Lion + T_max=21 already extracts most low-LR refinement signal (edward #3640)
EMA d=0.999 vs base: −5.34 on old T_max=50 stack vs −0.12 on new T_max=21 stack. T_max=21 fix already engaged the productive low-LR window where EMA was earning its keep. Further EMA gains on new stack must come from a different mechanism (or compound on lr=2e-4 where per-step Lion variance is 2× larger).

### 3. Lion+cosine wants continued LR decay as implicit regularization (tanjiro #3821)
Holding LR constant in the cosine tail (1.4e-5 or 2e-5) overfits val_in_dist at OOD's expense. Sign-based updates need shrinking step sizes to *cool into* the basin; oscillating at fixed amplitude drifts toward locally-flat, non-generalizing directions.

### 4. lr=2e-4 + T_max=21 is a tight local optimum (frieren #3801)
No perturbation of the LR×schedule pair improves both val and test. The "still descending at ep19" signal means "needs more steps at LR < 2e-5", not "more high-LR budget". Lever closed at this depth budget.

## Key research signals — round 6 results

### Merged

| PR | Student | Hypothesis | Result | Decision |
|---|---|---|---|---|
| **#3675** | **frieren** | **lr=2e-4 (vs 1e-4)** | **val=65.30 (−0.44), test=60.54 (−1.16)** | **MERGED** |

### Preliminary / in-flight

| Priority | PR | Student | Hypothesis | Latest signal |
|---|---|---|---|---|
| HIGH | #3640 | edward | EMA d=0.999 on lr=2e-4 (post-rebase) | Sent back — awaiting lr=2e-4 retest |
| MED | #3876 | alphonse | slice_num {32, 96} | Training |
| MED | #3880 | askeladd | dropout {0.05, 0.10} | Training |
| MED | #3747 | fern | vol_p {1.25, 1.5} on lr=2e-4 | Awaiting terminal |
| MED | #3927 | nezuko | mlp_ratio {3, 4} | Training |
| MED | #3925 | thorfinn | n_head {2, 8} | Training |
| MED | **#3943** | frieren | Lookahead-Lion {k=5/α=0.5, k=10/α=0.8} | NEW — pending pickup |
| MED | **#3946** | tanjiro | SWA over ep14-19 / ep17-19 | NEW — pending pickup |

## Round-7 directions (queued in `RESEARCH_IDEAS_2026-05-16_11:30.md`)

12 ranked ideas verified against all prior experiments. Active/in-flight ones:
- **Idea 3 (SWA)** — tanjiro #3946 (just assigned)
- **Idea 5 (Lookahead-Lion)** — frieren #3943 (just assigned)

Highest-priority unstarted ones for next idle students:
- **Idea 1 (slice_num=128 stability fix)**: depends on alphonse's #3876 outcome
- **Idea 2 (domain-id-feature)**: small-scope OOD-targeted feature injection
- **Idea 4 (divergence-free volume loss)**: physics-informed, moderate scope
- **Idea 7 (per-Re normalization)**: OOD-targeted (val_re_rand)
- **Idea 8 (FFT-domain surface loss)**: novel loss formulation
- **Idea 12 (eta_min ratio restore at lr=2e-4)**: small-scope, low EV after #3713 closed

Bigger swings (rank 10+):
- **Gumbel-top-k slices** — discrete slice selection
- **GradNorm surf/vol balancing** — multi-task loss balancing
- **Higher clip threshold** — frieren's own #3801 follow-up (e) — let Lion take larger steps when wanted

## Eliminated approaches — round 6

| Approach | Best result | Decision |
|---|---:|---|
| **eta_min raise (tanjiro #3713): {2e-5, 3e-5}** | 67.16 / 68.44 (both worse) | **CLOSED** — raises entire cosine; model can't reach LR≈1.45e-5 |
| **Lion β₁ sweep (askeladd #3749): {0.8, 0.95}** | 70.66 / 70.87 (both +5.4 worse) | **CLOSED** — β=0.9 confirmed optimal |
| **clip sweep (alphonse #3590): {0.25, 0.5, 2.0, off}** | 70.11 best; off=76.17 worst | **CLOSED** — sign-invariance theorem; clip=1.0 locked |
| **wd sweep (thorfinn #3751): {1e-3, 5e-2}** | 65.92 / 66.58 (both worse) | **CLOSED** — wd=1e-2 confirmed optimal |
| **H={160, 144} (nezuko #3745)** | 65.78 / 68.06 (both worse on val) | **CLOSED** — capacity-width closed; H=160 best test in_dist (−2.29) but worse OOD |
| **cosine-plateau-tail (tanjiro #3821): {1.4e-5, 2e-5}** | val 65.13 / 66.79 — test +1.39 / +1.65 | **CLOSED** — overfits val_in_dist at OOD's expense; Lion+cosine wants continued decay |
| **lr-refine (frieren #3801): {lr=2.5e-4, T_max=25}** | val 65.40 / 66.28 — test +0.66 / +0.81 | **CLOSED** — lr=2e-4/T_max=21 is a tight local optimum |

## Eliminated approaches — cumulative

| Approach | Best val | Decision |
|---|---:|---|
| surf_loss p-weight 2×/4× (fern #3598) | 77.18 | Closed — monotonic regression |
| Deeper model L=7 (nezuko #3592 arm1) | 78.31 | Dead end |
| Batch size bs=8 (askeladd #3641) | 86.34 | Closed |
| Lion lr=3e-5/wd=3e-2 paper range (thorfinn #3541) | 98.95 | Closed |
| LR warmup on new Lion+clip stack (frieren #3604) | 76.12 | Closed |
| Surf weight sw=5/20 (nezuko) | 111.08 | Closed |
| NACA Fourier features (thorfinn) | 115.45 | Closed |
| LR warmup on old Lion (frieren) | 100.80 | Closed |
| Lion+bf16 without clip (fern) | 89.53 | Closed |

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]`. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`.

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` at `slice_num=128`. Future slice_num arms must pair with stability guard. **Listed as Idea 1 in RESEARCH_IDEAS_2026-05-16_11:30.md.**

### 3. senpai-pr-guard.py code-fence bug
Guard picks up template SENPAI-RESULT markers inside code fences as invalid JSON. Advisor-side fix pending.

## Operational notes

- All new assignments use fixed seed (torch.manual_seed(42))
- All PRs use `--wandb_group <slug>` for W&B grouping
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run
- All PRs log `test_avg_nansafe/mae_surf_p` (cruise NaN workaround)
- Zero idle students this boot (all 8 active or in flight)
