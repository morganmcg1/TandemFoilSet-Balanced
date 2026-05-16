<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 12:00 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 6 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 6 in progress. SOTA: frieren #3675 (lr=2e-4) — val=65.2991, test=60.5400.**

This boot closed two more levers (wd, H=160/144) and sent edward back for EMA-on-lr=2e-4.

**Architectural map — what we now know:**
- **n_hidden** (H=128 vs 160/144): CLOSED — H=160 nearly ties SOTA on val, loses on test; H=144 is worse than both. Width is not the bottleneck.
- **wd** (1e-2 vs 1e-3/5e-2): CLOSED — wd=1e-2 is local optimum; both wd directions shift splits orthogonally to in-dist/OOD axis (not a regularization effect).
- **EMA** (d=0.999 on lr=1e-4 stack): preliminary — val margin shrinks from −7 (T_max=50) to −0.56 (T_max=21 = T_max=21 fix already captures most of the signal). RETESTING on lr=2e-4.
- **slice_num** {32, 96}: IN FLIGHT — never tested
- **dropout** {0.05, 0.10}: IN FLIGHT — never tested
- **n_head** {2, 8}: NEWLY ASSIGNED — never tested
- **mlp_ratio** {3, 4}: NEWLY ASSIGNED — never tested (standard transformer uses 4×, we use 2×)

The full current SOTA stack (7 levers stacked across 6 rounds):
1. **Lion optimizer** (PR #3387): sign-based update
2. **bf16 autocast** (PR #3427): 19 epochs in 30 min
3. **grad-clip(max_norm=1.0)** (PR #3427): 100% engagement — direction normalizer
4. **eta_min=1e-5** (PR #3427): floor
5. **T_max=21** (PR #3596): cosine fully traverses productive low-LR zone
6. **lr=2e-4** (PR #3675): 2× default LR — scales Lion sign-update magnitude
7. (under exploration) EMA d=0.999 on lr=2e-4 stack

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| alphonse | #3876 | `slice-num-sweep` | PhysicsAttention slice_num {32, 96} | WIP — Training |
| edward | #3640 | `ema-weights` | EMA d=0.999 on lr=2e-4+T_max=21 stack | WIP — Rerunning (sent back: was on lr=1e-4) |
| frieren | #3801 | `lion-lr-refine` | lr=2.5e-4 done (val=65.40, worse); T_max=25 arm nudged | WIP — Awaiting Arm 2 |
| tanjiro | #3821 | `cosine-plateau-tail` | Both arms done (W&B); awaiting terminal SENPAI-RESULT | WIP — Pending terminal |
| nezuko | #3927 | `mlp-ratio-sweep` | mlp_ratio {3, 4} on SOTA stack | NEW — just assigned |
| fern | #3747 | `vol-loss-p-weight` | Rerunning vol_p {1.25, 1.5} on lr=2e-4 (old run was lr=1e-4) | WIP — Rerunning |
| askeladd | #3880 | `dropout-sweep` | dropout {0.05, 0.10} on SOTA stack | WIP — Training |
| thorfinn | #3925 | `n-head-sweep` | n_head {2, 8} on SOTA stack | NEW — just assigned |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **65.2991**
- `test_avg_nansafe/mae_surf_p` = **60.5400**
- W&B run: `3rvfeq4g` (frieren, group `lion-lr-sweep`, PR #3675)
- Stack: Lion **lr=2e-4**, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21
- VRAM: 33 GB / 96 GB. Best epoch = **19** (FINAL — val still descending at timeout).

## Key research signals — round 6 results

### Merged

| PR | Student | Hypothesis | Result | Decision |
|---|---|---|---|---|
| **#3675** | **frieren** | **lr=2e-4 (vs 1e-4)** | **val=65.30 (−0.44), test=60.54 (−1.16)** | **MERGED** |

### Closed (this boot)

| PR | Student | Hypothesis | Best result | Decision |
|---|---|---|---|---|
| #3751 | thorfinn | wd {1e-3, 5e-2} | val=65.92 (wd=1e-3, best) | CLOSED — wd=1e-2 local optimum; lever exhausted |
| #3745 | nezuko | H=160 + calibrated T_max | val=65.78 (H=160, best) | CLOSED — capacity-width lever closed; T_max calibration confirmed |

### Preliminary / in-flight

| Priority | PR | Student | Hypothesis | Latest signal |
|---|---|---|---|---|
| HIGH | #3640 | edward | EMA d=0.999 on lr=2e-4+T_max=21 | Retesting — prev run was lr=1e-4 (val=65.18 on OLD stack) |
| HIGH | #3801 | frieren | lr=2.5e-4 done worse (+0.10); awaiting T_max=25 arm | Nudged — Arm 2 pending |
| MED | #3821 | tanjiro | plateau-tail: Arm 1 (1.4e-5) val=65.13 (mixed test); awaiting terminal | Pending terminal post |
| MED | #3876 | alphonse | slice_num {32, 96} | Training |
| MED | #3880 | askeladd | dropout {0.05, 0.10} | Training |
| MED | #3747 | fern | vol_p {1.25, 1.5} on lr=2e-4 stack | Re-running (old run was lr=1e-4) |
| NEW | #3925 | thorfinn | n_head {2, 8} on SOTA stack | NEW — just assigned |
| NEW | #3927 | nezuko | mlp_ratio {3, 4} on SOTA stack | NEW — just assigned |

## Critical insights

### EMA + T_max=21 mechanism overlap (confirmed this boot)

Edward's rebased EMA results show **EMA benefit shrinks dramatically on T_max=21 stack**:
- Old stack (T_max=50): EMA d=0.999 → val: 69.86 → 64.51 (−5.34)
- New stack (T_max=21): EMA d=0.999 → val: 65.74 → 65.18 (−0.56)

Implication: T_max=21 already drives LR into the productive low-noise zone where EMA was earning its keep. EMA benefit at T_max=50 was averaging across high-variance mid-training steps that T_max=21 now simply avoids.

**Question remaining**: Does EMA on lr=2e-4 (current SOTA) still give ~−0.5 val? At lr=2e-4 the Lion update magnitude is 2×, so more per-step variance — EMA might still help. Retesting this now.

### wd orthogonality insight (new this boot)

Both wd arms (lower AND higher) shift per-split test errors in the SAME direction (worse on geom_camber_rc, better on geom_camber_cruise). This rules out regularization-strength as the operative mechanism. wd=1e-2 is locally optimal — finer-grained wd tuning very unlikely to help.

### H=160 T_max calibration (confirmed this boot)

H=160's prior underperformance was 95% explained by misaligned T_max (cosine never reached productive low-LR zone). Calibrated H=160 closes to within 0.48 val of SOTA — nearly but not actually beating it. The capacity-width lever is not the path to next SOTA.

## Round-6 directions (in progress)

1. **EMA on lr=2e-4** (edward #3640 rerun): high-confidence expected win of ~−0.5 val if mechanism holds on new stack
2. **LR refinement** (frieren #3801): T_max=25 arm still pending
3. **Cosine-plateau-tail** (tanjiro #3821): Arm 1 val=65.13 but test regressed — mixed result pattern
4. **slice_num** (alphonse #3876): capacity in basis-vector dimension
5. **dropout** (askeladd #3880): first regularization test on SOTA stack
6. **n_head** (thorfinn #3925): NEW — attention partition dimension
7. **mlp_ratio** (nezuko #3927): NEW — MLP capacity dimension (baseline uses 2×, transformers standard is 4×)
8. **vol_p_weight** (fern #3747): finer sweep on lr=2e-4 after old-stack run showed mixed val/test

## Round-7 directions (speculative)

1. **Decouple eta_min from lr** — at lr=2e-4 the ratio eta_min/lr = 0.05 (half the previous 0.1). Test eta_min=2e-5 at lr=2e-4 to restore ratio. NOTE: must test ONLY in combination with lr=2e-4, not in isolation.
2. **H=160 + EMA stack** — if edward's EMA on lr=2e-4 wins, retest H=160 on top (H=160 showed better test on single_in_dist at cost of OOD; EMA might restore OOD).
3. **Different optimizers** — SOAP, Adan, Lion with custom momentum, after exhausting LR/wd levers.
4. **OOD-targeted augmentation** — geom_camber_rc is the persistent worst split (~68 test, ~12 points below in-dist). Multiple students have flagged this. Worth a dedicated augmentation study.
5. **n_layers sweep** — never tested on new stack (L=7 was dead end at old stack; L=4 might be better).
6. **Lookahead optimizer** — wrap Lion in Lookahead outer loop.
7. **SWA / Polyak averaging** — uniform averaging of last N epoch checkpoints (complement to EMA).
8. **Physics-informed loss terms** — mass/momentum conservation constraints.
9. **Researcher-agent generated ideas** — fresh hypothesis list in RESEARCH_IDEAS_2026-05-16.md (generation in progress).

## Eliminated approaches (round 6)

| Approach | Best result | Decision |
|---|---:|---|
| **eta_min raise (tanjiro #3713): {2e-5, 3e-5}** | 67.16 / 68.44 (both worse) | **CLOSED** — raises entire cosine second half |
| **Lion β₁ sweep (askeladd #3749): {0.8, 0.95}** | 70.66 / 70.87 (both +5.4 worse) | **CLOSED** — β=0.9 confirmed optimal |
| **clip sweep (alphonse #3590): {0.25, 0.5, 2.0, off}** | 70.11 best (clip=off worst 76.17) | **CLOSED** — sign-invariance theorem confirmed |
| **lr=2.5e-4 (frieren #3801 Arm 1)** | val=65.40 (+0.10 worse) | **Eliminated** — too high LR, Arm 2 T_max=25 pending |
| **H=160 + calibrated T_max (nezuko #3745)** | val=65.78 (+0.48 worse) | **CLOSED** — T_max calibration explained prior gap; capacity-width not the bottleneck |
| **H=144 + T_max=17 (nezuko #3745)** | val=68.06 (+2.76 worse) | **CLOSED** — worse than both neighbors |
| **wd {1e-3, 5e-2} (thorfinn #3751)** | val=65.92 (+0.62 worse) | **CLOSED** — wd=1e-2 locally optimal; lever orthogonal to OOD gap |

## Eliminated approaches (cumulative)

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
| Lion+clip full sweep (alphonse #3590) | 70.11 (clip=0.25/0.5); clip=off=76.17 (worst) | Sign-invariance theorem confirmed |
| Lion β₁ sweep (askeladd #3749) | 70.66 / 70.87 (both worse by >5) | β=0.9 confirmed optimal |
| wd sweep (thorfinn #3751): {1e-3, 5e-2} | 65.92 (wd=1e-3) — both worse | wd=1e-2 locally optimal; lever orthogonal to OOD |
| H=160/144 + calibrated T_max (nezuko #3745) | 65.78 (H=160) — both worse | T_max calibration confirmed prior gap; capacity-width closed |

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]`. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`.

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` at `slice_num=128`. Future slice_num arms must pair with stability guard.

### 3. senpai-pr-guard.py code-fence bug
Guard picks up template SENPAI-RESULT markers inside code fences as invalid JSON. Advisor-side fix pending.

## Operational notes

- All new assignments use fixed seed (torch.manual_seed(42))
- All PRs use `--wandb_group <slug>` for W&B grouping
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run
- All PRs log `test_avg_nansafe/mae_surf_p` (cruise NaN workaround)
- Watch for GitHub rate-limit issues during PR creation (label fix may be needed)
