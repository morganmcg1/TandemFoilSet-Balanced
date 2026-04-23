# SENPAI Research State

- **Updated:** 2026-04-23 (round 3b — post-review)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 93.127** (PR #11, `frieren/l1-sw1`, W&B run `yt7eup38`)
- Per-split val: in_dist=106.92 | camber_rc=106.14 | camber_cruise=73.28 | re_rand=86.16
- Config: L1 loss, **surf_weight=1**, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4, bs=4

**Key insight from round 3:** surf_weight=1 (no surface upweighting) dominates under L1 loss — volume supervision is load-bearing for the shared feature extractor. This is a departure from the MSE-era intuition.

**Scoring bug (GH #10) FIXED** — cherry-picked from thorfinn's PR #9 (commit 7d71abd). `test_avg/mae_surf_p` now produces finite numbers (except test_geom_camber_cruise NaN, unblocked separately).

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP | #14: sub-1 surf_weight micro-sweep + AMP compound | `frieren/...` |
| fern     | WIP (r3) | #12: AMP + accum rerun with surf_weight=1 | `fern/throughput-amp` |
| tanjiro  | WIP (r3b) | #15: H-flip augmentation (physics-confirmed, running next) | `tanjiro/horizontal-flip-augmentation` |
| nezuko   | WIP (r3b) | #6: LR floor + WSD on **sw=1** (rerun) | `nezuko/lr-schedule-sweep` |
| alphonse | WIP (r3b) | #7: Fourier m-extend {40,80,160} on **sw=1**, no FiLM | `alphonse/fourier-pe-film-re` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r3b) | #9: asinh + **sw=1** 3-seed compound | `thorfinn/pressure-target-reparam` |

**Idle students needing assignment:** none. All 7 GPUs occupied.

## ⚠️ Round-3 systemic footgun

Argparse default for `surf_weight` is still `10.0` in train.py. PR #11 established sw=1 as the winning recipe but did NOT change the default (it passed `--surf_weight 1` at runtime). **Three of four round-3b submissions (PRs #6, #7, #9) silently ran on sw=10** because students rebased the code but didn't realize the flag change was load-bearing. Every send-back now explicitly asks for `--surf_weight 1`. Future assignment templates should include this flag in the reproduce command.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #12 | fern     | AMP (bf16) + grad_accum=2 on **sw=1** | Beat **93.127** |
| #14 | frieren  | surf_weight sub-1 micro-sweep {0.25, 0.5, 1} + AMP compound | Beat **93.127** |
| #15 | tanjiro  | Horizontal-flip augmentation (x-flip, physics-confirmed) on **sw=1** | Beat **93.127** |
| #6  | nezuko   | Rerun: WSD@1e-3 + floor=1e-5 stacked on **sw=1** (2-seed) | Beat **93.127** |
| #7  | alphonse | Fourier m-saturation {40, 80, 160} at σ=1 on **sw=1**, no FiLM | Beat **93.127** |
| #8  | edward   | EMA 0.999 + wider grad-clip ({1, 5, 10, 50}) on L1 sw=1 | Beat **93.127** |
| #9  | thorfinn | asinh-on-**sw=1** 3-seed compound @ s∈{250,300,350,458} | Beat **93.127** |

---

## Recent Decisions Log

- **2026-04-23 21:40** Merged PR #3 (frieren L1): baseline set to 103.036.
- **2026-04-23 21:40** Closed PR #4 (fern capacity): undertrained, not undercapacity — reassigned to throughput (PR #12).
- **2026-04-23 22:20** Cherry-picked scoring.py fix from PR #9 (commit 7d71abd). Closed GH issue #10.
- **2026-04-23 22:20** Sent back PR #8 (edward): wider clip thresholds; rebase on L1.
- **2026-04-23 22:20** Sent back PR #9 (thorfinn): conflict with L1 merge; rebase + compound sweep.
- **2026-04-23 22:35** Sent back PR #7 (alphonse): Fourier σ=1 wins on MSE, needs L1 rebase + per-block FiLM.
- **Round 3:** Merged PR #11 (frieren surf_weight=1): **new baseline 93.127** (−9.62% vs prior).
- **Round 3:** Closed PR #5 (tanjiro channel-weighting): dead end under L1. Tanjiro idle.
- **Round 3:** Sent back PR #12 (fern AMP): 93.29 no longer beats new baseline; rerun with sw=1.
- **Round 3b (22:50):** Sent back PRs #6 (nezuko), #7 (alphonse), #9 (thorfinn) — all ran on sw=10 argparse default, missed the PR #11 runtime flag. Sent back #15 (tanjiro) with physics sign correction (x-flip: negate Ux not Uy; student caught the assignment bug).
- **Round 3b insight:** Asinh + L1 compound confirmed (−9 pts within sw=10). Seed variance at s=458 is std 2.07 — any effect < 3% needs multi-seed confirmation going forward.
- **Round 3b insight:** Fourier m is NOT saturated on L1. Monotonic m=10→20→40 val improvement (97.51→94.11→93.25). m=80+ is the obvious next step.
- **Round 3b insight:** Per-block FiLM alone is a regressor (−9.2% on sw=10); dropping it from next alphonse sweep.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 (round 3).

---

## Current Research Focus and Themes

The research has established a strong new operating point: L1 loss + surf_weight=1. The key insight is that under L1, volume supervision is equally load-bearing to surface supervision for the pressure metric — upweighting surface starves the shared trunk. This is the opposite of the MSE-era intuition.

**Highest-EV in-flight experiments:**

1. **thorfinn #9 r2 (asinh-on-L1, sw=1)** — asinh reparameterization showed 2.9% improvement even on MSE (100.034 val). Under L1 + sw=1, the compound effect could push to 88–90. This is the most promising pending PR.

2. **alphonse #7 r2 (Fourier σ-sweep + per-block FiLM on L1, sw=1)** — Fourier σ=1 showed −9.4% on MSE; orthogonal to L1. Per-block FiLM is the real test of Re-conditioning.

3. **fern #12 r3 (AMP + sw=1)** — throughput infrastructure that could add 4–5 epochs to every subsequent experiment. Force-multiplier if it holds at sw=1.

**Medium-EV:**

4. **edward #8 r2 (EMA + wider clip on L1, sw=1)** — EMA 0.999 is orthogonal to loss choice; clip value >1.0 should preserve gradient direction better.

5. **nezuko #6 r2 (schedule + seeds on L1, sw=1)** — WSD scheduler and LR floor at the new config.

---

## Potential Next Research Directions (Round 4+)

### Immediate high-EV (ready to assign once students idle)
- **sw sub-1 micro-sweep** (sw ∈ {0.25, 0.5}) — frieren's suggestion: edge of sweep, true minimum may be below 1. Quick 2-run check.
- **Horizontal-flip augmentation with Uy sign-flip** — physics-exact 2× data doubling. ~80 LOC. Orthogonal to all current changes.
- **Near-surface volume-band weighting** (3-tier: far-vol / near-vol / surf) — boundary-layer nodes need more gradient; sw=1 treats all volume equally. Target: 3–5% further gain.
- **SwiGLU feedforward replacement** — replace standard MLP in each TransolverBlock with SwiGLU (~30 LOC). Known wins in transformers. Orthogonal.

### Mid-term architectural
- **Cross-attention decoder with surface/volume query separation** — dedicated surface head at ~120 LOC. Directly targets the primary metric.
- **Attention temperature schedule** — anneal hardcoded `self.temperature=0.5` from high→0.5 in first 20% of training. ~10 LOC.
- **Sample-wise per-sample normalization** — addresses 10× per-sample y_std variance; moderate complexity.
- **Coordinate-based slice assignment** (slice projection from (x,z,is_surface) only) — more interpretable spatial partition; may improve OOD.

### Physics-informed
- **Kutta condition soft enforcement at trailing edge** — identify TE node, add |p_upper - p_lower| aux loss. ~150 LOC.
- **Learnable dsdf aux head** — predict signed distance as aux target; forces trunk to encode geometry cleanly. ~40 LOC.
- **Panel-method residual learning** — model predicts viscous correction on top of thin-airfoil prior. High impact, high complexity.

### Compounding plan after in-flight PRs land
If asinh (thorfinn) and AMP (fern) both merge, rerun with combined config as new baseline. Then:
1. H-flip augmentation as standalone PR.
2. Fourier + FiLM on the compounded config.
3. Capacity scaling (h=256) now feasible with AMP+accum — VRAM headroom exists.
