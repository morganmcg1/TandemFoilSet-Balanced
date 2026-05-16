<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~12:35 UTC) — #3748 nezuko MERGED (new baseline val 68.96); frieren #3843 lr=1e-4 is massive (val 65.41, pending terminal); #3817 alphonse closed (FiLM ablation); #3842/#3845 closed; 4 new R8 assignments; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 68.9592** (PR #3748 nezuko — output-only spectral norm + n_fourier=0 + FiLM + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14, run `u42jpd48`)
**test_avg/mae_surf_p = 60.8201** (same run, clean 4-split)

Per-split val: in_dist 77.84, camber_rc 81.38, camber_cruise 49.90, re_rand 66.71
Per-split test: in_dist 69.62, camber_rc 73.21, camber_cruise 40.68, re_rand 59.78

**Δ vs prior best (PR #3672 n_fourier=0, val 70.34 / test 61.63): −1.39 val / −0.81 test**

## Merged sequence (improvement cascade)

| PR | Description | val → val | test → test | Δ |
|----|-------------|-----------|-------------|---|
| #3098 | Huber loss | 135.23 → 96.05 | — | −29.1% |
| #3296 | NaN guard | — → 90.00 | first clean test | — |
| #3444 | cosine T_max=14 | 96.05 → 93.20 | 90.00 → 83.54 | −3.0% / −7.2% |
| #3537 | Lion optimizer | 93.20 → 77.58 | 83.54 → 68.88 | −16.8% / −17.5% |
| #3405 | FiLM+Lion+EMA | 77.58 → 71.65 | 68.88 → 62.11 | −7.9% / −9.8% |
| #3672 | n_fourier=0 | 71.65 → 70.34 | 62.11 → 61.63 | −1.8% / −0.8% |
| **#3748** | **spec_norm(output)** | **70.34 → 68.96** | **61.63 → 60.82** | **−2.0% / −1.3%** |

**Total improvement:** val 135 → 68.96 (−49%), test ~130 → 60.82 (−53%)

## PENDING MERGE — frieren #3843 lr=1e-4 (HUGE)

**Frieren's lr=1e-4 arm (W&B run `bw38ym4h`) on n_fourier=0 baseline WITHOUT spec_norm:**
- val 65.41 / test 56.06 → **−3.55 val / −4.76 test vs NEW spec_norm baseline**
- lr=2e-5 confirms opposite direction (val 78.93 — worse)
- Control arm lr=5e-5 (pqbyquwr) val 69.69 — good reproduction of substrate

**Waiting for terminal SENPAI-RESULT.** Merge immediately when posted.

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3954** | **nezuko** | **R8 H32: spec_norm + lr=1e-4 combined** | **Just assigned** |
| **#3955** | **alphonse** | **R8 H33: spec_norm n_power_iter sweep {1, 3, 5}** | **Just assigned** |
| **#3957** | **tanjiro** | **R8 H34: cosine T_max sweep {10, 14 ctrl, 20} under spec_norm** | **Just assigned** |
| **#3958** | **thorfinn** | **R8 H35: Lion wd sweep at lr=1e-4 {0.5e-3, 1e-3 ctrl, 2e-3}** | **Just assigned** |
| **#3843** | **frieren** | **Lion lr=1e-4 — AWAITING TERMINAL** | **WIP — URGENT nudge posted** |
| #3808 | fern | surf_weight sweep + confirmation arm w20 n_fourier=0 | WIP — asked for n_fourier=0 confirmation |
| #3712 | askeladd | Lion β1 {0.8, 0.9, 0.95} — β1=0.95 arm still missing | WIP — nudged |
| #3913 | edward | R8 H31: Reynolds-extremity WeightedRandomSampler α ∈ {0, 0.5, 1.0} | WIP |

## Closed this session

- **#3817 alphonse** — FiLM ablation: FiLM contributes −4.35 val / −4.56 test under n_fourier=0. Paper-critical confirmed. Arm A = baseline reproduction (val 70.05, lucky seed). No merge — no new mechanism.
- **#3842 tanjiro** — Sobolev finer sweep: catastrophic failure (val 212). Loss scaling broken at w=0.05+ on new spec_norm substrate. Informative.
- **#3845 thorfinn** — Train-time z-aug: p=0.5 val 93 (catastrophic). Same root cause as TTA failure — training dist AoA asymmetry. Informative.
- **#3786 edward** — Huber β: β=0.05 locally optimal in [0.05, 0.20]. Informative.

## Key findings (cumulative)

1. **FiLM on log(Re)** contributes −4.35 val / −4.56 test under n_fourier=0 (paper-critical ablation confirmed).
2. **EMA(0.997)** contributes +4.4 val on top of Lion.
3. **Fourier PE inert** under FiLM+Lion+EMA. n_fourier=0 wins.
4. **EMA decay robust** in [0.995, 0.997].
5. **Sobolev surface regularization**: pointed in right direction in R5 (small test gain), but catastrophically destabilizes at tighter weights on new substrate.
6. **LLRD doesn't transfer** from fine-tuning to scratch.
7. **LR warmup adds nothing** to Lion.
8. **Block-FiLM regresses.** Output-only FiLM is correct topology.
9. **Lookahead dead end.**
10. **TTA z-reflection fails** — AoA asymmetry.
11. **Huber β=0.05 locally optimal** in [0.05, 0.20].
12. **Seed noise floor ≈ 2.77 val** (two identical runs).
13. **Output-only spectral norm (−1.39 val / −0.81 test)**: Lipschitz constraint on head MLP reduces peak-pressure over-fitting. Arm C (output+FiLM spec_norm) hurts — FiLM's adaLN-Zero init incompatible with Lipschitz bounding.
14. **Lion lr=1e-4 wins massively (−3.55 val vs spec_norm baseline)**: sign-based updates tolerate 2× LR. lr=2e-5 significantly worse. Pending confirmation merge.
15. **Train-time z-aug fails** for same reason as TTA — AoA asymmetry is training-data deep, not inference-only.

## R8 hypothesis map (current round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| LR (pending merge) | lr=1e-4 → merge | #3843 frieren | val ~65 / test ~56 |
| Compound: spec_norm + lr=1e-4 | Stack two winners | #3954 nezuko | val ~62-65 |
| Spec_norm constraint | n_power_iter {1, 3, 5} | #3955 alphonse | marginal refinement |
| LR schedule | T_max {10, 14, 20} under spec_norm | #3957 tanjiro | marginal/informative |
| LR × wd | wd at lr=1e-4 {0.5, 1.0, 2.0}×1e-3 | #3958 thorfinn | calibration check |
| Re-stratified sampler | α ∈ {0, 0.5, 1.0} | #3913 edward | targets re_rand OOD |
| surf_weight confirmation | w=20 on n_fourier=0 | #3808 fern | +3 val candidate |
| Lion β1 completion | β1=0.95 missing arm | #3712 askeladd | paper ablation |
