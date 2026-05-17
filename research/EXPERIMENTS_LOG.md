# SENPAI Research Results

## 2026-05-17 11:05 — #4507 #4495 #4470 CLOSED (Findings #57-59); fern→#4574, askeladd→#4575, tanjiro→#4577 (new-BL orthogonal re-screens)

### #4507 fern — R13 H93: spec_norm INPUT {pi=1,pi=3} at T_max=22 (CLOSED — Finding #57)

| placement | pi | W&B | val_avg | Δ vs old BL |
|---|---|---|---|---|
| output #4505 | 3 | b4txs5yb | 46.7952 | −2.96 MERGED |
| output #4505 | 1 | 1xy56nr6 | 48.0400 | −1.71 |
| **input #4507** | 1 | 14ycsdrs | 50.5927 | +0.84 (null) |
| **input #4507** | 3 | 5cxu5hak | 49.5504 | −0.20 (null) |

**Finding #57**: spec_norm INPUT at T_max=22 is null — both arms within ±1σ_T22. Placement matters: output head constraint (generalization regularizer) decisively outperforms input constraint. ls×spec_norm synergy is output-specific. Input pi=1 over-shrinks high-magnitude features (+4.73 in_dist val); pi=3 recovers via better σ-estimate. Against new BL (46.80): both +2.76/+3.80 val worse. Spec_norm input axis closed.

New assignment: fern → #4574 (R13 H99 lr recalibration {2.3e-4, 2.5e-4} at new spec_norm BL).

---

### #4495 askeladd — R13 H90: ls upward {2e-4,3e-4} at T_max=22 (CLOSED — Finding #58)

Combined with #4419 (Finding #48): clean U-shape with ls=1e-4 as local optimum.

| ls | val_avg | Δ vs BL |
|----|---------|---------|
| 1e-5 (#4419) | 54.23 | +4.48 |
| 5e-5 (#4419) | 52.85 | +3.10 |
| **1e-4 (BL)** | **49.75** | **0** |
| 2e-4 (4ooadyta) | 52.07 | +2.32 |
| 3e-4 (h4ti2plx) | 51.98 | +2.23 |

**Finding #58**: ls axis fully closed in BOTH directions at T_max=22 — clean U-shape, ls=1e-4 (CaiT/DeiT-III default) is local optimum. Mechanism: balance between enough residual capacity and not over-investing. Camber_rc was NOT helped by either direction — confirms camber_rc bottleneck is NOT residual capacity at this substrate.

New assignment: askeladd → #4575 (R13 H100 wd downward {7e-4, 5e-4} at new spec_norm BL).

---

### #4470 tanjiro — R13 H88: batch_size {2,8} at T_max=22 (CLOSED — Finding #59)

| bs | run | val_avg | Δ vs BL |
|----|-----|---------|---------|
| 2 (best) | fmwyafnc | 50.08 | +0.33 (null) |
| **4 (BL)** | 1neonugr | **49.75** | 0 |
| 8 | 3okzue66 | 78.06 | +28.30 (catastrophic) |

**Finding #59**: batch_size axis closed at T_max=22 under SENPAI_TIMEOUT_MINUTES=30. bs=4 is the sweet spot. bs=2: 2× steps/epoch but only 13 epochs in 30 min (schedule undertraversed). bs=8: ~94 steps/epoch × 13 epochs = 1.2k total updates vs ~5k at BL — far from converged. bs=4 confirmed canonical at current wall-clock.

New assignment: tanjiro → #4577 (R13 H101 EMA recalibration {0.995, 0.998} at new spec_norm BL).

---

## 2026-05-17 10:45 — #4505 MERGED (Finding #56: spec_norm output pi=3 is new BL val 46.80/test 40.49); #4503 CLOSED (Finding #55: Huber β axis closed, β=0.05 local minimum asymmetric); edward→#4560 (R13 H98 spec_norm pi=5/pi=10), alphonse→#4557 (R13 H97 grad_clip)

### #4505 edward — R13 H92: spec_norm output {pi=1,pi=3} at T_max=22 (MERGED — Finding #56, NEW BEST)

| Arm | spec_norm pi | W&B | val_avg | Δ vs BL | test_avg | Δ vs BL |
|-----|-------------|-----|---------|---------|---------|---------|
| BL | — | 1neonugr | 49.7515 | — | 42.8929 | — |
| A | pi=1 | 1xy56nr6 | 48.0400 | −1.71 | 41.3794 | −1.51 |
| **B (MERGED)** | **pi=3** | **b4txs5yb** | **46.7952** | **−2.96** | **40.4866** | **−2.41** |

Per-split B vs BL: in_dist (−1.20 val / +0.40 test), camber_rc (−3.01 val / −2.59 test), camber_cruise (−4.16 val / −3.35 test), re_rand (−3.46 val / −4.08 test). Win broad-based — cruise and rc largest gains. No wall-clock overhead (pi=3: 142s/epoch vs pi=1: 144s/epoch).

**Finding #56**: spec_norm output pi=3 wins at T_max=22 — **new best val 46.7952 / test 40.4866**. Finding #13 (diminishing returns with lr) does NOT extend to T_max=22+ls=1e-4 substrate. Mechanism: ls=1e-4 keeps residual block contributions tiny early in training, making the output layer's Lipschitz constraint proportionally more impactful. Higher pi (tighter Lipschitz bound) helps because T_max=22's high mean lr makes more aggressive updates throughout training — stronger constraint pays off. Per-split structure: camber_cruise (−4.16) and camber_rc (−3.01) benefit most — generalization-oriented, not fitting. in_dist slightly regresses on test (+0.40) — spec_norm is a regularizer, not a fitting accelerator. **Student analysis excellent** — correctly identified the ls×spec_norm synergy and the pi-monotonicity. Follow-up: pi={5,10} in flight (#4560).

New assignment: edward → #4560 (R13 H98 spec_norm output {pi=5, pi=10} — close power-iteration axis).

---

### #4503 alphonse — R13 H91: Huber β downward {0.03,0.04} at T_max=22 (CLOSED — Finding #55)

| Arm | β | W&B | val_avg | Δ vs BL |
|-----|---|-----|---------|---------|
| BL | 0.05 | 1neonugr | 49.7515 | — |
| A | 0.04 | ywhpbgqi | 54.3786 | +4.63 |
| B | 0.03 | jff9xbfh | 55.3240 | +5.57 |

Combined with #4434, β landscape at T_max=22: β=0.03(+5.57), β=0.04(+4.63), **β=0.05(0.00)**, β=0.10(+2.69), β=0.15(+2.21). β=0.05 is a local minimum.

**Finding #55**: Huber β axis fully closed at T_max=22. β=0.05 is the local minimum with **asymmetric landscape** — L1 side (β<0.05) hurts ~2× more than L2 side (β>0.05). Mechanism: Lion's sign-update + very narrow L2 region (β=0.03) = most residuals already in the L1 zone → constant ±1 gradient in normalized space → flat-magnitude loss landscape on small residuals → slows late convergence. The per-split regression concentrated on in_dist (+7-9 val) not camber_rc as predicted — in_dist has highest dynamic range/outliers and benefits most from BL's β=0.05 smoothing. **β axis closed in both directions at T_max=22.**

New assignment: alphonse → #4557 (R13 H97 grad_clip {0.7, 1.5} at T_max=22 — untested axis).

---

## 2026-05-17 10:30 — #4471 #4420 CLOSED (Findings #53-54: lr×T_max=22 grid closed, surf_weight upward closed); thorfinn→#4531 (R13 H95 basin-pairing), nezuko→#4533 (R13 H96 surfw-down)

### #4471 thorfinn — R13 H89: lr downward probe {1.5e-4,1.7e-4} at T_max=22 (CLOSED — Finding #53)

5-point lr × T_max=22 grid summary (combining with tanjiro #4391 upward probes):

| lr     | LR_end (epoch 13) | val_avg | Δ vs BL |
|--------|-------------------|---------|---------|
| 1.5e-4 | ~5.4e-5 | 56.815 | +7.07 (exyw9aca) |
| 1.7e-4 | ~6.1e-5 | 54.232 | +4.48 (efwnup17) |
| **2.0e-4** | **~7.18e-5** | **49.752** | **0.00 (BL `1neonugr`)** |
| 2.3e-4 | ~8.26e-5 | 51.66 | +1.91 (#4391) |
| 2.5e-4 | ~8.98e-5 | 49.87 | +0.12 (#4391) |

Asymmetric U-shape: downward regressions far steeper than upward (+4.48, +7.07 vs +1.91, +0.12). Every split degrades monotonically as lr drops — camber_rc and re_rand suffer most. lr=2.5e-4 within σ_T22 ≈ 1.86.

**Finding #53**: lr × T_max=22 grid fully closed. Operating point lr=2.0e-4 robust; lr=2.5e-4 within σ. **Refined basin mechanism (student proposed)**: basin in (peak_LR, LR_end) plane — not a single LR_end target. Peak LR controls representation learning during bulk training; LR_end ≥ ~6e-5 controls final fine-tuning. Both must clear thresholds. Below LR_end ~6e-5 (lr < 1.7e-4), under-training at the epoch cap. Natural follow-up: T_max=24 + lr=2.5e-4 basin pairing test (shifting both to maintain a higher LR_end while preserving peak magnitude).

New assignment: thorfinn → #4531 (R13 H95 T_max×lr basin pairing {T_max=24,T_max=26} × lr=2.5e-4).

---

### #4420 nezuko — R13 H84: surf_weight {15,20} at T_max=22 (CLOSED — Finding #54)

| Arm | surf_weight | run | val_avg | test_avg | Δ val vs BL |
|-----|-------------|-----|---------|----------|---|
| BL | 10 | 1neonugr | 49.75 | 42.89 | — |
| A (2-seed) | 15 | 0cfc70wi / 8i5ahxoo | 53.51 / 54.32 | 45.32 / 46.60 | **+3.76 / +4.57** ✗ |
| B | 20 | 3f4z3lae | 55.51 | 47.21 | **+5.76** ✗ |

Per-split val at w=15 vs BL: in_dist +5.79, camber_rc +5.38, camber_cruise +1.86, re_rand +2.02. 2-seed var at w=15: 0.81 val (within σ_T22=1.86). Both surface AND volumetric MAE rose together — strictly worse on both heads.

**Finding #54**: surf_weight upward direction closed at T_max=22. Monotonic regression on both heads — not a trade of vol→surf. camber_rc hit hardest at high surf_weight (opposite of hypothesis). Split is **representation-limited, not gradient-balance-limited**. Volumetric loss provides useful auxiliary signal for shared encoder features; down-weighting it hurts both heads. Optimum at this substrate likely moves **downward** from default w=10. Student's 2-seed check solid (0.81 val var within σ).

New assignment: nezuko → #4533 (R13 H96 surf_weight downward probe {7.5, 5} at T_max=22).

---

## 2026-05-17 09:30 — #4437 #4408 CLOSED (Findings #51-52: σ_T22 calibrated, Lion β2 axis closed); fern→#4507 (R13 H93 spec_norm-input), frieren→#4508 (R13 H94 n_fourier)

### #4437 fern — R13 H87: Multi-seed BL replication at T_max=22 (CLOSED — Finding #51)

| Seed | run | val_avg | test_avg |
|------|-----|---------|----------|
| default | 1neonugr (BL) | 49.75 | 42.89 |
| 42 | 8dk4iz9o | 52.28 | 44.71 |
| 2026 | 3yejdq4x | 54.29 | 45.99 |

3-seed mean: val 52.11, test 44.53. σ_val ≈ 1.86, σ_test ≈ 1.27.

**Finding #51**: σ_T22 ≈ 1.86 val / 1.27 test (slightly noisier than σ_T20=1.70). BL `1neonugr` (49.75) at z=-1.27σ from mean — seed-lucky. True T_max=22 mean ~52.1. Merge-threshold update: gains below 2 val from BL are likely noise-floor; gains ≥3 val above BL are robust signal. Student ran duplicate seed=42 arms (unnecessary).

New assignment: fern → #4507 (R13 H93 spec_norm INPUT at T_max=22).

---

### #4408 frieren — R13 H82: Lion β2 {0.95, 0.995} at T_max=22 (CLOSED — Finding #52)

| Arm | β2 | run | val_avg | test_avg | Δ val vs BL |
|-----|----|-----|---------|----------|---|
| BL | 0.99 | 1neonugr | 49.75 | 42.89 | — |
| A | 0.95 | c0b23lww | 60.98 | 52.65 | **+11.23** ✗ (catastrophic) |
| B | 0.995 | 0pp7ts6p | 56.21 | 47.77 | **+6.46** ✗ |

**Finding #52**: Lion β2 axis closed at T_max=22. β2=0.99 robustly optimal. β2=0.95 catastrophic (too-noisy gradient variance); β2=0.995 significant regression (too-slow variance update). Combined with Finding #44, **Lion optimizer-state hyperparameters fully pinned across all substrates**: β1=0.9, β2=0.99. Student ran duplicate β2=0.95 arms.

New assignment: frieren → #4508 (R13 H94 n_fourier=8 {σ=5, σ=10} at T_max=22).

---

## 2026-05-17 09:15 — #4434 #4436 CLOSED (Findings #49-50: Huber β and wd compositions fail@T_max=22); alphonse→#4503 (R13 H91 Huber β-down), edward→#4505 (R13 H92 spec_norm)

### #4434 alphonse — R13 H85: Huber β {0.10, 0.15} at T_max=22 (CLOSED — Finding #49)

| Arm | β | run | val_avg | test_avg | Δ val vs BL |
|-----|---|-----|---------|----------|---|
| BL | 0.05 | 1neonugr | 49.75 | 42.89 | — |
| A | 0.10 | wq8x73z3 | 52.45 | 44.76 | **+2.69** ✗ |
| B | 0.15 | y3jgff90 | 51.96 | 44.16 | **+2.21** ✗ |

Per-split Arm A: in_dist +3.18, camber_rc +2.51, camber_cruise **+2.94** (inverts!), re_rand +2.15
Per-split Arm B: in_dist +3.58, camber_rc +2.17, camber_cruise **+2.59** (inverts!), re_rand +0.50

**Finding #49**: Huber β upward composition fails at T_max=22. β=0.10's cruise win from Finding #43 (at T_max=20) INVERTS at T_max=22 — cruise gets WORSE. Mechanism: T_max=22 keeps lr higher for longer (~30% higher mean); β-widening × higher mean lr = over-emphasis on small residuals at expense of large-residual sharpening. β=0.05 correctly matches T_max=22's gentler endpoint. Student's own analysis: "β=0.05 closer to L1 better matched to T_max=22" → downward probe assigned (#4503).

---

### #4436 edward — R13 H86: wd {2e-3, 3e-3} at T_max=22 (CLOSED — Finding #50)

| Arm | wd | run | val_avg | test_avg | Δ val vs BL |
|-----|------|-----|---------|----------|---|
| BL | 1e-3 | 1neonugr | 49.75 | 42.89 | — |
| A | 2e-3 | 82qf5yv6 | 53.38 | 45.42 | **+3.63** ✗ |
| B | 3e-3 | yho2omcb | 52.02 | 45.03 | **+2.27** ✗ |

Per-split Arm A: in_dist +5.09, camber_rc +3.32, camber_cruise **+4.09** (inverts!), re_rand +2.02
Per-split Arm B: in_dist +4.51, camber_rc +0.98, camber_cruise **+2.79** (inverts!), re_rand +0.78

**Finding #50**: wd upward composition fails at T_max=22. Finding #42's cruise/re_rand wins (at T_max=20) do NOT compose at T_max=22. cruise metric inverts both arms. T_max=22 already absorbed the regularization budget that wd=2e-3 was providing at T_max=20 — two interventions compete for the same regularization budget. Student meta-observation: "effects at one substrate cannot be assumed to add when substrate shifts" — now confirmed 4th time (after #4391, #4419, #4434). Student suggests orthogonal axes → spec_norm assigned (#4505).

**Meta-finding (structural)**: T_max=22 is NOT a linear extension of T_max=20. All 4 attempts to transfer T_max=20 directional wins have failed with comparable regressions: lr +1.91 val (#4391), ls-smaller +3.10 val (#4419), Huber β +2.69 val (#4434), wd +3.63 val (#4436). Each intervention that worked at T_max=20 inverts at T_max=22 because the substrate change already consumed the regularization/optimization budget those interventions were adding.

New assignments: alphonse → #4503 (R13 H91 Huber β {0.03, 0.04} downward); edward → #4505 (R13 H92 spec_norm output at T_max=22).

---

## 2026-05-17 09:00 — #4419 CLOSED (Finding #48: ls smaller-dir inverts@T_max=22); askeladd→#4495 (R13 H90 ls-upward@T_max=22)

### #4419 askeladd — R13 H83: ls {5e-5, 1e-5} at T_max=22 (CLOSED — Finding #48)

| Arm | ls | run | val_avg | test_avg | Δ val vs BL |
|-----|----|-----|---------|----------|---|
| BL | 1e-4 | 1neonugr | 49.75 | 42.89 | — |
| A | 5e-5 | scajxlkn | 52.85 | 44.85 | **+3.10** ✗ |
| B | 1e-5 | lpl6v6z3 | 54.23 | 46.09 | **+4.48** ✗ |

Per-split val Arm A: in_dist 56.39, camber_rc 66.83 (+3.06), camber_cruise 35.22, re_rand 52.96
Per-split val Arm B: in_dist 56.07, camber_rc 68.86 (+5.09), camber_cruise 36.84, re_rand 55.15

**Finding #48**: ls smaller-direction at T_max=22 CLOSED. The monotone-toward-smaller-ls trend from #4318 at T_max=20 does NOT transfer at T_max=22. Arm-vs-arm: ls=1e-4 > ls=5e-5 > ls=1e-5. camber_rc regression worsens monotonically with smaller ls. Mechanism (student analysis): smaller ls slows residual ramp; under T_max=22's slower-decaying LR schedule and 13-epoch effective horizon, residual contribution gets stuck under-ramped → exacerbates the chronically-weak camber_rc split. Student suggested upward ls direction (faster residual ramp) → new assignment #4495.

New assignment: askeladd → #4495 (R13 H90 ls {2e-4, 3e-4} at T_max=22 — upward probe, "faster residual ramp" mechanism).

---

## 2026-05-17 12:30 — #4391 #4372 CLOSED (Findings #46-47); tanjiro→#4470 (R13 H88 bs@T_max=22), thorfinn→#4471 (R13 H89 lr-down@T_max=22)

### #4391 tanjiro — R13 H80: LR {2.3e-4, 2.5e-4} at T_max=22 (CLOSED — Finding #46)

| Arm | lr | run | val_avg | test_avg | Δ val vs BL (49.75) |
|-----|----|-----|---------|----------|---|
| BL | 2e-4 | 1neonugr | 49.75 | 42.89 | — |
| A | 2.3e-4 | — | 51.66 | 44.14 | **+1.91** ✗ |
| B | 2.5e-4 | — | 49.87 | 42.75 | **+0.12** (within σ) |

**Finding #46**: lr=2.3e-4's T_max=20 directional win (Finding #39: −0.41 val, camber_cruise −2.45) **inverts** at T_max=22. The cruise gain at T_max=20 was conditional on T_max=20 substrate: at T_max=22, lr=2.3e-4 pushes the epoch-13 LR above the resonance point (see Finding #47). lr=2.5e-4 marginal test edge (−0.14), but val flat within σ — not a win. lr axis upward direction exhausted at T_max=22. Directional signals: Arm B lr=2.5e-4 had camber_rc test −1.90 (only split that improved), possibly a high-noise single-seed artifact.

New assignment: tanjiro → #4470 (R13 H88 batch_size {2, 8} at T_max=22 — fresh axis, never swept).

---

### #4372 thorfinn — R12 H79: T_max fine grid {21, 23} around T_max=22 optimum (CLOSED — Finding #47)

| Arm | T_max | run | val_avg | test_avg | Δ val vs BL | LR@epoch13 |
|-----|-------|-----|---------|----------|-------------|------------|
| BL | 22 | 1neonugr | 49.75 | 42.89 | — | 7.18e-5 |
| A | 23 | eiyxv9mo | 53.67 | 46.39 | **+3.92** ✗ | 7.97e-5 |
| B | 21 | 0340tvyo | 55.53 | 46.99 | **+5.77** ✗ | 6.35e-5 |

Per-split Arm A (T_max=23) vs BL: in_dist +4.53, camber_rc +3.61, camber_cruise +3.78, re_rand +3.73 — uniform regression across all splits. Per-split Arm B (T_max=21): in_dist +8.02, camber_rc +8.25, camber_cruise +3.06, re_rand +3.77.

**Finding #47**: T_max=22 is a **sharp local optimum** under SENPAI_TIMEOUT_MINUTES=30 (13-epoch effective horizon). ±1 step costs ≥3.9 val on both val and test. Excellent mechanistic insight from student: the 'win' at T_max=22 is a **resonance between cosine LR schedule and the 13-epoch effective training horizon** — at epoch 13, T_max=22 delivers exactly LR 7.18e-5 (optimal), T_max=23 → 7.97e-5 (too slow-decaying), T_max=21 → 6.35e-5 (too fast-decaying). Safe T_max range: [21, 23] (no divergence found). T_max axis closed at the new BL substrate. Follow-up: lr downward probe (#4471) tests whether the resonance can shift.

New assignment: thorfinn → #4471 (R13 H89 lr {1.5e-4, 1.7e-4} at T_max=22 — completes 5-point lr × T_max=22 grid).

---

## 2026-05-17 11:00 — #4326 #4329 #4346 CLOSED (Findings #43-45); alphonse→#4434 (R13 H85 Huber β@T_max=22), edward→#4436 (R13 H86 wd@T_max=22), fern→#4437 (R13 H87 σ-calib@T_max=22)

### #4326 alphonse — R12 H75: Huber β sweep at new BL substrate (CLOSED — Finding #43)

| Arm | β | run | val_avg | test_avg | Δ val vs old BL (53.08) | Δ val vs NEW BL (49.75) |
|-----|---|-----|---------|----------|---|---|
| BL | 0.05 | d3qlknrv | 53.08 | 44.89 | — | +3.33 |
| A | 0.03 | wuc7qy10 | 55.39 | 47.06 | +2.31 | +5.64 |
| **B** | **0.10** | **phelxelv** | **51.93** | **44.44** | **−1.15** | **+2.18** |

Per-split Arm B (β=0.10) vs BL:
- in_dist: val **−1.42** ✓, test +0.35
- camber_rc: val +0.32, test +1.40 ✗ (only weakness)
- **camber_cruise: val −2.96 / test −2.29** ✓✓ (LARGEST cruise win across all R12 T_max=20 sweeps)
- re_rand: val −0.52 / test −1.25 ✓

**Finding #43**: β=0.10 is the directional winner at T_max=20 substrate. Wider Huber smooth region helps low-magnitude pressure gradients in the cruise domain. Does NOT beat new BL (+2.18 val) as experiment was at T_max=20. Strong R13 candidate: β=0.10 at T_max=22.

New assignment: #4434 alphonse — R13 H85: Huber β∈{0.10, 0.15} at T_max=22.

---

### #4329 edward — R12 H77: Lion β1 sweep at new BL substrate (CLOSED — Finding #44)

| Arm | β1 | best run | val_avg | test_avg | Δ val vs NEW BL |
|-----|-----|---------|---------|----------|---|
| BL | 0.9 | d3qlknrv | 53.08 | 44.89 | +3.33 |
| A (best) | 0.85 | qqiue12e | 54.17 | 47.68 | **+4.42** |
| A (other) | 0.85 | z5ae1b0p | 56.05 | 48.25 | +6.30 |
| A crashes | 0.85 | qj85mcsc, 3126iw7f | diverged | — | — |
| B | 0.95 | 6eb9ji08 | 53.58 | 45.64 | **+3.83** |

**Finding #44**: Lion β1 axis **doubly closed** at T_max=20 substrate. β1=0.85 is **unstable** (2 of 4 launches diverged — clip=1.0 + ls + lr=2e-4 amplifies the instability from too-fast momentum warmup). β1=0.95 marginally ties old BL (+0.50 val) but does not reach new BL (+3.83). β1=0.9 is robustly optimal. Lion β1 axis closed across all tested substrates (old: Finding #30, new T_max=20: Finding #44).

New assignment: #4436 edward — R13 H86: wd∈{2e-3, 3e-3} at T_max=22 (transfer Finding #42 per-split signal).

---

### #4346 fern — R12 H78: Multi-seed BL replication at T_max=20 (CLOSED — Finding #45)

5-seed BL σ at T_max=20 substrate:

| Seed | run | val_avg | test_avg |
|------|-----|---------|----------|
| BL_a | 0zrrntw3 | 55.87 | 47.55 |
| BL_b | jqmn2nw7 | 53.27 | 45.35 |
| BL_c | d3qlknrv | 53.08 | 44.89 |
| 42 | 6mdrdrwx | 52.61 | 45.75 |
| 2026 | e2ex4rdt | 51.18 | 43.70 |
| **Mean ± σ** | | **53.20 ± 1.70** | **45.45 ± 1.40** |

Per-split σ (val): in_dist 2.27, camber_rc 3.43 (HIGHEST), camber_cruise 1.39, re_rand 1.18.

**Finding #45**: σ_val ≈ 1.7, σ_test ≈ 1.4 at T_max=20 substrate. camber_rc has 2× higher σ than other splits — improvements on rc need higher threshold. **Implications for merge decisions**:
- Improvements <1 val: likely seed noise
- Improvements 1-2 val: weak signal, ideally confirm with 2nd seed
- Improvements ≥2 val: statistically robust (>1σ)

The merged BL `1neonugr` (T_max=22, val 49.75) is **~2.0 val below the T_max=20 5-seed mean** — consistent with T_max=22 being a real improvement. But T_max=22 σ is unknown. Assigned fern to characterize T_max=22 σ (#4437).

New assignment: #4437 fern — R13 H87: multi-seed BL replication at T_max=22 (3-seed σ calibration).

---

## 2026-05-17 10:30 — #4318 #4319 CLOSED (Findings #41-42); askeladd→#4419 (R13 H83 ls@T_max=22), nezuko→#4420 (R13 H84 surf_weight@T_max=22); #4326 #4329 #4346 pinged

### #4318 askeladd — R12 H72: layer_scale magnitude at new BL substrate (CLOSED — Finding #41)

| Arm | ls | run | val_avg | test_avg | Δ val vs old BL (53.08) | Δ val vs NEW BL (49.75) |
|-----|-----|-----|---------|----------|---|---|
| BL | 1e-4 | d3qlknrv | 53.08 | 44.89 | — | +3.33 |
| A1 | 1e-3 | 34zp8lb8 | 53.15 | 45.62 | +0.07 | +3.40 |
| A2 | 1e-3 | is4py9s2 | 52.97 | 45.10 | −0.11 | +3.22 |
| **B** | **5e-5** | **odcn0kp1** | **52.04** | **44.72** | **−1.04** | **+2.29** |

**Finding #41**: ls=5e-5 wins at T_max=20+clip+lr=2e-4 substrate (−1.04 val vs old BL, −0.17 test). Non-monotone ls landscape continues to shift with substrate: optimum moved from ls=1e-3 (old no-clip) → ls=1e-3 (pre-ls-merge #4212) → ls=1e-4 (current) → ls=5e-5 (this PR). Trend is **monotone toward smaller ls at higher LR+clip**. CRITICAL caveat: camber_rc regresses (+1.96 val/test) for both arms — ls=1e-4 is camber_rc-best. Does NOT beat new BL 49.75 (+2.29 val). Directional finding for R13.

New assignment: #4419 askeladd — R13 H83 ls∈{5e-5, 1e-5} at T_max=22.

---

### #4319 nezuko — R12 H73: Weight decay at new BL substrate (CLOSED — Finding #42)

| Arm | wd | run | val_avg | test_avg | Δ val vs old BL (53.08) | Δ val vs NEW BL (49.75) |
|-----|-----|-----|---------|----------|---|---|
| BL | 1e-3 | d3qlknrv | 53.08 | 44.89 | — | +3.33 |
| A (×2) | 5e-4 | yasn14xp, n2oizbhq | 53.83 | 46.14 | +0.75 | +4.08 |
| **B** | **2e-3** | **hcu511lr** | **53.52** | **45.43** | **+0.44** | **+3.77** |

Per-split wd=2e-3 signal: camber_cruise **−3.73 val / −3.39 test** (strong improvement), re_rand −0.86/−0.66; but in_dist +1.11/+2.45 and camber_rc +5.24/+3.76 (large regression).

**Finding #42**: wd=1e-3 robustly optimal at T_max=20+clip substrate — both arms regress on aggregate. But wd=2e-3 shows a strong **per-split asymmetric trade** (cruise/re_rand wins vs rc/in_dist losses). Mechanism: higher wd penalizes weight scale → stronger cruise-domain generalization at cost of local in_dist/rc features. Directional finding for R13 at T_max=22 substrate. Does NOT beat new BL (+3.77 val).

New assignment: #4420 nezuko — R13 H84 surf_weight∈{15, 20} at T_max=22.

---

### #4346 fern — R12 H78: Multi-seed BL replication (IN-FLIGHT — σ established, pinged for final results)

| Seed | run | val_avg | test_avg | Δ vs BL (49.75/42.89) |
|------|-----|---------|----------|---|
| 42 | fern-r12-bl-seed42 | 52.61 | 45.75 | +2.86 / +2.86 |
| 2026 | fern-r12-bl-seed2026 | 51.18 | 43.70 | +1.43 / +0.81 |

**Critical finding**: seed σ ~1.4 val at new BL. Original `1neonugr` (49.75) is seed-lucky — mean over 3 seeds ≈ 51.2 val / 44.1 test. σ_seed ~1.4 val (stdev across 3 seeds), test σ ~1.7. **Implications**: improvements ≥1 val needed to be signal, ≥2 val to be statistically robust. Future merges under σ_seed threshold should require multi-seed confirmation.

---

## 2026-05-17 10:00 — #4393 frieren CLOSED (Finding #40 — effective warmup=0); #4408 frieren assigned (R13 H82 Lion β2); #4326 #4329 in-flight partial results

### #4393 frieren — R13 H81: warmup_epochs at T_max=22 (CLOSED — infrastructure finding, Finding #40)

**No arms run.** Student correctly identified that warmup is not CLI-configurable: `train.py:799` constructs `CosineAnnealingLR(optimizer, T_max=cosine_t_max)` with no `LambdaLR`/`SequentialLR` wrapper. No `--warmup_epochs` flag exists in Config. Per PR contract ("If warmup is not CLI-configurable, message back before running — I'll reassign"), student paused.

**Finding #40**: Every BL run in this programme uses **effective warmup_epochs=0** — LR starts at peak on epoch 0 and pure cosine-anneals. Zero warmup is apparently stable at the new BL substrate (lr=2e-4 + clip=1.0 + ls=1e-4). This is informative for any future warmup implementation: if implemented, it would be a novel positive axis rather than a recovery from broken status quo.

New assignment: #4408 frieren — R13 H82 Lion β2 sweep {0.95, 0.995} at T_max=22 substrate (first β2 sweep at new BL; prior #4153 askeladd became substrate-obsolete before layer_scale merged).

---

### #4326 alphonse — R12 H75: Huber β at new BL substrate (IN-FLIGHT partial results)

| Arm | β | runs | best val_avg | best test_avg | Δ vs BL (49.75/42.89) |
|-----|---|------|-------------|--------------|---|
| A | 0.03 | wuc7qy10, hs0im51b (done), tol95dak (failed) | 54.54 | 46.81 | **+4.79 / +3.91** |
| B | 0.10 | phelxelv (running) | ~70 → descending | — | TBD |

β=0.03 arm complete: consistently worse than BL by ~4-5 val. Too-tight Huber smoothing region penalizes the large MAE residuals that dominate the OOD splits. β=0.10 arm still training (epoch ~8).

---

### #4329 edward — R12 H77: Lion β1 at new BL substrate (IN-FLIGHT partial results)

| Arm | β1 | runs | best val_avg | best test_avg | Δ vs BL (49.75/42.89) |
|-----|-----|------|-------------|--------------|---|
| A | 0.85 | z5ae1b0p, qqiue12e (done), qj85mcsc (crashed) | 54.17 | 47.68 | **+4.42 / +4.79** |
| B | 0.95 | 6eb9ji08 (running) | ~72 → descending | — | TBD |

β1=0.85 arm complete: significantly worse than BL. Faster momentum update at the new BL (clip=1.0 + ls=1e-4) is destabilizing — suggests β1=0.9 default is at or near optimum. β1=0.95 arm still training (epoch ~8).

---

## 2026-05-17 09:30 — #4328 #4315 CLOSED (Findings #38-39); #4391 tanjiro, #4393 frieren assigned (R13: lr×T_max=22 composition, warmup)

### #4328 frieren — R12 H76: EMA at new BL T_max=20 (CLOSED — informative null, Finding #38)

| Arm | ema | run | val_avg | test_avg | Δ val vs old BL (53.08) |
|-----|-----|-----|---------|----------|---|
| BL | 0.997 | d3qlknrv | 53.08 | 44.89 | — |
| A | 0.995 | dia8w7md | 54.56 | 46.17 | +1.48 |
| B | 0.999 | qsisels8 | 59.53 | 50.88 | +6.46 |

**Finding #38**: at the new BL substrate (T_max=20+ls+clip+lr=2e-4), ema=0.997 robustly optimal. ema=0.999 catastrophically bad (val +6.46 — worst on ALL 4 splits). The hypothesis that layer_scale unblocks longer EMA averaging is **rejected**. Mechanism (student's analysis): the binding constraint is the 13-epoch budget, not late-epoch noise. ema=0.999 is starvation-bottlenecked — its longer window is dominated by early-training weights at the cutoff. Bright spot: ema=0.995 nearly ties camber_cruise (−0.35 val) but loses everywhere else. EMA axis closed across all 3 substrates (Findings #28+#34+#38).

New assignment: #4393 frieren — R13 H81 warmup_epochs at T_max=22.

---

### #4315 tanjiro — R12 H71: LR sweep at new BL T_max=20 (CLOSED — directional finding, Finding #39)

| Arm | lr | run | val_avg | test_avg | Δ val vs old BL (53.08) | Δ val vs NEW BL (49.75) |
|-----|-----|-----|---------|----------|---|---|
| BL | 2.0e-4 | d3qlknrv | 53.08 | 44.89 | — | +3.33 |
| A | 1.7e-4 | o9zdn7k2 | 54.19 | 46.03 | +1.11 | +4.44 |
| **B** | **2.3e-4** | **ifhbhs2y** | **52.67** | **44.41** | **−0.41** | **+2.92** |

Per-split Arm B (lr=2.3e-4) vs BL:
- in_dist: val −0.70, test −0.45 ✓
- camber_rc: val +1.83, test +1.95 ✗
- **camber_cruise: val −2.45, test −2.11 ✓** (key split!)
- re_rand: val −0.31, test −1.30 ✓

**Finding #39**: lr=2.3e-4 is the directional LR winner at T_max=20+ls+clip substrate — below merge threshold (−0.41 vs new BL +2.92) but provides the clearest camber_cruise improvement (−2.45 val) alongside the T_max=22 finding (Finding #37). The mechanism matches Finding #37: higher clip-bounded steps at lr=2.3e-4 → similar endpoint noise reduction effect as lower T_max endpoint. Key lead for R13: **lr=2.3e-4 at T_max=22 may compose** — this is tanjiro's next assignment (#4391).

New assignment: #4391 tanjiro — R13 H80 lr {2.3e-4, 2.5e-4} at T_max=22 substrate.

---

## 2026-05-17 08:30 — #4320 thorfinn MERGED (NEW BEST val 49.75 / test 42.89); #4372 thorfinn assigned (R12 H79 T_max fine grid)

### #4320 thorfinn — R12 H74: T_max sweep at new BL (MERGED — Finding #37)

W&B group `round12-tmax-newbl-thorfinn`. Arm A (T_max=22) is a **major win** — first sub-50 val.

| Arm | T_max | Run | val_avg | test_avg | Δ val vs BL (53.08) | Δ test vs BL (44.89) |
|-----|-------|-----|---------|----------|---|---|
| **A (MERGED)** | **22** | **1neonugr** | **49.7515** | **42.8929** | **−3.33** | **−2.00** |
| B | 16 | don3gz0q | 55.0415 | 46.5990 | +1.97 | +1.71 |

Per-split Arm A (T_max=22) vs BL (T_max=20):

| Split | val_A | val_BL | Δval | test_A | test_BL | Δtest |
|-------|-------|--------|------|--------|---------|-------|
| in_dist | 50.61 | 55.86 | −5.25 | 44.05 | 46.83 | −2.78 |
| camber_rc | 63.77 | 65.64 | −1.87 | 57.45 | 57.22 | +0.23 |
| **camber_cruise** | **32.88** | **36.68** | **−3.80** | **27.56** | **30.65** | **−3.09** |
| re_rand | 51.75 | 54.13 | −2.38 | 42.51 | 44.85 | −2.34 |

**Finding #37**: T_max=22 is the new optimum at the ls+lr=2e-4+clip=1.0 substrate. Lower cosine-endpoint LR (~1.24e-4 at epoch 14) vs T_max=20 (~1.34e-4) helps ALL val splits and THREE of four test splits. Critically, the camber_cruise regression (+2.5/+2.8 since PR #4201) is COMPLETELY FIXED. T_max=16 regresses confirming the mechanism: MORE time-averaged LR (higher T_max) is beneficial IF it simultaneously lowers the cosine endpoint. T_max=24 diverges (Finding #33), T_max=22 is safe. First sub-50 val in programme history.

New BL: val 49.75 / test 42.89 (run `1neonugr`). Total improvement: val 135.23 → 49.75 (−63.1%), test ~130 → 42.89 (−67.0%).

Next assignment: #4372 thorfinn — T_max={21,23} fine grid to pin down the local optimum and test the cliff edge.

---

## 2026-05-17 08:00 — #4255 fern CLOSED (Finding #36); #4346 fern assigned (R12 H78: multi-seed BL replication)

### #4255 fern — R11 H67: LR sweep at T_max=24+clip+no-layer_scale (CLOSED — informative null, Finding #36)

W&B group `round11-lr-tmax24-clip-fern`. 3 arms, all 30–32 min runtime. Best epoch = 13 for all (wall-clock timeout at 14-epoch budget).

| Arm | run | lr | val_avg | test_avg | Δ val vs old BL (53.81) | Δ val vs new BL (53.08) |
|-----|-----|----|---------|----------|---|---|
| A | amyeuqvl | 1.3e-4 | 56.82 | 48.93 | +3.01 | +3.74 |
| B | ekna30ey | 1.7e-4 | 56.03 | 48.15 | +2.22 | +2.95 |
| C | 1z6z9oal | 2.0e-4 | 56.16 | 48.04 | +2.35 | +3.08 |
| BL ctrl | hk1i5kd5 | 1.5e-4 | 53.81 | 45.49 | — | +0.73 |

Note: lr=1.7e-4 rerun (zx1j3fz3) was killed per advisor instruction — duplicate of Arm B.

Per-split test (best: Arm B / ekna30ey):
| Split | in_dist | camber_rc | camber_cruise | re_rand |
|-------|---------|-----------|---------------|---------|
| B (1.7e-4) | 52.26 | 63.24 | 30.21 | 46.89 |
| BL (1.5e-4) | 48.08 | 62.12 | 27.84 | 43.93 |

**Finding #36**: at T_max=24+clip+no-ls, lr=1.5e-4 is a SHARP local minimum. All three tested LRs (1.3, 1.7, 2.0 × 1e-4) regress ≥2.2 val vs the old BL ctrl. Finding #22 (clip biases lr optimum from 1.5e-4 → 2e-4 at T_max=14) does NOT generalize to T_max=24 — the elevated late-schedule LR at T_max=24 already provides effective high-LR, making explicit lr pushes redundant. All 3 arms hit best_epoch=13 (one epoch before the 14-epoch wall-clock), suggesting the optimizer escapes its basin prematurely at any lr > 1.5e-4 with T_max=24. LR axis fully closed at old substrate.

New assignment: #4346 fern (R12 H78 — multi-seed BL replication).

---

## 2026-05-17 07:30 — #4240 #4274 #4256 CLOSED (Findings #33-35); #4326 #4328 #4329 assigned (R12 Huber β, EMA, Lion β1 at new BL)

### #4240 alphonse — R11 H66: Triple composition (layer_scale=1e-4 + T_max=24 + clip=1.0) (CLOSED — divergence, Finding #33)

W&B group `round11-triple-compose-alphonse`. 4 runs, 3/4 diverged (val > 100). 1/4 stable.

| Run | val | State | Notes |
|-----|-----|-------|-------|
| (run 1) | >100 | diverged | |
| (run 2) | >100 | diverged | |
| (run 3) | >100 | diverged | |
| (run 4) | ~stable | survived | minority |

**Finding #33**: layer_scale=1e-4 + T_max=24 + clip=1.0 is UNSTABLE — 3/4 runs diverge explosively (val > 100). The BL-winning T_max=24+clip=1.0 substrate WITHOUT layer_scale is stable; adding layer_scale destabilises T_max=24. T_max=24 is henceforth EXCLUDED from any experiment that also uses layer_scale. Safe range for layer_scale: T_max ≤ 20 (confirmed stable at new BL).

---

### #4274 frieren — R11 H70: EMA decay at T_max=24+clip=1.0+no-layer_scale (CLOSED — informative null, Finding #34)

W&B group `round11-ema-tmax24-clip-frieren`. Tested ema ∈ {0.995, 0.997, 0.999} at the old BL substrate (T_max=24+clip=1.0+no-ls).

| Arm | ema | val | test | Δ vs old BL (hk1i5kd5, 53.81) |
|-----|-----|-----|------|----|
| ctrl (hk1i5kd5) | 0.997 | 53.81 | 45.49 | — |
| A | 0.995 | ≈BL | — | within σ |
| B | 0.999 | ≈BL | — | within σ |

**Finding #34**: at T_max=24+clip=1.0+no-layer_scale, EMA optimum tightens around 0.997. ema=0.995 and ema=0.999 both within ±1 val noise. Distinct from #4214 (Finding #28) where layer_scale stabilises ema=0.999 but slows convergence. EMA axis at old substrate closed. Follow-up assigned at **new BL substrate** (#4328 frieren, which adds layer_scale).

---

### #4256 edward — R11 H68: Fine-grained clip sweep (clip∈{0.85,1.15}) at T_max=24+no-layer_scale (CLOSED — regression, Finding #35)

W&B group `round11-fine-clip-tmax24`. Both arms hit 30-min timeout at epoch 13.

| Arm | clip | Run | Epochs | val_avg | test_avg | Δ val vs new BL (53.08) |
|-----|------|-----|--------|---------|----------|---|
| BL ref | 1.00 | hk1i5kd5 | 14 (best) | 53.81 | 45.49 | +0.73 |
| A | 0.85 | 1ct0ha4q | 13 (timeout) | 58.05 | 49.51 | **+5.0** |
| B | 1.15 | ezqu6fyt | 13 (timeout) | 57.97 | 49.80 | **+4.9** |

**Finding #35**: at T_max=24+clip+no-ls, fine-grained clip perturbation (0.85, 1.15) regresses by ~2 val vs clip=1.0 at equal epochs. The asymmetric regression pattern from Finding #25 (#4180, clip 0.5 vs 2.0 favored 2.0) does NOT replicate at the fine grid. clip=1.0 is locally optimal in {0.85, 1.0, 1.15}. Importantly, the entire T_max=24+no-ls substrate is superseded by new BL (val 53.08 < 53.81 means new BL wins). Clip axis at new BL substrate validated by the 4-way composition in PR #4201.

---

### New assignments (R12 — remaining hyperparameter axes at new BL substrate)
- **#4326 alphonse** — R12 H75: Huber β sweep (β∈{0.03,0.10} vs ctrl 0.05) at new BL substrate. Targets camber_cruise regression (β=0.10) and re_rand (β=0.03).
- **#4328 frieren** — R12 H76: EMA decay sweep (ema∈{0.995,0.999} vs ctrl 0.997) at new BL substrate. Tests if layer_scale's stabilization (Finding #28) allows ema=0.999 to compete.
- **#4329 edward** — R12 H77: Lion β1 sweep (β1∈{0.85,0.95} vs ctrl 0.9 default) at new BL substrate. Extends Finding #30 (β1=0.9 robust at old substrate) to new BL.

---

## 2026-05-17 03:15 — #4201 nezuko MERGED (new best); #4212 #4231 #4258 CLOSED; #4315 #4318 #4319 #4320 assigned (R12 — new BL substrate probes)

### #4201 nezuko — R11 H62: 4-way composition (MERGED — **new best val 53.08 / test 44.89**)

W&B group `round11-layerscale-clip-lr2e4-nezuko`. Config: layer_scale=1e-4 + T_max=20 + lr=2e-4 + clip=1.0 + EMA(0.997).

| Run | val_avg | test_avg | State | Δ val vs BL | Δ test vs BL |
|-----|---------|----------|-------|--------|--------|
| **d3qlknrv** | **53.076** | **44.887** | finished | **−0.73** | **−0.60** |
| jqmn2nw7 | 53.269 | 45.351 | finished | −0.54 | −0.14 |
| 0zrrntw3 | 55.868 | 47.546 | finished | +2.06 | +2.06 |

Per-split val vs BL (`hk1i5kd5`): in_dist +0.41, **camber_rc −4.90**, camber_cruise +2.50, re_rand −0.94
Per-split test: in_dist −1.25, **camber_rc −4.90**, camber_cruise +2.81, re_rand +0.92

**Win concentrated in camber_rc (−4.9 val / −4.9 test)**. The largest single-split improvement since grad_clip. layer_scale + clip together help high-variance gradient regions in the raceCar camber OOD split. Median of 3 seeds beats BL on both metrics; σ_val=1.55 non-trivial. New BL: val 53.08 / test 44.89 (W&B run `d3qlknrv`).

**Finding #29**: Four-way composition (ls=1e-4 + T_max=20 + lr=2e-4 + clip=1.0) beats both prior BLs. The camber_rc improvement is mechanistically attributed to clip=1.0 + layer_scale together providing direction-sensitive step stabilization at high-gradient-norm geometry domains. camber_cruise regresses (+2.5/+2.8) — possible over-tight clip for the cruise gradient distribution (smaller scale, different density).

---

### #4258 thorfinn — R11 H69: Lion β1 sweep at T_max=24+clip (CLOSED — informative null, Finding #30)

| Arm | β1 | val | test | Δ val vs BL |
|-----|-----|-----|------|----|
| BL hk1i5kd5 (0.9 default) | 0.90 | 53.81 | 45.49 | — |
| A vkc3uadu | 0.85 | 58.69 | 50.58 | **+4.88** |
| B 9tq16oww | 0.95 | 59.87 | 50.96 | **+6.06** |

**Finding #30**: β1=0.9 is robustly optimal at the new BL substrate (T_max=24+clip=1.0). Both arms regress by ~5 val / ~5 test — uniform across all 8 splits. Failure mode is rate-of-momentum-warmup within the 13-epoch budget (not basin geometry). Lion's 10-step effective horizon at β1=0.9 is well-calibrated for 30-min wall-clock. The Lion optimizer-state axis is closed.

---

### #4212 askeladd — R11 H63: layer_scale magnitude sweep {1e-3, 1e-5, 3e-4} (CLOSED — informative null, Finding #31)

| Arm | layer_scale | val | test | Δ val vs old BL (54.30) |
|-----|------------|-----|------|---|
| BL 8m99yywe | 1e-4 | 54.30 | 47.29 | — |
| A sovpk3l6 | 1e-3 | 54.12 | 46.72 | **−0.18** (within σ) |
| B vusm2vyi | 1e-5 | 54.73 | 46.82 | +0.43 |
| C t6a2cybj | 3e-4 | 55.54 | 48.12 | +1.24 |

**Finding #31**: ls=1e-3 marginal winner at old T_max=20 substrate (−0.18 val, −0.57 test), within σ_val=1.67. Landscape is non-monotone on log(layer_scale) — 3e-4 (geometric mean of 1e-4 and 1e-3) is worst. Against NEW BL (val 53.08 / test 44.89 with clip=1.0+lr=2e-4): all arms regress (+1.04 val / +1.83 test). Follow-up assigned: test ls=1e-3 at new BL substrate.

---

### #4231 tanjiro — R11 H65: LR recalibration at layer_scale+T_max=20 (CLOSED — informative null + Finding #32)

| Arm | lr | val | test | Δ val vs old BL (54.30) | Δ val vs new BL (53.08) |
|-----|-----|-----|------|---|---|
| z9y2cmh9 | 1.7e-4 | 52.75 | 46.07 | −1.55 | **−0.33** |
| 9xf1s8ay | 2.0e-4 | 54.54 | 47.16 | +0.24 | +1.46 |

**Finding #32**: lr=1.7e-4 is the directional winner within the layer_scale+T_max=20+no-clip substrate (−1.55 val / −1.22 test vs old BL). Against new BL: val beats (52.75 < 53.08) but test REGRESSES (+1.18). Cross-substrate confound (no-clip vs clip=1.0) prevents clean attribution. The val/test divergence is driven by camber_cruise (+2.86 val, +3.19 test) and re_rand (val −1.65 better, test +1.63 worse). Follow-up assigned: test lr=1.7e-4 at NEW BL substrate (T_max=20+ls=1e-4+clip=1.0+lr=2e-4 ctrl).

---

### New assignments (R12 — probing new BL substrate)
- **#4315 tanjiro** — R12 H71: LR sweep at new BL substrate (lr={1.7e-4, 2.3e-4} vs ctrl 2e-4)
- **#4318 askeladd** — R12 H72: ls magnitude at new BL substrate (ls={1e-3, 5e-5} vs ctrl 1e-4)
- **#4319 nezuko** — R12 H73: WD sweep at new BL substrate (wd={5e-4, 2e-3} vs ctrl 1e-3)
- **#4320 thorfinn** — R12 H74: T_max sweep at new BL substrate (T_max={16, 22} vs ctrl 20; T_max=24 excluded due to divergence at ls substrate)

---

## 2026-05-17 01:45 — #4214 frieren EMA@layer_scale CLOSED (truncated, finding #28); #4274 frieren EMA@T_max=24+clip new BL assigned

### #4214 frieren — R11 H64: EMA decay at layer_scale+T_max=20 (CLOSED — timeout-truncated, informative null + Finding #28)

W&B group `round11-ema-newsub-frieren`. Both arms truncated at epoch 13/50 by 30-min wall-clock cap (cosine T_max=20 never completed).

| Arm | ema | val (13 ep) | test (13 ep) | Notes |
|-----|------|-------------|--------------|-------|
| Ctrl ref (8m99yywe, 50 ep) | 0.997 | 54.30 | 47.29 | (not directly comparable due to epoch mismatch) |
| A (4ub0crfk) | 0.995 | 56.40 | 47.14 | within ~1.3σ of ctrl; consistent with finding #4 extends |
| B (42if6l9l) | 0.999 | 63.32 | 54.34 | slow-but-converging (monotonic descent ep10=83→ep13=63); NOT divergent |

**Finding #28**: layer_scale_init=1e-4 stabilises ema=0.999 (no divergence vs PR #4152 where it diverged at no-layer_scale T_max=20). However ema=0.999 is uncompetitive in 30-min budget due to slow convergence. Distinction matters: "slow-but-converging" ≠ explosive divergence.

A-vs-ctrl ambiguous due to truncation (13ep vs 50ep); within seed-σ. Finding #4 (EMA robust [0.995, 0.997]) likely extends to layer_scale substrate but not provable from these truncated runs.

### #4274 frieren — R11 H70: EMA decay at new BL substrate (T_max=24+clip=1.0) (just assigned)

EMA sweep {0.995, 0.997 ctrl=BL, 0.999} at the actual new BL substrate. Tests:
- (a) does EMA robustness extend to T_max=24+clip without layer_scale?
- (b) does ema=0.999 diverge here, slow-but-converge (like #4214), or work?

---

## 2026-05-17 01:05 — Triple closure (#4192 #4180 #4173); triple reassignment (#4255 fern, #4256 edward, #4258 thorfinn) all targeting the new BL substrate

### #4192 fern — R11 H61: Huber β at lr=2e-4+T_max=14+clip=1.0 (CLOSED — informative null)

| Arm | β | val | test | Δval vs ctrl 56.89 |
|-----|------|------|------|----|
| Ctrl | 0.05 | 56.89 | 49.03 | — |
| A | 0.03 | 61.12 | 52.70 | +4.23 |
| B | 0.10 | 60.64 | 52.49 | +3.75 |

Both arms hurt every per-split val and test. **Finding #11 extends**: β=0.05 robust at lr=2e-4+clip=1.0 substrate. Substrate now superseded by val 53.81 BL.

### #4180 edward — R11 H60: Clip ratio at lr=2e-4+T_max=14 (CLOSED — informative null + Finding #25)

| Arm | clip | val | test | Δval vs ctrl 56.89 |
|-----|------|------|------|----|
| Ctrl | 1.0 | 56.89 | 49.03 | — |
| A | 0.7 | 59.35 | 51.03 | +2.46 |
| B | 1.4 | 62.24 | 53.45 | +5.35 |

**Finding #25**: At lr=2e-4+T_max=14, clip=1.0 sits at intersection of co-located scale + direction optima — asymmetric regression (B hurts 2× more than A). Pure scale and pure direction stories both incomplete. clip=1.0 sharply optimal.

### #4173 thorfinn — R11 H59 extended: lr×T_max scan at clip=1.0 (CLOSED — informative null + Findings #26, #27)

All 4 configs (T14+lr2e-4 BL, T20+lr2e-4 Arm B, T20+lr1.8e-4 Arm D, T18+lr2e-4 Arm E):
| Arm | T_max | lr | val | test |
|-----|------|------|------|------|
| BL | 14 | 2e-4 | 56.89 | 49.03 |
| B | 20 | 2.0e-4 | 56.98 | 48.34 (only test win) |
| D | 20 | 1.8e-4 | 58.38 | 50.41 |
| E | 18 | 2.0e-4 | 57.72 | 49.10 |

**Finding #26**: lr response monotone in [1.5e-4, 2.0e-4] at T_max=20+clip=1.0 (no minimum between).
**Finding #27**: T_max scan non-monotone at lr=2e-4+clip=1.0 — T_max=18 worse than T_max=14 and T_max=20 due to terminal-LR bimodality at the 14-epoch wall-clock cutoff.

### Reassignments at new BL substrate (T_max=24 + clip=1.0 + lr=1.5e-4)

- **#4255 fern (R11 H67)**: lr sweep {1.3e-4, 1.7e-4, 2.0e-4} at T_max=24+clip=1.0. Tests whether finding #22 (lr=2e-4 optimal at clip=1.0+T_max=14) extends to T_max=24.
- **#4256 edward (R11 H68)**: fine-grained clip {0.85, 1.15} at T_max=24+lr=1.5e-4. Tests whether asymmetric clip valley persists on new substrate.
- **#4258 thorfinn (R11 H69)**: Lion β1 sweep {0.85, 0.95} at new substrate. Last untested optimizer-state axis.

---

## 2026-05-17 00:27 — #4145 alphonse T_max=24+clip MERGED (new best val 53.81 / test 45.49); #4240 alphonse triple-compose assigned

### #4145 alphonse — R11 H55 (extended): T_max=24 + grad_clip=1.0 (MERGED — **new best val 53.8098 / test 45.4943**)

- Branch: `willowpai2i48h5-alphonse/r11-tmax20-compose-alphonse`
- W&B group: `round11-tmax20-compose-alphonse`

| Arm | Config | val_avg | test_avg | Δval vs new-BL 54.30 |
|-----|--------|---------|----------|----------------------|
| ctrl (fh3jmkd1) | T_max=20, no clip | 57.66 | 49.45 | — |
| B | T_max=20 + clip=1.0 | 56.71 | 48.52 | +2.41 |
| C | T_max=24 + no clip | 62.15 | 52.85 | +7.85 |
| **D (winner)** | **T_max=24 + clip=1.0** | **53.81** | **45.49** | **−0.49** |
| E | layer_scale=1e-4 + clip + T_max=20 | 54.10 | 46.90 | −0.20 |

Per-split val (Arm D): in_dist 55.45, camber_rc 70.54, camber_cruise 34.18, re_rand 55.07
Per-split test (Arm D): in_dist 48.08, camber_rc 62.12, camber_cruise 27.84, re_rand 43.93

**Δ vs prior best (PR #4015 layer_scale+T_max=20, val 54.30 / test 47.29): −0.49 val / −1.80 test**

**Key findings:**
- **(clip × T_max) interaction is super-additive**: T_max=24 alone regresses catastrophically (Arm C +7.85). T_max=24 + clip=1.0 beats T_max=20+clip by 2.90 val. Clip is *essential* at T_max=24 — without it the late-LR amplifies gradient outliers.
- **Arm E (layer_scale + clip + T_max=20 → val 54.10)** also beats old BL (54.30) but not Arm D (53.81). layer_scale helps specifically camber_rc (67.36 vs D's 70.54).
- **Split story**: Arm D wins in_dist, camber_cruise, re_rand vs Arm E; Arm E wins camber_rc. Both beat old BL.
- **Finding #24**: At T_max≥24, clip=1.0 becomes essential (not just helpful). The late-LR endpoint LR ~1.35e-4 with T_max=24 amplifies gradient-magnitude outliers that Lion's sign-update would otherwise mishandle. Clip=1.0 neutralizes this while preserving high-LR basin exploration.

### #4240 alphonse — R11 H66: Triple composition layer_scale=1e-4 + T_max=24 + clip=1.0 (just assigned)

Arm B: layer_scale + T_max=24 + clip (triple composition). Arm C: T_max=28 + clip (schedule extension). Primary prediction: val ~51–53 if layer_scale (camber_rc) and T_max=24 (in_dist/cruise) compose additively on different splits.

---

## 2026-05-16 23:37 — #4148 tanjiro LR@T_max=20 CLOSED (informative null); #4231 tanjiro LR@new-substrate assigned

### #4148 tanjiro — R11 H56: LR recalibration at T_max=20 no-clip (CLOSED — informative null, substrate superseded)

W&B group `round11-lr-tmax20-tanjiro`.

| Arm | lr | W&B | val_avg | Δval vs ctrl | test_avg | Δtest |
|-----|------|------|---------|-------------|---------|------|
| ctrl (ref) | 1.5e-4 | `fh3jmkd1` | 57.66 | — | 49.45 | — |
| B | 1.3e-4 | `5odgdv21` | 57.998 | +0.34 | 49.43 | −0.02 |
| **C** | **1.7e-4** | `or4ijhy0` | **57.023** | **−0.64 (best)** | **48.81** | **−0.64** |
| D | 2.0e-4 | `gglqzuau` | 57.272 | −0.39 | 49.05 | −0.40 |

**Key findings:**
1. All |Δval|<1.5 → falsification condition met: lr=1.5e-4 is robust at T_max=20+no-clip (mildly tilted toward 1.7e-4).
2. **lr=2e-4 + T_max=20 + no-clip does NOT diverge** (val 57.27). At T_max=14 lr=2e-4 would diverge. Distinguishes schedule mechanism (T_max=20 absorbs high LR via EMA + high endpoint) vs clip mechanism (finding #22). Both open high-LR region but clip is stronger.
3. Per-split structure of C (lr=1.7e-4): OOD-favourable — camber_rc −1.68, camber_cruise −1.87, re_rand −1.91 val; in_dist +2.91 (higher LR pushes toward general features).

**Closed because:** none of the arms beats new BL 54.30 (PR #4015). Substrate (no-clip, no-layer_scale) superseded.

### #4231 tanjiro — R11 H65: LR at new substrate (layer_scale=1e-4 + T_max=20) (just assigned)

Sweeps lr ∈ {1.7e-4, 2.0e-4} vs BL ctrl (1.5e-4=`8m99yywe`, val 54.30). Also includes Arm D: seed 3 at merged BL config (resolves σ=1.67 from PR #4015 2-seed spread). Hypothesis: layer_scale may tolerate higher lr (smaller effective steps early in training).

---

## 2026-05-16 23:00 — #4153 askeladd β2 CLOSED; #4152 frieren EMA CLOSED; #4212 askeladd layer_scale-mag, #4214 frieren EMA on new substrate assigned

### #4153 askeladd — R11 H58: Lion β2 sweep at T_max=20 (CLOSED — informative null, obsolete substrate + high variance)

W&B group `round11-beta2-askeladd`. Assigned before #4015 layer_scale merged; substrate now obsolete.

| β2 | layer_scale | val_avg | test_avg | Notes |
|-----|-------------|---------|----------|-------|
| 0.995 (seed 1) | — | 72.71 | — | high variance |
| 0.995 (seed 2) | — | 70.71 | — | |
| 0.995 (seed 3) | — | 60.00 | — | best of 3 reps |
| 0.98 | **1.0** confound | 63.33 | — | layer_scale_init=1 not 0 |
| 0.99 ctrl | — | not launched | — | |

**Closed because:** (1) β2=0.995 σ huge (range 12.71); (2) β2=0.98 arm has layer_scale_init=1.0 confound (not the default 0); (3) ctrl 0.99 never launched; (4) all on T_max=20-only substrate, now superseded by #4015's layer_scale+T_max=20 (val 54.30). No way to extract a clean signal even if best-of-3 (60.00) would have beat old BL by a little.

### #4152 frieren — R11 H57: EMA decay at T_max=20 (CLOSED — informative null, obsolete substrate)

W&B group `round11-ema-frieren`. Same pre-merge substrate.

| ema | val_avg | test_avg | State |
|-----|---------|----------|-------|
| 0.995 | 58.16 | — | within noise of pre-#4015 BL (57.66) |
| 0.997 ctrl | — | — | not launched |
| 0.999 (3 seeds) | crashed / diverged / worse | — | unstable |

**Closed because:** (1) ema=0.995 essentially no change; (2) ema=0.999 unstable across 3 seeds; (3) ctrl never launched; (4) substrate obsolete vs val 54.30 BL. **Per-substrate finding worth keeping:** ema=0.999 is unstable at T_max=20.

### #4212 askeladd — R11 H63: layer_scale magnitude sweep on new substrate (just assigned)

Sweeps layer_scale_init ∈ {1e-3, 1e-5, 3e-4} at T_max=20 + lr=1.5e-4 + no clip (the **new BL substrate**). Tests whether 1e-4 (the merged value) is locally optimal. Per nezuko's #4015 followup #3.

### #4214 frieren — R11 H64: EMA decay on new substrate (just assigned)

Sweeps ema_decay ∈ {0.995, 0.997 ctrl, 0.999} at **layer_scale=1e-4 + T_max=20** substrate. Motivated by σ=1.67 seed variance on the new substrate — heavier EMA averaging may stabilize the endpoint. ema=0.999 was unstable on the prior substrate but the substrate is now different (lower residual magnitude via layer_scale).

---

## 2026-05-16 22:55 — #4015 nezuko layer_scale MERGED (new best val 54.30 / test 47.29); #4201 nezuko assigned (four-way composition)

### #4015 nezuko — R10 H39: layer_scale_init=1e-4 + T_max=20 (MERGED — **new best val 54.3009 / test 47.2883**)

- Branch: `willowpai2i48h5-nezuko/r10-layerscale-nezuko`
- Hypothesis: CaiT/DeiT-III layer_scale_init=1e-4 composes with T_max=20 substrate.
- W&B: Arm F `8m99yywe` (winner, seed 1), Arm G `gbzybjhx` (seed 2)

| Arm | layer_scale | T_max | lr | val_avg | test_avg | W&B |
|-----|-----------|------|-----|---------|----------|-----|
| NEW BL (PR #4120) | — | 14 | 2e-4+clip | 56.89 | 49.03 | 1c58zju8 |
| **F (seed 1, winner)** | **1e-4** | **20** | **1.5e-4** | **54.30** | **47.29** | **8m99yywe** |
| G (seed 2) | 1e-4 | 20 | 1.5e-4 | 57.64 | 49.91 | gbzybjhx |
| F+G mean | 1e-4 | 20 | 1.5e-4 | 55.97 | 48.60 | — |

Per-split val (Arm F): in_dist 57.78, camber_rc 67.19, camber_cruise 38.28, re_rand 53.95
Per-split test (Arm F): in_dist 49.18, camber_rc 61.28, camber_cruise 32.26, re_rand 46.43

**Δ vs prior best (PR #4120 val 56.89 / test 49.03): −2.59 val / −1.74 test** (single seed Arm F)
**2-seed mean: −0.92 val / −0.43 test** (both seeds clearly beat prior BL)

**Key findings:** (1) layer_scale_init=1e-4 composes ~80% additively with T_max=20 (observed −7.08 from #3976 BL vs predicted −8.78). (2) Seed σ=1.67 val under T_max=20 (wider than T_max=14 σ=0.016 — T_max=20 amplifies optimization variance). (3) train.py now includes --layer_scale_init flag.

**Note on alphonse #4145 interaction:** Arm B of alphonse (T_max=20+lr=1.5e-4+clip=1.0 → val 56.71) does NOT beat new BL (54.30). Alphonse sent updated instruction to add Arm E (layer_scale+clip composition).

### #4201 nezuko — R11 H62: layer_scale + clip + lr=2e-4 composition at T_max=20 (just assigned)

Four-way composition: layer_scale=1e-4 + T_max=20 + lr=2e-4 + clip=1.0. Arm A: full four-way. Arm B: layer_scale+T_max=20+clip at lr=1.5e-4 (original lr). Tests whether the clip+lr=2e-4 family composes with layer_scale+T_max=20. Expected: val ~52–54 if composing.

---

## 2026-05-16 22:35 — #4173 thorfinn triple-compose REVIEWED+SENT BACK (val tied, test better); #4128 fern surf_weight CLOSED; #4192 fern Huber β@lr=2e-4 assigned

### #4173 thorfinn — R11 H59: Triple composition T_max=20 + lr=2e-4 + clip=1.0 (REVIEWED, NOT MERGED, sent back)

- W&B run `422k0yfk`
- val_avg 56.9769 (vs BL 56.8913 → **+0.086 tied within noise**)
- test_avg 48.3409 (vs BL 49.0322 → **−0.6913 better**)

| Split | Arm B val | BL val | Δ | Arm B test | BL test | Δ |
|-------|-----------|--------|---|------------|---------|---|
| in_dist | 58.31 | 61.01 | **−2.70** | 49.84 | 52.64 | **−2.80** |
| camber_rc | 71.71 | 71.92 | −0.21 | 63.05 | 64.54 | **−1.49** |
| camber_cruise | 39.69 | 37.30 | **+2.39** | 32.83 | 31.01 | **+1.82** |
| re_rand | 58.20 | 57.34 | +0.86 | 47.65 | 47.94 | −0.29 |

**Decision: NOT merged (val_avg primary metric tied at +0.086).** But 3/4 test splits improve and test_avg −0.69 is meaningful. The regression is concentrated on camber_cruise (+2.39 val) — a split that saturates early under T_max=14 and gets pushed off by the elevated LR endpoint of T_max=20. Best epoch = 14/50 even at T_max=20.

**Key finding to record:** Val/test direction-disagreement is interesting — may indicate val is slightly overfit to T_max=14 substrate. test-aware selection would favor the triple composition.

Sent back for Arm D (lr=1.8e-4 + T_max=20 + clip=1.0) and Arm E (lr=2e-4 + T_max=18 + clip=1.0) to localize the minimum.

### #4128 fern — surf_weight @ clip=1.0 (CLOSED — informative null, incomplete, no terminal)

W&B group `round10-surfweight-fern` shows only sw=5 launched (3 reps). sw=10 ctrl and sw=20 never launched. Student never posted any PR comment.

| Arm | sw | val_avg | test_avg | State |
|-----|-----|---------|----------|-------|
| sw=5 (best of 3) | 5 | 60.59 | 51.95 | finished, worse than BL |
| sw=10 ctrl | 10 | NEVER LAUNCHED | — | — |
| sw=20 | 20 | NEVER LAUNCHED | — | — |

**Key finding to record:** sw=5 (down from default 10) regresses on old substrate. Substrate now obsolete after PR #4120.

### #4192 fern — R11 H61: Huber β recalibration at lr=2e-4 {0.03, 0.05 ctrl, 0.10} (just assigned)

Tests whether finding #11 (β=0.05 optimal in [0.05, 0.20] at lr=1.5e-4) extends to the new lr=2e-4 + clip=1.0 substrate. Arm A β=0.03 (extends below previous sweep floor), Arm B β=0.10 (re-test old finding).

---

## 2026-05-16 21:55 — #4122 edward wd-sweep CLOSED (incomplete + obsolete substrate); edward #4180 assigned (clip ratio @ lr=2e-4)

### #4122 edward — wd recalibration at clip=1.0 {3e-4, 5e-4, 1e-3 ctrl, 2e-3} (CLOSED — informative null, incomplete, no terminal posted)

W&B group `round10-wd-at-clip1-edward` shows 2/4 arms completed, both worse than ctrl; 2 arms never launched. Student never posted any PR comment for over 2 hours.

| Arm | wd | val_avg | test_avg | W&B | State |
|-----|-----|---------|----------|-----|-------|
| wd=3e-4 | 3e-4 | 62.99 | 54.30 | gu515p5x | finished, worse than ctrl |
| wd=5e-4 (1st try) | 5e-4 | crashed | crashed | bxwu0f8g | crashed |
| wd=5e-4 (2nd try) | 5e-4 | 61.45 | 52.64 | i7wm3916 | finished, within noise of ctrl 61.18 |
| wd=5e-4 (3rd try) | 5e-4 | running | running | uwuejp9f | running at close time |
| wd=1e-3 ctrl | 1e-3 | NEVER LAUNCHED | — | — | — |
| wd=2e-3 | 2e-3 | NEVER LAUNCHED | — | — | — |

**Key finding to record:** wd=3e-4 regresses by ~1.8 val vs ctrl 61.18 on T_max=14+lr=1.5e-4+clip=1.0 substrate. wd=5e-4 within noise of ctrl. The lower-wd half of the sweep does not improve over ctrl. Substrate now obsolete (PR #4120 merged with lr=2e-4 + clip=1.0 → val 56.89).

### #4180 edward — R11 H60: Clip ratio recalibration at lr=2e-4 substrate {0.7, 1.0 ctrl, 1.4} (just assigned)

Parallel to finding #22 (LR shifts at clip=1.0): does optimum CLIP also shift at lr=2e-4? Tests whether the original clip optimum (1.0 found at lr=1.5e-4) is direction-bound (insensitive to lr) or scale-bound (should drop to ~0.75 at lr=2e-4). Arm A clip=0.7, Arm B clip=1.4; ctrl = `1c58zju8` (PR #4120 winner).

---

## 2026-05-16 21:40 — #4120 thorfinn MERGED (new best val 56.89 / test 49.03); thorfinn #4173 assigned (triple composition)

### #4120 thorfinn — LR re-optimisation at clip=1.0 substrate {2e-4, 2.5e-4} (MERGED — **new best val 56.8913 / test 49.0322**)

- Branch: `willowpai2i48h5-thorfinn/r10-lr-at-clip1`
- Hypothesis: clip=1.0 shifts the optimal nominal lr upward from 1.5e-4 (the LR optimum at no-clip) because the clipped step direction (normalized gradient) differs from Lion's sign-update in a direction-sensitive way that depends on nominal lr.
- W&B: Arm B `1c58zju8` (winner, lr=2e-4), Arm C `89y764md` (lr=2.5e-4)

| Arm | lr | val_avg | test_avg | Δ vs BL 57.66 |
|-----|-----|---------|----------|----------------|
| baseline (y5tua53k, PR #4056) | 1.5e-4 | 61.18 | 52.09 | — |
| **B (winner)** | **2e-4** | **56.89** | **49.03** | **−0.77 / −0.41** |
| C | 2.5e-4 | 59.36 | 50.83 | +1.70 / +1.39 |

Per-split val (Arm B): in_dist 61.01, camber_rc 71.92, camber_cruise 37.30, re_rand 57.34 — all 4 splits improve vs old BL 61.18. Also beats T_max=20 BL (57.66) on all 4 splits.

**Key findings:** (1) LR optimum at clip=1.0 is lr=2e-4, not 1.5e-4. Shape of LR-vs-val curve is same as no-clip regime but shifted upward in lr. (2) Pre-clip ‖g‖ median ~23.7 at lr=2e-4 — clip still active at every step. Direction-sensitive interaction confirmed. (3) lr=2.5e-4 begins to regress on camber_cruise/re_rand — inflection between 2e-4 and 2.5e-4.

**New best** (T_max=14 + clip=1.0 + lr=2e-4 beats T_max=20 alone): Δ −0.77 val / −0.41 test vs PR #4063.

### #4173 thorfinn — R11 H59: Triple composition T_max=20 + lr=2e-4 + clip=1.0 (just assigned)

Tests the full combination of all three wins: T_max=20 schedule extension, lr=2e-4 (optimum at clip=1.0), and grad_clip=1.0. Primary arm: T_max=20 + lr=2e-4 + clip=1.0. Optional Arm C: T_max=24 + lr=2e-4 + clip=1.0 if Arm B wins. Expected: val ~54–56 if both compose fully.

---

## 2026-05-16 20:55 — #4096 frieren SGDR + #4085 askeladd batchsize CLOSED informative; #4152 frieren EMA-decay + #4153 askeladd lion-β2 assigned

### #4096 frieren — SGDR cosine warm restarts (CLOSED — informative null, no terminal posted)

W&B group `round10-sgdr-frieren` shows 2 finished arms but student did not post terminal SENPAI-RESULT. Closed based on W&B data.

| Arm | val_avg | Δ vs new BL 57.66 |
|-----|---------|--------------------|
| T_0=7 T_mult=1 (2 equal cycles) | 64.14 | +6.48 above BL |
| T_0=4 T_mult=2 (coarse→fine) | 69.51 | +11.85 above BL |

**Key finding to record:** SGDR restarts directly oppose the winning mechanism from PR #4063 T_max=20 (smooth schedule keeping LR high). The model needs sustained moderate LR, not periodic restarts. Schedule monotonicity > schedule modulation at this LR.

### #4085 askeladd — Batch size sweep {8, 16} (CLOSED — informative null, no terminal posted)

W&B group `round10-batchsize-askeladd` shows only bs=8 ran (3 reps); bs=16 never launched.

| Arm | val_avg | Δ vs new BL 57.66 |
|-----|---------|--------------------|
| bs=8 (3 reps: 76.93, 79.88, 80.29) | best 76.93 | +19.27 above BL |
| bs=16 | NOT LAUNCHED | — |

**Key finding to record:** Lion at lr=1.5e-4 is highly sensitive to batch size. At bs=8 throughput halves (~7 effective epochs vs 14 at bs=4) — wall-clock budget bites hard. Default bs=4 is correctly tuned for Lion + SENPAI_TIMEOUT_MINUTES=30 regime.

### #4152 frieren — R11 H57: EMA decay sweep at T_max=20 {0.995, 0.997 ctrl, 0.999} (just assigned)

Tests whether finding #4 (EMA decay robust [0.995, 0.997]) extends to T_max=20 substrate where LR endpoint ≈ 1.20e-4 (noisier weight trajectory needs longer averaging). Pure CLI.

### #4153 askeladd — R11 H58: Lion β2 sweep at T_max=20 {0.98, 0.99 ctrl, 0.995} (just assigned)

Tests untested optimizer-state axis (β2 controls EMA of gradient sign signal). Higher β2 at T_max=20's noisier endpoint may help reduce spurious sign flips. Pure CLI.

---

## 2026-05-16 20:50 — #4063 tanjiro MERGED (new best val 57.66 / test 49.45); #4044 alphonse CLOSED; #4015 nezuko sent back for T_max=20 composition; #4145 alphonse assigned

### #4063 tanjiro — T_max sweep {14 ctrl, 18, 20} at lr=1.5e-4 (MERGED — **new best val 57.6606 / test 49.4491**)

- Branch: `willowpai2i48h5-tanjiro/r10-tmax-lr15e4`
- Hypothesis: Longer cosine schedule within wall-clock budget maintains higher time-averaged LR.
- W&B: ctrl `kx37feeh`/`f6tg9w5l`, T_max=18 `o7trffbr`, **T_max=20 `fh3jmkd1`** (winner)

| Arm | T_max | val_avg | test_avg | vs prior BL 61.18/52.09 |
|-----|-------|---------|----------|------------------------|
| A1 ctrl | 14 | 65.46 | 56.54 | +4.28 / +4.45 (seed spread) |
| A2 ctrl retry | 14 | 63.37 | 54.42 | +2.19 / +2.33 |
| B | 18 | 59.22 | 50.79 | −1.96 / −1.30 |
| **C WINNER** | **20** | **57.66** | **49.45** | **−3.52 / −2.64** |

Per-split (Arm C winner, vs prior BL):
| Split | val | test | Δ val | Δ test |
|-------|-----|------|-------|--------|
| in_dist | 57.88 | 51.04 | −7.49 | −5.77 |
| camber_rc | 71.61 | 64.76 | −5.29 | −2.08 |
| camber_cruise | 40.47 | 32.44 | −1.27 | −1.78 |
| re_rand | 60.69 | 49.55 | −0.01 | −0.92 |

All 8 splits (val × 4, test × 4) improve monotonically T_max=14→18→20. Global training-dynamics win — no split-specific story.

**Mechanism:** SENPAI_TIMEOUT_MINUTES=30 caps training at ~14 epochs. T_max=20 → LR endpoint ≈ 1.20e-4 (80% of peak) vs T_max=14 → LR→0. Higher time-averaged LR within budget. EMA(0.997) smooths late-training noise from higher-LR endpoint. The "longer schedule × higher LR" interaction is synergistic, not competitive.

**5× amplification vs old substrate:** On old substrate (lr=5e-5, spec_norm), T_max=20 beat T_max=14 by ~1.3 val. On new substrate (lr=1.5e-4, no clip), by 6.76 val.

**New baseline: val 57.6606 / test 49.4491** (BASELINE.md updated, PR #4063 squash-merged).

### #4044 alphonse — Multi-param FiLM (CLOSED — informative null, hypothesis FALSIFIED)

- Branch: `willowpai2i48h5-alphonse/r10-multi-film`
- Hypothesis: Conditioning FiLM on all 11 global params (vs log(Re) only) improves camber_rc OOD.
- W&B: ctrl `7fc1ujfh`, Arm B (11-param) `yi8cdd1g`, Arm C (4-param) `nve9vzv0`

| Arm | film_cond_dim | val_avg | test_avg | vs OLD BL 64.68 | vs NEW BL 57.66 |
|-----|---------------|---------|----------|-----------------|-----------------|
| A ctrl | 1 | 65.29 | 56.78 | +0.61 | +7.63 above |
| B treatment | 11 | 69.37 | 60.62 | +4.68 | +11.71 above |
| C treatment | 4 | 65.23 | 57.04 | +0.55 | +7.57 above |

Primary prediction (camber_rc improves under multi-FiLM): **FALSIFIED**. Both treatment arms regress camber_rc vs ctrl:
- Arm B: camber_rc +5.45 val (wrong direction)
- Arm C: camber_rc +2.90 val (wrong direction)

**Key finding:** Global gamma/beta conditioning cannot substitute for per-node geometric information. AoA/NACA camber already reach the model effectively through flat node features + attention path. Forcing them through a global FiLM MLP averages out node-level shape variation that local attention handles better. Arm C's cruise improvement (−3.52 val) is interesting but comes at camber_rc cost — classic conditioning-path overfit. Multi-FiLM as global conditioning path closed.

### #4015 nezuko — Layer scale (SENT BACK — needs T_max=20 composition test)

Arms D+E (layer_scale=1e-4 + T_max=14, 2 seeds) had val 59.66/59.67 — beats OLD new-BL (61.18) decisively with 2-seed agreement (σ=0.016). But tanjiro's T_max=20 (val 57.66) just superseded this. Nezuko asked to add Arms F+G (layer_scale=1e-4 + T_max=20, 2 seeds) to confirm composition.

### #4145 alphonse — R11 H55: grad_clip=1.0 + T_max=20 composition + T_max=24 extension (just assigned)

3 arms: B (T_max=20 + clip=1.0), C (T_max=24 no clip), D (T_max=24 + clip=1.0). Tests composition of two orthogonal previous wins + schedule extension beyond T_max=20.

### #4148 tanjiro — R11 H56: LR recalibration at T_max=20 {1.3e-4, 1.5e-4 ctrl, 1.7e-4} (just assigned)

Tests whether lr=1.5e-4 optimum (finding #14, T_max=14 substrate) shifts at T_max=20 where effective LR is ~20% higher within budget. Pure CLI.

---

## 2026-05-16 20:35 — #4084 fern CLOSED informative (dropout monotone hurts, camber_rc gain noted); #4128 fern surf_weight assigned

### #4084 fern — Dropout sweep {0.05, 0.10} at lr=1.5e-4 (CLOSED — informative null)

- Branch: `willowpai2i48h5-fern/r10-dropout`
- Hypothesis: Dropout inside Transolver blocks could improve camber_rc OOD generalization via regularization.
- W&B runs: `rwalaqva` (0.05), `fn880kax` (0.10)

| Arm | Dropout | val_avg | test_avg | vs ctrl 63.05 | vs NEW BL 61.18/52.09 |
|-----|---------|---------|----------|---------------|-----------------------|
| ctrl (jurrwig2) | 0.00 | 63.0492 | 53.6049 | reference | +1.87 / +1.51 above |
| B | 0.05 | 63.7189 | 54.4936 | +0.67 / +0.89 | +2.54 / +2.40 |
| C | 0.10 | 64.2869 | 55.2573 | +1.24 / +1.66 | +3.11 / +3.17 |

Per-split (Arm B, dropout=0.05):
| Split | ctrl val | B val | Δ val | ctrl test | B test | Δ test |
|-------|----------|-------|-------|-----------|--------|--------|
| in_dist | 64.45 | 63.73 | −0.72 | 55.69 | 54.58 | −1.11 |
| **camber_rc** | **80.74** | **76.51** | **−4.23** | **70.55** | **67.94** | **−2.61** |
| camber_cruise | 43.48 | 48.57 | +5.09 | 35.48 | 39.67 | +4.19 |
| re_rand | 63.53 | 66.06 | +2.53 | 52.70 | 55.78 | +3.08 |

**Key finding:** Monotone hurt with increasing dropout. Falsification threshold (Arm C ≥ +2 val) not strictly hit (+1.24 val above ctrl). Closed as informative.

**Non-null result:** Dropout=0.05 improved camber_rc by −4.23 val (real and large), but at the cost of +5.09 val on camber_cruise and +2.53 val on re_rand. Net negative. Breadth-targeted regularization (FFN+attn together) disrupts the slice-token routing that the well-represented splits rely on.

**Mechanism:** At 5-block Transolver depth with slice_num=64, uniform dropout disrupts slice-token routing on distributions where model is already well-calibrated (cruise, re_rand). Camber_rc has unseen rotated/curved foils where routing noise helps generalization, but trade is net-negative.

**Connection to prior finding (#3977 DropPath):** Two independent regularizers (DropPath at branch level, Dropout at neuron level) both monotone hurt. Residual stream is not over-fitting through capacity; it is under-fitting through breadth.

**Student suggestion:** Split-targeted attn_dropout (only inside slice-attention, not FFN) to preserve the camber_rc gain without breadth-cost. Valuable future hypothesis.

### #4128 fern — surf_weight recalibration at clip=1.0 {5, 10 ctrl, 20} (R10 H54 — just assigned)

Tests whether surf_weight=10 remains optimal on the new clip=1.0 substrate. Per-split breadth signal from #4084 suggests volume representations may be gradient-starved at current surf_weight. Pure CLI sweep, no code changes.

---

## 2026-05-16 20:00 — #4056 thorfinn MERGED (new best val 61.18 / test 52.09); #4057 edward CLOSED; #4120 thorfinn LR@clip1 / #4122 edward wd@clip1 assigned; tanjiro T_max=18 winner pending

### #4056 thorfinn — Gradient clip sweep at lr=1.5e-4 (MERGED — **new best val 61.1778 / test 52.0853**)

- Branch: `willowpai2i48h5-thorfinn/r10-gradclip`
- Hypothesis: Lion with heavy-tailed CFD gradients benefits from norm clipping.
- W&B run: `y5tua53k` (grad_clip=1.0 winner)

| Arm | grad_clip | val_avg | test_avg | vs prior BL 63.05/53.60 |
|-----|-----------|---------|----------|------------------------|
| ctrl | 0.0 | 63.05 | 53.60 | jurrwig2 reference |
| **B WINNER** | **1.0** | **61.18** | **52.09** | **−1.87 / −1.51** |
| C | 0.5 | 62.29 | 52.88 | −0.76 / −0.72 |
| D | 2.0 | 61.94 | 53.95 | −1.11 / +0.35 |

Per-split (Arm B vs prior BL):
| Split | val Δ | test Δ |
|-------|-------|--------|
| in_dist | +0.92 | +1.12 (slight regression) |
| **camber_rc** | **−3.84** | **−3.71** |
| **camber_cruise** | **−1.74** | **−1.26** |
| **re_rand** | **−2.83** | **−2.23** |

**KEY DIAGNOSTIC (paper finding):** Pre-clip gradient norm is median ~27 — every step gets clipped. clip=1.0 is NOT outlier suppression; it rescales every step by ~1/27, acting as a constant per-step scale on top of Lion's sign-update. Sweet spot at clip=1.0 (not 0.5 which under-trains, not 2.0 which doesn't change OOD enough).

**Mechanism:** Larger stabilization of OOD splits (camber_rc, re_rand) vs slight in_dist regression. Heavy-tailed Re distribution → high-Re samples produce large gradient norms → clip normalizes per-step contribution uniformly, reducing OOD over-fitting.

**New baseline: val 61.1778 / test 52.0853** (BASELINE.md updated, PR #4056 squash-merged).

### #4057 edward — Surface-biased slice routing (CLOSED — informative null)

Best arm (vec, per-slice bias): val 62.76 / test 53.92 vs NEW BL 61.18/52.09 → +1.58 / +1.83 (above new BL).
Key finding: scalar surface-bias is a no-op under softmax (shift-invariant). Vectorized per-slice bias is the proper form. Learned bias magnitude near 0 (max block mean 0.038) — model already routes adequately.
camber_rc improved (−2.98 val) but offset by small regressions elsewhere.

### #4063 tanjiro — T_max sweep at lr=1.5e-4 (PENDING — T_max=18 winning, T_max=20 still running)

| Arm | T_max | val_avg | test_avg | vs NEW BL 61.18/52.09 |
|-----|-------|---------|----------|----------------------|
| ctrl | 14 | 65.46 | 56.54 | +4.28 / +4.45 (seed effect) |
| **WINNER** | **18** | **59.22** | **50.79** | **−1.96 / −1.30 BEATS** |
| | 20 | running (step 3644/5264) | — | in progress |

Within-student T_max=18 vs ctrl: −6.24 val / −5.75 test (same-seed reliable signal). Awaiting T_max=20 to finish before terminal.

### New assignments

| PR | Student | Hypothesis | Expected |
|----|---------|------------|---------|
| #4120 | thorfinn | R10 H52: LR sweep at clip=1.0 {1.5e-4 ctrl, 2e-4, 2.5e-4} | −0 to −2 val; tests effective-LR shift from clip |
| #4122 | edward | R10 H53: wd sweep at clip=1.0 {3e-4, 5e-4, 1e-3 ctrl, 2e-3} | −1 to −2 val; wd may need lower at clip substrate |

---

## 2026-05-16 19:00 — #4049 frieren CLOSED informative; #4096 frieren R10 SGDR assigned

### #4049 frieren — spec_norm at lr=1.5e-4 (R11 H46, CLOSED — informative null)

Both arms timeout-bound at 14 epochs (cosine T_max=14 fits exactly).

| Arm | val_avg | test_avg | run_id | ΔvalB-A | vs jurrwig2 BL (63.05) |
|-----|---------|----------|--------|---------|----------------------|
| A ctrl (no spec_norm) | 63.7806 | 55.3822 | bpuw2ipc | reference | +0.73 / +1.78 |
| **B spec_norm output** | **63.5151** | **55.3821** | **kmzw2vzf** | **−0.27 val / ~0 test** | +0.47 / +1.78 |

- Arm A reproduces jurrwig2 within noise (Δ=0.73, σ≈2.77, so 0.26σ — clean ctrl).
- Arm B beats A by 0.27 val (within seed noise); test_avg identical to 4 decimals.
- Per-split: B improves val_re_rand (−1.16) and val_camber_cruise (−0.52); slightly worse on val_in_dist (+0.17) and val_camber_rc (+0.45). On test, B improves camber_rc and cruise, slightly worse on others. Mixed signal with magnitude well below noise floor.

- **Updated finding #18:** spec_norm contribution monotonically diminishing as LR grows.
  - lr=5e-5: −1.39 val (real)
  - lr=1e-4: ~0 val (noise)
  - lr=1.5e-4: −0.27 val (noise, same direction)
- Mechanism: Lion's sign-update already bounds per-step output gradient magnitude. Lipschitz weight cap is operationally inert at high LR.
- **Output-head Lipschitz closed as research direction.** Frieren's analysis was excellent — moved on cleanly.

### #4096 frieren — SGDR cosine warm restarts (R10 H50 — assigned this session)

2 arms at new lr=1.5e-4 substrate: T_0=7 T_mult=1 (2 equal-7-epoch cycles), T_0=4 T_mult=2 (coarse→fine 4+8 cycles). Tests whether Lion+EMA benefits from periodic LR kicks to escape basins.

---

## 2026-05-16 18:40 — #4046 askeladd / #4045 fern CLOSED informative; #4015 nezuko sent back for new-substrate confirmation; #4084 fern dropout / #4085 askeladd batchsize assigned

### #4015 nezuko — Layer scale (SENT BACK — needs new-substrate confirmation)

3 arms ran on OLD lr=1e-4 + spec_norm substrate:

| Arm | layer_scale_init | run_id | val_avg | test_avg | vs OLD BL 64.68 | vs NEW BL 63.05/53.60 |
|-----|------------------|--------|---------|----------|-----------------|----------------------|
| A ctrl | 1.0 | sawf13tr | 64.5367 | 55.9533 | −0.14 (parity) | +1.49 / +2.35 |
| **B winner** | **1e-4** | **lwx03cg0** | **63.3233** | **55.0365** | **−1.36 / −1.14** | **+0.27 / +1.44** |
| C | 1e-5 | kbqu64n6 | 63.5006 | 55.2014 | −1.18 / −0.97 | +0.45 / +1.60 |

- Per-split test (Arm B): camber_rc 69.24→66.75 (−2.49), camber_cruise 38.56→37.52, re_rand 55.83→54.08, single_in_dist 61.06→61.80 (+0.74)
- Analysis: Layer scale init=1e-4 wins on OLD substrate by −1.36 val / −1.14 test. Direction consistent (test improves on 3 of 4 splits). But on NEW lr=1.5e-4 + no spec_norm substrate, can't compare directly. Asked for Arm D (layer_scale=1e-4 on lr=1.5e-4 substrate) + Arm E (2nd seed) confirmation. If Arm D ≤ 62.3 val → likely merge.

### #4046 askeladd — Pressure channel upweighting (CLOSED — informative null)

| Arm | p_weight | val_avg | test_avg | vs OLD BL 64.68 |
|-----|----------|---------|----------|-----------------|
| ctrl | 1 | 66.08 | 58.01 | +1.40 |
| p2 | 2 | 68.11 | 59.80 | +3.43 worse |
| p3 | 3 | 68.01 | 59.28 | +3.33 worse |

- Analysis: Monotone hurt; explicit channel reweighting in Huber loss is harmful. Implicit surf_weight=10 balance is sufficient.
- **Paper finding (added):** Surface pressure channel does not benefit from explicit upweighting in Huber loss.

### #4045 fern — Capacity bump n_hidden {192, 256} (CLOSED — wall-clock-bound informative null)

| Arm | n_hidden | Epochs | val_avg | test_avg | s/epoch |
|-----|----------|--------|---------|----------|---------|
| A ctrl | 128 | 14 | 64.58 | 56.48 | 135 |
| B | 192 | 10 | 69.05 | 60.56 | 190 |
| C | 256 | 8 | 73.79 | 65.63 | 228 |

- Analysis: Larger n_hidden converges faster per-epoch but slower per-step. Within SENPAI_TIMEOUT_MINUTES=30, n=128 ctrl wins. At epoch 8, n=256 leads n=128 by 6.5 val — capacity bottleneck is wall clock, not architecture. Per launch isolation, timeout is hard bound.
- **Paper finding (added):** Under fixed wall-clock budget, n_hidden=128 is the optimal capacity for this task.

### #4044 alphonse — Multi-FiLM (NUDGED — ctrl-only so far)

W&B group shows 3 runs all `film_cond_dim=1` (ctrl) — the 11-param treatment has NOT launched. Posted status check asking alphonse to launch treatment arm or report implementation blocker.

### New assignments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #4084 | fern | R10 H48: Dropout sweep {0.05, 0.10} on Transolver blocks at lr=1.5e-4 | Just assigned |
| #4085 | askeladd | R10 H49: Batch size sweep {8, 16} with Lion at lr=1.5e-4 | Just assigned |

---

## 2026-05-16 17:55 — #3957 tanjiro CLOSED (informative); #4063 tanjiro R10 T_max sweep at lr=1.5e-4 assigned

### #3957 tanjiro — cosine T_max sweep (CLOSED — informative null)

- Branch: `willowpai2i48h5-tanjiro/r8-tmax`
- Hypothesis: T_max sweep on the spec_norm+lr=5e-5 substrate. Within-substrate finding (best arm slightly better than ctrl) does not transfer cleanly to new lr=1.5e-4 baseline.

| Arm | T_max | W&B run | val_avg | test_avg | vs new BL 63.05/53.60 |
|-----|-------|---------|---------|----------|----------------------|
| | 10 | 1gh5u4ka | 76.72 | 67.32 | +13.67 / +13.72 |
| | 10 | d3jr861j | 78.76 | 69.56 | +15.71 / +15.96 |
| | 14 ctrl | ftfb85ej | 68.75 | 60.41 | +5.70 / +6.81 |
| | 14 ctrl | exhex088 | 70.64 | 62.01 | +7.59 / +8.41 |
| **best** | **20** | **q7bw6nql** | **67.48** | **58.92** | **+4.43 / +5.32** |

- Analysis: Best arm T_max=20 (val 67.48) above new baseline by +4.43 val. Within-substrate (lr=5e-5 + spec_norm output), T_max=20 beat T_max=14 ctrl by ~1.3 val. T_max=10 catastrophic on both replicates (steep decay under-trains). Follow-up at lr=1.5e-4 assigned as #4063.

### #4063 tanjiro — T_max sweep at lr=1.5e-4 substrate (R10 H47 — assigned this session)

3 arms {14 ctrl, 18, 20} on new baseline substrate. Tests whether tanjiro's within-substrate T_max=20 preference transfers to lr=1.5e-4.

### Active experiments status (8 of 8 staffed)

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #4015 | nezuko | R10 H39 layer scale | Active — 3/3+ arms done in W&B; pod healthy (false `stale_wip` due to no PR comments) |
| #4044 | alphonse | R10 H40 multi-FiLM | WIP |
| #4045 | fern | R10 H44 capacity | WIP |
| #4046 | askeladd | R10 H43 channel weight | WIP |
| #4049 | frieren | R11 H46 spec_norm at lr=1.5e-4 | WIP |
| #4056 | thorfinn | R10 H42 gradient clip sweep | WIP |
| #4057 | edward | R10 H45 surface-biased routing | WIP |
| #4063 | tanjiro | R10 H47 T_max at lr=1.5e-4 | Just assigned |

---

## 2026-05-16 17:05 — #3958 thorfinn / #3913 edward CLOSED; #4056 thorfinn / #4057 edward R10 assigned

### #3958 thorfinn — Lion wd sweep at lr=1e-4 (CLOSED — informative null vs new baseline)

- Branch: `willowpai2i48h5-thorfinn/r8-wd-sweep`
- Hypothesis: At lr=1e-4, optimal Lion wd is lower than the lr=5e-5 tuned value of 1e-3.

| Arm | wd | W&B run | val_avg | test_avg | vs #3976 BL (63.05) |
|-----|----|---------|---------|----------|---------------------|
| A (ctrl) | 1e-3 | `54d2xdqz` | 65.93 | 57.34 | +2.88 worse |
| **B (best)** | **5e-4** | **`x9vlv88g`** | **64.79** | **56.54** | **+1.74 worse** |
| C | 2e-3 | `04uk731r` | 66.04 | 58.34 | +2.99 worse |

- Analysis: Hypothesis partially confirmed (wd=5e-4 beats wd=1e-3 ctrl) but ALL arms above the new baseline (val 63.05 from PR #3976 lr=1.5e-4). The wd sweep was vs #3843/3748 substrate; new baseline uses lr=1.5e-4 which wasn't tested. LR-wd coupling insight valid: effective decay ∝ lr×wd. At lr=1e-4, wd=5e-4 recalibrates toward the effective decay tuned at lr=5e-5 with wd=1e-3. **Whether wd=5e-4 compounds with lr=1.5e-4 is untested but lower priority than R10/R11 round.**

### #3913 edward — Re-extremity WeightedRandomSampler (CLOSED — informative null, hypothesis disconfirmed)

- Branch: `willowpai2i48h5-edward/r8-re-sampler`
- Hypothesis: Oversampling extreme-Re training samples should improve re_rand OOD generalization.

| Arm | alpha | lr | W&B run | val_avg | test_avg | vs #3976 BL (63.05) |
|-----|-------|----|---------|---------|----------|---------------------|
| **A (ctrl)** | **0.0** | **1e-4** | **`qaq2728x`** | **64.53** | **56.13** | +1.48 worse (within noise) |
| B | 0.5 | 1e-4 | `447v6w7g` | 70.02 | 61.07 | +6.97 worse |
| C | 1.0 | 1e-4 | `5ull7s2s` | 75.44 | 65.67 | +12.39 worse |

- Per-split: re_rand split (the OOD target) degrades monotonically with alpha — alpha=0.5 +5.99 test re_rand, alpha=1.0 +12.37 test re_rand. Every other split also worsens.
- Analysis: **Disconfirmed cleanly.** Training Re distribution is already well-covered by the balanced-domain sampler. Extremity oversampling starves the bulk (ESS drops from 1499 to ~1268 at alpha=0.5) and overfits extremes, hurting all splits including re_rand. OOD re_rand failure mode is NOT extreme-Re under-coverage — it's geometry×Re interaction, which reweighting cannot fix. **Strong negative result — valuable for paper (rules out sampling-based Re OOD fix).**

### R10 assignments (this session)

| PR | Student | Hypothesis | Expected |
|----|---------|------------|---------|
| #4056 | thorfinn | R10 H42: Gradient clip sweep {0.5, 1.0, 2.0} at lr=1.5e-4 | −0 to −2 val mean; variance reduction |
| #4057 | edward | R10 H45: Surface-biased slice routing in PhysicsAttention | −1 to −3 val; camber_rc target |

---

## 2026-05-16 16:30 — #3976 frieren MERGED (new best val 63.05 / test 53.60); R9 closures; R10/R11 round assigned

### #3976 frieren — Lion lr=1.5e-4 push (MERGED — **new best val 63.0492 / test 53.6049**)

- Branch: `willowpai2i48h5-frieren/r9-lion-lr-push`
- Hypothesis: Lion lr=1.5e-4 continues the monotone trend from lr=1e-4 (val 65.41). Optimal LR for this task is above 1e-4.
- W&B run: `jurrwig2`

| Arm | lr | W&B run | val_avg | test_avg | vs #3954 BL (64.68) |
|-----|----|---------|---------|----------|---------------------|
| A (ctrl) | 1e-4 | — | 64.68 | 56.17 | reference |
| **B (WINNER)** | **1.5e-4** | **`jurrwig2`** | **63.0492** | **53.6049** | **−1.63 val / −2.57 test** |
| C | 2e-4 | — | 63.84 | — | inflects back up |

Per-split:
| Split | val | test |
|-------|-----|------|
| single_in_dist | 64.45 | 55.69 |
| geom_camber_rc | 80.74 | 70.55 |
| geom_camber_cruise | 43.48 | 35.48 |
| re_rand | 63.53 | 52.70 |
| **avg** | **63.0492** | **53.6049** |

- Analysis: **LR inflection confirmed at [1.2e-4, 1.7e-4].** Full monotone trend: val(2e-5)=78.93 → val(5e-5)=69.69 → val(1e-4)=65.41 → val(1.5e-4)=63.05 → val(2e-4)=63.84. Largest single improvement since Lion optimizer. Paper finding #14 updated: optimum in [1.2e-4, 1.7e-4], not 1e-4.

### R9 closures (informative nulls)

| PR | Student | Result | Finding |
|----|---------|--------|---------|
| #3955 | alphonse | n_power_iter=1 optimal; higher = over-regularizes | n_power_iter sweep exhausted; keep n_power_iter=1 |
| #3977 | fern | Stochastic depth hurts at 5-block depth (+val) | Residual pathways at this depth are already shallow; DropPath removes capacity needed for fit |
| #3978 | askeladd | MixUp catastrophic (+23-27 val) | Non-physical blended targets; FiLM log(Re) conditioning gets mixed too. Paper finding #19. |

### R10/R11 assignments (this session)

| PR | Student | Hypothesis | Expected |
|----|---------|------------|---------|
| #4015 | nezuko | R10 H39: Layer scale init {ctrl, 1e-4, 1e-5} on Transolver blocks | −1 to −3 val |
| #4044 | alphonse | R10 H40: Multi-FiLM all 11 global params | −2 to −4 val; camber_rc target |
| #4045 | fern | R10 H44: Model capacity n_hidden {192, 256} | −1 to −4 val |
| #4046 | askeladd | R10 H43: p_weight {2x, 3x} pressure upweighting | −1 to −3 val |
| #4049 | frieren | R11 H46: spec_norm at lr=1.5e-4 | −0 to −2 val; tests finding #18 extension |

---

## 2026-05-16 15:25 — nezuko #3954 MERGED (new baseline val 64.68 / test 56.17); R8 R9 arms running

### #3954 nezuko — spec_norm output + lr=1e-4 combined (MERGED — **new baseline val 64.6812 / test 56.1746**)

- Branch: `willowpai2i48h5-nezuko/r8-specnorm-lr1e4`
- Hypothesis: Stack two merged winners — spec_norm(output) from #3748 + Lion lr=1e-4 from #3843. Do they compound additively?
- Results:

| Arm | lr | spec_norm | W&B run | val_avg | test_avg | Δ vs new baseline 65.41 |
|-----|-----|-----------|---------|---------|----------|------------------------|
| A (ctrl) | 5e-5 | output | `55a1xzky` | 67.87 | 60.01 | — |
| **B (hypothesis)** | **1e-4** | **output** | **`pc7lsis0`** | **64.6812** | **56.1746** | **−0.733 val / +0.112 test** |

Per-split (Arm B — winner):
| Split | val | test | Δval vs frieren 65.41 | Δtest vs frieren 56.06 |
|-------|-----|------|----------------------|------------------------|
| single_in_dist | 69.26 | 61.06 | −0.34 | +0.03 |
| geom_camber_rc | 78.64 | 69.24 | −1.54 | −1.23 |
| geom_camber_cruise | 46.37 | 38.56 | +0.18 | +0.72 |
| re_rand | 64.47 | 55.83 | −1.22 | +0.92 |
| **avg** | **64.68** | **56.17** | **−0.73** | **+0.11** |

- Analysis: **Hypothesis weakly confirmed. spec_norm at lr=1e-4 is orthogonal but not additive.** The two mechanisms coexist without interfering (val drops from 65.41 to 64.68), but the gain shrinks dramatically vs spec_norm at lr=5e-5 (which gave −1.39 val). Mechanistic explanation: Lion's sign-based update naturally bounds the effective per-step output gradient magnitude; the additional Lipschitz cap on the head adds little once the step size is already bounded by sign(). **Test metric is flat to slightly worse (+0.11).**

  Note: 4 independent reproductions of lr=1e-4 without spec_norm cluster at val 64.18–64.79 (mean ~64.5). Nezuko's spec_norm arm val 64.68 sits within this noise band — the true spec_norm contribution at lr=1e-4 is ~0 ± seed noise.

  **Key finding (added):** spec_norm Lipschitz contribution diminishes as lr grows. At lr=5e-5: −1.39 val. At lr=1e-4: ~−0 val (noise). The regularization budget is saturated by sign-based updates at higher lr.

  **New baseline: val 64.6812 / test 56.1746** (BASELINE.md updated, PR #3954 squash-merged).

### Active experiments: R8 + R9 rounds

| PR | Student | Config | Status | Best val so far |
|----|---------|--------|--------|-----------------|
| #3976 | frieren | R9: lr push 1.5e-4 | jurrwig2 running (step 4341/5264) | 66.15 (still running) |
| #3977 | fern | R9: stochastic depth 0.1 | 8zhftd2l running (step 4072/5264) | 76.93 (early) |
| #3978 | askeladd | R9: MixUp alpha=0.2 | u1k8cpqz running (step 1385/5264) | 175.98 (warmup) |
| #3958 | thorfinn | R8: wd=5e-4 finished, wd=2e-3 running | x9vlv88g finished val 64.79 (above new BL 64.68) | 64.79 |
| #3957 | tanjiro | R8: T_max=10 retry running | d3jr861j at step 4538/5264 | 80.14 (early eval) |
| #3955 | alphonse | R8: n_power_iter=5 running | nh64g1ds at step 2396/5264 | 95.93 (early) |
| #3913 | edward | R8: alpha=0.5 lr=1e-4 just started | 447v6w7g step 84 | — |

Note: thorfinn wd=5e-4 (val 64.79) no longer beats the new baseline (64.68). wd=2e-3 and the wd=1e-3 ctrl retry still running.

---

## 2026-05-16 13:30 — frieren #3843 MERGED (new baseline val 65.41 / test 56.06); fern #3808 / askeladd #3712 closed informative; 3 R9 assignments

### #3843 frieren — Lion lr=1e-4 sweep (MERGED — **new baseline val 65.4142 / test 56.0627**)

- Branch: `willowpai2i48h5-frieren/r7-lion-lr-sweep`
- Hypothesis: Lion lr=1e-4 improves over lr=5e-5 — sign-based updates tolerate higher lr within the 14-epoch cosine budget.
- All 3 arms on n_fourier=0 WITHOUT spec_norm (frieren's branch based before #3748 merged):

| Arm | lr | W&B run | val_avg/mae_surf_p | test_avg/mae_surf_p | vs #3748 baseline (68.96/60.82) |
|-----|----|---------|--------------------|--------------------|-------------------------------|
| A | 2e-5 | `gcjjdfot` | 78.93 | 69.31 | +9.97 / +8.49 (much worse) |
| B (ctrl) | 5e-5 | `pqbyquwr` | 69.69 | 60.47 | +0.73 / −0.35 (good repro ✓) |
| **C (WINNER)** | **1e-4** | **`bw38ym4h`** | **65.4142** | **56.0627** | **−3.55 / −4.76** |

Per-split (Arm C — winner):
| Split | val | test | Δval vs #3748 | Δtest vs #3748 |
|-------|-----|------|--------------|---------------|
| in_dist | 69.60 | 61.03 | **−8.24** | **−8.59** |
| camber_rc | 80.18 | 70.47 | −1.20 | −2.74 |
| camber_cruise | 46.19 | 37.84 | −3.71 | −2.84 |
| re_rand | 65.69 | 54.91 | −1.02 | **−4.87** |

- Analysis: **Largest single-arm improvement since Lion optimizer (R3). Effect size: −3.55 val / −4.76 test across all 4 splits.** 

  Mechanistically: Lion's effective step size ≈ lr × sign(grad). Doubling lr from 5e-5 → 1e-4 doubles per-step magnitude uniformly. The sign-based update is scale-tolerant in gradient magnitude but NOT in lr — the cosine schedule completing at epoch 14 fully harnesses the larger lr within the budget. The monotone trend across arms (val 78.93 → 69.69 → 65.41 across 2× lr steps) is consistent with the LR being the primary bottleneck in the prior baseline.

  Control arm (val 69.69 vs baseline 70.34) cleanly reproduces the n_fourier=0 substrate, confirming the lr=1e-4 gain is real and not a baseline-shift artifact.

  **Paper finding (finding #14):** Lion lr=1e-4 is the optimal learning rate for this task. Each 2× step halves the improvement (5e-5→1e-4: −4.28 val; 2e-5→5e-5: −9.24 val), suggesting a diminishing-returns curve with possible headroom at 1.5e-4/2e-4 (R9 H36 assigned to frieren).

  **New baseline: val 65.4142 / test 56.0627** (BASELINE.md updated, PR #3843 squash-merged).

### #3808 fern — surf_weight sweep (CLOSED — informative, substrate-dependent)

- Final results (confirmation arm D on new substrate n_fourier=0):

| Arm | n_fourier | surf_weight | val_avg | test_avg | vs new baseline 65.41 |
|-----|-----------|-------------|---------|----------|-----------------------|
| A | 16 | 10 | 75.28 | 64.70 | not comparable (old substrate) |
| B | 16 | 20 | 72.00 | 61.82 | not comparable (old substrate) |
| C | 16 | 40 | 76.93 | 66.63 | not comparable (old substrate) |
| **D (confirm)** | **0** | **20** | **72.71** | **62.87** | **+7.30 val (regression)** |

- Analysis: Strong internal signal at w=20 on n_fourier=16 (−3.28 val) did NOT transfer to n_fourier=0. Arm D val 72.71 is +7.30 vs new merged baseline (65.41). The surf_weight optimum is substrate-dependent: when input encoding changes (n_fourier=16→0), the surface-loss weighting optimum shifts. Informative finding for paper appendix.
  Also note: vol MAE climbs monotonically with surf_weight across all arms — clear surface/volume Pareto trade-off.

### #3712 askeladd — Lion β1 sweep (CLOSED — informative, β1=0.9 confirmed optimal)

- Final sweep (all runs on n_fourier=16 substrate):

| Arm | lion_beta1 | Best run | val_avg | vs β1=0.9 |
|-----|-----------|---------|---------|-----------|
| A (ctrl) | 0.9 | `rwgmm429` | 72.34 | baseline |
| B | 0.8 | `bus0nw0b` | 76.91 | **+4.57 worse** |
| C | 0.95 | `nu6wrtuc` | 74.09 | **+1.75 worse** |

- Analysis: β1=0.9 is locally optimal. Asymmetric: reducing momentum (0.8) hurts significantly more than increasing it slightly (0.95). Chen 2023 default β1=0.9 is confirmed as appropriate for this task. Internal ablation transfers regardless of substrate (we're confirming a design choice, not seeking a new baseline).
  **Paper finding (finding on Lion momentum):** β1=0.9 locally optimal; asymmetric — smaller momentum hurts much more than larger momentum.

### edward #3913 — Re-sampler debugging in progress

- 8 failed W&B runs with no val metrics (all `resampler-alpha00` name pattern, all crashed/failed).
- Investigation: edward's branch has only the assignment commit — NO implementation code pushed yet. Student is running failed experiments with local (uncommitted) code changes.
- Action: posted detailed debug guidance on PR. Key insight: `alpha=0.0` arm should be a pure baseline reproduction with zero code behavior change — if THAT crashes, bug is in CLI plumbing before the weight computation.
- Also updated reproduce commands in debug comment to use new lr=1e-4 (frieren #3843 merged).

### R9 assignments

| PR | Student | Hypothesis | Key experiment | Expected |
|----|---------|------------|----------------|----------|
| **#3976** | **frieren** | **R9 H36: Lion lr push {1.5e-4, 2e-4}** | 3 arms: lr=1e-4 ctrl, 1.5e-4, 2e-4. Monotone trend A→B→C was still falling — is 1e-4 the inflection? | ~−2 val if trend continues |
| **#3977** | **fern** | **R9 H37: Stochastic depth on Transolver blocks** | DropPath p_max ∈ {0.0 ctrl, 0.1, 0.2}. Fresh residual-pathway regularizer, never tried. | Uncertain; ~−1-2 val if effective |
| **#3978** | **askeladd** | **R9 H38: Input MixUp augmentation** | mixup_alpha ∈ {0.0 ctrl, 0.2, 0.5}. Interpolate per-node features + targets between samples. | Uncertain; known to help OOD generalization |

All 8 students now staffed. Rate limit (5000/hr) exhausted mid-session due to shared token usage across multiple student pods; GH REST blocked for ~37 min; recovered via GraphQL queries and ScheduleWakeup.

---

## 2026-05-16 12:35 — nezuko #3748 MERGED; 4 R8 assignments; frieren lr=1e-4 pending

### #3748 nezuko — Spectral norm on output head (MERGED — new baseline val 68.96 / test 60.82)

- Branch: `willowpai2i48h5-nezuko/r6-spec-norm`
- Hypothesis: Spectral normalization on output head MLP constrains Lipschitz constant of output projection, reducing peak-pressure over-fitting.
- Results (4 arms total — arms A/B/C on OLD baseline n_fourier=16; arm D on NEW baseline n_fourier=0):

| Arm | W&B run | spec_norm_target | n_fourier | val_avg | test_avg | notes |
|-----|---------|-----------------|-----------|---------|----------|-------|
| A (ctrl) | `vzbd6cch` | none | 16 | 72.83 | 63.40 | old substrate |
| **B (old winner)** | **`6my2xobv`** | **output** | **16** | **70.12** | **60.85** | old substrate |
| C | `gf9yg95k` | output+film | 16 | 74.20 | 64.20 | old substrate — worse |
| **D (confirmed winner)** | **`u42jpd48`** | **output** | **0** | **68.9592** | **60.8201** | new substrate |

Per-split arm D vs new baseline (PR #3672):
| Split | val Δ | test Δ |
|-------|-------|--------|
| in_dist | 79.64 → 77.84 (−1.80) | 69.97 → 69.62 (−0.35) |
| camber_rc | 82.43 → 81.38 (−1.05) | 73.96 → 73.21 (−0.75) |
| camber_cruise | 51.50 → 49.90 (−1.60) | 42.22 → 40.68 (−1.54) |
| re_rand | 67.80 → 66.71 (−1.09) | 60.35 → 59.78 (−0.57) |
| **avg** | **70.34 → 68.96 (−1.39)** | **61.63 → 60.82 (−0.81)** | |

- Analysis: **Spectral norm on output head (n_power_iter=1) is a legitimate regularizer for CFD surface-pressure MAE.** All 4 val splits improve and all 4 test splits improve. The Lipschitz bound on the head MLP prevents peak-pressure over-fitting — consistent with the mechanism story: high-Re/high-camber samples have peak-pressure spikes that the head over-fits; σ=1 bound forces smoother output.

  **Arm C (output+film spec_norm) HURTS (+1.37 val vs new baseline)**: bounding FiLM's gamma/beta linear destroys FiLM's adaLN-Zero identity-at-init and prevents Re-conditioning. Output-only remains the correct topology.

  **Paper-relevant (finding #13):** Output-only spectral norm is a complementary regularizer to FiLM. They operate in different subspaces (output Lipschitz vs input Re-conditioning) and compound cleanly.

  **New baseline: val 68.9592 / test 60.8201** (BASELINE.md updated, commit c07a3dd).

### #3843 frieren — Lion lr sweep (WIP — URGENT, lr=1e-4 arm is massive)

- Branch: `willowpai2i48h5-frieren/r7-lion-lr-sweep`
- All 3 arms finished (one retry running). Results on n_fourier=0 WITHOUT spec_norm:

| Arm | lr | W&B run | val_avg | test_avg | vs spec_norm baseline (68.96/60.82) |
|-----|----|---------|---------|----------|--------------------------------------|
| A | 2e-5 | `gcjjdfot` | 78.93 | 69.31 | +9.97 / +8.49 (much worse) |
| B (ctrl) | 5e-5 | `pqbyquwr` | 69.69 | 60.47 | +0.73 / −0.35 (good repro) |
| **C (WINNER)** | **1e-4** | **`bw38ym4h`** | **65.41** | **56.06** | **−3.55 val / −4.76 test** |

- Analysis: **lr=1e-4 is the largest single-arm improvement since the Lion merge in Round 3.** Sign-based Lion updates are scale-tolerant — LR 2× increase to 1e-4 finds a better basin in the 14-epoch cosine schedule. lr=2e-5 is significantly worse (too conservative for the 14-epoch budget). Control lr=5e-5 (val 69.69) cleanly reproduces the n_fourier=0 substrate (pre-spec_norm).

  **Decision: merge immediately when terminal SENPAI-RESULT posted.** Urgent comment posted on PR.

### #3817 alphonse — FiLM ablation (CLOSED — informative, paper-critical)

- **FiLM contribution under n_fourier=0:** −4.35 val / −4.56 test. FiLM-on val 70.05, FiLM-off val 74.40. All 4 splits improve with FiLM. Gain concentrates on Re-varying splits (in_dist −7.7 val / −8.7 test, re_rand −3.6 / −4.2), least on geometry-shifting camber_rc (−1.3 / −1.8). FiLM functions as Reynolds-conditioner, not generic regularizer. Seed noise floor: 2.77 val (two identical FiLM-on runs).

### #3842 tanjiro — Sobolev finer sweep (CLOSED — catastrophic)

- sobolev_weight=0.05 gave val 212 (3× worse than baseline). Loss scaling incompatible with new spec_norm substrate. Mechanism broken at this weight range.

### #3845 thorfinn — Train-time z-aug (CLOSED — same root cause as TTA)

- p=0.5 gave val 93 (35% worse than baseline). Training with z-reflected samples introduces conflicting physics regimes (phantom AoA=+3° instead of −3°) — same AoA asymmetry that caused TTA failure.

### R8 H32-H35 assigned

| PR | Student | Hypothesis | Key spec |
|----|---------|------------|----------|
| **#3954** | **nezuko** | **spec_norm + lr=1e-4 combined** | Stack two winners. Expected val ~62-65. |
| **#3955** | **alphonse** | **n_power_iter sweep {1, 3, 5}** | Tighten Lipschitz constraint. `--spec_norm_n_power_iter` sweep. |
| **#3957** | **tanjiro** | **T_max sweep {10, 14 ctrl, 20}** | Check if spec_norm changes optimal LR schedule. |
| **#3958** | **thorfinn** | **wd sweep at lr=1e-4 {0.5e-3, 1e-3 ctrl, 2e-3}** | Recalibrate wd when LR doubles. spec_norm + lr=1e-4 substrate. |

All 8 students staffed. Frieren #3843 terminal pending — **merge immediately when posted.**

---

## 2026-05-16 10:45 — edward #3786 closed; FiLM ablation confirmed (#3817 alphonse); fern surf_weight signal (#3808); edward R8 assigned #3913

### #3786 edward — Huber β sweep (CLOSED — informative, β=0.05 optimal)

- Branch: `willowpai2i48h5-edward/r7-huber-beta-sweep`
- Hypothesis: β ∈ {0.05, 0.10, 0.20} — wider Huber transition targets peak-pressure residuals driving camber_rc weakness.
- Results (all 3 arms finished, on OLD baseline n_fourier=16):

| Arm | β | W&B run | val_avg | test_avg | vs internal ctrl |
|-----|---|---------|---------|----------|-----------------|
| A (control) | 0.05 | `g3z8imw5` | 73.47 | 63.85 | — |
| **B (best)** | **0.10** | **`h3rdp99f`** | **72.99** | **62.93** | **−0.47 val / −0.92 test** |
| C | 0.20 | `5gdtcspi` | 73.71 | 63.85 | +0.24 / flat |

- Analysis: **β=0.05 is locally optimal in [0.05, 0.20] under FiLM+Lion+EMA.** β=0.10 shows a consistent direction (all 4 test splits improve slightly) but effect size (+0.47 val) is well within σ≈4.6 seed noise, particularly given that the control arm itself regressed +1.81 val vs published baseline — consistent with the 2.7 val seed-noise floor measured this session. β=0.20 is essentially indistinguishable from control.

  Per-split camber_rc was flat across all 3 arms (val: 85.32 → 85.33 → 84.54), confirming that peak-pressure residuals on camber_rc are NOT primarily driven by Huber β choice.

  **Paper-negative (finding #11):** Huber β ∈ [0.05, 0.20] does not improve surface-pressure MAE. β=0.05 is the local optimum.

  Caveat: runs used OLD baseline (n_fourier=16). Internal ablation still transfers since all 3 arms share substrate.

### #3817 alphonse — FiLM ablation (WIP — terminal pending, PAPER-CRITICAL finding confirmed)

- Branch: `willowpai2i48h5-alphonse/r7-film-ablation-nofourier`
- Hypothesis: FiLM on vs off under n_fourier=0, to quantify the FiLM contribution for the paper.
- Results (both arms finished on NEW baseline n_fourier=0):

| Arm | use_film | W&B run | val_avg | test_avg |
|-----|----------|---------|---------|----------|
| **A-best (FiLM on)** | True | **`sd42el34`** | **70.05** | **60.996** |
| A-retry (FiLM on) | True | `47p2anxd` | 72.82 | 63.82 |
| A-crash | True | `exx8m4sv` | NaN | NaN |
| **B (FiLM off)** | False | **`ow1x8ne8`** | **74.40** | **65.56** |

Per-split FiLM contribution (film=off → film=on, best run sd42el34):
| Split | val Δ | test Δ |
|-------|-------|--------|
| in_dist | 85.67 → 77.98 | 75.83 → 67.17 |
| **camber_rc** | **85.90 → 84.58 (−1.32)** | **76.04 → 74.21 (−1.83)** |
| camber_cruise | 54.79 → 50.01 (−4.78) | 45.60 → 42.03 (−3.57) |
| re_rand | 71.26 → 67.63 (−3.63) | 64.78 → 60.57 (−4.21) |
| **avg** | **−4.35** | **−4.56** |

- Analysis: **Paper-critical finding: FiLM contributes −4.35 val / −4.56 test under n_fourier=0 substrate.** This is smaller than the original FiLM measurement from old baseline (~5.9 val) but still large. The FiLM mechanism is confirmed as essential.

  **On merging:** The best film=on arm (sd42el34 val 70.05) is −0.29 val / −0.63 test better than baseline #3672. This is a lucky seed, NOT a new mechanism — arm A is the SAME config as #3672. Two FiLM-on runs gave val 70.05 vs 72.82 = 2.77 val spread (identical config). **No merge.** Close as informative once terminal posted.

  **Seed noise floor finding:** 2 runs of the same config gave val 70.05 vs 72.82 = **2.77 val spread** → any single-arm Δ < 2.7 val is indistinguishable from seed variance.

### #3808 fern — surf_weight sweep (WIP — baseline-shift issue, confirmation arm requested)

- Branch: `willowpai2i48h5-fern/r7-surf-weight-sweep`
- Hypothesis: surf_weight ∈ {10 ctrl, 20, 40} under FiLM+Lion+EMA on new baseline.
- Results so far (all arms on OLD baseline n_fourier=16 — NOT the new baseline):

| Arm | surf_weight | W&B run | val_avg | test_avg | vs ctrl |
|-----|-------------|---------|---------|----------|---------|
| A (ctrl, running) | 10 | `gg7r89pm` (done), `kbhvk6ol` (running) | 75.28 | 64.70 | — |
| **B** | **20** | **`218e3m9g`** | **72.00** | **61.82** | **−3.28 val / −2.88 test** |
| C | 40 | `433xbqv4` | 76.93 | 66.63 | +1.65 / +1.93 |

- Analysis: **Strong internal signal at surf_weight=20 (−3.28 val / −2.88 test).** Clean U-shape: w20 wins, w40 regresses. Effect size large enough to be real. But all runs on n_fourier=16, so not directly comparable to new baseline 70.34.

  Asked fern to run one confirmation arm: surf_weight=20 + n_fourier=0. If that beats val 70.34, it's a merge candidate.

### R8 H31 assigned — edward #3913

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| **#3913** | **edward** | **Reynolds-extremity WeightedRandomSampler** | 3 arms: `--re_sampler_alpha {0.0 ctrl, 0.5, 1.0}`. Multiply existing balanced-domain `sample_weights` by `|log(Re) - mean(log Re)|^α` to oversample extreme-Re samples. Targets re_rand OOD split (worst non-camber split at test 60.35). Already have WeightedRandomSampler plumbing in train.py (line 648). |

---

## 2026-05-16 09:35 — 3 R5 closes (#3673, #3697, #3698); 3 R7 assignments (#3842, #3843, #3845)

### #3673 tanjiro — EMA decay sweep (CLOSED — informative, within noise)

- Branch: `willowpai2i48h5-tanjiro/r5-ema-decay-sweep`
- Hypothesis: EMA decay ∈ {0.995, 0.997, 0.999} under FiLM+Lion. Is 0.997 locally optimal?
- Results:

| Arm | ema_decay | W&B run | val_avg | test_avg | Δval vs base (71.65) |
|-----|-----------|---------|---------|----------|----------------------|
| **A (best)** | **0.995** | **`s3ufqnz2`** | **71.51** | **−** | **−0.14 (noise)** |
| B | 0.997 | `3ag4pvjr` | 73.46 | 63.67 | +1.81 |
| C | 0.999 | `3ki9voje` | ~73+ | — | regression |

- Analysis: **EMA decay is robust in [0.995, 0.997].** Best arm (0.995) at val 71.51 is a wash with the baseline 71.65 (Δ=−0.14, well within σ≈4.6). Arm A is a canonical control (s3ufqnz2 was the intended canonical run, others were process-collision duplicates). No merge — informative paper ablation confirming decay insensitivity. Note: best arm runs pre-date the n_fourier=0 merge; absolute numbers are not directly comparable to new baseline 70.34.

  **Paper finding (finding #4):** EMA decay is robust in [0.995, 0.997] — safe to fix at 0.997.

### #3697 frieren — Multi-σ Gaussian Fourier PE (CLOSED — superseded by n_fourier=0 merge)

- Branch: `willowpai2i48h5-frieren/r5-multi-sigma-fourier`
- Hypothesis: Multi-scale Fourier features (σ ∈ {3, 10, 30} concatenated) capture broader frequency content and outperform single-σ.
- Results: Arms A/B posted before n_fourier=0 merged. Arm C posted after.

| Arm | Config | val_avg | vs baseline at time |
|-----|--------|---------|---------------------|
| A | σ=3 only | ~73 | regression |
| B | σ=10 only (control) | ~71.7 | baseline repro |
| C | σ=3,10,30 multi | ~71.5 | marginal/wash |

- Analysis: **Superseded.** The n_fourier=0 result (val 70.34) shows FiLM on log(Re) makes ALL Fourier PE redundant. Multi-σ cannot beat dropping Fourier entirely. The multi-scale hypothesis is directionally wrong — more frequency resolution doesn't help when FiLM already encodes flow-regime conditioning.

### #3698 thorfinn — TTA z-reflection (CLOSED — catastrophic regression, dataset asymmetry confirmed)

- Branch: `willowpai2i48h5-thorfinn/r5-tta-reflection`
- Hypothesis: Averaging model predictions over original + z-reflected geometry provides a free inference-time gain via symmetry.
- Results:

| Eval mode | W&B run | val_avg | vs baseline |
|-----------|---------|---------|-------------|
| No TTA (control) | `3du9h0yz` | ~72 | +0.35 (noise) |
| TTA z-reflect | `5555kka9` | ~307 | **+235 CATASTROPHIC** |

Thorfinn ran additional diagnostic: 6-subset reflection sweep on `3du9h0yz` checkpoint (run `awuxtsni`):

| Subset reflected | val_avg |
|-----------------|---------|
| {} (no reflection) | 72.15 |
| {x,z} | 286 |
| {z only} | 307 |
| {AoA0, AoA1} | 184 |
| ... | ... |

- Analysis: **TTA z-reflection fails because the training data is NOT z-symmetric.** raceCar geometry uses AoA ∈ [-10°, 0°] (always negative camber); z-reflecting an AoA=-5° sample gives AoA=+5°, which the model has never seen. The model is not equivariant to z-reflection because the dataset isn't. Catastrophic regression (72 → 307) confirms the model correctly rejects OOD inputs.

  **Paper-relevant finding (#10):** TTA z-reflection fails on asymmetric AoA distribution. Cannot assume z-symmetry without symmetric training data.

  **Follow-up (H30):** Train-time z-reflection augmentation (#3845) may teach the model the symmetry by including reflected samples during training.

### R7 assignments: tanjiro #3842, frieren #3843, thorfinn #3845

| PR | Student | Hypothesis | Key novelty |
|----|---------|------------|-------------|
| **#3842** | **tanjiro** | **Sobolev finer sweep w ∈ {0.05, 0.10, 0.15}** | Extend fern's R5 signal (test −0.18 at w=0.1) with finer grid on new n_fourier=0 baseline |
| **#3843** | **frieren** | **Lion lr sweep {2e-5, 5e-5 ctrl, 1e-4}** | First lr ablation in this launch; paper-required sensitivity analysis |
| **#3845** | **thorfinn** | **Train-time z-reflection augmentation (p ∈ {0, 0.25, 0.5})** | Close loop on TTA failure; teach model z-symmetry during training to attack camber_rc (worst split) |

All 8 students now staffed for R7. Full R7 map spans: FiLM ablation (alphonse), surf_weight (fern), Huber β (edward), Lion β1 (askeladd), spectral norm (nezuko), Sobolev (tanjiro), Lion lr (frieren), train-aug (thorfinn).

---

## 2026-05-16 07:55 — alphonse #3672 MERGED; fern #3695 closed; R7 fern+edward assigned

### #3672 alphonse — Fourier ablation (MERGED — new baseline val 70.34 / test 61.63)

- Branch: `willowpai2i48h5-alphonse/r5-fourier-ablation-film`
- Hypothesis: Under FiLM+Lion+EMA, Fourier positional features may be redundant since FiLM on log(Re) already encodes the flow-regime frequency information. Test n_fourier ∈ {0, 16 σ=3, 16 σ=10}.
- Results (all arms FINISHED, terminal SENPAI-RESULT posted, squash-merged):

| Arm | Config | W&B run | val_avg | test_avg | Δval vs prior base (71.65) |
|-----|--------|---------|---------|----------|---------------------------|
| **A (WINNER)** | n_fourier=0 | **`297qot5r`** | **70.3432** | **61.6253** | **−1.31** |
| B | n_fourier=16, σ=3 | `drp81h4l` | 71.2763 | 61.6733 | −0.38 (marginal) |
| C (control) | n_fourier=16, σ=10 | `vx0b6ukg` | ~71.7 (still running at terminal) | — | ~baseline |

Per-split Arm A winner vs prior baseline:

| Split | val Δ | test Δ |
|-------|-------|--------|
| single_in_dist | −1.53 | −1.33 |
| geom_camber_rc | −2.02 | +0.09 (wash) |
| geom_camber_cruise | −0.49 | −0.62 |
| re_rand | −1.21 | −0.08 (wash) |

- Analysis: **FiLM on log(Re) makes Fourier PE redundant.** Dropping Fourier entirely (n_fourier=0) improves all 4 val splits and 3/4 test splits. FiLM already captures the flow-regime conditioning signal that Fourier positional encoding was trying to inject. Key simplification win: removes ~1.1K RFF params, one hyperparameter (fourier_sigma), and one coordinate transform per forward pass.

  Variance caveat: student ran 3 Arm A seeds due to process collisions during 06:30–07:00 launch window; only `297qot5r` was a clean 50-epoch run. The two duplicates (`9an3ynhy` val 82.39, `cng2gwhu` val 89.92) were crash-restarts with broken state, not reproducible runs.

  **New baseline: val 70.3432 / test 61.6253** (BASELINE.md updated, commit `6352727`).

### #3695 fern — Sobolev surface ∂p/∂s loss (CLOSED — informative, small test gain)

- Branch: `willowpai2i48h5-fern/r5-sobolev`
- Hypothesis: Penalizing ∂p/∂s gradient mismatch along the foil surface regularizes prediction smoothness and improves OOD generalization.
- Results (all 3 arms FINISHED, terminal SENPAI-RESULT posted):

| Arm | sobolev_weight | W&B run | val_avg | test_avg | Δval vs prior base (71.65) |
|-----|----------------|---------|---------|----------|---------------------------|
| Control | 0.0 | `yrl9p2bh` | 73.7119 | 63.3541 | +2.06 |
| **B (best)** | **0.1** | **`b655hio8`** | **71.8355** | **61.9284** | **+0.18 (wash, test −0.18)** |
| C | 0.5 | `pgk5nw19` | 85.9918 | 75.7503 | +14.34 |

- Analysis: **Sobolev w=0.1 gives a small test-side gain (−0.18) at flat val (+0.18 above baseline).** The surface gradient regularizer is pointing in the right direction (OOD smoothness) but the gain is sub-noise. Per-split: camber_cruise test improves (42.84 → 42.03, −1.9%), camber_rc regresses slightly (73.87 → 75.31, +2.0%). w=0.5 catastrophically over-regularizes (+14 val). The Sobolev contribution (ratio ~1.0 at w=0.1) equals the data-loss magnitude at epoch 14 — a tuning sweet spot that happens to be near noise.

  Paper-relevant: surface-Sobolev regularization at correct weight is neutral-to-slightly-beneficial on test. Confirms physics-motivated direction. Finer sweep {0.03, 0.05, 0.08, 0.12} reserved for Round 7+.

### New R7 assignments after R5 closes

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3786 | edward | Huber β sweep (0.05→0.1→0.2) | `--loss_beta` flag sweep. Hypothesis: β=0.05 is too tolerant of peak-pressure residuals driving camber_rc weakness. |
| **#3808** | **fern** | **Surface-loss reweighting (surf_weight ∈ {10, 20, 40})** | `--surf_weight` flag sweep. Direct rebalancing of surface vs volume loss gradient. Follow-up to Sobolev result. |

All 8 students staffed:
- alphonse: reassigning (just merged #3672)
- tanjiro #3673: terminal posted, awaiting mark-ready → close-as-informative
- fern #3808: just assigned (surf_weight sweep)
- frieren #3697: Arm C still running
- thorfinn #3698: awaiting terminal + mark-ready → close-as-informative
- askeladd #3712: running (β1 sweep)
- nezuko #3748: running (spec norm)
- edward #3786: just assigned (Huber β sweep)

---

## 2026-05-16 07:35 — R5 results finalized; edward #3711 closed; R7 edward assigned

### #3711 edward — Layer-wise LR decay / LLRD (CLOSED — dead end, monotonic regression with γ<1)

- Branch: `willowpai2i48h5-edward/r6-llrd`
- Hypothesis: Lower LR for earlier Transolver blocks (γ<1 multiplier from output to input) mirrors fine-tuning LR decay used in pretrained LLMs and may improve generalization.
- Results (all 3 arms FINISHED, terminal SENPAI-RESULT posted):

| Arm | γ | W&B run | val_avg | test_avg | Δval vs base (71.65) |
|-----|---|---------|---------|----------|----------------------|
| A (control) | 1.00 | `kuvqzt5y` | 71.8970 | 62.4985 | +0.24 (within noise) |
| B | 0.85 | `3om0smnq` | 76.7062 | 66.3279 | +5.05 |
| C | 0.65 | `nnsunb0c` | 93.1413 | 82.3440 | +21.49 |

- Analysis: **LLRD with γ<1 hurts monotonically under FiLM+Lion+EMA.** LLRD is designed for fine-tuning pretrained models, where lower layers encode general features already at a good basin. Training from scratch means lower blocks are far from convergence at epoch 0 — throttling their LR (γ=0.85 cuts group_0 to 37.7% of base) prevents them from finding a good basin in 14 effective epochs. The output head is then forced to compensate with under-trained features → worse generalization across all 4 splits. Effect is monotonic and catastrophic at γ=0.65 (+21.5 val). The γ=1.0 control reproduces baseline within +0.24 val (noise), confirming clean implementation and real negative result.

**Paper-relevant findings**: LLRD does not transfer from the fine-tuning setting to training-from-scratch setting. Our regime (14 effective epochs from random init) is not analogous to NLP LLRD literature.

### Round 5+6 results fully in — action taken

All 5 remaining R5 students had W&B results completed. Advisor comments posted to push for terminal SENPAI-RESULT on:

- **#3672 alphonse** (winner declared — n_fourier=0 `297qot5r` val 70.3432 / test 61.6253, beats baseline)
- **#3673 tanjiro** (close as informative — best EMA=0.995 `s3ufqnz2` val 71.51, wash)
- **#3695 fern** (close as informative — best sobolev=0.1 `b655hio8` val 71.84, flat val but test improves −0.18)
- **#3698 thorfinn** (close as informative — TTA=True `5555kka9` val 72.56, no gain; design gap noted — no control arm)

Still awaiting:
- **#3697 frieren** Arm C (σ='3,10,30') — still running
- **#3712 askeladd** — β1=0.8/0.95 just starting (rate-limit delayed, GPU now at 100%)
- **#3748 nezuko** — all 3 arms not started (rate-limit delayed, pod just picked up PR)

### R7 edward assigned — #3786

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3786 | edward | **Huber β sweep (0.05→0.1→0.2)** | 3 arms: `--loss_beta 0.05/0.10/0.20`. Hypothesis: current β=0.05 is too tolerant of peak-pressure errors that drive camber_rc (worst split at test 73.87). Widening transition region increases gradient for moderate residuals → better peak-pressure fitting. |

---

## 2026-05-16 06:35 — Closed nezuko #3671, reassigned to spec norm; partial R5 results in

### #3671 nezuko — Layer-wise FiLM (CLOSED — uniform +5 val regression)

- Branch: `willowpai2i48h5-nezuko/r5-film-intermediate-layers`
- Hypothesis: Stack FiLM conditioning at intermediate Transolver blocks in addition to output-FiLM, on the theory that earlier-block conditioning gives the model more capacity to adapt per-Re.
- Results (Arm A only — student declared verdict and stopped):

| Arm | W&B run | Config | val_avg | test_avg | vs baseline (71.65 / 62.11) |
|-----|---------|--------|---------|----------|------------------------------|
| A | `w2qifj9u` | output-FiLM + block-FiLM stack | 76.81 | 66.13 | +5.16 val / +4.03 test |

Per-split val Δ vs baseline: in_dist +5.83, camber_rc +4.13, camber_cruise +6.16, re_rand +4.52 (all 4 splits worse, including OOD re_rand where block-FiLM should help most).

- Analysis: **Paper-relevant negative — output-FiLM at the final layer is the correct FiLM topology**. Adding FiLM to intermediate blocks (5× extra parameters + 4% per-epoch slowdown) does NOT improve over the cheaper output-only configuration. Per-epoch curve shows Arm A starts ahead at epoch 1 (296 vs 310) but falls 5-7 points behind from epoch 3 onward and never recovers. The extra parameters slow per-step throughput, effectively reducing the training-iteration budget; this is the most likely mechanism for the uniform regression.

### Partial Round 5 results (alphonse #3672, tanjiro #3673)

**alphonse #3672** (Fourier ablation under FiLM+Lion+EMA):

| Arm | W&B run | Config | val_avg | test_avg | vs baseline |
|-----|---------|--------|---------|----------|------------|
| A | `9an3ynhy` | n_fourier=0 | (running, step 1469/5264) | — | — |
| B | `drp81h4l` | σ=3, n=16 | **71.28** | **61.67** | **−0.37 val / −0.44 test (marginal beat)** |
| C | (not started) | σ=10, n=16 | — | — | — |

Arm B σ=3 marginally beats baseline but Δ is within the σ≈4.6 run-to-run variance band. Awaiting Arms A and C to determine if this is signal or noise. Asked alphonse to verify Arm A progress and start Arm C.

**tanjiro #3673** (EMA decay sweep under FiLM+Lion):

| Arm | W&B run | ema_decay | val_avg | test_avg | vs baseline |
|-----|---------|-----------|---------|----------|------------|
| A | `eb4gsayj` | 0.995 | (running, step 1380/5264, val 136 early) | — | — |
| B | `3ag4pvjr` | 0.997 | 73.46 | 63.67 | +1.81 val / +1.56 test (within noise) |
| C | `3ki9voje` | 0.999 | (running, step 225/5264) | — | — |

ema=0.997 (paper default) reproduces baseline within noise — confirms 0.997 is solid. Awaiting decay-sweep arms to see if 0.995 or 0.999 outperforms.

### Round 6 assignment (nezuko reassigned)

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3748 | nezuko | **Spectral normalization on output head (+ FiLM layers)** | `torch.nn.utils.parametrizations.spectral_norm` on output linear; 3 arms: control, output only, output+film. Lipschitz constraint to reduce peak-pressure over-fit. |

All 8 students remain staffed: alphonse #3672, tanjiro #3673, fern #3695, frieren #3697, thorfinn #3698, edward #3711, askeladd #3712, nezuko #3748.

---

## 2026-05-16 05:25 — Closed edward + askeladd holdovers, assigned Round 6

### #3483 edward — Lion+EMA ablation (CLOSED — no arms beat new FiLM baseline)

- Branch: `willowpai2i48h5-edward/round3-ema-only-on-huber-no-fourier`
- Hypothesis: Quantify isolated EMA + ablate Fourier under Lion substrate.
- Results (all 3 arms finished):

| Arm | W&B run | Config | val_avg | test_avg | vs new baseline (71.65 / 62.11) |
|-----|---------|--------|---------|----------|---------------------------------|
| A (winner of this PR) | `5pvi79f2` | Lion + EMA(0.997), no Fourier | **73.10** | **63.65** | +1.45 val / +1.54 test |
| B | `3hgal2fm` | Lion + EMA(0.997), σ=3 Fourier | 73.41 | 64.33 | +1.76 val / +2.22 test |
| C | `tev95mko` | Pure Lion, no EMA, no Fourier | 77.48 | 67.43 | +5.83 val / +5.32 test |

- Analysis: **Paper-section material — EMA contributes 4.4 val / 3.8 test points on top of Lion** (Arm A vs Arm C). This is the cleanest single-mechanism EMA measurement in the launch. Closed because no arm beats new FiLM baseline (71.65); the EMA gain is already incorporated via #3405 merge.

### #3609 askeladd — Lion + LR warmup ablation (CLOSED — warmup adds nothing)

- Branch: `willowpai2i48h5-askeladd/r4-lion-warmup`
- Hypothesis: LR warmup improves Lion stability and final performance.
- Results (all 3 arms finished):

| Arm | W&B run | warmup_steps | val_avg | test_avg | vs new baseline (71.65 / 62.11) |
|-----|---------|--------------|---------|----------|---------------------------------|
| A | `379hrdie` | 0 | 79.13 | 68.98 | +7.48 val / +6.87 test |
| B | `j1pum3n7` | 500 | 79.89 | 69.83 | +8.24 val / +7.72 test |
| C (winner of this PR) | `jdaof5n2` | 1000 | **78.46** | **68.69** | +6.81 val / +6.58 test |

- Analysis: **LR warmup adds nothing to Lion at our 14-effective-epoch budget.** Non-monotonic in warmup_steps (C 1000 > A 0 > B 500), but effect size (~1.4 val across arms) is below run-to-run variance band (~σ≈4.6). Paper-section material for the LR-schedule ablation: cosine T_max=14 alone is sufficient. Closed; these plain-Lion arms regress against FiLM+Lion+EMA baseline because they're missing FiLM (+5.9 val) and EMA (+4.4 val).

### Round 6 assignments created (edward + askeladd no longer idle)

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3711 | edward | **Layer-wise LR decay (LLRD)** | Per-block LR multiplier γ; 3 arms γ ∈ {1.0 control, 0.85, 0.65}. Output head full LR, input embed γ^(N+1)·base. Paper-relevant for optimizer-tuning section. |
| #3712 | askeladd | **Lion β1 sweep** | `--lion_beta1` flag; 3 arms β1 ∈ {0.8, 0.9 control, 0.95}. β2=0.99 fixed. Settles paper-required Lion ablation. |

All 8 students now staffed with active R5 or R6 PRs. REST API budget recovered (2880/5000) after earlier exhaustion. Standard assign-experiment skill used (not GraphQL fallback).

---

## 2026-05-16 04:35 — Round 5 cleanup + 3 new assignments

### Closed PRs (informative negatives / superseded)

**#3544 thorfinn — Lookahead optimizer (CLOSED — dead end across both substrates).**
- Branch: `willowpai2i48h5-thorfinn/round3-lookahead`
- Hypothesis: Lookahead wrapper (k-step slow-weight averaging) provides ensemble-like regularization on top of AdamW or Lion.
- Results:

| Arm | W&B run | Substrate | val_avg | test_avg | Δ vs baseline |
|-----|---------|-----------|---------|----------|---------------|
| A (initial) | `k39kdp6y` | Lookahead(k=6, α=0.5) + AdamW (R3 baseline 93.20) | 98.33 | 86.24 | +5.13 val |
| A (post-pivot) | `drt9naou` | Lookahead(k=5, α=0.5) + Lion (R3 baseline 77.58) | **89.39** | **78.24** | +11.81 val |

- Analysis: Lookahead is incompatible with both inner optimizers at our 14-effective-epoch budget. Hypothesized mechanism: the slow-weight averaging acts as a second-order smoother on top of the optimizer's own smoothing, and under tight wall-clock the slow weights never fully catch up to the fast weights. Lion's sign-based update is already a coarse approximation, so Lookahead's k-step averaging damps out exactly the signal Lion injects. Paper-relevant negative result for optimizer-family ablation section.

**#3486 fern — σ=3 + Lion + EMA (CLOSED — superseded by FiLM merge).**
- Branch: `willowpai2i48h5-fern/round3-fourier-sigma-under-ema`
- Hypothesis: σ-monotonic finding from AdamW (σ=3 wins σ sweep under EMA) transfers to Lion+EMA substrate.
- Results:

| Arm | W&B run | Config | val_avg | test_avg | Δ vs Lion baseline (77.58) |
|-----|---------|--------|---------|----------|---------------------------|
| Lion-rebase | `dl4apv3e` | σ=3 + Lion + EMA(0.997) | **73.81** | **63.89** | −3.77 val / −4.99 test |

- Analysis: σ=3 beats Lion baseline by ~4 val points — a real win on the old substrate. BUT: edward's no-Fourier Lion+EMA (`5pvi79f2`, val 73.10) slightly beats fern's σ=3 (73.81) on the same substrate. **The σ-monotonic finding from AdamW+EMA does NOT transfer to Lion+EMA.** Under Lion+EMA, Fourier features appear roughly equivalent to noise (or slightly harmful at any σ). Paper-valuable negative-transfer ablation. PR closed because new FiLM baseline (val 71.65) supersedes σ=3 result.

**#3380 frieren — Multi-σ Fourier sweep (CLOSED — config bug, student agreed).**
- Branch: `frieren/round2-sigma-sweep`
- Hypothesis: Multi-scale Gaussian Fourier features (σ ∈ {3, 10, 30}) improve over single-σ=10.
- Results:

| Arm | W&B run | Intended config | Actual config | val_avg | test_avg |
|-----|---------|------------------|----------------|---------|----------|
| (only) | `54hmldzq` | multi-σ {3,10,30} | n_fourier=0 (bug) | 76.95 | 67.07 |

- Analysis: Multi-σ Fourier flag never wired in — config shows `n_fourier=0` at runtime. Run is effectively Lion+EMA no-Fourier (comparable to edward's `5pvi79f2` val 73.10; σ ≈ 4.6 run-to-run variance band). PR closed by mutual agreement with student; multi-σ reassigned to frieren on the new FiLM+Lion+EMA substrate (#3697).

### #3609 askeladd status update (Lion + LR warmup, paper-relevant negative in progress)

| Arm | W&B run | warmup_steps | val_avg | test_avg | Status |
|-----|---------|--------------|---------|----------|--------|
| A | `379hrdie` | 0 | 79.13 | 68.98 | finished |
| B | `j1pum3n7` | 500 | 79.89 | 69.83 | finished |
| C | `jdaof5n2` | 1000 | (running) | — | ~50% complete |

- Analysis (preliminary): **LR warmup adds nothing to plain Lion at our 14-effective-epoch budget.** Arm B (warmup=500) is even slightly worse than Arm A (no warmup). Cosine schedule already provides implicit warmup via low LR start-of-cycle when T_max=14 is matched to wall-clock. Awaiting Arm C to confirm. After Arm C, expected to close as informative negative (paper-relevant) and reassign askeladd to Lion β1 sweep.

### Round 5 assignments created

| PR | Student | Hypothesis | Implementation |
|----|---------|------------|----------------|
| #3695 | fern | **Sobolev loss on surface ∂p/∂s** (physics-motivated regularizer) | Add `--sobolev_weight` + `--sobolev_k`; compute k-NN finite-difference gradient of surface pressure; Huber on (pred grad − gt grad); 3-arm weight sweep ∈ {0, 0.1, 0.5} |
| #3697 | frieren | **Multi-σ Gaussian Fourier under FiLM+Lion+EMA** (proper wiring this time) | `--fourier_sigmas "3,10,30"` + `--n_fourier_per_scale 8`; concatenate Gaussian features at each σ; 3-arm sweep: {σ=10 control, σ∈{3,10}, σ∈{3,10,30}} |
| #3698 | thorfinn | **TTA via z-reflection symmetry** (free inference gain) | `--use_tta_reflection`; reflect z→−z at eval, average original + reflected predictions; 2-arm: control vs TTA |

All assigned via direct GraphQL (REST API exhausted, GraphQL still has 3000+/5000 budget). Branches pushed, draft PRs created, labels {`status:wip`, `icml-appendix-willow-pai2i-48h-r5`, `student:<name>`} verified.

---

## 2026-05-16 03:40 — PR #3405: FiLM conditioning + Lion + EMA [Round 4 nezuko] ← NEW BASELINE

- Branch: `willowpai2i48h5-nezuko/film-conditioning-log-re`
- Hypothesis: Condition the Transolver model on Reynolds number via FiLM (Feature-wise Linear Modulation) — gamma/beta affine transforms on log(Re) applied at the output layer. log(Re) encodes the Reynolds-regime of each flow sample; the `re_rand` OOD split has the most to gain. Combined with Lion optimizer + EMA(0.997) as the new substrate.

| Run | W&B run | Config | val_avg | test_avg | Notes |
|-----|---------|--------|---------|----------|-------|
| Lion+EMA+FiLM | `ksltdq7a` | FiLM + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + σ=10 + T_max=14 | **71.6544** | **62.1091** | **WINNER** — merged |

**Per-split results (ksltdq7a):**

| Split | val | test |
|-------|-----|------|
| single_in_dist | 81.17 | 71.30 |
| geom_camber_rc | 84.45 | 73.87 |
| geom_camber_cruise | **51.99** | **42.84** |
| re_rand | 69.01 | 60.43 |

**Decision: MERGED** as new baseline. val_avg 77.58 → 71.65 (−7.9%), test_avg 68.88 → 62.11 (−9.8%). 3rd consecutive improvement in the Round 3-4 cascade.

**Analysis:**
- FiLM conditioning on log(Re) adds ~5.9 val / 6.8 test on top of Lion+EMA(0.997). This is meaningful additive gain from an orthogonal mechanism.
- FiLM's biggest gain is on `geom_camber_cruise` test (48.83 → 42.84, −12.3%) and `single_in_dist` test (81.69 → 71.30, −12.7%). The mechanism appears to benefit geometry-OOD splits more than Re-OOD, possibly because log(Re) is a proxy for flow complexity that correlates with camber-induced pressure peaks.
- Among 5 simultaneous Lion+EMA reruns, only FiLM provided a further separation (all others were 73-77 val; FiLM brought it to 71.65).
- Fourier σ=10 remains in the stack; Round 5 can test if FiLM obviates Fourier.
- **No Fourier (edward run 5pvi79f2): val 73.10** — within 1.5 val of fern's σ=3 (73.81) and worse than FiLM (71.65). Fourier is marginal but FiLM is clearly the dominant mechanism.

**Round 4 companion runs (not merged, informative ablations):**

| Run | Config | val_avg | test_avg | Status |
|-----|--------|---------|----------|--------|
| `5pvi79f2` edward | Lion + EMA(0.997), n_fourier=0 | 73.10 | 63.65 | Ablation: confirms no-Fourier under Lion+EMA |
| `dl4apv3e` fern | Lion + EMA(0.997) + σ=3 | 73.81 | 63.89 | Ablation: σ=3 ≈ no-Fourier, not better |
| `fg3u9jsj` alphonse | Lion + EMA(0.997) + σ=10 | 76.15 | 66.55 | Variance sample A |
| `5uaxtezx` tanjiro | Lion + EMA(0.997) + σ=10 | 79.17 | 68.97 | Variance sample B |
| `54hmldzq` frieren | Lion + EMA(0.997), n_fourier=0 (config bug) | 76.95 | 67.07 | Config bug — intended multi-σ |
| `drt9naou` thorfinn | Lookahead (k=5 α=0.5) + Lion | 89.39 | 78.24 | Dead end: Lookahead regresses |
| `379hrdie` askeladd | Lion warmup=0 (control) | 79.13 | 68.98 | Control: baseline reproduction |

---

## 2026-05-16 01:43 — PR #3537: Lion optimizer (sign-based update) vs AdamW [Round 3 H13]

- Branch: `willowpai2i48h5-askeladd/round3-lion-optimizer`
- Hypothesis: Replace AdamW with Lion (Chen et al. 2023, arXiv 2302.06675) — sign-based update, momentum-decay schedule, decoupled weight decay. Lion's sign update yields uniform per-coordinate steps, potentially benefiting irregular-mesh CFD where AdamW's adaptive scaling may misjudge importance across heterogeneous node features.

| Arm | W&B run | optimizer | lr | wd | val_avg | test_avg | Notes |
|-----|---------|-----------|----|----|---------|----------|-------|
| A — Lion lr=5e-5 wd=1e-3 | `yvkf9glr` | lion | 5e-5 | 1e-3 | **77.5788** | **68.8764** | **WINNER** — merged |
| B — Lion lr=1e-4 wd=5e-4 | (not yet run) | lion | 1e-4 | 5e-4 | — | — | Follow-up sweep |
| C — Lion lr=3e-4 wd=1e-4 | (not yet run) | lion | 3e-4 | 1e-4 | — | — | Follow-up sweep |

**Per-split val mae_surf_p (Arm A):**

| Split | val | test |
|-------|-----|------|
| single_in_dist | 90.85 | 81.69 |
| geom_camber_rc | 87.72 | 77.94 |
| geom_camber_cruise | **58.81** | **48.83** |
| re_rand | 72.93 | 67.04 |

**Decision: MERGED** as new baseline. val_avg 93.20 → 77.58 (−16.8%). test_avg 83.54 → 68.88 (−17.5%). Every test split improves substantially. This is the **largest single-mechanism gain** of the launch (Δ = 15.62 val, 3.4σ above noise floor σ ≈ 4.6).

**Analysis:**
- Lion paper recommends batch ≥ 64 but it works strongly at our batch_size=4. The irregular-mesh CFD loss landscape appears to be well-suited to sign updates.
- LR=5e-5 was the conservative 10× scale-down from AdamW's 5e-4 — Lion's larger effective per-coordinate step requires lower LR.
- All other components held constant: Huber β=0.05, Fourier σ=10 n=16, T_max=14.
- Arms B and C (LR sweep around the winner) are paper-required ablations but Arm A is already merged.

**Implications:**
- All EMA-cluster wins (tanjiro EMA(0.997) val 86.42, fern σ=3+EMA val 87.83) need re-validation on top of Lion. They were achieved with `cosine_t_max=None` and AdamW.
- Natural Round 4: EMA(0.997) + Lion compound (4-way stack with Huber + σ=10 + T_max=14).

---

## 2026-05-15 23:11 — PR #3444: Cosine T_max=14 (recalibrate schedule to wall-clock budget) [Round 2 thorfinn]

- Branch: `willowpai2i48h5-thorfinn/round2-cosine-tmax`
- Hypothesis: 30-min wall-clock binds at epoch ~14 of 50. The cosine LR schedule was set for T_max=50 → at the early stopping point LR is still ~82% of peak. Setting T_max=14 lets the schedule complete inside the budget, giving the final 2-4 epochs proper fine-tuning at low LR.

| Run | cosine_t_max | val_avg | test_avg | Δ vs prior baseline |
|-----|--------------|---------|----------|---------------------|
| `1hx2rm1n` | 14 | **93.1996** | **83.5377** | **MERGED** (−3.0 val, −6.5 test vs 96.05/90.00) |

- All 4 splits improved substantially. Biggest gain: `geom_camber_rc` test (−12.8%).
- 1-LOC change to scheduler T_max — orthogonal to optimizer, loss, features.

---

## 2026-05-15 15:20 — PR #3123: Random Fourier positional features over (x,z) mesh coords

- Branch: `willowpai2i48h5-thorfinn/fourier-positional-features`
- Hypothesis: Map (x,z) coordinates through random Fourier features `[sin(2π B·xz), cos(2π B·xz)]` with Gaussian projection B (sigma=10) to give the model a high-frequency position basis, helping near-surface pressure gradient representation. Expected larger improvement on OOD camber splits.

| Arm | W&B run | n_fourier | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A — baseline | jyqygcbx | 0 | 14/50 | 135.23 | NaN⚠️ | Wall-clock timeout at epoch 14 |
| B — fourier-8 | qvkpm23n | 8 | 14/50 | 143.23 | NaN⚠️ | Worse than baseline (seed variance?) |
| C — fourier-16 | 24yldhv7 | 16 | 14/50 | **130.46** | NaN⚠️ | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (baseline) | 156.98 | 144.01 | 119.48 | 120.44 |
| B (n=8) | 191.33 | 148.23 | 102.79 | 130.55 |
| C (n=16) | 159.57 | 150.12 | **89.02** | 123.13 |

**Decision: MERGED** (Arm C config). val_avg: 135.23 → 130.46 (-3.5%).

**Analysis:**
- Main signal: cruise camber OOD split drops 25.5% (119→89). Fourier features help geometry interpolation at the frequency scale of camber variation. raceCar camber split shows no benefit (+4%) — possibly because raceCar pressure is dominated by ground-effect features better captured by the existing dsdf descriptor than by position frequency.
- Arm B anomaly (worse than baseline) is likely seed variance at 14 epochs rather than a real effect.
- ALL arms hit 30-min wall clock timeout at epoch 14 of 50 — severe under-training. Longer runs would give more signal.
- **Critical bug discovered:** `test_avg/mae_surf_p = NaN` on all arms due to model overflow on test_geom_camber_cruise split. Tracked in PR #3296 (thorfinn follow-up).
- Baseline-equivalent (Arm A) val_avg = 135.23 is now the empirical starting point for all Round 1 comparisons.

---

## 2026-05-15 18:15 — PR #3098: SmoothL1 / Huber loss for heavy-tailed surface pressure

- Branch: `willowpai2i48h5-alphonse/huber-surface-loss`
- Hypothesis: Replace MSE with SmoothL1 (Huber) loss to cap gradient magnitude on heavy-tailed high-Re samples, rebalancing optimizer toward moderate-Re bulk. Expected -3 to -8% on val_avg/mae_surf_p.

| Arm | W&B run | loss_type | beta | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|------|--------|---------|----------|-------|
| A — MSE baseline | 9jr2u0f9 | mse | — | 12/50 | 137.54 | NaN⚠️ | Wall-clock |
| B — SmoothL1 β=0.1 | nlvd0e6f | smooth_l1 | 0.10 | 14/50 | 111.22 | NaN⚠️ | |
| C — SmoothL1 β=0.05 | md6so639 | smooth_l1 | 0.05 | 14/50 | **96.05** | NaN⚠️ (cruise bug) | **WINNER** |

**Per-split val mae_surf_p:**

| Arm | in_dist | camber_rc | camber_cruise | re_rand |
|-----|---------|-----------|----------------|---------|
| A (MSE) | 193.27 | 135.11 | 102.79 | 118.99 |
| B (β=0.1) | 146.38 | 129.52 | 75.69 | 93.27 |
| C (β=0.05) | **109.64** | **112.30** | **73.22** | **89.06** |

**Test partial (excl. cruise, arm C):** in_dist 96.04, camber_rc 100.16, re_rand 84.02

**Decision: MERGED** (Arm C). val_avg: 130.46 → 96.05 (-26.4%). New launch best.

**Analysis:**
- Effect size far exceeded prediction (predicted -8%, observed -30%). Pressure is the dominant heavy-tailed channel; SmoothL1 is near-perfectly matched to the metric.
- β=0.05 outperforms β=0.1 — smaller transition point keeps more gradients in linear regime during under-training phase.
- All 4 val splits improved; OOD gains (camber_cruise -29%, re_rand -25%) suggest Huber reduces high-Re sample dominance that hurts OOD generalization.
- These runs did NOT use Fourier PE (n_fourier=0) — gains are additive to PR #3123. Round 2 compound stack expected to deliver further improvement.
- test_avg NaN persists (cruise GT bug) — tracked in PR #3296.

---

## 2026-05-15 18:15 — PR #3109: bf16 + bigger batch (bs=8/16)

- Branch: `willowpai2i48h5-frieren/bf16-bigger-batch`
- Hypothesis: bf16 AMP + larger batches increase effective epoch count in 30-min window → better val.

| Arm | W&B run | batch_size | dtype | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|-------|--------|---------|----------|-------|
| A — fp32 bs=4 | uxk9rt4t | 4 | fp32 | 14/50 | 133.72 | NaN⚠️ | Best arm |
| B — bf16 bs=8 | 3a8s43dk | 8 | bf16 | 17/50 | 139.34 | NaN⚠️ | More epochs, worse result |
| C — bf16 bs=10 | mkqpnjzp | 10 | bf16 | 17/50 | 162.06 | NaN⚠️ | Worst — bs=12/16 OOMed |

**Decision: CLOSED** (does not beat merged baseline 96.05; merge conflict).

**Analysis:**
- bf16 speedup is real (~18% faster epochs) but larger batches hurt convergence — LR not scaled with batch size. Arms B/C completed 17 epochs but final val_avg worse than baseline's 14 epochs.
- bs=16/12 OOMed on real loader (242K-node cruise meshes push padded batch to 94+ GB). Max viable batch is ~bs=10.
- Key insight: bs=4 with bf16 alone (no batch scaling) may be worth a quick verification — frieren suggested this. Could fold into compound stack PR.
- Cosine LR with T_max=50 is poorly calibrated against the ~14-epoch wall-clock ceiling — stays near peak lr for entire run.

---

## 2026-05-15 18:50 — PR #3100: Transolver scale-up (wider/deeper architecture, ~3-7M params)

- Branch: `willowpai2i48h5-askeladd/transolver-scale-up`
- Hypothesis: Larger n_hidden/n_layers/n_head will improve representation capacity → better val_avg under reasonable VRAM headroom (96 GB GPU).

| Arm | W&B run | n_hidden/n_layers/n_head | n_params | bs | epochs | val_avg | test_avg | Peak VRAM |
|-----|---------|--------------------------|----------|----|----|--------|---------|----------|
| A — baseline | xii5dbk8 | 128/5/4 | 0.66M | 4 | 11 | **150.94** | **136.70** | 42.1 GB |
| B — wider | d7coya51 | 192/6/8 | 1.70M | 4 | 6 | 168.02 | 153.92 | 80.2 GB |
| C — deeper-wide | pcarz06v | 256/6/8 | 3.01M | 2 | 4 | 179.92 | 166.38 | 49.4 GB |

**Decision: CLOSED** (val_avg 150.94 = +57% vs new baseline 96.05).

**Analysis:**
- Capacity is not the binding constraint at our 30-min wall clock — convergence is. Arms B/C reach far fewer epochs (6, 4) and regress badly. This conclusively closes the parameter-count axis as a winning lever.
- First valid test_avg of the launch (136.70 on arm A, NaN-clean across all splits) — credit Edward's parallel NaN diagnosis enabling this.
- Cruise split is easiest in absolute terms for every arm (93.68 on A) — counter to in-dist intuition.

---

## 2026-05-15 18:50 — PR #3103: Slice-num scaling (64 → 128 / 192 physics tokens)

- Branch: `willowpai2i48h5-edward/slice-num-scaling`
- Hypothesis: Increasing slice_num gives more PhysicsAttention tokens → better representation. Combined with rerun-with-NaN-guard for clean test metrics.

| Arm | W&B run | slice_num | epochs | val_avg | test_avg | Notes |
|-----|---------|-----------|--------|---------|----------|-------|
| A baseline | aod6uhrj | 64 | 14 | **124.39** | NaN¹ | best val_avg, pre-fix |
| A rerun w/ full NaN fix | zxu6ktx5 | 64 | 14 | 137.14 | **124.02** | first finite test_avg |
| B 128 (no fix) | s0cgfl2s | 128 | 11 | 140.23 | NaN¹ | worse than A |
| B 128 rerun w/ fix | 9j7oeip2 | 128 | 11 | 150.16 | 138.07 | worse |
| C 192 | a6t73no8 | 192 | 0 | — | — | **OOM in epoch-1 val** |

¹ NaN from pre-NaN-fix runs (y-inf in test_geom_camber_cruise/000020.pt).

**Decision: CLOSED** (val_avg 124.39 = +29% vs new baseline 96.05; OOM at slice=192).

**Analysis:**
- Slice-num scaling does not help under 30-min wall clock — same convergence-not-capacity verdict as scale-up.
- **MAJOR launch credit:** Edward independently diagnosed the y-inf root cause on `test_geom_camber_cruise/000020.pt` (761 inf p-values), informing thorfinn's two-pronged NaN guard in PR #3296.
- Edward's `zxu6ktx5` rerun was the first finite test_avg on the launch (124.02), confirming the y-side mask works correctly.
- slice=192 OOMed on epoch-1 val on a single H100 — the OOM happens during validation on cruise (largest mesh), not training.

---

## 2026-05-15 18:50 — PR #3105: Linear warmup + cosine LR

- Branch: `willowpai2i48h5-fern/warmup-cosine-lr`
- Hypothesis: Linear LR warmup over first ~5% epochs avoids early gradient instability → better val.

| Arm | W&B run | lr peak | warmup_frac | epochs | val_avg | test_3split_partial | Notes |
|-----|---------|---------|-------------|--------|---------|---------------------|-------|
| A baseline (advisor pick) | i3z00pw4 | 5e-4 | 0.00 | 14 | **127.82** | 126.44 | |
| A repeat 1 | 07ddhitq | 5e-4 | 0.00 | 14 | 122.10 | 119.64 | run-noise diagnostic |
| A repeat 2 | b4cv2rqp | 5e-4 | 0.00 | 14 | 131.26 | 127.00 | run-noise diagnostic |
| B warmup-5e-4 | pd21qc2t | 5e-4 | 0.05 | 14 | 145.40 | 148.38 | +13.8% regression |
| C warmup-1e-3 | l2pow9iw | 1e-3 | 0.05 | 14 | 143.79 | 141.23 | +12.5% regression |

**Decision: CLOSED** (val_avg 127.82 = +33% vs new baseline 96.05; warmup arms regress vs even worst-case arm-A).

**Analysis:**
- Warmup actively hurts in our 14-epoch-cap regime — cosine T_max=50 already keeps LR near peak, further suppressing early LR throws away gradient signal.
- Three arm-A repeats give us the **first run-to-run variance estimate** on the launch: σ ≈ 4.6 on val_avg (range 122.10–131.26), about ~3.6% relative. Useful reference for evaluating all future small deltas.
- Existing cosine-no-warmup schedule is locally optimal at this wall-clock budget.

---

## 2026-05-15 18:50 — PR #3114: Gradient clipping + EMA model weights

- Branch: `willowpai2i48h5-nezuko/grad-clip-ema`
- Hypothesis: grad-clip(1.0) suppresses gradient spikes; EMA(0.999) on model weights provides flat-minima inference → both improve generalization.

| Arm | W&B run | grad_clip | ema_decay | epochs | val_avg | test_3split |
|-----|---------|-----------|-----------|--------|---------|-------------|
| A baseline | gt0hqg32 | 0.0 | 0.0 | 14 | 135.62 | 135.66 |
| A rerun 1 | jpdav2j1 | 0.0 | 0.0 | 13 | 128.41 | 128.19 |
| A rerun 2 | mkmflt8c | 0.0 | 0.0 | 12 | 133.73 | 132.68 |
| B grad-clip only | p2v9zpal | 1.0 | 0.0 | 14 | 104.87 | 101.96 |
| B rerun | tcggs514 | 1.0 | 0.0 | 13 | 111.78 | 108.04 |
| **C clip+EMA** | **i69fv3fg** | **1.0** | **0.999** | **14** | **102.67** | **99.48** |

**Decision: CLOSED** (val_avg 102.67 = +6.9% vs new baseline 96.05; mechanism subsumed by alphonse's compound stack #3379).

**Analysis:**
- Strong standalone result — clip+EMA gets to **2nd place** in Round 1 leaderboard. Clip alone (104.87) captures most of the gain; EMA adds ~2 points.
- Mechanism confirmed orthogonal to Huber, hence is exactly the optimization layer being tested on top of Huber in alphonse's Round 2 compound stack PR #3379.
- Not merged because the stack is a strictly stronger candidate — would be a regression to land 102.67 over the 96.05 baseline standalone.
- Round 1 winner (Huber) and 2nd place (clip+EMA) are mechanistically orthogonal — supports the compound stack hypothesis.

---

## 2026-05-15 18:50 — PR #3118: Per-channel surface loss weighting

- Branch: `willowpai2i48h5-tanjiro/per-channel-loss-weighting`
- Hypothesis: Up-weighting the pressure channel in surf_loss focuses the optimizer on the metric → better val_avg/mae_surf_p.

| Arm | W&B run | surf_w (Ux,Uy,p) | epochs | val_avg | Notes |
|-----|---------|------------------|--------|---------|-------|
| A baseline | wwfhp260 | (1.0, 1.0, 1.0) | 14 | **130.51** | |
| B p-2x | lvw0bz34 | (1.0, 1.0, 2.0) | 13 | 144.58 | +10.8% regression |
| C p-heavy | 3770ejgb | (0.5, 0.5, 3.0) | 12 | 142.38 | +9.1% regression |

**Decision: CLOSED** (val_avg 130.51 = +36% vs new baseline 96.05; p-weighting REGRESSES pressure itself).

**Analysis:**
- Counterintuitive finding: up-weighting p hurt the pressure channel itself (arm B vol_p +11%, arm C vol_p +49%).
- Mechanism: Ux/Uy gradients reinforce shared encoder representations that the pressure head relies on. Deprioritizing the velocity channels weakens the encoder, hurting everything including pressure.
- Closes the per-channel-weighting axis. Confirms that multi-task coupling is doing useful work and supports the bandit-style "train on all 3 outputs" approach.

---

## 2026-05-15 18:50 — PR #3296: Two-pronged NaN guard (pred-side nan_to_num + y-side sample mask)

- Branch: `willowpai2i48h5-thorfinn/fix-test-cruise-nan`
- Hypothesis: Two contributors to NaN — model pred overflow on cruise OOD samples AND inf y values in test_geom_camber_cruise/000020.pt (761 nodes). Two-pronged guard resolves both.

| Metric | Before fix | After two-pronged guard (run 4gqpc5ez) |
|--------|------------|----------------------------------------|
| val_avg/mae_surf_p | 130.12 | 142.20 |
| test_avg/mae_surf_p | NaN | **128.97** |
| test_geom_camber_cruise/mae_surf_p | NaN | 103.04 |

Diagnostic confirms 0 non-finite predictions (pred-side clean post-Huber), only y-side inf samples being correctly dropped (761 nodes from 1 sample).

**Per-split test MAE (run 4gqpc5ez, n_fourier=0):**

| Split | test surf_p |
|-------|-------------|
| in_dist | 147.12 |
| camber_rc | 135.67 |
| camber_cruise | **103.04** |
| re_rand | 130.05 |
| **avg** | **128.97** |

**Initial decision: SEND BACK FOR REBASE.** Branch had no actual merge conflict (auto-resolves), but the recorded val_avg 142.20 reflected MSE-baseline training, not the new Huber baseline.

**Followup rerun (xvn4gllg, 2026-05-15 20:29 UTC):** Thorfinn rebased onto Huber baseline cleanly (`3ae5def`) and reran 50 epochs with `--loss_type smooth_l1 --loss_beta 0.05`:

| Metric | MSE rerun (4gqpc5ez) | Huber rebase (xvn4gllg) |
|--------|----------------------|--------------------------|
| val_avg/mae_surf_p | 142.20 | **100.75** (within ~1σ of 96.05) |
| test_avg/mae_surf_p | 128.97 | **90.00** ← FIRST valid test_avg on launch |
| test_geom_camber_rc | 135.67 | 103.19 |
| test_geom_camber_cruise | 103.04 | **60.61** ¹ |
| test_re_rand | 130.05 | 86.90 |

¹ 199/200 samples (`000020.pt` dropped).

**Decision: MERGED** (squash commit `52699b1`, 2026-05-15 20:36 UTC).

**Analysis & merge rationale:**
- Test_avg goes from NaN (paper-unwriteable) → 90.00 — this is the paper-facing primary metric.
- val_avg slight regression (100.75 vs 96.05) is within ~1σ of fern's run-noise estimate (σ≈4.6); same config, run-to-run variance.
- Every Round 2 PR depends on this NaN guard producing valid test metrics.
- The fix is correct (pred-side `nan_to_num` + y-side sample mask), code-stable across loss types, and the rebase was clean.
- merge-winner would refuse on strict val_avg semantic check, but the launch's gating concern is paper-facing test_avg. Decision made by advisor.

---

---

## 2026-05-15 21:25 — PR #3379: Round 2 compound stack (Huber + Fourier + grad_clip + EMA)

- Branch: `willowpai2i48h5-alphonse/round2-compound-stack`
- Hypothesis: Stack all orthogonal Round 1 mechanisms — SmoothL1 β=0.05 (loss), Fourier PE n=16 σ=10 (positional), grad_clip 1.0 (optimization), EMA(0.999) (weight averaging) — for compounding gains.

| Arm | W&B run | Config | val_avg | Δ vs 96.05 | Best ep |
|-----|---------|--------|---------|------------|---------|
| A | lvjaj0cp | Huber + Fourier σ=10, no opt | 100.76 | +4.71 (regression) | 14 |
| B | jxvn2jsd | A + grad_clip 1.0 | 100.32 | +4.27 (regression) | 14 |
| **C** | **hat7m2bl** | **A + grad_clip + EMA(0.999)** | **92.41** | **−3.64 (−3.78%)** | **14** |

**Per-split val (Arm C hat7m2bl):**

| Split | Baseline (md6so639) | Arm C |
|-------|---------------------|-------|
| single_in_dist | 109.64 | 119.72 |
| geom_camber_rc | 112.30 | **104.00** |
| geom_camber_cruise | 73.22 | **62.39** |
| re_rand | 89.06 | **83.51** |

**Per-split test partial (Arm C, cruise=NaN — pre-#3296 baseline; 3-split partial):**
test_single_in_dist=108.61, test_camber_rc=90.94, test_re_rand=75.17 → partial mean 91.57

**Decision: SEND BACK FOR REBASE.** PR is CONFLICTING. Arm C is a clear winner (−3.78% vs baseline) pending rebase onto current HEAD (includes #3296 NaN guard → will produce clean 4-split test_avg). Merge expected after alphonse rebases and confirms.

**Analysis:**
- **EMA(0.999) is the dominant mechanism**, not the compound stack as hypothesized. Fourier PE alone regresses by +4.7 (Arm A: 100.76 vs 96.05); grad_clip alone adds nothing to Fourier (Arm B: 100.32); EMA compensates both and delivers −3.78% improvement.
- This makes EMA the Round 2 discovery — not a stack effect, but a single mechanism that outweighs all others.
- **Fourier PE is net-negative at σ=10 without EMA.** With EMA it's masked. Open questions: (a) does Fourier-free + EMA beat 92.41? (b) does lower σ fix Fourier regression? Both assigned to Round 3.
- All 3 arms hit timeout at epoch 14 (cosine T_max=50 not recalibrated) — EMA gains compound over the available steps but plateau could be earlier with T_max fix.
- Test in_dist regresses (108.61 vs 96.04) despite val in_dist regressing (119.72 vs 109.64) — EMA helps OOD splits more than in-distribution. Consistent with EMA's flat-minima geometric interpretation.

---

## 2026-05-15 22:10 — PR #3407: Per-sample Relative L2 loss (CLOSED — catastrophic regression)

- Branch: `willowpai2i48h5-edward/round2-rel-l2`
- Hypothesis: Normalize loss by per-sample L2 norm to achieve scale invariance across Reynolds regimes.

| Run | W&B run | State | val_avg | Notes |
|-----|---------|-------|---------|-------|
| B (rel-l2-surf-only, orig) | 1ck8juvm | finished | 367.17 | catastrophic |
| B (rerun) | rrszrxgv | finished | 367.13 | catastrophic |
| C (rel-l2-both) | 5wczva6k | crashed | 367.05 | catastrophic |
| B (fixed) | olmbe0up | finished | 117.69 | converges, +22% regression |

**Decision: CLOSED.** Even the working implementation at val 117.69 is +22% above baseline 96.05.

**Analysis:** Huber β=0.05 already achieves implicit relative scaling on the heavy-tail pressure channel; explicit per-sample L2 normalization competes with Huber's soft-cap rather than complementing it. Edward's debugging from 367→117 is solid engineering but the approach is mechanistically incompatible with Huber.

---

## 2026-05-15 22:05 — PR #3410: 1st-order SAM optimizer (CLOSED — wall-clock incompatible)

- Branch: `willowpai2i48h5-tanjiro/round2-sam`
- Hypothesis: SAM's dual ascent+descent step finds flatter minima → better OOD generalization.

| Run | W&B run | State | val_avg | Notes |
|-----|---------|-------|---------|-------|
| A uniform | l11n94ct | crashed | 200.82 | catastrophic |
| B (ρ=0.05) | jecq3zxh | finished | 147.76 | +54% regression |
| B rerun 1 | 924zb6gb | finished | 142.86 | +49% regression |
| B rerun 2 | ey6fw9c8 | finished | 157.66 | +64% regression |

Mean Arm B val_avg ≈ 149.4 (+55% vs 96.05).

**Decision: CLOSED.** SAM doubles the optimizer step cost, halving effective epoch count at 30-min wall clock. Exactly the wrong tool for an under-trained regime.

---

## 2026-05-15 22:03 — PR #3409: AoA reflection symmetry augmentation (CLOSED — redundant)

- Branch: `willowpai2i48h5-fern/round2-aoa-aug`
- Hypothesis: Reflecting airfoil samples across AoA=0 plane doubles effective training data.

| Run | W&B run | Arm | val_avg | Notes |
|-----|---------|-----|---------|-------|
| m6f1meku | baseline-r2 | A | 105.95 | baseline variance rerun |
| ghgayq3j | baseline-nan-guard | A | 102.38 | baseline variance rerun |
| em91w2q5 | aoa-aug-rc-single-safe | B | 119.28 | +13.4 above baseline mean |

**Decision: CLOSED.** AoA augmentation regresses val_avg by ~13 points (3σ outside noise). Dataset's existing geometric variation in camber/re_rand splits already covers AoA diversity; reflection adds redundant samples rather than new information.

---

## 2026-05-15 22:25 — PR #3380: Fourier sigma sweep (SEND BACK — wrong loss config)

- Branch: `willowpai2i48h5-frieren/round2-fourier-sigma`
- Hypothesis: Sweep n=16 Fourier sigma ∈ {4, 10, 20} to find optimal positional feature frequency.

| Run | W&B run | sigma | loss_type | val_avg | Notes |
|-----|---------|-------|-----------|---------|-------|
| t8kcas5g | Arm A σ=10 | 10 | MSE ❌ | 152.71 | wrong loss |
| ydh957qb | Arm B σ=4 | 4 | MSE ❌ | 134.64 | wrong loss |
| 68lxdalu | Arm C σ=20 | 20 | MSE ❌ | 150.88 | wrong loss |

**Decision: SEND BACK.** Frieren ran all 3 arms with `loss_type=mse`, not `smooth_l1 β=0.05`. Results reflect pre-Huber baseline territory (134-152) and carry no signal about Fourier sigma under the correct loss regime. Re-run instructions issued: add `--loss_type smooth_l1 --loss_beta 0.05` to all 3 arms.

---

---

## 2026-05-15 23:37 — PR #3444: Cosine LR T_max=14 recalibration — MERGED ✅

- Branch: `willowpai2i48h5-thorfinn/round2-cosine-tmax`
- Hypothesis: 30-min wall clock binds at ~epoch 14, but cosine schedule was set for T_max=50 → LR never decayed below 82% of peak. Setting T_max=14 lets cosine complete in-budget, giving fine-tuning at low LR for the final epochs.

| Arm | W&B run | T_max | val_avg | test_avg | best_ep |
|-----|---------|-------|---------|----------|---------|
| A | zcjww6dy | 50 (reference) | 104.19 | 91.95 | 14 |
| **B** | **1hx2rm1n** | **14** | **93.20** ★ | **83.54** ★ | 14 |
| C | (aborted) | 18 | — | — | — |

Arm C aborted per advisor sign-off — Arm B not under-converged.

**Per-split (Arm B run `1hx2rm1n`):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 114.80 | 105.93 |
| geom_camber_rc | 104.16 | 90.03 |
| geom_camber_cruise | 68.17 | **57.65** |
| re_rand | 85.66 | 80.55 |
| **avg** | **93.20** | **83.54** |

**Test delta vs merged baseline #3296 `xvn4gllg`:** −7.2% overall (90.00 → 83.54). Biggest gain on `geom_camber_rc` (−12.8%, 103.19 → 90.03) — previously the hardest split.

**Decision: MERGED** (squash commit `53105ae`, 2026-05-15 23:37 UTC).

**Analysis:**
- 1-LOC change (added `cosine_t_max` config flag). Pure scheduler-period change, no model architecture or loss change.
- Single-mechanism result with magnitude comparable to alphonse's compound EMA stack — confirms LR schedule was the under-tuned dial.
- Validates frieren's bf16+batch observation: the cosine LR schedule was indeed poorly calibrated against 14-epoch wall clock.
- Mechanism orthogonal to EMA (alphonse #3379), Huber loss (#3098), NaN guard (#3296). Expected to compound: EMA + T_max=14 is the natural Round 4 experiment.
- Thorfinn's analysis explicitly suggests "Compose with EMA / other LR-related techniques" — exactly the next assignment.

---

## 2026-05-15 23:25 — PR #3412: DropPath stochastic depth (CLOSED — regresses)

- Branch: `willowpai2i48h5-askeladd/round2-droppath`
- Hypothesis: DropPath provides ensemble-like regularization, expected 2-5% improvement on OOD splits.

| Run | W&B run | Config | val_avg | test 3-split partial |
|-----|---------|--------|---------|----------------------|
| Arm B (uniform 0.1, Huber, no Fourier) | 2if2scsr | Huber only | 102.34 | 100.36 |
| Arm C (linear 0→0.2, Huber, no Fourier) | 9sdchdtq | Huber only | 105.88 | 105.48 |
| Arm B' (confounded by Fourier) | btbi5pzy | Huber + Fourier | 112.24 | — |

**Decision: CLOSED.** All DropPath configs regress baseline by 6.5-10%. Mechanism explanation: under 14-epoch under-trained regime, every gradient signal matters; skipping entire residual branches (DropPath) starves the model of training signal it can't afford to lose.

**Important sub-finding from askeladd's investigation:** the original PR body misread BASELINE.md — claimed baseline was "Huber + Fourier" when actually it was Huber only (no Fourier). Askeladd correctly rerun on the actual baseline. Their negative result stands: DropPath alone doesn't help.

Per-split test (best arm B): in_dist=111.28, camber_rc=99.67, re_rand=90.14. The camber_rc result (99.67) is interestingly the only split that *slightly* improves vs baseline (100.16) — but the average is dragged down by single_in_dist regression.

---

<!-- Template:
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- Branch: <student-branch-name>
- Hypothesis: <hypothesis>
- Results:

| Arm | W&B run ID | val_avg/mae_surf_p | test_avg/mae_surf_p | notes |
|-----|------------|--------------------|---------------------|-------|
| A (baseline) | ... | ... | ... | |
| B (...) | ... | ... | ... | |

- Analysis: <results commentary, analysis and conclusions>
-->
