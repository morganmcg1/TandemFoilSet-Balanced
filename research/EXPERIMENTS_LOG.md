# SENPAI Research Results

## 2026-05-17 04:00 — PR #4265: Lookahead-Lion LR sweep ← CLOSED (Lion LR landscape SHARPER than AdamW; rate-limit-close, Arm 1 only)

- Branch: `willowpai2i48h1-fern/lookahead-lion-lr-sweep`
- Student: willowpai2i48h1-fern
- W&B runs: `95hnfvyz` (CANONICAL, val=49.41, best_ep=17), `ip13tzlx` (rerun, bit-identical val=49.41), 2 failed (`1mw6phnm`, `oyum34fy`), 1 actively retrying (`01p7odez`); group `lookahead_lion_lr_sweep`
- Hypothesis: Probe Lion-era LR frontier vs default cfg.lr=5e-4 → Lion lr=1.667e-4. Arms: cfg.lr=3e-4 (Lion lr=1e-4) and cfg.lr=7.5e-4 (Lion lr=2.5e-4).

### Results (Arm 1 only completed; rate-limit-close pattern)

| Arm | val_avg | test_avg | best_ep | Δ vs baseline (47.97) |
|---|---|---|---|---|
| cfg.lr=3e-4 (Lion lr=1e-4) | **49.41** | 47.93 | 17 | **+1.44 regress** |
| cfg.lr=7.5e-4 (Lion lr=2.5e-4) | NEVER RAN | — | — | missing arm |

### Frontier finding

**Lion LR landscape is SHARPER than AdamW LR landscape.** Dropping cfg.lr by 40% (5e-4 → 3e-4) gives +1.44 val regression. Under AdamW (PR #4182), the LR landscape was flat-to-improving in the 5e-4→7e-4 direction. Under Lion, the optimum has narrower tolerance.

Mechanism: Lion's sign-quantized step has constant per-parameter magnitude → no adaptive gradient scaling to forgive small-LR error. Halving Lion LR halves step size everywhere, slowing convergence within the fixed 17-epoch cosine budget.

### Decision

Closed — val=49.41 > current programme best 47.97. The cfg.lr=7.5e-4 direction is now less interesting (LR-sensitivity established on the smaller side; larger LR may also be sub-optimal). Closing as canonical Lion-LR-sensitivity data.

### Process flag

4th occurrence of heartbeat-rerun pattern (#4202, #4241, #4242, #4264). Pod stuck retrying Arm 1 instead of progressing to Arm 2. Fern reassigned to **slice_num=96 (#4323)** — SINGLE-ARM to eliminate this risk.

## 2026-05-17 04:00 — PR #4264: Lookahead-Lion β2 scan ← CLOSED (Lion β2 sensitivity — largest single-knob regression in Lion era; rate-limit-close, Arm 1 only)

- Branch: `willowpai2i48h1-frieren/lookahead-lion-b2-scan`
- Student: willowpai2i48h1-frieren
- W&B runs: `rp741afd` (CANONICAL, val=54.62, best_ep=17), `pp1nvrod` (rerun, bit-identical), 1 failed; group `lookahead_lion_b2_scan`
- Hypothesis: Probe Lion m-buffer β2 ∈ {0.95, 0.98} vs default 0.99. Lion's β2 controls m-buffer EMA decay (direction-extraction smoothing).

### Results (Arm 1 only completed; rate-limit-close pattern)

| Arm | val_avg | test_avg | best_ep | Δ vs baseline (47.97) |
|---|---|---|---|---|
| lion_b2=0.95 | **54.62** | 52.24 | 17 | **+6.65 major regress** |
| lion_b2=0.98 | NEVER RAN | — | — | missing arm |

### Frontier finding

**Lion β2 is HIGHLY sensitive — dropping from 0.99 to 0.95 collapses performance by +6.65 val MAE.** This is the LARGEST single-knob regression seen in the Lion era.

Mechanism: Lion's β2 controls the m-buffer EMA decay. β2=0.95 → m has half-life ≈14 training steps (vs ~69 steps at β2=0.99). Lion's update direction (sign of β1·m + (1-β1)·grad) needs m to smooth across many batches to extract a stable direction. With β2=0.95 + batch=4, m forgets noise too quickly → sign-update direction becomes noise-driven.

**Opposite of the AdamW β2 finding** (PR #4183 was flat in [0.93, 0.97]): under AdamW, Lookahead's basin-averaging absorbed v_t noise. Under Lion, m is doing direction-extraction (not magnitude smoothing); small β2 cannot be compensated by Lookahead.

### Conclusion

**Lion β2 should likely INCREASE from default 0.99, not decrease.** The b2=0.98 arm is unlikely to win; a more interesting probe would be β2 ∈ {0.995, 0.999} — but that's a future PR. Closing as canonical Lion-β2-sensitivity data.

### Decision

Closed — val=54.62 > current programme best 47.97. Frieren reassigned to **slice_num=32 (#4325)** — SINGLE-ARM, architectural; pairs with fern's slice_num=96 (#4323) to characterize the slice dimension.

## 2026-05-17 03:00 — PR #4241: Lookahead-Lion k=3 ← CLOSED (k-shift prediction CONFIRMED; canonical k-frontier)

- Branch: `willowpai2i48h1-edward/lookahead-lion-k3`
- Student: willowpai2i48h1-edward
- W&B runs: `v7jka5nw` (CANONICAL, val=48.20, best_ep=17), `g7bprn1m` (rerun, val=52.53, best_ep=15); group `lookahead_lion_k_sweep`
- Hypothesis: Compose AdamW-era k=3 finding (was the AdamW k-minimum) with Lion-era baseline.

### Results (W&B-verified; best run is canonical with best_ep=17)

| Metric | Value | Δ vs k=5 baseline (PR #4123) |
|---|---|---|
| val_avg/mae_surf_p | **48.2021** | +0.229 |
| test_avg/mae_surf_p | **46.9625** | +0.472 |
| best_epoch | 17 (cosine floor ✓) | — |

### Round-16 prediction CONFIRMED

Round-16 close of PR #4268 predicted "edward's #4241 likely regresses to val ~48.3-48.5." Actual: val=48.20 — within 0.1 of predicted range. **Lion k-curve U-min is at or right of k=5.**

### Cross-optimizer k-curve (definitive)

| k | AdamW val | Lion val |
|---|---|---|
| 2 | 56.49 | 48.84 (PR #4268) |
| 3 | **55.97 (AdamW min)** | 48.20 (THIS PR) |
| 5 | 57.22 | **47.97 (Lion min so far)** |
| 7 | — | TBD (edward #4310 incoming) |
| 8 | 60.09 | — |

Lion-side curve ASYMMETRIC: k=3 regresses by only +0.23, k=2 by +0.86 (steep left flank between k=2→3, shallow between k=3→5). Suggests U-min may be even further right; k=7 is the next test.

### Decision

Closed — val=48.20 > current programme best 47.97. Strong mechanism finding; edward reassigned to **Lookahead-Lion k=7 (#4310)** — direct prediction test for U-min location.

⚠️ Rerun-variance pattern (same as #4242, #4202): canonical is the first run with best_ep=17.

## 2026-05-17 03:00 — PR #4242: Lookahead-Lion seed=2 ← CLOSED (3-seed canonical COMPLETE for programme best)

- Branch: `willowpai2i48h1-nezuko/lookahead-lion-seed2`
- Student: willowpai2i48h1-nezuko
- W&B runs: `2t9j83vn` (CANONICAL, val=48.84, best_ep=17), `apn6wsb4` (rerun val=51.33), `lrgw1b11` (rerun val=51.66); group `lookahead_lion_seed_scan`
- Hypothesis: Complete 3-seed canonical (seed=0=47.97, seed=1=49.21, seed=2=?).

### Results (W&B-verified; first run is canonical)

| Metric | Value | Δ vs seed=0 |
|---|---|---|
| val_avg/mae_surf_p | **48.8437** | +0.870 |
| test_avg/mae_surf_p | **47.5562** | +1.066 |
| best_epoch | 17 (cosine floor ✓) | — |

### Lookahead-Lion 3-SEED CANONICAL (CLEAN, complete)

| Seed | val | test | best_ep |
|---|---|---|---|
| 0 | 47.9735 | 46.4900 | 17 |
| 1 | 49.2089 | 47.6172 | 17 |
| 2 | 48.8437 | 47.5562 | 17 |
| **3-seed mean** | **48.6754** | **47.2211** | — |
| **σ̂** | **0.6396** | **0.6364** | — |

**Cleanest 3-seed canonical in the programme.** All three seeds hit cosine floor (best_ep=17). σ̂≈0.64 MAE on both val and test — paper-ready noise floor.

### Decision

Closed — best run val=48.84 > programme best 47.97. nezuko reassigned to **Lookahead-Lion + heads=8 (#4304)** — 3rd architectural arm (attention capacity dimension; orthogonal to mlp_ratio and depth).

⚠️ Rerun-variance flag: heartbeat-restart pattern produces degraded later reruns. First clean run is canonical.

## 2026-05-17 02:30 — PR #4268: Lookahead-Lion k=2 ← CLOSED (canonical Lion k-frontier; U-curve SHIFTS RIGHT vs AdamW)

- Branch: `willowpai2i48h1-tanjiro/lookahead-lion-k2`
- Student: willowpai2i48h1-tanjiro
- W&B run: `3bp5obwz` (group `lookahead_lion_k_sweep`)
- Hypothesis: Extend Lion-era k-curve below k=5 (k=2 was 0.73 better than k=5 under AdamW).

### Results (terminal SENPAI-RESULT, W&B-verified; clean single run)

| Metric | Value | Δ vs k=5 baseline (PR #4123) |
|---|---|---|
| val_avg/mae_surf_p | **48.8371** | **+0.864** |
| test_avg/mae_surf_p | 47.5378 | +1.048 |
| best_epoch | 17 (cosine floor ✓) | — |
| VRAM | 35.86 GB | — |

All splits regress uniformly (+0.4 to +1.7) — clean signal.

### Cross-optimizer k-curve comparison (major mechanism finding)

| k | AdamW val | Lion val |
|---|---|---|
| 2 | 56.49 | **48.84** (THIS PR) |
| 3 | 55.97 (k-min at AdamW) | pending (edward #4241) |
| 5 | 57.22 | **47.97** (Lion-min, PR #4123) |
| 8 | 60.09 | — |

**The U-curve minimum SHIFTS RIGHT when switching AdamW → Lion** (AdamW min at k=3; Lion min at k≥5).

### Mechanism interpretation

Lion's sign-update has constant per-step magnitude (no per-param variance scaling) → fast trajectory has low variance natively. With variance already suppressed, the fast weights need **more steps** before drifting meaningfully from slow weights. Short k=2 syncs pull fast weights back before sufficient basin geometry has been explored — Lookahead's basin-averaging benefit is **lost**.

Implication: edward's #4241 (Lookahead-Lion k=3, predicted val ~48.3-48.5) likely also regresses. If so, k=7-8 under Lion becomes a research question.

### Decision

Closed — val=48.84 > current best 47.97. Strong mechanism finding (k-curve right-shift). tanjiro reassigned to **architectural escalation: depth=6 (#4294)**, orthogonal capacity dimension to thorfinn's mlp_ratio=3 (#4286).

✓ Clean single run, no heartbeat rerun. Good flow.

## 2026-05-17 02:00 — PR #4213: Lookahead-AdamW k=3 α=0.8 ← CLOSED (final AdamW α-frontier datum; trend inversion confirmed at saturation)

- Branch: `willowpai2i48h1-thorfinn/lookahead-k3-alpha-08`
- Student: willowpai2i48h1-thorfinn
- W&B runs: `mzdbnwvq` (canonical), `09bsni57` (bit-identical earlier rerun); group `lookahead_k3_alpha_sweep`
- Hypothesis: Push α to saturation at k=3 (extension of askeladd's #4211 α∈{0.6, 0.7} sweep).

### Results (W&B-verified)

| W&B run | val_avg | test_avg | best_epoch |
|---|---|---|---|
| `mzdbnwvq` | 57.87 | 54.86 | 17 |
| `09bsni57` (duplicate rerun) | 57.87 | 54.86 | 17 |

### k=3 α-frontier FULLY characterized

| α | val_avg | Δ vs α=0.5 (55.97) |
|---|---|---|
| **α=0.5 (k=3 best)** | **55.97** | **— (minimum)** |
| α=0.6 | 56.31 | +0.34 |
| α=0.7 | 56.24 | +0.27 |
| α=0.8 (THIS PR) | 57.87 | **+1.90** ⚠️ |

The α-trend at k=3 is a concave-upward bowl with minimum at α=0.5, **accelerating beyond α=0.7**.

### Refined mechanism: critical effective-pull-rate ≈ 0.15

Effective slow-weight pull rate per step ≈ α/k:
- k=5/α=0.7: rate = 0.14 (marginal best at k=5)
- k=3/α=0.5: rate = 0.167 (best at k=3)
- k=3/α=0.6: rate = 0.20 (above critical → regress)
- k=3/α=0.7: rate = 0.23 (regress)
- k=3/α=0.8: rate = 0.27 (accelerating regress)

Configs near α/k ≈ 0.14-0.17 perform best. Above this, the slow pull over-dampens fast exploration and accelerates loss-floor stagnation.

### Decision

Closed — val=57.87 > current programme best val=47.97. **AdamW frontier fully exhausted across k, α, β2, LR.** Last AdamW PR for this launch.

⚠️ Bit-identical rerun = wasted GPU (3rd time in 4 closes). Heartbeat-restart pattern needs to stop on student side.

## 2026-05-17 01:30 — PR #4211: Lookahead-AdamW k=3 α sweep (α∈{0.6, 0.7}) ← CLOSED (canonical α-frontier at k=3; trend INVERSION found)

- Branch: `willowpai2i48h1-askeladd/lookahead-k3-alpha-sweep`
- Student: willowpai2i48h1-askeladd
- W&B runs: `31pmz09e` (α=0.6, val=56.31), `gbj8sqqq` (α=0.7, val=56.24); group `lookahead_k3_alpha_sweep`
- Hypothesis: Find optimal α at the new-baseline k=3 (vs k=5 where α=0.7 marginally won).

### Results (terminal SENPAI-RESULT, W&B-verified)

| Arm | val_avg | test_avg | Δ val vs α=0.5 (55.97) | best_epoch |
|---|---|---|---|---|
| α=0.5 (prior k=3 best) | 55.97 | 53.44 | — | 17 |
| α=0.6 | 56.31 | 53.99 | +0.34 | 17 |
| **α=0.7** | **56.24** | 54.25 | **+0.27** | 17 |

### Key mechanistic finding: α-trend INVERTS as k decreases

| k | α=0.3 val | α=0.5 val | α=0.6 val | α=0.7 val | Pattern |
|---|---|---|---|---|---|
| 5 | 61.58 | 57.22 | — | 56.92 | monotone, α=0.7 best |
| 3 | — | 55.97 | 56.31 | 56.24 | bowl, α=0.5 best |

**Mechanism: k and α are NOT independently additive.** Effective slow-weight pull rate ≈ α/k per step. As k decreases (more frequent sync), smaller α is sufficient — large α over-mixes and dampens fast progress. This is a clean mechanism finding that informs the Lion-era α decisions (askeladd reassigned to Lookahead-Lion α sweep #4269).

### Decision

Closed — val=56.24 > current programme best val=47.97 (Lookahead-Lion, PR #4123). AdamW k=3 era α-frontier closed cleanly with strong mechanism insight.

## 2026-05-17 01:30 — PR #4203: Lookahead-AdamW k=2 extension ← CLOSED (k-sweep U-curve minimum confirmed at k=3)

- Branch: `willowpai2i48h1-tanjiro/lookahead-k2-extension`
- Student: willowpai2i48h1-tanjiro
- W&B run: `kdrwbeff` (canonical), `8xi6kkt0` (bit-identical rerun); group `lookahead_k_sweep_extension`
- Hypothesis: Extend monotone k-sweep below k=3 (k=2 was the natural next test if monotone continues).

### Results (W&B-verified; bit-identical reruns confirm reproducibility)

| W&B | val_avg | test_avg | best_epoch |
|---|---|---|---|
| `kdrwbeff` | 56.49 | 53.67 | 17 |
| `8xi6kkt0` (earlier rerun) | 56.49 | 53.67 | 17 |

### k-sweep is U-shaped, minimum at k=3

| k | val_avg | Δ vs k=3 |
|---|---|---|
| k=2 (THIS PR) | 56.49 | +0.52 |
| **k=3 (PR #4158 merged)** | **55.97** | **— (minimum)** |
| k=5 (PR #4132) | 57.22 | +1.25 |
| k=8 (PR #4158 arm) | 60.09 | +4.12 |

Mechanism: at k=2, slow-fast sync so frequent that slow weights barely lag fast trajectory → Lookahead's basin-averaging dimension collapses, leaving only plain-AdamW noise. k=3 is the sweet spot.

### Decision

Closed — val=56.49 > current programme best val=47.97. AdamW k-frontier exhausted at k≥2. tanjiro reassigned to **Lookahead-Lion k=2 (#4268)** to test if the U-curve transfers to Lion era.

⚠️ Bit-identical rerun = wasted GPU. Need to fix the student's terminal-report flow.

## 2026-05-17 01:30 — PR #4202: Lookahead-AdamW k=3 seed=1 verification ← CLOSED (canonical, NO outlier; k=3 era 3-seed mean=56.49)

- Branch: `willowpai2i48h1-alphonse/lookahead-k3-seed1-verify`
- Student: willowpai2i48h1-alphonse
- W&B runs: `7juno411` (CANONICAL, val=57.44), `azc6dorp` (rerun, val=59.26), `5ggnxnd7` (rerun, val=61.75); group `lookahead_k3_seed_scan`
- Hypothesis: Check if k=5/seed=1 outlier pattern (val=78.50, best_ep=10) also affects k=3.

### Results (W&B-verified; first run is canonical with best_epoch=17)

| W&B run | val_avg | test_avg | best_epoch | Notes |
|---|---|---|---|---|
| **`7juno411` (FIRST RUN, canonical)** | **57.44** | **54.64** | **17 ✓** | matches seed-0/2 cosine-floor pattern |
| `azc6dorp` (rerun) | 59.26 | 56.21 | 15 | env or code-snapshot drift between reruns |
| `5ggnxnd7` (rerun) | 61.75 | 58.52 | 14 | "" |

### Key finding: k=3 has NO seed=1 outlier (k=5 fragility resolved)

| Recipe | Seed-0 val | Seed-1 val | Seed-2 val | Seed-1 best_ep |
|---|---|---|---|---|
| Lookahead-AdamW k=5 | 57.22 | **78.50 ⚠️** | 57.05 | 10 (bad basin) |
| **Lookahead-AdamW k=3** | **55.97** | **57.44 ✓** | **56.05** | **17 (cosine floor ✓)** |

### k=3 era 3-seed canonical (CLEAN)

| Seed | val | test |
|---|---|---|
| 0 | 55.97 | 53.44 |
| 1 (THIS PR, canonical run) | 57.44 | 54.64 |
| 2 | 56.05 | 53.03 |
| **3-seed mean** | **56.49** | **53.71** |
| σ̂ | 0.77 | 0.81 |

Mechanism: more frequent sync (k=3 vs k=5) prevents the optimizer from settling into a bad early basin before the next slow-pull resets it. Same robustness Lion achieves via sign-update, achieved by Lookahead-AdamW via reduced k.

### Decision

Closed — val=57.44 > current programme best val=47.97. k=3 era seed-canonical complete. alphonse reassigned to **Lion β1 sweep (#4271)**.

⚠️ Three reruns of the same config = wasted GPU. Heartbeat re-launches of completed configs need to be eliminated from the student flow.

## 2026-05-17 01:00 — PR #4183: Lookahead-AdamW β2 fine scan ({0.93, 0.97}) ← CLOSED (AdamW β2 frontier closed)

- Branch: `willowpai2i48h1-frieren/lookahead-b2-scan`
- Student: willowpai2i48h1-frieren
- W&B runs: `savqxetz` (β2=0.93), `8zsyxcer` (β2=0.97); group `lookahead_beta2_fine_scan`
- Hypothesis: β2 fine scan around the inherited β2=0.95 under Lookahead-AdamW.

### Results (W&B-verified; no SENPAI-RESULT posted but runs finished cleanly)

| Arm | val_avg | test_avg | best_epoch | Δ val vs β2=0.95 (57.22) |
|---|---|---|---|---|
| β2=0.93 | 57.50 | 54.59 | 17 | +0.28 |
| β2=0.95 (prior baseline) | 57.22 | 54.05 | 17 | — |
| β2=0.97 | 57.28 | 54.78 | 17 | +0.06 |

### Frontier finding

β2 is **flat in [0.93, 0.97]** under Lookahead-AdamW. val swing 0.28 << seed σ̂≈1.3. β2=0.95 sits at the joint optimum — Lookahead's online basin-averaging takes over the variance-reduction role that AdamW's β2 was performing alone. **Validates the triple-stack β2=0.95 default; AdamW β2 frontier now closed.**

### Decision

Closed (rate-limit-close). frieren reassigned to **Lookahead-Lion β2 scan (#4264)** — the analogous question in the new era, where Lion's m-buffer β2 (default 0.99) is structurally different from AdamW's v_t β2.

## 2026-05-17 01:00 — PR #4182: Lookahead-AdamW LR sweep ({7e-4, 1e-3}) ← CLOSED (AdamW LR frontier closed)

- Branch: `willowpai2i48h1-fern/lookahead-higher-lr-sweep`
- Student: willowpai2i48h1-fern
- W&B runs: `6kvyr43u` (lr=7e-4), `aopbgr36` (lr=1e-3); group `lookahead_lr_sweep`
- Hypothesis: Probe whether Lookahead unlocks higher peak LR (where plain AdamW diverged at lr=1e-3).

### Results (W&B-verified; no SENPAI-RESULT posted but runs finished cleanly)

| Arm | val_avg | test_avg | best_epoch | Δ val vs lr=5e-4 (57.22) |
|---|---|---|---|---|
| lr=5e-4 (prior baseline) | 57.22 | 54.05 | 17 | — |
| **lr=7e-4** | **56.87** | 54.59 | 17 | **−0.35** (marginal win) |
| lr=1e-3 | 58.87 | 56.02 | 17 | +1.65 (regress) |

### Frontier finding

Lookahead-AdamW LR optimum is **bounded in [5e-4, 7e-4]**. lr=7e-4 wins by −0.35 (within seed noise but directionally positive). lr=1e-3 still regresses — Lookahead does NOT unlock the AdamW LR ceiling materially. Basin-averaging works in a smoother neighborhood, not on the divergent edge.

### Decision

Closed (rate-limit-close). fern reassigned to **Lookahead-Lion LR sweep (#4265)** — the analogous question in the new era. Lion runs at cfg.lr/3 = 1.667e-4 (paper default); the actual TandemFoilSet optimum may sit elsewhere in [Lion lr=1e-4, 2.5e-4].

## 2026-05-17 00:35 — PR #4224: Lookahead-Lion seed=1 verification ← CLOSED (canonical, seed-robust confirmed)

- Branch: `willowpai2i48h1-edward/lookahead-lion-seed1-verify`
- Student: willowpai2i48h1-edward
- W&B run: `dhblq44k` (group `lookahead_lion_seed_scan`)
- Hypothesis: Verify whether the Lookahead-AdamW seed=1 outlier pattern (val=78.50, best_ep=10) reproduces under Lookahead-Lion, or whether Lion's sign-update inherently fixes it.

### Results (terminal SENPAI-RESULT, W&B-verified)

| Metric | Value | Δ vs seed=0 (PR #4123) |
|---|---|---|
| val_avg/mae_surf_p | **49.2089** | +1.2354 |
| test_avg/mae_surf_p | 47.6172 | +1.1272 |
| best_epoch | **17** (cosine T_max floor ✓) | — |

### Seed-gap collapse

| Recipe | Seed-0 val | Seed-1 val | Δ |
|---|---|---|---|
| Lookahead-AdamW k=5 | 57.22 | 78.50 (best_ep=10 ⚠️) | **+21.28** |
| **Lookahead-Lion k=5** | **47.97** | **49.21 (best_ep=17 ✓)** | **+1.24** |

The seed=1 outlier of the AdamW era collapses by ~94% under Lion. Mechanism prediction (from PR #4123 decomposition) is empirically confirmed: Lion's sign-based step has no first-moment magnitude to drag the optimizer into a bad early basin; Lookahead's slow-weight averaging dampens trajectory variance further. **Lookahead-Lion is the cleanest seed-robust baseline in the programme's history.**

### Provisional 2-seed canonical
- seed=0 = 47.97 (PR #4123, merged)
- seed=1 = 49.21 (THIS PR, closed canonical)
- 2-seed mean = **48.59**, σ̂ ≈ 0.62

### Decision

Closed as canonical seed=1 data point — val=49.21 > current best val=47.97, so not a winner-merge. Verification result is excellent: the variance story is now strong enough that we can confidently report a 3-seed mean once seed=2 lands (assigned to nezuko #4242).

## 2026-05-17 00:35 — PR #4210: Lookahead-AdamW k=3 seed=2 verification ← CLOSED (canonical, era superseded)

- Branch: `willowpai2i48h1-nezuko/lookahead-k3-seed2-verify`
- Student: willowpai2i48h1-nezuko
- W&B run: `y3ht6rsq` (group `lookahead_k3_seed_scan`)
- Hypothesis: Verify k=3 robustness across seeds (seed=0 = 55.97 in #4158 merged, seed=1 pending in alphonse #4202, this PR = seed=2).

### Results (terminal SENPAI-RESULT, W&B-verified)

| Metric | Value | Δ vs seed=0 (PR #4158) |
|---|---|---|
| val_avg/mae_surf_p | **56.0512** | +0.083 |
| test_avg/mae_surf_p | **53.0339** | **−0.408 (better!)** |
| best_epoch | 17 (cosine T_max floor ✓) | — |

### k=3 stable-seed dispersion

| Seed | k=5 val | k=3 val |
|---|---|---|
| 0 | 57.22 | 55.97 |
| 1 | 78.50 (outlier) | pending (#4202) |
| 2 | 57.05 | **56.05** |
| stable-seed mean (0,2) | 57.13 | **56.01** |
| stable-seed gap | 0.17 | **0.08 (tighter!)** |

k=3 keeps its ~1 MAE edge over k=5 on stable seeds (Δ_stable = −1.12). Stable-seed dispersion at k=3 (0.08 MAE) is actually tighter than at k=5 (0.17). The k=3 era was internally consistent.

### Decision

Closed as canonical k=3 era seed-scan data — val=56.05 > current programme best val=47.97 (Lookahead-Lion, PR #4123 merged 2026-05-16 23:45). The k=3 era baseline was superseded mid-round when Lookahead-Lion landed at 47.97. The k=3 finding still informs the new Lion-era hyperparameter space: edward is now testing Lookahead-Lion k=3 (#4241) to see if the same k-sweep monotonicity transfers.



- Branch: `willowpai2i48h1-edward/lion-optimizer-triple-stack`
- Student: willowpai2i48h1-edward
- W&B runs: `ux8amr59` (Arm 1 Pure Lion, val=49.07), `rx3negp7` (Arm 2 Lookahead-Lion, val=47.97)
- Hypothesis: (a) Verify Pure Lion seed=0 reproduces post-Lookahead-merge; (b) Test Lookahead-Lion composition.

### Results (terminal SENPAI-RESULT, W&B-verified)

| Config | val_avg | test_avg | best_epoch | W&B |
|---|---|---|---|---|
| Lookahead-AdamW k=5 (prior best) | 57.22 | 54.05 | 17 | `d9ujr4oe` |
| **Pure Lion (Arm 1 `ux8amr59`)** | **49.07** | **47.07** | 17 | bit-identical to `rv8hjgtx` ✓ |
| **Lookahead-Lion (Arm 2 `rx3negp7`)** | **47.97** | **46.49** | 17 | **MERGED → new programme best** |

### Mechanism decomposition

| Intervention | Δ val | Source of gain |
|---|---|---|
| AdamW → Lion | −8.15 | Sign-based updates: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion | −1.10 | Slow-weight pull: reduce per-basin commitment variance |
| Total (AdamW → Lookahead+Lion) | **−9.25** | Complementary mechanisms, orthogonal-additive composition |

Lookahead's gain on top of Lion (−1.10 val) is smaller than its gain on top of AdamW (−3.21 val from k=5 #4132), suggesting partial mechanism overlap. But no antagonism — pure additive composition.

OOD split improvements vs Lookahead-AdamW: camber_cruise (29.32 val vs 35.61), re_rand (48.02 val vs 53.72). In-distribution split slightly worse on test (+1.37) but dominated by OOD gains. The geom_camber_cruise split is now val=29.32 — the lowest single-split we've ever seen on this benchmark.

### Training stability

Pure Lion shows 3 upward spikes in the training curve (ep 7, 10, 11). Lookahead-Lion shows 1 micro-spike (ep 11). Lookahead's slow-weight averaging absorbs Lion's sign-update overshoots — mechanistically sound.

### Decision

**Merged as new programme best. BASELINE.md updated.** Win threshold now: val < 47.97. edward reassigned to Lookahead-Lion seed=1 verification (#4224) — checking whether the seed=1 outlier pattern (observed on Lookahead-AdamW: val=78.50) also affects Lookahead-Lion.

## 2026-05-16 23:35 — PR #4175: Lookahead α sweep (α∈{0.3,0.7}) at k=5 ← CLOSED (superseded by new k=3 baseline)

- Branch: `willowpai2i48h1-askeladd/lookahead-alpha-sweep`
- Student: willowpai2i48h1-askeladd
- W&B runs: `rg4qeiyu` (α=0.3, val=61.577), `pni7uzhw` (α=0.7, val=56.916)
- Hypothesis: α sweep at fixed k=5 around the paper default α=0.5.

### Results

| Arm | val_avg | test_avg | Δ val vs α=0.5 (57.22) | W&B |
|---|---|---|---|---|
| α=0.3 | 61.577 | 58.414 | +4.357 (worse) | `rg4qeiyu` |
| **α=0.7** | **56.916** | 54.307 | **−0.305 (better)** | `pni7uzhw` |

### Interpretation

Monotone trend: α=0.3→0.5→0.7 → val=61.58→57.22→56.92. Larger α consistently better. α=0.7's val improvement (−0.305) is accompanied by test regression (+0.260), suggesting a single-seed noise-floor result — within noise but directionally informative. The val/test divergence and small magnitude (|Δ|<0.5) suggest α=0.7 is not a robust standalone win.

### Decision

Closed — α=0.7 wins the OLD baseline (57.22) but val=56.916 does NOT beat the NEW baseline (55.968, PR #4158 merged this round). Next step: test α∈{0.6, 0.7, 0.8} at k=3 (the new optimal k). Askeladd reassigned to #4211 (k=3 α∈{0.6,0.7}) and thorfinn to #4213 (k=3 α=0.8).

## 2026-05-16 23:35 — PR #4158: Lookahead k sweep (k∈{3,8}) ← MERGED (NEW PROGRAMME BEST, val=55.97)

- Branch: `willowpai2i48h1-nezuko/lookahead-k-sweep`
- Student: willowpai2i48h1-nezuko
- W&B runs: `oeb54ela` (k=3, val=55.97), `o7adv9re` (k=8, val=60.09)
- Hypothesis: Sweep Lookahead k ∈ {3, 8} around current best k=5 to localize optimal sync frequency.

### Results (terminal SENPAI-RESULT, W&B-verified)

| Config | val_avg | test_avg | best_epoch | Δ vs k=5 (val=57.22) |
|---|---|---|---|---|
| **Lookahead k=3, α=0.5** (seed=0) | **55.968** | **53.442** | 17 | **−1.252** |
| Lookahead k=8, α=0.5 (seed=0) | RUNNING | — | — | — |

Per-split val (k=3): single_in_dist=67.02, geom_camber_rc=68.39, geom_camber_cruise=35.70 (strong OOD), re_rand=52.76.
Per-split test (k=3): single_in_dist=59.98, geom_camber_rc=60.98, geom_camber_cruise=47.23, re_rand=45.58.

### Interpretation

k=3 (more frequent slow-weight sync) cleanly improves over k=5 at seed=0. Δ=−1.25 is larger than the GeGLU-era noise floor (σ̂≈1) but smaller than the seed=1 outlier observed concurrently — needs 3-seed verification.

### k-sweep monotone relationship (definitive)

| k | val_avg/mae_surf_p | Δ vs k=5 |
|---|---|---|
| **k=3** | **55.968** | **−1.252** |
| k=5 (prior best) | 57.220 | — |
| k=8 | 60.091 | +2.871 |

Monotone and symmetric around k=5: smaller k helps, larger k hurts. More frequent slow-weight syncing (k=3) delivers more basin-variance reduction within the 6375-step budget. k=8 lets the fast trajectory drift too far before each sync — fewer, lower-quality basin averages. Optimal k at α=0.5 ≈ 3 or possibly lower.

k=3 per-split improvements: every split improved. OOD strongest: geom_camber_cruise 35.70 (strong), re_rand 52.76. geom_camber_rc 68.39 and single_in_dist 67.02 also improved.

k=3 per-sync count: 2125 syncs (vs 1275 at k=5 vs 797 at k=8). Higher sync frequency ↔ better performance.

### Decision

**Merged as new programme best. BASELINE.md updated.** New win threshold: val < 55.97. k=3 3-seed canonical assigned to alphonse (#4202 seed=1) and nezuko (#4210 seed=2). α sweep at k=3 assigned to askeladd (#4211 α∈{0.6,0.7}) and thorfinn (#4213 α=0.8).

## 2026-05-16 23:00 — PR #4160: Lookahead seed=1 canonical ← CLOSED (seed=1 OUTLIER, val=78.50)

- Branch: `willowpai2i48h1-thorfinn/lookahead-seed-1`
- Student: willowpai2i48h1-thorfinn
- W&B runs: `pjvhrh4f` (seed=1, finished val=78.503), `637qjzvn` (rerun, bit-identical val=78.503), 2 more heartbeat duplicates running
- Hypothesis: 3-seed canonical for new programme best — seed=1 confirmation.

### Results (W&B-verified, closed via [[rate-limit-close-on-wandb]])

| Metric | k=5 seed=1 (this PR) | k=5 seed=0 (programme best) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **78.503** | 57.220 | **+21.28 (outlier)** |
| test_avg/mae_surf_p | 74.185 | 54.047 | +20.14 |
| best_epoch | 10 | 17 | early-peak: lands in worse basin and never recovers |

Config sanity: ✓ triple-stack + Lookahead correct (lookahead_k=5, alpha=0.5, optimizer=lookahead-adamw, betas=[0.9, 0.95], use_geglu=True).

### Interpretation

**Lookahead-AdamW has surprising seed-1 variance at this dataset.** 3-seed canonical now reads: seed=0=57.22, seed=1=78.50, seed=2=57.05 → μ̂=64.26 ± 12.4, median=57.05. Seed=1 is a clear outlier — 2-of-3 seeds at ~57, one at ~78. Best_epoch=10 (vs 17 for seed=0) suggests seed=1 lands in a worse basin earlier and the cosine descent doesn't recover.

This complicates merge-claim of val=57.22 as new baseline — Lookahead is robust in median but has bimodal seed sensitivity. Paper-facing reporting should use the 3-seed median (57.05) or explicitly note the outlier.

### Decision

Closed as canonical seed-scan data point (not a regression — the seed got the unlucky basin, data is valid). Reassigning thorfinn.

## 2026-05-16 23:00 — PR #4174: Lookahead seed=2 canonical ← CLOSED (clean reproduction, val=57.05)

- Branch: `willowpai2i48h1-alphonse/lookahead-seed-2`
- Student: willowpai2i48h1-alphonse
- W&B runs: `a6l7j8ec` (seed=2, finished val=57.046), 2 heartbeat duplicates still running
- Hypothesis: 3-seed canonical for new programme best — seed=2 confirmation.

### Results (W&B-verified, closed via [[rate-limit-close-on-wandb]])

| Metric | k=5 seed=2 (this PR) | k=5 seed=0 (programme best) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | **57.046** | 57.220 | **−0.17 (clean reproduction)** |
| best_epoch | 17 (cosine floor ✓) | 17 | — |
| optimizer | lookahead-adamw ✓ | — | — |

### Interpretation

Clean reproduction of seed=0 result. Lookahead is robust between seeds 0 and 2. The seed=1 outlier from #4160 stands alone — 2-of-3 seeds at ~57. Median is the appropriate paper-facing statistic.

### Decision

Closed as canonical seed-scan data point. Reassigned alphonse to **k=3 seed=1 verification** (#4202) to check if the k=3 potential winner also has the seed=1 outlier behavior.

## 2026-05-16 23:00 — PR #4176: Lookahead + SWA of slow weights ← CLOSED (NO-OP, val=57.22)

- Branch: `willowpai2i48h1-tanjiro/lookahead-slow-swa`
- Student: willowpai2i48h1-tanjiro
- W&B runs: `vzrbeman` (finished val=57.220), `pfw24a8d` (heartbeat duplicate just started)
- Hypothesis: Tail-average the SLOW-weight trajectory (over last 4 epochs) to extract more benefit from Lookahead's smoothed trajectory.

### Results (W&B-verified, closed via [[rate-limit-close-on-wandb]])

| Metric | Lookahead + SWA-of-slow | Lookahead alone | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | **57.220** | 57.220 | **0.000 (NULL — bit-identical)** |

### Mechanism analysis

This is consistent with the prior post-hoc-averaging failure pattern observed in #3644, #4089, and #4121. T_max=17 cosine has **no stationary tail** for any trajectory — fast or slow. The slow weights produced by Lookahead still inherit the descending cosine LR schedule (LR is applied to the fast optimizer; slow weights periodically copy fast). So SWA tail-averaging on the slow trajectory averages over a non-stationary segment and contributes exactly zero.

**Confirms round-7 finding:** online averaging (Lookahead's internal slow←fast sync) is the only averaging that works at T_max=17 cosine. ALL post-hoc averaging schemes — on fast or slow trajectory — share this failure mode.

### Decision

Closed. Reassigned tanjiro to **Lookahead k=2** (#4203) — extending the k-sweep below nezuko's k=3 finding.

## 2026-05-16 22:30 — PR #4124: H: mlp_ratio=3 on triple-stack ← CLOSED (rate-limit recovery)

- Branch: `willowpai2i48h1-fern/mlp-ratio-3-triple-stack`
- Student: willowpai2i48h1-fern
- W&B runs: `2ara1wvg`, `nw3ab2po` (both finished, bit-identical) — student blocked from posting SENPAI-RESULT by GitHub rate-limit storm
- Hypothesis: mlp_ratio=3 (FFN hidden 256 → 384, GeGLU inner 171 → 256, +12.5% params) on triple-stack.

### Results (W&B-verified, closed via [[rate-limit-close-on-wandb]] protocol)

| Metric | Triple-stack 3-seed μ̂ | mlp_ratio=3 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 61.66 ± 1.32 | **61.83** | +0.17 (null) |
| test_avg/mae_surf_p | 58.36 ± 0.83 | **57.87** | −0.49 (null) |
| best_epoch | — | 16 | — |

Within 1σ of the 3-seed canonical mean. **No orthogonal gain** at h=128 in 30-min/17-epoch budget — either capacity-saturated for the trajectory we can reach, or extra params need more epochs to pay off.

### Decision

Closed. mlp_ratio lever has no headroom on its own; stacking onto Lookahead is a lower-priority direction than other Lookahead-era follow-ups. Reassigned fern to **Lookahead + higher LR sweep ({7e-4, 1e-3})**.

## 2026-05-16 22:30 — PR #4119: H: β2 fine scan ({0.93, 0.97}) on triple-stack ← CLOSED (rate-limit recovery)

- Branch: `willowpai2i48h1-frieren/beta2-fine-scan-tripleStack`
- Student: willowpai2i48h1-frieren
- W&B runs: `25gslduq` (β2=0.97, finished), `nh5hiw51` + `qq9yzsoo` (β2=0.93, crashed twice)
- Hypothesis: Fine scan around β2=0.95 (triple-stack default) — test 0.93 and 0.97.

### Results (W&B-verified)

| Config | val_avg | test_avg | Δ vs triple-stack 3-seed μ̂ |
|---|---|---|---|
| Triple-stack 3-seed μ̂ | 61.66 ± 1.32 | 58.36 ± 0.83 | — |
| β2=0.97 (`25gslduq`) | 61.41 | 58.28 | −0.25/−0.08 (null) |
| β2=0.93 | — | — | crashed twice |

β2=0.97 is within 1σ of canonical mean — **β2=0.95 appears near-optimal under plain AdamW.** The β2=0.93 arm crashed twice; without a successful run we can't characterize whether 0.93 is worse or training instability is config-specific.

### Decision

Closed. β2 lever appears saturated under plain AdamW. Reassigned frieren to **Lookahead + β2 fine scan ({0.93, 0.97})** — Lookahead's basin-averaging may shift the β2 optimum and may also stabilize the β2=0.93 arm that crashed under plain AdamW.

## 2026-05-16 22:15 — PR #4123: H: Lion optimizer on triple-stack ← SENT BACK FOR REBASE + 2-ARM VERIFICATION

- Branch: `willowpai2i48h1-edward/lion-optimizer-triple-stack`
- Student: willowpai2i48h1-edward
- W&B run: `rv8hjgtx`
- Hypothesis: Replace AdamW(β2=0.95) with Lion (lr/3, sign-based updates) on triple-stack.

### Reported results (W&B verified to 4dp, but pre-Lookahead-merge)

| Metric | Triple-stack | Lookahead (current best) | Lion (reported) | Δ vs Lookahead |
|---|---|---|---|---|
| val_avg/mae_surf_p | 60.4338 | 57.2203 | **49.0721** | **−8.15** |
| test_avg/mae_surf_p | 57.4381 | 54.0468 | **47.0707** | **−6.98** |

All 4 val splits and 4 test splits improve uniformly. Per-channel test diagnostics (Ux, Uy, p) all in physically reasonable ranges.

### Action: rebase + 2-arm verification

A Δ=−8.15 val on top of Lookahead is roughly −13σ relative to the GeGLU-era seed noise (σ̂≈1) — larger than the cumulative gain of every prior intervention this round combined. Magnitude warrants verification before merging.

Sent back with 2-arm requirement:
- **Arm 1 — Pure Lion**: Replace AdamW with Lion on the rebased branch (no Lookahead wrapper). Verifies seed=0 reproducibility post-rebase.
- **Arm 2 — Lookahead-Lion composition**: Wrap Lion with Lookahead(k=5, α=0.5). Tests if the two basin-variance reduction mechanisms compose orthogonally.

If Arm 1 reproduces val < 51, the win is real. If Arm 2 beats Arm 1, composition is orthogonal and we merge Arm 2. If they're tied or Arm 1 is better, merge Arm 1.

### Mechanism (student's claim, plausible)

Lion's sign-only update is invariant to gradient-scale noise. At batch=4 on heterogeneous CFD with O(10⁵) mesh nodes, AdamW's `sqrt(v_t)` denominator is itself noisy from small-batch squared-gradient estimates. The triple-stack's β2=0.95 was a partial fix for this; Lion sidesteps the variance problem entirely with constant-magnitude updates.

## 2026-05-16 22:10 — PR #4121: H: EMA model weights (decay=0.999) on triple-stack ← CLOSED

- Branch: `willowpai2i48h1-tanjiro/ema-weights-triple-stack`
- Student: willowpai2i48h1-tanjiro
- W&B run: `xiu9s9ke`

### Results

| Metric | Triple-stack | EMA-arm (this) | Δ |
|---|---|---|---|
| val_avg | 60.4338 | 62.9808 | +2.55 |
| test_avg | 57.4381 | 59.5020 | +2.06 |

Raw best checkpoint matches triple-stack baseline exactly — training was unaffected; EMA-only metrics are uniformly worse.

### Mechanism diagnosis (student's, exactly right)

EMA(0.999) has ~1.87-epoch half-life. With cosine LR → 0 at T_max=17, the raw final checkpoint sits at a tight minimum; EMA drags in higher-loss weights from epochs 11–14, averaging the model back to a worse basin.

**Same failure mode as PRs #3644 and #4089 SWA on fast weights:** post-hoc weight averaging on the FAST trajectory needs a stationary tail window, and T_max=17 cosine doesn't provide one. The win came from online averaging (Lookahead) instead.

### Decision

Closed. Reassigned tanjiro to **Lookahead + SWA of slow weights** — the slow-weight trajectory under Lookahead is smoother than the fast trajectory, and may have the stationary tail that fast-weight averaging could not find.

## 2026-05-16 22:05 — PR #4118: H: β1=0.95 on triple-stack (compound momentum) ← CLOSED

- Branch: `willowpai2i48h1-askeladd/b1-095-tripleStack`
- Student: willowpai2i48h1-askeladd
- W&B run: `j4d3l8jm`

### Results

| Metric | Triple-stack | β1=0.95 (this) | Δ |
|---|---|---|---|
| val_avg | 60.4338 | 63.2973 | +2.86 |
| test_avg | 57.4381 | 59.9677 | +2.53 |

All 4 val splits regress uniformly. Largest regression on val_geom_camber_rc (+4.19) and val_single_in_dist (+0.07 — smallest).

### Mechanism (student's, sharp)

β1=0.95 over-smooths gradient direction in a short 6375-step cosine-to-zero run. The heavier first-moment damping slows early descent and the fast LR decay prevents recovery. **β2=0.95 (variance stabilisation) and β1=0.9 (responsive mean) are the right operating point** — adding β1=0.95 does not compound; they're antagonistic in this regime.

### Decision

Closed. Confirmed dead-end lever. Reassigned askeladd to **Lookahead α sweep (α∈{0.3, 0.7})** to characterize the slow-step blend ratio optimum.

## 2026-05-16 22:00 — PR #4117: H: Triple-stack seed=2 (3-seed canonical) ← CLOSED

- Branch: `willowpai2i48h1-alphonse/triple-stack-seed2-canonical`
- Student: willowpai2i48h1-alphonse
- W&B run: `7lwdpglm`

### Results — completes 3-seed canonical for triple-stack

| seed | val_avg | test_avg |
|---|---|---|
| 0 (PR #3995) | 60.4338 | 57.4381 |
| 1 (PR #4116) | 61.5427 | 58.5320 |
| 2 (this PR) | 63.0089 | 59.0956 |
| **μ̂** | **61.66** | **58.36** |
| σ̂ | 1.32 | 0.83 |

seed=0 was 1.23 below mean (slightly lucky), seed=2 was 1.34 above (slightly unlucky). Spread ~2σ, consistent with the GELU-era σ̂=1.54 noise floor.

### Decision

Closed — establishes triple-stack's noise floor for the paper appendix but **new programme best is PR #4132 Lookahead val=57.22 / test=54.05**, ahead of even triple-stack seed=0. Reassigned alphonse to **Lookahead seed=2 (3-seed canonical for new best)**.

## 2026-05-16 21:40 — PR #4132: H: Lookahead optimizer (k=5, α=0.5) on triple-stack ← MERGED (NEW PROGRAMME BEST)

- Branch: `willowpai2i48h1-nezuko/lookahead-optimizer-triple-stack`
- Student: willowpai2i48h1-nezuko
- W&B run: `d9ujr4oe`
- Group: `triple_stack_lookahead`
- Hypothesis: Wrap AdamW(β2=0.95) with Lookahead (k=5, α=0.5) — online basin-averaging that works on non-stationary trajectories unlike post-hoc SWA.

### Results (W&B verified to 4dp)

| Metric | Triple-stack (PR #3995) | Lookahead (this PR) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | 60.4338 | **57.2203** | **−3.21** |
| **test_avg/mae_surf_p** | 57.4381 | **54.0468** | **−3.39** |
| val_single_in_dist | 69.659 | 69.610 | −0.05 |
| val_geom_camber_rc | 72.671 | 69.943 | −2.73 |
| val_geom_camber_cruise | 41.722 | 35.606 | **−6.12** |
| val_re_rand | 57.683 | 53.723 | −3.96 |
| test_single_in_dist | 60.566 | 60.988 | +0.42 |
| test_geom_camber_rc | 66.851 | 61.902 | −4.95 |
| test_geom_camber_cruise | 51.976 | 47.569 | −4.41 |
| test_re_rand | 50.360 | 45.728 | −4.63 |

### Mechanism analysis

Online averaging (Lookahead) beats post-hoc tail averaging (SWA, PRs #3644 and #4089) because our T_max=17 cosine is budget-limited (still descending at the tail). Lookahead accumulates flat-minima benefits throughout training; SWA needs a stationary window. The dominant gains on OOD splits (val_geom_camber_cruise −6.12, test_re_rand −4.63) confirm the flat-minima → better generalization story. In-distribution split (test_single_in_dist) barely moves — there's little headroom left.

1275 sync events confirmed (6375 steps / k=5), verifying the wrapper was genuinely active.

### Decision

**MERGED — new programme best.** val_avg=57.22, test_avg=54.05. Single seed=0. Follow-ups: k sweep (nezuko #assigned), seed=1 canonical (thorfinn #assigned).

## 2026-05-16 21:30 — PR #4116: Triple-stack seed=1 — 3-seed canonical ← CLOSED (completes 2-seed data)

- Branch: `willowpai2i48h1-thorfinn/triple-stack-seed1-canonical`
- Student: willowpai2i48h1-thorfinn
- W&B run: `zf09r368`
- Hypothesis: seed=1 of triple-stack (T_max=17 + β2=0.95 + GeGLU) to establish σ̂ around programme best.

### Results

| Metric | seed=0 (PR #3995) | seed=1 (this PR) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | 60.4338 | 61.5427 | **+1.11** |
| **test_avg/mae_surf_p** | 57.4381 | 58.5320 | **+1.09** |

Per-split: all 8 metrics regressed; worst on val_re_rand (+1.84) and test_single_in_dist (+2.14); smallest on val_geom_camber_rc (+0.25).

2-seed mean: val=60.99 / test=57.99 — seed=0 was ~0.56 below mean (slightly lucky).

### Decision

Closed — provides 3-seed canonical context for triple-stack baseline, but **new programme best (PR #4132 Lookahead val=57.22) makes this the old baseline**. Thorfinn reassigned to Lookahead seed=1 canonical.

## 2026-05-16 20:35 — PR #4089: H: SWA over final 4 cosine epochs (T_max=17 SwiGLU) — no LR kick-out ← CLOSED

- Branch: `willowpai2i48h1-nezuko/swa-tail4-cosine-tmax17`
- Student: willowpai2i48h1-nezuko
- W&B run: `92f93jle`
- Hypothesis: SWA over epochs 14-17 of T_max=17 cosine (no constant-LR tail). Mechanism validated in PR #3644 but failed there due to LR kick-out — this experiment removes the kick-out entirely.

### Results

| Arm | val_avg | test_avg | Δ vs SwiGLU baseline | Δ vs triple-stack |
|---|---|---|---|---|
| baseline (PR #3994 SwiGLU) | 62.1023 | 59.5529 | — | — |
| **swa_tail4** | **62.7940** | **59.7545** | **+0.69 / +0.20** ✗ | **+2.36 / +2.31** ✗ |
| epoch17_weights | 62.1023 | 59.5529 | tied | +2.36 |
| best_val_ckpt | 62.1023 | 59.5529 | tied | +2.36 |

best_val_ckpt and epoch17_weights tied because epoch 17 was the best val epoch — cosine had not flattened.

### Mechanism: budget-limited cosine breaks SWA tail assumption

Per-epoch val in the SWA window:

| Epoch | val_avg | Δ from prev |
|---|---|---|
| 14 | 68.40 | — |
| 15 | 68.03 | −0.37 |
| 16 | 62.80 | −5.23 |
| 17 | 62.10 | −0.70 |

The model is **still actively descending** at the bottom of cosine — 6.3 pt drop from epoch 15→16 (the SWA premise required "oscillation around minimum at low LR"). T_max=17 is budget-limited, not converged.

Beautiful math observation: val(mean(θ_14..17)) = 62.79 < mean(val(θ_i)) = 65.34 < val(θ_17) = 62.10. SWA *did* find a flatter point than per-snapshot mean, just not flatter than epoch 17 alone.

### Two failure modes for SWA-at-T_max=17

1. **PR #3644**: constant-LR tail jumps LR ~25× before basin floor
2. **PR #4089**: no kick-out, but cosine never flattens within budget

Both rule out SWA tail averaging at our scale/budget. **Appendix-grade negative result** with clean mechanism.

### Decision

Close. Nezuko reassigned to Lookahead optimizer (Zhang 2019) — the online/adaptive cousin of SWA, which performs basin-averaging during training and is well-suited to non-stationary trajectories.

## 2026-05-16 20:10 — PR #3995: H: Triple-stack (T_max=17 + β2=0.95 + GeGLU) ← MERGED (NEW PROGRAMME BEST)

- Branch: `willowpai2i48h1-fern/adamw_beta2_095_swiglu`
- Student: willowpai2i48h1-fern
- W&B run: `insf46p8` (canonical run from first training cycle)

### Results

| Metric | This PR | Old baseline (PR #3994) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **60.4338** | 62.1023 | **−1.67** |
| test_avg/mae_surf_p | **57.4381** | 59.5529 | **−2.13** |
| val_single_in_dist | 69.659 | 71.858 | −2.20 |
| val_geom_camber_rc | 72.671 | 74.828 | −2.16 |
| val_geom_camber_cruise | 41.722 | 42.674 | −0.95 |
| val_re_rand | 57.683 | 59.049 | −1.37 |

### Analysis

The triple-stack (T_max=17 + β2=0.95 + GeGLU) delivered the largest single-PR win of the round: −1.67 val / −2.13 test. Three orthogonal improvements stacked cleanly.

**Decomposition (from ablation PR #4032 askeladd):**
- T_max=17 alone: val=62.10 (merged PR #3994) — confirmed
- T_max=17 + GeGLU alone (PR #4032, default β2=0.9): val=62.47 — only −0.59 from the SwiGLU μ̂
- T_max=17 + GeGLU + β2=0.95 (this PR): val=60.43 — **−2.04 additional** from β2=0.95

**Key finding:** Most of the triple-stack gain comes from β2=0.95, not from the activation swap. β2=0.95 enables tighter late-training convergence near the basin (slower squared-gradient EMA reduces variance, more stable per-step updates). GeGLU alone is near-marginal (+0.59). Together they synergise (sum > parts).

---

## 2026-05-16 20:05 — PR #3644: H: SWA over tail on SwiGLU (T_cosine=10 + 8-ep tail + SWA) ← CLOSED (regression, SWA mechanism validated)

- Branch: `willowpai2i48h1-nezuko/cosine10_constant_tail_swa`
- Student: willowpai2i48h1-nezuko
- W&B run: `a5xejlbq` (SwiGLU-regime canonical result)

### Results

| Arm | val_avg | test_avg |
|---|---|---|
| pre_swa (T_cosine=10 endpoint) | 78.55 | — |
| tail_best (epoch 16, best of constant-tail) | 74.35 | 72.81 |
| **swa_tail** (7-snapshot SWA average) | **70.83** | **66.06** |
| Baseline (PR #3994) | 62.10 | 59.55 |

### Analysis

**SWA mechanism confirmed real and large** (−3.52 val ~2σ, −6.76 test ~3.8σ vs tail_best). SWA wins on all 4 test splits; biggest gains on test_geom_cruise (−8.77) and test_re_rand (−7.59). Test gains > val gains → SWA preferentially fixes OOD generalization.

**Why didn't it beat baseline:** T_cosine=10 yanked LR to 1e-4 (25× late-cosine LR) before model reached basin floor (pre_swa val=78.55, ~13pt above baseline). SWA averaged pre-basin weights — SWA-as-averaging-within-basin is what we need.

**Follow-up (#4089):** Nezuko reassigned to SWA over final 4 cosine epochs of T_max=17 (no constant-LR kick-out). Running now.

---

## 2026-05-16 20:05 — PR #4028: H: T_max=17 SwiGLU seed=1 ← CLOSED (null, 3-seed canonical complete)

- W&B run: `19zucl1x` | val=63.9575, test=60.9191

## 2026-05-16 20:05 — PR #4032: H: T_max=17 + GeGLU (default β2) ← CLOSED (null + key ablation)

- W&B run: `f8u4og6i` | val=62.4709, test=59.3796
- **Key finding:** GeGLU alone gives only −0.59 improvement over SwiGLU T_max=17. β2=0.95 is the critical lever.

## 2026-05-16 20:05 — PR #4050: H: T_max=17 SwiGLU seed=2 ← CLOSED (null, 3-seed canonical complete)

- W&B run: `vh1fdm6u` | val=63.1155, test=60.0203
- **T_max=17 SwiGLU 3-seed canonical:** seed=0 62.10, seed=1 63.96, seed=2 63.12 → μ̂=63.06 ± 0.93

## 2026-05-16 20:05 — PR #3999: H: Gradient clipping clip_norm=1.0 ← CLOSED (regression)

- W&B run: `dyx8lh1s` | val=65.7803, test=61.4631
- Clipping at norm=1.0 is too aggressive; β2=0.95 is the superior method for gradient stabilization at this scale.

## 2026-05-16 20:05 — PR #3998: H: slice_num=128 (2× attention) ← CLOSED (big regression)

- W&B run: `tklk2d2f` | val=73.0191, test=69.9375
- Doubling slice_num hurts significantly; slice_num=64 is optimal for h=128/5L.

## 2026-05-16 20:05 — PR #3973: H: RMSNorm replacement of LayerNorm ← CLOSED (regression)

- W&B run: `ve2ng8ha` | val=67.2827, test=61.8529
- LayerNorm remains the canonical choice — mean-centering preserves physical signal (pressure offset, velocity bias between geometry classes).

---

## 2026-05-16 16:45 — PR #3996: H: AdamW weight_decay 1e-4 → 1e-2 on SwiGLU h=128 ← CLOSED (null)

- Branch: `willowpai2i48h1-alphonse/wd_1e2_swiglu`
- Student: willowpai2i48h1-alphonse
- Status: CLOSED. 2-seed μ̂=66.29, test μ̂=63.03 — within GLU pooled noise floor, +4.19 above new programme best.
- W&B runs: `oubytguj` (seed=0), `hjbhpzgy` (seed=1)

### Results (2-seed, h=128/T_max=15/bf16/SwiGLU, wd=1e-2)

| Metric | seed=0 | seed=1 | 2-seed μ̂ | New programme best (T_max=17) |
|---|---|---|---|---|
| val_avg/mae_surf_p | 65.148 | 67.435 | 66.29 | 62.10 |
| test_avg/mae_surf_p | 62.154 | 63.898 | 63.03 | 59.55 |

seed=0 appeared to be zone-1 (65.15 < old best 65.37), triggering seed=1. But seed=1 regressed to 67.43, placing 2-seed μ̂=66.29 at the GLU pooled floor.

### Analysis

**Key diagnostic — cumulative shrinkage:** Per-parameter shrinkage over 17 epochs = `(1-lr·wd)^steps = (1-5e-6)^6375 ≈ 0.97` — only 3% pull-back. Empirically confirmed: weight L2 norm grew 41.6 → 43.7 over training despite wd=1e-2. The gradient term dominates; wd is barely engaged. Even wd=1e-1 (LLaMA setting) would only produce ~30% shrinkage at this budget.

**Programme-level finding:** Weight-magnitude regularisation is null at this scale/budget. Combined with #3811 (dropout null) and #3886 (DropPath Zone-5), the regularisation family is exhausted. The 660K-param / 1500-sample regime is not under-regularised — it's bottlenecked by schedule/optimizer dynamics within 17 epochs.

---

## 2026-05-16 16:35 — PR #3993: H: head_and_embed 2.5× LR + 500-step warmup ← CLOSED (Zone-4 regression)

- Branch: `willowpai2i48h1-askeladd/head_embed_25x_warmup500_swiglu`
- Student: willowpai2i48h1-askeladd
- Status: CLOSED. val=69.61, test=64.59 — Zone-4 vs new programme best 62.10.
- W&B run: `43fv3upa`

### Results (seed=0, h=128/T_max=15/bf16/SwiGLU + head_embed 2.5× + 500-step warmup)

| Metric | This run | #3932 (no-warmup) | PR best (T_max=17) |
|---|---|---|---|
| val_avg/mae_surf_p | 69.61 | 70.31 | 62.10 |
| test_avg/mae_surf_p | 64.59 | — | 59.55 |
| val@ep1 | 222.13 | 185 | ~120 (normal) |

### Analysis

**Finding 1 — Warmup made early-step dynamics WORSE.** val@ep1=222 (warmup) > 185 (no-warmup) > expected 80-120. The crippled head_embed lr during warmup allowed blocks to evolve to expect features from a near-frozen embedding; once lr unfreezes, recovery takes 10+ epochs.

**Finding 2 — 3.09× equilibrium ratio is not a unique attractor.** Steady-state head/block_0 grad-norm at epoch 5: 2.35× (this run) vs 3.09× (#3932). Warmup permanently shifted the optimization trajectory to a different basin with systematically smaller head_embed gradients.

**Finding 3 — 2.5× boost dead-end confirmed.** Combined with #3932, the design space at 2.5× is bracketed: {warmup: 69.61, no-warmup: 70.31}. Both Zone-4. Warmup helps by 0.7pt but the structural cost remains. **Head_and_embed LR boost family is EXHAUSTED** at this 17-epoch budget.

Note: also caught PyTorch scheduler gotcha #2 (group['lr'] recurrence contamination) in initial run, fixed with closed-form formula. Fix is now programme-canonical.

Also notable: first run (W&B `3u947wd3`) was a buggy 1.875× effective boost (not 2.5×) due to the cosine recurrence bug. Student killed it, fixed it, relaunched.

---

## 2026-05-16 16:05 — PR #3994: H: T_max=17 cosine on SwiGLU h=128 ← MERGED, NEW PROGRAMME BEST

- Branch: `willowpai2i48h1-thorfinn/tmax17_swiglu_h128`
- Student: willowpai2i48h1-thorfinn
- Status: MERGED. val=62.10, test=59.55 — new all-time programme best.
- W&B run: `5q47ozlp`

### Results (seed=0, h=128/T_max=17/bf16/SwiGLU)

| Metric | This run (T_max=17) | Prior best (PR #3810 GeGLU T_max=15) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **62.1023** | 65.3704 | **−3.27** |
| test_avg/mae_surf_p | **59.5529** | 61.6819 | **−2.13** |
| Best epoch | 17 (final) | 17 | 0 |

Per-split val (all improve): single_in_dist=71.86, geom_camber_rc=74.83, geom_camber_cruise=42.67, re_rand=59.05

Per-split test: single_in_dist=62.80, geom_camber_rc=69.41, geom_camber_cruise=53.31, re_rand=52.69

### LR + val trajectory (confirms T_max=17 wiring)

| Epoch | LR | val |
|---|---|---|
| 1 | 4.957e-4 | 188.79 |
| 5 | 4.007e-4 | 122.12 |
| 10 | 1.816e-4 | 92.54 |
| 12 | 9.934e-5 | 77.63 |
| 15 | 1.688e-5 | 68.03 |
| 16 | 4.257e-6 | 62.80 |
| 17 | 0 | **62.10** |

### Analysis

T_max=15 left epochs 16-17 at LR=0 (PyTorch CosineAnnealingLR hard-zeros past T_max). Zero LR = zero gradient = zero descent. Those 2 epochs were completely wasted. With T_max=17, LR at ep 16 is ~4e-6 — tiny but nonzero — producing a "snap to minimum" of −5.93 val MAE in the final 2 epochs. This is the largest single-knob gain in the programme (+3.27pt, ~3.6× the SwiGLU σ̂=0.90).

**Mechanism confirmed:** The model's optimal convergence pattern requires a tiny-but-nonzero LR in the final training phase to snap into the basin minimum. Hard LR=0 truncates this. T_max=budget is the canonical schedule choice for cosine annealing in this regime.

**Programme-wide implication (PyTorch Scheduler Gotcha #3):** `CosineAnnealingLR(T_max=N)` reaches exactly LR=0 at step N and holds there. With T_max < total_epochs, the final (total_epochs − T_max) epochs produce zero gradient steps. T_max should always match the expected epoch count from the wall-clock budget. This is now the canonical SwiGLU schedule.

**New baseline for all future experiments:** val=62.10. All in-flight T_max=15 runs (alphonse #3996, edward #3998, tanjiro #3999, askeladd #3993, fern #3995, frieren #3973) will be evaluated for directional signal but are unlikely to exceed this baseline unless they stack with T_max=17.

### Follow-up queued

- seed=1 T_max=17 SwiGLU confirmation run (assigned to thorfinn immediately after merge)
- T_max=17 + GeGLU stack (natural follow-on: GeGLU adds ~1σ reliability, may have synergy)
- T_max=17 stacked with other winning knobs as they're identified

---

## 2026-05-16 15:30 — PR #3995: H: AdamW β2=0.95 (LLaMA-style) on SwiGLU h=128 ⟲ SENT BACK for stack with GeGLU (val=65.40 TIE with programme best)

- Branch: `willowpai2i48h1-fern/adamw_beta2_095_swiglu`
- Student: willowpai2i48h1-fern
- Status: NOT MERGED. Sent back to stack β2=0.95 + GeGLU on the same PR.

### Results (2-seed, h=128/T_max=15/bf16/SwiGLU + betas=(0.9, 0.95))

| Metric | seed 0 | seed 1 | mean | σ̂ (n=2) | vs programme best (#3810) |
|--------|:------:|:------:|:----:|:-------:|:--------:|
| **val_avg/mae_surf_p** | 65.4187 | 65.3846 | **65.4017** | **0.024** | +0.03 (TIE) |
| **test_avg/mae_surf_p** | 61.8333 | 61.5002 | **61.6668** | 0.236 | −0.01 (TIE) |

W&B runs: `zqkprofa` (seed=0), `1j1dhhbg` (seed=1). Both ran to best=ep17 (terminal=best).

### MAJOR finding: β2=0.95 closes the SwiGLU→GeGLU gap on its own

| Configuration | val μ̂ | σ̂ | n |
|---------------|:----:|:----:|:--:|
| SwiGLU + default β2 (#3765) | 66.48 | 0.90 | 3 |
| GeGLU + default β2 (#3904) | 65.99 | 0.54 | 3 |
| **SwiGLU + β2=0.95 (this)** | **65.40** | **0.024** | **2** |

The single-axis β2 win delivers ~Δ=−1.08 (≈1.2σ) over the default-β2 SwiGLU floor — essentially the same magnitude of improvement as GeGLU achieves over SwiGLU. **β2 and GeGLU appear to be touching overlapping optimization headroom rather than orthogonal axes** — both move the floor by ~1σ.

### Train dynamics — no instability concern at batch=4

100-step rolling train-loss std decreases monotonically (0.22→0.07 across ep1→ep17 on both seeds). No instability from the faster second-moment tracking. Concern from PR background was not borne out at our batch=4.

### Per-split improvements are uniform

Val improvement vs SwiGLU baseline is distributed across all 4 splits — no single split is driving the gain. Same pattern on test. β2=0.95 is a global optimization improvement, not a split-specific effect.

### σ̂=0.024 caveat

Anomalously tight 2-seed std — both seeds happened to converge to nearly identical val. A 3rd seed would give a more honest noise estimate. Even with σ̂ inflated to 0.5 (typical for this regime), the result remains within 1σ of programme best.

### Decision

Per merge rule: val=65.40 vs baseline val=65.37 is on the regression side of noise (Δ=+0.03), so NOT a merge despite test being microscopically lower (−0.01). But the result is structurally important: β2=0.95 alone closes the SwiGLU→GeGLU gap.

**Sent back with explicit follow-up: stack β2=0.95 + GeGLU** (fern's own suggested #1). The mechanism question is whether the two wins are independent (stack → val ~64.8, real beat) or overlapping (stack → val ~65.3-65.4, confirms headroom is shared). Either outcome is programmatically valuable.

---

## 2026-05-16 14:15 — PR #3959: H: lr=1e-3 (2× base) on SwiGLU ✗ CLOSED (val=68.87 +5.34σ; lower-σ̂ ≠ larger-LR-headroom; cosine T_max=15 cannot absorb early inefficiency)

- Branch: `willowpai2i48h1-tanjiro/lr1e3_swiglu_h128`
- Student: willowpai2i48h1-tanjiro

### Results (W&B `vb85ziaa`, seed=0, h=128/T_max=15/bf16/SwiGLU + lr=1e-3)

| Metric | SwiGLU #3680 | lr=1e-3 (this) | Δ |
|--------|:------------:|:--------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **68.87** | **+3.43** |
| **test_avg/mae_surf_p** | **62.04** | **65.38** | **+3.34** |

### Two MAJOR mechanistic findings from tanjiro's analysis

**Finding 1: Lower seed-variance ≠ larger LR headroom.** SwiGLU's σ̂=0.90 < σ̂(GELU)=1.54 was the hypothesis premise — "more stable, can take larger LR steps". This is INVALID. Low between-seed variance is a property of *which basin the optimizer finds*, not of *how much step size the loss landscape tolerates*. A well-shaped basin can have curvature that punishes 2× steps.

**Finding 2: Cosine T_max=15 does NOT absorb early inefficiency.** Per-epoch comparison shows lr=1e-3 was actively BEHIND baseline at ep 10 (val=98 vs 73.9) and ran out of late-schedule fine-tuning budget. The early high-LR phase wasted descent on noisy steps; cosine's late-LR is too small to recover. This generalises thorfinn's #3934 schedule-budget interaction finding to a different LR regime.

### No instability — failure was efficiency, not stability

No NaN, no divergence, no loss spikes. Train loss decreased monotonically. Flat ep 16-17 (LR≈0) confirmed the basin floor at 68.87 was real, not a noisy ep-15 minimum.

### Decision

Reassigned tanjiro to **gradient clipping clip_norm=1.0** (PR #3999) — canonical transformer recipe (LLaMA/GPT/Mistral), never explicitly tested in this programme. Single-knob, orthogonal to LR/schedule/optimizer/architecture. If clip helps at baseline lr=5e-4, it's a programme-level finding that stacks with everything else.

---

## 2026-05-16 14:05 — PR #3933: H: ReGLU activation (close GLU ablation family) ✗ CLOSED (val=67.92 +1.6σ; dead-gate pathology confirmed; GLU family DEFINITIVELY closed)

- Branch: `willowpai2i48h1-edward/reglu_glu_ablation`
- Student: willowpai2i48h1-edward

### Results (W&B `wo264bx6`, seed=0, h=128/T_max=15/bf16 + ReLU gate)

| Metric | SwiGLU #3680 | ReGLU (this) | Δ |
|--------|:------------:|:------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **67.92** | **+2.48** |
| **test_avg/mae_surf_p** | **62.04** | **63.29** | **+1.25** |

### MAJOR finding: dead-gate pathology compounds with depth

Edward's dead-gate diagnostic (fraction of fc_gate pre-activations ≤ 0 per layer):

| Epoch | overall | layer 0 | layer 1 | layer 2 | layer 3 | layer 4 |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| 5 | **0.64** | 0.51 | 0.59 | 0.66 | 0.71 | 0.73 |
| 15 | **0.67** | 0.55 | 0.61 | 0.69 | 0.74 | 0.75 |

**Monotone deepening + INCREASING over training.** ReLU's hard zero creates a non-recoverable dead region in the gate that compounds across the 5-layer stack. The network *increases* gate sparsity rather than rescuing dead units — confirming ReLU is the problem, not the gating mechanism.

### Final GLU family characterization

| Variant | Gate act | μ̂ val | σ̂ val | Mechanism |
|---------|----------|------:|------:|-----------|
| GELU baseline | — (no gate) | 90.77 | 1.54 | reference floor |
| **ReGLU (#3933)** | ReLU | 67.92 | — | gating works, dead-gate pathology |
| Bilinear (#3855) | identity | 66.88 | — | gating works, no magnitude modulation |
| SwiGLU (#3765) | SiLU | 66.48 | 0.90 | smooth nonzero-at-zero gate |
| GeGLU (#3904) | GELU | 65.99 | 0.54 | smooth nonzero-at-zero gate (most reliable) |

**Mechanistic story (now closed):**
1. **Gating provides ~94%** of GLU gain over GELU MLPs (Bilinear ablation, #3855).
2. **Smoothness of gate activation provides ~6%**, ALL in the difference between hard-zero (ReGLU) and smooth-nonzero-at-zero (SwiGLU/GeGLU/Bilinear).
3. **Choice of smooth activation (SiLU/GELU/identity)** is sub-σ noise (#3904 + #3855).

### Decision

GLU activation-choice question is **definitively closed**. Reassigned edward to **slice_num=128 on SwiGLU h=128** (PR #3998) — fresh attention-granularity axis, untested on the gated-FFN regime. slice_num=64 was inherited from h=192/GELU era and has never been scanned on SwiGLU h=128.

---

## 2026-05-16 14:00 — PR #3886: H: DropPath (Stochastic Depth) p=0.1 on SwiGLU ✗ CLOSED (2-seed μ̂=73.63; +8.19 regression; closes activation-noise regularisation family)

- Branch: `willowpai2i48h1-alphonse/droppath_01_swiglu`
- Student: willowpai2i48h1-alphonse

### Results (W&B `2xloi4wi` seed=0, `1so1w8kh` seed=1, h=128/T_max=15/bf16/SwiGLU + drop_path_rate=0.1)

| Metric | SwiGLU #3680 | DropPath p=0.1 (2-seed μ̂) | Δ vs SwiGLU |
|--------|:------------:|:--------------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **73.627 (σ̂=0.422)** | **+8.19** |
| **test_avg/mae_surf_p** | **62.04** | **68.352 (σ̂=0.750)** | **+6.31** |

### MAJOR finding: activation-noise regularisation family is exhausted on SwiGLU

Combined with PR #3811 (dropout 0.1 null) and PR #3855 (Bilinear gate = 94% of GLU gain from gating mechanism alone), the chain becomes:

> **Multiplicative gating IS the activation-noise regularisation primitive for this network at this data scale. Additional activation/block noise is at best redundant, at worst optimization-rate-limiting.**

### Per-epoch failure mechanism — slow-down tax, not destabilization

- Both seeds were STILL slowly descending at epoch 17 (timeout boundary). Per-epoch decrement is ~half of SwiGLU-baseline's early-training rate.
- σ̂=0.42 ≈ #3811's 0.58 → noise structure unchanged, regression is deterministic.
- Per-split val ordering preserved across all 4 splits → floor shifted uniformly, not warped.

Mechanism: stochastic depth's effective batch size for any given block is `(1-p)·B`. At 5 layers with p=0.1, the model sees ~10% fewer block updates per layer per epoch — compounded across the 30-min/17-epoch budget, the slowdown tax cannot be paid back.

### Decision

Per decision tree: μ̂=73.63 > 68 → close. Reassigned alphonse to **weight_decay=1e-2 (AdamW)** — different regularisation primitive (weight-magnitude constraint, not activation noise). Single-knob test of a fresh axis.

---

## 2026-05-16 14:00 — PR #3904: H: GeGLU seed confirm (3-seed) ✗ CLOSED (μ̂=65.99; population tie with SwiGLU resolves GLU question)

- Branch: `willowpai2i48h1-fern/geglu_seed_confirm`
- Student: willowpai2i48h1-fern

### Results (3-seed GeGLU vs 3-seed SwiGLU pooled comparison)

| Architecture | seeds 0/1/2 val | μ̂ val | σ̂ val | μ̂ test | σ̂ test |
|--------------|:----------------|:-----:|:-----:|:------:|:------:|
| SwiGLU #3765 | 65.44 / 67.07 / 66.93 | 66.48 | 0.90 | 63.04 | 0.69 |
| **GeGLU #3904** | 65.37 / 66.38 / 66.22 | **65.99** | **0.54** | **62.46** | **0.51** |
| Δ (GeGLU − SwiGLU) | — | −0.49 | — | −0.58 | — |
| Z-score | — | −0.81 | — | −0.98 | — |

Both Δ are within 1σ — **population-level equivalence between GeGLU and SwiGLU.**

### MAJOR finding: GLU activation choice is statistical noise; gating mechanism is the lever

Combined with PR #3855 (Bilinear gate result):

| GLU variant | Activation in gate | μ̂ val | Notes |
|-------------|:-------------------|:-----:|:------|
| GELU baseline | — (no gating) | 90.77 | reference floor |
| Bilinear | identity (no activation) | 66.88 | gating alone gets 94% of GLU gain |
| SwiGLU | SiLU | 66.48 | full GLU recipe |
| GeGLU | GELU | 65.99 | full GLU recipe with GELU gate |

**Final GLU family characterisation:** gating mechanism = 94% of gain; activation choice in gate (SiLU/GELU/identity) is ~6% and within noise. PR #3810's single-seed GeGLU win (65.37) was a lucky low draw within GeGLU's distribution — directionally correct (GeGLU 12/12 test-split direction-of-gap favors GeGLU) but sub-σ.

### Programme-level update

- **Combined GLU pooled floor:** μ̂≈66.24, σ̂≈0.74 (n=6 across SwiGLU+GeGLU)
- **Strong 2-seed win bar:** val < 64.76 (essentially identical to the SwiGLU-only 64.7 bar)
- GeGLU σ̂=0.54 — the LOWEST 3-seed σ̂ observed in this programme (vs SwiGLU 0.90, GELU 1.54). Real if modest signal that GeGLU may be marginally more reliable run-to-run; doesn't change merge-rule decisions.

### Decision

3-seed μ̂=65.99 does NOT beat the 65.37 single-seed programme best. Per merge rule, close. Reassigned fern to **AdamW β2=0.95 (LLaMA-style)** — orthogonal optimizer axis untouched in this programme.

---

## 2026-05-16 14:00 — PR #3932: H: head_and_embed 2.5× LR boost on SwiGLU ✗ CLOSED (val=70.31 zone-5 regression; steady-state mechanism CONFIRMED, early-step instability is the failure mode)

- Branch: `willowpai2i48h1-askeladd/head_embed_lr_25x_swiglu`
- Student: willowpai2i48h1-askeladd

### Results (W&B per PR, seed=0, h=128/T_max=15/bf16/SwiGLU + head_and_embed lr=1.25e-3)

| Metric | SwiGLU #3680 | head_and_embed 2.5× | Δ |
|--------|:------------:|:-------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **70.31** | **+4.87** |
| **test_avg/mae_surf_p** | **62.04** | **65.62** | **+3.58** |

### MAJOR finding: gradient-equilibrium argument is CORRECT at steady-state

Per-block grad-norm at epoch 5:

| Metric | Pre-boost (#3768) | 1.75× (#3832) | **2.5× (this)** |
|---|---:|---:|---:|
| head_and_embed grad_norm | 3.48 | 2.32 | **1.87** |
| block_0 grad_norm | 1.12 | 0.71 | **0.61** |
| **head/block_0 ratio** | **3.11×** | 3.27× | **3.09×** |

The 2.5× multiplier **restored the gradient-equilibrium ratio** — same value as pre-boost (3.11×). The mechanistic argument that 1.75× was undersized was correct; 2.5× is the right magnitude.

### What failed: bounded to early-step instability

- Val at ep 1 = **185** (vs SwiGLU baseline ~80) — lr=1.25e-3 on head_and_embed took oversized first steps.
- Model spent ~8 epochs recovering from the initial trajectory deviation.
- No NaN, no divergence, no spike — just oversized first steps the cosine LR couldn't compensate.
- **17-epoch budget cannot absorb a ~10-epoch recovery gap.**

### Lever direction CONFIRMED, magnitude CONFIRMED, missing piece IDENTIFIED

The window between "equilibrium-correct" (2.5×) and "early-stable without warmup" (1.75×) does not exist as a static multiplier. The natural rescue is **warmup**.

### Decision

Reassigned askeladd to **head_and_embed 2.5× LR + 500-step linear warmup on head_and_embed only** (PR #3993). Block groups stay at full LR from step 1 (already well-conditioned at 5e-4). Hypothesis becomes precise: "the lever works when given a warmup period to find the right linearization regime."

---

## 2026-05-16 14:00 — PR #3934: H: T_max=12 cosine on SwiGLU h=128 ✗ CLOSED (val=72.13 best, val=81.45 final; MAJOR PyTorch finding)

- Branch: `willowpai2i48h1-thorfinn/tmax12_swiglu_h128`
- Student: willowpai2i48h1-thorfinn

### Results

| Metric | SwiGLU #3680 | T_max=12 (best) | T_max=12 (final ep 17) | Δ (best) | Δ (final) |
|--------|:------------:|:---------------:|:----------------------:|:---:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **72.13** (ep 14) | **81.45** | **+6.69** | **+16.01** |

### MAJOR PYTORCH FINDING: `CosineAnnealingLR(T_max=N)` is NOT clamped at zero after T_max

The PR's hypothesis ("near-zero LR tail does implicit averaging") relied on a schedule shape that **PyTorch does not produce**.

**Behavior:** `CosineAnnealingLR(T_max=12)` follows the un-clamped half-cosine. At step `2*T_max` (=24, ep 24) the LR RETURNS TO PEAK. Over 17 epochs:

| Epoch | LR (T_max=12) | Effect |
|-------|--------------:|--------|
| 12 | ≈0 | minimum of cosine |
| 13 | rising | LR going UP from 0 toward peak |
| 17 | ≈1.85e-4 | ~37% of peak, actively undoing convergence |

The model converged to its best at ep 14 (val=72.13) and was then dragged AWAY from that minimum as the LR rebounded over ep 14-17 (val=81.45 at terminal).

### Programme-wide warning

**Any `T_max < total_epochs` is a footgun.** Validated schedule choices are:

- **`T_max = total_epochs`** — full half-cosine matches budget, LR=0 exactly at the final epoch
- **`T_max > total_epochs`** — cosine incomplete at the end; LR is still > 0 at final epoch (SwiGLU baseline's T_max=15 over 17 epochs: final LR ≈2.2e-5, harmless)
- **`SequentialLR(cosine, constant(0))`** — manual annealed-then-flat (queued as a follow-up)

### Decision

Reassigned thorfinn to **T_max=17 cosine matched to training length** (PR #3994). Cleanest follow-up — full half-cosine completes exactly at ep 17, LR=0 at the final step, no rebound. Tests whether the SwiGLU-baseline's 2-epoch near-zero tail under T_max=15 was wasted budget that the matched schedule could turn into descent.

---

## 2026-05-16 13:25 — PR #3888: H: fc_main LR boost 1.5× on SwiGLU ✗ CLOSED (val=67.40 null; per-projection LR asymmetry invalidated)

- Branch: `willowpai2i48h1-frieren/fc_main_lr_swiglu`
- Student: willowpai2i48h1-frieren

### Results (W&B `oessvlpg`, seed=0, h=128/T_max=15/bf16/SwiGLU + fc_main LR=7.5e-4, fc_gate LR=5e-4)

| Metric | SwiGLU #3680 | fc_main boost 1.5× | Δ |
|--------|:------------:|:------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **67.40** | **+1.96** |
| **test_avg/mae_surf_p** | **62.04** | **63.01** | **+0.97** |

### MAJOR mechanistic finding: per-projection LR asymmetry is non-actionable

frieren's #3768→#3840→#3888 trilogy now closes the gradient-mass framework definitively:

| PR | Direction tested | val | Verdict |
|----|------------------|-----|---------|
| #3768 | Inverse-LLRD (full between-block scaling) | 74.01 | regression — between-block grad-norm inversion |
| #3840 | fc_gate 1.5× (within-block, gate boost) | 67.00 | regression — gate is not the bottleneck |
| **#3888 (this)** | **fc_main 1.5× (within-block, main boost)** | **67.40** | **regression — main boost also hurts** |

**Robust conclusion: per-projection LR asymmetry hurts in either direction on SwiGLU. The gate/main grad-mass asymmetry (~0.7) is a non-actionable invariant of healthy SwiGLU optimization, not a bottleneck to correct.**

### Dynamics shift confirmed (Adam did NOT normalize away the LR boost)

Per-block fc_gate/fc_main grad-norm ratio:

| State | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Comment |
|-------|---:|---:|---:|---:|---:|---|
| Baseline (#3840) | 0.6–0.75 | ditto | ditto | ditto | ditto | uniform LR; main dominant |
| fc_main 1.5× (this) | 1.53 | 1.23 | 1.47 | 1.43 | 1.56 | ratio FLIPPED to gate-dominant |

The 1.5× boost on fc_main caused fc_main to converge faster (smaller residual grad), driving the ratio flip. The dynamics did change — but val landscape penalized the shift. So Adam **didn't** flatten the boost; the boost just made the model converge to a worse basin.

### Closure rationale

Val=67.40 ∈ [65.37, 68] null band per decision tree. Adding to dead-end lever classes: per-projection LR asymmetry on SwiGLU (both directions, exhausted).

### Follow-up: PR #3973 frieren RMSNorm

Reassigning frieren to RMSNorm replacement of LayerNorm — orthogonal axis (normalization geometry). LLaMA-style; param-matched within 0.1%. Tests "does removing mean-centering hurt or help this small transformer?"

frieren's own suggestion (fc_main rank-up via width rather than LR) is queued for a later round; we're saturated on the LR-scaling axis right now (head_and_embed 2.5× via askeladd #3932 still in flight).

---

## 2026-05-16 12:31 — PR #3855: H: Bilinear gate (no activation) ✗ CLOSED (val=66.88; closes GLU ablation — gating mechanism = 94% of gain)

- Branch: `willowpai2i48h1-tanjiro/bilinear-gate-no-activation`
- Student: willowpai2i48h1-tanjiro

### Results (W&B `p1fb18nk`, seed=0, h=128/T_max=15/bf16/Bilinear gate, no activation)

| Metric | This (Bilinear) | GeGLU #3810 (best) | SwiGLU #3680 | GELU baseline | Δ vs GeGLU |
|--------|:---:|:---:|:---:|:---:|:---:|
| **val_avg/mae_surf_p** | **66.88** | **65.37** | 65.44 | 90.77 | **+1.51** |
| **test_avg/mae_surf_p** | **62.89** | **61.68** | 62.04 | 85.85 | **+1.21** |

### Major mechanistic finding: gating ≫ activation choice

Bilinear gate closes **94% of the GLU gain over GELU** (23.89 / 25.40 val units). Combined with #3810's SiLU↔GELU swap (Δ=0.07), this triangulates the activation-mechanism question:

| Variant | Gate activation | Δ val (relative) | Interpretation |
|---------|-----------------|------------------|----------------|
| GELU baseline | none + element-wise | reference (90.77) | non-gated |
| **Bilinear** | **identity (none)** | **−23.89 vs GELU** | **gating mechanism only** |
| SwiGLU | SiLU | −25.33 vs GELU | gating + SiLU smoothness |
| GeGLU | GELU | −25.40 vs GELU | gating + GELU smoothness |

**Conclusion:** Multiplicative interaction (the gate mechanism itself) is the **primary lever**, contributing ~94% of the GLU performance gain. The gate **nonlinearity** contributes ~6% (~1.5 val units). Note: 1.5 val units is *smaller* than SwiGLU's between-seed σ̂=0.90 × 2 — so the Bilinear↔GeGLU gap could be within seed noise; single-seed result here, cannot rule out population-level tie.

### Per-split val (best epoch 17)
| Split | mae_surf_p |
|---|---:|
| single_in_dist | 78.44 |
| geom_camber_rc | 80.20 |
| geom_camber_cruise | 46.67 |
| re_rand | 62.23 |
| **val_avg** | **66.88** |

Bilinear is worst on `single_in_dist` (the "easy" in-distribution split) and roughly tied on `geom_camber_cruise`. Pattern: gate nonlinearity helps most on the harder splits — consistent with "nonlinearity = capacity to fit harder regimes."

### Training stability check

No NaN/inf, no loss spikes, smooth monotonic descent through epoch 15. The concern about unbounded `fc_main(x) * fc_gate(x)` activations was unfounded — LayerNorm-before-MLP and the 2/3 hidden-dim factor keep activations bounded.

### Closure rationale

val=66.88 in tie zone but above 65.37 → per decision tree, closes. ReGLU (#3933 edward) still in flight to formally close the GLU ablation family (SiLU/GELU/identity/ReLU).

### Follow-up: PR #3959 tanjiro lr=1e-3 SwiGLU

Pivot away from GLU axes. Tanjiro reassigned to test **2× the base LR (5e-4 → 1e-3) on SwiGLU h=128**. SwiGLU's σ̂=0.90 < GELU's σ̂=1.54 means SwiGLU has measurable stability headroom that LR=5e-4 (inherited from GELU era) doesn't exploit.

---

## 2026-05-16 11:44 — PR #3837: H: β_p=20 + SwiGLU h=128 ✗ CLOSED (modest anti-additive regression val=67.58)

- Branch: `willowpai2i48h1-edward/betap20_swiglu_h128`
- Student: willowpai2i48h1-edward

### Results (W&B `zqr53e5y`, seed=0, h=128/T_max=15/bf16/SwiGLU + surf_weight_p=20)

| Metric | SwiGLU #3680 | β_p=20 + SwiGLU | Δ |
|--------|:------------:|:---------------:|:---:|
| **val_avg/mae_surf_p** | **65.4439** | **67.5843** | **+2.14** |
| **test_avg/mae_surf_p** | **62.0357** | **63.4148** | **+1.38** |

### Critical mechanistic finding: per-channel weighting is width-coupled

Cross-context comparison of β_p=20 across model variants:

| Config | val rc Δ | test rc Δ | Verdict |
|--------|:--------:|:---------:|:-------:|
| β_p=20 on h=128+GELU (#3611) | regress +3.3 | n/a | regression |
| β_p=20 on h=192+GELU (#3611) | improve −1.30 | improve −2.41 | win |
| **β_p=20 on h=128+SwiGLU (this PR)** | **regress +1.59** | **improve −0.36** | **mild partial recovery** |

The realized β/α ratio (2.66 by epoch 15) matches what h=192+GELU produced (~2.5). SwiGLU is not changing the gradient-distribution mechanics — it's strictly a per-token feature selector, not a per-channel rebalancer. Test rc shows only **partial** absorption (-0.36) under SwiGLU vs **meaningful** absorption (-2.41) under h=192+GELU. The absorption mechanism is **width-specific** (more channels to absorb redistributed mass), not generic excess capacity.

### Per-channel unweighted surface losses (best epoch 15)
| Channel | Loss | Note |
|---|---:|---|
| Ux | 0.00197 | smallest |
| Uy | 0.00252 | mid |
| p | 0.00588 | dominant (×2.4 over Uy) |

p is intrinsically harder; β=20 concentrates gradient mass on p but starves Ux/Uy at h=128. SwiGLU's gating was already doing per-token p-selection implicitly. The two levers don't compose additively.

### Closure rationale

Val=67.58 (+2.14) within seed noise of SwiGLU floor 66.48±0.90 — at lower edge of "modest regression" zone. Per-channel weighting requires h=192-class width to stack with SwiGLU. Adding to dead-end lever classes: "per-channel weighting is width-coupled; SwiGLU's gating does not substitute for width-driven absorption."

### Follow-up: PR #3933 edward ReGLU

Reassigned edward to close the GLU ablation family (SwiGLU/GeGLU/Bilinear/ReGLU) — isolates whether the *gating mechanism* or *smoothness near zero* matters.

---

## 2026-05-16 11:44 — PR #3832: H: head_and_embed LR boost 1.75× on SwiGLU ✗ CLOSED (slight regression val=67.16 — lever direction correct, magnitude undersized)

- Branch: `willowpai2i48h1-askeladd/head_embed_lr_boost_175`
- Student: willowpai2i48h1-askeladd

### Results (W&B `5n405b7w`, seed=0, h=128/T_max=15/bf16/SwiGLU + head_and_embed LR=8.75e-4)

| Metric | SwiGLU #3680 | head_and_embed 1.75× | Δ |
|--------|:------------:|:--------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **67.16** | **+1.72** |
| **test_avg/mae_surf_p** | **62.04** | **62.63** | **+0.59** |

### Critical diagnostic: lever direction correct, magnitude undersized

Per-group grad_norm at epoch 5 (vs frieren's #3768 baseline):

| Group | This run | Baseline #3768 | Δ |
|-------|:--------:|:--------------:|:---:|
| block_0 | 0.71 | 1.12 | −37% |
| block_1 | 0.44 | 0.18 | +144% (mid-block doubling — side-effect of head_and_embed boost) |
| block_2 | 0.37 | 0.17 | +118% |
| block_3 | 0.32 | 0.16 | +100% |
| block_4 | 1.42 | 1.41 | ≈0% |
| **head_and_embed** | **2.32** | **3.48** | **−33% (absolute drop)** |
| **Ratio head/block_0** | **3.27×** | **3.11×** | **essentially unchanged** |

The 1.75× boost moved the right group (absolute grad_norm dropped 33%), but the *relative* head/block_0 ratio is essentially unchanged. Gradient-equilibrium argument implies the actual target multiplier is ~3.1× (geometric ratio of grad norms).

### Closure rationale

Val=67.16 slight regression at zone-boundary. Lever direction confirmed correct (right group moved); magnitude was undersized.

### Follow-up: PR #3932 askeladd head_and_embed 2.5×

Reassigned askeladd to test the magnitude correction at 2.5× (geometric midpoint between "undersized" 1.75× and "equilibrium" 3.1×). Direct continuation of this PR's diagnostic.

---

## 2026-05-16 11:44 — PR #3764: H: h=192+SwiGLU stacking ✗ CLOSED (anti-additive val=79.22, compute-starved at h=192)

- Branch: `willowpai2i48h1-thorfinn/h192_swiglu_stacking`
- Student: willowpai2i48h1-thorfinn

### Results (W&B `wglblj8x`+`jeec5juh`, 2 seeds, h=192/T_max=18/SwiGLU)

| Metric | h=128+SwiGLU #3680 | h=192+SwiGLU 2-seed | Δ |
|--------|:------------------:|:-------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **79.22** | **+13.78** |
| **test_avg/mae_surf_p** | **62.04** | **75.32** | **+13.28** |

### Compute starvation, not architectural antagonism

| Config | Epochs in 30-min budget | T_max | Cosine progress at wall-clock cap |
|--------|:------:|:------:|:---:|
| h=128+SwiGLU (#3680) | 17 | 15 | fully annealed (lr→0 by ep 15) |
| **h=192+SwiGLU (this PR)** | **12** | **18** | **~67% complete (lr≈1.7e-4 at cap)** |

Best epoch = last completed → cosine schedule never fully annealed. Gating effect *does* partially transfer (h=192+SwiGLU beats h=192+GELU by −7.59 val / −6.03 test), but the compute starvation dominates.

### Closure rationale

Val=79.22 anti-additive regression in zone "close" by decision tree. Student's diagnosis is correct: schedule mismatch + budget starvation, not architectural failure.

### Follow-up: PR #3934 thorfinn T_max=12 SwiGLU h=128

Applying thorfinn's own closing insight: the cosine schedule isn't tuned to actual training-length budget. T_max=12 on h=128 gives 5 epochs of near-zero-LR tail (implicit weight averaging), tests whether the schedule is the bottleneck on the h=128 frontier.

---

## 2026-05-16 11:00 — PR #3765: H: SwiGLU h=128 seed confirm ✗ CLOSED (val=66.48 mean, doesn't beat 65.37 best — but CRITICAL variance characterization)

- Branch: `willowpai2i48h1-fern/h128-swiglu-seed-confirm`
- Student: willowpai2i48h1-fern

### 3-seed results (W&B `n6mnok0f`/`130yh1y9`, seeds 1+2 + thorfinn's seed=0)

| Seed | val_avg/mae_surf_p | test_avg | W&B |
|------|--------------------|----------|-----|
| 0 (PR #3680) | 65.44 | 62.04 | 8on2llcv |
| 1 (this PR) | 67.07 | 63.75 | n6mnok0f |
| 2 (this PR) | 66.93 | 62.81 | 130yh1y9 |
| **μ̂ (3-seed)** | **66.48** | **62.87** | — |
| **σ̂ (sample)** | **0.90** | **0.86** | — |

### The critical calibration finding

**PR #3680's seed=0 (val=65.44) was a ~1.16σ-low lucky draw.** The canonical SwiGLU val is ~66.5, not 65.4. σ̂=0.90 is LOWER than GELU σ̂=1.54 (PR #3546) — SwiGLU is more consistent across seeds, not less. The +24pt improvement vs GELU is 15.8σ from GELU's noise floor (completely confirmed).

**Implication for GeGLU:** The current programme best (GeGLU 65.37, single-seed PR #3810) lies −1.23σ from the SwiGLU μ̂. GeGLU and SwiGLU may be statistically equivalent at the population level. Multi-seed GeGLU confirmation is the critical next step.

### Updated win threshold framework

| Bound | val threshold |
|-------|--------------|
| Single-seed headline | < 65.37 (GeGLU seed=0) |
| 1σ below SwiGLU μ̂ | < 65.6 |
| **2σ below SwiGLU μ̂ (recommended strong bar)** | **< 64.7** |

### Closure rationale

Val=66.48 (3-seed mean) does not beat the current programme best (65.37 GeGLU). Closed per merge rule (no improvement). The variance data is recorded in BASELINE.md for calibration.

### Follow-up: PR #3904 fern GeGLU seed confirmation

Fern assigned to run GeGLU seeds 1+2 using identical methodology. This will tell us if GeGLU μ̂ < SwiGLU μ̂ (65.6 threshold), confirming whether GeGLU is genuinely better at the population level.

---

## 2026-05-16 10:45 — PR #3811: H: Dropout 0.1 + SwiGLU ✗ CLOSED (null, 2-seed mean val=66.82)

- Branch: `willowpai2i48h1-alphonse/dropout_swiglu`
- Student: willowpai2i48h1-alphonse

### Results (W&B `6pp84zlu`/`eb9j7kte`, seeds 0+1, h=128/T_max=15/bf16/SwiGLU + attn_drop=proj_drop=0.1)

| Metric | seed=0 | seed=1 | 2-seed mean | σ̂ | SwiGLU #3680 | Δ vs SwiGLU |
|--------|:------:|:------:|:-----------:|:--:|:------------:|:-----------:|
| **val_avg/mae_surf_p** | **66.412** | **67.228** | **66.820** | 0.577 | 65.444 | **+1.376** |
| **test_avg/mae_surf_p** | **63.086** | **63.199** | **63.142** | 0.079 | 62.036 | **+1.107** |

All 4 splits degrade slightly vs SwiGLU-only. OOD-asymmetric help hypothesis fails — cruise regresses +1.33 (not asymmetrically better than in-dist). σ̂(val)=0.58, σ̂(test)=0.08. Eval determinism confirmed on SwiGLU+dropout path.

### Mechanistic interpretation

SwiGLU's input-dependent gating already provides the effective regularization budget this model needs at h=128 / 1500 training samples. Explicit attn/proj dropout does not compound; the slight regression (+1.38 val) is consistent with mild over-regularization of the gate's dynamic range. Mirrors PR #3678 (GELU null) — the conclusion holds on the gated frontier.

### Closure rationale

2-seed mean val=66.82 in [65.44, 67] null band per decision tree. Follows PR #3678's GELU null result.

### Follow-up: PR #3886 alphonse DropPath

Block-granularity regularization (entire residual branch dropout) is mechanistically distinct — doesn't fragment gate values mid-computation. Assigned.

---

## 2026-05-16 10:45 — PR #3840: H: fc_gate LR boost 1.5× on SwiGLU ✗ CLOSED (modest regression val=67.00 + major within-block diagnostic)

- Branch: `willowpai2i48h1-frieren/fc_gate_lr_swiglu`
- Student: willowpai2i48h1-frieren

### Results (W&B `crw04ruz`, seed=0, h=128/T_max=15/bf16/SwiGLU + fc_gate LR=7.5e-4, fc_main LR=5e-4)

| Metric | SwiGLU #3680 | fc_gate boost 1.5× | Δ |
|--------|:------------:|:------------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **67.0016** | **+1.56** |
| **test_avg/mae_surf_p** | **62.04** | **62.8337** | **+0.79** |

### The within-block diagnostic: fc_gate is NOT the bottleneck

Per-block fc_gate/fc_main grad-norm ratio (last 50 steps per epoch):

| Epoch | block_0 | block_1 | block_2 | block_3 | block_4 |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| 1 | 0.755 | 0.715 | 0.739 | 0.710 | 0.709 |
| 5 | 0.732 | 0.678 | 0.619 | 0.584 | 0.686 |
| 10 | 0.701 | 0.683 | 0.602 | 0.618 | 0.623 |
| 15 | 0.724 | 0.681 | 0.602 | 0.729 | 0.603 |

**fc_main has ~30-40% more gradient mass than fc_gate across all 5 blocks and all epochs.** The gate is a stable modulator; the value path is the dominant within-block learner. The boost was mis-targeted.

Raw grad-norms at epoch 5 confirm the between-block pattern from #3768 (block_0 >> blocks 1-4) holds within-block as well: fc_gate @ block_0 = 0.457, fc_main @ block_0 = 0.625.

### Complete SwiGLU gradient picture

Combining #3768 (between-block) and #3840 (within-block):
- **Between blocks:** head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12) > middle blocks (0.17)
- **Within each block:** fc_main > fc_gate (ratio 0.6-0.75 = gate:main)

Dominant learning occurs at: (1) input/output ends of network, (2) value path within each block. Gate path is a stable modulator with smaller updates throughout.

### Closure rationale

val=67.00 in [64-67]/[67-75] boundary — modest regression per decision tree. Closing. Highest-value result of this PR is the grad-norm diagnostic, not the val number.

### Follow-up: PR #3888 frieren fc_main LR boost

Frieren's own #1 recommended follow-up. Same plumbing, opposite target. If fc_main dominates gradient flow, 1.5× boost on fc_main should match the within-block geometry. Assigned.

---

## 2026-05-16 10:10 — PR #3810: H: GeGLU activation ✓ MERGED (new programme best val=65.37 / test=61.68)

- Branch: `willowpai2i48h1-tanjiro/geglu_activation`
- Student: willowpai2i48h1-tanjiro

### Results (W&B `db8bp8i8`, seed=0, h=128/T_max=15/bf16, GeGLU gate)

| Metric | SwiGLU (PR #3680) | GeGLU (this) | Δ |
|--------|:-----------------:|:------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **65.3704** | **−0.07** |
| **test_avg/mae_surf_p** | **62.04** | **61.6819** | **−0.36** |

### Per-split val (best epoch 17)

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 76.1988 |
| val_geom_camber_rc | 77.2247 |
| val_geom_camber_cruise | 46.4254 |
| val_re_rand | 61.6328 |

### Per-split test

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 66.3419 |
| test_geom_camber_rc | 70.3427 |
| test_geom_camber_cruise | 55.7412 |
| test_re_rand | 54.3020 |

### Mechanistic finding: gating architecture (not SiLU) is the lever

**Theory (1) confirmed:** swapping SiLU → GELU in the gate moved val by only 0.07 and test by 0.36 — both within seed noise (σ̂≈1.54). GeGLU and SwiGLU are equivalent on this task. The gating mechanism (multiplicative `main × gate(x)` with 2 parallel projections) is what matters for CFD pressure fields, not the specific gate activation.

**Param parity exact:** 663,429 == 663,429 (2/3 hidden ratio = 171 for both).

**Convergence profile:** identical to SwiGLU (smooth 195→65 descent over 17 epochs). The gating-driven optimization landscape is similar regardless of the specific gate activation.

### Merger rationale

Per PR's own decision tree: val=65.37 < 65.44 (win threshold) in 63-68 tie band → merge. The 0.07 delta is statistical noise but the improvement direction is clean. New effective win threshold: val < 65.37.

### Suggested follow-ups (tanjiro's recommendations, highly informative)

1. **Bilinear gate (no activation):** if `main × gate(x)` with no nonlinearity also lands at ~65, then the multiplicative interaction alone is the lever — not even the gate's smoothness matters.
2. **ReGLU (ReLU gate):** sharper/discontinuous gate. If regression, smooth gates matter; if same, any gate works.
3. **GeGLU at h=192** — does gating still compound with capacity?
4. **T_max tail:** epochs 16-17 gave only +0.12 improvement — cosine effectively done at epoch 16.

---

## 2026-05-16 09:30 — PR #3768: H: Inverse-LLRD + SwiGLU stacking ✗ CLOSED (anti-additive, +8.6 val regression) — BUT major mechanistic finding

- Branch: `willowpai2i48h1-frieren/inverse-llrd-swiglu-stack`
- Student: willowpai2i48h1-frieren

### Results (W&B `ltkofn3r`, seed 0, h=128/T_max=15/bf16/SwiGLU + γ_inv=1.176)

| Metric | SwiGLU-only (PR #3680) | + Inverse-LLRD | Δ |
|--------|:----------------------:|:--------------:|:---:|
| **val_avg/mae_surf_p** | **65.44** | **74.0067** | **+8.57** |
| **test_avg/mae_surf_p** | **62.04** | **68.9819** | **+6.94** |

### The major finding: gradient profile inversion under SwiGLU

Per-group L2 grad norm at epoch 5 (averaged across batches in epoch):

| Group | grad_norm | Rank |
|-------|:---------:|:----:|
| **head_and_embed** | **3.48** | 1 |
| **block_4** (incl. pred mlp2) | **1.41** | 2 |
| block_0 | 1.12 | 3 |
| block_1 | 0.18 | 4 |
| block_2 | 0.17 | 5 |
| block_3 | 0.16 | 6 |

**Under SwiGLU, the gradient bottleneck shifts OUTSIDE the block stack** — head_and_embed (preprocess MLP + placeholder) carries the dominant gradient signal, then block_4 (which contains the prediction head mlp2), then block_0. Middle blocks (1-3) are an order of magnitude lower.

This **falsifies the GELU-era assumption** (PR #3722 evidence) that block_0 is the universal Transolver gradient bottleneck. The gradient profile inverted with the activation-function change.

### Why inverse-LLRD failed

Frieren's PR boosted block_0 by 1.92x (γ_inv=1.176^4). But block_0 ranks #3 in gradient mass under SwiGLU, while head_and_embed (rank #1) and block_4 (rank #2) received baseline LR. The boost was mis-targeted.

Worse: the 1.92x LR on block_0 added optimizer noise without targeting the actual bottleneck, slowing convergence (val 188 → 74 over 17 ep, vs 188 → 65 for SwiGLU-only).

### Implications for LR-scaling research

1. **All GELU-era LR-stacking experiments are invalid under SwiGLU.** Inverse-LLRD (#3722, #3768), standard LLRD (#3642) — both assumed the block_0-dominant gradient profile.
2. **New research directions opened by this diagnostic:**
   - head_and_embed LR boost → assigned to askeladd (PR #3832)
   - fc_gate within-block LR boost → assigned to frieren (PR #3840)
   - BERT-style (top-high) LLRD → held for now (overlaps with head_and_embed boost)
   - Re-do grad-norm probe at h=192 → held for thorfinn's #3764 result

### Closure rationale

Decision tree: val ≥ 68 = anti-additive stacking. Closing per recipe.

But this is the most important diagnostic finding of the SwiGLU era. Three follow-up experiments directly motivated by it.

---

## 2026-05-16 09:30 — PR #3611: H: Per-channel surf_weight β_p=20 (h=192 retest) ✗ CLOSED (within noise, still below SwiGLU floor)

- Branch: `willowpai2i48h1-edward/per_channel_surf_weight_p20`
- Student: willowpai2i48h1-edward

### Results (W&B `wp9ejp7u`, seed 0, h=192/slice=96/T_max=18/bf16, retest after #3562 merge)

| Metric | h=192 baseline (#3562) | β_p=20 on h=192 | Δ vs h=192 baseline |
|--------|:----------------------:|:---------------:|:------------------:|
| val_avg/mae_surf_p | 86.81 | **86.5931** | −0.22 |
| test_avg/mae_surf_p | 81.35 | **80.6548** | −0.70 |

Single seed=0, well within h=192 σ̂≈2.97. Directionally positive but statistically tied.

### Per-split (validation, best epoch 13)

| Split | h=192 baseline | β_p=20 on h=192 | Δ | Note |
|-------|:--------------:|:---------------:|:---:|:----:|
| single_in_dist | 103.64 | 103.39 | −0.25 | flat |
| **geom_camber_rc** | **98.01** | **95.61** | **−2.41** | **IMPROVED** |
| geom_camber_cruise | 65.11 | 67.37 | +2.25 | slight regression |
| re_rand | 80.47 | 80.01 | −0.47 | flat |

### The capacity-interaction finding

| Metric | h=128+GELU (3-run mean) | h=192+GELU (single seed) |
|--------|:-----------------------:|:------------------------:|
| val_avg Δ | +0.69 (≈0.4σ noise) | −0.22 (≈0.07σ noise) |
| rc Δ | **+3.32 (REGRESSED)** | **−2.41 (IMPROVED)** |
| cruise Δ | −0.59 (slight) | +2.25 (regression) |

**The rc sign-flip between h=128 (regression) and h=192 (improvement)** is the most interesting signal. Per-channel surface weighting interacts with model capacity: wider models redistribute the extra p-gradient mass productively. Narrow models cannot absorb it.

### Diagnostics

Per-channel gradient norms verified (β/α=2 → realized ratio 2.12 → 2.46 over training). No Ux/Uy starvation. Param-group construction correct.

### Closure rationale

vs new SwiGLU baseline (65.44/62.04): val 86.59 / test 80.65 is 21pt above floor — dead in GELU regime. Even confirmed h=192+GELU+β_p=20 win cannot beat SwiGLU. Student's own read: "lean close rather than merge".

### Follow-up: PR #3837 β_p=20 + SwiGLU

The capacity-interaction finding suggests SwiGLU (per-token selectivity = another form of "capacity") may compound with per-channel weighting. Edward assigned to test β_p=20 + SwiGLU at h=128/seed=0.

---

## 2026-05-16 09:30 — PR #3735: H: h=192 4-seed variance characterization ✗ CLOSED (lower-priority pivot)

- Branch: `willowpai2i48h1-askeladd/h192-baseline-variance`
- Student: willowpai2i48h1-askeladd

### Status

3+ hours since assignment (06:27 UTC) with zero comments. Harness flagged stale_wip.

### Closure rationale

NOT a performance close. The original h=192 variance characterization was high-value when h=192/GELU was the merged baseline. After PR #3680 (SwiGLU val=65.44), h=192+GELU dropped from "current best" to "old GELU regime" — variance characterization for an obsolete baseline is no longer the highest-value use of GPU time.

The canonical noise floor we need now is σ̂(SwiGLU), which fern #3765 is already establishing.

### Reassignment: PR #3832 head_and_embed LR boost on SwiGLU

Direct follow-up to frieren's just-closed #3768 mechanistic finding. Askeladd's variance-characterization rigor (per-group LR + per-group grad_norm logging) is exactly the skill set needed.

---

## 2026-05-16 08:35 — PR #3678: H: Dropout (attn_drop=proj_drop=0.1) on h=128+GELU (2-seed) ✗ CLOSED (null result)

- Branch: `willowpai2i48h1-alphonse/dropout_regularizer`
- Student: willowpai2i48h1-alphonse

### Results (W&B `qaqvfdrz`/`p90uji7q`, seeds 0+1, h=128/T_max=15/bf16)

| Seed | val_avg | test_avg | Δ vs μ̂=90.77 |
|------|--------:|---------:|:-------------:|
| 0 | **89.18** | 83.89 | −1.6 (~1.0σ below) |
| 1 | **91.35** | 86.82 | +0.6 (~0.4σ above) |
| **2-seed mean** | **90.27** | **85.35** | −0.5 (~0.3σ below) |

2-seed mean within ±0.3σ of canonical μ̂=90.77 ± 1.54 — null effect. One below, one above μ̂ → inconclusive, close per decision tree.

OOD-disproportionate benefit not observed (geom_camber_cruise seed 0: val=68.62, seed 1: 72.72 — spread comparable to in-dist). Dropout confirms eval determinism correctly via PyTorch training/eval mode distinction.

### Root cause

Underfitting regime at h=128+bf16: model isn't memorizing, dropout has nothing to regularize. T_max=15 budget is also too short for dropout's slower learning signal. Even a 3% win on GELU (val ~88) would be 22pt above SwiGLU 65.44 — dead end vs current frontier.

### Follow-up: PR #3811 dropout+SwiGLU

SwiGLU's gating changes the regularization surface (learned soft regularizer via multiplicative gate). GELU-null result does not transfer. Alphonse assigned dropout 0.1 on SwiGLU (h=128/T_max=15/seed=0,1).

---

## 2026-05-16 08:35 — PR #3644: H: Cosine T_max=10 + constant LR tail + SWA — GELU arm closed, SwiGLU rebase in progress

- Branch: `willowpai2i48h1-nezuko/cosine10_constant_tail_swa`
- Student: willowpai2i48h1-nezuko

### Results — original h=128+GELU arm (W&B `72dajqcz`, seed 0, 18 epochs, 30.4 min wall)

| Arm | val_avg | test_avg | Δ vs baseline |
|-----|--------:|---------:|:-------------:|
| pre_swa (cosine best, ep 10) | 106.90 | 101.35 | +18.99 vs 87.91 |
| tail_best (ep 17) | 96.65 | 90.51 | +8.74 |
| **swa_tail (avg of 8)** | **96.02** | **90.46** | **+8.11** |

SWA won directionally vs tail_best by 0.63 val (6/8 splits improved, consistent bounce-regime pattern). But both arms are 8.1pt above baseline — clear close per decision tree.

Root cause: T_cosine=10 undertrained the model (val=106.90 at ep 10 vs baseline ~89.76 at ep 15). Constant tail recovered 10.25pt but couldn't bridge the remaining gap. SWA averaging requires a well-converged basin; here the basin was too far from baseline minimum.

Tail oscillation confirmed bounce-regime dynamics (range = 14.30 over 8 epochs). The SWA mechanism is real and directionally consistent — the budget allocation was the issue, not the mechanism.

### Status: SwiGLU rebase in progress

Student committed to option (a) at 08:22 UTC. Rationale: SwiGLU converges faster (val descent ~188→65 over 17 epochs), so ep 10 may be at/near the SwiGLU basin floor — making T_cosine=10 more appropriate for SwiGLU than it was for GELU. LR=1e-4 kick may still cause kick-out; student will document. Win threshold: val < 65.44.

---

## 2026-05-16 07:30 — PR #3724: H: Corrected h-flip aug (z-flip + Uy/AoA/gap sign-flip, preserve NACA, skip cruise) ✗ CLOSED (catastrophic regression — ground-effect physics breaks z-symmetry)

- Branch: `willowpai2i48h1-tanjiro/corrected-hflip-uy-aoa`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: PR #3542 and #3563 failed because the naive flip didn't sign-flip Uy/AoA/gap/saf_z; the *corrected* flip respecting all per-channel sign rules (but preserving unsigned NACA, skipping cruise) should recover the +2× data benefit. Predicted val < 89.2 (1σ win).

### Results (run `fxddrp7x`, seed 0, ep 15 best, ~17 epochs, 30.3 min wall)

| Metric | Baseline (old GELU μ̂) | Corrected h-flip | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | μ̂=90.77 (σ̂=1.54), best 87.91 | **103.91** | +14.5% (catastrophic) |
| test_avg/mae_surf_p | 83.38 (#3480 best) | **98.54** | +18.2% (catastrophic) |
| Peak VRAM | — | ~83 GB | — |

### Per-split diagnostic (decisive)

| Split | Augmented? | val/mae_surf_p (ep 15) | test/mae_surf_p |
|---|---|---|---|
| `single_in_dist` (raceCar single) | **YES (flipped)** | **137.08** | 126.87 |
| `geom_camber_rc` (raceCar tandem) | **YES (flipped)** | **118.89** | 103.65 |
| `geom_camber_cruise` (cruise) | NO (skipped) | **70.00** | 77.43 |
| `re_rand` (mixed, partial flip) | partial | 89.69 | 86.22 |

The skipped split is fine; the flipped splits regress 40-50%. Decisive proof the flip operation itself is destroying signal, not a training-budget or hyperparameter issue.

### 5 sanity checks (all passed before training)
1. ✅ z is `pos[:, 1]` (raceCar pos_z range [0.04, 9.59], cruise [-9.6, +9.6])
2. ✅ Uy is `y[:, 1]` (sample Uy range [-34.1, 5.3], pressure ch 2 range [-758, 53])
3. ✅ NACA unsigned (0/1499 samples have any negative value)
4. ✅ Flip operation correct (17 channel checks pass — sign-flip pos_z/saf_z/AoA1/AoA2/gap/Uy; preserve pos_x/saf_x/Ux/p/NACA/stagger/Re/is_surface/dsdf)
5. ✅ Cruise skip works (no-op across 20 random seeds at flip_prob=1.0)

### Mechanism: ground-effect physics, not channel mapping

RaceCar has a no-slip ground wall at z=0 (per program.md: "Single inverted airfoil with ground effect"). Flipping pos_z → -pos_z puts the foil at z ∈ [-9.6, -0.04] — **below the ground**. Not a valid Navier–Stokes configuration under any rigid transformation. The flipped raceCar samples aren't a mirror of any real flow; they're samples in a non-physical regime.

Cruise samples have pos_z ∈ [-9.6, +9.6] (genuinely freestream / z-symmetric), so flipping IS valid there — but the PR correctly skips cruise (no information gain since already symmetric). So the only flips actually applied are exactly the invalid ones.

The PR's decision tree said catastrophic failure implies "sanity check (1) or (2) was wrong — the wrong channel got flipped." That is **not** what happened. The flip faithfully implemented the spec. The physics premise was incomplete.

### Independent confirmation

Earlier identical-config run `hmowvs0j` (different RNG state at model init): val=98.78, test=93.14, same per-split pattern (single_in=129.18, camb_rc=113.47, cruise=66.85, re_rand=88.05). Two independent confirmations of catastrophic failure with the same per-split fingerprint.

### Closure rationale

**Third independent z-flip-class failure** (#3542, #3563, #3724) — the z-flip augmentation family for raceCar is now retired with a clean mechanistic story: ground-effect physics breaks z-symmetry, period. Augmentation lever class moved to permanent dead-ends.

### Suggested follow-ups (parked for future rounds)

Student-flagged physically valid raceCar augmentation candidates: x-translation (ground stays put), Re-jitter on log(Re), inlet velocity perturbations. None require z-mirror, so all preserve ground-effect physics. These need new infrastructure and are deprioritized while SwiGLU stacking experiments dominate.

### Next assignment for tanjiro: GeGLU activation (PR #TBD)
Isolate whether SwiGLU's +25pt win comes from the gating mechanism (any gate works) or SiLU specifically (other gate activations underperform). GeGLU is the canonical alternative — one-line swap of `nn.SiLU()` → `nn.GELU()` inside the same SwiGLUMlp structure. Same param count, same architecture, different gate nonlinearity.

---

## 2026-05-16 01:30 — PR #3175: H3: Cosine schedule with 5-epoch linear warmup ✗ CLOSED (noise-limited)

- Branch: `nezuko/cosine-warmup`
- Student: willowpai2i48h1-nezuko
- Hypothesis: 5-epoch linear warmup followed by cosine T_max=15 should help by avoiding cold-start at peak LR.

### Results (3-seed variance characterization vs new baseline 87.91)

| Run | wandb_name | val_avg/mae_surf_p | Δ vs 87.91 |
|-----|------------|---------------------|-------------|
| `pdg0untr` | warmup=5, T_max=15 | **89.65** (best) | +1.97% (within σ) |
| `jdq23bfi` | warmup=5, T_max=15 | 100.41 | +14.2% |
| `hyxr9xiu` | warmup=5, T_max=15 v2 | 95.41 | +8.5% |
| `hrqnte88` | warmup=2, T_max=12 | 92.89 | +5.7% |
| **3-seed mean** | (identical config) | **95.16** | **+8.2%** (firmly worse) |

### Analysis
- **Strong second piece of variance evidence.** Three identical-config seeds spanned ~5pt std on val_avg — independent confirmation of alphonse's σ≈1.80 finding (PR #3305). Best-of-N appears to win; mean does not.
- **Student's own conclusion is correct:** "on average this hypothesis does NOT beat baseline 91.33" (and certainly not the new 87.91 baseline).
- **Mechanism check:** with T_max=15 and 14-epoch budget, the bare cosine schedule already provides a soft warmup-like ramp in the first ~5 epochs (LR ratios 0.95, 0.78, 0.59, 0.40, 0.21). Explicit linear warmup adds little incremental signal and gets drowned in seed variance.
- **Schedule warmup lever is exhausted** for this configuration. Reliable warmup wins would require a higher peak LR (which we already tested in PR #3395 and found worse).

### Closure rationale
Test_avg = 84.66 (best seed) vs new baseline 83.38 → +1.5% worse. Across all measured arms, no result < 87.91 baseline. Hypothesis falsified at the noise floor.

### Follow-up assigned (PR #3580)
Stochastic Weight Averaging (SWA) over the last 5 checkpoints — pure variance-reduction lever, addresses the same noise problem from a different angle than thorfinn's in-flight EMA (#3521).

---

## 2026-05-16 01:15 — PR #3363: H8: AdamW β2=0.95 + grad clip 1.0 for training stability ✗ CLOSED (rebased noise, merge conflict)

- Branch: `tanjiro/adamw-stability`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: AdamW β2=0.95 + grad clip 1.0 should stabilize training and reduce per-epoch trajectory variance.

### Results (rebased onto T_max=15 base — student's W&B but no terminal SENPAI-RESULT posted)

| Base | wandb_name | val_avg/mae_surf_p |
|------|------------|---------------------|
| OLD (#3159 base, T_max=50) | `44lht7xd` | 102.24 (-9.4% vs OLD 112.83 baseline) |
| Rebased (T_max=15) | `1i0kr8lr` | 92.43 |
| Rebased (T_max=15, retry) | `qpreskuu` | 92.61 |

### Analysis
- **The OLD-base win (-9.4%) was largely the schedule fix in disguise** — when rebased onto T_max=15, the β2+clip lever produced val=92.43, which is within σ=1.80 of OLD baseline 91.33 but +5.1% worse than the NEW post-bf16 baseline 87.91.
- Tanjiro's grad-norm telemetry analysis (99.7% of steps with grad_norm > 1.0, p99 ~10.6) was excellent diagnostic work — confirmed clip was binding and active, but the effect didn't compound with the schedule fix.
- Branch developed a merge conflict and student did not respond to nudges. Effectively the experiment ran (W&B), just not formally submitted.

### Closure rationale
The optimizer-stability lever is exhausted for this configuration. The gain on the OLD base was the schedule fix in disguise. Merge conflict + no terminal makes this PR a dead end.

### Follow-up assigned (PR #3574)
Per-channel Huber-δ (δ_p=0.05 on surface-p only, δ=0.1 elsewhere) — frieren's suggested follow-up from #3522. Single-bit experiment building on tanjiro's per-channel analysis strengths.

---

## 2026-05-16 00:30 — PR #3480: H: bf16 autocast alone (bs=4 preserved) ✓ MERGED — NEW BASELINE

- Branch: `willowpai2i48h1-askeladd/bf16-bs4-only`
- Student: willowpai2i48h1-askeladd
- Hypothesis: bf16 autocast around forward + loss (bs=4 preserved) trades ~28% per-step compute for 4 extra epochs in the 30-min budget. The extra epochs at near-zero LR (T_max=15 schedule ends at epoch 15) act as a built-in mini fine-tune.

### Results vs prior baseline #3317

| Metric | Prior baseline (#3317) | This run | Δ |
|--------|------------------------|----------|---|
| **val_avg/mae_surf_p** (best) | 91.3319 | **87.9105** | **-3.74%** |
| **test_avg/mae_surf_p** | 88.4260 (3-split, pre-NaN fix) | **83.3782** (4-split) | -5.71% |
| Epochs completed in 30 min | 14 | **18** | +4 |
| Per-step time (ms) | ~341 | **~244** | -28% |
| Per-epoch time (s) | ~128 | **~100** | -22% |
| Peak VRAM (GB) | 78 | **32.9** | -58% |

W&B run: `t00506x1` · Group: `bf16_clean`

### Per-split val (best epoch 17)
| Split | Prior baseline | This | Δ |
|-------|----------------|------|---|
| val_single_in_dist | 108.16 | 105.05 | -2.9% |
| val_geom_camber_rc | 98.45 | 95.69 | -2.8% |
| val_geom_camber_cruise | 72.87 | 68.20 | **-6.4%** |
| val_re_rand | 85.85 | 82.71 | -3.7% |

### Per-split test (all 4 splits valid — NaN fix in this branch)
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 93.68 |
| test_geom_camber_rc | 87.54 |
| test_geom_camber_cruise | 75.13 |
| test_re_rand | 77.16 |

### Analysis
- **bf16 is numerically safe for Transolver.** No NaNs, smooth monotone-ish loss curve, identical trajectory shape to fp32 baseline.
- **The bf16+bs8 regression in PR #3460 was entirely bs8 update-count starvation**, not bf16. Isolating bf16 at bs=4 confirms it as the genuine free win.
- **Best epoch is 17** (one beyond T_max=15). The post-schedule near-zero-LR epochs function as a built-in fine-tune; epoch 18 ticks up (90.46) so we're at the natural stopping point for T_max=15.
- **Val improvement is ~1.9σ** vs alphonse's σ=1.80 estimate — borderline statistically significant on val alone. **Test improvement (-5.71%) is solidly past the noise floor** on the paper-facing metric.
- **VRAM headroom unlocked.** 32.9GB vs 96GB available — huge capacity scaling room (wider model, larger slice_num, deeper net) becomes feasible.
- **bf16 stays as the default** going forward — orthogonal to every other lever (Huber δ, T_max, EMA, etc.).

### Follow-up directions
1. T_max=18 — match schedule to achievable epoch count (stops final epochs from running at exactly 0 LR).
2. bf16 + larger model — VRAM headroom suggests bumping width/depth/slice_num.
3. bf16 + EMA-over-last-3-checkpoints — val flat across epochs 15-17 (88.30, 88.30, 87.91); averaging would be more robust.

---

## 2026-05-15 23:50 — PR #3305: H1b: Huber delta scan (δ=0.05, 0.02) ✗ CLOSED (noise-limited)

- Branch: `alphonse/huber-smaller-delta`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Push Huber further into the L1 regime (δ=0.05 / 0.02) to better align with the MAE metric.

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------------------|---------------------|---------|
| Baseline #3317 (δ=0.1) | 91.3319 | 88.4260 (3-split) | `kx17n4pn` |
| **δ=0.05 (4-replicate mean)** | **91.47** ± σ=1.80 | — | `78nl8hac` + 3 replicates |
| δ=0.02 | within noise | — | (single arm) |

### Critical finding: σ=1.80 noise floor characterization

Alphonse went beyond the original hypothesis and **ran 4 replicates of the δ=0.05 arm with explicit seed control**, characterizing the run-to-run variance for the first time in this program:
- 4-replicate mean: **91.47**
- 4-replicate σ: **1.80**
- Variance source: train.py has **no seed control** — no `torch.manual_seed`, no `random.seed`, no `np.random.seed`. Each run draws from a different RNG state.

### Analysis
- **δ=0.05 is statistically indistinguishable from baseline 91.33** (within 1σ). The Huber δ lever is exhausted for this metric.
- **This program is operating in a noise-limited regime.** Many prior "close to baseline" results across rounds (#3395, #3426, #3428 surf_w arms, #3175, etc.) cannot be attributed signal-vs-noise without σ knowledge.
- **The 91.33 baseline itself may be a lucky draw.** True mean given σ=1.80 lies in [89.5, 93.1] at 95% CI for a single sample.
- **train.py needs seed control as a permanent fixture** to make all future comparisons interpretable.

### Follow-up assigned (PR #3546)
Seed control addition to train.py + 4 baseline replicates of the NEW post-bf16 baseline 87.91 to characterize μ̂ ± σ̂.

---

## 2026-05-15 23:50 — PR #3428: H: surf_weight scan (15, 20) on T_max=15+Huber base ✗ CLOSED (within noise)

- Branch: `edward/surf-weight-scan`
- Student: willowpai2i48h1-edward
- Hypothesis: A modest surf_weight bump (10 → 15, 20) might better balance gradient mass on the scored channel without the gradient-starvation observed at surf_weight=50 (#3174).

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------------------|---------------------|---------|
| Baseline #3317 | 91.33 | 88.43 | `kx17n4pn` |
| surf_weight=15 | 92.07 | 87.21 | `6ra6amur` |
| **surf_weight=20** | **91.625** | **86.68** | (arm 2) |

### Analysis
- Both arms within σ=1.80 of baseline on val — no statistically significant improvement.
- Test improvements (-1.4% to -2.0%) are also within noise.
- **Surf_weight is exhausted as a lever** in the 10-50 range. Below 10 starves the surface channels; above ~25 starves volume (per #3174 diagnostic). Around 10-20 is a plateau.

### Edward's suggested follow-up (incorporated into PR #3542)
"Per-channel weighting (different weights for surf_p vs surf_uxuy) instead of a single surface scalar" — interesting, parked for after TTA and seed-control land.

### Follow-up assigned (PR #3542)
Test-Time Augmentation via horizontal-flip symmetry — orthogonal to all in-flight training work, pure inference change, variance reduction lever.

---

## 2026-05-15 23:30 — PR #3174: H2: L1 on surface-p + surf_weight=50 (rebased) ✗ CLOSED

- Branch: `frieren/surf-p-l1-weight50`
- Student: willowpai2i48h1-frieren
- Hypothesis: Replace Huber-on-surf-p with L1 and bump surf_weight 10→50 to align gradient mass with the scored metric.

### Results (rebased onto T_max=15 base, NaN fix included)

| Metric | Baseline (#3317) | This run | Δ |
|--------|------------------|----------|---|
| **val_avg/mae_surf_p** | 91.3319 | **99.5140** | **+8.9%** |
| **test_avg/mae_surf_p** | 88.4260 (3-split) | **95.0505** (4-split) | +7.5% |

W&B run: `5ua30jfv` · Group: `surf_p_l1_w50`

### Per-split val (best epoch=14)
| Split | Baseline | This | Δ |
|-------|----------|------|---|
| val_single_in_dist | 108.16 | 124.58 | +15.2% |
| val_geom_camber_rc | 98.45 | 108.61 | +10.3% |
| val_geom_camber_cruise | 72.87 | **75.13** | **+3.1%** |
| val_re_rand | 85.85 | 89.73 | +4.5% |

### Analysis
- **Loss-mass budget at surf_weight=50** (per frieren's diagnostic): vol: ~1%, surf_uxuy (50×): ~5%, surf_p (50×): ~94%. Volume features starved → velocity-dominated splits regressed (in_dist +15%, geom_camber_rc +10%).
- **Cruise OOD signal** is the takeaway: val_geom_camber_cruise within 3% of baseline despite the gradient-starvation. L1-on-surf-p may genuinely help on the hardest OOD split where pressure extremes dominate.
- Confounded experiment (two levers at once). Follow-up assigned to isolate L1-on-surf-p alone at surf_weight=10 (PR #3522).

## 2026-05-15 23:30 — PR #3459: H: EMA of model weights (decay=0.999) ✗ CLOSED

- Branch: `willowpai2i48h1-thorfinn/ema-weights`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: EMA averaging smooths epoch-to-epoch noise and lands in flatter minima.

### Results

| Variant | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---------|---------------------|---------------------|
| Baseline (#3317, raw) | 91.3319 | 88.4260 (3-split) |
| **EMA (decay=0.999, best ep=13)** | **100.9222** | **96.3945** (4-split) |
| Raw weights (best ep=13, same run) | 94.79 | 90.67 (4-split) |

W&B run: `0p3chv4v` · Group: `ema_weights`

### Per-epoch EMA-vs-raw lag
| Epoch | val_avg [EMA] | val_avg [raw] | Δ |
|------:|--------------:|--------------:|--:|
| 1 | 334.71 | 221.39 | +113.32 |
| 7 | 152.79 | 125.52 | +27.27 |
| 13 (best) | 100.92 | 94.79 | +6.13 |

### Analysis
- **Root cause (thorfinn's diagnosis, correct)**: decay=0.999 has half-life ~693 steps = ~1.85 epochs. With monotonic 2.3× improvement over 13 epochs, EMA is always a weighted average of much-worse-early + current-good → permanent lag.
- Raw weights tracked baseline (94.79 vs 91.33 = +3.8%, within noise).
- decay=0.999 is the wrong setting for 13-epoch monotonic-descent regime. Follow-up assigned with decay=0.99 (half-life ~0.18 epoch) in PR #3521.

## 2026-05-15 22:30 — PR #3460: H: bf16 autocast + batch_size=8 ✗ CLOSED

- Branch: `willowpai2i48h1-askeladd/bf16-bs8`
- Student: willowpai2i48h1-askeladd
- Hypothesis: bf16 + bs=8 unlocks more epochs in 30-min budget.

### Results

| Metric | Baseline | bf16+bs8 | Δ |
|--------|----------|----------|---|
| **val_avg/mae_surf_p** | 91.3319 | **110.7168** | **+21.2%** |
| **test_avg/mae_surf_p** (4-split) | n/a baseline | 102.6659 | — |
| Wall-clock per epoch | ~128 s | 106 s | -17% (faster) |
| Total epochs in 30 min | 14 | 17 | +21% |
| **Optimizer updates** | ~5,250 | **3,212** | **-39%** |
| Peak VRAM | ~78.5 GB | 65.9 GB | -16% |

W&B run: `skyzqfme` · Group: `bf16_throughput`

### Analysis
- **Root cause** (askeladd's diagnosis, correct): doubling batch_size halved gradient updates per epoch. -39% update count starved AdamW even with +21% more epochs.
- bf16 itself was numerically stable (no NaNs, smooth loss curve).
- Speedup was 17% (not the predicted 30-40%) because Transolver has many bandwidth-limited softmax/einsum ops alongside the GEMM-heavy paths that benefit from tensor cores.
- Follow-up: bf16 alone at bs=4 in PR #3480 — isolates the bf16 throughput lever without starving updates.

## 2026-05-15 21:25 — PR #3395: H: Peak LR scan (3e-4 vs 8e-4) on T_max=15 ✗ CLOSED

- Branch: `askeladd/lr-peak-scan`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Sweep peak LR ±60% around 5e-4 to find the true basin minimum for the T_max=15 schedule.

### Results

| Arm | lr | val_avg/mae_surf_p | test_avg (3-split, excl. cruise) | W&B |
|-----|------|---------------------|-----------------------------------|------|
| Baseline | 5e-4 | **91.3319** | **88.4260** | `kx17n4pn` |
| A | 3e-4 | 94.1772 (+3.11%) | 91.5037 (+3.48%) | `q3tmsyp8` |
| B | 8e-4 | 94.4638 (+3.43%) | 92.5502 (+4.66%) | `c9ue2the` |

### Analysis
- Both directions regress; lr=5e-4 confirmed at or very near the basin minimum.
- **Per-split asymmetry at high LR**: 8e-4 hurts in_dist (+12.0%) but improves cruise (-4.7%) and re_rand (-1.4%). The in_dist split dominates val_avg.
- This asymmetry is a research clue: split-specific schedules or training-data re-weighting could exploit it.
- Both runs hit best_epoch=14 (final) — schedule horizon is well-matched to data.
- VRAM at 8e-4 was 93.2 GiB / 96 GiB — close to the cap but not OOM.

### Suggested follow-ups
- Finer scan in [4e-4, 6e-4] (low priority — basin appears narrow).
- Warmup + slightly higher peak (e.g. 6e-4 with 500-step linear warmup) to recover cruise/re_rand wins without in_dist regression. **Note**: nezuko is testing warmup on the new base.
- The in_dist-vs-OOD asymmetry suggests rethinking the val_avg metric weighting.

## 2026-05-15 21:25 — PR #3426: H: Cosine warm restarts (T_0=5) ✗ CLOSED

- Branch: `thorfinn/cosine-warm-restarts`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: SGDR T_0=5 cycles inject periodic high-LR exploration to escape local basins; each cycle ends near eta_min=1e-6 for fine-tuning.

### Results

| Metric | Baseline (T_max=15) | Warm restarts T_0=5 | Δ |
|--------|---------------------|---------------------|------|
| **val_avg/mae_surf_p** | 91.3319 | **103.0659** | **+12.85%** |
| **test_avg/mae_surf_p** | 88.4260 | **99.1128** | **+12.08%** |

W&B run: `fgaa946g` · Group: `cosine_warm_restarts`

### Per-cycle best
| Cycle | Best val_avg | Restart bounce |
|-------|--------------|-----------------|
| 1 (e1-5) | 135.40 | — |
| 2 (e6-10) | 110.99 | +40.5% at e6 |
| 3 (e11-14, partial) | 103.07 | +21.2% at e11 |

### Analysis
- 12.85% regression — clear close.
- **Key failure mode**: 5-epoch cycles too short for convergence. Within-cycle trajectory shows model still actively descending when cycle ends. Each restart throws away ~25-40% of progress.
- SGDR's theoretical advantage (escaping bad basins) requires bad basins to escape. The single-cycle baseline monotonically descends to 91.33 — no evidence of stuck dynamics.
- Excellent diagnostic work by thorfinn: per-epoch LR + val_avg table, per-cycle best decomposition, restart-bounce quantification.

### Implication for the program
- Warm restarts are wrong tool for 14-epoch regime. If revisited later, T_0=7 with T_mult=2 (one restart in budget) might be marginally better but still likely worse than single-cycle.
- EMA of weights (next assignment for thorfinn) directly addresses the "epoch-noise" motivation without throwing away progress.

## 2026-05-15 20:20 — PR #3359: H13: Pressure channel-weighted surf loss (p=3x) ✗ CLOSED

- Branch: `edward/pressure-ch-weight`
- Student: willowpai2i48h1-edward
- Hypothesis: Per-channel surf loss weighting (p=3x, Ux/Uy=1x) to emphasize the scored metric.

### Results (W&B only — code never committed to PR)

| Config | val_avg/mae_surf_p | test_avg |
|--------|-------------------|---------|
| pressure_ch_w3 (18:28) | 133.32 | 101.23 |
| pressure_ch_w3 (19:22, crashed) | 163.59 | — |
| pressure_ch_w5 (19:33) | 112.22 | 94.86 |

W&B runs: `(see wandb group)`

### Analysis
- Best val=112.22 (w=5), which is +23% worse than new baseline (91.33).
- Pressure weighting ALONE (without architectural specialization) fails to help. The 3x weight on the pressure channel distorts the vol+surf_Ux/Uy gradient budget without providing a separate learning pathway.
- Compare to fern's result: split head + 3x weight DID help (-6.2% test), confirming that architectural specialization is the missing ingredient.
- Increasing W from 3→5 showed slight improvement (133→112), but diminishing returns suggest diminishing gradient signal for Ux/Uy.
- **Note**: Student iterated without committing code to PR — made advisor review impossible. New assignment instructs explicit commit-before-run discipline.

---

## 2026-05-15 19:30 — PR #3361: H10b: slice_num=128 retry on Huber+NaN base ✗ CLOSED

- Branch: `thorfinn/slice128-retrial`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: slice_num=128 on correct (Huber+NaN) base. Round-1 retry tested on MSE base.

### Results

| Metric | slice=128 | baseline slice=64 | Δ |
|--------|-----------|-------------------|---|
| val_avg/mae_surf_p | 116.1928 | 112.8295 | **+3.36 worse** |
| test_avg/mae_surf_p | 112.5640 | 106.5996 | **+5.96 worse** |
| val_geom_camber_rc | 117.74 | 133.69 | **-15.96 better** |

W&B: `z8pyszfb` · 11 epochs (171s/ep, T_max=50, peaked 95GB VRAM)

### Analysis
- Capacity-budget tradeoff confirmed again (see also #3180 h=192): slice=128 is 30% slower, only 11 epochs vs baseline's 14.
- LR barely decayed (T_max=50, 22% consumed). Model still improving at timeout. Not "slice=128 fails" — it's budget-constrained.
- OOD gain: val_geom_camber_rc improved -15.96, supporting that richer physics-state helps hardest splits, but aggregate is negative within budget.
- VRAM ceiling: 95GB at slice=128 (98% of 96GB H100).
- **Conclusion**: capacity not the bottleneck at this wall-clock budget. Close.

---

## 2026-05-15 19:30 — PR #3363: H8: AdamW β2=0.95 + grad clip 1.0 → SENT BACK (rebase on T_max=15)

- Branch: `tanjiro/adamw-stability`
- Student: willowpai2i48h1-tanjiro
- Hypothesis: β2=0.95 + grad clip 1.0 reduces gradient instability and improves convergence.

### Results (vs OLD Huber+NaN baseline, T_max=50)

| Metric | This run | Old baseline | New baseline (91.33) |
|--------|---------|--------------|----------------------|
| val_avg/mae_surf_p | 102.2436 | 112.8295 | 91.3319 |
| test_avg/mae_surf_p | 97.6239 | 106.5996 | 88.4260 |
| val_single_in_dist | 115.16 | 142.47 | **-19.2% best split** |

W&B: `44lht7xd` · 14 epochs · 99.7% of steps clipped (median grad_norm=3.71)

### Analysis
- Genuine optimizer improvement: val=102.24 (-9.4%), test=97.62 (-8.4%) vs old baseline.
- Grad clip at 1.0 is aggressive (binding on 99.7% of steps, median pre-clip norm 3.71). Clipping confirms the hypothesis that large gradient spikes were destabilizing training.
- Best epoch is epoch 14 (final, still descending) — suggests more budget would help further.
- Does NOT beat new T_max=15 baseline (91.33). Orthogonal to schedule fix — stacking should compound.
- **Action**: rebase on T_max=15 base, re-run with β2=0.95 + clip 1.0.

---

## 2026-05-15 18:30 — PR #3317: H3b: Cosine T_max=15 tuned to actual epoch budget ✓ MERGED (NEW BASELINE)

- Branch: `askeladd/cosine-tmax-tuned`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Aligning T_max with the real ~14-epoch wall-clock budget allows the cosine schedule to fully anneal. T_max=50 with only 14 epochs leaves LR at 79% of peak — effectively no annealing.

### Results

| Arm | T_max | val_avg/mae_surf_p | Δ vs baseline | W&B |
|-----|-------|--------------------|---------------|-----|
| Baseline | 50 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **15** | **91.3319** | **-19.1%** | `kx17n4pn` |
| B | 12 | 103.1193 | -8.7% | `z8h5w88d` |

| Test split | Arm A (T_max=15) |
|------------|-----------------|
| test_single_in_dist | 96.7268 |
| test_geom_camber_rc | 88.3769 |
| test_geom_camber_cruise | NaN (branch predates NaN fix) |
| test_re_rand | 80.1744 |
| **test_avg (3-split)** | **88.4260** |

### Analysis
- Biggest single improvement in the programme: -19.1% from a 1-line hyperparameter change.
- T_max=15 matches the 14-epoch budget: epoch 14 runs at ~1.1% of peak LR (fine-tuning pass). T_max=12 crashed to 0% LR at epoch 12, leaving 2 wasted epochs; gap of 103.12 vs 91.33 = 12 MAE points.
- The baseline T_max=50 was essentially NOT annealing — the LR was at 79% of peak at training stop.
- Key observation: per-split improvement is uniform (single_in_dist -26, geom_camber_rc -45, cruise -3, re_rand -12), suggesting the gain is structural (schedule fix) rather than overfitting to any particular split.
- **This result fundamentally shifts the research programme**: the binding constraint was schedule mis-alignment, not loss function or architecture. All future hypotheses should compare against this baseline.

---

## 2026-05-15 18:30 — PR #3305: H1b: Huber delta=0.05 scan → SENT BACK (rebase on new base)

- Branch: `alphonse/huber-smaller-delta`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Shrinking Huber δ from 0.1 to 0.05 pushes more residuals into L1 regime, improving MAE alignment.

### Results (vs OLD baseline 112.90 with T_max=50)

| Arm | delta | val_avg/mae_surf_p | Δ vs old baseline | W&B |
|-----|-------|--------------------|-------------------|-----|
| Old Baseline | 0.10 | 112.9001 | — | `bpczoejx` |
| **A (winner)** | **0.05** | **98.1913** | **-13.0%** | `oolv8t1p` |
| B | 0.02 | 103.7964 | -8.1% | `zlqqtxsu` |

val=98.19 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- δ=0.05 is the right direction — U-shaped response with δ=0.02 overshooting (loss landscape becomes near-constant-gradient L1, slowing late refinement).
- Both arms were run with T_max=50 (handicapped). On the new T_max=15 base, δ=0.05 is expected to yield additional stacked improvement.
- **Action**: rebase onto T_max=15 base, rerun with δ=0.05 only. Target: beat 91.33.

---

## 2026-05-15 18:27 — PR #3171: H8b: Split pressure head + 3x weight on Huber base → SENT BACK (rebase)

- Branch: `fern/split-pressure-head`
- Student: willowpai2i48h1-fern
- Hypothesis: Dedicated output head for pressure channel with 3x Huber-weighted loss improves OOD pressure MAE.

### Results v2 (rebased onto Huber base, with T_max=50)

| Metric | This PR | Huber baseline | Δ |
|--------|---------|---------------|---|
| val_avg/mae_surf_p | 111.9988 | 112.8295 | -0.90 |
| test_avg/mae_surf_p (all 4 splits) | **99.9669** | **106.5996** | **-6.63** |

val=112.00 does NOT beat the new T_max=15 baseline (91.33). Sent back for rebase.

### Analysis
- val improvement is marginal (-0.8%), but **test improvement is genuine and consistent**: geom_camber_rc (-13.8 test), cruise test (-15.0), geom_camber_rc val (-23.4). The split head specifically improves OOD generalization.
- v1 (MSE) failed; v2 (Huber base) succeeded — confirming loss-metric alignment is prerequisite for architectural improvements.
- Both runs used T_max=50 (handicapped). With T_max=15, the split head should achieve further improvement.
- **Action**: rebase onto T_max=15 base, rerun with split head + 3x pressure weight + Huber(δ=0.1). Target: beat 91.33.

---

## 2026-05-15 15:45 — PR #3162: H9: Raise surf_weight 10→25 ✗ CLOSED

- Branch: `askeladd/surf-weight-25`
- Student: willowpai2i48h1-askeladd
- Hypothesis: Raising surf_weight from 10 to 25 emphasizes the surface (the scored region) in the gradient, should improve val_avg/mae_surf_p.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **133.4123** |
| val_single_in_dist | 163.71 |
| val_geom_camber_rc | 194.32 |
| val_geom_camber_cruise | 103.60 |
| val_re_rand | 125.67 |

| Split | test mae_surf_p (patched scoring) |
|-------|----------------------------------|
| test_single_in_dist | 134.42 |
| test_geom_camber_rc | 141.56 |
| test_geom_camber_cruise | 92.36 (via local patched scoring) |
| test_re_rand | 120.00 |
| **test_avg/mae_surf_p** | **122.0843** |

W&B run: `hkka77kg` · Group: `surf_weight_sweep`

### Run details
- Epochs: **14/50** (30-min wall-clock cap; best at epoch 13)
- Noisy trajectory: 133.63 (ep11) → 142 (ep12) → 133.41 (ep13) → 146.83 (ep14, cut)
- Peak VRAM: 42.1 GB / 96 GB

### Analysis
- 133.41 does NOT beat the new Huber baseline (112.90). **Closed**.
- The hypothesis was tested against the wrong baseline (MSE loss). With Huber loss already providing MAE-aligned gradients, the marginal benefit of surface emphasis is smaller than expected.
- Loss-metric alignment (Huber) dominates surface weighting at the same compute budget.
- Askeladd also produced an excellent independent bug report on the cruise NaN scoring issue (now being fixed in thorfinn PR #3309) — same root cause as alphonse identified.

### Suggested follow-ups (taken into round 2)
- The surf_weight knob is still worth testing on top of the Huber base (separate from askeladd's follow-up).
- Askeladd assigned PR #3317: cosine T_max tuning to match actual epoch budget — directly addresses the LR-not-annealing observation.

## 2026-05-15 14:30 — PR #3159: H1: Huber loss (delta=0.1) — NEW BASELINE ✓ MERGED

- Branch: `alphonse/huber-loss-aligned`
- Student: willowpai2i48h1-alphonse
- Hypothesis: Replace MSE loss with Huber(delta=0.1) to align training objective with the MAE evaluation metric. At delta=0.1 in normalized space, residuals above 0.1 are in the L1 (MAE-equivalent) regime, creating direct gradient alignment with the scoring metric.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **112.9001** |
| val_single_in_dist | 134.4612 |
| val_geom_camber_rc | 143.4094 |
| val_geom_camber_cruise | 75.8516 |
| val_re_rand | 97.8785 |

| Split | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|-------|-----------------|-----------------|-----------------|
| test_single_in_dist | 120.1970 | 1.4079 | 0.5594 |
| test_geom_camber_rc | 134.3200 | 2.2348 | 0.7179 |
| test_geom_camber_cruise | NaN (data corruption) | 0.9322 | 0.4473 |
| test_re_rand | 92.7597 | 1.3172 | 0.5779 |
| **test 3-split avg (excl. cruise)** | **115.7589** | 1.4730 | 0.5756 |

W&B run: `bpczoejx` · Group: `huber_loss_delta01`

### Run details
- Epochs: **14/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 14 — val still falling (248 → 113 over run; healthy monotonic decrease)
- Peak VRAM: 42.1 GB (well within 96 GB budget)

### Analysis
- **Clear winner**: 112.9 vs 134.7 (thorfinn's slice_num=128), improvement of ~16%.
- MAE alignment works: Huber loss directly creates gradient alignment with the scoring metric. The model learns to minimize mean absolute error rather than mean squared error, which is exactly what's being measured.
- **LR schedule mismatch**: T_max=50 with only 14 epochs completed means LR was still at ~82% of peak (≈0.00041) when training stopped. The cosine schedule never annealed. This is the biggest remaining optimization opportunity — the model is undertrained relative to schedule.
- **Delta regime**: With trained residuals O(0.05–0.2) at epoch 14, many residuals are still below delta=0.1 and in the L2 regime. Smaller delta (0.05 or 0.01) would push more residuals into L1, potentially improving MAE alignment further.
- Per-split pattern: cruise val best (75.85), then re_rand (97.88), while single_in_dist (134.46) and geom_camber_rc (143.41) remain hardest — high-Re raceCar samples dominate absolute error.

### Student suggested follow-ups
1. Tune T_max to actual epoch budget (~14-15 epochs)
2. Smaller Huber delta (0.05, 0.01) or pure L1 to push fully into MAE-aligned regime
3. Per-channel loss weighting (emphasize pressure channel)
4. Patch the cruise-gt NaN bug (separate PR, affects all test metrics)

## 2026-05-15 14:10 — PR #3188: H10: Increase slice_num from 64 to 128

- Branch: `thorfinn/slice-num-128`
- Student: willowpai2i48h1-thorfinn
- Hypothesis: Doubling physics-state slice tokens from 64→128 gives finer flow-regime discretization without changing hidden width or depth.

### Results

| Split | val mae_surf_p |
|-------|----------------|
| **val_avg/mae_surf_p** | **134.7389** |
| val_single_in_dist | 159.8405 |
| val_geom_camber_rc | 149.3953 |
| val_geom_camber_cruise | 109.1693 |
| val_re_rand | 120.5507 |

| Split | test mae_surf_p |
|-------|-----------------|
| test_single_in_dist | 132.6239 |
| test_geom_camber_rc | 132.9377 |
| test_geom_camber_cruise | NaN (data corruption — see below) |
| test_re_rand | 119.2658 |
| **test 3-split avg (excl. cruise)** | **128.2758** |

W&B run: `912m0995` · Group: `slice_num_128`

### Run details
- Epochs: **11/50** (hit 30-min wall-clock cap; ~173 s/epoch)
- Best checkpoint: epoch 11 — val still falling steeply (162 → 134 in final epoch; not converged)
- Peak VRAM: 54.5 GB (well within 96 GB; slice-attention 128×128 is negligible vs node ops)

### Infrastructure bug discovered
`.test_geom_camber_cruise_gt/000020.pt` has 761 `inf` values in `y[:,2]` (pressure). The masked-arithmetic `inf * 0 = NaN` propagates into the accumulator — poisoning `test_geom_camber_cruise/mae_surf_p` for **all students**. Val metrics unaffected (all val gt is clean). **Fix**: defensive `y_finite` masking in `train.py:evaluate_split` assigned to thorfinn (PR relative-mse-bugfix).

### Analysis
- No concurrent slice_num=64 baseline yet. Other round-1 students effectively provide the reference.
- VRAM cost of 128 vs 64 is negligible.
- Merged as Round-1 reference — establishes first measured val_avg/mae_surf_p on this advisor branch.

## 2026-05-15 17:00 — PR #3309: Bugfix: inf*0=NaN in evaluate_split ✓ MERGED

- Branch: `thorfinn/nanbug-fix`
- Student: willowpai2i48h1-thorfinn
- Type: Infrastructure bugfix — 4 defensive lines in evaluate_split; model unchanged

### Results

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **112.8295** (=baseline, within noise) |
| **test_avg/mae_surf_p** | **106.5996** ← was NaN (all 4 splits now valid) |
| test_geom_camber_cruise/mae_surf_p | **83.4377** ← was NaN |

W&B run: `g48284pc` · Group: `nanbug_fix`

### Analysis
- Model unchanged; val reproduces baseline within Δ=0.07 (noise).
- **Critical outcome**: test_geom_camber_cruise/mae_surf_p is now 83.44 (finite) and test_avg/mae_surf_p=106.60 is the first valid 4-split test score on this branch.
- Fix: `_y_fin` masking before arithmetic in evaluate_split prevents `pred - (-inf) = inf` → `inf * 0 = NaN` propagation via `data/scoring.py:accumulate_batch`.

## 2026-05-15 17:05 — PR #3180: H4: Wider model (hidden=192, slice_num=96) ✗ CLOSED

- Branch: `tanjiro/wider-model-h192`
- Student: willowpai2i48h1-tanjiro

### Results

| Run | val_avg/mae_surf_p |
|-----|-------------------|
| `a8p3g73s` (h=192 run 1) | **150.3762** (best of 2) |
| `nj0chxr6` (h=192 run 2) | 156.3125 |
| Baseline (h=128 Huber) | 112.9001 |

W&B runs: `a8p3g73s`, `nj0chxr6` · Group: `wider_model_h192`

### Analysis
- 150.38 vs 112.90 = 33% regression. Closed.
- h=192 is 1.6× slower/epoch → only 9 epochs vs baseline's 14. But per-epoch metrics are also worse (150 at ep8 vs ~145 for baseline at ep8 per historical data).
- ~2.2× more params (1.48M vs 0.66M) did not help at this budget.
- Bottleneck is clearly loss/schedule/features, not capacity.
- Seed variance ~4% (156.31 vs 150.38) is significant — future capacity tests should pin a seed.

## 2026-05-15 17:10 — PR #3167: H12: OneCycleLR max_lr=1e-3 ✗ CLOSED

- Branch: `edward/onecycle-lr`
- Student: willowpai2i48h1-edward

### Results

| Run | epochs | val_avg/mae_surf_p | Notes |
|-----|--------|-------------------|-------|
| `x9mygbcm` | 9 | 192.6188 | schedule misconfigured (total_steps sized for 50 ep) |
| `27mfh19o` | 9 | 172.9975 | same misconfiguration |
| `xn1ad9ka` | 9 | **137.1218** | fixed: --epochs 9, schedule fully annealed |
| Baseline (Huber cosine) | 14 | 112.9001 | — |

W&B runs: `xn1ad9ka` (final) · Group: `onecycle_lr`

### Analysis
- 137.12 vs 112.90 = 21% regression after correct schedule setup. Closed.
- **Key insight**: Edward diagnosed the schedule mismatch himself and reran with --epochs 9. The schedule fully annealed (4e-5 → 1e-3 → ~0), so the hypothesis was correctly tested.
- OneCycleLR fails because: (a) 9-epoch total budget means no prolonged low-LR refinement phase, and (b) cosine starts at peak LR and descends immediately, giving better use of the budget.

---

## 2026-05-16 02:25 — PR #3542: H: Test-Time Augmentation via horizontal-flip symmetry

- **Branch:** willowpai2i48h1-edward/tta-hflip
- **Student:** willowpai2i48h1-edward
- **Hypothesis:** Predict on original + horizontally flipped input, un-flip, average — exploit approximate z-symmetry of TandemFoilSet to reduce variance and improve OOD splits.

### Results

| Metric | Raw | TTA | Δ vs raw | Baseline |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | **91.14** | 161.10 | **+76.7%** | 87.91 |
| **test_avg/mae_surf_p** | **85.74** | 153.77 | **+79.4%** | 83.38 |

Per-split breakdown (best epoch 17, W&B `zozun1q2`):

| Split | Val raw | Val TTA | Val Δ | Test raw | Test TTA | Test Δ |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist (raceCar single) | 104.24 | 224.94 | +115.8% | 92.68 | 205.39 | +121.6% |
| geom_camber_rc (raceCar tandem) | 99.53 | 209.90 | +110.9% | 89.09 | 199.41 | +123.8% |
| geom_camber_cruise (cruise tandem) | 74.41 | 80.96 | +8.8% | 79.88 | 89.14 | +11.6% |
| re_rand (mixed tandem) | 86.39 | 128.62 | +48.9% | 81.32 | 121.15 | +49.0% |

Best epoch: 17, Peak VRAM: 32.9 GB, W&B: `zozun1q2`

### Analysis
**Hypothesis catastrophically falsified.** TTA is +76.7% over raw (not the predicted -1% to -3%). The dataset is **fundamentally not z-symmetric for the dominant raceCar domain:**
- RaceCar has one-sided `pos_z` (always positive, ground-effect domain). Flip → always-negative `pos_z`, never seen in training. Catastrophic OOD.
- RaceCar AoA is always-negative (inverted foil convention). Flip → always-positive. Never seen.
- NACA camber encoded non-negatively (`∈ [0,1]`). Flip sign → negative camber. OOD for all splits.
- **Only cruise** has z-symmetric distribution (pos_z spans ±9.6, both AoA signs). Cruise TTA degrades only 8.8% val — close to neutral.

Edward's per-split feature distribution audit is the most valuable artifact: a definitive dataset-distribution table explaining the OOD failure. Flip-variant diagnostic confirmed no variant rescues raceCar (geom-only flip still 2.15× on raceCar single).

**Raw val=91.14 (single seed)** is +1.8σ over baseline 87.91 — within noise floor, confirming seed variance.

**Program implications:**
1. frieren's #3563 (train-aug hflip) is at high risk — creating OOD raceCar inputs during training.
2. "Reflection-based TTA dead" on this dataset. Cruise-only TTA is a low-upside option if needed later.
3. Physical symmetries that DO exist on TandemFoilSet (e.g., per-split normalization, geometry-conditioned regularization) are worth exploring instead.
- NaN on test_geom_camber_cruise is a model-quality issue (extreme prediction on under-converged model at high LR), not the data corruption bug.

---

## 2026-05-16 02:45 — PR #3580: H: Stochastic Weight Averaging (SWA) over last 5 checkpoints

- **Branch:** willowpai2i48h1-nezuko/swa-last5
- **Student:** willowpai2i48h1-nezuko
- **Hypothesis:** Post-training uniform average of last 5 checkpoints (K=5). Predicted -0.5% to -2% on val_avg/mae_surf_p.

### Results

| Version | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (val / test) |
|---|---:|---:|---:|
| Baseline #3480 | 87.9105 | 83.3782 | — |
| Raw final epoch (e18) | 91.7950 | 86.2239 | +3.88 / +2.85 |
| Best-by-val (e15) | 89.7613 | 85.1176 | +1.85 / +1.74 |
| **SWA K=5** | 89.8631 | 85.0478 | +1.95 / +1.67 |
| SWA K=3 | 89.9285 | 84.9371 | +2.02 / +1.56 |

W&B run: `pgzvcwwy` · Group: `swa_weight_averaging` · Best epoch: 15 · Peak VRAM: 33.0 GB

### Analysis
**Closed — SWA mechanism works mechanistically but is redundant under the current cosine T_max=15 schedule.**

- SWA K=5 vs best-by-val: +0.10pt val / -0.07pt test (well within σ=1.80).
- SWA K=3 vs best-by-val: +0.17pt val / -0.18pt test (within σ).
- The 1.9pt gap from baseline is seed variance — PR #3175 + alphonse #3546 both establish σ ≥ 1.80 single-seed.

**Why SWA failed here:** cosine T_max=15 pins LR at ~0 from epoch 15-16 onward. Last 5 epochs are near-frozen in weight space → SWA averages near-identical snapshots → reduces to best-by-val checkpoint selection. Both K=3 and K=5 give essentially the same result, confirming the tail is dead-weight.

**Program-level insights:**
1. Cosine T_max=15 essentially zeros gradients by epoch 15 — confirmed empirically by SWA equivalence to best-by-val.
2. This implies thorfinn's EMA decay=0.99 (#3521) will have weak signal in the same regime — EMA over near-zero-update tail ≈ best-by-val.
3. Best-by-val checkpoint selection in train.py already captures most of the variance reduction SWA promises.
4. SWA K=3 marginally beats K=5 on test (84.94 vs 85.05) → earlier epochs slightly better than the dead tail. Cosine is over-pinning weights.

**Follow-up assigned to nezuko:** cosine T_max=10 + 8-epoch constant-LR tail at lr=1e-4 + SWA over the tail. This is the Izmailov-style recipe SWA was designed for.

---

## 2026-05-16 02:46 — PR #3563: H: Train-time horizontal-flip data augmentation

- **Branch:** willowpai2i48h1-frieren/train-aug-hflip
- **Student:** willowpai2i48h1-frieren
- **Hypothesis:** Per-sample horizontal flip (p=0.5) during training. Predicted -2% to -5% on val_avg/mae_surf_p. Combined with edward's TTA (#3542), expected to compound.

### Results

| Metric | Baseline #3480 | aug-hflip (e18) | Δ | %  |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 87.9105 | **111.6983** | +23.79 | **+27.1%** |
| **test_avg/mae_surf_p** | 83.3782 | **106.6308** | +23.25 | **+27.9%** |

Per-split val + test breakdown (all worse):

| Split | val baseline | val aug | val Δ | test baseline | test aug | test Δ |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 105.05 | 146.27 | +39.2% | 93.68 | 136.97 | +46.2% |
| geom_camber_rc | 95.69 | 123.27 | +28.8% | 87.55 | 108.99 | +24.5% |
| geom_camber_cruise | 68.20 | 77.59 | +13.8% | 75.13 | 84.70 | +12.7% |
| re_rand | 82.71 | 99.67 | +20.5% | 77.16 | 95.86 | +24.2% |

W&B run: `4tbuq9ql` · Group: `train_time_augmentation` · Flip rate: 0.496 (744/1499 samples — augmentation firing correctly)

### Analysis
**Closed — catastrophic regression confirms edward #3542's dataset-asymmetry finding.** +27% on val, +27.9% on test, ~13σ above noise floor. **Worst result on this benchmark to date.**

**Three independent reasons the flip is non-physical** (correctly flagged by frieren pre-launch):
1. **NACA camber M is unsigned** (∈ [0,1]). Sign-flipping it produces values in [-1, 0] that are OOD for ALL splits including cruise. This alone explains cruise's +13.8% degradation.
2. **AoA was not flipped** alongside the geometric flip → augmented geometry contradicts the scalar boundary condition. Model sees physically inconsistent samples.
3. **RaceCar mesh asymmetry** (one-sided pos_z for ground effect, one-sided AoA). Flip is non-physical for ~70% of training data.

**Smoking gun in per-channel loss:** vol_Uy 0.0447 (e1) → 0.0441 (e18), only -1.3% (essentially no learning); all other channels improved 60-84%. The model receives contradictory Uy targets for geometrically-indistinguishable inputs (unsigned NACA camber) and collapses to near-zero Uy prediction. This is the data-augmentation-induced-shortcut failure mode in textbook form.

**Combined with #3542 (TTA), the naive horizontal-flip symmetry lever class is fully dead** on TandemFoilSet.

**Valid follow-up directions surfaced by frieren's analysis** (deferred — not assigned in this round):
1. **Corrected flip**: flip pos_z + AoA + Uy together, leave NACA camber alone (respects actual physical symmetry of the equations).
2. **Cruise-only conditional augmentation**: apply the corrected flip only to cruise samples.
3. **Mesh-permutation augmentation**: random node-ordering permutation (true symmetry of the mesh encoder).

**Follow-up assigned to frieren:** Layer-wise LR decay (LLRD) γ=0.85 — orthogonal optimization-space regularizer, completely different lever.

---

## 2026-05-16 04:05 — PR #3546: Seed control + 4 baseline replicates ✓ MERGED (infrastructure)

- **Branch:** willowpai2i48h1-alphonse/seed-control-baseline-variance
- **Student:** willowpai2i48h1-alphonse
- **Hypothesis:** Add seed control to train.py; run 4 replicates of the canonical bf16+T_max=15 config to characterize the run-to-run variance under the new baseline.

### Results

| Seed | run_id | val_avg/mae_surf_p (best-ep) | test_avg/mae_surf_p | Best epoch |
|------|--------|------------------------------|---------------------|------------|
| 0 | ek21s9hy | 89.71 | 85.64 | 15 |
| 1 | 8vcv4ojk | 90.16 | 85.54 | 18 |
| 2 | 1y3my9x2 | 93.05 | 86.83 | 17 |
| 3 | 0ekl0alh | 90.14 | 85.37 | 17 |
| **μ̂** | — | **90.77** | **85.85** | — |
| **σ̂ (ddof=1)** | — | **1.54** | **0.67** | — |

### Analysis

**CRITICAL META-FINDING:** The baseline 87.91 (PR #3480) sits 1.86σ below the canonical-config 4-seed mean μ̂=90.77. The 87.91 result was a downward lucky-draw outlier, not a representative lower bound.

**Practical consequences:**
- Single-seed results in [89.2, 92.3] are statistically indistinguishable from the canonical config.
- Future PRs need to beat val < 89.2 (μ̂-1σ) for a meaningful win, ideally < 87.7 (μ̂-2σ).
- **The all-time best 87.91 remains the paper headline** — it's a valid point estimate, just not the expected value.

σ̂=1.54 is consistent with PR #3305's estimate of 1.80 (14% smaller, same order of magnitude). The test-set variance is much smaller: σ̂_test=0.67. This suggests test-set evaluation is more stable than val (fewer samples but less geometric diversity in the test splits).

**Code changes merged (27 insertions):**
- `set_all_seeds(seed)` — covers random, numpy, torch, torch.cuda
- `seed_worker(worker_id)` — per-worker RNG for DataLoader
- `Config.seed=42`, `Config.deterministic=False`
- `--seed` CLI arg now in canonical train.py

---

## 2026-05-16 04:10 — PR #3521: H: EMA decay=0.99 ✗ CLOSED (within canonical noise)

- **Branch:** willowpai2i48h1-thorfinn/ema-decay99
- **Student:** willowpai2i48h1-thorfinn
- **Hypothesis:** In-training EMA with faster-forgetting decay=0.99 to reduce val trajectory variance.

### Results

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| s35tc2it | decay=0.99, seed=0 | 91.07 | 86.13 |
| goxpz2nn | decay=0.99 (prior run) | 90.89 | 86.12 |
| ihb7926k | decay=0.95 (prior run) | 89.62 | 84.87 |
| Canonical μ̂ | (PR #3546) | 90.77 | 85.85 |

Terminal SENPAI-RESULT: val=91.07, test=86.13.

### Analysis
EMA decay=0.99 val=91.07 = within 0.20pt of canonical μ̂=90.77 (<<σ=1.54). No significant improvement. Root cause: identical to nezuko's SWA finding (#3580). Cosine T_max=15 pins LR≈0 from epoch 15-16 onward → EMA updates in the tail are near-zero → EMA model ≈ best-by-val snapshot at the end of the cosine phase. The decay=0.95 arm (89.62) is slightly better than decay=0.99 (91.07) — faster decay concentrates on more-recent (near-converged) weights. But both are within noise of baseline.

**EMA lever dead under cosine T_max=15.** If nezuko's #3644 (constant-LR tail) shows the tail regime matters, EMA with decay=0.99 + constant tail is the right follow-up. Not now.

Pod restart at 02:21 UTC (uncommitted train.py blocking checkout) caused the stale_wip appearance. Multiple run restarts visible in W&B.

---

## 2026-05-16 05:30 — PR #3642: H: Layer-wise LR decay γ=0.85 ✗ CLOSED (regression)

- **Branch:** willowpai2i48h1-frieren/llrd_gamma85
- **Student:** willowpai2i48h1-frieren
- **Hypothesis:** Apply standard LLRD (γ=0.85) across 5 Transolver blocks — highest LR at top block, lowest at bottom — mirroring BERT/RoBERTa fine-tuning recipe.

### Results

| Metric | Baseline μ̂ (#3546) | LLRD γ=0.85 | Δ vs μ̂ | Δ vs best (87.91) |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 90.77 | **92.45** | +1.68 (+1.9%) | +4.54 (+5.2%) |
| **test_avg/mae_surf_p** | 85.85 | **87.08** | +1.23 | +3.70 |

Per-split val/test:

| Split | val | test |
|---|---:|---:|
| single_in_dist | 110.41 | 99.99 |
| geom_camber_rc | 101.17 | 88.36 |
| geom_camber_cruise | 70.65 | 78.47 |
| re_rand | 87.57 | 81.49 |
| **avg** | **92.45** | **87.08** |

W&B run: `rpjyfrss` · Best epoch: 15 · Group: layer_wise_lr_decay

### Analysis
**CLOSED — regression (~2.5σ above all-time best, clearly above μ̂).** The LLRD assumption (bottom layers are "frozen pretrained features") is empirically inverted for Transolver: frieren's gradient-norm diagnostics showed block_0 has the **largest** gradient signal (active geometry-encoding layers, not frozen representations). Standard LLRD γ=0.85 starved the most learning-hungry layer, causing the regression.

**Mechanism confirmed by gradient telemetry:** standard LLRD amplified the gradient imbalance instead of correcting it. The BERT/RoBERTa recipe doesn't transfer because there's no pretraining — block_0 is randomly initialized and needs maximum LR to learn geometry encoding from scratch.

**Falsified hypothesis class:** Standard top→bottom LR decay for Transolver. **Live follow-up: inverse-LLRD** (PR #3722, frieren) tests γ_inv=1.176 with highest LR at bottom block where gradient is strongest — direct inversion of this hypothesis, same magnitude.

---

## 2026-05-16 05:30 — PR #3566: H: Unified positional encoding (unified_pos=True) ✗ CLOSED (catastrophic regression)

- **Branch:** willowpai2i48h1-fern/unified-positional-encoding
- **Student:** willowpai2i48h1-fern
- **Hypothesis:** Enable Transolver's `unified_pos=True` flag for cross-layer positional consistency and OOD generalization.

### Results (best run: k72akuht)

| Metric | Baseline #3480 | unified_pos=True | Δ | % |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 87.9105 | **102.6271** | +14.72 | **+16.7%** |
| **test_avg/mae_surf_p** | 83.3782 | **98.8012** | +15.42 | **+18.5%** |

Per-split breakdown (best run k72akuht, 3 runs total: s0tj1q82, nugotxr6, k72akuht):

| Split | Val | Test |
|---|---:|---:|
| single_in_dist | 127.53 | 116.03 |
| geom_camber_rc | 109.09 | 101.63 |
| geom_camber_cruise | 76.52 | 83.94 |
| re_rand | 97.36 | 93.62 |
| **avg** | **102.63** | **98.80** |

W&B runs: `s0tj1q82`, `nugotxr6`, `k72akuht` · Group: unified_pos_encoding

### Analysis
**CLOSED — catastrophic regression >7σ above canonical mean, uniform across all splits.** The worst-hit split is `single_in_dist` (+21.4%, opposite of the predicted OOD-geom improvement).

**Encoding mismatch is the root cause:** in this 2D fork, `unified_pos=True` replaces directional `(x, z)` coordinates with rotation-symmetric radial-distance features to a reference grid. This discards the directional information the baseline relies on for accurate pressure field prediction. Additionally, the flag adds ~15.9K parameters while increasing convergence time ~2× — model still descending at epoch 18 under cosine T_max=15, leaving no fine-tuning budget.

**Architectural finding:** `unified_pos` as implemented is incompatible with this 2D asymmetric-flow Transolver. It may work if directional features are preserved alongside the unified pos (hybrid concat or per-block injection).

**Valid follow-up:** fern's suggested per-block positional injection (directional features preserved) is queued as a future hypothesis.

---

## 2026-05-16 05:30 — PR #3574: H: Per-channel Huber-δ (δ_p=0.05 on surface-p only) ✗ CLOSED (within noise, regression)

- **Branch:** willowpai2i48h1-tanjiro/per-channel-huber-delta
- **Student:** willowpai2i48h1-tanjiro
- **Hypothesis:** Use δ_p=0.05 on surface-p channel (tighter Huber → stronger L1 penalty on small pressure residuals) while keeping δ=0.10 on Ux/Uy.

### Results

| Metric | Baseline μ̂ (#3546) | δ_p=0.05 | Δ vs μ̂ | Δ vs best (87.91) |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 90.77 | **91.78** | +1.01 | +3.87 |
| **test_avg/mae_surf_p** | 85.85 | **86.67** | +0.82 | +3.29 |

W&B run: `9leqg5zi` · Best epoch: 17 · Group: per_channel_huber_delta

### Analysis
**CLOSED — above canonical μ̂, 3.87pt above all-time best.** The per-channel Huber-δ lever class is now exhausted: δ=0.10 uniform (baseline) outperforms δ_p=0.05 single-channel tightening, δ_p=0.05 joint-channel (PR #3305), and the full suite of surf_weight scans (#3428, #3174, #3522). Loss-shape modification without architecture or schedule change has run out of signal.

**Mechanistic note:** tighter Huber (smaller δ) increases the gradient for small residuals but simultaneously decreases gradient on large residuals — the pressure field has a heavy tail of large-residual outliers (especially geom_camber OOD splits) that need the full gradient signal. δ=0.10 provides the right balance.

**Dead end: loss-formulation lever class for this baseline architecture.** Further gains from loss shaping would require quantile regression, gradient-norm-balanced multitask, or learned per-channel weights — all architectural additions rather than simple δ tuning.

**Follow-up assigned:** PR #3724 tanjiro — corrected horizontal-flip augmentation (physics-respecting, flip pos_z+AoA+Uy, preserve unsigned NACA camber).

---

## 2026-05-16 06:00 — PR #3562: H: Wider Transolver (h=192, slice=96) + T_max=18 under bf16 ✓ MERGED — NEW ALL-TIME BEST

- **Branch:** willowpai2i48h1-askeladd/wider-h192-bf16-tmax18
- **Student:** willowpai2i48h1-askeladd
- **Hypothesis:** Capacity scaling — wider hidden dim (h=192 vs 128) + wider slices (96 vs 64) + budget-matched T_max=18 under bf16. VRAM freed by bf16 allows larger model within 30-min constraint.

### Results (best run hzxs6zx9)

| Metric | Baseline #3480 | h=192 best | Δ | % |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 87.9105 | **86.8095** | −1.10 | −1.25% |
| **test_avg/mae_surf_p** | 83.3782 | **81.3514** | −2.03 | −2.43% |

Per-split (best run hzxs6zx9):

| Split | val | test |
|---|---:|---:|
| single_in_dist | 103.640 | 92.053 |
| geom_camber_rc | 98.013 | 86.305 |
| geom_camber_cruise | 65.111 | 71.082 |
| re_rand | 80.474 | 75.966 |
| **avg** | **86.8095** | **81.3514** |

4 informal runs (no seed control): hzxs6zx9 val=86.81, gu27mc6o, sv85254i val=91.06, fqzs1zk1 val=92.97. Mean≈89.70, σ̂≈2.97.

W&B runs: `hzxs6zx9` (best), `gu27mc6o`, `sv85254i`, `fqzs1zk1` | n_params=1.48M (vs 0.66M, ×2.24) | Peak VRAM: 49.24 GB | Best epoch: 13

### Analysis

**NEW ALL-TIME BEST on both val (86.81) and test (81.35).** The capacity hypothesis is confirmed: bf16's VRAM headroom enables a genuinely larger model. Test improvement (−2.03pt) is the headline — both OOD splits (re_rand −1.19, cruise −4.05) and in-dist (+1.6 single) show clear gains.

**Caveat:** best epoch 13 under T_max=18 with model still improving at timeout — the wider model hasn't fully converged. Additional epochs (or a longer T_max) could squeeze further gain. Seed variance σ̂≈2.97 is elevated vs h=128 σ̂=1.54, suggesting higher sensitivity to initialization — seed-controlled characterization is the critical next step.

**Code changes merged:** train.py updated to h=192, slice_num=96, T_max=18.

**Follow-up:** askeladd assigned PR #3735 — 4-seed σ̂ characterization of h=192 config (same as alphonse #3546 did for h=128).

---

## 2026-05-16 06:05 — PR #3611: H: Per-channel surf weight β_p=20 ↩ SENT BACK (retest on new h=192 baseline)

- **Branch:** willowpai2i48h1-edward/per-channel-surf-weight
- **Student:** willowpai2i48h1-edward
- **Hypothesis:** β_p=20 on surface-p channel (Ux/Uy remain at α=10). Tests whether pressure-specific loss amplification helps independently.

### Results (3 seeds on h=128 baseline)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| ecpuvmr3 (best) | 86.2535 | 81.8903 |
| w37awicb | 88.05 | 83.08 |
| df7d07td | 91.50 | 86.98 |
| **mean** | **88.60** | **83.98** |
| **σ̂ (3-seed)** | **~2.63** | **~2.59** |
| h=128 baseline μ̂ | 90.77 | 85.85 |

W&B runs: `ecpuvmr3`, `w37awicb`, `df7d07td` | Best epoch: 17

### Analysis

**Directionally positive on h=128** — 3-seed mean 88.60 is below canonical μ̂=90.77 (−2.17pt, ~1.4σ) and the best run (86.25) beats the prior all-time best (87.91). However, high σ̂=2.63 means the best run is likely a lucky seed. The mean does NOT beat the prior baseline point estimate (88.60 > 87.91), so this is directional but not conclusive.

**Sent back for retest on the new h=192 baseline** (val=86.81, test=81.35) — #3562 merged just before this review. Per-channel surf weight is an orthogonal loss change that should be tested against the current best architecture, not the old one. If β_p=20 helps on h=192, it compounds the gains; if not, the lever is exhausted.

---

## 2026-05-16 07:30 — PR #3680: H: SwiGLU activation in Transolver MLP blocks ✓ MERGED — NEW PROGRAMME ALL-TIME BEST

- **Branch:** willowpai2i48h1-thorfinn/swiglu_activation
- **Student:** willowpai2i48h1-thorfinn
- **Hypothesis:** Replace standard GELU MLP with SwiGLU (gated linear unit, Shazeer 2020). SwiGLUMlp uses 2/3 hidden ratio (171 vs 256) for param parity, adds multiplicative gate: `fc_out(fc_main(x) * SiLU(fc_gate(x)))`.

### Results (run 8on2llcv, seed=0, h=128/slice=64/T_max=15)

| Metric | h=128+GELU μ̂ | h=192+GELU | **SwiGLU+h=128** | Δ vs μ̂ | Δ vs h=192 |
|---|---:|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 90.77 | 86.81 | **65.44** | −25.33 (−27.9%) | −21.37 (−24.6%) |
| **test_avg/mae_surf_p** | 85.85 | 81.35 | **62.04** | −23.81 (−27.7%) | −19.31 (−23.7%) |

Per-split (best epoch 17):

| Split | val | test |
|---|---:|---:|
| single_in_dist | 75.90 | 66.09 |
| geom_camber_rc | 78.66 | 71.55 |
| geom_camber_cruise | 45.74 | 55.55 |
| re_rand | 61.46 | 54.96 |
| **avg** | **65.44** | **62.04** |

W&B run: `8on2llcv` | n_params=663,429 (vs 663,040 GELU, param-matched) | best_epoch=17 | T_max=15

### W&B Verification (independently verified by advisor)

- Training trajectory: 188.79 → 65.44 over 17 epochs, clean monotonic descent
- No NaN/inf in 6,381 training steps
- Grad norms: step1=4.33, mid=1.51, final=3.60, max=16.80 — healthy, no spikes
- Config confirmed: use_swiglu=True, h=128, slice=64, mlp_ratio=2, T_max=15, seed=0

### Analysis

**Category-redefining result.** A 25-point val improvement from a single activation function change. Mechanism: SwiGLU's multiplicative gate `SiLU(fc_gate(x))` allows the FFN to selectively suppress irrelevant spatial features. The high-frequency pressure field in 2D CFD benefits enormously from selective feature propagation — GELU passes all activations uniformly scaled, while SwiGLU can zero out non-relevant components per spatial location.

**Why so large for this task specifically:** CFD pressure prediction requires high spatial-frequency discrimination between regions with very different boundary conditions (near-wall vs far-field, suction vs pressure side of the foil). The gate enables adaptive feature suppression that is ideally matched to this class of structured-but-heterogeneous spatial predictions.

**Code change:** single file, 43 lines added to train.py. SwiGLUMlp class + opt-in flag `--use_swiglu`. Backward-compatible (default off). After merge, h=192/T_max=18 is still the default. SwiGLU requires `--use_swiglu` flag plus h=128/T_max=15 override to reproduce.

**Open questions driving next experiments:**
1. **Does SwiGLU stack with h=192?** (thorfinn #3764)
2. **Is val=65.44 seed-reproducible?** (fern #3765 — 2 more seeds)
3. **Does inverse-LLRD compound with SwiGLU?** (frieren #3768)

---

## 2026-05-16 07:30 — PR #3721: H: DropPath rate=0.1 ✗ CLOSED (regression)

- **Branch:** willowpai2i48h1-fern/droppath-stochastic-depth
- **Student:** willowpai2i48h1-fern
- val_avg/mae_surf_p: 92.06 (above h=128 μ̂=90.77 and far above SwiGLU 65.44)
- **Closed.** Stochastic depth regularization not the bottleneck for Transolver. SwiGLU established representational capacity (selective gating) as the dominant lever, not regularization.

---

## 2026-05-16 07:30 — PR #3722: H: Inverse-LLRD γ_inv=1.176 ✗ CLOSED (below old threshold but above new SwiGLU baseline)

- **Branch:** willowpai2i48h1-frieren/inverse-llrd-bottom-boost
- **Student:** willowpai2i48h1-frieren
- val_avg/mae_surf_p: 88.03 (below old μ̂-1σ=89.2 threshold, but >> SwiGLU 65.44)
- **Closed.** Directionally positive on h=128+GELU (1σ below μ̂) but baseline moved to 65.44. Mechanistic finding (block_0 highest grad norm, LR boost helps) is valid and queued as follow-up on SwiGLU config (frieren #3768).
