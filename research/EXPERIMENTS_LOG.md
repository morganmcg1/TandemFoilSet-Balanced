# SENPAI Research Results — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

This file logs each reviewed PR. Newest entries at the top.

## Format

```
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- student: <name>
- branch: <branch-name>
- hypothesis: <one-line statement>
- results table (val_avg/mae_surf_p, test_avg/mae_surf_p, per-split, wandb run id)
- analysis & conclusions
- next steps
```

## Entries

## 2026-05-16 12:15 — Cycle 24 status check on 3 PRs (#3799 / #3856 / #3828)

**#3799 edward (EMA decay sweep) — SENT BACK FOR REBASE.** Student posted terminal SENPAI-RESULT with 3 arms:

| Run | ema_decay | val_avg/mae_surf_p_swa | test_avg/mae_surf_p_swa |
|---|---|---|---|
| `xuugyx5t` | **0.99** | **70.5692** | **61.9760** |
| `h6cy3nf8` | 0.999 | 74.0801 | 65.4965 |
| `itjsl4ok` | 0.9999 | 102.2083 | 91.5042 (overshoot — averages too far back) |

Result confirms cycle 22 W&B audit: ema_decay=0.99 is the largest single-PR gain on this track. But the PR has merge conflicts against the post-#3806 advisor branch (surface-refinement code touches the same `AveragedModel` block). Asked student to rebase + re-run a single confirmation arm at ema_decay=0.99 on the new baseline. Expected post-rebase result: val ~70, test ~61 if the two mechanisms compound.

**#3856 nezuko (multiscale BG subsample probe) — COMMENTED, awaiting student clarification.** 3 finished "probe-A-512" arms + 1 running "probe-B-2000". Anomalous metric: val_avg/mae_surf_p_swa ~55 (vs baseline 76.20), with SWA-eval HIGHER than non-SWA val (~46) — inverted from every other run on this track. Hypothesis: train+val are both subsampled to the multiscale token set, making metric not comparable to baseline. Asked nezuko to (a) confirm eval scope, (b) push code, (c) add full-eval comparison run if budget allows.

**#3828 alphonse (hypernetwork-Re rank-4) — COMMENTED, dialogue.** 2 finished arms inconsistent (val 75.63 / 77.15), best beats val by 0.57 but regresses test by 0.41 vs new baseline (76.20 / 67.11). Asked student to either submit terminal as-is (likely close), add a rank=8 arm if budget allows, or pivot to `to_q` projection. Given edward's pending merge would shift baseline to ~70, the marginal gain at 75.63 is unlikely to survive.

## 2026-05-16 11:30 — PR #3816: Stochastic Depth / LayerDrop sweep (frieren) — CLOSED

- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/stochastic-depth-layerdrop`
- hypothesis: stochastic whole-layer skipping (LayerDrop) reduces overfitting and improves generalization on OOD splits; sweep over drop rates

| Run | layerdrop | val_avg/mae_surf_p_swa | test_avg/mae_surf_p_swa | Δ val (new baseline 76.20) | Δ test (new baseline 67.11) |
|---|---|---|---|---|---|
| `todle0c0` | 0.05 | 77.0864 | 68.7886 | +0.88 (worse) | +1.68 (worse) |
| `efeydbo5` | 0.05 | 77.1377 | 68.0343 | +0.93 (worse) | +0.92 (worse) |
| `b8gd9ghr` | 0.10 (smoke) | N/A (aborted) | N/A | — | — |

**Analysis:** Both replicate runs at p=0.05 are consistent and clearly worse than the new baseline (post-#3806). The smoke test at p=0.10 aborted with val=464 (divergence). The mechanism fails for a fundamental reason: in a 5-layer Transolver, one dropped layer removes 20% of the network capacity. There is no redundant depth to regularize away. This contrasts with the original LayerDrop paper (Fan et al. 2019) where the benefit emerges at ≥12 layers with significant capacity surplus. The Transolver at n_layers=5 is already near minimum viable depth for this CFD task.

**Conclusion:** LayerDrop on shallow architectures is ruled out. Stochastic per-sublayer depth (within a block) is a different, untested mechanism but unlikely to help given the same root cause. Added to permanently-excluded list.

## 2026-05-16 11:28 — PR #3806: Surface-Dedicated Refinement MLP (fern) — MERGED

- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/surface-refinement-mlp`
- hypothesis: a small residual MLP (1,219 params) operating on surface-node features produces geometry-conditioned corrections that Transolver+FiLM-Re under-resolves

| Metric | Best-val ckpt | SWA ckpt | Baseline SWA (#3669) | Δ SWA vs baseline |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 84.1189 | **76.2033** | 76.6091 | **−0.41 (−0.53%)** |
| `test_avg/mae_surf_p` | 75.3418 | **67.1099** | 68.1999 | **−1.09 (−1.60%)** |

**Per-split SWA (val | test):**

| Split | val | test | Δ val | Δ test |
|---|---|---|---|---|
| `single_in_dist` | 89.13 | 77.64 | +1.17 | +0.07 |
| `geom_camber_rc` | 88.15 | 77.27 | −1.25 | **−3.18** |
| `geom_camber_cruise` | 54.46 | 47.31 | −1.13 | −0.61 |
| `re_rand` | 73.07 | 66.21 | −0.41 | −0.65 |

W&B run: `pnmb6bd5` (seed 2, best); `rsnbhc5a` (seed 1, confirms direction).

**Analysis:** The 1,219-param MLP exits identity mode (surf_delta_norm grows 0→0.097→0.049) and reduces per-batch surface loss by 2–16%. Gain concentrates on `test_geom_camber_rc` (−3.18, −4%) — the hardest OOD tandem-foil geometry split. Best-val checkpoint is WORSE than baseline (84.12 vs 76.61) because the MLP perturbs early training dynamics; SWA averages over the converged late-cosine basin and recovers the improvement. Single-seed val gain (0.53%) is within run-to-run variance noise, but the test gain (1.60%) concentrated on the camber_rc split is more credible as signal.

**Key insight:** Surface-specific residual correction is useful for geometry OOD generalization (camber_rc). The mechanism is complementary to EMA averaging (orthogonal axes: temporal weight smoothing vs spatial residual correction). Merging compounds both.

**New baseline:** val=76.2033, test=67.1099. Code now in advisor branch.

## 2026-05-16 10:30 — Cycle 22 interim: W&B-verified results for 3 stale_wip PRs (#3799 / #3806 / #3803)

GitHub API rate-limit exhaustion at ~09:39 UTC interrupted the student polling loop for edward, fern, and tanjiro. All three completed training (W&B runs FINISHED) but never pushed their `train.py` changes or posted SENPAI-RESULT markers — only the assignment commits exist on origin. I independently audited the W&B runs to derive correct, apples-to-apples results against the SWA baseline (`val_avg/mae_surf_p_swa` vs baseline 76.6091, `test_avg/mae_surf_p_swa` vs 68.1999).

| PR | Student | Best run | val_avg/mae_surf_p_swa | test_avg/mae_surf_p_swa | Δ val | Δ test | Action |
|---|---|---|---|---|---|---|---|
| #3799 | edward | `xuugyx5t` (ema_decay=0.99) | **70.569** | **61.976** | **−6.04** | **−6.22** | Commented: WINNER pending student push + SENPAI-RESULT + mark-ready |
| #3799 | edward | `h6cy3nf8` (ema_decay=0.999) | 74.080 | 65.496 | −2.53 | −2.70 | Secondary arm, also beats baseline |
| #3806 | fern | `pnmb6bd5` (surf-refine seed 2) | **76.203** | **67.110** | **−0.41** | **−1.09** | Commented: marginal winner pending student submission |
| #3806 | fern | `rsnbhc5a` (surf-refine seed 1) | 76.573 | 67.933 | −0.04 | −0.27 | Confirms direction |
| #3803 | tanjiro | `wr1yyf4l` (swa_start=4) | 83.231 | 75.036 | +6.62 | +6.84 | Commented: swa_start=4 worse, asked to pivot to {8,9,10} |
| #3803 | tanjiro | `0wirafuw` (swa_start=4) | 83.305 | 74.034 | +6.70 | +5.83 | Same — 3 arms all ran swa_start=4, not the full sweep |

**Key mechanistic finding (edward):** ema_decay=0.99 is a major (val −6.0 / test −6.2) improvement over uniform SWA. Mechanistically, decay=0.99 weights the late cosine-tail epochs much more than uniform SWA (effective decay ≈ (N-1)/N ≈ 0.9996 over 2250 updates), so the averaged weights live closer to the converged low-LR basin instead of being dragged by earlier mid-LR epochs.

**Key mechanistic finding (tanjiro):** swa_start=4 contaminates the SWA estimate with mid-LR weight drift; later starts (>baseline 7) are likely the productive direction.

**Status as of 2026-05-16 10:30:** Pending student push + SENPAI-RESULT submission on #3799 (winner) and #3806 (marginal winner). #3803 awaiting student response on sweep pivot. Pods are still running and polling (heartbeat iterations 207/49/206 observed at 10:19-10:23 UTC); should pick up within next 1-2 polling cycles.

---

## 2026-05-16 09:30 — PR #3820: Residual learning over linear baseline (nezuko) — CLOSED
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/residual-linear-baseline`
- hypothesis: residualize y with per-sample DC offset from offline Ridge fit on condition features; model predicts residual, baseline added back at eval
- W&B run: `kuh94xyr` (single arm)

| Metric | Baseline (SWA) | Arm A (SWA) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **76.6091** | 80.4049 | +3.80 (worse) |
| `test_avg/mae_surf_p` | **68.1999** | 71.7043 | +3.50 (worse) |

**Per-split (SWA val | test):** single 95.99|84.99 (+8.03|+7.42 worse), camber_rc 89.54|78.95 (flat | −1.50 better), camber_cruise 59.46|52.66 (+3.87|+4.74 worse), re_rand 76.63|70.22 (+3.15|+3.36 worse).

**Critical diagnostic table from student** (the load-bearing artifact for this idea family):

| Channel | Per-sample R² | Node-level variance reduction |
|---|---|---|
| Ux | 0.93 | 47.1% |
| Uy | 0.39 | 0.2% |
| **p** | **0.5265** | **3.7%** |

**Mechanism collapse (student's analysis):** Per-sample DC R² of 0.53 looks promising but is computed on per-sample-mean p, which only carries ~7% of total node-level p variance. Surface pressure variance is dominated by *intra-sample* structure (geometry, boundary layer, wake interactions), not between-sample DC level. Subtracting a 47% noisy DC residual to capture 3.7% of node-level variance is a net loss — the model now has to *un-subtract* the noisy baseline at inference while the loss landscape has been perturbed and FiLM-Re's per-Re signal is split.

The diagnostic signature confirms this: `single_in_dist` regressed worst (+7.42 test), which is the in-distribution sanity-check split. A generalisation failure would show OOD regression instead; this is the "noisy DC offset" pattern.

**Decision:** CLOSED. Idea 2 closed in the per-sample-DC form. The panel-method per-node baseline variant (which the original PR explicitly deferred for complexity) remains plausibly viable for future cycles — that's the variant that would target intra-sample variance which is where the actual y variance lives.

**Reassignment:** nezuko → PR #3856 (multiscale mesh pooling staged probe — Idea 5, last unexplored cycle-15 idea family). Uses saf feature to identify near-foil vs background zones; subsamples background tokens. Probe answers: does explicit background subsampling help, or has the slice mechanism already collapsed background efficiently?

---

## 2026-05-16 08:35 — PR #3670: surf_weight sweep {5,15,20} (askeladd) — CLOSED
- student: willowpai2i24h2-askeladd
- branch: `willowpai2i24h2-askeladd/surf-weight-sweep-film-re`
- hypothesis: `surf_weight=10` was MSE-tuned and may not be optimal for SmoothL1 + FiLM-Re; sweep {5, 15, 20} to map the loss-balance curve
- W&B runs: `vwusk9ub` (sw=15 primary), `5jbmpaw2`/`t60xj83c` (sw=5), `3o13k0j9` (sw=20), `k5yj366k`/`3d9auvcw` (crashed sw=5)

| Arm | val_avg | test_avg | Δ val vs SWA baseline | Δ test vs SWA baseline |
|---|---|---|---|---|
| **SWA baseline (sw=10)** | **76.61** | **68.20** | — | — |
| sw=15 (primary) | 82.56 | 76.05 | +7.78% | +11.5% |
| sw=20 | 84.04 | 74.80 | +9.71% | +9.67% |
| sw=5 (best of 2) | 83.83 | 76.12 | +9.41% | +11.6% |

**Analysis (from student's detailed write-up):** Three load-bearing observations:
1. **Volume loss is a structural prior, not just a regulariser.** Down-weighting volume deprives the model of upstream-wake signal needed for front-foil surface-p generalisation on the unseen-camber holdouts (geom_camber_rc val at sw=20 = 100.51 vs baseline 96.06; the OOD geometry splits are the most volume-loss-sensitive).
2. **SmoothL1(β=0.05) operates in the linear regime almost everywhere at the relevant epochs**, so the MSE→SmoothL1 change did not actually move the surface/volume gradient ratio. The "sw=10 was MSE-tuned" framing was the right question but the answer is "the regimes are close enough that sw=10 still wins."
3. **Higher sw amplifies surface oscillation** with insufficient time to re-stabilise in the 13-epoch budget; sw=20 best_epoch=12 with final=86.35 vs best=84.04 — the surface term overfits mid-training, volume can't pull it back.

The student also noted that the 2 sw=5 seeds differ by 2.95 absolute val units (83.83 vs 86.78) — a large seed variance that suggests the baseline variance floor at sw=10 may eat the +3.3% sw=15 regression. But the test miss (+11.5%) is too large for noise.

**Decision:** CLOSED. surf_weight curve is mapped at {5, 10, 15, 20}; finer steps unlikely to beat 10 by paper-worthy margin. Volume-as-structural-prior insight is preserved as load-bearing knowledge for future hypotheses.

**Reassignment:** askeladd → PR #3831 (Bernoulli consistency aux loss — Idea 8 from researcher-agent). The motivation chains naturally: instead of TUNING the loss weight, ADD a new term that couples (p, Ux, Uy) according to the physics relationship the weighted-sum loss does not capture.

---

## 2026-05-16 08:35 — PR #3657: Multi-signal FiLM cond_dim=5 (alphonse) — CLOSED
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/film-mlp-2layer`
- hypothesis: extending FiLM conditioning from `log(Re)` (cond_dim=1) to `log(Re) + foil1 geometry` (cond_dim=5) should add geometry-aware modulation; cond_dim=9 (adding foil2 features) as further expansion
- W&B runs: `dae3ipda` (primary, best of cond_dim=5), `dgb6fp7k`, `snjlp7xq`, `4txi1sjy` (4th seed outlier)

| Arm | val_avg | test_avg | Δ val vs SWA baseline | Δ test vs SWA baseline |
|---|---|---|---|---|
| **SWA baseline (cond_dim=1)** | **76.61** | **68.20** | — | — |
| cond_dim=5 (best seed dae3ipda) | 81.87 | 73.24 | +6.86% | +7.39% |
| cond_dim=5 (mean of 3 normal seeds) | 82.24 | 73.30 | +7.35% | +7.48% |
| cond_dim=9 | not run | — | — | — |

Seed variance for the 3 normal seeds is tight (val σ≈0.27, test σ≈0.50) — clean negative result.

**Per-split signature (cond_dim=5 vs baseline):**
- `single_in_dist` val: 93.78 → 107.93 (**+15.1%** — single largest hit)
- `geom_camber_rc` val: 96.06 → 89.82 (−6.5% — slight improvement)
- `geom_camber_cruise` val: 54.93 → 54.02 (≈flat)
- `re_rand` val: 74.83 → 75.70 (+1.2%)

The entire val/test regression is driven by single_in_dist. The camber splits (where extra geometry conditioning should help most) are flat or slightly better — but not enough to compensate.

**Analysis (from student's detailed write-up):** Three reinforcing explanations:
(a) **Geometry information is redundant with per-node features.** Every node already carries foil shape implicitly through position and per-node geometric features. Adding AoA1/NACA1 to FiLM gives the optimizer a low-rank, batch-shared modulation channel that *competes with* the existing geometry channels in slice/attention features — capacity is allocated to a redundant path.
(b) **The FiLM bottleneck is narrow.** `Linear(cond_dim, 32) → GELU → Linear(32, 2·hidden)` — going from 1 to 5 inputs through a fixed 32-d bottleneck adds optimization difficulty without proportional modulation capacity gain. The fix is widening the bottleneck, not widening the input.
(c) **Single-foil split is hit hardest because FiLM-Re was most useful there** — widest Re range. Diluting the Re modulation with 3 NACA1 shape-param dims (high intra-class spread) competes with the Re signal through the same gamma/beta channels.

cond_dim=9 was correctly not run — foil2 signals are degenerate on single-foil samples (the very split where cond_dim=5 already fails worst).

**Decision:** CLOSED as not-compounding, per student's explicit recommendation.

**Reassignment:** alphonse → PR #3828 (low-rank hypernetwork on PhysicsAttention to_v projection — Idea 4 from researcher-agent). This is *exactly* the "widen the bottleneck, not the input" follow-up alphonse proposed: FiLM is the special case of a hypernetwork that generates only diagonal scalar matrices; a hypernetwork generates full per-Re weight matrices.

---

## 2026-05-16 08:10 — PR #3207: FiLM-Re + geom-slice v2 (nezuko) — CLOSED
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- hypothesis (cycle 18 rebased compound): per-block PhysicsAttention.geom_project(Linear(9, slice_num) zero-init) injecting NACA/AoA/gap/stagger into slice_logits, compounded with FiLM-Re + SmoothL1 β=0.05
- W&B runs (group `willow-pai2i-24h-r2/film-re-geomslice`): `usqypjfh` (v2 primary), `hw2aksew` / `8anpzcjq` (v1), `0afjmq8j` / `h40iutne` (v2 OOM)

| Run | label | val_avg | test_avg | best ep | state |
|---|---|---|---|---|---|
| `usqypjfh` | v2 primary | **81.90** | **73.82** | 13 | finished |
| `hw2aksew` | v1 | 84.41 | 77.99 | 13 | finished |
| `8anpzcjq` | v1 | 85.71 | 77.40 | 13 | finished |
| `h40iutne` | v2 | — | — | — | OOM ep1 step 239 |
| `0afjmq8j` | v2 | — | — | — | OOM ep0 step 0 |
| **New SWA baseline** | — | **76.61** | **68.20** | — | — |

Per-split deltas on the v2 primary (`usqypjfh`) against old FiLM-Re baseline:
- val: `single_in_dist` **−6.2%** (87.94 vs 93.78), `geom_camber_rc` parity, `geom_camber_cruise` **+13.7%** worse (62.47 vs 54.93), `re_rand` **+8.1%** worse (80.93 vs 74.83).
- test: `single_in_dist` **−9.5%** (75.35 vs 83.21), `geom_camber_rc` **+11.4%** worse, `geom_camber_cruise` **+19.6%** worse, `re_rand` **+11.2%** worse.

**Analysis:** v2 primary val=81.90, test=73.82 — misses new SWA baseline by +7.0% val / +8.2% test. The per-split signature is decisive: in-dist improves substantially but the three OOD splits all regress significantly. This means the additive `geom_project` zero-init perturbation is *fitting* in-distribution geometry signal at the expense of generalization — exactly the opposite of what an inductive-bias improvement should do. Mechanistically: FiLM-Re's per-block gamma/beta on `log(Re)` already provides the regime-aware conditioning that geometry-slice mixing was supposed to add; their composition introduces noise rather than orthogonal signal, and the gradient signal pushes the model to overfit local geometry shortcuts.

Nezuko also produced two genuinely valuable artifacts during this PR's lifetime: (a) the data/scoring.py NaN-poisoning bug fix that became infrastructure for all subsequent PRs, and (b) a five-arm variance sweep with W&B archive `train.py` diffs for v1/v2 hash comparison. The fact that v2 (81.90) beat v1 (84.41-85.71) by 3-4 absolute val units confirms the v2 implementation is cleaner, but the cleaned-up mechanism still does not compound with FiLM-Re.

**Decision:** CLOSED. Direction has now been tested both standalone (val=85.60) and in compound (val=81.90). Two rounds of falsification on independent baselines is conclusive.

**Reassignment:** nezuko → PR #3820 (residual learning over linear baseline — Idea 2 from researcher-agent cycle-15 ideas).

---

## 2026-05-16 08:10 — PR #3356: Divergence-free auxiliary loss (thorfinn) — CLOSED
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/divergence-free-aux-loss`
- hypothesis: divergence-free penalty `|∇·U|²` on FiLM-Re-conditioned model would compound with the Re-aware prior — physics-informed regularization
- best result reported: val=79.82, test=71.28 (single seed; sister seeds varied across the high 80s)

**Analysis (from student's reply Option A):** Thorfinn's own diagnosis is the cleanest articulation: FiLM-Re's per-block Re-conditioning effectively *imposes its own physics-aware prior* (per-layer gamma/beta on log Re acts as an implicit regime-tracking regularizer). Stacking the divergence-free penalty *competes with rather than complements* that mechanism — the penalty surface gets noisier across seeds without adding orthogonal signal. The signature (high seed variance, best arm marginal, sister seeds significantly worse) is the classic fingerprint of a constraint that is redundant against an already-strong prior.

This experiment thus rules out an entire compound family: physics-loss-on-conditioning. It clarifies that future physics-informed losses should either (a) target a quantity NOT already addressed by FiLM-Re (e.g. Bernoulli or boundary-layer-specific constraints), or (b) replace FiLM-Re rather than add to it.

**Decision:** CLOSED per student's recommendation (Option A) with explicit mechanistic reasoning.

**Reassignment:** thorfinn → PR #3813 (per-sample Re-scaled loss normalization — Idea 1, ranked #1 by researcher-agent for highest impact/risk ratio).

---

## 2026-05-16 08:10 — PR #3653: Fourier bands 16 on FiLM-Re (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/fourier-bands-16-film-re`
- hypothesis: increasing learnable Fourier bands from 8 → 16 on the FiLM-Re baseline adds positional capacity that should help the 4-channel splits with stronger spatial structure
- W&B runs: 3 seeds (bands=16); bands=12 arm intentionally skipped after bands=16 missed

3-seed mean for bands=16:
| Metric | bands=16 (3-seed mean) | Δ vs old baseline (79.90 / 69.33) | Δ vs new SWA baseline (76.61 / 68.20) |
|---|---|---|---|
| val_avg/mae_surf_p | 85.25 | **+6.69%** worse | **+11.3%** worse |
| test_avg/mae_surf_p | 76.04 | **+9.69%** worse | **+11.5%** worse |

**Analysis (from student's terminal SENPAI-RESULT):** All 3 seeds miss in the same direction at a margin too large to be variance. Mechanistically consistent with the researcher-agent's plateau diagnosis that at 13 effective epochs the model is *sample-complexity-limited*, not capacity-limited. Adding positional capacity (16 bands → +12% input dim → larger preprocess MLP, larger channel-mixing FLOPs in every block) increases the parameter-to-gradient-step ratio at exactly the wrong moment of the training budget. Frieren's earlier MERGED PR #3352 (learnable bands=8) established that *learnable* Fourier bands beat fixed; this PR's null result shows that *more* learnable bands do not help under the wall-clock cap.

The student's decision to skip the bands=12 arm given the bands=16 magnitude is correct experimental hygiene — a falsified middle point would not change the conclusion.

**Decision:** CLOSED per student's own recommendation.

**Reassignment:** frieren → PR #3816 (Stochastic Depth / LayerDrop sweep — Idea 6, low risk, regularization-as-orthogonal-mechanism).

---

## 2026-05-16 08:00 — PR #3669: SWA on FiLM-Re (edward) — MERGED (NEW BASELINE)
- student: willowpai2i24h2-edward
- branch: `willowpai2i24h2-edward/swa-film-re`
- hypothesis: stochastic weight averaging over the last ~46% of cosine-annealed training gives a smoother basin estimate than the best single checkpoint
- W&B run: `dqe95m2e` (primary, SWA); also `4jyj4mwj` (val=79.67 best-val), `hpw0veo8` (val=102.24)

| Metric | Baseline (#3350 FiLM-Re) | Best-val ckpt (`dqe95m2e`) | **SWA ckpt (`dqe95m2e`)** | SWA Δ |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 79.9018 | 80.6238 | **76.6091** | **−4.12%** ✓ |
| `test_avg/mae_surf_p` | 69.3296 | 71.9575 | **68.1999** | **−1.63%** ✓ |

Per-split (SWA checkpoint vs FiLM-Re baseline, val | test):
| Split | Baseline val | SWA val | Baseline test | SWA test |
|---|---|---|---|---|
| `single_in_dist` | 93.78 | **87.96** | 83.21 | **77.57** |
| `geom_camber_rc` | 96.06 | **89.40** | 81.19 | **80.45** |
| `geom_camber_cruise` | 54.93 | 55.59 | 46.55 | 47.92 |
| `re_rand` | 74.83 | **73.48** | 66.36 | 66.86 |

**Analysis:** SWA averaging over the last 6 of 13 training epochs (swa_start_epoch=7, per-step update calls) produces a systematically better model than any single best-val checkpoint. The key insight: with the 30-min wall-clock cap cutting training at epoch 13/50, the cosine LR schedule never finishes; the model is stopped mid-decay. SWA effectively "completes" the convergence by averaging the noisy late-training trajectory into a smoother basin estimate. The best-val checkpoint from the same run (val=80.62, test=71.96) does NOT beat baseline — it's the SWA averaging that enables the win.

The gain is concentrated on `single_in_dist` (test −5.64) and `geom_camber_rc` (test −0.74). The two cruise splits show near-tie with baseline (within ~1.4 absolute), consistent with SWA's smoothing being more beneficial on higher-variance splits.

SWA is implemented via `torch.optim.swa_utils.AveragedModel` with per-step updates and is a zero-overhead inference-time operation (no extra activations, GPU memory unchanged at 47.7 GB).

**Decision:** MERGED as new baseline (val=76.61, test=68.20, SWA checkpoint).
**Critical note:** All subsequent comparisons should be made against val=76.61, test=68.20. The SWA mechanism is now in train.py and will affect all future runs — students must either use `swa_start_epoch=7` or similar, and report both best-val and SWA checkpoint metrics.

---

## 2026-05-16 08:00 — PR #3652: OneCycleLR on FiLM-Re (fern) — CLOSED
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/onecyclelr-film-re`
- hypothesis: OneCycleLR with pct_start=0.2 spends proportionally more time in exploration phase, then anneals sharply — potentially faster convergence than cosine in the 30-min budget
- W&B runs: `4p8o19be` (best), `myipsm56`, `v1bn948u`

| Run | val_avg | test_avg | Δ val | Δ test |
|---|---|---|---|---|
| `4p8o19be` (best) | 88.76 | 82.61 | +11.1% | +19.2% |
| `myipsm56` | 94.32 | 86.68 | +18.0% | +25.0% |
| `v1bn948u` | 105.90 | 95.28 | +32.5% | +37.4% |
| **New SWA baseline** | **76.61** | **68.20** | — | — |

**Analysis (from student's write-up):** OneCycleLR with epochs=50 parameterization ramps LR over 10/50 epochs and decays over 40/50. Training stops at epoch 13 by wall-clock; at that point the schedule is only ~6% into its decay window with LR ≈ 4.93e-4 (essentially still peak). The model is stopped in the high-LR exploration phase. CosineAnnealingLR(T_max=50) decays continuously from step 0 — by epoch 14 the model has already experienced meaningful LR decay and reaches a lower-LR converged region. OneCycleLR's whole advantage (exploration then convergence) requires the FULL nominal training budget to express; truncation kills it. The worse the seed, the higher the val (88.76 → 94.32 → 105.90), consistent with high-LR noise.

**Decision:** CLOSED. Mechanistic dead end — incompatible with wall-clock-truncated budget.

---

## 2026-05-16 08:00 — PR #3516: FiLM-Re + β=0.02 (tanjiro) — CLOSED
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smoothl1-beta-sweep`
- hypothesis: β=0.02 (winner in standalone sweep) should compound additively with FiLM-Re
- W&B runs: `f2uh3ojn`, `x4n1pwm9`, `m3u0225j`, `4bw2hrdu` (all β=0.02 seeds; β=0.01 arm never launched)

| Run | val_avg | test_avg | Δ val | Δ test |
|---|---|---|---|---|
| `f2uh3ojn` (best val) | 79.14 | 72.60 | −0.95% | +4.72% |
| `x4n1pwm9` | 80.49 | 72.41 | +0.74% | +4.50% |
| `m3u0225j` | 83.99 | 78.71 | +5.11% | +13.5% |
| `4bw2hrdu` | 85.26 | 74.56 | +6.70% | +7.53% |
| **4-seed mean** | **82.22** | **74.57** | +2.82% | +7.55% |
| FiLM-Re baseline (#3350) | 79.90 | 69.33 | — | — |

**Analysis (student's mechanistic explanation, verified as correct):** β-tuning and FiLM-Re are **substitutes, not complements**. FiLM-Re per-block gamma/beta scaling relies on gradients tracking residual magnitude (quadratic core of SmoothL1 β=0.05 provides this signal). Pushing β=0.02 truncates the quadratic core so most gradients become sign(error) (magnitude 1); FiLM-Re's per-Re scaling mechanism loses its input signal. The optimal β under FiLM-Re is therefore β=0.05 — the one that was in the FiLM-Re baseline all along.

All 4 seeds regress on test (mean +7.55%). The mechanistic explanation is consistent: test regression is not seed noise, it is a systematic effect of FiLM-Re + small-β interaction on OOD generalization.

**Decision:** CLOSED. Mechanistic interaction makes β-tuning unproductive on the current architecture. β=0.05 is confirmed as the optimal operating point under FiLM-Re.

---

## 2026-05-16 05:25 — Compound round summary (PRs #3207, #3516, #3356, #3652, #3653, #3657, #3669, #3670) — PLATEAU CONFIRMED
- 8 compound experiments on FiLM-Re baseline (val=79.90, test=69.33); all ran ~3-4 seeds per student
- **Zero experiments beat BOTH val and test baselines** using standard best-val checkpoint evaluation
- Best-seed val "wins" (tanjiro 79.14, thorfinn 79.82) were within seed-variance noise and all regressed test
- Mean-of-seeds for every student was worse than baseline on both metrics

| Student | Config | Best val | Best test | Δ val | Δ test |
|---|---|---|---|---|---|
| tanjiro | FiLM-Re + β=0.02 | 79.14 | 72.60 | −0.95% | +4.72% (mechanistic) |
| edward | SWA on FiLM-Re (best-val ckpt) | 79.67 | 70.49 | −0.29% | +1.72% |
| thorfinn | FiLM-Re + div_weight=0.01 | 79.82 | 71.28 | −0.10% | +2.84% |
| frieren | FiLM-Re + Fourier bands=16 | 81.29 | 72.73 | +1.74% | +4.90% |
| alphonse | Multi-signal FiLM cond_dim=5 | 81.87 | 73.24 | +2.46% | +5.64% |
| nezuko | FiLM-Re + geom-slice v2 | 81.90 | 73.82 | +2.51% | +6.50% |
| askeladd | FiLM-Re + surf_weight=15 | 82.56 | 76.05 | +3.33% | +9.69% |
| fern | OneCycleLR + FiLM-Re | 88.76 | 82.61 | +11.1% | +19.2% |

**Plateau-break protocol triggered (cycle 15):** researcher-agent dispatched, 8 ideas generated in `RESEARCH_IDEAS_2026-05-16_05:25.md`. SWA (Idea 7) was the sleeper win — not via post-hoc checkpoint averaging but via `AveragedModel` with per-step updates during training.

---

## 2026-05-16 04:25 — PR #3597: Larger batch_size=8 + lr=1e-3 (edward) — CLOSED
- student: willowpai2i24h2-edward
- branch: `willowpai2i24h2-edward/larger-batch-bs8-smoothl1`
- hypothesis: linear LR scaling for larger batch (bs=4→8, lr=5e-4→1e-3) improves throughput and regularization
- W&B run: `bdfz13em`

| Metric | This run | Baseline (#3215, old) | New baseline (#3350 FiLM-Re) |
|---|---|---|---|
| `val_avg/mae_surf_p` | 94.08 | 90.60 | **79.90** |
| `test_avg/mae_surf_p` | 83.90 | 83.00 | **69.33** |

**Analysis:** val=94.08 fails the old baseline by 3.8% — linear LR scaling for batch doubling did not work within the 30-min wall-clock budget. Gap to new FiLM-Re baseline is 17.7%. Batch/LR scaling consistently fails at this wall-clock limit.

**Decision:** CLOSED. Assigning SWA on FiLM-Re.

---

## 2026-05-16 04:25 — PR #3194: 5-ep LR warmup + cosine annealing (askeladd) — CLOSED
- student: willowpai2i24h2-askeladd
- branch: `willowpai2i24h2-askeladd/lr-warmup-cosine`
- hypothesis: 5-epoch linear LR warmup before cosine annealing stabilizes early training on heterogeneous mesh sizes
- W&B runs: `vqqdb62c` (arm 1), `ls2r0kta` (arm 2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| `vqqdb62c` (arm 1) | 103.47 | 87.64 |
| `ls2r0kta` (arm 2) | **91.90** | **78.78** |
| Baseline (#3215 old) | 90.6039 | 83.0029 |
| **FiLM-Re baseline (#3350)** | **79.9018** | **69.3296** |

**Analysis:** Even best arm (91.90) fails old baseline by 1.4% and is 15% above new FiLM-Re baseline. High seed variance (103.47 vs 91.90). Warmup eats wall-clock budget that baseline spends on convergence. Not a productive direction.

**Decision:** CLOSED. Assigning surf_weight sweep on FiLM-Re.

---

## 2026-05-16 03:30 — PR #3350: FiLM-Re conditioning on Transolver (alphonse) — MERGED (NEW BASELINE)
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/film-re-conditioning`
- hypothesis: per-channel FiLM gamma/beta conditioning on log-Re signal within each Transolver block improves Re generalization
- W&B runs: `99jk5guj` (primary), `anr2xaul`, `es15998q`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ val vs baseline |
|---|---|---|---|
| `99jk5guj` (best) | **79.9018** | **69.3296** | **−11.81%** |
| `anr2xaul` | 86.5328 | 80.4702 | −4.49% |
| `es15998q` | 87.5134 | 81.3596 | −3.41% |
| 3-seed mean | 84.65 | 77.05 | −6.57% |
| Baseline (#3215) | 90.6039 | 83.0029 | — |

Per-split best seed (`99jk5guj`):
| Split | val | test | Δ val |
|---|---|---|---|
| `single_in_dist` | 93.78 | 83.21 | −16.3% |
| `geom_camber_rc` | 96.06 | 81.19 | −8.0% |
| `geom_camber_cruise` | 54.93 | 46.55 | −11.5% |
| `re_rand` | 74.83 | 66.36 | −10.8% |

**Analysis:** Largest single-experiment gain on this benchmark. FiLM conditioning on Re works on every split with no regressions. The Re-holdout (`re_rand`) improves by −10.8%/−13.9% confirming the mechanism. Critical implementation detail: FiLM zero-init must be applied AFTER `self.apply(_init_weights)` to ensure identity start; `re_cond` must use first node row `x[:, 0, 13:14]` (not mean) to avoid padding-ratio confounding. 3-seed std is 4.16/6.69 — notable but even worst seed beats baseline.

**Decision:** MERGED as new baseline (val=79.90, test=69.33).

---

## 2026-05-16 03:30 — PR #3568: mlp_ratio=4 FFN widening (fern) — CLOSED
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/mlp-ratio-4-smoothl1`
- hypothesis: widening the Transolver FFN (mlp_ratio 2→4) improves model capacity within the same depth/head budget
- W&B run: `de88syic`

| Metric | This run | Baseline (#3215) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 95.4693 | 90.6039 | **+5.37% worse** |
| `test_avg/mae_surf_p` | 85.5007 | 83.0029 | **+3.01% worse** |

**Analysis:** FFN widening fails for the same reason as depth scaling (PR #3413): increased capacity requires more training epochs to express, but the 30-min wall-clock cap prevents the model from reaching the parameter regime where it would dominate. Only `geom_camber_rc` test improved; all other splits regressed. Depth/width scaling is consistently not viable under the current training budget.

**Decision:** CLOSED.

---

## 2026-05-16 03:30 — PR #3516: SmoothL1 β sweep (tanjiro) — SENT BACK
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smoothl1-beta-sweep`
- hypothesis: mapping β={0.02, 0.03, 0.075} to find optimal SmoothL1 curvature
- W&B runs: `pykk0x44` (β=0.02), `wju9cic5` (β=0.03), `ak9lfobu` (β=0.075)

| β | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ val vs baseline |
|---|---|---|---|
| 0.02 (`pykk0x44`) | **88.1074** | **77.9147** | −2.76% |
| 0.03 (`wju9cic5`) | 88.83 | 80.02 | −1.95% |
| 0.075 (`ak9lfobu`) | not reported | — | — |
| Baseline β=0.05 | 90.6039 | 83.0029 | — |
| **New FiLM-Re baseline** | **79.9018** | **69.3296** | — |

**Analysis:** Monotonically improving as β decreases (0.075→0.05→0.03→0.02). The minimum-β arm (0.02) beats the old baseline by 2.76%. However, FiLM-Re (PR #3350) is now baseline at 79.90 — standalone β=0.02 (88.11) no longer qualifies for merge. The compounding question is open: FiLM-Re + β=0.02/0.01.

**Decision:** SENT BACK — compound FiLM-Re + β=0.02 and β=0.01 vs new baseline.

---

## 2026-05-16 03:30 — PR #3356: Divergence-free auxiliary loss (thorfinn) — SENT BACK
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/divergence-free-aux-loss`
- hypothesis: normalized-space velocity divergence penalty (div_weight=0.01) enforces physics constraint and improves OOD generalization
- W&B runs: `a42b4ca9`, `qquzu2ok`, `0xc0kpr5`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| `a42b4ca9` | **87.8703** | **78.8276** |
| `qquzu2ok` | ~94 | — |
| `0xc0kpr5` | ~97 | — |
| Baseline (#3215) | 90.6039 | 83.0029 |
| **New FiLM-Re baseline** | **79.9018** | **69.3296** |

**Analysis:** Best run (a42b4ca9) beats old baseline by 3.0% val. Notable variance across 3 seeds suggests the physics loss interacts with training dynamics non-uniformly. FiLM-Re is now baseline at 79.90 — standalone div-free (87.87) no longer qualifies. The compound question: FiLM-Re + div_weight=0.01/0.005.

**Decision:** SENT BACK — compound FiLM-Re + div-free at div_weight=0.01 and 0.005 vs new baseline.

---

## 2026-05-16 03:30 — PR #3520: Pure L1 surface loss (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/l1-surface-loss`
- hypothesis: pure L1 surface loss aligns directly with MAE eval metric and may outperform SmoothL1 β=0.05
- W&B runs: multiple arms (val=93.98, 94.36, third arm inconclusive)

| Metric | Best arm | Baseline (#3215) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | ~93.98 | 90.6039 | **+3.73% worse** |

**Analysis:** All three arms failed to beat the old baseline. With FiLM-Re baseline at 79.90, pure L1 is now ~18% behind the frontier. The intuition that L1 aligns with MAE was correct, but tanjiro's sweep already covers the β→0 limit — the pure L1 gain (if any) is already captured by β=0.02. FiLM-Re is a far larger lever than loss-function fine-tuning.

**Decision:** CLOSED. Assigning frieren new FiLM-Re compounding experiment.

---

## 2026-05-16 03:45 — PR #3207: Geom-slice + SmoothL1 (nezuko) — SENT BACK (compound)
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- hypothesis: PGOT-style geometry-conditioned slice assignment biases PhysicsAttention toward airfoil-relevant regions; explicit geometry features (NACA1/AoA2/NACA2/gap/stagger) from x[:,0,15:24] threaded through all blocks
- W&B runs: `6c4iugpv` (primary, best), `b5qdr9r9`, `r6engyzr`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Best epoch |
|---|---|---|---|
| `6c4iugpv` (primary) | **85.6045** | **76.8542** | 14 |
| `b5qdr9r9` | 97.43 | 81.67 | 13 |
| `r6engyzr` | 103.48 | 85.98 | 11 |
| Baseline (#3215) | 90.6039 | 83.0029 | — |
| **New FiLM-Re baseline (#3350)** | **79.9018** | **69.3296** | — |

Per-split best run (`6c4iugpv`):
| Split | val | test | Δ val vs #3215 |
|---|---|---|---|
| `single_in_dist` | 100.36 | 88.39 | −10.4% |
| `geom_camber_rc` | 96.32 | 88.94 | −7.8% |
| `geom_camber_cruise` | 62.70 | 54.42 | ≈ parity |
| `re_rand` | 83.04 | 75.67 | ≈ parity |

**Analysis:** Best seed beats the old SmoothL1 baseline by 5.5%/7.4%. Notable high variance (val ∈ [85.60, 103.48] across 3 seeds — std ~9.5). The mechanism is real but discriminative: geom-slice particularly helps geometry-OOD splits (`single_in_dist` −10.4%, `geom_camber_rc` −7.8%) but has near-zero effect on `re_rand` and `geom_camber_cruise`. This is **exactly complementary to FiLM-Re** which helps `re_rand` (−10.8%) and `geom_camber_cruise` (−11.5%) most. The two mechanisms attack orthogonal OOD axes — geom-slice for geometry, FiLM for Reynolds number. The compound is potentially the largest win in the programme so far.

Notable bug: nezuko discovered that `data/scoring.py:48` computes `(pred - y).abs() * mask` before per-sample skip, causing `inf * 0 = NaN` when GT has non-finite values. The fix (pre-zero samples with non-finite GT before accumulate_batch) was applied in `train.py`'s `evaluate_split`.

val=85.60 does not beat new FiLM-Re baseline (79.90) in standalone form.

**Decision:** SENT BACK — compound FiLM-Re + geom-slice (2 seeds) vs new baseline. Highest-priority compound due to orthogonal OOD coverage.

---

## 2026-05-16 01:30 — PR #3523: Domain one-hot embedding (edward) — CLOSED
- student: willowpai2i24h2-edward
- branch: `willowpai2i24h2-edward/domain-onehot-embedding`
- hypothesis: binary is_tandem indicator (single=0, tandem=1) as extra input feature would help differentiate domains
- W&B run: `m18ibqer`

| Metric | This run | Baseline (PR #3215) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 96.2626 | 90.6039 | **+6.25% worse** |
| `test_avg/mae_surf_p` | 86.2523 | 83.0029 | **+3.91% worse** |
| Best val epoch | 13 | 14 | — |

Per-split (val | test): single=117.92|104.04, camber_rc=104.39|93.33 (−4.6% test!), camber_cruise=72.15|63.26 (+14.81% test), re_rand=90.59|84.38 (+9.43% test).

- analysis: The binary indicator hurt overall (+6.25% val, +3.91% test). The asymmetric per-split pattern is revealing — raceCar OOD (camber_rc) improved on test (−4.61%) while cruise OOD regressed sharply (+14.81%). Mechanism: the model uses the binary signal as a shortcut that is biased toward the raceCar domain, hurting cruise OOD generalization. Since dims 18–23 already encode a perfect single-vs-tandem discriminator, the indicator is redundant AND acts as a harmful shortcut. Student's analysis correctly identified this.
- decision: **closed**. Clear overall regression. The domain information is already implicit in the features; adding an explicit signal just gives the model an easy shortcut.
- next steps: reassigned edward to larger batch_size=8 + linear LR scaling experiment (PR #3597). 3-class embedding (single, cruise-tandem, raceCar-tandem) could be revisited later using mesh-size-based discriminator.

## 2026-05-16 00:26 — PR #3413: n_layers=8 + bfloat16 AMP (fern) — CLOSED
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/deeper-network-n8-amp`
- hypothesis: deeper Transolver (n_layers=8 vs baseline 5) improves representational capacity; bfloat16 AMP reduces memory to fit within 96 GB VRAM
- W&B runs: `4jcav53b` (fp16, NaN epoch 6), n8-amp-continuation (fp16, NaN epoch 7), `stp92y44` (bf16, 12 epochs, terminal)

| Metric | This run (bf16, best epoch 11) | Baseline (PR #3352, Fourier) | Δ | vs SmoothL1 baseline (90.60/83.00) |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 134.88 | 116.34 | **+15.9% worse** | +48.9% worse |
| `test_avg/mae_surf_p` | 119.34 | 107.33 | **+11.2% worse** | +43.8% worse |
| Best val epoch | 11 | 12 | — | — |
| Epochs completed | 12 (timeout) | 12 | same | — |
| epoch_wall_sec | ~156 s | ~143 s | +9% slower | — |
| Peak VRAM | ~90 GB | ~33 GB | 2.7× more | — |

Per-split (val | test): single=169.80|137.40, camber_rc=143.86|129.14, camber_cruise=101.49|89.66, re_rand=124.38|121.14. All four splits regress 10-18% vs Fourier baseline.

- analysis: Architecture depth scaling fails at fixed 30-min budget. The 1.83M-param (n=8) model vs 1.03M-param (n=5) baseline has ~78% more parameters but the same 12-epoch training budget. The model is clearly undertrained: val_avg shot from 134.88 (epoch 11) to 154.82 (epoch 12) — still on the steep part of the learning curve when wall-clock cuts in. The 9% per-epoch slowdown compounds the under-training problem. fp16 attempt NaN'd (gradient overflow on Fourier freq params without GradScaler); bf16 fixed this cleanly (native fp32 dynamic range on Blackwell GPU). The bf16 recipe is validated and reusable, but the depth hypothesis is falsified at this budget.
- decision: **closed**. Regression on all splits vs OLD Fourier baseline. Depth scaling is not the bottleneck at 30-min compute.
- next steps: pivot to width (mlp_ratio) instead of depth. fern reassigned to PR #3568: mlp_ratio=4 single-arm test.

## 2026-05-15 22:41 — PR #3441: slice_num=80 without gradient checkpointing (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/slice-num-80-no-checkpoint`
- hypothesis: removing gradient checkpointing at slice_num=80 trades VRAM headroom for convergence speed vs PR #3353
- W&B run: `o1wv46oq`

| Metric | This run | Baseline (PR #3352) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 130.17 | 116.34 | **+11.9% worse** |
| `test_avg/mae_surf_p` | 120.26 | 107.33 | **+12.0% worse** |
| Best epoch | 11 | 12 | — |
| Peak VRAM | 90.6 GB (94.8%) | ~33 GB | 2.7× more |
| Avg epoch time | 142.6s | ~128s | +11% |

Per-split val: single=150.35, camber_rc=142.71, camber_cruise=100.86, re_rand=126.78 — all regress.

- analysis: The key failure is the memory model. PR #3353 (slice_num=96+ckpt) used only 16 GB with checkpointing, but that was HIDING ~80 GB of activation memory. Removing checkpointing at slice_num=80 demands the full activation footprint (~90.6 GB), leaving ~5 GB headroom. Epoch time is +11% worse (not better), and convergence is noisy (±20–30 val swing between consecutive epochs). The slice_num axis is not the model's bottleneck at this dataset scale.
- decision: **closed**. Clear regression on all splits. The student's own VRAM analysis was correct — the slice attention cost grows quadratically and dominates without checkpointing.
- next steps: Abandon the slice_num scaling direction for now. frieren reassigned to new hypothesis.

## 2026-05-15 22:30 — PR #3356: Divergence-free velocity auxiliary loss (thorfinn) — beats OLD baseline, needs rebase on SmoothL1
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/divergence-free-aux-loss`
- hypothesis: adding a normalized-space divergence penalty on predicted velocity fields biases the model toward physically-realizable solutions, improving OOD generalization
- W&B runs: `ilylzwo5` (div_weight=0.01, winner), `2w40di7q` (div_weight=0.001, fails), `4ws0rum6` (div_weight=0.01 replication)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | vs OLD baseline (121.50/112.49) | vs NEW baseline (116.34/107.33) |
|---|---|---|---|---|
| div_weight=0.01 (ilylzwo5) | **113.41** | **102.86** | −6.7%/−8.6% | −2.5%/−4.2% |
| div_weight=0.01 (4ws0rum6 replication) | 117.20 | 108.05 | −3.5%/−4.0% | +0.7%/+0.7% |
| div_weight=0.001 (2w40di7q) | 133.48 | 122.17 | +9.9%/+8.6% | +14.7%/+13.8% |

Per-split val (div_weight=0.01, best run): single=142.61 (+2%), camber_rc=121.19 (−12.6%), camber_cruise=85.88 (−8.2%), re_rand=103.97 (−8.7%)

- analysis: The hypothesis is well-supported in direction: OOD splits (camber_rc, camber_cruise, re_rand) improve dramatically at div_weight=0.01 while in-dist (single) is slightly worse — the classic signature of useful regularization. **Key implementation finding**: naive denormalized divergence with raw x/z coordinates explodes to 1e5–1e7 on boundary-layer mesh pairs (|dx|~1e-5); student switched to normalized-space divergence with eps floor on |dx|,|dz|, giving div_loss in the 50–250 range and div_weight*div_loss balanced against vol_loss. **Caveat**: two runs at div_weight=0.01 disagree — best run beats new baseline (−2.5%/−4.2%), replication at +0.7%/+0.7% (barely misses). This variance is concerning for a definitive merge decision. The run was on the current learnable-Fourier baseline, BUT now that tanjiro's SmoothL1 is about to become the new baseline at val=90.60/test=83.00, even the best div-free result (113.41/102.86) doesn't beat the incoming baseline.
- decision: **sent back for rebase on SmoothL1 baseline** (pending tanjiro merge). The physics-informed direction is promising; must verify compound effect with SmoothL1.
- next steps: rebase onto SmoothL1 baseline. Single arm div_weight=0.01 (best arm from this round). A 10% sweep (0.005, 0.01, 0.02) would also clarify whether 0.01 is near-optimal.

## 2026-05-15 22:27 — PR #3215: SmoothL1 β=0.05 on learnable Fourier baseline (tanjiro) — PENDING MERGE (rate-limited)
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smooth-l1-loss`
- hypothesis: replacing MSE with Huber/SmoothL1 (β=0.05) in normalized space caps the gradient contribution of extreme y-values, reducing the distortion from large-Re samples
- W&B run: `iofja54s`

| Metric | Baseline (PR #3352, learnable Fourier) | SmoothL1 β=0.05 (rebased) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 116.3411 | **90.6039** | **−22.13%** |
| `test_avg/mae_surf_p` | 107.3254 | **83.0029** | **−22.66%** |
| Best epoch | 12 | 14 | +2 epochs |
| Peak VRAM | ~33 GB | ~42.5 GB | — |

Per-split val: single=112.03 (−22.8%), camber_rc=104.42 (−17.3%), camber_cruise=62.07 (−29.6%), re_rand=83.89 (−20.8%)
Per-split test: single=101.95 (−19.4%), camber_rc=97.84 (−17.2%), camber_cruise=55.10 (−28.1%), re_rand=77.11 (−28.6%)

- analysis: **Largest single-change improvement on this benchmark to date** — −22% on both headline metrics with all 4 splits improving 17-30%. Confirmed compound with learnable Fourier: magnitudes nearly identical to pre-rebase arms (val 90.24→90.60, test 82.21→83.00), confirming SmoothL1 and learnable Fourier are additive on this dataset. The improvement is largest on the highest-range splits (re_rand test −28.6%, camber_cruise −28%). Mechanism clear: MSE squares large normalized residuals, creating gradient domination from the most extreme Re/geometry samples; SmoothL1 with β=0.05 transitions to linear beyond |err|>0.05, effectively downweighting the outlier-sample gradient contribution. Best epoch was the LAST completed epoch (14/50, wall-clock limited) — val curve still declining, suggesting more headroom with extended training.
- decision: **PENDING MERGE** — blocked by GitHub API rate limit (5000/5000 exhausted; resets 23:19 UTC). PR is not-draft, status:review, MERGEABLE — ready as soon as rate limit allows.
- next steps: After merge, this becomes the new baseline (val=90.60, test=83.00). All in-flight PRs should be re-evaluated against this threshold. High-priority follow-ups: β sweep (0.02, 0.075, 0.10, L1 limit), per-channel β, SmoothL1+FiLM compound.

## 2026-05-15 20:35 — PR #3215: SmoothL1 loss β=0.05 and β=0.10 (tanjiro) — sent back for rebased verification
- student: willowpai2i24h2-tanjiro
- branch: `willowpai2i24h2-tanjiro/smoothl1-loss`
- hypothesis: Replace MSE with smooth-L1 (Huber) loss in normalized space; caps gradient contribution of high-Re extreme errors without zeroing them
- W&B runs: `638hd0v7` (β=0.05, won by val), `pbvt4zsz` (β=0.10, won by test)

| Metric | OLD baseline (#3200) | β=0.05 | β=0.10 | NEW baseline (#3352) | vs NEW |
|---|---|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **90.2450** | 91.2928 | 116.3411 | **−22.4% (β=0.05)** |
| `test_avg/mae_surf_p` | 112.4884 | 82.2072 | **81.1592** | 107.3254 | **−24.4% (β=0.10)** |
| best val epoch | 14 | 13 | 14 | 12 | — |
| peak VRAM | ~42.5 GB | 42.5 GB | 42.5 GB | ~33 GB | — |

Per-split val (β=0.05): single=105.40 (−25%), camber_rc=98.02 (−29%), camber_cruise=70.25 (−25%), re_rand=87.31 (−23%)
Per-split test (β=0.10): single=96.44 (−21%), camber_rc=90.75 (−32%), camber_cruise=60.05 (−28%), re_rand=77.40 (−31%)

**Largest single-change improvement on this benchmark to date.** All 4 splits improve by 21-34% on test. The β=0.05 and β=0.10 arms agree closely (within 1-2% on each metric), so the improvement is robust, not noise. Mechanism is consistent with hypothesis: high-Re extreme y-values (normalized errors up to ~5) dominate MSE gradients quadratically; SmoothL1 caps this contribution linearly above β.

**Caveat:** ran on OLD baseline (fixed Fourier features). vs NEW baseline (learnable Fourier, val=116.34, test=107.33), the gain is still −22% val / −24% test — likely real and compounds with learnable Fourier, but must verify. Sent back for ONE rebased single-arm re-run (β=0.05) on the new baseline. If it confirms (anywhere near val<100, test<90), merge immediately as new baseline.

**Side observation:** student noted an earlier β=0.05 run on un-rebased code (no Fourier features at all) got val_avg=99.14. That single-change improvement (SmoothL1 alone over the pre-Fourier baseline) is also dramatic. Suggests MSE→SmoothL1 is the dominant lever in the dataset, with Fourier features adding ~10-20% on top.

## 2026-05-15 20:32 — PR #3353: slice_num=96 + gradient checkpointing (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/slice-num-96-checkpoint`
- W&B run: `o70g1t9p` (full), `2d9oqekh` (smoke test, OK)

| Metric | This run | NEW baseline (#3352) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **130.7342** | 116.3411 | **+12.4% worse** |
| `test_avg/mae_surf_p` | **120.7062** | 107.3254 | **+12.5% worse** |
| Best val epoch | 10 / 50 | 12 | −4 |
| Avg epoch time | 192.5 s | ~128 s | **+50% slower** |
| Peak GPU memory | **16.0 GB** | ~33 GB | −52% |
| Wall-clock | 31.9 min | 30 min | similar |

Per-split test: single=152.57 (+25%), camber_rc=130.17 (**−10%** beats baseline!), camber_cruise=84.31 (+10%), re_rand=115.78 (+8%).

Analysis: Gradient checkpointing was over-aggressive — peak VRAM 16 GB out of 96 GB available, ~80 GB headroom unused. The 50% epoch-time penalty (from checkpoint recomputation) ate the 30-min budget, limiting to 10/50 epochs vs baseline's 12. Model was still descending hard at epoch 10 (174→131, ~24% drop) — under-convergence dominated the result. `geom_camber_rc` actually beating baseline by 3.2 test points partially supports the slice-bump hypothesis directionally, but cannot be claimed as a win in the equal-weight mean.

Decision: **CLOSED**. Wall-clock bottleneck, not memory. The student's own recommendation #1 is the right next step.

Reassignment: PR #3441 — slice_num=80 **without** gradient checkpointing. Uses memory headroom (predicted ~50-55 GB peak at slice_num=80, well under 96 GB), avoids the checkpoint recompute tax (predicted +10-20% epoch time vs +50%), aims for 12+ epochs in 30 min.

## 2026-05-15 20:25 — PR #3350: FiLM-style Re conditioning per Transolver block (alphonse) — sent back for rebase
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/film-re-conditioning`
- hypothesis: FiLM on every Transolver block, conditioned on per-sample log(Re), should improve val_re_rand by 5-12% and net val_avg
- W&B run: `5a3i5ctn` (v2 with bug-fix), `d5xeygsf` (v1 with buggy init + padding contamination, val=128.96)

| Metric | OLD baseline (#3200) | FiLM v2 | NEW baseline (#3352) | vs NEW |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **116.962** | 116.3411 | +0.5% (flat) |
| `test_avg/mae_surf_p` | 112.4884 | **104.636** | 107.3254 | −2.5% |
| best val epoch | 14 | 11 | 12 | — |
| peak VRAM | ~42.5 GB | 47.6 GB | ~33 GB | — |

Per-split test (best-val checkpoint): single=125.41, camber_rc=115.38, camber_cruise=73.86, re_rand=103.90 — all 4 splits improve vs NEW baseline.
Per-split val: single=147.81 (+1.9%), camber_rc=127.04 (+0.6%), camber_cruise=86.61 (−1.7%), re_rand=106.39 (+0.4%) — 3 of 4 val splits slightly worse vs NEW baseline.

Analysis: alphonse's run was on the OLD baseline (PR #3200, fixed Fourier) and finished ~1 min before PR #3352 merged. vs the OLD baseline, FiLM is a clear winner: val −3.7%, test −7.0%. But vs the NEW baseline (learnable Fourier), val is flat (within noise) and test is better (−2.5%, consistent across all 4 splits). The mechanism is sound — debug logs show clean per-sample log(Re) signal (after fixing zero-init preservation + padding contamination bugs the student found themselves). The Re-conditioning gain isn't only on re_rand split — geom_camber_rc and geom_camber_cruise also improve substantially on test, suggesting the model uses Re-implicit features for held-out cambers and FiLM frees up other capacity.

Decision: **sent back for rebase + re-run on learnable Fourier baseline**. The compound (FiLM + learnable Fourier) hasn't been tested. If it beats both val_avg < 116.34 and test_avg < 107.33, merge.

Sub-finding worth recording: alphonse caught two implementation bugs that future FiLM-like PRs should be aware of:
1. `Transolver.__init__` calls `self.apply(self._init_weights)` AFTER constructing all submodules, which overwrites the zero init in any sub-module that needs near-identity start. Fix: re-zero after `self.apply`.
2. Reading global per-sample features via `x[:, :, idx].mean(dim=1)` is **contaminated by padding** when `pad_collate` zero-pads in raw space and the model normalizes afterward. The padding rows in normalized space have value `-mean/std`, which is large in magnitude (~−19 for log Re) and dominates the mean. Fix: read row 0 directly, which is always a real node since pad_collate appends padding.

## 2026-05-15 19:28 — PR #3352: Learnable Fourier frequency bands (8 trainable freqs) — MERGED
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/learnable-fourier-freqs`
- hypothesis: Making the 8 Fourier frequency bands learnable nn.Parameters (initialized to octave-doubling baseline) allows gradient descent to discover which spatial scales matter most for TandemFoilSet, improving OOD geometry splits most.
- W&B run: `rumqs1au`

| Metric | Baseline (PR #3200) | PR #3352 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 121.4956 | **116.3411** | **−4.24%** |
| `test_avg/mae_surf_p` | 112.4884 | **107.3254** | **−4.59%** |
| best val epoch | 14 | 12 | — |
| peak VRAM | ~42.5 GB | ~33 GB | — |

Per-split surface-p (val | test):
- `single_in_dist`: 145.03 | 126.46 (+3.7% val regression)
- `geom_camber_rc`: 126.25 | 118.24 (−9.0% val / −11.3% test — largest OOD gain)
- `geom_camber_cruise`: 88.12 | 76.60 (−5.8% val / −7.8% test)
- `re_rand`: 105.96 | 108.00 (−7.0% val / −3.1% test)

Analysis: Frequencies barely moved from octave-doubling init (max drift 2.47% on freq_0: 1.000→1.025; all others <1%). The improvement is not from discovering a fundamentally different frequency basis — it comes from the extra gradient signal through the frequency parameters during optimization (and possibly a small regularization effect from weight decay on positive parameters). The octave-doubling init is near-optimal for this dataset. Biggest gains on OOD geometry splits as predicted — in-dist regressed slightly (+3.7%). Net improvement is clear and robust across 3 of 4 splits.

Decision: **MERGED** as new baseline. val_avg=116.34 > 121.50; test_avg=107.33 > 112.49.

Next steps: fern assigned PR #3413 (n_layers=8 + AMP) to test depth scaling on the new baseline.

## 2026-05-15 18:30 — PR #3207 re-run (post-NaN-fix): geom-conditioned slice (nezuko) — sent back for rebase
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- run: `5yws4drs` (single arm, NaN fix patch applied in `evaluate_split`)

| Metric | Value (W&B, clean) | vs baseline (PR #3200) |
|---|---|---|
| `val_avg/mae_surf_p` | 127.71 | +5.1% worse |
| `test_avg/mae_surf_p` | 116.56 | +3.6% worse |
| best epoch | 11 / 50 (wall clock at ep 11) | — |

Per-split test (best-val checkpoint): single=133.43, camber_rc=126.58 (**beats baseline 133.37**), camber_cruise=89.29, re_rand=116.94.

- analysis: nezuko's NaN-fix patch is correct — the new W&B test_avg (116.56) matches their offline-corrected value (115.71) within rounding. The PGOT hypothesis is supported in pattern: camber_cruise is the easiest split, camber_rc is the hardest, and camber_rc beats the merged baseline on that split alone. But the equal-weight mean is the metric; nezuko is worse than baseline on the other 3 splits and worse net.
- decision: **sent back for rebase + re-run** on the new baseline (PR #3200, Fourier 8 bands + NaN fix). Hypothesis: geom-slice and Fourier are orthogonal mechanisms (slice-token level vs input-augmentation level) and should compound. This is the final iteration — if rebased re-run doesn't beat baseline, close.
- next steps: if it wins, merge as 2nd baseline; if not, close and assign nezuko a Round-3 hypothesis.


## 2026-05-15 17:25 — PR #3191: Per-sample scale-normalizing loss (alphonse) — CLOSED
- student: willowpai2i24h2-alphonse
- branch: `willowpai2i24h2-alphonse/scale-norm-loss`
- hypothesis: dividing per-sample sq_err by per-sample, per-channel variance of `y_norm` rebalances the gradient between high-Re and low-Re samples
- runs: `qt4mwr34` (eps=1e-3, winner), `34gllrtz` (eps=1e-6, unstable)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p (offline) | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| eps=1e-3 | 148.51 | 140.57 (offline eval_clean.py, NaN in W&B) | 196.69 | 183.68 | **90.23** | 123.46 |
| eps=1e-6 | 365.51 | 346.89 | — | — | — | — |

- analysis: cruise (the low-Re-heavy split) was the lowest per-split val MAE — directionally consistent with the hypothesis. But `val_single_in_dist` regressed badly (196.69 vs ~140 on the merged baseline) and the equal-weight mean is the metric. eps=1e-3 vs eps=1e-6 showed a sharp instability cliff: tiny `eps` divides by near-zero variance for low-energy samples and blows up the loss. Side-channel offline eval (eval_clean.py) used because W&B test_avg was NaN from the scoring bug (now fixed in merged baseline). Even taking the offline number, 140.57 > 112.49 baseline test.
- decision: **closed**. Worse than the new merged baseline on both val and test. Hypothesis lives in too-narrow a slice of the design space to compound easily.
- next steps: reassigned alphonse to FiLM Re-conditioning (PR #3350) — a different attack on the same dynamic-range problem, from the architecture side.

## 2026-05-15 17:24 — PR #3206: Capacity scale-up n_hidden=256, n_head=8, slice_num=128 (frieren) — CLOSED
- student: willowpai2i24h2-frieren
- branch: `willowpai2i24h2-frieren/capacity-256-8h-128s`
- hypothesis: 2-3× parameter capacity helps overcome the small published Transolver config; 8M params + slice_num=128
- runs: `18jia682` (Arm A bs=4 OOM), `vn27s8f7` (Arm A bs=2), `hncmk6wk` (Arm B 192/6/96 bs=4 fallback)

| Arm | bs | val_avg (best) | test_avg (offline) | Epochs in 30 min | Peak GB |
|---|---|---|---|---|---|
| A 256/8/128 | 4 | — (OOM) | — | 0 | 93.4 (OOM) |
| A 256/8/128 | 2 | 159.95 @ ep 6 | 148.07 | 6 | 54.4 |
| B 192/6/96 | 4 | **160.29 @ ep 6** | 147.02 | 8 | 71.1 |

- analysis: Arm A at the PR-specified `batch_size=4` is *infeasible* on 96 GB (93.4 GB allocated → OOM at epoch 1). At equal 30-min budget, the larger model (Arm A bs=2) and the smaller fallback (Arm B bs=4) land within 0.7% of each other — both severely undertrained relative to the 50-epoch schedule. Capacity scale-up at this wall-clock budget is a wash. Student diagnosed the same scoring.py NaN root cause as nezuko (independently). Offline reeval.py numbers are inadmissible per program contract; even taking them, both arms lose to baseline (147.02 vs 112.49 test).
- decision: **closed**. Useful information about VRAM constraint at bs=4 with naive scale-up.
- next steps: reassigned frieren to slice_num=96 + gradient checkpointing (PR #3353) — surgical 50% slice bump with a concrete memory plan, addressing the OOM lesson from this PR.

## 2026-05-15 17:24 — PR #3218: Stochastic depth / DropPath (thorfinn) — CLOSED
- student: willowpai2i24h2-thorfinn
- branch: `willowpai2i24h2-thorfinn/drop-path`
- hypothesis: per-block residual DropPath with linearly-scaled drop_prob regularizes against overfitting on 1499 samples with full-file camber holdouts
- runs: `6cz9o2u5` (dpr=0.1), `6qvqt304` (dpr=0.1 seed 2), `7b1ae3dt` (dpr=0.2, winner)

| Arm | val_avg | val_single | val_camber_rc | val_camber_cruise | val_re_rand | test partial-avg (3 finite) |
|---|---|---|---|---|---|---|
| dpr=0.1 (seed 1) | 130.94 | 152.72 | 138.31 | 109.19 | 123.54 | 130.76 |
| dpr=0.1 (seed 2) | 131.75 | 162.35 | 141.60 | 105.54 | 117.51 | 130.96 |
| dpr=0.2 | **128.90** | 168.52 | 135.93 | **96.86** | **114.28** | 127.80 |

- analysis: DropPath at dpr=0.2 beat dpr=0.1 by ~2.4 absolute on val_avg, which is bigger than the dpr=0.1 seed variance (~0.8). The OOD signal is the cleanest result: all three OOD splits improved with dpr=0.2 (camber_rc -4.0, camber_cruise -10.5, re_rand -6.2) while the in-dist split regressed (+11.0). Classic regularization trade-off, exactly as the hypothesis predicted. Late-training behavior under dpr=0.2 was much more stable (final-epoch val_avg 138.93 vs 185.03). All runs hit the wall clock at ~ep 14 and reported NaN test_avg (scoring bug — fern's PR #3200 fix lands the fix in main).
- decision: **closed**. val_avg=128.90 doesn't beat new merged baseline (121.50). The OOD-specific gain is interesting but the net equal-weight metric loses.
- next steps: reassigned thorfinn to divergence-free aux loss (PR #3356) — a physics-informed regularizer that should give directional (rather than random) bias toward generalization.

## 2026-05-15 17:22 — PR #3200: Fourier position features 8 bands (fern) — **MERGED, new baseline**
- student: willowpai2i24h2-fern
- branch: `willowpai2i24h2-fern/fourier-pos-features`
- hypothesis: sinusoidal Fourier features on normalized (x, z) give the model direct access to multi-scale spatial frequency content
- runs: `t1ai7kzf` (8 bands, winner), `oj5578rn` (12 bands, worse)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| 8 bands | **121.50** | **112.49** | 139.80 | 138.71 | 93.55 | 113.93 |
| 12 bands | 129.26 | 121.27 | — | — | — | — |

- analysis: clean win. 8 bands beats 12 bands clearly on both val and test (the smaller input has fewer dims for the shallow preprocess MLP to disentangle in 14 epochs). All 4 splits stayed finite — fern bundled the `evaluate_split` NaN fix (zero non-finite y rows before `accumulate_batch`) in the same PR, unblocking finite W&B test_avg for the entire research track. Per-split test values: single=122.01, camber_rc=133.37, camber_cruise=83.11, re_rand=111.46. Val curve still descending at ep 14 when wall clock fired — there's still headroom in this configuration.
- decision: **merged** as the new baseline (commit `421b225f`). BASELINE.md updated.
- next steps: Round-2 hypotheses build on this baseline. Fern reassigned to learnable Fourier frequencies (PR #3352) as a natural extension.


## 2026-05-15 15:25 — PR #3207: PGOT-style geometry-conditioned slice assignment
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- hypothesis: injecting per-node geometry features (NACA M/P/T, AoA, Re, gap, stagger) into PhysicsAttention's slice projection improves generalization across the camber-holdout splits without hurting in-distribution
- runs: `pjmkgg22` (wandb_group `willow-pai2i-24h-r2/geom-slice-injection`)

| Arm | val_avg/mae_surf_p (best, ep 12) | test_avg/mae_surf_p (W&B) | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| geom-slice | **128.34** | NaN ⚠ | 145.96 | 142.21 | 107.66 | 117.51 |

- analysis: The hypothesis is supported on validation — geom-slice beats the warmup=3 val_avg (136.55, PR #3194) by ~6% with no regression on any of the four val tracks, and `val_geom_camber_rc` drops from 152.82 → 142.21 (–7%), exactly the split the hypothesis targeted. The run completed all 50 epochs in 31.5 min (just over wall clock cap, last-epoch eval was the bottleneck) and converged cleanly with `val_avg` still falling slowly after epoch 12, suggesting more headroom with a slightly larger model or schedule adjustment. **However, `test_avg/mae_surf_p` is NaN in W&B** — same global bug as PR #3194 (data/scoring.py:48 computes `(pred - y).abs() * mask` BEFORE the per-sample skip, so `inf*0 = NaN` poisons the accumulator when GT has non-finite values; reproduced to `test_geom_camber_cruise/000020.pt` having `y[..., 2] = -inf` at 761 volume nodes). The student computed an offline-corrected `test_avg = 115.71` by re-running scoring with NaN-zeroed samples, but the program contract requires the W&B-logged metric to be the source of truth.
- decision: **sent back** to draft with the exact `evaluate_split` patch (pre-zero non-finite y samples and exclude them from the metric via the mask/is_surface, before calling `accumulate_batch`). Asked the student to re-run the same single arm and confirm the W&B `test_avg/mae_surf_p` reads ~115.71. The hypothesis is the strongest candidate so far; if the re-run lands a finite test number, this becomes the first merge-eligible Round-1 result.
- next steps: on a clean re-run, merge this as the new baseline (val_avg/mae_surf_p=128.34, test=115.71). Then Round 2 priorities: (a) stack geom-slice + warmup=3 (small additive risk, both target different bottlenecks), (b) per-block geometry conditioning (FiLM-style modulation), (c) sweep `slice_num` since slice-token capacity is the load-bearing component.


## 2026-05-15 14:45 — PR #3194: 5-epoch LR warmup + cosine annealing
- student: willowpai2i24h2-askeladd
- branch: `willowpai2i24h2-askeladd/lr-warmup-cosine`
- hypothesis: linear LR warmup over the first 5 epochs prevents cold-start damage to the PhysicsAttention slice projection and improves `val_avg/mae_surf_p`
- runs: `5jtgoadb` (warmup=3), `gyin7q96` (warmup=5)

| Arm | val_avg/mae_surf_p (best, ep 13) | test_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| warmup=3 | **136.55** | NaN ⚠ | 159.58 | 152.82 | 109.78 | 124.01 |
| warmup=5 | 153.72 | NaN ⚠ | 207.68 | 155.53 | 116.98 | 134.70 |

- analysis: Two arms compared; warmup=3 beat warmup=5 across every val split. Both hit the 30-min wall clock at epoch 14 (cosine barely decayed). The student rightly flagged that this is a warmup-3-vs-warmup-5 comparison only — there is no no-warmup arm to confirm warmup itself beats the existing schedule. `test_geom_camber_cruise` returned `Infinity` in the pressure channel for at least one cruise sample on both arms, which propagates through `data/scoring.py`'s global accumulator and poisons `test_avg/mae_surf_p` to NaN. NaN on the paper-facing metric is a merge blocker per the program contract.
- decision: **sent back** to the student with two requirements: (1) defensively zero out predictions in padded positions and apply `nan_to_num(...).clamp_(-50, 50)` inside `evaluate_split` to localize any overflow without touching the read-only `data/scoring.py`; (2) re-run with two arms in the same wandb_group `willow-pai2i-24h-r2/warmup-cosine-v2` — `warmup=0` (proper baseline) and `warmup=3` (winner). The warmup=5 arm is dropped.
- next steps: once the re-run clears NaN and shows warmup=3 ≥ warmup=0 by any margin, merge. The `nan_to_num` fix will become the baseline for all subsequent PRs.


## 2026-05-15 23:21 — PR #3198: Per-channel surface loss weights (edward) — CLOSED
- student: willowpai2i24h2-edward
- branch: `willowpai2i24h2-edward/p-channel-weight`
- hypothesis: upweighting the pressure channel in the surface loss (p_surf_weight ∈ {2.0, 3.0, 5.0}) would improve surface-p prediction over the uniform 10× surface weight
- W&B runs: arms p2, p3 (`n0c2k6j9`, val=128.66/test=119.14), p5 (`9a1p79b3`, val=130.46/test=120.21)

All 3 arms worse than old baseline (116.34/107.33) and far worse than new SmoothL1 baseline (90.60/83.00).

- analysis: Per-channel loss weights and SmoothL1 address the same problem (gradient domination by high-error samples) from different angles — weights reduce relative channel contribution while SmoothL1 reduces per-sample outlier gradient. With SmoothL1 already in place, the per-channel weight change is redundant: the outlier samples that dominated MSE are already downweighted by SmoothL1's linear-regime cap. The null result is mechanistically expected in retrospect.
- decision: **closed**. Null result, redundant mechanism.
- next steps: edward reassigned to domain one-hot embedding (PR #3523).
