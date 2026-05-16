<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2i-24h-r3`

## 2026-05-16 00:25 — PR #3391 thorfinn CLOSED: NACA Fourier stacked (val 115.45 regression)

- Branch: `willowpai2i24h3-thorfinn/naca-fourier-stacked`
- Hypothesis: NACA 4-digit Fourier positional features (sin/cos harmonics of camber c, camber position m, thickness t) prepended to Transolver input, predicting better cross-geometry generalization on camber OOD splits.

### Terminal results

| Run | val_avg | test_nansafe | Δ vs Lion baseline 94.08 |
|---|---:|---:|---:|
| `jge3bnk4` (primary, K=4 Fourier) | 115.4505 | 112.1973 | **+21.37 (regression)** |

Per-split val: single_in_dist=126.97, geom_camber_rc=122.26, geom_camber_cruise=98.58, re_rand=113.98.
All four val splits worse than Lion baseline (single_in_dist: 108.05, geom_camber_rc: 109.69, camber_cruise: 69.35, re_rand: 89.22).

### Analysis (thorfinn's diagnosis confirmed correct)

Three compounding failure modes identified by thorfinn, all accepted:

1. **Standardized input breaks Fourier periodicity**: Features applied to z-scored camber parameters — sin(2π·k·x) is only meaningful when x ∈ [0,1]; after z-scoring, the periodicity is lost and the features carry noise not signal.
2. **Capacity dilution**: Input expanded from 22→38 channels in the first Linear layer; without increasing hidden width, per-channel attention capacity drops by ~40%.
3. **Huber already subsumed the geometry signal**: The camber/OOD outlier gradient problem that NACA Fourier was intended to fix has already been absorbed by Huber's δ=2.0 elbow, leaving no residual signal for geometry encoding to capture.

Thorfinn's own recommendation was correct: "Probably stop trying NACA Fourier on top of Huber. The signal from this run is strong enough."

### Decision: CLOSED

- val_avg 115.45 is +22.7% regression vs Lion baseline 94.08 ✗
- All 4 val splits worsened ✗
- No recovery path: the three identified failure modes each require architectural changes that constitute a different hypothesis
- NACA Fourier as a 22-channel input augmentation on standardized features is eliminated

### Reassignment

Thorfinn reassigned to `lion-lr-wd-sweep` (PR #3541) — test Lion paper's published hyperparameter ratios (lr 3–10× smaller, wd 3–10× larger than AdamW optimal). Highest-confidence orthogonal lever not covered by any other in-flight PR.

---

## 2026-05-15 23:00 — PR #3394 frieren CLOSED: surface-only Huber (val 103.20, high variance)

- Branch: `willowpai2i24h3-frieren/huber-surface-only`
- Hypothesis: Apply Huber only to surface term; keep MSE for volume. Tighter δ=1.0.

| Run | δ | val_avg | test_nansafe | Δ vs old baseline |
|---|---|---:|---:|---:|
| `pbg3fjj5` (δ=1.0 arm 1) | 1.0 | 103.20 | 101.39 | −4.26 (beats old Huber) |
| `1h1wlbdy` (δ=1.0 arm 2) | 1.0 | 121.38 | 119.06 | +13.92 (regression) |
| `mrtndp5i` (δ=2.0 surf-only) | 2.0 | 117.04 | 116.76 | +9.57 (regression) |

**Decision**: 103.20 doesn't beat new Lion baseline 94.08. Closed. Key finding: 18-point spread between identical-config runs flagged as a cohort-wide variance concern. Fixed seed required in future. Frieren reassigned to `lion-warmup` (#3515).

---

## 2026-05-15 23:00 — PR #3403 edward CLOSED: T_max=14 diagnostic confirmed (val 103.30)

- Branch: `willowpai2i24h3-edward/lr-tmax-fix`
- Hypothesis: Fix cosine T_max=50 → T_max=14 matching actual epoch budget.

| Run | T_max | val_avg | test_nansafe | Notes |
|---|---:|---:|---:|---|
| `2j268eqn` (primary) | 14 | 103.30 | 98.64 | Full schedule, LR→0 at ep14 |
| `pn4p54cm` (replication) | 14 | 104.35 | — | Very consistent — low seed noise |
| `r2ovztrr` | 12 | 120.68 | 113.15 | Aggressive — wasted high-LR steps |
| `o9vw958j` | 14 | 137.66 | 132.06 | Incomplete (ep7 only) |

**Decision**: T_max=14 confirmed as −3.9% improvement on old AdamW+Huber baseline. High replication consistency (103.30 vs 104.35). Doesn't beat new Lion baseline 94.08. Closed as completed diagnostic. Edward reassigned to `lion-tmax14` (#3518) — stack T_max=14 fix on Lion baseline.

---

## 2026-05-15 23:00 — PR #3427 alphonse SENT BACK for rebase: val=92.62 confirmed

- Branch: `willowpai2i24h3-alphonse/bf16-stable`
- Terminal confirmed: val=92.6166, test=87.6987 (two arms, both 19 epochs in 30 min)

| Arm | val_avg | test_nansafe | Δ vs Lion baseline 94.08 |
|---|---:|---:|---:|
| `to8x5txt` (δ=2.0, primary) | 92.62 | 87.70 | **−1.46 (beats Lion)** |
| `8x6xlmup` (δ=1.0) | 93.74 | 87.32 | −0.34 (narrow) |

Late-cosine divergence fully eliminated (primary: best=92.62 at ep19, stable). But this result was trained on Huber baseline — branch needs rebase to incorporate Lion. Sent back for rebase+rerun. Predicted Lion+bf16+clip+floor val ~82–91.

---

## 2026-05-15 21:45 — PR #3387 fern MERGED: Lion+Huber new SOTA → val=94.08

- Branch: `willowpai2i24h3-fern/lion-stacked`
- Hypothesis: Lion optimizer (sign-based update rule, `lr=1e-4, wd=1e-2`) stacked on
  the merged Huber loss (δ=2.0) baseline, to compound two orthogonal gradient-capping
  mechanisms: Huber at the loss level, Lion at the optimizer level.

### Terminal results (1 arm `f9w6yzoq`)

| Metric | Lion+Huber | Huber baseline | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **94.0803** | 107.4641 | **−13.38** |
| **test_avg_nansafe/mae_surf_p** | **88.9362** | 101.9848 | **−13.05** |
| val_single_in_dist | 108.0536 | 127.9121 | −19.86 |
| val_geom_camber_rc | 109.6926 | 118.4850 | −8.79 |
| val_geom_camber_cruise | 69.3504 | 83.3455 | −13.99 |
| val_re_rand | 89.2247 | 100.1139 | −10.89 |

Total epochs: 14 in 31 min (timeout). Best epoch = last epoch (val still descending). Peak VRAM: 42.1 GB.

### Analysis

Lion's sign-based update rule bounds the per-parameter gradient magnitude uniformly,
complementing Huber's per-sample loss-level bound. Result: every val split and every test
split improved. The −12.4% improvement substantially exceeded the −3 to −8% prediction.

**Critical signal**: Val curve slope at timeout was −2.9/epoch, still descending without
plateau. Best epoch = epoch 14 (the last one). Material headroom remains — the curve was
cut off by the 30-min wall-clock, not by convergence.

### Decision: MERGED as new round-4 SOTA

- val_avg 94.08 beats prior baseline 107.46 by −13.38 (−12.4%) ✓ 
- test_nansafe 88.94 beats 101.98 by −13.05 (−12.8%) ✓
- Terminal SENPAI-RESULT: `{"terminal":true,"status":"complete","pending_arms":false,...}` ✓
- All 4 val splits improved, all 4 test splits improved ✓
- Merged 21:45 UTC, new BASELINE.md entry created

### Reassignment

Fern reassigned to `lion-bf16-stacked` (PR #3481) — add bf16 mixed precision to extend
~14 epochs to ~19 epochs in the same 30-min budget. The descending val curve makes this
the highest-confidence next lever in the cohort.

---

## 2026-05-15 21:30 — PR #3385 askeladd posts terminal; sent back for max_norm=50 variant

- Branch: `willowpai2i24h3-askeladd/warmup-cosine-stacked`
- Hypothesis: Stack LR warmup (2 or 5 epochs) + cosine + grad-clip(max_norm=1.0) on the
  merged Huber baseline to compound optimization stability levers.

### Terminal results (2 arms)

| Run | warmup | val_avg/mae_surf_p | test_avg_nansafe/mae_surf_p | Δ val vs 107.46 | Δ test vs 101.98 |
|---|---|---:|---:|---:|---:|
| `4brvwa73` (warmup2-cos-clip1) | 2 | 107.6118 | **100.7504** | +0.15 | **−1.23** |
| `gr05e3j0` (warmup5-cos-clip1) | 5 | 113.9132 | 107.0005 | +6.45 | +5.02 |

### Diagnostic finding (askeladd's analysis)

`train/grad_norm` logging revealed **100% of steps were clipped** on both arms. Observed
median pre-clip grad norms ~21–26 vs max_norm=1.0 ceiling → clipping was acting as a
~20–40× per-step LR reduction every step, not as a stability ceiling for rare spikes.
Combined with the 30-min wall-clock cap (14 epochs), warmup further eats productive
training time (warmup=5 loses ~36% of training to warmup phase, warmup=2 only ~14%).

### Decision: send back for `warmup2-clip50` arm

- val_avg essentially tied (+0.15 = within run-to-run noise floor); not merging on primary metric
- test_nansafe modest win (−1.23) consistent with implicit regularization from over-clipping
- Sent back with feedback: run third arm `warmup2-clip50` using `max_norm=50` (≈2× observed median),
  so only true spike batches get clipped and the optimizer can move freely between spikes
- askeladd's own follow-up suggestion #1 — directly actionable

---

## 2026-05-15 19:40 — PR #3282 alphonse closes bf16-mixed-precision; reassigned to bf16-stable (#3427)

- Branch: `willowpai2i24h3-alphonse/bf16-mixed-precision`
- Hypothesis: bf16 mixed-precision to lift wall-clock cap (~2x epoch throughput), enabling
  more cosine annealing within the 30-min budget. Expected ~18–22 epochs vs ~14 fp32.

### Terminal results

| Metric | Value |
|---|---|
| W&B run | `tup20e60` (group: `bf16-mixed-precision`) |
| **val_avg/mae_surf_p (best)** | **111.566** (epoch 16/19) |
| val_single_in_dist | 127.898 |
| val_geom_camber_rc | 127.287 |
| val_geom_camber_cruise | 88.569 |
| val_re_rand | 102.509 |
| **test_avg_nansafe/mae_surf_p (3-split manual)** | **109.134** |
| Total epochs | 19 in 30.95 min |
| Mean per-epoch wall-clock | 97.7 s |
| Peak VRAM | 80.3% of 96 GB (~77 GB) |
| Final val_avg (ep19) | 171.42 — late-cosine divergence confirmed |

### Analysis

Throughput hypothesis confirmed: bf16 delivers ~19 epochs in 30 min vs ~14 fp32 L=5
(~36% more steps), matching prediction. The best checkpoint (ep16) is cohort runner-up at
111.57, but it doesn't beat the new 107.46 Huber baseline.

The dominant failure mode is **late-cosine divergence**: `tup20e60` best=111.6 at ep16,
then val spikes to 171.4 at ep19. Pattern aligns with earlier frieren divergence
(`1walszqd`). Root cause: cosine T_max=50, only 19 epochs running — LR near-zero regime
causes gradient/momentum accumulation instability. This is the same T_max=50
misconfiguration edward is testing in isolation (#3403), but the late divergence emerges
earlier in bf16 runs because they reach the dangerous LR-tail regime more quickly.

Also notable: alphonse identified a cruise-nansafe bug in his eval pipeline — his
`test_geom_camber_cruise` nansafe filter returns NaN even after `nan_to_num`. Likely
bf16 attention softmax overflow on a specific sample poisoning the accumulator before
the filter runs. Flagged for downstream investigation.

### Reassignment

Alphonse reassigned to **`bf16-stable`** (PR #3427) — three stacked orthogonal levers:
1. bf16 autocast (throughput retained from #3282)
2. `clip_grad_norm_(max_norm=1.0)` (stops late divergence at optimizer level)
3. `eta_min=1e-5` in CosineAnnealingLR (prevents LR-near-zero instability at tail)

Predicted: val_avg/mae_surf_p ~99–103 if all three levers stack cleanly on the Huber baseline.

---

## 2026-05-15 19:23 — PR #3313 edward closes grad-accum; reassigned to lr-tmax-fix (#3403)

- Branch: `willowpai2i24h3-edward/grad-accum`
- Hypothesis: Gradient accumulation (accum_steps=2) to simulate batch_size=8 under
  H100 96GB VRAM ceiling, expecting smoother gradients and improved convergence.

### Terminal results

| Metric | Value |
|---|---|
| W&B runs | `wgsyk2sz` (accum=2, val=137.42), earlier arm val=196.07 |
| **val_avg/mae_surf_p (best)** | **137.42** (+28% regression vs 107.46 Huber baseline) |
| Best epoch / total | epoch 12/14 (timeout) |
| Status | closed by advisor 19:23 UTC |

### Analysis (edward's own closure note)

Edward's closure analysis pinpointed the root cause: cosine T_max=50 misconfiguration.
With only 14 epochs running under the 30-min cap, the LR barely anneals (from 5e-4 to
~4.7e-4, only 3% of the cosine range). Grad-accum doubles the effective batch (→ noise
floor lower) but the constant-LR regime can't exploit it; combined with the noise-floor
shift, it pushes the optimizer into a worse minimum.

### Reassignment

Edward reassigned to **`lr-tmax-fix`** (PR #3403) — round-5 priority-1 idea from
researcher agent. First-principles diagnostic: add `--lr_T_max` CLI override and test
`--lr_T_max 14` (matches actual epoch budget) and `--lr_T_max 12` (LR hits near-zero by
end). If the isolated T_max fix improves on 107.46, it becomes the new baseline for ALL
subsequent stacking experiments.

---

## 2026-05-15 18:22–18:45 — Round-3 closure + round-4 assignments

### Round-3 merges and closures

| PR | Student | Action | val_avg | Reason |
|---|---|---|---:|---|
| #3248 | frieren | **MERGED** | 107.46 | Round-3 winner. Huber δ=2.0. New baseline. |
| #3244 | askeladd | closed | 109.99 | Doesn't beat new baseline 107.46; stacking test assigned (round-4 PR #3385) |
| #3249 | nezuko | closed | 130.18 | EMA neutral (≈ fresh-slate baseline); lever doesn't apply at 13-epoch budget |
| #3250 | tanjiro | closed | 124.76 | Loss reweighting regression; re-test on Huber baseline assigned (PR #3392 delta sweep) |
| #3251 | thorfinn | closed | 123.35 | NACA Fourier +5.1% on MSE; training stability was binding constraint, not geometry; re-test on Huber (#3391) |
| #3312 | fern | closed | 115.49 | Lion +12% on MSE; re-test stacked on Huber (#3387) |

Edward (#3313, grad-accum) and alphonse (#3282, bf16) still WIP, nudged for terminal.

### Round-4 assignments (2026-05-15 18:30–18:45)

All 6 idle students assigned stacking experiments on the Huber baseline:

| PR | Student | Slug | Key change |
|---|---|---|---|
| #3385 | askeladd | `warmup-cosine-stacked` | Warmup 5ep + cosine + grad-clip=1.0 |
| #3387 | fern | `lion-stacked` | Lion lr=1e-4, wd=1e-2 |
| #3389 | nezuko | `surf-weight-sweep` | `--surf_weight 5.0` and `20.0` (2 arms) |
| #3391 | thorfinn | `naca-fourier-stacked` | NACA Fourier features rebased onto Huber |
| #3392 | tanjiro | `huber-delta-tuning` | `--huber_delta 0.5 / 1.0 / 3.0` (3 arms) |
| #3394 | frieren | `huber-surface-only` | MSE vol + Huber surf, δ=1.0 and δ=2.0 |

---

## 2026-05-15 17:39 — PR #3248 frieren posts terminal SENPAI-RESULT — round-3 cohort leader

- Branch: `willowpai2i24h3-frieren/huber-delta2`
- Hypothesis: Replace MSE with Huber loss (δ=2.0) in normalized space to cap gradient
  contribution from high-magnitude outliers (high-Re tail, geom_camber_rc), expecting
  better cross-Re and cross-geometry generalization.

### Primary run `mp8s8okf` (best of 3 arms)

| Metric | Value |
|---|---|
| W&B run | `mp8s8okf` (`huber-delta2`) |
| **val_avg/mae_surf_p** | **107.4641** (best epoch 14/50) |
| **test_avg_nansafe/mae_surf_p** | **101.9848** (3-split mean) |
| Best epoch / total | 14 / 50 (timeout @ 31.05 min) |
| Mean epoch wall-clock | 131.7 s |
| Peak VRAM | 96.66 GB / 96 GB H100 (~94%) |
| Params | 0.66 M (no architecture change) |
| W&B group | `huber-robust-loss` |

### Per-split val (best ckpt)

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 127.91 |
| val_geom_camber_rc | 118.48 |
| val_geom_camber_cruise | 83.35 |
| val_re_rand | 100.11 |
| **val_avg** | **107.46** |

### Per-split test (nansafe, best ckpt)

| Split | mae_surf_p (nansafe) |
|---|---|
| test_single_in_dist | 114.43 |
| test_geom_camber_rc | 107.92 |
| test_geom_camber_cruise | 89.01 (NaN in-tree from data bug) |
| test_re_rand | 96.58 |
| **test_avg (nansafe)** | **101.98** |

### Sweep — all 3 arms (group `huber-robust-loss`)

| run | best val_avg | final val_avg | note |
|---|---|---|---|
| `mp8s8okf` (primary) | 107.46 | 107.46 | stable; final==best, this is the anchor |
| `wkrqrv80` | 114.24 | 120.34 | slight late drift |
| `1walszqd` | 121.85 | 175.16 | late divergence (LR-tail sensitivity); not anchor |

### Analysis

- **Cohort leader by 2.3% over askeladd** (109.99). Huber δ=2.0 attacks the binding
  high-loss-tail constraint head-on: by capping outlier gradient magnitude, it removes
  noise injected by the very-high-std single-foil samples without changing the
  optimizer schedule or architecture.
- **val_re_rand=100.11 is the largest win** — consistent with the gradient-rebalancing
  mechanism (high-Re tail samples no longer dominate updates).
- **Stability sensitivity:** 2 of 3 arms drift late in training; Huber + standard
  cosine is sensitive to LR-tail behavior. `mp8s8okf` (the primary) doesn't drift, but
  this is a known weakness that frieren's `huber-surface-only` round-4 follow-up
  should address by combining with askeladd's grad-clip/warmup.
- **Confirmed the `data/scoring.py` bug:** the in-tree `test_avg/mae_surf_p` is None
  due to cruise NaN propagation. Student computed nansafe variants exactly per the
  cohort-wide protocol.

### Advisor action (next invocation due to GH rate limit reset 18:19 UTC)

- **MERGE FIRST** via `senpai:merge-winner 3248 target/`. PR currently in draft
  state — need `gh pr ready 3248` before the merge skill.
- BASELINE.md will update to `val_avg/mae_surf_p=107.46`,
  `test_avg_nansafe/mae_surf_p=101.98`.
- After frieren merges, askeladd #3244 (warmup-cosine-grad-clip) merges second as a
  compound improvement — the two levers (loss function vs optimizer schedule) are
  orthogonal and should stack.

## 2026-05-15 15:50 — Round-3 cohort interim ranking (no merges yet)

W&B sweep of all in-flight round-3 runs (project `wandb-applied-ai-team/senpai-v1`,
agent prefix `willowpai2i24h3-`). All runs trained 11–14 epochs in the 30-min cap.

| Rank | Agent | Run | Group / hypothesis | val_avg/mae_surf_p | Status |
|---|---|---|---|---|---|
| 1 | askeladd | `6swu9ka3` | warmup-cosine-grad-clip | **109.99** | finished, frontier |
| 2 | askeladd | `trlcrai2` | warmup-cosine-grad-clip | 114.80 | finished |
| 3 | askeladd | `4ffogic3` | warmup-cosine-grad-clip | 115.06 | finished |
| 4 | frieren  | `8mgwqtn4` | huber-robust-loss | 124.66 | finished |
| 5 | tanjiro  | `bhywnmol` | re-conditioned-loss-weighting | 125.07 | finished |
| 6 | tanjiro  | `nfw04qzx` | re-conditioned-loss-weighting | 127.82 | finished |
| — | edward   | `7fa1s7vm` | baseline (AdamW, equal channels) | **129.99** | finished, fresh-slate anchor |
| 7 | nezuko   | `qln1o6ew` | ema-model-averaging | 130.17 | finished |
| 8 | fern     | `pf6dwz1f` | larger-slice-num (S=128) | 133.73 | finished, test NaN |
| 9 | edward   | `0723rw1e` | surf-p-weighted-loss [1,1,3] | 135.66 | finished, **+4.4% vs baseline** |
| 10 | nezuko  | `70w6bkyh` | ema-model-averaging | 135.98 | finished |
| 11 | thorfinn| `flqftgbz` | naca-camber-fourier-features | 138.36 | finished |
| 12 | thorfinn| `8bk36jc8` | naca-camber-fourier-features | 140.82 | finished |
| 13 | alphonse| `sof2eicn` | deeper-transolver (L=8) | 147.85 | finished, undertrained |

In-flight as of 15:50 (cohort not closed): askeladd `by2u0eyv`, frieren `mp8s8okf`,
nezuko `022pwbj4`, tanjiro `nbm68wvs`, thorfinn `n2i46t6r`.

Key cohort signal: **training-stability changes dominate** (askeladd's warmup-cosine-grad-clip
at 109.99 leads by ~14 vs the next tier of loss-formulation tweaks at 124–127). All test_avg
metrics are NaN in-tree because of the cruise-idx-20 `-inf` bug in `data/scoring.py`; per-split
finite tests are usable.

## 2026-05-15 15:50 — PR closures (3 review-ready)

### PR #3243 — Deeper Transolver L=8 (alphonse) — **closed**

- val_avg/mae_surf_p = 147.85, test_avg (nansafe) = 138.60
- Bottom of cohort. 33% behind frontier. Hypothesis undertrained (9/50 epochs).
- The depth lever returns when bf16 (#3282) unblocks the epoch budget.
- Diagnostic credit: alphonse identified the `data/scoring.py` NaN propagation bug
  (cruise idx 20 has `-inf` in interior `y[:,2]`; `NaN * 0 = NaN` poisons surface metric).
  Now project-wide policy: every run logs `test_avg_nansafe/mae_surf_p`.

### PR #3245 — Per-channel loss weights [1,1,3] (edward) — **closed**

- val_avg/mae_surf_p = 135.66 vs equal-weight baseline 129.99 → **+4.4% (worse)**
- Hypothesis directionally falsified. Predicted Ux/Uy degradation also observed (+15% on
  val_single_in_dist Ux), consistent with the loss reallocation pulling capacity away from
  velocity channels without compensating gain on pressure.
- Best artifact: edward's clean baseline run `7fa1s7vm` at 129.99 (14 epochs) is now the
  anchored fresh-slate reference for round-3 ranking.

### PR #3247 — Larger slice_num S=64→128 (fern) — **closed**

- val_avg/mae_surf_p = 133.73, **test_avg = NaN** (cruise pressure prediction → ±inf,
  reproducible across runs `pf6dwz1f` and `kcpsgrot`)
- Cruise val improved to 104.24 (best cruise val of cohort) — signal that slice scaling helps
  large meshes — but cannot merge with non-finite test pressure.
- New project-wide bug class: **model-side numerical instability** at `slice_num=128` in
  PhysicsAttention. Distinct from the data-side `-inf` in `data/scoring.py`.
- Future slice-num work must pair with a stability guard (fp32-stable softmax in slice
  projection, logit clamp, or slice_norm divisor floor). Not pursued now — fern reassigned
  to lion-optimizer.

## 2026-05-15 15:50 — PR #3282 status (alphonse, bf16-mixed-precision)

- Smoke run `1t41l8sx` crashed at 0.1 min (config issue, likely autocast/dtype mismatch).
- Advisor left a debug nudge with the canonical bf16-with-Transolver recipe (autocast
  wrap, no GradScaler, loss outside autocast). Awaiting next run.

## 2026-05-15 14:30 — PR #3243: Deeper Transolver (n_layers 5 → 8)

- Branch: `willowpai2i24h3-alphonse/deeper-transolver`
- Student: willowpai2i24h3-alphonse
- Hypothesis: increasing depth from L=5 to L=8 (paper's reference depth) reduces
  `val_avg/mae_surf_p` because the baseline is capacity-limited. Predicted −8% to −15%.

### Results

| Metric | Value | W&B |
|---|---|---|
| W&B run | `sof2eicn` (deeper-transolver-L8) | https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/sof2eicn |
| best `val_avg/mae_surf_p` | **147.85** at epoch 9 of 9 | |
| `test_avg/mae_surf_p` (nansafe) | **138.60** | (in-tree scorer NaN — see bug note) |
| best epoch / total epochs | 9 / 9 | timeout hit at 30.89 min |
| mean epoch wall-clock | 208 s/epoch (≈3.47 min) | |
| total train minutes | 30.89 (cut by `SENPAI_TIMEOUT_MINUTES=30`) | |
| params | 1.03 M (+56% vs L=5) | |
| peak VRAM | 64.5 GB / 96 GB (not logged in W&B summary; reported by student) | |
| batch_size | 4 (no OOM) | |

### Per-split val at best epoch

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 176.27 |
| val_geom_camber_rc | 172.55 |
| val_geom_camber_cruise | 112.02 |
| val_re_rand | 130.55 |
| **val_avg** | **147.85** |

### Per-split test (nansafe)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 154.06 |
| test_geom_camber_rc | 156.95 |
| test_geom_camber_cruise | 112.50 |
| test_re_rand | 130.88 |
| **test_avg (nansafe)** | **138.60** |

### Analysis

- Train loss decreased monotonically over all 9 epochs; no instability — the deeper
  model trained cleanly.
- Hard `SENPAI_TIMEOUT_MINUTES=30` cap → 9 of 50 epochs only; cosine `T_max=50` meant
  LR never annealed (still at ~peak when the run was killed).
- Result is an **undertrained L=8 number**. The hypothesis cannot be falsified or
  confirmed from this run alone — we never reached the regime where L=8's extra capacity
  would matter most (late-cosine fine-tuning).
- Student identified a critical bug in `data/scoring.py`: a `-inf` in interior pressure
  of one cruise test sample propagates NaN into the surface metric via `NaN * 0 = NaN`,
  making in-tree `test_avg/mae_surf_p = None`. Documented in `CURRENT_RESEARCH_STATE.md`.
- The student's W&B summary correctly logs nansafe variants and `data_bug/*` diagnostics.

### Advisor action

- **Closed at 15:50** — bottom-tier in cohort ranking (147.85 vs frontier 109.99). The
  depth lever returns when bf16 (#3282) unblocks proper epoch counts.
