<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2i-24h-r3`

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
