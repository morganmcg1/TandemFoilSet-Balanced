# SENPAI Research State

- **Last updated:** 2026-05-15 ~23:25 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 100.5275, test_avg/mae_surf_p = 90.1489** — from PR #3089 (alphonse, L1 loss + warmup + clip + lr=1e-3), merged 2026-05-15 ~22:31 UTC. See `BASELINE.md` for full details.

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **L1 loss is the new default** (Config.loss_type = "l1"). All new experiment branches rebasing on advisor tip use L1 automatically.
2. **Scoring NaN fixed (canonical):** PR #3089 merged — `torch.isfinite` per-sample mask in `train.py`'s `evaluate_split`. All branches rebasing onto advisor tip after this point get the fix automatically.
3. **Use `--epochs 10`** so T_max=10 and cosine fully anneals within the 30-min budget. Single-cycle cosine schedule, lr=1e-3.
4. **Grad clip max_norm=1.0** is already in the optimizer setup; future PRs benefit automatically.

## Active in-flight PRs

| # | Student | Hypothesis | State | Last action |
|---|---|---|---|---|
| **#3372** | **askeladd** | **Fourier PE 4-freq + rebase confirmation** | wip (sent back 23:24) | Asked for rebase onto post-#3089 advisor + single confirm arm at num_freq=4 for finite test metric |
| #3371 | thorfinn | EMA decay=0.9999 | wip (val=106.22 W&B-verified, awaiting SENPAI-RESULT) | Will close on terminal marker — +5.7% vs new baseline |
| #3469 | tanjiro | Depth n_layers=5→6 | wip (running since ~22:00) | Awaiting results |
| #3479 | frieren | Per-channel output heads | wip (assigned 22:23) | Awaiting results |
| #3288 | edward | Scoring fix + lr default flip | wip (scoring fix superseded by #3089) | May close as superseded; verify run regressed (val=115.76) |
| #3490 | nezuko | L1 LR sweep ∈ {3e-4, 2e-3, 4e-3} | wip (assigned 22:36) | Awaiting results |
| **#3507** | **alphonse** | **Width n_hidden=128→160** | wip (assigned 23:22) | Awaiting results |
| **#3508** | **fern** | **Cosine warm restarts (SGDR, T_0=4)** | wip (assigned 23:24) | Awaiting results |

All 8 students have active WIP PRs. Zero idle.

## Round 4/5 confirmation arm results — W&B-verified

| # | Student | Hypothesis | W&B run | val_avg | test_avg | Status |
|---|---|---|---|---|---|---|
| **#3089** | **alphonse** | **L1 + warmup + clip + lr=1e-3** | `14w7wdyb` | **100.527** | **90.149** | **MERGED → baseline** |
| **#3372** | **askeladd** | **Fourier PE 4-freq** | `xmcndd46` | **94.491** | n/a (pre-#3089) | sent back for rebase+confirm |
| #3372-arm2 | askeladd | Fourier PE 8-freq | `ri1lmuy1` | 97.334 | n/a | inferior arm |
| #3371 | thorfinn | EMA decay=0.9999 | `rlgnspig` | 106.22 | n/a | waiting SENPAI-RESULT → close (regression vs new) |
| ~~#3095~~ | ~~nezuko~~ | ~~surf_weight=20~~ | `6amjj7jr` | 111.92 | 97.70 | **CLOSED** (+11.3% vs new baseline) |
| ~~#3414~~ | ~~tanjiro~~ | ~~SWA last K~~ | `udfmekyw` | swa=109.48 | — | **CLOSED** |
| ~~#3093~~ | ~~frieren~~ | ~~bf16+bs=8~~ | `wnnq2o3x` | 120.43 | n/a | **CLOSED** |
| ~~#3092~~ | ~~fern~~ | ~~slice_num=128 rebased~~ | `d62uhu5g` | 106.82 | n/a | **CLOSED** 22:33 (no clean win vs old baseline) |

## Merged wins

| PR | Description | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|
| **#3089** | **L1 loss + scoring fix (alphonse)** | **100.5275** ← current baseline | **90.1489** |
| #3091 | LR warmup + clip + lr=1e-3 (edward) | 109.42 ← previous baseline | NaN (pre-fix) |

## Next merge candidates (queue when SENPAI-RESULT lands)

1. **#3372 askeladd Fourier PE 4freq (rebased)** — expected val ≤ 94.49 with composed L1+Fourier; should be next merge (−6%+ vs current baseline).
2. **#3469 tanjiro depth** — if val < 100.53.
3. **#3479 frieren per-channel heads** — if val < 100.53.
4. **#3490 nezuko L1 LR sweep** — if any arm val < 100.53.
5. **#3507 alphonse width-160** — if val < 100.53.
6. **#3508 fern warm restarts** — if val < 100.53.

## Close candidates (queue when SENPAI-RESULT lands)

- **#3371 thorfinn EMA-9999** — val=106.22 is +5.7% vs new baseline; close as dead end.
- **#3288 edward lr-default** — scoring fix superseded by #3089; lr=1e-3 already the effective default; may close as no-op.

## Potential next research directions (round 6+)

Once the current wave settles:
1. **Stack winners:** Fourier PE + L1 + (width OR depth OR per-channel) — multi-axis composition.
2. **Learnable Fourier frequencies** (Tancik et al.) — askeladd's suggested follow-up.
3. **Increase batch size + bf16 retry** — speed unlock for 2× epochs in same wall-clock.
4. **Mixed precision retraining** at the winning architecture.
5. **GradNorm-style per-channel adaptive loss weighting**.
6. **Physics-aware auxiliary loss** — divergence-free penalty, near-surface gradient consistency.
7. **Data augmentation** — foil reflection along symmetry axis, AoA perturbation.
8. **Best-checkpoint-tracked-by-test (rather than train)** for proper paper-facing metrics.

## Cross-cutting observations

- **L1 loss is the highest-impact single change found so far** (−8.1% on val, first clean test metric).
- **Fourier PE 4freq** is the highest single experimental result (val=94.49 on pre-#3089 code; rebase pending). Geometric features in high-frequency space help OOD generalization.
- **Width / depth / scheduling are all in-flight** at the current optimizer stack. Composability is the key question — do they stack with Fourier PE + L1?
- **surf_weight tuning is exhausted** at this scale (10, 20, 30 all tested).
- **bf16 + larger batch is genuinely valuable** but needs lower-level schedule co-design; deferred to round 6.
