# SENPAI Research State

- **Last updated:** 2026-05-16 ~00:35 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 96.0997, test_avg/mae_surf_p = 85.5256** — from PR #3507 (alphonse, n_hidden=160), merged 2026-05-16 ~00:30 UTC. See `BASELINE.md` for full details.

Per-split test: single_in_dist=103.75, geom_camber_rc=92.42, geom_camber_cruise=61.38, re_rand=84.55.

Previous baseline: val=100.5275, test=90.1489 (PR #3089 alphonse L1 loss).

## Width scaling curve so far

| n_hidden | params | val_avg | test_avg | Δ val vs 128 | PR |
|---|---|---|---|---|---|
| 128 | 662k | 100.527 | 90.149 | baseline | #3089 (MERGED) |
| 160 | 1.03M | 96.100 | 85.526 | −4.4% | #3507 (MERGED) |
| 192 | ~1.49M | — | — | in-flight | #3552 (alphonse) |

Val curve still descending at epoch 10 for both merged widths → continued width scaling expected to help.

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **L1 loss is the default** (Config.loss_type = "l1"). Advisor branch uses L1 everywhere.
2. **n_hidden=160 is the new default** (Config.n_hidden = 160). All branches rebase to get this.
3. **Scoring NaN fixed** (torch.isfinite per-sample mask in train.py's evaluate_split).
4. **Use `--epochs 10`** (or `--epochs 8` for very wide models > 185s/epoch) so cosine fully anneals.
5. **Grad clip max_norm=1.0**, warmup 2 epochs, lr=1e-3, batch=4.

## Active in-flight PRs

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3372** | **askeladd** | **Fourier PE 4-freq (rebased)** | wip (sent back 23:24) | expected ≤94.49 + finite test |
| #3469 | tanjiro | Depth n_layers=5→6 | wip (training at 00:27+, GPU 100%) | awaiting results |
| #3479 | frieren | Per-channel heads (rebased) | wip (sent back 00:32) | needs rebase onto n_hidden=160 + L1 |
| #3288 | edward | Scoring fix + lr default | wip (superseded; may close) | verify run val=115.76 (regressed) |
| #3490 | nezuko | L1 LR sweep {3e-4, 2e-3, 4e-3} | wip (training since 00:20+) | awaiting results |
| **#3552** | **alphonse** | **Width n_hidden=192** | wip (assigned 00:33) | awaiting results |
| #3508 | fern | Cosine warm restarts (SGDR T_0=4) | wip (training since 00:20+) | awaiting results |
| #3524 | thorfinn | Huber loss β=1.0 | wip (training since 00:20+) | awaiting results |

All 8 students have active WIP PRs.

## Merged wins (cumulative)

| PR | Description | val_avg | test_avg |
|---|---|---|---|
| **#3507** | **Width n_hidden=160 (alphonse)** | **96.0997** ← current | **85.5256** |
| #3089 | L1 loss + scoring fix (alphonse) | 100.5275 | 90.1489 |
| #3091 | LR warmup + clip + lr=1e-3 (edward) | 109.42 | NaN |

## Closed experiments

| PR | Hypothesis | val | Reason |
|---|---|---|---|
| #3095 | nezuko surf_weight=20 | 111.92 | +11.3% vs new baseline |
| #3371 | thorfinn EMA-9999 | 106.22 | +5.7% vs new baseline; training not converged enough |
| #3414 | tanjiro SWA | swa=109.48 | EMA/SWA premature at 10 epochs |
| #3093 | frieren bf16+bs=8 | 120.43 | MSE+bf16+clip incompatible schedule |
| #3092 | fern slice_num=128 | 106.82 | No win vs baseline |

## Pending merge candidates (when SENPAI-RESULT lands)

1. **#3372 askeladd Fourier PE 4freq (rebased + L1)** — expected val < 94.49; should stack with L1 for new record.
2. **#3552 alphonse width-192** — if val < 96.10.
3. **#3469 tanjiro depth n_layers=6** — if val < 96.10.
4. **#3490 nezuko LR sweep** — if any arm val < 96.10.
5. **#3508 fern warm restarts** — if val < 96.10.
6. **#3524 thorfinn Huber** — if val < 96.10.
7. **#3479 frieren per-channel heads (rebased)** — if val < 96.10 after rebase.

## Potential next research directions (round 6+)

1. **Stack width=192 + Fourier PE** — largest two independent gains; compose after both PRs land.
2. **Depth+width composition** (n_layers=6 + n_hidden=192) — once tanjiro's depth result is in.
3. **n_hidden=224** — continue width curve if 192 still has headroom and VRAM allows.
4. **bf16 retry at n_hidden=160** — correct bf16 schedule co-design (lower LR or smaller warmup); 2× epochs in budget.
5. **Physics-aware auxiliary loss** — divergence-free penalty for incompressible flow.
6. **Data augmentation** — foil reflection / AoA perturbation; doubles effective training set.
7. **Learnable Fourier frequencies** — askeladd's suggested follow-up from #3372.
8. **Longer training (15–20 epochs)** — val curves are still descending at epoch 10; more epochs = more gains at cost of multiple 30-min slots or a single overnight run.

## Cross-cutting observations

- **Width scaling is the primary driver** so far: 128→160→(192 in-flight) gives compounding improvements.
- **L1 loss gave 8.1% gain** on top of the base optimizer — now canonical.
- **Fourier PE is the highest single-arm result** (val=94.49 on pre-#3089 code); rebase expected to push even lower with L1.
- **EMA/SWA don't help** with 10-epoch budget; training still descending.
- **Per-channel heads** gave a val=106.65 result on stale code; hypothesis is promising but needs clean re-eval on n_hidden=160 baseline.
