# SENPAI Research State

- **Last updated:** 2026-05-16 ~02:35 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 88.2442, test_avg/mae_surf_p = 77.0880** — from PR #3372 (askeladd, Fourier PE 4-freq), merged 2026-05-16 ~02:25 UTC. See `BASELINE.md` for full details.

Per-split test: single_in_dist=87.88, geom_camber_rc=82.70, geom_camber_cruise=59.41, re_rand=78.36.

Baseline progression (val_avg/mae_surf_p):
- #3091: 109.42 (warmup + clip + lr=1e-3, MSE)
- #3089: 100.53 (L1 loss + scoring fix)
- #3507: 96.10 (n_hidden=160 width scaling)
- **#3372: 88.24 (Fourier PE 4-freq) ← CURRENT**

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

## Active in-flight PRs (status as of 02:35 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3372** | askeladd | Fourier PE 4-freq | **MERGED** 02:25 → new baseline | 88.244 🏆 |
| **#3479** | frieren | Per-channel heads compose test | **WIP** (sent back 02:28; rebase onto Fourier-PE advisor needed) | 95.60 (pre-compose) |
| #3469 | tanjiro | Depth n_layers=5→6 | **CLOSED** 01:38 (stale baseline) | — |
| #3288 | edward | Scoring fix verify | **CLOSED** 02:30 (superseded) | — |
| #3490 | nezuko | L1 LR sweep | **CLOSED** 02:30 (best 98.88) | — |
| #3508 | fern | Warm restarts SGDR | **CLOSED** 02:30 (best 100.79) | — |
| #3524 | thorfinn | Huber loss β=1.0 | **CLOSED** 02:30 (best 101.44) | — |
| #3552 | alphonse | Width n_hidden=192 | **CLOSED** 02:30 (best 102.73) | — |
| **#3633** | askeladd | **Learnable Fourier freqs** | WIP (assigned 02:32) | awaiting |
| **#3632** | tanjiro | **Coord noise augmentation** | WIP (assigned 02:32) | awaiting |
| **#3634** | fern | **slice_num 64→96** | WIP (assigned 02:32) | awaiting |
| **#3635** | edward | **Depth n_layers=6 on new stack** | WIP (assigned 02:32) | awaiting |
| **#3636** | nezuko | **num_freq sweep {2, 6}** | WIP (assigned 02:32) | awaiting |
| **#3637** | thorfinn | **Width n_hidden=176** | WIP (assigned 02:32) | awaiting |
| **#3638** | alphonse | **Pressure channel weight p_weight=3** | WIP (assigned 02:32) | awaiting |

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

## Merge resolution status (2026-05-16 02:30 UTC)

All round-2 PRs resolved:
- ✅ **MERGED #3372** (askeladd Fourier PE 4-freq) → new baseline val=88.24 / test=77.09
- ↩️ **SENT BACK #3479** (frieren per-channel heads) — needs rebase + compose test onto Fourier-PE advisor
- ❌ **CLOSED #3469** (tanjiro depth-6, stale code)
- ❌ **CLOSED #3490** (nezuko LR sweep, best 98.88)
- ❌ **CLOSED #3508** (fern warm restarts, best 100.79)
- ❌ **CLOSED #3524** (thorfinn Huber, best 101.44)
- ❌ **CLOSED #3552** (alphonse width-192, best 102.73)
- ❌ **CLOSED #3288** (edward scoring verify, superseded)

## Active round-3 hypotheses (assigned 02:32 UTC)

| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #3633 | askeladd | Learnable Fourier frequency scales | −1–3% (PE specialization) |
| #3632 | tanjiro | Coord noise augmentation (Gaussian jitter) | −1–3% (OOD robustness) |
| #3634 | fern | slice_num 64→96 | −1–4% (finer physics attention) |
| #3635 | edward | n_layers=6 on new stack (--epochs 8) | −1–5% (depth + Fourier PE) |
| #3636 | nezuko | num_freq sweep {2, 6} | brackets optimum, possible improvement |
| #3637 | thorfinn | n_hidden=176 (width sweet spot) | −1–2% if 160 was under-capacity |
| #3638 | alphonse | p_weight=3.0 pressure upweighting | −2–5% (direct metric alignment) |
| #3479 | frieren | Per-channel heads compose (rebase WIP) | ~−0.5% if composes |

## Potential next research directions (round 4+)

1. **Stack per-channel heads + Fourier PE** — if #3479 frieren compose succeeds
2. **Num_freq=3 or 5** — once nezuko's sweep pins the bracket
3. **Learnable freq + per-channel heads** — compose of two architecture wins
4. **Longer training** — val curves still descending at 10 epochs; test with `--epochs 12` or a two-run chained approach when budget allows
5. **bf16 training** — 2× throughput → 20 epochs in 30 min; needs careful schedule co-design
6. **Physics-informed loss** — divergence-free penalty (∇·u=0); deferred from this round
7. **Data augmentation at scale** — AoA jitter, camber symmetry, Re perturbation
8. **Asymmetric head capacity** — wider/deeper p-head vs Ux/Uy (frieren's follow-up)
9. **n_head increase** (4→8) — more attention heads at n_hidden=160; head_dim=20 might be too narrow
10. **MLP ratio increase** (2→4) — larger feed-forward in TransolverBlocks

## Cross-cutting observations

- **Width-160 + L1 + Fourier PE composes spectacularly** — the +8.1% L1 gain (#3089), +4.4% width gain (#3507), and +8.2% Fourier PE gain (#3372 unmerged) are all roughly orthogonal levers that have STACKED cleanly to bring val from 109.42 → 88.24 (a 19.4% improvement in 3 merges).
- **L1 loss gave 8.1% gain** on top of the base optimizer — now canonical.
- **Width scaling has a sweet spot near 160.** Width-192 at --epochs 8 regressed to val=102.73 (over-parameterized for budget); the 168s/epoch at width-160 is the right wall-clock vs capacity tradeoff at the 30-min budget.
- **Per-channel heads compose with width** but the marginal gain shrinks (val −2.5% at width-128 → val −0.5% at width-160) — width absorbed most of the per-channel upside.
- **Huber β=1.0 worse than L1** at this scale (104.43 vs 96.10). L1 stays canonical.
- **Cosine warm restarts (SGDR) worse than vanilla cosine** at this budget (100.79 vs 96.10). Cyclic LR resets restart from too-high LR for the under-converged solver state.
- **Higher LR (2e-3) worse than 1e-3** for L1 (98.88 vs 96.10). 1e-3 + warmup is well-tuned.
- **EMA/SWA don't help** with 10-epoch budget; training still descending.
- **Val curves still descending at epoch 10** for every experiment → longer training is an unexplored axis.
