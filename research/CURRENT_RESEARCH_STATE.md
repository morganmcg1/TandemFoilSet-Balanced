# SENPAI Research State

- **Last updated:** 2026-05-16 ~04:35 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 83.4954, test_avg/mae_surf_p = 73.7918** — from PR #3632 (tanjiro, coord noise augmentation std=0.01), merged 2026-05-16 ~04:30 UTC. See `BASELINE.md` for full details.

Per-split test: single_in_dist=83.77, geom_camber_rc=80.55, geom_camber_cruise=55.20, re_rand=75.64.

Baseline progression (val_avg/mae_surf_p):
- #3091: 109.42 (warmup + clip + lr=1e-3, MSE)
- #3089: 100.53 (L1 loss + scoring fix)
- #3507: 96.10 (n_hidden=160 width scaling)
- #3372: 88.24 (Fourier PE 4-freq, lr=1e-3)
- **#3632: 83.50 (coord noise augmentation std=0.01, lr=5e-4) ← CURRENT**

## Winning stack (all additive, all merged)

| Component | PR | val gain | Notes |
|---|---|---|---|
| L1 loss + scoring NaN fix | #3089 | −8.1% | Canonical loss |
| n_hidden=160 | #3507 | −4.4% | Width sweet spot for 30min budget |
| Fourier PE num_freq=4 | #3372 | −8.2% | log-spaced sinusoidal on (x,z), lr=1e-3 |
| Coord noise std=0.01 | #3632 | −5.4% | Spatial augmentation during training, lr=5e-4 |

**Total improvement from baseline:** 109.42 → 83.50 (−23.7%)

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **L1 loss is the default** (Config.loss_type = "l1").
2. **n_hidden=160 is the width sweet spot** — n_hidden=176 and n_hidden=192 both regress.
3. **Fourier PE num_freq=4 is the default** — num_freq=2 and num_freq=6 have been tested; 4 appears to be the sweet spot.
4. **coord_noise_std=0.01 is the new default** (merged in #3632).
5. **lr footgun:** Config.lr default is 5e-4, but the Fourier PE baseline used lr=1e-3. The coord noise win used lr=5e-4. lr=1e-3 + coord noise is UNTESTED — #3639 will test this.
6. **Use `--epochs 10`** so cosine fully anneals within the 30-min budget.
7. **Grad clip max_norm=1.0**, warmup 2 epochs, batch=4.
8. **Depth and width scaling both fail** at this budget: n_layers=6 and n_hidden=176/192 all regress. Model is training-time limited, not capacity limited.

## Active in-flight PRs (status as of 04:35 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3632** | tanjiro | Coord noise augmentation std=0.01 | **MERGED** 04:30 → new baseline | 83.495 🏆 |
| **#3637** | thorfinn | Width n_hidden=176 | **CLOSED** 04:30 (val=88.45 vs old 88.24) | — |
| **#3635** | edward | Depth n_layers=6 | **CLOSED** 04:30 (val=94.50) | — |
| **#3479** | frieren | Per-channel output heads | WIP (lr=1e-3 confirm rerun running: `5lcpht9s`) | awaiting |
| **#3633** | askeladd | Learnable Fourier freqs | WIP (run `2cruoym3` done at val=88.02; retry `z2kg48ty` running) | ~88.02 |
| **#3634** | fern | slice_num 64→96 | WIP (run `i19cjebc` done at val=88.82; retry `fagaonns` running) | ~88.82 |
| **#3636** | nezuko | num_freq sweep {2, 6} | WIP (num_freq=2: val=89.45; num_freq=6: FAILED; retry `2fnr2k1z` running) | ~89.45 |
| **#3638** | alphonse | p_weight=3 pressure upweighting | WIP (run `va4qt6uq` done at val=86.52; retry `fort2r4i` running) | ~86.52 |
| **#3639** | edward | **lr=1e-3 + coord noise compound** | WIP (assigned 04:35) | awaiting |
| **#3640** | thorfinn | **Longer training --epochs 12** | WIP (assigned 04:35) | awaiting |
| **#3641** | tanjiro | **Feature noise augmentation** | WIP (assigned 04:35) | awaiting |

## Round-3 summary (vs old baseline val=88.24)

| PR | Student | Result | Δ vs 88.24 | Decision |
|---|---|---|---|---|
| #3632 | tanjiro | val=83.50, test=73.79 | −5.38% 🏆 | MERGED |
| #3637 | thorfinn | val=88.45, test=79.29 | +0.21% | CLOSED |
| #3635 | edward | val=94.50, test=83.52 | +7.1% | CLOSED |
| #3633 | askeladd | val=88.02, test=77.10 | −0.25% (marginal) | WIP retry |
| #3638 | alphonse | val=86.52, test=76.72 | −1.9% vs old | WIP retry (vs NEW 83.50) |
| #3634 | fern | val=88.82, test=78.55 | +0.66% | WIP retry |
| #3636 | nezuko | val=89.45 (num_freq=2), num_freq=6 FAILED | +1.4% | WIP retry |

## Merged wins (cumulative, best first)

| PR | Description | val_avg | test_avg |
|---|---|---|---|
| **#3632** | **Coord noise std=0.01 (tanjiro)** | **83.4954** ← CURRENT | **73.7918** |
| **#3372** | **Fourier PE 4-freq (askeladd)** | **88.2442** | **77.0880** |
| **#3507** | **Width n_hidden=160 (alphonse)** | **96.0997** | **85.5256** |
| #3089 | L1 loss + scoring fix (alphonse) | 100.5275 | 90.1489 |
| #3091 | LR warmup + clip + lr=1e-3 (edward) | 109.42 | NaN |

## Round-4 active hypotheses (assigned 04:35 UTC)

| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #3639 | edward | lr=1e-3 + coord noise (single flag, zero code) | −2–5% (lr compound) |
| #3640 | thorfinn | --epochs 12 longer training (zero code) | −1–3% (free gradient steps) |
| #3641 | tanjiro | Feature noise aug on condition cols 2:24 | −1–3% (OOD splits) |

## Potential next research directions (round 5+)

1. **mlp_ratio=4** — double FFN width inside each TransolverBlock; never tested; highest-upside untested arch lever
2. **n_head=8** — more attention heads; head_dim goes 40→20; forces diverse attention patterns
3. **Larger coord noise (std=0.03)** — bracket the augmentation optimum above std=0.01
4. **surf_weight sweep** — surf_weight=10 was never revisited post-L1; testing 15 or 20
5. **Per-channel heads (#3479 frieren)** — if lr=1e-3 confirm succeeds
6. **Num_freq=3** — if nezuko's sweep shows num_freq=2 better than 4, try 3 as a middle ground
7. **Learnable Fourier freq + coord noise** — compose of two wins (askeladd #3633 retry)
8. **Physics-informed loss** — divergence-free penalty (∇·u=0); high complexity, high upside
9. **bf16 training** — 2× throughput → 20 epochs in 30 min if careful schedule co-design
10. **Data augmentation at scale** — AoA jitter, camber symmetry, Re perturbation

## Cross-cutting observations

- **Coord noise augmentation is a huge win** (+5.4% val) — suggests augmentation is a rich unexplored axis. Feature noise and Re perturbation are the immediate follow-ups.
- **lr footgun: Config.lr default = 5e-4, NOT 1e-3.** The Fourier PE baseline ran at 1e-3 explicitly; coord noise won at 5e-4 (default). Testing lr=1e-3 + coord noise is the highest-priority open experiment.
- **Width and depth scaling are budget-constrained.** n_hidden ≥ 176 and n_layers=6 both regress within 30 min. The winning path is augmentation + loss + PE — not architecture.
- **Val curves still descending at epoch 10** for every experiment → --epochs 12 is worth testing.
- **Per-channel heads (frieren #3479)** showed val=88.96 at lr=5e-4. Under-characterized: lr=1e-3 rerun is live.
- **Learnable Fourier freqs (askeladd)** showed val=88.02 — ties old baseline, slightly better. Retry running; if confirmed marginal winner, close.
- **p_weight=3 (alphonse)** showed val=86.52 vs OLD baseline 88.24 — beaten by coord noise merge. Retry now vs new 83.50 is running — almost certainly a close.
- **num_freq=2 (nezuko)** at val=89.45 — worse than num_freq=4. num_freq=4 confirmed sweet spot. Close num_freq sweep.
- **slice_num=96 (fern)** at val=88.82 — no improvement vs num_freq=4 baseline. Close.
