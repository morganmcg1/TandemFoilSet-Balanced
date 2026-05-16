# SENPAI Research State

- **Last updated:** 2026-05-16 ~08:00 UTC
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

## Active in-flight PRs (status as of 08:00 UTC)

| # | Student | Hypothesis | State | val_avg/mae_surf_p |
|---|---|---|---|---|
| **#3632** | tanjiro | Coord noise augmentation std=0.01 | **MERGED** 04:30 → baseline | 83.495 🏆 |
| **#3716** | fern | n_head=8 (attention diversity) | CLOSED (val=93.17) | — |
| **#3715** | askeladd | mlp_ratio=4 (FFN capacity) | **CLOSED 08:00** (val=93.17, +9.68) | — |
| **#3692** | tanjiro | Feature condition noise aug cols 2:24 | **CLOSED 08:00** (val=85.98, +2.48) | — |
| **#3690** | edward | lr=1e-3 + coord noise | stale_wip — W&B FAIL (`96tusrhs` 86.32 / `x0icixhu` 87.54); advisor commented | close on submission |
| **#3691** | thorfinn | --epochs 12 longer training | stale_wip — W&B `zqxkh9np` val=**82.50**, test=**74.10**; val wins −1.2%, test +0.4% | merge candidate (val), test caveat |
| **#3714** | alphonse | surf_weight=15 sweep | WIP — W&B FAIL (`j8rnxpc4` 88.23 / `84azuean` 88.27); advisor commented | close on submission |
| **#3717** | frieren | coord_noise_std sweep (0.03, 0.005) | WIP — W&B FAIL (std=0.03 `mynslale` 86.29; std=0.005 arm not launched); advisor asked for std=0.005 | awaiting |
| **#3718** | nezuko | AoA jitter augmentation | WIP — W&B FAIL (`4h64yzzl` std=0.02 84.96 — marginal); advisor commented | close on submission |
| **#3741** | fern | eta_min=1e-5 cosine floor | WIP (no run started yet) | awaiting |
| **#3814** | askeladd | **SwiGLU FFN (round-5)** | WIP (assigned 08:00) | awaiting |
| **#3815** | tanjiro | **TTA coord noise K=4/K=8 (round-5)** | WIP (assigned 08:00) | awaiting |

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

## Round-4 active hypotheses (8 PRs, all running or polling)

### Round-4a (assigned 04:40 UTC, runs started 05:22)
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #3690 | edward | lr=1e-3 + coord noise (single flag, zero code) | −2–5% (lr compound) |
| #3691 | thorfinn | --epochs 12 longer training (zero code) | −1–3% (free gradient steps) |
| #3692 | tanjiro | Feature noise aug on condition cols 2:24 | −1–3% (OOD splits) |

### Round-4b (assigned 05:28-30 UTC, after round-3 closes)
| PR | Student | Hypothesis | Expected gain | Status |
|---|---|---|---|---|
| #3714 | alphonse | surf_weight=15 sweep (single flag) | −1–3% (direct metric alignment) | running |
| #3715 | askeladd | mlp_ratio=4 (FFN capacity, untested arch lever) | −2–5% (if FFN-bottlenecked) | running |
| #3716 | fern | n_head=8 (attention head diversity) | −1–3% (speculative) | CLOSED (val=93.17, head_dim=20 too narrow) |
| #3717 | frieren | coord_noise_std sweep (0.03, 0.005) | −0–2% (pin augmentation optimum) | running |
| #3718 | nezuko | AoA jitter augmentation std=0.02 | −1–3% (OOD splits) | running |

### Round-4c (assigned 05:55 UTC, after #3716 close)
| PR | Student | Hypothesis | Expected gain | Status |
|---|---|---|---|---|
| #3741 | fern | eta_min=1e-5 cosine LR floor | −0.5–2% (meaningful gradients in last epoch) | running |

## Round-5 (assigned 08:00 UTC — plateau-protocol tier change)

7 of 8 round-4 experiments failed to beat baseline (only thorfinn #3691 epochs=12 won at W&B val≈82.50). Per plateau protocol, round-5 moves up a tier from incremental tuning to architecture/inference/loss reformulation. Hypotheses ranked on public-literature merit + on-task fit only (see `RESEARCH_IDEAS_2026-05-16_07:25.md`).

| PR | Student | Hypothesis | Expected gain | Risk |
|---|---|---|---|---|
| #3814 | askeladd | **SwiGLU FFN (param-matched)** | −1–4% val (exploratory; LM-literature signal) | MED |
| #3815 | tanjiro | **TTA coord noise K=4/K=8** | −1–4% val (inference-only; OOD splits) | LOW |

### Round-5 backlog (top 4 unassigned, ordered by priority)
1. **onecycle-lr** — schedule change; super-convergence for short-budget training; pairs with lr=5e-4
2. **asinh-output-norm** — target transform for heavy-tailed y (per-sample std 164→2077)
3. **dsdf-clip** — 2-line input regularization (clip dims 4-11 to ±3σ)
4. **per-domain-output-norm** — separate y_stats per (single, racecar, cruise) domain

## Round-4a / Round-4b closeout (W&B-verified, awaiting student submissions)

All 7 round-4 follow-ups regressed vs baseline 83.4954. The single near-miss (nezuko AoA std=0.02 at val=84.96, +1.47) reaffirms that augmentation on flow-condition scalars is at best neutral. Augmentation gains in this launch are concentrated on coord noise (#3632); architecture scaling (FFN width, attention heads, depth, hidden width) is exhausted at the 30-min budget.

The only winner is thorfinn #3691 (epochs=12 / slower cosine), at W&B val≈82.50, test≈74.10 — pending terminal SENPAI-RESULT submission to merge.

## Round-4 W&B verdicts (07:25 UTC; awaiting student SENPAI-RESULT submissions)

| PR | Student | Best run | val_avg | test_avg | Δ val vs 83.50 | Verdict |
|---|---|---|---|---|---|---|
| #3691 | thorfinn | `zqxkh9np` | 82.500 | 74.102 | −1.2% | **val WIN, test +0.4% — review** |
| #3718 | nezuko | `4h64yzzl` | 84.962 | 75.557 | +1.8% | close (marginal) |
| #3692 | tanjiro | `xu5e6cul` | 85.980 | 75.300 | +3.0% | close |
| #3717 | frieren | `mynslale` | 86.289 | 77.073 | +3.3% | close |
| #3690 | edward | `96tusrhs` | 86.319 | 75.853 | +3.4% | close |
| #3714 | alphonse | `j8rnxpc4` | 88.227 | 77.436 | +5.7% | close |
| #3692 | tanjiro | `yg32qo3i` | 89.187 | 79.578 | +6.8% | (same PR alt arm) |
| #3715 | askeladd | `0ezsswb4` | 93.174 | 83.387 | +11.6% | close (large regression) |
| #3741 | fern | (none yet) | — | — | — | run not yet started |

**Plateau signal:** 7 of 8 round-4 experiments outright failed; thorfinn's val win comes with a test regression. The model's training-time-limited regime + augmentation saturation hypothesis is confirmed — incremental architecture/loss/aug tweaks are exhausted. Time to escalate strategy tier per Plateau Protocol.

## Potential next research directions (round 5+, ESCALATED per Plateau Protocol)

Round 4 had 7 failures and 1 marginal val-only win. The incremental neighborhood is exhausted. The next round must use **bigger swings** — change tier of the strategy rather than tune within the current one.

### Tier change candidates (high upside, higher risk)
1. **bf16 mixed-precision** — 2× throughput. At constant 30-min budget, ~20 epochs instead of 10. Combined with cosine schedule co-design (T_max=20), this could unlock a different regime entirely. Currently the model is training-time-limited; this is the single largest training-side intervention not yet tried.
2. **Physics-informed loss term** — divergence-free penalty (∇·u=0) on velocity field, OR mass conservation constraint. Adds an inductive bias the L1 pointwise loss cannot impose. High complexity but high upside.
3. **Camber symmetry augmentation** — reflect (x→x, z→−z) along chord and flip AoA sign. Doubles effective training data along a symmetry the dataset definitely respects. Combined with coord noise, could unlock substantial OOD generalization on geom splits.
4. **Multi-scale Fourier PE** — frequencies span multiple decades (e.g., 4 log-spaced from 1 to 32) rather than current narrow band. Could let early layers attend to far-field while preserving boundary-layer detail.
5. **Slice-token-aware attention** — re-examine the slice mechanism. Currently slice_num=64 is hardcoded. Try slice_num=32 (faster) or learnable slice positions (more expressive). The slice mechanism is the architectural distinguisher of Transolver and least explored.
6. **Test-time augmentation** — average predictions over coord-noise variants at inference. Free win at val/test if it works.
7. **Sliding-window training schedule** — train epochs 0–5 at full resolution, then fine-tune epochs 6–10 with reduced lr and no augmentation. Decouples augmentation regime from final convergence.

### Single-flag confirmatory experiments (low risk, complementary)
8. **--epochs 12 retry** — confirm or refute thorfinn's val-only win. If real, merge despite test-side noise.
9. **slice_num=32** — half the slice count; never tested.
10. **gradient accumulation batch_size=8 effective** — increase effective batch without VRAM cost.

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
