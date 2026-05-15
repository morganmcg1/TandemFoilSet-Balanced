# SENPAI Research State

- **Updated:** 2026-05-15 20:35
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 open for `test_geom_camber_cruise` NaN bug.

## Current baseline

**`val_avg/mae_surf_p = 98.88`** — CosineAnnealingWarmRestarts(T_0=5, T_mult=2), PR #3320, **merged** (2026-05-15 20:25)

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 116.36 |
| val_geom_camber_rc | 108.40 |
| val_geom_camber_cruise | 77.91 |
| val_re_rand | 92.87 |
| **val_avg** | **98.88** |

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | **98.88** | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 3 portfolio (active experiments)

| Student | PR | Change | Rationale |
|---|---|---|---|
| nezuko | #3431 | EMA weights (decay=0.999) | Smooth out warm-restart oscillations; may improve best checkpoint quality |
| alphonse | #3436 | CosineAnnealingWarmRestarts T_0=3 | More frequent restarts (2→3 cycles in 14-epoch budget); direct T_0 sweep |
| edward | #3434 | L1 surface loss (vs MSE) | Align training objective with MAE metric; L1 minimizer = L1 metric |
| thorfinn | #3416 | Per-channel surf loss: p×3 | Directly weight primary metric channel harder vs Ux/Uy |

## Round 2 WIP still completing (multi-arm, stale labels)

| Student | PR | Change | Status |
|---|---|---|---|
| tanjiro | #3360 | grad clip max_norm=0.5 | Stale WIP — ran training (GPU evidence), no comment yet; nudged with new-baseline note |
| askeladd | #3307 | OneCycleLR right-sized | WIP draft — schedule-sizing fix pending |
| fern | #3139 | surf_weight=25 | WIP multi-arm — all arms regressing (best 141.69), likely closeable |
| frieren | #3146 | slice_num=128 | WIP — single arm at 136.57, regression |

## Round 2 final results (closed without merge)

| PR | Change | val_avg | Δ | Decision |
|---|---|---|---|---|
| #3381 edward | n_hidden=192 | 126.44 | +7.9% | closed — width not bottleneck (per-epoch identical) |
| #3112 alphonse | bf16 autocast | 114.34 (best), ~122 (mean) | mean +4.3% | closed — speed benefit, accuracy neutral-to-negative |
| #3310 edward | n_layers=6 | 127.23 | +8.6% | closed — depth costs more epochs than it gains |
| #3308 thorfinn | AdamW beta2=0.95 | 134.89 | +15.1% | closed — grad noise grows, not shrinks |
| #3306 tanjiro | grad clip max_norm=100 | 124.31 | +6.1% | closed — confirms tight clip = gradient normalizer |
| #3153 nezuko | Huber vol loss | 127.22 (confounded) | +8.6% | closed — missing grad clip, high variance |
| #3164 thorfinn | dropout=0.05 | 142.51 | +21.7% | closed — no overfit in this budget |
| #3133 edward | n_layers=7 | 146.62 | +25.1% | closed — unstable without clip |
| #3125 askeladd | lr=1e-3 + warmup + cosine | 135.06 | +15.3% | closed — cosine horizon mismatch |

## Key research findings

1. **Warm restarts is the dominant lever** — T_0=5 T_mult=2 gives 15.6% improvement, replicated 3× with <3 pp variance. Mechanism: multiple escape-from-local-minima within 14-epoch budget.
2. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; loosening regresses. Testing max_norm=0.5 (tanjiro #3360).
3. **Model is NOT capacity-limited** — width (n_hidden=192) and depth (n_layers=6,7) both fail in this budget because they slow per-epoch time without proportional quality gain.
4. **Second-moment adaptation** — beta2=0.95 fails (more gradient noise); baseline beta2=0.999 is near-optimal for this regime.
5. **bf16 is speed-neutral** — 29% more epochs in 30 min but accuracy neutral on average; not yet worth merging.
6. **High run variance** — ~15-pp spread in ≤14-epoch runs; warm restarts reduced this (3-pp spread). EMA may reduce further.

## Potential next research directions (round 4+)

1. **Compound winners** — if round-3 experiments win, stack them (EMA + per-channel p-weighting, or T_0=3 + L1 surf)
2. **Higher LR on warm-restarts baseline** — askeladd's OneCycleLR with 1e-3 peak is in flight; could also test lr=1e-3 with existing warm restarts
3. **FiLM conditioning** — global features (Re, AoA, NACA) broadcast through all per-node MLPs; FiLM encode once and modulate cheaply
4. **bf16 + warm restarts** — if VRAM becomes limiting or we want more epochs; worth revisiting once model is bigger
5. **Gradient direction exploration** — if max_norm=0.5 wins further: try Lion/SignSGD (pure gradient direction)
6. **Data efficiency** — is the ~1500-sample training set limiting? Curriculum or importance sampling
