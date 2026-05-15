# SENPAI Research State

- **Date:** 2026-05-15 17:25
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3200 (fern) — Fourier position features (8 bands)** — merged 2026-05-15 17:22

- `val_avg/mae_surf_p` = **121.4956**
- `test_avg/mae_surf_p` = **112.4884**
- W&B run: `t1ai7kzf`
- The `evaluate_split` NaN fix is bundled — all subsequent PRs inherit finite W&B test_avg

Per-split surface-p (val | test): single=139.80|122.01, camber_rc=138.71|133.37, camber_cruise=93.55|83.11, re_rand=113.93|111.46.

## Round 1 summary

| PR | Student | Hypothesis | Status | val_avg |
|---|---|---|---|---|
| #3191 | alphonse | Per-sample scale-normalizing loss | **closed** | 148.51 (worse) |
| #3194 | askeladd | 5-epoch LR warmup + cosine | **rebase requested** (re-run on new baseline) | 136.55 (pre-rebase) |
| #3198 | edward | Per-channel surface loss weights | WIP (stale — no results yet, pod up) | — |
| #3200 | fern | Fourier position features (8 bands) | **MERGED → new baseline** | **121.50** |
| #3206 | frieren | Capacity scale-up n_hidden=256 | **closed** (OOM @ bs=4) | 160.29 |
| #3207 | nezuko | PGOT-style geom-conditioned slice | WIP (re-running with NaN fix) | 128.34 (pre-rebase) |
| #3215 | tanjiro | SmoothL1 (Huber) loss | WIP (stale — no results yet, pod up) | — |
| #3218 | thorfinn | DropPath stochastic depth | **closed** | 128.90 (worse) |

**Key Round-1 learnings:**

1. Fourier features were the biggest single lever (input augmentation > architecture changes at this budget).
2. All non-trivial runs hit the 30-min wall clock with val still descending — optimizer/schedule that get more out of the fixed budget are high-leverage (askeladd's warmup re-run is in this lane).
3. `test_geom_camber_cruise` is consistently the *easiest* split (~83-110) and `val_single_in_dist` consistently the *hardest* (~140-160). Counterintuitive — likely high-Re extreme samples in the in-distribution holdout dominate.
4. Regularization (DropPath) helps OOD splits but hurts in-dist — *directional* regularizers (physics-informed) may avoid the trade-off; that's the thorfinn re-assignment.
5. Naive capacity scale-up at bs=4 OOMs on 96 GB. Memory-conscious capacity bumps (slice_num=96 + gradient checkpointing — frieren re-assignment) are the right way to test capacity.

## Round 2 assignments (idle students reassigned)

| PR | Student | Hypothesis | Angle | Status |
|---|---|---|---|---|
| #3350 | alphonse | FiLM-style Reynolds conditioning on each Transolver block | arch/conditioning | WIP |
| #3352 | fern | Learnable Fourier frequency bands (8 trainable freqs) | features | WIP |
| #3353 | frieren | `slice_num=96` with gradient checkpointing (memory-safe) | arch | WIP |
| #3356 | thorfinn | Divergence-free velocity auxiliary loss (physics-informed) | loss/physics | WIP |

Full hypothesis details + code snippets in `research/RESEARCH_IDEAS_2026-05-15_16:35.md`.

## Round 1 carry-overs still WIP

- **PR #3194 (askeladd, warmup-cosine):** sent rebase note after fern merged. Re-run two arms (`warmup=0`, `warmup=3`) on the new baseline. Targets val_avg < 121.50.
- **PR #3207 (nezuko, geom-conditioned slice):** re-running with NaN fix. Targets val_avg < 121.50 (their pre-rebase value of 128.34 already beat warmup=3, so this is a real candidate).
- **PR #3215 (tanjiro, SmoothL1):** stale WIP, pod up. No reported terminal results yet.
- **PR #3198 (edward, channel weights):** stale WIP, pod up. No reported terminal results yet.

If #3215 or #3198 produce results that beat the new baseline, merge sequentially. If they fail, close and reassign in Round 3.

## Potential next research directions (post-Round 2)

If Round 2 surfaces winners, Round 3 will compound. Anticipated follow-ups depending on which lever moves:

- **If FiLM (alphonse) wins:** extend to FiLM on the slice projection (not just LN), or condition on geometry features (NACA M/P/T) in addition to Re.
- **If learnable Fourier (fern) wins:** look at learned frequencies — if they collapse to a narrow band, try a different parameterization (multi-scale Gabor); if they spread, sweep `N_FOURIER_BANDS` up.
- **If slice_num=96 + checkpoint (frieren) wins:** try `slice_num=128` (with checkpointing always on), or extend to `n_layers=8` paired with the memory plan that worked.
- **If divergence-free (thorfinn) wins:** add other physics constraints (no-slip at surface nodes, far-field BC), or upgrade the finite-difference approximation to use a learned local-neighborhood kernel.
- **Carry-over (askeladd warmup):** if warmup beats no-warmup on the new baseline, stack with OneCycleLR + gradient clipping (ideas file #1).
- **Carry-over (nezuko geom-slice):** if it wins post-rebase, extend to per-block geometry conditioning.

**Plateau response (5+ failed experiments in a row):** move to a different architecture entirely — GNN over the mesh, Galerkin transformer, spectral-conv hybrid. Reconsider the data normalization (per-sample relative scoring vs global stats).
