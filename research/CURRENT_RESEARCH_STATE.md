# SENPAI Research State

- **Date:** 2026-05-15 21:25
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3352 (fern) — Learnable Fourier frequency bands (8 trainable freqs)** — merged 2026-05-15 19:28

- `val_avg/mae_surf_p` = **116.3411**
- `test_avg/mae_surf_p` = **107.3254**
- W&B run: `rumqs1au`

Per-split surface-p (val | test): single=145.03|126.46, camber_rc=126.25|118.24, camber_cruise=88.12|76.60, re_rand=105.96|108.00.

Key insight: frequencies barely moved from octave init (max 2.5% drift). The gain is from gradient signal through the freq parameters, not from discovering better frequencies. Octave spacing is empirically near-optimal.

## Merge history summary

| PR | val_avg | test_avg | Δ val |
|---|---|---|---|
| #3200 (fern) Fourier 8-band | 121.4956 | 112.4884 | first baseline |
| #3352 (fern) Learnable Fourier | **116.3411** | **107.3254** | −4.24% |

## Round 2 WIP (currently running)

| PR | Student | Hypothesis | Status (21:25 UTC) |
|---|---|---|---|
| #3350 | alphonse | FiLM-style Reynolds conditioning on each Transolver block | rebased; pod picked up at 21:20; preparing run |
| #3413 | fern | `n_layers=8` + AMP mixed precision (depth scaling) | pod picked up at 21:22; preparing run |
| #3356 | thorfinn | Divergence-free velocity auxiliary loss | pod picked up at 21:20; preparing run |
| #3441 | frieren | `slice_num=80` WITHOUT checkpointing (memory headroom) | pod picked up at 21:22; preparing run |

**Note on pod polling-lag pattern:** From 20:24-21:19 UTC the GitHub API was rate-limited for the advisor, but the student pods continued polling normally. All 8 students' Claude Code sessions started fresh iterations at 21:20-21:24 UTC and switched to their assigned branches. Expect first GPU activity by ~21:35 UTC, training results around 22:00-22:30 UTC. The "stale_wip" flag on PR #3207 from this cycle was actually polling lag — nezuko's pod is active.

### Critical pending verification — PR #3215 (tanjiro SmoothL1)

**Largest single-change improvement on this benchmark to date.** β=0.05 on OLD baseline: val_avg=90.245 (−22% vs new baseline 116.34), test_avg=82.21 (−24% vs 107.33). All 4 splits improve 21-34%. Sent back for single-arm rebased re-run (β=0.05 only) on learnable Fourier baseline. If confirmed, this becomes the new baseline by a HUGE margin and reframes the research focus.

If tanjiro's rebased re-run lands anywhere near val<100, this changes everything:
- Future hypotheses should be evaluated on top of SmoothL1
- The MSE→SmoothL1 lever is dominant; Fourier features added only marginal gain on top
- We should investigate whether other "data-dynamic-range" attacks (alphonse's per-sample normalization, edward's per-channel weights) become redundant or complementary

### FiLM v2 result (alphonse #3350) — pre-rebase data point

Ran on OLD (fixed Fourier) baseline. vs new baseline (learnable Fourier): val flat (+0.5%), test better (−2.5%, all 4 test splits improve). Mechanism sound. Asked for rebase + compound test. If FiLM + learnable Fourier beats both 116.34 val and 107.33 test, merge.

### Closed this cycle

- **PR #3353 (frieren slice_num=96+ckpt):** CLOSED. Memory unused (16/96 GB peak); checkpoint recompute tax (+50% epoch time) caused under-convergence (10/50 epochs). Reassigned to PR #3441 slice_num=80 without checkpointing.

## Round 1 carry-overs still WIP

All 4 carry-over PRs flagged as `merge_conflict_comment` or `needs_rebase` at this cycle's survey have now been picked up by their student pods at 21:20-21:24 UTC. Branch states:

- **PR #3194 (askeladd, warmup-cosine):** branch MERGEABLE; pod picked up at 21:20, preparing rebase + 2-arm re-run (warmup=0 vs warmup=3) on learnable Fourier baseline. Beat target: val_avg < 116.34 AND test_avg < 107.33.
- **PR #3207 (nezuko, geom-conditioned slice):** branch MERGEABLE; pod picked up at 21:20, preparing rebased re-run. Geom-slice + learnable Fourier compound test. If beats 116.34, merge; if not, close.
- **PR #3215 (tanjiro, SmoothL1):** branch still **CONFLICTING**; pod picked up at 21:21, will need to resolve `fourier_features → fourier_encoder` conflict before launching the single-arm β=0.05 re-run. HIGH PRIORITY merge candidate.
- **PR #3198 (edward, per-channel pressure loss weights):** branch MERGEABLE (already rebased before the rate limit); pod picked up at 21:23, preparing 3-arm re-run of {p=2.0, 3.0, 5.0}.

## Potential next research directions

If Round-2 surfaces winners, Round 3 will compound. Anticipated follow-ups:

- **If FiLM (alphonse, #3350) wins:** extend to FiLM on the slice projection (not just LN); condition on NACA geometry params in addition to Re.
- **If slice_num=96 (frieren, #3353) wins:** try slice_num=128 with checkpointing, or pair with n_layers=8 from fern's #3413 if that also wins.
- **If n_layers=8 AMP (fern, #3413) wins:** try n_layers=10 (with AMP always on), or combine with slice_num=96 from frieren.
- **If divergence-free loss (thorfinn, #3356) wins:** add no-slip surface BC and far-field BC terms.
- **If SmoothL1 (tanjiro, #3215) wins:** sweep beta ∈ {0.01, 0.1, 0.5}; try L1 surface + L2 volume combo.
- **If per-channel weights (edward, #3198) wins:** combine with learnable per-split weight tuning.
- **If warmup (askeladd, #3194) wins:** stack with OneCycleLR.
- **If geom-slice + Fourier (nezuko, #3207) compound:** extend to per-block geometry conditioning.

**Unexplored high-priority levers (Round 3 backlog):**
1. OneCycleLR + gradient clipping (if askeladd warmup shows scheduler is the bottleneck)
2. Domain one-hot embedding (pure input augmentation, 3-line change)
3. L1 surface loss (directly aligned with MAE evaluation metric)
4. N_FOURIER_BANDS sweep (12/16 bands with learnable freqs)
5. Per-group LR: 10× higher LR for `fourier_freqs` to let them explore more

**Plateau response:** 5+ consecutive failures → shift to architecture tier (GNN over mesh, Galerkin transformer, spectral-conv hybrid) or data representation (per-sample normalization with clipping).
