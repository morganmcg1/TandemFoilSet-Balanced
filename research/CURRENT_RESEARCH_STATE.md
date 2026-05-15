# SENPAI Research State

- **Date:** 2026-05-15 19:35
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

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3350 | alphonse | FiLM-style Reynolds conditioning on each Transolver block | WIP |
| #3353 | frieren | `slice_num=96` with gradient checkpointing | WIP |
| #3356 | thorfinn | Divergence-free velocity auxiliary loss | WIP |
| #3413 | fern | `n_layers=8` + AMP mixed precision (depth scaling) | WIP (just assigned) |

## Round 1 carry-overs still WIP

- **PR #3194 (askeladd, warmup-cosine):** rebased onto Fourier baseline. Running warmup=0 vs warmup=3 arms. Branch is MERGEABLE. Beat target: val_avg < 116.34. Wait for terminal result.
- **PR #3207 (nezuko, geom-conditioned slice):** sent back for rebase onto learnable Fourier baseline. Final iteration — geom-slice + learnable Fourier compound test. If beats 116.34, merge; if not, close.
- **PR #3215 (tanjiro, SmoothL1 beta=0.05):** started training at 18:39 UTC on the Fourier baseline. W&B run `638hd0v7`. Results expected soon. Second arm (beta=0.1) follows.
- **PR #3198 (edward, per-channel pressure loss weights):** 3 arms (p_surf_weight=2.0, 3.0, 5.0) running sequentially since 18:40 UTC. arm_p2 done, arm_p3/p5 in queue.

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
