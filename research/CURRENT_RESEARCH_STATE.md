# SENPAI Research State

- **Date:** 2026-05-15 20:27
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
| #3350 | alphonse | FiLM-style Reynolds conditioning on each Transolver block | re-run on new baseline (rebase requested 20:25) |
| #3353 | frieren | `slice_num=96` with gradient checkpointing | WIP (pod picked up at it.72, auto-rebasing) |
| #3356 | thorfinn | Divergence-free velocity auxiliary loss | WIP |
| #3413 | fern | `n_layers=8` + AMP mixed precision (depth scaling) | WIP (just assigned 19:35) |

### FiLM v2 result (alphonse #3350) — pre-rebase data point

Ran on OLD (fixed Fourier) baseline so not apples-to-apples vs current head:
- val_avg = 116.96 (vs new baseline 116.34 → +0.5%, flat within noise)
- test_avg = 104.64 (vs new baseline 107.33 → **−2.5%**, all 4 test splits improve)

FiLM mechanism is sound (student fixed two impl bugs: zero-init preserved after self.apply; row-0 read instead of mean-over-nodes to avoid padding contamination of log(Re) signal). Asking for rebase + compound test on learnable Fourier. If FiLM + learnable Fourier beats both 116.34 val and 107.33 test, merge.

## Round 1 carry-overs still WIP

- **PR #3194 (askeladd, warmup-cosine):** sent back for rebase + re-run on learnable Fourier baseline (20:25). Two arms: warmup=0 vs warmup=3. Beat target: val_avg < 116.34 AND test_avg < 107.33.
- **PR #3207 (nezuko, geom-conditioned slice):** sent back earlier for rebase onto learnable Fourier baseline. Final iteration — geom-slice + learnable Fourier compound test. If beats 116.34, merge; if not, close.
- **PR #3215 (tanjiro, SmoothL1):** previous run (W&B `638hd0v7`, beta=0.05 on fixed Fourier baseline) didn't post terminal results before new baseline merged. Sent back at 20:25 for rebase + re-run with beta=0.05 and beta=0.10 arms.
- **PR #3198 (edward, per-channel pressure loss weights):** previous sweep (3 arms, p_surf_weight=2.0/3.0/5.0 on fixed Fourier baseline) didn't post terminal results. Sent back at 20:25 for rebase + re-run.

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
