# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 ~23:35 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current baseline

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | PR |
|--------|-------------------|--------------------|----|
| Fourier pos encoding L=6 mf32 + BF16 | **103.24** | **90.83** | #1386 ✅ MERGED |
| BF16 + scoring fix (frieren rerun) | 120.40 | 106.67 | #1541 |

Per-test-split (new baseline): single_in_dist=105.79, geom_camber_rc=102.99, geom_camber_cruise=64.21, re_rand=90.31

## Active experiments

| PR | Student | Config | Best W&B val (latest runs) | Status |
|----|---------|--------|---------------------------|--------|
| #1357 | askeladd | Huber δ=1.0 + BF16 | **98.79** (`m733u17z`) — **BEATS baseline** | WIP; nudged to post SENPAI-RESULT |
| #1367 | fern | Dropout=0.2 + BF16 | **98.96** (`otwlgvo7`) — **BEATS baseline** | WIP; nudged to post SENPAI-RESULT; also running dropout+Fourier compound |
| #1604 | alphonse | Asinh transform + Fourier (rebasing) | 106.48 pre-Fourier | WIP; nudged to rebase + rerun |
| #1583 | thorfinn | T_max=18 + Fourier (rebasing) | 112.02 pre-Fourier | WIP; nudged to rebase + rerun |
| #1607 | edward | EMA weight avg + Fourier (rebasing) | 112.14 pre-Fourier | WIP; nudged to rebase + rerun |
| #1400 | tanjiro | Aux surf-p head + Fourier compound | 118.88 pre-Fourier; new compound running | WIP; running fourier+aux compound |
| #1690 | nezuko | Fourier follow-up: L=8 + concat-raw | new assignment | WIP — just assigned |
| #1694 | frieren | n_head=8 (wider attention) + n_hidden=192 secondary | new assignment | WIP — just assigned |

**Closed this cycle:** #1624 frieren AdamW betas (−36% regression, direction exhausted)

## Students running Fourier compounds (from W&B)
Observed in W&B mid-training (~23:15 UTC):
- `p8qfs3xn` fern/dropout-0.2-bf16-fourier — compound in flight
- `xd6973hg` tanjiro/aux-surf-p-lambda2-bf16-fourier — compound in flight
- `bzxxg31v` askeladd/huber-delta-1-bf16 — likely Fourier base rerun

## Priority merge queue (pending formal SENPAI-RESULT submission)

Both of these beat the current Fourier baseline (103.24):
1. **askeladd #1357**: val=98.79, test=88.90 (Huber BF16 without Fourier) — ~4.2% below baseline
2. **fern #1367**: val=98.96, test=88.74 (dropout=0.2 BF16 without Fourier) — ~4.1% below baseline

After these merge, the advisor branch gains both Huber + Dropout on top of Fourier. The dropout+Fourier and Huber+Fourier compounds fern/tanjiro/askeladd are running will then tell us whether these gains stack.

## Key findings (all rounds)

1. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test — biggest single gain. Foundational input feature.
2. **BF16:** ~4 extra epochs (18 vs ~14) in 30-min window
3. **Scoring bug fixed (PR #1541):** NaN guard in scoring.py
4. **Huber > MSE:** ~10% val gain (pre-Fourier base), ~4% above Fourier baseline
5. **Dropout=0.2 > 0.1:** ~7.7% val gain (pre-Fourier), ~4% above Fourier baseline
6. **Frequency scaling:** Fourier max_freq=1000→32 flipped −8% to +14%
7. **Capacity-up architecture loses in 18 epochs:** slice_num=128 −9.5%, frieren AdamW betas −36%
8. **surf_weight, OneCycleLR, warmup-only, raw Fourier, naive aux head:** all fail standalone

## Potential Round 3+ directions

**High priority — compositions:**
- Fourier + Huber + Dropout (triple) — all three orthogonal improvements
- Fourier + T_max=18 (if T_max wins on rebase)
- Fourier + EMA (if EMA wins on rebase)

**Fourier variants (nezuko #1690):**
- L=8 (more octaves)
- Concat raw positions + L=6 (preserve both representations)

**Attention (frieren #1694):**
- n_head=8 (finer slice specialization)
- n_hidden=192 (wider embeddings)

**Architecture (after compositions saturate):**
- n_layers=6 (deeper stack)
- Dropout sweep 0.15/0.20/0.25/0.30

**Loss formulation:**
- Huber δ sweep: 0.5, 1.0, 2.0
- Per-sample y normalization
- Sobolev / divergence-free penalty on (Ux, Uy)
