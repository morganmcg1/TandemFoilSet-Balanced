# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 ~23:15 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current baseline

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | PR |
|--------|-------------------|--------------------|----|
| Fourier pos encoding L=6 mf32 + BF16 | **103.24** | **90.83** | #1386 ✅ MERGED |
| BF16 + scoring fix (frieren rerun) | 120.40 | 106.67 | #1541 |

Per-test-split (new baseline): single_in_dist=105.79, geom_camber_rc=102.99, geom_camber_cruise=64.21, re_rand=90.31

## Round 2 active experiments

| PR | Student | Config | W&B val (best finished run) | Status |
|----|---------|--------|----------------------------|--------|
| #1357 | askeladd | Huber δ=1.0 + BF16 (rebase) | 98.79 (`m733u17z`) — **BEATS baseline** | WIP; new run in flight |
| #1367 | fern | Dropout=0.2 + BF16 (rebase) | 98.96 (`otwlgvo7`) — **BEATS baseline** | WIP; new run in flight |
| #1604 | alphonse | Asinh transform on p target | 106.48 (`8lsszzwj`) — below new baseline | WIP; no SENPAI-RESULT yet |
| #1583 | thorfinn | CosineAnnealingLR T_max=18 | 110.72 (`se2af891`) — below new baseline | WIP; new run in flight |
| #1607 | edward | EMA weight averaging at eval | 112.14 (`bdjvz5qy`) — below new baseline | WIP; new run in flight |
| #1624 | frieren | AdamW betas (0.9, 0.95) + new arm (0.9, 0.98) | 141.04 — worse than baseline | WIP; new run in flight |
| #1400 | tanjiro | Aux surf-p head λ=2 + BF16 | 118.88 (`nzxjwa7n`) — below new baseline | WIP; new run in flight |
| #1386 | nezuko | Fourier L=6 mf32 BF16 | 103.24 ✅ | MERGED |

**Note:** Multiple students started new runs at ~22:51-22:55 UTC, suggesting they completed first arms and are running additional experiments. Results should land ~23:21-23:25 UTC.

## Key findings (all rounds)

1. **Fourier positional encoding (max_freq=32, normalized):** −14.8% test, biggest single gain. All splits improve; largest on cruise geometry (−20.9%). Foundational input feature change.
2. **BF16 buys ~4 extra epochs** (18 vs ~14) in the 30-min window — foundational
3. **Scoring bug fixed (PR #1541):** `data/scoring.py` now guards `0×inf=NaN`
4. **Huber > MSE:** ~10% val gain (pre-BF16), targets high-Re outliers directly. With BF16, expect ~98-99 val
5. **Dropout=0.2 > 0.1:** ~7.7% val gain (pre-BF16). With BF16, expect ~98-99 val
6. **Frequency scaling matters enormously:** Fourier L=6 max_freq=1000 was −8% worse; max_freq=32 is −14% better
7. **Capacity-up architecture loses in 18 epochs:** slice_num=128 −9.5%, aux head starved for epochs
8. **surf_weight, OneCycleLR, warmup-only, AdamW betas(0.9,0.95):** no improvement

## Upcoming priority: compound the top wins

After round-2 PRs merge (Huber + Dropout both likely beat ~103.24 too):

1. **Fourier + Huber:** Input encoding × loss robustness — orthogonal improvements likely to stack
2. **Fourier + Dropout:** Fourier features + regularization — should stack (one fixes input, one fixes training dynamics)
3. **Fourier + Huber + Dropout:** triple combo — the three biggest independent gains
4. **Fourier + Dropout sweep:** 0.15, 0.20, 0.25 — find optimal
5. **Fourier L sweep:** L=8, L=12 with max_freq=32 — marginal but worth screening

## Potential Round 3+ directions

**Compositions (high priority — each tested win is likely orthogonal):**
- Fourier L=6 + Huber δ=1.0 + BF16
- Fourier L=6 + Dropout=0.2 + BF16
- Fourier L=6 + Huber + Dropout + BF16 (triple)
- Fourier L=6 + T_max=18 (if T_max wins)

**Fourier tuning:**
- L=8 or L=12 (same max_freq=32, more octaves)
- min_freq tuning (try 2.0 or π instead of 1.0)
- Concatenate raw positions + Fourier features (don't replace)
- Random Fourier basis (Gaussian weights) vs fixed log-spaced

**Architecture (after compositions saturate):**
- n_hidden=192 or 256 (VRAM has 63 GB free at current config)
- n_layers=6 or 7
- n_head=8 (currently 4)
- Deeper preprocessing MLP (fun_dim-based changes)

**Loss formulation:**
- Huber δ sweep: 0.5, 1.0, 2.0 — find optimum
- Per-sample y normalization
- Sobolev / divergence-free penalty on (Ux, Uy)
- Adaptive loss weighting from gradient norms

**Optimization:**
- Lion or Adan optimizer
- Larger batch + linear LR scaling
- T_max=18 tuning (if it provides meaningful gain over the new baseline)
