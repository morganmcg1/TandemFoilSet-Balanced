# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 ~21:10 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock. Treat experiments as isolated for git and experiment artifacts; do not cross-reference unrelated branches.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current baseline

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | PR |
|--------|-------------------|--------------------|----|
| BF16 + scoring fix (frieren rerun) | **120.40** | **106.67** | #1541 |

All 4 test splits now finite. Per-test: single_in_dist=125.29, rc=113.23, cruise=81.16, re_rand=106.99

## Round 1 results — summary

| PR | Student | Config | val_avg | test_avg | Status |
|----|---------|--------|---------|----------|--------|
| #1371 | frieren | BF16 autocast | 123.72 | NaN | ✅ MERGED (superseded by #1541) |
| #1541 | frieren | Scoring fix + BF16 rerun | **120.40** | **106.67** | ✅ MERGED — current baseline |
| #1367 | fern | Dropout=0.2+clip=1.0 | **113.86** (pre-BF16) | — | ♻ Rebasing to add BF16 |
| #1357 | askeladd | Huber δ=1.0 | **107.91** (pre-BF16) | — | ♻ Sent back for BF16+scoring rerun |
| #1412 | thorfinn | Warmup-5ep+BF16 | 123.10 | NaN | ✗ CLOSED (within noise of baseline) |
| #1352 | alphonse | surf_weight=30 | 120.88 | NaN | ✗ CLOSED (direction exhausted) |
| #1365 | edward | OneCycleLR max_lr=1e-3 | 128.89 | NaN | ✗ CLOSED (schedule mismatched to budget) |
| #1386 | nezuko | Fourier pos encoding | 123.10 | NaN | ♻ Retry with max_freq=32 + BF16 |
| #1400 | tanjiro | Aux surf-p head λ=2 | 132.48 | NaN | ♻ Rebasing for BF16 combo |

**Two clear winners pending rebase + rerun:**
1. **askeladd Huber δ=1.0** (107.91 pre-BF16) — biggest single-experiment gain, ~10.4% below baseline
2. **fern Dropout=0.2** (113.86 pre-BF16) — ~5.5% below baseline

Both should land below 110 after BF16+scoring rerun. These are the priority merge candidates.

## Active experiments (Round 2 — assigned 2026-05-12 21:05)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #1604 | alphonse | Asinh transform on p target (compress high-Re tail) | WIP — new |
| #1607 | edward | EMA weight averaging at eval (smooth val wobble) | WIP — new |
| #1609 | frieren | slice_num=64→128 (double physics-token resolution) | WIP — new |
| #1583 | thorfinn | CosineAnnealingLR T_max=18 (match reachable epochs) | WIP — from cycle 5 |
| #1367 | fern | Dropout=0.2 + BF16 + scoring fix (rebase) | WIP |
| #1357 | askeladd | Huber δ=1.0 + BF16 + scoring fix (rebase) | WIP — sent back |
| #1400 | tanjiro | Aux surf-p head λ=2 + BF16 combo | WIP |
| #1386 | nezuko | Fourier max_freq=32 + normalized positions + BF16 | WIP |

All 8 students have active assignments.

## Key findings from round 1

1. **BF16 buys ~4 extra epochs** (18 vs ~14) in the 30-min window — foundational
2. **Scoring bug fixed (PR #1541):** `data/scoring.py` now guards `0×inf=NaN` — `test_avg/mae_surf_p` is usable
3. **Huber > MSE:** ~10% gain, biggest single lever — targets high-Re outliers directly
4. **Dropout=0.2 > 0.1:** ~5-8% gain, helps every split (loss-landscape smoother, not classic regularizer)
5. **surf_weight, OneCycleLR, raw Fourier, naive aux head, warmup-only:** all fail to beat BF16+scoring baseline standalone
6. **T_max mismatch:** Schedule decays only 36% in 18 BF16 epochs — late-epoch LR wobble visible

## Potential next research directions (round 3+)

**After Round 2 winners merge:**
- **Compound the two biggest wins:** Huber + Dropout=0.2 + BF16 + scoring fix
- **Triple combos:** Huber + Dropout=0.2 + T_max=18 (if T_max wins)
- **Dropout sweep around 0.2:** 0.15, 0.25, 0.30
- **Huber δ sweep:** 0.5, 1.0, 2.0 — find optimum

**Architecture-level (after compositions saturate):**
- **n_hidden=192 or 256:** wider model, BF16 leaves 63 GB free
- **n_layers=6 or 7:** deeper Transolver stack
- **Different attention head count:** n_head=8 (currently 4)

**Loss formulation (orthogonal to model):**
- **Per-sample y normalization:** divide each sample's y by its own std
- **Sobolev / divergence-free penalty** on (Ux, Uy) — physics-informed
- **Adaptive loss weighting:** auto-balance vol/surf weights from gradient norms

**Optimization:**
- **AdamW betas (0.9, 0.95)** — short-training tweak
- **Larger batch + linear LR scaling**
- **Lion / Adan optimizer**

**Data augmentation:**
- **Mesh dropout / subset sampling** for training robustness
- **Coordinate jittering** for spatial invariance
