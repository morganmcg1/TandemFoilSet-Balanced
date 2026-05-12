# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 ~20:00 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock. Treat experiments as isolated for git and experiment artifacts; do not cross-reference unrelated branches.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current baseline

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | PR |
|--------|-------------------|--------------------|----|
| BF16 autocast (merged) | **123.72** | NaN* | #1371 |

*NaN due to pre-existing data bug in `test_geom_camber_cruise/000020.pt` — frieren fixing in PR #1541.

## Round 1 results so far

| PR | Student | Config | val_avg | 3-split test | Status |
|----|---------|--------|---------|-------------|--------|
| #1371 | frieren | BF16 autocast | **123.72** | 121.90 | ✅ MERGED — new baseline |
| #1367 | fern | Dropout=0.2+clip=1.0 | **113.86** | 114.77 | ♻ Rebasing (conflict w/ #1371) |
| #1412 | thorfinn | Warmup-5ep+cosine | 135.37 | 131.12 | ♻ Rebasing to add BF16+warmup combo |
| #1357-#1400 | others | 5 hypotheses | — | — | WIP/no results yet (2+ hr) |

**fern dropout=0.2** (113.86) is the current leading unmerged result — 7.7% below the BF16 baseline and still descending at the cap. Priority merge candidate when rebase lands.

## Key findings from round 1

1. **BF16 buys ~4 extra epochs** (18 vs ~14) in the 30-min window — free gains
2. **Dropout=0.2** unexpectedly improves every split, not just OOD — likely acting as loss-landscape smoother in the low-epoch regime rather than classic regularization
3. **Warmup=5 > warmup=3** across 3/4 splits (−6.3% on val_avg). Untested with BF16
4. **Critical bug:** `data/scoring.py` + bad GT in `test_geom_camber_cruise/000020.pt` poisons test_avg — fix in progress (#1541)

## Active experiments

| PR | Student | Hypothesis | Expected ETA |
|----|---------|-----------|-------------|
| #1541 | frieren | Fix scoring.py NaN + BF16 baseline rerun | ~30 min |
| #1367 | fern | Dropout=0.2 + BF16 (rebase) | ~30 min |
| #1412 | thorfinn | Warmup-5 + BF16 combo | ~30 min |
| #1352 | alphonse | surf_weight=30 | still running? |
| #1357 | askeladd | Huber loss δ=1.0 | still running? |
| #1365 | edward | OneCycleLR max_lr=1e-3 | still running? |
| #1386 | nezuko | Fourier pos encoding | still running? |
| #1400 | tanjiro | Aux surf-p head | still running? |

## Potential next research directions (round 2+)

- **dropout sweep around 0.2:** fern confirmed 0.2 > 0.1; try 0.15, 0.25, 0.3 to find the optimum
- **dropout=0.2 + warmup-5 + BF16 triple combo:** after each lands, combine them
- **slice_num sweep (64→96→128):** Transolver "physics tokens" count is likely high-leverage; VRAM permits (frieren only used 33 GB with BF16)
- **Bigger model n_hidden=192 or 256:** BF16 at 33 GB leaves 63 GB free; a 2× wider model is possible
- **Output-space transforms:** asinh on p target to compress high-Re tail
- **Per-sample y normalization:** divide each sample's y by its own std before loss
- **Sobolev / divergence-free penalty** on (Ux, Uy)
- **EMA model** for evaluation — averages out the noise in the short 30-min runs
- **surf_weight tune** combined with dropout: both work in the same direction (emphasize surface)
