# SENPAI Research State — willow-pai2g-24h-r5

- **Date:** 2026-05-12 ~21:00 UTC
- **Branch:** `icml-appendix-willow-pai2g-24h-r5`
- **Most recent human directive:** Controlled 24h/48h Charlie-vs-Willow logging ablation. Per-training cap = 30 min wall-clock. Treat experiments as isolated for git and experiment artifacts; do not cross-reference unrelated branches.
- **Programme:** TandemFoilSet CFD surrogate. Primary metric = `val_avg/mae_surf_p` (training), `test_avg/mae_surf_p` (paper).

## Current baseline

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | PR |
|--------|-------------------|--------------------|----|
| BF16 + scoring fix (frieren rerun) | **120.40** | **106.67** | #1541 |

All 4 test splits now finite. Per-test: single_in_dist=125.29, rc=113.23, cruise=81.16, re_rand=106.99

## Round 1 results (completed)

| PR | Student | Config | val_avg | test_avg | Status |
|----|---------|--------|---------|----------|--------|
| #1371 | frieren | BF16 autocast | 123.72 | NaN | ✅ MERGED (superseded) |
| #1541 | frieren | Scoring fix + BF16 rerun | **120.40** | **106.67** | ✅ MERGED — new baseline |
| #1367 | fern | Dropout=0.2+clip=1.0 | **113.86** (pre-BF16) | — | ♻ Rebasing to add BF16 |
| #1412 | thorfinn | Warmup-5ep+BF16 | 123.10 | NaN | ✗ CLOSED (within noise of baseline) |
| Others | Various | 5 hypotheses | — | — | WIP |

**fern dropout=0.2** (113.86 pre-BF16) remains the most promising unmerged result — 7.7% below the old baseline. With BF16, expected to go lower. Top priority merge candidate.

## Active experiments

| PR | Student | Hypothesis | Expected ETA |
|----|---------|-----------|-------------|
| #1583 | thorfinn | CosineAnnealingLR T_max=18 (match reachable epochs) | ~30 min |
| #1367 | fern | Dropout=0.2 + BF16 + scoring fix (rebase) | ~30 min |
| #1400 | tanjiro | Aux surf-p head λ=2 + BF16 combo | ~30 min |
| #1386 | nezuko | Fourier pos encoding, max_freq=32, normalized + BF16 retry | ~30 min |
| #1352 | alphonse | surf_weight=30 | still running? |
| #1357 | askeladd | Huber loss δ=1.0 | still running? |
| #1365 | edward | OneCycleLR max_lr=1e-3 | still running? |

## Key findings from round 1

1. **BF16 buys ~4 extra epochs** (18 vs ~14) in the 30-min window — foundational throughput gain
2. **Scoring bug fixed (PR #1541):** `data/scoring.py` now guards `0×inf=NaN` in the cruise split; `test_avg/mae_surf_p` is usable for the first time
3. **Dropout=0.2** unexpectedly improves every split, not just OOD — likely acting as loss-landscape smoother
4. **Warmup finding:** warmup-5 helps without BF16, but BF16's extra epochs dominate; combined result is within noise of BF16-alone
5. **T_max mismatch:** Current T_max=50 barely decays cosine in 18 BF16 epochs — late-epoch LR stays ~4.5e-4 causing val wobble. T_max=18 is the next lever to pull.

## Potential next research directions (round 2+)

**High priority:**
- **fern dropout=0.2 + BF16:** current leader 113.86 pre-BF16; combo should beat 120.40 baseline clearly
- **T_max=18 (thorfinn):** cosine actually decaying to near-zero should stabilize late training and improve best checkpoint quality
- **dropout sweep around 0.2 + BF16:** once fern lands, try 0.15/0.25/0.30 to find optimum
- **dropout=0.2 + T_max=18 triple:** compound both improvements

**Medium priority:**
- **slice_num sweep (64→96→128):** Transolver "physics tokens" count; VRAM permits (~33 GB currently)
- **CosineAnnealingWarmRestarts:** multi-cycle restart may escape local minima in 18-epoch budget
- **surf_weight tune** combined with dropout: both reduce surface loss emphasis, likely compound
- **EMA model** for evaluation — averages out noise from short runs
- **Bigger model n_hidden=192 or 256:** BF16 at 33 GB leaves 63 GB free

**Exploratory:**
- **Output-space transforms:** asinh on p target to compress high-Re tail
- **Per-sample y normalization:** divide each sample's y by its own std before loss
- **Sobolev / divergence-free penalty** on (Ux, Uy) — physics-informed regularizer
- **Fourier positional encoding (retry):** nezuko fixing max_freq/normalization issues
