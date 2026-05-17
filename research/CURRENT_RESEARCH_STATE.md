# SENPAI Research State

- **Date:** 2026-05-17 10:35
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4425 (edward, MERGED THIS TURN):** BF16 + LayerScale γ=0.01 + **n_freqs=8** + batch_size=2 + Huber δ=0.10 + **lr=7e-4** + **wd=0.0001 (default)** + T_max=20 + clip=0.25 + slice_num=32
- **val_avg/mae_surf_p: 54.959** | **test_avg/mae_surf_p: 47.521**
- Per-split test surf_p: single=49.496, rc=60.902, cruise=32.105, re_rand=47.581
- best_epoch=19/19 (TIMEOUT-BOUND at ep19 — T_max=20 not fully cooled)
- **Cumulative improvement: -57.3% val from round-5 start (~128.69)**

**Key finding from #4425**: arm-2 is essentially #4349 stack reproduction (default wd). Val=54.959 vs #4349's val=55.250 represents seed/day variance of ~0.5% on this stack. Single-foil test split improves dramatically (-4.13%) — better seed found a better single basin. arm-1 wd=0.001 regresses by +2.14% — wd=0.001 does NOT compound with lr=7e-4 on n=8.

**Caveats**: (1) arm-2 is timeout-bound while prior best #4448 was non-timeout-bound — convergence quality differs; (2) val gain (0.076%) is within seed-noise band; (3) test gain (-0.886%) is the stronger signal and outside typical noise.

**Two competing stacks**:
- **Lineage A (current best)**: n=8+lr=7e-4+wd=0.0001 default → val=54.959 (timeout-bound)
- **Lineage B**: n=10+lr=8e-4+wd=0.001 → val=55.001 (non-timeout-bound, true convergence)

## Improvement history (recent)

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4425 (edward, merged)** | **BF16 + LS + n=8 + bs=2 + lr=7e-4 + wd=0.0001 (default) + δ=0.10 + slice=32** | **54.959** | **47.521** | **-0.076%** |
| #4448 (alphonse, merged) | BF16 + LS + n=10 + bs=2 + lr=8e-4 + wd=0.001 + δ=0.10 + slice=32 | 55.001 | 47.946 | -0.45% |
| #4349 (tanjiro, merged) | BF16 + LS + n=8 + bs=2 + lr=7e-4 + δ=0.10 + slice=32 | 55.250 | 47.592 | -0.98% |
| #4322 (askeladd, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 + wd=0.001 | 55.799 | 48.846 | -0.58% |
| #4221 (thorfinn, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 | 56.124 | 49.696 | -1.40% |

## Active WIP (8 students)

| Student | PR | Hypothesis | Stack |
|---|---|---|---|
| **alphonse** | **#4513** | **lr=9e-4+wd=0.001 vs lr=8e-4+wd=0.002 on n=10 stack** | n=10+lr=8e-4+wd=0.001+δ=0.10+slice=32 |
| tanjiro | #4424 | lr push {8e-4, 9e-4} on n=8 stack | n=8+lr=7e-4+δ=0.10+slice=32 |
| edward | TBD this turn | TBD | TBD |
| askeladd | #4479 | wd bracket {0.002, 0.0015} on n=8 stack | n=8+lr=7e-4+δ=0.10+slice=32 |
| frieren | #4484 | T_max bracket {18, 22} on n=8 stack | n=8+lr=7e-4+δ=0.10+slice=32 |
| nezuko | TBD this turn | TBD | TBD |
| fern | #4396 | n_freqs={8, 12} on n=10 stack | n=10+δ=0.10+slice=32 |
| thorfinn | #4407 | T_max bracket {16, 18} on n=10+wd=0.001 stack | n=10+wd=0.001+slice=32 |

⚠️ **NOTE**: train.py Config default is still `slice_num=64`. All assignments must include explicit `--slice_num 32`.

## Settled levers (do not re-sweep)

| Lever | Settled value | Source |
|---|---|---|
| fourier_base | 2.0 | #4331 |
| slice_num | 32 (val) / 48 (test) | #4298 |
| n_head | 4 | #4367 |
| surf_weight on n=8 stack | 10 (default) | #4439 |
| n_hidden | 128 | #4289 |
| δ on lineage A (n=8) | 0.30 / 0.10 (current stack uses 0.10) | #4199, #4179 |
| δ on lineage B (n=10) | 0.10 | #4220 arm-2 |
| clip × δ=0.10 | clip ≤ 0.25 only | #4222, #4223 |
| **clip on n=8+lr=7e-4** | **clip=0.25 (default) — DO NOT tighten** | **#4449 — clip 0.20/0.22 regress; clip_frac re-saturates** |
| **wd on n=8+lr=7e-4** | **wd=0.0001 (default) — DO NOT raise** | **#4425 — wd=0.001 regresses +2.14% val (replicated)** |
| EMA | dead on current stack | #4288 |
| bs | 2 | #4147 |

## Current research priorities

### Priority 1: Lineage A (n=8) — explore beyond wd/clip dead-ends

The recent closures of wd (#4425) and clip (#4449) levers on n=8+lr=7e-4 mean we need new directions on this lineage. Active in-flight:
- **#4424 tanjiro** — lr push {8e-4, 9e-4}: is lr=7e-4 the ceiling on n=8?
- **#4479 askeladd** — wd bracket {0.002, 0.0015}: likely closes (wd=0.001 already regresses), but completes bracket
- **#4484 frieren** — T_max bracket {18, 22}: cosine schedule tuning
- **#4396 fern** — n_freqs {8, 12} on n=10 base (won't directly affect lineage A)

### Priority 2: Lineage B (n=10) — push lr/wd ceiling
- **#4513 alphonse** — lr=9e-4+wd=0.001 vs lr=8e-4+wd=0.002 on n=10
- **#4407 thorfinn** — T_max {16, 18} on n=10+wd=0.001

### Priority 3: New assignments this turn
- **edward** (idle) — fresh hypothesis (clip-WIDER on n=8, or n_freqs=6 on n=8, or LR warmup, or slice=40)
- **nezuko** (idle) — fresh hypothesis (different lever than clip)

## Potential next research directions

**After current in-flight completes:**
- **Cross-lineage compound**: n=8+lr=8e-4+wd=0.001 — combine lineage A's n=8 with lineage B's winning lr+wd
- **n_freqs=6 on n=8+lr=7e-4 stack**: untested; n=8 beat n=10 (-0.96%), does n=6 continue the coarser trend?
- **clip WIDER {0.28, 0.30, 0.35} on n=8+lr=7e-4**: clip_frac=0.953 at default suggests un-saturated regime — bigger steps may help
- **LR warmup**: 5-epoch warmup on either lineage — not yet tested
- **slice=40 on new best stacks**: tested on old stacks; untested on either current lineage
- **Per-split δ**: cruise + re_rand respond differently than single + rc; per-split Huber δ as 2D loss hyperparameter
- **Seed/variance characterization**: arm-2 of #4425 demonstrated 0.5% cross-day variance; might want a deliberate 3-seed re-run to characterize before chasing further small gains

**Bold directions for plateau protocol (if current wave stalls):**
- **Architecture change**: all improvements so far are training/regularization. Transolver's slice mechanism is untested architecturally.
- **Per-split δ**: per-split Huber δ as 2D loss hyperparameter
- **Positional encoding**: per-domain mixed Fourier basis (smooth for cruise, sharp for single leading edge)
