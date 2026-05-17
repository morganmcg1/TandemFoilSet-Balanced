# SENPAI Research State

- **Date:** 2026-05-17 10:00
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4448 (alphonse, MERGED THIS TURN):** BF16 + LayerScale γ=0.01 + **n_freqs=10** + batch_size=2 + Huber δ=0.10 + **lr=8e-4** + **wd=0.001** + T_max=20 + clip=0.25 + slice_num=32
- **val_avg/mae_surf_p: 55.001** | **test_avg/mae_surf_p: 47.946**
- Per-split test surf_p: single=51.626, rc=60.718, cruise=31.954, re_rand=47.486
- best_epoch=20/22 (**NON-TIMEOUT — true convergence**)
- **Cumulative improvement: -57.3% val from round-5 start (~128.69)**

**Key finding from #4448**: lr=8e-4 + wd=0.001 compound on n=10 is the first non-timeout-bound win. clip_frac=0.957 at convergence (healthy range). arm-2 (lr=7e-4+wd=0.001 on n=10) regresses to 55.860 — the lr=8e-4 component is load-bearing, not wd alone.

**Prior best (PR #4349):** n=8+lr=7e-4+δ=0.10+slice=32 — val=55.250 / test=47.592

**NOTE**: Two active lineages: (A) n=8+lr=7e-4, (B) n=10+lr=8e-4+wd=0.001. Both are competitive.

## Improvement history (recent)

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4448 (alphonse, merged)** | **BF16 + LS + n=10 + bs=2 + lr=8e-4 + wd=0.001 + δ=0.10 + slice=32** | **55.001** | **47.946** | **-0.45%** |
| #4349 (tanjiro, merged) | BF16 + LS + n=8 + bs=2 + lr=7e-4 + δ=0.10 + slice=32 | 55.250 | 47.592 | -0.98% |
| #4322 (askeladd, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 + wd=0.001 | 55.799 | 48.846 | -0.58% |
| #4221 (thorfinn, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 | 56.124 | 49.696 | -1.40% |
| #4103 (tanjiro, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 | 56.92 | 49.32 | -0.33% |

## Active WIP (8 students)

| Student | PR | Hypothesis | Stack |
|---|---|---|---|
| **alphonse** | **#4513** | **lr=9e-4+wd=0.001 vs lr=8e-4+wd=0.002 on n=10 stack** | n=10+lr=8e-4+wd=0.001+δ=0.10+slice=32 |
| **tanjiro** | **#4424** | **lr push {8e-4, 9e-4} on new best n=8 stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **edward** | **#4425** | **wd={0.001, 0.0001} compound on n=8 stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **askeladd** | **#4479** | **wd bracket {0.002, 0.0015} on n=8 stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **frieren** | **#4484** | **T_max bracket {18, 22} on new best n=8 stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **nezuko** | **#4449** | **clip={0.20, 0.22} transfer test on new best n=8 stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| fern | #4396 | n_freqs={8, 12} on n=10 stack | n=10+δ=0.10+slice=32 |
| thorfinn | #4407 | T_max bracket {16, 18} on n=10+wd=0.001 stack | n=10+wd=0.001+slice=32 |

⚠️ **NOTE**: train.py Config default is still `slice_num=64` (not 32 despite #4221 intent). All assignments must include explicit `--slice_num 32`.

## Settled levers (do not re-sweep)

| Lever | Settled value | Source |
|---|---|---|
| fourier_base | 2.0 | #4331 (2nd confirmation across stacks) |
| slice_num | 32 (val) / 48 (test) | #4298 — slice=40 worst; sweet spot 32-48 |
| n_head | 4 | #4367 — n_head=2 +1.05%, n_head=8 +6.45% |
| surf_weight on n=8 stack | **10 (default)** | #4439 — sw=11 +0.85%, sw=13 +6.1%; destabilization via clip-saturation |
| n_hidden | 128 | #4289 — wall-clock-bound not capacity-bound |
| δ on lineage A (n=8) | 0.30 | #4199, #4179 |
| δ on lineage B (n=10) | 0.10 | #4220 arm-2 — δ=0.05 regresses |
| clip × δ=0.10 | clip ≤ 0.25 only | #4222, #4223 — clip≥1.0 collapses clip_frac |
| EMA | dead on current stack | #4288 — anti-additive with δ=0.10 |
| bs | 2 | #4147 — bs=1 regresses (step-count optimum) |

## Current research priorities

### Priority 1: Push lr ceiling on both lineages

**Lineage B (n=10+lr=8e-4+wd=0.001, current new best):**
- **#4513 alphonse** — lr=9e-4+wd=0.001 vs lr=8e-4+wd=0.002: can lr exceed 8e-4? Can wd=0.002 compound with lr=8e-4?

**Lineage A (n=8+lr=7e-4, prior best):**
- **#4424 tanjiro** — lr push {8e-4, 9e-4}: is lr=7e-4 the optimum on n=8, or can we push higher?
- **#4425 edward** — wd compound {0.001, 0.0001}: does wd=0.001 (won on n=10) also help n=8?
- **#4479 askeladd** — wd bracket {0.002, 0.0015}: complements edward's bracket; wd=0.002 was n=10 local optimum

### Priority 2: Schedule and regularization tuning on n=8 lineage

- **#4449 nezuko** — clip={0.20, 0.22}: clip=0.20 won on n=10; does it transfer to n=8+lr=7e-4? clip_frac=0.953 at baseline — tightening may re-saturate.
- **#4484 frieren** — T_max bracket {18, 22}: tests both directions of cosine schedule around T_max=20 on n=8

### Priority 3: Close out n=10 stack residual tests

- fern #4396 — n_freqs={8, 12} on n=10 base
- thorfinn #4407 — T_max {16, 18} on n=10+wd=0.001 stack

## Potential next research directions

**After current in-flight completes:**
- **Cross-lineage compound**: n=8+lr=8e-4+wd=0.001 — combines lineage A's n=8 with lineage B's lr+wd
- **n_freqs=6 on new best stacks**: n=8 beat n=10 on lineage A; does n=6 continue the coarser-Fourier trend?
- **T_max on n=10 new best stack**: once thorfinn closes T_max on old n=10, apply winning schedule to n=10+lr=8e-4+wd=0.001
- **slice=40 on new best stacks**: tested on old stacks; untested on either current lineage
- **LR warmup**: 5-epoch warmup on either lineage — not yet tested

**Bold directions for plateau protocol (if current wave stalls):**
- **Architecture change**: all improvements so far are training/regularization. Transolver's slice mechanism is untested architecturally.
- **Per-split δ**: cruise + re_rand respond differently than single + rc; per-split Huber δ as a 2D loss hyperparameter
- **Positional encoding**: per-domain mixed Fourier basis (smooth for cruise, sharp for single leading edge)
