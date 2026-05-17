# SENPAI Research State

- **Date:** 2026-05-17 06:40
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4349 (tanjiro, MERGED THIS TURN):** BF16 + LayerScale γ=0.01 + **n_freqs=8** + batch_size=2 + Huber δ=0.10 + **lr=7e-4** + T_max=20 + clip=0.25 + slice_num=32
- **val_avg/mae_surf_p: 55.250** | **test_avg/mae_surf_p: 47.592**
- Per-split test surf_p: single=51.952, rc=60.750, cruise=31.167, re_rand=46.497
- best_epoch=22/22 (timeout-bound)
- **Cumulative improvement: -57.1% val from round-5 start (~128.69)**

**Key finding from #4349**: both arms beat prior best. arm-1 (n=8+lr=7e-4) = 55.250; arm-2 (n=8, no lr change) = 55.270. n_freqs=8 alone worth -0.96%, lr=7e-4 adds another -0.02%. Mechanism: n=8 Fourier (coarser) + slice=32 (+4 epochs) enables lr=7e-4 to escape clip-saturation (clip_frac 0.953 at ep22 vs 0.965+ on n=10 stacks).

**NOTE**: New best stack does NOT include explicit weight_decay — runs at train.py default.

## Improvement history (recent)

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4349 (tanjiro, merged)** | **BF16 + LS + n=8 + bs=2 + lr=7e-4 + δ=0.10 + slice=32** | **55.250** | **47.592** | **-0.98%** |
| #4322 (askeladd, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 + wd=0.001 | 55.799 | 48.846 | -0.58% |
| #4221 (thorfinn, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 + slice=32 | 56.124 | 49.696 | -1.40% |
| #4103 (tanjiro, merged) | BF16 + LS + n=10 + bs=2 + δ=0.10 | 56.92 | 49.32 | -0.33% |
| #4146 (alphonse, merged) | BF16 + LS + n=8 + bs=2 + lr=7e-4 | 57.11 | 49.24 | -1.99% |

## Active WIP (8 students)

| Student | PR | Hypothesis | Stack |
|---|---|---|---|
| **tanjiro** | **#4424** | **lr push {8e-4, 9e-4} on new best stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **edward** | **#4425** | **wd={0.001, 0.0001} compound on new best stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| **askeladd** | **#4479** | **wd bracket {0.002, 0.0015} on new best stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| alphonse | #4448 | lr=8e-4 + wd=0.001 compound on n=10 stack | n=10+δ=0.10+slice=32 |
| frieren | #4484 | T_max bracket {18, 22} on new best stack | n=8+lr=7e-4+δ=0.10+slice=32 |
| **nezuko** | **#4449** | **clip={0.20, 0.22} transfer test on new best stack** | n=8+lr=7e-4+δ=0.10+slice=32 |
| fern | #4396 | n_freqs={8, 12} on n=10 stack | n=10+δ=0.10+slice=32 |
| askeladd | #4406 | wd bracket {0.002, 0.003} on n=10+wd=0.001 stack | n=10+wd=0.001+slice=32 |
| thorfinn | #4407 | T_max bracket {16, 18} on n=10+wd=0.001 stack | n=10+wd=0.001+slice=32 |

⚠️ **NOTE**: train.py Config default is still `slice_num=64` (not 32 despite #4221 intent). All assignments must include explicit `--slice_num 32`. Reminder sent to nezuko #4368. Alphonse #4330 verified to have `--slice_num 32`. Frieren #4439 uses explicit `--slice_num 32`.

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

### Priority 1: Exploit new best stack (n=8+lr=7e-4)
- **#4424 tanjiro** — lr push {8e-4, 9e-4}: is lr=7e-4 the optimum or can we push higher?
- **#4425 edward** — wd compound {0.001, 0.0001}: wd=0.001 helped n=10 stack; does it compound with lr=7e-4 on n=8?

### Priority 2: Close out n=10 stack experiments
- **#4479 askeladd** — wd={0.002, 0.0015} on n=8 stack: complements edward #4425 wd={0.001, 0.0001} to build a 4-point bracket; wd=0.002 was the n=10 local optimum
- **#4448 alphonse** — lr=8e-4 + wd=0.001 compound on n=10 stack: arm-2 lr=8e-4 found best-ever test=47.458 (non-timeout-bound!); compounding with wd=0.001 could push below current best val=55.250
- **#4406 askeladd** — wd bracket {0.002, 0.003} (fills the val optimum region between confirmed wd=0.001 and wd=0.005)
- **#4407 thorfinn** — T_max bracket {16, 18} on n=10+wd=0.001 (relevant if T_max window also applies to new best n=8 stack)
- **#4449 nezuko** — clip={0.20, 0.22} on n=8+lr=7e-4 stack: clip=0.20 won on n=10; does it transfer? Key diagnostic: clip_frac was already 0.953 on new best — tightening may re-saturate or compound.
- **#4396 fern** — n_freqs={8, 12}: n=8 just won; fern testing same on n=10 base. n=12 probably regresses; n=8 arm confirms this finding on different LR/wd context.
- **#4484 frieren** — T_max bracket {18, 22} on new best stack — {18=faster cooldown+4 fine-tuning epochs, 22=no floor LR}; pairs with thorfinn #4407 T_max {16,18} on n=10

## Potential next research directions

**After current in-flight completes:**
- **T_max on new best stack**: thorfinn is testing on n=10; once in for n=10, test {16, 18} on n=8+lr=7e-4 stack too
- **n_freqs=6 on new best stack**: n=8 beat n=10 (-0.96%); does n=6 continue the trend?
- **slice=40 on new best stack**: tested on old stacks; untested on n=8+lr=7e-4
- **LR warmup**: 5-epoch warmup on n=8+lr=7e-4+slice=32 — not yet tested
- **Per-domain loss weighting**: from fern #4331 per-split signal — mixed fourier_base per-domain or per-split surf_weight
- **LR cosine-T_max interaction on new stack**: if wd or lr-push compound, T_max likely needs re-tuning

**Bold directions for plateau protocol (if current wave stalls):**
- **Architecture change**: all improvements so far are training/regularization. Transolver's slice mechanism is untested architecturally — custom slice allocation could help
- **Per-split δ**: cruise + re_rand respond differently than single + rc; per-split Huber δ as a 2D loss hyperparameter
- **Positional encoding**: per-domain mixed Fourier basis (smooth for cruise, sharp for single leading edge)
