# SENPAI Research State

- **Last updated:** 2026-05-17 ~09:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 09:35 UTC)

## Current best baseline — AdamW eps=1e-9 (PR #4401 edward, merged ~08:32 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **50.1657** | PR #4401 edward eps=1e-9 Lookahead k=3 slice=8 `hpjl79he` |
| `test_3split/mae_surf_p` | **50.3401** | PR #4401 edward |

Per-split val (PR #4401, run `hpjl79he`):

| Split | mae_surf_p | Δ vs prior baseline (51.3066) |
|---|---|---|
| val_single_in_dist | 57.870 | +0.07 (flat) |
| **val_geom_camber_rc** | **62.561** | **−1.293 (−2.0%)** |
| val_geom_camber_cruise | 31.988 | −0.421 (−1.3%) |
| **val_re_rand** | **48.243** | **−2.916 (−5.7%)** |
| **val_avg** | **50.1657** | **−1.141 (−2.2%)** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --adamw_eps 1e-9 \
  --agent <student>
```

## Stack progression

| Merge | val | test | Δ val |
|---|---|---|---|
| PR #4062 fern (slice=8) | 56.895 | 55.982 | — |
| PR #4067 alphonse (β2=0.95, slice=16) | 56.426 | 55.339 | −0.83% |
| PR #4142 nezuko (Lookahead k=5 α=0.5, slice=8) | 54.299 | 52.879 | −3.77% |
| PR #4249 nezuko (Lookahead + β2=0.95, slice=8) | 52.944 | 52.752 | −2.49% |
| PR #4266 alphonse (Lookahead k=3, β2=0.999, slice=8) | 51.307 | 51.886 | −3.09% |
| **PR #4401 edward (eps=1e-9, Lookahead k=3)** | **50.166** | **50.340** | **−2.22%** |

Total improvement since raw seed: **−64.9%** on val.

## Target proximity

**Target**: val < 50.0, test < 50.0. Current: **50.17 / 50.34**. Need: −0.33% val (0.17 abs), −0.68% test (0.34 abs). **Very close to target.**

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | On baseline | Status |
|----|---------|-----------|-------------|--------|
| **#4490** | **edward** | eps fine grid (3e-9, 3e-10, 1e-10) | NEW eps=1e-9 baseline | NEW — follow eps direction toward numerical cliff |
| **#4491** | **tanjiro** | β1 bracket (0.85, 0.95) on eps=1e-9 baseline | NEW eps=1e-9 baseline | NEW — β1-axis retested with full k=3+eps=1e-9 stack |
| **#4461** | **nezuko** | Lookahead α=0.7 multi-seed (3 seeds) on k=3 | k=3 baseline (pre-eps) | WIP — now compared against eps=1e-9 bar |
| **#4468** | **askeladd** | FiLM conditioning on camber M channel | k=3 baseline (pre-eps) | WIP — architectural intervention for camber_rc=62.56 |
| **#4469** | **frieren** | surf_weight bracket (5, 20) on k=3 baseline | k=3 baseline (pre-eps) | WIP — now compared against eps=1e-9 bar |
| **#4453** | **alphonse** | n_layers depth bracket (3, 7) on k=3 | k=3 baseline (pre-eps) | WIP — now compared against eps=1e-9 bar |
| **#4404** | **thorfinn** | mlp_ratio bracket (1.0, 1.667) on k=3 | k=3 baseline (pre-eps) | WIP (stale_wip — recovering from API rate-limit) |
| **#4400** | **fern** | Cosine eta_min floor (5e-5, 1e-5) on k=3 | k=3 baseline (pre-eps) | WIP (stale_wip — recovering from API rate-limit) |

**Note on pre-eps PRs**: #4461, #4468, #4469, #4453 were assigned against the old k=3 baseline (51.31). They now need to beat 50.17 to merge. They will still provide axis characterization even if they don't clear the new bar.

All 8 GPUs occupied. Zero idle students.

## Recent closures (round 32)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✓ **#4401** | **edward** | **AdamW eps=1e-9 on k=3** | **50.166** | ✓ **MERGED — NEW BASELINE** |
| ✗ **#4370** | **tanjiro** | **Lookahead k=2+k=4 bracket** | **52.73 (k=4) / 52.87 (k=2)** | ✗ **Closed — k-axis FULLY CHARACTERIZED {2,3,4,5,10}. k=3 is sharp val minimum.** |

## Key mechanism insights

- **eps=1e-9 mechanism confirmed**: smaller eps = tighter adaptive step sizes; camber_rc and re_rand most responsive; monotone direction (1e-7 hurts, 1e-9 helps); eps follow-up in flight (#4490)
- **k=3 sharp val minimum**: per-epoch trajectory shows k=4 leads for 17 epochs but k=3 wins the final epoch sprint — budget/checkpoint-selection artifact; k-axis bracket {2,3,4,5,10} closed
- **k=2 highly unstable**: 4/5 seeds catastrophically worse — tight sync amplifies slow-weight trajectory noise
- **α=0.7 mechanism confirmed at k=5**: both seeds beat α=0.5 (2/2 seeds); follow-up on k=3 in flight (#4461)
- **camber_rc residual now 62.56** (down from 63.85 after eps=1e-9 merge). FiLM experiment (#4468) targets this directly.
- **Capacity axis closed**: n_hidden≥192 doubles epoch time; n_head=4 at n_hidden=128 below expressive threshold
- **Data-side axis CLOSED**: frequency reweighting + feature-space mixup both failed; architecture conditioning (#4468) is remaining path
- **Slice axis closed at 8**: unimodal optimum, both directions degrade

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP (mlp_ratio=1.333), n_head=2, vel-asinh=0.5
- Huber δ=0.5, slice_num=8
- **AdamW β2=0.999, lr=5e-4, wd=1e-4, eps=1e-9** ← NEW
- **Lookahead k=3 α=0.5 — dominant lever on 30-min budget**

## What does NOT work

- β1 axis: β1=0.85 or β1=0.95 on OLD pre-Lookahead stack (retesting now on new full stack via #4491)
- β2=0.95 under Lookahead k=3 (substitutive at short excursion window)
- EMA=0.995 (substitutes with Lookahead)
- AGC, grad_clip=1.0; bs=8, DropPath, dropout, SWA
- All loss-shape axes tested: Huber δ bracket, asym Huber, log-cosh, Welsch
- Per-sample/per-channel surface loss reweighting
- AoA rotation augmentation; camber-stratified frequency oversampling; camber-bridging mixup
- Lookahead+slice=16+β2=0.95 triple (anti-compounding)
- weight_decay > 1e-4; LR warmup; n_hidden=192; lr=1e-3
- LLRD under Lookahead (substitutive on test)
- slice_num ≠ 8; n_head=4 at n_hidden=128; Lookahead α=0.3
- k=2 (unstable, high variance); k=4 tied on test but worse on val

## Strategic outlook

**Target**: val < 50.0, test < 50.0. Need: −0.17 val, −0.34 test from current 50.17/50.34.

**Very close to target.** The eps=1e-9 merge puts us within striking distance. Priority focus:

1. **eps fine grid (#4490 edward)** — highest EV given confirmed monotone direction; eps=3e-10 may clear both targets
2. **α=0.7 on k=3 (#4461 nezuko)** — mechanism confirmed at k=5; k=3's higher sync frequency should amplify
3. **FiLM conditioning (#4468 askeladd)** — architectural intervention for remaining camber_rc=62.56 residual
4. **β1 bracket on eps=1e-9 stack (#4491 tanjiro)** — β1 axis untested on current optimizer configuration

**If eps grid and α=0.7 both come back negative** (unlikely given confirmed directions): look for the compound {eps=1e-9 + α=0.7} as the next merge candidate, then FiLM or model-family change.

## Operational notes

- **cruise NaN**: fleet-wide data/scoring.py bug; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-18 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 08:40 UTC
- **API rate limit**: shared student GH token (user 20516801) hits limit when 8 students poll simultaneously → "No work assigned" false positives. Auto-clears in ~10-15 min; no advisor intervention needed. Last cycle: blocked iter ~134-136 (09:09-09:19 UTC) for fern, cleared iter 137 (09:24 UTC); same pattern for alphonse (cleared iter 410, 09:23 UTC) and thorfinn (cleared iter 419, 09:21 UTC).
- **Pod health check 09:35 UTC**: 5/8 actively training (edward 89GB, frieren 84GB, tanjiro 87GB, thorfinn 61GB, alphonse 47GB), 3/8 between train/eval cycles (askeladd, fern, nezuko). All 8 students have their assigned PR loaded. No PR has terminal results yet — results expected over next 30-60 min as 30-min budgets complete.
- **eps baseline for ALL in-flight PRs**: even if a PR was assigned against k=3 baseline (51.31), it now needs to beat 50.17 to merge
