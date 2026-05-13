# SENPAI Research State

- **As of:** 2026-05-13 12:20 (MERGED #1968 thorfinn max_lr=1e-3 val=59.39 SUB-60 MILESTONE 17th merge; CLOSED #1540 askeladd EMA null on OneCycleLR; assigned thorfinn #2261 max_lr=1.2e-3, askeladd #2263 AdamW β1=0.95)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=59.39 / test=51.40 (thorfinn #1968 OneCycleLR max_lr=1e-3 on bs=1+beta=0.5). This is the 17th merge — **SUB-60 VAL MILESTONE HIT**.

**Key diagnostics:**
- OneCycleLR max_lr=1e-3: still descending at final epoch 21 (best=21/21), suggesting the LR upper bracket is still open.
- max_lr bracket: 6e-4 < 8e-4 < 1e-3 (all confirmed); 1.2e-3 next.
- rc split (val=70.02) still the hardest — improved from 72.79 → 70.02 with higher max_lr; architecture (#2217 edward) should help.
- cruise split (val=46.67) slightly worsened at 1e-3 (+0.77) — may be sensitive to high peak LR. Something to track.
- EMA axis CLOSED: EMA is mechanistically null on OneCycleLR (deep anneal replaces noise averaging). 
- Sub-58 val is the next target.

## Merged recipe (current advisor base — 16 effective merges)

1. **#1512** (NaN fix) — baseline=123.99
2. **#1513** (bf16 autocast) — 24% speedup
3. **#1416** (unified_pos=True, ref=8) — cruise OOD
4. **#1577** (seed=42 + surf_weight=10 rollback) — val=116.43
5. **#1542** (T_max=15) — val=114.81
6. **#1374** (Huber beta=1.0) — val=110.59
7. **#1696** (grad-clip max_norm=1.0) — val=96.78
8. **#1762** (surf_weight=5.0) — val=90.58
9. **#1695** (T_max=18) — val=84.67
10. **#1855** (eta_min=5e-5) — val=83.95
11. **#1812** (lr-warmup-1ep) — val=82.56
12. **#1972** (batch_size=4→2) — val=76.24
13. **#2036** (batch_size=2→1) — val=70.30
14. **#2012** (smooth_l1 beta=1.0→0.5) — val=66.32
15. **#2014** (OneCycleLR max_lr=8e-4, T_MAX_EPOCHS=21) — val=60.98
16. **#1968** (OneCycleLR max_lr=1e-3) — val=59.39 **CURRENT BEST (sub-60)**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, wd=1e-4, surf_weight=5.0, seed=42, batch_size=1, OneCycleLR(max_lr=1e-3, pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=10.0, T_MAX_EPOCHS=21), AdamW(0.9, 0.999, lr=5e-4 initial), loss=F.smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0)`

**NOTE:** cfg.lr (5e-4) is the AdamW initial parameter but is overridden from step 1 by OneCycleLR. max_lr=1e-3 is the effective peak LR. Experiments targeting "lr" must target "max_lr" in the OneCycleLR call, not cfg.lr.

## Confirmed null results / closed axes

- **max_norm axis CLOSED**: 1.0 optimum.
- **surf_weight axis CLOSED**: 5 optimum (tested on SequentialLR — may need revisit on OneCycleLR).
- **depth axis CLOSED**: 5 optimum.
- **AdamW β1 axis CLOSED**: 0.9 optimum.
- **AdamW β2 axis SEMI-OPEN**: 0.999 best on SequentialLR; tanjiro #2125 testing β2={0.95,0.99} on OneCycleLR HEAD.
- **slice_num axis FULLY CLOSED**: 64 confirmed optimum.
- **lr lower-bound (SequentialLR era) CLOSED**: moot now; max_lr bracket open on OneCycleLR.
- **loss beta axis OPEN**: 1.0→0.5 merged; beta=0.25 (fern #2164) in flight.
- **eta_min axis CLOSED (SequentialLR era)**: moot now (no CosineAnnealing).
- **ref axis CLOSED**: ref=8 optimum.
- **mlp_ratio axis**: upper CLOSED (4 worse), lower OPEN (1 < 2 on SequentialLR; frieren #1992 retesting on OneCycleLR HEAD).
- **wd axis FULLY CLOSED**: 1e-4 optimum.
- **n_head axis FULLY CLOSED**: n_head=4 unimodal optimum.
- **CAWR schedules CLOSED** for 30-min budget.
- **SequentialLR T_max axis CLOSED**: replaced by OneCycleLR entirely.
- **cfg.lr axis MOOT** on new HEAD: OneCycleLR overrides it. max_lr bracket now open.
- **n_hidden axis**: 192/256 tested on EARLY baseline (val ~90-100); REOPEN on OneCycleLR HEAD. Edward #2217 testing.
- **max_lr axis (OneCycleLR): 6e-4 < 8e-4 < 1e-3** confirmed (all tested). Lower bracket closed at 6e-4. 1.2e-3 in-flight (thorfinn #2261). Pattern: each +25% step improved val; stopping condition unclear — continue until regression.
- **pct_start axis (OneCycleLR) BRACKETED**: 0.05 (nezuko #2212, 1-epoch warmup), 0.15 (alphonse #2242, 3-epoch warmup) both in-flight against default 0.10.
- **EMA axis CLOSED on OneCycleLR**: EMA is mechanistically null — OneCycleLR's deep anneal already provides smooth late-training weights (no noisy LR rebound). Do not revisit unless schedule changes.
- **AdamW β1 axis REOPEN on OneCycleLR HEAD**: β1=0.9 was confirmed on SequentialLR (~5e-4 peak). At max_lr=1e-3, β1=0.95 newly relevant (high peak LR → more benefit from stable first-moment direction). Testing via askeladd #2263.

## Themes

1. **Schedule + max_lr — ACTIVE.** OneCycleLR now on max_lr=1e-3 (17th merge). Upper bracket continues at 1.2e-3. EMA CLOSED (null on OneCycleLR).
2. **pct_start bracket.** Nezuko #2212 (0.05, 1-epoch warmup) + alphonse #2242 (0.15, 3-epoch warmup). Default 0.10 current base.
3. **Architecture.** Edward #2217 (n_hidden=192). rc split (70.02) is hardest — wider model most likely to help.
4. **Loss beta bracket.** Fern #2164 (beta=0.25). If lower β wins here as it did 1.0→0.5, continued downward.
5. **AdamW moments.** Tanjiro #2125 (β2={0.95,0.99} on OneCycleLR). Askeladd #2263 (β1=0.95 — first moment, untested on new HEAD).
6. **FFN capacity.** Frieren #1992 (mlp_ratio=1 rerun on max_lr=1e-3 HEAD).

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **OneCycleLR max_lr=1e-3 (thorfinn #1968)** | **59.39** | **51.40** | **MERGED — CURRENT BEST** |
| OneCycleLR max_lr=8e-4 (nezuko #2014) | 60.98 | 52.48 | MERGED → superseded |
| EMA on OneCycleLR (askeladd #1540) | 61.03 | 52.45 | CLOSED — mechanistic null |
| OneCycleLR max_lr=6e-4 (alphonse #2106) | 62.49 | 54.95 | CLOSED — regression +2.5% |
| loss-beta-0-5 + bs=1 (edward #2012) | 66.32 | 59.68 | MERGED → superseded |

## Active student assignments (all 8)

### max_lr bracket (upper side, highest confidence)
- **PR #2261 — `max-lr-1-2e-3` (thorfinn)** — **WIP (new 12:20)** — max_lr=1e-3→1.2e-3 upper bracket continuation

### pct_start bracket
- **PR #2212 — `onecycle-pct-start-0-05` (nezuko)** — **WIP** — pct_start=0.10→0.05 (1-epoch warmup)
- **PR #2242 — `onecycle-pct-start-0-15` (alphonse)** — **WIP** — pct_start=0.10→0.15 (3-epoch warmup)

### Architecture
- **PR #2217 — `n-hidden-192-onecycle` (edward)** — **WIP** — n_hidden=128→192; rc-split improvement expected

### AdamW moments
- **PR #2263 — `adamw-beta1-0-95` (askeladd)** — **WIP (new 12:20)** — β1=0.9→0.95 (first-moment momentum at high peak LR)
- **PR #2125 — `adamw-beta2-0-95` (tanjiro)** — **WIP** — β2={0.95,0.99} rerun on max_lr=1e-3 HEAD

### Loss + FFN
- **PR #2164 — `loss-beta-0-25-bs1` (fern)** — **WIP** — smooth_l1 beta=0.5→0.25
- **PR #1992 — `mlp-ratio-1` (frieren)** — **WIP** — mlp_ratio=2→1 rerun on max_lr=1e-3 HEAD

## Next frontier

- Sub-58 val is the new target
- max_lr bracket: 1.2e-3 result will tell us if the optimum is still climbing or plateaued
- pct_start bracket closure: {0.05, 0.10, 0.15} — good experiment design, results imminent
- Architecture (n_hidden=192): rc split at 70.02 is the dominant drag; wider model most likely to crack it
- AdamW moments both in-flight — if either β1 or β2 wins, revisit the other in the same direction
- After max_lr bracket closes: consider div_factor or final_div_factor tuning (untested)
