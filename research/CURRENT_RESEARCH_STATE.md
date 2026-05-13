# SENPAI Research State

- **As of:** 2026-05-13 12:05 (CLOSED #2106 alphonse max_lr=6e-4 clear regression +2.5%; assigned alphonse #2242 pct_start=0.15 bracket; lower max_lr bracket fully closed, upper in-flight via thorfinn #1968; pct_start now bracketed: 0.05 (nezuko #2212), 0.15 (alphonse #2242))
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm. Local JSONL metrics only.

## Current focus

TandemFoilSet surrogate, primary metric `val_avg/mae_surf_p`. **CURRENT BEST:** val=60.98 / test=52.48 (nezuko #2014 OneCycleLR on bs=1+beta=0.5). This is the 16th merge — biggest single improvement: −8% val / −12% test vs previous best.

**Key diagnostics:**
- OneCycleLR with T_MAX_EPOCHS=21 solves the schedule-alignment problem. Super-convergence schedule (peak lr=8e-4 at ~ep2, cosine descent to 3.2e-6).
- `cfg.lr` (5e-4) is now IRRELEVANT — OneCycleLR overrides optimizer lr from step 1. Experiments targeting cfg.lr are moot after rebase.
- `T_max` (CosineAnnealingLR) no longer exists in train.py. T_max-based experiments are moot.
- rc split (val=72.79) is still the hardest; biggest test gain was rc (−10.12). Architecture (n_hidden=192) may help.
- Sub-60 val milestone is the new target.

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
15. **#2014** (OneCycleLR max_lr=8e-4, T_MAX_EPOCHS=21) — val=60.98 **CURRENT BEST**

**Current `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, wd=1e-4, surf_weight=5.0, seed=42, batch_size=1, OneCycleLR(max_lr=8e-4, pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=10.0, T_MAX_EPOCHS=21), AdamW(0.9, 0.999, lr=5e-4 initial), loss=F.smooth_l1_loss(beta=0.5), clip_grad_norm_(max_norm=1.0)`

**NOTE:** cfg.lr (5e-4) is the AdamW initial parameter but is overridden from step 1 by OneCycleLR. max_lr=8e-4 is the effective peak LR. This distinction is CRITICAL for new experiments — any experiment targeting "lr" must target "max_lr" in the OneCycleLR call, not cfg.lr.

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
- **max_lr axis (OneCycleLR) LOWER CLOSED**: 6e-4 < 8e-4 (alphonse #2106 confirmed +2.5% regression, uniform across all splits). Optimum is at 8e-4 or above. Upper bracket (1e-3) in-flight via thorfinn #1968.
- **pct_start axis (OneCycleLR) BRACKETED**: 0.05 (nezuko #2212, 1-epoch warmup), 0.15 (alphonse #2242, 3-epoch warmup) both in-flight against default 0.10.

## Themes

1. **Schedule optimization — MERGED.** OneCycleLR(max_lr=8e-4, T_MAX_EPOCHS=21) replaces SequentialLR. Huge win.
2. **max_lr bracket (OneCycleLR).** Lower closed (6e-4 regressed +2.5%). Upper: thorfinn #1968 (max_lr=1e-3) in flight.
3. **pct_start bracket.** Nezuko #2212 (pct_start=0.05, 1-epoch warmup) + alphonse #2242 (pct_start=0.15, 3-epoch warmup). Bracket: {0.05, 0.10 current, 0.15}.
4. **Architecture.** Edward #2217 (n_hidden=192). rc split (72.79) is hardest — wider model may help.
5. **EMA stacking.** Askeladd #1540. Rebased onto OneCycleLR HEAD, actively training.
6. **Loss beta bracket.** Fern #2164 (beta=0.25). Mechanism: narrower quadratic zone = more L1 character.
7. **AdamW β2 bracket.** Tanjiro #2125 (β2={0.95,0.99} rerun on OneCycleLR HEAD — higher max_lr makes β2 even more relevant).
8. **FFN capacity.** Frieren #1992 (mlp_ratio=1 rerun on OneCycleLR HEAD).

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **OneCycleLR max_lr=8e-4 (nezuko #2014)** | **60.98** | **52.48** | **MERGED — CURRENT BEST** |
| OneCycleLR max_lr=6e-4 (alphonse #2106) | 62.49 | 54.95 | CLOSED — regression +2.5% |
| loss-beta-0-5 + bs=1 (edward #2012) | 66.32 | 59.68 | MERGED → superseded |
| batch-size-1 (alphonse #2036) | 70.30 | 61.39 | MERGED → superseded |

## Active student assignments (all 8)

### pct_start bracket (OneCycleLR warmup length)
- **PR #2212 — `onecycle-pct-start-0-05` (nezuko)** — **WIP** — pct_start=0.05 (1-epoch warmup, shorter)
- **PR #2242 — `onecycle-pct-start-0-15` (alphonse)** — **WIP (new 12:05)** — pct_start=0.15 (3-epoch warmup, longer; β2 half-life argument)

### max_lr bracket upper side
- **PR #1968 — `max_lr-1e-3` (thorfinn)** — **WIP** — max_lr=8e-4→1e-3 upper bracket

### Architecture
- **PR #2217 — `n-hidden-192-onecycle` (edward)** — **WIP** — n_hidden=128→192; VRAM headroom at 8.5/98GB

### EMA + loss + β2 + FFN
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — EMA decay=0.999 on OneCycleLR HEAD (rebased)
- **PR #2164 — `loss-beta-0-25-bs1` (fern)** — **WIP** — smooth_l1 beta=0.5→0.25
- **PR #2125 — `adamw-beta2-0-95` (tanjiro)** — **WIP** — β2={0.95,0.99} rerun on OneCycleLR HEAD
- **PR #1992 — `mlp-ratio-1` (frieren)** — **WIP** — mlp_ratio=2→1 rerun on OneCycleLR HEAD

## Next frontier after current round

- Sub-60 val via EMA + upper-max_lr + architecture
- pct_start bracket closure: {0.05, 0.10, 0.15} — one should emerge as winner
- max_lr upper side: if 1e-3 beats 8e-4, next bracket is {8e-4, 1e-3, 1.2e-3}
- EMA stacking: rebased on OneCycleLR; if val drops below 59, it's the biggest lever remaining
- n_hidden=192: VRAM headroom allows; if it wins, next is 256
