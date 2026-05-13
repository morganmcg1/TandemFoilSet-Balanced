# SENPAI Research State

- **As of:** 2026-05-13 11:30 (MERGED #2014 nezuko OneCycleLR val=60.98 HUGE WIN; CLOSED #2162 T_max=20 moot; REDIRECTED #1968 thorfinn to max_lr=1e-3, #2106 alphonse to max_lr=6e-4; assigned nezuko #2212 pct_start=0.05, edward #2217 n_hidden=192; 16 effective merges)
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
- **AdamW β2 axis SEMI-OPEN**: 0.999 best on SequentialLR; tanjiro #2125 testing β2={0.95,0.99} on new OneCycleLR HEAD.
- **slice_num axis FULLY CLOSED**: 64 confirmed optimum.
- **lr lower-bound (SequentialLR era) CLOSED**: 5e-4 optimum lower-side. Moot now (max_lr bracket open).
- **loss beta axis OPEN**: 1.0→0.5 merged; beta=0.25 (fern #2164) in flight.
- **eta_min axis CLOSED (SequentialLR era)**: 5e-5 optimum. Moot now (no CosineAnnealing).
- **ref axis CLOSED**: ref=8 optimum.
- **mlp_ratio axis**: upper CLOSED (4 worse), lower OPEN (1 < 2 on SequentialLR; frieren #1992 retesting on OneCycleLR HEAD).
- **wd axis FULLY CLOSED**: 1e-4 optimum.
- **warmup length axis CLOSED (SequentialLR era)**: 1-epoch optimum. Now controlled by pct_start in OneCycleLR.
- **n_head axis FULLY CLOSED**: n_head=4 unimodal optimum.
- **CAWR schedules CLOSED** for 30-min budget.
- **SequentialLR T_max axis CLOSED**: replaced by OneCycleLR entirely.
- **cfg.lr axis MOOT** on new HEAD: OneCycleLR overrides it. max_lr bracket now open.
- **n_hidden axis**: 192/256 tested on EARLY baseline (val ~90-100); REOPEN on OneCycleLR HEAD.

## Themes

1. **Schedule optimization — MERGED.** OneCycleLR(max_lr=8e-4, T_MAX_EPOCHS=21) replaces SequentialLR. Huge win.
2. **max_lr bracket (OneCycleLR).** Thorfinn #1968 (max_lr=1e-3 upper), alphonse #2106 (max_lr=6e-4 lower).
3. **pct_start bracket.** Nezuko #2212 (pct_start=0.05).
4. **Architecture.** Edward #2217 (n_hidden=192, VRAM headroom at 8.5/98GB).
5. **EMA stacking.** Askeladd #1540. Actively training on bs=1+beta=0.5 HEAD (~11:19 finish expected). Result imminent.
6. **Loss beta bracket.** Fern #2164 (beta=0.25).
7. **AdamW β2 bracket.** Tanjiro #2125 (β2={0.95,0.99} rerun on new HEAD).
8. **FFN capacity.** Frieren #1992 (mlp-ratio-1 rerun on new HEAD).

## Leaderboard (val_avg/mae_surf_p)

| Lever | val_avg | test_avg | Status |
|---|---|---|---|
| **OneCycleLR (nezuko #2014)** | **60.98** | **52.48** | **MERGED — CURRENT BEST** |
| loss-beta-0-5 + bs=1 (edward #2012) | 66.32 | 59.68 | MERGED → superseded |
| batch-size-1 (alphonse #2036) | 70.30 | 61.39 | MERGED → superseded |
| adamw-beta2-0-95 (tanjiro #2125) | 69.74 | 62.37 | SENT BACK — rerun on new HEAD |
| lr-4e-4-bs1 (alphonse #2106) | 69.75 | 61.03 | REDIRECTED → max_lr=6e-4 bracket |
| lr-7e-4 (thorfinn #1968) | 79.77 | 72.06 | REDIRECTED → max_lr=1e-3 bracket |

## Active student assignments (all 8)

### max_lr bracket (OneCycleLR tune — highest confidence)
- **PR #2106 — `max_lr-6e-4` (alphonse)** — **WIP (redirected 11:10)** — max_lr lower bracket on OneCycleLR
- **PR #1968 — `max_lr-1e-3` (thorfinn)** — **WIP (redirected 11:10)** — max_lr upper bracket on OneCycleLR

### pct_start bracket
- **PR #2212 — `onecycle-pct-start-0-05` (nezuko)** — **WIP (new)** — pct_start=0.05 (1 epoch warmup vs current 2.1)

### Architecture
- **PR #2217 — `n-hidden-192-onecycle` (edward)** — **WIP (new)** — wider attention n_hidden=128→192

### EMA + loss + β2 + FFN
- **PR #1540 — `ema-weights` (askeladd)** — **WIP** — Training on bs=1+beta=0.5 HEAD; result imminent (~11:19 finish)
- **PR #2164 — `loss-beta-0-25-bs1` (fern)** — **WIP** — beta=0.25 bracket
- **PR #2125 — `adamw-beta2-0-95` (tanjiro)** — **WIP** — β2={0.95,0.99} rerun on new HEAD (will need rebase)
- **PR #1992 — `mlp-ratio-1` (frieren)** — **WIP** — mlp_ratio=1 rerun on new HEAD (will need rebase)

## Next frontier after current round

- Sub-60 val via EMA + max_lr tuning + architecture
- max_lr bracket closure: {6e-4, 8e-4, 1e-3} — one or more should beat 60.98
- EMA expected sub-59 val if mechanism compounds (1-3% improvement on 60.98)
- pct_start tuning: 0.05 vs 0.1 — extra annealing time
- n_hidden=192: VRAM headroom allows it; was tested on OLD baseline only
