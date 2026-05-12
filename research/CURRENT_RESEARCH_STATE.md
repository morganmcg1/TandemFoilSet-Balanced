# SENPAI Research State

- **Date:** 2026-05-12 23:15
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 128.0916** (PR #1482, merged)  
Config: 3-ep warmup + lr=1e-3 + cosine(T_max=47, eta_min=1e-6), bs=4, chan_w=[1,1,5], wd=1e-4, ~0.66M model, 14 epochs (timeout-cut)  
Test NaN on cruise (model-level batch sensitivity at lr=1e-3 — NOT data bug); bs=1 test_avg = 117.40  
**Note:** Measured WITHOUT chan_w (pre-#1464 base). Advisor branch now has BOTH chan_w + warmup. True stacked floor unmeasured — expected < 128.09. askeladd's #1536 (sent back for rebase) will measure this.

**Known test NaN bug:** `data/scoring.py` `0*NaN` propagation from `test_geom_camber_cruise/000020.pt` NaN p-channel GT. Affects test_avg but not val_avg (4 val splits are clean). Fix in train.py `evaluate_split` (guard by askeladd — code not yet pushed). First finite test_avg = 133.04 (askeladd's unofficial 3-run mean 130.0).

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Round |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + rebase + re-run with lr=1e-3 | Bug fix / measurement | 2-revised (sent back) |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | 3 (training) |
| #1524 | tanjiro | Stack chan_w + grad-accum=4 + lr=1e-3 + T_max=14 | Stacking / opt | 2-revised (training) |
| #1489 | thorfinn | Stack chan_w + per-sample AoA flip p=0.25 | Stacking / aug | 2-revised (training) |
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency | 1 (training, rate-limit retries) |
| #1573 | frieren | Warmup + lr=7.5e-4 + gradient clipping | Stability / optimization | 3 (training) |
| #1603 | edward | EMA weights (decay=0.999) — v2 with snapshot fix | Inference-time averaging | 3 (training fix) |
| #1681 | nezuko | Higher weight decay (wd=1e-4 → 5e-4) | Regularization | 3 (just assigned) |

## Recent decisions

- **#1485 (nezuko) CLOSED**: slice_num=128 stacked on floor → +25.4% regression (160.67 vs 128.09). Wall-clock budget is the binding constraint, not capacity. Revisit only with AMP (#1477).
- **#1536 (askeladd) SENT BACK**: Code not pushed; branch pre-#1482 (would revert warmup). NaN guard logic correct — test_avg 133.04 is first-ever finite result. Needs rebase + push + re-run at lr=1e-3.
- **#1603 (edward) v1 regression**: -7.8% from random-init EMA bias. Edward applied fix (snapshot model on first post-warmup step), v2 running.

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4% on floor, PR #1464 merged, floor 133.94).
2. **Warmup + lr=1e-3 is a confirmed win** (+4.4%, PR #1482 merged, floor 128.09). val_re_rand most improved (−14%).
3. **chan_w + warmup are now STACKED in advisor train.py** — new experiments start with both. True stacked floor unmeasured (expected < 128.09).
4. **chan_w response curve is non-monotonic** — p=10 is 14% WORSE than p=5 (PR #1531 closed). Over-weighting pressure starves velocity convergence.
5. **surf_weight=30 alone is WORSE than baseline** (-18% regression, PR #1468 closed).
6. **Grad-accum=4 + sqrt-LR beats pre-chan_w floor by 2.4%** at half the VRAM of bs=8. Stacking with chan_w + T_max=14 in progress (#1524 revised).
7. **Per-sample AoA flip p=0.25 fixes Uy regression** (−50%). Primary metric flat without chan_w stack. Stacking in progress (#1489 revised).
8. **Cosine T_max=50 barely decays in 14-epoch budget** — set T_max≈14 for meaningful LR decay.
9. **pad_collate makes batch scaling expensive** — bs=8 uses 84 GB. Grad-accum is the correct lever.
10. **224-7-8 model at bs=2 only reached 6 epochs — undertrained, inconclusive** (PR #1526 closed). Retry after fern's AMP/bf16 result.
11. **Test NaN Type 1** (data bug): 0×NaN from 000020.pt corrupt p-channel. Fix in train.py evaluate_split (#1536 sent back, correct logic confirmed).
12. **Test NaN Type 2** (numerical): lr=1e-3 causes non-finite attention weights for specific bs=4 batches in test_geom_camber_cruise. Fix: lr=7.5e-4 + gradient clipping (#1573 frieren in progress).
13. **VRAM budget:** bs=4 baseline uses ~42 GB; slice_num=128 uses ~55 GB. bf16 (fern, GPU 99% active) expected ~21-25 GB.
14. **slice_num=128 doesn't compound with stacked floor at 30-min cap** (+25.4% regression, PR #1485 closed). Need AMP first.
15. **EMA random-init bias**: literal `copy.deepcopy(model)` after random init has stale random weights in EMA for many steps. Fix: snapshot model into EMA on first post-warmup step (#1603 edward v2 in progress).
16. **First finite test_avg on branch: 133.04** (askeladd's unofficial run at old config). True clean test_avg at floor config = unmeasured.

## Round-3 hypothesis pipeline

### High priority (active)
- **askeladd NaN guard rebased + rerun** (#1536 sent back): first confirmed clean test_avg measurement at floor config. Critical unlock.
- **alphonse decoupled surf/vol chan_w** (#1559 training): [1,1,5] surf only — addresses Ux degradation at p=10.
- **tanjiro chan_w + grad-accum + T_max=14** (#1524 revised training): stack two orthogonal levers + fix LR decay.
- **thorfinn chan_w + per-sample AoA flip** (#1489 revised training): orthogonal stacking.
- **frieren lr=7.5e-4 + gradclip** (#1573 training): fix test NaN Type 2 while staying near peak LR.
- **edward EMA weights v2** (#1603 fix running): snapshot fix applied, v2 in flight.
- **fern AMP bf16** (#1477 training, GPU 99% active): unlock 2x faster training → more epochs, enables 224-7-8 retry.
- **nezuko weight decay 5e-4** (#1681 just assigned): regularization to reduce final-epoch val noise.

### Next round priorities (if current WIPs complete)
- If AMP (fern) wins: retry 224-7-8 + slice_num=128 with bf16 (both need the VRAM headroom).
- Stack best winners: chan_w + warmup + AMP + EMA + WD (if wins compound).
- Sort-by-size sampler (reduce pad_collate waste, enables higher effective batch).
- SmoothL1/Huber loss for pressure channel (handles outlier p spikes in training, orthogonal to chan_w).
- If NaN guard (#1536) merges cleanly, propagate to all future PRs via advisor train.py commit.
- Dual surface/volume heads (AB-UPT style) if loss-alignment levers saturate.
- Fourier positional encoding for (x, z) coordinates.
- Lookahead optimizer wrapper (orthogonal to AdamW + schedule, 0.5-2% typical gain).
