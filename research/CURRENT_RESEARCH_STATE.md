# SENPAI Research State

- **Last updated:** 2026-05-17 ~11:50 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 11:05 UTC)

## Current best baseline — n_layers=4 (PR #4453 alphonse, merged ~10:45 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **50.1193** | PR #4453 alphonse n_layers=4 Lookahead k=3 slice=8 `uiy4eks9` |
| `test_3split/mae_surf_p` | **50.2103** | PR #4453 alphonse (manual 3-split, cruise NaN excluded) |

Per-split val (PR #4453, run `uiy4eks9`):

| Split | mae_surf_p | Δ vs prior baseline (50.1657) |
|---|---|---|
| val_single_in_dist | 60.392 | +2.52 (regressed vs eps baseline) |
| **val_geom_camber_rc** | **60.666** | **−1.895 (−3.0%)** ← biggest gain |
| val_geom_camber_cruise | 31.444 | −0.544 |
| val_re_rand | 48.656 | +0.41 |
| **val_avg** | **50.1193** | **−0.047 (−0.09%)** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --n_layers 4 \
  --agent <student>
```

**Note on eps**: this baseline uses DEFAULT eps=1e-8. The eps=1e-9 axis (PR #4401) and n_layers=4 axis are INDEPENDENT. The n_layers=4 + eps=1e-9 stack is the immediate highest-priority follow-up (PR #4552).

## Stack progression

| Merge | val | test | Δ val |
|---|---|---|---|
| PR #4062 fern (slice=8) | 56.895 | 55.982 | — |
| PR #4067 alphonse (β2=0.95, slice=16) | 56.426 | 55.339 | −0.83% |
| PR #4142 nezuko (Lookahead k=5 α=0.5, slice=8) | 54.299 | 52.879 | −3.77% |
| PR #4249 nezuko (Lookahead + β2=0.95, slice=8) | 52.944 | 52.752 | −2.49% |
| PR #4266 alphonse (Lookahead k=3, β2=0.999, slice=8) | 51.307 | 51.886 | −3.09% |
| PR #4401 edward (eps=1e-9, Lookahead k=3) | 50.166 | 50.340 | −2.22% |
| **PR #4453 alphonse (n_layers=4, default eps)** | **50.119** | **50.210** | **−0.09%** |

Total improvement since raw seed: **−64.9%** on val.

## Target proximity

**Target**: val < 50.0, test < 50.0. Current: **50.12 / 50.21**. Need: −0.24% val (0.12 abs), −0.42% test (0.21 abs).

**Critically close.** The n_layers=4 merge and prior eps=1e-9 win are INDEPENDENT axes. If they stack even at 60% additivity, the combined config should land at **val≈49.9, test≈49.9** — clearing both targets simultaneously. This is the campaign's primary hypothesis right now (#4552).

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | On baseline | Status |
|----|---------|-----------|-------------|--------|
| **#4552** | **alphonse** | **n_layers=4 + eps=1e-9 stack (2 seeds)** | **n_layers=4 baseline** | **TARGET-CLEARING RUN. Highest EV of campaign.** |
| **#4554** | **askeladd** | weight decay bracket (5e-5, 3e-4) on n_layers=4 | n_layers=4 baseline | wd untested on current stack |
| **#4573** | **tanjiro** | n_head bracket (1, 4) on n_layers=4 — architecture axis | n_layers=4 baseline | Assigned R41 after closing #4491 (β1 null) |
| **#4585** | **edward** | grad_clip bracket (2.0, 10.0) on n_layers=4 — dynamics axis | n_layers=4 baseline | NEW — Assigned R42 after closing #4490 (eps fine grid null) |
| **#4534** | **nezuko** | Lookahead k=4 retest on eps=1e-9 (2 seeds) | eps=1e-9+n_layers=5 | WIP |
| **#4595** | **frieren** | asinh_p_scale bracket (0.5, 2.0) on n_layers=4 — target normalization | n_layers=4 baseline | NEW — Assigned R43 after closing #4469 (surf_weight null) |
| **#4570** | **thorfinn** | lr bracket (3e-4, 7e-4) on n_layers=4 baseline | n_layers=4 baseline | lr axis untested on current config |
| **#4569** | **fern** | T_max-matched cosine (epochs=20) + eta_min bracket | n_layers=4 baseline | Schedule-fix from #4400 T_max discovery |

**Note on pre-n_layers=4 PRs**: #4534 was assigned against older baselines (val=50.17). It now needs to beat **50.12** to merge. Most will still provide axis characterization even if they don't clear the new bar.

## Recent closures (rounds 40–41)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✓ **#4453** | **alphonse** | **n_layers=4 depth bracket** | **50.12** | ✓ **MERGED — NEW BASELINE** |
| ✗ **#4468** | **askeladd** | **FiLM conditioning on camber M** | **56.89** | ✗ **Closed — FiLM conditioning CLOSED** |
| ✗ **#4400** | **fern** | **eta_min floor** | **52.28** | ✗ **Closed — T_max mismatch makes eta_min mechanically inactive; follow-up #4569 tests T_max-matched schedule** |
| ✗ **#4404** | **thorfinn** | **mlp_ratio bracket (1.0, 1.667)** | **54.54 mean** | ✗ **Closed — throughput noise dominates; best run 50.06 is cherry-pick. FFN-width axis CLOSED at 1.333.** |
| ✗ **#4491** | **tanjiro** | **β1 bracket (0.85, 0.95) on eps=1e-9 stack** | **51.10 (best arm)** | ✗ **Closed R41 — both arms regress. β1-axis CLOSED at default 0.9 on full stack. Architecture axis (n_head) assigned as #4573.** |
| ✗ **#4490** | **edward** | **eps fine grid (3e-9, 3e-10, 1e-10) on eps=1e-9 stack** | **50.73 (best arm B)** | ✗ **Closed R42 — eps=1e-9 is a SHARP PEAK, not a transition. bf16 cliff falsified at 1e-10. val-test decorrelate (Arm B best val/worst test). eps-axis FULLY CLOSED at 1e-9. Dynamics axis (grad_clip) assigned as #4585.** |
| ✗ **#4469** | **frieren** | **surf_weight bracket (5, 20) on k=3 baseline** | **52.22/52.24** | ✗ **Closed R43 — both arms regress ~4% vs current baseline. Notable OOD-vs-in-dist tradeoff in Arm B (sw=5): cruise −1.18%, re_rand −1.87%, in_dist +8.16%. surf_weight axis CLOSED at default 10. Target-normalization axis (asinh_p_scale) assigned as #4595.** |

## Prior closures (rounds 37/32)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✗ **#4461** | **nezuko** | Lookahead α=0.7 (3 seeds on k=3) | 53.67 mean | ✗ Closed — α-axis CLOSED at k=3 (α=0.5 optimal) |
| ✓ **#4401** | **edward** | AdamW eps=1e-9 on k=3 | 50.166 | ✓ MERGED |
| ✗ **#4370** | **tanjiro** | Lookahead k=2+k=4 bracket | 52.73/52.87 | ✗ Closed — k-axis characterized; k=4 retesting on eps=1e-9 stack (#4534) |

## Key mechanism insights

- **n_layers=4 mechanism confirmed**: depth-axis points DOWN at 30-min wall-clock budget. n=4 (22 epochs) beats n=5 (17 epochs). n=7 (10 epochs) collapses. More Lookahead syncs > deeper per-block capacity. The depth bracket is now characterized: n=4 is the sweet spot (n=3 slightly worse on val despite more epochs, n=7 catastrophic).
- **val_geom_camber_rc most improved by depth reduction**: 63.85 → 60.67 (−3.18 absolute, −5.0% over n=5 on old stack). The hardest split is most responsive to update-count increases.
- **eps=1e-9 mechanism confirmed AND fully characterized**: eps=1e-9 is a SHARP PEAK (not a plateau). #4490 fine grid found all of {3e-9, 3e-10, 1e-10} regress vs baseline. The bf16 cliff hypothesis is FALSIFIED — 1e-10 trains with NaN-free grad norms (max 72.5, lowest of all arms). The mechanism is per-parameter step floor: below 1e-9, smallest-sqrt(v̂) params take outsized steps. eps-axis FULLY CLOSED at 1e-9.
- **k=3 sharp val minimum**: per-epoch trajectory shows k=4 leads for 17 epochs but k=3 wins the final epoch sprint — budget/checkpoint-selection artifact; k-axis bracket {2,3,4,5,10} closed
- **k=2 highly unstable**: 4/5 seeds catastrophically worse — tight sync amplifies slow-weight trajectory noise
- **α-axis on k=3 CLOSED**: α=0.3 (failed), α=0.5 (optimal), α=0.7 (PR #4461, +4.60% worse, 3 seeds). The k=5+α=0.7 win does NOT transfer to k=3.
- **FiLM conditioning CLOSED**: scalar M conditioning fails. M is already in input features; post-block modulation destroys residual structure; camber_rc gap is about tandem wake interactions, not M-extrapolation.
- **Capacity axis closed**: n_hidden≥192 doubles epoch time; n_head=4 at n_hidden=128 below expressive threshold
- **FFN-width axis CLOSED at mlp_ratio=1.333** (PR #4404): mlp_ratio=1.0/1.667 both regress at median; per-epoch wall time is attention/IO-bound not FFN-bound; throughput variance dominates
- **T_max scheduling gap discovered** (PR #4400 fern): cfg.epochs=50 is T_max but only ~18-22 epochs fit in wall-clock. All experiments train at ~70% of peak LR at termination. T_max-matched cosine is now being tested (PR #4569)
- **Data-side axis CLOSED**: frequency reweighting + feature-space mixup both failed; FiLM conditioning also closed
- **Slice axis closed at 8**: unimodal optimum, both directions degrade

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP (mlp_ratio=1.333), n_head=2, vel-asinh=0.5
- Huber δ=0.5, slice_num=8
- **n_layers=4** ← NEW (was 5)
- AdamW β2=0.999, lr=5e-4, wd=1e-4
- **Lookahead k=3 α=0.5 — dominant lever on 30-min budget**
- eps=1e-9 (independently valid on n_layers=5 stack; stacking with n_layers=4 unverified but expected)

## What does NOT work

- FiLM conditioning on camber M (scalar M conditioning, post-block residual modulation)
- β1 axis: β1=0.85 and β1=0.95 both regress on both old stack AND new eps+Lookahead stack (#4491, R41). β1-axis CLOSED at default 0.9.
- β2=0.95 under Lookahead k=3 (substitutive at short excursion window)
- n_layers=7 (too few epochs, severe under-training at 30-min budget)
- mlp_ratio ≠ 1.333 (both 1.0 and 1.667 regress at median; FFN-width axis closed)
- eta_min tuning without T_max matching (T_max=50 makes eta_min mechanically inactive at ~18 epoch cutoff)
- EMA=0.995 (substitutes with Lookahead)
- AGC, grad_clip=1.0; bs=8, DropPath, dropout, SWA
- All loss-shape axes tested: Huber δ bracket, asym Huber, log-cosh, Welsch
- Per-sample/per-channel surface loss reweighting
- AoA rotation augmentation; camber-stratified frequency oversampling; camber-bridging mixup
- weight_decay > 1e-4; LR warmup; n_hidden=192; lr=1e-3
- LLRD under Lookahead (substitutive on test)
- slice_num ≠ 8; n_head=4 at n_hidden=128; Lookahead α=0.3, α=0.7 (k=3)
- k=2 (unstable); k=4 tied test on OLD stack (retesting on eps+n_layers=4 stack #4534)

## Strategic outlook

**Target**: val < 50.0, test < 50.0. Current: **50.12 / 50.21**. Gap: 0.12 val, 0.21 test.

1. **#4552 alphonse — n_layers=4 + eps=1e-9 stack** ← **HIGHEST PRIORITY, expected to clear BOTH targets** if axes compound at ≥60% additivity (expected val≈49.9, test≈49.9)
2. **#4554 askeladd — wd bracket on n_layers=4** ← orthogonal probe; wd untested on current stack
3. **#4573 tanjiro — n_head bracket (1, 4) on n_layers=4** ← architecture axis untested at new depth; n_head=4 lost on n_layers=5 but depth change may shift optimum
4. **#4570 thorfinn — lr bracket on n_layers=4** ← lr axis may have shifted at new epoch budget (~22 epochs vs 17)
5. **#4569 fern — T_max-matched cosine** ← tests whether schedule alignment unlocks lower final LR convergence
6. **#4490 edward — eps fine grid** ← tells us the eps floor on n_layers=5; informs eps stacking on n_layers=4
5. **#4534 nezuko — k=4 retest** ← k-axis with smoother eps trajectory
6. **#4469/#4404/#4400** — axis characterization on old stack; useful reference data even if they don't clear new bar

**If #4552 clears both targets**: look for further gains on combined stack. Possible next axes: mlp_ratio at n_layers=4, lr reduction (3e-4 vs 5e-4), β1 fine-grid.

**If #4552 does NOT clear targets**: eps and n_layers gains are partially substitutive. Next bold move: n_layers=3 + eps=1e-9 (even more epochs), or architecture changes (cross-attention between two-body patches for camber_rc).

## Operational notes

- **cruise NaN**: fleet-wide data/scoring.py bug; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min; ~22 epochs at n_layers=4/slice=8 (~82s/epoch); ~17 epochs at n_layers=5
- **GPU utilization**: 100% — all 8 students assigned as of 10:50 UTC
- **API rate limit**: shared student GH token (user 20516801) hits limit when 8 students poll simultaneously. Cycles ~50 min usage / 10-15 min blocked. No advisor intervention possible — auto-clears.
- **New baseline bar for ALL in-flight PRs**: must beat val=50.12 / test=50.21 to merge
