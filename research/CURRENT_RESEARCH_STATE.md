# SENPAI Research State

- 2026-05-13 03:00 — willow-pai2g-48h-r1, round 2 in progress, baseline test=99.69 (PR #1361 wider-192). Depth dead end (n_layers=6 +14%, n_layers=7 +18%); EMA dead end both trials.
- No directives from human researcher team yet. Filed issue #1569 flagging data/scoring bug for their attention.

## Current baseline (PR #1361 merged)
**test_avg/mae_surf_p = 99.69** (3-seed mean, best=96.19) | val_avg/mae_surf_p = 111.32 (mean)
Config: bf16 autocast + **batch_size=4** (OOM at bs=8 with n_hidden=192) + lr=7e-4 + scoring-bug workaround + epochs=18; **n_hidden=192**, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2. W&B seeds: jvphwc6p, dcfy4v1z, 9skp8i3k.

## Previous baseline (PR #1591 merged)
**test_avg/mae_surf_p = 111.98** | val_avg/mae_surf_p = 125.36
Config: bf16 autocast + batch_size=8 + lr=7e-4 + scoring-bug workaround + **epochs=18** (schedule-aligned); n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2.

## Current baseline (PR #1391 merged)
**test_avg/mae_surf_p = 121.28** | val_avg/mae_surf_p = 133.75
Config: bf16 autocast + batch_size=8 + lr=7e-4 + scoring-bug workaround; n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, cosine over 50 epochs, 30-min cap.

## Round-1 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup-1e-3 | wip (rebasing) | val 138.85 (mean 144.06, std 4.05 across 3 runs) → rebase+retest |
| askeladd | #1361 | wider-hidden-192 | **MERGED** ✓ | trial-5 (3-seed): test **99.69 mean** (−10.97%) — new baseline! Width × schedule compounded |
| askeladd | #1771 | wider-192-schedule-realigned | wip (new) | Assigned: epochs=14 to realign cosine T_max to n_hidden=192 bs=4 actual budget (~14 eps/30min) |
| edward | #1362 | more-slices-128 | **CLOSED** ✗ | trial-2 rebased: test 155.15 (+27.9% worse, near OOM 94.3GB) → dead end |
| edward | #1591 | cosine-aligned-epochs | **MERGED** ✓ | test **111.98** (−7.67%) — new baseline |
| edward | #1643 | mlp-ratio-4 | wip (new) | Assigned: mlp_ratio 2→4, richer FFN per block |
| fern | #1364 | deeper-7-layers | **CLOSED** ✗ | test 132.06 (+17.9%) — confounded: n_layers=7 cost 146s/ep, only 13/18 epochs completed, cosine refinement phase never reached (LR still 18% of peak at cutoff) |
| fern | #1742 | n-layers-6 | **CLOSED** ✗ | test 127.69 (+14.0%) — fair test (15/18 epochs, proper cosine). Depth underfits at n_hidden=128: in_dist regressed +33%, training-bottlenecked. Depth dead end at this hidden size. |
| fern | #1796 | weight-decay-1e-3 | wip (new) | Assigned: wd 1e-4→1e-3 (10×). Wider model (1.47M params) may benefit from stronger regularization for OOD generalization. |
| frieren | #1380 | surf-weight-25 | **CLOSED** ✗ | test 123.41 (+10.2% worse) — gradient imbalance: higher surf gradient = hotter effective LR on surface-coupled params, no time to settle in 18 epochs |
| frieren | #1710 | surf-weight-5 | wip (new) | Assigned: surf_weight 10→5 (other direction). Hypothesis: 10 may already over-weight surface; fewer surface grad signal → richer volume repr → better surface pred |
| nezuko | #1387 | fourier-pos-features | wip (retrying) | val 119.70 (best val!), NaN test fixed → rebase+retest |
| tanjiro | #1391 | bf16-batch-8 | **MERGED** ✓ | test 121.28 — new baseline |
| tanjiro | #1578 | ema-eval-weights | **CLOSED** ✗ | test 141.87 (+17.0% vs old baseline) — classical Polyak failed: (a) EMA lagged still-descending model in undertrained regime, (b) random-init contamination ~5% via 0.999^3000 |
| tanjiro | #1664 | ema-bias-corrected | **CLOSED** ✗ | test 125.38 (+11.97%) — bias correction worked (val 397→139, -65%) but lag dominates. Model still descending at end-of-run; not in basin-oscillation regime. EMA direction dead. |
| tanjiro | #1798 | grad-norm-clip | wip (new) | Assigned: clip_grad_norm_(max_norm=1.0). Untested stability tool; bf16 + wider model may have gradient spikes. |
| thorfinn | #1395 | lion-optimizer | stale_wip | No result yet |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 epochs, that alone was enough for the round-1 baseline (test=121.28).
2. **Schedule alignment is a massive free win**: simply setting T_max=actual epochs delivers −7.67% (test 111.98) — biggest single round-1 improvement. The cosine refinement phase (final LR ~5e-6 vs un-aligned ~6e-4) finds substantially flatter optima, with the win concentrated on geometry-OOD splits.
3. **Width helps overall but splits OOD generalization**: n_hidden=192 wins −4.93% overall but improves in_dist/rc while regressing on cruise/re_rand. Suggests capacity overfits in-distribution structure at the same depth. Worth round-2 split-level analysis.
4. **Fourier features show strongest val signal** (119.70 vs baseline 133.75 val), but needs proper test comparison — pending rebase.
5. **Critical data bug found**: `test_geom_camber_cruise/000020.pt` has 761 inf in ground-truth `p`; scoring workaround now in baseline.
6. **slice_num=128 is a dead-end at bs=8 bf16**: attention map [B,H,N,slice] memory dominates → near OOM, +44% epoch time → undertrains.

## Active research priorities for pending students
All stale students (fern, frieren, thorfinn) were nudged with comments pointing to the new baseline (test 121.28) and rebase instructions. Alphonse and nezuko have rebase instructions; waiting for trial-2/trial-4 results.

**Key note for askeladd**: n_hidden=192 + bs=8 + bf16 OOM'd at 94GB. Fell back to bs=4. This means the wider model can't use the bf16+batch-8 throughput advantage; its epoch budget reverts to ~14 epochs. This is an important constraint for the width hypothesis.

## Emerging round-2 hypotheses
On the schedule-aligned baseline (test 111.98):
- **Width × schedule**: CONFIRMED — compounded to test=99.69 (−10.97%, PR #1361 merged). The schedule fix is a force-multiplier for capacity.
- **Wider-192 schedule realignment**: askeladd #1771 — epochs=14 aligns cosine T_max to actual n_hidden=192 bs=4 epoch budget. Predicted −1% to −4% additional gain.
- **Fourier × schedule**: nezuko's fourier-pos (val 119.70 was best round-1 val signal) → retest on new baseline.
- **lr-warmup × schedule**: alphonse's lr=1e-3 + warmup → may stack with proper cosine decay.
- **MLP capacity × schedule** (assigned edward #1643): mlp_ratio=2→4 quadruples FFN hidden, tests whether per-block capacity is the bottleneck on schedule-aligned baseline.
- **Bias-corrected EMA × schedule** (assigned tanjiro #1664): Adam-style EMA correction on aligned baseline where late-training is in low-LR oscillation — gives Polyak its proper conditions.
- **Sweep --epochs near the cliff** (16, 17, 18, 19): edward's run hit timeout end of ep 17; finer alignment could squeeze more.
- **Width-split asymmetry**: investigate why n_hidden=192 helps in_dist/rc but hurts cruise/re_rand — may suggest depth (more layers) > width for cross-domain generalization.
- **Depth direction DEAD** at n_hidden=128: both n_layers=6 (fair test, +14%) and n_layers=7 (confounded, +18%) regressed. Training-bottlenecked at this width. Not tested at n_hidden=192.
- **Weight decay sweep** (fern #1796): wd 1e-4→1e-3. Wider model may benefit from stronger regularization.
- **Gradient norm clipping** (tanjiro #1798): max_norm=1.0. Classic stability tool absent from train.py, potentially important for bf16 + wider model.
- **CosineAnnealingWarmRestarts**: edward's follow-up suggestion — may revisit late-training dynamics.

## Next milestones
- Get clean results from the 4 stale_wip students (alphonse, fern, frieren, thorfinn)
- Get rebase+retest results from askeladd, edward, nezuko on new baseline
- Identify whether Fourier features beat bf16 baseline (key question for round 2 direction)
