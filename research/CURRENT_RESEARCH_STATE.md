# SENPAI Research State

- 2026-05-12 20:10 — willow-pai2g-48h-r1, round 1 in progress
- No directives from human researcher team yet. Filed issue #1569 flagging data/scoring bug for their attention.

## Current baseline (PR #1391 merged)
**test_avg/mae_surf_p = 121.28** | val_avg/mae_surf_p = 133.75
Config: bf16 autocast + batch_size=8 + lr=7e-4 + scoring-bug workaround; n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, cosine over 50 epochs, 30-min cap.

## Round-1 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup-1e-3 | stale_wip | No result yet |
| askeladd | #1361 | wider-hidden-192 | wip (retrying) | val 140-148 (NaN test fixed) → rebase+retest |
| edward | #1362 | more-slices-128 | wip (retrying) | test 129.60 (worse than baseline) → rebase+retest |
| fern | #1364 | deeper-7-layers | stale_wip | No result yet |
| frieren | #1380 | surf-weight-25 | stale_wip | No result yet |
| nezuko | #1387 | fourier-pos-features | wip (retrying) | val 119.70 (best val!), NaN test fixed → rebase+retest |
| tanjiro | #1391 | bf16-batch-8 | **MERGED** ✓ | test 121.28 — new baseline |
| thorfinn | #1395 | lion-optimizer | stale_wip | No result yet |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 epochs, that alone was enough for the current best test score.
2. **Fourier features show strongest val signal** (119.70 vs baseline 133.75 val), but needs proper test comparison — sending back for rebase.
3. **Undertraining is the main confound**: cosine T_max=50 barely decays in 10-17 epochs; all round-1 models are severely undertrained.
4. **Critical data bug found**: `test_geom_camber_cruise/000020.pt` has 761 inf in ground-truth `p`; scorig workaround now in baseline.

## Active research priorities for pending students
Students still stale (alphonse, fern, frieren, thorfinn) need to complete their round-1 runs and rebase onto the new baseline before submitting.

## Emerging round-2 hypotheses
- **Fourier + bf16**: Most promising combo given nezuko's val signal and tanjiro's throughput win
- **Wider model (192) + bf16**: Width hypothesis needs a fair test on the new baseline
- **Cosine schedule fix**: T_max aligned to actual epoch budget (key undertraining mitigation — alphonse's `lr-warmup-1e-3` directly tests this)
- **More physics tokens (slice_num=128) + bf16**: Edward's hypothesis needs fair test on new baseline
- **Combined wins**: After confirming which individual changes work, combine the winners for round 3+
- **Deeper model (7 layers)**: Fern's result pending — depth may compound well with throughput improvements

## Next milestones
- Get clean results from the 4 stale_wip students (alphonse, fern, frieren, thorfinn)
- Get rebase+retest results from askeladd, edward, nezuko on new baseline
- Identify whether Fourier features beat bf16 baseline (key question for round 2 direction)
