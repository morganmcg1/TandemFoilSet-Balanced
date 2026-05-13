# SENPAI Research State

- 2026-05-13 02:35 — willow-pai2g-48h-r1, round 2 in progress. **New baseline: test=93.29 (PR #1387 Fourier+wider-192)**. Depth dead at n_hidden=128 (both +14% and +18%); EMA dead both trials. Fourier × width compounds −6.42%.
- No directives from human researcher team yet. Filed issue #1569 flagging data/scoring bug for their attention.

## Current baseline (PR #1387 merged)
**test_avg/mae_surf_p = 93.29** | val_avg/mae_surf_p = 103.29
Config: bf16 autocast + **batch_size=4** + lr=7e-4 + scoring-bug workaround + epochs=18; **n_hidden=192**, n_layers=5, n_head=4, **space_dim=34 (Fourier L=8)**, slice_num=64, mlp_ratio=2. W&B run: nh6alavj.

Per-split: in_dist=97.57, rc=106.32, cruise=72.25, re_rand=97.04.

## Previous baselines
- PR #1361 (wider-192): test=99.69 (3-seed mean) | Config: n_hidden=192, bs=4, epochs=18, no Fourier
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128, bs=8, epochs=18
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128, bs=8

## Round-2 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup-1e-3 | wip | val 138.85 → rebasing/retesting on new baseline |
| askeladd | #1771 | wider-192-schedule-realigned | wip | epochs=14 to realign cosine T_max to actual bs=4 budget |
| edward | #1643 | mlp-ratio-4 | wip | mlp_ratio 2→4 on n_hidden=192+Fourier baseline |
| fern | #1796 | weight-decay-1e-3 | wip | wd 1e-4→1e-3 on n_hidden=192+Fourier baseline |
| frieren | #1710 | surf-weight-5 | wip | surf_weight 10→5 on n_hidden=192+Fourier baseline |
| nezuko | #1387 | fourier-pos-features | **MERGED** ✓ | test **93.29** (−6.42%) — new baseline! Fourier × width compounds. |
| nezuko | #1862 | n-layers-6-fourier-wider | wip (new) | Assigned: n_layers 5→6 on Fourier+wider baseline. Prior depth failures were at n_hidden=128; revisit at n_hidden=192 + Fourier. |
| tanjiro | #1798 | grad-norm-clip | wip | clip_grad_norm_(max_norm=1.0) on n_hidden=192+Fourier baseline |
| thorfinn | #1395 | lion-optimizer | stale_wip | No result yet |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 epochs — the round-1 win.
2. **Schedule alignment is a massive free win**: T_max=actual epochs delivers −7.67% (test 111.98) — the cosine refinement phase (final LR ~5e-6 vs unaligned ~6e-4) finds substantially flatter optima.
3. **Width × schedule compounds**: n_hidden=192 on aligned baseline → −10.97% (test 99.69). The schedule fix is a force-multiplier for capacity.
4. **Fourier × width compounds**: NeRF-style log-scale Fourier encoding (L=8) on wider baseline → −6.42% (test 93.29). High-frequency spatial basis helps with near-foil pressure gradients. in_dist wins most (−16.3%), OOD wins more modest (−2 to −3%).
5. **Critical data bug**: `test_geom_camber_cruise/000020.pt` has 761 inf ground-truth `p`; scoring workaround now in baseline.
6. **Depth dead at n_hidden=128**: n_layers=6 (+14%) and n_layers=7 (+18%) both regressed — training-bottlenecked at narrow width. **Untested at n_hidden=192 + Fourier** (nezuko #1862 now testing).
7. **EMA dead end**: Both classical Polyak and bias-corrected EMA failed — model still descending at end-of-run, not in basin-oscillation regime. EMA requires converged training to yield gains.
8. **surf_weight=25 dead end**: Higher surf gradient acts like hotter effective LR on surface params → no time to settle in 18 epochs. Lower direction (surf_weight=5) still testing.

## Active stacking opportunities (given test=93.29 baseline)
- **Depth at n_hidden=192 + Fourier** (nezuko #1862): n_layers=6. If depth now works at richer representation, could compound further. Most important open architecture question.
- **Schedule realignment at bs=4** (askeladd #1771): epochs=14 aligns cosine T_max to actual ~14 eps/30min budget. Should squeeze 1-4% more.
- **MLP capacity** (edward #1643): mlp_ratio=4 on new baseline. Orthogonal to Fourier (different part of network).
- **Regularization** (fern #1796): wd-1e-3 at 1.49M params may improve OOD generalization.
- **Gradient stability** (tanjiro #1798): grad-clip max_norm=1.0. bf16 + wider + Fourier may have gradient spikes.
- **Fourier L sweep**: L=8 is a single point in frequency space. L=16 (space_dim=66) could capture finer spatial resolution. L=4 is cheaper ablation. Not yet assigned.
- **n_head=8**: n_head=4 with n_hidden=192 gives head_dim=48. n_head=8 gives head_dim=24 but 2× more attention patterns. Orthogonal to Fourier. Not yet assigned.
- **lr-warmup on new baseline** (alphonse #1359): if warmup helps on narrow model, may still help on Fourier+wider.

## Next milestones
- Nezuko #1862: n_layers=6 result — determines if depth is viable at n_hidden=192+Fourier
- Edward #1643: mlp_ratio=4 result — capacity lever orthogonal to Fourier/width
- Askeladd #1771: schedule realignment result — should give "free" 1-4% on top of current baseline
- Thorfinn #1395: lion optimizer result (long overdue, may need to close if no progress)
