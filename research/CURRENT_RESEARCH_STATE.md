# SENPAI Research State

- 2026-05-13 04:10 — willow-pai2g-48h-r1, round 2 in progress. **Baseline: test=83.77 (PR #1395 Lion, provisional Lion-only)**. Lion+Fourier compound confirmation still pending. Stack so far: wider-192 → Fourier L=8 → Lion lr=1.5e-4.
- No directives from human researcher team yet.

## Current baseline (PR #1395 merged — provisional Lion-only result)
**test_avg/mae_surf_p ≈ 83.77** (Lion alone on n_hidden=192, no Fourier) | val ≈ 92.70
Config: bf16 autocast + **batch_size=4** + **Lion lr=1.5e-4** + scoring-bug workaround + epochs=18; **n_hidden=192**, n_layers=5, n_head=4, **space_dim=34 (Fourier L=8)**, slice_num=64, mlp_ratio=2. W&B run: xhg3h5mi.

Per-split (Lion-only): in_dist=90.07, rc=98.72, cruise=60.96, re_rand=85.32.

Note: The merged train.py now has Lion+Fourier stacked. The Lion+Fourier combined result still unverified (expected ≤ 83.77). Several in-flight runs will confirm this implicitly.

## Previous baselines
- PR #1387 (Fourier+wider): test=93.29 | space_dim=34, n_hidden=192, AdamW lr=7e-4
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-2 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup | **CLOSED** ✗ | lr=3e-4 + 2-epoch warmup: test=88.37 (+5.5% vs Lion baseline). Lion at 1.5e-4 is near-optimal; warmup redundant for sign-momentum. |
| alphonse | #1945 | n-hidden-256 | **wip** (new) | n_hidden 192→256 (33% wider). Lion memory enables it: ~43GB at 192 → ~57GB at 256. Capacity stacking test. |
| askeladd | #1771 | wider-192-schedule-realigned | **CLOSED** ✗ | test 104.86 (+5.19%) — T_max=14 worse than T_max=18. |
| askeladd | #1877 | lion-bs-8-sqrt2-lr | wip | Lion memory enables bs=8. bs=8 + lr=2.1e-4 (√2-scaled). |
| edward | #1643 | mlp-ratio-4 | wip | mlp_ratio 2→4 on Lion+Fourier+wider baseline |
| fern | #1796 | weight-decay-1e-3 | **wip (rebase+rerun)** | First run was on stale AdamW baseline (lr=7e-4, 101GB peak, test=100.10 ≈ tie vs old). Sent back to rebase + rerun under Lion+Fourier. Lion paper rec specifically calls for 3-10× larger wd than AdamW. |
| frieren | #1710 | surf-weight-5 | **CLOSED** ✗ | test=94.71 (+1.5% vs Fourier base, +13% vs Lion base). Surf-weight lever closed in both directions. |
| frieren | #1887 | fourier-L-16 | **wip** (new) | fourier_L 8→16, space_dim 34→66. Double frequency resolution ceiling test. |
| nezuko | #1862 | n-layers-6-fourier-wider | wip | n_layers 5→6 on Fourier+wider+Lion baseline |
| tanjiro | #1798 | grad-norm-clip | wip | grad-clip max_norm=1.0 on Lion+Fourier+wider baseline |
| thorfinn | #1395 | lion-optimizer | **MERGED** ✓ | test **83.77** (−10.2% vs Fourier baseline) — HUGE win! New best. |
| thorfinn | #1876 | n-head-8 | wip | n_head 4→8 on Lion+Fourier+wider baseline. First clean Lion+Fourier stacked confirmation. |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 — the round-1 win.
2. **Schedule alignment is a massive free win**: T_max=actual epochs delivers −7.67% (test 111.98). T_max=14 over-corrects — current T_max=18 with 14-15 completion is already optimal (PR #1771 CLOSED).
3. **Width × schedule compounds**: n_hidden=192 → −10.97% (test 99.69).
4. **Fourier × width compounds**: NeRF L=8 → −6.42% (test 93.29). High-freq spatial basis helps near-foil pressure gradients.
5. **Lion optimizer is the single biggest lever**: −15.97% from swapping optimizer (test 83.77 from n_hidden=192 only). Sign-momentum particularly well-suited. ALL splits improve.
6. **Memory: Lion opens batch-size budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4. Enough headroom for bs=8 or n_hidden=256.
7. **EMA dead end**: Model still descending at end-of-run. Both Polyak variants failed.
8. **Depth dead at n_hidden=128**: n_layers=6 (+14%) and n_layers=7 (+18%) regressed. Retesting at n_hidden=192+Fourier (nezuko #1862).
9. **Surf-weight lever CLOSED**: Both directions (5 and 25) tie-or-lose. Default surf_weight=10 is in a robust local optimum. Do not revisit.
10. **LR-warmup lever CLOSED for Lion**: lr=3e-4 + 2-epoch warmup regressed +5.5% (test=88.37). Lion's sign-momentum is inherently stable; warmup is redundant. Lion lr=1.5e-4 is near-optimal.

## Active stacking opportunities (vs test≈83.77 provisional baseline)
- **n_head=8** (thorfinn #1876): first Lion+Fourier stacked confirmation + attention diversity. Most important near-term.
- **Lion bs=8** (askeladd #1877): leverage Lion's memory advantage. Potential −3-8%.
- **Fourier L=16** (frieren #1887): double frequency resolution. Fast single-line test of Fourier ceiling.
- **n_layers=6** (nezuko #1862): depth revisit at full stack. May compound with Lion.
- **mlp_ratio=4** (edward #1643): FFN capacity, orthogonal to optimizer changes.
- **n_hidden=256** (alphonse #1945): ~57GB peak, 33% wider, clean capacity test. Primary near-term priority after n_head result.
- **wd=1e-3 under Lion** (fern #1796 rerun): stale AdamW run rebasing onto Lion. OOD improvement signal (cruise -5.7%, re_rand -2.5%) may be real.
- **Lion wd=1e-2** (per Lion paper rec): unassigned. Potentially large for OOD splits.
- ~~LR-warmup (closed)~~: Lion inherently stable, lever exhausted.

## Next milestones
- Confirm Lion+Fourier compound baseline (thorfinn #1876 will give this as a by-product)
- Fourier L=16 result (frieren #1887): fast test of frequency ceiling
- Lion bs=8 result (askeladd #1877): if large batches help, potential −3-8%
- n_layers=6 result (nezuko #1862): viability of depth at full stack
