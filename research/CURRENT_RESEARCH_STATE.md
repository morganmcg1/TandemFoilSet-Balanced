# SENPAI Research State

- 2026-05-13 03:25 — willow-pai2g-48h-r1, round 2 in progress. **New baseline: test=83.77 (PR #1395 Lion optimizer)**. Note: 83.77 is Lion without Fourier; Lion+Fourier compound result pending from ongoing runs. Stack so far: wider-192 → Fourier L=8 → Lion lr=1.5e-4.
- No directives from human researcher team yet.

## Current baseline (PR #1395 merged — provisional Lion-only result)
**test_avg/mae_surf_p ≈ 83.77** (Lion alone on n_hidden=192, no Fourier) | val ≈ 92.70
Config: bf16 autocast + **batch_size=4** + **Lion lr=1.5e-4** + scoring-bug workaround + epochs=18; **n_hidden=192**, n_layers=5, n_head=4, **space_dim=34 (Fourier L=8)**, slice_num=64, mlp_ratio=2. W&B run: xhg3h5mi.

Per-split (Lion-only): in_dist=90.07, rc=98.72, cruise=60.96, re_rand=85.32.

Note: The merged train.py now has Lion + Fourier stacked. The Lion+Fourier combined result is still being measured (expected: ≤ 83.77).

## Previous baselines
- PR #1387 (Fourier+wider): test=93.29 | space_dim=34, n_hidden=192, AdamW lr=7e-4
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-2 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup-1e-3 | wip | lr=1e-3 + warmup on new baseline |
| askeladd | #1771 | wider-192-schedule-realigned | **CLOSED** ✗ | test 104.86 (+5.19%) — T_max=14 worse than T_max=18. Current 14-15/18 epochs is already optimal operating point. |
| askeladd | TBD | lion-bs-8-sqrt2-lr | wip (assigning) | Lion memory budget (43GB vs 94GB AdamW) enables bs=8. bs=8 + lr=2.1e-4 (√2-scaled from 1.5e-4). |
| edward | #1643 | mlp-ratio-4 | wip | mlp_ratio 2→4 on Lion+Fourier+wider baseline |
| fern | #1796 | weight-decay-1e-3 | wip | wd 1e-4→1e-3 on Lion+Fourier+wider baseline |
| frieren | #1710 | surf-weight-5 | wip | surf_weight 10→5 on Lion+Fourier+wider baseline |
| nezuko | #1862 | n-layers-6-fourier-wider | wip (new) | n_layers 5→6 on Fourier+wider baseline (pre-Lion merge — will inherit Lion from branch) |
| tanjiro | #1798 | grad-norm-clip | wip | grad-clip max_norm=1.0 on Lion+Fourier+wider baseline |
| thorfinn | #1395 | lion-optimizer | **MERGED** ✓ | test **83.77** (−10.2% vs Fourier baseline) — HUGE win! New best. |
| thorfinn | TBD | n-head-8 | wip (assigning) | n_head 4→8 on Lion+Fourier+wider baseline. First clean Lion+Fourier stacked confirmation + attention diversity test. |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 — the round-1 win.
2. **Schedule alignment is a massive free win**: T_max=actual epochs delivers −7.67% (test 111.98). T_max=14 over-corrects — current T_max=18 with 14-15 completion is already optimal (PR #1771 CLOSED).
3. **Width × schedule compounds**: n_hidden=192 → −10.97% (test 99.69).
4. **Fourier × width compounds**: NeRF L=8 → −6.42% (test 93.29). High-freq spatial basis helps near-foil pressure gradients.
5. **Lion optimizer is the single biggest lever**: −15.97% from just swapping optimizer (test 83.77 from n_hidden=192 only). Sign-momentum particularly well-suited. ALL splits improve. Much larger than expected.
6. **Memory: Lion opens batch-size budget**: AdamW at n_hidden=192 bs=4 used ~94 GB. Lion uses only ~43 GB (no second-moment buffer) — enough headroom for bs=8.
7. **EMA dead end**: Model still descending at end-of-run, not basin-oscillating. Both Polyak variants failed.
8. **Depth dead at n_hidden=128**: n_layers=6 (+14%) and n_layers=7 (+18%) regressed. Retesting at n_hidden=192+Fourier (nezuko #1862).

## Active stacking opportunities (vs test≈83.77 provisional baseline)
- **n_head=8** (thorfinn): first Lion+Fourier stacked run + attention diversity test. Most important near-term.
- **Lion bs=8** (askeladd): leverage Lion's memory advantage for larger batches + √2 lr scaling.
- **n_layers=6** (nezuko #1862): depth revisit at n_hidden=192+Fourier. May compound further with Lion.
- **mlp_ratio=4** (edward #1643): FFN capacity, orthogonal to all optimizer changes.
- **Fourier L=16**: double frequency resolution (L=8→16). Not yet assigned.
- **Lion wd sweep** (wd=1e-2 per Lion paper recommendation): potentially huge for OOD. Not yet assigned.

## Next milestones
- Establish actual Lion+Fourier stacked baseline (thorfinn n_head=8 run will give this implicitly)
- Lion bs=8 result (askeladd): if large batches help with Lion optimizer, potential −3-8% more
- n_layers=6 result (nezuko): determines if depth is viable at full stack
