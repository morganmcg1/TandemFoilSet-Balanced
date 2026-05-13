# SENPAI Research State

- 2026-05-13 06:00 — willow-pai2g-48h-r1, round 2 continued. **Baseline: test=83.77 (PR #1395 Lion-only, provisional)**. Lion+Fourier compound confirmation still pending — in-flight PRs will establish this implicitly.
- No directives from human researcher team yet.

## Current baseline (PR #1395 merged — provisional Lion-only result)
**test_avg/mae_surf_p ≈ 83.77** (Lion alone on n_hidden=192, no Fourier) | val ≈ 92.70
Config: bf16 autocast + **batch_size=4** + **Lion lr=1.5e-4** + scoring-bug workaround + epochs=18; **n_hidden=192**, n_layers=5, n_head=4, **space_dim=34 (Fourier L=8)**, slice_num=64, mlp_ratio=2. W&B run: xhg3h5mi.

Per-split (Lion-only): in_dist=90.07, rc=98.72, cruise=60.96, re_rand=85.32.

Note: The merged train.py has Lion+Fourier stacked. The Lion+Fourier compound result is being established implicitly by in-flight PRs #1877, #1887, #1945, #1798.

## Previous baselines
- PR #1387 (Fourier+wider): test=93.29 | space_dim=34, n_hidden=192, AdamW lr=7e-4
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-2 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup+3e-4 | **CLOSED** ✗ | +5.5% regression. Lion lr=1.5e-4 near-optimal; warmup redundant. |
| alphonse | #1945 | n-hidden-256 | **wip** | n_hidden 192→256 (~57GB at bs=4). Capacity stacking test. |
| askeladd | #1771 | schedule-realigned | **CLOSED** ✗ | T_max=14 worse than T_max=18. |
| askeladd | #1877 | lion-bs-8-sqrt2-lr | **CLOSED** ✗ | +6.5% regression. 2.1× fewer steps starvation. rc split -4.5% = OOD signal real. |
| askeladd | #1980 | gradient-accumulation | **wip** (new) | accum=2, eff_bs=8, same step count as bs=4. Better gradient quality without step starvation. |
| edward | #1643 | mlp-ratio-4 | **CLOSED** ✗ | +10.0% regression. Per-epoch cost +11% → 13 epochs (horizon-vs-capacity). |
| edward | #1973 | cosine-eta-min | **wip** (new) | eta_min=lr/10 (1.5e-5) floor in cosine schedule. Zero overhead. |
| fern | #1796 | weight-decay-1e-3 | **CLOSED** ✗ | +0.8% regression. Per-split sign-flipped vs AdamW. wd lever exhausted. |
| fern | #1969 | decoupled-weight-decay | **wip** (new) | Zero wd on biases/norm scales; wd=1e-4 on weights only. Standard Transformer practice. |
| frieren | #1710 | surf-weight-5 | **CLOSED** ✗ | +13% vs Lion base. Surf-weight lever closed. |
| frieren | #1887 | fourier-L-16 | **wip** | fourier_L 8→16, space_dim 34→66. Frequency ceiling test. |
| nezuko | #1862 | n-layers-6-fourier-wider | **CLOSED** ✗ | +14.7% regression (triple-confirmed dead end across widths). Horizon-vs-depth. |
| nezuko | #1967 | slice-num-96 | **wip** (new) | slice_num 64→96 (1.5× physics-attention slots). Orthogonal capacity axis. |
| tanjiro | #1798 | grad-norm-clip | **wip** | max_norm=1.0 on full Lion+Fourier+wider stack. |
| thorfinn | #1395 | lion-optimizer | **MERGED** ✓ | test **83.77** (−10.2% vs Fourier base). New best. |
| thorfinn | #1876 | n-head-8 | **CLOSED** ✗ | +25.4% regression. head_dim 48→24 below Transolver threshold; +33% per-epoch cost. |
| thorfinn | #1971 | lion-beta2-0999 | **wip** (new) | beta2 0.99→0.999. Longer sign-momentum horizon (~1000 steps). Paper alt value. |

## Key research findings so far
1. **Throughput matters more than architecture at 30-min budget**: bf16+batch-8 gets 17 epochs vs 10-11 — the round-1 win.
2. **Schedule alignment is a massive free win**: T_max=actual epochs delivers −7.67% (test 111.98). T_max=14 over-corrects — current T_max=18 with 14-15 completion is already optimal.
3. **Width × schedule compounds**: n_hidden=192 → −10.97% (test 99.69).
4. **Fourier × width compounds**: NeRF L=8 → −6.42% (test 93.29). High-freq spatial basis helps near-foil pressure gradients.
5. **Lion optimizer is the single biggest lever**: −15.97% from swapping optimizer (test 83.77). Sign-momentum particularly well-suited. ALL splits improve.
6. **Memory: Lion opens batch-size budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4. Enough headroom for bs=8 or n_hidden=256.
7. **EMA dead end**: Model still descending at end-of-run. Both Polyak variants failed.
8. **Depth dead in ALL tested configs**: n_layers=6/7 at n_hidden=128 failed (+14%, +18%). n_layers=6 at n_hidden=192+Fourier failed (+14%). **Horizon-vs-depth tradeoff** — per-epoch cost +20-33% compresses 30-min budget below schedule refinement zone. CLOSED permanently at 30-min budget.
9. **Surf-weight lever CLOSED**: Both directions (5, 25) tie-or-lose. Default surf_weight=10 is robust local optimum.
10. **LR-warmup lever CLOSED for Lion**: lr=3e-4 + 2-epoch warmup regressed +5.5%. Lion's sign-momentum is inherently stable; warmup redundant.
11. **wd magnitude lever CLOSED**: wd=1e-3 under Lion+Fourier gave +0.8% aggregate regression; per-split effects sign-flip vs AdamW run (OOD exhausted under Lion). Pivot to decoupled-wd (structurally different).
12. **n_head=8 CLOSED at n_hidden=192**: head_dim 48→24 below Transolver ≥32 threshold + +33% per-epoch cost. Revisit only if n_hidden=256 lands (head_dim=32 at n_head=8).
13. **mlp_ratio=4 CLOSED**: +11% per-epoch cost → 13 epochs, schedule undertrained. Third "horizon-vs-capacity" failure. Pattern: any capacity change that slows per-epoch >10% hurts without schedule realignment.
14. **bs=8 OOM concern overstated**: mlp_ratio=4 + n_hidden=192 + bs=8 = 50.6 GB peak. bs=8 + Lion at default config also ran at ~89 GB (not OOM). Prior OOM cliff was likely a transient spike under AdamW.
15. **Batch-size lever closed as direct increase**: bs=8 fails due to 2.1× step starvation (+6.5% regression). Gradient accumulation (same steps, eff_bs=8) is the structurally correct alternative. Assigned askeladd #1980.

## Active hypotheses in-flight
| PR | Student | Hypothesis | Memory est. | Expected gain |
|---|---|---|---|---|
| #1980 | askeladd | Gradient accumulation (accum=2, eff_bs=8) | ~43 GB | −2% to −5% |
| #1887 | frieren | Fourier L=16 (space_dim 34→66) | ~50 GB | −1% to −5% |
| #1945 | alphonse | n_hidden=256 (33% wider) | ~57 GB | −3% to −8% |
| #1798 | tanjiro | grad-norm-clip max_norm=1.0 | ~43 GB | −1% to −3% |
| #1967 | nezuko | slice_num 64→96 | ~45-48 GB | −2% to −6% |
| #1969 | fern | decoupled weight decay | ~43 GB | −1% to −4% |
| #1971 | thorfinn | lion beta2=0.999 | ~43 GB | −2% to −5% |
| #1973 | edward | cosine eta_min=lr/10 | ~43 GB | −1% to −3% |

## Most important pending result
**Lion+Fourier compound confirmation** — any of the 4 original in-flight PRs (#1877, #1887, #1945, #1798) will give us this. Until then, the 83.77 baseline remains provisional. If the compound degrades, we need to investigate the interaction (may relate to the oscillatory behavior seen in thorfinn's n_head=8 run, which was the only result on the full Lion+Fourier stack).

## Next milestones
- **n_hidden=256** result (alphonse #1945): width has been the dominant lever; if Lion memory enables 256, expect −5% to −10%
- **Lion bs=8** result (askeladd #1877): gradient accuracy + OOD signal
- **Fourier L=16** result (frieren #1887): fast single-line test of frequency ceiling
- **Lion+Fourier compound confirmation** — implicit in any of the above
