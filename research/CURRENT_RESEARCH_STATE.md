# SENPAI Research State

- 2026-05-13 07:15 — willow-pai2g-48h-r1, round 2 continued. **NEW BEST: test=80.62 (PR #1980 grad-accum=2)**. Cumulative gain from PR #1391: 121.28 → 80.62 = −33.5%. All 8 students assigned (#1945 #1887 #1967 #1969 #1798 #2009 #2010 #2030).
- No directives from human researcher team yet.

## Current baseline (PR #1980 merged — gradient accumulation accum=2)
**test_avg/mae_surf_p = 80.62** | val = 90.82
Config: bf16 + **batch_size=4** + **accumulation_steps=2** (eff_bs=8) + **Lion lr=1.5e-4** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2. W&B run: 6qxwtm0v.

Per-split: in_dist=82.23, rc=93.60, cruise=61.57, re_rand=85.06.

**Mechanism**: Gradient accumulation improves Lion's sign vote via tighter per-micro-batch padding on variable-length TandemFoilSet meshes. Free: no memory increase, same per-epoch time.

## Previous baselines
- PR #1395 (Lion optimizer): test=83.77 | Lion lr=1.5e-4, no accumulation
- PR #1387 (Fourier+wider): test=93.29 | AdamW lr=7e-4, space_dim=34, n_hidden=192
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-2 status
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #1359 | lr-warmup+3e-4 | **CLOSED** ✗ | +5.5% vs Lion baseline. Warmup redundant for sign-momentum. |
| alphonse | #1945 | n-hidden-256 | **wip** | n_hidden 192→256. In-flight (83 GB GPU, training). |
| askeladd | #1771 | schedule-realigned | **CLOSED** ✗ | T_max=14 worse than T_max=18. |
| askeladd | #1877 | lion-bs-8-sqrt2-lr | **CLOSED** ✗ | +6.5% regression. Step-count starvation. |
| askeladd | #1980 | gradient-accumulation | **MERGED** ✓ | test **80.62** (−3.77% vs 83.77). New best! in_dist −8.7%, rc −5.2%. |
| askeladd | #2009 | grad-accum-4 | **wip** (new) | accum=4, eff_bs=16. Tests noise-vs-starvation limit at higher accumulation. |
| edward | #1643 | mlp-ratio-4 | **CLOSED** ✗ | +10% regression. Per-epoch cost +11% → 13 epochs. |
| edward | #1973 | cosine-eta-min | **CLOSED** ✗ | +2.94%. eta_min=lr/10 raised LR 75% higher than expected at ep14. |
| edward | #2010 | swiglu-activation | **wip** (new) | GELU→SiLU in all MLP blocks. On grad-accum=2 stack. |
| fern | #1796 | weight-decay-1e-3 | **CLOSED** ✗ | +0.8%. wd magnitude lever exhausted. |
| fern | #1969 | decoupled-weight-decay | **wip** | Zero wd on biases/norms. In-flight (94 GB GPU). |
| frieren | #1710 | surf-weight-5 | **CLOSED** ✗ | +13% vs Lion base. Dead end. |
| frieren | #1887 | fourier-L-16 | **wip** | Fourier L=16. In-flight (75 GB GPU). |
| nezuko | #1862 | n-layers-6-fourier-wider | **CLOSED** ✗ | +14.7%. Depth dead (horizon-vs-depth). |
| nezuko | #1967 | slice-num-96 | **wip** | slice_num 64→96. In-flight (91 GB GPU). |
| tanjiro | #1798 | grad-norm-clip | **wip** | max_norm=1.0. In-flight (65 GB GPU). |
| thorfinn | #1395 | lion-optimizer | **MERGED** ✓ | test 83.77 (−10.2% vs Fourier baseline). |
| thorfinn | #1876 | n-head-8 | **CLOSED** ✗ | +25.4%. head_dim<32 + per-epoch cost. |
| thorfinn | #1971 | lion-beta2-0999 | **CLOSED** ✗ | +5.99%. Horizon (~1000 steps) exceeded training budget. |
| thorfinn | #2030 | drop-path-stochastic-depth | **wip** (new) | DropPath rate=0.1 linear schedule, both residuals. Orthogonal regularizer. |

## Key research findings so far
1. **Throughput matters at 30-min budget**: bf16+batch-8 → 17 epochs → round-1 win.
2. **Schedule alignment**: T_max=actual epochs → −7.67%. T_max=14 over-corrects.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%. High-freq basis helps near-foil.
5. **Lion optimizer biggest lever**: sign-momentum → −15.97% (83.77 from 99.69).
6. **Lion opens memory budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4.
7. **Depth dead at all tested widths**: horizon-vs-depth tradeoff — per-epoch cost compresses budget. CLOSED permanently.
8. **Surf-weight lever CLOSED**: Default=10 is robust optimum.
9. **LR-warmup CLOSED for Lion**: Sign-momentum inherently stable.
10. **wd magnitude lever CLOSED**: wd=1e-3 under Lion flips per-split signs (OOD exhausted). Decoupled-wd still in-flight.
11. **n_head=8 CLOSED at n_hidden=192**: head_dim<32 + per-epoch cost. Revisit with n_hidden=256.
12. **mlp_ratio=4 CLOSED**: +11% per-epoch cost → undertraining.
13. **Batch-size lever CLOSED (direct)**: bs=8 = 2.1× fewer steps. Starvation > gradient quality.
14. **Gradient accumulation (accum=2) WINS**: −3.77%, 43 GB unchanged. Mechanism: tighter micro-batch padding reduces sign-vote noise for Lion. **New best: test=80.62**.
15. **LR floor (eta_min=lr/10) CLOSED**: Raises LR at epoch 14 by 75% — overshoots refinement window under truncated cosine.
16. **Lion beta2 horizon lever CLOSED**: beta2=0.999 horizon (~1000 steps) exceeds our 1170-1316 step training budget. Buffer never equilibrates → sign-vote signal degradation across all splits. beta2=0.99 is well-matched to truncated training.

## Active hypotheses in-flight
| PR | Student | Hypothesis | Status | Expected gain |
|---|---|---|---|---|
| #1945 | alphonse | n_hidden=256 | Training (83 GB) | −3% to −8% on new baseline |
| #1887 | frieren | Fourier L=16 | Training (75 GB) | −1% to −5% |
| #1967 | nezuko | slice_num=96 | Training (91 GB) | −2% to −6% |
| #1969 | fern | Decoupled weight decay | Training (94 GB) | −1% to −4% |
| #1798 | tanjiro | Grad-norm-clip | Training (65 GB) | −1% to −3% |
| #1971 | thorfinn | lion_beta2=0.999 | Training (94 GB) | −2% to −5% |
| #2009 | askeladd | Grad-accum=4 (eff_bs=16) | Starting | win or informative loss |
| #2010 | edward | SiLU activation (GELU→SiLU) | Starting | −0.5% to −2.5% |
| #2030 | thorfinn | DropPath rate=0.1 linear schedule | Starting | −0.5% to −2.5% |

## Key open questions
1. **Does n_hidden=256 compound with grad-accum?** Width was the dominant lever; Lion's memory savings should enable 256.
2. **Does Fourier L=16 add anything?** Frequency ceiling test.
3. **Do the in-flight PRs (#1887, #1967, #1969, #1798, #1971) beat the NEW grad-accum baseline?** They were assigned before the merge — their results vs old baseline (83.77) may need reinterpretation.
4. **Grad-accum=4 saturation test**: If accum=4 regresses, accum=2 is the optimal lever value.

## Next milestones
- n_hidden=256 result (alphonse #1945) — most anticipated
- Fourier L=16 result (frieren #1887)
- SiLU result (edward #2010) — fast, orthogonal
- Grad-accum=4 saturation test (askeladd #2009)
- DropPath stochastic depth (thorfinn #2030) — pure structural regularizer, orthogonal
