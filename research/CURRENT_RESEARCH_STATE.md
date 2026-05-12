# SENPAI Research State

- **Date:** 2026-05-12 20:05
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 133.9353** (PR #1464, merged)  
Config: chan_w=[1,1,5], bs=4, lr=5e-4, ~0.66M baseline model, 14 epochs (timeout-cut)  
Test 3-split avg: 125.48 (excl. cruise NaN)

**Known test NaN bug:** `data/scoring.py` `0*NaN` propagation from `test_geom_camber_cruise/000020.pt` NaN p-channel GT. Affects test_avg but not val_avg (4 val splits are clean). Documented by 2 students (alphonse + thorfinn). Fix = one line in `data/scoring.py` (protected file) — will address in a dedicated bug-fix PR.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Round |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard fix + clean floor rerun | Bug fix / measurement | 2 |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | 3 |
| #1524 | tanjiro | Stack chan_w + grad-accum=4 + lr=1e-3 + T_max=14 | Stacking / opt | 2-revised |
| #1489 | thorfinn | Stack chan_w + per-sample AoA flip p=0.25 | Stacking / aug | 2-revised |
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency | 1 |
| #1482 | frieren | 3-epoch warmup + peak lr=1e-3 + cosine | Optimization | 1 |
| #1485 | nezuko | slice_num 64 → 128 | Physics-token resolution | 1 |
| #1526 | edward | Model scaling: n_hidden=224, n_layers=7 (~3.4M) | Capacity | 2 |

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4% on floor, PR #1464). Primary metric alignment via loss weighting is a strong lever.
2. **chan_w response curve is non-monotonic** — p=10 is 14% WORSE than p=5 (PR #1531 closed). Optimum is p≈5. Mae_surf_Ux degraded +27.5% at p=10 — over-weighting pressure starves velocity convergence.
3. **surf_weight=30 alone is WORSE than baseline** (-18% regression, PR #1468 closed). Channel weighting targets pressure only; uniform surface upweighting is too coarse and hurts vol convergence.
4. **Grad-accum=4 + sqrt-LR beats pre-chan_w floor by 2.4%** at half the VRAM of bs=8 (PR #1524 round 1 sent back). Stacking with chan_w in progress.
5. **Per-sample AoA flip p=0.25 fixes Uy regression** (mae_surf_Uy cut in half from per-batch p=0.5). Primary metric flat — stacking with chan_w in progress (#1489 revised).
6. **pad_collate makes batch scaling expensive** — bs=8 uses 84 GB. Grad-accum is the correct lever.
7. **256-8-8 OOMs at bs=4**; 224-7-8 (~3.4M) is the correct intermediate test for model capacity.
8. **Test NaN in test_geom_camber_cruise** — 0×NaN through scoring. Fix in progress (askeladd PR #1536).
9. **VRAM budget:** bs=4 baseline uses ~42 GB. bf16 (fern) could halve this and enable 256-8-8 or higher effective batch.
10. **Cosine T_max=50 barely decays in 14-epoch budget** — set T_max to actual epoch count for meaningful LR decay (tanjiro #1524 revised includes this).

## Round-2 hypothesis pipeline

### High priority (stack winners)
- **alphonse decoupled surf/vol chan_w** (#1559): apply [1,1,5] only to surface portion of sq_err — addresses Ux degradation seen in p=10 sweep. Expected −2% to −7% on floor.
- **tanjiro chan_w + grad-accum + T_max=14** (#1524 revised): stack two orthogonal levers + fix LR decay schedule. Highest expected value of WIPs.
- **thorfinn chan_w + per-sample AoA flip** (#1489 revised): orthogonal stacking — input distribution × gradient direction.
- **askeladd scoring-nan-guard** (#1536): applies 3-line train.py mask, reruns floor config → first clean test_avg. High-value measurement unlock.

### Medium priority
- **tanjiro grad-accum**: once confirmed, effective bs=16 should compound with any loss/architecture change.
- **edward model-224-7-8**: if 3.4M params improve on 0.66M, the capacity bottleneck is confirmed.

### Potential round-3 directions
- If AMP (fern) wins: retry 256-8-8 with bf16 (activations halved, ~47 GB → fits).
- Stack best winners: chan_w=[1,1,5|10] + warmup (if frieren wins) + AMP (if fern wins).
- Sort-by-size sampler (batching by mesh size reduces pad_collate waste).
- If augmentation pays off: per-sample per-domain AoA flip at p=0.25.
- Dual surface/volume heads (AB-UPT style) if loss-alignment levers saturate.
- Fourier positional encoding for (x, z).
- If NaN guard (#1536) merges cleanly, propagate train.py fix to all future PRs via baseline.
</content>
