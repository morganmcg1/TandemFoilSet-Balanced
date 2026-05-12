# SENPAI Research State

- **Date:** 2026-05-12 20:25
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 128.0916** (PR #1482, merged)  
Config: 3-ep warmup + lr=1e-3 + cosine(T_max=47, eta_min=1e-6), bs=4, ~0.66M model, 14 epochs (timeout-cut)  
Test NaN on cruise (model-level batch sensitivity at lr=1e-3 — NOT data bug); bs=1 test_avg = 117.40  
**Note:** Measured WITHOUT chan_w (pre-#1464 base). Advisor branch now has BOTH chan_w + warmup. True stacked floor unmeasured — expected < 128.09.

**Known test NaN bug:** `data/scoring.py` `0*NaN` propagation from `test_geom_camber_cruise/000020.pt` NaN p-channel GT. Affects test_avg but not val_avg (4 val splits are clean). Documented by 2 students (alphonse + thorfinn). Fix = one line in `data/scoring.py` (protected file) — will address in a dedicated bug-fix PR.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Round |
|---|---|---|---|---|
| PR | Student | Hypothesis | Lever | Round |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard fix + clean floor rerun | Bug fix / measurement | 2 |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | 3 |
| #1524 | tanjiro | Stack chan_w + grad-accum=4 + lr=1e-3 + T_max=14 | Stacking / opt | 2-revised |
| #1489 | thorfinn | Stack chan_w + per-sample AoA flip p=0.25 | Stacking / aug | 2-revised |
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency | 1 (WIP — gpu 89 GB active) |
| #1573 | frieren | Warmup + lr=7.5e-4 + gradient clipping | Stability / optimization | 3 |
| #1485 | nezuko | slice_num 64 → 128 | Physics-token resolution | 1 (WIP — gpu 85 GB active) |
| #1526 | edward | Model scaling: n_hidden=224, n_layers=7 (~3.4M) | Capacity | 2 |

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4% on floor, PR #1464 merged, floor 133.94).
2. **Warmup + lr=1e-3 is a confirmed win** (+4.4%, PR #1482 merged, floor 128.09). Val_re_rand most improved (−14%). Test NaN at bs=4 due to numerical boundary at lr=1e-3.
3. **chan_w + warmup are now STACKED in advisor train.py** — new experiments start with both. True stacked floor unmeasured (frieren's run lacked chan_w; expected <128.09).
4. **chan_w response curve is non-monotonic** — p=10 is 14% WORSE than p=5 (PR #1531 closed). Over-weighting pressure starves velocity convergence.
5. **surf_weight=30 alone is WORSE than baseline** (-18% regression, PR #1468 closed).
6. **Grad-accum=4 + sqrt-LR beats pre-chan_w floor by 2.4%** at half the VRAM of bs=8. Stacking with chan_w + T_max=14 in progress (#1524 revised).
7. **Per-sample AoA flip p=0.25 fixes Uy regression** (−50%). Primary metric flat without chan_w stack. Stacking in progress (#1489 revised).
8. **Cosine T_max=50 barely decays in 14-epoch budget** — set T_max≈14 for meaningful LR decay.
9. **pad_collate makes batch scaling expensive** — bs=8 uses 84 GB. Grad-accum is the correct lever.
10. **256-8-8 OOMs at bs=4**; 224-7-8 (~3.4M) is the correct intermediate test for model capacity.
11. **Test NaN Type 1** (data bug): 0×NaN from 000020.pt corrupt p-channel. Fix in train.py evaluate_split (#1536).
12. **Test NaN Type 2** (numerical): lr=1e-3 causes non-finite attention weights for specific bs=4 batch compositions in test_geom_camber_cruise. Fix: lr=7.5e-4 + gradient clipping (#frieren follow-up).
13. **VRAM budget:** bs=4 baseline uses ~42 GB. bf16 (fern, WIP at 89 GB) is actively running.

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
