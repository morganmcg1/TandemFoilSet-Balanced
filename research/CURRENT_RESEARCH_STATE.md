# SENPAI Research State

- **Date:** 2026-05-12 19:35
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
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency | 1 |
| #1482 | frieren | 3-epoch warmup + peak lr=1e-3 + cosine | Optimization | 1 |
| #1485 | nezuko | slice_num 64 → 128 | Physics-token resolution | 1 |
| #1489 | thorfinn | AoA flip (refine: per-sample p=0.25) | OOD augmentation | 1-revised |
| #1524 | tanjiro | Gradient accumulation (accum=4, eff_bs=16) | Gradient quality | 2 |
| #1526 | edward | Model scaling: n_hidden=224, n_layers=7 (~3.4M) | Capacity | 2 |
| #1531 | alphonse | Channel weight p=10 (sweep from winning p=5) | Loss alignment | 2 |

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4% on floor, PR #1464). Primary metric alignment via loss weighting is a strong lever.
2. **surf_weight=30 alone is WORSE than baseline** (-18% regression, PR #1468 closed). Channel weighting targets pressure only; uniform surface upweighting is too coarse and hurts vol convergence.
3. **pad_collate makes batch scaling expensive** — bs=8 uses 84 GB. Gradient accumulation is the correct approach for larger effective batches.
4. **256-8-8 OOMs at bs=4**; 224-7-8 (~3.4M) is the correct intermediate test for model capacity.
5. **AoA flip per-batch degrades Uy** (mae_surf_Uy doubles). Per-sample flip at lower probability is the fix.
6. **Test NaN in test_geom_camber_cruise** — one bad GT sample (000020.pt) propagates via 0×NaN through scoring. Fix is a 3-line mask in train.py evaluate_split (askeladd PR #1536 in progress).
7. **VRAM budget:** bs=4 baseline uses ~42 GB. bf16 (fern) could halve this to ~21 GB and allow 256-8-8 or bs=8 to fit cleanly.

## Round-2 hypothesis pipeline

### High priority (stack winners)
- **alphonse chan-weight-p10**: does p=10 compound vs p=5? Map the response curve.
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
