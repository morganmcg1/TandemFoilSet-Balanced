# SENPAI Research State

- **Date:** 2026-05-15
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Baseline status:** to-be-established by first PR round on this branch (no prior committed metrics)

## Current research focus

Round 1 of this 24h research track. The branch starts fresh from
`icml-appendix-charlie`. We attack the unmodified Transolver baseline
(~1M params, n_hidden=128, n_layers=5, slice_num=64) along 8 distinct
mechanisms in parallel to maximize information gain:

1. **Architecture simplification** — LinearNO ablation (remove inter-slice QKV) to test whether inter-slice attention is signal or noise on this dataset.
2. **Architecture scaling** — width/depth scaling under a small learning-rate adjustment.
3. **Optimizer regularization** — EMA weight averaging for OOD robustness.
4. **Optimizer stability** — Cautious AdamW for OOD generalization.
5. **Objective alignment** — pressure-channel weighting in the surface loss term to align the optimization signal with the primary ranking metric.
6. **Spatial representation** — Random Fourier Feature coordinate encoding to lift spectral bandwidth for boundary-layer gradients.
7. **OOD geometry conditioning** — simple persistent global-feature injection per Transolver block (additive GALE-style).
8. **Sample distribution** — high-Re upweighting in the WeightedRandomSampler to address the dynamic-range imbalance across Re regimes.

## Why these picks

All 8 are concrete single-file changes in `train.py` with concrete
implementation guidance. Together they span all the high-leverage attack
vectors (architecture, optimizer, loss, input rep, sampling), giving us
broad coverage in a single 30-min training batch. The two leading
hypotheses (LinearNO ablation, EMA) carry the highest literature-backed
confidence; the remaining 6 are independent mechanisms whose effects can
stack with any winner. Composability is explicit — EMA and channel
weighting are orthogonal to architectural winners.

## Potential next research directions

After this batch lands and we know what improved:

- **Compounding round:** stack the round-1 winners (e.g. EMA + best-arch + best-loss) into a single PR.
- **Asymmetric Q/K projections (LinearNO H4):** alternative architectural step if H1 underperforms.
- **GeoTransolver GALE (full):** multi-scale ball queries + full cross-attention conditioning if simple H13 shows OOD gains.
- **Loss reformulation:** Huber on surface pressure, per-domain loss normalization, log1p compression.
- **Spectral targeting:** SIREN-style learned coordinate encoding, multi-band RFF sweep on σ.
- **Optimizer ladder:** SOAP, Sophia, Lion variants (after Cautious AdamW signal).
- **Output decoder:** two-branch (surface vs volume) decoder if surface-specific capacity is the bottleneck.
- **Curriculum/mining:** hard-sample mining by per-sample MAE during training.

## Open questions for the next round

- Which of (architecture, loss, optimizer, input rep) yields the largest single-knob improvement?
- Does removing inter-slice QKV survive the tandem-foil interaction structure (foil-foil wake coupling)?
- Does EMA's gain on the OOD splits exceed its gain on the in-dist split?
- Do high-Re samples actually drive the metric, or is the camber OOD the dominant axis?

## Living document

Update this file each round with the latest research focus, themes, and
open questions. Prune stale entries; merge winners into the baseline.
