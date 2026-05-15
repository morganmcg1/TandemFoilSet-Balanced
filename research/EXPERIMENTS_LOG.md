# SENPAI Research Results — `icml-appendix-willow-pai2i-48h-r3`

Logged per advisor review of each PR.

## 2026-05-15 — Launch: round 3 of willow-pai2i-48h begins

All 8 students idle; no PRs in flight. First round of assignments dispatched (each PR runs dual-arm baseline + variant in the same wandb_group, since no canonical baseline run exists yet on this branch state).

| Student | Hypothesis | Family |
|---|---|---|
| alphonse | Wider hidden + more heads (128→192, 4→6) | Capacity |
| askeladd | LR warmup + peak 5e-4→1e-3 (3-epoch linear warmup) | Optimization |
| edward | Per-channel loss weighting (p ×3 in MSE) | Loss formulation |
| fern | Huber loss instead of MSE (delta=1.0) | Loss formulation |
| frieren | Per-sample loss normalization (equal-weight per sample) | Loss formulation |
| nezuko | Depth scaling (5→8 layers) | Capacity |
| tanjiro | MLP ratio (2→4) | Capacity |
| thorfinn | Fourier position features + slice_num 64→96 | Inputs |
