# SENPAI Research State
- 2026-04-27 23:30 — round 1 in flight; PR #355 first to land, sent back due to test-eval NaN
- Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val splits); ranking final metric is `test_avg/mae_surf_p`

## Active risk: non-finite predictions can poison `test_avg/mae_surf_p`
- PR #355 returned with `test_geom_camber_cruise/mae_surf_p = NaN` (val on the same split was clean at 99.19, but at least one *test* sample triggers a non-finite prediction on the pressure channel; only the p channel is affected, velocity MAE is finite).
- `data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions** (and is read-only).
- Fix lives in `train.py:evaluate_split` — apply `pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)` immediately after the model forward. This is a no-op for healthy predictions and converts pathological ones into a large finite penalty rather than NaN.
- This may affect other round-1 PRs once they land. If a returning PR also reports test NaN, route the same fix.

## Baseline (default `train.py`)
- Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- AdamW lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, cosine annealing over `epochs`
- No metrics measured yet on this branch — round 1 will establish the baseline numbers and start beating them.

## Round 1 hypothesis portfolio (8 experiments — all assigned, status:wip)
Coverage chosen so that each lever is moved independently and likely-orthogonal gains can compound across rounds:

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #350 | alphonse  | bigger-transolver-bf16   | Architecture (capacity, n_hidden 128→256, n_head 4→8) + bf16 autocast | Default Transolver may underfit 24→3 regression on 1499 samples; 96 GB GPU is under-utilized |
| #351 | askeladd  | surf-weight-50           | Loss balance (surf_weight 10→50) | Surface pressure dominates the metric; volume nodes still dominate gradient at 10× |
| #352 | edward    | smoothl1-surface         | Loss form (SmoothL1 beta=1 on surface) | Eval is MAE, MSE over-penalizes high-Re outliers — SmoothL1 closer to MAE-shaped gradient |
| #353 | fern      | warmup-cosine-1e3        | LR schedule (5-ep warmup + cosine to 1e-5, peak 1e-3) | Transformers like warmup; lets us safely raise peak LR |
| #354 | frieren   | slice-128-heads-8        | Slice/head count (slice 64→128, n_head 4→8) | Finer physics-aware slice tokens for 242K-node meshes |
| #355 | nezuko    | mlp-ratio-4              | MLP capacity (mlp_ratio 2→4) | Standard transformer recipe; only place with per-node nonlinearity — **sent back 04-27** (val_avg/mae_surf_p=129.24 @ ep13/50 timeout-cut; test NaN on cruise camber pressure; re-run with `nan_to_num` guard) |
| #356 | tanjiro   | ema-eval                 | Checkpoint selection (EMA 0.999 shadow for val + saved ckpt) | Free smoothing of noisy iterate; better best-checkpoint pick |
| #357 | thorfinn  | channel-weighted-loss    | Per-channel surface weights ([1,1,5] for Ux,Uy,p) | Up-weight `p` channel directly inside surface loss — aligns with ranking metric |

## Potential next directions (Round 2+)
- Combine the strongest two winners (capacity × loss alignment).
- SwiGLU MLP at matched param count vs. plain GELU `mlp_ratio=4` — modern transformer recipe; nezuko flagged it.
- Width × MLP ratio sweep (`n_hidden ∈ {160, 192}` × `mlp_ratio=4`) once mlp_ratio=4 is confirmed.
- Optimizer changes (Lion, Adan, SOAP).
- Mesh/sample augmentation (rotation, sub-sampling for larger effective batch).
- Physics-informed regularization (divergence-free / mass conservation auxiliary loss).
- Multi-scale slice attention (mix slice_num=32, 64, 128 across layers).
- Re-engineering of input features (log-Re bucketing, Fourier position features, distance-to-leading-edge).
- Per-domain conditioning (single vs raceCar tandem vs cruise tandem).
- Throughput: 30-min timeout currently fits ~12–13 epochs of 50; gradient accumulation, smaller batch with memory-light forward, or slice subsampling could buy more epochs.
- Train/val mismatch diagnostics: which split is currently the worst and why?
