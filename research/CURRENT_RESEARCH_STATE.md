# SENPAI Research State

- **Last updated:** 2026-05-15 (initial state for `icml-appendix-charlie-pai2i-24h-r2`)
- **Most recent research direction from human researcher team:** none yet (no open issues).
- **Current research focus and themes:** establish strong baseline numbers and probe orthogonal improvement axes in parallel.

## Branch context
This is `icml-appendix-charlie-pai2i-24h-r2`, the round-2 advisor branch for the Charlie launch. Local JSONL metrics only; no remote experiment tracking. Students branch off this branch and PRs target it.

## Round 1 plan (8 students, 8 hypothesis families)
Each student covers one orthogonal lever. Designed so winners can compound (loss + arch + schedule are largely additive).

| Student | Lever | Hypothesis |
|---------|-------|-----------|
| alphonse | Architecture width | Scale `n_hidden` 128 → 192 (∼2.25× params). |
| askeladd | Architecture depth | Scale `n_layers` 5 → 8 (∼1.6× params). |
| edward | Attention capacity | `slice_num` 64 → 192, `n_head` 4 → 8. Give physics attention more slots. |
| fern | Loss reformulation | Replace MSE with SmoothL1 (Huber) — robust to extreme high-Re pressure values. |
| frieren | Surface bias ramp | `surf_weight` 10 → 30 + per-channel pressure emphasis (2× on p). |
| nezuko | Position encoding | Add Fourier features on (x, z) coords (32 frequencies). |
| tanjiro | Schedule + warmup | 100 epochs with 5-epoch linear warmup + cosine, lr 5e-4 → 7e-4. |
| thorfinn | Throughput (bf16) | Add `torch.cuda.amp.autocast(bfloat16)` + grad scaler, batch 4 → 6. |

## Next research directions (candidates for round 2)
- Geometry-aware augmentation: reflect tandem foils about the chord line, AoA sign-flip with target sign-flip.
- Signed distance / per-node curvature / normals as extra input features.
- Two-stage residual: train volume head, then surface head conditional on volume features.
- Loss in log-magnitude space for pressure (sign + log|p|).
- GNN / FNO / GINO alternative architectures.
- Mixup or sample-level interpolation across nearby NACA codes.
- Test-time augmentation: average prediction over chord-reflected geometry.

Round 2 will pick from these once round 1 results identify which axes had movement. The researcher-agent has been dispatched in parallel to produce a richer literature-anchored list (file in `research/RESEARCH_IDEAS_2026-05-15_initial.md`).
