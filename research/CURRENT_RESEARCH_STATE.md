# SENPAI Research State

- 2026-05-12 — willow-pai2g-48h-r1 launch
- No directives from human researcher team yet.
- No baseline run on `icml-appendix-willow-pai2g-48h-r1` yet; the unmodified Transolver in `train.py` is the implicit starting point (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; AdamW lr=5e-4, wd=1e-4, batch=4, surf_weight=10, cosine over 50 epochs, hard 30-min wall clock per run).

## Research focus
Round 1 establishes the lay of the land across orthogonal levers for `val_avg/mae_surf_p` and the matching `test_avg/mae_surf_p`. Each hypothesis targets a different axis so we can pick the best directions for round 2 without redundant work.

## Round 1 hypothesis matrix (8 students)
| Student | Slug | Lever | Predicted delta on val_avg/mae_surf_p |
|---------|------|-------|---------------------------------------|
| alphonse | `lr-warmup-1e-3` | Optimization schedule | -5 to -10% |
| askeladd | `wider-hidden-192` | Width capacity | -3 to -7% |
| edward | `more-slices-128` | Physics-token resolution | -2 to -6% |
| fern | `deeper-7-layers` | Depth capacity | -3 to -6% |
| frieren | `surf-weight-25` | Loss balance to primary metric | -2 to -6% |
| nezuko | `fourier-pos-features` | Input representation | -5 to -10% |
| tanjiro | `bf16-batch-8` | Throughput → more effective epochs | -3 to -7% |
| thorfinn | `lion-optimizer` | Optimizer geometry | -2 to -5% |

## Potential round-2 directions (informed by round-1 outcomes)
- If width/depth helps: scale further and combine with throughput-enabling changes (bf16 + larger batch).
- If surface-weight helps: try per-channel scaling (push p relative to Ux/Uy) and surface-aware loss masks (e.g. log-loss or huber).
- If Fourier features help: try learned Fourier (NeRF-style) and combine with unified-pos grid.
- If slice_num helps: vary heads and dim_head together; try slice annealing (start small, grow).
- Optimizer winners motivate trying schedule-free, AdEMAMix, or curvature-aware variants.
- Cross-split disagreements (Re vs geom-camber tracks) → consider domain-conditional heads or auxiliary classifier features.
- Plateau Protocol: switch tier — multiscale physics (graph-conv local mixing + Transolver global), spectral decoders, or PDE-residual auxiliaries.
