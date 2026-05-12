# SENPAI Research Results

<!-- Round 1 experiments are in progress — results will be logged here as PRs are reviewed. -->

## Round 1 — In Progress (2026-05-12)

8 PRs assigned to 8 students. Awaiting results.

| PR | Student | Hypothesis | Expected signal |
|----|---------|------------|-----------------|
| #1456 | alphonse | bf16-amp — more epochs via bf16 AMP | Throughput increase, same/better val metrics |
| #1457 | askeladd | surf-weight-50 — surf_weight 10→50 | Lower mae_surf_p, possibly higher mae_vol_p |
| #1458 | edward | wider-deeper — n_hidden=256, n_layers=6, n_head=8 | Better OOD generalization from more capacity |
| #1460 | fern | relative-l2-loss — per-sample relative L2 | Improved cruise and Re holdout splits |
| #1462 | frieren | warmup-cosine — 2-epoch LR warmup | Smoother convergence, better late-epoch metrics |
| #1467 | nezuko | more-slices-128 — slice_num 64→128 | Better OOD geometry routing |
| #1473 | tanjiro | huber-loss — Huber delta=0.5 | Stable training, less sensitivity to leading-edge outliers |
| #1479 | thorfinn | grad-clip-1 — gradient clipping norm=1.0 | Diagnostic: reveals if baseline was gradient-unstable |
