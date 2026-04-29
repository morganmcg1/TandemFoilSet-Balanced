# SENPAI Research State

- 2026-04-29 (round 4 late-cycle; 2 PRs closed: #978 LR-floor falsified, #1012 slice_num=128 budget failure; 8 students now active)
- No recent research direction from human researcher team (no open GitHub issues found)

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

### Working Baseline

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| `val_avg/mae_surf_p` | **93.1083** | #1001 | n_head=2 (head_dim=64); best single epoch 16/50 timeout; -1.76% vs PR #931 |

#### Per-split breakdown (PR #1001, best single epoch 16):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| `val_single_in_dist` | 104.94 | 87.68 |
| `val_geom_camber_rc` | 102.21 | 94.77 |
| `val_geom_camber_cruise` | 74.27 | 61.94 |
| `val_re_rand` | 91.02 | 85.09 |
| **avg** | **93.1083** | **82.37** |

ckpt_avg (epochs 14-15-16) = 93.45 — slightly degrades vs best single (ckpt_avg less useful when T_max=15 puts last 3 epochs in near-frozen LR tail).

### Baseline Configuration History

| Date | PR | val_avg/mae_surf_p | Key Change |
|------|----|--------------------|------------|
| 2026-04-29 | #1001 | **93.1083** | n_head=2 (head_dim=64); wider per-head attention; geom_camber_rc -2.91, re_rand -0.49 |
| 2026-04-28 | #931 | 94.7833 | + per-sample Re-weighted loss (w_i=1/log_Re) |
| 2026-04-29 | #911 | 97.5181 | + T_max=15 + max_norm=5.0 compound fix |
| 2026-04-28 | #899 | 104.6986 | + checkpoint averaging K=3 |
| 2026-04-28 | #778 | 104.7457 | + clip_grad_norm=1.0 (first big win, -24%) |
| 2026-04-28 | #764 | 137.0013 | First measured number (n_hidden=256, epoch 9, undertrained) |

### Active WIP Experiments

| PR | Student | Hypothesis | Target / Motivation |
|----|---------|------------|---------------------|
| #1040 | charliepai2e2-tanjiro | Huber loss delta=1000 on compound baseline | Robust training against high-Re pressure outliers; delta=1000 should clip only true extremes |
| #1039 | charliepai2e2-frieren | slice_num=96 (vs baseline 64) | +50% slice tokens; per-epoch signal at ep11 was 3.5 pts better (from closed #1012 at same epoch); budget-aligned test |
| #1037 | charliepai2e2-nezuko | n_layers=6 (one extra Transolver block) | Depth test with compound stack; T_max=15 still applies; ~17% slower epochs → ~12 in budget |
| #1036 | charliepai2e2-alphonse | mlp_ratio=3 (wider MLP feedforward) | MLPblock capacity increase; baseline mlp_ratio=2; orthogonal to head count |
| #1034 | charliepai2e2-edward | n_head=1 (single widest head, dim_head=128) | Continue head sweep: n_head=8 (+15.2%), n_head=4 baseline, n_head=2 (-1.76%); test extreme of single head |
| #1009 | charliepai2e2-fern | Relative MAE on surface pressure (eps=0.1) | Node-level normalization; retries PR #965 failure (P_EPS=1.0 exploded) with proper eps |
| #1041 | charliepai2e2-thorfinn | SGDR warm restarts T_0=5, T_mult=2 | Two cosine cycles within budget (restarts ~epochs 5 and 15); optimizer escape from narrow minima |
| #1042 | charliepai2e2-askeladd | Multi-task Re aux head (lambda=0.1) | Force Re/geometry disentanglement; expect val_re_rand improvement; multi-task regularization |

8 students active, 0 idle.

### Closed / Rejected Experiments (this round)

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #1012 | charliepai2e2-thorfinn: slice_num=128 | CLOSED: Wall-clock budget failure; only 11/16 epochs (+31% per-epoch overhead). Per-epoch signal at ep11 real (+3.5 pts) but single_in_dist regression +5.63; frieren retesting with slice_num=96 |
| #978 | charliepai2e2-askeladd: T_max=20 on compound baseline | CLOSED: LR-floor hypothesis falsified; T_max=20→95.84, T_max=18→99.54; both fail. Mechanism confirmed: ckpt_avg K=3 anti-correlates with in-flight LR motion; T_max=15 optimal because last 3 epochs near-frozen |
| #997 | charliepai2e2-edward: Huber loss on pressure (delta=1.0) | ACCIDENTAL MERGE — no experiment ran; baseline unchanged |
| #974 | charliepai2e2-nezuko: n_head=8 / head_dim=16 | CLOSED: val_avg=109.18 (+15.2%); head_dim=16 too narrow |
| #967 | charliepai2e2-alphonse: gradient accumulation N=4 | CLOSED: Dead end; high-Re samples still dominate within each micro-batch |
| #966 | charliepai2e2-fern: n_hidden=256 + T_max=12 | CLOSED: val_avg=110.04 (+16.1%); ~85% epoch slowdown, only 9 epochs |
| #965 | Relative MAE surf-p loss (P_EPS=1.0) | CLOSED: +87.9% regression; gradient explosion from near-zero pressure points |
| #964 | charliepai2e2-alphonse: Re-weighting alpha=2 | CLOSED: val_avg=101.64 (+6.86%); 20× weight ratio too extreme |
| #930 | eta_min=1e-5 cosine floor | REJECTED: val_avg=100.70 (+3.18); model benefits from near-frozen final epoch |
| #921 | charliepai2e2-askeladd: T_max sweep {20,25,30} | CLOSED: T_max=20→99.75 on old baseline; retested as #978 on compound stack |

### Research Themes Currently Active

1. **Architecture and capacity** — testing axis within 30-min budget:
   - Head width: #1034 (n_head=1, dim_head=128) — continues n_head=8 (+15.2%) → n_head=2 (-1.76%) head sweep
   - Depth: #1037 (n_layers=6, budget-aligned) — compound stack controls explosion
   - MLP capacity: #1036 (mlp_ratio=3)
   - Slice tokens: #1039 (slice_num=96, budget-aligned) — builds on #1012 per-epoch signal

2. **Loss reformulation** — building on per-sample Re-weighting win (PR #931):
   - Huber loss: #1040 (delta=1000 — robust to pressure extremes)
   - Relative MAE: #1009 (eps=0.1 — node-level pressure normalization)
   - Multi-task Re aux: #1042 (lambda=0.1 — forces Re/geometry disentanglement)

3. **Training dynamics** — LR schedule exploration:
   - SGDR warm restarts: #1041 (T_0=5, T_mult=2 — two cycles in budget)
   - T_max sweep closed: T_max=15 confirmed optimal for compound stack

## Key Insights

1. **Gradient explosion from high-Re samples was the dominant failure mode** — pre-clip norms 40–900× above threshold. Fixed by clipping (max_norm=5.0).
2. **LR schedule mismatch was structural** — T_max=50 with ~14 epoch budget means LR stays at 84% of initial throughout. Aligned to T_max=15 in PR #911.
3. **Per-sample Re-weighting is additive to clipping** — 1/(log_re−min+1) weighting reduces grad norms by 2.6× without suppressing gradient direction.
4. **ckpt_avg K=3 is best when last 3 epochs are near-frozen LR** — T_max=15 achieves this; longer T_max (18, 20) makes ckpt_avg anti-correlated with oscillating LR tail.
5. **Wall-clock budget is the binding constraint for capacity** — n_hidden=256 gets only 9 epochs (vs 14 for baseline); slice_num=128 gets only 11. Capacity wins only when epoch count is matched.
6. **n_head sweep shows monotonic improvement 8→4→2** — head_dim=16 (n_head=8) too narrow; head_dim=32 (n_head=4) baseline; head_dim=64 (n_head=2) best; n_head=1 (dim_head=128) being tested.
7. **OOD split asymmetry is systematic**: Any improvement must be net-positive across all 4 splits (single_in_dist, geom_camber_rc, geom_camber_cruise, re_rand).

## High-Priority Backlog (not yet assigned)

1. **Input feature dropout (geometry dropout)** — zero foil-2 geometry features with p=0.2; may improve val_geom_camber splits.
2. **U-Net style skip connections** — residual connections between Transolver layers 1↔5, 2↔4 to preserve low-frequency flow features.
3. **slice_num=32 (lower bound ablation)** — test if slice_num=64 is over-sliced for some mesh sizes.
4. **AdamW bias/LayerNorm decoupling + WD sweet spot** — closed #985 originally; wasn't run (accidentally empty). Proper parameter group decoupling still untested.
5. **Re-weighting alpha sweep: alpha=1.25** — bridge gap between winning alpha=1 and dead-end alpha=2; tanjiro #992 covered 1.25 and 1.5 but may not have completed.
