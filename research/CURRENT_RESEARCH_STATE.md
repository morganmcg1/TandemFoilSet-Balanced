# SENPAI Research State

- **Date:** 2026-05-12 22:15
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — triple compound winner)

**PR #1606 — fern EMA weights decay=0.999** (merged 2026-05-12 22:10, stacked on top of #1436):
- `val_avg/mae_surf_p = 92.3452` (epoch 17; vs 96.49 Huber+bf16 → −4.3%)
- `test_avg/mae_surf_p = 81.6297` (vs 86.33 → −5.4%)
- Config: EMA shadow (decay=0.999) + Huber loss (β=1.0) + bf16 autocast + NaN scoring fix, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- ~17 epochs / 30 min (bf16 = ~110 s/epoch with diagnostic pass)
- EMA-vs-live gap at epoch 17: live test=104.70, EMA test=81.63 (+28% better)

**Cumulative compounding (3 merges so far):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber | 96.49 | 86.33 | Smooth L1 loss → loss-shape MAE alignment |
| PR #1606 fern EMA | **92.35** | **81.63** | Weight averaging → reduces noise ball at eval |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1544 | `mlp_ratio=4` (2× MLP width) | Architecture (capacity) | WIP | tested on bf16-only baseline; may need rebase |
| askeladd | #1427 | `surf_weight=30` (3×) | Loss weighting | WIP (rebasing) | pinged with Huber+bf16 baseline; rebase requested |
| edward | #1546 | `n_layers=8` (Transolver paper default) | Architecture (depth) | WIP | tested on bf16-only baseline; may need rebase |
| fern | #1626 | EMA without diagnostic pass (more epochs) | Training efficiency | WIP | −25 s/epoch → 21-22 epochs vs 17; EMA mechanism confirmed |
| frieren | #1442 | Wider `n_hidden=192` | Architecture (width) | WIP (rebased 21:13) | rerun at bs=4 on bf16; mechanism test clean |
| nezuko | #1445 | Per-channel surf weights `(0.5, 0.5, 2.0)` | Loss / metric alignment | WIP (rebasing) | pinged with Huber+bf16 baseline; rebase requested |
| tanjiro | #1534 | Gradient clipping `max_norm=1.0` | Gradient stability | WIP (rebased 21:18) | v2 rerun on bf16+Huber; 100% steps clipped in fp32 run |
| thorfinn | #1629 | Dropout=0.1 | OOD regularization | WIP | H8 from round-1 pool; first regularization direction tested |

**Critical baseline note**: All PRs must now beat `val_avg/mae_surf_p < 92.3452` to merge directly. PRs started against the bf16-only (109.29) or Huber+bf16 (96.49) baseline will need rebase + retest if they fall between those thresholds.

## Closed hypotheses

- **slice_num=96** (#1550, thorfinn) — val=120.69, +10.4% worse than bf16-only baseline. Two-run confirmation. Two mechanisms: +20% per-epoch cost (15 epochs vs 18), and slower per-step convergence as 96-group slice partition needs more gradient steps. OOD splits worse, not better. slice_num=64 is well-calibrated for our 30-min budget.
- **slice_num=128** (#1451, thorfinn) — confounded by bs=2 OOM. Superseded.
- **batch_size=8** (#1447, tanjiro) — dataloader bottleneck; no per-epoch speedup. Closed.
- **lr=1e-3 + warmup** (#1430, edward) — cosine T_max mismatch; schedule under-annealed. Closed.

## Key observations

1. **bf16 is the dominant lever** — 18 epochs/30 min vs 11-14 for fp32. Merged.
2. **Huber loss is the second lever** — loss-shape alignment with MAE metric; ~4 epochs of effective speedup vs MSE. Merged.
3. **EMA weight averaging is the third lever** — reduces the SGD noise ball at eval; EMA consistently outperforms live weights from epoch 9+ (epoch 17: −25 MAE). Merged.
4. **All three stack orthogonally** — compounding from val~160 to val=92.35 confirms each lever is mostly independent. The remaining headroom from these three stacked should be explored before declaring a local minimum.
5. **Gradient norms are massive without clipping** — 5250/5250 steps clipped at max_norm=1.0, max norm 837. Acts as full gradient normalization. Tanjiro's retest on bf16 will show whether smoother trajectory compounds with the existing stack.
6. **Per-epoch throughput is king** — any lever that doesn't speed up wall-clock per-epoch or improve sample-efficiency struggles. Architecture levers (wider, deeper, more slices) face this headwind.
7. **val_single_in_dist is hardest** (~112-175 MAE across runs). OOD camber_cruise is easiest (58-87 MAE). In_dist being hardest is likely extreme-Re / extreme-p samples in the in-distribution set, not an overfitting artifact.

## Potential next directions (post current round)

- **EMA decay sweep** (0.9995) — if ema-no-diag shows continued improvement, test longer half-life
- **Grad-clip + EMA compounding** — tanjiro's v2 will reveal if both stack
- **Architecture at EMA baseline** — mlp_ratio=4 (alphonse) and n_layers=8 (edward) may compound once they rebase to EMA+Huber+bf16
- **LR schedule alignment** — cosine T_max matched to actual epoch budget (~20 epochs for EMA-no-diag)
- **Re-conditioning** — explicit Re-aware embeddings or log-Re positional encoding (OOD re_rand split underperforms)
- **Surface-aware decoder / dual-head** — separate volume and surface heads may improve surface-pressure alignment
- **Spectral / Fourier neural operator hybrids** — fresh architecture direction if attention-based plateau
- **Test-time augmentation** using physical symmetries (mirroring flow domain)
