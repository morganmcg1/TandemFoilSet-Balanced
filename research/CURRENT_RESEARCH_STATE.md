# SENPAI Research State

- **Date**: 2026-04-27
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet (fresh launch).

## Current research focus

This is the first round on the willow-pai2d-r1 advisor branch. The Transolver
baseline trains MSE on `[Ux, Uy, p]` while the ranking metric only scores
**surface pressure MAE**. The opening theme is therefore *aligning the
training objective with the evaluation metric* — pressure-channel weighting,
Huber loss, surface weight tuning — paired with a few well-known
generalization levers (model scale, LR warmup, EMA + grad clip, Fourier
features) so multiple ideas advance in parallel.

## Themes in flight

| Theme | Slot |
|---|---|
| Reference baseline (clean number to compare to) | alphonse |
| Loss / metric alignment | askeladd, edward, thorfinn |
| Capacity scaling | fern |
| Optimization & schedule | frieren |
| Stability / regularization | nezuko |
| Spatial inductive bias | tanjiro |

## Potential next directions (round 2+)

- **Stack winners** from round 1 (e.g. Huber + surf-weight + EMA in one PR)
  if multiple beat baseline.
- **Test-time augmentation**: mirror-flip x to exploit symmetry on cruise foils
  (raceCar is inverted and asymmetric in z so this is split-specific).
- **Per-domain or per-Re curriculum / sampling**: weight high-Re samples
  higher, since pressure variance scales with Re² and high-Re drives the tail
  of `mae_surf_p`.
- **Surface-only auxiliary head**: a small MLP head that consumes surface
  node features and outputs a refined `p` correction, trained on surface loss
  only. Decouples surface fidelity from volume reconstruction.
- **Mesh-aware encoders**: kNN message passing over the local mesh (GAT or
  PointNet-style) before the slice attention, so each node sees its local
  geometry instead of relying purely on dsdf/saf features.
- **Gradient-based features**: precompute |∇x|, |∇z| of the dsdf field per
  sample and feed as extra channels — encodes surface curvature.
- **Loss in physical units** rather than normalized — currently MSE is in
  the `(y - mean)/std` space, which equally weights all magnitude regimes;
  switching to per-domain or per-Re-bucket normalization could change which
  samples dominate.
- **Bigger architectural swings** if simple levers plateau: GNO/GNOT-style
  geometry-aware attention, Galerkin Transformer, FNO with neural ops on the
  mesh, or a hierarchical multi-scale slice transformer.

## Notes

- 30-minute training cap per run (`SENPAI_TIMEOUT_MINUTES=30`), 50 epochs
  default. Several round-1 interventions (e.g. larger model, more slices) may
  not reach 50 epochs; that's expected — students should report `best epoch`
  alongside metrics so we know whether timeout was binding.
- One hypothesis per PR. Sweeps are allowed when localized (e.g. surf-weight
  ∈ {15, 25, 40}) under a single `--wandb_group`.
