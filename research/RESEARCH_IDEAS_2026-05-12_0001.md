# Round 1 Research Ideas — 2026-05-12

Fresh research track. Goal: establish a baseline number on this hardware/time budget (30-min training cap) and screen 7 orthogonal single-variable interventions. Each hypothesis is intentionally isolated so we can attribute any movement on `val_avg/mae_surf_p` cleanly.

Each PR runs the default screening recipe (`--epochs=50`, `SENPAI_TIMEOUT_MINUTES=30`, AdamW, CosineAnnealing). Whatever epoch count the runtime actually reaches becomes the screening signal.

## H1 — Baseline reference (alphonse, slug `baseline-ref`)
- **Why:** No prior PRs on this branch — we need a stable anchor for `val_avg/mae_surf_p` at the 30-min cap before any comparison is meaningful.
- **Change:** None. Default `train.py`.

## H2 — Surface weight 20 (askeladd, slug `surf-weight-20`)
- **Why:** `surf_weight=10` is the default and was never tuned for this branch. The primary ranking metric is surface pressure; weighting surface MSE harder relative to volume should directly move the metric.
- **Change:** `Config.surf_weight: float = 10.0` → `20.0` in `train.py`.

## H3 — Huber loss (edward, slug `huber-loss`)
- **Why:** Per-sample y std spans an order of magnitude even within a split (50 → 2000+) — high-Re samples dominate MSE gradients. Huber (smooth L1) clips outlier gradients while remaining squared near zero, which often improves median accuracy on heavy-tailed regression targets.
- **Change:** Replace the `sq_err = (pred - y_norm) ** 2` MSE losses (both `vol_loss` and `surf_loss`) with `F.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')` inside the training step and inside `evaluate_split`. Apply identically (masked sum / count). Keep the same surf_weight multiplier.

## H4 — lr 1e-3 with linear warmup (fern, slug `lr1e3-warmup-cosine`)
- **Why:** Default `lr=5e-4` with no warmup; transformer-style models often prefer higher peak lr with a short warmup, especially when the schedule has to fit in a fixed wall clock. Faster convergence per epoch matters at the 30-min cap.
- **Change:** Set `Config.lr: float = 5e-4` → `1e-3`. Replace `CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)` with a `SequentialLR` of `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=3)` followed by `CosineAnnealingLR(T_max=MAX_EPOCHS - 3, eta_min=0)`.

## H5 — Weight decay 5e-4 (frieren, slug `wd5e-4`)
- **Why:** Stronger weight decay tends to help OOD generalization, which is exactly what the two camber holdouts (`val_geom_camber_rc/cruise`) measure. Cheap to test.
- **Change:** `Config.weight_decay: float = 1e-4` → `5e-4` in `train.py`.

## H6 — slice_num 128 (nezuko, slug `slice128`)
- **Why:** Physics attention groups N mesh nodes into `slice_num` learned slice tokens. Default 64 may be too few when mesh size is up to 242K (cruise samples). More slices = finer physical structure represented; should especially help cruise splits.
- **Change:** `slice_num=64` → `128` in `model_config` in `train.py`. Watch VRAM peak; if it OOMs, drop `batch_size` to 2 temporarily.

## H7 — hidden 192 (tanjiro, slug `hidden192`)
- **Why:** Default model is only ~0.7M params — extremely small relative to the variability across three physical domains. Widening hidden from 128 to 192 grows it to ~1.5M, still well within 96GB and ~30-min budgets.
- **Change:** `n_hidden=128` → `192` in `model_config` in `train.py`. Keep `n_head=4` (so dim_head=48); `slice_num=64`. Confirm no OOM.

## H8 — Unified positional encoding (thorfinn, slug `unified-pos`)
- **Why:** `Transolver` already supports `unified_pos=True` with a `ref^3`-feature grid encoding, but the default is `False` (raw (x,z) positions go through the preprocess MLP). Mesh nodes are irregular; a regularized fixed-grid encoding can give a stronger spatial inductive bias.
- **Change:** Set `unified_pos=True, ref=8` in `model_config` in `train.py`. Note: `space_dim=2` here; verify `Transolver.preprocess` input dim path. If the path requires a 2D grid (not 3D `ref**3`), modify `Transolver.__init__` to use `ref**space_dim` instead of `ref**3` so the math matches the 2D mesh.

## Expected magnitudes

- H2, H5, H6: small (~1-5%) improvements likely if any.
- H4: high variance — could regress if warmup misspecified; could win 5-10% if it converges further per epoch.
- H7: modest (~3-10%) improvement expected from capacity, but slower per epoch may eat into budget.
- H3: large effect on per-channel MAE balance possible; surface p could go either way.
- H8: untested direction — exploratory.
- H1: control.
