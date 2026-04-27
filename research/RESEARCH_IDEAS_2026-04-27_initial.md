# Round-1 hypotheses — willow-pai2d-r1 (2026-04-27)

The default Transolver baseline trains a per-channel-equal MSE loss in
normalized space, while only **surface pressure MAE** is the ranking metric.
That mismatch is the single biggest lever for round 1: any change that pushes
the optimization closer to the evaluation metric should help directly.

Round 1 covers eight low-complexity, mostly orthogonal interventions plus an
explicit baseline. They are designed to be largely additive — multiple winners
can stack in later rounds.

## Slot 1 — `alphonse/baseline-default`

**Run the default Transolver as-is** to establish a reference number for
round 1. No code changes; just run with the canonical command.

## Slot 2 — `askeladd/pressure-channel-loss-weight`

The MSE term currently averages `Ux`, `Uy`, `p` equally per node. The metric
only cares about `p`. **Weight pressure 5x in the per-node MSE** so the
gradient pushes harder on the channel that defines our score.

```python
ch_w = torch.tensor([1.0, 1.0, 5.0], device=sq_err.device)        # Ux, Uy, p
sq_err_w = sq_err * ch_w                                          # [B, N, 3]
vol_loss = (sq_err_w * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1) / 3
surf_loss = (sq_err_w * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1) / 3
```

## Slot 3 — `edward/huber-loss`

MSE optimizes squared error, but our metric is MAE. **Replace MSE with
SmoothL1 (Huber, beta=1.0)** in normalized space. This better aligns the
training objective with the evaluation metric and is more robust to the
high-Re outliers that dominate the pressure tail.

## Slot 4 — `fern/wider-deeper-transolver`

The default model is small (~2.5M params, hidden=128, 5 layers). With 96 GB
VRAM per GPU there is room to scale. **`hidden=192`, `n_layers=6`, `n_head=6`,
`slice_num=96`, `mlp_ratio=2`** — keep batch_size=4 to stay within VRAM.

## Slot 5 — `frieren/lr-warmup-and-higher-peak`

No warmup currently. Cosine starts at `5e-4` and decays. **Add a 5-epoch
linear warmup from `1e-5` to `1e-3`, then cosine decay over the remaining 45
epochs to 0.** Higher peak with proper warmup is a well-known recipe for
transformer-style models trained on small datasets.

## Slot 6 — `nezuko/ema-and-grad-clip`

Two stability/generalization tricks:

1. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before
   each `optimizer.step()`.
2. Maintain an EMA shadow copy of the weights with decay `0.9999`. Validate
   and run final test on EMA weights, save EMA weights as the artifact.

These are nearly free at training time and routinely give ~1–3% wins on
small-data transformers.

## Slot 7 — `tanjiro/fourier-features-on-positions`

Inputs include node `(x, z)` at dims 0–1. Transformers struggle with raw
high-frequency spatial inputs. **Concatenate Fourier features
`[sin(2^k π x), cos(2^k π x), sin(2^k π z), cos(2^k π z)] for k=0..7`** to the
24-dim feature vector before the encoder MLP. Update `space_dim` /
`X_DIM`-style accounting accordingly. This trick helps NeRF-style coordinate
MLPs and should help the slice-attention encoder pick up sharper spatial
gradients near the foil surface where pressure is most variable.

## Slot 8 — `thorfinn/surf-weight-sweep`

Current `surf_weight=10`. A short, three-value sweep to localize the optimum
without spending too many slots:

- 15
- 25
- 40

Run all three under `--wandb_group surf-weight-sweep` so they're grouped.
Pick the best `val_avg/mae_surf_p` for the merge decision.
