# Assignment: frieren — Auxiliary Surface Decoder Head (architectural pivot)

**Branch (use exactly):** `charliepai2g48h3-frieren/auxsurfhead-nlayers2-slicenum16-epochs46`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

The current architecture treats surface and volume nodes uniformly through Transolver — all nodes flow through the same attention/MLP stack, and the surface vs volume separation only happens in the loss (via `vol_mask` / `surf_mask`).

**Hypothesis:** Adding a small parallel **auxiliary surface-only decoder head** that operates on the same final features but has its own MLP layers specifically tuned for surface prediction will:

1. Act as **multi-task regularization** — gradient signal from the aux head shapes the shared encoder to produce features that work well for surface prediction specifically.
2. **Directly target the primary metric** — val/test_avg/mae_surf_p is computed on surface nodes only; an aux head dedicated to surface gives the architecture a focused capacity tap.
3. **Reuse all baseline machinery** — Lion, L1, cosine, surf_weight=10 all unchanged. The aux head adds ~5-20K params (negligible vs 361K baseline).

This is an **architectural pivot** orthogonal to all closed axes (capacity, schedule, loss-weight, loss-form, optimizer, post-hoc averaging, schedule-floor). It's a direct attack on the primary metric via dedicated decoder capacity.

## Why this is the right next bet

All closed axes from this launch:
- **Capacity** (n_hidden, mlp_ratio, depth) — REFUTED
- **Schedule shape** (warmup HEAD, truncated cosine TAIL) — REFUTED
- **Schedule floor** (eta_min=5e-6 #2861) — STRUCTURALLY DEAD for Lion (sign-update oscillation)
- **Loss-weight** — SATURATED
- **Loss-form** (Huber d=5.0 and d=0.1) — CLOSED
- **Optimizer** (AdamW lr=3e-4 and 1e-3) — CLOSED at 30-min budget
- **Post-hoc averaging (SWA, your work #2857)** — CLOSED for 46-epoch schedule (still-descending trajectory invalidates flat-region premise)

What remains: **architectural changes** and **data-side levers**. Aux surface head is the cleanest architectural intervention — small code change, focused on the primary metric, well-validated in multi-task ML literature.

## Implementation

In `train.py` (or model file, wherever `model_config` lives):

Add to Config:
```python
aux_surf_head: bool = False           # Enable auxiliary surface decoder head
aux_surf_weight: float = 1.0          # Loss weight for aux head (applied to its surface loss only)
aux_surf_hidden: int = 64             # Hidden dim for the aux head MLP
```

In the model definition (likely a Transolver model class), add an aux decoder applied to final features:
```python
if cfg.aux_surf_head:
    # Aux head: takes the same final features that feed the main decoder,
    # produces a parallel surface-only prediction.
    self.aux_surf_decoder = nn.Sequential(
        nn.Linear(self.n_hidden, cfg.aux_surf_hidden),
        nn.GELU(),
        nn.Linear(cfg.aux_surf_hidden, len(output_fields)),  # 3 outputs: Ux, Uy, p
    )

def forward(self, x, ...):
    # ... existing forward returning final features `h` and main_pred = self.decoder(h) ...
    if self.aux_surf_head_enabled:
        aux_pred = self.aux_surf_decoder(h)
        return main_pred, aux_pred
    return main_pred, None
```

In the training loop:
```python
main_pred, aux_pred = model(x, ...)

# Main loss (unchanged)
abs_err = (main_pred - y_norm).abs()
vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss

# Aux head loss (surface only)
if cfg.aux_surf_head and aux_pred is not None:
    aux_abs_err = (aux_pred - y_norm).abs()
    aux_surf_loss = (aux_abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    loss = loss + cfg.aux_surf_weight * aux_surf_loss
```

**Eval uses ONLY the main head** — aux head is a training-time regularizer only. Do not modify val/test eval functions.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name auxsurfhead-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --aux_surf_head true \
  --aux_surf_weight 1.0 \
  --aux_surf_hidden 64
```

## Baseline to beat

PR #2468 (n_layers=2 + slice_num=16 + epochs=46, **Lion + L1 + cosine, surf_weight=10**):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **35.256** |
| val_single_in_dist | 36.476 |
| val_geom_camber_rc | 48.297 |
| val_geom_camber_cruise | 18.326 |
| val_re_rand | 37.923 |
| **test_avg/mae_surf_p** | **30.245** |

## Per-run constraints

- Hard timeout: 30 minutes per training execution (`SENPAI_TIMEOUT_MINUTES=30`).
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT add/import/configure/query/log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Terminal result format

Post a comment with a single-line `SENPAI-RESULT` marker:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Also log the **aux head's training loss** in metrics.jsonl so we can see whether the aux head is actually learning surface structure (i.e., aux_surf_loss should decrease over training).

## Suggested follow-ups

If aux_surf_weight=1.0 wins: test `aux_surf_weight=2.0` (stronger regularization) or `aux_surf_hidden=128` (more aux capacity)
If aux_surf_weight=1.0 is neutral: try `aux_surf_weight=0.5` (weaker, less interference) and `aux_surf_weight=2.0` (stronger)
If aux_surf_weight=1.0 loses: try gating aux at different layers (apply aux head to intermediate features, not just final)

## EV assessment

**Medium-high.** Multi-task auxiliary heads are reliable regularizers in dense prediction tasks. The hypothesis directly targets the primary metric (surface p MAE) via dedicated capacity. Implementation is straightforward (~30 lines), zero per-epoch overhead (~5-20K extra params on 361K baseline), no interference with eval. Best case: +0.5-1.5% val on top of strong baseline. Worst case: ~0% delta (regularization neutral). Reasonable downside: <+2% if aux gradient noise destabilizes shared encoder.
