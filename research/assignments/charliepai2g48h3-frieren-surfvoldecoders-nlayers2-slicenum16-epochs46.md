# Assignment: frieren — Specialized Surface/Volume Decoders (architectural pivot)

**Branch (use exactly):** `charliepai2g48h3-frieren/surfvoldecoders-nlayers2-slicenum16-epochs46`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

Aux-head result (#2871): aux head at weight=1.0 learned surface structure (aux_surf_loss tracks main surf_loss tightly) but did NOT improve the shared encoder. Root cause: `surf_weight=10` already provides 10× surface gradient weight — adding another surface-only loss is redundant gradient signal in the same direction, not complementary regularization.

Your own recommendation: "Feature-routing changes that SEPARATE surface and volume processing, rather than adding parallel heads on the same features."

**Hypothesis:** Replacing the SHARED final decoder with TWO specialized decoders — one for surface nodes, one for volume nodes — allows the readout function (features → physical quantities) to specialize per node type without shared parameters. Surface nodes need different readout from volume nodes (surface pressure field has different spatial correlations, sharper gradients at leading/trailing edges). Specialized decoders give the architecture this inductive bias directly — without redundant gradient signal.

Mechanistically DIFFERENT from the aux head:
- **Aux head:** adds a second surface prediction in parallel; gradient flows back through both decoders.
- **Specialized decoders:** REPLACES the shared decoder with two heads gated by mask. Each decoder receives gradient ONLY from its own node type. Gradient on the shared encoder is the same composition as baseline, but decoder weights are specialized.

## Implementation

Find the final decoder in the model (likely an `nn.Linear` or small MLP mapping `n_hidden → len(output_fields)` in the Transolver model class). Replace it with two specialized decoders:

**Add a config flag:**
```python
specialized_decoders: bool = False   # Use separate surf/vol decoders instead of shared
```

**In `__init__`:**
```python
if cfg.specialized_decoders:
    self.surf_decoder = nn.Sequential(
        nn.Linear(self.n_hidden, self.n_hidden),
        nn.GELU(),
        nn.Linear(self.n_hidden, len(output_fields)),
    )
    self.vol_decoder = nn.Sequential(
        nn.Linear(self.n_hidden, self.n_hidden),
        nn.GELU(),
        nn.Linear(self.n_hidden, len(output_fields)),
    )
else:
    # original decoder unchanged
```

**In `forward` (where `h` → `pred`):**
```python
if self.specialized_decoders_enabled:
    surf_pred = self.surf_decoder(h)     # (B, N, n_outputs) — all nodes
    vol_pred = self.vol_decoder(h)       # (B, N, n_outputs)
    surf_m = surf_mask.unsqueeze(-1).float()
    vol_m = vol_mask.unsqueeze(-1).float()
    pred = surf_pred * surf_m + vol_pred * vol_m
else:
    pred = self.decoder(h)               # original path unchanged
```

The mask ensures surface nodes use ONLY `surf_decoder` and volume nodes use ONLY `vol_decoder`. Backprop through the masked sum routes surface gradients through `surf_decoder` only, volume gradients through `vol_decoder` only.

**Eval uses the same logic** — no changes to val/test eval functions.

**Important:** If the current decoder is a single Linear, still use the 2-layer MLP form (`n_hidden → n_hidden → n_outputs`) for BOTH decoders to give them capacity to specialize.

**At training end, please report:**
1. The L2 norm of `surf_decoder` final-layer weights vs `vol_decoder` final-layer weights
2. The cosine similarity between the two decoders' final-layer weight matrices (tells us whether the decoders actually specialized or converged to the same function)

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name surfvoldecoders-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --specialized_decoders true
```

## Baseline to beat

PR #2468 (n_layers=2 + slice_num=16 + epochs=46, **Lion + L1 + cosine, surf_weight=10, shared decoder**):

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
- Two-decoder forward ~doubles decoder cost, but decoder is <1% of total compute. Expect <5% slowdown (same as your aux-head observation: 36-38s vs baseline 35.1s/epoch). Report honestly if >40s/epoch.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Include the **per-split val and test table** (all 4 splits) and the decoder weight cosine similarity check.

## Suggested follow-ups

- **If wins:** test deeper specialized decoders (3 layers each) — more capacity for specialization.
- **If neutral (within ±1% val):** report cosine similarity. If decoders converge to ~same weights, mechanism is dead at this readout depth. Try specialization at deeper transformer layer.
- **If loses:** try single-Linear specialized decoders (less capacity). If that also fails, pivot to data-side levers.

## EV assessment

**Medium-high.** Architectural pivot family is wide open. Specialized decoders is the simplest non-aux-head change targeting surface-vs-volume asymmetry without redundant gradient signal. Implementation ~20 lines. Worst case: ~0% to mild +1% loss (decoders converge to same function). Best case: 1-3% val improvement from readout specialization.
