# Research Ideas — 2026-05-13 15:00

Generated for round-44 reassignment: idle students edward, frieren, tanjiro.
Baseline: val_avg/mae_surf_p = 42.3455 (PR #2307), test_avg = 38.5059.
Advisor config: slice_num=24, n_hidden=96, n_layers=4, n_head=2, LayerScale γ=1e-4, L1, warmup-3-cosine.

---

## Hypothesis 1 — mlp_ratio 2→1 (FFN contraction) [assign to edward]

### What it is
Reduce the Transolver feed-forward network expansion ratio from mlp_ratio=2 to mlp_ratio=1, halving the FFN intermediate dimension from 192 to 96.

### Mechanism
At n_hidden=96, mlp_ratio=2 means each transformer block's FFN expands to 192 and back. Reducing to ratio=1 (FFN stays at 96, no expansion) cuts the two FFN linear layers roughly in half. The FFN dominates per-epoch wallclock because it operates on all N nodes × all slices. Budget-bound wins are confirmed 2-for-2 (depth PR #2268, width PR #2290): the primary bottleneck is epochs-within-30-min. mlp_ratio=4 was a hard budget CLIFF (+42.5% LOSS); the downward direction has NEVER been probed.

Expected per-epoch saving: ~15-20% (FFN is two n_hidden→mlp_ratio*n_hidden linear layers per block × n_layers=4; ratio=1 vs ratio=2 halves both). At ~30.8 s/epoch baseline, saving ~5-6 s/epoch → ~64 epochs vs ~58, gaining ~6 extra training epochs in the 30-min cap.

### Predicted delta
Win: -2% to -5% on val_avg/mae_surf_p. Per-split: uniform improvement consistent with budget-bound pattern (all 4 splits improved under depth-down and width-down). The OOD split val_geom_camber_rc (60.83) is the primary bottleneck — more epochs may help routing generalization.

### Freshness
mlp_ratio downward direction has NEVER been tested in any round. mlp_ratio=4 was tested (budget cliff LOSS). mlp_ratio=2 is the baseline. mlp_ratio=1 is a fresh axis.

### Code change (1 line in train.py model_config dict)
```python
# Before
model_config = dict(
    ...
    mlp_ratio=2,
    ...
)

# After
model_config = dict(
    ...
    mlp_ratio=1,
    ...
)
```

### Failure mode
If FFN capacity is the representational bottleneck (not budget), then val loss will stagnate at the same epoch count relative to n_hidden=96/mlp_ratio=2 — the extra epochs won't help. Observable: if best_epoch/total_epochs ratio stays the same as baseline and per-epoch val loss does NOT improve in absolute terms, this is capacity-limited rather than budget-limited. In that case, close and note "FFN capacity is load-bearing at n_hidden=96."

### Reproduce command
```bash
cd target && python train.py --agent edward --experiment_name "edward/mlp-ratio-1" --epochs 70
```

---

## Hypothesis 2 — surf_weight 10→20 (loss gradient rebalancing) [assign to frieren]

### What it is
Double the surface loss weight in the combined training objective from surf_weight=10 to surf_weight=20, more directly aligning the gradient signal with the primary evaluation metric mae_surf_p.

### Mechanism
The training loss is `L = L1_vol + surf_weight * L1_surf`. The primary metric is mae_surf_p (surface pressure only). At surf_weight=10, the gradient budget is split between the full-volume field (74K-242K nodes, mostly volume) and the surface (relatively few surface nodes). Increasing to 20 doubles the gradient emphasis on surface accuracy. The prior surf_weight=30 LOSS occurred under a substantially different model config (pre-slice_num=24, larger n_hidden, pre-LayerScale) with a much heavier per-block budget — gradient imbalance was more disruptive there. The current lean config (n_hidden=96, n_layers=4, slice_num=24) is more regularized and may tolerate higher surface emphasis without the instability that caused the round-N surf_weight=30 LOSS. surf_weight=20 is the exact intermediate value between the closed 10 (baseline) and closed 30 (LOSS) — it has never been tried.

### Predicted delta
Win: -2% to -4% on val_avg/mae_surf_p, especially val_geom_camber_rc (OOD bottleneck, 60.83) and val_geom_camber_cruise (27.65 — the cruise OOD split). The val_single_in_dist (35.48) may be less affected because it is in-distribution. Volume metrics (mae_vol_*) may degrade slightly — this is acceptable given the primary metric is surface-only. Per-epoch cost: unchanged (same forward/backward pass, just a scalar multiplier change).

### Freshness
surf_weight=20 is genuinely fresh — the only two tested values are 10 (baseline) and 30 (LOSS round-N). This is a 1-line change at the exact midpoint of the closed range.

### Code change (1 line in train.py)
```python
# Before
surf_weight = 10.0

# After  
surf_weight = 20.0
```

### Failure mode
If the vol-surf gradient imbalance is model-config-agnostic (i.e., still destructive even at the leaner current config), then val_avg will increase (+loss) with deterioration especially on val_re_rand (multi-domain) and val_single_in_dist. Observable: if val_single_in_dist improves but val_re_rand and val_avg worsen, this is the bimodal in-dist/OOD failure mode (8× confirmed closed class). Close immediately in that case.

### Reproduce command
```bash
cd target && python train.py --agent frieren --experiment_name "frieren/surf-weight-20" --epochs 70
```

---

## Hypothesis 3 — Separate surface / volume output heads (bold swing) [assign to tanjiro]

### What it is
Replace the single final output projection with two independent linear heads — one dedicated to surface nodes and one to volume nodes — selected by the `is_surface` mask at inference. The shared encoder/attention backbone is unchanged; only the final decoder layer is bifurcated.

### Mechanism
The current model uses a single output head that must simultaneously reconstruct surface pressure (the primary metric) and the full volume field (secondary). These two tasks have different difficulty profiles: surface nodes have sharp boundary-layer gradients and strong geometry dependence; volume nodes are smoother and less geometry-sensitive. A single linear map from the same hidden representation must serve both regimes. The primary training objective is mae_surf_p (surface pressure MAE), but parameter allocation in the final head is blind to this asymmetry.

Separating into `surface_head(hidden)` and `vol_head(hidden)` allows each head to specialize. The surface head can learn a surface-specific projection (steeper pressure gradients, boundary conditions); the volume head handles the bulk field. The two heads share the full encoder, so the shared representation is still trained jointly — the bifurcation only affects the final linear map (~n_hidden × 3 parameters per head, negligible param count increase from 576K to ~576K + 96×3 = 576,288 total, effectively no param change).

The `is_surface` mask is already available in `pad_collate` output and used in scoring. The only change is in the forward pass of the model output layer and the loss computation.

### Code sketch (~15 lines in train.py)
```python
# In model __init__ (replace single output_fc):
self.surface_head = nn.Linear(n_hidden, out_dim)
self.vol_head = nn.Linear(n_hidden, out_dim)

# In model forward (replace single output projection):
# hidden: [B, N, n_hidden]
# is_surface: [B, N] bool, passed through model kwargs
pred_surf = self.surface_head(hidden)   # [B, N, 3]
pred_vol = self.vol_head(hidden)         # [B, N, 3]
# Select per-node:
is_surf_expanded = is_surface.unsqueeze(-1).expand_as(pred_surf)  # [B, N, 3]
pred = torch.where(is_surf_expanded, pred_surf, pred_vol)          # [B, N, 3]

# In the training loop, pass is_surface into the model forward call.
# Loss and metrics remain unchanged — pred is still [B, N, 3] in normalized space.
```

The `is_surface` flag is already normalized and padded by `pad_collate` — no data loader changes needed.

### Predicted delta
Win: -3% to -8% on val_avg/mae_surf_p if the model benefits from head specialization. The primary bottleneck val_geom_camber_rc (60.83) involves unseen front-foil camber — OOD geometry — where surface boundary conditions are most different. A dedicated surface head may better generalize surface pressure patterns independently of the volume field. Failure risk: if the shared representation requires the surface head to regularize via the volume-field gradient (physics consistency), decoupling the heads could increase surface error. Observable: vol metrics stable or degraded; surface metrics improved if hypothesis holds.

### Freshness
Architectural bifurcation of the decoder has never been tested in any round. This is the first experiment that changes the output structure of the model. Bold because it is mechanistically motivated by the surface/volume metric asymmetry, but risky because physics consistency between surface BCs and volume field may require the shared head.

### Failure mode
If surface pressure accuracy depends on consistent co-prediction with volume (i.e., the model learns surface p partly via volume divergence constraints implicitly encoded in the shared head), decoupling heads will degrade both. Observable: both mae_surf_p AND mae_vol_p worsen vs baseline. In that case, close and note "shared decoder enforces implicit volume-surface physics consistency."

### Reproduce command
```bash
cd target && python train.py --agent tanjiro --experiment_name "tanjiro/split-surface-vol-heads" --epochs 70
```

---

## Summary ranking by expected impact

| Rank | Student | Hypothesis | Mechanism | Predicted delta | Complexity |
|------|---------|-----------|-----------|----------------|-----------|
| 1 | edward | mlp_ratio 2→1 | Budget-bound: ~6 extra epochs → more routing improvement | -2% to -5% | 1 line |
| 2 | tanjiro | Separate surface/volume heads | Decoder specialization for primary metric (surface) | -3% to -8% (high variance) | ~15 lines |
| 3 | frieren | surf_weight 10→20 | Direct gradient alignment to mae_surf_p objective | -2% to -4% | 1 line |

Note: tanjiro ranks #2 by ceiling despite higher variance — the bold swing is appropriate given round-44 three-consecutive-close pattern.

---

## Closed axes (do NOT re-propose)

- All optimizer variants (Lion, Sophia, RAdam, LAMB, Lookahead, NAdam, SGD-momentum, amsgrad, β1/β2 variants)
- Huber-β loss family (both softening AND berHu amplification) — pure L1 is GLOBAL optimum; loss-shape axis FULLY CLOSED
- n_head axis (n_head=2 is interior optimum; {1,2,4} all probed)
- LayerScale init variants (γ=1e-4 uniform optimal)
- DropPath (4-attempt pod-stall pattern; ABANDONED)
- RMSNorm (LOSS vs LayerNorm)
- n_hidden=64 (LOSS round-44) AND n_hidden=96 is the width floor — width-down axis CLOSED
- mlp_ratio=4 (budget cliff LOSS +42.5%)
- All averaging-style regularizations (8× bimodal class): EMA, grad-clip, lr-down, Lookahead, coord-jitter, fun-jitter, warmup-5
- Broadcast-scalar prior corruption: gap/stagger jitter, NACA jitter, input-coord augmentation
- GELU→SiLU (LOSS +13.91%)
- LR floor eta_min=5e-5 (LOSS)
- Warmup duration closed at warmup-3
- SGDR warm restarts (LOSS with L1 sign-gradient regime)
- surf_weight=30 (LOSS, prior config)
- Batch=8 (step-count starvation)

## In-flight (do NOT overlap)

- askeladd #2362: slice_num=16
- thorfinn #2298: epochs=90
- nezuko #2289: n_layers=3
- alphonse #2283: ReLU² activation
- fern #1775: wd=5e-5
