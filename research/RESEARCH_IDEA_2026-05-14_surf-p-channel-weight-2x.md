# Round 134 — Surface pressure channel loss weight 2× (per-channel reweighting)

## Hypothesis

Modify the surface loss to **weight the pressure channel 2× higher than Ux/Uy** within `surf_loss`. The primary validation/test metric is `mae_surf_p` (surface pressure MAE only), but the current training loss averages MAE equally across (Ux, Uy, p) channels — meaning only 1/3 of the surface-loss gradient targets the metric we actually optimize for. Reweighting p to 2× concentrates gradient signal on the metric while preserving the regularizing effect of fitting all channels.

## Why this might WIN

1. **Direct test of student-suggested 'loss function' direction.** Student of #2922 (Lookahead-Lion LOSS): *"the cruise↔in_dist tradeoff is NOT optimization-trajectory-driven; it lives in the LOSS LANDSCAPE itself. Future attacks should target either the loss function (per-domain reweighting, OOD-aware loss terms) or representation (architectures that explicitly separate domain-conditioned features) rather than optimizer dynamics."* This experiment is the most direct test of "per-channel reweighting" within the loss-function family.

2. **The primary metric is `mae_surf_p` — we should explicitly optimize it.** Current surf_loss = (mae_Ux + mae_Uy + mae_p) / 3 gives p only 33% of the gradient weight. New formulation (mae_Ux + mae_Uy + 2·mae_p) / 4 gives p 50% — gradient signal is concentrated where the metric is.

3. **Surf_weight tuning already showed loss-axis is sensitive.** #2910 surf_weight=20 (vs baseline 10) was LOSS uniform +5.85% — too much surface weight broke everything. But per-channel reweighting is a different axis: it keeps total surf_loss magnitude similar while redistributing within the 3 channels. We're not changing surf-vs-vol balance, only Ux/Uy-vs-p balance within surf.

4. **Conservative 2× factor avoids the #2910 trap.** A 3× or 5× would dominate the loss with pressure (potentially destabilizing Ux/Uy convergence which the body MLP needs to learn the underlying flow). 2× is the smallest meaningful tilt toward p.

5. **Zero new params, single-line change.** Pure loss-function edit. No architecture, no scheduler, no optimizer change.

## Why this might LOSS

1. **Joint multitask training may be load-bearing.** Predicting Ux, Uy, p simultaneously forces the model to learn the underlying flow field; degrading Ux/Uy fitting could degrade the shared representation that gives good p predictions. Mitigation: 2× is conservative.

2. **Surface loss magnitude shifts.** Going from (Ux+Uy+p)/3 to (Ux+Uy+2p)/4 changes total magnitude by factor 4/3 if all channels have similar MAE magnitudes (likely). This effectively raises surf_weight from 10 to ~13.3 — within the regime that produced WIN in #2879 baseline (which used surf_weight=10).

3. **In_dist may regress.** The meta-signal series suggests interventions that shift capacity allocation hurt in_dist. Per-channel reweighting IS a form of capacity allocation (redirecting body MLP gradient toward p-channel features). Could repeat the meta-signal pattern.

## Falsifiable predictions

- **WIN** (val < 30.5605): Direct metric-channel reweighting helps. Try 1.5× and 3× to characterize the axis.
- **PARTIAL** (val_in_dist drops via cruise+RC trade-off): meta-signal repeats. Capacity reallocation through loss reweighting acts like capacity reallocation through architecture.
- **WASH** (val ≈ 30.5605 ± 0.3%): Per-channel weighting has no detectable effect at 2×. Try 3× or close axis.
- **LOSS** (val > 31.0): Joint multitask supervision is load-bearing. Confirms the model needs full Ux/Uy/p supervision; close per-channel loss axis.

## Implementation

### Step 1: Locate the surf_loss / vol_loss computation in `train.py`

The current loss block looks something like:

```python
# Existing — surface and volume losses, both averaged across channels
surf_loss = (pred[is_surface] - y[is_surface]).abs().mean()  # mean over masked tokens × channels
vol_loss = (pred[is_volume]  - y[is_volume]).abs().mean()
loss = vol_loss + cfg.surf_weight * surf_loss
```

(Exact form depends on how channels are handled — may use a `mask.sum().clamp(min=1)` normalization. Find it.)

### Step 2: Compute per-channel surface MAE, then 2× weight on p

Replace the surf_loss line with per-channel breakdown:

```python
# Per-channel surface MAE, with 2x weight on pressure channel
diff_surf = (pred - y).abs()  # [B, N, 3] or wherever the channel dim is
# Apply surface mask BEFORE channel-wise reduction
# Channels: 0=Ux, 1=Uy, 2=p

# Compute per-channel mean of surface tokens (use the same mask normalization as the current surf_loss)
mae_Ux = (diff_surf[..., 0] * is_surface_mask).sum() / is_surface_mask.sum().clamp(min=1)
mae_Uy = (diff_surf[..., 1] * is_surface_mask).sum() / is_surface_mask.sum().clamp(min=1)
mae_p  = (diff_surf[..., 2] * is_surface_mask).sum() / is_surface_mask.sum().clamp(min=1)

# Weighted average — pressure gets 2× weight
surf_loss = (mae_Ux + mae_Uy + 2.0 * mae_p) / 4.0
```

The total denominator (4.0) keeps the magnitude comparable to the current `(Ux + Uy + p) / 3` formulation, so `surf_weight=10` continues to scale surf vs vol appropriately.

**Important:** Keep using the exact same mask normalization (`.sum() / mask.sum().clamp(min=1)`) as the current code — don't introduce mask-handling bugs.

### Step 3: Volume loss unchanged

Keep `vol_loss = mean over all volume tokens, all channels` — same as baseline.

### Step 4: Startup diagnostic

```python
print(f"Loss formulation: surf_loss = (mae_Ux + mae_Uy + 2.0 * mae_p) / 4.0")
print(f"Surface p channel effective weight: {(2.0/4.0)/(1.0/3.0):.2f}x vs baseline (1.5x relative)")
print(f"Total loss: vol_loss + {cfg.surf_weight} * surf_loss")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 5: Per-epoch logging

In addition to the regular train/surf and train/vol losses, log the **per-channel** train MAEs (mae_Ux, mae_Uy, mae_p) every epoch. Watch:
- Does train/mae_p improve faster than baseline?
- Does train/mae_Ux or train/mae_Uy regress?
- Are the per-channel ratios stable across training?

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 |

Current loss: `vol_loss + 10 * (mae_Ux + mae_Uy + mae_p) / 3` (per-channel equal weight, surf-to-vol balance 10:1).

After change: `vol_loss + 10 * (mae_Ux + mae_Uy + 2.0 * mae_p) / 4.0` (p weighted 2× within surf_loss; total surf magnitude similar).

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/surf-p-channel-weight-2x" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — per-channel weighting is hardcoded in the loss computation. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Per-channel train MAE table:** train/mae_Ux, train/mae_Uy, train/mae_p at ep1, ep10, ep30, ep60. Does p improve faster while Ux/Uy regress?
4. **Per-channel val MAE table:** val_avg/mae_surf_Ux, val_avg/mae_surf_Uy, val_avg/mae_surf_p at best epoch. The PR's success is val_avg/mae_surf_p improving without Ux/Uy getting catastrophic.
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap
8. **Meta-signal check:** does this experiment join the cruise WIN / in_dist LOSS pattern, break it uniformly (like Lookahead), or break it in the WIN direction (no meta-signal)?
9. **Plain-language verdict:** WIN / WASH / LOSS, and the qualitative shape: did the metric channel improve at the cost of the other channels (expected) or did all channels move together (capacity-shared)?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
