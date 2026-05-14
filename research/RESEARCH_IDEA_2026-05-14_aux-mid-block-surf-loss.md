# Round 137 — Auxiliary mid-block surface loss (deep supervision at block 2)

## Hypothesis

**Add an auxiliary surface loss applied to block-2 (mid-network) features via a small auxiliary Linear projection**, with weight α=0.1. Tests whether intermediate-layer features can be guided toward better surface representations via deep supervision, attacking the meta-signal at a structural/local level instead of via output-loss reshaping.

This is the **second STRUCTURAL/LOCAL intervention** in this launch (paralleling #2946 askeladd's separate-surf-vol-heads). Direct motivation from #2937 decisive student insight: *"Future cruise/in_dist attacks need to be STRUCTURAL/LOCAL (per-region weighting, geometry-conditioned re-weighting, surface-vs-volume specialization), not uniform global re-shaping."* Deep supervision is a depth-axis STRUCTURAL intervention that complements the head-axis split.

Architecture:
- Block 2 outputs intermediate features `x_mid` (shape: B, N, 96)
- Add auxiliary head: `aux_head_surf = nn.Linear(96, 3)`
- Compute auxiliary surface loss: `aux_loss = L1(aux_head_surf(x_mid)[surf_mask], y_surf)`
- Total loss: `loss = vol_loss + 10 * surf_loss + 0.1 * aux_surf_loss`

## Why this might WIN

1. **#2937 student STRUCTURAL/LOCAL recommendation, depth-axis variant.** Per-region weighting at the OUTPUT was #2933/#2941 (closed). Per-region head split is #2946. Per-region per-DEPTH supervision is the natural next axis — deep supervision is the canonical depth-axis loss intervention.

2. **The output prediction is surface-dominated (val_avg/mae_surf_p is primary metric).** Mid-block features may be under-specialized for surface prediction because only the OUTPUT layer sees the surface loss gradient. Adding mid-block surface supervision provides direct gradient pull on intermediate features.

3. **Deep supervision is well-documented for improving fine-grained predictions** (Xie & Tu 2015 HED, Wang et al. 2015 DSN). The pattern: auxiliary classifiers/regressors at intermediate layers improve final-layer quality by ensuring features are usable earlier.

4. **Tiny param overhead.** New Linear(96, 3) = 96*3 + 3 = 291 params (0.07% of total). Negligible capacity addition.

5. **Aux loss weight=0.1 is conservative.** Strong enough to provide gradient signal but small enough that the primary surf_loss dominates. If this is the right axis, follow-ups can sweep weight.

6. **Mid-block supervision specifically addresses the in_dist regression pattern.** The meta-signal shows in_dist gets hit hardest when capacity is shifted around — likely because in_dist's higher-frequency features are most sensitive to intermediate representation quality. Mid-block supervision regularizes those intermediate representations.

## Why this might LOSS

1. **Aux loss may over-constrain mid features.** Forcing block-2 features to be linearly mappable to surface prediction may restrict the freedom block-2 needs to build useful representations for blocks 2-3.

2. **The 60-epoch budget may be too tight to benefit.** Deep supervision typically helps in longer training regimes.

3. **The mid-block features may already be well-formed.** If block 2 features are nearly linearly mappable to surface prediction already, aux loss adds noise without signal.

4. **Lion + auxiliary loss interaction.** Lion's sign-step on combined gradients may distort the relative weights of primary vs auxiliary signals.

## Falsifiable predictions

- **WIN** (val < 30.5605): Deep supervision helps. Try aux weight=0.05 and 0.2 sweeps; try aux supervision at block 1 and 3 too.
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): Aux supervision is neutral. Move to other structural axes.
- **LOSS** (val > 31.0): Aux supervision actively hurts. Closes deep-supervision axis from the moderate-weight direction; could try lower weight 0.01 but lower priority.

## Implementation

### Step 1: Locate the Transolver forward loop in `train.py`

The model has 4 blocks (n_layers=4). Block iteration typically looks like:
```python
for block in self.blocks:
    x = block(x, fx=fx, T=T)
```

### Step 2: Capture block-2 output

Modify the forward loop to capture block-2's output:
```python
mid_features = None
for i, block in enumerate(self.blocks):
    x = block(x, fx=fx, T=T)
    if i == 1:  # block 2 = index 1 (0-indexed: blocks 0, 1, 2, 3)
        mid_features = x  # shape: (B, N, n_hidden=96)
```

Note: with 4 blocks (indices 0-3), block-2 = index 1 places aux supervision at 50% depth (halfway through). If preferred, use index 2 (block 3, 75% depth) — student can choose based on their architecture inspection.

### Step 3: Add the auxiliary head

In `__init__`:
```python
self.aux_head_surf = nn.Linear(n_hidden, 3)
```

Initialize with same scheme as the primary head (Kaiming or default).

### Step 4: Compute auxiliary loss

In the training loop, after computing primary surf_loss:
```python
aux_pred = self.aux_head_surf(mid_features)  # (B, N, 3)
# Apply same surface masking as primary surf_loss
aux_pred_surf = aux_pred[surf_mask]  # only surface tokens
aux_loss = F.l1_loss(aux_pred_surf, y_surf)  # same shape/masking as primary surf_loss
total_loss = vol_loss + 10 * surf_loss + 0.1 * aux_loss
```

### Step 5: Startup diagnostics

```python
print(f"Deep supervision: aux_head_surf at block 2 (index 1), weight=0.1")
print(f"aux_head_surf: nn.Linear({n_hidden}, 3) = {96*3 + 3} new params (~291)")
print(f"Total param count: {sum(p.numel() for p in model.parameters())}")  # expect ~408,231 (+291)
print(f"Total loss: vol_loss + 10*surf_loss + 0.1*aux_surf_loss")
print(f"Structural axis: depth-axis surface supervision (paralleling head-axis #2946 surf-vol split)")
```

### Step 6: Per-epoch logging

Track aux_loss separately from primary surf_loss at ep1, 5, 10, 30, 60. Compute the RATIO aux_loss / surf_loss over time. If aux_loss converges to a much smaller value than surf_loss at the same surf_weight scaling, the mid-features are easily mappable to surface prediction (aux is doing little work). If aux_loss STAYS large, mid-features need significant guidance (aux is doing real work).

### Step 7: Aux-head weight divergence diagnostic

At ep60, compute the L2 norm of `aux_head_surf.weight` vs the primary head's weight. If they're very similar, aux is redundant. If they diverge significantly, aux is learning a distinct intermediate projection.

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 (single output head) |

After change: +291 params for aux_head_surf, total ~408,231.

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/aux-mid-block-surf-loss" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — aux head + supervision hardcoded in model + training loop. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. Param count confirmation (expect ~408,231, +291)
4. Epochs completed (target: 60), sec/epoch, peak GPU memory
5. Train→val loss gap at convergence
6. **Aux loss diagnostic:** trajectory of aux_loss / surf_loss ratio over training. Did aux loss drop to a much smaller value than primary (=mid features are well-formed, aux is redundant) or stay comparable (=mid features needed real guidance)?
7. **Aux-head weight divergence:** L2 norm of aux_head_surf vs primary head's weight. (Similar = redundant, different = learning distinct projection.)
8. **Meta-signal check:** does aux supervision differentially affect surface-dominated splits vs volume-dominated splits? Or move uniformly?
9. **Block placement check (if time allows):** if WIN, the natural next is moving aux supervision earlier (block 1) or later (block 3) to map the depth-axis sensitivity. Don't run multiple in this PR — propose for followup.
10. **Plain-language verdict:** WIN (deep supervision helps; depth-axis open) / WASH (aux supervision is redundant; mid features already well-formed) / LOSS (aux over-constrains mid features).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
