# Round 138 — Volume aux supervision at block-1 (mid-depth, target inversion)

## Hypothesis

Replace #2952's SURFACE aux head with a VOLUME aux head at the SAME block-1 placement and SAME weight α=0.1. Tests the orthogonality interpretation of #2952/#2961 closures — does the aux mechanism work when the auxiliary target ALIGNS with the dominant gradient direction?

## Motivation

After #2952 (surf aux at 50% depth, +1.89% LOSS) and #2961 (surf aux at 75% depth, +6.53% LOSS), depth-axis surface deep supervision is DECISIVELY closed with depth-monotonic worsening.

Mechanism diagnosed: aux_head_surf learned a near-orthogonal projection on the p channel regardless of placement (cos similarity ~ 0.05 at 50% depth, -0.10 at 75% depth). Aux/surf ratio 2.5-3.7× → mid features are NOT linearly mappable to surface output. Aux gradient COMPETED with primary gradient.

**Direct test:** Replace target with VOLUME field. Volume tokens are ~70% of nodes, dominate the gradient signal, and mid-block features ARE shaped primarily by volume_loss gradient. The aux head should:
1. Be near-REDUNDANT (high cos similarity with primary volume head)
2. Have aux/vol ratio ~0.5-1.0× (aux solves a "similar" problem to primary)
3. Either WIN (alignment helps) or wash (aux is redundant)

If volume aux ALSO LOSSES, the aux mechanism is fundamentally broken at this scale/budget regardless of target — closes the entire aux-supervision-axis comprehensively.

## Architecture

```python
# REPLACE #2952's surface aux head with:
self.aux_head_vol = nn.Linear(n_hidden, 3)  # +291 params, same shape as aux_head_surf

# In forward at block-1 output (index 1, 50% depth):
aux_pred = self.aux_head_vol(mid_features)  # (B, N, 3)
aux_pred_vol = aux_pred[~surf_mask]  # VOLUME tokens (vs aux_pred[surf_mask] in #2952)
aux_loss = F.l1_loss(aux_pred_vol, y_vol)
total_loss = vol_loss + 10 * surf_loss + 0.1 * aux_loss
```

ONLY change from #2952: target is volume (~70% of nodes) instead of surface (~30%).

## Why this might WIN

1. **Aux ALIGNS with dominant gradient direction.** Volume is ~70% of nodes; vol_loss dominates the gradient signal at mid-blocks already. aux_head_vol fits a problem the network is already shaped toward — high cos similarity expected.

2. **No representational competition.** Surface aux at block-1 competes with the primary surf gradient at the head. Volume aux at block-1 reinforces the primary vol gradient — should regularize, not redirect.

3. **Tests an explicit orthogonality interpretation.** If vol aux WINS, the aux mechanism is fine, surf aux failed because surface features are minority-population-orthogonal in mid-blocks. Strong mechanistic finding.

4. **Same +291 params, same diagnostics.** Reuses #2952 measurement infrastructure.

## Why this might LOSS

1. **Same UNDERFIT regime.** Even an "aligned" aux stealing gradient may slow primary convergence in 60ep budget.

2. **Volume targets already dominate primary loss.** Aux on volume = double-weighted volume → may UNDERWEIGHT surface (the primary metric).

3. **Aux on dominant target = redundant gradient injection.** No new information; just noise around the natural learning trajectory.

4. **Fundamental aux mechanism broken.** If WIN/LOSS both worse than baseline, axis is fully closed.

## Falsifiable predictions

- **WIN** (val < 30.5605): aux mechanism works when target ALIGNS with dominant gradient direction. Strong evidence that the surface aux failures were orthogonality-driven, not aux-mechanism-driven. Followup: vol-aux α sweep, vol-aux at deeper depths.
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): aligned aux is wash. Closes aux-axis at the WIN side.
- **LOSS** (val > 31.0): aux mechanism fundamentally broken regardless of target. Closes AUX-SUPERVISION-AXIS comprehensively across BOTH target types.

## Key diagnostics

1. **aux/vol ratio trajectory** (primary mechanistic test):
   - Predicted (if aux aligns): aux/vol ratio ~0.5-1.0× throughout training
   - If observed: ratio > 2× → vol features ALSO not redundantly mappable → aux mechanism broken

2. **Cosine similarity of aux head vs primary final head at ep60:**
   - Predicted (if aux aligns): cos sim > 0.5 on all 3 channels (especially Ux/Uy which are dominant in vol field)
   - If observed: cos sim ~ 0 → aux STILL learns orthogonal → mechanism broken

3. **Train→val gap:** Should be SAME or smaller than #2952 (0.024). If aux ALIGNS, no gradient theft.

## Implementation

1-line change from #2952's existing block-1 capture infrastructure: replace `surf_mask` with `~surf_mask` and `y_surf` with `y_vol` in the aux loss computation. Replace `aux_head_surf` with `aux_head_vol` (both nn.Linear(96, 3)).

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/volume-aux-at-block-1" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Reporting

1. val_avg/test_avg vs baseline + per-split
2. aux/vol ratio trajectory (vs #2952/#2961 aux/surf ratio reference)
3. Aux head weight divergence vs primary head — especially cosine similarity per channel
4. Param count (+291, unchanged from #2952/#2961)
5. Train→val gap at convergence
6. Meta-signal check (cruise WIN / in_dist LOSS pattern)
7. **CRITICAL VERDICT:** WIN (alignment fixes aux) / WASH (aux redundant when aligned) / LOSS (aux mechanism fundamentally broken regardless of target).
