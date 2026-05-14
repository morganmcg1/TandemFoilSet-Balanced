# Round 136 — Surface p-channel weight 2× MEAN form (clean disambiguation of #2933 confound)

## Hypothesis

**Re-test the per-channel p-weight 2× intervention but using the explicit `/4` MEAN-form normalization** to preserve baseline surf_loss magnitude. This isolates the "p-weight only" lever from the "effective surf_weight bump" confound that contaminated #2933.

Specifically: `surf_loss = (mae_Ux + mae_Uy + 2*mae_p) / 4` (channel weights [1, 1, 2]/4).

This is **student of #2933's followup #1**, directly motivated by their self-diagnosed confound. Their literal recommendation: *"Isolate the p-weight effect from the surf-weight effect: rerun with `surf_loss = (mae_Ux + mae_Uy + 2*mae_p) / 4` (the PR's explicit /4 form). If that also regresses, the p-weight lever itself is bad. If it's neutral or better, the confound was the whole story."*

## Why this might WIN

1. **Cleanly tests the p-weight lever in isolation.** #2933 used sum-form which gave a ~33% magnitude bump and effective surf_weight ≈ 13.3. This experiment fixes that — surf_loss magnitude matches baseline.

2. **#2933 already showed cruise-WIN on the p-weight intervention.** -0.96% cruise improvement. If the cruise direction is signal-bearing and not just an artifact of the surf_weight bump, the /4 form should preserve cruise-WIN while reducing the in_dist-LOSS magnitude.

3. **Sets up the per-channel WEIGHT axis cleanly.** If WIN or WASH, the axis stays open at tighter p-only weights (1.25, 1.5). If LOSS, the per-channel WEIGHT axis (mean-form) closes definitively — independent of surf_weight magnitude.

4. **Direct attack on the val_avg primary metric.** p is the dominant val_avg term (Ux/Uy MAEs are ~0.5 / ~0.25 vs p ~30) — reweighting p has the largest leverage on val_avg by construction.

## Why this might LOSS

1. **#2910 surf_weight=20 was LOSS even at uniform per-channel weights.** Surface gradient magnitude near the airfoil already dominates the model's attention; further weighting may overcommit capacity there.

2. **The cruise-WIN may have come from the surf_weight bump, not p reweighting.** If so, /4 form will show NO cruise-WIN, AND the in_dist-LOSS may also reduce — revealing the original effect was entirely the confound.

3. **Train per-channel MAE shows all 3 surface channels drop monotonically together** (per #2933 student diagnostic). No bottleneck on p specifically — reweighting may not unlock anything new.

4. **The cruise/in_dist tradeoff is structural** per #2922. Per-channel reweighting may just move along the same trade-off curve.

## Falsifiable predictions

- **WIN** (val < 30.5605): The /4 form unlocks the p-weight lever cleanly. Sets up tighter p-only weights as next axis.
- **PARTIAL** (val ≈ 30.5605-31.0): The /4 form partially recovers vs #2933. Confounded but not entirely. Closes WEIGHT axis at p=2x; suggests smaller p-bump might be neutral.
- **WASH** (val ≈ 30.5605 ± 0.3%): Confound was the entire story. Pure p-weight lever has zero sign for val_avg. Close per-channel-WEIGHT axis.
- **LOSS** (val > 31.0): Per-channel p-reweighting actively hurts regardless of surf_loss magnitude. CLEAN closure of the mean-form WEIGHT axis. Move to per-domain reweighting (student followup #2) next round.

## Implementation

### Step 1: Locate surf_loss in `train.py`

The baseline form is:
```python
# Baseline (current code)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum()
# This is equivalent to: surf_loss = mae_Ux + mae_Uy + mae_p
```

(Confirmed by #2933 student's inspection — mask denominator does NOT include channel dim.)

### Step 2: Replace with /4 mean-form weighted

```python
# This PR
mae_Ux = (sq_err[..., 0] * surf_mask).sum() / surf_mask.sum()
mae_Uy = (sq_err[..., 1] * surf_mask).sum() / surf_mask.sum()
mae_p  = (sq_err[..., 2] * surf_mask).sum() / surf_mask.sum()
surf_loss = (mae_Ux + mae_Uy + 2 * mae_p) / 4
```

**Key:** the `/4` divisor preserves baseline surf_loss magnitude (baseline is sum form `mae_Ux + mae_Uy + mae_p`, this PR is mean form `(mae_Ux + mae_Uy + 2*mae_p)/4`). The total loss formula `vol_loss + 10 * surf_loss` should give an effective surf_weight unchanged at 10.

### Step 3: Startup diagnostics

```python
print(f"surf_loss form: MEAN (mae_Ux + mae_Uy + 2*mae_p)/4 — channel weights [0.25, 0.25, 0.5]")
print(f"vs baseline sum (mae_Ux + mae_Uy + mae_p) — channel weights [1, 1, 1] (effective [0.33, 0.33, 0.33])")
print(f"vs #2933 sum-form (mae_Ux + mae_Uy + 2*mae_p) — channel weights [1, 1, 2] (effective [0.25, 0.25, 0.5])")
print(f"p RELATIVE weight: 50% (same as #2933)")
print(f"p ABSOLUTE weight: 0.5 vs baseline 0.33 vs #2933 2.0 — magnitude preserved by /4")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Verify surf_loss magnitude vs baseline

At ep1, log `surf_loss` value. Compare to baseline ep1 value. If matched (within ~5%), the /4 normalization is working correctly. If differs significantly, the magnitude confound persists.

### Step 5: Per-channel MAE tracking (same as #2933)

Log train and val per-channel MAEs at ep1, 5, 10, 30, 60. Compare to #2933 trajectory to see if the /4 form changes the per-channel descent dynamics.

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

For comparison:
- #2933 sum-form (effective surf_weight ≈ 13.3): val 31.9074 (+4.41% LOSS), test 26.5558 (+0.15% WASH), cruise -0.96% WIN, in_dist +9.60% LOSS
- #2910 surf_weight=20 (uniform channels): val ~32.35 (+5.85% LOSS, recall)

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/surf-p-weight-2x-mean-form" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160 AND vs #2933 31.9074 / 26.5558 (the prior sum-form arm)
2. Per-split val + test breakdown
3. **surf_loss magnitude check at ep1:** does it match baseline within ~5%? (If yes, /4 normalization worked.)
4. Per-channel train/val MAE trajectory at ep1, 5, 10, 30, 60. Did Ux/Uy regress while p improved? Or did p improve uniformly with others?
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence
8. **Confound disambiguation:** is the +4.41% val regression of #2933 attributable to (a) p-weight lever itself, (b) effective surf_weight bump, or (c) both? Use the /4 form result to attribute.
9. **Meta-signal check:** does cruise-WIN / in_dist-LOSS pattern persist under /4 form? If yes, the pattern is intrinsic to per-channel p-reweighting (independent of surf_loss magnitude). If no, the pattern was tied to magnitude not channel reweighting.
10. **Plain-language verdict:** WIN (p-weight is good when normalized) / WASH (confound was the story) / LOSS (p-weight lever is bad regardless of magnitude). State which.

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
