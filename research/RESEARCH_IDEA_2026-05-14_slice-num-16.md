# Round 135 — slice_num 24→16 (downward sweep, student followup #1)

## Hypothesis

**Reduce slice_num from baseline 24 to 16**, keeping n_head=2 and all other hyperparameters fixed. Tests whether the slice-routing alphabet is at, above, or below peak capacity at the baseline (24). #2923 (slice_num 24→32) was LOSS +6.02%, so larger isn't better. The complementary question — does smaller help? — is exactly what student of #2923 explicitly recommended as their #1 followup.

## Why this might WIN

1. **Student of #2923 explicitly recommended this as their highest-priority followup.** Verbatim: *"slice_num=16 (downward sweep, complementary to slice_num=32 which failed) — if 24→32 hurt, does 24→16 help? Tests whether baseline is at peak of capacity curve or above-peak."* This is the cleanest directional test of slice-routing capacity remaining after the upward sweep failure.

2. **Block-3 head entropy diagnostic suggests under-utilization at slice_num=24.** Student of #2923 reported block-3 head entropy collapsed (range 0.01-0.92 nats vs 3.466 ceiling for 32 tokens, 2.77 ceiling for 16 tokens). At baseline 24, the actual routing distribution likely has low effective rank (head 0 and head 1 both routing to ~few slices). Reducing the alphabet to 16 forces sharper, more-decisive routing because each slice has more responsibility — analogous to how reducing k in k-means forces tighter clusters.

3. **The block-0 vs block-3 contrast suggests the network only needs ~16 slices.** Block-0 entropy range was 1.32-1.71 nats (active routing). Block-3 entropy was 0.01-0.92 (collapsed). If even with 24 tokens block-3 is routing to only ~2-3 effective slices, smaller alphabet (16) won't degrade block-3 (already over-provisioned there) but may concentrate signal at block-0.

4. **Param count decrease is tiny.** slice_num is the dimension of `to_q` `to_k` `to_v` `to_out` projections in PhysicsAttention; the param delta is in the small projection matrices. Net change ~-1.5K params (rough), making this nearly capacity-matched while reducing routing alphabet.

5. **Loss-axis-adjacent: tests REPRESENTATION (matches student insight from #2922).** Per #2922 student's decisive insight, the cruise↔in_dist tradeoff lives in the LOSS LANDSCAPE / REPRESENTATION space. Slice_num governs the representational granularity of the per-token routing — testing it is testing the discrete representation alphabet.

## Why this might LOSS

1. **Could underprovision block-0.** Block-0 has active head differentiation (1.32-1.71 nats entropy range), suggesting it's currently using meaningful slice diversity. Reducing to 16 might force collapse there too. Mitigation: 16 is still a substantial alphabet (vs e.g. 8 which would be aggressive).

2. **The trade-off may be insensitive to alphabet size at this depth.** With 4 blocks and only 96-dim hidden, the bottleneck may not be routing granularity. Could be WASH.

3. **Trade-off may persist (meta-signal).** Like most interventions in this launch, slice_num change is a CAPACITY-SHIFT (alters routing-token count). Per #2918 student insight, meta-signal applies specifically to capacity-allocation-shifting interventions. Cruise WIN / in_dist LOSS may repeat.

## Falsifiable predictions

- **WIN** (val < 30.5605): Baseline (24) is above peak; smaller alphabet sharpens routing. Try slice_num=12 or 8 to characterize the axis.
- **PARTIAL** (in_dist LOSS, cruise WIN): Meta-signal repeats. Capacity reallocation through routing-alphabet narrowing acts like capacity reallocation through head splitting.
- **WASH** (val ≈ 30.5605 ± 0.3%): Routing alphabet doesn't matter at this scale. Close downward sweep axis.
- **LOSS** (val > 31.0): Block-0 head differentiation depended on alphabet size 24; smaller alphabet broke it. slice_num axis is closed (both directions exhausted).

## Implementation

### Step 1: Locate slice_num in `train.py`

The current Transolver config passes `slice_num=24` to the model. Look for the line that constructs `Transolver(...)` and includes `slice_num=24` (or similar).

### Step 2: Change to slice_num=16

```python
# Before
model = Transolver(..., slice_num=24, ...)

# After
model = Transolver(..., slice_num=16, ...)
```

That's the **only change**. Keep n_head=2, dim_head=48, n_layers=4, n_hidden=96. All other hyperparameters identical to baseline #2879.

### Step 3: Startup diagnostics

```python
print(f"Routing alphabet: slice_num=16 (vs baseline 24)")
print(f"Entropy ceiling: {math.log(16):.4f} nats (vs baseline {math.log(24):.4f})")
print(f"Per-head dim: dim_head=48, n_head=2 (unchanged)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # ~406k (slight decrease)
```

### Step 4: Per-block routing-entropy logging (optional but very useful)

If feasible, log block-0 through block-3 head-routing entropy at ep1, 10, 30, 60. The student of #2923 demonstrated this is informative — replicate for the downward sweep to characterize whether collapse persists or sharpens.

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 (slice_num=24, n_head=2, dim_head=48) |

For comparison, #2923 slice_num=32: val 32.3998 (+6.02% LOSS), test 27.2641 (+2.82% LOSS).

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-tanjiro \
    --experiment_name "charliepai2g48h5-tanjiro/slice-num-16" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — slice_num is hardcoded in the model config. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. Param count confirmation (expect ~406k, slight decrease vs baseline)
4. Epochs completed (target: 60), sec/epoch, peak GPU memory
5. **Routing-entropy diagnostic** (if feasible): per-block (0-3) per-head entropy at ep1, 10, 30, 60. Does the smaller alphabet sharpen routing at block-0 (which had active differentiation at slice_num=24) or collapse it?
6. **Comparison to #2923 (slice_num=32):** does the trend hold (smaller=better, larger=worse), or is 24 still optimal?
7. **Meta-signal check:** does this experiment join the cruise WIN / in_dist LOSS pattern, break it uniformly, or break it in the WIN direction?
8. **Plain-language verdict:** WIN / WASH / LOSS — was baseline (24) above peak, at peak, or below peak of the routing-alphabet capacity curve?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
