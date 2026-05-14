# Round 137 — Separate output heads for surface vs volume tokens (first STRUCTURAL/LOCAL attack on meta-signal)

## Hypothesis

**Split the single output Linear `nn.Linear(n_hidden, 3)` into TWO separate output heads** — one for surface tokens (`head_surf`), one for volume tokens (`head_vol`). Surface mask routes surface tokens through `head_surf`; volume mask routes volume tokens through `head_vol`. Each head learns its own per-channel weights specialized to its physical region.

This is the **first STRUCTURAL/LOCAL intervention** in this launch, directly motivated by #2937 student's decisive insight: *"Future cruise/in_dist attacks need to be STRUCTURAL/LOCAL (per-region weighting, geometry-conditioned re-weighting, surface-vs-volume specialization), not uniform global re-shaping."* Of those three categories, surface-vs-volume specialization is the most direct and tractable to implement.

## Why this might WIN

1. **#2937 student explicitly recommended surface-vs-volume specialization.** Verbatim above.

2. **Direct physical motivation.** Surface tokens encode boundary-layer physics (high velocity gradients, pressure peaks at leading/trailing edge); volume tokens encode far-field flow (smoother gradients). A single linear projection forces the head to learn a compromise between these two regimes. Separate heads let each specialize.

3. **Loss is dominated by surface (10× weight).** With `surf_weight=10`, the shared head's gradient is dominated by surface tokens. Volume tokens get only 1/11 of the gradient pull — likely under-specialized. A separate vol head removes this asymmetric gradient competition.

4. **Tiny param add (~600 params, 0.15% of total).** `nn.Linear(96, 3)` = 96*3 + 3 = 291 params. Doubling = 582 new params. Negligible capacity change.

5. **Attacks the META-SIGNAL directly.** The cruise/in_dist tradeoff has been robust to capacity-axis interventions and global loss-shape interventions. Per-region head specialization is a NEW axis: changes the model's internal differentiation between surface (where most of the val_avg signal lives — p MAE dominates) and volume.

6. **Lion-compatible.** No optimizer interaction quirks; just adds 600 params with Kaiming-style init.

7. **Zero gradient-flow cost** — both heads run on the same input tensor, then masked-combined for the final prediction.

## Why this might LOSS

1. **Shared head may already work fine.** Linear projections have low capacity; whatever the unified head was learning may already be the right thing. Splitting may just halve effective per-channel data per head without helping.

2. **The body's residual stream is shared.** Surface and volume tokens use the same body. Specialization at only the head layer may be insufficient — the bottleneck may be earlier in the network.

3. **Could over-fit cruise to surface, hurt in_dist.** If `head_vol` learns to fit the simpler far-field but `head_surf` over-fits to in_dist training surface, in_dist may regress.

4. **Adds new parameters** — even 600 params can shift training dynamics under Lion's sign-step at our small total of 407k.

## Falsifiable predictions

- **WIN** (val < 30.5605): Surface/volume specialization helps. Follow-up: try per-region body branches (heavier intervention), or per-channel-per-region (3 weights × 2 regions = 6 separate output paths).
- **PARTIAL** (val WIN, test LOSS or vice versa): One region of specialization is signal, the other isn't. Probably surf-only head split helps; vol-only split is noise.
- **WASH** (val ≈ 30.5605 ± 0.3%): Head separation doesn't help at this scale. The bottleneck is upstream. Move to body-level specialization next.
- **LOSS** (val > 31.0): Specialization hurts — shared head is correctly compromising between regimes. Closes the surface-vs-volume head-specialization axis. Move to per-block depth specialization next.

## Implementation

### Step 1: Locate the head in `train.py`

Find the output projection layer in the `Transolver` model — typically a single line like:
```python
self.head = nn.Linear(n_hidden, 3)
```
(Names may differ — could be `out_proj`, `output`, `final`, etc. There may be a final normalization before it; keep that.)

### Step 2: Replace single head with two heads

```python
# Before
self.head = nn.Linear(n_hidden, 3)

# After
self.head_surf = nn.Linear(n_hidden, 3)
self.head_vol  = nn.Linear(n_hidden, 3)
```

### Step 3: Route surface vs volume tokens in forward()

```python
# Before
pred = self.head(x)  # [B, N, 3]

# After
pred_surf = self.head_surf(x)  # [B, N, 3]
pred_vol  = self.head_vol(x)   # [B, N, 3]
# Combine: surface tokens use head_surf, volume tokens use head_vol
# surf_mask, vol_mask are [B, N] indicator tensors (0/1), available in the data dict
# Use the masks from the data dict (typically `data["surface_mask"]` and `data["volume_mask"]` or similar)
pred = (
    pred_surf * surf_mask.unsqueeze(-1)
    + pred_vol * vol_mask.unsqueeze(-1)
)
```

If surface and volume masks are NOT mutually exclusive (some tokens neither surface nor volume), the convention should fall back to one of them — typically `head_vol` for "general" tokens. Test both conventions and report.

### Step 4: Startup diagnostics

```python
print(f"Output heads: SPLIT (head_surf + head_vol), each nn.Linear({n_hidden}, 3)")
print(f"vs baseline: single nn.Linear({n_hidden}, 3)")
print(f"New params: {sum(p.numel() for p in self.head_surf.parameters())} (head_surf) + {sum(p.numel() for p in self.head_vol.parameters())} (head_vol) = +{2 * (n_hidden * 3 + 3)} total (~600)")
print(f"Total param count: {sum(p.numel() for p in model.parameters())}")  # expect ~408,531
print(f"Routing: surf_mask routes to head_surf, vol_mask routes to head_vol")
```

### Step 5: Per-region per-epoch diagnostic

Log mean per-channel MAE separately for surface tokens (via `head_surf`) and volume tokens (via `head_vol`) at ep1, 5, 10, 30, 60. Compare to baseline trajectory. Look for evidence that surface and volume convergence rates diverge under split heads — the key signal that specialization is happening.

### Step 6: Head-weight divergence diagnostic

At ep60, compute the L2 norm of the difference between `head_surf.weight` and `head_vol.weight`. If specialization is happening, expect divergence ≥ 0.5 (heads differ substantially). If divergence ≤ 0.1, the two heads are learning ~the same projection and the intervention is null.

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 (single shared head) |

After change: 2 heads (head_surf, head_vol), param count ~408,531 (single head was 96*3+3=291 params; doubled = 582 total).

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/separate-surf-vol-heads" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — head split hardcoded in model construction. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. Param count confirmation (expect ~408,531, +591 vs baseline)
4. Epochs completed (target: 60), sec/epoch, peak GPU memory
5. Train→val loss gap at convergence
6. **Per-region convergence diagnostic:** trajectory of surface-token MAE vs volume-token MAE at ep1, 5, 10, 30, 60 (separately per channel). Did specialization actually occur?
7. **Head-weight divergence diagnostic:** L2 norm of `head_surf.weight - head_vol.weight` at ep60. (If close to 0, the intervention is null.)
8. **Meta-signal check:** does this STRUCTURAL intervention move the cruise/in_dist tradeoff (e.g., differentially affect surface-dominated splits vs volume-dominated splits)? Or does it move it uniformly like global re-shape interventions?
9. **Plain-language verdict:** WIN (surface/volume specialization helps) / WASH (head capacity not bottleneck) / LOSS (shared head was correctly compromising). State which.

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
