# Round 121 — Block-asymmetric MLP ratio [3, 3, 4, 4]

## Hypothesis

Use **block-asymmetric mlp_ratio = [3, 3, 4, 4]** — narrow MLPs in the shallow blocks (blocks 0, 1) and wider MLPs in the deep blocks (blocks 2, 3). Tests whether wider MLP depth-late can preserve the geometric-OOD improvement seen at uniform mlp_ratio=4 (#2889) WITHOUT sacrificing in-distribution fit (where uniform mlp_ratio=4 regressed +10.88%).

## Why this might WIN

1. **Directly exploits novel signal from #2889.** Uniform mlp_ratio=4 (#2889) improved val_geom_camber_cruise -3.60% (rare OOD WIN) and val_geom_camber_rc -0.74% while regressing val_single_in_dist +10.88% and val_re_rand +5.72%. This is the FIRST experiment to show geometric-OOD improvement at in-dist cost — opposite of the typical pattern. The signal suggests the wider MLP IS useful, but uniformly applied it hurts more than it helps.

2. **Depth-progressive geometric specialization is a known pattern.** Deep ViT/ConvNeXt analyses (e.g., Raghu 2021, Caron 2021) consistently show shallow blocks learn position/edges/textures while deep blocks specialize on object-level features. For TandemFoilSet, the "object-level" features ARE the geometric configuration (airfoil pair + gap + stagger). Routing extra MLP capacity to deep blocks aligns with this depth-progressive specialization.

3. **Recovers param count near baseline.** Expected ~446k params (vs 482k uniform mlp_ratio=4, 408k baseline). Avoids the capacity-saturation signature that hurt #2889 (gate_zero_frac rose +27-48% on every block).

4. **Concretely matches the #2889 student's #1 suggested follow-up.** The student observed the OOD improvement and proposed exactly this experiment as the cleanest test.

5. **Zero new module types — just a config tweak.** No new layers, no new gates, no new losses. Pure capacity-distribution test along the depth axis. Minimal implementation risk.

## Why this might LOSS

1. **The novel signal might be a fluke.** A single experiment showing OOD WIN at in-dist LOSS could be noise. Camber_cruise has been plateaued near 17 for many experiments; -3.60% could be within run-to-run variance (we'd need more reps to be sure).

2. **Could fail to recover in-dist.** If the in-dist regression in #2889 came from BOTH wider MLPs in shallow blocks AND general capacity saturation, narrowing shallow blocks may not be enough.

3. **Asymmetric architecture may interact poorly with cosine LR.** Late-training plateau may be driven by deeper blocks' under-training; widening them late may worsen this.

## Falsifiable predictions

- **WIN** (val < 30.5605): Asymmetric capacity routing helps; try [3,3,4,5] or [2,3,4,5] next, or apply asymmetry to other components (slice_num, num_heads).
- **WASH** (val ≈ 30.5605 ± 0.5%): Asymmetric capacity is roughly equivalent to uniform mlp_ratio=3; the novel #2889 OOD signal was likely noise. Close axis.
- **LOSS** (val > 30.5605 + 1%): Asymmetric routing does not preserve the OOD gain; or wider deep blocks alone introduce their own training pathology. Close axis; the #2889 OOD signal may need a different intervention (e.g., geometry-conditioned routing instead of capacity).

## Implementation

### Step 1: Make `mlp_ratio` accept a per-block list

In `Transolver.__init__` (around line 254-262 in `train.py`):

```python
# BEFORE
self.blocks = nn.ModuleList([
    TransolverBlock(
        num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
        act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
        slice_num=slice_num, last_layer=(i == n_layers - 1),
        use_se=(i == n_layers - 1),
    )
    for i in range(n_layers)
])

# AFTER
# Accept either int or list — broadcast int to list of length n_layers.
if isinstance(mlp_ratio, (list, tuple)):
    assert len(mlp_ratio) == n_layers, f"mlp_ratio list len {len(mlp_ratio)} != n_layers {n_layers}"
    per_block_mlp_ratios = list(mlp_ratio)
else:
    per_block_mlp_ratios = [mlp_ratio] * n_layers

self.blocks = nn.ModuleList([
    TransolverBlock(
        num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
        act=act, mlp_ratio=per_block_mlp_ratios[i], out_dim=out_dim,
        slice_num=slice_num, last_layer=(i == n_layers - 1),
        use_se=(i == n_layers - 1),
    )
    for i in range(n_layers)
])
```

### Step 2: Set per-block mlp_ratio in the model config

In the `model_config` dict (around line 515-525 in `train.py`):

```python
# BEFORE
mlp_ratio=3,

# AFTER
mlp_ratio=[3, 3, 4, 4],
```

### Step 3: Verify SwiGLU sizing per block

In `TransolverBlock.__init__` line 213, `SwiGLUMLP` uses `hidden_swiglu = round(d * mlp_ratio * 2 / 3 / 8) * 8`. Per block:

| Block | mlp_ratio | hidden_swiglu | SwiGLU params |
|---|---|---|---|
| 0 | 3 | 192 | ~55,296 |
| 1 | 3 | 192 | ~55,296 |
| 2 | 4 | 256 | ~74,336 |
| 3 | 4 | 256 | ~74,336 |

Expected total SwiGLU = 259,264. Expected total model params ≈ **446,000** (between 407,940 baseline and 482,180 uniform mlp_ratio=4).

### Step 4: Startup diagnostic prints

After model construction add:

```python
print(f"Block-asymmetric mlp_ratio: {per_block_mlp_ratios}")
print(f"Per-block hidden_swiglu: {[round(96*r*2/3/8)*8 for r in per_block_mlp_ratios]}")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")
```

### Step 5: Diagnostics to log

Per-block diagnostics (same as #2889 — they will be different per block due to different widths):
- `block_<i>/gate_zero_frac` — expect lower for asymmetric than uniform mlp_ratio=4 (smaller width in shallow blocks)
- `block_<i>/gate_std`, `block_<i>/value_abs_mean` — should be stable

## Baseline (current best — PR #2879 Round 118)

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p | **30.5605** | best ep58/70 |
| test_avg/mae_surf_p | **26.5160** | from best-val ep58 |
| Param count | **407,940** | uniform mlp_ratio=3 |
| val_single_in_dist | **23.3997** | |
| val_geom_camber_rc | **46.0708** | |
| val_geom_camber_cruise | **17.8657** | |
| val_re_rand | **34.9057** | |

**Target to beat:** val_avg/mae_surf_p < **30.5605**

### Reference: #2889 uniform mlp_ratio=4 LOSS (the run we are exploiting)

| Split | val_mae (#2889) | Δ vs #2879 |
|---|---|---|
| val_single_in_dist | 25.9467 | **+10.88%** ❌ |
| val_geom_camber_rc | 45.7313 | -0.74% ✓ |
| val_geom_camber_cruise | 17.2232 | **-3.60%** ✓✓ |
| val_re_rand | 36.9007 | +5.72% ❌ |

**Hypothesis:** [3,3,4,4] preserves the camber_cruise/camber_rc improvement (the novel signal) and recovers val_single_in_dist (the main failure mode).

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/asym-mlp-ratio-3344" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 70
```

**IMPORTANT**: Use canonical hyperparameters (lr=1.5e-4, weight_decay=3e-4, epochs=70). The asymmetric mlp_ratio is HARDCODED in the model_config dict as `mlp_ratio=[3, 3, 4, 4]`; no new CLI flag needed.

**No W&B / wandb** — local JSONL only. `SENPAI_TIMEOUT_MINUTES=30` hard cap.

## Reporting

Post results as a PR comment including:

1. **val_avg/test_avg vs baseline #2879 (30.5605 / 26.5160)**.
2. **Per-split val + test breakdown** with delta vs #2879 baseline AND vs #2889 uniform mlp_ratio=4 reference. This is the critical comparison — does asymmetric recover in-dist AND preserve the camber_cruise gain?
3. **Per-block gate_zero_frac** comparison vs both #2879 (uniform=3) and #2889 (uniform=4). Expected: blocks 0/1 like #2879, blocks 2/3 like #2889 (but possibly improved due to smaller upstream context).
4. **Param count confirmation** — expect ~446,000.
5. Total epochs reached, sec/epoch, peak GPU memory.
6. Training-loss-vs-val-loss gap.
7. **Plain-language verdict on the novel signal** — did asymmetric capacity routing recover in-dist while preserving the camber-cruise gain? If yes → continue this axis with [3,3,4,5] or [2,3,4,5]. If no → close axis.

Use the terminal result marker:
```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
