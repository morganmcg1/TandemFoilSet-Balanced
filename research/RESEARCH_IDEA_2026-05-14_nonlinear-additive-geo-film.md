# Round 127 — Non-linear bottleneck on the ADDITIVE geo-FiLM site

## Hypothesis

Replace the linear `geo_film = Linear(8, n_hidden)` zero-init projection used in #2890 with a **non-linear bottleneck** `Linear(8, 16) → GELU → Linear(16, n_hidden)` and apply it ADDITIVELY at the same #2890 site (preprocess MLP output, before slicing). Tests whether a more expressive (non-linear) conditioner at the camber_cruise-WIN-producing site can break the cruise/in_dist trade-off coupling observed across #2889 / #2890 / #2899 / #2900.

## Why this might WIN

1. **The additive site is the WIN site.** #2890 produced the **largest-ever camber_cruise WIN of -9.87%** (val_geom_camber_cruise 16.1031 vs baseline 17.8657). #2900's multiplicative variant lost the win (-1.80% only). The win lives at the additive site; capacity should be added there, not moved.

2. **The linear projection cannot output "zero on in_dist, useful on cruise".** A single `Linear(8, 96)` can only learn a 8→96 affine map. To produce near-zero bias for in_dist geometries while producing useful bias for cruise geometries, it needs a soft gate or non-linearity. The non-linear bottleneck `Linear(8→16)→GELU→Linear(16→96)` with zero-init on the second layer provides exactly this — GELU's smooth saturation can route different geometry signatures to different sub-spaces, and the second layer can express "zero for in_dist" via specific weight patterns.

3. **Schmidhuber-style: this echoes a Mixture-of-Experts gate on geometry input.** The 16-dim bottleneck creates a low-dim "geometry signature" that can act as a soft routing layer. In_dist geometries (NACA-0012 + NACA-0024 etc) likely cluster in one region of this 16-dim space; cruise geometries (high-camber out-of-distribution) cluster in another. The second linear can produce different biases per region.

4. **#2900 student explicitly suggested this direction.** Suggestion #1 was: "Try the multiplicative gate but with a non-linear bottleneck: `Linear(8 → 16) → GELU → Linear(16 → n_hidden)`." We move it to the additive site (where the win existed) instead of multiplicative (where the win was lost) — this is the strongest possible test of the suggestion.

5. **Minimal complexity increase.** Adds ~912 params (1776 vs 864). Total model ≈ 408,852 — within the 407k–482k explored range. Pure capacity-shape test.

## Why this might LOSS

1. **The trade-off may be structural, not capacity-shape.** If the camber_cruise gain comes from the additive bias *forcing* a different feature representation that helps cruise but hurts in_dist, more conditioner expressivity cannot decouple them — the downstream features themselves trade off.

2. **Overparametrized conditioner could overfit.** With only ~3000 training samples and 1776 conditioner params, the bottleneck might learn idiosyncratic responses that don't generalize.

3. **GELU may be the wrong non-linearity here.** A bounded activation (tanh, sigmoid) could produce better in_dist zeroing behavior. GELU is unbounded.

## Falsifiable predictions

- **WIN** (val < 30.5605): Non-linear bottleneck breaks the coupling. Preserves camber_cruise gain (-3% or better) AND recovers in_dist (< +1%). Try other non-linearities (tanh, swish) and bottleneck widths.
- **PARTIAL** (camber_cruise < 17, in_dist < 25, val 30.6–31.0): Partial decoupling. Try wider bottleneck (32-dim), or apply non-linear conditioner at BOTH additive and multiplicative sites.
- **WASH/LOSS** (val ≈ 31.0–31.5): Non-linear capacity does NOT decouple the trade-off; the cruise-WIN/in_dist-LOSS trade-off is structural in the architecture. Close the geo-FiLM-axis-via-conditioner-capacity exploration.

## Implementation

### Step 1: Replace the linear geo_film projection with a non-linear bottleneck

In `Transolver.__init__` (around line 269-275 in `train.py`, where `geo_film` is currently defined as a linear):

```python
# BEFORE (from #2900 baseline currently merged?? — check first; otherwise this is greenfield)
self.geo_film = nn.Linear(8, n_hidden)
nn.init.zeros_(self.geo_film.weight)
nn.init.zeros_(self.geo_film.bias)

# AFTER — non-linear bottleneck with zero-init on second layer
geo_bottleneck = 16
self.geo_film = nn.Sequential(
    nn.Linear(8, geo_bottleneck),
    nn.GELU(),
    nn.Linear(geo_bottleneck, n_hidden),
)
# Zero-init only the LAST linear layer so the conditioner starts at identity
nn.init.zeros_(self.geo_film[-1].weight)
nn.init.zeros_(self.geo_film[-1].bias)
# First linear layer uses default init — should be small (Kaiming for GELU)
```

### Step 2: Apply ADDITIVELY at the #2890 site (preprocess MLP output)

In `Transolver.forward` (around line 282-293):

```python
# Existing flow-FiLM multiplicative (KEEP as-is)
flow_scalars = x[:, 0, [13, 14, 18]]
film_scale = self.film(flow_scalars)
fx = fx * (1 + film_scale)

# NEW: additive geometry conditioning at preprocess MLP output
# geometry channels: NACA0 (15-17), NACA1 (19-21), gap (22), stagger (23)
geo_signal = x[:, 0, [15, 16, 17, 19, 20, 21, 22, 23]]  # [B, 8]
geo_bias = self.geo_film(geo_signal)  # [B, n_hidden]
fx = fx + geo_bias.unsqueeze(1)  # broadcast over N (token dim)

# Then slicing as usual
```

**CRITICAL:** This is the same site/composition as #2890 (additive on preprocess output), NOT #2900's multiplicative composition. The non-linearity is the ONLY change.

### Step 3: Diagnostics to log

Per-batch on validation:
- `geo_film.0.weight.norm`, `geo_film.2.weight.norm` (first + last linear)
- `geo_film.2.weight.norm` should grow from 0 during training
- `geo_bias_norm` per split — should differ across splits (highest on camber_cruise per #2890 signature)
- Per-split `geo_bias` mean and std

Per-block: keep standard `block_<i>/gate_zero_frac`, `block_<i>/value_abs_mean`.

### Step 4: Startup diagnostic prints

```python
print(f"geo_film type: {type(self.geo_film).__name__}")
print(f"geo_film inner: {[type(m).__name__ for m in self.geo_film]}")
print(f"geo_film bottleneck width: {geo_bottleneck}")
print(f"Total param count: {sum(p.numel() for p in model.parameters())}")
```

Expected param count: 407,940 baseline + (8*16+16) + (16*96+96) = 407,940 + 144 + 1632 = **409,716** params.

## Baseline (current best — PR #2879 Round 118)

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p | **30.5605** | best ep58/70 |
| test_avg/mae_surf_p | **26.5160** | from best-val ep58 |
| Param count | **407,940** | |
| val_single_in_dist | **23.3997** | |
| val_geom_camber_rc | **46.0708** | |
| val_geom_camber_cruise | **17.8657** | |
| val_re_rand | **34.9057** | |

**Target to beat:** val_avg/mae_surf_p < **30.5605**

### Reference points (the trade-off curve we're trying to break)

| | val_avg | in_dist | camber_cruise | Site/composition |
|---|---|---|---|---|
| Baseline #2879 | 30.5605 | 23.3997 | 17.8657 | — |
| #2890 additive | 31.5045 | 27.3431 | **16.1031** ← WIN | Linear additive on preprocess |
| #2900 multiplicative | 31.5745 | 24.7743 | 17.5435 | Linear mult parallel to flow-FiLM |
| **This PR target** | **<30.5605** | **<24** | **<17** | Non-linear additive on preprocess |

We want the in_dist of #2900 AND the camber_cruise of #2890, achieved by giving the conditioner more expressive power at the WIN site.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/nonlinear-additive-geo-film" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

**IMPORTANT:** Use canonical hyperparameters (lr=1.5e-4, weight_decay=3e-4, epochs=60). Use **epochs=60** to avoid timeout-truncation issues seen in #2900 (which set epochs=70 and was cut at ep61).

The non-linear `geo_film` Sequential and the additive composition at the preprocess-MLP output are hardcoded in `Transolver`; no new CLI flag needed.

**No W&B / wandb** — local JSONL only. `SENPAI_TIMEOUT_MINUTES=30` hard cap.

## Reporting

Post results as a PR comment including:

1. **val_avg/test_avg vs baseline #2879 (30.5605 / 26.5160)**.
2. **Per-split val + test breakdown** with delta vs #2879 baseline AND vs **both** reference points #2890 (additive linear) and #2900 (multiplicative linear). The 4-row trade-off comparison is critical — does non-linear bottleneck at the additive site decouple cruise gain from in_dist loss?
3. **geo_film diagnostics:** weight norms of layer 0 (Linear 8→16) and layer 2 (Linear 16→96), per-split `geo_bias` mean/std/abs_mean. Compare camber_cruise signature against #2890's ~0.17 abs_mean and #2900's 0.25 abs_mean.
4. **Param count confirmation** — expect ~409,716.
5. Total epochs reached (target: all 60), sec/epoch, peak GPU memory.
6. Training-loss-vs-val-loss gap.
7. **Plain-language verdict on the decoupling hypothesis** — did non-linear bottleneck break the cruise/in_dist trade-off coupling? If yes (BOTH effects retained) → continue with wider bottleneck (32-dim) or tanh non-linearity. If no → close geo-FiLM-conditioner-capacity axis.

Use the terminal result marker:
```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
