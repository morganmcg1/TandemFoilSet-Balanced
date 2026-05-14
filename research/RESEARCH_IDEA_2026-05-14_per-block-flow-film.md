# Round 132 — Per-block flow-FiLM (multiplicative, replace single init)

## Hypothesis

**Replace** the single initial flow-FiLM (currently applied once at preprocess-MLP output) with **4 per-block flow-FiLM modules** — one Linear(3, n_hidden) zero-init projection per TransolverBlock, applied multiplicatively to the block's input. Tests whether **distributing the geometry-conditioning capacity across depth** can break the cruise/in_dist trade-off coupling observed across 10 consecutive meta-signal experiments.

## Why this might WIN

1. **Directly tests the "capacity allocation in conditioning pathway" diagnosis.** The student of #2912 (droppath LOSS) explicitly diagnosed: *"Cruise gains in prior experiments came from capacity reallocation across the geometry-conditioning pathway, not co-adaptation between blocks ... If revisiting this trade-off, target the FiLM/geo-conditioning sites (per-channel scale, per-block FiLM, separate cruise vs in_dist gates)."* This experiment is the most direct test of that hypothesis.

2. **Conditioning at depth-0 is a known bottleneck for transformer FiLM.** Single-site FiLM (applied once at the embedding) cannot adapt features that emerge in deeper layers. Per-block FiLM is the standard fix from DiT (Peebles & Xie 2022), AdaIN-style image transformers, and conditioned U-Nets. Our 4-block transformer has *4 different feature regimes* across depth — each could benefit from re-conditioning on the flow scalars.

3. **The 10-experiment meta-signal cluster is conditioning-pathway-shaped.**
   - Geo-FiLM additive linear (#2890): cruise WIN -9.87%, in_dist LOSS +16.85%
   - Geo-FiLM multiplicative linear (#2900): cruise WIN -1.80%, in_dist LOSS +5.88%
   - Geo-FiLM nonlinear additive (#2911): uniform LOSS (capacity-shape failure)
   - These all modify *what features at depth-0 see geometry*; none modify *how depth-1,2,3 reconcile geometry with attention's slice-routing*. Per-block FiLM operates at every depth.

4. **Schmidhuber-style: this is "conditional batch norm" / FiLM at every block.** A very classic, well-established design that's been used in conditional GANs, conditional U-Nets, BigGAN, ImageNet-conditional classifiers since 2017. It is *not* a novel architectural mod — it's a recipe gap we've left on the table.

5. **Minimal complexity increase.** Each per-block FiLM is Linear(3, 96) = 288 weights + 96 bias = 384 params. 4 blocks → 1536 params. Replacing the existing single FiLM (384 params) gives net +1152 params (~0.28% increase). Total ~409,092 params.

6. **Zero-init each FiLM preserves baseline at step 0.** Each FiLM starts at `1 * (1 + 0) = 1`, so the model begins exactly at baseline behavior. Training learns to use per-block conditioning gradually.

## Why this might LOSS

1. **The trade-off may not be conditioning-capacity-limited.** If the cruise/in_dist coupling is driven by the slicing/routing layer (which lives inside attention, NOT in the conditioning pathway), more FiLM capacity won't help.

2. **4 conditioning sites on 4 blocks may overfit at 3000 samples.** Each FiLM has only 384 params, but the combinatorial interaction of 4 independent conditioning maps could create high-variance routing per geometry. With only ~3000 training points, this could memorize idiosyncratic per-sample conditioning rather than generalize.

3. **Multiplicative composition is the LOSS-coupled side of the 2-point trade-off curve.** #2900 multiplicative was less bad than #2890 additive on cruise (-1.80% vs -9.87%) but still net LOSS. If per-block multiplicative composition inherits this property, we may get a less-bad-but-still-net-LOSS result. Mitigation: a partial improvement could still beat baseline given the distribution across blocks compounds differently.

4. **Could shift the trade-off to a NEW axis** — e.g., camber_rc (the hardest split) might suffer if per-block FiLM disproportionately specializes on cruise+in_dist splits.

## Falsifiable predictions

- **WIN** (val < 30.5605): Per-block FiLM breaks the conditioning-capacity ceiling. Try `LayerNorm`-style FiLM (replace 1+x with γ,β pair) and per-block geometry-FiLM as composites.
- **PARTIAL** (in_dist within ±1% AND camber_cruise improved): trade-off partially decoupled by depth-distributed conditioning. Try widening per-block FiLM to 2-layer MLP, or add geometry channels to per-block FiLM input.
- **WASH** (val ≈ 30.5605 ± 0.5%): Per-block FiLM has no effect at this scale. Try LayerNorm-style γ,β FiLM or apply to LN_2 (MLP side) only before closing.
- **LOSS** (val > 31.0): Conditioning-capacity is not the bottleneck. Combined with #2911 closure (conditioner-expressivity-capacity), this CLOSES the entire flow-FiLM-pathway-capacity axis cluster. 111th taxon.

## Implementation

### Step 1: Replace `self.film` with `self.block_films`

In `Transolver.__init__` (around where `self.film` is currently defined, near line 269-275 in `train.py`):

```python
# BEFORE
self.film = nn.Linear(3, n_hidden)
nn.init.zeros_(self.film.weight)
nn.init.zeros_(self.film.bias)

# AFTER
self.block_films = nn.ModuleList([
    nn.Linear(3, n_hidden) for _ in range(n_layers)
])
for film in self.block_films:
    nn.init.zeros_(film.weight)
    nn.init.zeros_(film.bias)
```

### Step 2: Apply per-block FiLM at each block input

In `Transolver.forward`, REMOVE the existing single-site flow-FiLM application (currently right after preprocess MLP):

```python
# REMOVE
# flow_scalars = x[:, 0, [13, 14, 18]]
# film_scale = self.film(flow_scalars)
# fx = fx * (1 + film_scale)
```

Then INSIDE the block-loop, BEFORE each `block(fx)` call:

```python
flow_scalars = x[:, 0, [13, 14, 18]]   # [B, 3] — keep outside loop or compute once
for i, block in enumerate(self.blocks):
    film_scale = self.block_films[i](flow_scalars)   # [B, n_hidden]
    fx = fx * (1 + film_scale.unsqueeze(1))           # broadcast over token dim
    fx = block(fx)
```

**Note:** `flow_scalars` is computed ONCE outside the loop (it's static per-batch). Only the per-block FiLM module changes between iterations.

**CRITICAL:** Multiplicative composition `fx * (1 + film_scale)` — same as the current baseline FiLM. NOT additive. This is the conservative composition that won in baseline.

### Step 3: Diagnostics

Add at training start:
```python
print(f"Per-block FiLM: {len(model.block_films)} modules, "
      f"each Linear({model.block_films[0].in_features}, {model.block_films[0].out_features})")
print(f"Total flow-FiLM params: {sum(p.numel() for m in model.block_films for p in m.parameters())}")
print(f"Total model params: {sum(p.numel() for p in model.parameters())}")
```

Per-epoch validation:
- `block_films.{0,1,2,3}.weight.norm()` — confirms each FiLM module learns non-zero weights
- Per-split `film_scale` mean and std at each block — does deeper conditioning differ from depth-0?
- Per-split `||film_scale||` magnitude at each block — does cruise have larger conditioning magnitude at deep blocks?

### Step 4: Sanity check at step 0

With all FiLM weights/biases zero-init, the first forward pass produces `fx * (1 + 0) = fx` exactly — identity composition. Verify validation metrics at step 0 (epoch 0) closely match a baseline #2879 fresh-init run.

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

**Beat:** `val_avg/mae_surf_p < 30.5605`

### Reference: prior FiLM-axis experiments to compare against

| Experiment | val_avg Δ | in_dist Δ | camber_cruise Δ | FiLM site/composition |
|---|---|---|---|---|
| Baseline #2879 | — | — | — | Single flow-FiLM (mult) at preprocess output |
| #2890 additive geo-FiLM | +3.09% LOSS | +16.85% LOSS | **-9.87% WIN** | Linear geo-FiLM (additive) at preprocess output |
| #2900 multiplicative geo-FiLM | +3.32% LOSS | +5.88% LOSS | **-1.80% WIN** | Linear geo-FiLM (mult) parallel to flow-FiLM |
| #2911 nonlinear geo-FiLM | +7.97% LOSS | +14.40% LOSS | +7.04% LOSS | Nonlinear bottleneck (additive) at preprocess output — 109th taxon CLOSED |
| **This PR target** | **<0% WIN** | **<0% WIN** | **<0% WIN** | Per-block flow-FiLM (mult, 4 sites) replacing single-site |

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/per-block-flow-film" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — `self.block_films` ModuleList is hardcoded in Transolver. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **FiLM-pathway-capacity trade-off table:** the 4-row table above with this PR's row appended. Does distributing FiLM across depth retain ANY meta-signal element (cruise WIN) while NOT regressing in_dist?
4. Param count confirmation (~409,092 — +1152 vs baseline)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train-loss vs val-loss gap
7. **Per-block FiLM diagnostic:** per-block `||film_scale||` mean across the validation set. If deeper blocks consistently have larger conditioning magnitude, conditioning capacity at depth IS the bottleneck. If all blocks have similar magnitudes, the depth-distribution doesn't matter and we should fall back to single-site capacity tweaks.
8. **Plain-language verdict:** did per-block FiLM break the cruise/in_dist coupling? If yes (BOTH improve OR cruise WIN with in_dist held) → try wider FiLM, LN-style γ,β, or geometry-conditioned per-block FiLM. If WASH → try LN-style FiLM. If LOSS → close FiLM-pathway-capacity axis cluster (combined with #2911 = 111th taxon).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
